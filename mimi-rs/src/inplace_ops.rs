use crate::error::check_same_shape;
use crate::{Backend, BackendF, Result, Tensor, WithDType, WithDTypeF};

impl<T: WithDType, B: Backend<T>> Tensor<T, B> {
    pub fn inplace_add(&mut self, other: &Self) -> Result<()> {
        check_same_shape(&self.shape, &other.shape, "inplace_add")?;
        self.data.add_assign(&other.data, self.elem_count())?;
        Ok(())
    }

    pub fn add_(&mut self, lhs: &Self, rhs: &Self) -> Result<()> {
        check_same_shape(&lhs.shape, &rhs.shape, "add_")?;
        check_same_shape(&self.shape, &lhs.shape, "add_ (output)")?;
        self.data.add(&lhs.data, &rhs.data, self.elem_count())?;
        Ok(())
    }

    pub fn mul_(&mut self, lhs: &Self, rhs: &Self) -> Result<()> {
        check_same_shape(&lhs.shape, &rhs.shape, "mul_")?;
        check_same_shape(&self.shape, &lhs.shape, "mul_ (output)")?;
        self.data.mul(&lhs.data, &rhs.data, self.elem_count())?;
        Ok(())
    }

    pub fn transpose_(&mut self, src: &Self, dim1: usize, dim2: usize) -> Result<()> {
        let dims = src.dims();
        if dim1 == dim2 {
            self.data.copy(&src.data, self.elem_count())?;
        } else {
            self.data.transpose(&src.data, dim1, dim2, dims)?;
        }
        Ok(())
    }

    pub fn copy_(&mut self, src: &Self) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "copy_")?;
        self.data.copy(&src.data, self.elem_count())?;
        Ok(())
    }

    pub fn fill_(&mut self, value: T) -> Result<()> {
        self.data.fill(value, self.elem_count())?;
        Ok(())
    }

    pub fn scale_(&mut self, src: &Self, m: T) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "scale_")?;
        self.data.scale(&src.data, m, self.elem_count())?;
        Ok(())
    }
}

impl<T: WithDTypeF, B: BackendF<T>> Tensor<T, B> {
    pub fn cos_(&mut self, src: &Self) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "cos_")?;
        self.data.cos(&src.data, self.elem_count())?;
        Ok(())
    }

    pub fn sin_(&mut self, src: &Self) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "sin_")?;
        self.data.sin(&src.data, self.elem_count())?;
        Ok(())
    }

    pub fn silu_(&mut self, src: &Self) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "silu_")?;
        self.data.silu(&src.data, self.elem_count())?;
        Ok(())
    }

    pub fn softmax_(&mut self, src: &Self) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "softmax_")?;
        let dim_m1 = self.shape.dims().last().copied().unwrap_or(1);
        let d = self.elem_count() / dim_m1;
        self.data.softmax(&src.data, dim_m1, d)?;
        Ok(())
    }

    /// Apply causality mask in-place.
    /// Shape: (batch * heads, seq_q, seq_kv)
    /// Masks positions where key position > query position + offset (sets to -inf).
    /// offset: starting position of the first query token (for KV cache generation).
    pub fn apply_causality_mask_(&mut self, offset: usize) -> Result<()> {
        let (bh, t1, t2) = self.dims3()?;
        self.data.apply_causality_mask(bh, t1, t2, offset)?;
        Ok(())
    }

    pub fn rms_norm_(&mut self, src: &Self, alpha: &Self, eps: f32) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "rms_norm_ src")?;
        if eps <= 0.0 {
            crate::bail!("rms_norm_ eps must be positive");
        }
        let dim_m1 = self.shape.dims().last().copied().unwrap_or(1);
        let d = self.elem_count() / dim_m1;
        let expected_shape_alpha = dim_m1.into();
        check_same_shape(&alpha.shape, &expected_shape_alpha, "rms_norm_ alpha")?;
        self.data.rms_norm(&src.data, &alpha.data, dim_m1, d, eps)?;
        Ok(())
    }

    pub fn matmul_(&mut self, lhs: &Self, rhs: &Self, rhs_t: bool) -> Result<()> {
        let lhs_dims = lhs.dims();
        let rhs_dims = rhs.dims();

        if lhs_dims.len() < 2 || rhs_dims.len() < 2 {
            crate::bail!(
                "matmul requires at least 2D tensors, got lhs {:?}, rhs {:?}",
                lhs.shape(),
                rhs.shape()
            );
        }

        // Extract M, K from lhs (last two dimensions)
        let lhs_m = lhs_dims[lhs_dims.len() - 2];
        let lhs_k = lhs_dims[lhs_dims.len() - 1];

        // Extract K, N from rhs (last two dimensions), accounting for transpose
        let (rhs_k, rhs_n) = if rhs_t {
            (rhs_dims[rhs_dims.len() - 1], rhs_dims[rhs_dims.len() - 2])
        } else {
            (rhs_dims[rhs_dims.len() - 2], rhs_dims[rhs_dims.len() - 1])
        };

        if lhs_k != rhs_k {
            crate::bail!(
                "matmul inner dimension mismatch: lhs {:?}, rhs {:?}, rhs_t={rhs_t}",
                lhs.shape(),
                rhs.shape()
            );
        }

        // Compute batch dimensions
        let lhs_batch_dims = &lhs_dims[..lhs_dims.len() - 2];
        let rhs_batch_dims = &rhs_dims[..rhs_dims.len() - 2];
        let lhs_batch: usize = lhs_batch_dims.iter().product::<usize>().max(1);
        let rhs_batch: usize = rhs_batch_dims.iter().product::<usize>().max(1);

        // Check batch dimensions are compatible
        if rhs_batch != 1 && rhs_batch != lhs_batch {
            crate::bail!(
                "matmul batch dimension mismatch: lhs {:?}, rhs {:?}",
                lhs.shape(),
                rhs.shape()
            );
        }

        let (m, n, k) = (lhs_m, rhs_n, lhs_k);
        let dst_elems = lhs_batch * m * n;

        if dst_elems > self.data.len() {
            crate::bail!(
                "matmul dst is too small, dst {} < {dst_elems}, lhs {:?} rhs {:?}",
                self.data.len(),
                lhs.shape(),
                rhs.shape()
            );
        }

        // For row-major contiguous tensors, gemm strides:
        // - cs = column stride (moving to next column in same row) = 1 for row-major
        // - rs = row stride (moving to next row in same column) = num_cols for row-major
        // - dst: m×n, cs = 1, rs = n
        // - lhs: m×k, cs = 1, rs = k
        // - rhs: k×n (or n×k if transposed), cs = 1, rs = n (or swapped if transposed)
        let (dst_cs, dst_rs) = (1, n);
        let (lhs_cs, lhs_rs) = (1, lhs_k);
        let (rhs_cs, rhs_rs) = if rhs_t {
            // rhs is stored as n×k but we want k×n, swap strides
            // Original: cs=1, rs=k. After transpose: cs=k, rs=1
            (rhs_dims[rhs_dims.len() - 1], 1)
        } else {
            (1, rhs_n)
        };

        // rhs matrix size for batch stride
        let rhs_mat_size = rhs_dims[rhs_dims.len() - 2] * rhs_dims[rhs_dims.len() - 1];
        let b_stride = if rhs_batch == 1 { 0 } else { rhs_mat_size };

        self.data.gemm(
            (&lhs.data, 0),
            (&rhs.data, 0),
            m,
            n,
            k,
            lhs_batch,
            b_stride,
            (dst_cs, dst_rs),
            (lhs_cs, lhs_rs),
            (rhs_cs, rhs_rs),
        )?;

        Ok(())
    }

    pub fn rope_(&mut self, src: &Self, cos: &Self, sin: &Self, pos: usize) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "rope_ src")?;
        check_same_shape(&cos.shape, &sin.shape, "rope_ cos/sin")?;
        let (b, h, t, d) = self.dims4()?;
        let (max_pos, d_over_2) = cos.dims2()?;
        if d_over_2 * 2 != d {
            crate::bail!(
                "rope_ requires even d dimension, got d={d}, {:?} {:?}",
                self.shape(),
                cos.shape()
            );
        }
        if pos + t > max_pos {
            crate::bail!(
                "rope_ position out of range, pos={pos} + t={t} > max_pos={max_pos}, {:?} {:?}",
                self.shape(),
                cos.shape()
            );
        }
        self.data.rope(&src.data, &cos.data, &sin.data, b, h, t, d, pos)?;
        Ok(())
    }

    pub fn rope_i_(&mut self, src: &Self, cos: &Self, sin: &Self, pos: usize) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "rope_i_ src")?;
        check_same_shape(&cos.shape, &sin.shape, "rope_i_ cos/sin")?;
        let (b, h, t, d) = self.dims4()?;
        let (max_pos, d_over_2) = cos.dims2()?;
        if d_over_2 * 2 != d {
            crate::bail!(
                "rope_i_ requires even d dimension, got d={d}, {:?} {:?}",
                self.shape(),
                cos.shape()
            );
        }
        if pos + t > max_pos {
            crate::bail!(
                "rope_i_ position out of range, pos={pos} + t={t} > max_pos={max_pos}, {:?} {:?}",
                self.shape(),
                cos.shape()
            );
        }
        self.data.rope_i(&src.data, &cos.data, &sin.data, b, h, t, d, pos)?;
        Ok(())
    }
}
