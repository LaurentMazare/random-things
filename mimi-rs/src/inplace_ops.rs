use crate::error::check_same_shape;
use crate::{Backend, Result, Tensor, WithDType, WithDTypeF};

impl<T: WithDType, B: Backend> Tensor<T, B> {
    pub fn inplace_add(&mut self, other: &Self) -> Result<()> {
        check_same_shape(&self.shape, &other.shape, "inplace_add")?;
        let len = self.elem_count();
        B::add_assign(&mut self.data, &other.data, len)?;
        Ok(())
    }

    pub fn add_(&mut self, lhs: &Self, rhs: &Self) -> Result<()> {
        check_same_shape(&lhs.shape, &rhs.shape, "add_")?;
        check_same_shape(&self.shape, &lhs.shape, "add_ (output)")?;
        let len = self.elem_count();
        B::add(&mut self.data, &lhs.data, &rhs.data, len)?;
        Ok(())
    }

    pub fn mul_(&mut self, lhs: &Self, rhs: &Self) -> Result<()> {
        check_same_shape(&lhs.shape, &rhs.shape, "mul_")?;
        check_same_shape(&self.shape, &lhs.shape, "mul_ (output)")?;
        let len = self.elem_count();
        B::mul(&mut self.data, &lhs.data, &rhs.data, len)?;
        Ok(())
    }

    pub fn transpose_(&mut self, src: &Self, dim1: usize, dim2: usize) -> Result<()> {
        let dims = src.dims();
        let len = self.elem_count();
        if dim1 == dim2 {
            B::copy(&mut self.data, &src.data, len)?;
        } else {
            B::transpose(&mut self.data, &src.data, dim1, dim2, dims)?;
        }
        Ok(())
    }

    pub fn copy_(&mut self, src: &Self) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "copy_")?;
        let len = self.elem_count();
        B::copy(&mut self.data, &src.data, len)?;
        Ok(())
    }

    pub fn fill_(&mut self, value: T) -> Result<()> {
        let len = self.elem_count();
        B::fill(&mut self.data, value, len)?;
        Ok(())
    }

    pub fn scale_(&mut self, src: &Self, m: T) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "scale_")?;
        let len = self.elem_count();
        B::scale(&mut self.data, &src.data, m, len)?;
        Ok(())
    }
}

impl<T: WithDTypeF, B: Backend> Tensor<T, B> {
    pub fn cos_(&mut self, src: &Self) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "cos_")?;
        let len = self.elem_count();
        B::cos(&mut self.data, &src.data, len)?;
        Ok(())
    }

    pub fn sin_(&mut self, src: &Self) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "sin_")?;
        let len = self.elem_count();
        B::sin(&mut self.data, &src.data, len)?;
        Ok(())
    }

    pub fn silu_(&mut self, src: &Self) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "silu_")?;
        let len = self.elem_count();
        B::silu(&mut self.data, &src.data, len)?;
        Ok(())
    }

    pub fn softmax_(&mut self, src: &Self) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "softmax_")?;
        let dim_m1 = self.shape.dims().last().copied().unwrap_or(1);
        let d = self.elem_count() / dim_m1;
        B::softmax(&mut self.data, &src.data, dim_m1, d)?;
        Ok(())
    }

    /// Apply causality mask in-place.
    /// Shape: (batch * heads, seq_q, seq_kv)
    /// Masks positions where key position > query position + offset (sets to -inf).
    /// offset: starting position of the first query token (for KV cache generation).
    pub fn apply_causality_mask_(&mut self, offset: usize) -> Result<()> {
        let (bh, t1, t2) = self.dims3()?;
        B::apply_causality_mask(&mut self.data, bh, t1, t2, offset)?;
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
        B::rms_norm(&mut self.data, &src.data, &alpha.data, dim_m1, d, eps)?;
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
        let storage_len = B::storage_len(&self.data);

        if dst_elems > storage_len {
            crate::bail!(
                "matmul dst is too small, dst {} < {dst_elems}, lhs {:?} rhs {:?}",
                storage_len,
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

        B::gemm(
            &mut self.data,
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
        B::rope(&mut self.data, &src.data, &cos.data, &sin.data, b, h, t, d, pos)?;
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
        B::rope_i(&mut self.data, &src.data, &cos.data, &sin.data, b, h, t, d, pos)?;
        Ok(())
    }

    pub fn sqr_(&mut self, src: &Self) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "sqr_")?;
        let len = self.elem_count();
        B::sqr(&mut self.data, &src.data, len)?;
        Ok(())
    }

    pub fn sqrt_(&mut self, src: &Self) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "sqrt_")?;
        let len = self.elem_count();
        B::sqrt(&mut self.data, &src.data, len)?;
        Ok(())
    }

    pub fn abs_(&mut self, src: &Self) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "abs_")?;
        let len = self.elem_count();
        B::abs(&mut self.data, &src.data, len)?;
        Ok(())
    }

    pub fn gelu_erf_(&mut self, src: &Self) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "gelu_erf_")?;
        let len = self.elem_count();
        B::gelu_erf(&mut self.data, &src.data, len)?;
        Ok(())
    }

    pub fn elu_(&mut self, src: &Self, alpha: f32) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "elu_")?;
        let len = self.elem_count();
        B::elu(&mut self.data, &src.data, alpha, len)?;
        Ok(())
    }

    pub fn relu_(&mut self, src: &Self) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "relu_")?;
        let len = self.elem_count();
        B::relu(&mut self.data, &src.data, len)?;
        Ok(())
    }

    pub fn tanh_(&mut self, src: &Self) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "tanh_")?;
        let len = self.elem_count();
        B::tanh(&mut self.data, &src.data, len)?;
        Ok(())
    }

    pub fn sigmoid_(&mut self, src: &Self) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "sigmoid_")?;
        let len = self.elem_count();
        B::sigmoid(&mut self.data, &src.data, len)?;
        Ok(())
    }

    pub fn reduce_max_(&mut self, src: &Self, dim: usize) -> Result<()> {
        let src_dims = src.dims();
        let dim_size = src_dims[dim];
        let outer_size: usize = src_dims[..dim].iter().product::<usize>().max(1);
        let inner_size: usize = src_dims[dim + 1..].iter().product::<usize>().max(1);
        B::reduce_max(&mut self.data, &src.data, dim_size, outer_size, inner_size)?;
        Ok(())
    }

    pub fn reduce_min_(&mut self, src: &Self, dim: usize) -> Result<()> {
        let src_dims = src.dims();
        let dim_size = src_dims[dim];
        let outer_size: usize = src_dims[..dim].iter().product::<usize>().max(1);
        let inner_size: usize = src_dims[dim + 1..].iter().product::<usize>().max(1);
        B::reduce_min(&mut self.data, &src.data, dim_size, outer_size, inner_size)?;
        Ok(())
    }

    pub fn reduce_argmin_(&mut self, src: &Self, dim: usize) -> Result<()> {
        let src_dims = src.dims();
        let dim_size = src_dims[dim];
        let outer_size: usize = src_dims[..dim].iter().product::<usize>().max(1);
        let inner_size: usize = src_dims[dim + 1..].iter().product::<usize>().max(1);
        B::reduce_argmin(&mut self.data, &src.data, dim_size, outer_size, inner_size)?;
        Ok(())
    }

    pub fn broadcast_add_(&mut self, lhs: &Self, rhs: &Self) -> Result<()> {
        let dst_shape = self.dims().to_vec();
        let (lhs_strides, rhs_strides) =
            compute_broadcast_strides(&dst_shape, lhs.dims(), rhs.dims())?;
        B::broadcast_add(
            &mut self.data,
            &lhs.data,
            &rhs.data,
            &dst_shape,
            &lhs_strides,
            &rhs_strides,
        )?;
        Ok(())
    }

    pub fn broadcast_sub_(&mut self, lhs: &Self, rhs: &Self) -> Result<()> {
        let dst_shape = self.dims().to_vec();
        let (lhs_strides, rhs_strides) =
            compute_broadcast_strides(&dst_shape, lhs.dims(), rhs.dims())?;
        B::broadcast_sub(
            &mut self.data,
            &lhs.data,
            &rhs.data,
            &dst_shape,
            &lhs_strides,
            &rhs_strides,
        )?;
        Ok(())
    }

    pub fn broadcast_mul_(&mut self, lhs: &Self, rhs: &Self) -> Result<()> {
        let dst_shape = self.dims().to_vec();
        let (lhs_strides, rhs_strides) =
            compute_broadcast_strides(&dst_shape, lhs.dims(), rhs.dims())?;
        B::broadcast_mul(
            &mut self.data,
            &lhs.data,
            &rhs.data,
            &dst_shape,
            &lhs_strides,
            &rhs_strides,
        )?;
        Ok(())
    }

    pub fn broadcast_div_(&mut self, lhs: &Self, rhs: &Self) -> Result<()> {
        let dst_shape = self.dims().to_vec();
        let (lhs_strides, rhs_strides) =
            compute_broadcast_strides(&dst_shape, lhs.dims(), rhs.dims())?;
        B::broadcast_div(
            &mut self.data,
            &lhs.data,
            &rhs.data,
            &dst_shape,
            &lhs_strides,
            &rhs_strides,
        )?;
        Ok(())
    }
}

/// Compute broadcast strides for lhs and rhs given the output shape.
/// Returns (lhs_strides, rhs_strides) where stride is 0 for broadcast dimensions.
fn compute_broadcast_strides(
    out_shape: &[usize],
    lhs_shape: &[usize],
    rhs_shape: &[usize],
) -> crate::Result<(Vec<usize>, Vec<usize>)> {
    let out_rank = out_shape.len();
    let lhs_rank = lhs_shape.len();
    let rhs_rank = rhs_shape.len();

    let mut lhs_strides = vec![0usize; out_rank];
    let mut rhs_strides = vec![0usize; out_rank];

    // Compute strides for lhs (right-aligned)
    let lhs_offset = out_rank - lhs_rank;
    let mut lhs_stride = 1usize;
    for i in (0..lhs_rank).rev() {
        let out_idx = i + lhs_offset;
        if lhs_shape[i] == out_shape[out_idx] {
            lhs_strides[out_idx] = lhs_stride;
            lhs_stride *= lhs_shape[i];
        } else if lhs_shape[i] == 1 {
            lhs_strides[out_idx] = 0; // broadcast dimension
        } else {
            crate::bail!(
                "broadcast shape mismatch: lhs dim {} is {} but output is {}",
                i,
                lhs_shape[i],
                out_shape[out_idx]
            );
        }
    }

    // Compute strides for rhs (right-aligned)
    let rhs_offset = out_rank - rhs_rank;
    let mut rhs_stride = 1usize;
    for i in (0..rhs_rank).rev() {
        let out_idx = i + rhs_offset;
        if rhs_shape[i] == out_shape[out_idx] {
            rhs_strides[out_idx] = rhs_stride;
            rhs_stride *= rhs_shape[i];
        } else if rhs_shape[i] == 1 {
            rhs_strides[out_idx] = 0; // broadcast dimension
        } else {
            crate::bail!(
                "broadcast shape mismatch: rhs dim {} is {} but output is {}",
                i,
                rhs_shape[i],
                out_shape[out_idx]
            );
        }
    }

    Ok((lhs_strides, rhs_strides))
}
