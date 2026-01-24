use rayon::prelude::*;

use crate::error::check_same_shape;
use crate::{Error, Result};
use crate::{Tensor, WithDType, WithDTypeF};

impl<T: WithDType> Tensor<T> {
    pub fn inplace_add(&mut self, other: &Tensor<T>) -> Result<()> {
        check_same_shape(&self.shape, &other.shape, "inplace_add")?;
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += *b;
        }
        Ok(())
    }

    pub fn add_(&mut self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Result<()> {
        if lhs.shape != rhs.shape {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: lhs.shape.clone(),
                rhs: rhs.shape.clone(),
                op: "add_",
            }
            .bt());
        }
        if self.shape != lhs.shape {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape.clone(),
                rhs: lhs.shape.clone(),
                op: "add_ (output)",
            }
            .bt());
        }
        for ((a, b), c) in self.data.iter_mut().zip(lhs.data.iter()).zip(rhs.data.iter()) {
            *a = *b + *c;
        }
        Ok(())
    }

    pub fn mul_(&mut self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Result<()> {
        if lhs.shape != rhs.shape {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: lhs.shape.clone(),
                rhs: rhs.shape.clone(),
                op: "mul_",
            }
            .bt());
        }
        if self.shape != lhs.shape {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape.clone(),
                rhs: lhs.shape.clone(),
                op: "mul_ (output)",
            }
            .bt());
        }
        for ((a, b), c) in self.data.iter_mut().zip(lhs.data.iter()).zip(rhs.data.iter()) {
            *a = *b * *c;
        }
        Ok(())
    }

    pub fn transpose_(&mut self, src: &Tensor<T>, dim1: usize, dim2: usize) -> Result<()> {
        let dims = src.dims();
        if dim1 == dim2 {
            self.data.copy_from_slice(&src.data);
        } else {
            let (dim1, dim2) = (usize::min(dim1, dim2), usize::max(dim1, dim2));
            // d_i: product of dimensions before dim1
            // d_j: product of dimensions between dim1 and dim2
            // d_k: product of dimensions after dim2
            // d1: size of dimension dim1
            // d2: size of dimension dim2
            let d_i: usize = dims[..dim1].iter().product();
            let d_j: usize = dims[dim1 + 1..dim2].iter().product();
            let d_k: usize = dims[dim2 + 1..].iter().product();
            let d1 = dims[dim1];
            let d2 = dims[dim2];

            for i in 0..d_i {
                for a1 in 0..d1 {
                    for j in 0..d_j {
                        for a2 in 0..d2 {
                            for k in 0..d_k {
                                let src_idx = i * d1 * d_j * d2 * d_k
                                    + a1 * d_j * d2 * d_k
                                    + j * d2 * d_k
                                    + a2 * d_k
                                    + k;
                                let dst_idx = i * d2 * d_j * d1 * d_k
                                    + a2 * d_j * d1 * d_k
                                    + j * d1 * d_k
                                    + a1 * d_k
                                    + k;
                                self.data[dst_idx] = src.data[src_idx];
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn copy_(&mut self, src: &Tensor<T>) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "copy_")?;
        self.data.copy_from_slice(&src.data);
        Ok(())
    }

    pub fn fill_(&mut self, value: T) -> Result<()> {
        self.data.fill(value);
        Ok(())
    }

    pub fn inplace_scale(&mut self, m: T) -> Result<()> {
        for a in self.data.iter_mut() {
            *a *= m;
        }
        Ok(())
    }
}

impl<T: WithDTypeF> Tensor<T> {
    pub fn cos_(&mut self, src: &Tensor<T>) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "cos_")?;
        for (a, b) in self.data.iter_mut().zip(src.data.iter()) {
            *a = b.cos();
        }
        Ok(())
    }

    pub fn sin_(&mut self, src: &Tensor<T>) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "sin_")?;
        for (a, b) in self.data.iter_mut().zip(src.data.iter()) {
            *a = b.sin();
        }
        Ok(())
    }

    pub fn exp_(&mut self, src: &Tensor<T>) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "exp_")?;
        for (a, b) in self.data.iter_mut().zip(src.data.iter()) {
            *a = b.exp();
        }
        Ok(())
    }

    pub fn silu_(&mut self, src: &Tensor<T>) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "silu_")?;
        for (a, b) in self.data.iter_mut().zip(src.data.iter()) {
            *a = *b / (T::one() + (-*b).exp());
        }
        Ok(())
    }

    pub fn softmax_(&mut self, src: &Tensor<T>) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "softmax_")?;
        let dim_m1 = self.shape.dims().last().copied().unwrap_or(1);
        src.data.par_chunks(dim_m1).zip(self.data.par_chunks_mut(dim_m1)).for_each(|(src, dst)| {
            let mut max = T::neg_infinity();
            for &v in src.iter() {
                max = T::max(v, max)
            }
            for (s, d) in src.iter().zip(dst.iter_mut()) {
                *d = (*s - max).exp();
            }
            let sum_exp = dst.iter().map(|v| <T as WithDTypeF>::to_f32(*v)).sum::<f32>();
            for d in dst.iter_mut() {
                *d = T::from_f32(d.to_f32() / sum_exp)
            }
        });
        Ok(())
    }

    /// Causal softmax: applies causal mask before softmax.
    /// Input shape: (batch, heads, seq_q, seq_kv)
    /// q_offset: the position of the first query token (for cached generation)
    /// Masks positions where query_pos < key_pos (future positions)
    pub fn softmax_causal_(&mut self, src: &Tensor<T>, q_offset: usize) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "softmax_causal_")?;
        let dims = self.dims();
        if dims.len() != 4 {
            crate::bail!("softmax_causal_ requires 4D tensor, got {:?}", self.shape());
        }
        let (_b, _h, seq_q, seq_kv) = (dims[0], dims[1], dims[2], dims[3]);

        // Process each row (seq_q position) separately
        src.data
            .par_chunks(seq_kv)
            .zip(self.data.par_chunks_mut(seq_kv))
            .enumerate()
            .for_each(|(row_idx, (src, dst))| {
                // row_idx maps to which query position within the batch*heads*seq_q
                let q_pos = row_idx % seq_q + q_offset;

                // Apply causal mask and find max
                let mut max = T::neg_infinity();
                for (kv_pos, &v) in src.iter().enumerate() {
                    if kv_pos <= q_pos {
                        max = T::max(v, max);
                    }
                }

                // Compute exp with mask
                let mut sum_exp: f32 = 0.0;
                for (kv_pos, (s, d)) in src.iter().zip(dst.iter_mut()).enumerate() {
                    if kv_pos <= q_pos {
                        *d = (*s - max).exp();
                        sum_exp += d.to_f32();
                    } else {
                        *d = T::zero();
                    }
                }

                // Normalize
                if sum_exp > 0.0 {
                    for d in dst.iter_mut() {
                        *d = T::from_f32(d.to_f32() / sum_exp);
                    }
                }
            });
        Ok(())
    }

    pub fn rms_norm_(&mut self, src: &Tensor<T>, alpha: &Tensor<T>, eps: f32) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "rms_norm_ src")?;
        if eps <= 0.0 {
            crate::bail!("rms_norm_ eps must be positive");
        }
        let dim_m1 = self.shape.dims().last().copied().unwrap_or(1);
        let expected_shape_alpha = dim_m1.into();
        check_same_shape(&alpha.shape, &expected_shape_alpha, "rms_norm_ alpha")?;
        src.data.par_chunks(dim_m1).zip(self.data.par_chunks_mut(dim_m1)).for_each(|(src, dst)| {
            let sum2 = src.iter().map(|&v| v.to_f32() * v.to_f32()).sum::<f32>();
            let m = (sum2 / dim_m1 as f32 + eps).sqrt();
            for ((d, s), alpha) in dst.iter_mut().zip(src.iter()).zip(alpha.data.iter()) {
                *d = T::from_f32((*s).to_f32() / m * (*alpha).to_f32())
            }
        });
        Ok(())
    }

    pub fn matmul_(&mut self, lhs: &Tensor<T>, rhs: &Tensor<T>, rhs_t: bool) -> Result<()> {
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

        let lhs_strides = lhs.shape().stride_contiguous();
        let rhs_strides = rhs.shape().stride_contiguous();
        let (dst_rs, dst_cs) = (n, 1);

        let (lhs_stride_m2, lhs_stride_m1) = {
            let l = lhs_strides.len();
            (lhs_strides[l - 2], lhs_strides[l - 1])
        };
        let (lhs_rs, lhs_cs) = (lhs_stride_m2, lhs_stride_m1);

        let (rhs_stride_m2, rhs_stride_m1) = {
            let l = rhs_strides.len();
            (rhs_strides[l - 2], rhs_strides[l - 1])
        };
        let (rhs_rs, rhs_cs) =
            if rhs_t { (rhs_stride_m1, rhs_stride_m2) } else { (rhs_stride_m2, rhs_stride_m1) };

        // rhs matrix size for indexing (original layout, before considering transpose)
        let rhs_mat_size = rhs_dims[rhs_dims.len() - 2] * rhs_dims[rhs_dims.len() - 1];

        for b_idx in 0..lhs_batch {
            let dst = &mut self.data[b_idx * m * n..(b_idx + 1) * m * n];
            let lhs = &lhs.data[b_idx * m * k..(b_idx + 1) * m * k];
            let rhs_idx = if rhs_batch == 1 { 0 } else { b_idx };
            let rhs_ptr = unsafe { rhs.data.as_ptr().add(rhs_idx * rhs_mat_size) };
            unsafe {
                gemm::gemm(
                    /* m: usize = */ m,
                    /* n: usize = */ n,
                    /* k: usize = */ k,
                    /* dst: *mut T = */ dst.as_mut_ptr(),
                    /* dst_cs: isize = */ dst_cs as isize,
                    /* dst_rs: isize = */ dst_rs as isize,
                    /* read_dst: bool = */ false,
                    /* lhs: *const T = */ lhs.as_ptr(),
                    /* lhs_cs: isize = */ lhs_cs as isize,
                    /* lhs_rs: isize = */ lhs_rs as isize,
                    /* rhs: *const T = */ rhs_ptr,
                    /* rhs_cs: isize = */ rhs_cs as isize,
                    /* rhs_rs: isize = */ rhs_rs as isize,
                    /* alpha: T = */ T::zero(),
                    /* beta: T = */ T::one(),
                    /* conj_dst: bool = */ false,
                    /* conj_lhs: bool = */ false,
                    /* conj_rhs: bool = */ false,
                    gemm::Parallelism::Rayon(crate::utils::get_num_threads()),
                )
            }
        }

        Ok(())
    }

    pub fn rope_(
        &mut self,
        src: &Tensor<T>,
        cos: &Tensor<T>,
        sin: &Tensor<T>,
        pos: usize,
    ) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "rope_ src")?;
        check_same_shape(&cos.shape, &sin.shape, "rope_ cos/sin")?;
        let (_b, _h, t, d) = self.dims4()?;
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
        let cos = &cos.data[pos * d / 2..];
        let sin = &sin.data[pos * d / 2..];
        src.data
            .par_chunks(t * d)
            .zip(self.data.par_chunks_mut(t * d))
            .for_each(|(src, dst)| {
                for i_t in 0..t {
                    for i_d in 0..d / 2 {
                        let i1 = i_t * d + i_d;
                        let i2 = i1 + d / 2;
                        let i_cs = i_t * (d / 2) + i_d;
                        let (src_i1, src_i2) = (src[i1], src[i2]);
                        dst[i1] = src_i1 * cos[i_cs] - src_i2 * sin[i_cs];
                        dst[i2] = src_i1 * sin[i_cs] + src_i2 * cos[i_cs];
                    }
                }
            });

        Ok(())
    }

    pub fn rope_i_(
        &mut self,
        src: &Tensor<T>,
        cos: &Tensor<T>,
        sin: &Tensor<T>,
        pos: usize,
    ) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "rope_i_ src")?;
        check_same_shape(&cos.shape, &sin.shape, "rope_i_ cos/sin")?;
        let (_b, _h, t, d) = self.dims4()?;
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
        let cos = &cos.data[pos * d / 2..];
        let sin = &sin.data[pos * d / 2..];
        src.data
            .par_chunks(t * d)
            .zip(self.data.par_chunks_mut(t * d))
            .for_each(|(src, dst)| {
                for i_over_2 in 0..t * d / 2 {
                    let i = 2 * i_over_2;
                    let (s_i, s_ip) = (src[i], src[i + 1]);
                    dst[i] = s_i * cos[i_over_2] - s_ip * sin[i_over_2];
                    dst[i + 1] = s_i * sin[i_over_2] + s_ip * cos[i_over_2];
                }
            });
        Ok(())
    }
}
