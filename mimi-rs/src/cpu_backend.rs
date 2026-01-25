use crate::{Result, WithDType, WithDTypeF};
use rayon::prelude::*;

impl crate::Backend for () {
    type Storage<T: WithDType> = Vec<T>;

    fn storage_len<T: WithDType>(storage: &Self::Storage<T>) -> usize {
        storage.len()
    }

    unsafe fn alloc_uninit<T: WithDType>(len: usize, _: &Self) -> Result<Self::Storage<T>> {
        Ok(vec![T::zero(); len])
    }

    fn from_vec<T: WithDType>(v: Vec<T>, _: &Self) -> Result<Self::Storage<T>> {
        Ok(v)
    }

    fn data<T: WithDType>(src: &Self::Storage<T>, len: usize) -> Result<std::borrow::Cow<'_, [T]>> {
        Ok(std::borrow::Cow::Borrowed(&src[..len]))
    }

    fn add_assign<T: WithDType>(
        dst: &mut Self::Storage<T>,
        s: &Self::Storage<T>,
        l: usize,
    ) -> Result<()> {
        s[..l].iter().zip(dst[..l].iter_mut()).for_each(|(src, dst)| *dst += *src);
        Ok(())
    }

    fn mul_assign<T: WithDType>(
        dst: &mut Self::Storage<T>,
        s: &Self::Storage<T>,
        l: usize,
    ) -> Result<()> {
        s[..l].iter().zip(dst[..l].iter_mut()).for_each(|(src, dst)| *dst *= *src);
        Ok(())
    }

    fn scale<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        m: T,
        l: usize,
    ) -> Result<()> {
        for (d, s) in dst[..l].iter_mut().zip(&src[..l]) {
            *d = *s * m
        }
        Ok(())
    }

    fn add<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        l: usize,
    ) -> Result<()> {
        for ((d, l), r) in dst[..l].iter_mut().zip(&lhs[..l]).zip(&rhs[..l]) {
            *d = *l + *r
        }
        Ok(())
    }

    fn mul<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        l: usize,
    ) -> Result<()> {
        for ((d, l), r) in dst[..l].iter_mut().zip(&lhs[..l]).zip(&rhs[..l]) {
            *d = *l * *r
        }
        Ok(())
    }

    fn maximum<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        l: usize,
    ) -> Result<()> {
        for ((d, l), r) in dst[..l].iter_mut().zip(&lhs[..l]).zip(&rhs[..l]) {
            *d = if *l > *r { *l } else { *r }
        }
        Ok(())
    }

    fn minimum<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        l: usize,
    ) -> Result<()> {
        for ((d, l), r) in dst[..l].iter_mut().zip(&lhs[..l]).zip(&rhs[..l]) {
            *d = if *l < *r { *l } else { *r }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn copy2d<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        d1: usize,
        d2: usize,
        dst_s: usize,
        src_s: usize,
        dst_o: usize,
        src_o: usize,
    ) -> Result<()> {
        for i1 in 0..d1 {
            let dst_idx = i1 * dst_s + dst_o;
            let src_idx = i1 * src_s + src_o;
            let dst = &mut dst[dst_idx..dst_idx + d2];
            let src = &src[src_idx..src_idx + d2];
            dst.copy_from_slice(src)
        }
        Ok(())
    }

    fn transpose<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim1: usize,
        dim2: usize,
        dims: &[usize],
    ) -> Result<()> {
        if dim1 == dim2 {
            dst.copy_from_slice(src);
        } else {
            let (dim1, dim2) = (usize::min(dim1, dim2), usize::max(dim1, dim2));
            let d_i = dims[..dim1].iter().product::<usize>();
            let d_j = dims[dim1 + 1..dim2].iter().product::<usize>();
            let d_k = dims[(dim2 + 1)..].iter().product::<usize>();
            let d1 = dims[dim1];
            let d2 = dims[dim2];
            // Inefficient, we should blit the data where possible.
            // i: pre
            for i in 0..d_i {
                for a1 in 0..d1 {
                    // j: mid
                    for j in 0..d_j {
                        for a2 in 0..d2 {
                            // k: post
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
                                dst[dst_idx] = src[src_idx]
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn copy<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        l: usize,
    ) -> Result<()> {
        dst[..l].copy_from_slice(&src[..l]);
        Ok(())
    }

    fn fill<T: WithDType>(dst: &mut Self::Storage<T>, v: T, l: usize) -> Result<()> {
        dst[..l].fill(v);
        Ok(())
    }

    fn rope<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        cos: &Self::Storage<T>,
        sin: &Self::Storage<T>,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        pos: usize,
    ) -> Result<()> {
        if dst.len() != b * h * t * d {
            crate::bail!("rope unexpected size for dst {} {b} {h} {t} {d}", dst.len())
        }
        if src.len() != b * h * t * d {
            crate::bail!("rope unexpected size for src {} {b} {h} {t} {d}", src.len())
        }
        let cos = &cos[pos * d / 2..];
        let sin = &sin[pos * d / 2..];
        dst.par_chunks_mut(t * d).zip(src.par_chunks(t * d)).for_each(|(dst, src)| {
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

    fn rope_i<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        cos: &Self::Storage<T>,
        sin: &Self::Storage<T>,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        pos: usize,
    ) -> Result<()> {
        if dst.len() != b * h * t * d {
            crate::bail!("rope-i unexpected size for dst {} {b} {h} {t} {d}", dst.len())
        }
        if src.len() != b * h * t * d {
            crate::bail!("rope-i unexpected size for src {} {b} {h} {t} {d}", src.len())
        }
        let cos = &cos[pos * d / 2..];
        let sin = &sin[pos * d / 2..];
        dst.par_chunks_mut(t * d).zip(src.par_chunks(t * d)).for_each(|(dst, src)| {
            for i_over_2 in 0..t * d / 2 {
                let i = 2 * i_over_2;
                let (s_i, s_ip) = (src[i], src[i + 1]);
                dst[i] = s_i * cos[i_over_2] - s_ip * sin[i_over_2];
                dst[i + 1] = s_i * sin[i_over_2] + s_ip * cos[i_over_2];
            }
        });
        Ok(())
    }

    fn gemm<T: WithDType>(
        dst: &mut Self::Storage<T>,
        (lhs, lhs_o): (&Self::Storage<T>, usize),
        (rhs, rhs_o): (&Self::Storage<T>, usize),
        m: usize,
        n: usize,
        k: usize,
        lhs_b: usize,
        b_stride: usize,
        (dst_cs, dst_rs): (usize, usize),
        (lhs_cs, lhs_rs): (usize, usize),
        (rhs_cs, rhs_rs): (usize, usize),
    ) -> Result<()> {
        let lhs = &lhs[lhs_o..];
        let rhs = &rhs[rhs_o..];
        for b_idx in 0..lhs_b {
            let dst = &mut dst[b_idx * m * n..(b_idx + 1) * m * n];
            let lhs = &lhs[b_idx * m * k..(b_idx + 1) * m * k];
            let rhs = &rhs[b_idx * b_stride..];
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
                    /* rhs: *const T = */ rhs.as_ptr(),
                    /* rhs_cs: isize = */ rhs_cs as isize,
                    /* rhs_rs: isize = */ rhs_rs as isize,
                    /* alpha: T = */ T::zero(),
                    /* beta: T = */ T::one(),
                    /* conj_dst: bool = */ false,
                    /* conj_lhs: bool = */ false,
                    /* conj_rhs: bool = */ false,
                    gemm::Parallelism::Rayon(get_num_threads()),
                )
            }
        }
        Ok(())
    }

    fn index_select<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        ids: &[u32],
        dim: usize,
        dims: &[usize],
    ) -> Result<()> {
        let left_size: usize = dims[..dim].iter().product();
        let right_size: usize = dims[dim + 1..].iter().product::<usize>().max(1);
        let src_dim_size = dims[dim];

        for left in 0..left_size {
            for (i, &idx) in ids.iter().enumerate() {
                let idx = idx as usize;
                let src_offset = left * src_dim_size * right_size + idx * right_size;
                let dst_offset = left * ids.len() * right_size + i * right_size;
                dst[dst_offset..dst_offset + right_size]
                    .copy_from_slice(&src[src_offset..src_offset + right_size]);
            }
        }
        Ok(())
    }

    fn cos<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        l: usize,
    ) -> Result<()> {
        for (d, s) in dst[..l].iter_mut().zip(&src[..l]) {
            *d = s.cos();
        }
        Ok(())
    }

    fn sin<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        l: usize,
    ) -> Result<()> {
        for (d, s) in dst[..l].iter_mut().zip(&src[..l]) {
            *d = s.sin();
        }
        Ok(())
    }

    fn silu<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        l: usize,
    ) -> Result<()> {
        for (d, s) in dst[..l].iter_mut().zip(&src[..l]) {
            *d = *s / (T::one() + (T::zero() - *s).exp())
        }
        Ok(())
    }

    fn apply_causality_mask<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        bh: usize,
        t1: usize,
        t2: usize,
        offset: usize,
    ) -> Result<()> {
        for idx_b in 0..bh {
            for idx1 in 0..t1 {
                // Query at position offset + idx1 can attend to keys at positions 0..=offset+idx1
                // So mask positions where idx2 > offset + idx1
                for idx2 in (offset + idx1 + 1)..t2 {
                    let idx = idx_b * t1 * t2 + idx1 * t2 + idx2;
                    dst[idx] = T::neg_infinity()
                }
            }
        }
        Ok(())
    }

    fn softmax<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
    ) -> Result<()> {
        let src = &src[..d * dim_m1];
        let dst = &mut dst[..d * dim_m1];
        src.par_chunks(dim_m1).zip(dst.par_chunks_mut(dim_m1)).for_each(|(src, dst)| {
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

    fn rms_norm<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        alpha: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
        eps: f32,
    ) -> Result<()> {
        let src = &src[..d * dim_m1];
        let dst = &mut dst[..d * dim_m1];
        src.par_chunks(dim_m1).zip(dst.par_chunks_mut(dim_m1)).for_each(|(src, dst)| {
            let sum2 = src.iter().map(|&v| v.to_f32() * v.to_f32()).sum::<f32>();
            let m = (sum2 / dim_m1 as f32 + eps).sqrt();
            for ((d, s), alpha) in dst.iter_mut().zip(src.iter()).zip(alpha) {
                *d = T::from_f32((*s).to_f32() / m * (*alpha).to_f32())
            }
        });
        Ok(())
    }

    fn layer_norm<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        weight: &Self::Storage<T>,
        bias: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
        eps: f32,
    ) -> Result<()> {
        let src = &src[..d * dim_m1];
        let dst = &mut dst[..d * dim_m1];
        let weight = &weight[..dim_m1];
        let bias = &bias[..dim_m1];
        src.par_chunks(dim_m1).zip(dst.par_chunks_mut(dim_m1)).for_each(|(src, dst)| {
            // Compute mean
            let sum: f32 = src.iter().map(|&v| v.to_f32()).sum();
            let mean = sum / dim_m1 as f32;

            // Compute variance
            let var: f32 = src.iter().map(|&v| {
                let diff = v.to_f32() - mean;
                diff * diff
            }).sum::<f32>() / dim_m1 as f32;

            let inv_std = 1.0 / (var + eps).sqrt();

            // Normalize and apply weight/bias
            for i in 0..dim_m1 {
                let normalized = (src[i].to_f32() - mean) * inv_std;
                dst[i] = T::from_f32(normalized * weight[i].to_f32() + bias[i].to_f32());
            }
        });
        Ok(())
    }

    fn sqr<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        l: usize,
    ) -> Result<()> {
        for (d, s) in dst[..l].iter_mut().zip(&src[..l]) {
            *d = *s * *s;
        }
        Ok(())
    }

    fn sqrt<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        l: usize,
    ) -> Result<()> {
        for (d, s) in dst[..l].iter_mut().zip(&src[..l]) {
            *d = s.sqrt();
        }
        Ok(())
    }

    fn abs<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        l: usize,
    ) -> Result<()> {
        for (d, s) in dst[..l].iter_mut().zip(&src[..l]) {
            *d = s.abs();
        }
        Ok(())
    }

    fn gelu_erf<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        l: usize,
    ) -> Result<()> {
        // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
        let sqrt_2_inv = std::f32::consts::FRAC_1_SQRT_2;
        for (d, s) in dst[..l].iter_mut().zip(&src[..l]) {
            let x: f32 = <T as WithDTypeF>::to_f32(*s);
            let erf_val = libm::erff(x * sqrt_2_inv);
            *d = T::from_f32(x * 0.5 * (1.0 + erf_val));
        }
        Ok(())
    }

    fn elu<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        alpha: f32,
        l: usize,
    ) -> Result<()> {
        for (d, s) in dst[..l].iter_mut().zip(&src[..l]) {
            let x: f32 = <T as WithDTypeF>::to_f32(*s);
            *d = T::from_f32(if x > 0.0 { x } else { alpha * (x.exp() - 1.0) });
        }
        Ok(())
    }

    fn relu<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        l: usize,
    ) -> Result<()> {
        for (d, s) in dst[..l].iter_mut().zip(&src[..l]) {
            *d = T::max(*s, T::zero());
        }
        Ok(())
    }

    fn tanh<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        l: usize,
    ) -> Result<()> {
        for (d, s) in dst[..l].iter_mut().zip(&src[..l]) {
            *d = s.tanh();
        }
        Ok(())
    }

    fn sigmoid<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        l: usize,
    ) -> Result<()> {
        for (d, s) in dst[..l].iter_mut().zip(&src[..l]) {
            *d = T::one() / (T::one() + (T::zero() - *s).exp());
        }
        Ok(())
    }

    fn reduce_max<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut max_val = T::neg_infinity();
                for d in 0..dim_size {
                    let src_idx = outer * dim_size * inner_size + d * inner_size + inner;
                    max_val = T::max(max_val, src[src_idx]);
                }
                let dst_idx = outer * inner_size + inner;
                dst[dst_idx] = max_val;
            }
        }
        Ok(())
    }

    fn reduce_min<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut min_val = T::infinity();
                for d in 0..dim_size {
                    let src_idx = outer * dim_size * inner_size + d * inner_size + inner;
                    min_val = T::min(min_val, src[src_idx]);
                }
                let dst_idx = outer * inner_size + inner;
                dst[dst_idx] = min_val;
            }
        }
        Ok(())
    }

    fn reduce_argmin<T: WithDTypeF>(
        dst: &mut Self::Storage<i64>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut min_val = T::infinity();
                let mut min_idx: usize = 0;
                for d in 0..dim_size {
                    let src_idx = outer * dim_size * inner_size + d * inner_size + inner;
                    if src[src_idx].to_f32() < min_val.to_f32() {
                        min_val = src[src_idx];
                        min_idx = d;
                    }
                }
                let dst_idx = outer * inner_size + inner;
                dst[dst_idx] = min_idx as i64;
            }
        }
        Ok(())
    }

    fn reduce_sum<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut sum = T::zero();
                for d in 0..dim_size {
                    let src_idx = outer * dim_size * inner_size + d * inner_size + inner;
                    sum += src[src_idx];
                }
                let dst_idx = outer * inner_size + inner;
                dst[dst_idx] = sum;
            }
        }
        Ok(())
    }

    fn broadcast_add<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        dst_shape: &[usize],
        lhs_strides: &[usize],
        rhs_strides: &[usize],
    ) -> Result<()> {
        broadcast_binary_op(dst, lhs, rhs, dst_shape, lhs_strides, rhs_strides, |a, b| a + b)
    }

    fn broadcast_sub<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        dst_shape: &[usize],
        lhs_strides: &[usize],
        rhs_strides: &[usize],
    ) -> Result<()> {
        broadcast_binary_op(dst, lhs, rhs, dst_shape, lhs_strides, rhs_strides, |a, b| a - b)
    }

    fn broadcast_mul<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        dst_shape: &[usize],
        lhs_strides: &[usize],
        rhs_strides: &[usize],
    ) -> Result<()> {
        broadcast_binary_op(dst, lhs, rhs, dst_shape, lhs_strides, rhs_strides, |a, b| a * b)
    }

    fn broadcast_div<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        dst_shape: &[usize],
        lhs_strides: &[usize],
        rhs_strides: &[usize],
    ) -> Result<()> {
        broadcast_binary_op(dst, lhs, rhs, dst_shape, lhs_strides, rhs_strides, |a, b| a / b)
    }

    #[allow(clippy::too_many_arguments)]
    fn conv1d<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        kernel: &Self::Storage<T>,
        batch: usize,
        in_channels: usize,
        out_channels: usize,
        length: usize,
        out_length: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<()> {
        let in_c_per_group = in_channels / groups;

        // Initialize output to zero
        dst.iter_mut().for_each(|v| *v = T::zero());

        // Reorder input from [B, C, L] to [B, L, C] for better memory access in the inner loop
        let mut src_reordered = vec![T::zero(); batch * length * in_channels];
        for b in 0..batch {
            for l in 0..length {
                for c in 0..in_channels {
                    let src_idx = b * in_channels * length + c * length + l;
                    let dst_idx = b * length * in_channels + l * in_channels + c;
                    src_reordered[dst_idx] = src[src_idx];
                }
            }
        }

        // Process each kernel offset
        for k_offset in 0..kernel_size {
            // Parallelize over output channels
            (0..out_channels).into_par_iter().for_each(|out_c| {
                let g = out_c / (out_channels / groups);
                let in_c_start = g * in_c_per_group;

                // Gather kernel weights for this output channel and kernel offset
                // kernel layout: [out_channels, in_c_per_group, kernel_size]
                let k_cont: Vec<T> = (0..in_c_per_group)
                    .map(|ic| {
                        let k_idx =
                            out_c * in_c_per_group * kernel_size + ic * kernel_size + k_offset;
                        kernel[k_idx]
                    })
                    .collect();

                for b in 0..batch {
                    let dst_base = b * out_channels * out_length + out_c * out_length;

                    for ol in 0..out_length {
                        let src_l = ol * stride + k_offset * dilation;

                        // Check padding bounds
                        if src_l < padding || src_l >= padding + length {
                            continue;
                        }
                        let src_l = src_l - padding;

                        // Compute dot product over input channels
                        let src_base = b * length * in_channels + src_l * in_channels + in_c_start;
                        let mut d = T::zero();
                        for ic in 0..in_c_per_group {
                            d += src_reordered[src_base + ic] * k_cont[ic];
                        }

                        // Accumulate into output
                        // Safety: each out_c is processed by a different thread, so no races
                        let dst_idx = dst_base + ol;
                        unsafe {
                            let ptr = dst.as_ptr().add(dst_idx) as *mut T;
                            *ptr += d;
                        }
                    }
                }
            });
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_transpose1d<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        kernel: &Self::Storage<T>,
        batch: usize,
        in_channels: usize,
        out_channels: usize,
        length: usize,
        out_length: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        _output_padding: usize,
        groups: usize,
    ) -> Result<()> {
        let in_c_per_group = in_channels / groups;
        let out_c_per_group = out_channels / groups;

        // Initialize output to zero
        dst.iter_mut().for_each(|v| *v = T::zero());

        // Reorder input from [B, C, L] to [B, L, C] for contiguous memory access
        let mut src_reordered = vec![T::zero(); batch * length * in_channels];
        for b in 0..batch {
            for l in 0..length {
                for c in 0..in_channels {
                    let src_idx = b * in_channels * length + c * length + l;
                    let dst_idx = b * length * in_channels + l * in_channels + c;
                    src_reordered[dst_idx] = src[src_idx];
                }
            }
        }

        // Process each kernel offset
        for k_offset in 0..kernel_size {
            // Parallelize over output channels
            (0..out_channels).into_par_iter().for_each(|out_c| {
                let g = out_c / out_c_per_group;
                let oc_in_group = out_c % out_c_per_group;
                let in_c_start = g * in_c_per_group;

                // Gather kernel weights for this output channel and kernel offset
                // Kernel layout: [in_channels, out_channels/groups, kernel_size]
                let k_cont: Vec<T> = (0..in_c_per_group)
                    .map(|ic| {
                        let in_c = in_c_start + ic;
                        let k_idx = in_c * out_c_per_group * kernel_size
                            + oc_in_group * kernel_size
                            + k_offset;
                        kernel[k_idx]
                    })
                    .collect();

                for b in 0..batch {
                    for il in 0..length {
                        let out_pos_raw = il * stride + k_offset;

                        // Check padding bounds
                        if out_pos_raw < padding || out_pos_raw >= out_length + padding {
                            continue;
                        }
                        let out_pos = out_pos_raw - padding;

                        // Compute dot product over input channels
                        let src_base = b * length * in_channels + il * in_channels + in_c_start;
                        let mut d = T::zero();
                        for ic in 0..in_c_per_group {
                            d += src_reordered[src_base + ic] * k_cont[ic];
                        }

                        // Accumulate into output
                        // Safety: each out_c is processed by a different thread, so no races
                        let dst_idx = b * out_channels * out_length + out_c * out_length + out_pos;
                        unsafe {
                            let ptr = dst.as_ptr().add(dst_idx) as *mut T;
                            *ptr += d;
                        }
                    }
                }
            });
        }
        Ok(())
    }
}

/// Helper function for broadcast binary operations.
fn broadcast_binary_op<T: WithDTypeF>(
    dst: &mut [T],
    lhs: &[T],
    rhs: &[T],
    dst_shape: &[usize],
    lhs_strides: &[usize],
    rhs_strides: &[usize],
    op: impl Fn(T, T) -> T,
) -> Result<()> {
    let total_elems: usize = dst_shape.iter().product();
    let rank = dst_shape.len();

    for (dst_idx, dst) in dst.iter_mut().enumerate().take(total_elems) {
        // Convert linear index to multi-dimensional indices
        let mut remaining = dst_idx;
        let mut lhs_idx = 0usize;
        let mut rhs_idx = 0usize;

        for d in 0..rank {
            let stride: usize = dst_shape[d + 1..].iter().product::<usize>().max(1);
            let coord = remaining / stride;
            remaining %= stride;

            lhs_idx += coord * lhs_strides[d];
            rhs_idx += coord * rhs_strides[d];
        }

        *dst = op(lhs[lhs_idx], rhs[rhs_idx]);
    }

    Ok(())
}

pub(crate) fn get_num_threads() -> usize {
    use std::str::FromStr;
    // Respond to the same environment variable as rayon.
    match std::env::var("RAYON_NUM_THREADS").ok().and_then(|s| usize::from_str(&s).ok()) {
        Some(x) if x > 0 => x,
        Some(_) | None => num_cpus::get(),
    }
}
