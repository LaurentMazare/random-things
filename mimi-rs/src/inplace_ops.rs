use crate::error::check_same_shape;
use crate::{Backend, Result, Tensor, WithDType, WithDTypeF};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnaryOp {
    Cos,
    Sin,
    Sqr,
    Sqrt,
    Abs,
    GeluErf,
    Elu { alpha: f32 },
    Relu,
    Tanh,
    Sigmoid,
}

impl UnaryOp {
    pub fn as_str(&self) -> &'static str {
        match self {
            UnaryOp::Cos => "cos",
            UnaryOp::Sin => "sin",
            UnaryOp::Sqr => "sqr",
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Abs => "abs",
            UnaryOp::GeluErf => "gelu_erf",
            UnaryOp::Elu { .. } => "elu",
            UnaryOp::Relu => "relu",
            UnaryOp::Tanh => "tanh",
            UnaryOp::Sigmoid => "sigmoid",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    Minimum,
}

impl BinaryOp {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
            BinaryOp::Maximum => "maximum",
            BinaryOp::Minimum => "minimum",
        }
    }
}

impl<T: WithDType, B: Backend> Tensor<T, B> {
    pub(crate) fn inplace_binary(&self, other: &Self, op: BinaryOp) -> Result<()> {
        check_same_shape(&self.shape, &other.shape, op.as_str())?;
        let len = self.elem_count();
        let mut dst = self.storage_mut()?;
        let src = other.storage()?;
        B::bin_assign(&mut *dst, &*src, len, op)?;
        Ok(())
    }

    pub fn inplace_add(&self, other: &Self) -> Result<()> {
        self.inplace_binary(other, BinaryOp::Add)
    }

    pub fn inplace_sub(&self, other: &Self) -> Result<()> {
        self.inplace_binary(other, BinaryOp::Sub)
    }

    pub fn inplace_mul(&self, other: &Self) -> Result<()> {
        self.inplace_binary(other, BinaryOp::Mul)
    }

    pub fn inplace_div(&self, other: &Self) -> Result<()> {
        self.inplace_binary(other, BinaryOp::Div)
    }

    pub fn inplace_maximum(&self, other: &Self) -> Result<()> {
        self.inplace_binary(other, BinaryOp::Maximum)
    }

    pub fn inplace_minimum(&self, other: &Self) -> Result<()> {
        self.inplace_binary(other, BinaryOp::Minimum)
    }

    pub fn binary_(&self, lhs: &Self, rhs: &Self, op: BinaryOp) -> Result<()> {
        check_same_shape(&lhs.shape, &rhs.shape, op.as_str())?;
        check_same_shape(&self.shape, &lhs.shape, op.as_str())?;
        let len = self.elem_count();
        let mut dst = self.storage_mut()?;
        let lhs_data = lhs.storage()?;
        let rhs_data = rhs.storage()?;
        B::binary(&mut *dst, &*lhs_data, &*rhs_data, len, op)?;
        Ok(())
    }

    pub fn add_(&self, lhs: &Self, rhs: &Self) -> Result<()> {
        self.binary_(lhs, rhs, BinaryOp::Add)
    }

    pub fn sub_(&self, lhs: &Self, rhs: &Self) -> Result<()> {
        self.binary_(lhs, rhs, BinaryOp::Sub)
    }

    pub fn mul_(&self, lhs: &Self, rhs: &Self) -> Result<()> {
        self.binary_(lhs, rhs, BinaryOp::Mul)
    }

    pub fn div_(&self, lhs: &Self, rhs: &Self) -> Result<()> {
        self.binary_(lhs, rhs, BinaryOp::Div)
    }

    pub fn maximum_(&self, lhs: &Self, rhs: &Self) -> Result<()> {
        self.binary_(lhs, rhs, BinaryOp::Maximum)
    }

    pub fn minimum_(&self, lhs: &Self, rhs: &Self) -> Result<()> {
        self.binary_(lhs, rhs, BinaryOp::Minimum)
    }

    pub fn transpose_(&self, src: &Self, dim1: usize, dim2: usize) -> Result<()> {
        let dims = src.dims();
        let len = self.elem_count();
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        if dim1 == dim2 {
            B::copy(&mut *dst, &*src_data, len)?;
        } else {
            B::transpose(&mut *dst, &*src_data, dim1, dim2, dims)?;
        }
        Ok(())
    }

    pub fn copy_(&self, src: &Self) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "copy_")?;
        let len = self.elem_count();
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        B::copy(&mut *dst, &*src_data, len)?;
        Ok(())
    }

    pub fn fill_(&self, value: T) -> Result<()> {
        let len = self.elem_count();
        let mut dst = self.storage_mut()?;
        B::fill(&mut *dst, value, len)?;
        Ok(())
    }

    pub fn scale_(&self, src: &Self, m: T) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "scale_")?;
        let len = self.elem_count();
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        B::scale(&mut *dst, &*src_data, m, len)?;
        Ok(())
    }
}

impl<T: WithDTypeF, B: Backend> Tensor<T, B> {
    pub fn unary_(&self, src: &Self, op: UnaryOp) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, op.as_str())?;
        let len = self.elem_count();
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        B::unary(&mut *dst, &*src_data, len, op)?;
        Ok(())
    }

    pub fn cos_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::Cos)
    }

    pub fn sin_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::Sin)
    }

    pub fn gelu_erf_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::GeluErf)
    }

    pub fn elu_(&self, src: &Self, alpha: f32) -> Result<()> {
        self.unary_(src, UnaryOp::Elu { alpha })
    }

    pub fn abs_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::Abs)
    }

    pub fn sqr_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::Sqr)
    }

    pub fn sqrt_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::Sqrt)
    }

    pub fn relu_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::Relu)
    }

    pub fn tanh_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::Tanh)
    }

    pub fn sigmoid_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::Sigmoid)
    }

    pub fn silu_(&self, src: &Self) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "silu_")?;
        let len = self.elem_count();
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        B::silu(&mut *dst, &*src_data, len)?;
        Ok(())
    }

    pub fn softmax_(&self, src: &Self) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "softmax_")?;
        let dim_m1 = self.shape.dims().last().copied().unwrap_or(1);
        let d = self.elem_count() / dim_m1;
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        B::softmax(&mut *dst, &*src_data, dim_m1, d)?;
        Ok(())
    }

    /// Apply causality mask in-place.
    /// Shape: (batch * heads, seq_q, seq_kv) or (batch, heads, seq_q, seq_kv)
    /// Masks positions where key position > query position + offset (sets to -inf).
    /// offset: starting position of the first query token (for KV cache generation).
    pub fn apply_causality_mask_(&self, offset: usize) -> Result<()> {
        let dims = self.dims();
        let (bh, t1, t2) = match dims.len() {
            3 => (dims[0], dims[1], dims[2]),
            4 => (dims[0] * dims[1], dims[2], dims[3]),
            _ => crate::bail!(
                "apply_causality_mask expects 3D or 4D tensor, got shape {:?}",
                self.shape()
            ),
        };
        let mut dst = self.storage_mut()?;
        B::apply_causality_mask(&mut *dst, bh, t1, t2, offset)?;
        Ok(())
    }

    pub fn rms_norm_(&self, src: &Self, alpha: &Self, eps: f32) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "rms_norm_ src")?;
        if eps <= 0.0 {
            crate::bail!("rms_norm_ eps must be positive");
        }
        let dim_m1 = self.shape.dims().last().copied().unwrap_or(1);
        let d = self.elem_count() / dim_m1;
        let expected_shape_alpha = dim_m1.into();
        check_same_shape(&alpha.shape, &expected_shape_alpha, "rms_norm_ alpha")?;
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        let alpha_data = alpha.storage()?;
        B::rms_norm(&mut *dst, &*src_data, &*alpha_data, dim_m1, d, eps)?;
        Ok(())
    }

    pub fn layer_norm_(&self, src: &Self, weight: &Self, bias: &Self, eps: f32) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "layer_norm_ src")?;
        if eps <= 0.0 {
            crate::bail!("layer_norm_ eps must be positive");
        }
        let dim_m1 = self.shape.dims().last().copied().unwrap_or(1);
        let d = self.elem_count() / dim_m1;
        let expected_shape = dim_m1.into();
        check_same_shape(&weight.shape, &expected_shape, "layer_norm_ weight")?;
        check_same_shape(&bias.shape, &expected_shape, "layer_norm_ bias")?;
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        let weight_data = weight.storage()?;
        let bias_data = bias.storage()?;
        B::layer_norm(&mut *dst, &*src_data, &*weight_data, &*bias_data, dim_m1, d, eps)?;
        Ok(())
    }

    pub fn matmul_(&self, lhs: &Self, rhs: &Self, rhs_t: bool) -> Result<()> {
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
        let dst_data = self.storage()?;
        let storage_len = B::storage_len(&*dst_data);
        drop(dst_data);

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

        let mut dst = self.storage_mut()?;
        let lhs_data = lhs.storage()?;
        let rhs_data = rhs.storage()?;
        B::gemm(
            &mut *dst,
            (&*lhs_data, 0),
            (&*rhs_data, 0),
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

    pub fn rope_(&self, src: &Self, cos: &Self, sin: &Self, pos: usize) -> Result<()> {
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
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        let cos_data = cos.storage()?;
        let sin_data = sin.storage()?;
        B::rope(&mut *dst, &*src_data, &*cos_data, &*sin_data, b, h, t, d, pos)?;
        Ok(())
    }

    pub fn rope_i_(&self, src: &Self, cos: &Self, sin: &Self, pos: usize) -> Result<()> {
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
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        let cos_data = cos.storage()?;
        let sin_data = sin.storage()?;
        B::rope_i(&mut *dst, &*src_data, &*cos_data, &*sin_data, b, h, t, d, pos)?;
        Ok(())
    }

    pub fn reduce_max_(&self, src: &Self, dim: usize) -> Result<()> {
        let src_dims = src.dims();
        let dim_size = src_dims[dim];
        let outer_size: usize = src_dims[..dim].iter().product::<usize>().max(1);
        let inner_size: usize = src_dims[dim + 1..].iter().product::<usize>().max(1);
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        B::reduce_max(&mut *dst, &*src_data, dim_size, outer_size, inner_size)?;
        Ok(())
    }

    pub fn reduce_min_(&self, src: &Self, dim: usize) -> Result<()> {
        let src_dims = src.dims();
        let dim_size = src_dims[dim];
        let outer_size: usize = src_dims[..dim].iter().product::<usize>().max(1);
        let inner_size: usize = src_dims[dim + 1..].iter().product::<usize>().max(1);
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        B::reduce_min(&mut *dst, &*src_data, dim_size, outer_size, inner_size)?;
        Ok(())
    }

    pub fn reduce_argmin_<U: crate::WithDTypeF>(
        dst: &Tensor<i64, B>,
        src: &Tensor<U, B>,
        dim: usize,
    ) -> Result<()> {
        let src_dims = src.dims();
        let dim_size = src_dims[dim];
        let outer_size: usize = src_dims[..dim].iter().product::<usize>().max(1);
        let inner_size: usize = src_dims[dim + 1..].iter().product::<usize>().max(1);
        let mut dst_data = dst.storage_mut()?;
        let src_data = src.storage()?;
        B::reduce_argmin(&mut *dst_data, &*src_data, dim_size, outer_size, inner_size)?;
        Ok(())
    }

    pub fn reduce_sum_(&self, src: &Self, dim: usize) -> Result<()> {
        let src_dims = src.dims();
        let dim_size = src_dims[dim];
        let outer_size: usize = src_dims[..dim].iter().product::<usize>().max(1);
        let inner_size: usize = src_dims[dim + 1..].iter().product::<usize>().max(1);
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        B::reduce_sum(&mut *dst, &*src_data, dim_size, outer_size, inner_size)?;
        Ok(())
    }

    pub fn broadcast_binary_(&self, lhs: &Self, rhs: &Self, op: BinaryOp) -> Result<()> {
        let dst_shape = self.dims().to_vec();
        let (lhs_strides, rhs_strides) =
            compute_broadcast_strides(&dst_shape, lhs.dims(), rhs.dims())?;
        let mut dst = self.storage_mut()?;
        let lhs_data = lhs.storage()?;
        let rhs_data = rhs.storage()?;
        B::broadcast_binary(
            &mut *dst,
            &*lhs_data,
            &*rhs_data,
            &dst_shape,
            &lhs_strides,
            &rhs_strides,
            op,
        )?;
        Ok(())
    }

    pub fn broadcast_add_(&self, lhs: &Self, rhs: &Self) -> Result<()> {
        self.broadcast_binary_(lhs, rhs, BinaryOp::Add)
    }

    pub fn broadcast_sub_(&self, lhs: &Self, rhs: &Self) -> Result<()> {
        self.broadcast_binary_(lhs, rhs, BinaryOp::Sub)
    }

    pub fn broadcast_mul_(&self, lhs: &Self, rhs: &Self) -> Result<()> {
        self.broadcast_binary_(lhs, rhs, BinaryOp::Mul)
    }

    pub fn broadcast_div_(&self, lhs: &Self, rhs: &Self) -> Result<()> {
        self.broadcast_binary_(lhs, rhs, BinaryOp::Div)
    }

    pub fn broadcast_maximum_(&self, lhs: &Self, rhs: &Self) -> Result<()> {
        self.broadcast_binary_(lhs, rhs, BinaryOp::Maximum)
    }

    pub fn broadcast_minimum_(&self, lhs: &Self, rhs: &Self) -> Result<()> {
        self.broadcast_binary_(lhs, rhs, BinaryOp::Minimum)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_(
        &self,
        src: &Self,
        kernel: &Self,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<()> {
        let src_dims = src.dims();
        let kernel_dims = kernel.dims();
        if src_dims.len() != 3 {
            crate::bail!(
                "conv1d input must be 3D (batch, in_channels, length), got {:?}",
                src.shape()
            );
        }
        if kernel_dims.len() != 3 {
            crate::bail!(
                "conv1d kernel must be 3D (out_channels, in_channels/groups, kernel_size), got {:?}",
                kernel.shape()
            );
        }

        let batch = src_dims[0];
        let in_channels = src_dims[1];
        let length = src_dims[2];
        let out_channels = kernel_dims[0];
        let kernel_size = kernel_dims[2];

        // Compute output length
        let out_length = (length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

        let dst_dims = self.dims();
        if dst_dims != [batch, out_channels, out_length] {
            crate::bail!(
                "conv1d output shape mismatch: expected {:?}, got {:?}",
                [batch, out_channels, out_length],
                dst_dims
            );
        }

        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        let kernel_data = kernel.storage()?;
        B::conv1d(
            &mut *dst,
            &*src_data,
            &*kernel_data,
            batch,
            in_channels,
            out_channels,
            length,
            out_length,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose1d_(
        &self,
        src: &Self,
        kernel: &Self,
        stride: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
    ) -> Result<()> {
        let src_dims = src.dims();
        let kernel_dims = kernel.dims();
        if src_dims.len() != 3 {
            crate::bail!(
                "conv_transpose1d input must be 3D (batch, in_channels, length), got {:?}",
                src.shape()
            );
        }
        if kernel_dims.len() != 3 {
            crate::bail!(
                "conv_transpose1d kernel must be 3D (in_channels, out_channels/groups, kernel_size), got {:?}",
                kernel.shape()
            );
        }

        let batch = src_dims[0];
        let in_channels = src_dims[1];
        let length = src_dims[2];
        let out_channels = kernel_dims[1] * groups;
        let kernel_size = kernel_dims[2];

        // Compute output length for transposed convolution
        // out_length = (length - 1) * stride - 2 * padding + kernel_size + output_padding
        let out_length = (length - 1) * stride + kernel_size + output_padding - 2 * padding;

        let dst_dims = self.dims();
        if dst_dims != [batch, out_channels, out_length] {
            crate::bail!(
                "conv_transpose1d output shape mismatch: expected {:?}, got {:?}",
                [batch, out_channels, out_length],
                dst_dims
            );
        }

        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        let kernel_data = kernel.storage()?;
        B::conv_transpose1d(
            &mut *dst,
            &*src_data,
            &*kernel_data,
            batch,
            in_channels,
            out_channels,
            length,
            out_length,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
        )
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
