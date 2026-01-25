use crate::{Backend, Dim, Error, Result, Tensor, WithDType, WithDTypeF};

fn check_same_shape<T: WithDType, B: Backend>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    op: &'static str,
) -> Result<()> {
    if a.shape != b.shape {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: a.shape.clone(),
            rhs: b.shape.clone(),
            op,
        }
        .bt());
    }
    Ok(())
}

impl<T: WithDType, B: Backend> Tensor<T, B> {
    pub fn add(&self, other: &Self) -> Result<Self> {
        check_same_shape(self, other, "add")?;
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.add_(self, other)?;
        Ok(result)
    }

    pub fn mul(&self, other: &Self) -> Result<Self> {
        check_same_shape(self, other, "mul")?;
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.mul_(self, other)?;
        Ok(result)
    }

    pub fn transpose<D1: Dim, D2: Dim>(&self, dim1: D1, dim2: D2) -> Result<Self> {
        let dim1 = dim1.to_index(self.shape(), "transpose dim1")?;
        let dim2 = dim2.to_index(self.shape(), "transpose dim2")?;
        let mut new_dims = self.dims().to_vec();
        new_dims.swap(dim1, dim2);
        let mut result = unsafe { Tensor::alloc_uninit(new_dims.into(), self.device()) }?;
        result.transpose_(self, dim1, dim2)?;
        Ok(result)
    }

    pub fn copy(&self) -> Result<Self> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.copy_(self)?;
        Ok(result)
    }

    pub fn full_like(&self, value: T) -> Result<Self> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.fill_(value)?;
        Ok(result)
    }

    pub fn scale(&self, m: T) -> Result<Self> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.scale_(self, m)?;
        Ok(result)
    }
}

impl<T: WithDTypeF, B: Backend> Tensor<T, B> {
    pub fn cos(&self) -> Result<Self> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.cos_(self)?;
        Ok(result)
    }

    pub fn sin(&self) -> Result<Self> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.sin_(self)?;
        Ok(result)
    }

    pub fn silu(&self) -> Result<Self> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.silu_(self)?;
        Ok(result)
    }

    pub fn softmax(&self) -> Result<Self> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.softmax_(self)?;
        Ok(result)
    }

    /// Apply causality mask and return a new tensor.
    /// Shape: (batch * heads, seq_q, seq_kv) or (batch, heads, seq_q, seq_kv)
    /// Masks positions where key position > query position + offset (sets to -inf).
    /// offset: starting position of the first query token (for KV cache generation).
    pub fn apply_causality_mask(&self, offset: usize) -> Result<Self> {
        let mut result = self.copy()?;
        result.apply_causality_mask_(offset)?;
        Ok(result)
    }

    pub fn rms_norm(&self, alpha: &Self, eps: f32) -> Result<Self> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.rms_norm_(self, alpha, eps)?;
        Ok(result)
    }

    pub fn rope(&self, cos: &Self, sin: &Self, pos: usize) -> Result<Self> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.rope_(self, cos, sin, pos)?;
        Ok(result)
    }

    pub fn rope_i(&self, cos: &Self, sin: &Self, pos: usize) -> Result<Self> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.rope_i_(self, cos, sin, pos)?;
        Ok(result)
    }

    fn matmul_with_t(&self, other: &Self, rhs_t: bool) -> Result<Self> {
        if self.shape.rank() < 2 || other.shape.rank() < 2 {
            return Err(Error::MatmulShapeMismatch {
                lhs: self.shape.clone(),
                rhs: other.shape.clone(),
                msg: "matmul requires at least 2D tensors",
            }
            .bt());
        }

        let lhs_dims = self.dims();
        let rhs_dims = other.dims();

        // Get M, K from lhs (last two dims)
        let lhs_m = lhs_dims[lhs_dims.len() - 2];
        let lhs_k = lhs_dims[lhs_dims.len() - 1];

        // Get K, N from rhs (last two dims), accounting for transpose
        let (rhs_k, rhs_n) = if rhs_t {
            (rhs_dims[rhs_dims.len() - 1], rhs_dims[rhs_dims.len() - 2])
        } else {
            (rhs_dims[rhs_dims.len() - 2], rhs_dims[rhs_dims.len() - 1])
        };

        if lhs_k != rhs_k {
            return Err(Error::MatmulShapeMismatch {
                lhs: self.shape.clone(),
                rhs: other.shape.clone(),
                msg: "inner dimensions do not match in matmul",
            }
            .bt());
        }

        // Check batch dimensions are compatible
        // rhs can be 2D (no batch) which broadcasts to any lhs batch
        let lhs_batch = &lhs_dims[..lhs_dims.len() - 2];
        let rhs_batch = &rhs_dims[..rhs_dims.len() - 2];
        if !rhs_batch.is_empty() && lhs_batch != rhs_batch {
            return Err(Error::MatmulShapeMismatch {
                lhs: self.shape.clone(),
                rhs: other.shape.clone(),
                msg: "batch dimensions do not match in matmul",
            }
            .bt());
        }

        // Build output shape: lhs batch dims + [M, N]
        let mut target_shape = lhs_batch.to_vec();
        target_shape.push(lhs_m);
        target_shape.push(rhs_n);

        let dev = self.device();
        let mut result = unsafe { Self::alloc_uninit(target_shape.into(), dev) }?;
        result.matmul_(self, other, rhs_t)?;
        Ok(result)
    }

    pub fn matmul(&self, other: &Self) -> Result<Self> {
        self.matmul_with_t(other, false)
    }

    pub fn matmul_t(&self, other: &Self) -> Result<Self> {
        self.matmul_with_t(other, true)
    }

    // ========================================================================
    // Convolution operations (todo stubs)
    // ========================================================================

    /// 1D convolution.
    pub fn conv1d(
        &self,
        _kernel: &Self,
        _bias: Option<&Self>,
        _stride: usize,
        _padding: usize,
        _dilation: usize,
        _groups: usize,
    ) -> Result<Self> {
        todo!("conv1d")
    }

    /// 1D transposed convolution.
    pub fn conv_transpose1d(
        &self,
        _kernel: &Self,
        _bias: Option<&Self>,
        _stride: usize,
        _padding: usize,
        _output_padding: usize,
        _groups: usize,
    ) -> Result<Self> {
        todo!("conv_transpose1d")
    }

    // ========================================================================
    // Additional operations (todo stubs)
    // ========================================================================

    /// Element-wise square.
    pub fn sqr(&self) -> Result<Self> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.sqr_(self)?;
        Ok(result)
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Result<Self> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.sqrt_(self)?;
        Ok(result)
    }

    /// Element-wise absolute value.
    pub fn abs(&self) -> Result<Self> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.abs_(self)?;
        Ok(result)
    }

    /// Sum along dimensions, keeping the dimensions.
    pub fn sum_keepdim(&self, _dims: impl Into<Vec<usize>>) -> Result<Self> {
        todo!("sum_keepdim")
    }

    /// Maximum value along dimension.
    pub fn max(&self, _dim: impl Dim) -> Result<Self> {
        todo!("max")
    }

    /// Minimum value along dimension.
    pub fn min(&self, _dim: impl Dim) -> Result<Self> {
        todo!("min")
    }

    /// Argmin along dimension.
    /// Note: Returns indices encoded as the same type T for simplicity.
    /// A proper implementation would return integer indices.
    pub fn argmin(&self, _dim: impl Dim) -> Result<Self> {
        todo!("argmin")
    }

    /// Broadcast multiplication.
    pub fn broadcast_mul(&self, _other: &Self) -> Result<Self> {
        todo!("broadcast_mul")
    }

    /// Broadcast division.
    pub fn broadcast_div(&self, _other: &Self) -> Result<Self> {
        todo!("broadcast_div")
    }

    /// Broadcast addition.
    pub fn broadcast_add(&self, _other: &Self) -> Result<Self> {
        todo!("broadcast_add")
    }

    /// Broadcast subtraction.
    pub fn broadcast_sub(&self, _other: &Self) -> Result<Self> {
        todo!("broadcast_sub")
    }

    /// Flatten all dimensions into a single dimension.
    pub fn flatten_all(&self) -> Result<Self> {
        self.reshape(vec![self.elem_count()])
    }

    /// Flatten dimensions from start to end (inclusive) into a single dimension.
    pub fn flatten<D: Dim>(&self, start_dim: D, end_dim: D) -> Result<Self> {
        let start_dim = start_dim.to_index(self.shape(), "flatten start_dim")?;
        let end_dim = end_dim.to_index(self.shape(), "flatten end_dim")?;
        let dims = self.dims();
        if start_dim > end_dim {
            crate::bail!("flatten: start_dim {start_dim} > end_dim {end_dim}");
        }
        let flat_size: usize = dims[start_dim..=end_dim].iter().product();
        let mut new_dims = Vec::with_capacity(dims.len() - (end_dim - start_dim));
        new_dims.extend_from_slice(&dims[..start_dim]);
        new_dims.push(flat_size);
        new_dims.extend_from_slice(&dims[end_dim + 1..]);
        self.reshape(new_dims)
    }

    /// GELU activation with erf.
    pub fn gelu_erf(&self) -> Result<Self> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.gelu_erf_(self)?;
        Ok(result)
    }

    /// ELU activation.
    pub fn elu(&self, alpha: f32) -> Result<Self> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.elu_(self, alpha)?;
        Ok(result)
    }

    /// ReLU activation.
    pub fn relu(&self) -> Result<Self> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.relu_(self)?;
        Ok(result)
    }

    /// Tanh activation.
    pub fn tanh(&self) -> Result<Self> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.tanh_(self)?;
        Ok(result)
    }

    /// Sigmoid activation.
    pub fn sigmoid(&self) -> Result<Self> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.sigmoid_(self)?;
        Ok(result)
    }

    /// Expand tensor to a new shape (broadcasting).
    pub fn expand(&self, _shape: impl Into<crate::Shape>) -> Result<Self> {
        todo!("expand")
    }

    /// Repeat tensor along dimensions.
    pub fn repeat(&self, _repeats: impl Into<Vec<usize>>) -> Result<Self> {
        todo!("repeat")
    }

    /// Ensure tensor is contiguous in memory.
    pub fn contiguous(&self) -> Result<Self> {
        todo!("contiguous")
    }

    /// Where condition: select from self or other based on condition.
    /// The condition should be a tensor of the same shape where non-zero values
    /// select from self, and zero values select from other.
    pub fn where_cond(&self, _condition: &Self, _other: &Self) -> Result<Self> {
        todo!("where_cond")
    }

    /// Create a tensor of zeros with the same shape.
    pub fn zeros_like(&self) -> Result<Self> {
        Self::zeros(self.shape().clone(), self.device())
    }

    /// Transpose (swap last two dimensions).
    pub fn t(&self) -> Result<Self> {
        let rank = self.rank();
        if rank < 2 {
            crate::bail!("t requires at least 2 dimensions");
        }
        self.transpose(rank - 2, rank - 1)
    }

    /// Unsqueeze: add a dimension of size 1 at the given position.
    pub fn unsqueeze(&self, _dim: impl Dim) -> Result<Self> {
        todo!("unsqueeze")
    }

    /// Pad with zeros along a dimension.
    pub fn pad_with_zeros(&self, _dim: impl Dim, _left: usize, _right: usize) -> Result<Self> {
        todo!("pad_with_zeros")
    }

    /// Pad by replicating boundary values.
    pub fn pad_with_same(&self, _dim: impl Dim, _left: usize, _right: usize) -> Result<Self> {
        todo!("pad_with_same")
    }
}
