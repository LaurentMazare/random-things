use crate::{Backend, BackendF, Dim, Error, Result, Tensor, WithDType, WithDTypeF};

fn check_same_shape<T: WithDType, B: Backend<T>>(
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

impl<T: WithDType, B: Backend<T>> Tensor<T, B> {
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

impl<T: WithDTypeF, B: BackendF<T>> Tensor<T, B> {
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

    /// Causal softmax for autoregressive attention.
    /// Input shape: (batch, heads, seq_q, seq_kv)
    /// q_offset: position of first query token (for cached generation)
    pub fn softmax_causal(&self, _q_offset: usize) -> Result<Self> {
        todo!()
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
}
