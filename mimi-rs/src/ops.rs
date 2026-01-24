use crate::shape::Dim;
use crate::{Error, Result};
use crate::{Tensor, WithDType, WithDTypeF};

impl<T: WithDType> Tensor<T> {
    pub fn add(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        if self.shape != other.shape {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape.clone(),
                rhs: other.shape.clone(),
                op: "add",
            }
            .bt());
        }
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone()) };
        result.add_(self, other)?;
        Ok(result)
    }

    pub fn mul(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        if self.shape != other.shape {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape.clone(),
                rhs: other.shape.clone(),
                op: "mul",
            }
            .bt());
        }
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone()) };
        result.mul_(self, other)?;
        Ok(result)
    }

    pub fn transpose<D1: Dim, D2: Dim>(&self, dim1: D1, dim2: D2) -> Result<Tensor<T>> {
        let dim1 = dim1.to_index(self.shape(), "transpose dim1")?;
        let dim2 = dim2.to_index(self.shape(), "transpose dim2")?;
        let mut new_dims = self.dims().to_vec();
        new_dims.swap(dim1, dim2);
        let mut result = unsafe { Tensor::alloc_uninit(new_dims.into()) };
        result.transpose_(self, dim1, dim2)?;
        Ok(result)
    }

    pub fn copy(&self) -> Result<Tensor<T>> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone()) };
        result.copy_(self)?;
        Ok(result)
    }

    pub fn full_like(&self, value: T) -> Result<Tensor<T>> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone()) };
        result.fill_(value)?;
        Ok(result)
    }

    pub fn scale(&self, m: T) -> Result<Tensor<T>> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone()) };
        for (dst, src) in result.data.iter_mut().zip(self.data.iter()) {
            *dst = *src * m;
        }
        Ok(result)
    }
}

impl<T: WithDTypeF> Tensor<T> {
    pub fn cos(&self) -> Result<Tensor<T>> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone()) };
        result.cos_(self)?;
        Ok(result)
    }

    pub fn sin(&self) -> Result<Tensor<T>> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone()) };
        result.sin_(self)?;
        Ok(result)
    }

    pub fn exp(&self) -> Result<Tensor<T>> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone()) };
        result.exp_(self)?;
        Ok(result)
    }

    pub fn silu(&self) -> Result<Tensor<T>> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone()) };
        result.silu_(self)?;
        Ok(result)
    }

    pub fn softmax(&self) -> Result<Tensor<T>> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone()) };
        result.softmax_(self)?;
        Ok(result)
    }

    /// Causal softmax for autoregressive attention.
    /// Input shape: (batch, heads, seq_q, seq_kv)
    /// q_offset: position of first query token (for cached generation)
    pub fn softmax_causal(&self, q_offset: usize) -> Result<Tensor<T>> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone()) };
        result.softmax_causal_(self, q_offset)?;
        Ok(result)
    }

    pub fn rms_norm(&self, alpha: &Tensor<T>, eps: f32) -> Result<Tensor<T>> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone()) };
        result.rms_norm_(self, alpha, eps)?;
        Ok(result)
    }

    pub fn rope(&self, cos: &Tensor<T>, sin: &Tensor<T>, pos: usize) -> Result<Tensor<T>> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone()) };
        result.rope_(self, cos, sin, pos)?;
        Ok(result)
    }

    pub fn rope_i(&self, cos: &Tensor<T>, sin: &Tensor<T>, pos: usize) -> Result<Tensor<T>> {
        let mut result = unsafe { Tensor::alloc_uninit(self.shape.clone()) };
        result.rope_i_(self, cos, sin, pos)?;
        Ok(result)
    }

    fn matmul_with_t(&self, other: &Tensor<T>, rhs_t: bool) -> Result<Tensor<T>> {
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

        let mut result = unsafe { Tensor::alloc_uninit(target_shape.into()) };
        result.matmul_(self, other, rhs_t)?;
        Ok(result)
    }

    pub fn matmul(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        self.matmul_with_t(other, false)
    }

    pub fn matmul_t(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        self.matmul_with_t(other, true)
    }
}
