use crate::{Result, Tensor, WithDTypeF};

pub struct RmsNorm<T: WithDTypeF> {
    weight: Tensor<T>,
    eps: f32,
}

impl<T: WithDTypeF> RmsNorm<T> {
    pub fn new(weight: Tensor<T>, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        x.rms_norm(&self.weight, self.eps)
    }
}

pub struct Linear<T: WithDTypeF> {
    weight: Tensor<T>,
}

impl<T: WithDTypeF> Linear<T> {
    pub fn new(weight: Tensor<T>) -> Self {
        Self { weight }
    }

    pub fn forward(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        // weight: (out_features, in_features)
        // x: (..., in_features)
        // output: (..., out_features)
        x.matmul_t(&self.weight)
    }
}
