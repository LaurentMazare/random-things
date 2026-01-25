use crate::{Backend, Result, Tensor, WithDTypeF};

pub struct RmsNorm<T: WithDTypeF, B: Backend> {
    weight: Tensor<T, B>,
    eps: f32,
}

impl<T: WithDTypeF, B: Backend> RmsNorm<T, B> {
    pub fn new(weight: Tensor<T, B>, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        x.rms_norm(&self.weight, self.eps)
    }
}

pub struct Linear<T: WithDTypeF, B: Backend> {
    weight: Tensor<T, B>,
}

impl<T: WithDTypeF, B: Backend> Linear<T, B> {
    pub fn new(weight: Tensor<T, B>) -> Self {
        Self { weight }
    }

    pub fn forward(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        // weight: (out_features, in_features)
        // x: (..., in_features)
        // output: (..., out_features)
        x.matmul_t(&self.weight)
    }
}
