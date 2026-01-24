pub mod backend;
pub mod cpu_backend;
pub mod dtype;
pub mod error;
pub mod inplace_ops;
pub mod llama;
pub mod nn;
pub mod ops;
pub mod shape;
pub mod tensor;
pub mod utils;

pub use backend::{Backend, BackendF};
pub use dtype::{DType, WithDType, WithDTypeF};
pub use error::{Error, Result};
pub use shape::{D, Dim, Shape};
pub use tensor::Tensor;

pub type CpuDevice = ();
pub type CpuBackend<T> = Vec<T>;
pub type CpuTensor<T> = Tensor<T, CpuBackend<T>>;
