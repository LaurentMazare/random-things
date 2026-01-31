pub mod backend;
pub mod cpu_backend;
pub mod display;
pub mod dtype;
pub mod error;
pub mod inplace_ops;
pub mod models;
pub mod nn;
pub mod ops;
pub mod shape;
pub mod tensor;
pub mod utils;

pub use backend::Backend;
pub use dtype::{DType, WithDType, WithDTypeF};
pub use error::{Error, Result};
pub use shape::{D, Dim, Shape};
pub use tensor::Tensor;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CpuDevice;
pub type CpuTensor<T> = Tensor<T, CpuDevice>;

pub const CPU: CpuDevice = CpuDevice;

pub(crate) use inplace_ops::{BinaryOp, UnaryOp};
