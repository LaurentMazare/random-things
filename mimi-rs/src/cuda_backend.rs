#![allow(unused)]
use crate::{BinaryOp, Result, UnaryOp, WithDType, WithDTypeF};
use cudarc::cublas::{Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceRepr, DeviceSlice,
    LaunchConfig,
};
use half::{bf16, f16};
use std::sync::Arc;

#[derive(Clone)]
pub struct Device {
    cuda: Arc<CudaContext>,
    default_stream: Arc<CudaStream>,
    blas: Arc<cudarc::cublas::CudaBlas>,
}

impl Device {
    pub fn new(ordinal: usize) -> Result<Self> {
        let cuda = cudarc::driver::CudaContext::new(ordinal)?;
        let default_stream = cuda.default_stream();
        let blas = cudarc::cublas::CudaBlas::new(default_stream.clone())?;
        Ok(Self { cuda, default_stream, blas: Arc::new(blas) })
    }
}

impl crate::Backend for Device {
    type Storage<T: WithDType> = CudaSlice<T>;

    fn storage_len<T: WithDType>(storage: &Self::Storage<T>) -> usize {
        storage.len()
    }

    unsafe fn alloc_uninit<T: WithDType>(len: usize, dev: &Self) -> Result<Self::Storage<T>> {
        crate::bail!("not implemented yet")
    }

    fn from_vec<T: WithDType>(v: Vec<T>, dev: &Self) -> Result<Self::Storage<T>> {
        crate::bail!("not implemented yet")
    }

    fn fill<T: WithDType>(dst: &mut Self::Storage<T>, elem: T, len: usize) -> Result<()> {
        crate::bail!("not implemented yet")
    }

    fn copy<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()> {
        crate::bail!("not implemented yet")
    }

    fn data<T: WithDType>(src: &Self::Storage<T>, len: usize) -> Result<std::borrow::Cow<'_, [T]>> {
        crate::bail!("not implemented yet")
    }

    fn inplace_unary<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        len: usize,
        op: UnaryOp,
    ) -> Result<()> {
        crate::bail!("not implemented yet")
    }

    fn bin_assign<T: WithDType>(
        dst: &mut Self::Storage<T>,
        s: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()> {
        crate::bail!("not implemented yet")
    }

    fn unary<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
        op: UnaryOp,
    ) -> Result<()> {
        crate::bail!("not implemented yet")
    }

    fn binary<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()> {
        crate::bail!("not implemented yet")
    }

    fn scale<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        v: T,
        len: usize,
    ) -> Result<()> {
        crate::bail!("not implemented yet")
    }

    fn transpose<T: WithDType>(
        dst: &mut Self::Storage<T>,
        s: &Self::Storage<T>,
        dim1: usize,
        dim2: usize,
        dims: &[usize],
    ) -> Result<()> {
        crate::bail!("not implemented yet")
    }

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
        crate::bail!("not implemented yet")
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
        crate::bail!("not implemented yet")
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
        crate::bail!("not implemented yet")
    }

    fn gemm<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: (&Self::Storage<T>, usize),
        rhs: (&Self::Storage<T>, usize),
        m: usize,
        n: usize,
        k: usize,
        lhs_b: usize,
        b_stride: usize,
        dst_strides: (usize, usize),
        lhs_strides: (usize, usize),
        rhs_strides: (usize, usize),
    ) -> Result<()> {
        crate::bail!("not implemented yet")
    }

    fn index_select<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        ids: &[u32],
        dim: usize,
        dims: &[usize],
    ) -> Result<()> {
        crate::bail!("not implemented yet")
    }

    fn apply_causality_mask<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        bh: usize,
        t1: usize,
        t2: usize,
        offset: usize,
    ) -> Result<()> {
        crate::bail!("not implemented yet")
    }

    fn softmax<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
    ) -> Result<()> {
        crate::bail!("not implemented yet")
    }

    fn rms_norm<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        alpha: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
        eps: f32,
    ) -> Result<()> {
        crate::bail!("not implemented yet")
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
        crate::bail!("not implemented yet")
    }

    fn reduce_max<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        crate::bail!("not implemented yet")
    }

    fn reduce_min<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        crate::bail!("not implemented yet")
    }

    fn reduce_argmin<T: WithDTypeF>(
        dst: &mut Self::Storage<i64>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        crate::bail!("not implemented yet")
    }

    fn reduce_sum<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        crate::bail!("not implemented yet")
    }

    fn broadcast_binary<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        dst_shape: &[usize],
        lhs_strides: &[usize],
        rhs_strides: &[usize],
        op: BinaryOp,
    ) -> Result<()> {
        crate::bail!("not implemented yet")
    }

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
        crate::bail!("not implemented yet")
    }

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
        output_padding: usize,
        groups: usize,
    ) -> Result<()> {
        crate::bail!("not implemented yet")
    }
}
