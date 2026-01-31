#![allow(unused)]
use crate::{BinaryOp, DType, Result, UnaryOp, WithDType, WithDTypeF};
use cudarc::cublas::{Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, DeviceRepr, DeviceSlice,
    LaunchConfig, PushKernelArg,
};
use half::{bf16, f16};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct Device {
    cuda: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: Arc<cudarc::cublas::CudaBlas>,
    /// Cache for loaded PTX modules
    modules: Arc<Mutex<HashMap<&'static str, Arc<cudarc::driver::CudaModule>>>>,
}

impl Device {
    pub fn new(ordinal: usize) -> Result<Self> {
        let cuda = cudarc::driver::CudaContext::new(ordinal)?;
        let stream = cuda.default_stream();
        let blas = cudarc::cublas::CudaBlas::new(stream.clone())?;
        Ok(Self {
            cuda,
            stream,
            blas: Arc::new(blas),
            modules: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    fn get_or_load_module(&self, ptx: &'static str) -> Result<Arc<cudarc::driver::CudaModule>> {
        let mut modules = self.modules.lock().unwrap();
        if let Some(module) = modules.get(ptx) {
            return Ok(module.clone());
        }
        let module = self.cuda.load_module(ptx.into())?;
        modules.insert(ptx, module.clone());
        Ok(module)
    }

    fn get_func(&self, name: &str, ptx: &'static str) -> Result<CudaFunction> {
        let module = self.get_or_load_module(ptx)?;
        let func = module.load_function(name)?;
        Ok(func)
    }
}

/// CUDA storage that holds both the device data and a reference to the device.
pub struct Storage<T: WithDType> {
    pub data: CudaSlice<T>,
    pub device: Device,
}

impl<T: WithDType> Storage<T> {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

fn kernel_name<T: WithDType>(base_name: &str) -> String {
    let dtype_str = match T::DTYPE {
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::F32 => "f32",
        DType::I64 => "i64",
        DType::U8 => "u8",
    };
    format!("{base_name}_{dtype_str}")
}

// Reduced precision settings
static MM_F16_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static MM_BF16_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static MM_F32_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn gemm_reduced_precision_f32() -> bool {
    MM_F32_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_gemm_reduced_precision_f32(b: bool) {
    MM_F32_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

pub fn gemm_reduced_precision_f16() -> bool {
    MM_F16_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_gemm_reduced_precision_f16(b: bool) {
    MM_F16_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

pub fn gemm_reduced_precision_bf16() -> bool {
    MM_BF16_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_gemm_reduced_precision_bf16(b: bool) {
    MM_BF16_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

impl crate::Backend for Device {
    type Storage<T: WithDType> = Storage<T>;

    fn storage_len<T: WithDType>(storage: &Self::Storage<T>) -> usize {
        storage.len()
    }

    unsafe fn alloc_uninit<T: WithDType>(len: usize, dev: &Self) -> Result<Self::Storage<T>> {
        let data = unsafe { dev.stream.alloc::<T>(len) }?;
        Ok(Storage { data, device: dev.clone() })
    }

    fn from_vec<T: WithDType>(v: Vec<T>, dev: &Self) -> Result<Self::Storage<T>> {
        let data = dev.stream.clone_htod(&v)?;
        Ok(Storage { data, device: dev.clone() })
    }

    fn fill<T: WithDType>(dst: &mut Self::Storage<T>, elem: T, len: usize) -> Result<()> {
        let kname = kernel_name::<T>("fill");
        let func = dst.device.get_func(&kname, crate::cuda_kernels::FILL)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&elem);
        launch_args.arg(&len);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn copy<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()> {
        let src_slice = src.data.slice(..len);
        let mut dst_slice = dst.data.slice_mut(..len);
        dst.device.stream.memcpy_dtod(&src_slice, &mut dst_slice)?;
        Ok(())
    }

    fn data<T: WithDType>(src: &Self::Storage<T>, len: usize) -> Result<std::borrow::Cow<'_, [T]>> {
        let data = src.device.stream.clone_dtoh(&src.data)?;
        Ok(std::borrow::Cow::Owned(data))
    }

    fn inplace_unary<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        len: usize,
        op: UnaryOp,
    ) -> Result<()> {
        crate::bail!("inplace_unary not implemented yet")
    }

    fn bin_assign<T: WithDType>(
        dst: &mut Self::Storage<T>,
        s: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()> {
        crate::bail!("bin_assign not implemented yet")
    }

    fn unary<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
        op: UnaryOp,
    ) -> Result<()> {
        crate::bail!("unary not implemented yet")
    }

    fn binary<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()> {
        crate::bail!("binary not implemented yet")
    }

    fn scale<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        v: T,
        len: usize,
    ) -> Result<()> {
        crate::bail!("scale not implemented yet")
    }

    fn transpose<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim1: usize,
        dim2: usize,
        dims: &[usize],
    ) -> Result<()> {
        crate::bail!("transpose not implemented yet")
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
        crate::bail!("copy2d not implemented yet")
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
        _pos: usize,
    ) -> Result<()> {
        let kname = kernel_name::<T>("rope");
        let func = dst.device.get_func(&kname, crate::cuda_kernels::ROPE)?;
        let bh = (b * h) as u32;
        let td = (t * d) as u32;
        let d = d as u32;
        // The kernel processes bh * td / 2 elements (each thread handles 2 elements)
        let cfg = LaunchConfig::for_num_elems(bh * td / 2);
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&cos.data);
        launch_args.arg(&sin.data);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&bh);
        launch_args.arg(&td);
        launch_args.arg(&d);
        unsafe { launch_args.launch(cfg) }?;
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
        _pos: usize,
    ) -> Result<()> {
        let kname = kernel_name::<T>("rope_i");
        let func = dst.device.get_func(&kname, crate::cuda_kernels::ROPE)?;
        let bh = (b * h) as u32;
        let td = (t * d) as u32;
        // The kernel processes bh * td / 2 elements (each thread handles 2 elements)
        let cfg = LaunchConfig::for_num_elems(bh * td / 2);
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&cos.data);
        launch_args.arg(&sin.data);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&bh);
        launch_args.arg(&td);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
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
        crate::bail!("gemm not implemented yet")
    }

    fn index_select<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        ids: &[u32],
        dim: usize,
        dims: &[usize],
    ) -> Result<()> {
        crate::bail!("index_select not implemented yet")
    }

    fn apply_causality_mask<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        bh: usize,
        t1: usize,
        t2: usize,
        offset: usize,
    ) -> Result<()> {
        crate::bail!("apply_causality_mask not implemented yet")
    }

    fn softmax<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
    ) -> Result<()> {
        crate::bail!("softmax not implemented yet")
    }

    fn rms_norm<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        alpha: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
        eps: f32,
    ) -> Result<()> {
        crate::bail!("rms_norm not implemented yet")
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
        crate::bail!("layer_norm not implemented yet")
    }

    fn reduce_max<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        crate::bail!("reduce_max not implemented yet")
    }

    fn reduce_min<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        crate::bail!("reduce_min not implemented yet")
    }

    fn reduce_argmin<T: WithDTypeF>(
        dst: &mut Self::Storage<i64>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        crate::bail!("reduce_argmin not implemented yet")
    }

    fn reduce_sum<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        crate::bail!("reduce_sum not implemented yet")
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
        crate::bail!("broadcast_binary not implemented yet")
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
        crate::bail!("conv1d not implemented yet")
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
        crate::bail!("conv_transpose1d not implemented yet")
    }
}
