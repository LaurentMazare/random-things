#![allow(unused)]
#![allow(clippy::too_many_arguments)]
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

/// Implementation of GEMM using cuBLAS for f32.
fn gemm_f32(
    dst: &mut Storage<f32>,
    lhs: (&Storage<f32>, usize),
    rhs: (&Storage<f32>, usize),
    m: usize,
    n: usize,
    k: usize,
    lhs_b: usize,
    b_stride: usize,
    (_dst_cs, dst_rs): (usize, usize),
    (lhs_m1, lhs_m2): (usize, usize),
    (rhs_m1, rhs_m2): (usize, usize),
) -> Result<()> {
    use cudarc::cublas::sys::cublasOperation_t;

    // Determine transposition and leading dimension for rhs (A in cuBLAS terms)
    let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        crate::bail!("non-contiguous matmul rhs m:{m} n:{n} k:{k} strides:({rhs_m1}, {rhs_m2})")
    };

    // Determine transposition and leading dimension for lhs (B in cuBLAS terms)
    let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        crate::bail!("non-contiguous matmul lhs m:{m} n:{n} k:{k} strides:({lhs_m1}, {lhs_m2})")
    };

    let gemm = GemmConfig {
        alpha: 1.0f32,
        beta: 0.0f32,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda,
        ldb,
        ldc: dst_rs as i32,
        transa,
        transb,
    };

    let cfg = StridedBatchedConfig {
        batch_size: lhs_b as i32,
        gemm,
        stride_a: b_stride as i64,
        stride_b: (m * k) as i64,
        stride_c: (m * n) as i64,
    };

    let lhs_view = lhs.0.data.slice(lhs.1..);
    let rhs_view = rhs.0.data.slice(rhs.1..);

    unsafe {
        dst.device.blas.gemm_strided_batched(cfg, &rhs_view, &lhs_view, &mut dst.data)?;
    }

    Ok(())
}

/// Implementation of GEMM using cuBLAS for f16.
fn gemm_f16(
    dst: &mut Storage<f16>,
    lhs: (&Storage<f16>, usize),
    rhs: (&Storage<f16>, usize),
    m: usize,
    n: usize,
    k: usize,
    lhs_b: usize,
    b_stride: usize,
    (_dst_cs, dst_rs): (usize, usize),
    (lhs_m1, lhs_m2): (usize, usize),
    (rhs_m1, rhs_m2): (usize, usize),
) -> Result<()> {
    use cudarc::cublas::sys::cublasOperation_t;

    let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        crate::bail!("non-contiguous matmul rhs m:{m} n:{n} k:{k} strides:({rhs_m1}, {rhs_m2})")
    };

    let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        crate::bail!("non-contiguous matmul lhs m:{m} n:{n} k:{k} strides:({lhs_m1}, {lhs_m2})")
    };

    let gemm = GemmConfig {
        alpha: f16::ONE,
        beta: f16::ZERO,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda,
        ldb,
        ldc: dst_rs as i32,
        transa,
        transb,
    };

    let cfg = StridedBatchedConfig {
        batch_size: lhs_b as i32,
        gemm,
        stride_a: b_stride as i64,
        stride_b: (m * k) as i64,
        stride_c: (m * n) as i64,
    };

    let lhs_view = lhs.0.data.slice(lhs.1..);
    let rhs_view = rhs.0.data.slice(rhs.1..);

    unsafe {
        dst.device.blas.gemm_strided_batched(cfg, &rhs_view, &lhs_view, &mut dst.data)?;
    }

    Ok(())
}

/// Implementation of GEMM using cuBLAS for bf16.
fn gemm_bf16(
    dst: &mut Storage<bf16>,
    lhs: (&Storage<bf16>, usize),
    rhs: (&Storage<bf16>, usize),
    m: usize,
    n: usize,
    k: usize,
    lhs_b: usize,
    b_stride: usize,
    (_dst_cs, dst_rs): (usize, usize),
    (lhs_m1, lhs_m2): (usize, usize),
    (rhs_m1, rhs_m2): (usize, usize),
) -> Result<()> {
    use cudarc::cublas::sys::cublasOperation_t;

    let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        crate::bail!("non-contiguous matmul rhs m:{m} n:{n} k:{k} strides:({rhs_m1}, {rhs_m2})")
    };

    let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        crate::bail!("non-contiguous matmul lhs m:{m} n:{n} k:{k} strides:({lhs_m1}, {lhs_m2})")
    };

    let gemm = GemmConfig {
        alpha: bf16::ONE,
        beta: bf16::ZERO,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda,
        ldb,
        ldc: dst_rs as i32,
        transa,
        transb,
    };

    let cfg = StridedBatchedConfig {
        batch_size: lhs_b as i32,
        gemm,
        stride_a: b_stride as i64,
        stride_b: (m * k) as i64,
        stride_c: (m * n) as i64,
    };

    let lhs_view = lhs.0.data.slice(lhs.1..);
    let rhs_view = rhs.0.data.slice(rhs.1..);

    unsafe {
        dst.device.blas.gemm_strided_batched(cfg, &rhs_view, &lhs_view, &mut dst.data)?;
    }

    Ok(())
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
        let (kname, alpha) = match op {
            UnaryOp::Cos => (kernel_name::<T>("inplace_cos"), None),
            UnaryOp::Sin => (kernel_name::<T>("inplace_sin"), None),
            UnaryOp::Sqr => (kernel_name::<T>("inplace_sqr"), None),
            UnaryOp::Sqrt => (kernel_name::<T>("inplace_sqrt"), None),
            UnaryOp::Abs => (kernel_name::<T>("inplace_abs"), None),
            UnaryOp::GeluErf => (kernel_name::<T>("inplace_gelu_erf"), None),
            UnaryOp::Elu { alpha } => (kernel_name::<T>("inplace_elu"), Some(alpha)),
            UnaryOp::Relu => (kernel_name::<T>("inplace_relu"), None),
            UnaryOp::Silu => (kernel_name::<T>("inplace_silu"), None),
            UnaryOp::Tanh => (kernel_name::<T>("inplace_tanh"), None),
            UnaryOp::Sigmoid => (kernel_name::<T>("inplace_sigmoid"), None),
        };
        let func = dst.device.get_func(&kname, crate::cuda_kernels::ARITHMETIC)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&len);
        launch_args.arg(&mut dst.data);
        if let Some(ref alpha) = alpha {
            launch_args.arg(alpha);
        }
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn bin_assign<T: WithDType>(
        dst: &mut Self::Storage<T>,
        s: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()> {
        let kname = match op {
            BinaryOp::Add => kernel_name::<T>("assign_add"),
            BinaryOp::Sub => kernel_name::<T>("assign_sub"),
            BinaryOp::Mul => kernel_name::<T>("assign_mul"),
            BinaryOp::Div => kernel_name::<T>("assign_div"),
            BinaryOp::Maximum => kernel_name::<T>("assign_maximum"),
            BinaryOp::Minimum => kernel_name::<T>("assign_minimum"),
        };
        let func = dst.device.get_func(&kname, crate::cuda_kernels::ARITHMETIC)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&len);
        launch_args.arg(&s.data);
        launch_args.arg(&mut dst.data);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn unary<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
        op: UnaryOp,
    ) -> Result<()> {
        let (kname, alpha) = match op {
            UnaryOp::Cos => (kernel_name::<T>("unary_cos"), None),
            UnaryOp::Sin => (kernel_name::<T>("unary_sin"), None),
            UnaryOp::Sqr => (kernel_name::<T>("unary_sqr"), None),
            UnaryOp::Sqrt => (kernel_name::<T>("unary_sqrt"), None),
            UnaryOp::Abs => (kernel_name::<T>("unary_abs"), None),
            UnaryOp::GeluErf => (kernel_name::<T>("unary_gelu_erf"), None),
            UnaryOp::Elu { alpha } => (kernel_name::<T>("unary_elu"), Some(alpha)),
            UnaryOp::Relu => (kernel_name::<T>("unary_relu"), None),
            UnaryOp::Silu => (kernel_name::<T>("unary_silu"), None),
            UnaryOp::Tanh => (kernel_name::<T>("unary_tanh"), None),
            UnaryOp::Sigmoid => (kernel_name::<T>("unary_sigmoid"), None),
        };
        let func = dst.device.get_func(&kname, crate::cuda_kernels::ARITHMETIC)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&len);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        if let Some(ref alpha) = alpha {
            launch_args.arg(alpha);
        }
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn binary<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()> {
        let kname = match op {
            BinaryOp::Add => kernel_name::<T>("binary_add"),
            BinaryOp::Sub => kernel_name::<T>("binary_sub"),
            BinaryOp::Mul => kernel_name::<T>("binary_mul"),
            BinaryOp::Div => kernel_name::<T>("binary_div"),
            BinaryOp::Maximum => kernel_name::<T>("binary_maximum"),
            BinaryOp::Minimum => kernel_name::<T>("binary_minimum"),
        };
        let func = dst.device.get_func(&kname, crate::cuda_kernels::ARITHMETIC)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&len);
        launch_args.arg(&lhs.data);
        launch_args.arg(&rhs.data);
        launch_args.arg(&mut dst.data);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn scale<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        v: T,
        len: usize,
    ) -> Result<()> {
        let kname = kernel_name::<T>("scale");
        let func = dst.device.get_func(&kname, crate::cuda_kernels::ARITHMETIC)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&len);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&v);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn transpose<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim1: usize,
        dim2: usize,
        dims: &[usize],
    ) -> Result<()> {
        let numel: usize = dims.iter().product();
        if dim1 == dim2 || dims.iter().filter(|v| **v != 1).count() <= 1 {
            // Simple copy when no real transpose needed
            let src_slice = src.data.slice(..numel);
            let mut dst_slice = dst.data.slice_mut(..numel);
            dst.device.stream.memcpy_dtod(&src_slice, &mut dst_slice)?;
        } else {
            let (dim1, dim2) = (usize::min(dim1, dim2), usize::max(dim1, dim2));
            let d_i: usize = dims[..dim1].iter().product();
            let d_j: usize = dims[dim1 + 1..dim2].iter().product();
            let d_k: usize = dims[(dim2 + 1)..].iter().product();
            let d1 = dims[dim1] as u32;
            let d2 = dims[dim2] as u32;
            let d_i = d_i as u32;
            let d_j = d_j as u32;
            let d_k = d_k as u32;

            let kname = kernel_name::<T>("transpose");
            let func = dst.device.get_func(&kname, crate::cuda_kernels::LAYOUT)?;
            let cfg = LaunchConfig::for_num_elems(numel as u32);
            let mut launch_args = dst.device.stream.launch_builder(&func);
            launch_args.arg(&numel);
            launch_args.arg(&d1);
            launch_args.arg(&d2);
            launch_args.arg(&d_i);
            launch_args.arg(&d_j);
            launch_args.arg(&d_k);
            launch_args.arg(&src.data);
            launch_args.arg(&mut dst.data);
            unsafe { launch_args.launch(cfg) }?;
        }
        Ok(())
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
        // Dispatch to type-specific GEMM implementations
        // We use pointer casting since we know the exact type from DTYPE
        match T::DTYPE {
            DType::F32 => {
                // SAFETY: T::DTYPE == F32 guarantees T is f32
                let dst = unsafe { &mut *(dst as *mut Storage<T> as *mut Storage<f32>) };
                let lhs_storage = unsafe { &*(lhs.0 as *const Storage<T> as *const Storage<f32>) };
                let rhs_storage = unsafe { &*(rhs.0 as *const Storage<T> as *const Storage<f32>) };
                gemm_f32(
                    dst,
                    (lhs_storage, lhs.1),
                    (rhs_storage, rhs.1),
                    m,
                    n,
                    k,
                    lhs_b,
                    b_stride,
                    dst_strides,
                    lhs_strides,
                    rhs_strides,
                )
            }
            DType::F16 => {
                let dst = unsafe { &mut *(dst as *mut Storage<T> as *mut Storage<f16>) };
                let lhs_storage = unsafe { &*(lhs.0 as *const Storage<T> as *const Storage<f16>) };
                let rhs_storage = unsafe { &*(rhs.0 as *const Storage<T> as *const Storage<f16>) };
                gemm_f16(
                    dst,
                    (lhs_storage, lhs.1),
                    (rhs_storage, rhs.1),
                    m,
                    n,
                    k,
                    lhs_b,
                    b_stride,
                    dst_strides,
                    lhs_strides,
                    rhs_strides,
                )
            }
            DType::BF16 => {
                let dst = unsafe { &mut *(dst as *mut Storage<T> as *mut Storage<bf16>) };
                let lhs_storage = unsafe { &*(lhs.0 as *const Storage<T> as *const Storage<bf16>) };
                let rhs_storage = unsafe { &*(rhs.0 as *const Storage<T> as *const Storage<bf16>) };
                gemm_bf16(
                    dst,
                    (lhs_storage, lhs.1),
                    (rhs_storage, rhs.1),
                    m,
                    n,
                    k,
                    lhs_b,
                    b_stride,
                    dst_strides,
                    lhs_strides,
                    rhs_strides,
                )
            }
            _ => crate::bail!("GEMM not supported for dtype {:?}", T::DTYPE),
        }
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
        // dim_m1 is ncols (last dimension), d is nrows
        let ncols = dim_m1 as i32;
        let nrows = d as u32;

        let kname = kernel_name::<T>("softmax");
        let func = dst.device.get_func(&kname, crate::cuda_kernels::REDUCE)?;

        // Kernel uses: row = blockDim.x*blockIdx.x + threadIdx.x, tid = threadIdx.y
        // One row per block, 32 threads per row for warp-based reduction
        let block_dim = (1, 32, 1);
        let grid_dim = (nrows, 1, 1);
        let cfg = LaunchConfig { block_dim, grid_dim, shared_mem_bytes: 0 };

        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&ncols);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn rms_norm<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        alpha: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
        eps: f32,
    ) -> Result<()> {
        // dim_m1 is ncols (last dimension), d is nrows
        let ncols = dim_m1 as i32;
        let nrows = d;

        let kname = kernel_name::<T>("rmsnorm");
        let func = dst.device.get_func(&kname, crate::cuda_kernels::REDUCE)?;

        // Kernel uses: row = blockIdx.x*blockDim.y + threadIdx.y, tid = threadIdx.x
        // blockDim.x threads collaborate on each row
        const WARP_SIZE: u32 = 32;
        let block_size = WARP_SIZE;
        let block_size_i32 = block_size as i32;
        let rows_per_block = 4u32;
        let block_dim = (block_size, rows_per_block, 1);
        let grid_dim = (nrows.div_ceil(rows_per_block as usize) as u32, 1, 1);
        let cfg = LaunchConfig { block_dim, grid_dim, shared_mem_bytes: 0 };

        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&alpha.data);
        launch_args.arg(&ncols);
        launch_args.arg(&block_size_i32);
        launch_args.arg(&eps);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
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
        // dim_m1 is ncols (last dimension), d is nrows
        let ncols = dim_m1 as i32;
        let nrows = d;

        let kname = kernel_name::<T>("layernorm");
        let func = dst.device.get_func(&kname, crate::cuda_kernels::REDUCE)?;

        // Kernel uses: row = blockIdx.x*blockDim.y + threadIdx.y, tid = threadIdx.x
        // blockDim.x threads collaborate on each row
        const WARP_SIZE: u32 = 32;
        let block_size = WARP_SIZE;
        let block_size_i32 = block_size as i32;
        let rows_per_block = 4u32;
        let block_dim = (block_size, rows_per_block, 1);
        let grid_dim = (nrows.div_ceil(rows_per_block as usize) as u32, 1, 1);
        let cfg = LaunchConfig { block_dim, grid_dim, shared_mem_bytes: 0 };

        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&weight.data);
        launch_args.arg(&bias.data);
        launch_args.arg(&ncols);
        launch_args.arg(&block_size_i32);
        launch_args.arg(&eps);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
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
