use half::{bf16, f16};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F16,
    BF16,
    F32,
}

pub trait WithDType:
    Sized + Copy + num_traits::NumAssign + 'static + Clone + Send + Sync + std::fmt::Debug
{
    const DTYPE: DType;
    const BYTE_SIZE: usize;
    fn from_be_bytes(dst: &mut [Self], src: &[u8]);
    /// Convert a little-endian byte slice to a Vec of Self.
    /// This handles alignment safely by reading bytes individually.
    fn vec_from_le_bytes(src: &[u8]) -> Vec<Self>;
}

pub trait WithDTypeF: WithDType + num_traits::Float {
    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
}

impl WithDType for f16 {
    const DTYPE: DType = DType::F16;
    const BYTE_SIZE: usize = 2;

    fn from_be_bytes(dst: &mut [Self], src: &[u8]) {
        for (i, v) in dst.iter_mut().enumerate() {
            *v = f16::from_bits(u16::from_be_bytes([src[2 * i + 1], src[2 * i]]))
        }
    }

    fn vec_from_le_bytes(src: &[u8]) -> Vec<Self> {
        let len = src.len() / Self::BYTE_SIZE;
        let mut dst: Vec<Self> = Vec::with_capacity(len);
        // SAFETY: We allocate `len` elements, initialize all bytes via copy, then set length.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                dst.spare_capacity_mut().as_mut_ptr().cast::<u8>(),
                len * Self::BYTE_SIZE,
            );
            dst.set_len(len);
        }
        dst
    }
}

impl WithDTypeF for f16 {
    fn to_f32(self) -> f32 {
        f16::to_f32(self)
    }

    fn from_f32(v: f32) -> Self {
        f16::from_f32(v)
    }
}

impl WithDType for bf16 {
    const DTYPE: DType = DType::BF16;
    const BYTE_SIZE: usize = 2;

    fn from_be_bytes(dst: &mut [Self], src: &[u8]) {
        for (i, v) in dst.iter_mut().enumerate() {
            *v = bf16::from_bits(u16::from_be_bytes([src[2 * i + 1], src[2 * i]]))
        }
    }

    fn vec_from_le_bytes(src: &[u8]) -> Vec<Self> {
        let len = src.len() / Self::BYTE_SIZE;
        let mut dst: Vec<Self> = Vec::with_capacity(len);
        // SAFETY: We allocate `len` elements, initialize all bytes via copy, then set length.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                dst.spare_capacity_mut().as_mut_ptr().cast::<u8>(),
                len * Self::BYTE_SIZE,
            );
            dst.set_len(len);
        }
        dst
    }
}

impl WithDTypeF for bf16 {
    fn to_f32(self) -> f32 {
        bf16::to_f32(self)
    }

    fn from_f32(v: f32) -> Self {
        bf16::from_f32(v)
    }
}

impl WithDType for f32 {
    const DTYPE: DType = DType::F32;
    const BYTE_SIZE: usize = 4;

    fn from_be_bytes(dst: &mut [Self], src: &[u8]) {
        for (i, v) in dst.iter_mut().enumerate() {
            *v = f32::from_bits(u32::from_be_bytes([
                src[4 * i + 3],
                src[4 * i + 2],
                src[4 * i + 1],
                src[4 * i],
            ]))
        }
    }

    fn vec_from_le_bytes(src: &[u8]) -> Vec<Self> {
        let len = src.len() / Self::BYTE_SIZE;
        let mut dst: Vec<Self> = Vec::with_capacity(len);
        // SAFETY: We allocate `len` elements, initialize all bytes via copy, then set length.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                dst.spare_capacity_mut().as_mut_ptr().cast::<u8>(),
                len * Self::BYTE_SIZE,
            );
            dst.set_len(len);
        }
        dst
    }
}

impl WithDTypeF for f32 {
    fn to_f32(self) -> f32 {
        self
    }

    fn from_f32(v: f32) -> Self {
        v
    }
}

// TODO(laurent): Instead of doing the conversions here, it would be better and simpler to handle
// it on device, so to have in the backend trait a cast method and only allow tensors to be created
// in their native dtype.
/// Convert bytes from a source dtype to Vec<T> where T: WithDTypeF.
/// This handles conversion through f32 as an intermediate type.
pub fn convert_bytes_to_vec<T: WithDTypeF>(src: &[u8], src_dtype: DType) -> Vec<T> {
    match src_dtype {
        DType::F32 => {
            let f32_vec = f32::vec_from_le_bytes(src);
            if T::DTYPE == DType::F32 {
                // SAFETY: T is f32, we can transmute Vec<f32> to Vec<T>
                unsafe { std::mem::transmute::<Vec<f32>, Vec<T>>(f32_vec) }
            } else {
                f32_vec.into_iter().map(T::from_f32).collect()
            }
        }
        DType::F16 => {
            let f16_vec = f16::vec_from_le_bytes(src);
            if T::DTYPE == DType::F16 {
                // SAFETY: T is f16, we can transmute Vec<f16> to Vec<T>
                unsafe { std::mem::transmute::<Vec<f16>, Vec<T>>(f16_vec) }
            } else {
                f16_vec.into_iter().map(|v| T::from_f32(v.to_f32())).collect()
            }
        }
        DType::BF16 => {
            let bf16_vec = bf16::vec_from_le_bytes(src);
            if T::DTYPE == DType::BF16 {
                // SAFETY: T is bf16, we can transmute Vec<bf16> to Vec<T>
                unsafe { std::mem::transmute::<Vec<bf16>, Vec<T>>(bf16_vec) }
            } else {
                bf16_vec.into_iter().map(|v| T::from_f32(v.to_f32())).collect()
            }
        }
    }
}
