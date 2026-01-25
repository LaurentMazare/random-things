use crate::{Backend, Result, Shape, Tensor, WithDType};

pub struct MmapedFiles {
    mmaps: Vec<(std::path::PathBuf, memmap2::Mmap)>,
}

impl MmapedFiles {
    pub fn load_from_files<P: AsRef<std::path::Path>>(file_paths: &[P]) -> Result<Self> {
        let mut mmaps = Vec::new();
        for path in file_paths {
            let path = path.as_ref();
            let file = std::fs::File::open(path)?;
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
            mmaps.push((path.to_path_buf(), mmap));
        }
        Ok(Self { mmaps })
    }
}

#[derive(yoke::Yokeable)]
pub struct TensorData<'a> {
    pub data: &'a [u8],
    pub shape: Shape,
    pub dtype: crate::DType,
}

pub struct VarBuilder<'a, B: Backend> {
    tensor_data: std::collections::HashMap<String, TensorData<'a>>,
    device: B,
}

fn load_tensor_data(mmaps: &MmapedFiles) -> Result<std::collections::HashMap<String, TensorData<'_>>> {
    let mut tensor_data = std::collections::HashMap::new();
    for (_path, mmap) in mmaps.mmaps.iter() {
        let tensors = safetensors::SafeTensors::deserialize(mmap)?;
        for (name, tensor) in tensors.iter() {
            let shape: Shape = tensor.shape().into();
            let data = tensor.data();
            let dtype = match tensor.dtype() {
                safetensors::Dtype::F32 => crate::DType::F32,
                safetensors::Dtype::F16 => crate::DType::F16,
                safetensors::Dtype::BF16 => crate::DType::BF16,
                _ => continue,
            };
            let td = TensorData { data, shape, dtype };
            tensor_data.insert(name.to_string(), td);
        }
    }
    Ok(tensor_data)
}

impl<'a, B: Backend> VarBuilder<'a, B> {
    pub fn load(mmaped_files: &'a MmapedFiles, device: B) -> Result<Self> {
        let tensor_data = load_tensor_data(mmaped_files)?;
        Ok(Self { tensor_data, device })
    }

    pub fn get_tensor(&self, name: &str) -> Option<&TensorData<'a>> {
        self.tensor_data.get(name)
    }

    pub fn device(&self) -> &B {
        &self.device
    }

    pub fn tensor<T: WithDType>(
        &self,
        name: &str,
        shape: impl Into<Shape>,
    ) -> Result<Tensor<T, B>> {
        let td = self.tensor_data.get(name);
        make_tensor(td, name, shape, &self.device)
    }
}

fn make_tensor<T: WithDType, B: Backend>(
    td: Option<&TensorData<'_>>,
    name: &str,
    shape: impl Into<Shape>,
    device: &B,
) -> Result<Tensor<T, B>> {
    let td = match td {
        Some(t) => t,
        None => crate::bail!("tensor '{name}' not found"),
    };
    if td.dtype != T::DTYPE {
        crate::bail!(
            "dtype mismatch for tensor '{name}': expected {:?}, found {:?}",
            T::DTYPE,
            td.dtype
        );
    }
    let shape = shape.into();
    if td.shape != shape {
        crate::bail!(
            "shape mismatch for tensor '{name}': expected {shape:?}, found {:?}",
            td.shape
        );
    }
    let data = T::vec_from_le_bytes(td.data);
    let tensor = Tensor::from_vec(data, shape, device)?;
    Ok(tensor)
}

// Inner yokeable struct that holds the borrowed tensor data
#[derive(yoke::Yokeable)]
struct VarBuilderYoke<'a> {
    tensor_data: std::collections::HashMap<String, TensorData<'a>>,
}

/// A self-contained VarBuilder that owns its memory-mapped files.
/// This wrapper uses yoke to safely combine owned MmapedFiles with borrowed tensor data.
pub struct VB<B: Backend> {
    yoke: yoke::Yoke<VarBuilderYoke<'static>, Box<MmapedFiles>>,
    device: B,
}

impl<B: Backend> VB<B> {
    pub fn load<P: AsRef<std::path::Path>>(file_paths: &[P], device: B) -> Result<Self> {
        let mmaps = MmapedFiles::load_from_files(file_paths)?;
        let yoke = yoke::Yoke::try_attach_to_cart(Box::new(mmaps), |mmaps| -> Result<_> {
            let tensor_data = load_tensor_data(mmaps)?;
            Ok(VarBuilderYoke { tensor_data })
        })?;
        Ok(Self { yoke, device })
    }

    pub fn get_tensor(&self, name: &str) -> Option<&TensorData<'_>> {
        self.yoke.get().tensor_data.get(name)
    }

    pub fn device(&self) -> &B {
        &self.device
    }

    pub fn tensor<T: WithDType>(&self, name: &str, shape: impl Into<Shape>) -> Result<Tensor<T, B>> {
        let td = self.yoke.get().tensor_data.get(name);
        make_tensor(td, name, shape, &self.device)
    }
}
