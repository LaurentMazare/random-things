use crate::{Backend, DType, Result, Shape, WithDType, shape::Dim};
use std::cell::{Ref, RefCell, RefMut};
use std::sync::Arc;

#[derive(Clone)]
pub struct Tensor<T: WithDType, B: Backend> {
    pub(crate) data: Arc<RefCell<B::Storage<T>>>,
    pub(crate) shape: Shape,
    pub(crate) device: B,
    _marker: std::marker::PhantomData<T>,
}

pub enum TypedTensor<'a, B: Backend> {
    F16(&'a Tensor<half::f16, B>),
    BF16(&'a Tensor<half::bf16, B>),
    F32(&'a Tensor<f32, B>),
    I64(&'a Tensor<i64, B>),
    U8(&'a Tensor<u8, B>),
}

impl<T: WithDType, B: Backend> Tensor<T, B> {
    pub fn dtype(&self) -> DType {
        T::DTYPE
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn elem_count(&self) -> usize {
        self.shape.elem_count()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn dim(&self, index: impl Dim) -> Result<usize> {
        self.shape.dim(index)
    }

    pub fn device(&self) -> &B {
        &self.device
    }

    /// Borrow the underlying storage immutably.
    /// Returns an error if the storage is currently mutably borrowed.
    pub fn storage(&self) -> Result<Ref<'_, B::Storage<T>>> {
        self.data.try_borrow().map_err(|_| {
            crate::Error::Msg("tensor storage is currently mutably borrowed".to_string()).bt()
        })
    }

    /// Borrow the underlying storage mutably.
    /// Returns an error if the storage is currently borrowed (mutably or immutably).
    pub fn storage_mut(&self) -> Result<RefMut<'_, B::Storage<T>>> {
        self.data
            .try_borrow_mut()
            .map_err(|_| crate::Error::Msg("tensor storage is currently borrowed".to_string()).bt())
    }

    /// Get the raw Arc<RefCell<...>> for direct access.
    pub fn storage_arc(&self) -> &Arc<RefCell<B::Storage<T>>> {
        &self.data
    }

    pub fn zeros(shape: impl Into<Shape>, device: &B) -> Result<Self> {
        Self::full(T::zero(), shape, device)
    }

    pub fn to_vec(&self) -> Result<Vec<T>> {
        let len = self.elem_count();
        let data = self.storage()?;
        let data_cow = B::data(&*data, len)?;
        Ok(data_cow.into_owned())
    }

    pub fn full(value: T, shape: impl Into<Shape>, device: &B) -> Result<Self> {
        let shape: Shape = shape.into();
        let size = shape.elem_count();
        let mut data = unsafe { B::alloc_uninit(size, device)? };
        B::fill(&mut data, value, size)?;
        Ok(Tensor {
            data: Arc::new(RefCell::new(data)),
            shape,
            device: device.clone(),
            _marker: std::marker::PhantomData,
        })
    }

    /// Reshape the tensor to a new shape with the same number of elements.
    /// This operation shares the underlying data (no copy).
    #[tracing::instrument(skip_all)]
    pub fn reshape(&self, shape: impl crate::shape::ShapeWithOneHole) -> Result<Self> {
        let shape = shape.into_shape(self.elem_count())?;
        if shape.elem_count() != self.elem_count() {
            crate::bail!(
                "reshape: cannot reshape tensor of {} elements to shape {:?} ({} elements)",
                self.elem_count(),
                shape,
                shape.elem_count()
            );
        }
        // Share the underlying data instead of copying
        Ok(Tensor {
            data: Arc::clone(&self.data),
            shape,
            device: self.device.clone(),
            _marker: std::marker::PhantomData,
        })
    }

    /// Extract a slice of the tensor along a given dimension.
    #[tracing::instrument(skip_all)]
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let dims = self.dims();
        if dim >= dims.len() {
            crate::bail!("narrow: dim {} out of range for rank {}", dim, dims.len());
        }
        let dim_size = dims[dim];
        if start + len > dim_size {
            crate::bail!("narrow: start {start} + len {len} exceeds dim size {dim_size}");
        }

        // Compute output shape
        let mut out_dims = dims.to_vec();
        out_dims[dim] = len;
        let out_shape = crate::Shape::from(out_dims);

        let result = unsafe { Self::alloc_uninit(out_shape, self.device()) }?;

        // Copy using copy2d
        let outer_size: usize = dims[..dim].iter().product::<usize>().max(1);
        let inner_size: usize = dims[dim + 1..].iter().product::<usize>().max(1);

        {
            let src_data = self.storage()?;
            let mut dst_data = result.storage_mut()?;
            B::copy2d(
                &mut dst_data,
                &*src_data,
                outer_size,            // d1: number of outer blocks
                len * inner_size,      // d2: elements per block to copy
                len * inner_size,      // dst_s: stride in output
                dim_size * inner_size, // src_s: stride in source
                0,                     // dst_o: start at beginning of output
                start * inner_size,    // src_o: offset by start in source
            )?;
        }

        Ok(result)
    }

    /// # Safety
    /// The returned tensor's data is uninitialized.
    pub unsafe fn alloc_uninit(shape: impl Into<Shape>, dev: &B) -> Result<Self> {
        let shape = shape.into();
        let size = shape.elem_count();
        let data = unsafe { B::alloc_uninit(size, dev)? };
        Ok(Tensor {
            data: Arc::new(RefCell::new(data)),
            shape,
            device: dev.clone(),
            _marker: std::marker::PhantomData,
        })
    }

    pub fn index_select(&self, indices: &[u32], dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "index_select dim")?;
        let dim_size = self.dim(dim)?;
        for (i, &idx) in indices.iter().enumerate() {
            if idx as usize >= dim_size {
                crate::bail!(
                    "index_select: index {idx} at position {i} is out of bounds for dimension {dim} with size {dim_size}"
                );
            }
        }

        // Calculate output shape
        let mut out_dims: Vec<usize> = self.dims().to_vec();
        out_dims[dim] = indices.len();
        let out_shape = Shape::from(out_dims);

        // Allocate output
        let dev = self.device();
        let out: Self = unsafe { Tensor::alloc_uninit(out_shape, dev) }?;
        {
            let src_data = self.storage()?;
            let mut dst_data = out.storage_mut()?;
            B::index_select(&mut dst_data, &*src_data, indices, dim, self.dims())?;
        }
        Ok(out)
    }

    pub fn from_vec<S: Into<Shape>>(data: Vec<T>, shape: S, dev: &B) -> Result<Self> {
        let shape = shape.into();
        if data.len() != shape.elem_count() {
            crate::bail!(
                "from_vec: data length {} does not match shape {:?} with {} elements",
                data.len(),
                shape,
                shape.elem_count()
            );
        }
        let data = B::from_vec(data, dev)?;
        Ok(Tensor {
            data: Arc::new(RefCell::new(data)),
            shape,
            device: dev.clone(),
            _marker: std::marker::PhantomData,
        })
    }

    /// Concatenate tensors along a given dimension.
    pub fn cat(tensors: &[&Self], dim: impl Dim) -> Result<Self> {
        if tensors.is_empty() {
            crate::bail!("cat requires at least one tensor");
        }

        let first = tensors[0];
        let rank = first.rank();
        let dim = dim.to_index(first.shape(), "cat dim")?;

        for (i, t) in tensors.iter().enumerate().skip(1) {
            if t.rank() != rank {
                crate::bail!("cat: tensor {i} has rank {} but expected {rank}", t.rank());
            }
            for d in 0..rank {
                if d != dim && t.dims()[d] != first.dims()[d] {
                    crate::bail!(
                        "cat: tensor {i} has shape {:?} but expected dimension {d} to be {}",
                        t.shape(),
                        first.dims()[d]
                    );
                }
            }
        }

        // Calculate output shape
        let cat_dim_size: usize = tensors.iter().map(|t| t.dims()[dim]).sum();
        let mut out_dims: Vec<usize> = first.dims().to_vec();
        out_dims[dim] = cat_dim_size;
        let out_shape = Shape::from(out_dims);

        // Allocate output
        let dev = first.device();
        let out: Self = unsafe { Tensor::alloc_uninit(out_shape, dev) }?;

        // Copy data from each tensor using copy2d
        // For contiguous tensors, data is laid out as: [outer dims][cat dim][inner dims]
        let outer_size: usize = if dim == 0 { 1 } else { out.dims()[..dim].iter().product() };
        let inner_size: usize = out.dims()[dim + 1..].iter().product::<usize>().max(1);

        let mut cat_offset = 0;
        {
            let mut out_data = out.storage_mut()?;
            for tensor in tensors {
                let t_cat_size = tensor.dims()[dim];
                let src_data = tensor.storage()?;
                // Copy using copy2d: outer_size rows of (t_cat_size * inner_size) elements
                B::copy2d(
                    &mut out_data,
                    &*src_data,
                    outer_size,                // d1: number of outer blocks
                    t_cat_size * inner_size,   // d2: elements per block from this tensor
                    cat_dim_size * inner_size, // dst_s: stride in output
                    t_cat_size * inner_size,   // src_s: stride in source
                    cat_offset * inner_size,   // dst_o: offset in output
                    0,                         // src_o: offset in source
                )?;
                cat_offset += t_cat_size;
            }
        }

        Ok(out)
    }

    /// Stack tensors along a new dimension.
    /// All tensors must have the same shape.
    /// The new dimension is inserted at position `dim`.
    pub fn stack(tensors: &[&Self], dim: impl Dim) -> Result<Self> {
        if tensors.is_empty() {
            crate::bail!("stack requires at least one tensor");
        }

        let first = tensors[0];
        // For stack, dim can be 0..=rank (inserting a new dimension)
        let dim = dim.to_index_plus_one(first.shape(), "stack dim")?;

        // All tensors must have the same shape
        for (i, t) in tensors.iter().enumerate().skip(1) {
            if t.shape() != first.shape() {
                crate::bail!(
                    "stack: tensor {i} has shape {:?} but expected {:?}",
                    t.shape(),
                    first.shape()
                );
            }
        }

        // Unsqueeze each tensor at dim, then concatenate
        let unsqueezed: Vec<Self> =
            tensors.iter().map(|t| t.unsqueeze(dim)).collect::<Result<Vec<_>>>()?;
        let unsqueezed_refs: Vec<&Self> = unsqueezed.iter().collect();
        Self::cat(&unsqueezed_refs, dim)
    }

    pub fn downcast(&self) -> Result<TypedTensor<'_, B>> {
        use crate::error::Context;
        let slf = self as &dyn std::any::Any;
        let tt = match T::DTYPE {
            DType::F16 => TypedTensor::F16(slf.downcast_ref().context("downcast to f16")?),
            DType::BF16 => TypedTensor::BF16(slf.downcast_ref().context("downcast to bf16")?),
            DType::F32 => TypedTensor::F32(slf.downcast_ref().context("downcast to f32")?),
            DType::I64 => TypedTensor::I64(slf.downcast_ref().context("downcast to i64")?),
            DType::U8 => TypedTensor::U8(slf.downcast_ref().context("downcast to u8")?),
        };
        Ok(tt)
    }
}
