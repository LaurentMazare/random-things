use crate::{shape::Dim, Backend, DType, Result, Shape, WithDType};

/// Copy-on-write mutable storage that can either own or borrow data.
pub enum CowMut<'a, S> {
    Owned(S),
    Borrowed(&'a mut S),
}

impl<'a, S> CowMut<'a, S> {
    pub fn as_ref(&self) -> &S {
        match self {
            Self::Owned(s) => s,
            Self::Borrowed(s) => s,
        }
    }

    pub fn as_mut(&mut self) -> &mut S {
        match self {
            Self::Owned(s) => s,
            Self::Borrowed(s) => s,
        }
    }
}

pub struct Tensor<'a, T: WithDType, B: Backend> {
    pub(crate) data: CowMut<'a, B::Storage<T>>,
    pub(crate) shape: Shape,
    pub(crate) device: B,
}

impl<T: WithDType, B: Backend> Clone for Tensor<'static, T, B>
where
    B::Storage<T>: Clone,
{
    fn clone(&self) -> Self {
        Tensor {
            data: CowMut::Owned(self.data.as_ref().clone()),
            shape: self.shape.clone(),
            device: self.device.clone(),
        }
    }
}

impl<'a, T: WithDType, B: Backend> Tensor<'a, T, B> {
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

    pub fn storage(&self) -> &B::Storage<T> {
        self.data.as_ref()
    }

    pub fn storage_mut(&mut self) -> &mut B::Storage<T> {
        self.data.as_mut()
    }

    pub fn to_vec(&self) -> Result<Vec<T>> {
        let len = self.elem_count();
        let data_cow = B::data(self.data.as_ref(), len)?;
        Ok(data_cow.into_owned())
    }

    /// Reshape the tensor to a new shape with the same number of elements.
    #[tracing::instrument(skip_all)]
    pub fn reshape(&self, shape: impl crate::shape::ShapeWithOneHole) -> Result<Tensor<'static, T, B>> {
        let shape = shape.into_shape(self.elem_count())?;
        if shape.elem_count() != self.elem_count() {
            crate::bail!(
                "reshape: cannot reshape tensor of {} elements to shape {:?} ({} elements)",
                self.elem_count(),
                shape,
                shape.elem_count()
            );
        }
        let elem_count = shape.elem_count();
        let mut res = unsafe { Tensor::alloc_uninit(shape, self.device())? };
        B::copy(res.data.as_mut(), self.data.as_ref(), elem_count)?;
        Ok(res)
    }

    /// Extract a slice of the tensor along a given dimension.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Tensor<'static, T, B>> {
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

        let mut result = unsafe { Tensor::alloc_uninit(out_shape, self.device()) }?;

        // Copy using copy2d
        let outer_size: usize = dims[..dim].iter().product::<usize>().max(1);
        let inner_size: usize = dims[dim + 1..].iter().product::<usize>().max(1);

        B::copy2d(
            result.data.as_mut(),
            self.data.as_ref(),
            outer_size,            // d1: number of outer blocks
            len * inner_size,      // d2: elements per block to copy
            len * inner_size,      // dst_s: stride in output
            dim_size * inner_size, // src_s: stride in source
            0,                     // dst_o: start at beginning of output
            start * inner_size,    // src_o: offset by start in source
        )?;

        Ok(result)
    }

    pub fn index_select(&self, indices: &[u32], dim: impl Dim) -> Result<Tensor<'static, T, B>> {
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
        let mut out = unsafe { Tensor::alloc_uninit(out_shape, dev) }?;
        B::index_select(out.data.as_mut(), self.data.as_ref(), indices, dim, self.dims())?;
        Ok(out)
    }

    /// Concatenate tensors along a given dimension.
    pub fn cat(tensors: &[&Self], dim: impl Dim) -> Result<Tensor<'static, T, B>> {
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
        let mut out = unsafe { Tensor::alloc_uninit(out_shape, dev) }?;

        // Copy data from each tensor using copy2d
        // For contiguous tensors, data is laid out as: [outer dims][cat dim][inner dims]
        let outer_size: usize = if dim == 0 { 1 } else { out.dims()[..dim].iter().product() };
        let inner_size: usize = out.dims()[dim + 1..].iter().product::<usize>().max(1);

        let mut cat_offset = 0;
        for tensor in tensors {
            let t_cat_size = tensor.dims()[dim];
            // Copy using copy2d: outer_size rows of (t_cat_size * inner_size) elements
            B::copy2d(
                out.data.as_mut(),
                tensor.data.as_ref(),
                outer_size,                // d1: number of outer blocks
                t_cat_size * inner_size,   // d2: elements per block from this tensor
                cat_dim_size * inner_size, // dst_s: stride in output
                t_cat_size * inner_size,   // src_s: stride in source
                cat_offset * inner_size,   // dst_o: offset in output
                0,                         // src_o: offset in source
            )?;
            cat_offset += t_cat_size;
        }

        Ok(out)
    }

    /// Stack tensors along a new dimension.
    /// All tensors must have the same shape.
    /// The new dimension is inserted at position `dim`.
    pub fn stack(tensors: &[&Self], dim: impl Dim) -> Result<Tensor<'static, T, B>> {
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
        let unsqueezed: Vec<Tensor<'static, T, B>> =
            tensors.iter().map(|t| t.unsqueeze(dim)).collect::<Result<Vec<_>>>()?;
        let unsqueezed_refs: Vec<&Tensor<'static, T, B>> = unsqueezed.iter().collect();
        Tensor::cat(&unsqueezed_refs, dim)
    }
}

// Methods that always return owned tensors
impl<T: WithDType, B: Backend> Tensor<'static, T, B> {
    pub fn zeros(shape: impl Into<Shape>, device: &B) -> Result<Self> {
        Self::full(T::zero(), shape, device)
    }

    pub fn full(value: T, shape: impl Into<Shape>, device: &B) -> Result<Self> {
        let shape: Shape = shape.into();
        let size = shape.elem_count();
        let mut data = unsafe { B::alloc_uninit(size, device)? };
        B::fill(&mut data, value, size)?;
        Ok(Tensor {
            data: CowMut::Owned(data),
            shape,
            device: device.clone(),
        })
    }

    /// # Safety
    /// The returned tensor's data is uninitialized.
    pub unsafe fn alloc_uninit(shape: impl Into<Shape>, dev: &B) -> Result<Self> {
        let shape = shape.into();
        let size = shape.elem_count();
        let data = unsafe { B::alloc_uninit(size, dev)? };
        Ok(Tensor {
            data: CowMut::Owned(data),
            shape,
            device: dev.clone(),
        })
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
            data: CowMut::Owned(data),
            shape,
            device: dev.clone(),
        })
    }
}

// Methods for creating borrowed tensors
impl<'a, T: WithDType, B: Backend> Tensor<'a, T, B> {
    /// Create a tensor that borrows existing storage.
    pub fn from_storage(storage: &'a mut B::Storage<T>, shape: impl Into<Shape>, device: &B) -> Result<Self> {
        let shape = shape.into();
        let storage_len = B::storage_len(storage);
        if storage_len < shape.elem_count() {
            crate::bail!(
                "from_storage: storage length {} is less than shape {:?} with {} elements",
                storage_len,
                shape,
                shape.elem_count()
            );
        }
        Ok(Tensor {
            data: CowMut::Borrowed(storage),
            shape,
            device: device.clone(),
        })
    }

    /// Convert to an owned tensor by copying if borrowed.
    pub fn into_owned(self) -> Result<Tensor<'static, T, B>> {
        match self.data {
            CowMut::Owned(data) => Ok(Tensor {
                data: CowMut::Owned(data),
                shape: self.shape,
                device: self.device,
            }),
            CowMut::Borrowed(storage) => {
                let elem_count = self.shape.elem_count();
                let mut new_data = unsafe { B::alloc_uninit(elem_count, &self.device)? };
                B::copy(&mut new_data, storage, elem_count)?;
                Ok(Tensor {
                    data: CowMut::Owned(new_data),
                    shape: self.shape,
                    device: self.device,
                })
            }
        }
    }

    /// Check if the tensor owns its data.
    pub fn is_owned(&self) -> bool {
        matches!(self.data, CowMut::Owned(_))
    }
}
