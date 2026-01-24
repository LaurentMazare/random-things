use crate::{shape::Dim, Backend, DType, Result, Shape, WithDType};

#[derive(Clone)]
pub struct Tensor<T: WithDType, B: Backend<T>> {
    pub(crate) data: B,
    pub(crate) shape: Shape,
    _marker: std::marker::PhantomData<T>,
}

impl<T: WithDType, B: Backend<T>> Tensor<T, B> {
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

    pub fn device(&self) -> &B::Device {
        self.data.device()
    }

    pub fn storage(&self) -> &B {
        &self.data
    }

    pub fn storage_mut(&mut self) -> &mut B {
        &mut self.data
    }

    pub fn zeros(shape: impl Into<Shape>, device: &B::Device) -> Result<Self> {
        Self::full(T::zero(), shape, device)
    }

    pub fn to_vec(&self) -> Result<Vec<T>> {
        let len = self.elem_count();
        let data_cow = self.data.data(len)?;
        Ok(data_cow.into_owned())
    }

    pub fn full(value: T, shape: impl Into<Shape>, device: &B::Device) -> Result<Self> {
        let shape: Shape = shape.into();
        let size = shape.elem_count();
        let mut data = unsafe { B::alloc_uninit(size, device)? };
        data.fill(value, size)?;
        Ok(Tensor { data, shape, _marker: std::marker::PhantomData })
    }

    /// Reshape the tensor to a new shape with the same number of elements.
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
        let elem_count = shape.elem_count();
        let mut res = unsafe { Self::alloc_uninit(shape, self.device())? };
        // TODO(laurent): avoid copying data if possible.
        res.data.copy(&self.data, elem_count)?;
        Ok(res)
    }

    /// Extract a slice of the tensor along a given dimension.
    pub fn narrow(&self, _dim: usize, _start: usize, _len: usize) -> Result<Self> {
        todo!("narrow")
    }

    /// # Safety
    /// The returned tensor's data is uninitialized.
    pub unsafe fn alloc_uninit(shape: Shape, dev: &B::Device) -> Result<Self> {
        let size = shape.elem_count();
        let data = unsafe { B::alloc_uninit(size, &dev)? };
        Ok(Tensor { data, shape, _marker: std::marker::PhantomData })
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
        let mut out: Self = unsafe { Tensor::alloc_uninit(out_shape, dev) }?;
        out.data.index_select(&self.data, indices, dim)?;
        Ok(out)
    }

    pub fn from_vec<S: Into<Shape>>(data: Vec<T>, shape: S, dev: &B::Device) -> Result<Self> {
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
        let shape = shape.into();
        Ok(Tensor { data, shape, _marker: std::marker::PhantomData })
    }

    /// Concatenate tensors along a given dimension.
    pub fn cat(tensors: &[&Self], dim: impl Dim) -> Result<Self> {
        todo!()
        // if tensors.is_empty() {
        //     crate::bail!("cat requires at least one tensor");
        // }

        // let first = tensors[0];
        // let rank = first.rank();
        // let dim = dim.to_index(first.shape(), "cat dim")?;

        // for (i, t) in tensors.iter().enumerate().skip(1) {
        //     if t.rank() != rank {
        //         crate::bail!("cat: tensor {i} has rank {} but expected {rank}", t.rank());
        //     }
        //     for d in 0..rank {
        //         if d != dim && t.dims()[d] != first.dims()[d] {
        //             crate::bail!(
        //                 "cat: tensor {i} has shape {:?} but expected dimension {d} to be {}",
        //                 t.shape(),
        //                 first.dims()[d]
        //             );
        //         }
        //     }
        // }

        // // Calculate output shape
        // let cat_dim_size: usize = tensors.iter().map(|t| t.dims()[dim]).sum();
        // let mut out_dims: Vec<usize> = first.dims().to_vec();
        // out_dims[dim] = cat_dim_size;
        // let out_shape = Shape::from(out_dims);

        // // Allocate output
        // let mut out = unsafe { Tensor::alloc_uninit(out_shape) };

        // // Copy data from each tensor
        // // For contiguous tensors, data is laid out as: [outer dims][cat dim][inner dims]
        // // outer_size = product of dimensions before cat dim
        // // inner_size = product of dimensions after cat dim
        // let outer_size: usize = if dim == 0 { 1 } else { out.dims()[..dim].iter().product() };
        // let inner_size: usize = out.dims()[dim + 1..].iter().product::<usize>().max(1);

        // let mut cat_offset = 0;
        // for tensor in tensors {
        //     let t_cat_size = tensor.dims()[dim];

        //     for outer in 0..outer_size {
        //         for cat_idx in 0..t_cat_size {
        //             let src_offset = outer * (t_cat_size * inner_size) + cat_idx * inner_size;
        //             let dst_offset =
        //                 outer * (cat_dim_size * inner_size) + (cat_offset + cat_idx) * inner_size;

        //             out.data[dst_offset..dst_offset + inner_size]
        //                 .copy_from_slice(&tensor.data[src_offset..src_offset + inner_size]);
        //         }
        //     }
        //     cat_offset += t_cat_size;
        // }

        // Ok(out)
    }
}
