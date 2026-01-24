use crate::{shape::Dim, DType, Result, Shape, WithDType};

#[derive(Clone)]
pub struct Tensor<T: WithDType> {
    pub data: Vec<T>,
    pub shape: Shape,
}

impl<T: WithDType> Tensor<T> {
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

    pub fn zeros(shape: impl Into<Shape>) -> Self {
        let shape: Shape = shape.into();
        let size = shape.elem_count();
        let data = vec![T::zero(); size];
        Tensor { data, shape }
    }

    pub fn full(shape: impl Into<Shape>, value: T) -> Self {
        let shape: Shape = shape.into();
        let size = shape.elem_count();
        let data = vec![value; size];
        Tensor { data, shape }
    }

    /// # Safety
    /// The returned tensor's data is uninitialized.
    pub unsafe fn alloc_uninit(shape: Shape) -> Self {
        let size = shape.elem_count();
        let mut data: Vec<T> = Vec::with_capacity(size);
        #[allow(clippy::uninit_vec)]
        unsafe {
            data.set_len(size);
        }
        Tensor { data, shape }
    }

    /// Reshape the tensor to a new shape with the same number of elements.
    pub fn reshape(&self, shape: impl crate::shape::ShapeWithOneHole) -> Result<Tensor<T>> {
        let shape = shape.into_shape(self.elem_count())?;
        if shape.elem_count() != self.elem_count() {
            crate::bail!(
                "reshape: cannot reshape tensor of {} elements to shape {:?} ({} elements)",
                self.elem_count(),
                shape,
                shape.elem_count()
            );
        }
        Ok(Tensor { data: self.data.clone(), shape })
    }

    /// Extract a slice of the tensor along a given dimension.
    pub fn narrow(&self, _dim: usize, _start: usize, _len: usize) -> Result<Tensor<T>> {
        todo!("narrow")
    }

    /// Index select: gather slices from self using indices along the given dimension.
    /// For self with shape (d0, d1, ..., d_dim, ..., d_n) and indices of length k,
    /// returns a tensor of shape (d0, d1, ..., k, ..., d_n).
    pub fn index_select(&self, indices: &[usize], dim: impl Dim) -> Result<Tensor<T>> {
        let dim = dim.to_index(self.shape(), "index_select dim")?;
        let dim_size = self.dim(dim)?;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= dim_size {
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
        let mut out = unsafe { Tensor::alloc_uninit(out_shape) };

        // Data layout: [outer dims][dim][inner dims]
        let outer_size: usize = if dim == 0 { 1 } else { self.dims()[..dim].iter().product() };
        let inner_size: usize = self.dims()[dim + 1..].iter().product::<usize>().max(1);

        for outer in 0..outer_size {
            for (out_idx, &src_idx) in indices.iter().enumerate() {
                let src_offset = outer * (dim_size * inner_size) + src_idx * inner_size;
                let dst_offset = outer * (indices.len() * inner_size) + out_idx * inner_size;

                out.data[dst_offset..dst_offset + inner_size]
                    .copy_from_slice(&self.data[src_offset..src_offset + inner_size]);
            }
        }

        Ok(out)
    }

    /// Concatenate tensors along a given dimension.
    pub fn cat(tensors: &[&Tensor<T>], dim: impl Dim) -> Result<Tensor<T>> {
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
        let mut out = unsafe { Tensor::alloc_uninit(out_shape) };

        // Copy data from each tensor
        // For contiguous tensors, data is laid out as: [outer dims][cat dim][inner dims]
        // outer_size = product of dimensions before cat dim
        // inner_size = product of dimensions after cat dim
        let outer_size: usize = if dim == 0 { 1 } else { out.dims()[..dim].iter().product() };
        let inner_size: usize = out.dims()[dim + 1..].iter().product::<usize>().max(1);

        let mut cat_offset = 0;
        for tensor in tensors {
            let t_cat_size = tensor.dims()[dim];

            for outer in 0..outer_size {
                for cat_idx in 0..t_cat_size {
                    let src_offset = outer * (t_cat_size * inner_size) + cat_idx * inner_size;
                    let dst_offset =
                        outer * (cat_dim_size * inner_size) + (cat_offset + cat_idx) * inner_size;

                    out.data[dst_offset..dst_offset + inner_size]
                        .copy_from_slice(&tensor.data[src_offset..src_offset + inner_size]);
                }
            }
            cat_offset += t_cat_size;
        }

        Ok(out)
    }
}
