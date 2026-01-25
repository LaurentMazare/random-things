use crate::Result;

pub trait Backend: Sized + Clone + 'static {
    type Storage<T: crate::WithDType>: Sized;

    fn storage_len<T: crate::WithDType>(storage: &Self::Storage<T>) -> usize;

    fn storage_is_empty<T: crate::WithDType>(storage: &Self::Storage<T>) -> bool {
        Self::storage_len::<T>(storage) == 0
    }

    /// # Safety
    /// This function allocates an unitialized block of memory. It is the responsibility of the
    /// caller to set the memory before using or returning the block.
    unsafe fn alloc_uninit<T: crate::WithDType>(len: usize, dev: &Self)
    -> Result<Self::Storage<T>>;

    // TODO(laurent): Add a from_slice variant.
    fn from_vec<T: crate::WithDType>(v: Vec<T>, dev: &Self) -> Result<Self::Storage<T>>;

    fn cst<T: crate::WithDType>(v: T, len: usize, dev: &Self) -> Result<Self::Storage<T>> {
        let mut res = unsafe { Self::alloc_uninit(len, dev)? };
        Self::fill(&mut res, v, len)?;
        Ok(res)
    }

    fn fill<T: crate::WithDType>(dst: &mut Self::Storage<T>, elem: T, len: usize) -> Result<()>;

    fn copy<T: crate::WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()>;

    fn data<T: crate::WithDType>(
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<std::borrow::Cow<'_, [T]>>;

    fn add_assign<T: crate::WithDType>(
        dst: &mut Self::Storage<T>,
        s: &Self::Storage<T>,
        len: usize,
    ) -> Result<()>;

    fn mul_assign<T: crate::WithDType>(
        dst: &mut Self::Storage<T>,
        s: &Self::Storage<T>,
        len: usize,
    ) -> Result<()>;

    fn add<T: crate::WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        len: usize,
    ) -> Result<()>;

    fn mul<T: crate::WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        len: usize,
    ) -> Result<()>;

    fn maximum<T: crate::WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        len: usize,
    ) -> Result<()>;

    fn minimum<T: crate::WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        len: usize,
    ) -> Result<()>;

    fn scale<T: crate::WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        v: T,
        len: usize,
    ) -> Result<()>;

    fn transpose<T: crate::WithDType>(
        dst: &mut Self::Storage<T>,
        s: &Self::Storage<T>,
        dim1: usize,
        dim2: usize,
        dims: &[usize],
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn copy2d<T: crate::WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        d1: usize,
        d2: usize,
        dst_s: usize,
        src_s: usize,
        dst_o: usize,
        src_o: usize,
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn rope<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        cos: &Self::Storage<T>,
        sin: &Self::Storage<T>,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        pos: usize,
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn rope_i<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        cos: &Self::Storage<T>,
        sin: &Self::Storage<T>,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        pos: usize,
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn gemm<T: crate::WithDType>(
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
    ) -> Result<()>;

    fn index_select<T: crate::WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        ids: &[u32],
        dim: usize,
        dims: &[usize],
    ) -> Result<()>;

    // Methods requiring T: WithDTypeF
    fn cos<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()>;

    fn sin<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()>;

    fn silu<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()>;

    fn apply_causality_mask<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        bh: usize,
        t1: usize,
        t2: usize,
        offset: usize,
    ) -> Result<()>;

    fn softmax<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
    ) -> Result<()>;

    fn rms_norm<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        alpha: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
        eps: f32,
    ) -> Result<()>;

    fn sqr<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()>;

    fn sqrt<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()>;

    fn abs<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()>;

    fn gelu_erf<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()>;

    fn elu<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        alpha: f32,
        len: usize,
    ) -> Result<()>;

    fn relu<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()>;

    fn tanh<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()>;

    fn sigmoid<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()>;

    /// Reduce max along a dimension.
    /// dst has shape with the reduced dimension removed.
    /// dim_size is the size of the dimension being reduced.
    /// outer_size is the product of dimensions before the reduced dim.
    /// inner_size is the product of dimensions after the reduced dim.
    fn reduce_max<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()>;

    /// Reduce min along a dimension.
    fn reduce_min<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()>;

    /// Reduce argmin along a dimension.
    /// Returns i64 indices.
    fn reduce_argmin<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<i64>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()>;

    /// Reduce sum along a dimension.
    fn reduce_sum<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()>;

    /// Broadcast binary operation: addition.
    /// lhs_strides and rhs_strides have 0 for broadcast dimensions.
    fn broadcast_add<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        dst_shape: &[usize],
        lhs_strides: &[usize],
        rhs_strides: &[usize],
    ) -> Result<()>;

    /// Broadcast binary operation: subtraction.
    fn broadcast_sub<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        dst_shape: &[usize],
        lhs_strides: &[usize],
        rhs_strides: &[usize],
    ) -> Result<()>;

    /// Broadcast binary operation: multiplication.
    fn broadcast_mul<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        dst_shape: &[usize],
        lhs_strides: &[usize],
        rhs_strides: &[usize],
    ) -> Result<()>;

    /// Broadcast binary operation: division.
    fn broadcast_div<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        dst_shape: &[usize],
        lhs_strides: &[usize],
        rhs_strides: &[usize],
    ) -> Result<()>;

    /// 1D convolution.
    /// src: (batch, in_channels, length)
    /// kernel: (out_channels, in_channels/groups, kernel_size)
    /// dst: (batch, out_channels, out_length)
    #[allow(clippy::too_many_arguments)]
    fn conv1d<T: crate::WithDTypeF>(
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
    ) -> Result<()>;

    /// 1D transposed convolution.
    /// src: (batch, in_channels, length)
    /// kernel: (in_channels, out_channels/groups, kernel_size)
    /// dst: (batch, out_channels, out_length)
    #[allow(clippy::too_many_arguments)]
    fn conv_transpose1d<T: crate::WithDTypeF>(
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
    ) -> Result<()>;
}
