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
}
