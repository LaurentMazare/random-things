use crate::{Result, WithDType, WithDTypeF};

pub trait Backend: Sized + 'static {
    type Device;
    type S<T: WithDType>;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// # Safety
    /// This function allocates an unitialized block of memory. It is the responsibility of the
    /// caller to set the memory before using or returning the block.
    unsafe fn alloc_uninit<T: WithDType>(len: usize, dev: &Self::Device) -> Result<Self::S<T>>;

    fn from_vec<T: WithDType>(v: Vec<T>, dev: &Self::Device) -> Result<Self::S<T>>;

    fn cst<T: WithDType>(v: T, len: usize, dev: &Self::Device) -> Result<Self::S<T>> {
        let mut res = unsafe { Self::alloc_uninit(len, dev)? };
        res.fill(v, len)?;
        Ok(res)
    }

    fn device(&self) -> &Self::Device;
    fn fill<T: WithDType>(dst: &mut Self::S<T>, elem: T, len: usize) -> Result<()>;
    fn copy<T: WithDType>(dst: &mut Self::S<T>, src: &Self::S<T>, len: usize) -> Result<()>;
    fn data<T: WithDType>(src: &Self::S<T>, len: usize) -> Result<std::borrow::Cow<'_, [T]>>;

    fn add_assign<T: WithDType>(dst: &mut Self::S<T>, s: &Self::S<T>, len: usize) -> Result<()>;
    fn mul_assign<T: WithDType>(dst: &mut Self::S<T>, s: &Self::S<T>, len: usize) -> Result<()>;
    fn add<T: WithDType>(
        dst: &mut Self::S<T>,
        lhs: &Self::S<T>,
        rhs: &Self::S<T>,
        len: usize,
    ) -> Result<()>;
    fn mul<T: WithDType>(
        dst: &mut Self::S<T>,
        lhs: &Self::S<T>,
        rhs: &Self::S<T>,
        len: usize,
    ) -> Result<()>;
    fn scale<T: WithDType>(dst: &mut Self::S<T>, src: &Self::S<T>, v: T, len: usize) -> Result<()>;

    fn transpose<T: WithDType>(
        dst: &mut Self::S<T>,
        s: &Self::S<T>,
        dim1: usize,
        dim2: usize,
        dims: &[usize],
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn copy2d<T: WithDType>(
        dst: &mut Self::S<T>,
        src: &Self::S<T>,
        d1: usize,
        d2: usize,
        dst_s: usize,
        src_s: usize,
        dst_o: usize,
        src_o: usize,
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn rope<T: WithDTypeF>(
        dst: &mut Self::S<T>,
        src: &Self::S<T>,
        cos: &Self::S<T>,
        sin: &Self::S<T>,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        pos: usize,
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn rope_i<T: WithDTypeF>(
        dst: &mut Self::S<T>,
        src: &Self::S<T>,
        cos: &Self::S<T>,
        sin: &Self::S<T>,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        pos: usize,
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &mut self,
        lhs: (&Self, usize),
        rhs: (&Self, usize),
        m: usize,
        n: usize,
        k: usize,
        lhs_b: usize,
        b_stride: usize,
        _: (usize, usize),
        _: (usize, usize),
        _: (usize, usize),
    ) -> Result<()>;

    fn index_select<T: WithDType>(
        dst: &mut Self::S<T>,
        src: &Self::S<T>,
        ids: &[u32],
        dim: usize,
        dims: &[usize],
    ) -> Result<()>;
    fn cos<T: WithDTypeF>(dst: &mut Self::S<T>, src: &Self::S<T>, len: usize) -> Result<()>;
    fn sin<T: WithDTypeF>(dst: &mut Self::S<T>, src: &Self::S<T>, len: usize) -> Result<()>;
    fn silu<T: WithDTypeF>(dst: &mut Self::S<T>, src: &Self::S<T>, len: usize) -> Result<()>;
    fn apply_causality_mask<T: WithDTypeF>(
        dst: &mut Self::S<T>,
        bh: usize,
        t1: usize,
        t2: usize,
        offset: usize,
    ) -> Result<()>;

    fn softmax<T: WithDType>(
        dst: &mut Self::S<T>,
        src: &Self::S<T>,
        dim_m1: usize,
        d: usize,
    ) -> Result<()>;

    fn rms_norm<T: WithDType>(
        dst: &mut Self::S<T>,
        src: &Self::S<T>,
        alpha: &Self::S<T>,
        dim_m1: usize,
        d: usize,
        eps: f32,
    ) -> Result<()>;
}
