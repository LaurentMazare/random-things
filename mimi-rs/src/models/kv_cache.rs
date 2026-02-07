use crate::{Backend, Result, Shape, Tensor, TensorView, WithDType, shape::Dim};

pub struct Cache<T: WithDType, B: Backend> {
    all_data: Tensor<T, B>,
    dim: usize,
    current_seq_len: usize,
    max_seq_len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: WithDType, B: Backend> Cache<T, B> {
    pub fn new<S: Into<Shape>, D: Dim>(dim: D, shape: S, dev: &B) -> Result<Self> {
        let shape = shape.into();
        let dim = dim.to_index(&shape, "kv-cache")?;
        let max_seq_len = shape.dims()[dim];
        let all_data = Tensor::zeros(shape, dev)?;
        Ok(Self { all_data, dim, current_seq_len: 0, max_seq_len, _phantom: Default::default() })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn all_data(&self) -> &Tensor<T, B> {
        &self.all_data
    }

    pub fn current_data(&self) -> Result<TensorView<T, B>> {
        let view = TensorView::from(&self.all_data);
        view.narrow(self.dim, 0, Some(self.current_seq_len))
    }

    pub fn append(&mut self, src: &Tensor<T, B>) -> Result<()> {
        let seq_len = src.dim(self.dim)?;
        if self.current_seq_len + seq_len > self.max_seq_len {
            crate::bail!(
                "kv-cache: above max-seq-len {}+{seq_len}>{}",
                self.current_seq_len,
                self.max_seq_len
            )
        }
        self.all_data.slice_assign(src, self.dim, self.current_seq_len)?;
        self.current_seq_len += seq_len;
        Ok(())
    }
}

pub struct KvCache<T: WithDType, B: Backend> {
    k: Cache<T, B>,
    v: Cache<T, B>,
}

impl<T: WithDType, B: Backend> KvCache<T, B> {
    pub fn new<S: Into<Shape>, D: Dim>(dim: D, shape: S, dev: &B) -> Result<Self> {
        let shape = shape.into();
        let dim = dim.to_index(&shape, "kv-cache")?;
        let k = Cache::new(dim, &shape, dev)?;
        let v = Cache::new(dim, &shape, dev)?;
        Ok(Self { k, v })
    }

    pub fn k(&self) -> &Cache<T, B> {
        &self.k
    }

    pub fn v(&self) -> &Cache<T, B> {
        &self.v
    }

    pub fn append(
        &mut self,
        k: &Tensor<T, B>,
        v: &Tensor<T, B>,
    ) -> Result<(TensorView<T, B>, TensorView<T, B>)> {
        self.k.append(k)?;
        self.v.append(v)?;
        let k = self.k.current_data()?;
        let v = self.v.current_data()?;
        Ok((k, v))
    }
}
