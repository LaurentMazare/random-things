use super::rope::RotaryEmbedding;
use crate::nn::var_builder::Path;
use crate::{Backend, Result, Tensor, WithDTypeF};

/// State for StreamingMultiheadAttention.
pub struct StreamingMHAState<T: WithDTypeF, B: Backend> {
    /// KV cache: shape [2, batch_size, sequence_length, num_heads, dim_per_head]
    pub cache: Tensor<T, B>,
    /// Current end position (number of tokens seen so far).
    pub current_end: usize,
}

#[allow(clippy::type_complexity)]
fn complete_kv<T: WithDTypeF, B: Backend>(
    cache: &Tensor<T, B>,
    current_end: usize,
    k: &Tensor<T, B>,
    v: &Tensor<T, B>,
) -> Result<(Tensor<T, B>, Tensor<T, B>, usize)> {
    let t = k.dim(1usize)?;

    // cache[0, :, current_end..current_end+t] = k
    // cache[1, :, current_end..current_end+t] = v
    cache
        .narrow(0, 0..1)?
        .contiguous()?
        .reshape(cache.dims()[1..].to_vec())?
        .slice_set(k, 1usize, current_end)?;
    cache
        .narrow(0, 1..2)?
        .contiguous()?
        .reshape(cache.dims()[1..].to_vec())?
        .slice_set(v, 1usize, current_end)?;

    let new_end = current_end + t;
    // valid = cache[:, :, :new_end]
    let cache_shape = cache.dims().to_vec();
    let keys = cache
        .narrow(0, 0..1)?
        .contiguous()?
        .reshape(cache_shape[1..].to_vec())?
        .narrow(1, 0..new_end)?
        .contiguous()?;
    let values = cache
        .narrow(0, 1..2)?
        .contiguous()?
        .reshape(cache_shape[1..].to_vec())?
        .narrow(1, 0..new_end)?
        .contiguous()?;
    Ok((keys, values, new_end))
}

fn materialize_causal_mask<T: WithDTypeF, B: Backend>(
    num_queries: usize,
    num_keys: usize,
    device: &B,
) -> Result<Tensor<T, B>> {
    let shift = num_keys - num_queries;
    // Upper-left triangular mask (causal)
    let mut data = Vec::with_capacity(num_queries * num_keys);
    for q in 0..num_queries {
        for k in 0..num_keys {
            if k <= q + shift {
                data.push(T::from_f32(0.0));
            } else {
                data.push(T::from_f32(f32::NEG_INFINITY));
            }
        }
    }
    Tensor::from_vec(data, (num_queries, num_keys), device)
}

/// Streaming multi-head attention (used by the flow LM transformer).
pub struct StreamingMultiheadAttention<T: WithDTypeF, B: Backend> {
    in_proj_weight: Tensor<T, B>,
    out_proj_weight: Tensor<T, B>,
    pub embed_dim: usize,
    pub num_heads: usize,
    name: String,
}

impl<T: WithDTypeF, B: Backend> StreamingMultiheadAttention<T, B> {
    pub fn load(vb: &Path<B>, embed_dim: usize, num_heads: usize) -> Result<Self> {
        let out_dim = 3 * embed_dim;
        let in_proj_weight = vb.pp("in_proj").tensor("weight", (out_dim, embed_dim))?;
        let out_proj_weight = vb.pp("out_proj").tensor("weight", (embed_dim, embed_dim))?;
        let name = vb.prefix();
        Ok(Self { in_proj_weight, out_proj_weight, embed_dim, num_heads, name })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
    ) -> StreamingMHAState<T, B> {
        let dim_per_head = self.embed_dim / self.num_heads;
        let cache = Tensor::full(
            T::from_f32(f32::NAN),
            (2, batch_size, sequence_length, self.num_heads, dim_per_head),
            self.in_proj_weight.device(),
        )
        .unwrap();
        StreamingMHAState { cache, current_end: 0 }
    }

    pub fn forward(
        &self,
        query: &Tensor<T, B>,
        rope: &RotaryEmbedding<T, B>,
        state: &mut StreamingMHAState<T, B>,
    ) -> Result<Tensor<T, B>> {
        let (b, t, _) = query.dims3()?;
        let d = self.embed_dim / self.num_heads;
        let offset = state.current_end;

        let projected = query.matmul_t(&self.in_proj_weight)?;
        // Reshape to [b, t, 3, num_heads, d]
        let packed = projected.reshape((b, t, 3, self.num_heads, d))?;
        // q, k, v each [b, t, num_heads, d]
        let q = packed.narrow(2, 0..1)?.contiguous()?.reshape((b, t, self.num_heads, d))?;
        let k = packed.narrow(2, 1..2)?.contiguous()?.reshape((b, t, self.num_heads, d))?;
        let v = packed.narrow(2, 2..3)?.contiguous()?.reshape((b, t, self.num_heads, d))?;

        // Apply RoPE: q, k are [b, t, h, d]
        let (q, k) = rope.forward(&q, &k, offset)?;

        // Complete KV cache: k, v are [b, t, h, d]
        let (k, v, new_end) = complete_kv(&state.cache, offset, &k, &v)?;
        state.current_end = new_end;

        // Causal mask
        let kv_len = k.dim(1usize)?;
        let mask = materialize_causal_mask::<T, B>(t, kv_len, query.device())?;

        // Transpose to [b, h, t, d] for attention
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Scaled dot-product attention
        let scale = T::from_f32(1.0 / (d as f32).sqrt());
        let attn = q.matmul_t(&k)?.scale(scale)?;
        let attn = attn.broadcast_add(&mask)?;
        let attn = attn.softmax()?;
        let x = attn.matmul(&v)?;

        // Back to [b, t, h*d]
        let x = x.transpose(1, 2)?.reshape((b, t, self.embed_dim))?;
        crate::ops::matmul_t(&x, &self.out_proj_weight)
    }
}
