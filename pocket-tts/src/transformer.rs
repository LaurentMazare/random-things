use anyhow::Result;
use mimi::nn::var_builder::Path;
use mimi::{CpuDevice, CpuTensor};

/// KV cache for attention layers.
pub struct KvCache {
    k: Option<CpuTensor<f32>>,
    v: Option<CpuTensor<f32>>,
    max_seq_len: usize, // 0 = unlimited
}

impl KvCache {
    pub fn new(max_seq_len: usize) -> Self {
        Self { k: None, v: None, max_seq_len }
    }

    /// Append new K,V to cache and return full K,V.
    /// Returns (full_k, full_v, offset) where offset is the position before append.
    pub fn append(
        &mut self,
        new_k: CpuTensor<f32>,
        new_v: CpuTensor<f32>,
    ) -> Result<(CpuTensor<f32>, CpuTensor<f32>, usize)> {
        let (k, v, offset) = match (self.k.take(), self.v.take()) {
            (Some(prev_k), Some(prev_v)) => {
                let offset = prev_k.dims()[2]; // seq_len dim
                let k = CpuTensor::cat(&[&prev_k, &new_k], 2)?;
                let v = CpuTensor::cat(&[&prev_v, &new_v], 2)?;
                (k, v, offset)
            }
            _ => (new_k, new_v, 0),
        };

        // Trim to max_seq_len if needed
        let (k, v) = if self.max_seq_len > 0 {
            let seq_len = k.dims()[2];
            if seq_len > self.max_seq_len {
                let start = seq_len - self.max_seq_len;
                let k = k.narrow(2, start..seq_len)?.contiguous()?;
                let v = v.narrow(2, start..seq_len)?.contiguous()?;
                (k, v)
            } else {
                (k, v)
            }
        } else {
            (k, v)
        };

        self.k = Some(k.copy()?);
        self.v = Some(v.copy()?);
        Ok((k, v, offset))
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }

    pub fn current_len(&self) -> usize {
        self.k.as_ref().map_or(0, |k| k.dims()[2])
    }
}

/// Precomputed rotary embeddings.
pub struct RotaryEmbedding {
    cos: CpuTensor<f32>,
    sin: CpuTensor<f32>,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_len: usize, max_period: f32) -> Result<Self> {
        let dev = &mimi::CPU;
        let half_dim = dim / 2;
        let mut cos_data = vec![0f32; max_len * half_dim];
        let mut sin_data = vec![0f32; max_len * half_dim];

        for pos in 0..max_len {
            for i in 0..half_dim {
                let freq = 1.0 / max_period.powf(2.0 * i as f32 / dim as f32);
                let angle = pos as f32 * freq;
                cos_data[pos * half_dim + i] = angle.cos();
                sin_data[pos * half_dim + i] = angle.sin();
            }
        }

        let cos = CpuTensor::from_vec(cos_data, (max_len, half_dim), dev)?;
        let sin = CpuTensor::from_vec(sin_data, (max_len, half_dim), dev)?;
        Ok(Self { cos, sin })
    }
}

/// Multi-head attention with KV cache and RoPE.
pub struct Attention {
    in_proj_weight: CpuTensor<f32>,
    out_proj_weight: CpuTensor<f32>,
    out_proj_bias: Option<CpuTensor<f32>>,
    num_heads: usize,
    head_dim: usize,
    kv_cache: KvCache,
}

impl Attention {
    pub fn load(
        vb: &Path<CpuDevice>,
        d_model: usize,
        num_heads: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        let head_dim = d_model / num_heads;
        let in_proj_weight: CpuTensor<f32> =
            vb.tensor("self_attn.in_proj.weight", (3 * d_model, d_model))?;
        let out_proj_weight: CpuTensor<f32> =
            vb.tensor("self_attn.out_proj.weight", (d_model, d_model))?;
        let out_proj_bias = if vb.contains("self_attn.out_proj.bias") {
            Some(vb.tensor("self_attn.out_proj.bias", (d_model,))?)
        } else {
            None
        };
        Ok(Self {
            in_proj_weight,
            out_proj_weight,
            out_proj_bias,
            num_heads,
            head_dim,
            kv_cache: KvCache::new(max_seq_len),
        })
    }

    /// xs: (batch, seq, d_model), returns (batch, seq, d_model)
    pub fn forward(
        &mut self,
        xs: &CpuTensor<f32>,
        rope: &RotaryEmbedding,
    ) -> Result<CpuTensor<f32>> {
        let dims = xs.dims();
        let (b, s, _d) = (dims[0], dims[1], dims[2]);

        // Project QKV: (b, s, 3*d_model)
        let qkv = xs.matmul_t(&self.in_proj_weight)?;

        // Split into Q, K, V each (b, s, d_model)
        let d_model = self.num_heads * self.head_dim;
        let q = qkv.narrow(2, ..d_model)?.contiguous()?;
        let k = qkv.narrow(2, d_model..2 * d_model)?.contiguous()?;
        let v = qkv.narrow(2, 2 * d_model..3 * d_model)?.contiguous()?;

        // Reshape to (b, s, num_heads, head_dim) -> (b, num_heads, s, head_dim)
        let q = q
            .reshape((b, s, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, s, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, s, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // KV cache: get offset, then append
        let offset = self.kv_cache.current_len();

        // Apply RoPE to Q and K
        let q = q.rope_i(&rope.cos, &rope.sin, offset)?;
        let k = k.rope_i(&rope.cos, &rope.sin, offset)?;

        // Append to KV cache
        let (k, v, _offset) = self.kv_cache.append(k, v)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt();
        // q: (b, heads, s_q, head_dim), k: (b, heads, s_kv, head_dim)
        // attn_weights: (b, heads, s_q, s_kv)
        let attn_weights = q.matmul_t(&k)?.scale(1.0 / scale)?;

        // Apply causal mask
        let attn_weights = attn_weights.apply_causality_mask(offset)?;

        // Softmax over last dim
        let attn_weights = attn_weights.softmax()?;

        // attn_weights @ v: (b, heads, s_q, head_dim)
        let attn_out = attn_weights.matmul(&v)?;

        // Reshape back: (b, heads, s_q, head_dim) -> (b, s_q, d_model)
        let attn_out = attn_out.transpose(1, 2)?.contiguous()?;
        let attn_out = attn_out.reshape((b, s, d_model))?;

        // Output projection
        let out = attn_out.matmul_t(&self.out_proj_weight)?;
        let out = if let Some(bias) = &self.out_proj_bias {
            out.broadcast_add(bias)?
        } else {
            out
        };

        Ok(out)
    }

    pub fn reset_state(&mut self) {
        self.kv_cache.reset();
    }
}

/// Layer scale: learnable per-channel scaling.
pub struct LayerScale {
    scale: CpuTensor<f32>,
}

impl LayerScale {
    pub fn load(vb: &Path<CpuDevice>, dim: usize) -> Result<Self> {
        let scale: CpuTensor<f32> = vb.tensor("scale", (dim,))?;
        Ok(Self { scale })
    }

    pub fn forward(&self, xs: &CpuTensor<f32>) -> Result<CpuTensor<f32>> {
        Ok(xs.broadcast_mul(&self.scale)?)
    }
}

/// Layer normalization with weight and bias.
pub struct LayerNorm {
    weight: CpuTensor<f32>,
    bias: CpuTensor<f32>,
    eps: f32,
}

impl LayerNorm {
    pub fn load(vb: &Path<CpuDevice>, dim: usize) -> Result<Self> {
        let weight: CpuTensor<f32> = vb.tensor("weight", (dim,))?;
        let bias: CpuTensor<f32> = vb.tensor("bias", (dim,))?;
        Ok(Self { weight, bias, eps: 1e-5 })
    }

    pub fn forward(&self, xs: &CpuTensor<f32>) -> Result<CpuTensor<f32>> {
        Ok(xs.layer_norm(&self.weight, &self.bias, self.eps)?)
    }
}

/// Single transformer layer: norm1 → attn → (layer_scale) → add → norm2 → ffn → (layer_scale) → add
pub struct TransformerLayer {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    linear1_weight: CpuTensor<f32>,
    linear1_bias: Option<CpuTensor<f32>>,
    linear2_weight: CpuTensor<f32>,
    linear2_bias: Option<CpuTensor<f32>>,
    layer_scale_1: Option<LayerScale>,
    layer_scale_2: Option<LayerScale>,
}

impl TransformerLayer {
    pub fn load(
        vb: &Path<CpuDevice>,
        d_model: usize,
        num_heads: usize,
        dim_feedforward: usize,
        max_seq_len: usize,
        use_layer_scale: bool,
    ) -> Result<Self> {
        let norm1 = LayerNorm::load(&vb.pp("norm1"), d_model)?;
        let attn = Attention::load(vb, d_model, num_heads, max_seq_len)?;
        let norm2 = LayerNorm::load(&vb.pp("norm2"), d_model)?;

        let linear1_weight: CpuTensor<f32> =
            vb.tensor("linear1.weight", (dim_feedforward, d_model))?;
        let linear1_bias = if vb.contains("linear1.bias") {
            Some(vb.tensor("linear1.bias", (dim_feedforward,))?)
        } else {
            None
        };
        let linear2_weight: CpuTensor<f32> =
            vb.tensor("linear2.weight", (d_model, dim_feedforward))?;
        let linear2_bias = if vb.contains("linear2.bias") {
            Some(vb.tensor("linear2.bias", (d_model,))?)
        } else {
            None
        };

        let layer_scale_1 = if use_layer_scale {
            Some(LayerScale::load(&vb.pp("layer_scale_1"), d_model)?)
        } else {
            None
        };
        let layer_scale_2 = if use_layer_scale {
            Some(LayerScale::load(&vb.pp("layer_scale_2"), d_model)?)
        } else {
            None
        };

        Ok(Self {
            norm1,
            attn,
            norm2,
            linear1_weight,
            linear1_bias,
            linear2_weight,
            linear2_bias,
            layer_scale_1,
            layer_scale_2,
        })
    }

    pub fn forward(
        &mut self,
        xs: &CpuTensor<f32>,
        rope: &RotaryEmbedding,
    ) -> Result<CpuTensor<f32>> {
        // Self-attention block
        let residual = xs;
        let h = self.norm1.forward(xs)?;
        let h = self.attn.forward(&h, rope)?;
        let h = if let Some(ls) = &self.layer_scale_1 { ls.forward(&h)? } else { h };
        let xs = residual.add(&h)?;

        // FFN block
        let residual = &xs;
        let h = self.norm2.forward(&xs)?;
        let h = h.matmul_t(&self.linear1_weight)?;
        let h = if let Some(bias) = &self.linear1_bias {
            h.broadcast_add(bias)?
        } else {
            h
        };
        let h = h.gelu_erf()?;
        let h = h.matmul_t(&self.linear2_weight)?;
        let h = if let Some(bias) = &self.linear2_bias {
            h.broadcast_add(bias)?
        } else {
            h
        };
        let h = if let Some(ls) = &self.layer_scale_2 { ls.forward(&h)? } else { h };
        let xs = residual.add(&h)?;

        Ok(xs)
    }

    pub fn reset_state(&mut self) {
        self.attn.reset_state();
    }
}

/// Full transformer: multiple layers + shared RoPE.
pub struct Transformer {
    layers: Vec<TransformerLayer>,
    rope: RotaryEmbedding,
}

impl Transformer {
    pub fn load(
        vb: &Path<CpuDevice>,
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        dim_feedforward: usize,
        max_seq_len: usize,
        max_period: f32,
        use_layer_scale: bool,
    ) -> Result<Self> {
        let head_dim = d_model / num_heads;
        let rope = RotaryEmbedding::new(head_dim, 8192, max_period)?;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer = TransformerLayer::load(
                &vb.pp(format!("layers.{i}")),
                d_model,
                num_heads,
                dim_feedforward,
                max_seq_len,
                use_layer_scale,
            )?;
            layers.push(layer);
        }

        Ok(Self { layers, rope })
    }

    /// xs: (batch, seq, d_model) -> (batch, seq, d_model)
    pub fn forward(&mut self, xs: &CpuTensor<f32>) -> Result<CpuTensor<f32>> {
        let mut xs = xs.copy()?;
        for layer in &mut self.layers {
            xs = layer.forward(&xs, &self.rope)?;
        }
        Ok(xs)
    }

    pub fn reset_state(&mut self) {
        for layer in &mut self.layers {
            layer.reset_state();
        }
    }
}
