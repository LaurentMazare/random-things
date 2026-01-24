use crate::nn::{Linear, RmsNorm};
use crate::{Result, Tensor, WithDTypeF};

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl Config {
    /// Llama 3 8B configuration
    pub fn llama3_8b() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 14336,
            vocab_size: 128256,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            max_position_embeddings: 8192,
        }
    }

    /// TinyLlama 1.1B configuration
    /// https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
    pub fn tiny_llama_1_1b() -> Self {
        Self {
            hidden_size: 2048,
            intermediate_size: 5632,
            vocab_size: 32000,
            num_hidden_layers: 22,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            head_dim: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 2048,
        }
    }

    /// SmolLM 135M configuration
    /// https://huggingface.co/HuggingFaceTB/SmolLM-135M
    pub fn smol_lm_135m() -> Self {
        Self {
            hidden_size: 576,
            intermediate_size: 1536,
            vocab_size: 49152,
            num_hidden_layers: 30,
            num_attention_heads: 9,
            num_key_value_heads: 3,
            head_dim: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 2048,
        }
    }

    /// SmolLM 360M configuration
    /// https://huggingface.co/HuggingFaceTB/SmolLM-360M
    pub fn smol_lm_360m() -> Self {
        Self {
            hidden_size: 960,
            intermediate_size: 2560,
            vocab_size: 49152,
            num_hidden_layers: 32,
            num_attention_heads: 15,
            num_key_value_heads: 5,
            head_dim: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 2048,
        }
    }

    /// Tiny test configuration for quick testing (only ~1M params)
    pub fn tiny_test() -> Self {
        Self {
            hidden_size: 64,
            intermediate_size: 128,
            vocab_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 512,
        }
    }
}

pub struct Attention<T: WithDTypeF> {
    q_proj: Linear<T>,
    k_proj: Linear<T>,
    v_proj: Linear<T>,
    o_proj: Linear<T>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_kv_groups: usize,
}

impl<T: WithDTypeF> Attention<T> {
    pub fn new(
        q_proj: Linear<T>,
        k_proj: Linear<T>,
        v_proj: Linear<T>,
        o_proj: Linear<T>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let num_kv_groups = num_heads / num_kv_heads;
        Self { q_proj, k_proj, v_proj, o_proj, num_heads, num_kv_heads, head_dim, num_kv_groups }
    }

    pub fn forward(
        &self,
        x: &Tensor<T>,
        cos: &Tensor<T>,
        sin: &Tensor<T>,
        pos: usize,
        kv_cache: Option<(&Tensor<T>, &Tensor<T>)>,
    ) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>)> {
        let (b, seq_len, _hidden) = x.dims3()?;

        // Project to Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape: (b, seq_len, num_heads * head_dim) -> (b, num_heads, seq_len, head_dim)
        let q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?;
        let q = q.transpose(1, 2)?;

        let k = k.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?;
        let k = k.transpose(1, 2)?;

        let v = v.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.transpose(1, 2)?;

        // Apply RoPE
        let q = q.rope(cos, sin, pos)?;
        let k = k.rope(cos, sin, pos)?;

        // Handle KV cache - cache BEFORE repeat_kv to save memory
        let (k_cache, v_cache, k, v) = match kv_cache {
            Some((prev_k, prev_v)) => {
                let k_cat = Tensor::cat(&[prev_k, &k], 2)?;
                let v_cat = Tensor::cat(&[prev_v, &v], 2)?;
                (k_cat.copy()?, v_cat.copy()?, k_cat, v_cat)
            }
            None => (k.copy()?, v.copy()?, k, v),
        };

        // Repeat KV heads for grouped query attention
        let k = self.repeat_kv(&k)?;
        let v = self.repeat_kv(&v)?;

        // Scaled dot-product attention
        // Q: (b, num_heads, seq_len, head_dim)
        // K: (b, num_heads, kv_len, head_dim)
        // V: (b, num_heads, kv_len, head_dim)
        let scale = T::from_f32(1.0 / (self.head_dim as f32).sqrt());
        let k_t = k.transpose(2, 3)?;
        let attn_weights = q.matmul(&k_t)?;
        let attn_weights = attn_weights.scale(scale)?;

        // Apply causal mask and softmax
        let attn_weights = attn_weights.softmax_causal(pos)?;

        // Attention output
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: (b, num_heads, seq_len, head_dim) -> (b, seq_len, hidden_size)
        let attn_output = attn_output.transpose(1, 2)?;
        let attn_output = attn_output.reshape((b, seq_len, self.num_heads * self.head_dim))?;

        // Output projection
        let output = self.o_proj.forward(&attn_output)?;

        Ok((output, k_cache, v_cache))
    }

    fn repeat_kv(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        if self.num_kv_groups == 1 {
            return x.copy();
        }
        // x: (b, num_kv_heads, seq_len, head_dim)
        // output: (b, num_heads, seq_len, head_dim)
        // Each KV head is repeated num_kv_groups times
        let (b, num_kv_heads, seq_len, head_dim) = x.dims4()?;
        let num_heads = num_kv_heads * self.num_kv_groups;

        let mut out =
            unsafe { Tensor::alloc_uninit((b, num_heads, seq_len, head_dim).into()) };

        let head_size = seq_len * head_dim;
        for b_idx in 0..b {
            for kv_head in 0..num_kv_heads {
                let src_start = (b_idx * num_kv_heads + kv_head) * head_size;
                let src = &x.data[src_start..src_start + head_size];

                // Repeat this KV head num_kv_groups times
                for g in 0..self.num_kv_groups {
                    let dst_head = kv_head * self.num_kv_groups + g;
                    let dst_start = (b_idx * num_heads + dst_head) * head_size;
                    out.data[dst_start..dst_start + head_size].copy_from_slice(src);
                }
            }
        }

        Ok(out)
    }
}

pub struct Mlp<T: WithDTypeF> {
    gate_proj: Linear<T>,
    up_proj: Linear<T>,
    down_proj: Linear<T>,
}

impl<T: WithDTypeF> Mlp<T> {
    pub fn new(gate_proj: Linear<T>, up_proj: Linear<T>, down_proj: Linear<T>) -> Self {
        Self { gate_proj, up_proj, down_proj }
    }

    pub fn forward(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        // SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        let gate = self.gate_proj.forward(x)?;
        let gate = gate.silu()?;
        let up = self.up_proj.forward(x)?;
        let hidden = gate.mul(&up)?;
        self.down_proj.forward(&hidden)
    }
}

pub struct TransformerBlock<T: WithDTypeF> {
    attn: Attention<T>,
    mlp: Mlp<T>,
    input_layernorm: RmsNorm<T>,
    post_attention_layernorm: RmsNorm<T>,
}

impl<T: WithDTypeF> TransformerBlock<T> {
    pub fn new(
        attn: Attention<T>,
        mlp: Mlp<T>,
        input_layernorm: RmsNorm<T>,
        post_attention_layernorm: RmsNorm<T>,
    ) -> Self {
        Self { attn, mlp, input_layernorm, post_attention_layernorm }
    }

    pub fn forward(
        &self,
        x: &Tensor<T>,
        cos: &Tensor<T>,
        sin: &Tensor<T>,
        pos: usize,
        kv_cache: Option<(&Tensor<T>, &Tensor<T>)>,
    ) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>)> {
        // Pre-norm architecture
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let (attn_out, k_cache, v_cache) = self.attn.forward(&x, cos, sin, pos, kv_cache)?;
        let x = residual.add(&attn_out)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&x)?;
        let x = residual.add(&mlp_out)?;

        Ok((x, k_cache, v_cache))
    }
}

pub struct Llama<T: WithDTypeF> {
    embed_tokens: Tensor<T>,
    layers: Vec<TransformerBlock<T>>,
    norm: RmsNorm<T>,
    lm_head: Linear<T>,
    cos_cache: Tensor<T>,
    sin_cache: Tensor<T>,
}

pub struct KvCache<T: WithDTypeF> {
    kvs: Vec<(Tensor<T>, Tensor<T>)>,
}

impl<T: WithDTypeF> Llama<T> {
    pub fn new(
        embed_tokens: Tensor<T>,
        layers: Vec<TransformerBlock<T>>,
        norm: RmsNorm<T>,
        lm_head: Linear<T>,
        cos_cache: Tensor<T>,
        sin_cache: Tensor<T>,
    ) -> Self {
        Self { embed_tokens, layers, norm, lm_head, cos_cache, sin_cache }
    }

    pub fn forward(
        &self,
        tokens: &[usize],
        pos: usize,
        kv_caches: Option<&KvCache<T>>,
    ) -> Result<(Tensor<T>, KvCache<T>)> {
        // Token embedding: (seq_len,) -> (1, seq_len, hidden_size)
        let mut x = self.embed_tokens.index_select(tokens, 0)?;
        x = x.reshape((1, tokens.len(), ()))?;

        // Run through transformer layers
        let mut kvs = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            let kv_cache = kv_caches.map(|c| (&c.kvs[i].0, &c.kvs[i].1));
            let (new_x, k_cache, v_cache) =
                layer.forward(&x, &self.cos_cache, &self.sin_cache, pos, kv_cache)?;
            x = new_x;
            kvs.push((k_cache, v_cache));
        }

        // Final norm
        let x = self.norm.forward(&x)?;

        // LM head: (1, seq_len, hidden_size) -> (1, seq_len, vocab_size)
        let logits = self.lm_head.forward(&x)?;

        Ok((logits, KvCache { kvs }))
    }
}

pub fn precompute_freqs_cis<T: WithDTypeF>(
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
) -> Result<(Tensor<T>, Tensor<T>)> {
    let half_dim = head_dim / 2;
    let mut freqs = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
        freqs.push(freq);
    }

    let mut cos_data = Vec::with_capacity(max_seq_len * half_dim);
    let mut sin_data = Vec::with_capacity(max_seq_len * half_dim);

    for pos in 0..max_seq_len {
        for &freq in &freqs {
            let angle = pos as f32 * freq;
            cos_data.push(T::from_f32(angle.cos()));
            sin_data.push(T::from_f32(angle.sin()));
        }
    }

    let shape = (max_seq_len, half_dim).into();
    let cos = Tensor { data: cos_data, shape: crate::Shape::from((max_seq_len, half_dim)) };
    let sin = Tensor { data: sin_data, shape };

    Ok((cos, sin))
}
