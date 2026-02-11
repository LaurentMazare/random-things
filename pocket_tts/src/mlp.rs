use mimi::nn::{Linear, var_builder::Path};
use mimi::{Backend, Result, Tensor, WithDTypeF};

fn modulate<T: WithDTypeF, B: Backend>(
    x: &Tensor<T, B>,
    shift: &Tensor<T, B>,
    scale: &Tensor<T, B>,
) -> Result<Tensor<T, B>> {
    let one_plus_scale = scale.add_scalar(T::from_f32(1.0))?;
    x.broadcast_mul(&one_plus_scale)?.broadcast_add(shift)
}

// TODO(laurent): this is not the typical RMSNorm implementation.
// Make the version in mimi-rs flexible enough so that it could be
// used here.

pub struct RMSNorm<T: WithDTypeF, B: Backend> {
    alpha: Tensor<T, B>,
    eps: f32,
}

impl<T: WithDTypeF, B: Backend> RMSNorm<T, B> {
    pub fn load(vb: &Path<B>, dim: usize) -> Result<Self> {
        let alpha = vb.tensor("alpha", (dim,))?;
        Ok(Self { alpha, eps: 1e-5 })
    }

    #[tracing::instrument(name = "rms-norm", skip_all)]
    pub fn forward(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        // Python version uses variance (not mean-of-squares):
        //   var = eps + x.var(dim=-1, keepdim=True)   # unbiased (N-1)
        //   y = x * (alpha * rsqrt(var))
        let last_dim = x.dim(mimi::D::Minus1)?;
        let n = last_dim as f32;
        let data = x.to_vec()?;
        let alpha_data = self.alpha.to_vec()?;
        let eps = self.eps;

        let mut out = vec![T::from_f32(0.0); data.len()];
        for (chunk_in, chunk_out) in data.chunks_exact(last_dim).zip(out.chunks_exact_mut(last_dim))
        {
            // Unbiased variance: sum((x - mean)^2) / (N - 1)
            let mean: f32 = chunk_in.iter().map(|&v| v.to_f32()).sum::<f32>() / n;
            let var: f32 = chunk_in
                .iter()
                .map(|&v| {
                    let d = v.to_f32() - mean;
                    d * d
                })
                .sum::<f32>()
                / (n - 1.0);
            let rsqrt = 1.0 / (var + eps).sqrt();
            for ((o, &x_val), &a) in
                chunk_out.iter_mut().zip(chunk_in.iter()).zip(alpha_data.iter())
            {
                *o = T::from_f32(x_val.to_f32() * a.to_f32() * rsqrt);
            }
        }

        Tensor::from_vec(out, x.shape().clone(), x.device())
    }
}

// ---- LayerNorm ----

pub struct LayerNorm<T: WithDTypeF, B: Backend> {
    weight: Option<Tensor<T, B>>,
    bias: Option<Tensor<T, B>>,
    eps: f32,
}

impl<T: WithDTypeF, B: Backend> LayerNorm<T, B> {
    pub fn load(vb: &Path<B>, channels: usize, elementwise_affine: bool) -> Result<Self> {
        let (weight, bias) = if elementwise_affine {
            (Some(vb.tensor("weight", (channels,))?), Some(vb.tensor("bias", (channels,))?))
        } else {
            (None, None)
        };
        Ok(Self { weight, bias, eps: 1e-6 })
    }

    #[tracing::instrument(name = "layer-norm", skip_all)]
    pub fn forward(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        match (&self.weight, &self.bias) {
            (Some(w), Some(b)) => x.layer_norm(w, b, self.eps),
            _ => {
                // Manual layer norm without affine: (x - mean) / sqrt(var + eps)
                // Use the layer_norm primitive with ones/zeros
                let dim = x.dim(mimi::D::Minus1)?;
                let dev = x.device();
                let ones_data = vec![T::from_f32(1.0); dim];
                let zeros_data = vec![T::from_f32(0.0); dim];
                let ones = Tensor::from_vec(ones_data, (dim,), dev)?;
                let zeros = Tensor::from_vec(zeros_data, (dim,), dev)?;
                x.layer_norm(&ones, &zeros, self.eps)
            }
        }
    }
}

// ---- TimestepEmbedder ----

pub struct TimestepEmbedder<T: WithDTypeF, B: Backend> {
    linear1: Linear<T, B>,
    linear2: Linear<T, B>,
    rms_norm: RMSNorm<T, B>,
    freqs: Tensor<T, B>,
    _frequency_embedding_size: usize,
}

impl<T: WithDTypeF, B: Backend> TimestepEmbedder<T, B> {
    pub fn load(vb: &Path<B>, hidden_size: usize, frequency_embedding_size: usize) -> Result<Self> {
        let mlp = vb.pp("mlp");
        let linear1 = Linear::load_b(&mlp.pp("0"), frequency_embedding_size, hidden_size)?;
        let linear2 = Linear::load_b(&mlp.pp("2"), hidden_size, hidden_size)?;
        let rms_norm = RMSNorm::load(&mlp.pp("3"), hidden_size)?;

        let freqs = vb.tensor("freqs", (frequency_embedding_size / 2,))?;

        Ok(Self {
            linear1,
            linear2,
            rms_norm,
            freqs,
            _frequency_embedding_size: frequency_embedding_size,
        })
    }

    #[tracing::instrument(name = "ts-embedder", skip_all)]
    pub fn forward(&self, t: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        // t: [..., 1] -> frequency embedding
        let args = t.broadcast_mul(&self.freqs)?;
        let cos = args.cos()?;
        let sin = args.sin()?;
        let embedding = Tensor::cat(&[&cos, &sin], embedding_last_dim(&cos)?)?;

        // MLP: linear -> silu -> linear -> rmsnorm
        let mut x = self.linear1.forward(&embedding)?;
        x = x.silu()?;
        x = self.linear2.forward(&x)?;
        x = self.rms_norm.forward(&x)?;
        Ok(x)
    }
}

fn embedding_last_dim<T: WithDTypeF, B: Backend>(t: &Tensor<T, B>) -> Result<usize> {
    Ok(t.rank() - 1)
}

// ---- ResBlock ----

pub struct ResBlock<T: WithDTypeF, B: Backend> {
    in_ln: LayerNorm<T, B>,
    mlp_linear1: Linear<T, B>,
    mlp_linear2: Linear<T, B>,
    ada_ln_silu_linear: Linear<T, B>,
}

impl<T: WithDTypeF, B: Backend> ResBlock<T, B> {
    pub fn load(vb: &Path<B>, channels: usize) -> Result<Self> {
        let in_ln = LayerNorm::load(&vb.pp("in_ln"), channels, true)?;
        let mlp = vb.pp("mlp");
        let mlp_linear1 = Linear::load_b(&mlp.pp("0"), channels, channels)?;
        let mlp_linear2 = Linear::load_b(&mlp.pp("2"), channels, channels)?;
        let ada = vb.pp("adaLN_modulation");
        let ada_ln_silu_linear = Linear::load_b(&ada.pp("1"), 3 * channels, channels)?;
        Ok(Self { in_ln, mlp_linear1, mlp_linear2, ada_ln_silu_linear })
    }

    #[tracing::instrument(name = "resblock", skip_all)]
    pub fn forward(&self, x: &Tensor<T, B>, y: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let ada = self.ada_ln_silu_linear.forward(&y.silu()?)?;
        let channels = x.dim(mimi::D::Minus1)?;
        let shift_mlp = ada.narrow(ada.rank() - 1, 0..channels)?.contiguous()?;
        let scale_mlp = ada.narrow(ada.rank() - 1, channels..2 * channels)?.contiguous()?;
        let gate_mlp = ada.narrow(ada.rank() - 1, 2 * channels..3 * channels)?.contiguous()?;

        // h = modulate(ln(x), shift, scale)
        let h = self.in_ln.forward(x)?;
        let h = modulate(&h, &shift_mlp, &scale_mlp)?;

        // MLP
        let h = self.mlp_linear1.forward(&h)?;
        let h = h.silu()?;
        let h = self.mlp_linear2.forward(&h)?;
        x.add(&gate_mlp.broadcast_mul(&h)?)
    }
}

// ---- FinalLayer ----

pub struct FinalLayer<T: WithDTypeF, B: Backend> {
    norm_final: LayerNorm<T, B>,
    linear_weight: Tensor<T, B>,
    linear_bias: Tensor<T, B>,
    ada_ln_silu_linear_weight: Tensor<T, B>,
    ada_ln_silu_linear_bias: Tensor<T, B>,
}

impl<T: WithDTypeF, B: Backend> FinalLayer<T, B> {
    pub fn load(vb: &Path<B>, model_channels: usize, out_channels: usize) -> Result<Self> {
        let norm_final = LayerNorm::load(&vb.pp("norm_final"), model_channels, false)?;
        let linear_weight = vb.pp("linear").tensor("weight", (out_channels, model_channels))?;
        let linear_bias = vb.pp("linear").tensor("bias", (out_channels,))?;

        let ada = vb.pp("adaLN_modulation");
        let ada_ln_silu_linear_weight =
            ada.pp("1").tensor("weight", (2 * model_channels, model_channels))?;
        let ada_ln_silu_linear_bias = ada.pp("1").tensor("bias", (2 * model_channels,))?;

        Ok(Self {
            norm_final,
            linear_weight,
            linear_bias,
            ada_ln_silu_linear_weight,
            ada_ln_silu_linear_bias,
        })
    }

    pub fn forward(&self, x: &Tensor<T, B>, c: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let ada = c.silu()?.matmul_t(&self.ada_ln_silu_linear_weight)?;
        let ada = ada.broadcast_add(&self.ada_ln_silu_linear_bias)?;
        let model_channels = x.dim(mimi::D::Minus1)?;
        let shift = ada.narrow(ada.rank() - 1, 0..model_channels)?.contiguous()?;
        let scale = ada.narrow(ada.rank() - 1, model_channels..2 * model_channels)?.contiguous()?;

        let x = self.norm_final.forward(x)?;
        let x = modulate(&x, &shift, &scale)?;
        let x = x.matmul_t(&self.linear_weight)?;
        x.broadcast_add(&self.linear_bias)
    }
}

// ---- SimpleMLPAdaLN ----

pub struct SimpleMLPAdaLN<T: WithDTypeF, B: Backend> {
    time_embeds: Vec<TimestepEmbedder<T, B>>,
    cond_embed_weight: Tensor<T, B>,
    cond_embed_bias: Tensor<T, B>,
    input_proj_weight: Tensor<T, B>,
    input_proj_bias: Tensor<T, B>,
    res_blocks: Vec<ResBlock<T, B>>,
    final_layer: FinalLayer<T, B>,
    pub num_time_conds: usize,
}

impl<T: WithDTypeF, B: Backend> SimpleMLPAdaLN<T, B> {
    pub fn load(
        vb: &Path<B>,
        in_channels: usize,
        model_channels: usize,
        out_channels: usize,
        cond_channels: usize,
        num_res_blocks: usize,
        num_time_conds: usize,
    ) -> Result<Self> {
        let mut time_embeds = Vec::new();
        for i in 0..num_time_conds {
            time_embeds.push(TimestepEmbedder::load(
                &vb.pp("time_embed").pp(i),
                model_channels,
                256,
            )?);
        }

        let cond_embed_weight =
            vb.pp("cond_embed").tensor("weight", (model_channels, cond_channels))?;
        let cond_embed_bias = vb.pp("cond_embed").tensor("bias", (model_channels,))?;

        let input_proj_weight =
            vb.pp("input_proj").tensor("weight", (model_channels, in_channels))?;
        let input_proj_bias = vb.pp("input_proj").tensor("bias", (model_channels,))?;

        let mut res_blocks = Vec::new();
        for i in 0..num_res_blocks {
            res_blocks.push(ResBlock::load(&vb.pp("res_blocks").pp(i), model_channels)?);
        }

        let final_layer = FinalLayer::load(&vb.pp("final_layer"), model_channels, out_channels)?;

        Ok(Self {
            time_embeds,
            cond_embed_weight,
            cond_embed_bias,
            input_proj_weight,
            input_proj_bias,
            res_blocks,
            final_layer,
            num_time_conds,
        })
    }

    /// Forward pass.
    /// c: conditioning from AR transformer
    /// s: start time tensor
    /// t: target time tensor
    /// x: input tensor [N, C]
    #[tracing::instrument(name = "mlp-adaln", skip_all)]
    pub fn forward(
        &self,
        c: &Tensor<T, B>,
        s: &Tensor<T, B>,
        t: &Tensor<T, B>,
        x: &Tensor<T, B>,
    ) -> Result<Tensor<T, B>> {
        // input_proj(x)
        let mut x = x.matmul_t(&self.input_proj_weight)?;
        x = x.broadcast_add(&self.input_proj_bias)?;

        // Combine time conditions: average of time embeddings
        let ts = [s, t];
        let mut t_combined = self.time_embeds[0].forward(ts[0])?;
        for (embed, &t_input) in self.time_embeds[1..].iter().zip(ts[1..].iter()) {
            t_combined = t_combined.add(&embed.forward(t_input)?)?;
        }
        let scale = T::from_f32(1.0 / self.num_time_conds as f32);
        t_combined = t_combined.scale(scale)?;

        // cond_embed(c) + t_combined
        let c = c.matmul_t(&self.cond_embed_weight)?;
        let c = c.broadcast_add(&self.cond_embed_bias)?;
        let y = t_combined.add(&c)?;

        // Res blocks
        for block in &self.res_blocks {
            x = block.forward(&x, &y)?;
        }

        // Final layer
        self.final_layer.forward(&x, &y)
    }
}
