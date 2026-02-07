use anyhow::Result;
use mimi::nn::var_builder::Path;
use mimi::{CpuDevice, CpuTensor};

/// Timestep embedder: sinusoidal frequencies → MLP → RMSNorm
pub struct TimestepEmbedder {
    /// Precomputed frequency buffer (half_dim,)
    freqs: CpuTensor<f32>,
    mlp_0_weight: CpuTensor<f32>,
    mlp_0_bias: CpuTensor<f32>,
    mlp_2_weight: CpuTensor<f32>,
    mlp_2_bias: CpuTensor<f32>,
    norm_weight: CpuTensor<f32>,
}

impl TimestepEmbedder {
    pub fn load(vb: &Path<CpuDevice>, hidden_size: usize) -> Result<Self> {
        let frequency_dim = 256;
        let half_dim = frequency_dim / 2;

        // Load precomputed frequencies from checkpoint
        let freqs: CpuTensor<f32> = vb.tensor("freqs", (half_dim,))?;

        let mlp_0_weight: CpuTensor<f32> =
            vb.tensor("mlp.0.weight", (hidden_size, frequency_dim))?;
        let mlp_0_bias: CpuTensor<f32> = vb.tensor("mlp.0.bias", (hidden_size,))?;
        let mlp_2_weight: CpuTensor<f32> =
            vb.tensor("mlp.2.weight", (hidden_size, hidden_size))?;
        let mlp_2_bias: CpuTensor<f32> = vb.tensor("mlp.2.bias", (hidden_size,))?;
        // RMSNorm weight stored as "mlp.3.alpha" in the checkpoint
        let norm_weight: CpuTensor<f32> = vb.tensor("mlp.3.alpha", (hidden_size,))?;

        Ok(Self { freqs, mlp_0_weight, mlp_0_bias, mlp_2_weight, mlp_2_bias, norm_weight })
    }

    /// Embed a single timestep scalar into a vector of size hidden_size.
    /// Returns (1, hidden_size).
    pub fn forward(&self, t: f32) -> Result<CpuTensor<f32>> {
        // t * freqs -> (half_dim,)
        let t_freqs = self.freqs.scale(t)?;

        // Concat [cos(t*f), sin(t*f)] -> (frequency_dim,)
        let cos = t_freqs.cos()?;
        let sin = t_freqs.sin()?;
        let emb = CpuTensor::cat(&[&cos, &sin], 0)?;

        // Reshape to (1, frequency_dim) for matmul
        let emb = emb.unsqueeze(0)?;

        // MLP: Linear -> SiLU -> Linear
        let h = emb.matmul_t(&self.mlp_0_weight)?;
        let h = h.broadcast_add(&self.mlp_0_bias)?;
        let h = h.silu()?;
        let h = h.matmul_t(&self.mlp_2_weight)?;
        let h = h.broadcast_add(&self.mlp_2_bias)?;

        // RMSNorm (eps=1e-5 matching Python's RMSNorm default)
        Ok(h.rms_norm(&self.norm_weight, 1e-5)?)
    }
}

/// AdaLN residual block: LayerNorm → modulate → Linear → SiLU → Linear → gate * h + x
pub struct ResBlock {
    norm_weight: CpuTensor<f32>,
    norm_bias: CpuTensor<f32>,
    linear1_weight: CpuTensor<f32>,
    linear1_bias: CpuTensor<f32>,
    linear2_weight: CpuTensor<f32>,
    linear2_bias: CpuTensor<f32>,
    adaln_weight: CpuTensor<f32>,
    adaln_bias: CpuTensor<f32>,
}

impl ResBlock {
    pub fn load(vb: &Path<CpuDevice>, dim: usize) -> Result<Self> {
        // Checkpoint uses "in_ln" for the LayerNorm
        let norm_weight: CpuTensor<f32> = vb.tensor("in_ln.weight", (dim,))?;
        let norm_bias: CpuTensor<f32> = vb.tensor("in_ln.bias", (dim,))?;
        // Checkpoint uses "mlp.0" / "mlp.2" for the two linear layers
        let linear1_weight: CpuTensor<f32> = vb.tensor("mlp.0.weight", (dim, dim))?;
        let linear1_bias: CpuTensor<f32> = vb.tensor("mlp.0.bias", (dim,))?;
        let linear2_weight: CpuTensor<f32> = vb.tensor("mlp.2.weight", (dim, dim))?;
        let linear2_bias: CpuTensor<f32> = vb.tensor("mlp.2.bias", (dim,))?;
        // Checkpoint uses "adaLN_modulation.1" for the AdaLN linear
        let adaln_weight: CpuTensor<f32> =
            vb.tensor("adaLN_modulation.1.weight", (3 * dim, dim))?;
        let adaln_bias: CpuTensor<f32> =
            vb.tensor("adaLN_modulation.1.bias", (3 * dim,))?;
        Ok(Self {
            norm_weight,
            norm_bias,
            linear1_weight,
            linear1_bias,
            linear2_weight,
            linear2_bias,
            adaln_weight,
            adaln_bias,
        })
    }

    /// x: (1, dim), cond: (1, dim)
    pub fn forward(&self, x: &CpuTensor<f32>, cond: &CpuTensor<f32>) -> Result<CpuTensor<f32>> {
        let dim = x.dims()[1];

        // Compute modulation params from conditioning
        let mod_params = cond.silu()?;
        let mod_params = mod_params.matmul_t(&self.adaln_weight)?;
        let mod_params = mod_params.broadcast_add(&self.adaln_bias)?;

        // Split into shift, scale, gate (each dim)
        let shift = mod_params.narrow(1, ..dim)?.contiguous()?;
        let scale = mod_params.narrow(1, dim..2 * dim)?.contiguous()?;
        let gate = mod_params.narrow(1, 2 * dim..3 * dim)?.contiguous()?;

        // LayerNorm → modulate
        let h = x.layer_norm(&self.norm_weight, &self.norm_bias, 1e-5)?;
        // modulate: h * (1 + scale) + shift
        let ones = CpuTensor::<f32>::full(1.0, scale.dims(), &mimi::CPU)?;
        let h = h.broadcast_mul(&ones.add(&scale)?)?.broadcast_add(&shift)?;

        // Linear → SiLU → Linear
        let h = h.matmul_t(&self.linear1_weight)?;
        let h = h.broadcast_add(&self.linear1_bias)?;
        let h = h.silu()?;
        let h = h.matmul_t(&self.linear2_weight)?;
        let h = h.broadcast_add(&self.linear2_bias)?;

        // gate * h + x
        let h = h.broadcast_mul(&gate)?;
        Ok(x.add(&h)?)
    }
}

/// Final layer: LayerNorm(no affine) → modulate → Linear(dim → out_dim)
pub struct FinalLayer {
    linear_weight: CpuTensor<f32>,
    linear_bias: CpuTensor<f32>,
    adaln_weight: CpuTensor<f32>,
    adaln_bias: CpuTensor<f32>,
    dim: usize,
}

impl FinalLayer {
    pub fn load(vb: &Path<CpuDevice>, dim: usize, out_dim: usize) -> Result<Self> {
        let linear_weight: CpuTensor<f32> = vb.tensor("linear.weight", (out_dim, dim))?;
        let linear_bias: CpuTensor<f32> = vb.tensor("linear.bias", (out_dim,))?;
        // Checkpoint uses "adaLN_modulation.1" for the AdaLN linear
        let adaln_weight: CpuTensor<f32> =
            vb.tensor("adaLN_modulation.1.weight", (2 * dim, dim))?;
        let adaln_bias: CpuTensor<f32> =
            vb.tensor("adaLN_modulation.1.bias", (2 * dim,))?;
        Ok(Self { linear_weight, linear_bias, adaln_weight, adaln_bias, dim })
    }

    /// x: (1, dim), cond: (1, dim) → (1, out_dim)
    pub fn forward(&self, x: &CpuTensor<f32>, cond: &CpuTensor<f32>) -> Result<CpuTensor<f32>> {
        let dim = self.dim;

        // Compute modulation
        let mod_params = cond.silu()?;
        let mod_params = mod_params.matmul_t(&self.adaln_weight)?;
        let mod_params = mod_params.broadcast_add(&self.adaln_bias)?;

        let shift = mod_params.narrow(1, ..dim)?.contiguous()?;
        let scale = mod_params.narrow(1, dim..2 * dim)?.contiguous()?;

        // LayerNorm without affine parameters
        let h = layer_norm_no_affine(x, 1e-5)?;

        // Modulate: h * (1 + scale) + shift
        let ones = CpuTensor::<f32>::full(1.0, scale.dims(), &mimi::CPU)?;
        let h = h.broadcast_mul(&ones.add(&scale)?)?.broadcast_add(&shift)?;

        // Final linear
        let h = h.matmul_t(&self.linear_weight)?;
        Ok(h.broadcast_add(&self.linear_bias)?)
    }
}

/// Layer norm without affine (weight=1, bias=0).
fn layer_norm_no_affine(x: &CpuTensor<f32>, eps: f32) -> Result<CpuTensor<f32>> {
    let dim = *x.dims().last().unwrap();
    let dev = &mimi::CPU;
    let ones = CpuTensor::<f32>::full(1.0, (dim,), dev)?;
    let zeros = CpuTensor::<f32>::zeros((dim,), dev)?;
    Ok(x.layer_norm(&ones, &zeros, eps)?)
}

/// SimpleMLPAdaLN: flow-matching MLP with AdaLN conditioning.
pub struct SimpleMLPAdaLN {
    input_proj_weight: CpuTensor<f32>,
    input_proj_bias: CpuTensor<f32>,
    cond_embed_weight: CpuTensor<f32>,
    cond_embed_bias: CpuTensor<f32>,
    time_embeds: Vec<TimestepEmbedder>,
    res_blocks: Vec<ResBlock>,
    final_layer: FinalLayer,
}

impl SimpleMLPAdaLN {
    pub fn load(
        vb: &Path<CpuDevice>,
        input_dim: usize,
        hidden_dim: usize,
        cond_dim: usize,
        depth: usize,
        num_timestep_conds: usize,
        output_dim: usize,
    ) -> Result<Self> {
        let input_proj_weight: CpuTensor<f32> =
            vb.tensor("input_proj.weight", (hidden_dim, input_dim))?;
        let input_proj_bias: CpuTensor<f32> = vb.tensor("input_proj.bias", (hidden_dim,))?;
        let cond_embed_weight: CpuTensor<f32> =
            vb.tensor("cond_embed.weight", (hidden_dim, cond_dim))?;
        let cond_embed_bias: CpuTensor<f32> = vb.tensor("cond_embed.bias", (hidden_dim,))?;

        let mut time_embeds = Vec::with_capacity(num_timestep_conds);
        for i in 0..num_timestep_conds {
            time_embeds.push(TimestepEmbedder::load(
                &vb.pp(format!("time_embed.{i}")),
                hidden_dim,
            )?);
        }

        let mut res_blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            res_blocks.push(ResBlock::load(&vb.pp(format!("res_blocks.{i}")), hidden_dim)?);
        }

        let final_layer = FinalLayer::load(&vb.pp("final_layer"), hidden_dim, output_dim)?;

        Ok(Self {
            input_proj_weight,
            input_proj_bias,
            cond_embed_weight,
            cond_embed_bias,
            time_embeds,
            res_blocks,
            final_layer,
        })
    }

    /// Run flow MLP.
    /// x: (1, input_dim), cond: (1, cond_dim), timesteps: &[f32] (one per timestep embedder)
    /// Returns (1, output_dim)
    pub fn forward(
        &self,
        x: &CpuTensor<f32>,
        cond: &CpuTensor<f32>,
        timesteps: &[f32],
    ) -> Result<CpuTensor<f32>> {
        // Project input
        let mut h = x.matmul_t(&self.input_proj_weight)?;
        h = h.broadcast_add(&self.input_proj_bias)?;

        // Compute timestep embeddings and average them
        let mut time_emb: Option<CpuTensor<f32>> = None;
        for (i, te) in self.time_embeds.iter().enumerate() {
            let emb = te.forward(timesteps[i])?;
            time_emb = Some(match time_emb {
                Some(prev) => prev.add(&emb)?,
                None => emb,
            });
        }
        let time_emb = time_emb.unwrap();
        let num_conds = self.time_embeds.len() as f32;
        let time_emb = time_emb.scale(1.0 / num_conds)?;

        // Compute conditioning embedding
        let cond_emb = cond.matmul_t(&self.cond_embed_weight)?;
        let cond_emb = cond_emb.broadcast_add(&self.cond_embed_bias)?;

        // Combined conditioning = time_emb + cond_emb
        let combined_cond = time_emb.add(&cond_emb)?;

        // Run through residual blocks (conditioning is passed separately, NOT added to x)
        for block in &self.res_blocks {
            h = block.forward(&h, &combined_cond)?;
        }

        // Final layer
        self.final_layer.forward(&h, &combined_cond)
    }
}
