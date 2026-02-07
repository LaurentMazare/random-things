use anyhow::Result;
use mimi::nn::var_builder::Path;
use mimi::{CpuDevice, CpuTensor};

use crate::flow_mlp::SimpleMLPAdaLN;
use crate::transformer::{LayerNorm, Transformer};

/// FlowLM: text-conditioned latent generation with flow matching.
pub struct FlowLMModel {
    /// Embedding mean for latent normalization
    emb_mean: CpuTensor<f32>,
    /// Embedding std for latent normalization
    emb_std: CpuTensor<f32>,
    /// BOS embedding (beginning of sequence)
    bos_emb: CpuTensor<f32>,
    /// Text token embedding table [vocab_size+1, d_model]
    conditioner_embed_weight: CpuTensor<f32>,
    /// Projects latent dim → d_model
    input_linear_weight: CpuTensor<f32>,
    /// Main transformer
    transformer: Transformer,
    /// Output normalization
    out_norm: LayerNorm,
    /// EOS prediction head
    out_eos_weight: CpuTensor<f32>,
    out_eos_bias: CpuTensor<f32>,
    /// Flow network for LSD decode
    flow_net: SimpleMLPAdaLN,
    /// Speaker projection weight [d_model, speaker_dim]
    /// Used for voice cloning from raw audio (not needed for pre-projected voice embeddings)
    #[allow(dead_code)]
    speaker_proj_weight: CpuTensor<f32>,

    // Config
    d_model: usize,
    latent_dim: usize,
}

impl FlowLMModel {
    pub fn load(vb: &Path<CpuDevice>) -> Result<Self> {
        // Config values from b6369a24.yaml
        let d_model = 1024;
        let num_heads = 16;
        let num_layers = 6;
        let hidden_scale = 4;
        let dim_feedforward = d_model * hidden_scale; // 4096
        let latent_dim = 32;
        let vocab_size = 4001; // 4000 + 1 for padding/EOS
        let speaker_dim = 512;
        let max_period = 10000.0;

        // Flow MLP config
        let flow_depth = 6;
        let flow_dim = 512;
        let num_timestep_conds = 2;

        let emb_mean: CpuTensor<f32> = vb.tensor("emb_mean", (latent_dim,))?;
        let emb_std: CpuTensor<f32> = vb.tensor("emb_std", (latent_dim,))?;
        let bos_emb: CpuTensor<f32> = vb.tensor("bos_emb", (latent_dim,))?;

        let conditioner_embed_weight: CpuTensor<f32> =
            vb.tensor("conditioner.embed.weight", (vocab_size, d_model))?;
        let input_linear_weight: CpuTensor<f32> =
            vb.tensor("input_linear.weight", (d_model, latent_dim))?;

        let transformer = Transformer::load(
            &vb.pp("transformer"),
            d_model,
            num_heads,
            num_layers,
            dim_feedforward,
            0, // unlimited context for FlowLM
            max_period,
            false, // no layer scale
        )?;

        let out_norm = LayerNorm::load(&vb.pp("out_norm"), d_model)?;

        let out_eos_weight: CpuTensor<f32> = vb.tensor("out_eos.weight", (1, d_model))?;
        let out_eos_bias: CpuTensor<f32> = vb.tensor("out_eos.bias", (1,))?;

        let flow_net = SimpleMLPAdaLN::load(
            &vb.pp("flow_net"),
            latent_dim,    // input_dim
            flow_dim,      // hidden_dim
            d_model,       // cond_dim
            flow_depth,    // depth
            num_timestep_conds,
            latent_dim,    // output_dim
        )?;

        let speaker_proj_weight: CpuTensor<f32> =
            vb.tensor("speaker_proj_weight", (d_model, speaker_dim))?;

        Ok(Self {
            emb_mean,
            emb_std,
            bos_emb,
            conditioner_embed_weight,
            input_linear_weight,
            transformer,
            out_norm,
            out_eos_weight,
            out_eos_bias,
            flow_net,
            speaker_proj_weight,
            d_model,
            latent_dim,
        })
    }

    /// Embed text token IDs into (1, n_tokens, d_model).
    pub fn embed_text(&self, token_ids: &[u32]) -> Result<CpuTensor<f32>> {
        // Index select from embedding table
        let emb = self.conditioner_embed_weight.index_select(token_ids, 0)?;
        // Result is (n_tokens, d_model), unsqueeze to (1, n_tokens, d_model)
        Ok(emb.unsqueeze(0)?)
    }

    /// Project speaker embedding (1, T, speaker_dim) → (1, T, d_model).
    /// Used for voice cloning from raw audio (not needed for pre-projected voice embeddings)
    #[allow(dead_code)]
    pub fn project_speaker(&self, speaker_emb: &CpuTensor<f32>) -> Result<CpuTensor<f32>> {
        Ok(speaker_emb.matmul_t(&self.speaker_proj_weight)?)
    }

    /// Feed a tensor through the transformer and update KV cache.
    /// embeddings: (1, seq, d_model)
    pub fn feed_embeddings(&mut self, embeddings: &CpuTensor<f32>) -> Result<CpuTensor<f32>> {
        self.transformer.forward(embeddings)
    }

    /// Normalize a latent code using emb_mean and emb_std.
    pub fn normalize_latent(&self, latent: &CpuTensor<f32>) -> Result<CpuTensor<f32>> {
        // (latent - mean) / std
        Ok(latent.broadcast_sub(&self.emb_mean)?.broadcast_div(&self.emb_std)?)
    }

    /// Denormalize a latent code.
    pub fn denormalize_latent(&self, latent: &CpuTensor<f32>) -> Result<CpuTensor<f32>> {
        // latent * std + mean
        Ok(latent.broadcast_mul(&self.emb_std)?.broadcast_add(&self.emb_mean)?)
    }

    /// Project latent to transformer dimension.
    /// latent: (1, latent_dim) → (1, 1, d_model)
    pub fn project_latent(&self, latent: &CpuTensor<f32>) -> Result<CpuTensor<f32>> {
        let h = latent.matmul_t(&self.input_linear_weight)?;
        // (1, d_model) → (1, 1, d_model)
        Ok(h.unsqueeze(1)?)
    }

    /// Get the BOS embedding projected to transformer dimension.
    /// Returns (1, 1, d_model)
    /// Note: bos_emb is NOT normalized before projection (matches Python behavior).
    pub fn get_bos_embedding(&self) -> Result<CpuTensor<f32>> {
        let bos = self.bos_emb.unsqueeze(0)?; // (1, latent_dim)
        self.project_latent(&bos)
    }

    /// Check EOS from transformer output.
    /// h: (1, 1, d_model) → returns EOS logit
    pub fn check_eos(&self, h: &CpuTensor<f32>) -> Result<f32> {
        let h = self.out_norm.forward(h)?;
        // Take last token: (1, d_model)
        let seq_len = h.dims()[1];
        let last = h.narrow(1, seq_len - 1..seq_len)?.contiguous()?;
        let last = last.reshape((1, self.d_model))?;

        // EOS logit: linear(last) → scalar
        let logit = last.matmul_t(&self.out_eos_weight)?;
        let logit = logit.broadcast_add(&self.out_eos_bias)?;

        let data = logit.to_vec()?;
        Ok(data[0])
    }

    /// Get the conditioning vector from transformer output for flow decoding.
    /// h: (1, seq, d_model) → (1, d_model) (last token after norm)
    pub fn get_flow_conditioning(&self, h: &CpuTensor<f32>) -> Result<CpuTensor<f32>> {
        let h = self.out_norm.forward(h)?;
        let seq_len = h.dims()[1];
        let last = h.narrow(1, seq_len - 1..seq_len)?.contiguous()?;
        Ok(last.reshape((1, self.d_model))?)
    }

    /// LSD flow decode: given noise and conditioning, run flow_net to produce latent.
    /// noise: (1, latent_dim), cond: (1, d_model)
    /// Returns (1, latent_dim) denormalized latent.
    pub fn lsd_decode(
        &self,
        noise: &CpuTensor<f32>,
        cond: &CpuTensor<f32>,
        num_steps: usize,
    ) -> Result<CpuTensor<f32>> {
        // LSD decode: Lagrangian Self Distillation
        // v_t takes (cond, s, t, x) where s=start time, t=target time
        let mut current = noise.copy()?;
        for i in 0..num_steps {
            let s = i as f32 / num_steps as f32;
            let t = (i + 1) as f32 / num_steps as f32;
            let timesteps = vec![s, t];
            let flow_dir = self.flow_net.forward(&current, cond, &timesteps)?;
            let dx = flow_dir.scale(1.0 / num_steps as f32)?;
            current = current.add(&dx)?;
        }
        self.denormalize_latent(&current)
    }

    pub fn reset_state(&mut self) {
        self.transformer.reset_state();
    }

    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    #[allow(dead_code)]
    pub fn d_model(&self) -> usize {
        self.d_model
    }
}
