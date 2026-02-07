use anyhow::Result;
use mimi::{CpuDevice, CpuTensor};
use mimi::nn::var_builder::Path;
use rand::Rng;

use crate::flow_lm::FlowLMModel;
use crate::mimi_dec::MimiDecoder;

/// Full TTS model: FlowLM + MimiDecoder.
pub struct TTSModel {
    flow_lm: FlowLMModel,
    mimi_decoder: MimiDecoder,
}

impl TTSModel {
    pub fn load(vb: &Path<CpuDevice>) -> Result<Self> {
        let flow_lm = FlowLMModel::load(&vb.pp("flow_lm"))?;
        let mimi_decoder = MimiDecoder::load(vb)?;
        Ok(Self { flow_lm, mimi_decoder })
    }

    /// Generate audio from text.
    ///
    /// Returns f32 audio samples at 24kHz.
    pub fn generate(
        &mut self,
        text: &str,
        voice_emb: &CpuTensor<f32>,
        tokenizer: &sentencepiece::SentencePieceProcessor,
        temperature: f32,
        lsd_decode_steps: usize,
        eos_threshold: f32,
    ) -> Result<Vec<f32>> {
        let dev = &mimi::CPU;

        // Reset model state
        self.flow_lm.reset_state();
        self.mimi_decoder.reset_state();

        // Prepare text: capitalize first letter, ensure punctuation
        let text = prepare_text(text);
        tracing::info!("prepared text: {text}");

        // Tokenize text
        let token_ids = tokenizer.encode(&text)?;
        let token_ids: Vec<u32> = token_ids.into_iter().map(|p| p.id as u32).collect();
        tracing::info!("tokenized: {} tokens", token_ids.len());

        let n_tokens = token_ids.len();

        // Max generation length: ceil((n_tokens / 3.0 + 2.0) * 12.5)
        let max_gen_len = ((n_tokens as f64 / 3.0 + 2.0) * 12.5).ceil() as usize;
        tracing::info!("max generation length: {max_gen_len}");

        // Step 1: Feed voice embedding then text embeddings through transformer.
        // In Python, voice (audio_conditioning) is fed first, then text tokens are fed second.
        // The voice embedding from .safetensors files is already projected to d_model (1024) space.
        // We feed them separately to match the Python ordering in the KV cache.
        let _ = self.flow_lm.feed_embeddings(voice_emb)?;
        let text_emb = self.flow_lm.embed_text(&token_ids)?;
        let _ = self.flow_lm.feed_embeddings(&text_emb)?;

        // Step 2: Feed BOS and start autoregressive generation
        let bos_emb = self.flow_lm.get_bos_embedding()?;
        let mut h = self.flow_lm.feed_embeddings(&bos_emb)?;

        let mut all_audio = Vec::new();
        let mut rng = rand::rng();
        let latent_dim = self.flow_lm.latent_dim();

        for step in 0..max_gen_len {
            // Check EOS: stop when logit exceeds threshold (Python: out_eos > threshold)
            let eos_logit = self.flow_lm.check_eos(&h)?;
            if eos_logit > eos_threshold {
                tracing::info!("EOS at step {step}, logit={eos_logit:.3}");
                break;
            }

            // Get conditioning from transformer output
            let cond = self.flow_lm.get_flow_conditioning(&h)?;

            // Sample noise: N(0, std^2) where std = temp^0.5
            let std = temperature.sqrt();
            let noise_data: Vec<f32> = (0..latent_dim)
                .map(|_| sample_normal(&mut rng, std))
                .collect();
            let noise = CpuTensor::from_vec(noise_data, (1, latent_dim), dev)?;

            // LSD flow decode: noise → latent
            let latent = self.flow_lm.lsd_decode(&noise, &cond, lsd_decode_steps)?;

            // Decode latent → audio frame
            let audio_frame = self.mimi_decoder.decode_latent(&latent)?;

            // Collect audio samples
            let audio_data = audio_frame.to_vec()?;
            all_audio.extend_from_slice(&audio_data);

            // Feed latent back into transformer for next step
            let latent_norm = self.flow_lm.normalize_latent(&latent)?;
            let latent_proj = self.flow_lm.project_latent(&latent_norm)?;
            h = self.flow_lm.feed_embeddings(&latent_proj)?;
        }

        tracing::info!("generated {} audio samples ({:.2}s at 24kHz)",
            all_audio.len(), all_audio.len() as f64 / 24000.0);

        Ok(all_audio)
    }
}

/// Prepare text for TTS: capitalize first letter, ensure punctuation at end.
/// Matches Python's prepare_text_prompt().
fn prepare_text(text: &str) -> String {
    let mut text = text.trim().to_string();
    // Collapse whitespace
    text = text.replace('\n', " ").replace('\r', " ").replace("  ", " ");
    if text.is_empty() {
        return ".".to_string();
    }

    // Capitalize first letter
    let mut chars = text.chars();
    if let Some(first) = chars.next() {
        if !first.is_uppercase() {
            text = first.to_uppercase().to_string() + chars.as_str();
        }
    }

    // Ensure punctuation at end: if last char is alphanumeric, add period
    let last_char = text.chars().last().unwrap();
    if last_char.is_alphanumeric() {
        text.push('.');
    }

    // Pad short text with leading spaces (Python: 8 spaces if < 5 words)
    let word_count = text.split_whitespace().count();
    if word_count < 5 {
        text = format!("        {text}");
    }

    text
}

/// Sample from normal distribution N(0, std^2) using Box-Muller transform.
fn sample_normal(rng: &mut impl Rng, std: f32) -> f32 {
    loop {
        let u1: f32 = rng.random();
        let u2: f32 = rng.random();
        if u1 <= f32::EPSILON {
            continue;
        }
        let n = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        return n * std;
    }
}
