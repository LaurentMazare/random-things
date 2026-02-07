use anyhow::Result;
use mimi::nn::var_builder::Path;
use mimi::{CpuDevice, CpuTensor};

use crate::conv::StreamingConvTranspose1d;
use crate::seanet::SEANetDecoder;
use crate::transformer::Transformer;

/// Mimi decoder: latent codes → audio.
pub struct MimiDecoder {
    /// DummyQuantizer: Conv1d(latent_dim, decoder_dim, 1, no bias)
    quantizer_weight: CpuTensor<f32>,
    /// Upsample: ConvTranspose1d(decoder_dim, decoder_dim, k=32, stride=16, groups=decoder_dim)
    upsample: StreamingConvTranspose1d,
    /// Decoder transformer
    decoder_transformer: Transformer,
    /// SEANet decoder
    seanet_decoder: SEANetDecoder,

    #[allow(dead_code)]
    decoder_dim: usize,
}

impl MimiDecoder {
    pub fn load(vb: &Path<CpuDevice>) -> Result<Self> {
        let latent_dim = 32;
        let decoder_dim = 512;

        // DummyQuantizer output projection: Conv1d(32, 512, 1, no bias)
        let quantizer_weight: CpuTensor<f32> =
            vb.tensor("mimi.quantizer.output_proj.weight", (decoder_dim, latent_dim, 1))?;

        // Upsample: ConvTranspose1d(512, 512, k=32, stride=16, groups=512, no bias)
        // Note: pocket-tts uses the double "convtr.convtr" path for the mimi upsample
        let upsample = StreamingConvTranspose1d::load(
            &vb.pp("mimi.upsample.convtr"),
            decoder_dim,
            decoder_dim,
            32,  // kernel_size
            16,  // stride
            decoder_dim, // groups (depthwise)
            false, // no bias
        )?;

        // Decoder transformer: d_model=512, 2 layers, 8 heads, context=250, layer_scale=0.01
        let decoder_transformer = Transformer::load(
            &vb.pp("mimi.decoder_transformer.transformer"),
            decoder_dim, // d_model
            8,           // num_heads
            2,           // num_layers
            2048,        // dim_feedforward
            250,         // context window
            10000.0,     // max_period
            true,        // use layer_scale
        )?;

        // SEANet decoder
        let seanet_decoder = SEANetDecoder::load(
            &vb.pp("mimi.decoder"),
            decoder_dim,  // dimension (input channels)
            64,           // n_filters
            &[6, 5, 4],  // ratios
            1,            // n_residual_layers
            7,            // kernel_size
            3,            // residual_kernel_size
            3,            // last_kernel_size
            2,            // dilation_base
            2,            // compress
        )?;

        Ok(Self {
            quantizer_weight,
            upsample,
            decoder_transformer,
            seanet_decoder,
            decoder_dim,
        })
    }

    /// Decode a latent code to audio.
    /// latent: (1, latent_dim) → audio: (1, 1, 1920) (one frame of 24kHz audio at 12.5 fps)
    pub fn decode_latent(&mut self, latent: &CpuTensor<f32>) -> Result<CpuTensor<f32>> {
        // Reshape latent from (1, latent_dim) to (1, latent_dim, 1) for conv1d
        let latent = latent.unsqueeze(2)?;

        // DummyQuantizer: Conv1d(32, 512, 1) - manual since we just have weight
        let h = latent.conv1d(
            &self.quantizer_weight,
            None, // no bias
            1,    // stride
            0,    // padding
            1,    // dilation
            1,    // groups
        )?;
        // h: (1, 512, 1)

        // Upsample: ConvTranspose1d → (1, 512, 16)
        let h = self.upsample.forward(&h)?;

        // Decoder transformer expects (batch, seq, d_model)
        // h is (1, 512, T) → transpose to (1, T, 512)
        let h = h.transpose(1, 2)?.contiguous()?;

        // Run decoder transformer
        let h = self.decoder_transformer.forward(&h)?;

        // Transpose back to (1, 512, T) for SEANet
        let h = h.transpose(1, 2)?.contiguous()?;

        // SEANet decoder → audio
        let audio = self.seanet_decoder.forward(&h)?;

        Ok(audio)
    }

    pub fn reset_state(&mut self) {
        self.upsample.reset_state();
        self.decoder_transformer.reset_state();
        self.seanet_decoder.reset_state();
    }
}
