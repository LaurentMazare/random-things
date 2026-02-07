use anyhow::Result;
use clap::Parser;
use mimi::nn::VB;
use mimi::CPU;

mod conv;
mod flow_lm;
mod flow_mlp;
mod mimi_dec;
mod seanet;
mod transformer;
mod tts;
mod wav;


#[derive(Parser)]
#[command(name = "pocket-tts")]
#[command(about = "Text-to-speech generation using Pocket-TTS")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Generate speech from text
    Generate {
        /// Text to synthesize
        #[arg(long)]
        text: String,

        /// Predefined voice name (e.g., alba)
        #[arg(long, default_value = "alba")]
        voice: String,

        /// Output WAV file path
        #[arg(long, default_value = "./output.wav")]
        output: String,

        /// Sampling temperature
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,

        /// Number of LSD decode steps
        #[arg(long, default_value_t = 1)]
        lsd_decode_steps: usize,

        /// EOS threshold (lower = longer speech)
        #[arg(long, default_value_t = -4.0)]
        eos_threshold: f32,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Generate { text, voice, output, temperature, lsd_decode_steps, eos_threshold } => {
            generate(&text, &voice, &output, temperature, lsd_decode_steps, eos_threshold)?;
        }
    }

    Ok(())
}

fn generate(
    text: &str,
    voice: &str,
    output_path: &str,
    temperature: f32,
    lsd_decode_steps: usize,
    eos_threshold: f32,
) -> Result<()> {
    // Download model weights from HuggingFace
    tracing::info!("downloading model weights...");
    let api = hf_hub::api::sync::Api::new()?;

    // Main model weights
    let weights_repo = api.model("kyutai/pocket-tts".to_string());
    let weights_path = weights_repo.get("tts_b6369a24.safetensors")?;
    tracing::info!("weights: {}", weights_path.display());

    // Tokenizer
    let tokenizer_repo = api.model("kyutai/pocket-tts-without-voice-cloning".to_string());
    let tokenizer_path = tokenizer_repo.get("tokenizer.model")?;
    tracing::info!("tokenizer: {}", tokenizer_path.display());

    // Voice embedding
    let voice_path = tokenizer_repo.get(&format!("embeddings/{voice}.safetensors"))?;
    tracing::info!("voice: {}", voice_path.display());

    // Load tokenizer
    tracing::info!("loading tokenizer...");
    let tokenizer = sentencepiece::SentencePieceProcessor::open(&tokenizer_path)?;

    // Load voice embedding from safetensors
    tracing::info!("loading voice embedding...");
    let voice_vb = VB::load(&[&voice_path], CPU)?;
    let voice_root = voice_vb.root();
    // Voice embedding shape: (1, T, 1024) - but stored as "audio_prompt"
    // We need to figure out the shape dynamically
    let voice_td = voice_root.get_tensor("audio_prompt")
        .ok_or_else(|| anyhow::anyhow!("voice file missing 'audio_prompt' tensor"))?;
    let voice_shape = voice_td.shape.clone();
    tracing::info!("voice embedding shape: {:?}", voice_shape);
    let voice_emb: mimi::CpuTensor<f32> = voice_root.tensor("audio_prompt", voice_shape)?;

    // Load model
    tracing::info!("loading model...");
    let vb = VB::load(&[&weights_path], CPU)?;
    let vb_root = vb.root();
    let mut model = tts::TTSModel::load(&vb_root)?;

    // Generate audio
    tracing::info!("generating audio for: \"{}\"", text);
    let audio = model.generate(
        text,
        &voice_emb,
        &tokenizer,
        temperature,
        lsd_decode_steps,
        eos_threshold,
    )?;

    // Write WAV file
    tracing::info!("writing WAV to {output_path}");
    wav::write_wav(
        std::path::Path::new(output_path),
        &audio,
        24000, // sample rate
        1,     // mono
    )?;

    tracing::info!("done!");
    Ok(())
}
