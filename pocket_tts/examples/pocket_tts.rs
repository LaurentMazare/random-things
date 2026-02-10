use anyhow::{Context, Result};
use clap::Parser;
use mimi::nn::VB;
use mimi::{Backend, Tensor};
use pocket_tts::flow_lm::FlowLMConfig;
use pocket_tts::mimi::MimiConfig;
use pocket_tts::tts_model::{TTSConfig, TTSModel, prepare_text_prompt};

#[derive(Parser, Debug)]
#[command(name = "pocket-tts")]
#[command(about = "Generate speech from text using Pocket TTS")]
struct Args {
    /// Text to synthesize
    text: String,

    /// Output WAV file path
    #[arg(short, long, default_value = "output.wav")]
    output: std::path::PathBuf,

    /// Voice to use
    #[arg(short, long, default_value = "alba")]
    voice: String,

    /// Sampling temperature
    #[arg(short, long, default_value_t = 0.7)]
    temperature: f32,

    /// Use the cpu device even if cuda is available
    #[arg(long, default_value_t = false)]
    cpu: bool,

    #[arg(long)]
    chrome_tracing: bool,
}

const VOICES: &[&str] =
    &["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"];

fn download_files(
    voice: &str,
) -> Result<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf)> {
    use hf_hub::{Repo, RepoType, api::sync::Api};
    let repo_id = "kyutai/pocket-tts-without-voice-cloning";
    println!("Downloading from {repo_id}...");
    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

    let model_path = repo.get("tts_b6369a24.safetensors").context("model weights not found")?;
    println!("  Model weights: {}", model_path.display());

    let tokenizer_path = repo.get("tokenizer.model").context("tokenizer not found")?;
    println!("  Tokenizer: {}", tokenizer_path.display());

    let voice_file = format!("embeddings/{voice}.safetensors");
    let voice_path =
        repo.get(&voice_file).with_context(|| format!("voice embedding '{voice}' not found"))?;
    println!("  Voice embedding: {}", voice_path.display());

    Ok((model_path, tokenizer_path, voice_path))
}

fn remap_key(name: &str) -> Option<String> {
    // Skip keys we don't need
    if name.contains("flow.w_s_t")
        || name.contains("quantizer.vq")
        || name.contains("quantizer.logvar_proj")
        || name.contains("learnt_padding")
    {
        return None;
    }

    let mut name = name.to_string();

    // Order matters: more specific replacements first
    name = name.replace(
        "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
        "flow_lm.speaker_proj_weight",
    );
    name = name.replace(
        "flow_lm.condition_provider.conditioners.transcript_in_segment.",
        "flow_lm.conditioner.",
    );
    name = name.replace("flow_lm.backbone.", "flow_lm.transformer.");
    name = name.replace("flow_lm.flow.", "flow_lm.flow_net.");
    name = name.replace("mimi.model.", "mimi.");

    Some(name)
}

fn init_tracing() -> tracing_chrome::FlushGuard {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::{prelude::*, registry::Registry};

    let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
    Registry::default().with(chrome_layer).init();
    guard
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _guard = if args.chrome_tracing { Some(init_tracing()) } else { None };

    if !VOICES.contains(&args.voice.as_str()) {
        anyhow::bail!("Unknown voice '{}'. Available voices: {}", args.voice, VOICES.join(", "));
    }

    #[cfg(feature = "cuda")]
    {
        if args.cpu {
            println!("Using CPU backend");
            run_for_device(args, mimi::CPU)?;
        } else {
            println!("Using CUDA backend");
            let dev = mimi::cuda_backend::Device::new(0)?;
            unsafe {
                dev.disable_event_tracking();
            }
            run_for_device(args, dev)?;
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        println!("Using CPU backend");
        run_for_device(args, mimi::CPU)?;
    }

    Ok(())
}

struct Rng {
    inner: rand::rngs::StdRng,
    distr: rand_distr::Normal<f32>,
}

impl Rng {
    pub fn new(temperature: f32) -> Result<Self> {
        use rand::SeedableRng;
        let std = temperature.sqrt();
        let distr = rand_distr::Normal::new(0f32, std)?;
        Ok(Self { inner: rand::rngs::StdRng::seed_from_u64(42), distr })
    }
}

impl pocket_tts::flow_lm::Rng for Rng {
    fn sample(&mut self) -> f32 {
        use rand::Rng;
        self.inner.sample(self.distr)
    }
}

fn run_for_device<Dev: Backend>(args: Args, dev: Dev) -> Result<()> {
    let (model_path, tokenizer_path, voice_path) = download_files(&args.voice)?;

    // Load model weights with key remapping
    println!("\nLoading model...");
    let vb = VB::load_with_key_map(&[&model_path], dev.clone(), remap_key)?;
    let root = vb.root();

    let tokenizer_str = tokenizer_path.to_str().context("invalid tokenizer path")?;

    let cfg = TTSConfig {
        flow_lm: FlowLMConfig {
            d_model: 1024,
            num_heads: 16,
            num_layers: 6,
            dim_feedforward: 4096,
            max_period: 10000.0,
            n_bins: 4000,
            tokenizer_path: tokenizer_str.to_string(),
            lut_dim: 1024,
            flow_dim: 512,
            flow_depth: 6,
            ldim: 32,
        },
        mimi: MimiConfig {
            channels: 1,
            sample_rate: 24000,
            frame_rate: 12,
            dimension: 512,
            quantizer_dimension: 32,
            quantizer_output_dimension: 512,
            n_filters: 64,
            n_residual_layers: 1,
            ratios: vec![6, 5, 4],
            kernel_size: 7,
            last_kernel_size: 3,
            residual_kernel_size: 3,
            dilation_base: 2,
            compress: 2,
            transformer_d_model: 512,
            transformer_num_heads: 8,
            transformer_num_layers: 2,
            transformer_layer_scale: 0.01,
            transformer_context: 250,
            transformer_max_period: 10000.0,
            transformer_dim_feedforward: 2048,
        },
        temp: args.temperature,
        lsd_decode_steps: 1,
        noise_clamp: None,
        eos_threshold: -4.0,
    };

    let mut rng = Rng::new(args.temperature)?;

    let model: TTSModel<f32, Dev> = TTSModel::load(&root, &cfg)?;
    println!("Model loaded successfully!");

    // Prepare text
    let (text, frames_after_eos) = prepare_text_prompt(&args.text);
    println!("Text: {text:?}");

    // Tokenize
    let tokens = model.flow_lm.conditioner.tokenize(&text);
    let num_tokens = tokens.len();
    println!("Tokens: {num_tokens}");

    // Max generation frames
    let max_frames = ((num_tokens as f64 / 3.0 + 2.0) * 12.5).ceil() as usize;
    println!("Max generation frames: {max_frames}");

    // Budget for transformer state: text tokens + voice frames + generation frames
    let seq_budget = num_tokens + 512 + max_frames;

    // Init states
    let mut tts_state = model.init_flow_lm_state(1, seq_budget)?;
    let mut mimi_state = model.init_mimi_state(1, 250)?;

    // Load voice embedding
    println!("Loading voice embedding...");
    let voice_vb = VB::load(&[&voice_path], dev.clone())?;
    let voice_names = voice_vb.tensor_names();
    let voice_key = voice_names.first().context("no tensors found in voice embedding file")?;
    let voice_td = voice_vb.get_tensor(voice_key).context("voice tensor not found")?;
    let voice_shape = &voice_td.shape;
    let voice_dims = voice_shape.dims();
    println!("  Voice tensor '{voice_key}': shape {voice_dims:?}");

    // Load as raw tensor and reshape to [1, T, dim]
    let voice_emb: Tensor<f32, Dev> = voice_vb.tensor(voice_key, voice_shape.clone())?;
    let voice_emb = if voice_dims.len() == 2 {
        voice_emb.reshape((1, voice_dims[0], voice_dims[1]))?
    } else {
        voice_emb
    };

    // Prompt with audio conditioning
    println!("Prompting with voice conditioning ({} frames)...", voice_emb.dim(1usize)?);
    println!("{voice_emb}");
    model.prompt_audio(&mut tts_state, &voice_emb)?;

    // Prompt with text
    println!("Prompting with text...");
    model.prompt_text(&mut tts_state, &tokens)?;

    // Auto-regressive generation
    println!("\nGenerating speech...");
    let gen_start = std::time::Instant::now();

    // BOS marker: NaN tensor [1, 1, ldim]
    let ldim = cfg.flow_lm.ldim;
    let nan_data: Vec<f32> = vec![f32::NAN; ldim];
    let mut prev_latent: Tensor<f32, Dev> = Tensor::from_vec(nan_data, (1, 1, ldim), &dev)?;

    let mut eos_countdown: Option<usize> = None;

    let (latent_tx, latent_rx) = std::sync::mpsc::channel();
    let model = std::sync::Arc::new(model);
    let jh = std::thread::spawn({
        let model = model.clone();
        move || {
            let mut audio_chunks: Vec<Tensor<f32, Dev>> = Vec::new();
            while let Ok(next_latent) = latent_rx.recv() {
                // Decode latent to audio
                let audio_chunk = model.decode_latent(&next_latent, &mut mimi_state)?;
                audio_chunks.push(audio_chunk);
            }
            let gen_elapsed = gen_start.elapsed();
            println!(
                "Generated {} frames in {:.2}s",
                audio_chunks.len(),
                gen_elapsed.as_secs_f64()
            );

            // Concatenate audio
            let audio_refs: Vec<&Tensor<f32, Dev>> = audio_chunks.iter().collect();
            let audio = Tensor::cat(&audio_refs, 2)?;
            let audio = audio.narrow(0, ..1)?.contiguous()?;
            println!("Output audio shape: {:?}", audio.dims());

            let pcm = audio.to_vec()?;
            let duration = pcm.len() as f64 / 24000.0;
            println!("Audio duration: {duration:.2}s");

            // Write WAV
            let output_file = std::fs::File::create(&args.output)?;
            let mut writer = std::io::BufWriter::new(output_file);
            kaudio::wav::write_pcm_as_wav(&mut writer, &pcm, 24000, 1)?;
            println!("Written to {}", args.output.display());

            Ok::<_, anyhow::Error>(())
        }
    });

    for step in 0..max_frames {
        let (next_latent, is_eos) = model.generate_step(&mut tts_state, &prev_latent, &mut rng)?;
        println!("Step {step}:\n{next_latent}");
        latent_tx.send(next_latent.clone()).unwrap();

        if is_eos && eos_countdown.is_none() {
            println!("  EOS detected at step {step}");
            eos_countdown = Some(frames_after_eos);
        }

        if let Some(ref mut countdown) = eos_countdown {
            if *countdown == 0 {
                println!("  Stopping after {frames_after_eos} frames past EOS");
                break;
            }
            *countdown -= 1;
        }

        prev_latent = next_latent;

        if (step + 1) % 25 == 0 {
            println!("  Step {}/{max_frames}", step + 1);
        }
    }
    std::mem::drop(latent_tx); // Close channel to signal generation thread to finish
    jh.join().unwrap()?;

    Ok(())
}
