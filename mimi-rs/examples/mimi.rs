use anyhow::{Context, Result};
use clap::Parser;
use mimi::Tensor;
use mimi::models::mimi::{Config, Mimi};
use mimi::nn::VB;

#[derive(Parser, Debug)]
#[command(name = "mimi")]
#[command(about = "Run Mimi audio tokenizer model")]
struct Args {
    /// Number of chunks to process (each chunk is 1920 samples)
    #[arg(short, long, default_value_t = 10)]
    num_chunks: usize,

    /// Number of codebooks to use (default: 8)
    #[arg(short, long, default_value_t = 8)]
    codebooks: usize,
}

fn download_model() -> Result<std::path::PathBuf> {
    use hf_hub::{Repo, RepoType, api::sync::Api};
    let repo_id = "kyutai/moshiko-candle-q8";
    println!("Downloading model from {repo_id}...");
    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

    let model_path = repo
        .get("tokenizer-e351c8d8-checkpoint125.safetensors")
        .context("model.safetensors not found")?;
    println!("  Found model.safetensors at {}", model_path.display());

    Ok(model_path)
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Mimi Audio Tokenizer Example");
    println!("============================");
    println!("Chunks to process: {}", args.num_chunks);
    println!("Codebooks: {}", args.codebooks);

    // Download model weights
    let model_path = download_model()?;

    // CPU device
    let dev = ();

    // Load model
    println!("\nLoading model weights...");
    let vb = VB::load(&[model_path], dev)?;
    let config = Config::v0_1_no_weight_norm(Some(args.codebooks));
    println!("Config: sample_rate={}, frame_rate={}", config.sample_rate, config.frame_rate);

    let mut model: Mimi<f32, ()> = Mimi::load(&vb.root(), config, &dev)?;
    println!("Model loaded successfully!");

    // Process chunks of zeros
    // Mimi expects audio at 24kHz, mono channel
    // Each chunk of 1920 samples = 80ms of audio at 24kHz
    let chunk_size = 1920;
    let total_samples = chunk_size * args.num_chunks;

    println!(
        "\nProcessing {} samples ({} chunks of {} samples each)...",
        total_samples, args.num_chunks, chunk_size
    );
    println!("Total duration: {:.2}ms at 24kHz", total_samples as f64 / 24.0);

    let start = std::time::Instant::now();

    for chunk_idx in 0..args.num_chunks {
        // Create a chunk of zeros: shape [batch=1, channels=1, time=1920]
        let audio_data: Vec<f32> = vec![0.0; chunk_size];
        let audio: Tensor<f32, ()> = Tensor::from_vec(audio_data, (1, 1, chunk_size), &dev)?;

        // Encode the audio to codes
        let codes = model.encode(&audio)?;

        // codes shape should be [batch, num_codebooks, time_frames]
        let code_dims = codes.dims();
        if chunk_idx == 0 {
            println!("\nFirst chunk output:");
            println!("  Input shape: [1, 1, {}]", chunk_size);
            println!("  Output codes shape: {:?}", code_dims);

            // Print first few codes
            let codes_vec = codes.to_vec()?;
            let num_to_show = codes_vec.len().min(16);
            println!("  First {} code values: {:?}", num_to_show, &codes_vec[..num_to_show]);
        }

        if (chunk_idx + 1) % 10 == 0 || chunk_idx == args.num_chunks - 1 {
            println!("  Processed chunk {}/{}", chunk_idx + 1, args.num_chunks);
        }
    }

    let elapsed = start.elapsed();
    let samples_per_sec = total_samples as f64 / elapsed.as_secs_f64();
    let realtime_factor = samples_per_sec / 24000.0;

    println!("\nPerformance:");
    println!("  Total time: {:.2}s", elapsed.as_secs_f64());
    println!("  Samples/sec: {:.0}", samples_per_sec);
    println!("  Realtime factor: {:.2}x (>1 means faster than realtime)", realtime_factor);

    // Also test decode on a simple code tensor
    println!("\nTesting decode...");
    let test_codes: Vec<i64> = vec![0; args.codebooks * 2]; // 2 time frames
    let test_codes: Tensor<i64, ()> = Tensor::from_vec(test_codes, (1, args.codebooks, 2), &dev)?;
    let decoded = model.decode(&test_codes)?;
    println!("  Decoded shape: {:?}", decoded.dims());

    println!("\nDone!");
    Ok(())
}
