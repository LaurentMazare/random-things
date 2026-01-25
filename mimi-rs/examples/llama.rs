use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use mimi::models::llama::{
    precompute_freqs_cis, Attention, Config, KvCache, Llama, Mlp, TransformerBlock,
};
use mimi::nn::{Linear, RmsNorm};
use mimi::{Backend, Tensor, WithDType, WithDTypeF};
use rand::Rng;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModelSize {
    /// Tiny test model (~1M params, 2 layers) - for quick testing (no weights)
    Test,
    /// SmolLM 135M - small but capable
    Smol135m,
    /// SmolLM 360M
    Smol360m,
    /// TinyLlama 1.1B
    TinyLlama,
}

impl ModelSize {
    fn hf_repo(&self) -> Option<&'static str> {
        match self {
            ModelSize::Test => None,
            ModelSize::Smol135m => Some("HuggingFaceTB/SmolLM-135M"),
            ModelSize::Smol360m => Some("HuggingFaceTB/SmolLM-360M"),
            ModelSize::TinyLlama => Some("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        }
    }

    fn config(&self) -> Config {
        match self {
            ModelSize::Test => Config::tiny_test(),
            ModelSize::Smol135m => Config::smol_lm_135m(),
            ModelSize::Smol360m => Config::smol_lm_360m(),
            ModelSize::TinyLlama => Config::tiny_llama_1_1b(),
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "llama")]
#[command(about = "Run Llama model inference in autoregressive mode")]
struct Args {
    /// Model size to use
    #[arg(short = 's', long, value_enum, default_value_t = ModelSize::Smol135m)]
    model_size: ModelSize,

    /// Number of tokens to generate
    #[arg(short, long, default_value_t = 50)]
    max_tokens: usize,

    /// Text prompt (or comma-separated token ids if --raw-tokens is set)
    #[arg(short, long, default_value = "The quick brown fox")]
    prompt: String,

    /// Interpret prompt as comma-separated token IDs instead of text
    #[arg(long, default_value_t = false)]
    raw_tokens: bool,

    /// Sampling temperature (0 = greedy/argmax, higher = more random)
    #[arg(short, long, default_value_t = 0.7)]
    temperature: f32,

    /// Verbose output (show tensor loading)
    #[arg(short, long, default_value_t = false)]
    verbose: bool,
}

/// Load a tensor from safetensors data, converting to f32
fn load_tensor<B: Backend>(
    tensors: &safetensors::SafeTensors,
    name: &str,
    dev: &B,
) -> Result<Tensor<f32, B>> {
    let view = tensors.tensor(name).with_context(|| format!("tensor {name} not found"))?;
    let shape: Vec<usize> = view.shape().to_vec();

    let data: Vec<f32> = match view.dtype() {
        safetensors::Dtype::F32 => {
            let bytes = view.data();
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        }
        safetensors::Dtype::F16 => {
            let bytes = view.data();
            bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect()
        }
        safetensors::Dtype::BF16 => {
            let bytes = view.data();
            bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect()
        }
        dtype => anyhow::bail!("unsupported dtype: {:?}", dtype),
    };
    let data = Tensor::from_vec(data, shape, dev)?;
    Ok(data)
}

/// Container for memory-mapped safetensor files
struct SafeTensorFiles {
    // Keep mmaps alive while SafeTensors references them
    _mmaps: Vec<memmap2::Mmap>,
    data: Vec<&'static [u8]>,
}

impl SafeTensorFiles {
    fn new(paths: Vec<std::path::PathBuf>) -> Result<Self> {
        let mut mmaps = Vec::new();
        let mut data = Vec::new();

        for path in paths {
            let file = std::fs::File::open(&path)
                .with_context(|| format!("failed to open {}", path.display()))?;
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            // SAFETY: We keep the mmap alive in self._mmaps
            let static_slice: &'static [u8] =
                unsafe { std::slice::from_raw_parts(mmap.as_ptr(), mmap.len()) };
            mmaps.push(mmap);
            data.push(static_slice);
        }

        Ok(Self { _mmaps: mmaps, data })
    }
}

/// Load all tensors from safetensor files into a hashmap
fn load_all_tensors<B: Backend>(
    files: &SafeTensorFiles,
    verbose: bool,
    dev: &B,
) -> Result<HashMap<String, Tensor<f32, B>>> {
    let mut all_tensors = HashMap::new();

    for data in &files.data {
        let tensors = safetensors::SafeTensors::deserialize(data)?;
        for name in tensors.names() {
            let tensor = load_tensor(&tensors, name, dev)?;
            if verbose {
                println!("  Tensor {}: {:?}", name, tensor.shape());
            }
            all_tensors.insert(name.to_string(), tensor);
        }
    }

    Ok(all_tensors)
}

fn get_tensor<T: WithDType, B: Backend>(
    tensors: &HashMap<String, Tensor<T, B>>,
    name: &str,
) -> Result<Tensor<T, B>> {
    match tensors.get(name) {
        None => anyhow::bail!("missing tensor: {name}"),
        Some(tensor) => Ok(tensor.copy()?),
    }
}

fn create_attention_from_weights<T: WithDTypeF, B: Backend>(
    tensors: &HashMap<String, Tensor<T, B>>,
    prefix: &str,
    config: &Config,
) -> Result<Attention<T, B>> {
    let q_proj = Linear::new(get_tensor(tensors, &format!("{prefix}.q_proj.weight"))?);
    let k_proj = Linear::new(get_tensor(tensors, &format!("{prefix}.k_proj.weight"))?);
    let v_proj = Linear::new(get_tensor(tensors, &format!("{prefix}.v_proj.weight"))?);
    let o_proj = Linear::new(get_tensor(tensors, &format!("{prefix}.o_proj.weight"))?);

    Ok(Attention::new(
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    ))
}

fn create_mlp_from_weights<T: WithDTypeF, B: Backend>(
    tensors: &HashMap<String, Tensor<T, B>>,
    prefix: &str,
) -> Result<Mlp<T, B>> {
    let gate_proj = Linear::new(get_tensor(tensors, &format!("{prefix}.gate_proj.weight"))?);
    let up_proj = Linear::new(get_tensor(tensors, &format!("{prefix}.up_proj.weight"))?);
    let down_proj = Linear::new(get_tensor(tensors, &format!("{prefix}.down_proj.weight"))?);

    Ok(Mlp::new(gate_proj, up_proj, down_proj))
}

fn create_transformer_block_from_weights<T: WithDTypeF, B: Backend>(
    tensors: &HashMap<String, Tensor<T, B>>,
    layer_idx: usize,
    config: &Config,
) -> Result<TransformerBlock<T, B>> {
    let prefix = format!("model.layers.{layer_idx}");

    let attn = create_attention_from_weights(tensors, &format!("{prefix}.self_attn"), config)?;
    let mlp = create_mlp_from_weights(tensors, &format!("{prefix}.mlp"))?;

    let input_layernorm = RmsNorm::new(
        get_tensor(tensors, &format!("{prefix}.input_layernorm.weight"))?,
        config.rms_norm_eps,
    );
    let post_attention_layernorm = RmsNorm::new(
        get_tensor(tensors, &format!("{prefix}.post_attention_layernorm.weight"))?,
        config.rms_norm_eps,
    );

    Ok(TransformerBlock::new(attn, mlp, input_layernorm, post_attention_layernorm))
}

fn load_llama_from_weights<T: WithDTypeF, B: Backend>(
    tensors: &HashMap<String, Tensor<T, B>>,
    config: &Config,
    dev: &B,
) -> Result<Llama<T, B>> {
    println!("  Loading embed_tokens...");
    let embed_tokens = get_tensor(tensors, "model.embed_tokens.weight")?;

    println!("  Loading {} transformer layers...", config.num_hidden_layers);
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        if i % 5 == 0 {
            println!("    Layer {}/{}...", i, config.num_hidden_layers);
        }
        layers.push(create_transformer_block_from_weights(tensors, i, config)?);
    }

    println!("  Loading final norm and lm_head...");
    let norm = RmsNorm::new(get_tensor(tensors, "model.norm.weight")?, config.rms_norm_eps);

    // lm_head might be tied to embed_tokens in some models
    let lm_head_weight = match tensors.get("lm_head.weight") {
        Some(tensor) => tensor.copy()?,
        None => embed_tokens.copy()?,
    };
    let lm_head = Linear::new(lm_head_weight);

    println!("  Computing RoPE frequencies...");
    let (cos_cache, sin_cache) = precompute_freqs_cis(
        config.head_dim,
        config.max_position_embeddings,
        config.rope_theta,
        dev,
    )?;

    Ok(Llama::new(embed_tokens, layers, norm, lm_head, cos_cache, sin_cache))
}

fn create_llama_zeros<T: WithDTypeF, B: Backend>(config: &Config, dev: &B) -> Result<Llama<T, B>> {
    let embed_tokens = Tensor::zeros((config.vocab_size, config.hidden_size), dev)?;

    let layers: Vec<TransformerBlock<T, B>> = (0..config.num_hidden_layers)
        .map(|_| {
            let hidden_size = config.hidden_size;
            let num_heads = config.num_attention_heads;
            let num_kv_heads = config.num_key_value_heads;
            let head_dim = config.head_dim;
            let intermediate_size = config.intermediate_size;

            let q_proj = Linear::new(Tensor::zeros((num_heads * head_dim, hidden_size), dev)?);
            let k_proj = Linear::new(Tensor::zeros((num_kv_heads * head_dim, hidden_size), dev)?);
            let v_proj = Linear::new(Tensor::zeros((num_kv_heads * head_dim, hidden_size), dev)?);
            let o_proj = Linear::new(Tensor::zeros((hidden_size, num_heads * head_dim), dev)?);
            let attn =
                Attention::new(q_proj, k_proj, v_proj, o_proj, num_heads, num_kv_heads, head_dim);

            let gate_proj = Linear::new(Tensor::zeros((intermediate_size, hidden_size), dev)?);
            let up_proj = Linear::new(Tensor::zeros((intermediate_size, hidden_size), dev)?);
            let down_proj = Linear::new(Tensor::zeros((hidden_size, intermediate_size), dev)?);
            let mlp = Mlp::new(gate_proj, up_proj, down_proj);

            let input_layernorm =
                RmsNorm::new(Tensor::zeros((hidden_size,), dev)?, config.rms_norm_eps);
            let post_attention_layernorm =
                RmsNorm::new(Tensor::zeros((hidden_size,), dev)?, config.rms_norm_eps);

            Ok::<_, anyhow::Error>(TransformerBlock::<T, B>::new(
                attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
            ))
        })
        .collect::<Result<Vec<_>>>()?;

    let norm = RmsNorm::new(Tensor::zeros((config.hidden_size,), dev)?, config.rms_norm_eps);
    let lm_head = Linear::new(Tensor::zeros((config.vocab_size, config.hidden_size), dev)?);

    let (cos_cache, sin_cache) = precompute_freqs_cis(
        config.head_dim,
        config.max_position_embeddings,
        config.rope_theta,
        dev,
    )?;

    Ok(Llama::new(embed_tokens, layers, norm, lm_head, cos_cache, sin_cache))
}

struct ModelFiles {
    safetensor_paths: Vec<std::path::PathBuf>,
    tokenizer_path: std::path::PathBuf,
}

fn download_model(repo_id: &str) -> Result<ModelFiles> {
    use hf_hub::{api::sync::Api, Repo, RepoType};
    println!("Downloading model from {repo_id}...");
    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

    // Download tokenizer
    let tokenizer_path = repo.get("tokenizer.json").context("tokenizer.json not found")?;
    println!("  Found tokenizer.json");

    // Get list of safetensor files
    let mut safetensor_paths = Vec::new();

    // Try single model.safetensors first
    match repo.get("model.safetensors") {
        Ok(path) => {
            println!("  Found model.safetensors");
            safetensor_paths.push(path);
        }
        Err(_) => {
            // Try sharded format with different shard counts
            for i in 1..=10 {
                for total in [2, 3, 4, 5, 6, 7, 8] {
                    if let Ok(path) = repo.get(&format!("model-{i:05}-of-{total:05}.safetensors")) {
                        println!("  Found {}", path.display());
                        safetensor_paths.push(path);
                        break;
                    }
                }
            }
        }
    }

    if safetensor_paths.is_empty() {
        anyhow::bail!("No safetensor files found in {repo_id}");
    }

    Ok(ModelFiles { safetensor_paths, tokenizer_path })
}

fn sample_token<B: Backend>(
    logits: &Tensor<f32, B>,
    temperature: f32,
    rng: &mut impl Rng,
) -> Result<u32> {
    // logits shape: (1, seq_len, vocab_size)
    let vocab_size = logits.dims()[2];
    let data = logits.to_vec()?;
    // Get logits for the last token
    let start = data.len() - vocab_size;
    let last_logits = &data[start..];

    if temperature <= 0.0 {
        // Pure argmax
        let logit = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);
        Ok(logit)
    } else {
        // Temperature-scaled sampling with top-p
        let mut probs: Vec<(usize, f32)> =
            last_logits.iter().enumerate().map(|(i, &v)| (i, v / temperature)).collect();

        // Softmax
        let max_logit = probs.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = probs.iter().map(|(_, v)| (*v - max_logit).exp()).sum();
        for (_, v) in probs.iter_mut() {
            *v = (*v - max_logit).exp() / sum_exp;
        }

        // Sort by probability descending for top-p sampling
        probs.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Top-p (nucleus) sampling with p=0.9
        let top_p = 0.9;
        let mut cumsum = 0.0;
        let mut cutoff_idx = probs.len();
        for (i, (_, p)) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Renormalize the top-p tokens
        let top_probs = &probs[..cutoff_idx];
        let sum: f32 = top_probs.iter().map(|(_, p)| p).sum();

        // Sample from the top-p distribution
        let mut rng_val: f32 = rng.random();
        for (idx, p) in top_probs {
            rng_val -= p / sum;
            if rng_val <= 0.0 {
                return Ok(*idx as u32);
            }
        }

        // Fallback to first token
        Ok(top_probs[0].0 as u32)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = args.model_size.config();

    println!("Model: {:?}", args.model_size);
    println!("Config: {:?}", config);

    // CPU Device
    let dev = ();

    let (model, tokenizer) = if let Some(repo_id) = args.model_size.hf_repo() {
        let model_files = download_model(repo_id)?;

        println!("Loading tokenizer...");
        let tokenizer = tokenizers::Tokenizer::from_file(&model_files.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        println!("Loading weights...");
        let files = SafeTensorFiles::new(model_files.safetensor_paths)?;
        let tensors = load_all_tensors(&files, args.verbose, &dev)?;
        println!("  Loaded {} tensors", tensors.len());
        let model = load_llama_from_weights(&tensors, &config, &dev)?;

        (model, Some(tokenizer))
    } else {
        println!("Using zero-initialized weights (test mode)");
        (create_llama_zeros(&config, &dev)?, None)
    };

    // Tokenize the prompt
    let mut tokens: Vec<u32> = if args.raw_tokens {
        args.prompt.split(',').filter_map(|s| s.trim().parse().ok()).collect()
    } else if let Some(ref tokenizer) = tokenizer {
        let encoding = tokenizer
            .encode(args.prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        encoding.get_ids().to_vec()
    } else {
        // Test mode without tokenizer
        vec![1, 2, 3]
    };

    if tokens.is_empty() {
        tokens = vec![1]; // Default BOS token
    }

    println!("\nPrompt: \"{}\"", args.prompt);
    println!("Tokenized: {:?} ({} tokens)", tokens, tokens.len());
    println!("Generating {} tokens (temperature={})...\n", args.max_tokens, args.temperature);

    // Autoregressive generation loop
    let mut rng = rand::rng();
    let mut kv_cache: Option<KvCache<f32, ()>> = None;
    let mut pos = 0;
    let mut generated_tokens = Vec::new();
    let mut autoregressive_start: Option<std::time::Instant> = None;

    for step in 0..args.max_tokens {
        let input_tokens: Vec<u32> = if kv_cache.is_none() {
            // First forward pass: process all prompt tokens
            tokens.clone()
        } else {
            // Subsequent passes: only process the last generated token
            vec![*tokens.last().unwrap()]
        };

        let (logits, new_kv_cache) = model.forward(&input_tokens, pos, kv_cache.as_ref())?;
        kv_cache = Some(new_kv_cache);
        pos += input_tokens.len();

        // Start timing after the first forward pass (prefill)
        if autoregressive_start.is_none() {
            autoregressive_start = Some(std::time::Instant::now());
        }

        // Sample next token
        let next_token = sample_token(&logits, args.temperature, &mut rng)?;
        tokens.push(next_token);
        generated_tokens.push(next_token);

        // Decode and print the new token
        if let Some(ref tokenizer) = tokenizer {
            let decoded = tokenizer
                .decode(&[next_token], false)
                .unwrap_or_else(|_| format!("[{}]", next_token));
            print!("{}", decoded);
            use std::io::Write;
            std::io::stdout().flush().ok();
        } else {
            print!("[{}]", next_token);
        }

        // Stop if we hit an EOS token (common values: 0, 1, 2 depending on tokenizer)
        if next_token == 0 || next_token == 1 || next_token == 2 {
            println!("\n\n(stopped at EOS token)");
            break;
        }

        if step == args.max_tokens - 1 {
            println!("\n\n(reached max tokens)");
        }
    }

    // Print final summary
    let autoregressive_elapsed =
        autoregressive_start.map(|start| start.elapsed()).unwrap_or_default();
    let tokens_per_second = if autoregressive_elapsed.as_secs_f64() > 0.0 {
        generated_tokens.len() as f64 / autoregressive_elapsed.as_secs_f64()
    } else {
        0.0
    };
    println!(
        "\nGenerated {} tokens in {:.2}s ({:.2} tokens/sec)",
        generated_tokens.len(),
        autoregressive_elapsed.as_secs_f64(),
        tokens_per_second
    );
    println!("Token IDs: {:?}", generated_tokens);

    if let Some(ref tokenizer) = tokenizer {
        let full_text =
            tokenizer.decode(&tokens, false).unwrap_or_else(|_| "(decode error)".to_string());
        println!("\nFull text:\n{}", full_text);
    }

    Ok(())
}
