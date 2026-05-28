#![allow(dead_code, unused_variables, unused_imports)]

mod config;
mod layers;
mod cache;
mod gpu;
mod gpu_model;
mod model;
mod quantize;
mod sampling;
mod weights;
mod generation;

use std::io::{self, Write};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let use_gpu = args.iter().any(|a| a == "--gpu");

    let model_dir = args.iter()
        .filter(|a| !a.starts_with("--") && *a != &args[0])
        .next()
        .cloned()
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap();
            format!("{}/Downloads/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08", home)
        });

    let sink_size = 4;
    let window_size = 64;

    if use_gpu {
        println!("Loading model (GPU/Metal) from: {}", model_dir);
        let start = Instant::now();

        let (wts, config) = weights::ModelWeights::load(&model_dir);
        let gpu_model = gpu_model::GpuLlamaModel::new(&config, &wts);

        let tokenizer_path = std::path::Path::new(&model_dir).join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .expect("Failed to load tokenizer.json");

        println!("Model loaded in {:.2}s", start.elapsed().as_secs_f64());

        println!("{}", "=".repeat(60));
        println!("STREAMING ATTENTION SINKS GENERATION (Metal GPU)");
        println!("  Sink tokens: {}", sink_size);
        println!("  Window size: {}", window_size);
        println!("  Total attention per token: {}", sink_size + window_size);
        println!("{}", "=".repeat(60));

        let gen_start = Instant::now();
        generate_gpu(
            "Once upon a time",
            &tokenizer,
            &gpu_model,
            200,
            sink_size,
            window_size,
        );
        println!("\nTotal time: {:.2}s", gen_start.elapsed().as_secs_f64());
    } else {
        println!("Loading model (CPU/Accelerate) from: {}", model_dir);
        let start = Instant::now();

        let (tokenizer, model, config) = model::load_model(&model_dir);
        println!("Model loaded in {:.2}s", start.elapsed().as_secs_f64());

        println!("{}", "=".repeat(60));
        println!("STREAMING ATTENTION SINKS GENERATION");
        println!("  Sink tokens: {}", sink_size);
        println!("  Window size: {}", window_size);
        println!("  Total attention per token: {}", sink_size + window_size);
        println!("{}", "=".repeat(60));

        let gen_start = Instant::now();
        generation::generate_streaming(
            "Once upon a time",
            &tokenizer,
            &model,
            200,
            sink_size,
            window_size,
        );
        println!("\nTotal time: {:.2}s", gen_start.elapsed().as_secs_f64());
    }
}

fn generate_gpu(
    prompt: &str,
    tokenizer: &tokenizers::Tokenizer,
    model: &gpu_model::GpuLlamaModel,
    max_tokens: usize,
    sink_size: usize,
    window_size: usize,
) {
    let mut kv_cache = cache::StreamingKVCache::new(sink_size, window_size);

    let encoding = tokenizer.encode(prompt, true).expect("Failed to encode");
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();

    print!("{}", prompt);
    io::stdout().flush().unwrap();

    // Prefill: process prompt tokens one at a time through GPU
    let mut logits = Vec::new();
    for &tok in &input_ids {
        logits = model.forward_single_token(tok as usize, &mut kv_cache);
    }

    // Decode loop
    let start_time = Instant::now();
    let mut tokens_generated = 0;

    for _ in 0..max_tokens {
        let next_token = sampling::min_p_sampling(&logits, 0.1);

        let tok_str = tokenizer.decode(&[next_token as u32], false).unwrap_or_default();
        print!("{}", tok_str);
        io::stdout().flush().unwrap();
        tokens_generated += 1;

        logits = model.forward_single_token(next_token, &mut kv_cache);
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    let tps = if elapsed > 0.0 { tokens_generated as f64 / elapsed } else { 0.0 };

    println!("\n\n[Streaming Sinks Generation - Metal GPU]");
    println!("  Tokens: {}", tokens_generated);
    println!("  Throughput: {:.2} tok/s", tps);
    println!("  Window: {} sink + {} recent = {} total attention", sink_size, window_size, sink_size + window_size);
    println!("  Elapsed: {:.2}s", elapsed);
}
