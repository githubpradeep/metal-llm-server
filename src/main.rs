#![allow(dead_code, unused_variables, unused_imports)]

mod config;
mod batch_engine;
mod layers;
mod cache;
mod gpu;
mod gpu_model;
mod gemma4_config;
mod gemma4_gpu_model;
mod gemma4_mtp;
mod kv_pool;
mod metrics;
mod model;
mod quantize;
mod sampling;
mod scheduler;
mod weights;
mod generation;
mod server;

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
        // Detect if this is a Gemma4 model by checking for text_config in config.json
        let config_path = std::path::Path::new(&model_dir).join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .expect("Failed to read config.json");
        let is_gemma4 = config_str.contains("\"gemma4\"") || config_str.contains("text_config");

        if is_gemma4 {
            println!("Loading Gemma4 model (GPU/Metal) from: {}", model_dir);
            let start = Instant::now();

            let gpu_model = gemma4_gpu_model::Gemma4GpuModel::new(&model_dir);
            let assistant_dir = args.iter()
                .position(|a| a == "--assistant-dir")
                .and_then(|i| args.get(i + 1))
                .cloned();

            let tokenizer_path = std::path::Path::new(&model_dir).join("tokenizer.json");
            let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
                .expect("Failed to load tokenizer.json");

            println!("Model loaded in {:.2}s", start.elapsed().as_secs_f64());

            // Serve mode: start OpenAI-compatible HTTP server
            if args.iter().any(|a| a == "--serve") {
                let port: u16 = args.iter()
                    .position(|a| a == "--port")
                    .and_then(|i| args.get(i + 1))
                    .and_then(|p| p.parse().ok())
                    .unwrap_or(8080);

                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(server::run_server(gpu_model, tokenizer, port));
                return;
            }

            // Interactive generation mode
            let mut gpu_model = gpu_model;
            println!("{}", "=".repeat(60));
            println!("GEMMA4 E4B GENERATION (Metal GPU, Q4_0)");
            println!("{}", "=".repeat(60));

            let gen_start = Instant::now();
            //let prompt = "<start_of_turn>user\n A train leaves at 8:15 AM and arrives at 11:47 AM. How long was the journey?<end_of_turn>\n<start_of_turn>model\n";
            //let prompt = "<start_of_turn>user\n Write a short essay about the benefits of exercise. Include an introduction, 3 key points, and a conlcusion. <end_of_turn>\n<start_of_turn>model\n";
            let prompt = "<start_of_turn>user\n Implement bubble sort in python <end_of_turn>\n<start_of_turn>model\n";
            if let Some(assistant_dir) = assistant_dir {
                println!("Loading Gemma4 MTP assistant from: {}", assistant_dir);
                let mut assistant = gemma4_mtp::Gemma4MtpAssistant::new(&assistant_dir, &gpu_model);
                generate_gemma4_gpu_mtp(prompt, &tokenizer, &mut gpu_model, &mut assistant, 1000);
            } else {
                generate_gemma4_gpu(prompt, &tokenizer, &mut gpu_model, 1000);
            }
            println!("\nTotal time: {:.2}s", gen_start.elapsed().as_secs_f64());
        } else {
            println!("Loading model (GPU/Metal) from: {}", model_dir);
            let start = Instant::now();

            let (wts, config) = weights::ModelWeights::load(&model_dir);
            let mut gpu_model = gpu_model::GpuLlamaModel::new(&config, &wts);

            let tokenizer_path = std::path::Path::new(&model_dir).join("tokenizer.json");
            let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
                .expect("Failed to load tokenizer.json");

            println!("Model loaded in {:.2}s", start.elapsed().as_secs_f64());

            println!("{}", "=".repeat(60));
            println!("FULL CONTEXT GENERATION (Metal GPU, Q4_0)");
            println!("  Max context: {}", config.max_position_embeddings);
            println!("{}", "=".repeat(60));

            let gen_start = Instant::now();
            generate_gpu(
                "Once upon a time",
                &tokenizer,
                &mut gpu_model,
                200,
            );
            println!("\nTotal time: {:.2}s", gen_start.elapsed().as_secs_f64());
        }
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

fn parse_mtp_draft_steps() -> usize {
    std::env::var("LLAMA_MTP_DRAFT_STEPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(4)
}

fn print_gemma_token(
    token: usize,
    tokenizer: &tokenizers::Tokenizer,
    eos_tokens: &[usize],
) -> bool {
    if eos_tokens.contains(&token) {
        return false;
    }
    let tok_str = tokenizer.decode(&[token as u32], false).unwrap_or_default();
    print!("{}", tok_str);
    io::stdout().flush().unwrap();
    true
}

fn generate_gemma4_gpu_mtp(
    prompt: &str,
    tokenizer: &tokenizers::Tokenizer,
    model: &mut gemma4_gpu_model::Gemma4GpuModel,
    assistant: &mut gemma4_mtp::Gemma4MtpAssistant,
    max_tokens: usize,
) {
    let encoding = tokenizer.encode(prompt, true).expect("Failed to encode");
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();

    print!("{}", prompt);
    io::stdout().flush().unwrap();

    let token_ids: Vec<usize> = input_ids.iter().map(|&t| t as usize).collect();
    let mut logits = model.forward_prefill(&token_ids);

    let start_time = Instant::now();
    let mut tokens_generated = 0usize;
    let mut drafted_total = 0usize;
    let mut accepted_total = 0usize;
    let mut rejected_total = 0usize;
    let draft_steps = parse_mtp_draft_steps();
    let eos_tokens: &[usize] = &[1, 106];

    'outer: while tokens_generated < max_tokens {
        let first_token = sampling::argmax(&logits);
        if !print_gemma_token(first_token, tokenizer, eos_tokens) {
            break;
        }
        tokens_generated += 1;
        logits = model.forward_single_token(first_token);
        if tokens_generated >= max_tokens {
            break;
        }

        let mut draft_token = first_token;
        let mut draft_activation = model.last_hidden_activation();
        let mut drafted = Vec::with_capacity(draft_steps);
        for _ in 0..draft_steps {
            let draft = assistant
                .draft_next(draft_token, &draft_activation, model)
                .expect("MTP assistant draft failed");
            draft_token = draft.token_id;
            draft_activation = draft.projected_activation;
            drafted.push(draft_token);
        }
        drafted_total += drafted.len();

        for drafted_token in drafted {
            if tokens_generated >= max_tokens {
                break 'outer;
            }

            let verifier_token = sampling::argmax(&logits);
            if verifier_token == drafted_token {
                if !print_gemma_token(drafted_token, tokenizer, eos_tokens) {
                    break 'outer;
                }
                accepted_total += 1;
                tokens_generated += 1;
                logits = model.forward_single_token(drafted_token);
            } else {
                if !print_gemma_token(verifier_token, tokenizer, eos_tokens) {
                    break 'outer;
                }
                rejected_total += 1;
                tokens_generated += 1;
                logits = model.forward_single_token(verifier_token);
                continue 'outer;
            }
        }

        if tokens_generated >= max_tokens {
            break;
        }

        let bonus_token = sampling::argmax(&logits);
        if !print_gemma_token(bonus_token, tokenizer, eos_tokens) {
            break;
        }
        tokens_generated += 1;
        logits = model.forward_single_token(bonus_token);
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    let tps = if elapsed > 0.0 { tokens_generated as f64 / elapsed } else { 0.0 };
    let accept_rate = if drafted_total > 0 {
        accepted_total as f64 * 100.0 / drafted_total as f64
    } else {
        0.0
    };

    println!("\n\n[Gemma4 E4B Generation - Metal GPU, MTP assistant]");
    println!("  Tokens: {}", tokens_generated);
    println!("  Throughput: {:.2} tok/s", tps);
    println!("  Context length: {} tokens", model.num_items());
    println!("  Drafted: {}", drafted_total);
    println!("  Accepted: {} ({:.1}%)", accepted_total, accept_rate);
    println!("  Rejected cycles: {}", rejected_total);
    println!("  Draft steps: {}", draft_steps);
    println!("  Elapsed: {:.2}s", elapsed);
}

fn generate_gpu(
    prompt: &str,
    tokenizer: &tokenizers::Tokenizer,
    model: &mut gpu_model::GpuLlamaModel,
    max_tokens: usize,
) {
    let encoding = tokenizer.encode(prompt, true).expect("Failed to encode");
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();

    print!("{}", prompt);
    io::stdout().flush().unwrap();

    // Batched prefill: process all prompt tokens at once
    let token_ids: Vec<usize> = input_ids.iter().map(|&t| t as usize).collect();
    let mut logits = model.forward_prefill(&token_ids);

    // Decode loop
    let start_time = Instant::now();
    let mut tokens_generated = 0;

    for _ in 0..max_tokens {
        let next_token = sampling::min_p_sampling(&logits, 0.1);

        let tok_str = tokenizer.decode(&[next_token as u32], false).unwrap_or_default();
        print!("{}", tok_str);
        io::stdout().flush().unwrap();
        tokens_generated += 1;

        logits = model.forward_single_token(next_token);
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    let tps = if elapsed > 0.0 { tokens_generated as f64 / elapsed } else { 0.0 };

    println!("\n\n[Full Context Generation - Metal GPU, Q4_0]");
    println!("  Tokens: {}", tokens_generated);
    println!("  Throughput: {:.2} tok/s", tps);
    println!("  Context length: {} tokens", model.num_items());
    println!("  Elapsed: {:.2}s", elapsed);
}

fn generate_gemma4_gpu(
    prompt: &str,
    tokenizer: &tokenizers::Tokenizer,
    model: &mut gemma4_gpu_model::Gemma4GpuModel,
    max_tokens: usize,
) {
    let encoding = tokenizer.encode(prompt, true).expect("Failed to encode");
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();

    print!("{}", prompt);
    io::stdout().flush().unwrap();

    // Prefill
    let token_ids: Vec<usize> = input_ids.iter().map(|&t| t as usize).collect();
    let mut logits = model.forward_prefill(&token_ids);

    // Decode loop
    let start_time = Instant::now();
    let mut tokens_generated = 0;

    // Gemma4 stop tokens: <eos> (1), <end_of_turn> (107)
    let eos_tokens: &[usize] = &[1, 106];

    for _ in 0..max_tokens {
        let next_token = sampling::min_p_sampling(&logits, 0.1);

        // Stop at EOS or end-of-turn
        if eos_tokens.contains(&next_token) {
            break;
        }

        let tok_str = tokenizer.decode(&[next_token as u32], false).unwrap_or_default();
        print!("{}", tok_str);
        io::stdout().flush().unwrap();
        tokens_generated += 1;

        logits = model.forward_single_token(next_token);
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    let tps = if elapsed > 0.0 { tokens_generated as f64 / elapsed } else { 0.0 };

    println!("\n\n[Gemma4 E4B Generation - Metal GPU, Q4_0]");
    println!("  Tokens: {}", tokens_generated);
    println!("  Throughput: {:.2} tok/s", tps);
    println!("  Context length: {} tokens", model.num_items());
    println!("  Elapsed: {:.2}s", elapsed);
}
