#![allow(dead_code, unused_variables, unused_imports)]

mod config;
mod batch_engine;
mod layers;
mod cache;
mod gpu;
mod gpu_model;
mod gemma4_config;
mod gemma4_gpu_model;
mod gemma4_assistant_config;
mod gemma4_assistant_model;
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

            let tokenizer_path = std::path::Path::new(&model_dir).join("tokenizer.json");
            let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
                .expect("Failed to load tokenizer.json");

            // Optional MTP assistant / draft model
            let assistant_dir = args.iter()
                .position(|a| a == "--assistant-dir")
                .and_then(|i| args.get(i + 1))
                .cloned();

            let assistant = assistant_dir.map(|dir| {
                gemma4_assistant_model::Gemma4AssistantGpuModel::new(&dir)
            });

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
            if assistant.is_some() {
                println!("GEMMA4 E4B GENERATION (Metal GPU, Q4_0, MTP speculative decoding)");
            } else {
                println!("GEMMA4 E4B GENERATION (Metal GPU, Q4_0)");
            }
            println!("{}", "=".repeat(60));

            let prompt = "<start_of_turn>user\n Write a short essay about the benefits of exercise. Include an introduction, 3 key points, and a conclusion.<end_of_turn>\n<start_of_turn>model\n";
            let gen_start = Instant::now();
            if let Some(mut assistant) = assistant {
                generate_gemma4_gpu_speculative(
                    prompt,
                    &tokenizer,
                    &mut gpu_model,
                    &mut assistant,
                    1000,
                    6, // max draft tokens
                );
            } else {
                generate_gemma4_gpu(
                    prompt,
                    &tokenizer,
                    &mut gpu_model,
                    1000,
                );
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

fn generate_gemma4_gpu_speculative(
    prompt: &str,
    tokenizer: &tokenizers::Tokenizer,
    model: &mut gemma4_gpu_model::Gemma4GpuModel,
    assistant: &mut gemma4_assistant_model::Gemma4AssistantGpuModel,
    max_tokens: usize,
    max_draft_tokens: usize,
) {
    use rand::Rng;
    use crate::sampling::{softmax, argmax};

    let encoding = tokenizer.encode(prompt, true).expect("Failed to encode");
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();

    print!("{}", prompt);
    io::stdout().flush().unwrap();

    // Prefill
    let token_ids: Vec<usize> = input_ids.iter().map(|&t| t as usize).collect();
    let logits = model.forward_prefill(&token_ids);

    let start_time = Instant::now();
    let mut tokens_generated = 0;
    let mut accepted_draft_tokens = 0;
    let mut drafted_total = 0;

    let eos_tokens: &[usize] = &[1, 106];

    // next_token is the next token to emit; it has not been emitted yet.
    let mut next_token = sampling::min_p_sampling(&logits, 0.1);
    while tokens_generated < max_tokens && !eos_tokens.contains(&next_token) {
        let tok_str = tokenizer.decode(&[next_token as u32], false).unwrap_or_default();
        print!("{}", tok_str);
        io::stdout().flush().unwrap();
        tokens_generated += 1;

        // Forward the main model to get logits for the next position and the
        // post-final-norm hidden state that seeds the assistant.
        let (mut current_logits, main_hidden) =
            model.forward_single_token_with_hidden_state(next_token);

        // Ask the assistant to draft future tokens from this position.
        let main_kv = model.assistant_kv_view();
        let position_id = main_kv.seq_len.saturating_sub(1);
        let drafts = assistant.draft_tokens(
            next_token,
            &main_hidden,
            &model.embed_tokens_f16,
            model.config.hidden_size,
            &main_kv,
            position_id,
            max_draft_tokens,
            eos_tokens,
        );

        // Standard speculative decoding acceptance.
        // For each draft token x drawn from draft distribution p, we compare it to
        // the main model distribution q at the same position. The token is accepted
        // with probability min(1, q(x) / p(x)). If rejected, we sample a replacement
        // from (q - p)^+ instead. If every draft token is accepted, we sample one
        // bonus token from the main model at the final position.
        let mut all_drafts_accepted = true;
        let draft_ids: Vec<usize> = drafts.iter().map(|(t, _)| *t).collect();
        if mtp_debug_enabled() {
            eprintln!(
                "[spec] seed={} drafts={:?} main_argmax={}",
                next_token,
                draft_ids,
                argmax(&current_logits)
            );
        }

        for (draft_token, draft_logits) in drafts {
            if tokens_generated >= max_tokens {
                all_drafts_accepted = false;
                break;
            }

            drafted_total += 1;
            let q = softmax(&current_logits);
            let p = softmax(&draft_logits);

            let ratio = if p[draft_token] > 1e-12 {
                q[draft_token] / p[draft_token]
            } else {
                f32::INFINITY
            };
            let accept_prob = ratio.min(1.0);
            let u: f32 = rand::thread_rng().gen();

            if u < accept_prob {
                // Accept the draft token: emit it now.
                accepted_draft_tokens += 1;

                let draft_str = tokenizer.decode(&[draft_token as u32], false).unwrap_or_default();
                print!("{}", draft_str);
                io::stdout().flush().unwrap();
                tokens_generated += 1;

                if eos_tokens.contains(&draft_token) || tokens_generated >= max_tokens {
                    next_token = draft_token;
                    all_drafts_accepted = false; // no bonus token when we hit EOS/max
                    break;
                }

                // Advance the main model to get logits for the following position.
                current_logits = model.forward_single_token(draft_token);
            } else {
                // Reject: sample a replacement from (q - p)^+.
                // The replacement becomes the seed for the next iteration.
                all_drafts_accepted = false;
                let mut replacement_probs = vec![0.0f32; q.len()];
                let mut sum = 0.0f32;
                for i in 0..q.len() {
                    let val = (q[i] - p[i]).max(0.0);
                    replacement_probs[i] = val;
                    sum += val;
                }
                next_token = if sum > 1e-12 {
                    for prob in replacement_probs.iter_mut() {
                        *prob /= sum;
                    }
                    sampling::multinomial_sample(&replacement_probs)
                } else {
                    argmax(&current_logits)
                };
                break;
            }
        }

        // If every draft token was accepted, sample one bonus token from the main
        // model at the final position. It becomes the seed for the next iteration.
        if all_drafts_accepted && tokens_generated < max_tokens {
            next_token = sampling::min_p_sampling(&current_logits, 0.1);
        }
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    let tps = if elapsed > 0.0 { tokens_generated as f64 / elapsed } else { 0.0 };

    println!("\n\n[Gemma4 E4B Generation - Metal GPU, Q4_0, MTP speculative]");
    println!("  Tokens: {}", tokens_generated);
    println!("  Throughput: {:.2} tok/s", tps);
    println!(
        "  Drafted: {}, Accepted: {}, Acceptance rate: {:.2}%",
        drafted_total,
        accepted_draft_tokens,
        if drafted_total > 0 {
            100.0 * accepted_draft_tokens as f64 / drafted_total as f64
        } else {
            0.0
        }
    );
    println!("  Context length: {} tokens", model.num_items());
    println!("  Elapsed: {:.2}s", elapsed);
}

fn mtp_debug_enabled() -> bool {
    match std::env::var("MTP_DEBUG") {
        Ok(v) => !v.is_empty() && v != "0",
        Err(_) => false,
    }
}
