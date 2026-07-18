#![allow(dead_code, unused_variables, unused_imports)]

mod config;
mod batch_engine;
mod layers;
mod cache;
mod ggml_gemv;
mod ggml_flash_attn;
mod ggml_flash_attn_ext;
mod gguf;
mod gpu;
mod gpu_model;
mod gemma4_config;
mod gemma4_gpu_model;
mod gemma4_mtp;
mod decode_fused;
mod speculative;
mod kv_pool;
mod metrics;
mod model;
mod mtp_serve;
mod quantize;
mod sampling;
mod scheduler;
mod weights;
mod generation;
mod server;
mod token_printer;

use std::io::{self, Write};
use std::time::Instant;
use rand::Rng;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let use_gpu = args.iter().any(|a| a == "--gpu");

    if args.iter().any(|a| a == "--bench-matvec") {
        let ctx = gpu::MetalContext::new();
        ctx.bench_matvec();
        return;
    }
    if args.iter().any(|a| a == "--bench-mul-mm") {
        let ctx = gpu::MetalContext::new();
        ctx.bench_mul_mm();
        return;
    }

    let bench_decode = args.iter().any(|a| a == "--bench-decode");
    let bench_decode_tokens: usize = args
        .iter()
        .position(|a| a == "--bench-decode-tokens")
        .and_then(|i| args.get(i + 1))
        .and_then(|n| n.parse().ok())
        .unwrap_or(256);
    let bench_prefill = args.iter().any(|a| a == "--bench-prefill");
    let bench_prefill_tokens: Vec<usize> = args
        .iter()
        .position(|a| a == "--bench-prefill-tokens")
        .and_then(|i| args.get(i + 1))
        .map(|s| parse_prefill_token_sizes(s))
        .unwrap_or_else(|| vec![128, 256, 512]);

    let model_dir = args.iter()
        .filter(|a| !a.starts_with("--") && *a != &args[0])
        .next()
        .cloned()
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap();
            format!("{}/Downloads/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08", home)
        });

    // Dev helper: build the embedded GGUF tokenizer and sanity-check encode/decode.
    if args.iter().any(|a| a == "--gguf-tok-test") {
        let tok = gguf::build_tokenizer_from_gguf(&model_dir);
        // Control-token atomicity: each should encode to exactly one id.
        for ctrl in ["<|turn>", "<turn|>"] {
            let ids = tok.encode(ctrl, false).expect("encode failed").get_ids().to_vec();
            println!("ctrl {:?} -> {:?} (atomic={})", ctrl, ids, ids.len() == 1);
        }
        for sample in [
            "<|turn>user\nHello, world!<turn|>\n<|turn>model\n",
            "The quick brown fox jumps over the lazy dog.",
        ] {
            let enc = tok.encode(sample, true).expect("encode failed");
            let ids = enc.get_ids();
            let decoded_no_special = tok.decode(ids, true).unwrap_or_default();
            println!("--- sample ---");
            println!("text:           {:?}", sample);
            println!("ids:            {:?}", ids);
            println!("decoded(skip):  {:?}", decoded_no_special);
            println!("roundtrip_ok:   {}", decoded_no_special == sample);
        }
        return;
    }

    // Dev helper: validate the native K-quant matvec kernels (Q4_K / Q6_K)
    // against the CPU dequant reference, using real tensors from the GGUF.
    if args.iter().any(|a| a == "--gguf-kquant-test") {
        let g = gguf::Gguf::open(&model_dir);
        let ctx = gpu::MetalContext::new();
        let candidates = [
            "token_embd.weight",
            "output.weight",
            "blk.0.attn_v.weight",
            "blk.0.ffn_down.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.attn_q.weight",
        ];
        let mut tested = 0;
        for name in candidates {
            if !g.has_tensor(name) {
                continue;
            }
            let t = g.tensor_type(name);
            let fmt = match t {
                gguf::ggml_type::Q4_K => gpu::weight_fmt::Q4_K,
                gguf::ggml_type::Q6_K => gpu::weight_fmt::Q6_K,
                _ => continue,
            };
            let info = g.tensor(name).unwrap();
            let k = info.ne0(); // reduction / in-dim
            if k % 256 != 0 {
                continue;
            }
            let m = info.n_rows().min(2048); // test a leading row block
            let (_, bpb) = gguf::type_block_spec(t);
            let row_bytes = (k / 256) * bpb;
            let raw = &g.tensor_raw(name)[..m * row_bytes];

            // Deterministic input vector.
            let x: Vec<f32> = (0..k).map(|j| ((j % 17) as f32 - 8.0) * 0.05).collect();

            // CPU reference.
            let w = gguf::dequant_type_to_f32(t, raw, m * k);
            let mut y_ref = vec![0.0f32; m];
            for r in 0..m {
                let mut acc = 0.0f32;
                let base = r * k;
                for j in 0..k {
                    acc += w[base + j] * x[j];
                }
                y_ref[r] = acc;
            }

            // GPU kernel.
            let w_view = gpu::BufferView::from_buffer(ctx.buffer_from_bytes(raw)).with_format(fmt);
            let x_buf = ctx.buffer_from_slice(&x);
            let y_buf = ctx.buffer_empty(m);
            let cmd = ctx.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            ctx.encode_matvec_qk_at_view(enc, &w_view, &x_buf, 0, &y_buf, 0, m as u32, k as u32, 1);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            let y_gpu =
                unsafe { std::slice::from_raw_parts(y_buf.contents() as *const f32, m) };

            let mut max_abs = 0.0f32;
            let mut max_rel = 0.0f32;
            for r in 0..m {
                let d = (y_gpu[r] - y_ref[r]).abs();
                max_abs = max_abs.max(d);
                let denom = y_ref[r].abs().max(1e-3);
                max_rel = max_rel.max(d / denom);
            }
            println!(
                "{:<26} type={:<5} m={:<6} k={:<6} max_abs_err={:.3e} max_rel_err={:.3e} {}",
                name,
                gguf::ggml_type_name(t),
                m,
                k,
                max_abs,
                max_rel,
                if max_rel < 1e-3 { "OK" } else { "FAIL" }
            );
            tested += 1;

            // Batched mul_mm (prefill) vs CPU reference when seq > 8.
            let seq_len: u32 = 16;
            if crate::ggml_gemv::should_use_mul_mm(k as u32, seq_len) {
                let x_batch: Vec<f32> = (0..seq_len as usize)
                    .flat_map(|s| (0..k).map(move |j| ((j + s * 7) % 17) as f32 * 0.05 - 0.4))
                    .collect();
                let mut y_ref = vec![0.0f32; m * seq_len as usize];
                for s in 0..seq_len as usize {
                    for r in 0..m {
                        let mut acc = 0.0f32;
                        let base_w = r * k;
                        let base_x = s * k;
                        for j in 0..k {
                            acc += w[base_w + j] * x_batch[base_x + j];
                        }
                        y_ref[s * m + r] = acc;
                    }
                }
                let x_buf = ctx.buffer_from_slice(&x_batch);
                let y_buf = ctx.buffer_empty(m * seq_len as usize);
                let cmd = ctx.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                ctx.encode_mul_mm_kquant_at_view(
                    enc,
                    &w_view,
                    &x_buf,
                    &y_buf,
                    m as u32,
                    k as u32,
                    seq_len,
                );
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                let y_gpu = unsafe {
                    std::slice::from_raw_parts(y_buf.contents() as *const f32, m * seq_len as usize)
                };
                let mut mm_max_rel = 0.0f32;
                for i in 0..y_ref.len() {
                    let d = (y_gpu[i] - y_ref[i]).abs();
                    let denom = y_ref[i].abs().max(1e-3);
                    mm_max_rel = mm_max_rel.max(d / denom);
                }
                println!(
                    "  mul_mm seq={:<3} max_rel_err={:.3e} {}",
                    seq_len,
                    mm_max_rel,
                    if mm_max_rel < 1e-3 { "OK" } else { "FAIL" }
                );
            }
        }
        if tested == 0 {
            println!("No Q4_K/Q6_K tensors found in {} (nothing to test).", model_dir);
        }
        return;
    }

    // Dev helper: load a GGUF model on GPU and greedy-decode a short prompt.
    if args.iter().any(|a| a == "--gguf-gen") {
        let mut model = gemma4_gpu_model::Gemma4GpuModel::load_from_gguf(&model_dir);
        let tok = gguf::build_tokenizer_from_gguf(&model_dir);
        let prompt = "<|turn>user\nWhat is the capital of France? Answer in one sentence.<turn|>\n<|turn>model\n";
        let ids: Vec<usize> = tok
            .encode(prompt, true)
            .expect("encode")
            .get_ids()
            .iter()
            .map(|&t| t as usize)
            .collect();
        println!("Prompt ids ({}): {:?}", ids.len(), ids);
        let mut next = model.forward_prefill_sample_last(&ids, 0.0, 0.0, 0);
        let eos: &[usize] = &[1, 106];
        let printer = token_printer::TokenPrinter::spawn(&tok);
        for _ in 0..60 {
            if eos.contains(&next) {
                break;
            }
            printer.send(next as u32);
            next = model.forward_single_token_sample(next, 0.0, 0.0, 0);
        }
        let out = printer.finish();
        println!("\n=== GGUF greedy generation ===\n{}", out);
        return;
    }

    let sink_size = 4;
    let window_size = 64;

    if use_gpu {
        // A `.gguf` path is loaded directly (weights + embedded tokenizer).
        let is_gguf = model_dir.ends_with(".gguf");
        // Otherwise, detect a Gemma4 HF model dir by checking config.json.
        let is_gemma4 = is_gguf || {
            let config_path = std::path::Path::new(&model_dir).join("config.json");
            let config_str = std::fs::read_to_string(&config_path)
                .expect("Failed to read config.json");
            config_str.contains("\"gemma4\"") || config_str.contains("text_config")
        };

        if is_gemma4 {
            let start = Instant::now();
            let (mut gpu_model, tokenizer) = if is_gguf {
                println!("Loading Gemma4 model (GGUF) from: {}", model_dir);
                let model = gemma4_gpu_model::Gemma4GpuModel::load_from_gguf(&model_dir);
                let tokenizer = gguf::build_tokenizer_from_gguf(&model_dir);
                (model, tokenizer)
            } else {
                println!("Loading Gemma4 model (GPU/Metal) from: {}", model_dir);
                let model = gemma4_gpu_model::Gemma4GpuModel::new(&model_dir);
                let tokenizer_path = std::path::Path::new(&model_dir).join("tokenizer.json");
                let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
                    .expect("Failed to load tokenizer.json");
                (model, tokenizer)
            };

            println!("Model loaded in {:.2}s", start.elapsed().as_secs_f64());

            if bench_decode {
                bench_decode_gemma4(
                    &tokenizer,
                    &mut gpu_model,
                    bench_decode_tokens,
                );
                return;
            }

            if bench_prefill {
                bench_prefill_gemma4(
                    &tokenizer,
                    &mut gpu_model,
                    &bench_prefill_tokens,
                );
                return;
            }

            // Serve mode: start OpenAI-compatible HTTP server
            if args.iter().any(|a| a == "--serve") {
                let port: u16 = args.iter()
                    .position(|a| a == "--port")
                    .and_then(|i| args.get(i + 1))
                    .and_then(|p| p.parse().ok())
                    .unwrap_or(8080);

                // Optional MTP speculative decoding: `--serve --mtp <draft_path>`.
                let serve_mtp_path = args.iter()
                    .position(|a| a == "--mtp")
                    .and_then(|i| args.get(i + 1))
                    .cloned();

                let mtp_assistant = serve_mtp_path.map(|draft_path| {
                    println!("Loading MTP draft head from: {}", draft_path);
                    gemma4_mtp::Gemma4MtpAssistant::new(&gpu_model.ctx, &draft_path, &gpu_model)
                });

                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(server::run_server_with_mtp(
                    gpu_model,
                    tokenizer,
                    port,
                    mtp_assistant,
                ));
                return;
            }

            // Interactive generation mode
            let mtp_draft_path = args.iter()
                .position(|a| a == "--mtp")
                .and_then(|i| args.get(i + 1))
                .cloned();

            if let Some(draft_path) = mtp_draft_path {
                println!("{}", "=".repeat(60));
                println!("GEMMA4 E2B GENERATION (Metal GPU, MTP)");
                println!("{}", "=".repeat(60));

                let gen_start = Instant::now();
                let mut assistant = gemma4_mtp::Gemma4MtpAssistant::new(&gpu_model.ctx, &draft_path, &gpu_model);
                generate_gemma4_gpu_mtp(
                    //"<start_of_turn>user\n Write a short essay about the benefits of exercise. Include an introduction, 3 key points, and a conclusion.<end_of_turn>\n<start_of_turn>model\n",
                    "<start_of_turn>user\n def bubble_sort<end_of_turn>\n<start_of_turn>model\n",

                    &tokenizer,
                    &mut gpu_model,
                    &mut assistant,
                    1000,
                );
                println!("\nTotal time: {:.2}s", gen_start.elapsed().as_secs_f64());
            } else {
                println!("{}", "=".repeat(60));
                println!("GEMMA4 E4B GENERATION (Metal GPU, Q4_0)");
                println!("{}", "=".repeat(60));

                let gen_start = Instant::now();
                generate_gemma4_gpu(
                    "<start_of_turn>user\n Write a short essay about the benefits of exercise. Include an introduction, 3 key points, and a conclusion.<end_of_turn>\n<start_of_turn>model\n",
                    &tokenizer,
                    &mut gpu_model,
                    1000,
                );
                println!("\nTotal time: {:.2}s", gen_start.elapsed().as_secs_f64());
            }
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
    let mut next_token = sampling::min_p_sampling(&logits, 0.1);

    let printer = token_printer::TokenPrinter::spawn(tokenizer);

    // Decode loop: GPU first, print on background thread
    let start_time = Instant::now();
    let mut tokens_generated = 0;

    for _ in 0..max_tokens {
        printer.send(next_token as u32);
        tokens_generated += 1;

        logits = model.forward_single_token(next_token);
        next_token = sampling::min_p_sampling(&logits, 0.1);
    }

    printer.finish();
    let elapsed = start_time.elapsed().as_secs_f64();
    let tps = if elapsed > 0.0 { tokens_generated as f64 / elapsed } else { 0.0 };

    println!("\n\n[Full Context Generation - Metal GPU, Q4_0]");
    println!("  Tokens: {}", tokens_generated);
    println!("  Throughput: {:.2} tok/s", tps);
    println!("  Context length: {} tokens", model.num_items());
    println!("  Elapsed: {:.2}s", elapsed);
}

/// llama.cpp-style benchmark: prefill and decode timed separately, no stdout I/O
/// inside the decode loop (matches `Prompt: X t/s | Generation: Y t/s` reporting).
fn bench_decode_gemma4(
    tokenizer: &tokenizers::Tokenizer,
    model: &mut gemma4_gpu_model::Gemma4GpuModel,
    gen_tokens: usize,
) {
    // Plain prompt for token count; wrap in Gemma chat template so greedy decode
    // does not immediately sample <end_of_turn> (happens without <start_of_turn>model).
    let plain = "Write a short essay about the benefits of exercise. Include an introduction, 3 key points, and a conclusion.";
    let prompt = format!(
        "<start_of_turn>user\n{plain}<end_of_turn>\n<start_of_turn>model\n"
    );
    let encoding = tokenizer.encode(prompt.as_str(), true).expect("Failed to encode");
    let token_ids: Vec<usize> = encoding.get_ids().iter().map(|&t| t as usize).collect();
    let prompt_tokens = token_ids.len();

    // Prefill
    let prefill_start = Instant::now();
    let mut rng = rand::thread_rng();
    let seed: u32 = rng.gen();
    let mut next_token =
        model.forward_prefill_sample_last(&token_ids, 0.0, 0.0, seed);
    let prefill_secs = prefill_start.elapsed().as_secs_f64();
    let prefill_tps = if prefill_secs > 0.0 {
        prompt_tokens as f64 / prefill_secs
    } else {
        0.0
    };

    // Decode — no print/flush/tokenizer in the timed section
    let decode_start = Instant::now();
    let mut generated = 0usize;
    let eos_tokens: &[usize] = &[1, 106];
    for _ in 0..gen_tokens {
        if eos_tokens.contains(&next_token) {
            break;
        }
        generated += 1;
        let seed: u32 = rng.gen();
        next_token = model.forward_single_token_sample(next_token, 0.0, 0.0, seed);
    }
    let decode_secs = decode_start.elapsed().as_secs_f64();
    let decode_tps = if decode_secs > 0.0 {
        generated as f64 / decode_secs
    } else {
        0.0
    };

    println!("\n=== Gemma4 E4B decode benchmark (Metal GPU) ===");
    println!("  Prompt tokens: {}", prompt_tokens);
    println!("  Generated tokens: {}", generated);
    println!("  Context after bench: {} tokens", model.num_items());
    println!(
        "  [ Prompt: {:.1} t/s | Generation: {:.1} t/s ]",
        prefill_tps, decode_tps
    );
    println!("  (llama.cpp reports the same Prompt/Generation split; no stdout in decode loop)");
}

fn parse_prefill_token_sizes(s: &str) -> Vec<usize> {
    s.split(',')
        .filter_map(|p| p.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .collect()
}

/// Timed parallel prefill via KV-pool path (same as server), no decode.
fn bench_prefill_gemma4(
    tokenizer: &tokenizers::Tokenizer,
    model: &mut gemma4_gpu_model::Gemma4GpuModel,
    sizes: &[usize],
) {
    let filler = "The quick brown fox jumps over the lazy dog. ";
    println!("\n=== Gemma4 parallel prefill benchmark (Metal GPU) ===");
    println!(
        "  max_parallel_prefill_seq={}  PREFILL_MUL_MM={}  PREFILL_FLASH_ATTN={}  PREFILL_GATE_UP_STACKED={}  PREFILL_QKV_STACKED={}  PREFILL_QKV_HSD={}",
        model.max_parallel_prefill_seq(),
        std::env::var("PREFILL_MUL_MM").unwrap_or_else(|_| "1".into()),
        std::env::var("PREFILL_FLASH_ATTN").unwrap_or_else(|_| "1 (default)".into()),
        std::env::var("PREFILL_GATE_UP_STACKED").unwrap_or_else(|_| "1".into()),
        std::env::var("PREFILL_QKV_STACKED").unwrap_or_else(|_| "1".into()),
        std::env::var("PREFILL_QKV_HSD").unwrap_or_else(|_| "1".into())
    );

    for &target in sizes {
        model.reset_legacy_state();
        let mut text = String::from("<start_of_turn>user\n");
        while tokenizer
            .encode(text.as_str(), true)
            .map(|e| e.get_ids().len())
            .unwrap_or(0)
            < target
        {
            text.push_str(filler);
        }
        text.push_str("<end_of_turn>\n<start_of_turn>model\n");
        let mut token_ids: Vec<usize> = tokenizer
            .encode(text.as_str(), true)
            .expect("Failed to encode bench prefill prompt")
            .get_ids()
            .iter()
            .map(|&t| t as usize)
            .collect();
        // Exact length for fair tile-alignment A/B (flash FC needs kv%64==0, q%8==0).
        if std::env::var("BENCH_PREFILL_EXACT").is_ok() && token_ids.len() > target {
            token_ids.truncate(target);
        }
        let actual = token_ids.len();

        let mut kv_pool = model.create_kv_pool(1, model.kv_capacity);
        let slot = kv_pool
            .allocate()
            .expect("Failed to allocate prefill benchmark KV slot");

        let start = Instant::now();
        model
            .forward_prefill_chunked_with_kv_slot(&token_ids, &mut kv_pool, slot)
            .expect("prefill benchmark failed");
        let secs = start.elapsed().as_secs_f64();
        let tps = if secs > 0.0 { actual as f64 / secs } else { 0.0 };

        println!(
            "  target={:<4} actual={:<4} prefill={:.1} tok/s  ({:.2} ms)",
            target,
            actual,
            tps,
            secs * 1000.0
        );
    }
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

    // Prefill: advance intermediate tokens, GPU-sample the last (no full-vocab readback).
    let token_ids: Vec<usize> = input_ids.iter().map(|&t| t as usize).collect();
    let mut rng = rand::thread_rng();
    let seed: u32 = rng.gen();
    let mut next_token =
        model.forward_prefill_sample_last(&token_ids, 0.7, 0.1, seed);

    // Decode loop
    let start_time = Instant::now();
    let mut tokens_generated = 0;

    // Gemma4 stop tokens: <eos> (1), <end_of_turn> (106)
    let eos_tokens: &[usize] = &[1, 106];

    let printer = token_printer::TokenPrinter::spawn(tokenizer);

    for _ in 0..max_tokens {
        if eos_tokens.contains(&next_token) {
            break;
        }

        printer.send(next_token as u32);
        tokens_generated += 1;

        let seed: u32 = rng.gen();
        next_token = model.forward_single_token_sample(next_token, 0.7, 0.1, seed);
    }

    printer.finish();
    let elapsed = start_time.elapsed().as_secs_f64();
    let tps = if elapsed > 0.0 { tokens_generated as f64 / elapsed } else { 0.0 };

    println!("\n\n[Gemma4 E4B Generation - Metal GPU, Q4_0]");
    println!("  Tokens: {}", tokens_generated);
    println!("  Throughput: {:.2} tok/s", tps);
    println!("  Context length: {} tokens", model.num_items());
    println!("  Elapsed: {:.2}s", elapsed);
}

// ─── MTP Draft/Verify generation helpers ─────────────────────────────────────

fn parse_mtp_draft_steps() -> usize {
    std::env::var("LLAMA_MTP_DRAFT_STEPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(4)
}

fn parse_mtp_adaptive() -> bool {
    std::env::var("LLAMA_MTP_ADAPTIVE")
        .map(|value| value == "1" || value.to_ascii_lowercase() == "true")
        .unwrap_or(false)
}

/// Optional draft confidence floor (0 = disabled). When set, stop the draft
/// chain early if the greedy token's softmax probability is below this value.
fn parse_mtp_p_min() -> f32 {
    std::env::var("LLAMA_MTP_P_MIN")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|&v| v > 0.0 && v < 1.0)
        .unwrap_or(0.0)
}

/// Google-style heuristic: don't draft N-1 tail tokens when recent accept depth is low.
fn adaptive_draft_tail_steps(accept_history: &[usize], max_steps: usize) -> usize {
    if max_steps <= 1 {
        return 0;
    }
    if accept_history.is_empty() {
        return max_steps - 1;
    }
    let window = accept_history.len().min(12);
    let recent = &accept_history[accept_history.len() - window..];
    let avg = recent.iter().sum::<usize>() as f64 / window as f64;
    if avg >= 3.0 {
        max_steps - 1
    } else if avg >= 2.0 {
        (max_steps - 1).min(2)
    } else if avg >= 1.2 {
        1
    } else {
        // Keep one tail token as a probe. Returning zero is self-locking:
        // cycles with only d0 can accept at most one token, so their average
        // can never reach the 1.2 threshold needed to enable tails again.
        1
    }
}

fn context_limited_draft_tail_steps(tail_steps: usize, context_len: usize, sliding_window: usize) -> usize {
    if tail_steps == 0 {
        return 0;
    }
    if context_len > sliding_window.saturating_mul(2) {
        return 0;
    }
    if context_len > sliding_window {
        return tail_steps.min(1);
    }
    if context_len > sliding_window * 3 / 4 {
        return tail_steps.min(2);
    }
    tail_steps
}

fn effective_draft_tail_steps(
    accept_history: &[usize],
    max_steps: usize,
    mtp_adaptive: bool,
    context_len: usize,
    sliding_window: usize,
) -> usize {
    let tail = if mtp_adaptive {
        adaptive_draft_tail_steps(accept_history, max_steps)
    } else {
        max_steps.saturating_sub(1)
    };
    context_limited_draft_tail_steps(tail, context_len, sliding_window)
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

    // Verify defaults to parallel prefill (`forward_verify_parallel`).
    // Opt-in: MTP_VERIFY_SEQUENTIAL=1, MTP_VERIFY_DECODE_BATCH=1.
    let use_parallel_prefill = std::env::var("MTP_PREFILL_PARALLEL")
        .map(|v| v == "1")
        .unwrap_or(false);

    let mut kv_pool = model.create_kv_pool(1, model.kv_capacity);
    let kv_slot = kv_pool
        .allocate()
        .expect("failed to allocate MTP KV slot");

    let logits = if use_parallel_prefill {
        eprintln!("  [MTP] Parallel prefill + decode-faithful verify");
        let logits = model
            .forward_prefill_pool(&token_ids, &mut kv_pool, kv_slot)
            .expect("MTP parallel prefill failed");
        model
            .alias_kv_from_pool(&kv_pool, kv_slot)
            .expect("alias MTP KV failed");
        logits
    } else {
        eprintln!("  [MTP] Decode prefill + decode-faithful verify");
        model.forward_prefill(&token_ids)
    };

    // Upstream: keep id_last out of KV until verify batch [id_last, drafts…].
    let mut id_last = sampling::argmax(&logits);
    let mut mtp_hidden = model.last_hidden_activation();

    if !print_gemma_token(id_last, tokenizer, &[1, 106]) {
        return;
    }

    let start_time = Instant::now();
    let mut tokens_generated = 1usize;
    let mut drafted_total = 0usize;
    let mut accepted_total = 0usize;
    let mut rejected_total = 0usize;
    let mut main_forwards = 0usize;
    let draft_steps = parse_mtp_draft_steps();
    let mtp_adaptive = parse_mtp_adaptive();
    let mtp_p_min = parse_mtp_p_min();
    let mtp_debug = std::env::var("LLAMA_MTP_DEBUG").is_ok();
    let mut mtp_debug_cycles = 0usize;
    let mut accept_history: Vec<usize> = Vec::new();
    let mut draft_us = 0u128;
    let mut verify_us = 0u128;
    let mut main_other_us = 0u128;
    let eos_tokens: &[usize] = &[1, 106];

    'outer: while tokens_generated < max_tokens {
        let tail_steps = effective_draft_tail_steps(
            &accept_history,
            draft_steps,
            mtp_adaptive,
            model.kv_seq_len as usize,
            model.config.sliding_window,
        );
        let n_draft = tail_steps + 1;

        let t_draft = Instant::now();
        let drafted = assistant
            .draft_chain(id_last, &mtp_hidden, n_draft, model, mtp_p_min)
            .expect("MTP assistant draft failed");
        drafted_total += drafted.len();
        draft_us += t_draft.elapsed().as_micros();

        if mtp_debug && mtp_debug_cycles < 5 {
            eprintln!(
                "MTP debug cycle {}: id_last={} n_past={} drafts={:?}",
                mtp_debug_cycles, id_last, model.total_tokens, drafted
            );
            mtp_debug_cycles += 1;
        }

        if drafted.is_empty() {
            // Rare fallback: append id_last with decode. Mixing writers can
            // disturb a parallel session — keep drafts non-empty in practice.
            let t0 = Instant::now();
            let next_logits = model.forward_single_token(id_last);
            main_forwards += 1;
            main_other_us += t0.elapsed().as_micros();
            if use_parallel_prefill {
                let _ = kv_pool.with_slot_mut(kv_slot, |s| {
                    s.seq_len = model.kv_seq_len;
                    s.total_tokens = model.total_tokens;
                });
            }
            mtp_hidden = model.last_hidden_activation();
            id_last = sampling::argmax(&next_logits);
            if !print_gemma_token(id_last, tokenizer, eos_tokens) {
                break;
            }
            tokens_generated += 1;
            accept_history.push(0);
            rejected_total += 1;
            continue;
        }

        let mut verify_batch = Vec::with_capacity(drafted.len() + 1);
        verify_batch.push(id_last);
        verify_batch.extend_from_slice(&drafted);

        let t_verify = Instant::now();
        let verify_tokens = model
            .forward_verify_batch(&verify_batch)
            .expect("MTP verify batch failed");
        if use_parallel_prefill {
            let _ = kv_pool.with_slot_mut(kv_slot, |s| {
                s.seq_len = model.kv_seq_len;
                s.total_tokens = model.total_tokens;
            });
        }
        main_forwards += 1;
        verify_us += t_verify.elapsed().as_micros();

        let mut ids: Vec<usize> = Vec::with_capacity(drafted.len() + 1);
        let mut n_draft_accepted = 0usize;
        for i in 0..drafted.len() {
            let pred = verify_tokens[i];
            ids.push(pred);
            if mtp_debug && mtp_debug_cycles <= 5 {
                eprintln!(
                    "  draft[{}]: drafted={} verifier={} match={}",
                    i,
                    drafted[i],
                    pred,
                    pred == drafted[i]
                );
            }
            if pred != drafted[i] {
                break;
            }
            n_draft_accepted += 1;
        }
        if n_draft_accepted == drafted.len() {
            ids.push(verify_tokens[drafted.len()]);
        }

        accepted_total += n_draft_accepted;
        accept_history.push(n_draft_accepted);
        if n_draft_accepted < drafted.len() {
            rejected_total += 1;
        }

        let rewind = (drafted.len() - n_draft_accepted) as u32;
        if rewind > 0 {
            model.truncate_kv(rewind);
            if use_parallel_prefill {
                let _ = kv_pool.with_slot_mut(kv_slot, |s| {
                    s.seq_len = model.kv_seq_len;
                    s.total_tokens = model.total_tokens;
                });
            }
        }

        let i_h = n_draft_accepted.min(verify_batch.len() - 1);
        mtp_hidden = model.prefill_hidden_activation_at(i_h);

        for &tok in &ids {
            if tokens_generated >= max_tokens {
                break 'outer;
            }
            if !print_gemma_token(tok, tokenizer, eos_tokens) {
                break 'outer;
            }
            tokens_generated += 1;
        }

        id_last = *ids.last().unwrap();
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    let tps = if elapsed > 0.0 { tokens_generated as f64 / elapsed } else { 0.0 };
    let tokens_per_forward = if main_forwards > 0 {
        tokens_generated as f64 / main_forwards as f64
    } else {
        0.0
    };
    let accept_rate = if drafted_total > 0 {
        accepted_total as f64 * 100.0 / drafted_total as f64
    } else {
        0.0
    };
    let assistant_passes = assistant.gpu_passes;
    let total_us = draft_us + verify_us + main_other_us;
    let draft_pct = if total_us > 0 {
        draft_us as f64 * 100.0 / total_us as f64
    } else {
        0.0
    };
    let verify_pct = if total_us > 0 {
        verify_us as f64 * 100.0 / total_us as f64
    } else {
        0.0
    };
    let main_pct = if total_us > 0 {
        main_other_us as f64 * 100.0 / total_us as f64
    } else {
        0.0
    };

    println!("\n\n[Gemma4 E2B Generation - Metal GPU, MTP assistant]");
    println!("  Tokens: {}", tokens_generated);
    println!("  Throughput: {:.2} tok/s", tps);
    println!("  Main-model forwards: {}", main_forwards);
    println!("  Assistant GPU passes: {} ({:.1}x main forwards)", assistant_passes, assistant_passes as f64 / main_forwards.max(1) as f64);
    println!("  Tokens / main forward: {:.2} (need ~2.0+ for 2x speedup)", tokens_per_forward);
    println!(
        "  Wall time: draft {:.0}% | verify {:.0}% | main-other {:.0}%",
        draft_pct, verify_pct, main_pct
    );
    println!("  Context length: {} tokens", model.num_items());
    println!("  Drafted: {}", drafted_total);
    println!("  Accepted: {} ({:.1}%)", accepted_total, accept_rate);
    println!("  Rejected cycles: {}", rejected_total);
    println!(
        "  Draft steps: {} (max), adaptive={}",
        draft_steps, mtp_adaptive
    );
    println!("  Elapsed: {:.2}s", elapsed);
}
