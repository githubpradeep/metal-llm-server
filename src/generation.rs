use std::io::{self, Write};
use std::time::Instant;

use tokenizers::Tokenizer;

use crate::cache::StreamingKVCache;
use crate::model::LlamaForCausalLM;
use crate::sampling::min_p_sampling;
use crate::token_printer;

/// Generate text using streaming attention sinks.
/// The attention window is bounded to (sink_size + window_size) tokens,
/// so each forward pass does attention over a FIXED number of tokens
/// regardless of how long the generation runs.
pub fn generate_streaming(
    prompt: &str,
    tokenizer: &Tokenizer,
    model: &LlamaForCausalLM,
    max_tokens: usize,
    sink_size: usize,
    window_size: usize,
) -> String {
    let mut kv_cache = StreamingKVCache::new(sink_size, window_size);

    // Encode prompt
    let encoding = tokenizer.encode(prompt, true).expect("Failed to encode prompt");
    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

    let mut result = prompt.to_string();
    print!("{}", prompt);
    io::stdout().flush().unwrap();

    // Prefill: feed prompt (chunked if needed)
    let logits = if input_ids.len() <= sink_size + window_size {
        // Prompt fits in window — single prefill pass
        model.forward(&[input_ids.clone()], &mut kv_cache)
    } else {
        // Prompt exceeds window — feed in chunks
        let chunk_size = window_size;
        let mut last_logits = None;
        for chunk in input_ids.chunks(chunk_size) {
            let chunk_vec: Vec<i64> = chunk.to_vec();
            last_logits = Some(model.forward(&[chunk_vec], &mut kv_cache));
        }
        last_logits.unwrap()
    };

    // Get logits for last token
    let seq_len = logits.shape()[1];
    let vocab_size = logits.shape()[2];
    let last_logits: Vec<f32> = (0..vocab_size)
        .map(|v| logits[[0, seq_len - 1, v]])
        .collect();

    // Decode loop: sample + forward on main thread; decode/print on background thread
    let start_time = Instant::now();
    let mut tokens_generated = 0;
    let mut current_logits = last_logits;
    let mut next_token = min_p_sampling(&current_logits, 0.1);
    let printer = token_printer::TokenPrinter::spawn(tokenizer);

    for _ in 0..max_tokens {
        printer.send(next_token as u32);
        tokens_generated += 1;

        let logits = model.forward(&[vec![next_token as i64]], &mut kv_cache);
        current_logits = (0..vocab_size).map(|v| logits[[0, 0, v]]).collect();
        next_token = min_p_sampling(&current_logits, 0.1);
    }

    result.push_str(&printer.finish());
    let elapsed = start_time.elapsed().as_secs_f64();
    let tps = if elapsed > 0.0 {
        tokens_generated as f64 / elapsed
    } else {
        0.0
    };

    println!("\n\n[Streaming Sinks Generation]");
    println!("  Tokens: {}", tokens_generated);
    println!("  Throughput: {:.2} tok/s", tps);
    println!(
        "  Window: {} sink + {} recent = {} total attention",
        sink_size,
        window_size,
        sink_size + window_size
    );
    println!("  Elapsed: {:.2}s", elapsed);

    result
}

/// Speculative decoding where the draft model uses a small streaming window
/// and the verifier uses a full unbounded KV cache.
///
/// Note: Full speculative decoding requires running the model with two different
/// cache types simultaneously. For production use, you'd want the model to accept
/// a generic cache trait.
#[allow(dead_code)]
pub fn generate_spec_sinks(
    prompt: &str,
    tokenizer: &Tokenizer,
    model: &LlamaForCausalLM,
    max_tokens: usize,
    sink_size: usize,
    window_size: usize,
    _speculation_length: usize,
) -> String {
    println!("[Note: Speculative decoding requires dual-cache support.]");
    println!("[Falling back to streaming generation.]");
    generate_streaming(prompt, tokenizer, model, max_tokens, sink_size, window_size)
}
