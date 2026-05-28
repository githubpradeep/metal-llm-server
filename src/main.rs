#![allow(dead_code, unused_variables, unused_imports)]

mod config;
mod layers;
mod cache;
mod model;
mod quantize;
mod sampling;
mod weights;
mod generation;

use std::time::Instant;

fn main() {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap();
            format!("{}/Downloads/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08", home)
        });

    println!("Loading model from: {}", model_dir);
    let start = Instant::now();

    let (tokenizer, model, config) = model::load_model(&model_dir);
    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f64());

    let sink_size = 4;
    let window_size = 64;

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
