# Llama 3.2 Inference with Streaming Attention Sinks (Rust)

A Rust port of `llama3.2_model_numpy_sinks.py` — Llama 3.2 1B inference using streaming attention sinks for constant-memory, constant-speed text generation.

## Features

- **Streaming KV Cache with Attention Sinks**: Keeps the first N "sink" tokens and the most recent M tokens, evicting the middle. This bounds memory and compute regardless of generation length.
- **Fused Gate+Up Projection**: Single BLAS call for the MLP gate and up projections.
- **Optimized Single-Token Decode**: Fused Q×K→softmax→V attention for the common decode case.
- **Causal Masking**: Proper causal attention for multi-token prefill.
- **Min-p Sampling**: Filters low-probability tokens before sampling.
- **Safetensors Loading**: Loads sharded or single-file safetensors weights (f32, f16, bf16).

## Building

```bash
# On macOS with Xcode 16+ / macOS 26 SDK, set deployment target:
export MACOSX_DEPLOYMENT_TARGET=15.0

cargo build --release
```

## Running

```bash
# Default model path (~/Downloads/hub/models--meta-llama--Llama-3.2-1B/...)
cargo run --release

# Custom model path
cargo run --release -- /path/to/llama-3.2-1b
```

The model directory should contain:
- `config.json`
- `tokenizer.json`
- `model.safetensors` (or sharded with `model.safetensors.index.json`)

## Architecture

```
src/
├── main.rs        # Entry point
├── config.rs      # LlamaConfig deserialization
├── weights.rs     # Safetensors loading (f32/f16/bf16)
├── cache.rs       # KVCache + StreamingKVCache (attention sinks)
├── layers.rs      # Linear, RMSNorm, RotaryEmb, Attention, MLP, DecoderLayer
├── model.rs       # LlamaModel, LlamaForCausalLM, load_model()
├── sampling.rs    # min_p_sampling, argmax
└── generation.rs  # generate_streaming, generate_spec_sinks
```

## Configuration

Default generation parameters:
- **Sink size**: 4 tokens (always kept in cache)
- **Window size**: 64 tokens (most recent tokens kept)
- **Max tokens**: 200
- **Min-p threshold**: 0.1
