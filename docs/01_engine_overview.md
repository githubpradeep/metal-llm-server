# Engine Overview

## What This Is

`llama-sinks` is a from-scratch LLM inference engine and OpenAI-compatible server for Apple Silicon, written in Rust with Metal GPU compute shaders. It runs Google's **Gemma4 E4B** (4.5B) and **Llama 3.2 1B** models entirely on GPU using **Q4_0 4-bit quantized weights**, achieving ~29-33 tok/s on Apple M1 Pro.

## Engines

| Engine | Models | Architecture | Status |
|--------|--------|-------------|--------|
| **Gemma4 GPU** (`gemma4_gpu_model.rs`) | Gemma4 E4B | 42 layers, sliding + full attention, PLE, QK-norm, logit softcapping, shared KV | Primary |
| **Llama GPU** (`gpu_model.rs`) | Llama 3.2 1B | 16 layers, GQA, RoPE | Legacy |
| **Llama CPU** (`model.rs` + `layers.rs`) | Llama 3.2 1B | Apple Accelerate BLAS | Reference |

## Architecture

```
Client (OpenAI SDK / curl)
    │
    ▼
┌──────────────────────┐
│ HTTP Server (axum)   │  server.rs
│ POST /v1/chat        │
│ GET /v1/models       │
│ GET /health, /metrics│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Scheduler            │  scheduler.rs
│ Request admission    │
│ Fair prefill/decode  │
│ Timeouts, cancel     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Batch Engine         │  batch_engine.rs
│ KV pool bridge       │
│ Batch prefill/decode │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ GPU Model (Metal)    │  gemma4_gpu_model.rs
│ Forward pass         │
│ Single cmd buffer    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Metal Shaders        │  shaders/*.metal
│ Q4 matvec (8+ vars)  │
│ Flash attention      │
│ RMSNorm, RoPE, etc.  │
│ Mega decode kernel   │
└──────────────────────┘
```

## Key Components

### Runtime

| Component | File | Role |
|-----------|------|------|
| **Server** | `server.rs` | Axum HTTP server, chat completions (sync + SSE), tokenization, stop detection, tool call parsing |
| **Scheduler** | `scheduler.rs` | Admission control, KV pool slot management, round-robin prefill with fair token budgets, timeout/cancellation |
| **Batch Engine** | `batch_engine.rs` | Bridges scheduler to model, manages batch sizing for prefill/decode |
| **KV Pool** | `kv_pool.rs` | GPU KV cache slot allocation/release per request, supports f16/Q8_0/Q4_0 cache formats |
| **Metrics** | `metrics.rs` | Prometheus counters/gauges for throughput, latency, batch sizes, active requests |

### Model & Inference

| Component | File | Role |
|-----------|------|------|
| **GPU Model (Gemma4)** | `gemma4_gpu_model.rs` | Core inference — weight loading, Q4 cache, forward pass, prefill, decode, batching |
| **GPU Context** | `gpu.rs` | Metal device/queue, ~100 compiled pipeline states, encoder helpers for all operations |
| **Config (Gemma4)** | `gemma4_config.rs` | Deserializes Gemma4 config: per-layer attention types, head dims, RoPE, PLE, KV sharing, softcapping |
| **Config (Llama)** | `config.rs` | Llama config deserialization |
| **Weights** | `weights.rs` | SafeTensors loading (sharded), bf16/f16 → f32 conversion |
| **Sampling** | `sampling.rs` | Greedy, temperature, min-p, top-k, repetition/frequency penalty, multinomial |
| **Mega Decode** | `mega_decode.rs` | Single-dispatch kernel — builds GPU op graph once, replays per token |
| **Flash Attention** | `ggml_flash_attn.rs` | GGML-style flash attention port |
| **GGML GEMV** | `ggml_gemv.rs` | GGML-style Q4 matvec port |

### Shaders (`src/shaders/`)

| Shader | Contents |
|--------|----------|
| `llama.metal` | ~50+ kernels: Q4 matvec (8 variants), attention, RMSNorm, RoPE, SiLU, GeLU, embedding gather, KV cache ops, PLE, sampling |
| `decode_mega.metal` | Single-dispatch mega-kernel for entire per-token forward pass |
| `ggml_mul_mv_q4.metal` | llama.cpp Q4 matvec port |
| `ggml_flash_attn.metal` | llama.cpp flash attention port |

## Data Flow (Single Token Decode)

```
Token ID ──► CPU Embedding Lookup ──► GPU hidden_buf
                    │
                    ▼
        ┌───────────────────────────────────┐
        │   SINGLE METAL COMMAND BUFFER      │
        │                                     │
        │  ┌─── For each of 42 layers ────┐  │
        │  │                               │  │
        │  │  RMSNorm → QKV projections    │  │
        │  │  RoPE → KV cache append       │  │
        │  │  Attention (sliding or full)  │  │
        │  │  O projection → residual add  │  │
        │  │  RMSNorm → Gate/Up projection │  │
        │  │  SiLU → Down → residual add   │  │
        │  │  PLE embedding add            │  │
        │  └───────────────────────────────┘  │
        │                                     │
        │  Final RMSNorm → LM head → logits   │
        └───────────────────────────────────┘
                    │
                    ▼
        CPU Min-p Sampling ──► Next Token ID
```

## Gemma4 E4B Specifics

- **42 transformer layers**: alternating sliding-attention (local) and full-attention (global) every 4 layers
- **Shared KV**: layers 24-41 share KV caches with earlier layers of same type
- **PLE** (Per-Layer Embedding): token identity embedding injected each layer
- **QK-norm**: RMSNorm applied to Q and K before attention
- **Logit softcapping**: tanh-based logit capping in attention output and final head
- **Configurable KV cache**: f16, Q8_0, or Q4_0 quantized KV cache

## Performance

| Metric | Value |
|--------|-------|
| Decode throughput (M1 Pro, Gemma4 Q4_0) | ~29-33 tok/s |
| Decode throughput (M1 Pro, Llama 3.2 Q4_0) | ~73 tok/s |
| Weight format | Q4_0 (0.56 bytes/weight) |
| Total GPU memory (Gemma4) | ~3 GB |
| Total GPU memory (Llama 3.2) | ~1.7 GB |
| Prefill | Chunked, configurable tokens/tick |

## Running

```bash
# Server mode with Gemma4
cargo run --release -- serve --model /path/to/gemma4-e4b

# Interactive CLI mode
cargo run --release -- --gpu

# With custom kernel selection
MATVEC_KERNEL=r8 FLASH_ATTN=1 cargo run --release -- serve
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_KV_POOL_SLOTS` | 8 | Max concurrent requests |
| `LLAMA_KV_CACHE_TYPE` | f16 | KV cache format (f16/Q8_0/Q4_0) |
| `LLAMA_PREFILL_TOKENS_PER_TICK` | 512 | Chunked prefill token budget |
| `LLAMA_MAX_PREFILL_SEQ` | 2048 | Max prefill context length |
| `MATVEC_KERNEL` | r2 | Q4 matvec variant (r1/r2/fast/r4/r8/splitk/lc/ggml) |
| `FLASH_ATTN` | 1 | Use flash attention |
| `MEGA_KERNEL` | 0 | Use decode mega-kernel |
| `METAL_N_CB` | 3 | Parallel command buffers for CPU-GPU overlap |
| `PROFILE_PHASES` | 0 | Log per-phase timing |
