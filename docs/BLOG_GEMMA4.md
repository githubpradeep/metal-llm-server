# Building a Production LLM Inference Engine from Scratch in Rust + Metal

**How I built a full Gemma4 E4B inference server — from raw transformer math to Metal GPU kernels to an OpenAI-compatible API — running at 14 tok/s on a MacBook Pro.**

---

## Table of Contents

1. [Why Build This](#why-build-this)
2. [What This Engine Actually Does](#what-this-engine-actually-does)
3. [The Gemma4 Architecture: What Makes It Different](#the-gemma4-architecture)
4. [Part 1: Loading Model Weights](#part-1-loading-model-weights)
5. [Part 2: The Forward Pass — Token by Token](#part-2-the-forward-pass)
6. [Part 3: Metal GPU Kernels](#part-3-metal-gpu-kernels)
7. [Part 4: Quantization (Making It Fit)](#part-4-quantization)
8. [Part 5: The KV Cache](#part-5-the-kv-cache)
9. [Part 6: Sampling — Turning Logits into Text](#part-6-sampling)
10. [Part 7: The Server (OpenAI API)](#part-7-the-server)
11. [Part 8: Benchmarking and Evaluation](#part-8-benchmarking)
12. [Performance Breakdown](#performance-breakdown)
13. [What I Learned](#what-i-learned)
14. [Running It Yourself](#running-it-yourself)

---

## Why Build This

Every tutorial about LLMs starts with `from transformers import AutoModelForCausalLM` and ends with `model.generate()`. You get text out, but you've learned nothing about what happened between those two lines.

I wanted to understand every byte. What happens when you type a prompt and the model responds? Where does the time go? Why is it slow? What does the GPU actually *do*?

So I built the entire inference stack from scratch:
- No PyTorch, no ONNX, no llama.cpp
- Raw Metal compute shaders for Apple Silicon
- Custom Q4 quantization
- OpenAI-compatible API server
- Runs a 4B parameter model on a laptop at interactive speed

This post walks through every layer of that stack, from math to metal.

---

## What This Engine Actually Does

The engine takes Google's Gemma4 E4B model (a 4.5 billion parameter dense language model optimized for edge/on-device deployment) and runs it on a MacBook Pro M1 with:

- **~14 tokens/second** decode throughput
- **~3 GB** total memory for model weights (Q4 quantized + f16 sensitive layers + PLE tables in bf16)
- **~88 MB** for the KV cache (1024 context)
- **OpenAI-compatible API** — drop-in replacement for any OpenAI client
- **Streaming SSE** responses, just like ChatGPT

Here's what using it looks like:

```bash
# Start the server
cargo run --release -- --gpu --serve ~/models/gemma-4-e4b-it

# Talk to it (any OpenAI client works)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Write a haiku about Rust"}]}'
```

And you get back a streaming response, token by token, from a model running entirely on your local GPU.

---

## The Gemma4 Architecture

Before we look at code, let's understand what Gemma4 E4B actually is. It's a **dense transformer** — not a Mixture of Experts (the larger 26B A4B variant is MoE, but E4B is fully dense). "E4B" stands for "Edge 4 Billion" — designed specifically for on-device deployment.

Despite being dense, it has several architectural innovations that make it punch above its weight class.

### Model Specifications

| Property | Value |
|----------|-------|
| Architecture | Dense transformer (not MoE) |
| Total parameters | ~4.5B (effective), ~8B including PLE tables |
| Hidden dimension | 2560 |
| Layers | 42 |
| Attention heads | 20 |
| KV heads | 4 (GQA with 5 groups) |
| KV-shared layers | Layers 24–41 share KV with earlier layers |
| Intermediate (MLP) size | 10240 |
| Vocabulary | 262,144 tokens |
| Max context | 131,072 tokens (architecturally) |
| Head dimensions | 128 (sliding) / 512 (global) |
| PLE dimension | 256 per layer |

### What Makes Gemma4 Special

**1. Mixed attention: Sliding window + Full attention**

Not all layers are equal. Gemma4 alternates between two types:
- **Sliding window layers** (majority): Attend only to the last 512 tokens. Head dim = 128. Cheap and fast.
- **Full attention layers** (every ~6th layer): Attend to the entire context. Head dim = 512. Expensive but gives long-range awareness.

This hybrid approach gives you long-context understanding without the quadratic cost of full attention on every layer.

**2. KV Sharing**

Layers 24–41 (roughly the second half) don't compute their own keys and values. Instead, they reuse the KV cache from an earlier layer. This halves the KV compute for those layers while maintaining quality — the model learns which layers can share representations.

```rust
pub has_kv: bool,           // false for shared KV layers (layers 24-41)
pub kv_source_layer: usize, // which layer's KV cache to use
```

**3. Per-Layer Embeddings (PLE)**

This is the most unusual part. In standard transformers, each layer receives only the output of the previous layer. In Gemma4, each layer *also* receives a per-layer input computed from:
- The raw token embedding (looked up in a separate per-layer embedding table)
- A learned projection of the main hidden state

These are combined and gated before being added to the residual stream. Think of it as giving each layer a direct "shortcut" to the original token meaning, not just the increasingly transformed hidden state. This is what makes the PLE embedding tables ~3.5 GB — they store a unique 256-dim vector per token per layer.

**4. QK-Norm**

Before computing attention scores, both queries and keys are RMS-normalized per-head. This stabilizes training at scale and eliminates the need for the traditional `1/sqrt(d)` scaling — the model uses `attention_scale = 1.0`.

**5. Post-norm architecture**

Gemma4 applies RMSNorm *after* the attention/MLP output, not before the residual connection. Combined with a learnable `layer_scalar` per layer that dampens signal growth across the 42 layers.

**6. Logit softcapping**

The final logits are passed through `cap * tanh(logits / cap)` (cap=30) to prevent extreme values from dominating sampling.

---

## Part 1: Loading Model Weights

The model comes as SafeTensors files — a simple binary format that stores named tensors with their shapes and data types.

### The Challenge

Gemma4 E4B has ~4.5 billion dense parameters (plus ~3.5 GB of PLE embedding tables). In bf16 (the native training format), the core weights are ~9 GB. On a 16 GB MacBook, we can't fit that alongside the OS, the KV cache, and the app itself.

Solution: **quantize to Q4_0 on load** — 4 bits per weight, reducing ~9 GB to ~2.5 GB for the main weights.

### The Loading Pipeline

```rust
pub fn new(model_dir: &str) -> Self {
    // 1. Read config.json → understand architecture
    let config: Gemma4Config = serde_json::from_str(&config_str);
    
    // 2. For each layer's weight tensor:
    //    - Read bf16/f16 from SafeTensors
    //    - Convert to f32 (CPU)
    //    - Quantize to Q4_0 (CPU)
    //    - Upload to GPU buffer (Metal)
    
    // 3. Some weights stay in f16 (quality-sensitive):
    //    - Embedding tables (token lookups are exact)
    //    - O-projection (attention output, amplifies errors)
    //    - Some early/late layers
}
```

### Weight Caching

Converting and quantizing 42 layers × 7 weight matrices takes about 45 seconds. So on the first load, we serialize the GPU-ready buffers to a binary cache file. Second load: 3 seconds.

```rust
// First load: safetensors → quantize → save cache (~45s)
fn load_from_safetensors(model_dir: &str) -> Self { ... }

// Subsequent loads: read cache directly into GPU buffers (~3s)
fn load_from_cache(model_dir: &str, cache_path: &Path) -> Self { ... }
```

---

## Part 2: The Forward Pass

This is the core of the engine. Given a token ID, produce logits (a probability distribution over the next token).

### High-Level Flow

```
Token ID (e.g., 42)
    │
    ▼
┌─────────────────────────────┐
│  Embedding Lookup            │  embed_tokens[42] * sqrt(hidden_size)
│  → [2560] vector             │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  PLE Pre-computation         │  Per-layer context from embedding
│  → [42 × 256] vectors       │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐  ×42 layers
│  Transformer Layer           │
│  ┌───────────────────┐      │
│  │ Attention Block    │      │
│  │ Q projection       │      │  (always computed)
│  │ K/V projection     │      │  (only layers 0-23; layers 24-41 reuse earlier KV)
│  │ QK-Norm + RoPE     │      │
│  │ KV Cache Append    │      │
│  │ Attention Score    │      │
│  │ Output Projection  │      │
│  └───────────────────┘      │
│  ┌───────────────────┐      │
│  │ MLP Block          │      │
│  │ Gate + Up proj     │      │
│  │ GeLU activation    │      │
│  │ Down projection    │      │
│  └───────────────────┘      │
│  ┌───────────────────┐      │
│  │ PLE Integration    │      │
│  │ Gate → GeLU → Proj │      │
│  └───────────────────┘      │
│  × layer_scalar             │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Final RMSNorm               │
│  LM Head (vocab projection)  │
│  Logit Softcapping           │
│  → [262144] logits           │
└─────────────────────────────┘
```

### Operations Per Token

Let's count the work for a single token:

| Operation | Per Layer | Total (42 layers) |
|-----------|-----------|-------------------|
| RMSNorm | 5 | 210 |
| Matrix-vector multiply (Q4) | 4-5 | ~150 (fewer for KV-shared layers) |
| Matrix-vector multiply (f16) | 2-3 | ~100 |
| Rotary embedding | 1-2 | ~66 (Q always, K only for layers with own KV) |
| Attention (dot product over KV) | 1 | 42 |
| GeLU activation | 2 | 84 |
| Vector add (residual) | 3 | 126 |
| Vector scale | 1 | 42 |
| Vector copy | 3 | 126 |

Note: Layers 24–41 skip K/V projection and KV cache append (they reuse earlier layers' cache), reducing total compute by ~15%.

**~800–1000 GPU kernel dispatches per token**, all encoded into a single Metal command buffer.

### The Single Command Buffer Pattern

This is the most important optimization. Instead of submitting each operation to the GPU separately (which would add ~5-10μs overhead per dispatch × 1000 = 5-10ms wasted), we encode the *entire forward pass* into one command buffer:

```rust
let cmd = self.ctx.queue.new_command_buffer();

// PLE pre-pass
let ple_enc = cmd.new_compute_command_encoder();
// ... encode 5-6 PLE operations ...
ple_enc.end_encoding();

// 42 transformer layers
for layer_idx in 0..42 {
    let encoder = cmd.new_compute_command_encoder();
    // ... encode ~25 operations per layer ...
    encoder.end_encoding();
}

// Single submit, single wait
cmd.commit();
cmd.wait_until_completed();  // GPU processes everything in one shot

// Final norm + LM head (separate small cmd buffer)
// ... produce logits ...
```

The GPU hardware sees the full dependency graph and can pipeline operations internally — starting the next operation as soon as its inputs are ready, without waiting for a CPU round-trip.

---

## Part 3: Metal GPU Kernels

Every operation (matmul, norm, attention, etc.) is a Metal compute shader. Let's look at the key ones.

### Matrix-Vector Multiply (Q4 Quantized)

This is where 70%+ of the time goes. Each layer has weight matrices that we multiply by the activation vector.

The key insight: **inference is memory-bandwidth-bound, not compute-bound.** The GPU can multiply numbers faster than it can read them from memory. So the optimization target is: *read fewer bytes*.

Q4_0 format: every 32 weights are stored as 1 f16 scale + 16 packed bytes (2 weights per byte) = 18 bytes per 32 weights.

```metal
kernel void matvec_q4(
    device const uchar* W [[buffer(0)]],    // Quantized weights
    device const float* x [[buffer(1)]],    // Input activation vector
    device float* y [[buffer(2)]],          // Output vector
    constant uint& M [[buffer(3)]],         // Output dimension
    constant uint& K [[buffer(4)]],         // Input dimension
    uint tgid [[threadgroup_position_in_grid]],   // Which output row
    uint tid [[thread_index_in_threadgroup]]       // Lane within SIMD group
) {
    uint row = tgid;
    uint num_groups = K / 32;  // Number of Q4 groups per row
    uint bytes_per_row = num_groups * 18;  // 18 bytes per group
    
    float acc = 0.0;
    
    // Each thread handles a subset of groups (cooperative parallelism)
    for (uint g = tid; g < num_groups; g += 32) {
        uint block_offset = row * bytes_per_row + g * 18;
        
        // Read scale (f16 → f32)
        half scale_h = *reinterpret_cast<device const half*>(&W[block_offset]);
        float scale = float(scale_h);
        
        // Read 16 packed bytes = 32 quantized weights
        device const uchar* quants = &W[block_offset + 2];
        uint base = g * 32;
        
        float local_acc = 0.0;
        for (uint i = 0; i < 16; i++) {
            uchar packed = quants[i];
            // Low nibble (subtract 8 to center around 0)
            local_acc += float(int(packed & 0x0F) - 8) * x[base + i*2];
            // High nibble
            local_acc += float(int(packed >> 4) - 8) * x[base + i*2 + 1];
        }
        acc += local_acc * scale;
    }
    
    // SIMD reduction: sum across all 32 threads in the group
    acc = simd_sum(acc);
    
    if (tid == 0) {
        y[row] = acc;
    }
}
```

**Why this is fast:**
- Each thread reads Q4 data (0.5625 bytes/weight) instead of f32 (4 bytes/weight) = 7x less memory traffic
- 32 threads cooperate on one output row via hardware SIMD reduction
- `simd_sum` is a single-cycle cross-lane operation (no shared memory needed)

### RMSNorm

Root Mean Square Layer Normalization — simpler than LayerNorm (no mean subtraction):

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum x_i^2 + \epsilon}} \cdot \gamma$$

```metal
kernel void rmsnorm(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    // Step 1: Compute sum of squares (cooperative across threads)
    float sum_sq = 0.0;
    for (uint i = tid; i < dim; i += 32) {
        sum_sq += x[i] * x[i];
    }
    sum_sq = simd_sum(sum_sq);  // Reduce across SIMD group
    
    // Step 2: Compute scale = 1/sqrt(mean_sq + eps)
    float scale = rsqrt(sum_sq / float(dim) + eps);
    
    // Step 3: Apply scale and learned weight
    for (uint i = tid; i < dim; i += 32) {
        out[i] = x[i] * scale * weight[i];
    }
}
```

### Rotary Position Embeddings (RoPE)

RoPE encodes position by rotating pairs of dimensions:

```metal
kernel void rotary(
    device float* q [[buffer(0)]],    // Query/Key vector
    device const float* cos [[buffer(1)]],  // cos(pos * freq)
    device const float* sin [[buffer(2)]],  // sin(pos * freq)
    constant uint& head_dim [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint half_dim = head_dim / 2;
    
    for (uint i = tid; i < half_dim; i += 32) {
        float x0 = q[i];
        float x1 = q[i + half_dim];
        
        // Rotation: [x0, x1] → [x0*cos - x1*sin, x0*sin + x1*cos]
        q[i]            = x0 * cos[i] - x1 * sin[i];
        q[i + half_dim] = x0 * sin[i] + x1 * cos[i];
    }
}
```

Gemma4 adds complexity: **partial rotary** for full-attention layers (only 25% of dimensions get rotated) and different `theta` values for sliding vs full attention.

### Attention

For single-token decode, attention computes how much the current query "looks at" each cached key:

```metal
kernel void attention_f16(
    device const float* q,           // Current query [head_dim]
    device const half* k_cache,      // All past keys [seq_len, head_dim]
    device const half* v_cache,      // All past values [seq_len, head_dim]
    device float* output,            // Attention output [head_dim]
    constant uint& seq_len,
    constant uint& head_dim,
    ...
) {
    // 1. Compute attention scores: score[t] = dot(q, k_cache[t])
    // 2. Softmax over all positions
    // 3. Weighted sum of values: out = sum(score[t] * v_cache[t])
}
```

For GQA (Grouped Query Attention), multiple query heads share the same KV head. With 20 query heads and 4 KV heads, that's 5 queries per KV pair — a 5x saving in KV cache memory and KV compute.

---

## Part 4: Quantization

### Why Q4?

Memory bandwidth on M1 Pro: ~200 GB/s. Model weights for Gemma4 E4B (dense, ~4.5B params):

| Format | Size | Theoretical Max tok/s |
|--------|------|----------------------|
| f32 | ~18 GB | Doesn't fit |
| bf16 | ~9 GB | ~22 tok/s |
| f16 | ~9 GB | ~22 tok/s |
| **Q4_0** | **~2.5 GB** | **~80 tok/s** |

Q4 gives us the best throughput because we read the least data per token. The quality loss is minimal for a dense 4B model — and we compensate by keeping quality-sensitive layers in f16.

### Q4_0 Format

```
Group of 32 weights:
┌────────┬─────────────────────────────────────┐
│ scale  │ 16 packed bytes (2 weights per byte) │
│ (f16)  │ each weight: 4 bits, range [0, 15]  │
│ 2 bytes│ 16 bytes                             │
└────────┴─────────────────────────────────────┘
Total: 18 bytes per 32 weights = 4.5 bits/weight
```

### Quantization Code

```rust
fn quantize_q4_0(data: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    let group_size = 32;
    let groups_per_row = cols / group_size;
    let bytes_per_row = groups_per_row * 18;
    let mut output = vec![0u8; rows * bytes_per_row];
    
    for row in 0..rows {
        for g in 0..groups_per_row {
            let start = row * cols + g * group_size;
            let group = &data[start..start + group_size];
            
            // Find max absolute value → determines scale
            let max_abs = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = max_abs / 7.0;  // Map to [-7, 7] → stored as [1, 15]
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            
            // Write f16 scale
            let out_offset = row * bytes_per_row + g * 18;
            let scale_f16 = f32_to_f16(scale);
            output[out_offset] = (scale_f16 & 0xFF) as u8;
            output[out_offset + 1] = (scale_f16 >> 8) as u8;
            
            // Pack 32 weights into 16 bytes
            for i in 0..16 {
                let v0 = ((group[i*2] * inv_scale).round() as i32 + 8).clamp(0, 15) as u8;
                let v1 = ((group[i*2+1] * inv_scale).round() as i32 + 8).clamp(0, 15) as u8;
                output[out_offset + 2 + i] = v0 | (v1 << 4);
            }
        }
    }
    output
}
```

### Mixed Precision Strategy

Not all layers are equal. We keep certain weights in f16 for quality:
- **O-projection** (attention output): errors here propagate through all subsequent layers
- **Embedding tables**: lookup must be exact, no approximation
- **First/last few layers**: most sensitive to quantization noise

```rust
// Decision per layer during loading:
let use_f16 = layer_idx < 2 || layer_idx >= num_layers - 2;  // First/last 2 layers

if use_f16 {
    layer.q_proj = ctx.buffer_from_f32_as_f16(&weight_data);   // 2 bytes/weight
} else {
    layer.q_proj = ctx.buffer_from_f32_as_q4(&weight_data, out_dim, in_dim);  // 0.56 bytes/weight
}
```

---

## Part 5: The KV Cache

The KV cache is what makes autoregressive generation work. Without it, generating 100 tokens would require re-processing all previous tokens at each step — quadratic cost.

### How It Works

During the forward pass, each layer produces a key vector and a value vector for the current token. These are appended to a growing cache:

```
After generating 5 tokens, layer 0's KV cache looks like:

K cache: [k₀, k₁, k₂, k₃, k₄]   ← one key per past token
V cache: [v₀, v₁, v₂, v₃, v₄]   ← one value per past token

Attention for token 5:
  score[t] = dot(q₅, kₜ) for t = 0..4
  output = softmax(scores) @ V_cache
```

### Implementation

```rust
// Pre-allocate at startup (fixed capacity)
let kv_capacity = 1024;  // Max context length

// Per layer: K and V buffers sized for max context
// Shape: [num_kv_heads × kv_capacity × head_dim] in f16
let k_cache: Buffer = ctx.buffer_empty(num_kv_heads * kv_capacity * head_dim);
let v_cache: Buffer = ctx.buffer_empty(num_kv_heads * kv_capacity * head_dim);
```

The append operation writes the new K/V at position `seq_len` and increments the counter:

```metal
kernel void kv_append_f16(
    device const float* new_kv,     // New key or value [num_heads × head_dim]
    device half* cache,             // Existing cache [num_heads × capacity × head_dim]
    constant uint& num_heads,
    constant uint& head_dim,
    constant uint& capacity,
    constant uint& position,        // Where to write (current seq_len)
    uint tid [[thread_position_in_grid]]
) {
    uint head = tid / head_dim;
    uint dim = tid % head_dim;
    uint cache_idx = head * capacity * head_dim + position * head_dim + dim;
    cache[cache_idx] = half(new_kv[tid]);
}
```

### KV Cache Memory Math

For Gemma4 E4B with f16 KV cache:

```
Per token, per layer:
  K: num_kv_heads × head_dim × 2 bytes
  V: num_kv_heads × head_dim × 2 bytes

Sliding layers (head_dim=128): 4 × 128 × 2 × 2 = 2,048 bytes/token/layer
Full attention layers (head_dim=512): 4 × 512 × 2 × 2 = 8,192 bytes/token/layer

For 1024 context:
  ~34 sliding layers × 1024 × 2048 = 71 MB
  ~8 full layers × 1024 × 8192 = 67 MB
  Total: ~138 MB (but sliding layers only keep 512 tokens → actual ~88 MB)
```

### Sliding Window Optimization

Sliding window layers only attend to the last 512 tokens. This means:
- We still allocate the full capacity for simplicity
- But attention only reads the relevant window
- Long contexts don't slow down sliding layers

```rust
// For sliding window layers:
let effective_kv_seq = if !is_full {
    attn_kv_seq.min(self.config.sliding_window as u32)  // Cap at 512
} else {
    attn_kv_seq  // Full attention sees everything
};
```

---

## Part 6: Sampling

Once we have logits (a score for each of the 262,144 possible next tokens), we need to pick one. This is where "creativity" vs "determinism" lives.

### Temperature

Temperature scales the logits before softmax. Higher temperature → more uniform distribution → more random choices:

```rust
// Temperature 0 = always pick highest logit (greedy)
if temperature < 1e-6 {
    return argmax(logits);
}

// Apply temperature: divide logits by temperature before softmax
let scaled = (logits[i] - max_logit) / temperature;
probs[i] = scaled.exp();
```

### Min-P Sampling

Min-P is an alternative to Top-K and Top-P that adapts to the distribution shape:

```rust
// Min-P: keep tokens with probability > min_p × max_probability
let p_max = probs.iter().max();
let threshold = p_max * min_p;  // e.g., min_p = 0.05

for p in probs.iter_mut() {
    if *p < threshold {
        *p = 0.0;  // Filter out unlikely tokens
    }
}
// Renormalize and sample from remaining
```

Why Min-P beats Top-K:
- Top-K(50) always keeps 50 tokens regardless of confidence
- Min-P(0.05) keeps 3 tokens when the model is confident, 200 tokens when it's uncertain
- It adapts to the model's certainty automatically

### Logit Softcapping

Gemma4 applies `cap × tanh(logits / cap)` before sampling. This squashes extreme logits:

```rust
let cap = 30.0;  // From config
for l in logits.iter_mut() {
    let x = (*l / cap).clamp(-10.0, 10.0);  // Prevent overflow
    *l = cap * x.tanh();
}
```

This prevents any single token from having a probability arbitrarily close to 1.0, maintaining diversity in generation.

---

## Part 7: The Server

The server wraps the inference engine in an OpenAI-compatible HTTP API using `axum` (a Rust async web framework).

### Architecture

```
Client (curl, Python, any OpenAI SDK)
    │
    ▼  HTTP POST /v1/chat/completions
┌───────────────────────────────┐
│  Axum HTTP Server (tokio)      │
│  - Parse request JSON          │
│  - Apply chat template         │
│  - Tokenize prompt             │
└───────────────┬───────────────┘
                │ Mutex<Model>
                ▼
┌───────────────────────────────┐
│  GPU Inference Engine          │
│  - Prefill (all prompt tokens) │
│  - Decode loop (one at a time) │
│  - Stream tokens back via SSE  │
└───────────────────────────────┘
```

### Chat Template

Gemma4 uses a specific format for multi-turn conversations:

```rust
fn apply_chat_template(messages: &[Message]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str("<start_of_turn>user\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<end_of_turn>\n");
            }
            "user" => {
                prompt.push_str("<start_of_turn>user\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<end_of_turn>\n");
            }
            "assistant" => {
                prompt.push_str("<start_of_turn>model\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<end_of_turn>\n");
            }
            _ => {}
        }
    }
    prompt.push_str("<start_of_turn>model\n");
    prompt
}
```

### Streaming SSE

For streaming responses, we use Server-Sent Events — the same protocol OpenAI uses:

```rust
async fn chat_completions_stream(state: Arc<AppState>, request: ChatCompletionRequest)
    -> Sse<impl Stream<Item = Result<Event, Infallible>>>
{
    let stream = async_stream::stream! {
        // Prefill
        let mut logits = model.forward_prefill(&token_ids);
        
        // Decode loop
        for _ in 0..max_tokens {
            let next_token = sample(&logits, temperature, min_p);
            if is_eos(next_token) { break; }
            
            let text = tokenizer.decode(&[next_token as u32]);
            
            // Send SSE chunk
            yield Ok(Event::default().data(json!({
                "choices": [{"delta": {"content": text}}]
            }).to_string()));
            
            logits = model.forward_single_token(next_token);
        }
        
        // Final chunk with finish_reason
        yield Ok(Event::default().data("[DONE]"));
    };
    
    Sse::new(stream)
}
```

---

## Part 8: Benchmarking

### Performance Metrics

From our evaluation run (27 diverse prompts across 10 categories):

| Metric | Value |
|--------|-------|
| Decode throughput | ~11 tok/s (with overhead) |
| Peak decode | ~14 tok/s (pure generation) |
| Prefill throughput | ~14 tok/s (sequential, not yet parallel) |
| Total tokens generated | 7,929 |
| Total time | 732 seconds |
| Errors | 0 |

### Quality Evaluation (LLM-as-a-Judge)

We ran the model through a comprehensive eval covering writing, reasoning, math, coding, extraction, STEM, humanities, instruction following, safety, and multi-turn conversation.

**Overall score: 8.0/10** (judged by Claude Opus 4.6)

| Category | Score | Notes |
|----------|-------|-------|
| Writing | 9.3 | Excellent — best category |
| STEM Knowledge | 8.7 | Accurate, good pedagogy |
| Humanities | 8.6 | Correct history/philosophy |
| Extraction | 8.5 | Clean structured outputs |
| Math | 8.3 | Correct methods, sometimes truncated |
| Safety | 8.1 | Appropriate refusals |
| Conversation | 8.1 | Good multi-turn context |
| Coding | 7.1 | Solid Python, weaker Rust |
| Reasoning | 6.8 | Struggles with multi-step logic |
| Instruction Following | 6.8 | Usually great, occasionally total failure |

### Comparison: Us vs llama.cpp

| | This Engine | llama.cpp (Metal) |
|---|---|---|
| Decode speed | ~14 tok/s | ~18 tok/s |
| Code size | ~3K lines Rust | ~150K lines C/C++ |
| Models supported | Gemma4 only | 50+ architectures |
| Dependencies | 0 (just Metal) | 0 (just Metal) |
| Time to build | 2 weeks | 2+ years |

We're ~78% of llama.cpp's speed with 2% of the code. The remaining gap is largely kernel micro-optimization (tiled matmul, better memory access patterns) that takes months to tune.

---

## Performance Breakdown

### Where Does the Time Go?

For a single token generation at ~14 tok/s (71ms per token):

```
Matrix-vector multiplies (Q4):    ~50ms (70%)
Matrix-vector multiplies (f16):   ~12ms (17%)
Attention (over KV cache):         ~4ms (6%)
RMSNorm + activations:             ~3ms (4%)
CPU overhead (embed, rotary calc): ~2ms (3%)
```

The overwhelming bottleneck is weight reads. 42 layers × 7 weight matrices × variable sizes = reading ~2.5 GB of Q4 data from memory per token.

### The Roofline

```
M1 Pro specs:
  Memory bandwidth: ~200 GB/s
  GPU compute (FP32): 4.1 TFLOPS

Model weight reads per token: ~2.5 GB (Q4) + ~0.5 GB (f16 layers)
Theoretical max throughput: 200 / 3.0 ≈ 67 tok/s

Actual: ~14 tok/s = 21% of theoretical peak
```

Why only 20%? Several reasons:
1. **Memory access pattern isn't perfectly sequential** — KV cache reads interleave with weight reads
2. **Kernel launch gaps** — even with single command buffer, encoder boundaries add small gaps
3. **SIMD divergence** — Q4 decode has branches that reduce occupancy
4. **CPU→GPU sync** — embedding lookup + rotary computation on CPU before GPU starts

The path to 40+ tok/s: parallel prefill (matmul for prompt), better kernel fusion, potentially int8 KV cache.

---

## What I Learned

### 1. The transformer is simpler than you think

Strip away the framework abstractions and a transformer is just:
- Matrix multiplies (linear projections)
- Element-wise operations (norms, activations, residual adds)
- One attention mechanism (dot product + softmax + weighted sum)

That's it. Everything else is bookkeeping.

### 2. Memory bandwidth is the only constraint that matters for inference

Compute is cheap. Reading weights from memory is expensive. Every optimization in this project reduces bytes read:
- Q4 quantization: 7x reduction
- f16 KV cache: 2x vs f32
- Sliding window: attend to fewer cached positions

### 3. Apple Silicon's unified memory is perfect for local inference

No PCIe bottleneck, no GPU memory shortage (shared pool), no data copies. The hardware is designed for exactly this workload.

### 4. The hard part isn't the math — it's the shapes

Getting every buffer size, offset, stride, and dimension right across 42 layers with mixed head dimensions, shared KV layers, partial rotary, and per-layer embeddings... that's where 80% of debugging time went.

### 5. Build from the model card, not the code

Reading the Gemma4 paper and model card was more useful than reading the HuggingFace implementation. The paper tells you *what* to compute. The code tells you one specific *how* (tangled with framework abstractions).

### 6. You can ship a useful product with surprisingly little code

The entire engine is ~3000 lines of Rust + ~500 lines of Metal shaders. That's it. No build system complexity, no Python dependency hell, no CUDA driver issues. `cargo build --release` and you have a binary.

---

## Running It Yourself

### Prerequisites

- macOS 15+ with Apple Silicon (M1/M2/M3/M4)
- Rust toolchain: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- Gemma4 E4B model (from HuggingFace, ~10 GB download)

### Download the Model

```bash
# Using huggingface-cli
pip install huggingface-hub
huggingface-cli download google/gemma-4-e4b-it --local-dir ~/models/gemma-4-e4b-it
```

### Build and Run

```bash
cd llama_sinks_rust
export MACOSX_DEPLOYMENT_TARGET=15.0

# Build
cargo build --release

# Run interactively
cargo run --release -- --gpu ~/models/gemma-4-e4b-it

# Run as API server
cargo run --release -- --gpu --serve ~/models/gemma-4-e4b-it

# Custom port
cargo run --release -- --gpu --serve --port 9090 ~/models/gemma-4-e4b-it
```

### Use the API

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

response = client.chat.completions.create(
    model="gemma-4-e4b",
    messages=[{"role": "user", "content": "Explain recursion in one sentence."}],
    temperature=0.7,
    max_tokens=200,
)
print(response.choices[0].message.content)
```

### Run the Evaluation

```bash
# Start server in one terminal
cargo run --release -- --gpu --serve ~/models/gemma-4-e4b-it

# Run eval in another terminal
python3 benchmarks/eval_run.py
# → outputs benchmarks/eval_outputs.json
```

### Project Structure

```
llama_sinks_rust/
├── src/
│   ├── main.rs              CLI entry point, model detection, generation loop
│   ├── gemma4_config.rs     Config structs (architecture hyperparameters)
│   ├── gemma4_gpu_model.rs  The core: model loading, forward pass, KV cache
│   ├── gpu.rs               Metal context: buffer management, kernel encoding
│   ├── server.rs            Axum HTTP server (OpenAI API, SSE streaming)
│   ├── sampling.rs          Temperature, Min-P, multinomial sampling
│   ├── shaders/
│   │   └── llama.metal      All GPU compute kernels (matvec, attention, etc.)
│   └── ...                  (Llama 3.2 code, CPU fallback, etc.)
├── benchmarks/
│   ├── eval_prompts.json    27 evaluation prompts across 10 categories
│   ├── eval_run.py          Evaluation harness (hits server, collects outputs)
│   ├── eval_outputs.json    Results from latest eval run
│   └── EVAL_JUDGE_PROMPT.md Instructions for LLM-as-a-Judge scoring
└── docs/
    ├── BLOG.md              Original Llama 3.2 blog post (73 tok/s journey)
    └── BLOG_GEMMA4.md       This post
```

---

## What's Next

The engine currently handles single-user inference. The [production roadmap](../.kiro/specs/production-inference-server/design.md) adds:

1. **Parallel prefill** — process all prompt tokens in one matmul (10x prefill speedup)
2. **Request queue + KV cache pool** — multiple concurrent users
3. **Continuous batching** — batch decode tokens from different users into one GPU pass
4. **FlashAttention** — tiled attention for 4K+ context without quadratic memory
5. **Speculative decoding** — 2-3x decode speedup using a draft model

But even today, it's a complete, working inference engine that you can use as:
- A learning tool (understand every byte of LLM inference)
- A local chat assistant (no internet, no API keys, full privacy)
- A foundation for custom inference optimizations

The full source is in this repository. Build it, break it, learn from it.
