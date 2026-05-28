# Building a 73 tok/s LLM Inference Engine in Rust + Metal: From Scratch to GPU

**A step-by-step guide to implementing Llama 3.2 1B inference in Rust, optimizing from 5.5 tok/s (Python) to 73+ tok/s (Metal GPU with int4 quantization).**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Phase 1: Pure Rust with BLAS (5.5 tok/s)](#phase-1-pure-rust-with-blas)
4. [Phase 2: Metal GPU Compute (9.7 → 19.2 tok/s)](#phase-2-metal-gpu-compute)
5. [Phase 3: SIMD-Group Optimization (26.9 tok/s)](#phase-3-simd-group-optimization)
6. [Phase 4: Half-Precision Weights (51.6 tok/s)](#phase-4-half-precision-weights)
7. [Phase 5: Int4 Quantization (73.5 tok/s)](#phase-5-int4-quantization)
8. [KV Cache: Sinks vs Full Context](#kv-cache-sinks-vs-full-context)
9. [Lessons Learned](#lessons-learned)
10. [Running the Code](#running-the-code)

---

## Introduction

This project implements Llama 3.2 1B inference entirely from scratch in Rust — no PyTorch, no ONNX runtime, no llama.cpp. Starting from a Python+NumPy reference implementation running at 5.5 tokens/second, we progressively optimized to 73+ tokens/second on Apple Silicon using Metal compute shaders.

The goal was to understand every layer of the inference stack: from transformer math to GPU kernel optimization.

**Final performance:**
- 73.5 tok/s decode throughput
- 200 tokens generated in 2.7 seconds
- 13.4x faster than the Python baseline
- Comparable to llama.cpp Metal performance

---

## Architecture Overview

Llama 3.2 1B has:
- 16 transformer decoder layers
- 2048 hidden dimension
- 32 attention heads (64-dim each)
- 8 KV heads (grouped-query attention, 4 groups)
- 8192 intermediate MLP dimension
- 128,256 vocabulary size
- RoPE positional embeddings

Each token generation requires:
1. **Embedding lookup** — index into (128256, 2048) table
2. **16 decoder layers**, each with:
   - RMSNorm → Q/K/V projection → Rotary → KV cache → Attention → O projection → Residual
   - RMSNorm → Gate+Up projection → SiLU activation → Down projection → Residual
3. **Final RMSNorm** → LM head projection → Sampling

That's **7 matrix-vector multiplications per layer × 16 layers + 1 LM head = 113 matmuls per token**.

---

## Phase 1: Pure Rust with BLAS

### The Bottleneck: Memory Bandwidth

For autoregressive decode (generating one token at a time), each matmul is a **matrix-vector product**: a (M, K) weight matrix times a (K,) activation vector. The weight matrix is read entirely from memory for every single token.

For Llama 3.2 1B in f32:
- Total weight parameters: ~1.24 billion
- Memory per token: ~4.7 GB read from RAM
- Apple Silicon memory bandwidth: ~100 GB/s
- Theoretical max: ~100/4.7 ≈ 21 tok/s (f32)

### Implementation

```rust
// Linear layer using Apple Accelerate BLAS
pub fn forward_vec(&self, x: &[f32], y: &mut [f32]) {
    unsafe {
        cblas_sgemv(
            101, 111,  // RowMajor, NoTrans
            self.out_features as i32,
            self.in_features as i32,
            1.0, self.weights.as_ptr(), self.in_features as i32,
            x.as_ptr(), 1,
            0.0, y.as_mut_ptr(), 1,
        );
    }
}
```

### Result: 5.5 tok/s

Same as Python+NumPy because both use the same BLAS library (Accelerate) for the matmuls. The Rust overhead (allocation, indexing) is negligible compared to the BLAS compute time.

**Key insight:** When your code is BLAS-bound, the language doesn't matter. You need to change the compute paradigm.

---

## Phase 2: Metal GPU Compute

### Why GPU?

Apple Silicon has a unified memory architecture — CPU and GPU share the same physical RAM. This means:
- Zero-copy buffer sharing between CPU and GPU
- No PCIe transfer overhead
- GPU has higher memory bandwidth utilization due to massive parallelism

### The Naive Approach (9.7 tok/s)

First attempt: dispatch each operation as a separate Metal command buffer with `wait_until_completed()` after each one.

```rust
// Each matvec = new command buffer + wait
pub fn matvec(&self, w_buf: &Buffer, x_buf: &Buffer, y_buf: &Buffer, m: u32, k: u32) {
    let cmd = self.queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    // ... encode kernel ...
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();  // BLOCKING!
}
```

With ~192 operations per token, that's 192 GPU round-trips. Each round-trip has ~5-10μs of overhead = ~1-2ms wasted per token.

### Single Command Buffer (19.2 tok/s)

The fix: encode ALL operations for one token into a single command buffer, submit once, wait once.

```rust
let cmd = self.ctx.queue.new_command_buffer();
let encoder = cmd.new_compute_command_encoder();

for layer in &self.layers {
    self.ctx.encode_rmsnorm(encoder, ...);
    self.ctx.encode_matvec(encoder, ...);  // Q proj
    self.ctx.encode_matvec(encoder, ...);  // K proj
    self.ctx.encode_matvec(encoder, ...);  // V proj
    self.ctx.encode_rotary(encoder, ...);
    self.ctx.encode_kv_append(encoder, ...);
    self.ctx.encode_attention(encoder, ...);
    self.ctx.encode_matvec(encoder, ...);  // O proj
    self.ctx.encode_vec_add(encoder, ...);
    // ... MLP ...
}

encoder.end_encoding();
cmd.commit();
cmd.wait_until_completed();  // ONE wait for entire forward pass
```

**2x speedup** just from eliminating dispatch overhead.

---

## Phase 3: SIMD-Group Optimization

### The Problem

The naive matvec kernel: 1 thread per output row, sequentially iterating over K=2048 elements.

```metal
// SLOW: one thread does all the work for one row
kernel void matvec_naive(..., uint gid [[thread_position_in_grid]]) {
    float acc = 0.0;
    for (uint k = 0; k < K; k++) {
        acc += W[row * K + k] * x[k];
    }
    y[gid] = acc;
}
```

### The Fix: Cooperative SIMD Groups

Use 32 threads (one SIMD group) per row. Each thread handles K/32 elements, then hardware-level `simd_sum` reduces across all lanes in a single cycle.

```metal
kernel void matvec(...,
    uint tgid [[threadgroup_position_in_grid]],  // which row
    uint tid [[thread_index_in_threadgroup]]      // lane in SIMD group
) {
    float acc = 0.0;
    
    // Each thread handles every 32nd group of 4 elements
    uint k = tid * 4;
    uint stride = 32 * 4;  // 128 elements per full iteration
    
    for (; k + 3 < K; k += stride) {
        float4 w = *reinterpret_cast<device const float4*>(&W[row_offset + k]);
        float4 xv = *reinterpret_cast<device const float4*>(&x[k]);
        acc += dot(w, xv);  // Hardware float4 dot product
    }
    
    acc = simd_sum(acc);  // Hardware cross-lane reduction (1 cycle!)
    
    if (tid == 0) y[tgid] = acc;
}
```

**40% speedup** from better GPU utilization.

---

## Phase 4: Half-Precision Weights

### The Insight

Apple Silicon GPU reads f16 at the same bandwidth as f32 in terms of bytes/second. But f16 is half the bytes per element. So you get **2x the elements per second**.

### Implementation

Store weights as `half` (f16), keep activations as `float` (f32) for precision:

```metal
kernel void matvec_f16(
    device const half* W [[buffer(0)]],   // f16 weights (2 bytes each)
    device const float* x [[buffer(1)]],  // f32 activations
    device float* y [[buffer(2)]],        // f32 output
    ...
) {
    // Read half4 (8 bytes = 4 weights) instead of float4 (16 bytes)
    half4 w = *reinterpret_cast<device const half4*>(&W[row_offset + k]);
    float4 xv = *reinterpret_cast<device const float4*>(&x[k]);
    acc += dot(float4(w), xv);  // Convert to f32 for accumulation
}
```

On the Rust side, convert f32 weights to f16 during model loading:

```rust
pub fn buffer_from_f32_as_f16(&self, data: &[f32]) -> Buffer {
    let f16_data: Vec<u16> = data.iter().map(|&v| f32_to_f16(v)).collect();
    self.device.new_buffer_with_data(f16_data.as_ptr() as *const _, ...)
}
```

**92% speedup** — nearly doubled throughput by halving memory reads.

---

## Phase 5: Int4 Quantization

### Q4_0 Format

Each group of 32 weights is stored as:
- 1 × f16 scale (2 bytes)
- 16 × packed byte pairs (16 bytes) — each byte holds two 4-bit values

Total: 18 bytes per 32 weights = **0.5625 bytes per weight** (vs 2 bytes for f16, 4 bytes for f32).

### Quantization

```rust
fn quantize_q4_0(data: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    for each group of 32 values:
        scale = max_abs / 7.0
        for each value:
            quantized = round(value / scale) + 8  // maps to [0, 15]
            pack two 4-bit values into one byte
        store: [f16 scale][16 packed bytes]
}
```

### Dequantization in the GPU Kernel

```metal
kernel void matvec_q4(...) {
    for (uint g = tid; g < num_groups; g += SIMD_SIZE) {
        half scale_h = *reinterpret_cast<device const half*>(&W_q4[block_offset]);
        float scale = float(scale_h);
        
        device const uchar* quants = &W_q4[block_offset + 2];
        
        float local_acc = 0.0;
        for (uint i = 0; i < 16; i += 4) {
            uchar p0 = quants[i];
            // Low nibble = first weight, high nibble = second
            local_acc += float(int(p0 & 0x0F) - 8) * x[base];
            local_acc += float(int(p0 >> 4) - 8) * x[base + 1];
            // ... unrolled for 4 bytes (8 weights) per iteration
        }
        acc += local_acc * scale;  // Deferred scale multiplication
    }
}
```

**41% speedup over f16** — reading 4x less data from memory.

---

## KV Cache: Sinks vs Full Context

The KV cache stores past keys and values so attention can look back at previous tokens. Two strategies:

### Streaming Attention Sinks (Fixed Window)

Keep first N "sink" tokens + last M "window" tokens. Evict the middle.

```
[sink₁ sink₂ sink₃ sink₄ | recent₁ recent₂ ... recent₆₄]
                          ↑ evict everything between
```

- **Pro:** Constant memory, constant speed per token
- **Con:** Loses middle context, may degrade quality for long sequences
- **Use case:** Infinite-length generation, chatbots

### Full Context (Unbounded)

Keep all tokens. Attention cost grows linearly.

```
[token₁ token₂ token₃ ... token_N]  ← grows with each token
```

- **Pro:** Perfect context, best quality
- **Con:** Attention cost grows as O(N) per token, memory grows
- **Use case:** Short-to-medium generation (< 2048 tokens)

### Switching Between Them

The model is identical — only the cache policy changes:

```rust
// Sinks: capacity = 128, evict when full
let kv_capacity = 128;
if kv_seq > 68 { self.evict_kv_cache(4, 64); }

// Full context: capacity = 2048, never evict
let kv_capacity = config.max_position_embeddings;  // 2048
// No eviction needed
```

---

## Lessons Learned

### 1. Profile Before Optimizing

Each optimization targeted a specific, measured bottleneck:
- Dispatch overhead → single command buffer
- Low GPU utilization → SIMD groups
- Memory bandwidth → f16 → Q4

### 2. Apple Silicon's AMX Coprocessor is Fast

Our scalar int8 kernel on CPU was **slower** than Accelerate's f32 sgemv because Accelerate uses the AMX coprocessor (dedicated matrix multiply hardware). Don't try to beat hardware-accelerated BLAS with scalar code.

### 3. Memory Bandwidth is King for Inference

Autoregressive LLM inference is **memory-bound**, not compute-bound. Each token reads the entire model's weights from memory. The optimization path is: reduce bytes read per token.

| Format | Bytes/weight | Relative bandwidth |
|--------|-------------|-------------------|
| f32 | 4 | 1x |
| f16 | 2 | 2x |
| Q4_0 | 0.5625 | 7.1x |

### 4. Unified Memory is a Superpower

On Apple Silicon, CPU and GPU share memory. This means:
- Weight buffers created once, used by both CPU (for loading) and GPU (for compute)
- No PCIe copies, no staging buffers
- KV cache lives on GPU permanently

### 5. Single Command Buffer >> Many Small Ones

GPU dispatch has fixed overhead (~5-10μs per command buffer). With 192 operations per token, that's 1-2ms of pure overhead. Batching into one submission eliminated this entirely.

---

## Running the Code

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Rust toolchain (`rustup`)
- Llama 3.2 1B model weights in safetensors format

### Build

```bash
cd llama_sinks_rust
export MACOSX_DEPLOYMENT_TARGET=15.0
cargo build --release
```

### Run

```bash
# GPU mode (Metal, ~73 tok/s)
cargo run --release -- --gpu

# CPU mode (Accelerate BLAS, ~5.5 tok/s)
cargo run --release

# Custom model path
cargo run --release -- --gpu /path/to/llama-3.2-1b
```

### Project Structure

```
src/
├── main.rs          Entry point, CLI args, generation loop
├── config.rs        LlamaConfig deserialization
├── weights.rs       Safetensors loading (f32/f16/bf16)
├── cache.rs         CPU KV cache (for CPU mode)
├── layers.rs        CPU layers (Linear, RMSNorm, Attention, MLP)
├── model.rs         CPU model (LlamaForCausalLM)
├── quantize.rs      BLAS-backed linear (CPU mode)
├── sampling.rs      Min-p sampling
├── generation.rs    CPU generation loop
├── gpu.rs           Metal context, pipelines, encode methods
├── gpu_model.rs     GPU model with Q4 weights + full context KV cache
└── shaders/
    └── llama.metal  All Metal compute kernels
```

---

## Performance Summary

| Optimization | tok/s | Speedup | Key Technique |
|---|---|---|---|
| Python + NumPy | 5.5 | 1x | Baseline |
| Rust + Accelerate BLAS | 5.5 | 1x | Same BLAS, same speed |
| Metal GPU (naive) | 9.7 | 1.8x | GPU parallelism |
| Single command buffer | 19.2 | 3.5x | Eliminate dispatch overhead |
| SIMD-group matvec | 26.9 | 4.9x | Cooperative dot product |
| f16 weights | 51.6 | 9.4x | Halve memory bandwidth |
| **Q4_0 int4** | **73.5** | **13.4x** | Quarter memory bandwidth |

Total journey: **5.5 → 73.5 tok/s** in a from-scratch implementation.
