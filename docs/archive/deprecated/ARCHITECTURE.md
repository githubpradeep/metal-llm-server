# Architecture Documentation

## Overview

This is a from-scratch Llama 3.2 1B inference engine in Rust with two backends:
- **CPU**: Apple Accelerate BLAS (sgemv/sgemm)
- **GPU**: Metal compute shaders with Q4_0 quantized weights

## Data Flow (GPU Mode, Single Token Decode)

```
Token ID (usize)
    │
    ▼
┌─────────────────────┐
│ Embedding Lookup    │  CPU: index into f32 table, write to GPU buffer
│ (128256 × 2048)     │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Rotary cos/sin      │  CPU: compute for current position, write to GPU
│ (64 floats each)    │
└─────────────────────┘
    │
    ▼
╔═════════════════════════════════════════════════════════╗
║  SINGLE METAL COMMAND BUFFER (all 16 layers + LM head) ║
╠═════════════════════════════════════════════════════════╣
║                                                         ║
║  ┌─── Layer i (×16) ───────────────────────────────┐   ║
║  │                                                  │   ║
║  │  buf_copy: hidden → residual                     │   ║
║  │  rmsnorm: hidden → normed                        │   ║
║  │  matvec_q4: Q_proj(normed) → q_buf              │   ║
║  │  matvec_q4: K_proj(normed) → k_buf              │   ║
║  │  matvec_q4: V_proj(normed) → v_buf              │   ║
║  │  apply_rotary: q_buf, k_buf (in-place)          │   ║
║  │  kv_cache_append: k_buf → k_cache[i]            │   ║
║  │  kv_cache_append: v_buf → v_cache[i]            │   ║
║  │  attention_single_token: q,K,V → attn_out       │   ║
║  │  matvec_q4: O_proj(attn_out) → o_out            │   ║
║  │  vec_add: residual + o_out → hidden              │   ║
║  │  buf_copy: hidden → residual                     │   ║
║  │  rmsnorm: hidden → normed                        │   ║
║  │  matvec_q4: gate_proj(normed) → gate_buf         │   ║
║  │  matvec_q4: up_proj(normed) → up_buf             │   ║
║  │  silu_mul: silu(gate) * up → silu_buf            │   ║
║  │  matvec_q4: down_proj(silu_buf) → down_buf       │   ║
║  │  vec_add: residual + down_buf → hidden           │   ║
║  │                                                  │   ║
║  └──────────────────────────────────────────────────┘   ║
║                                                         ║
║  rmsnorm: hidden → normed (final)                       ║
║  matvec_q4: lm_head(normed) → logits                   ║
║                                                         ║
╚═════════════════════════════════════════════════════════╝
    │
    ▼
┌─────────────────────┐
│ Read logits to CPU  │  128,256 floats
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Min-p Sampling      │  CPU: filter + multinomial sample
└─────────────────────┘
    │
    ▼
Next Token ID
```

## Memory Layout

### Weight Storage (Q4_0)

Each weight matrix is stored as packed Q4_0 blocks:

```
Row 0: [block_0][block_1]...[block_{K/32 - 1}]
Row 1: [block_0][block_1]...[block_{K/32 - 1}]
...

Each block (18 bytes):
┌──────────┬──────────────────────────────────┐
│ f16 scale│ 16 bytes: 32 packed 4-bit values  │
│ (2 bytes)│ byte[i] = (q[2i+1] << 4) | q[2i] │
└──────────┴──────────────────────────────────┘

Dequantization: value = (nibble - 8) * scale
```

### KV Cache Layout

```
Per layer, per K and V:
Buffer shape: (num_kv_heads, capacity, head_dim) = (8, 2048, 64)

Head 0: [pos_0][pos_1]...[pos_{seq-1}][unused...]
Head 1: [pos_0][pos_1]...[pos_{seq-1}][unused...]
...
Head 7: [pos_0][pos_1]...[pos_{seq-1}][unused...]

Each position: 64 × f32 = 256 bytes
Total per layer: 8 × 2048 × 64 × 4 = 4 MB
Total all layers: 16 × 4 MB = 64 MB (for full 2048 context)
```

### Scratch Buffers (Pre-allocated, Reused)

```
hidden_buf:    2048 × f32 = 8 KB
normed_buf:    2048 × f32 = 8 KB
residual_buf:  2048 × f32 = 8 KB
q_buf:         2048 × f32 = 8 KB   (32 heads × 64 dim)
k_buf:          512 × f32 = 2 KB   (8 heads × 64 dim)
v_buf:          512 × f32 = 2 KB
attn_out_buf:  2048 × f32 = 8 KB
gate_buf:      8192 × f32 = 32 KB
up_buf:        8192 × f32 = 32 KB
silu_buf:      8192 × f32 = 32 KB
down_buf:      2048 × f32 = 8 KB
logits_buf:  128256 × f32 = 500 KB
```

## Metal Compute Kernels

### matvec_q4 (Hot Path)

The most performance-critical kernel. Called 113 times per token.

**Dispatch:** M threadgroups × 32 threads each (one SIMD group per output row)

**Algorithm:**
1. Each thread handles K/(32×32) = 2 groups of 32 weights
2. For each group: read f16 scale + 16 packed bytes
3. Unrolled dequant: 4 bytes (8 weights) per iteration
4. Deferred scale: accumulate integer products, multiply scale once
5. `simd_sum` reduces across 32 lanes (hardware, 1 cycle)
6. Lane 0 writes output

**Memory access pattern:**
- Weight reads: sequential within a row (good for GPU cache lines)
- Activation reads: strided by SIMD_SIZE groups (but x is small, fits in L1)

### attention_single_token

**Dispatch:** num_heads threadgroups × 64 threads each

**Algorithm per head:**
1. Compute Q·K^T scores (distributed across threads)
2. Parallel max reduction (shared memory)
3. Softmax: exp(score - max), parallel sum reduction
4. Normalize scores
5. Weighted sum of V values (distributed across head_dim)

**Shared memory:** 2560 floats for scores + 256 for reductions

### rmsnorm

**Dispatch:** 1 threadgroup × 256 threads (for single vector)

**Algorithm:**
1. Parallel sum of squares (each thread handles dim/256 elements)
2. Shared memory reduction to get total
3. Compute inv_rms = rsqrt(mean + eps)
4. Parallel normalize: out[i] = x[i] * inv_rms * weight[i]

## Grouped-Query Attention (GQA)

Llama 3.2 uses GQA: 32 query heads share 8 KV heads (4 groups).

Instead of physically copying KV heads (which wastes memory bandwidth), we use **virtual indexing**:

```metal
uint kv_h = h / num_kv_groups;  // Map query head → KV head
// Query head 0,1,2,3 → KV head 0
// Query head 4,5,6,7 → KV head 1
// ...
```

This saves copying 4× the KV cache data on every attention call.

## Rotary Position Embeddings (RoPE)

Applied in-place to Q and K after projection:

```metal
// For each (head, dimension_pair):
float q1 = q[base + d];
float q2 = q[base + d + half_dim];
q[base + d]            = q1 * cos - q2 * sin;
q[base + d + half_dim] = q2 * cos + q1 * sin;
```

cos/sin are precomputed on CPU for the current position (trivial cost).

## Quantization Details

### Why Q4_0 Works

For a 1B model, 4-bit quantization introduces minimal quality loss because:
- The weight distributions are approximately Gaussian
- Per-group scaling (32 weights per group) captures local variance
- The model has enough parameters that individual weight precision matters less

### Memory Savings

| Component | f32 | f16 | Q4_0 |
|-----------|-----|-----|------|
| Layer weights (16 layers) | 3.8 GB | 1.9 GB | 0.54 GB |
| LM head | 1.0 GB | 0.5 GB | 0.14 GB |
| Embeddings | 1.0 GB | 1.0 GB | 1.0 GB (kept f32) |
| KV cache (200 tokens) | 6.4 MB | 6.4 MB | 6.4 MB (kept f32) |
| **Total** | **5.8 GB** | **3.4 GB** | **1.7 GB** |

## Performance Model

For single-token decode, time per token ≈ bytes_read / bandwidth:

```
Q4_0 weights per token: ~0.54 GB (all layer weights)
Apple Silicon bandwidth: ~100 GB/s (base M-series)
Theoretical max: 100 / 0.54 ≈ 185 tok/s

Achieved: 73.5 tok/s = 40% of theoretical
Overhead: kernel dispatch, attention, non-matmul ops, sampling
```

The gap between theoretical and achieved is due to:
- Attention kernel (reads KV cache, not just weights)
- Non-matmul operations (RMSNorm, SiLU, residual adds)
- Command buffer submission overhead
- Sampling on CPU (reading logits back)
