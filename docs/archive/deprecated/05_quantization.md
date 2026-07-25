# Quantization Format and MatMul Path

## One-Sentence Summary

Weights are stored in **Q4_0** format (GGUF/ggml compatible): 4-bit symmetric per-block quantization with f16 scale, 32 elements per block, 18 bytes per block. **Activations remain f32** throughout. Dequantization happens **on-the-fly inside the matmul Metal kernel** — the GPU never materializes a full f32 weight matrix.

---

## What is Quantized?

**Weights only.** All linear projection matrices are quantized:

| Projection | Shape (Gemma4 E4B) | Quantized |
|------------|-------------------|-----------|
| `q_proj` | (2560 or 10240) × 2560 | Q4_0 |
| `k_proj` | (512 or 2048) × 2560 | Q4_0 |
| `v_proj` | (512 or 2048) × 2560 | Q4_0 |
| `o_proj` | 2560 × (2560 or 10240) | Q4_0 |
| `gate_proj` | 10240 × 2560 | Q4_0 |
| `up_proj` | 10240 × 2560 | Q4_0 |
| `down_proj` | 2560 × 10240 | Q4_0 |
| `lm_head` | 262144 × 2560 | Q4_0 |
| PLE projections | 256 × 2560, 2560 × 256 | Q4_0 |

Activations (`hidden_states`, Q, K, V, attention output, MLP intermediates) are all **f32** throughout.

---

## Q4_0 Block Format

| Field | Size | Layout |
|-------|------|--------|
| `d` (scale) | 2 bytes | f16 (IEEE 754 half-precision) |
| `qs` (quants) | 16 bytes | 32 × 4-bit signed values, packed 2 per byte |

```
One block (32 weights, 18 bytes):
┌───── 2 bytes ─────┐────────── 16 bytes ──────────┐
│   f16 scale (d)   │  nibble pairs packed in bytes │
└────────────────────┴──────────────────────────────┘
```

Byte `i` of `qs` (for `i=0..15`):
- low nibble = quantized value for element `i`
- high nibble = quantized value for element `i+16`

Reconstruction:
```
value = (nibble - 8) × scale
      = (nibble - 8) × (max_abs / 7)
```

**Source**: `llama.metal:85-86`, `ggml_mul_mv_q4.metal:57-60` (`struct block_q4_0`)

### Constants

| Constant | Value | Source |
|----------|-------|--------|
| `Q4_GROUP_SIZE` | 32 | `llama.metal:91` |
| `Q4_BLOCK_BYTES` | 18 | `llama.metal:92` |
| Scale range divider | 7.0 (max value of 4-bit signed = 7) | `gpu.rs:5035` |
| Zero point | none (symmetric, shifted by -8) | — |
| Effective bitrate | 18 × 8 / 32 = **4.5 bits/weight** | — |

### Key properties

| Property | Value |
|----------|-------|
| Symmetric? | **Yes** (no zero point; `nibble - 8` centers at 0) |
| Block size | 32 elements |
| Per-tensor/per-channel/per-block | **Per-block** (each group of 32 has its own f16 scale) |
| Zero point | None (symmetric quant) |

---

## Q3_0 Block Format (optional, used on some layers)

| Field | Size | Layout |
|-------|------|--------|
| `d` (scale) | 2 bytes | f16 |
| `qs_low` | 8 bytes | 32 × 2-bit low bits, packed 4 per byte |
| `qs_high` | 4 bytes | 32 × 1-bit high bit, packed 8 per byte |

Scale = `max_abs / 3`, reconstruct: `(q - 4) × scale`. 14 bytes per 32 weights ~ 3.5 bpw.

Controlled by env `Q3_LAYER_RANGE` (e.g., `Q3_LAYER_RANGE=0-5`). See `gpu.rs:4939`.

---

## Scale layout

Scales are **inline**, stored at the start of each block:

```
Row r: [block_0: d(2B) | qs(16B)] [block_1: d(2B) | qs(16B)] ... [block_n: d(2B) | qs(16B)]
```

Row stride: `(K / 32) × 18` bytes.

A weight matrix of shape `(M, K)` occupies:
```
total_bytes = M × (K / 32) × 18
```

Compare with f16: `M × K × 2` bytes (4.56× larger), f32: `M × K × 4` bytes (9.1× larger).

---

## Memory Read for One Dot Product

To compute `y[r] = sum_{i=0}^{K-1} W[r][i] × x[i]` (one row of the output):

1. **Read scale**: 2 bytes (f16) at position `r × row_bytes + g × 18`
2. **Read 16 packed bytes** at offset `r × row_bytes + g × 18 + 2`
3. **Read 32 floats of x** from `x[g × 32 .. g × 32 + 32]` (128 bytes)
4. **Repeat** for `g = 0..(K/32)` groups

Total reads per row: `(18 + 128) × (K/32)` bytes = `(18 + 128) × K/32`.

For a typical decode matvec `M=2560, K=2560`:
- Weight reads: `2560 × 80 × 18 = 3,686,400` bytes (~3.7 MB)
- Activation reads: `2560 × 80 × 128 = 26,214,400` bytes (~26 MB, but x is small and cached)
- Output writes: `2560 × 4 = 10,240` bytes

The kernel reads **exactly** these bytes — no decompressed f32 weight buffer is ever materialized.

---

## Quantization Path: Rust (CPU, model load)

### Step 1: Load f32 weights from safetensors
`gemma4_gpu_model.rs:1031-1051`: decode safetensor tensor bytes to `Vec<f32>`.

### Step 2: Quantize to Q4_0
`gpu.rs:5015-5058` `fn quantize_q4_0(data, rows, cols)`:

```
for each row:
  for each group of 32:
    1. Find max_abs in group
    2. scale = max_abs / 7.0
    3. Store scale as f16 (2 bytes)
    4. for i in 0..16:
         q_lo = round(v[i] / scale) + 8, clamp to [0,15]
         q_hi = round(v[i+16] / scale) + 8, clamp to [0,15]
         byte[i] = q_lo | (q_hi << 4)
```

### Step 3: Upload to Metal buffer
`gpu.rs:975-983` `fn buffer_from_f32_as_q4`:
```rust
let q4_data = quantize_q4_0(data, rows, cols);
let byte_len = q4_data.len() as u64;
self.device.new_buffer_with_data(q4_data.as_ptr(), byte_len, ...)
```

### Step 4: Dispatch as GPU buffer
The resulting `Buffer` is stored in `Gemma4GpuLayer` fields and dispatched as `buffer(0)` in matmul kernels.

---

## MatMul Path: Metal GPU (decode, M=1)

### Dispatch

`gpu.rs:1499-1539` `encode_matvec_q4_at` → selects kernel variant → dispatches.

### Kernel: `matvec_q4_fast` (default decode kernel)

`llama.metal:269-320` `matvec_q4_fast_body<ROWS=4>`:

Per threadgroup: 8 simdgroups × 32 lanes = 256 threads.
Each simdgroup owns 4 output rows. All lanes in the simdgroup cooperate on the K loop.

```
for each group g (0..K/32):
    lane g loads 8 float4 from x[g*32..g*32+32]  → xv0..xv7 (registers)
    for each owned row r (0..3):
        read f16 scale from W_q4[row * row_bytes + g * 18]  → scale (register)
        call q4_dot_vec_fast(qs_ptr, xv0..xv7)              → partial dot (register)
        acc[r] += partial_dot * scale
```

### `q4_dot_vec_fast` (the core dequant + dot)

`llama.metal:220-258`:

```
4 chunks of 4 bytes each:
  chunk 0: read 4 bytes (q0..q3)
    qi = uint4(q[0], q[1], q[2], q[3])
    flo = float4(qi & 0xF) - 8    // dequant low nibbles
    fhi = float4(qi >> 4) - 8     // dequant high nibbles
    acc += flo * xv0 + fhi * xv4  // dot with x[0..3] and x[16..19]

  chunk 1: read bytes 4-7 → dot with x[4..7], x[20..23]
  chunk 2: read bytes 8-11 → dot with x[8..11], x[24..27]
  chunk 3: read bytes 12-15 → dot with x[12..15], x[28..31]

return acc.x + acc.y + acc.z + acc.w
```

### Reduction

After the K loop, each lane holds a partial sum. `simd_sum(acc[r])` reduces across 32 lanes.
Lane 0 writes the result: `y[base_row + r] = simd_sum(acc[r])`.

### Accumulation and output dtypes

| Component | Dtype |
|-----------|-------|
| Weight scale | f16 (loaded as float) |
| Dequantized weight | float (computed in-register) |
| Activation (x) | f32 |
| Dot product | float (32-bit, in register accumulator) |
| Partial sum per lane | float |
| Output (y) | f32 |

**Everything is float32** for the accumulation path. The only half-precision data is the stored scales.

---

## Complete End-to-End Trace

Here is the full path for `attn_out = o_proj @ attn_input` during decode:

```
1. model load:  safetensors → f32 Vec → quantize_q4_0() → Metal Buffer (GPU.rs:5015-5058, 975-983)
2. forward:     encode_matvec_q4_at(encoder, &layer.o_proj, &attn_out_buf, &o_out_buf, hidden_size, q_out)
                (GPU.rs:1499-1539)
3. dispatch:    8 threadgroups × 256 threads (2560 rows, ROWS=4, 8 SG/TG)
                (GPU.rs:1537: matvec_q4_dispatch)
4. per thread:  loop over K/32 groups, each lane loads 8 float4 from x
                (llama.metal:294-303)
5. per group:   read f16 scale + 16 packed bytes from weight buffer
                (llama.metal:305-309)
6. dequant:     unpack 4 bytes → 2 float4 via nibble extraction, subtract 8
                (llama.metal:220-258 q4_dot_vec_fast)
7. dot:         multiply by 8 float4 of x, accumulate in float register
                (llama.metal:231, 239, 247, 255)
8. scale:       multiply partial dot by f16 scale → float
                (llama.metal:309)
9. reduce:      simd_sum across 32 lanes → float per output element
                (llama.metal:315)
10. write:      lane 0 writes y[row] = float
                (llama.metal:317)
```

---

## Kernel Variants

| Variant | ROWS | SG/TG | Description | Best for |
|---------|------|-------|-------------|----------|
| `r1` | 1 | 8 | One row per SIMD group | Small M (narrows, e.g., 256) |
| `r2` | 2 | 4 | Two rows per SIMD group | Medium M |
| `fast` (default) | 4 | 8 | Four rows per SIMD group, 8 SG | Decode shapes |
| `r8` | 8 | 4 | Eight rows per SIMD group | Large M |
| `lc` | 4 | 2 | llama.cpp-style, 2 SG × 4 rows | General (legacy) |
| `splitk` | 4 | 8 | Split-K across K, partial results per TG | Very wide M (lm_head) |
| `ggml` | 4 | 2 | ggml-metal `mul_mv_q4_0_f32` kernel | All-round (auto default) |
| `ggml-ext` | 4 | 2 | Extended ggml with runtime nsg | Tuning |

Selected at runtime via `MATVEC_KERNEL` env var. Default is `Auto` which benchmarks at startup.

---

## Dual and Fused Kernels

The codebase has optimized fused kernels that combine multiple matmuls into one dispatch:

- **`matvec_q4_dual`**: `y0 = W0 × x`, `y1 = W1 × x` with single x load (used for `gate+up`, `k+v`)
- **`matvec_q4_dual_gelu`**: computes `y = GeLU(W0 × x) × (W1 × x)` in one kernel, no intermediate buffer
- **`matvec_q4_interleaved_gelu`**: same but W0 and W1 are interleaved row-wise in one buffer

---

## Related Files

| File | Lines | Role |
|------|-------|------|
| `src/shaders/llama.metal` | 85-92 | Q4_0 constants |
| `src/shaders/llama.metal` | 96-258 | `q4_dot_vec`, `q4_dot_vec_fast` |
| `src/shaders/llama.metal` | 269-321 | `matvec_q4_fast_body` |
| `src/shaders/ggml_mul_mv_q4.metal` | 57-76 | `block_q4_0` struct, `block_q_n_dot_y` |
| `src/gpu.rs` | 40-45 | `weight_buf_is_q4` detection |
| `src/gpu.rs` | 975-983 | `buffer_from_f32_as_q4` |
| `src/gpu.rs` | 1499-1539 | `encode_matvec_q4_at` dispatch |
| `src/gpu.rs` | 4898-4933 | `f32_to_f16`, `f16_to_f32` |
| `src/gpu.rs` | 5015-5058 | `quantize_q4_0` (CPU) |
| `src/gemma4_gpu_model.rs` | 1083-1130 | Weight loading with quantization |
| `src/quantize.rs` | 1-92 | CPU fallback (Accelerate BLAS, f32 weights) |
