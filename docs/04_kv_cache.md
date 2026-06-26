# KV Cache Layout

## One-Sentence Summary

The KV cache is stored as two **separate flat buffers** (one for K, one for V) with layout `(num_kv_heads, capacity, row)` where `row` is either `head_dim` floats (f32/f16) or `groups_per_row * bytes_per_group` bytes (Q8_0/Q4_0). **Head-major, token-mid, dim-fastest** — `head_dim` changes fastest in memory.

---

## Cache Types

| Type | `bytes_per_row(head_dim)` | Element size | Precision | Source |
|------|---------------------------|--------------|-----------|--------|
| `F16` | `head_dim * 2` | 2 bytes | half | `gemma4_config.rs:83` |
| `Q8_0` | `(head_dim / 32) * 34` | 34 B / 32 elems | 8-bit int + f16 scale | `gemma4_config.rs:84` |
| `Q4_0` | `(head_dim / 32) * 18` | 18 B / 32 elems | 4-bit int + f16 scale | `gemma4_config.rs:85` |

Controlled by env `LLAMA_KV_CACHE_TYPE` (default: `F16`).

---

## Q4_0 on-disk format (per group of 32)

```
[2-byte f16 scale][16 bytes: 2×4-bit per byte] = 18 bytes
```

Each byte packs two nibbles: `q_lo | (q_hi << 4)`, dequantized as `(nibble - 8) * scale`.
See `llama.metal:3253` `q4_0_read()`.

## Q8_0 on-disk format (per group of 32)

```
[2-byte f16 scale][32 bytes: signed int8] = 34 bytes
```

Dequantized as `int8_val * scale`. See `llama.metal:3243` `q8_0_read()`.

---

## Offset Formula

The canonical offset for K/V element `(kv_head, token_pos, dim)` in cache buffer:

### f32 / f16 cache

```
offset = kv_head * capacity * head_dim
       + token_pos * head_dim
       + dim
```

**Proof** (`llama.metal:1944`):
```metal
uint dst_offset = h * capacity * head_dim + cur_seq * head_dim + d;
//                ──── head stride ────   ── token stride ──  dim
```

### Q8_0 cache

```
offset = kv_head * capacity * row_bytes
       + token_pos * row_bytes
       + group * 34
       + 2 + d_in_group
```

Proof (`llama.metal:3000`):
```metal
uint base_offset = h * capacity * row_bytes + cur_seq * row_bytes + g * 34;
uint row_bytes = groups_per_row * 34;
```

Where `row_bytes = (head_dim / 32) * 34`.

### Q4_0 cache

```
offset = kv_head * capacity * row_bytes
       + token_pos * row_bytes
       + group * 18
       + 2 + nibble_index
```

Proof (`llama.metal:3045`, `q4_0_read` at `llama.metal:3253`):
```metal
uint base_offset = h * capacity * row_bytes + cur_seq * row_bytes + g * 18;
uint row_bytes = groups_per_row * 18;
```

---

## Stride Table

| Access | Stride (f32/f16) | Stride (Q8_0) | Stride (Q4_0) |
|--------|-------------------|---------------|---------------|
| `dim` (fastest) | 1 element | 1 byte* | 1 byte* |
| `token → token+1` | `head_dim` elems | `row_bytes` bytes | `row_bytes` bytes |
| `head → head+1` | `capacity × head_dim` elems | `capacity × row_bytes` bytes | `capacity × row_bytes` bytes |

\* Quantized reads dequantize on-the-fly via helper functions.

---

## Dimension Order (Fastest → Slowest)

```
dim → token_pos → kv_head
```

This is **head-major** layout: all dimensions for a single K/V head's current tokens are contiguous in memory.

---

## Q: Where is token position 700, layer 12, head 3 K/V stored?

### f16 cache
```
k_offset = 3 * capacity * head_dim + 700 * head_dim + d
v_offset = 3 * capacity * head_dim + 700 * head_dim + d
```

The K and V are in **separate buffers** (`k_cache[layer]` and `v_cache[layer]`), each indexed identically.

### Q4_0 cache
```
row_bytes = (head_dim / 32) * 18
k_offset = 3 * capacity * row_bytes + 700 * row_bytes + g * 18 + 2 + nibble
v_offset = 3 * capacity * row_bytes + 700 * row_bytes + g * 18 + 2 + nibble
```

---

## Are K and V separate or interleaved?

**Separate.** Two independent buffers (`k_cache` and `v_cache`), each with the same layout. Never interleaved.

Source: `kv_pool.rs:46-47` (`k_cache: Vec<Buffer>, v_cache: Vec<Buffer>`), `cache.rs:7-8` (`key_cache: Vec<Option<Vec<f32>>>, value_cache: Vec<Option<Vec<f32>>>`).

---

## Is layout layer-major, token-major, or head-major?

**Head-major** (within a layer). There is no inter-layer stride — each layer has its own independent K/V buffer.

Within a buffer:
- `kv_head` changes **slowest** (strided by `capacity × row_bytes`)
- `token_pos` changes **mid** (strided by `row_bytes`)
- `dim` changes **fastest** (contiguous)

---

## What dimension changes fastest?

`dim` (the element within a head's key/value vector). This matches the attention kernel's inner loop: it reads `K_cache[k_offset + d]` for `d = 0..head_dim` (contiguous access, good cache line utilization).

See `llama.metal:1734`:
```metal
uint k_offset = k_head_base + kv * head_dim;
for (uint d = tid; d < head_dim; d += tg_size) {
    partial_dot += Q[q_offset + d] * K_cache[k_offset + d];
}
```

---

## How does GQA/MQA affect cache reads?

GQA reduces KV cache reads by `num_kv_groups`. For Gemma4 E4B with `num_heads=20` and `num_kv_heads=4`, `num_kv_groups=5`. Each KV head is shared by 5 query heads.

### Single-TG-per-head decode (`attention_single_token_offset`)
One threadgroup per **query head**. Each threadgroup independently reads the same KV cache line. **KV bandwidth is NOT shared** — total KV reads = `num_heads × kv_seq × head_dim` (redundant). See `llama.metal:2738`.

### GQA-aware decode (`attention_single_token_offset_f16_gqa`)
One threadgroup per **KV head**, one simdgroup (32 lanes) per query head in the group. KV cache row is loaded **once** per threadgroup and reused by all query heads via `simd_sum`. Total KV reads = `num_kv_heads × kv_seq × head_dim` — **5× less** for Gemma4.

Proof (`llama.metal:2909-2916`):
```metal
const uint kv_h = tgid;       // One TG per KV head
if (kv_h >= num_kv_heads) return;
const uint h = kv_h * num_kv_groups + group;  // Each simdgroup handles one Q head
```
Dispatch: `threadgroups = num_kv_heads, tg_size = num_kv_groups × 32` (`gpu.rs:3336`).

### GQA flash decode Q4_0 (`attention_flash_decode_q4_0_gqa`)
Same principle — one threadgroup per KV head. KV tiles are loaded into threadgroup memory once per tile, then each simdgroup computes its own Q head's dot product. See `llama.metal:5164`.

---

## During decode, are reads contiguous or strided?

**Strided** along the token dimension. The attention kernel loops over `kv = 0..kv_seq`, computing `k_offset = k_head_base + kv * head_dim`, then reading `head_dim` contiguous floats from that offset. The inner `dim` loop is contiguous; the outer `kv` loop strides by `head_dim` elements.

```
Memory access pattern for one decode step:
K_cache: [h0_t0_d0 d1 d2...] [gap] [h0_t1_d0 d1 d2...] [gap] ...
         └── contiguous ──┘  stride=head_dim  └── contiguous ──┘
```

This means one decode step reads **every** cached token position — O(kv_seq) bandwidth per decode token.

With sliding window, `kv_start` limits the range to the last `sliding_window` tokens, but reads within that range are still strided.

---

## What happens as context grows?

- **KV cache memory** grows as `num_kv_heads × capacity × row_bytes` per layer. For Gemma4 with `kv_capacity=4096`, `head_dim=128` (sliding) or `512` (full), `num_kv_heads=4`, `num_layers=42`:
  - F16: ~672 MB total (42 × 4 × 4096 × 128 × 2B + 42 × 4 × 4096 × 512 × 2B)
  - Q4_0: ~378 MB total
- **Decode latency** grows **linearly** with `kv_seq` — each decode step attends to all cached tokens
- **Prefill** uses causal attention (O(n²) compute, O(n) cache writes via batch append kernels)
- **StreamingKVCache** (CPU path, `cache.rs`) evicts middle tokens when `seq > sink_size + window_size`, keeping only `sink_size` head tokens + `window_size` tail tokens
- **Sliding window** (GPU path, `gemma4_gpu_model.rs:2906-2917`) limits attention to last `sliding_window` tokens for sliding layers, but keeps writing to the flat cache at position `kv_seq`
- **CPU `kv_seq_lens`** updated per-layer; for the streaming path all layers share the same seq_len
- **Buffer capacity** is fixed at allocation; the streaming path reallocates (doubles) when exceeded

### Context → Decode throughput (Gemma4 E4B, Q4_0 weights, f16 KV cache, Apple M1 Pro)

| Context (tokens) | Decode tok/s | Relative |
|:----------------:|:------------:|:--------:|
|       128        |    3.1       |   1.0×   |
|       512        |    1.1       |   0.35×  |
|      1024        |    0.6       |   0.19×  |
|      2048        |    0.3       |   0.10×  |
|      4096        |    0.1       |   0.03×  |

Throughput scales as ~1/kv_seq, confirming the O(n) attention cost per decode step.

---

## Allocation

### GPU pool (`kv_pool.rs:61-110`)
`KvCachePool::new()` allocates `num_slots × num_layers` Metal buffers. Each buffer:
```
byte_len = num_kv_heads * max_seq_len * bytes_per_row
```
With `max_seq_len` typically 4096 (capped at `gemma4_gpu_model.rs:1202`).

### GPU model (`gpu_model.rs:110-117`)
Simple model: one buffer per layer per K/V:
```rust
k_cache.push(ctx.buffer_empty(num_kv_heads * kv_capacity * head_dim));
v_cache.push(ctx.buffer_empty(num_kv_heads * kv_capacity * head_dim));
```

### CPU streaming (`cache.rs:81-86`)
Dynamic allocation on first use, `vec![0.0f32; num_kv_heads * cap * head_dim]`, grows by 2× when exceeded.

---

## Write Paths

### Single-token append (decode)
`kv_cache_append_f16` (`llama.metal:1952`): one thread per element (`num_kv_heads × head_dim` threads). Each thread computes:
```metal
uint dst_offset = h * capacity * head_dim + cur_seq * head_dim + d;
cache[dst_offset] = half(new_data[h * head_dim + d]);
```

### Batch append (prefill)
`kv_cache_batch_append_f16` (`llama.metal:2007`): one thread per element (`num_kv_heads × seq_len × head_dim` threads). Each thread:
```metal
uint dst_offset = h * capacity * head_dim + (cur_seq + s) * head_dim + d;
```

### Fused KV append + attention
`attention_flash_decode_fused_q4_0` (`llama.metal:5350`): after attention completes, the first simdgroup per KV head appends the new K/V data at position `cur_seq`.

---

## Read Paths (Decode Attention)

All decode attention kernels follow the same pattern:

```
for kv in 0..kv_seq:
    actual_pos = kv_start + kv
    for d in 0..head_dim:
        dot += Q[h * head_dim + d] * K_cache[k_head_base + actual_pos * head_dim + d]
    softmax_update(...)
    for d in 0..head_dim:
        output[d] += weight * V_cache[v_head_base + actual_pos * head_dim + d]
```

The key insight: **each decode step re-reads the entire cached sequence** from position `kv_start` to `kv_seq-1`. There is no incremental maintenance — attention is recomputed from scratch every step.

---

## Position Indexing / Rotary Embeddings

Two approaches:

1. **CPU precomputed** (`gpu_model.rs:158-168`): compute cos/sin for current position, write to GPU buffer
2. **GPU computed** (`rope_fill_decode`, `llama.metal:1787`): fills packed cos/sin for all layers at decode position in one dispatch

Rotary is applied in-place to Q and K buffers **before** writing to KV cache. The cache stores **rotated** keys. This means:
- Cached keys already have position-dependent rotation
- Position `pos` of a key is `kv_start + offset` within the current context
- Sliding window layers rotate at the **global** position, not the window-relative position

---

## Key Source Files

| File | Lines | Role |
|------|-------|------|
| `src/kv_pool.rs` | 61-110 | GPU KV pool allocation |
| `src/gpu_model.rs` | 110-117 | Simple GPU KV allocation |
| `src/cache.rs` | 1-212 | CPU streaming KV cache (flat vec) |
| `src/layers.rs` | 265-455 | CPU attention + GQA cache reads |
| `src/gemma4_gpu_model.rs` | 1195-1226 | Gemma4 GPU KV alloc |
| `src/gemma4_gpu_model.rs` | 2780-3029 | Gemma4 GPU KV append + attention dispatch |
| `src/gemma4_config.rs` | 64-87 | `KvCacheType` enum + `bytes_per_row` |
| `src/shaders/llama.metal` | 1928-1960 | `kv_cache_append`, `kv_cache_append_f16` |
| `src/shaders/llama.metal` | 2893-2962 | GQA-aware decode attention (f16) |
| `src/shaders/llama.metal` | 5164-5263 | GQA flash decode attention (Q4_0) |
| `src/shaders/llama.metal` | 5350-5437 | Fused KV append + flash decode (Q4_0) |
| `src/gpu.rs` | 2822-2890 | `encode_kv_append*` dispatch |
| `src/gpu.rs` | 3199-3338 | `encode_attention_with_offset*` dispatch |
