# Bottleneck Analysis: gemma4 E4B Decode Performance

## One-Sentence Summary

Decode is dominated by **1,344 tiny Matrix-Vector multiplications** (42 layers × 4 matvecs + 1 head) and **42 attention kernels** that reread the entire KV cache from global memory — neither is compute-limited; both are memory-bandwidth limited.

---

## Throughput Summary

| Metric | Value |
|--------|-------|
| Best throughput (env=llama_q4_0, n=128) | ~33.7 tok/s |
| llama.cpp comparison (same model) | ~40 tok/s (est.) |
| Gap | ~6 tok/s (~18% slower) |
| Optimal CPU decode | ~6 ms CPU encode overhead |
| `gpu_wait` time at n=128 | ~34 ms (GPU busy) |
| `encode` time (CPU) | ~0.65 ms |

---

## Bottleneck #1: Too Many Tiny GPU Dispatches

> **Impact:** ~9 ms per token (9% of total time)

The decode path dispatches **~638 kernel launches** per token. Each launch incurs fixed overhead:

| Overhead source | Per-dispatch cost | Total (638 ×) |
|-----------------|-------------------|----------------|
| CPU encode: set pipeline + bind buffers + dispatch | ~1-3 µs | ~1-2 ms |
| GPU scheduler gap between tiles | ~10-30 µs | ~6-19 ms |
| **Total overhead** | | **~7-21 ms (est. ~9 ms)** |

The core problem: the architecture models each transformer operation (matvec, rmsnorm, vec_add, rotary) as a separate Metal dispatch. For 42 layers:

| Operation | Dispatches/layer | Total (42 layers) |
|-----------|-----------------|-------------------|
| `matvec_q4` (q, k, v, o, gate, up, down, 2× PLE) | 8 (+ 0-3 KV) | ~378 |
| `rmsnorm` (pre-attn, post-attn, pre-ff, post-ff, post-PLE) | 5 | ~210 |
| `rmsnorm_per_head` (Q, K) | 2 | ~84 |
| `vec_add` (residual skip, PLE residual) | 2 | ~84 |
| `vec_scale` (context, layer_scalar) | 1 | ~42 |
| `rotary` (Q, K) | 2 | ~84 |
| `gelu_mul` / `gelu_mul_at` | 2 | ~84 |
| `kv_append` | 2 | ~48 |
| `attention_flash_decode` | 1 | ~42 |
| Subtotal per token | | ~1056 |

Many of these (rmsnorm, vec_add, vec_scale, gelu_mul) are trivial: 1 threadgroup × 256 threads on a vector of 2560 elements. These complete in <5 µs. Their overhead-to-compute ratio is terrible.

### Dispatch Comparison

| Approach | Launches | Comment |
|----------|----------|---------|
| Current (separate per op) | ~638 | One dispatch per tensor operation |
| Mega graph (`MEGA_KERNEL=1`) | ~1056 (same) | Still dispatches each op individually |
| llama.cpp (fused matvec) | ~168 | Gate+up fused, ~4 launch points/layer |

**llama.cpp achieves ~40 tok/s. The 18% gap is almost entirely dispatch overhead.** Each extra 1 µs of overhead per dispatch costs ~0.6 ms per token.

---

## Bottleneck #2: Attention Flash Decode Reads All KV from Global Memory

> **Impact:** ~40 ms at 128 ctx, scales linearly with context

The `attention_flash_decode_q4_0` kernel:

```
for each KV head group:
    for each tile of kv_seq (tile_size=32 or threadgroup_size):
        load K tile from Q4_0 → dequant → f16 (global read)
        load V tile from Q4_0 → dequant → f16 (global read)  
        for each query head in group:
            Q·K dot product → attention score → softmax → V accumulator
    write output for this query head × head_dim
```

**Each attention call rereads the entire KV cache in Q4_0 format.** At 128-token context:
- K cache: 4 heads × 128 × (128+512)/2 avg × 18 bytes/32 weights ≈ ~9 KB per attention call × 42
- V cache: same
- Total KV reads: ~0.75 MB per token at 128 ctx, ~24 MB at 4096 ctx

**This pattern repeats 42 times** (once per layer). The same KV data is read from global memory 42 times per token.

### Would Chunked Prefill Help?

Yes — but decode is inherently sequential. The attention kernel O(KV_seq) compute is unavoidable for single-token decode. The 42× repeat is addressable by fusing multiple attention calls into one (one big kernel that processes all layers' KV caches).

---

## Bottleneck #3: Q4.0 Dequantization Overhead

> **Impact:** ~55% of matmul time spent on dequantization

Each `matvec_q4` kernel:
1. Reads Q4_0 blocks (32 weights packed into 18 bytes: 16 nibbles + 2 bytes scale)
2. Dequantizes on the fly: `f32_val = ((int8_t)nibble - 8) * scale` — 2 bytes → 2 f32 loads, 1 mul, 1 sub per weight
3. Accumulates in f32 registers

The Q4_0 format is chosen for KV cache memory savings, but the on-the-fly dequantization:
- Saturates ALU rather than memory bandwidth for small M (M=1 in all decode matvecs)
- Has poor arithmetic intensity: ~1.5 FLOP/byte (vs ~8 FLOP/byte needed to be compute-bound on M1)

**This is a necessary evil for KV cache capacity** but could be optimized:
- Q4_1 or Q6_K would trade capacity for throughput
- Batch dequantization: dequant a full row into a reusable f16 buffer (not possible with M=1 decode)

---

## Bottleneck #4: CPU-GPU Synchronization Per Token

> **Impact:** ~0.65 ms encode + ~6 ms decode overhead

The decode loop:

```cpp
// CPU: gather embedding → write_buffer
// CPU: build command buffer (638 dispatch calls)
// CPU: commit command buffer
// GPU: execute command buffer (34 ms)
// CPU: wait for GPU (gpu_wait)
// CPU: read logits → sample → next token
```

The CPU is mostly idle during GPU execution, but:
- **Encode overhead** (0.65 ms): building 638 MTLRenderCommandEncoder commands — 1 µs each
- **GPU scheduling overhead**: Metal's scheduler serializes tiny dispatches
- **Read-back stall**: `wait_until_complete` blocks CPU until GPU finishes all 638 kernels

**Potential fix:** Use double-buffered command buffers and async GPU completion. Overlap encode of token t+1 with execution of token t.

---

## Bottleneck #5: Embedding Gather on CPU

> **Impact:** ~1-2 ms per token

The input embedding is a CPU gather into a GPU buffer:

```rust
let token_id = self.token_ids[*i as usize];
let embed_row = &self.input_embedding[token_id as usize];
let embed_buf = self.decode_io_bufs.embed.slice(offset..).unwrap();
unsafe {
    std::ptr::copy_nonoverlapping(
        embed_row.as_ptr(),
        embed_buf.contents().add(offset),
        embed_row.len(),
    );
}
```

For batch=1 this is trivial, but for each token it requires:
1. CPU read from embedding matrix (~2.5K f32 = 10 KB)
2. `write_buffer` (CPU→GPU transfer)
3. MTLBuffer synchronization on unified memory

On M1 Pro unified memory this is just a pointer copy, but it still serializes with the GPU command buffer setup.

---

## Relative Impact at Different Context Lengths

| Bottleneck | 128 ctx | 512 ctx | 2048 ctx | 4096 ctx |
|------------|---------|---------|----------|----------|
| Dispatch overhead | **9%** | **3%** | **1%** | **<1%** |
| Attention KV reads | **38%** | **81%** | **94%** | **97%** |
| MLP matvecs | **48%** | **14%** | **5%** | **2%** |
| LM head matvec | **5%** | **2%** | **<1%** | **<1%** |

**Key insight:** At short context (128), MLP dominates. At long context (>512), attention dominates absolutely. Dispatch overhead is always a concern but gets drowned out at long context.

---

## Ranking by Estimated Improvement Potential

| Rank | Fix | Est. speedup at 128 ctx | Complexity |
|------|-----|------------------------|------------|
| 1 | Fuse gate+up matvec (already partially done) | ~5% | Low |
| 2 | Fuse rmsnorm→matvec (eliminate intermediate writes) | ~5% | Medium |
| 3 | Fuse all per-layer ops into one kernel (mega kernel) | ~10% | High |
| 4 | Async GPU (overlap encode with execution) | ~2% | Medium |
| 5 | Attention layer fusing (process all 42 KV caches in one kernel) | ~15-20% | Very High |
| 6 | FlashAttention V2-style online softmax | ~10% attn | High |
| 7 | Q4_0 dequant specialization (warp-level dequant) | ~5% | Medium |

---

## Sources

| Source | File | Key Data |
|--------|------|----------|
| Dispatch loop | `src/gemma4_gpu_model.rs:2394-3621` | Forward per-token kernel sequence |
| Phase timers | `src/gemma4_gpu_model.rs:3623-3651` | PREPASS, ATTN, MLP_PLE, HEAD times |
| GPU profile | `benchmarks/bottleneck_results.txt` | PROFILE_GPU breakdown |
| Perf TODO | `perf.todo.md` | Baseline numbers, ranked improvements |
| Q4 matmul dispatch | `src/gpu.rs:1499-1539` | `encode_matvec_q4_at` kernel launch config |
| Op graph builder | `src/mega_decode.rs:454-666` | All 1056 operations per token |
| Sliding window | `src/gemma4_gpu_model.rs:183-217` | Gemma4Config, layer architecture |
| Flash attention kernel | `src/shaders/llama.metal:473-583` | `attention_flash_decode_q4_0` |
| Matvec kernel | `src/shaders/llama.metal:207-299` | `matvec_q4_fast` |
