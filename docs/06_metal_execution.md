# Metal Execution: Per-Token Kernel Launch Table

## One-Sentence Summary

One decode token launches **~660 tiny Metal kernel dispatches** (42 layers × ~15 dispatches + global overhead). The GPU is bottlenecked by dispatch overhead and KV cache attention bandwidth, not compute.

---

## Per-Token Kernel Launch Table

Table for Gemma4 E4B default config (`FUSED_QKV=0, FLASH_ATTN=1, MLP_GATE_UP_GGML=1`).
Times from `bottleneck_results.txt` + `perf.todo.md`; 128-token context, Apple M1 Pro.

### Global (once per token)

| # | Kernel | Grid | Total time | % of decode |
|---|--------|------|------------|-------------|
| 1 | `rope_fill_decode` | layers × half_dim | — | — |
| 2 | `matvec_q4` (PLE ctx proj) | 10752×2560 | — | — |
| 3 | `vec_scale` | 10752 | — | — |
| 4 | `rmsnorm_per_head` (PLE) | 42×256 | — | — |
| 5 | `vec_add` | 10752 | — | — |
| 6 | `vec_scale` | 10752 | — | — |
| | **∑ Prep/embed/PLE** | | **~3 ms** | **~9%** |

### Per KV layer (24 layers, has_kv=true) — ×24

| # | Kernel | Grid | Time/call | Total/layer |
|---|--------|------|-----------|-------------|
| 7 | `rmsnorm` | 1 TG × 256 thr | — | — |
| 8 | `matvec_q4` (q_proj) | q_out × hidden | — | — |
| 9 | `matvec_q4` (k_proj) | kv_out × hidden | — | — |
| 10 | `matvec_q4` (v_proj) | kv_out × hidden | — | — |
| 11 | `rmsnorm_per_head` (Q) | 20 heads × 128/512 | — | — |
| 12 | `apply_rotary` (Q) | num_heads × half_dim | — | — |
| 13 | `rmsnorm_per_head` (K) | 4 heads × 128/512 | — | — |
| 14 | `apply_rotary` (K) | num_kv_heads × half_dim | — | — |
| 15 | `rmsnorm_per_head_noweight` (V) | 4 heads × 128/512 | — | — |
| 16 | `kv_cache_append_q4_0` (K) | num_kv_heads × groups | — | — |
| 17 | `kv_cache_append_q4_0` (V) | num_kv_heads × groups | — | — |
| 18 | `attention_flash_decode_q4_0` | num_heads × tile_kv | **~1.6 ms** | **~50%** |
| 19 | `matvec_q4` (o_proj) | hidden × q_out | — | — |
| 20 | `rmsnorm` (post-attn) | 1 TG × 256 thr | — | — |
| 21 | `vec_add` (residual) | hidden | — | — |
| 22 | `rmsnorm` (pre-ff) | 1 TG × 256 thr | — | — |
| 23 | `matvec_q4_dual_ggml` (gate+up) | inter × hidden × 2 | — | — |
| 24 | `gelu_mul` | inter | — | — |
| 25 | `matvec_q4` (down) | hidden × inter | — | — |
| 26 | `rmsnorm` (post-ff) | 1 TG × 256 thr | — | — |
| 27 | `vec_add` (residual) | hidden | — | — |
| 28 | `matvec_q4` (PLE gate) | ple_dim × hidden | — | — |
| 29 | `gelu_mul_at` (PLE) | ple_dim | — | — |
| 30 | `matvec_q4` (PLE proj) | hidden × ple_dim | — | — |
| 31 | `rmsnorm` (post-PLE) | 1 TG × 256 thr | — | — |
| 32 | `vec_add` (PLE residual) | hidden | — | — |
| 33 | `vec_scale` (layer_scalar) | hidden | — | — |
| | **∑ Attention + MLP + PLE** | | **~1.35 ms** | **~40% + ~51%** |

### Per shared-KV layer (18 layers, has_kv=false) — ×18

Same as above minus kernels 9, 10, 13, 14, 15, 16, 17 = **26 kernels/layer**.

### Final (once per token)

| # | Kernel | Grid | Total time | % |
|---|--------|------|------------|---|
| 34 | `rmsnorm` (final) | 1 TG × 256 thr | — | — |
| 35 | `matvec_q4` (lm_head) | 262144×2560 | **~5 ms** | **~15%** |
| 36 | `sample_token` | 1 TG × 256 thr | — | — |
| | **∑ Head** | | **~5 ms** | **~15%** |

---

## Aggregated Times at 128-token Context

| Phase | ms/token | % | Kernel count | 
|-------|----------|---|-------------|
| Prep/PLE | ~3 | 9% | 5 |
| Attention (42 layers) | ~40 | 38% | 42 × ~8 = 336 |
| MLP+per-layer PLE (42 layers) | ~56 | 54% | 42 × ~7 = 294 |
| Head (lm_head) | ~5 | 5% | 3 |
| **Total** | **~104** | **100%** | **~638** |

The huge kernel count (638 dispatches/token) is the primary bottleneck. Each dispatch has fixed overhead:
- CPU encode: ~1-3 µs (set pipeline, bind buffers, dispatch)
- GPU scheduling: ~10-30 µs gap between kernels
- **Total dispatch overhead** at 638 × ~15 µs ≈ **9.6 ms** (9% of decode time)

---

## How Context Growth Affects Time

From benchmarks at different context lengths:

| Context | Total ms/token | Attention ms | % attention | MLP+Head ms (fixed) |
|---------|---------------|-------------|-------------|-------------------|
| 128 | 104 | 40 | 38% | 64 |
| 512 | ~330 | ~266 | 81% | 64 |
| 1024 | ~590 | ~526 | 89% | 64 |
| 2048 | ~1110 | ~1046 | 94% | 64 |
| 4096 | ~2150 | ~2086 | 97% | 64 |

Attention scales **linearly** with context. MLP and head are constant-cost.

---

## Bottleneck Summary

| Bottleneck | Impact | Evidence |
|------------|--------|----------|
| **Too many tiny kernels** | ~9 ms overhead | 638 dispatches per token at ~15 µs each |
| **Attention KV reads** | ~40 ms at 128 ctx, linear with length | Each token reads all cached K/V from global memory |
| **MLP weight bandwidth** | ~49 ms fixed | 4 matmuls (gate+up+down+ple) per layer × 42 layers |
| **CPU-GPU sync** | ~0.6 ms encode | `PROFILE_GPU=1` shows `encode=0.65ms` at n=128 |
| **Buffer writes (embed)** | 2 × `write_buffer` | CPU embedding gather writes to GPU buffer each token |

---

## Key Source Files

| File | Lines | Role |
|------|-------|------|
| `src/gemma4_gpu_model.rs` | 2394-3621 | `forward_single_token_inner` — full kernel launch sequence |
| `src/gemma4_gpu_model.rs` | 3623-3651 | Phase profiling accumulators |
| `src/gemma4_gpu_model.rs` | 3653-3683 | Per-token decode profiling |
| `src/gpu.rs` | 5121-5200 | `GpuTimestampProfiler` — GPU timestamp sampling |
| `src/gpu.rs` | 1499-1539 | Q4 matmul dispatch |
| `src/mega_decode.rs` | 454-666 | Mega graph op builder (shows all ops) |
| `perf.todo.md` | 1-140 | Performance roadmap with measurements |
| `benchmarks/bottleneck_results.txt` | 1-33 | GPU timestamp ablation results |
