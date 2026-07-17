# Failed Experiments Log

Goal: Close the ~7 tok/s gap between our Metal GPU inference and llama.cpp on
Gemma4 E4B Q4_K_M (M1 Pro). Llama.cpp gets 53.9 tok/s flat regardless of context
length. We match at 25 tok (54.7) but degrade to 47.9 at 200 tok (−12%).

## 1. Dispatch Overhead

**What**: Counted dispatch calls (455/token, ~5.2 μs each = 2.38 ms total
overhead per token). Measured GPU compute time at ~18 ms.

**Result**: 2.38 ms overhead is real but not the gap — at 200 tok the gap is
2.6 ms vs our 25-tok baseline. Dispatch overhead is constant; the gap grows
with context, meaning the attention kernel itself scales worse than llama.cpp's.

**Conclusion**: Not the bottleneck.

---

## 2. `fastMathEnabled=true`

**What**: Set Metal pipeline `fastMathEnabled` to true.

**Result**: Zero performance change. Kernel is bandwidth-bound, not compute-bound.

**Conclusion**: Dead end.

---

## 3. `KQ_NR0=2` (matvec tile geometry)

**What**: Changed matvec tile height from 4 to 2 (fewer output rows per
threadgroup).

**Result**: Slower (42.8 vs 46.1 tok/s). Reverted.

**Conclusion**: 4 is optimal for bandwidth-bound matvec on M1 Pro.

---

## 4. `FUSED_DECODE=0` (disable mega-kernel)

**What**: Disabled the fused decode mega-kernel, running individual ops instead.

**Result**: Same ~47.x tok/s. The bottleneck exists in both paths.

**Conclusion**: Not a mega-kernel issue.

---

## 5. V accumulation stride fix (simdgroup-partitioned access)

**What**: Fixed V-value write-back in the flash attention kernel so each
simdgroup writes to its own partition of threadgroup memory (non-overlapping
float4 lanes).

**Result**: ~47.1 tok/s — within noise of baseline. The stride fix was
correctness-preserving but didn't change performance.

**Conclusion**: Not a shared-memory bank-conflict issue.

---

## 6. Non-flash single-pass attention (tile-free for kv_seq ≤ 256)

**What**: Bypassed the flash-attention per-tile softmax. Loaded all K,V into
shared memory at once, did one large softmax over the full kv_seq.

**Result**: ~47.3 tok/s — same as flash. At 200 tok the tile overhead is
negligible.

**Conclusion**: Flash vs non-flash isn't the differentiator.

---

## 7. `ATTENTION_KERNEL=ggml` (32-thread vec kernel, NWG=1, NSG=1)

**What**: Used the reference llama.cpp vec kernel directly (1 simdgroup, 32
threads per WG, Q into half shared memory, dequantize-on-the-fly).

**Result**: Same throughput as the 256-thread fused flash kernel (~47.x tok/s).
Thread utilization is NOT the bottleneck — both 32-thread and 256-thread
kernels give identical results.

**Conclusion**: Not about thread count or simdgroup count.

---

## 8. `ATTENTION_KERNEL=mwg` (multi-WG vec kernel, NWG=32)

**What**: Implemented llama.cpp's `kernel_flash_attn_ext_vec` pattern with
NWG=32 workgroups (each 32 threads), partitioning the KV cache across groups,
then a reduce kernel to combine. Ported `flash_attn_ext_vec_multi_wg_impl` and
`flash_attn_ext_vec_reduce_impl` from reference.

**Details**:
- 32 workgroups × 32 threads = 1024 threads total per head
- Each WG handles 1/32 of KV cache (C=32 tokens at a time)
- `C × NWG = 1024` tokens per iteration across all WGs
- Small shared memory (~1.8 KB vs ~2.8 KB for vec, ~5 KB for fused)
- Temp buffer: `nrows × DV × NWG + S/M` (~1 MB for E4B)

**Result**: **Worse** — 40.38 tok/s at 441 tok context (vs ~46 estimated for
existing kernel). Root cause: at kv_seq=441, each WG processes only
`ceil(441/1024) = 1` loop iteration. 32 separate WG dispatches × overhead >
benefit. MWG only helps when `kv_seq >> NWG × C ≈ 1024` (e.g., 4k+ contexts).

**Conclusion**: NWG=32 is wrong for our test range (25–441 tok). Would only
help at very long contexts.

---

## 9. MWG scratch buffer bug (`DV4` vs `DV`)

**What**: While implementing MWG, used `DV4` (head_dim/4 = 64) instead of `DV`
(head_dim = 256) for the S/M data offset in the temp buffer.

**Result**: S/M would overlap with O data — garbage output. Fixed during
development, never benchmarked in broken state.

**Conclusion**: Fixed as part of MWG implementation; not a performance factor.

---

## 10. MWG reduce kernel `kernel void` vs `void`

**What**: Declared `flash_attn_ext_vec_reduce_impl` as `kernel void` template,
but entry points call it directly — Metal doesn't allow calling `kernel
functions from other kernel functions.

**Result**: Metal compilation error at runtime (`call to kernel function
flash_attn_ext_vec_reduce_impl`). Fixed by changing to `void`.

**Conclusion**: Fixed during development; not a performance factor.

---

## 11. GQA tiled attention on production fused path (default-on)

**What**: Default `ATTENTION_GQA_Q4` on; wired tiled GQA kernel into
`decode_fused` primary path (was dead code behind `full_fused` only).
Dispatches 2 threadgroups/KV head instead of 8/Q head; loads KV tiles to
threadgroup memory once per tile.

**Result**: **Correctness regression** on E2B Q4_K_M — garbage output
(`くださいまして` loop) at 59.6 tok/s. Root cause: decomposed GQA path
(`encode_fused_attn_q4_gqa_has_kv`) replaced `full_fused` on KV-owning
layers; it appends KV to Q4 cache *before* attention instead of attending
with f32 K/V for `cur_seq` then appending (as `full_fused` does). Reverted:
GQA opt-in (`ATTENTION_GQA_Q4=1`) and **shared-KV layers only**; KV-owning
layers keep `encode_attention_full_fused_q4_0`.

**Follow-up**: Shared-layer GQA still garbage with `ATTENTION_GQA_Q4=1` —
root cause was **splitting QK-norm+RoPE from attention**. Separate
`rmsnorm` + `apply_rotary` + `attention_flash_decode_q4_0_gqa` (even with
partitioned `shared_exp` and device KV reads) does not match the fused
`attention_flash_decode_qknorm_rope_q4_0` kernel. Fix: new fused kernel
`attention_flash_decode_qknorm_rope_q4_0_gqa_{h128,h256,h512}` — same
`flash_load_q_qknorm_rope_hd` + flash attention as the working per-head
kernel, but one threadgroup per KV head (4 query heads share KV reads).
Pending re-benchmark.

---

## 12. llama.cpp shared-KV anchor layers + cache row_bytes

**What**: Fixed `kv_source_layer` for shared layers 24–41 to match llama.cpp
`n_layer_kv_from_start - (is_swa ? 2 : 1)` (anchors 22/23 for E4B, not
same-type scan). Attention cache reads use anchor layer `row_bytes`.

**Result**: Pending benchmark + correctness check on shared full layers.

---

## 13. llama.cpp `flash_attn_ext_vec` MWG (ggml path, `ATTENTION_KERNEL=ggml`)

**What**: Ported llama.cpp's multi-WG vec attention (`NWG=32` + temp buffer +
reduce kernel) into `ggml_flash_attn.metal` / `ggml_flash_attn.rs`. Wired into
`decode_fused` and legacy decode via `encode_attention_ggml_q4_0`. All layers
use ggml MWG for every decode token when `ATTENTION_KERNEL=ggml`.

**Result**: Flat ~49 tok/s regardless of context length — no degradation vs
fused baseline at 200 tok, but also no short-context win. Slightly below
llama.cpp (53.9) and below fused at 25 tok (54.7).

**Conclusion**: ggml MWG is context-stable but not faster than fused flash at
short ctx. Useful as the long-context leg of a hybrid.

---

## 14. `ATTENTION_KERNEL=auto` hybrid (fused <128, ggml MWG ≥128)

**What**: Hybrid routing via `attention_use_ggml_for_layer_kv(has_kv, kv_seq)`:
fused `full_fused` / `qknorm_rope` below 128 KV tokens, decomposed norm/RoPE +
ggml MWG at/above 128. Applied to **all layers** (KV-owning and shared-KV).

**Result** (bench-decode, E4B Q4_K_M, Q4_0 KV):
- `specialized` (fused only): ~54.7 @ 25 tok gen, ~47.9 @ 200 tok gen
- `ggml` (always): ~48.9 @ 25, ~49.1 @ 200
- `auto` v1 (shared layers always ggml): ~50.7 @ 25, ~49.1 @ 200
- `auto` v2 (all layers switch by kv_seq): **~53.5 @ 25, ~50.0 @ 200**

Best throughput profile so far: near-fused short ctx, +~2 tok/s long ctx vs
fused-only. Still ~4 tok/s below llama.cpp at 200+ tok.

**Conclusion**: Hybrid routing works for throughput; threshold tuning (64/256)
not yet explored.

---

## 15. Hybrid auto KV append correctness bug (fixed `cd8f7d9`)

**What**: Interactive essay generation with `ATTENTION_KERNEL=auto` produced
garbage after an initially coherent opening — repetitive "benefits a powerful
benefits…" then endless `###` blocks (~415 tok, 49.15 tok/s, ctx 464).

**Root cause**: At `kv_seq ≥ 128` the path switches from fused flash (KV
append inline) to decomposed + ggml MWG. `fused_kv_attention_enabled()` stays
`true` for `auto` mode, so the decomposed branch skipped explicit
`encode_kv_append` — ggml attention read the cache without the current token's
K/V.

**Fix**: `needs_explicit_kv_append(has_kv, effective_kv_seq)` — returns true
when ggml is active even if fused KV append is otherwise enabled. Applied in
`decode_fused.rs`, `gemma4_gpu_model.rs` (legacy + batch decode).

**Result**: Coherent 358-token essay at **48.52 tok/s** (ctx 407). Hybrid
routing + correctness both working.

**Conclusion**: Any future kernel switch must reconcile KV append semantics
(fused inline append vs explicit append before attention).

---

## Summary (updated)

| # | Experiment | Tok/s | vs llama.cpp (53.9) | Note |
|---|-----------|-------|---------------------|------|
| — | Baseline fused (25 tok) | 54.7 | +0.8 | Ties at short context |
| — | Baseline fused (200 tok) | 47.9 | −6.0 (12%) | Gap opens with context |
| 1 | Dispatch overhead | 47.x | −6.x | Constant ~2.4 ms |
| 2 | fastMathEnabled | 47.x | −6.x | No effect |
| 3 | KQ_NR0=2 | 42.8 | −11.1 | Actually worse |
| 4 | FUSED_DECODE=0 | 47.x | −6.x | No effect |
| 5 | V stride fix | 47.1 | −6.8 | Within noise |
| 6 | Non-flash (tile-free) | 47.3 | −6.6 | Same as flash |
| 7 | GGML vec (32-thread) | 47.x | −6.x | Same throughput |
| 8 | MWG old (llama.metal, NWG=32) | 40.38 | −13.5 | Worse; wrong for short ctx |
| 9 | DV4 bug | — | — | Fixed (correctness) |
| 10 | kernel void bug | — | — | Fixed (compilation) |
| 11 | GQA tiled (default-on) | 59.6 | +6.0 | Garbage output; reverted |
| 12 | Shared-KV anchor fix | — | — | Pending benchmark |
| 13 | ggml MWG always | ~49.1 | −4.8 | Flat; context-stable |
| 14 | auto hybrid v2 | 53.5 / 50.0 | −0.4 / −3.9 | Best profile; 25/200 tok |
| 15 | Hybrid KV append bug | 48.5 | −5.4 | Fixed; essay coherent |

Current best config: `ATTENTION_KERNEL=auto` + Q4_0 KV + fused decode executor.
~50 tok/s at 200+ tok context, ~4 tok/s below llama.cpp. Short-context peak
~53.5 tok/s (not sustained through long generation).

---

## Path to 60 tok/s — pending experiments

Target: ~60 tok/s sustained decode on E4B Q4_K_M (M1 Pro). Current ~50 tok/s
with hybrid auto leaves ~10 tok/s (~17 ms/token) to find. Likely not a single
kernel change — need phase timing to locate the gap.

### E16. Prefill phase timing @ 4k (done 2026-07-11) + decode scaling

Cool E2B Q4_K_M ablation (`PROFILE_ABLATE`, see `benchmarks/prefill_phase_4k.txt`):

| Bucket | Δms @4k | Share | Note |
|--------|---------|-------|------|
| MLP | 4488 | 54% | gate∥up 3119 (ex-gelu ~2650), gelu 467, down 1764 |
| Attn | 2930 | 35% | flash 2107, qkv 821, o 779 |
| PLE | 1555 | 19% | |
| Head | 402 | 5% | |
| CB/embed/rope | 135 | 1.6% | `SKIP_all` floor |
| f16 cast | ~−100 | wash | `SKIP_cast` / `PREFILL_MLP_F16=0` |

Gap vs llama ~585: ~1.24 s. Tile align / kvpad FC: **wash** (exact 4096 ≈ 4112).
Pad/mask / SWA-narrow: low ROI. Next prefill lever: MLP non-matmul + PLE.

Decode (separate): short-prompt bench 51/46/41 @25/200/400 gen; long-ctx chat
decode ~27 @0.8k / ~18 @3k / ~13 @6k → ~10 @31k matches logs.

### E22. Prefill MLP/PLE (2026-07-12) — PLE f32→Q4 was the gap

Gate∥up mul_mm already at peak (~2.64s theory @3.22 TFLOPS matches
`mlp_gate` ex-gelu). `PREFILL_MLP_GATE_F16_DST=1` **worse** (~−10%).

**Win:** PLE `inp_gate`/`proj` are **F32** on Q4_K_M but `qw()` requantized
them to Q4_0 → slow `projection_q4_batch`. Keep dense **f16** + `mul_mm_f16`
(same fix class as `per_layer_model_proj`). Delete `model.q4cache` after.

Result (exact 4096): PLE Δ **~1555→~230 ms**; prefill **~530–572 tok/s**
(was ~500–520). Correctness: `Hello.` + mid-SWA `ZEBRA42` OK. Gap to llama
585 now ~15–55 tok/s depending on thermal.

### E23. Prefill flash h256 NSG=8 (2026-07-12) — matches llama @4k

llama-bench pp4096 FA=1: **593.9 tok/s**. After PLE f16 baseline ~533, flash Δ
~2480 ms. Raised Metal/host NSG for h256 **4→8** (24 KB smem; h512 stays 4).
Cool exact-4096: **581–591 tok/s**, flash Δ ~1900 ms. Correctness: Hello /
ZEBRA42 / short needle OK. Remaining gap to llama is noise/thermal.

### E17. Hybrid threshold sweep (not started)

Sweep auto switch threshold: 64, 128 (current), 256. Measure tok/s at 200 and
400 tok generation; verify text quality at each threshold.

### E18. ggml vs specialized at 400–512 tok (not started)

Force `ATTENTION_KERNEL=ggml` vs `specialized` at long context to confirm which
path is structurally slower and whether hybrid should switch earlier or later.

### E19. KV layout / ggml MWG tiling for E4B head_dim (not started)

Microbench attention kernel only (no MLP/logits). Compare current ggml MWG vs
variants tuned for head_dim=256 and typical kv_seq 400–800 (tile size, prefetch,
loop order). Check row_bytes / group-of-32 packing matches llama.cpp.

### E20. MLP variant sweep (not started)

Toggle `MLP_GELU_F16`, `MLP_GATE_UP_GGML`, `FUSED_MLP_GELU_DOWN` at 200 tok
decode. MLP is ~half of per-layer work; 10–15% savings there ≈ +2–3 tok/s.

### E21. Command-buffer pipelining / micro-batching (not started)

Confirm one CB per token with no implicit device waits. Try batch-2/4 decode to
test occupancy vs dispatch overhead tradeoff.

Unresolved hypotheses (unchanged): KV-cache Q4_0 write bandwidth during decode,
pipeline bubbles between attention and MLP, K-norm/RoPE path differences vs
llama.cpp. Phase timing (E16) should narrow these.

---

## MTP (E2B Q4_K_M + F16 draft head) — verify path optimization (2026-07-17)

Goal: MTP ≥ non-MTP baseline (~44 tok/s, `ATTENTION_KERNEL=auto`, Q4_0 KV,
8192 ctx). Started at ~25 tok/s (sequential verify, 90% of wall in verify).

### M1. Parallel prefill verify now correct + default

Earlier garbage traced to two bugs (fixed prior session): draft-head attention
scratch sized to `hidden_head` instead of `max_head_dim` (512), and f16 MLP
cast feeding the f32 matvec fallback when `should_use_mul_mm` was false.
`MTP_VERIFY_CROSSCHECK=1` now passes on every cycle (parallel == sequential,
all rows). `forward_verify_parallel` (batched prefill chunk) is the default;
`MTP_VERIFY_SEQUENTIAL=1` / `MTP_VERIFY_DECODE_BATCH=1` opt back.

### M2. K-quant `mul_mv_ext` small-batch kernels (batch 2–8)

Ported llama.cpp `kernel_mul_mv_ext_q4x4_f32` (r1ptg=2..5, nxpsg=8) for
Q4_K/Q6_K into `ggml_mul_mv_q4.metal` (`matvec_ggml_ext_q{4,6}K_nx8_r{2..5}`).
Weight row dequantized once, dotted against all batch rows. Routed in
`encode_prefill_kquant_projection` + stacked gate/up for `2 ≤ seq ≤ 8`.
(`mul_mm` at these sizes is **worse**: `MUL_MM_MIN_SEQ=1` → 20 tok/s.)

### M3. Batched lm_head for verify rows

Verify computed logits per row with `encode_matvec_auto_at_view` — the
~440 MB vocab matrix was read once *per row*. Replaced with one
`encode_prefill_projection_auto_batch_view` over all rows. +1 tok/s.

### M4. Tiled flash_attn_ext for small q (default `TILED_EXT_MIN_Q=2`)

Biggest win. Per-row causal attention (one dispatch per q row ×
per-row KV reads) was the verify bottleneck. The tiled ext kernel already
handled small q fine — the `q_len ≥ 20` gate was just llama.cpp's vec/tiled
switch, but our sub-20 fallback is much worse than their vec path. Lowering
the gate to 2 shares KV tile loads across verify rows: seq=3 verify GPU
44 → 36 ms; e2e 37.7 → 42.4 tok/s. Crosscheck still passes.
(`MTP_VERIFY_DECODE_FA=1` per-row decode attention: 73 ms — far worse.)

### Results (399-token essay, adaptive draft, ~42% accept, 1.85 tok/forward)

| Config | tok/s |
|--------|-------|
| Non-MTP baseline (auto) | 43.5–44.5 |
| MTP sequential verify (old default) | 23.5–26 |
| MTP parallel verify + ext matvec | 34.8 |
| + batched lm_head | 37.7 |
| + tiled ext attention (new default) | **42.4** (auto) / **43.1** (specialized) |

Draft-steps sweep (2/3/4/6/7): flat 42–43.8; `p_min` 0.3/0.5 raises accept to
44–48% but lowers tok/s (draft passes cost more than they save). Verify cap
`MAX_MTP_VERIFY_SEQ=8` → max draft steps 7.

### Remaining gap to >45 tok/s

Verify seq=3 is ~36 ms vs ~22 ms single decode (1.6x for 3 rows). Ablation:
MLP ≈ 12 ms of it (batched ext matvec already; gate∥up + down at batch 3 cost
~1.6x batch-1 despite weight reuse — bandwidth model says should be ~1.1x).
Acceptance is the structural limit: at 42% accept and 1.85 tok/forward, even
free batching caps at ~1.85× per-forward cost. Next levers: draft head quality
(accept ~42% → 60%+), or shave verify MLP (fused gate∥up ext kernel for
batch 2–8, analogous to `matvec_ggml_q4_K_gelu_mul`).
