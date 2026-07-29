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

### M5. h_nextn pre-final-norm "fix" — WRONG for Gemma4 (reverted)

The generic `llama-graph.h` comment says `t_h_nextn` is "hidden state before
final output norm", but **gemma4.cpp sets it AFTER `output_norm`** (post-final-
norm, the LM-head input) — matching transformers/vLLM/SGLang for this arch.
Tried switching all MTP h_nextn capture sites from `normed_buf` to
`hidden_buf`: accept rate collapsed 42.5% → 21.8%, tok/s 41 → 30. Reverted.
Our existing post-norm capture was already correct. Kept: last-row fix in
`forward_prefill_parallel_self` (was reading row 0 instead of the last row for
multi-token prefill).

### M6. Draft confidence normalized over top-k=10 (llama.cpp parity)

llama.cpp's draft-mtp sampler is `top_k=10`: `cur_p->data[0].p` (the p_min
gate) is the greedy token's softmax over the **top 10 candidates**, not the
full vocab. Ours was full-vocab softmax — systematically lower p, so p_min cut
drafts too early. Now `draft_token_confidence` normalizes over top-k
(`LLAMA_MTP_DRAFT_TOP_K`, default 10; 0 = full vocab).

Sweep (adaptive, auto, 442-tok essay): p_min 0.3 → 44.0% accept / 40.6 tok/s;
0.5 → 46.5% / 39.6–43.7; 0.75 → 50.5% / 43.4. Baseline greedy no-p_min:
42.5% / 41.1–43.5. Accept rises with p_min but tok/s stays within run-to-run
noise (±2) — the extra draft passes still roughly cancel the accept gain.
p_min remains opt-in.

### M7. Fused gate∥up+GeLU ext matvec for verify seq 2–8 (default on)

**What**: Ported the decode-style fused Q4_K gate∥up+GeLU pattern onto the
K-quant `mul_mv_ext` path (`matvec_ggml_ext_q4K_gelu_nx8_r{2..5}`). Each TG
dequants gate row i + up row i, shares activation tiles across the batch, and
writes `GeLU(gate·x)*(up·x)` directly — skips the 2·M·batch intermediate and
the separate `gelu_mul_stacked` dispatch. Wired in `encode_prefill_mlp_gate_up`
when `PREFILL_GATE_UP_EXT_GELU=1` (default), both weights Q4_K, seq ∈ [2,8].

**Result** (adaptive MTP, auto, 442-tok essay, 2 runs each):
- fused ON:  42.4 / 43.3 tok/s (accept 42.5%, identical draft path)
- fused OFF: 42.1 / 41.9 tok/s
- Essay coherent; accept rate unchanged (same greedy tokens).

**Conclusion**: Correct and ~0.5–1.5 tok/s — within run noise / tiny. Expected:
weight bandwidth for gate+up is unchanged (still read both matrices once); only
activation scratch + one gelu dispatch are saved. Does **not** close the
"batch-3 MLP ≈ 1.6× batch-1" gap — that cost is the three weight streams
(gate/up/down) plus occupancy, not the gelu glue. Keep default-on as a clean
path; next verify MLP lever needs a different angle (down-proj / occupancy /
phase timing).

### Remaining gap to >45 tok/s

Verify seq=3 is ~36 ms vs ~22 ms single decode (1.6x for 3 rows). Ablation:
MLP ≈ 12 ms of it (batched ext matvec already; gate∥up + down at batch 3 cost
~1.6x batch-1 despite weight reuse — bandwidth model says should be ~1.1x).
Acceptance is the structural limit: at 42% accept and 1.85 tok/forward, even
free batching caps at ~1.85× per-forward cost. M7 fused gelu was a wash for
e2e. Next levers: draft head quality (accept ~42% → 60%+), or deeper verify
phase timing to find where the 1.6× MLP tax actually lives.

---

## E24. Q2_R32 residual/additive weights (2026-07-28) — kernel loses to Q4

**What**: Implemented a true two-stage binary residual/additive format:

`w[i] ≈ d0·sign0[i] + d1·sign1[i]`

The first plane quantizes the weight and the second starts from its residual;
four Lloyd/least-squares refinement iterations optimize assignments and scales.
Each 32-weight block stores two f16 scales plus two 32-bit sign planes:
12 bytes / 32 weights = exactly **3.00 bpw**. Added CPU quant/dequant, Metal
batch-1/batched matvec, fused gate+up+GeLU, correctness tests, a real-tensor
quality probe, and `--bench-residual-matvec`.

**Kernel result** (E2B shapes, M1 Pro, lower ms is better):

| Shape | Q2_R32 | Q3_0 | Q4_0 |
|---|---:|---:|---:|
| 1536×1536 q/o | 0.047 | 0.032 | **0.023** |
| 8192×1536 gate/up | 0.077 | 0.095 | **0.063** |
| 1536×8192 down | 0.073 | 0.092 | **0.059** |
| 262144×1536 lm_head | 1.900 | 2.483 | **1.471** |

Despite 33% fewer bytes than Q4_0, two binary-plane dot products cost more than
the highly optimized nibble decoder. Q2_R32 is 22–104% slower than Q4_0 on the
representative shapes. Vectorized sign decoding and shared activation loads
improved it over Q3_0 for large matrices but did not beat Q4_0.

**Quality result** (requantizing `blk.0.ffn_gate.weight` from the available
Q4_K GGUF, so these are additional errors):

| Format | bpw | relative MSE |
|---|---:|---:|
| Q2_R32 | 3.00 | 0.118267 |
| Q3_0 | 3.50 | 0.058685 |
| Q4_0 | 4.50 | 0.010943 |

**Conclusion**: Do not route the model to Q2_R32. The kernel-level gate fails
before end-to-end integration, and quality is poor on top of Q4_K. A viable
sub-4-bit speed path on M1 Pro needs a hardware-friendly vector/codebook lookup
or integer-dot design that beats the Q4 nibble kernel in isolation, plus
quantization from BF16/F16 rather than an already quantized GGUF.

---

## E25. Adaptive GGML attention MWG (2026-07-29) — small decode win

**What**: Compiled `flash_attn_ext_vec` main/reduce pairs for NWG
4/8/16/32 at head dimensions 128/256/512. Dispatch now selects NWG from both
active KV length and query-head count:

`nwg = clamp_pow2(max(ceil(kv_seq/32), ceil(128/num_heads)), 4, 32)`

The 128-total-workgroup occupancy floor is important on M1 Pro. For Gemma4's
eight query heads this selects NWG=16 through 512 KV tokens, then NWG=32.
`ATTENTION_GGML_NWG=4|8|16|32` forces a variant for benchmarking. Scratch
remains sized for the maximum NWG=32.

**Correctness fix**: llama.cpp's reduce kernel assumed NWG=simd width=32.
For smaller NWGs, lanes `iwg >= NWG` previously indexed outside the partial
S/M and output arrays. Inactive lanes now contribute the online-softmax
identity (`S=0`, `M=-inf`, output=0), allowing the same 32-lane simd reduction
for all variants.

**Result** (E2B Q4_K_M, Q4_0 KV, M1 Pro):

| Mode | 200-token generation | 400-token generation |
|---|---:|---:|
| auto hybrid, forced NWG=32 (old behavior) | 44.5 tok/s | 44.0 tok/s |
| auto hybrid, NWG=16 (adaptive choice ≤512) | **45.0 tok/s** | **44.4 tok/s** |

In forced-GGML isolation at 200 tokens, NWG 4/8/16/32 measured
41.8/43.8/**45.8**/43.4 tok/s, confirming that workgroup occupancy matters
more than merely assigning one 32-token KV chunk per workgroup. Gains in the
production hybrid are modest (~0.4–0.5 tok/s) because fused attention still
handles KV <128 and attention is only part of decode time.

**Verification**: all Metal variants compile; the policy unit test passes; a
436-token adaptive essay remained coherent and completed at 45.60 tok/s.

**Conclusion**: Keep adaptive MWG as the default GGML leg. It removes the
fixed-NWG=32 oversubscription penalty but does not close the remaining
llama.cpp gap by itself.

---

## E26. MTP host offload + ANE measurement (`MTP_BACKEND=ane`) (2026-07-29)

**What**: Move the Gemma4 MTP draft trunk off the full-Metal path
(`MTP_BACKEND=ane|cpu|host`, default `metal`): pre_proj / norms / Q / O / FFN on
host Accelerate `sgemv` (AMX). The 262k lm_head + post_proj + argmax stay on
Metal in one command buffer.

**ANE verdict — the Neural Engine is slower here, measured, not assumed.**
`tools/ane_mtp_probe.py` builds the whole draft step (pre_proj → 4 layers
*including* attention over the target KV → output_norm → post_proj, i.e. one
prediction per draft step) and times CoreML fp16:

| kv_len | CPU_AND_NE | CPU_ONLY | ALL (GPU+ANE) |
|---|---:|---:|---:|
| 192 | 1.97 ms | **1.22 ms** | 3.56 ms |
| 512 | 1.97 ms | **1.30 ms** | 3.19 ms |
| 2048 | 3.23 ms | **2.01 ms** | 4.25 ms |

ANE loses at every context length. The draft is batch-1 with hidden 256 — memory
bound, far below ANE's arithmetic-intensity sweet spot, and it pays per-inference
dispatch. So there is no CoreML wiring: it cannot win, and the Metal fused draft
step is ~2.6 ms *including* lm_head.

**The real regression was GPU syncs, not the trunk math.** The first version
dispatched Metal attention per layer (write Q → encode → wait → read araw), so a
draft step went from 1 sync to ~5 → 36.8 tok/s. Fix: the target's KV is frozen
during a draft chain (only verify appends), so `KvMirror` snapshots the needed
rows to host f32 **once per chain**, dequantizing only positions appended since
the last snapshot. Metal buffers are `StorageModeShared`, so this is a
memcpy+dequant off unified memory — no command buffer, no fence. Attention then
runs on Accelerate (`K·q`, softmax, `Vᵀ·p`). Rewound rows are dropped and
re-read. `MTP_HOST_KV_MAX` (default 32768) falls back to Metal attention beyond
that KV length to bound mirror memory.

**Result** (E2B Q4_K_M + F16 MTP, bubble_sort, 160 tok, all 110/200 accepted =
55.0%, byte-identical output):

| Config | tok/s |
|---|---:|
| metal (default), 3 runs | 49.8 / 52.4 / 51.8 |
| ane/host + KV mirror, 3 runs | 50.6 / 50.5 / 39.0 |
| ane/host, `MTP_HOST_KV_MAX=0` (per-layer Metal attn) | 38.3 |
| ane/host, first version (no mirror) | 36.8 |

The `MTP_HOST_KV_MAX=0` row is the clean A/B: same code, mirror disabled, and
~12 tok/s evaporates. Draft wall share went 30% → 23%, matching metal.

**Measurement warning**: this machine drifts hard. During this session load
average was 6.5 with WindowServer at 38% (GPU contention), Cursor, Chrome and an
EDR agent live; identical metal binaries measured 55.8, then 49–52, then 41–49.
Single runs are worthless here. `benchmarks/mtp_metal_sweep.sh` interleaves
configs round-robin and scores best-of-N; use it instead of consecutive blocks.

**Conclusion**: The mirror brings host offload to rough parity with the fused
Metal draft in good runs, but it is *less stable* (one run at 39.0) because the
trunk now competes for CPU with the rest of the process, whereas the Metal draft
rides the GPU. Keep `metal` as default; `ane/host` is a correct opt-in path.
ANE is closed as a speed lever for MTP draft — the probe, not intuition, settles
it. Remaining host-side headroom is small: weights are dequantized to F32 for
`sgemv` (2× the bandwidth of the Metal F16 path), so the trunk floor is
~0.6–1 ms plus ~1 ms of lm_head + sync.

---

## E27. MTP draft depth 4 → 3 (2026-07-29) — +5 tok/s, one env var

**What**: The default `LLAMA_MTP_DRAFT_STEPS=4` is too deep for this workload.
Swept depth / adaptive / p_min on the Metal draft path, interleaved best-of-N
(`benchmarks/mtp_metal_sweep.sh`, E2B Q4_K_M + F16 MTP, bubble_sort 160 tok).

| Config | best tok/s | all runs | tok/fwd | fwds |
|---|---:|---|---:|---:|
| steps=4 (default) | 54.67 | 48.1 / 54.7 / 52.0 / 53.8 | 3.20 | 50 |
| **steps=3** | **59.62** | 56.4 / 57.0 / 59.6 / 58.8 | 2.91 | 55 |
| steps=2 | 57.96 | 57.2 / 57.2 / 58.0 / 57.6 | 2.42 | 66 |
| steps=4 + adaptive | 58.99 | 54.5 / 55.9 / 59.0 / 59.0 | 2.42 | 66 |

Fixed depths 5/6/7 were all *worse* than 4 (35–45 tok/s), and `p_min` 0.5/0.75
landed at/below baseline.

Confirmed twice more. Second interleaved sweep (contended machine, WindowServer
42%): steps=3 ran 55.0 / 57.2 / 55.2 / 58.5 vs steps=4 at 48.1 / 52.7 / 50.3 /
51.9 — *every* depth-3 run beat *every* depth-4 run, so the effect survives the
noise. On a quiet machine with warm page cache, depth 3 reached **61.4 tok/s**
(53.8 / 55.9 / 61.1 / 61.4 as cache warmed) against the 55.8 previously seen at
depth 4. Host backend under the same setting stayed below metal (52.9–57.8).

**Why deeper loses**: a rejected draft costs a full draft GPU pass *and* a verify
row. At ~2.2 accepted/cycle, depth 4 wastes ~45% of its draft passes. Depth 3
gives up a little tok/forward (2.91 vs 3.20) but cuts wasted draft passes and
shrinks the verify batch, and 55 cheap forwards beat 50 expensive ones.

`LLAMA_MTP_ADAPTIVE=1` finds nearly the same operating point on its own — its
heuristic clamps tails to ≤2 once the 12-cycle accept average sits under 3.0,
which is why steps=4/6/7 adaptive all converge to 2.42 tok/fwd and 66 forwards.
It is the safer choice when accept rate varies; fixed depth 3 was slightly faster
and more consistent here.

**Not contradicting M6**: that sweep (2/3/4/6/7 flat at 42–43.8) ran a 399-token
essay at ~42% accept. This prompt accepts 55%, which moves the optimum. Depth is
workload-dependent — sweep it per workload rather than trusting the default.

**Conclusion**: Default 4 leaves ~5 tok/s on the table for short code prompts.
Use `LLAMA_MTP_DRAFT_STEPS=3` (or `LLAMA_MTP_ADAPTIVE=1`). Not yet flipped as the
built-in default — needs validation across more prompt types first.
**Superseded by E28**: once extra verify rows became nearly free, depth 4 (the
existing default) became the optimum again.

---

## E28. Narrow-N simdgroup matmul for MTP verify (2026-07-29) — +18% e2e

**Tooling first**: added `--bench-verify` (prefill a fixed context, then time
`forward_verify_batch` at each batch size, rewinding the KV after each call so
every sample runs at the same KV length). Deterministic to ~±1%, and the
`PROFILE_ABLATE` buckets work on it — this replaced noisy whole-generation A/B
testing. `--bench-mv-ext` shapes moved from gemma-4-12b to E2B and gained `mm`
and `narrow` columns. The `mul_mm` accuracy check in `--gguf-kquant-test` was
reporting FAIL on a correct kernel: per-element relative error with a `1e-3`
floor is meaningless for half-MMA paths, so it and the new narrow check now use
relative L2.

**Diagnosis** (E2B Q4_K_M, ctx 512): verify at batch 1/2/3/4/6/8 =
22.0/34.0/37.9/43.2/59.8/69.7 ms. Ablation at batch 4: MLP 20.6 ms (gate∥up
11.5, down 9.4), attention 13.5, floor 5.9, PLE/head ~1. About half the ~7 ms
marginal cost per draft row is MLP, and it matched the kernel microbench exactly
(gate∥up ext matvec marginal 0.032 ms/row × 35 layers = 1.12 ms/row).

Root cause: `matvec_ggml_ext_q4x4` reads each weight row once per batch but
evaluates the batch with **scalar dot products**. Measured as GB/s of weight
traffic (flat = perfect amortization) it falls 56→20 from batch 2 to 8. The
marginal row runs at ~50% of fp32 FMA peak — near optimal *for scalar code*, so
the fix had to change the instruction mix, not the memory layout. Note that
activation traffic is `m·k·r1ptg·4` bytes regardless of `nxpsg`/`nypsg`, so tile
geometry cannot help; `MV_EXT_NSG` 1/2/4/8 was a wash, ruling out occupancy.

The wide prefill `mul_mm` is perfectly flat in batch but always computes a
32-column tile, wasting 4–8× the work at N≤8 — which is why `MUL_MM_MIN_SEQ=1`
was catastrophic in M2. It does show the matrix units are ~5.6× more
FLOP-efficient per output column than the scalar dots.

**Kernel**: `mul_mm_narrow_{q4_K,q6_K}_f32` — 32 weight rows × 8 batch rows,
NK=32. The first attempt used one simdgroup per threadgroup (to keep ceil(m/32)
threadgroups for the 1536-row down/o_proj shapes) and only reached ~30 GB/s:
with a single simdgroup the `sa` staging barriers have no other warp to hide
behind. **The fix was to split K rather than M across 4 simdgroups**, each with
a private slice of the staging tiles, so the K loop needs only
`simdgroup_barrier` and the four partial tiles are summed once at the end. That
lifted gate∥up 30→41, down Q4_K 15→33 and lm_head 45→57 GB/s, all flat in batch.

**Routing** (`MUL_MM_NARROW=0` to disable, `MUL_MM_NARROW_MIN_SEQ` default 3):
Q4_K only, batch 3–8. Q6_K keeps the ext matvec — `mul_vec_q6_K` folds scales in
at the end instead of fully dequantizing and stays ahead (44 vs 33 GB/s at batch
4). Batch 2 also keeps the ext matvec (48 vs 42). The fused gate∥up+GeLU ext
kernel (M7) now yields to narrow + a separate gelu dispatch above the threshold.

**Result** (bubble_sort, E2B Q4_K_M + F16 MTP, best of 2 interleaved runs):

| draft steps | verify batch | narrow off | narrow on |
|---|---|---:|---:|
| 3 | 4 | 58.0 | 63.0 |
| 4 | 5 | 53.9 | **63.7** |
| 5 | 6 | 45.3 | 60.6 |
| 6 | 7 | 42.1 | 58.9 |

Verify is now nearly flat from batch 3 to 6 (35.2/36.3/35.2/37.3 ms, was
33.1/38.3/45.7/54.6), so extra draft rows are close to free and the throughput
peak moved from steps=3 back to the existing **default steps=4**. Verify fell
from 83% to 74% of wall time; draft is the next target at 26%.

**Verification**: `--gguf-kquant-test` narrow rel_l2 1.5e-3 (Q4_K) / 2.9e-4
(Q6_K) at every batch 1–8, matching the existing prefill `mul_mm`, and covering
partial M tiles (m=256) and partial N tiles. `MTP_VERIFY_CROSSCHECK=1` reports
all rows matching the fused-sequential reference on every cycle. Generated
bubble_sort is correct and accept rate is unchanged at 54.3%. Prompt prefill is
unaffected (480→486 and 653→660 tok/s at 147/537 tokens).

**Conclusion**: The batch dimension belongs on the matrix units; the batch
*threshold* is just where a kernel stops being worth its threadgroup staging.
Keep narrow default-on for Q4_K at batch ≥3.

### E28 follow-up — depth across prompt types (2026-07-29)

`--prompt` presets + `benchmarks/mtp_prompt_sweep.sh` (2 interleaved reps,
narrow on). Best-of-2:

| Prompt | accept@4 | best config | best tok/s | steps=3 | steps=4 |
|---|---:|---|---:|---:|---:|
| bubble_sort (code) | 54% | **steps=4** | 67.3 | 66.3 | 67.3 |
| fibonacci (code) | 64% | **steps=4** | 69.8 | 69.1 | 69.8 |
| essay (prose) | 26% | **adaptive** | 44.1 | 42.0 | 39.1 |
| explain (prose) | 30% | **adaptive** | 45.5 | 44.4 | 42.0 |
| qa (factual) | 39% | **adaptive** | 52.0 | 51.2 | 51.6 |
| json (structured) | 42% | steps=3† | 59.9 | 59.9 | 56.5 |

† json finishes in ~40 tokens — too short for a clean depth call.

**Pattern**: after E28, fixed depth 4 wins on high-accept code (≥54%). On
low-accept prose (≤30%) fixed 4 wastes draft passes and loses to both fixed 3
and adaptive; adaptive is the clear winner because it clamps tails once the
12-cycle accept average drops. Mid-accept (~39%) is a wash between 3/4/adaptive.

**Recommendation**: keep built-in default `LLAMA_MTP_DRAFT_STEPS=4` (correct for
code after narrow verify). Prefer `LLAMA_MTP_ADAPTIVE=1` when the workload's
accept rate is unknown or known-low (essay/chat). Do not flip the global
default to 3 — that only helps the low-accept regime and costs ~0–1 tok/s on
code.
