# Performance TODO — surpass llama.cpp on Gemma4 E4B Q4_0

> Target: beat llama.cpp decode throughput (~40 tok/s on this machine).
> Current best known config: `MLP_GATE_UP_GGML=1 FUSED_QKV=0` → ~33.7 tok/s.
> Gap: ~6 tok/s (~18%).

## Baseline

- **Model:** `google/gemma-4-E4B-it` Q4_0 split cache
- **GPU:** Apple M1 Pro
- **Benchmark:** `./target/release/llama-sinks --gpu <model> --bench-decode --bench-decode-tokens 200`
- **Best result:** 33.7 tok/s generation (context ~248 tokens)
- **llama.cpp reference:** ~40 tok/s

## Where the time goes

From `PROFILE_PHASES=1` at 128-token context:

| Phase | ms/token |
|---|---|
| prepass (embed/ple setup) | ~3 |
| attention | ~40 |
| mlp_ple (MLP + per-layer embeddings) | ~56 |
| head (lm_head) | ~5 |
| **total** | **~104** |

Ablations:
- Skipping PLE saves ~7 ms.
- Skipping MLP saves ~32 ms → MLP itself is ~49 ms.
- Attention is the second-largest block and grows with context.

## Done

- [x] Fused gate+up+GeLU ggml Q4_0 kernel (parity with old path, removes scratch/dispatch).
- [x] Experimental SIMD-reduction GQA attention kernel (`ATTENTION_GQA_F16=1`) — currently loses to the tiled flash-decode kernel, so default-off.
- [x] Identified `FUSED_QKV=0` as a quick win (+0.5–0.8 tok/s).
- [x] Tiled GQA-aware flash-decode for q4_0 KV (`attention_flash_decode_q4_0_gqa`, enabled via `ATTENTION_GQA_Q4=1`). One threadgroup per KV head processes all query heads in the group; quantized K and V tiles are loaded into threadgroup memory once per tile and reused, cutting KV cache device reads by ~4×. Observed +1.0–1.2 tok/s (~3.5–5%) on decode benchmarks (248 tokens: ~31.8 → ~32.9 tok/s; 1000 tokens: ~25.4 → ~26.6 tok/s). Default-off pending broader validation.

## TODO (ranked)

### 1. Full MLP block fusion — HIGH
**Why:** MLP is the biggest single block (~49 ms). The current ggml path uses separate dispatches for pre-FF RMSNorm, gate+up+GeLU, and down. Fusing the whole block eliminates `normed_buf`/`gelu_buf` round-trips and dispatch bubbles.

**Files:** `src/shaders/ggml_mul_mv_q4.metal` or new shader, `src/gpu.rs`, `src/gemma4_gpu_model.rs`

**Notes:**
- The existing packed `FUSED_MLP_GELU_DOWN=1` path already fuses gate∥up+GeLU+down but is slower than the separate-ggml path.
- Goal: a ggml-Q4_0 kernel using **separate** gate/up weights that streams gate → up → GeLU → down in one dispatch.
- Down projection needs the full gelu vector; use a tile/reduction design or two-phase kernel (gelu tile + partial down sums).
- Can be staged: first fuse rmsnorm+gate+up+GeLU, then add down.

**Expected gain:** +1–2 tok/s.

---

### 2. PLE optimization — MEDIUM
**Why:** Per-layer embeddings cost ~7 ms at 128-token context (~12% of mlp_ple phase).

**Files:** `src/shaders/llama.metal`, `src/gemma4_gpu_model.rs`

**Notes:**
- Current PLE does a gather + context projection per layer.
- Options: fuse PLE gather/context-proj into the MLP dispatch, or store PLE table in a tiled layout with better cache locality.

**Expected gain:** +0.5–1.5 tok/s.

---

### 3. Metal System Trace profiling — MEDIUM
**Why:** End-to-end effective bandwidth (~80 GB/s) is well below isolated matvec bandwidth (~160 GB/s). Idle bubbles between dispatches/CBs are the suspected culprit.

**Files:** N/A ( Instruments / `xctrace` )

**Notes:**
- Capture a trace of the best known config.
- Look for gaps where the GPU is idle between kernels.
- Identify cheap fusion candidates (e.g., residual adds, small elementwise kernels).

**Expected gain:** unknown; likely small but cheap wins.

---

### 4. Command-buffer scheduling — MEDIUM
**Why:** Current default is `METAL_N_CB=2`. More granular splits or a single CB per token might reduce wait overhead.

**Files:** `src/gemma4_gpu_model.rs`

**Notes:**
- `METAL_N_CB=3/4` showed no gain in quick tests, but a full single-CB-per-token or a ring of CBs across tokens could help.
- Risk: CPU encode must stay ahead of GPU execution.

**Expected gain:** +0.5–1.5 tok/s.

---

### 5. Weight format upgrade — LOW/MEDIUM
**Why:** MLP weights are the dominant bandwidth. Q4_0 uses ~0.56 byte/weight. A block layout with larger blocks (e.g., Q4_K-style) could cut bytes by 10–15%.

**Files:** quant cache path, shaders

**Notes:**
- Requires new dequant kernels and accuracy validation.
- Large implementation cost; do after fusion wins are exhausted.

**Expected gain:** +1–2 tok/s if feasible.

---

### 6. Speculative decoding — LONG-TERM
**Why:** Biggest theoretical multiplier for per-request throughput.

**Files:** new module, sampling path

**Notes:**
- Needs a small draft model and acceptance sampling.
- Out of scope until single-token decode is within striking distance of llama.cpp.

## Quick env reference

```bash
# Current best known config
MLP_GATE_UP_GGML=1 FUSED_QKV=0 \
  ./target/release/llama-sinks --gpu <model> --bench-decode --bench-decode-tokens 200

# Profile phases (slows decode, use for diagnosis)
PROFILE_PHASES=1 MLP_GATE_UP_GGML=1 FUSED_QKV=0 \
  ./target/release/llama-sinks --gpu <model> --bench-decode --bench-decode-tokens 128

# Tiled GQA-aware q4_0 flash-decode (small but measurable win)
ATTENTION_GQA_Q4=1 MLP_GATE_UP_GGML=1 FUSED_QKV=0 \
  ./target/release/llama-sinks --gpu <model> --bench-decode --bench-decode-tokens 200

# Experimental GQA SIMD kernel (currently slower)
ATTENTION_GQA_F16=1 MLP_GATE_UP_GGML=1 FUSED_QKV=0 \
  ./target/release/llama-sinks --gpu <model> --bench-decode --bench-decode-tokens 200
```

## Next action

Start with **#1 (full MLP block fusion)** — MLP is still the largest single phase and the tiled GQA attention win is now in place.
