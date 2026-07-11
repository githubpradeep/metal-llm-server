# Performance Improvement Plan — Close the Gap and Go Beyond llama.cpp

**Baseline (M1 Pro, Gemma-4 E4B):**

| Model / path | You | llama.cpp | Gap |
|--------------|-----|-----------|-----|
| Q4_K_M (community GGUF) | **28.8 tok/s** | ~32 tok/s | −3.2 tok/s (~11%) |
| Q4_0 (split cache) | **33.7 tok/s** | ~34–40 tok/s | −0.5 to −6 tok/s |

K-quant kernels are **correct** (pass `--gguf-kquant-test`). Remaining work is **fusion, dispatch reduction, and orchestration** — not more matvec porting.

---

## North-star targets (realistic)

| Milestone | Target | Meaning |
|-----------|--------|---------|
| **M1** | **≥32 tok/s** | llama.cpp parity on Q4_K_M |
| **M2** | **≥38 tok/s** | Beat llama.cpp on Q4_0 |
| **M3** | **≥45 tok/s** | Strong E4B decode (~50–60% of roofline) |
| **M4** | **≥55 tok/s** | Near roofline for this model class |
| **M5** | **2× effective tok/s** | Speculative decoding (uzu-style) |

**100 tok/s** single-stream on E4B Q4_K_M is not a sensible near-term target. Treat **~55–60 tok/s** as the physics ceiling for this model on M1 Pro; **100** needs speculation, batching, or a smaller model.

Roofline reference (`docs/BLOG_GEMMA4.md`):
- Q4_0 E4B theoretical: ~80 tok/s (200 GB/s ÷ ~2.5 GB weights)
- Mixed Q4 + f16 layers: ~67 tok/s (200 GB/s ÷ ~3 GB)

---

## How to measure (every phase)

```bash
# Correctness gate (must pass after any kernel change)
cargo run --release -- --gguf-kquant-test ~/Downloads/gemma-4-E4B-it-Q4_K_M.gguf

# Primary benchmark (Q4_K_M)
LLAMA_KV_CACHE_TYPE=q4_0 cargo run --release -- --gpu ~/Downloads/gemma-4-E4B-it-Q4_K_M.gguf --bench-decode --bench-decode-tokens 128

# Q4_0 comparison
LLAMA_KV_CACHE_TYPE=q4_0 cargo run --release -- --gpu <Q4_0_model_or_cache> --bench-decode --bench-decode-tokens 200

# Phase breakdown
PROFILE_PHASES=1 LLAMA_KV_CACHE_TYPE=q4_0 cargo run --release -- --gpu ... --bench-decode --bench-decode-tokens 128

# llama.cpp reference
llama-bench -m ~/Downloads/gemma-4-E4B-it-Q4_K_M.gguf -p 48 -n 128 -ctk q4_0 -ctv q4_0 -fa 1
```

Always use `cargo run --release --` so Rust + shader changes stay in sync (stale `./target/release/llama-sinks` can show wrong results).

---

## Where time goes today (Q4_0, PROFILE_PHASES=1, 128 context)

| Phase | ms/token |
|-------|----------|
| prepass (embed/ple setup) | ~3 |
| attention | ~40 |
| mlp_ple (MLP + per-layer embeddings) | ~56 |
| head (lm_head) | ~5 |
| **total** | **~104** |

Ablations:
- Skipping PLE saves ~7 ms.
- Skipping MLP saves ~32 ms → MLP itself is ~49 ms.
- ~638–1056 kernel dispatches per token (`docs/08_bottleneck_analysis.md`).

---

## Phase 0: Instrument (1–2 days)

**Goal:** Know where the 34.7 ms/token goes on Q4_K_M, not just Q4_0.

| Task | Action |
|------|--------|
| Profile Q4_K_M | `PROFILE_PHASES=1` on Q4_K_M GGUF; compare to Q4_0 table above |
| Dispatch count | Add debug counter: kernels encoded per token (or Metal System Trace) |
| Metal trace | `xctrace` one 128-token bench; find idle gaps between kernels |
| Ablation matrix | `PROFILE_ABLATE=mlp\|attn\|ple` on Q4_K_M |

**Exit:** Table like “MLP 52%, attention 35%, PLE 8%, overhead 5%” for **Q4_K_M specifically**.

---

## Phase 1: llama parity on Q4_K_M (M1 → ≥32 tok/s)

**Gap to close:** ~3.2 tok/s (~3.5 ms/token).  
**Theme:** Enable existing fusion on the K-quant path; remove redundant dispatches.

### 1a. Fused QKV for K-quant (+1–1.5 tok/s)

Today `FUSED_QKV` is **Q4_0 only** (`src/gemma4_gpu_model.rs` ~3227).

| Work | Details |
|------|---------|
| Add `rmsnorm_qkv_q4_K` / `rmsnorm_qkv_q6_K` variants | One dispatch: norm + Q (Q4_K) + K/V (Q6_K where needed) |
| Wire in decode path | `layer.weight_format.is_kquant()` branch |
| Test | `--gguf-kquant-test` + generation sanity |

**Files:** `src/shaders/llama.metal`, `src/gpu.rs`, `src/gemma4_gpu_model.rs`

### 1b. Fuse pre-FF norm into Q4_K gelu_mul (+0.5–1 tok/s)

Today K-quant MLP = `rmsnorm` → `matvec_q4_K_gelu_mul` → `matvec_q6_K` (3 dispatches).

| Work | Details |
|------|---------|
| `rmsnorm_matvec_q4_K_gelu_mul` kernel | Fold RMSNorm into existing fused gelu kernel (read hidden once) |
| Keep Q6_K down separate | Still 2 dispatches/layer vs 3 |

**Files:** `src/shaders/ggml_mul_mv_q4.metal`, `src/gpu.rs`, `src/gemma4_gpu_model.rs`

### 1c. Turn on validated attention wins (+0.5–1 tok/s)

| Env | Status | Action |
|-----|--------|--------|
| `FUSED_Q_ATTN=1` | On for quant + Q4 KV | Verify on Q4_K_M bench |
| `FUSED_K_ATTN=1` | Partial | Ensure K-norm+RoPE+KV append fused for all KV layers |
| `ATTENTION_GQA_Q4=1` | +1 tok/s on Q4_0, default-off | Re-benchmark on Q4_K_M; enable if OK |

### 1d. Quick env sweep (+0–0.5 tok/s)

```bash
FUSED_QKV=1 FUSED_Q_ATTN=1 FUSED_K_ATTN=1 ATTENTION_GQA_Q4=1 \
  cargo run --release -- --gpu ~/Downloads/gemma-4-E4B-it-Q4_K_M.gguf --bench-decode --bench-decode-tokens 128
```

**Phase 1 exit:** ≥32 tok/s on Q4_K_M, all kquant tests OK, coherent generation.

---

## Phase 2: Beat llama.cpp on Q4_0 (M2 → ≥38 tok/s)

**Gap:** ~6 tok/s on the best-known path.

### 2a. Fix or replace `FUSED_MLP_GELU_DOWN` (+1–2 tok/s)

`perf.todo.md` notes the packed fused MLP path is **slower** than separate ggml — fix or delete.

| Approach | Description |
|----------|-------------|
| **A. Fix packed path** | Profile why `rmsnorm_mlp_gelu_down_q4_packed` loses; likely extra memory traffic |
| **B. uzu-style 3-kernel MLP** | `rmsnorm+gate∥up+gelu` (1) + `down` (1) with **interleaved weights** via `PACKED_MLP_GATE_UP` |
| **C. Two-phase down** | Tile gelu → partial down sums in threadgroup memory |

Default config target:

```bash
MLP_GATE_UP_GGML=1 PACKED_MLP_GATE_UP=1 FUSED_QKV=0 FUSED_RMSNORM_ACC=1
```

### 2b. Fuse residual + RMSNorm (+0.5–1 tok/s)

~210 trivial `rmsnorm` + ~84 `vec_add` dispatches per token.

| Kernel | Fuses |
|--------|-------|
| `rmsnorm_acc` | residual add + RMSNorm (enable everywhere) |
| `rmsnorm_acc_per_head` | Q/K norm + residual where safe |

**Target:** Cut norm/add dispatches by ~50%.

### 2c. Default-on GQA attention (+1 tok/s)

Ship `ATTENTION_GQA_Q4=1` after Q4_K_M + Q4_0 validation.

### 2d. PLE compression (+0.5–1.5 tok/s)

PLE costs ~7 ms/token at 128 context.

| Option | Effort |
|--------|--------|
| Fuse PLE gate+proj into post-MLP residual pass | Medium |
| GPU embedding lookup (remove CPU memcpy) | Medium |
| `FUSED_MLP_PLE` block | Higher |

**Phase 2 exit:** ≥38 tok/s Q4_0, ≥34 tok/s Q4_K_M.

---

## Phase 3: Structural wins — uzu-style orchestration (M3 → ≥45 tok/s)

**Theme:** Fewer, bigger units of work — not one mega-kernel, but an **encoder** like uzu.

**Reference:** `reference/uzu/crates/backend-uzu/src/backends/common/encoder.rs`

### 3a. Layer block encoder (biggest structural win)

Replace 100+ `encode_*` calls per layer with a **LayerBlock**:

```
Per layer (target: ~8–12 dispatches, not ~25):
  1. fused_qkv OR norm+qkv
  2. fused_k_attn OR k_norm+rope+kv
  3. attention
  4. fused_o_residual OR o+norm+add
  5. fused_mlp (norm+gate∥up+gelu)
  6. down matvec
  7. fused_post_mlp_residual
  8. ple (or fused with 7)
```

**Start:** Extend `src/mega_decode.rs` beyond Q4_0-only.

### 3b. Extend MEGA_KERNEL to Q4_K_M

Today `MEGA_KERNEL=1` bails on K-quant (`src/gemma4_gpu_model.rs` ~2612).

| Step | Work |
|------|------|
| 1 | Mega graph for K-quant MLP + attention encodes |
| 2 | Replay graph per token (single encode loop) |
| 3 | Compare dispatch count vs default path |

**Expected:** +1–2 tok/s from dispatch amortization.

### 3c. GPU sampling (+0.3–0.5 tok/s)

Move min-p/greedy to GPU (`sample_min_p` exists; CPU path default for scheduler).

**Reference:** `reference/uzu/crates/backend-uzu/src/backends/metal/kernel/sampling/unified_sampling.metal`

### 3d. GPU embedding lookup (+0.5–1 tok/s)

Today: CPU mmap → dequant → `write_buffer` every token. Upload embed table once; GPU gather kernel.

**Phase 3 exit:** ≥45 tok/s Q4_0, ≥38 tok/s Q4_K_M, dispatch count <300/token.

---

## Phase 4: Roofline push (M4 → ≥55 tok/s)

**Theme:** Bandwidth efficiency — currently ~40–50% of roofline.

### 4a. Matmul router (uzu-inspired)

Replace single `MATVEC_KERNEL` env with **(M,N,K, quant_type) → pipeline** table.

| Piece | Source |
|-------|--------|
| GEMV tile selection | `reference/uzu/crates/backend-uzu/src/backends/metal/kernel/matmul/gemv/policy.rs` |
| Keep ggml Q4_K/Q6_K inner loops | Already ported from llama.cpp (`reference/llama.cpp/ggml/src/ggml-metal/`) |
| M1/M2/M3 Pro tier tables | Benchmark hot Gemma shapes |

Hot shapes: `(m,k) = (10240,2560)`, `(2560,10240)`, `(2048,2560)`, `(2560,2048)`.

### 4b. BF16 activations (optional, +10–15% bandwidth)

Uzu uses BF16 activations; this engine uses F32 throughout. Halve activation traffic; need BF16 RMSNorm/attention paths.

### 4c. Q6_K → Q4_0 down conversion at load time

`Q6K_TO_Q4=1` showed marginal gain before; revisit after fusion. Unified Q4 down enables full MLP fusion.

**Phase 4 exit:** ≥55 tok/s on Q4_0 split cache.

---

## Phase 5: Beyond single-stream (M5 — 2× effective throughput)

| Technique | Effective gain | Reference |
|-----------|----------------|-----------|
| **Speculative decoding** | 1.5–2.5× tok/s | `reference/uzu/` trie + speculators |
| **Async decode overlap** | +10–20% | uzu `async_generate` |
| **Batch decode** | Aggregate throughput | `src/scheduler.rs`, batch engine |
| **Draft model** | 2×+ with acceptance | `.kiro/specs/production-inference-server/requirements.md` |

Only invest after **M3** — speculation multiplies a fast base.

---

## Priority matrix

```
Impact ▲
       │  [1a Fused QKV K-quant]  [2a Fix MLP fusion]
       │  [1b Norm+gelu_mul fuse]   [3a Layer block encoder]
       │  [1c Attention defaults] [3b Mega K-quant]
       │  [2b rmsnorm_acc]        [4a Matmul router]
       │  [2c GQA default-on]     [5 Speculative]
       └──────────────────────────────────────────► Effort
              LOW                              HIGH
```

**Recommended order:**

1. Phase 0 (instrument Q4_K_M)
2. Phase 1a → 1b → 1c (parity on Q4_K_M)
3. Phase 2a → 2b → 2c (beat llama on Q4_0)
4. Phase 3a (layer encoder)
5. Phase 3b–3d, then Phase 4
6. Phase 5 when base decode ≥40 tok/s

---

## Risk register

| Risk | Mitigation |
|------|------------|
| Fusion breaks Q4_K accuracy | `--gguf-kquant-test` + `--gguf-gen` after every kernel change |
| `FUSED_MLP_GELU_DOWN` stays slow | Abandon; build uzu-style 2-dispatch MLP instead |
| Mega kernel unmaintainable | Use as **replay graph**, not one Metal shader |
| Q6_K down blocks MLP fusion | Load-time Q6→Q4 down quant for decode-only |
| Chasing 100 tok/s | Reframe: **55 = excellent**, **2× spec = beyond** |

---

## Success criteria

| Phase | Metric | Correctness |
|-------|--------|-------------|
| 1 | ≥32 tok/s Q4_K_M | kquant test OK, essay gen OK |
| 2 | ≥38 tok/s Q4_0 | Match/exceed llama-bench tg |
| 3 | ≥45 tok/s Q4_0, <300 dispatches/token | Profile phases stable |
| 4 | ≥55 tok/s Q4_0 | Within 80% of roofline |
| 5 | ≥1.8× effective tok/s | Speculative acceptance >60% |

---

## Already done (don't redo)

- [x] Port llama.cpp Q4_K / Q6_K matvec kernels (`src/shaders/ggml_mul_mv_q4.metal`)
- [x] Fix `nb01` byte strides for K-quant (`src/ggml_gemv.rs`, `src/gpu.rs`)
- [x] Q4_K gate∥up+GeLU fused kernel (`matvec_ggml_q4_K_gelu_mul`)
- [x] Re-enable fused Q attention for K-quant layers (`src/gemma4_gpu_model.rs`)
- [x] Correctness: `--gguf-kquant-test` passes (max_rel_err ~1e-4)
- [x] Packed gate∥up for Q4_0 (uzu pattern, `PACKED_MLP_GATE_UP`)
- [x] `ATTENTION_GQA_Q4` (+1 tok/s on Q4_0, needs default-on validation)
- [x] Fused gate+up+GeLU ggml Q4_0 kernel (`MLP_GATE_UP_GGML=1`)

---

## Key files

| Area | Path |
|------|------|
| Decode path doc | `docs/02_decode_path.md` |
| Bottleneck analysis | `docs/08_bottleneck_analysis.md` |
| K-quant kernels | `src/shaders/ggml_mul_mv_q4.metal` |
| K-quant args/dispatch | `src/ggml_gemv.rs` |
| Decode orchestration | `src/gemma4_gpu_model.rs` |
| GPU encode/dispatch | `src/gpu.rs` |
| Mega decode graph | `src/mega_decode.rs`, `src/shaders/decode_mega.metal` |
| llama.cpp reference | `reference/llama.cpp/ggml/src/ggml-metal/` |
| uzu reference | `reference/uzu/crates/backend-uzu/` |
| Legacy perf notes | `perf.todo.md` |

---

## uzu vs mega-metal (summary)

| Lever | uzu | mega-metal (Q4_K_M) |
|-------|-----|---------------------|
| Gate∥up fusion | Native, all models | Partial (Q4_K gelu_mul yes; down Q6_K separate) |
| Matmul | Own GEMV router + tuned tiles | ggml port (correct) |
| Dispatches | ~fewer effective steps; 1 CB/pass | ~100–660/token |
| Sampling | GPU | CPU |
| Speculation | Built-in | None |
| MLP per layer | 3 steps (up∥gate → act → down) | 3+ (norm + gelu_mul + Q6_K down) |

---

## Quick env reference

```bash
# Q4_0 best known
MLP_GATE_UP_GGML=1 FUSED_QKV=0 \
  cargo run --release -- --gpu <model> --bench-decode --bench-decode-tokens 200

# Profile phases
PROFILE_PHASES=1 MLP_GATE_UP_GGML=1 \
  cargo run --release -- --gpu <model> --bench-decode --bench-decode-tokens 128

# GQA attention win (validate then default-on)
ATTENTION_GQA_Q4=1 MLP_GATE_UP_GGML=1 \
  cargo run --release -- --gpu <model> --bench-decode --bench-decode-tokens 200
```

**Next action:** Phase 1a — Fused QKV for K-quant.
