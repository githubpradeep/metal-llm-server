# 16 — Glossary, Drills, and Final Exam

Use after Parts 0–V. Drills come with answer keys at the end; the exam does
not — for those, verify in the code.

---

# Part A — Glossary

## A.1 Concepts

| Term | Meaning in this engine |
|---|---|
| **Prefill** | multi-token forward that fills KV; compute-bound; `mul_mm` + tiled attention |
| **Decode** | one-new-token forward; bandwidth-bound; matvec + flash decode |
| **Ridge point** | ~16 FLOP/byte on M1 Pro; below it a kernel is memory-bound (00c A.1) |
| **Arithmetic intensity** | FLOPs ÷ bytes moved; ~3.5 for Q4_0 matvec, ~3.5·S for batch S |
| **GQA** | fewer KV heads than Q heads; `num_kv_groups = n_q / n_kv` (5 on E4B) |
| **SWA / sliding layer** | attends the last `sliding_window` keys; head_dim 128; θ≈10⁴ |
| **Full / global layer** | attends all keys; head_dim 512; θ≈10⁶; partial rotary 0.25 |
| **Shared KV layer** | `has_kv = false`; computes Q and attends against `kv_source_layer` |
| **Anchor layer** | the KV-owning layer a shared layer reads; must be the same type |
| **QK-norm** | per-head RMSNorm on Q and K before RoPE; makes `attention_scale = 1.0` |
| **V-norm** | per-head RMSNorm on V with **no** learned weight |
| **PLE** | per-layer embedding: token identity + context projection, injected per layer |
| **`layer_scalar`** | per-layer constant multiply on the residual (depth stabilization) |
| **Softcap** | `cap · tanh(logits / cap)`, cap = 30 |
| **Online / flash softmax** | running `(m, ℓ, acc)` with retroactive rescale by `exp(m_old − m_new)` |
| **Residual stream** | the `hidden`-wide bus every sub-block adds into; identity is the default |
| **Continuous batching** | each tick: decode all `Decoding` requests, then prefill within a budget |
| **Water-filling** | fair prefill allocation: equal shares, redistribute what small requests don't use |
| **MTP** | speculative decoding: draft head proposes, base model verifies in one batch |
| **`h_nextn`** | the hidden state the draft head is seeded with — **post**-final-norm on Gemma4 |
| **Accept rate** | fraction of drafted tokens the base model confirms (~42% here) |

## A.2 Formats

| Term | Meaning |
|---|---|
| **Q4_0** | 32 weights → 18 B; `w = d·(code − 8)`; symmetric, zero-point 8 |
| **Q8_0** | 32 values → 34 B (KV cache option) |
| **Q4_K** | 256 weights → 144 B; affine: `w = (d·sc)·code − dmin·m`; 6-bit sub-scales |
| **Q6_K** | 256 weights → 210 B; 6-bit codes split across `ql`/`qh`; zero-point 32 |
| **K-quant** | umbrella for Q4_K/Q6_K; `WeightFormat::KQuant` is the per-layer tag |
| **`bytes_per_row`** | KV row stride: F16 `hd·2`, Q8_0 `(hd/32)·34`, Q4_0 `(hd/32)·18` |
| **The size trap** | Q4_0 and Q4_K of the same shape have identical byte length |

## A.3 Code identifiers worth memorizing

| Identifier | Where | Role |
|---|---|---|
| `MetalContext` | `gpu.rs` | device, queue, ~150 pipelines, all `encode_*` |
| `BufferView` | `gpu.rs` | `(buffer, offset, length, format)` — format is a correctness invariant |
| `Gemma4GpuLayer` | `gemma4_gpu_model.rs` ~845 | per-layer weights + `has_kv`, `kv_source_layer`, `head_dim` |
| `forward_single_token_inner` | ~3364 | the decode trace (Ch 09) |
| `encode_fused_decode_layer` | `decode_fused.rs` ~117 | per-layer fused encode; kernel policy |
| `fused_decode_eligible` | `decode_fused.rs` ~101 | four conditions; one F16 layer disables all |
| `needs_explicit_kv_append` | `gpu.rs` ~243 | unifies kernel choice with append side effects |
| `attention_use_ggml_for_layer_kv` | `gpu.rs` | the hybrid routing predicate |
| `KvCachePool` / `KvCacheSlot` | `kv_pool.rs` | per-request K/V buffers, free list |
| `alias_kv_from_pool` | `gemma4_gpu_model.rs` | points model caches at a slot |
| `plan_prefill_round` | `scheduler.rs` ~476 | water-filling |
| `prepare_decode_token` | `scheduler.rs` ~594 | sampling + stopping policy |
| `compute_stream_deltas` | `server.rs` ~526 | SSE delta computation |
| `forward_verify_parallel` | `gemma4_gpu_model.rs` ~5141 | MTP batched verify |

## A.4 GPU vocabulary

| Term | Meaning |
|---|---|
| **Simdgroup** | 32 lanes in lockstep; `simd_sum` / `simd_max` reduce across them |
| **Threadgroup** | 64–1024 threads; owns threadgroup memory (~32 KB cap); the unit of cooperation |
| **`NSG`** | simdgroups per threadgroup (4 default; 8 for h256 prefill flash) |
| **`NR0`** | output rows per simdgroup in matvec (4; `KQ_NR0 = 4`) |
| **`NWG`** | workgroups splitting the KV axis in MWG attention (32) |
| **`NQPTG` / `NCPSG`** | query rows per TG (8) and keys per chunk (64) in tiled prefill attention |
| **Function constant** | compile-time pipeline specialization (`has_kvpad`, `bc_mask`, `bc_inp/out`) |
| **`dispatch_threads`** | Metal computes TG count; threads past your data still run → need a bounds guard |
| **`dispatch_thread_groups`** | you compute the TG count; use when a TG *owns* something |

## A.5 Environment knobs (behaviour-changing)

| Env | Default | Effect |
|---|---|---|
| `ATTENTION_KERNEL` | specialized | `auto` = fused < 128 KV, ggml MWG ≥ 128; `ggml`, `mwg` |
| `LLAMA_KV_CACHE_TYPE` | f16 | `q4_0` for ~3.5× less KV memory and traffic |
| `FUSED_DECODE` | 1 | fused per-layer executor |
| `ATTENTION_GQA_Q4` | 0 | GQA-tiled attention (shared-KV layers only) |
| `METAL_N_CB` | 1 | split a token across command buffers for CPU/GPU overlap |
| `TILED_EXT_MIN_Q` | 2 | min query rows for the tiled ext attention kernel |
| `MUL_MM_MIN_SEQ` | ~16 | min sequence length before switching to matmul |
| `LLAMA_KV_POOL_SLOTS` | 4 | concurrency ceiling |
| `LLAMA_QUEUE_DEPTH` | 32 | admission queue capacity |
| `LLAMA_CTX_SIZE` | 16384 | KV capacity per slot |
| `MTP_VERIFY_CROSSCHECK` | 0 | verify parallel == sequential, every row |
| `PROFILE_{DECODE,PHASES,GPU,DISPATCHES,ABLATE}` | off | measurement modes (mutually exclusive in places) |

---

# Part B — Drills

## Drill A — Shapes (E4B)

For a **sliding** layer and a **full** layer, write from memory:

1. `q_out_dim`, `kv_out_dim`
2. Q4_0 KV bytes for one position (K only)
3. MLP gate weight bytes in Q4_0
4. Which kernel suffix (`h128`/`h256`/`h512`) the attention dispatch uses

## Drill B — One decode token

Without notes, list in order:

1. The two CPU gathers and their scale factors
2. Everything encoded before the layer loop
3. Inside one KV-owning fused layer: attention, MLP, PLE sub-blocks
4. What a shared-KV layer skips
5. Final norm / lm_head / softcap, and what each `DecodeMode` does
6. Which metadata gets bumped, and where it is stored in the server path

## Drill C — Hybrid routing and the append bug

1. What `ATTENTION_KERNEL=auto` does at `effective_kv_seq` 50 vs 200
2. Why the fused-append assumption and the ggml path disagreed
3. How `needs_explicit_kv_append` unifies them
4. Why the symptom appeared ~128 tokens into a generation, not at token 1

## Drill D — Metal geometry

For `matvec_ggml_q4_0` with `NR0 = 4`, `NSG = 2`:

1. Threads per threadgroup?
2. Output rows per threadgroup?
3. `first_row` for `tgpig.x = 2, sgitg = 1`?
4. For `M = 10240`, how many threadgroups?
5. Which lane writes the result, and why only one?

## Drill E — Kernel family selection

Name the dominant kernel family for:

1. Decode Q projection, Q4_K weights
2. Prefill MLP gate at S = 4096
3. Prefill attention at S = 4096
4. MTP verify MLP at S = 3
5. MTP verify attention at S = 3
6. Decode attention, Q4_0 KV, `has_kv`, kv_seq 60

## Drill F — Serving

1. Queue depth versus pool slots: who waits where, and which is the real limit?
2. Why do decode rounds run before prefill rounds in each tick?
3. What changes when `--mtp` is enabled for `--serve`?
4. Two ways a request can be cancelled, and who sets the flag in each

## Drill G — Numerics

1. The online softmax update when the new tile's max exceeds the running max
2. Q4_0 dequant with the `−8` factored out of the sum
3. Why Gemma's `attention_scale` can be 1.0
4. Why the embedding is scaled by `√hidden_size`

## Drill H — Memory arithmetic

1. F16 KV for 4 slots at 16384 context, E4B (mixed head_dim, 24 owning layers)
2. The same in Q4_0
3. Weight bytes streamed per decode token, and the implied rate ceiling
4. `lm_head` bytes in Q4_0, and its share of a 20 ms token

## Drill I — Prefill layout

List, in order, every layout transformation in one prefill layer from `hidden`
to the attention call, naming the kernel for each. Then the transformations
after attention back into the residual.

## Drill J — Failure attribution

For each symptom, name the layer (product/model/device) and the first thing you
would check:

1. Output coherent for ~500 tokens, then repetitive
2. Two concurrent requests each see the other's context
3. Correct at head_dim 128, garbage at 512
4. 429 responses under load with the GPU mostly idle
5. Fluent output that scores worse than llama.cpp on every prompt

## Drill K — Design defence

Justify each in one sentence, then name what it costs:

1. Single-threaded scheduler with no locks
2. CPU-side embedding gather
3. `format` tag on `BufferView` instead of inferring from length
4. Sizing scratch buffers to the maximum over layers
5. Sequential (non-batched) MTP serving

## Drill L — Predict a measurement

Predict the direction and rough magnitude, then check `AGENTS.md`:

1. `fastMathEnabled = true`
2. `KQ_NR0` from 4 to 2
3. `NSG` from 4 to 8 for prefill h256
4. Fusing GeLU into the ext matvec
5. `MUL_MM_MIN_SEQ = 1` (matmul at batch 1–8)

---

# Part C — Final exam (20 questions, closed book)

90 minutes, no repo, no notes. Then verify each answer **in the code**, not in
the chapter prose. Chapter references are where to look if you were stuck.

**Foundations**

1. Why is decode bandwidth-bound and prefill compute-bound? Give the arithmetic
   intensity for one MLP gate projection in both regimes, and the batch size
   where they cross. (00c A, 10)
2. Write the online-softmax rescaling update for a tile whose max exceeds the
   running max, and show what breaks if you rescale `ℓ` but not `acc`. (00b 4.5, 07)
3. `partial_rotary_factor = 0.25` on full layers. How is it implemented without
   a branch in the consuming kernel? (00b 5.2, 02 F)

**Weights and shapes**

4. Q4_0 and Q4_K tensors of identical shape have identical byte length. Show the
   arithmetic, and name what the engine uses to tell them apart. (03, 05 E.4)
5. `layer_head_dim` returns two values in one model. Which, when, and name three
   things that must be sized to the larger one. (02 C, 17 A)
6. Compute `byte_len()` for a `[10240, 2560]` Q6_K tensor. (03)
7. Derive the affine Q4_K dequant formula and explain why Q6_K needs no `dmin`.
   (05 E)

**GPU**

8. In `rmsnorm`, why is the barrier *inside* the reduction loop, and why does
   removing it "work" at 32 threads? (00c B.3, 08 A)
9. What does `rmsnorm_acc` save versus `rmsnorm` + `vec_add`, in bytes per
   token? (08 B)
10. When would you use `dispatch_threads` rather than `dispatch_thread_groups`,
    and what must the kernel then contain? (00c D)
11. Derive the Q4_0 KV byte address for head 2, position 100, group 3, head_dim
    128, capacity 8192. (06 C)
12. Why is `capacity` — not `seq_len` — the stride between heads, and what is
    the failure mode if you use the wrong one? (06 C)

**Attention**

13. In `flash_decode_full_fused_q4_0`, what single branch decides whether K comes
    from f32 scratch or the quantized cache, and why must it exist? (07 D.6)
14. Which heads may append K/V in the fused kernel, and what expression enforces
    it? (07 D.10)
15. State the `needs_explicit_kv_append` invariant, the bug that motivated it,
    and why the failure appeared partway through a generation. (07 F.5)
16. MWG with `NWG = 32`, `C = 32`: at what `kv_seq` does it start to pay off?
    Show the arithmetic and connect it to the measured 40.4 tok/s at 441. (07 F.4)

**Paths**

17. List every layout transition in one prefill layer, in order, with kernels.
    (10 C)
18. Why does prefill commit roughly one command buffer per layer and decode one
    per token? (00c C.2, 09 C, 10 F)

**Serving**

19. Justify `sort`, `dedup_by_key`, `.rev()`, and `swap_remove` in the
    scheduler's reap loop — one sentence each. Then hand-run water-filling with
    budget 300 and requests needing 40 / 40 / 900. (12)
20. At 42% accept rate and 1.85 tokens per forward, what is the ceiling on MTP
    speedup with a *free* verify? Name the two levers that could move it, and
    say which one `AGENTS.md` M5/M6 attacked. (14)

Scoring: for each answer you could not produce, reread that chapter's cited
code — not its prose.

---

# Part D — Capstone

Whiteboard, 30 minutes, out loud, no notes:

```text
Client request  →  …  →  a sampled token streaming back
```

Your drawing must include:

- which thread owns the scheduler and how it talks to the HTTP task
- where KV memory lives, who allocates it, and how the model reaches it
- one concrete Metal dispatch for attention, with its threadgroup geometry
- one env var that changes that dispatch, and the condition it tests
- one historical bug that degraded quality without crashing, and its fix
- one measurement you would take first if throughput dropped 20%

If you can do that while naming real files and functions, you own the system.

---

# Part E — Answer keys (Drills A–E, H)

**Drill A (E4B).** Sliding: `q_out = 2560`, `kv_out = 512`; full:
`q_out = 10240`, `kv_out = 2048`. Q4_0 KV bytes per position (K only): sliding
`4 × 72 = 288`, full `4 × 288 = 1152`. Gate in Q4_0:
`10240 × (2560/32) × 18 ≈ 14.7 MB`. Suffix: sliding → `h128`, full → `h512`.

**Drill C.** `auto` runs fused below 128 effective KV and ggml MWG at or above.
The bug: `fused_kv_attention_enabled()` stayed true in `auto` mode, so the host
skipped the explicit append while the ggml kernel — which never appends — was
actually selected. `needs_explicit_kv_append` derives append from the same
predicate as kernel selection. The symptom appeared ~128 tokens in because that
is where routing switches; before it, the fused kernel appended correctly.

**Drill D.** 64 threads per TG (`NSG × 32`); 8 rows per TG (`NSG × NR0`);
`first_row = (2 × 2 + 1) × 4 = 20`; `ceil(10240 / 8) = 1280` threadgroups; lane
0 writes after `simd_sum`, because all 32 lanes hold the same reduced value and
32 writes to one address would be redundant traffic (and racy for
read-modify-write patterns).

**Drill E.** (1) `matvec_ggml_q4_K` plus fused variants. (2) `mul_mm_q4_K_f32`.
(3) tiled `flash_attn_ext` (8×64 tiles). (4) `matvec_ggml_ext_q4K_nx8_r3`,
optionally the fused GeLU variant. (5) tiled `flash_attn_ext`, enabled by
`TILED_EXT_MIN_Q = 2`. (6) `flash_decode_full_fused_q4_0_h128` — rung 1 of the
ladder, since kv_seq 60 < 128 keeps it off the ggml path.

**Drill H.** F16 at 16384 context, E4B, 24 owning layers with mixed head_dim:
~1.06 GB per slot ⇒ ~4.2 GB for 4 slots. Q4_0: ~0.30 GB per slot ⇒ ~1.2 GB.
Weights ~2.5 GB per token ⇒ ~12.5 ms at 200 GB/s ⇒ ~80 tok/s ceiling.
`lm_head` ≈ 378 MB ⇒ ~1.9 ms ⇒ ~10% of a 20 ms token.

---

**Back to index:** [README.md](README.md)
