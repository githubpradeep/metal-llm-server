# 16 — Glossary & Whiteboard Drills

## Glossary

| Term | Meaning here |
|------|----------------|
| **Prefill** | Process prompt tokens; fill KV; optionally produce first logits |
| **Decode** | Generate one new token using KV |
| **GQA** | Grouped-query attention — fewer KV heads than Q heads |
| **SWA** | Sliding-window attention layer |
| **Shared KV** | Layer reuses earlier layer’s K/V cache |
| **PLE** | Per-layer embedding residual path |
| **QK-Norm** | RMSNorm on Q and K before attention |
| **Softcap** | `cap*tanh(x/cap)` on logits |
| **Q4_0 / Q4_K / Q6_K** | ggml weight block formats |
| **Matvec** | Matrix×vector (decode) |
| **mul_mm** | Matrix×matrix (prefill) |
| **mul_mv_ext** | Small-batch matvec sharing dequant |
| **Flash attention** | Tiled online-softmax attention |
| **MWG** | Multi-workgroup attention + reduce |
| **Fused kernel** | Multiple math stages in one Metal dispatch |
| **KV slot** | One request’s GPU KV cache in the pool |
| **Queue depth** | Max waiting HTTP inference requests |
| **Continuous batching** | Mix prefill/decode of different requests per tick |
| **MTP** | Speculative multi-token draft + verify |
| **h_nextn** | Hidden state used to condition the draft head |
| **CB** | Metal command buffer |
| **PSO** | Pipeline state object (compiled kernel) |

---

## Drill A — 10-minute system sketch

Without notes, draw:

1. Client → server → queue → scheduler → batch engine → model → Metal.  
2. Label where `LLAMA_KV_POOL_SLOTS` and `LLAMA_QUEUE_DEPTH` apply.  
3. Mark MTP’s serial scheduler as a side path.

---

## Drill B — One layer

Draw one Gemma4 layer with:

- norms, QKV, QK-norm, RoPE, attention, O, MLP, PLE, scalar  
- annotate `has_kv=false` differences  

---

## Drill C — Hybrid attention bug

Explain in ≤5 sentences:

> Why `ATTENTION_KERNEL=auto` produced coherent text then collapsed into
> repetitive garbage after context crossed ~128, and what `needs_explicit_kv_append`
> does.

---

## Drill D — Memory

Estimate (rough is fine):

> 4 slots × 42 layers × 4 kv heads × 8192 ctx × Q4_0 row for hd=128  
> (ignore full layers first, then discuss how full hd=512 changes it)

---

## Drill E — Prefill vs decode optimization

Pick each idea as mainly **prefill**, **decode**, or **both**:

- Raise flash NSG for h256  
- `ATTENTION_KERNEL=auto`  
- PLE keep f16  
- dual gate∥up matvec  
- tiled ext `TILED_EXT_MIN_Q=2`  
- increase KV pool slots  

---

## Drill F — Code tour (timed)

30 minutes with the repo:

| Minute | Open |
|--------|------|
| 0–5 | `scheduler.rs` `run` |
| 5–10 | `kv_pool.rs` `new` / `allocate` |
| 10–15 | `gpu.rs` `attention_use_ggml_for_layer_kv` |
| 15–20 | one `attention_flash_decode*` in `llama.metal` |
| 20–25 | `batch_engine.rs` |
| 25–30 | `AGENTS.md` summary table |

Close laptop; recite what each does.

---

## Answer keys (short)

**C:** Auto switches to ggml MWG at kv_seq≥128; fused flag still looked “on” so
explicit KV append was skipped; attention read cache without current token;
fix forces append whenever ggml path is active.

**E:** NSG prefill; auto decode; PLE f16 prefill; dual matvec decode; tiled min_q
MTP/prefill small-q; pool slots serving concurrency.

---

## You finished the curriculum when…

- [ ] You can teach Ch 01 and Ch 12 to someone else with only a whiteboard.  
- [ ] You can navigate from an HTTP request to a Metal attention kernel.  
- [ ] You can discuss AGENTS experiments without reading them line-by-line.  
- [ ] You know what you still *don’t* know (mega-kernel parity, exact GQA status, …).

Return to [`README.md`](README.md) or [`../CURRICULUM.md`](../CURRICULUM.md).
