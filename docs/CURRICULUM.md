# Curriculum: Understanding `llama-sinks` (in depth)

This is a **textbook**, not a tour. Goal: after working through it, you can
derive attention math on a whiteboard, explain Metal threadgroup geometry,
trace one Gemma4 decode token through `decode_fused.rs` into a specific
`.metal` kernel, and explain why a hybrid KV-append bug produces coherent
then garbage text.

Binary: **`llama-sinks`**. Target: **Gemma4 E2B / E4B** on Apple Silicon via
**GGUF + Metal**.

---

## How this edition differs from a “map”

Earlier docs named files and drew boxes. That is orientation, not learning.
This edition teaches:

1. **First principles** — transformers, GPU execution, Softmax / FlashAttention
2. **Mental models** — what must be true in your head before you open a file
3. **Code as proof** — exact functions, kernels, env predicates in *this* repo
4. **Failure modes** — bugs from `AGENTS.md` that teach invariants

Every chapter now does concrete work with the code. What you *do* per chapter:

| Chapter | What you do with it |
|---------|---------------------|
| [00b](textbook/00b_transformers_first_principles.md) | Hand-compute RoPE and a two-tile online softmax |
| [00c](textbook/00c_gpu_and_metal_fundamentals.md) | Compute arithmetic intensity; derive dispatch geometry |
| [03](textbook/03_weights_and_gguf.md) | Parse the GGUF header; compute `byte_len` for every quant type |
| [04](textbook/04_metal_runtime.md) | Add a kernel end to end, host + shader |
| [05](textbook/05_quantization_matmul.md) | Derive Q4_0; walk ggml GEMV lane-by-lane |
| [06](textbook/06_kv_cache.md) | Derive the cache byte address for any (head, pos, block) |
| [07](textbook/07_attention.md) | Walk `flash_decode_full_fused_*` line-by-line |
| [08](textbook/08_mlp_norms_ple.md) | Read the RMSNorm kernel; predict the MLP branch and dispatch count |
| [09](textbook/09_decode_path.md) | Trace one token through `forward_single_token_inner` |
| [10](textbook/10_prefill_path.md) | Follow every layout transition in one prefill layer |
| [11](textbook/11_server_api.md) | Trace JSON → token ids → SSE deltas, including the shrink bug |
| [12](textbook/12_scheduler_batching.md) | Hand-run water-filling; justify `sort`/`dedup`/`rev`/`swap_remove` |
| [13](textbook/13_kv_pool.md) | Compute total KV memory for your config |
| [14](textbook/14_mtp.md) | Hand-run verify/accept/rewind on a mismatch |
| [15](textbook/15_optimization_lab.md) | Run an ablation and write an `AGENTS.md` entry |
| [17](textbook/17_tensor_shapes.md) | Derive every shape from the config accessors |

Do not skim the GPU chapters. Read 05/06/07 with `llama.metal` and
`ggml_mul_mv_q4.metal` open on a second screen; the code blocks are quoted
with line numbers so you can jump straight to them.

If a chapter feels dense: good. Work the drills at the end with the code open.

---

## How to use this

1. Read **Part 0** even if you “know transformers.” The rest assumes that math.
2. For each chapter: read prose → open cited files → do the checklist **cold**.
3. Canonical path: [`textbook/`](textbook/). Ignore
   [`archive/deprecated/`](archive/deprecated/).
4. Living perf diary: root [`AGENTS.md`](../AGENTS.md) — read after Ch 15.

Index: [`textbook/README.md`](textbook/README.md).

---

## Syllabus (≈ 40–60 hours of serious study)

| Part | Chapters | You should be able to |
|------|----------|------------------------|
| **0 — Foundations** | 00–00c | Derive attention; explain Metal threads/simdgroups; bandwidth vs FLOPs |
| **I — Map** | 01 | Draw client → Metal; know which layer owns which bug |
| **II — Model** | 02–03, 17 | Gemma4 quirks (GQA, SWA, PLE, shared KV); load GGUF; know shapes |
| **III — GPU** | 04–08 | Trace encode → dispatch; dequant on the fly; KV layout; flash/MWG |
| **IV — Paths** | 09–10 | One-token decode call graph; prefill tiling; hybrid routing |
| **V — Serve** | 11–13 | Request → slot → schedule → SSE; continuous batching |
| **VI — Advanced** | 14–16 | MTP draft/verify; measurement method; whiteboard drills |

---

## Suggested study plan

| Block | Focus |
|-------|--------|
| 1–2 days | Ch 00–00c (foundations — do not skip) |
| 1 day | Ch 01–02 (system + Gemma4 arch) |
| 1–2 days | Ch 03 + 17 + 04–05 (weights, shapes, Metal runtime, quant matmul) |
| 2 days | Ch 06–08 (KV, attention, MLP/PLE) — hardest GPU material |
| 1–2 days | Ch 09–10 with code open (decode + prefill) |
| 1–2 days | Ch 11–13 (server stack — 11 and 12 are long) |
| 1–2 days | Ch 14–15 (MTP + optimization lab) |
| 1 day | Ch 16 drills — redraw everything from memory |

---

## The bar

For any subsystem you claim to understand:

1. **Draw** the dataflow (boxes + arrows + tensor shapes).
2. **Name** the Rust type and the Metal `kernel void` entry point(s).
3. **State one invariant** (e.g. “shared-KV layers never append”).
4. **Describe one real bug** that violated it (`AGENTS.md`).
5. **Name the env var** that switches the path (if any).

Reciting README bullets is not the bar.
