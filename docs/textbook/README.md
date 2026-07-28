# Textbook Index — `llama-sinks`

Start at [../CURRICULUM.md](../CURRICULUM.md).

~11 000 lines across 20 chapters. Every chapter is a tutorial: derivations,
line-by-line code walks with quoted source and line numbers, worked numbers,
failure modes, and exercises. None is a summary. Read them with the cited files
open.

**Budget 3–4 h each:**
[00b](00b_transformers_first_principles.md) transformer math derived from
scratch (attention from "content-addressed lookup", RoPE's relative-position
proof, online softmax by hand),
[07](07_attention.md) the fused flash decode kernel line by line, plus MWG and
the hybrid,
[05](05_quantization_matmul.md) Q4_0/Q4_K/Q6_K, GEMV lane geometry, `mul_mm`
tiles.

**Budget 2–3 h each:**
[00c](00c_gpu_and_metal_fundamentals.md) (the roofline derived, dispatch
geometry),
[08](08_mlp_norms_ple.md), [09](09_decode_path.md), [10](10_prefill_path.md),
[12](12_scheduler_batching.md), [02](02_gemma4_architecture.md),
[14](14_mtp.md).

**Where the hard-won knowledge lives:** every chapter ties its subject to the
`AGENTS.md` experiment log — what was tried, what failed, and why. Chapter
[15](15_optimization_lab.md) is the method itself, and
[16](16_glossary_and_drills.md) is the glossary, drills with answer keys, and a
20-question closed-book exam.

Outdated notes: [`../archive/deprecated/`](../archive/deprecated/) — **do not study**.

---

## Part 0 — Foundations (read first)

| # | File | Topic |
|---|------|--------|
| 00 | [00_how_to_study.md](00_how_to_study.md) | Four reading modes, verification habits, route map |
| 00b | [00b_transformers_first_principles.md](00b_transformers_first_principles.md) | Tokens → attention → residual stack → AR decode |
| 00c | [00c_gpu_and_metal_fundamentals.md](00c_gpu_and_metal_fundamentals.md) | GPU hierarchy, Metal API, bandwidth math, Apple Silicon |

## Part I — Orientation

| # | File | Topic |
|---|------|--------|
| 01 | [01_system_map.md](01_system_map.md) | Module inventory, three entry paths, full decode/prefill call chains, navigation recipes |

## Part II — Model & Weights

| # | File | Topic |
|---|------|--------|
| 02 | [02_gemma4_architecture.md](02_gemma4_architecture.md) | SWA/full, GQA, QK-norm, RoPE, PLE, shared KV, softcap |
| 03 | [03_weights_and_gguf.md](03_weights_and_gguf.md) | GGUF byte-by-byte, block specs, dtype traps, the weight cache, load verification |
| 17 | [17_tensor_shapes.md](17_tensor_shapes.md) | Deriving every shape; per-layer head_dim; memory arithmetic |

## Part III — GPU Engine

| # | File | Topic |
|---|------|--------|
| 04 | [04_metal_runtime.md](04_metal_runtime.md) | `MetalContext`, pipelines, encode pattern, args-struct contract, add-a-kernel tutorial |
| 05 | [05_quantization_matmul.md](05_quantization_matmul.md) | Q4_0/Q4_K/Q6_K, GEMV, mul_mm, ext matvec |
| 06 | [06_kv_cache.md](06_kv_cache.md) | Address math derived, append kernel line by line, fused vs explicit, SWA, failure modes |
| 07 | [07_attention.md](07_attention.md) | Flash decode, online softmax, ggml MWG, hybrid auto |
| 08 | [08_mlp_norms_ple.md](08_mlp_norms_ple.md) | RMSNorm kernel walk, 4-branch MLP, PLE pre-pass + per-layer block |

## Part IV — Forward Paths

| # | File | Topic |
|---|------|--------|
| 09 | [09_decode_path.md](09_decode_path.md) | One-token decode: call graph through fused executor |
| 10 | [10_prefill_path.md](10_prefill_path.md) | Layout journey, segments, strided append, flash_attn_ext, phase timing |

## Part V — Serving

| # | File | Topic |
|---|------|--------|
| 11 | [11_server_api.md](11_server_api.md) | Prompt encoding, validation, backpressure, SSE frames and delta diffing |
| 12 | [12_scheduler_batching.md](12_scheduler_batching.md) | Tick loop, sampling policy, water-filling, cancellation |
| 13 | [13_kv_pool.md](13_kv_pool.md) | Memory arithmetic, slot lifecycle, aliasing, capacity planning, batch semantics |

## Part VI — Advanced

| # | File | Topic |
|---|------|--------|
| 14 | [14_mtp.md](14_mtp.md) | Draft chain, verify/accept/rewind, the speculative cost model and its ceiling |
| 15 | [15_optimization_lab.md](15_optimization_lab.md) | Measurement rules, knob catalog, case studies, benchmarking discipline |
| 16 | [16_glossary_and_drills.md](16_glossary_and_drills.md) | Glossary, 12 drills with keys, final exam, capstone |

---

## Code → chapter map

| Code | Chapters |
|------|----------|
| `server.rs` | 11 |
| `scheduler.rs` | 12 |
| `kv_pool.rs` / `batch_engine.rs` | 13 |
| `gemma4_config.rs` / `gemma4_gpu_model.rs` | 02, 09, 10, 17 |
| `decode_fused.rs` | 09, 07, 08 |
| `gpu.rs` | 04, 05, 07 |
| `gguf.rs` | 03, 05 |
| `shaders/*.metal` | 00c, 05–08 |
| `mtp_serve.rs` / `speculative.rs` / `gemma4_mtp.rs` | 14 |
| `AGENTS.md` | 15 |
