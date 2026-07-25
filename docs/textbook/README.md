# Textbook Index

Canonical student path for `llama-sinks`. Start at [../CURRICULUM.md](../CURRICULUM.md).

Outdated notes live under [`../archive/deprecated/`](../archive/deprecated/) —
**do not study them.**

## Part I — Orientation

| # | File | Topic |
|---|------|--------|
| 00 | [00_how_to_study.md](00_how_to_study.md) | How to study AI-assisted systems code |
| 01 | [01_system_map.md](01_system_map.md) | End-to-end map: client → Metal |

## Part II — Model & Weights

| # | File | Topic |
|---|------|--------|
| 02 | [02_gemma4_architecture.md](02_gemma4_architecture.md) | Sliding/full attn, GQA, PLE, QK-norm, softcap, shared KV |
| 03 | [03_weights_and_gguf.md](03_weights_and_gguf.md) | GGUF mmap, HF path, weight formats |
| 17 | [17_tensor_shapes.md](17_tensor_shapes.md) | E4B shape cheatsheet (current defaults) |

## Part III — GPU Engine

| # | File | Topic |
|---|------|--------|
| 04 | [04_metal_runtime.md](04_metal_runtime.md) | `MetalContext`, command buffers, pipelines, env flags |
| 05 | [05_quantization_matmul.md](05_quantization_matmul.md) | Q4_0 / Q4_K / Q6_K, matvec, mul_mm, ext |
| 06 | [06_kv_cache.md](06_kv_cache.md) | Layout, append, sliding window, shared layers |
| 07 | [07_attention.md](07_attention.md) | Flash decode, ggml MWG, hybrid auto, GQA |
| 08 | [08_mlp_norms_ple.md](08_mlp_norms_ple.md) | RMSNorm, GeLU MLP, PLE block |

## Part IV — Forward Paths

| # | File | Topic |
|---|------|--------|
| 09 | [09_decode_path.md](09_decode_path.md) | One-token decode call graph |
| 10 | [10_prefill_path.md](10_prefill_path.md) | Chunked / batched prefill, flash_attn_ext |

## Part V — Serving

| # | File | Topic |
|---|------|--------|
| 11 | [11_server_api.md](11_server_api.md) | Axum routes, chat template, SSE |
| 12 | [12_scheduler_batching.md](12_scheduler_batching.md) | Admission, prefill/decode rounds, continuous batching |
| 13 | [13_kv_pool.md](13_kv_pool.md) | Slots, allocate/release, BatchEngine bridge |

## Part VI — Advanced

| # | File | Topic |
|---|------|--------|
| 14 | [14_mtp.md](14_mtp.md) | Draft head, verify, serial MTP scheduler |
| 15 | [15_optimization_lab.md](15_optimization_lab.md) | How we measure; experiment log method |
| 16 | [16_glossary_and_drills.md](16_glossary_and_drills.md) | Glossary + whiteboard drills |

## Quick file → chapter map

| Code | Start here |
|------|------------|
| `server.rs` | 11 |
| `scheduler.rs` | 12 |
| `kv_pool.rs` / `batch_engine.rs` | 13 |
| `gemma4_gpu_model.rs` | 02, 09, 10, 17 |
| `gpu.rs` | 04, 07 |
| `decode_fused.rs` | 09 |
| `gguf.rs` / `weights.rs` | 03 |
| `shaders/*.metal` | 05–08 |
| `mtp_serve.rs` / `speculative.rs` | 14 |
| `AGENTS.md` | 15 |
