# Gemma4 Metal Inference Server (Rust)

Local LLM inference server for **Gemma4** on Apple Silicon. Metal GPU
kernels, OpenAI-compatible API, KV pooling, continuous batching, and chunked
prefill. Loads community GGUF directly with mmap zero-copy weights.

Binary name: `llama-sinks`.

## Models (GGUF preferred)

The primary path is a **community GGUF** (weights + embedded tokenizer). Point
`--gpu` at a `.gguf` file:

| Model | Example GGUF | Notes |
|-------|----------------|-------|
| **Gemma4 E2B** | `gemma-4-E2B-it-Q4_K_M.gguf` | Smaller; good for prefill/decode tuning |
| **Gemma4 E4B** | `gemma-4-E4B-it-Q4_K_M.gguf` | Larger; original target |

`Q4_K_M` (mixed Q4_K / Q6_K) is the usual quant. Weights are read directly from
the GGUF and uploaded as native K-quant (Q4_K / Q6_K) Metal buffers via
memory-mapped zero-copy — cold load is sub-second and there is no separate
weight cache to build or maintain.

HF safetensors directories (`config.json` + `tokenizer.json` + weights) still load,
but GGUF is what we use day-to-day.

### Get a GGUF

From Hugging Face (example — pick a repo that hosts the file you want):

```bash
# Example layout
mkdir -p ~/Downloads/models/e2b

# Download with huggingface-cli / browser, then:
ls ~/Downloads/models/e2b/gemma-4-E2B-it-Q4_K_M.gguf
```

Or convert / obtain any Gemma4 Instruct `Q4_K_M` GGUF compatible with llama.cpp.

## Build

```bash
git clone git@github.com:githubpradeep/metal-llm-server.git
cd metal-llm-server
export MACOSX_DEPLOYMENT_TARGET=15.0
cargo build --release
```

## Run server (GGUF)

```bash
ATTENTION_KERNEL=auto LLAMA_KV_CACHE_TYPE=q4_0 LLAMA_CTX_SIZE=200000 LLAMA_MAX_PREFILL_SEQ=4096 \
  ./target/release/llama-sinks --gpu ~/Downloads/models/e2b/gemma-4-E2B-it-Q4_K_M.gguf --serve
```

Listens on `http://0.0.0.0:8080` by default (`--port N` to change).

### Important env vars

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLAMA_KV_CACHE_TYPE` | `f16` | KV quant: `q4_0` (recommended), `q8_0`, or omit for f16 |
| `LLAMA_CTX_SIZE` | `16384` | KV capacity / context window (max `200000`) |
| `LLAMA_MAX_PREFILL_SEQ` | engine default | Max tokens per prefill chunk (e.g. `4096`) |
| `ATTENTION_KERNEL` | `specialized` | Decode attention: `auto` (hybrid), `ggml`, or `specialized` |
| `LLAMA_QUEUE_DEPTH` | server default | Admission queue depth |
| `LLAMA_KV_POOL_SLOTS` | server default | Concurrent KV slots |
| `LLAMA_REQUEST_TIMEOUT_SECS` | server default | Per-request timeout |
| `LLAMA_PREFILL_TOKENS_PER_TICK` | server default | Prefill budget per scheduler tick |

Prefill tuning (defaults are usually fine):

| Variable | Default | Meaning |
|----------|---------|---------|
| `PREFILL_FLASH_ATTN` | on | Tiled `flash_attn_ext` for prefill (`0` = legacy) |
| `PREFILL_MUL_MM` | on | K-quant matrix-matrix for long seq |
| `PREFILL_MLP_F16` | on | f16 RHS activations for MLP mul_mm |
| `PREFILL_GATE_UP_STACKED` | on | Stacked gate∥up mul_mm |
| `PREFILL_QKV_STACKED` | on | Stacked Q∥K∥V mul_mm |
| `PREFILL_TIMING` | off | Print CPU/GPU prefill phase timings |
| `PROFILE_ABLATE` | off | Skip `attn` / `mlp` / `ple` for ablation |
| `BENCH_PREFILL_EXACT` | off | Truncate bench prompts to exact target length |

## Benchmarks

Prefill:

```bash
LLAMA_KV_CACHE_TYPE=q4_0 LLAMA_MAX_PREFILL_SEQ=4096 \
./target/release/llama-sinks \
  --gpu ~/Downloads/models/e2b/gemma-4-E2B-it-Q4_K_M.gguf \
  --bench-prefill --bench-prefill-tokens 2048,4096
```

Decode:

```bash
LLAMA_KV_CACHE_TYPE=q4_0 \
./target/release/llama-sinks \
  --gpu ~/Downloads/models/e2b/gemma-4-E2B-it-Q4_K_M.gguf \
  --bench-decode --bench-decode-tokens 25,200
```

Ballpark on Apple M1 Pro (E2B Q4_K_M, Q4_0 KV, cool machine): prefill ~580–590
tok/s @ 4k (matches llama.cpp FA); decode ~45–50 tok/s at short context (falls
with long context). Numbers move with thermal state — warm up or take the second
run.

## API example

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4",
    "messages": [{"role": "user", "content": "Explain KV cache reuse in one sentence."}],
    "max_tokens": 64,
    "temperature": 0.0
  }'
```

Streaming: add `"stream": true` and use `curl -N`.

Endpoints: `/v1/chat/completions`, `/v1/models`, `/health`, `/metrics`.

## HF safetensors (optional)

```bash
hf download google/gemma-4-e4b-it --local-dir ~/models/gemma-4-e4b-it

./target/release/llama-sinks --gpu ~/models/gemma-4-e4b-it --serve
```

Directory must contain `config.json`, `tokenizer.json`, and safetensors weights.
Prefer GGUF for quantized Metal runs.

## Tests

```bash
cargo test
```

With the server up:

```bash
python3 benchmarks/server_regression.py --port 8080 --requests 10 --max-tokens 32 --timeout 300
python3 benchmarks/prefill_correctness.py --port 8080 --max-tokens 24 --timeout 300
```

## Architecture

```text
src/main.rs                 CLI (GGUF vs HF dir, --serve, benches)
src/server.rs               OpenAI-compatible HTTP + runtime config
src/scheduler.rs            Prefill/decode scheduling
src/gemma4_gpu_model.rs     Load (GGUF/HF), prefill, decode, batching
src/gguf.rs                 GGUF parse + embedded tokenizer
src/gpu.rs                  Metal pipelines / encoders
src/shaders/                Metal kernels (mul_mm, flash_attn_ext, …)
benchmarks/                 Regression / correctness / stress scripts
```

## Known limits

- Local single-node server; not a clustered / multi-GPU runtime.
- Gemma4-focused (E2B / E4B); not a general multi-arch runtime.
- Context via `LLAMA_CTX_SIZE` (default 16k, up to 200k); quality may drop past
  the model’s trained length.
- Long prompts are chunked (`LLAMA_MAX_PREFILL_SEQ`); multi-chunk prefill is
  slower per token than a single 4k bench chunk.
- Decode throughput falls as KV grows; short-context benches overstate long-chat
  tok/s.

## Positioning

> A local Metal Gemma inference server with production-oriented continuous
> batching, native GGUF loading, and mmap zero-copy weights (cold load
> ~0.3s). Decode throughput tracks llama.cpp within a few tok/s at the
> context lengths we exercise.
