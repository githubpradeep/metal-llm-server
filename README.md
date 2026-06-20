# Gemma4 Metal Inference Server (Rust)

Alpha-stage local LLM inference server for Gemma4 E4B on Apple Silicon. The goal
is to keep the stack understandable while implementing production-oriented
serving pieces: an OpenAI-compatible API, Metal GPU kernels, KV pooling,
continuous batching, chunked prefill, and scheduler observability.

This is suitable for experimentation and technical preview use. It is not yet a
production-stable serving system.

## Current Status

Implemented:

- Gemma4 E4B GPU inference on Metal.
- Q4_0 model weights with selected f16 paths for quality-sensitive operations.
- OpenAI-compatible `/v1/chat/completions`, `/v1/models`, `/health`, and
  `/metrics` endpoints.
- Non-streaming and SSE streaming chat completions.
- KV cache pooling for concurrent requests.
- Real decode batching across active requests.
- Real multi-request batched prefill across KV slots.
- Chunked prefill for long prompts.
- Mixed prefill/decode scheduling with a per-tick prefill token budget.
- Denser prefill scheduling so one long prompt does not monopolize a tick when
  other prompts are prefilling.
- Runtime knobs for queue depth, KV pool slots, request timeout, and prefill
  token budget.
- Rich Prometheus-style metrics for batch size and latency.
- 4096-token server context cap for Gemma4 E4B in the current build.
- Single-request parallel prefill chunk path optimized into one Metal command
  buffer across the layer stack.

Validated locally:

- Full regression suite: health/models, structured errors, sync chat, chunked
  prefill, streaming, 10-way concurrency, prefill correctness, mixed fairness,
  stress, and idle metrics.
- Sequential vs concurrent greedy correctness across varied prompt shapes and
  reversed request order.
- Stress tests for mixed prompt lengths, staggered arrivals, stream
  cancellation, client timeout pressure, and scheduler idle recovery.
- Long-context acceptance around 3500 prompt tokens and structured rejection
  above the 4096-token context limit.

Still pending:

- Make KV capacity configurable and memory-aware instead of hardcoding 4096.
- Move the single-command-buffer optimization into the multi-request batched
  prefill path.
- Use or remove the experimental tiled projection kernels after benchmarking.
- Benchmark and tune the tiled FlashAttention-style attention kernels further.
- Add deeper logits/top-k debug correctness checks.
- Publish stable benchmark numbers with exact hardware, model, and runtime
  configuration.

## Build

```bash
git clone git@github.com:githubpradeep/metal-llm-server.git
cd metal-llm-server
export MACOSX_DEPLOYMENT_TARGET=15.0
cargo build --release
```

The model directory should contain:

- `config.json`
- `tokenizer.json`
- `model.safetensors` or sharded safetensors with `model.safetensors.index.json`

## Download Model From Hugging Face

Install the Hugging Face CLI:

```bash
python3 -m pip install -U huggingface_hub
```

If the model is gated, log in first and make sure you have accepted the model
license on Hugging Face:

```bash
hf auth login
```

Download the model to a local directory:

```bash
mkdir -p ~/models

hf download google/gemma-4-e4b-it \
  --local-dir ~/models/gemma-4-e4b-it
```

Then point the server at that directory:

```bash
export MODEL_DIR=~/models/gemma-4-e4b-it
```

If you use a different Gemma4 E4B checkpoint or an already-downloaded local
snapshot, set `MODEL_DIR` to the folder containing `config.json`,
`tokenizer.json`, and the safetensors files.

## Run Server

```bash
export MODEL_DIR=/path/to/gemma-model-dir

MACOSX_DEPLOYMENT_TARGET=15.0 \
LLAMA_QUEUE_DEPTH=64 \
LLAMA_KV_POOL_SLOTS=32 \
LLAMA_PREFILL_TOKENS_PER_TICK=128 \
LLAMA_REQUEST_TIMEOUT_SECS=300 \
cargo run --release -- --gpu --serve "$MODEL_DIR" --port 8080
```

Important runtime knobs:

```bash
LLAMA_QUEUE_DEPTH=64
LLAMA_KV_POOL_SLOTS=32
LLAMA_REQUEST_TIMEOUT_SECS=300
LLAMA_PREFILL_TOKENS_PER_TICK=128
```

`LLAMA_PREFILL_TOKENS_PER_TICK` controls how much prompt prefill work the
scheduler can submit per tick. Lower values improve decode interleaving under
long prefill load; higher values may improve prompt throughput.

## API Example

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-e4b-q4",
    "messages": [{"role": "user", "content": "Explain KV cache reuse in one sentence."}],
    "max_tokens": 64,
    "temperature": 0.0
  }'
```

Streaming:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-e4b-q4",
    "messages": [{"role": "user", "content": "Count from one to five."}],
    "max_tokens": 32,
    "stream": true
  }'
```

## Test Commands

Core Rust tests:

```bash
cargo test
```

Full server regression after starting the server:

```bash
python3 benchmarks/server_regression.py \
  --port 8080 \
  --requests 10 \
  --max-tokens 32 \
  --timeout 300 \
  --prefill-correctness \
  --mixed-fairness \
  --stress
```

Standalone correctness:

```bash
python3 benchmarks/prefill_correctness.py \
  --port 8080 \
  --max-tokens 24 \
  --timeout 300
```

Standalone stress:

```bash
python3 benchmarks/stress_scheduler.py \
  --port 8080 \
  --requests 24 \
  --timeout 300
```

Long-context smoke:

```bash
python3 benchmarks/mixed_batching_fairness.py \
  --port 8080 \
  --stream-tokens 32 \
  --prefill-words 900 \
  --prefill-max-tokens 1 \
  --timeout 300 \
  --max-stream-gap 8.0
```

This has produced prompts around 3500 tokens locally. Prompts above 4096 tokens
are expected to fail cleanly with `context_length_exceeded`.

## Metrics

The server exposes Prometheus-style metrics at `/metrics`.

Useful batching and latency metrics:

```bash
curl -s http://127.0.0.1:8080/metrics \
  | grep -E 'llama_(prefill|decode)_batch_items_(avg|max)|llama_.*latency_ms_(avg|max)'
```

Examples:

- `llama_prefill_batch_items_avg`
- `llama_prefill_batch_items_max`
- `llama_decode_batch_items_avg`
- `llama_decode_batch_items_max`
- `llama_request_latency_ms_avg`
- `llama_prefill_latency_ms_avg`
- `llama_decode_latency_ms_avg`
- `llama_decode_compute_latency_ms_avg`

## Architecture

Key files:

```text
src/main.rs                 CLI entry point and model detection
src/server.rs               OpenAI-compatible HTTP server and runtime config
src/scheduler.rs            Request admission, prefill/decode scheduling
src/batch_engine.rs         KV pool bridge for batched model calls
src/kv_pool.rs              Per-request KV cache slots
src/gemma4_gpu_model.rs     Gemma4 model loading, prefill, decode, batching
src/gpu.rs                  Metal pipeline setup and encoder helpers
src/shaders/llama.metal     Metal compute kernels
src/metrics.rs              Prometheus-style counters and gauges
benchmarks/                 Regression, correctness, stress, fairness scripts
docs/                       Architecture notes and blog posts
```

## Known Limits

- Alpha software. Expect sharp edges.
- Gemma4 E4B-focused; not a general model runtime.
- Current server context cap is 4096 tokens.
- Near-4096 prompts can be slow enough to require a larger
  `LLAMA_REQUEST_TIMEOUT_SECS`.
- Multi-request batched prefill is real, but the latest single-command-buffer
  optimization is not yet applied to that path.
- Tiled projection kernels are present but not currently used in the hot path.
- Decode and causal attention use tiled FlashAttention-style kernels by default
  (`FLASH_ATTN=legacy` to revert to the older per-token attention path).

## Positioning

Recommended public framing:

> Alpha: a local Metal Gemma inference server with production-oriented
> continuous batching experiments.

Avoid calling it production-ready until long-context performance, configurability,
and broader soak testing are complete.
