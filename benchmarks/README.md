# Gemma4 Server Benchmarks and Regression Checks

Start the server first:

```bash
export MODEL_DIR=/path/to/gemma-model-dir

MACOSX_DEPLOYMENT_TARGET=15.0 \
LLAMA_QUEUE_DEPTH=64 \
LLAMA_KV_POOL_SLOTS=32 \
LLAMA_PREFILL_TOKENS_PER_TICK=128 \
LLAMA_REQUEST_TIMEOUT_SECS=300 \
cargo run --release -- --gpu --serve "$MODEL_DIR" --port 8080
```

## Full Strong Regression

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

This checks:

- health and model listing
- structured request errors
- sync chat
- streaming chat
- chunked prefill
- concurrent requests
- sequential vs concurrent prefill correctness
- reversed-order batched prefill correctness
- mixed prefill/decode fairness
- scheduler stress
- idle metrics

## Prefill Correctness

```bash
python3 benchmarks/prefill_correctness.py \
  --port 8080 \
  --max-tokens 24 \
  --timeout 300
```

This compares greedy sequential baseline outputs against concurrent requests
across tiny, short, medium, chunk-boundary, and long prompts. It also verifies
that prefill batch counters prove multi-request prefill batching happened.

## Scheduler Stress

```bash
python3 benchmarks/stress_scheduler.py \
  --port 8080 \
  --requests 24 \
  --timeout 300
```

This exercises mixed prompt lengths, staggered arrivals, stream cancellation,
client timeout pressure, batching metrics, and final idle gauges.

## Mixed Prefill/Decode Fairness

```bash
python3 benchmarks/mixed_batching_fairness.py \
  --port 8080 \
  --stream-tokens 64 \
  --prefill-words 180 \
  --prefill-max-tokens 1 \
  --timeout 300 \
  --max-stream-gap 5.0
```

Long-context smoke around 3500 prompt tokens:

```bash
python3 benchmarks/mixed_batching_fairness.py \
  --port 8080 \
  --stream-tokens 32 \
  --prefill-words 900 \
  --prefill-max-tokens 1 \
  --timeout 300 \
  --max-stream-gap 8.0
```

Prompts above 4096 tokens should be rejected cleanly with
`context_length_exceeded`.

## Prefill Throughput Benchmark

```bash
python3 benchmarks/prefill_benchmark.py \
  --port 8080 \
  --sizes 32,64,128,256,512 \
  --repeats 3
```

This uses `/metrics` deltas to report prompt prefill tokens/sec, prefill
chunks, and end-to-end request latency for each prompt size.

## Metrics

```bash
curl -s http://127.0.0.1:8080/metrics \
  | grep -E 'llama_(prefill|decode)_batch_items_(avg|max)|llama_.*latency_ms_(avg|max)'
```

Useful metrics:

- `llama_prefill_batch_items_avg`
- `llama_prefill_batch_items_max`
- `llama_decode_batch_items_avg`
- `llama_decode_batch_items_max`
- `llama_request_latency_ms_avg`
- `llama_request_latency_ms_max`
- `llama_prefill_latency_ms_avg`
- `llama_prefill_latency_ms_max`
- `llama_decode_latency_ms_avg`
- `llama_decode_latency_ms_max`
- `llama_decode_compute_latency_ms_avg`
- `llama_decode_compute_latency_ms_max`

## Runtime Knobs

```bash
LLAMA_QUEUE_DEPTH=64
LLAMA_KV_POOL_SLOTS=32
LLAMA_REQUEST_TIMEOUT_SECS=300
LLAMA_PREFILL_TOKENS_PER_TICK=128
```

`LLAMA_PREFILL_TOKENS_PER_TICK` caps scheduler prefill work per tick. Lower
values improve decode interleaving under long prefill load; higher values may
improve prompt throughput.
