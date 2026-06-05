# Gemma4 E4B Inference Quality Benchmark

## Local server regression smoke test

Start the server first:

```bash
cargo run --release -- --port 8080
```

Then run the regression suite:

```bash
python3 benchmarks/server_regression.py --port 8080 --requests 10 --max-tokens 32
```

This checks health/model listing, structured request errors, non-streaming chat,
streaming chat, concurrent requests, stop-token trimming, and idle scheduler
gauges.

## Prefill throughput benchmark

Start the server first, then run:

```bash
python3 benchmarks/prefill_benchmark.py --port 8080 --sizes 32,64,128,256,512 --repeats 3
```

The benchmark uses `/metrics` deltas to report prompt prefill tokens/sec,
prefill chunks, and end-to-end request latency for each prompt size.
Set `LLAMA_MAX_PREFILL_SEQ=64` or `LLAMA_MAX_PREFILL_SEQ=128` before starting
the server to compare prefill chunk sizes.

## How to compare your local model against llama.cpp

### Step 1: Run the Colab script (llama.cpp reference on GPU)
Open `benchmark_llamacpp_colab.ipynb` in Google Colab with a T4/A100 GPU.
It will:
- Download Gemma4 E4B Q4 GGUF
- Run greedy decode on 10 fixed prompts
- Save the outputs to `reference_outputs.json`

### Step 2: Run the local test against your server
```bash
# Start your server first:
# cargo run --release -- --gpu --serve ~/Downloads/models--google--gemma-4-E4B-it/...

# Then run the comparison:
python3 benchmarks/benchmark_local.py
```

### Step 3: Compare
```bash
python3 benchmarks/compare_outputs.py
```

This compares token-by-token at temperature=0 (greedy).
Any differences indicate quantization divergence (expected for first few tokens)
or implementation bugs (if outputs completely differ).
