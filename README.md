# llama-sinks

**Fast local Gemma 4 inference on Apple Silicon.**

A from-scratch Rust + Metal engine with an OpenAI-compatible server. Load a
community GGUF, hit `/v1/chat/completions`, and chat — no Python, no CUDA, no
cloud.

Supports the Gemma 4 instruct family we care about day-to-day:

| Model | Kind | Typical Q4 size | MTP draft | Good for |
|-------|------|-----------------|-----------|----------|
| **E2B** | Dense ~2.3B | ~3 GB | Yes (F16) | Fast laptop chat; best-tested MTP path |
| **E4B** | Dense ~4.5B | ~5–6 GB | Yes (F16) | Default quality / speed balance |
| **12B** | Dense ~12B | ~7–8 GB | Yes (F16) | Stronger coding & reasoning on 16–32 GB Macs |
| **26B-A4B** | MoE ~26B / ~4B active | ~16 GB (UD-Q4_K_M) | Yes (sidecar) | Best quality we run locally; sparse experts |

Binary name: `llama-sinks`.

---

## Why use this instead of wrapping llama.cpp?

Most “local LLM on Mac” stacks are thin shells around llama.cpp or MLX. This
one is a **dedicated Metal runtime** built around Gemma 4:

1. **Native Metal path** — K-quant matvecs, flash attention, fused decode, and
   Gemma-specific pieces (SWA / shared-KV, PLE on edge models, A4B MoE) live in
   our shaders — not a generic portable graph.
2. **OpenAI API that behaves like a server** — KV pool, continuous batching,
   chunked prefill, admission control, `/metrics`. Point Cursor, Continue,
   Open WebUI, or `curl` at `localhost:8080`.
3. **Sub-second GGUF load** — mmap zero-copy weights. No separate “convert then
   cache” dance for day-to-day Q4_K_M runs.
4. **Long context on small Macs** — Q4_0 KV cache + hybrid attention
   (`ATTENTION_KERNEL=auto`) keep tok/s usable as context grows.
5. **Real MoE for 26B-A4B** — shared FFN ∥ routed experts with an LFU expert
   slot cache so A4B can run on unified memory instead of pretending MoE is
   just a dense MLP.
6. **MTP speculative decoding** — optional draft-head verify loop (`--mtp`) for
   E2B / E4B / 12B / A4B when the matching F16 MTP GGUF is present.
7. **Throughput in llama.cpp’s ballpark** — on M1 Pro class hardware, E2B
   prefill ~580–590 tok/s @ 4k and decode ~45–50 tok/s at short context (cool
   machine; thermals matter).

If you need every architecture under the sun, use llama.cpp. If you want a
**Gemma 4–first Metal server you can read and hack**, use this.

---

## Requirements

- Apple Silicon Mac (M1 / M2 / M3 / M4)
- macOS with Metal (we build with `MACOSX_DEPLOYMENT_TARGET=15.0`)
- Rust toolchain (`rustup`)
- [`huggingface-cli`](https://huggingface.co/docs/huggingface_hub/guides/cli)
  (or the `hf` CLI) for downloads
- Disk: ~4 GB (E2B) up to ~17 GB (A4B UD-Q4_K_M), plus headroom for KV

**Text-only for now.** Gemma 4 GGUFs may ship `mmproj` vision/audio sidecars;
this engine does not load them yet.

---

## 1. Install Hugging Face CLI

```bash
pip install -U "huggingface_hub[cli]"
# or: brew install huggingface-cli
hf --version
```

---

## 2. Download a GGUF from Hugging Face

We recommend **Unsloth** instruct GGUFs. Pick one size:

### E2B (~3 GB) — fastest

```bash
mkdir -p ~/models/gemma-4-e2b
hf download unsloth/gemma-4-E2B-it-GGUF \
  gemma-4-E2B-it-Q4_K_M.gguf \
  --local-dir ~/models/gemma-4-e2b
```

Repo: [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF)

### E4B (~5–6 GB) — recommended default

```bash
mkdir -p ~/models/gemma-4-e4b
hf download unsloth/gemma-4-E4B-it-GGUF \
  gemma-4-E4B-it-Q4_K_M.gguf \
  --local-dir ~/models/gemma-4-e4b
```

Repo: [unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF)

### 12B (~7–8 GB)

```bash
mkdir -p ~/models/gemma-4-12b
hf download unsloth/gemma-4-12B-it-GGUF \
  gemma-4-12b-it-Q4_K_M.gguf \
  --local-dir ~/models/gemma-4-12b
```

Repo: [unsloth/gemma-4-12B-it-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-GGUF)

### 26B-A4B MoE (~16 GB) — Unsloth Dynamic Q4

```bash
mkdir -p ~/models/gemma-4-a4b
hf download unsloth/gemma-4-26B-A4B-it-GGUF \
  gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  --local-dir ~/models/gemma-4-a4b
```

Repo: [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)

**RAM note:** A4B UD-Q4_K_M wants a comfortable **16 GB+** machine (expert LFU
cache + KV). On a busy 16 GB Mac, close other apps and keep context modest
(`LLAMA_CTX_SIZE=8192` or `16384`). See [`docs/09_gemma4_moe.md`](docs/09_gemma4_moe.md)
for how MoE works in this engine.

### Which should I pick?

| Your Mac | Start with |
|----------|------------|
| 8 GB | E2B |
| 16 GB | E4B or 12B; A4B possible but tight |
| 32 GB+ | 12B or A4B |

---

## 3. Build

```bash
git clone git@github.com:githubpradeep/metal-llm-server.git
cd metal-llm-server
export MACOSX_DEPLOYMENT_TARGET=15.0
cargo build --release
```

Binary: `./target/release/llama-sinks`.

---

## 4. Run the server

Recommended defaults for chat:

```bash
export ATTENTION_KERNEL=auto
export LLAMA_KV_CACHE_TYPE=q4_0
export LLAMA_CTX_SIZE=32768
export LLAMA_MAX_PREFILL_SEQ=4096

# Swap the path for the model you downloaded
./target/release/llama-sinks \
  --gpu ~/models/gemma-4-e4b/gemma-4-E4B-it-Q4_K_M.gguf \
  --serve
```

Listens on `http://0.0.0.0:8080` (`--port N` to change).

### One-liners per model

```bash
# E2B
./target/release/llama-sinks --gpu ~/models/gemma-4-e2b/gemma-4-E2B-it-Q4_K_M.gguf --serve

# E4B
./target/release/llama-sinks --gpu ~/models/gemma-4-e4b/gemma-4-E4B-it-Q4_K_M.gguf --serve

# 12B
./target/release/llama-sinks --gpu ~/models/gemma-4-12b/gemma-4-12b-it-Q4_K_M.gguf --serve

# 26B-A4B (MoE)
./target/release/llama-sinks --gpu ~/models/gemma-4-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --serve
```

(Still export `ATTENTION_KERNEL=auto` and `LLAMA_KV_CACHE_TYPE=q4_0` in the same
shell for best results.)

---

## 5. MTP speculative decoding (optional)

**MTP** (multi-token prediction) uses a small **draft head** GGUF alongside the
main model. The draft proposes several tokens; the base model verifies them in
a batched forward. Accepted tokens are emitted without paying full single-token
decode for each one.

When it helps: interactive chat / coding where accept rate stays healthy
(~40%+). On E2B Q4_K_M + F16 draft we typically land near non-MTP decode
throughput or a bit under/over depending on accept rate — see `AGENTS.md` MTP
notes for the latest numbers.

### Download the matching draft head

Draft files live in the same Unsloth repos as the base GGUFs (often named
`mtp-gemma-4-…`):

```bash
# E2B (~160 MB)
hf download unsloth/gemma-4-E2B-it-GGUF \
  mtp-gemma-4-E2B-it-F16.gguf \
  --local-dir ~/models/gemma-4-e2b

# E4B (~160 MB)
hf download unsloth/gemma-4-E4B-it-GGUF \
  mtp-gemma-4-E4B-it-F16.gguf \
  --local-dir ~/models/gemma-4-e4b

# 12B (~800 MB)
hf download unsloth/gemma-4-12B-it-GGUF \
  mtp-gemma-4-12b-it-F16.gguf \
  --local-dir ~/models/gemma-4-12b


```

Always pair a draft with **the same model family** as the base GGUF (E2B draft
with E2B base, etc.).

### Serve with MTP

Pass `--mtp <draft.gguf>` together with `--serve`:

```bash
export ATTENTION_KERNEL=auto
export LLAMA_KV_CACHE_TYPE=q4_0

# E2B + MTP (most exercised path)
./target/release/llama-sinks \
  --gpu ~/models/gemma-4-e2b/gemma-4-E2B-it-Q4_K_M.gguf \
  --mtp ~/models/gemma-4-e2b/mtp-gemma-4-E2B-it-F16.gguf \
  --serve

# E4B + MTP
./target/release/llama-sinks \
  --gpu ~/models/gemma-4-e4b/gemma-4-E4B-it-Q4_K_M.gguf \
  --mtp ~/models/gemma-4-e4b/mtp-gemma-4-E4B-it-F16.gguf \
  --serve

# 12B + MTP
./target/release/llama-sinks \
  --gpu ~/models/gemma-4-12b/gemma-4-12b-it-Q4_K_M.gguf \
  --mtp ~/models/gemma-4-12b/mtp-gemma-4-12b-it-F16.gguf \
  --serve
```

Same OpenAI API as without MTP — clients do not change.

**Important:** MTP serve uses a **serial** scheduler (one request at a time).
Concurrent chat slots fall back to FIFO queueing. For multi-user batching,
omit `--mtp`.

### CLI generate with MTP

```bash
ATTENTION_KERNEL=auto LLAMA_KV_CACHE_TYPE=q4_0 \
  ./target/release/llama-sinks \
  --gpu ~/models/gemma-4-e2b/gemma-4-E2B-it-Q4_K_M.gguf \
  --mtp ~/models/gemma-4-e2b/mtp-gemma-4-E2B-it-F16.gguf
```

### MTP environment knobs

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLAMA_MTP_DRAFT_STEPS` | (engine default) | Max draft tokens per verify (capped by verify seq ≤ 8) |
| `LLAMA_MTP_ADAPTIVE` | off | Adaptive draft length |
| `LLAMA_MTP_P_MIN` | off | Stop drafting when top-token confidence &lt; threshold |
| `LLAMA_MTP_DRAFT_TOP_K` | `10` | Softmax over top-k for draft confidence (`0` = full vocab) |
| `LLAMA_MTP_DEBUG` | off | Verbose draft/accept logging |
| `MTP_VERIFY_CROSSCHECK` | off | Assert parallel verify == sequential (debug) |

Start simple: base Q4_K_M + matching F16 draft, `ATTENTION_KERNEL=auto`, no extra
MTP env vars. Add `LLAMA_MTP_ADAPTIVE=1` / `LLAMA_MTP_P_MIN=0.5` only when
tuning accept rate vs speed.

---

## 6. Talk to it

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4",
    "messages": [{"role": "user", "content": "Explain KV cache reuse in one sentence."}],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

Streaming: add `"stream": true` and use `curl -N`.

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/chat/completions` | Chat (sync or SSE) |
| `GET /v1/models` | List models |
| `GET /health` | Liveness |
| `GET /metrics` | Prometheus-style metrics |

Any OpenAI-compatible client works if you set `base_url=http://127.0.0.1:8080/v1`.

---

## Quick CLI generate (no server)

```bash
ATTENTION_KERNEL=auto LLAMA_KV_CACHE_TYPE=q4_0 \
  ./target/release/llama-sinks \
  --gpu ~/models/gemma-4-e4b/gemma-4-E4B-it-Q4_K_M.gguf
```

---

## Benchmarks

```bash
# Prefill
LLAMA_KV_CACHE_TYPE=q4_0 LLAMA_MAX_PREFILL_SEQ=4096 \
./target/release/llama-sinks \
  --gpu ~/models/gemma-4-e2b/gemma-4-E2B-it-Q4_K_M.gguf \
  --bench-prefill --bench-prefill-tokens 2048,4096

# Decode
LLAMA_KV_CACHE_TYPE=q4_0 ATTENTION_KERNEL=auto \
./target/release/llama-sinks \
  --gpu ~/models/gemma-4-e2b/gemma-4-E2B-it-Q4_K_M.gguf \
  --bench-decode --bench-decode-tokens 25,200
```

Ballpark **M1 Pro, E2B Q4_K_M, Q4_0 KV, cool machine**: prefill ~580–590 tok/s
@ 4k; decode ~45–50 tok/s at short context (falls as KV grows).

---

## Important environment variables

| Variable | Recommended | Meaning |
|----------|-------------|---------|
| `ATTENTION_KERNEL` | `auto` | Hybrid fused / ggml attention |
| `LLAMA_KV_CACHE_TYPE` | `q4_0` | KV quant (`q4_0` / `q8_0` / omit for f16) |
| `LLAMA_CTX_SIZE` | `32768` | Context / KV capacity (max `200000`) |
| `LLAMA_MAX_PREFILL_SEQ` | `4096` | Prefill chunk size |
| `LLAMA_KV_POOL_SLOTS` | (default) | Concurrent request slots |
| `LLAMA_QUEUE_DEPTH` | (default) | Admission queue depth |

A4B MoE knobs (optional): `MOE_EXPERT_SLOTS`, `MOE_PROFILE=1`, `MOE_DISABLE=1`
— see [`docs/09_gemma4_moe.md`](docs/09_gemma4_moe.md).

---


## Architecture (short)

```text
Client  →  axum OpenAI API  →  scheduler / KV pool
                →  Gemma4 Metal model (GGUF mmap)
                →  shaders/*.metal
```

| Area | Docs |
|------|------|
| Engine overview | [`docs/01_engine_overview.md`](docs/01_engine_overview.md) |
| Decode path | [`docs/02_decode_path.md`](docs/02_decode_path.md) |
| A4B MoE deep dive | [`docs/09_gemma4_moe.md`](docs/09_gemma4_moe.md) |

---

## Known limits

- **Apple Silicon + Metal only** — not CUDA / Linux GPU.
- **Gemma 4 text** — E2B / E4B / 12B / 26B-A4B; not a general multi-arch zoo.
- **No vision/audio** yet (ignore `mmproj` sidecars).
- **MTP serve is serial** — with `--mtp`, requests run one-at-a-time (FIFO).
  Omit `--mtp` for multi-slot continuous batching.
- Context quality past the model’s trained window may drop even if
  `LLAMA_CTX_SIZE` allows larger KV.
- Decode tok/s falls as context grows; short-context benches overstate long chat.

---

## License / models

Engine code: see repository license. Model weights are Google Gemma 4 (check
each Hugging Face card for terms). Unsloth GGUFs are redistributions of those
weights in quantized form.
