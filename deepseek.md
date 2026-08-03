# DeepSeek V4 Flash — local instructions (this branch)

`llama-sinks` (this repo) runs **Gemma 4** today. It does **not** load DeepSeek V4
yet. On the `deepseek` branch we use the vendored **DwarfStar / ds4** engine under
`reference/ds4/` to run DeepSeek-V4-Flash on Apple Silicon.

This note is for our machines (including **16 GB M1 Pro**). Official ds4 docs
target 96–128 GB resident; below that you must use **SSD streaming**.

---

## Reality check (16 GB)

| Fact | Implication |
|------|-------------|
| Flash IQ2 GGUF ≈ **81–87 GB** on disk | Does not fit in RAM |
| Dense (non-expert) core in typical IQ2 is still large (~Q8 attn/shared) | Even streaming needs headroom for dense + KV + OS |
| `--ssd-streaming` pages **routed experts** from SSD | Generation will be slow; goal is “it runs”, not 30 tok/s |
| Prefer **short context** (`--ctx 512`–`2048`) and `--nothink` first | Leaves RAM for expert cache |

If streaming OOMs or thrash-swaps the machine, stop and use the DeepSeek API, or
stick to Gemma 4 A4B in `llama-sinks`.

---

## 1. Prerequisites

- Apple Silicon Mac + Xcode CLT
- ~100 GB free on the **internal** SSD (fast NVMe)
- Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
hf --version
```

Optional: `HF_TOKEN` if downloads are rate-limited.

---

## 2. Build ds4 (from this repo)

```bash
cd reference/ds4
make -j
ls -la ds4 ds4-server
```

Metal is the default on macOS. Run binaries from `reference/ds4` so relative
`metal/` shader paths resolve (or set absolute Metal source paths if you move
the binary).

Upstream project: [antirez/ds4](https://github.com/antirez/ds4) (DwarfStar).
Only use **ds4-specific** GGUFs from [antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf)
— generic llama.cpp DeepSeek GGUFs will not load correctly.

---

## 3. Download Flash weights (prefer 0731)

**Latest GA checkpoint** is DeepSeek-V4-Flash-**0731** (July 2026 re-post-train).
Prefer files with `0731` in the name.

### Recommended for ≤128 GB / streaming (smallest)

```bash
mkdir -p ~/models/dsv4-0731
cd ~/models/dsv4-0731

hf download antirez/deepseek-v4-gguf \
  DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --local-dir .
```

~86.7 GB. Asymmetric quant: routed experts IQ2 / Q2_K; attn / shared / out stay
higher precision.

### Slightly better quality (~98 GB)

```bash
hf download antirez/deepseek-v4-gguf \
  DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf \
  --local-dir ~/models/dsv4-0731
```

Last 6 expert layers Q4_K; rest IQ2. Still stream on low RAM.

### Convenience symlink (optional)

```bash
cd /path/to/mega-metal-llm-server/reference/ds4
ln -sfn ~/models/dsv4-0731/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  ./ds4flash.gguf
```

### Scripted download (preview-era names)

`reference/ds4/download_model.sh` can fetch older non-0731 imatrix files:

```bash
cd reference/ds4
./download_model.sh q2-imatrix    # ~81 GB preview IQ2
# ./download_model.sh mtp         # optional MTP sidecar (~3.5 GB)
```

Prefer the **explicit `hf download …0731…`** commands above until the script
defaults to 0731.

**Do not download** Q4-full (~165 GB) or Pro (~430 GB+) for a 16 GB Mac.

---

## 4. First run (16 GB — SSD streaming)

Close browsers and other heavy apps. Start **cold**, short context, no thinking:

```bash
cd /path/to/mega-metal-llm-server/reference/ds4

./ds4 \
  -m ~/models/dsv4-0731/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --ssd-streaming \
  --ctx 1024 \
  --nothink \
  --temp 1.0 \
  --top-p 1.0 \
  -n 64 \
  -p "Say hello in one short sentence."
```

Notes:

- `--ssd-streaming` keeps dense weights resident and pages routed experts.
- Omit `--ssd-streaming-cache-experts …` first so ds4 auto-sizes the expert
  cache from free Metal memory. On 16 GB it will be tiny.
- If auto cache is still too large / `mlock` fails, try an explicit small budget
  (ds4 converts this to a count of full experts):

```bash
./ds4 \
  -m ~/models/dsv4-0731/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --ssd-streaming \
  --ssd-streaming-cache-experts 2GB \
  --ctx 512 \
  --nothink \
  -n 32 \
  -p "Hello"
```

Watch the startup **cache report** line. Prefer a lockable expert cache over a
huge pageable one (paging experts through swap kills tok/s and can wedge the
machine).

### Expected behavior on 16 GB

- Prefill: painful but may finish for short prompts.
- Decode: often **≪ 1–few tok/s** when expert miss rate is high.
- Success = coherent short answers without OOM / kernel thrash.

---

## 5. Server mode (optional)

```bash
cd reference/ds4
mkdir -p /tmp/ds4-kv

./ds4-server \
  -m ~/models/dsv4-0731/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --ssd-streaming \
  --ctx 2048 \
  --kv-disk-dir /tmp/ds4-kv \
  --kv-disk-space-mb 4096 \
  --host 127.0.0.1 \
  --port 8000
```

Then hit the OpenAI-compatible endpoints ds4-server exposes (see
`reference/ds4/README.md`). Keep `--ctx` small on 16 GB.

Disk-backed KV is useful so long agent prompts do not eat the expert cache.

---

## 6. If you have 96–128 GB (resident path)

Skip streaming when the model fits:

```bash
cd reference/ds4
./download_model.sh q2-imatrix   # or hf download the 0731 IQ2 file
./ds4 -p "Explain Redis streams in one paragraph."
# or
./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

Official sampling tip: `temperature = 1.0`, `top_p = 1.0`. Think-max wants a
large context (hundreds of K) — not for 16 GB.

---

## 7. Optional MTP (ds4)

```bash
cd reference/ds4
./download_model.sh mtp
# or: hf download antirez/deepseek-v4-gguf DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf --local-dir ./gguf

./ds4 \
  -m ./ds4flash.gguf \
  --ssd-streaming \
  --mtp ./gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --mtp-draft 2 \
  --ctx 1024 \
  --nothink \
  -p "Hello"
```

On 16 GB, add MTP only after base streaming works — it costs more memory and
I/O.

---

## 8. What *not* to do

| Don’t | Why |
|-------|-----|
| Point `llama-sinks --gpu` at a DeepSeek GGUF | Wrong arch / tensors; unsupported |
| `llama-gguf r` on the 81 GB file | Can allocate anonymous heap and OOM the Mac |
| `mlock` / force full resident on 16 GB | Impossible |
| Start with Think Max + 100k ctx on 16 GB | Dense + KV + indexer will not fit |
| Use random HF GGUFs not from antirez/ds4 | Layout / quants / metadata mismatch |

---

## 9. Relation to llama-sinks / Gemma MoE

| | Gemma 4 A4B (`llama-sinks`) | DeepSeek-V4-Flash (`reference/ds4`) |
|--|-----------------------------|--------------------------------------|
| Engine | This repo’s Metal path | DwarfStar (`ds4`) |
| MoE | Softmax top-k + shared FFN + LFU slots | √softplus / hash-MoE + CSA/HCA + mHC |
| Fits 16 GB? | Yes (designed for it) | Only via SSD expert streaming (slow) |
| Docs | `README.md`, `docs/09_gemma4_moe.md` | `reference/ds4/README.md`, this file |

Porting Flash into `llama-sinks` is a separate multi-month effort (CSA/HCA,
mHC, new quants, routing). Until then: **run Flash via ds4; keep Gemma for
day-to-day local chat.**

---

## 10. Checklist

1. [ ] `cd reference/ds4 && make -j`
2. [ ] Download `…imatrix-0731.gguf` (~87 GB) to `~/models/dsv4-0731`
3. [ ] Run with `--ssd-streaming --ctx 1024 --nothink -n 64`
4. [ ] Confirm coherent short output
5. [ ] Only then raise `--ctx` or enable thinking / MTP

---

## Links

- Engine (vendored): `reference/ds4/`
- Upstream: https://github.com/antirez/ds4
- GGUFs: https://huggingface.co/antirez/deepseek-v4-gguf
- Official weights: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
- Gemma local server: [`README.md`](README.md)
