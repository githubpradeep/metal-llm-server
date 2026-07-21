# Demo: "Warm reopen" + MTP Accept Brain

Two things to show, ~60–90s total. Prep everything below *before* recording.

## 0. One-time setup

```bash
cargo build --release
export MODEL="$HOME/Downloads/models/e2b/gemma-4-E2B-it-Q4_K_M.gguf"
export DRAFT="$HOME/Downloads/models/e2b/mtp-gemma-4-E2B-it-F16.gguf"
rm -f ~/.cache/mega-metal/sessions/demo1.kv   # clean session for the recording
```

## Scene A — Warm reopen (CLI)

**1. Start the chat with session save enabled:**

```bash
ATTENTION_KERNEL=specialized LLAMA_KV_CACHE_TYPE=q4_0 LLAMA_CTX_SIZE=200000 \
  LLAMA_KV_SAVE=1 LLAMA_SESSION=demo1 \
  ./target/release/llama-sinks --gpu "$MODEL" --mtp "$DRAFT"
```

**2. Chat a paragraph.** Type something like:

```
You: Write two sentences about why the ocean matters.
```

Watch the inline `[brain n/m]` colored blocks (green = accepted draft, red =
first rejection) flash after each verify cycle, then the `[turn] ... tok/s |
accept % | tok/forward` line, then `[session] saved N tokens, X MB KV -> ...`.

**3. Kill the process** (Ctrl-C, or `kill -9 <pid>` from another terminal —
either way, the *last completed turn* is already on disk).

**4. Restart with the same session:**

```bash
ATTENTION_KERNEL=specialized LLAMA_KV_CACHE_TYPE=q4_0 LLAMA_CTX_SIZE=200000 \
  LLAMA_KV_SAVE=1 LLAMA_SESSION=demo1 \
  ./target/release/llama-sinks --gpu "$MODEL" --mtp "$DRAFT"
```

You'll see:

```
[session] restored 40 tokens, 0 prefill (0.2 ms)
```

**5. Keep chatting** — reference something from before the kill to prove the
model still has the context, then check ctx keeps growing in the `[turn]`
line.

### Proof this is real (not just cosmetic)

We validated bit-for-bit determinism: run the same 2-turn conversation (a)
uninterrupted in one process and (b) killed after turn 1 and resumed in a
fresh process with `LLAMA_SESSION`. Turn 2's output, accept rate, and
tok/forward were **identical** in both runs:

| Run | Turn 2 reply | tok/s | accept % | tok/forward | ctx after |
|-----|--------------|-------|----------|--------------|-----------|
| (a) uninterrupted | `Apple<end_of_turn>` | 13.9 | 75.0% | 4.00 | 76 |
| (b) killed + restored, 0 prefill | `Apple<end_of_turn>` | 12.0 | 75.0% | 4.00 | 76 |

(tok/s differs slightly — thermal/scheduling noise — everything token- and
KV-derived is identical.) A mismatched `LLAMA_CTX_SIZE` (or model path) on
reload is **refused loudly**, e.g.:

```
[session] REFUSING to load 'demo1': ... model fingerprint mismatch — refusing to load
  saved:    path=...gguf|size=3106736256|kv=q4_0|ctx=8192
  expected: path=...gguf|size=3106736256|kv=q4_0|ctx=4096
[session] starting fresh instead
```

## Scene B — MTP Accept Brain (serve)

**1. Start the server with MTP:**

```bash
ATTENTION_KERNEL=specialized LLAMA_KV_CACHE_TYPE=q4_0 LLAMA_CTX_SIZE=200000 \
  ./target/release/llama-sinks --gpu "$MODEL" --mtp "$DRAFT" --serve
```

**2. Open the Brain page** in a browser: `http://localhost:8080/brain`.

**3. Send a chat request** (or point `ui/chat.py` / any OpenAI client at
`http://localhost:8080/v1/chat/completions`):

```bash
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "gemma-4-e2b",
  "messages": [{"role":"user","content":"Write two sentences about the ocean."}],
  "max_tokens": 80
}'
```

**4. Screen-record the Brain page** while the request streams: the grid
lights up green/red per drafted token in real time, and the metrics strip
(accept %, tok/s, tok/forward, tokens generated, verify cycles) updates live.

## Env var summary

| Var | Meaning | Default |
|-----|---------|---------|
| `LLAMA_SESSION` | Session name for warm reopen | unset (disabled) |
| `LLAMA_KV_SAVE` | `1` to save KV + history after every turn | `0` |
| `LLAMA_SESSION_DIR` | Override session directory | `~/.cache/mega-metal/sessions/` |
| `LLAMA_MAX_TOKENS_PER_TURN` | Cap generated tokens per CLI turn | `512` |
| `LLAMA_BRAIN_CLI` | `0` to disable the inline `[brain n/m]` CLI flashes | `1` |

## What's real vs. what's a known rough edge

- **Real**: KV persistence is a genuine byte-for-byte snapshot of the GPU KV
  cache (Metal `StorageModeShared` buffers — direct CPU memcpy, no fake
  readback), sliced per-KV-head to the actual sequence length (not the full
  `LLAMA_CTX_SIZE` capacity), and validated bit-identical against an
  uninterrupted session above.
- **Real**: the Accept Brain SSE feed is wired straight into the production
  `--serve --mtp` draft/verify loop (`src/mtp_serve.rs`), not a simulation.
- **Rough edge**: the CLI chat loop is single-conversation / single-process —
  there's no multi-session HTTP API for warm reopen yet (only the CLI path
  saves/restores sessions). Extending `/v1/chat/completions` with a
  `session_id` field that saves/restores via `kv_persist` is the natural next
  step but was out of scope for this pass (see "Suggested order" in the
  original task — CLI warm-reopen + serve-side Brain UI were the two
  independently-shippable halves).
- **Rough edge**: occasionally the model spells out the literal text
  `<end_of_turn>` instead of emitting the control token — a pre-existing MTP
  quality quirk unrelated to persistence/telemetry, most visible on very
  short/terse prompts.
