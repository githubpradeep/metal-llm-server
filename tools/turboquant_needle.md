# P0 — TurboQuant needle quality gate

Generation correctness beats attention cosine similarity. Run this before
claiming a `(K_bits, V_bits, residual_window)` config is demo-safe.

## Procedure

1. Build release with TQ wired:
   ```bash
   cargo build --release
   ```

2. Pick a config (start with the old-branch validated pair):
   ```bash
   export ATTENTION_KERNEL=specialized
   export LLAMA_KV_CACHE_TYPE=turboquant
   export TURBOQUANT_K_BITS=3
   export TURBOQUANT_V_BITS=2
   export TURBOQUANT_RESIDUAL_WINDOW=128
   export LLAMA_CTX_SIZE=8192
   MODEL="$HOME/Downloads/models/e2b/gemma-4-E2B-it-Q4_K_M.gguf"
   ```

3. Build a long prompt with a buried fact, e.g. pad with filler paragraphs and
   insert once:
   ```
   The secret project code name is AURORA-7749.
   ```
   Target total tokens ≈ 2048, 4096, then 8192 (stay under `LLAMA_CTX_SIZE`).

4. Ask greedily:
   ```
   What is the secret project code name? Reply with only the code name.
   ```

5. **PASS** iff the completion contains `AURORA-7749` exactly.
   Log: bits, rw, ctx, KV MB (from the runtime meter), tok/s, PASS/FAIL.

## Sweep matrix (minimum)

| K | V | rw | 2k | 4k | 8k |
|---|---|----|----|----|----|
| 3 | 2 | 128 | ? | ? | ? |
| 4 | 4 | 128 | ? | ? | ? |
| 6 | 4 | 128 | ? | ? | ? (if bits supported) |
| 2 | 2 | 128 | expect FAIL | | |
| 3 | 2 | 0 | expect FAIL / derail | | |

Fill the table from real runs; do not advertise FAIL configs.

## Baseline control

Repeat the same prompt with `LLAMA_KV_CACHE_TYPE=q4_0` — must PASS. If Q4 fails,
the prompt/tokenization is the bug, not TQ.

## Automation sketch

A future `tools/turboquant_needle.sh` can:
1. Generate padded prompt via Python/tokenizers
2. Pipe into `llama-sinks` CLI chat (or a small `--prompt`/`--ngen` mode)
3. Grep the needle in stdout
4. Append a TSV row to `benchmarks/turboquant_needle.tsv`

Until that script exists, run manually and paste results into the table above
or into `docs/TURBOQUANT.md`.
