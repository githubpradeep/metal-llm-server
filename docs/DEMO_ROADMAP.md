# Demo Roadmap

## Phase 1 — shipped: "Warm reopen" + MTP Accept Brain

- **Warm reopen** (`src/kv_persist.rs`): kill the CLI MTP chat process, restart it
  with the same `LLAMA_SESSION`, and it restores the KV cache + token history
  straight to GPU with **zero re-prefill** of the restored prefix. Verified
  greedy-token-identical vs an uninterrupted session (same reply, same accept
  rate, same tok/forward — see `docs/DEMO.md`).
- **MTP Accept Brain** (`src/mtp_serve.rs` SSE + `ui/brain.html`): live view of
  every draft/verify cycle while `--serve --mtp` is running — a block per
  drafted token, green if the big model kept it, red on first rejection, plus
  a metrics strip (accept %, tok/s, tok/forward).
- CLI chat also prints a compact colored accept/reject strip per cycle
  (`LLAMA_BRAIN_CLI=0` to disable).

See `docs/DEMO.md` for the exact commands and a 60–90s screen-record script.

## Phase 2 — NOT implemented, stub only

### Moonshot: extreme KV compression ("turboquant")

One-liner: compress the on-disk (and eventually on-GPU) KV cache far below
Q4_0 — think learned/vector-quantized KV codebooks or residual coding — so a
200k-token session fits on disk/unified memory in tens of MB instead of
hundreds, making warm-reopen sessions and long-context chat viable at scale.

Not started. No format changes, no kernels, no integration in this pass —
`kv_persist.rs`'s on-disk format has a `kv_type` byte reserved specifically so
a future turboquant codec can be added as a new tag without breaking the
existing Q4_0/Q8_0/F16 session files.
