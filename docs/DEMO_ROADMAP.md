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

## Phase 2 — TurboQuant KV (in progress on `feature/turboquant-kv`)

One-liner: **“100k-token Gemma chat in ~2 GB-class KV — needle still found. Watch the meter.”**

See `docs/TURBOQUANT.md` for config, phase map (P0–P4), and honest claim bounds.

| Phase | Goal |
|---|---|
| P0 | Needle quality gate (`tools/turboquant_needle.md`) |
| P1–P2 | Port from `mega-kernel-gguf-turboquant-1`: Lloyd–Max V3 + Metal fused attn, residual window, asymmetric K/V |
| P3 | Live KV meter + `kv_persist` type tag + CLI/Brain |
| P4 | Disk-backed cold TQ pages for 1M-class docs (stub — placement only changes speed) |

`kv_persist.rs` keeps a `kv_type` byte so TQ sessions do not silently load as Q4_0.
