# TurboQuant KV compression (P0–P4)

Dense-model analog of Colibri’s “experts on disk”: **KV is the expert**.
Compress the long-context cache so large chats fit in tiny unified memory, with
a live meter and warm-reopen persistence.

Paper / refs: [TurboQuant](https://arxiv.org/abs/2504.19874),
`reference/turboquant-pytorch`, `reference/turboquant-mlx`, prior branch
`mega-kernel-gguf-turboquant-1`.

## Honest claim (demo)

> **100k-token Gemma chat in ~2 GB-class KV (SWA + TQ on full layers) — needle still found. Watch the meter.**

Do **not** promise “1M tokens in 2 GB resident” without the P4 disk tier.
vs F16 the quality-safe configs are ~2–5×; vs our existing Q4_0 another ~1.3–2×.

## Config

```bash
ATTENTION_KERNEL=specialized \
LLAMA_KV_CACHE_TYPE=turboquant \
TURBOQUANT_K_BITS=3 TURBOQUANT_V_BITS=2 \
TURBOQUANT_RESIDUAL_WINDOW=128 \
TURBOQUANT_HOT_WINDOW=2048 \
LLAMA_CTX_SIZE=8192 \
./target/release/llama-sinks \
  --gpu "$HOME/Downloads/models/e2b/gemma-4-E2B-it-Q4_K_M.gguf"
```

| Env | Meaning |
|---|---|
| `LLAMA_KV_CACHE_TYPE=turboquant` | Enable TQ path |
| `TURBOQUANT_BITS` | Symmetric bit-width (fallback) |
| `TURBOQUANT_K_BITS` / `V_BITS` | Asymmetric (prefer K ≥ V) |
| `TURBOQUANT_RESIDUAL_WINDOW` | Recent tokens kept fp32 in Haar frame (default 128) |
| `TURBOQUANT_HOT_WINDOW` | Model-frame Q4_0 ring for fast attn/prefill (default **2048**; `0` = pure TQ) |
| `TURBOQUANT_DUAL_WRITE` | `1` = also pack TQ cold while in hot window (needed before ctx > hot). Default off for speed. |

**Recommended demo:** `K3/V2` + `rw=128` + `hot=2048` (decode ≈ Q4 flash while ctx ≤ hot).  
**Quality-safer:** `K6/V4` + `rw=128` if available / when raising bits.

**Speed model:** While `attn_kv_seq ≤ TURBOQUANT_HOT_WINDOW`, fused decode uses the same Q4 attention stack as `LLAMA_KV_CACHE_TYPE=q4_0` on the hot ring — including `ATTENTION_KERNEL=auto` (fused &lt;128, ggml MWG ≥128). The first token past the window **spills** hot Q4 → TQ cold once (`turboquant_spill_q4_to_v3`), then decode uses `turboquant_attn_v3`. Prefill is parallel (zero-alloc alias) while the prompt fits in the hot window.

Set `TURBOQUANT_DUAL_WRITE=1` to also pack TQ during the hot window (avoids a big spill; slight hot-path tax).

**MTP:** Allowed when `TURBOQUANT_HOT_WINDOW>0` (verify stays on the hot/Q4 path). Pure TQ (`HOT_WINDOW=0`) still refuses `--mtp`.

## Phase map

| Phase | Status | What |
|---|---|---|
| **P0** | Harness ready | Needle gate: `tools/turboquant_needle.md` + `tools/turboquant_needle.sh` (run after build) |
| **P1** | Shipped | Lloyd–Max V3 + residual window (CPU codebooks on GPU path) |
| **P2** | Shipped | Metal fused rotate+quant + `turboquant_attn_v3` with **device** score buffer (8–16k ctx) |
| **P3** | Shipped | KV meter on CLI/scheduler; `kv_persist` TQ type tag |
| **P3b** | Shipped | Hybrid Q4 hot window + fused decode + parallel prefill-in-hot + faster `attn_v3` |
| **P3c** | Shipped | Auto spill hot→TQ at boundary (`turboquant_spill_q4_to_v3`); zero-alloc TQ prefill alias |
| **P4** | Stub only | Disk-backed cold TQ pages for 1M-class docs — see below |

## Design (runtime)

```
decode write K,V (post QK-norm / RoPE)
        │
        ├─► Q4 hot ring (last HOT tokens, model frame) ── Q4 flash while ctx ≤ HOT
        │
        ▼
 residual window (last W tokens, fp32 rotated)
        │ older
        ▼
 TurboQuant V3: rotate → Lloyd–Max → bitpack (asymmetric K/V)
        │
        ▼
 turboquant_attn_v3 (cold / ctx > HOT): rotate Q, score, unrotate O
```

No QJL (softmax amplifies QJL noise — community + our refs agree).

## P4 stub — disk tier (not implemented)

Cold compressed rows live on SSD; GPU holds residual window + hot TQ pages.
Prefetch next pages while attending. Same quality contract as RAM-resident TQ;
placement only changes speed. Enables the stretch claim:
“1M-token book on disk, GPU holds a sliding compressed working set.”

## Quality gate (P0)

Before advertising a bit-width:

1. Hide a unique needle in a long prompt (≥2k–8k tokens).
2. Ask the model to recall it greedily.
3. PASS = exact string in completion.
4. Log: bits, rw, KV MB, vs F16 MB, tok/s, PASS/FAIL.

See `tools/turboquant_needle.md` for the procedure.
