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
# CLI
ATTENTION_KERNEL=auto \
LLAMA_KV_CACHE_TYPE=turboquant \
TURBOQUANT_K_BITS=3 TURBOQUANT_V_BITS=2 \
TURBOQUANT_RESIDUAL_WINDOW=128 \
TURBOQUANT_HOT_WINDOW=2048 \
LLAMA_CTX_SIZE=8192 \
./target/release/llama-sinks \
  --gpu "$HOME/Downloads/models/e2b/gemma-4-E2B-it-Q4_K_M.gguf"

# OpenAI server (same KV env)
ATTENTION_KERNEL=auto \
LLAMA_KV_CACHE_TYPE=turboquant \
TURBOQUANT_K_BITS=3 TURBOQUANT_V_BITS=2 \
TURBOQUANT_HOT_WINDOW=2048 \
LLAMA_CTX_SIZE=8192 LLAMA_KV_POOL_SLOTS=4 \
./target/release/llama-sinks \
  --gpu "$HOME/Downloads/models/e2b/gemma-4-E2B-it-Q4_K_M.gguf" --serve
```

| Env | Meaning |
|---|---|
| `LLAMA_KV_CACHE_TYPE=turboquant` | Enable TQ path |
| `TURBOQUANT_BITS` | Symmetric bit-width (fallback) |
| `TURBOQUANT_K_BITS` / `V_BITS` | Asymmetric (prefer K ≥ V) |
| `TURBOQUANT_RESIDUAL_WINDOW` | Recent tokens kept fp32 in Haar frame (default 128) |
| `TURBOQUANT_HOT_WINDOW` | Decode spill threshold / short-ctx Q4 window (default **2048**; `0` = pure TQ) |
| `TURBOQUANT_PREFILL_Q4` | Prefill with full-ctx Q4 flash then spill (default **on**; `0` = legacy per-row cold attn) |
| `TURBOQUANT_DUAL_WRITE` | `1` = also pack TQ cold while in hot window. Default off for speed. |
| `LLAMA_KV_POOL_SLOTS` | Serve concurrency (default 4). Each slot owns its own Q4 hot ring (~25 MB @2048 on E2B). |

**Recommended demo:** `K3/V2` + `rw=128` + `hot=2048` (decode ≈ Q4 flash while ctx ≤ hot).  
**Quality-safer:** `K6/V4` + `rw=128` if available / when raising bits.

**Speed model:** While `attn_kv_seq ≤ TURBOQUANT_HOT_WINDOW`, decode stays on the Q4 flash stack (same as `q4_0`, including `ATTENTION_KERNEL=auto`). Prefill (default **`TURBOQUANT_PREFILL_Q4=1`**) allocates a Q4 ring for the full ctx, attends with that flash stack for the **entire** prompt, then **spills once** to TQ cold if `seq > HOT_WINDOW` — so ~7k prompts are Q4-prefill-fast, not per-row `attn_v3`. Set `TURBOQUANT_PREFILL_Q4=0` to restore legacy past-hot per-row cold attention (slow). Peak RAM during prefill ≈ full Q4 KV + TQ cold allocation.

**`--serve`:** Per-slot hot rings + residual + spill flag live in `KvCachePool` (not model-global). Concurrent requests do not share a hot ring. When the scheduler batches multiple TQ slots in one tick, prefill/decode run **serially per slot** via the single-slot swap path (correctness). With default prefill-Q4, each slot’s hot ring is sized to ctx; long prefills spill once at the end.

Set `TURBOQUANT_DUAL_WRITE=1` to also pack TQ during the hot window (avoids a big spill; slight hot-path tax).

**MTP:** Allowed when `TURBOQUANT_HOT_WINDOW>0` (verify stays on the hot/Q4 path). Pure TQ (`HOT_WINDOW=0`) still refuses `--mtp`. Same gate for `--serve --mtp`.

## Phase map

| Phase | Status | What |
|---|---|---|
| **P0** | Harness ready | Needle gate: `tools/turboquant_needle.md` + `tools/turboquant_needle.sh` (run after build) |
| **P1** | Shipped | Lloyd–Max V3 + residual window (CPU codebooks on GPU path) |
| **P2** | Shipped | Metal fused rotate+quant + `turboquant_attn_v3` with **device** score buffer (8–16k ctx) |
| **P3** | Shipped | KV meter on CLI/scheduler; `kv_persist` TQ type tag |
| **P3b** | Shipped | Hybrid Q4 hot window + fused decode + parallel prefill-in-hot + faster `attn_v3` |
| **P3c** | Shipped | Auto spill hot→TQ at boundary (`turboquant_spill_q4_to_v3`); zero-alloc TQ prefill alias |
| **P3d** | Shipped | Serve path: per-slot hot/rw/spill in `KvCachePool`; batched hot prefill/decode; cold via spill + single-slot |
| **P3e** | Shipped | Chunked past-hot prefill: split at `HOT_WINDOW`, spill once, batched QKV/MLP + per-row `attn_v3` |
| **P3f** | Shipped | Prefill-Q4 (default): full-ctx Q4 flash for whole prompt, one spill to TQ if `seq > HOT` |
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
