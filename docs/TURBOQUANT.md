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
| `TURBOQUANT_PREFILL_Q4` | **Opt-in only** (`=1`): Q4-flash whole prompt then spill. Default **off** — past-hot uses TQ multi-query causal attn. |
| `TURBOQUANT_GQA` | Default **on**. SWA `h256` decode/prefill share KV across Q heads (E2B `8:1`). Set `=0` to force per-Q-head TG. |
| `TURBOQUANT_DUAL_WRITE` | `1` = also pack TQ cold while in hot window. Default off for speed. |
| `LLAMA_KV_POOL_SLOTS` | Serve concurrency (default 4). Each slot owns its own Q4 hot ring (~25 MB @2048 on E2B). |

**Recommended demo:** `K3/V2` + `rw=128` + `hot=2048` (decode ≈ Q4 flash while ctx ≤ hot).  
**Quality-safer:** `K6/V4` + `rw=128` if available / when raising bits.

**Speed model:** While `attn_kv_seq ≤ TURBOQUANT_HOT_WINDOW`, decode stays on the Q4 flash stack. Past-hot **prefill** batch-rotates Q/K/V, packs each cold chunk in two dispatches, and attends with **`turboquant_attn_v3_causal`** / **`_gqa_h256`** (flash online softmax, half centroids, GQA KV share on SWA). Spill once at the hot boundary. On E2B K3/V2, the 7,127-token needle improved from **167.45s → ~52s** and still returns `AURORA-7749`. Past-hot **decode** uses fused flash `attn_v3` (no device score buffer); SWA layers use GQA (`8:1` on E2B). Global `h512` stays per-Q-head (TG smem). `TURBOQUANT_PREFILL_Q4=1` is an explicit escape hatch only.

**`--serve`:** Per-slot hot rings + residual + spill flag live in `KvCachePool`. Multi-slot TQ runs serially per slot. Past-hot prefill uses TQ multi-query causal attention on each slot’s cold cache.

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
| **P3f** | Opt-in only | `TURBOQUANT_PREFILL_Q4=1` Q4-flash whole prompt (escape hatch; default off) |
| **P3g** | Shipped | `turboquant_attn_v3_causal`: multi-query TQ flash for past-hot prefill chunks |
| **P3h** | Shipped | Native TQ prefill: batched f16 Haar matmuls, chunk-wide K/V packing, four-query tiles, SIMD-pair score mapping |
| **P3i** | Shipped | Parity path: flash online decode (no score buffer), half centroids, GQA `h256` for groups 2/4/8 (E2B 8:1). Past-hot decode **~23 vs ~13.5 tok/s** (`GQA=1` vs `0`); 7k needle still `AURORA-7749` |
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
