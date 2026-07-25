# 17 — Tensor Shapes Cheatsheet (E4B decode)

Current defaults as of the GGUF + `LLAMA_CTX_SIZE` era. Always re-check
`gemma4_config` metadata in **your** GGUF — E2B differs.

---

## Model config (typical E4B)

| Property | Typical value | Notes |
|----------|---------------|--------|
| `hidden_size` | 2560 | Residual width |
| `num_hidden_layers` | 42 | |
| `num_attention_heads` | 20 | Q heads |
| `num_key_value_heads` | 4 | GQA → 5 groups |
| `head_dim` (sliding) | 128 | |
| `global_head_dim` (full) | 512 | `max_head_dim` for scratch |
| `intermediate_size` | 10240 | Or per-layer `intermediate_sizes` (E2B) |
| `vocab_size` | ~262144 | |
| `hidden_size_per_layer_input` (PLE) | 256 | |
| `final_logit_softcapping` | 30.0 | |
| `kv_capacity` | `LLAMA_CTX_SIZE` (default **16384**, cap **200000**) | Not hard-capped at 4096 anymore |
| `kv_cache_type` | env `LLAMA_KV_CACHE_TYPE` (default f16; prefer **q4_0**) | |
| Weights | GGUF K-quants (Q4_K / Q6_K in Q4_K_M) | Not “everything is Q4_0” |

---

## Per-layer projection widths

| Layer type | `q_out` | `kv_out` |
|------------|---------|----------|
| Sliding (hd=128) | 20×128 = **2560** | 4×128 = **512** |
| Full (hd=512) | 20×512 = **10240** | 4×512 = **2048** |

```text
head_dim = layer_head_dim(i)
q_out    = num_heads * head_dim
kv_out   = num_kv_heads * head_dim
```

---

## Decode buffers (one token)

| Tensor | Shape (E4B) | Where |
|--------|-------------|--------|
| token id | scalar | CPU |
| `hidden_buf` | `[2560]` f32 | GPU |
| Q | `[q_out]` f32 | GPU scratch |
| K, V (if `has_kv`) | `[kv_out]` f32 | GPU scratch |
| attn out | `[q_out]` → O → `[2560]` | GPU |
| gate / up | `[intermediate]` | GPU |
| down | `[2560]` | GPU |
| PLE layer vec | `[256]` | GPU |
| logits | `[vocab]` f32 | GPU → CPU (server) |
| K/V cache row | `num_kv_heads × row_bytes` at `pos` | per layer buffer |

---

## KV row bytes

```text
F16:  head_dim * 2
Q8_0: (head_dim/32) * 34
Q4_0: (head_dim/32) * 18

buffer_bytes(layer) = num_kv_heads * kv_capacity * bytes_per_row(head_dim)
```

Pool cost ≈ `slots × Σ_layers 2 × buffer_bytes(layer)` (K and V).

---

## Prefill (chunk length S)

| Tensor | Shape |
|--------|-------|
| hidden | `[S, 2560]` |
| Q | `[S, q_out]` (or stacked QKV layout) |
| K/V append | `S` rows into cache at `cur_seq .. cur_seq+S` |
| attn | causal over `past + S` |

Scratch must fit `max(S) = LLAMA_MAX_PREFILL_SEQ` (and verify batch for MTP).

---

## Shared-KV reminder

Layers with `has_kv=false` still allocate Q of size `q_out` for that layer’s
head_dim, but **do not** write K/V; they read `k_cache[kv_source_layer]` /
`v_cache[kv_source_layer]` (source row_bytes must match).

---

## Checklist

- [ ] Quote sliding vs full `q_out`/`kv_out` without notes.
- [ ] Know ctx is env-driven (16k default), not a fixed 4096.
- [ ] Estimate pool bytes order-of-magnitude for your slots×ctx×dtype.

**Prev:** [16_glossary_and_drills.md](16_glossary_and_drills.md) · **Index:** [README.md](README.md)
