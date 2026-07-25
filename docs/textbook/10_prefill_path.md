# 10 — Prefill Path

## One-sentence summary

Prefill consumes many prompt tokens at once: build Q/K/V for a chunk, **batch
append** into KV, run **causal** flash attention (tiled ext), run wide MLP
`mul_mm`, optionally skip lm_head on intermediate chunks — then the scheduler
advances a cursor until the prompt is exhausted.

---

## Why chunk

```text
LLAMA_MAX_PREFILL_SEQ  → max tokens per chunk (GPU scratch / kernel limits)
LLAMA_PREFILL_TOKENS_PER_TICK → fair share across concurrent prefills
```

Long prompts = multiple chunks. Intermediate chunks: `want_logits=false`
(fill KV only). Last chunk: logits for first generated token.

```mermaid
flowchart LR
  P["prompt tokens"] --> C1["chunk 0"]
  C1 --> C2["chunk 1"]
  C2 --> Cn["chunk n want_logits"]
  Cn --> S["sample → decode phase"]
```

---

## Prefill vs decode differences

| | Prefill | Decode |
|--|---------|--------|
| Sequence | `S` tokens | 1 token |
| Matmul | `mul_mm` / stacked | matvec |
| Attention | causal over chunk (+ past KV) | attend past only |
| KV write | batch append S rows | append 1 row |
| Cost | ~O(S) … O(S²) attn | O(kv_seq) attn |

---

## High-level encode (one chunk)

```text
embed all token rows → hidden[S, H]   (or staged)
for layer:
  norm → stacked QKV mul_mm (or projections)
  QK-norm + RoPE (GPU rope helpers)
  batch KV append for layers with has_kv
  flash_attn_ext causal (tiled)
  O proj mul_mm
  residual / norms
  MLP gate∥up mul_mm → gelu → down
  PLE for all rows
final norm + lm_head  if want_logits
```

Host orchestration: `forward_prefill_chunk*_with_kv_slot`, parallel variants,
batch multi-slot `forward_prefill_batch_with_kv_slots`.

---

## flash_attn_ext (prefill)

Shaders: `ggml_flash_attn_ext.metal`  
Glue: `ggml_flash_attn_ext.rs`

Ideas:

- Tile KV in shared memory  
- Mask fill / padding helpers for non-multiples  
- Head-dim specialized (`h256`, `h512`)  
- NSG (simdgroups) tuning — E23 raised h256 NSG 4→8 to match llama @4k  

`TILED_EXT_MIN_Q`: below this, older per-row paths may run — MTP set default
low (2) because verify batches are small but still benefit from tiling (M4).

---

## Scheduler integration

`scheduler.rs` `prefill_active_round`:

1. Plan which active Prefilling requests get tokens this tick (`plan_prefill_round`).  
2. Build `PrefillInput { slot, token_ids, want_logits }`.  
3. `BatchEngine::prefill_batch`.  
4. Advance cursors; on complete → sample → `ActivePhase::Decoding`.

Fairness: round-robin index `next_prefill_index` so one long prompt cannot starve others forever when token budget is set.

---

## Profiling

```text
PREFILL_TIMING=1
PROFILE_ABLATE=attn|mlp|ple|...
BENCH_PREFILL_EXACT=1
```

`--bench-prefill --bench-prefill-tokens 2048,4096` for cold/hot tok/s.

Phase table intuition (E16): MLP > Attn > PLE > head on long prefill.

---

## Checklist

- [ ] Why intermediate chunks skip lm_head.  
- [ ] Prefill attention is causal+past, not decode-style.  
- [ ] Role of `LLAMA_MAX_PREFILL_SEQ` vs tokens per tick.  
- [ ] Name the ext shader file and one tuning win (NSG).

**Next:** [11_server_api.md](11_server_api.md)
