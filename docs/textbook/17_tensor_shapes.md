# 17 — Shapes: Derive Them, Don't Memorize Them

Most confusion in this codebase is shape confusion. This chapter gives you the
derivation rules and the config functions that produce every number, so you can
recompute any shape at a whiteboard instead of grepping for it.

Source of truth: `src/gemma4_config.rs`. Read it alongside this chapter.

---

## Part A — The config is not uniform, and that is the whole point

The first thing to unlearn: **Gemma4 layers do not all have the same shapes.**
Four things vary per layer, and each has an accessor:

```107:148:src/gemma4_config.rs
pub fn layer_intermediate_size(&self, layer_idx: usize) -> usize {
    self.intermediate_sizes.get(layer_idx).copied().unwrap_or(self.intermediate_size)
}

pub fn layer_num_kv_heads(&self, layer_idx: usize) -> usize {
    self.num_key_value_heads_per_layer.get(layer_idx).copied().unwrap_or(self.num_key_value_heads)
}

pub fn layer_num_kv_groups(&self, layer_idx: usize) -> usize {
    let kv = self.layer_num_kv_heads(layer_idx);
    if kv == 0 { 1 } else { self.num_attention_heads / kv }
}

pub fn is_full_attention(&self, layer_idx: usize) -> bool {
    self.layer_types.get(layer_idx).map_or(false, |t| t == "full_attention")
}

pub fn layer_head_dim(&self, layer_idx: usize) -> usize {
    if self.is_full_attention(layer_idx) { self.global_head_dim } else { self.head_dim }
}
```

The last one is the surprise, and it explains a lot of the codebase:

> **Full-attention layers use `global_head_dim` (default 512); sliding layers
> use `head_dim`.**

That is why Metal pipelines come in `h128` / `h256` / `h512` variants, why
`pipeline_for(head_dim)` exists on the host side, why `rope_max_head_dim` is a
thing, and why the MTP draft-head scratch bug (`AGENTS.md` M1) was "sized to
`hidden_head` instead of `max_head_dim = 512`." A scratch buffer sized for the
sliding head_dim overflows the moment a full-attention layer runs.

**Rule: any code that touches head_dim must ask the layer, never the config
root.** `config.head_dim` is the sliding value, not "the" value.

The `unwrap_or` pattern in each accessor means: per-layer array if present,
otherwise the uniform default. Both are legal checkpoints.

---

## Part B — Derivation rules

Given `hidden_size (H)`, `num_attention_heads (nh)`,
`num_key_value_heads (nkv)`, `layer_head_dim (hd)`,
`layer_intermediate_size (I)`, `hidden_size_per_layer_input (ple)`,
`vocab_size (V)`, `num_hidden_layers (L)`:

```text
q_out         = nh  * hd
kv_out        = nkv * hd
num_kv_groups = nh / nkv          # query heads sharing one KV head

q_proj        : [q_out,  H]
k_proj        : [kv_out, H]       # absent when !has_kv
v_proj        : [kv_out, H]       # absent when !has_kv
o_proj        : [H,  q_out]

q_norm        : [hd]              # applied per (token, head)
k_norm        : [hd]

gate_proj     : [I, H]
up_proj       : [I, H]
down_proj     : [H, I]

ple_gate      : [ple, H]
ple_proj      : [H, ple]
ple_model_proj: [L * ple, H]      # the pre-pass weight (Ch 08 Part D.2)

lm_head       : [V, H]
```

Two invariants to check whenever you touch attention:

```text
nh % nkv == 0                  # GQA must divide evenly
hd % 32 == 0                   # required by quantized KV (Ch 13 Part A.3)
```

### B.1 Activations, decode vs prefill

| Tensor | Decode | Prefill (seq = S) |
|---|---|---|
| hidden | `[H]` | `[S, H]` |
| normed | `[H]` | `[S, H]` |
| q | `[nh, hd]` | `[S, nh, hd]` → transposed to `[nh, S, hd]` |
| k, v | `[nkv, hd]` | `[S, nkv, hd]` → `[nkv, S, hd]` |
| scores | `[kv_seq]` per head | `[S, kv_seq]` per head (causally masked) |
| attn_out | `[nh, hd]` | `[nh, S, hd]` → back to `[S, nh, hd]` |
| gate/up | `[I]` | `[S, I]` |
| logits | `[V]` | `[V]` for the last row only (`want_logits`) |

The transposes in the prefill column are the two per-layer layout flips from
Ch 10 Part C/E. Decode needs none: with one token, `[nh, hd]` and
`[1, nh, hd]` are the same bytes.

### B.2 The complete tensor catalogue

Every learned tensor in a Gemma4 layer, with its formula and its E4B sliding /
full values. `H` = hidden, `nh` = query heads, `nkv` = KV heads, `hd` = layer
head_dim, `I` = layer intermediate, `ple` = PLE dim, `V` = vocab.

| Tensor | Shape `[out, in]` | E4B sliding | E4B full |
|---|---|---|---|
| `input_layernorm` | `[H]` | 2560 | 2560 |
| `q_proj` | `[nh·hd, H]` | [2560, 2560] | [10240, 2560] |
| `k_proj` | `[nkv·hd, H]` | [512, 2560] | [2048, 2560] |
| `v_proj` | `[nkv·hd, H]` | [512, 2560] | [2048, 2560] |
| `q_norm` (per head) | `[hd]` | 128 | 512 |
| `k_norm` (per head) | `[hd]` | 128 | 512 |
| `o_proj` | `[H, nh·hd]` | [2560, 2560] | [2560, 10240] |
| `post_attention_norm` | `[H]` | 2560 | 2560 |
| `pre_feedforward_norm` | `[H]` | 2560 | 2560 |
| `gate_proj` | `[I, H]` | [10240, 2560] | same |
| `up_proj` | `[I, H]` | [10240, 2560] | same |
| `down_proj` | `[H, I]` | [2560, 10240] | same |
| `post_feedforward_norm` | `[H]` | 2560 | 2560 |
| `ple_gate` | `[ple, H]` | [256, 2560] | same |
| `ple_proj` | `[H, ple]` | [2560, 256] | same |
| `post_per_layer_input_norm` | `[H]` | 2560 | 2560 |
| `layer_scalar` | scalar | 1 | 1 |

Model-level tensors (not per layer):

| Tensor | Shape | E4B |
|---|---|---|
| `token_embd` | `[V, H]` | [262144, 2560] |
| `per_layer_token_embd` | `[V, L·ple]` | [262144, 10752] |
| `per_layer_model_projection` | `[L·ple, H]` | [10752, 2560] |
| `per_layer_projection_norm` (per layer) | `[ple]` | 256 |
| `final_norm` | `[H]` | 2560 |
| `lm_head` | `[V, H]` | [262144, 2560] (may be tied) |

Two entries repay attention. `per_layer_token_embd` is `[262144, 10752]` — an
enormous table, but only one row is gathered per token (Ch 09 Part B), so its
size costs memory, not bandwidth. And note that four of the seventeen per-layer
tensors are `[H]` norm vectors: negligible bytes, but each one is a separate
dispatch's worth of correctness (Ch 02 Part K).

Shared-KV layers are missing `k_proj` and `v_proj` entirely — the tensors do not
exist in the file, so a loader that assumes they do will fail at load rather
than silently (Ch 03 Part C).

### B.3 KV cache

```text
bytes_per_row = F16 : hd * 2
                Q8_0: (hd/32) * 34
                Q4_0: (hd/32) * 18

per layer per tensor = nkv * capacity * bytes_per_row
total                = slots * L * 2 * (that)
```

Note `capacity`, not `seq_len` — Ch 06 Part C.1.

---

## Part C — Worked: E4B

Approximate values; always confirm against your checkpoint's metadata.

```text
H   = 2560      nh  = 20        nkv = 4         L = 42
hd  = 128 (sliding) / 512 (full)
I   = 10240     ple = 256       V   = 262144
sliding_window ≈ 512-1024 (from metadata)
```

Derived, sliding layer (hd = 128):

```text
num_kv_groups = 20 / 4 = 5
q_out         = 20 * 128 = 2560
kv_out        =  4 * 128 =  512

q_proj  : [2560, 2560]    o_proj : [2560, 2560]
k_proj  : [512,  2560]    v_proj : [512,  2560]
gate/up : [10240, 2560]   down   : [2560, 10240]
ple_gate: [256, 2560]     ple_proj: [2560, 256]
lm_head : [262144, 2560]
```

Derived, full-attention layer (hd = 512):

```text
q_out  = 20 * 512 = 10240
kv_out =  4 * 512 =  2048
q_proj : [10240, 2560]   o_proj : [2560, 10240]
```

**Four times the attention projection work on full layers.** That is not a
typo, and it is why full-attention layers are more expensive than SWA layers
for reasons beyond the attention span itself.

### C.1 Size checks worth doing once

`lm_head` in Q4_0: `262144 * 2560 / 32 * 18 ≈ 378 MB`. In Q6_K:
`262144 * 2560 / 256 * 210 ≈ 551 MB`. Either way it is the largest single
tensor and the reason `want_logits` (Ch 12 E.3) and the batched verify lm_head
(Ch 14, M3) matter so much.

MLP weights per layer in Q4_0:
`3 * 10240 * 2560 / 32 * 18 ≈ 44 MB`. Times 42 layers ≈ 1.85 GB — the bulk of
the ~2.5 GB read per decode token. **This is the number that makes decode
bandwidth-bound.** Everything in Ch 05 follows from it.

---

## Part D — Worked: E2B

```text
H = 2048        nh = 8          nkv = 4 (check config)     L ≈ 30-34
hd = 256 (sliding) / 512 (full)
I ≈ 8192        ple = 256       V = 262144
```

Do the derivation yourself — that is Exercise 1. Note that E2B's *sliding*
head_dim (256) equals E4B's `hd/2` for full layers, which is why both models
exercise the `h256` pipelines and why benchmarks in `AGENTS.md` alternate
between them.

---

---

## Part D.1 — Scratch inventory: the max-over-layers rule

Decode scratch buffers are allocated once at load and reused for every layer of
every token. Since layers differ (Part A), each buffer must be sized for the
**maximum** over layers, and the constructor does exactly that:

```1365:1367:src/gemma4_gpu_model.rs
        let max_head_dim = config.global_head_dim;
        let max_q_out = num_heads * max_head_dim;
        let max_kv_out = (0..num_layers).map(|i| config.layer_num_kv_heads(i) * config.layer_head_dim(i)).max().unwrap_or(num_kv_heads_max * max_head_dim);
```

Read the third line carefully: it does not assume `nkv` is uniform either. It
takes the max of the *product* over layers, which is the only correct thing to
do when both factors can vary per layer (E2B).

| Scratch buffer | Size | E4B floats |
|---|---|---|
| `hidden_buf`, `normed_buf`, `residual_buf` | `H` | 2560 |
| `q_buf`, `q_normed_buf` | `max_q_out` | 20 × 512 = 10240 |
| `k_buf`, `v_buf`, `k_normed_buf` | `max_kv_out` | 4 × 512 = 2048 |
| `attn_out_buf` | `max_q_out` | 10240 |
| MLP mid | `max_intermediate_size()` | 10240 (E2B: per-layer max) |
| `ple_token_id_buf`, `ple_context_proj_buf`, `ple_combined_buf` | `L · ple` | 10752 |
| `decode_rope_{cos,sin}_packed` | `L · rope_max_head_dim` | 42 × 512 = 21504 |
| logits | `V` | 262144 |

Total decode scratch is on the order of a few hundred KB plus the 1 MB logits
buffer — trivial next to 2.5 GB of weights, which is why generous
max-based sizing is the right call. The prefill scratch is a different story: it
scales with `max_prefill_seq`, and that is what caps chunk size (Ch 10 Part B).

**The failure this rule prevents.** Size `q_buf` as `nh × config.head_dim`
(= 2560) instead of `nh × global_head_dim` (= 10240) and everything works until
the first full-attention layer, which writes 10240 floats into a 2560-float
buffer and corrupts whatever follows. `AGENTS.md` M1 is this exact bug in the
MTP draft head: "draft-head attention scratch sized to `hidden_head` instead of
`max_head_dim` (512)." Symptom was garbage output, cause was one multiplication.

## Part E — Shape debugging protocol

When output is wrong, or a buffer overruns:

**1. Print, don't infer.** For layer 0 and one full-attention layer:

```text
layer_idx, is_full_attention, layer_head_dim, layer_num_kv_heads,
layer_num_kv_groups, q_out_dim, kv_out_dim, layer_intermediate_size,
weight_format, has_kv, kv_source_layer
```

**2. Check the invariants:** `nh % nkv == 0`, `hd % 32 == 0`,
`q_out == nh * hd`, `kv_out == nkv * hd`.

**3. Check scratch sizing against the max, not the typical.** Every scratch
buffer must be sized for `max_head_dim` (512) and `max_intermediate_size()`,
not layer 0's values. `max_intermediate_size()` exists precisely for this.

**4. Check the norm batch counts.** QK-norm reduces over `hd` with
`S * nh` independent rows (Ch 10 Part C.2). PLE's pre-pass norm reduces over
`ple` with `L` rows (Ch 08 Part D.2). Same kernel, different contract.

**5. Check who owns head_dim on shared-KV layers.** The anchor's, not the
layer's (Ch 06 Part F).

---

## Part F — Exercises

1. Derive every weight shape for E2B, both layer types, from the config values
   in Part D. Verify against `tools/gguf_inspect.py` output.

2. A scratch buffer is sized `nh * config.head_dim` floats. What is the first
   layer index at which it overflows for E4B, and by how much?

3. Compute total Q4_0 weight bytes read per decode token for E4B: attention
   projections (both layer types, weighted by how many of each), MLP, PLE,
   lm_head. Compare with the ~2.5 GB figure.

4. For a full-attention E4B layer with Q4_0 KV: how many bytes does one
   position of K occupy? One position of K+V? At 8192 capacity, one layer?

5. `nh = 20`, `nkv = 3`. Which invariant fails, and where in the code would it
   first manifest?

6. Why is `[nh, hd]` and `[1, nh, hd]` the same for decode but not for
   prefill? Express the difference as a stride.

7. `ple_model_proj` is `[L * ple, H]`. For E4B compute its shape and its Q4_0
   byte size, then explain why E22 wanted it kept as f16 instead.

8. `per_layer_token_embd` is `[262144, 10752]`. Compute its size in Q4_0 and in
   F16. Then explain why a tensor larger than `lm_head` costs almost nothing per
   token.

9. Take the `max_kv_out` expression in Part D.1 and evaluate it for a
   hypothetical model with `num_key_value_heads_per_layer = [4, 4, 2, 8]` and
   `layer_types = [sliding, full, sliding, sliding]`, `head_dim = 128`,
   `global_head_dim = 512`. Which layer determines the max?

10. Add up the seventeen per-layer tensors' byte sizes for one E4B sliding layer
    in Q4_0. What fraction is the MLP? Compare with the ~75% claim in Ch 00b
    Part 6.4.

11. Given only `q_proj.n_rows() = 10240` and `H = 2560`, decide whether the
    layer is sliding or full — and say what additional fact you need if
    `global_head_dim` were 256 instead of 512.

---

## Checklist

- [ ] I can derive every weight shape from H, nh, nkv, hd, I, ple, V.
- [ ] I know full-attention layers use `global_head_dim`, and what breaks if I
      forget.
- [ ] I know both invariants (`nh % nkv`, `hd % 32`) and who enforces them.
- [ ] I can compute KV bytes per layer per slot for any KV type.
- [ ] I know why scratch must be sized to max, not typical.
- [ ] I can compute the ~2.5 GB/token figure that makes decode
      bandwidth-bound.

**Next:** [16_glossary_and_drills.md](16_glossary_and_drills.md)
