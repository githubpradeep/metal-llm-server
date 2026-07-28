# 02 — Gemma4 Architecture: The Contract the Engine Must Honour

Every strange branch in `gemma4_gpu_model.rs` exists because Gemma4 is not a
Llama clone. This chapter is the contract: what the architecture demands, why
it demands it, and which line of code satisfies each demand.

Primary types: `Gemma4Config` / `Gemma4TextConfig` in `src/gemma4_config.rs`
(179 lines — read all of it, it is the smallest high-value file in the repo).
GGUF derivation: `gemma4_config_from_gguf` in `gemma4_gpu_model.rs`.

Prerequisite: [00b](00b_transformers_first_principles.md) Part 7.

---

# Part A — The five things that make Gemma4 different

Before the details, the list. If you internalize only this, you can predict
most of the engine's structure:

| # | Feature | Consequence for the engine |
|---|---|---|
| 1 | **Two attention geometries** (sliding vs full, with *different head_dim*) | every attention kernel compiled 3× by head_dim; scratch sized to the max |
| 2 | **Shared KV layers** | `has_kv` / `kv_source_layer` on every layer; append skipped for 40%+ of layers |
| 3 | **QK-norm and V-norm** | per-head RMSNorm dispatches; `attention_scale = 1.0` |
| 4 | **PLE (per-layer embeddings)** | a GPU pre-pass plus a third residual branch per layer |
| 5 | **Norms on both sides of every sub-block, plus `layer_scalar`** | 3 `rmsnorm_acc` per layer, and a per-layer constant multiply |

Numbers 1 and 2 are structural (they change what data exists). Numbers 3–5 are
numerical (they change the arithmetic). Structural mistakes crash or produce
obvious garbage; numerical mistakes produce fluent, wrong text. Ch 15 Part B.5
is about catching the second kind.

---

# Part B — Spec snapshot

Always verify against your GGUF; these are typical E4B values.

| Property | E4B (typical) | Why the engine cares |
|---|---|---|
| dense transformer (no MoE) | yes | no routing, no expert gather |
| `hidden_size` | 2560 | residual width; every norm's `dim` |
| `num_hidden_layers` | 42 | loop bound; × per-layer dispatch count |
| `num_attention_heads` | 20 | threadgroup count in decode attention |
| `num_key_value_heads` | 4 | GQA groups = 5; KV cache width |
| `head_dim` (sliding) | 128 | `h128` kernel family |
| `global_head_dim` (full) | 512 | `h512` kernel family; scratch max |
| `intermediate_size` | 10240 | MLP width; ~75% of weight bytes |
| `intermediate_sizes[]` | (E2B) per-layer | scratch must use the **max** |
| `vocab_size` | 262144 | `lm_head` rows; ~380 MB in Q4_0 |
| `hidden_size_per_layer_input` | 256 | PLE dim |
| `sliding_window` | often 512 | SWA mask; `kv_start` in decode |
| `num_kv_shared_layers` | 18 | layers 24–41 read layers 22/23-ish caches |
| `final_logit_softcapping` | 30.0 | `cap·tanh(x/cap)` |
| `tie_word_embeddings` | often true | `lm_head` may be `token_embd` |

E2B differs in more than scale: it uses `intermediate_sizes[]` (some layers
"double-wide") and may use `num_key_value_heads_per_layer[]`. **Never carry E4B
numbers in your head while debugging E2B** — Ch 17 Part A is about exactly this
non-uniformity.

Read the defaults directly:

```61:68:src/gemma4_config.rs
fn default_rope_factor() -> f64 { 1.0 }

fn default_global_head_dim() -> usize { 512 }
fn default_hidden_size_per_layer() -> usize { 256 }
fn default_max_pos() -> usize { 131072 }
fn default_final_logit_softcapping() -> f32 { 30.0 }
fn default_rope_theta() -> f64 { 10000.0 }
fn default_partial_rotary() -> f64 { 1.0 }
```

Every one of those defaults is a decision that fires when a GGUF omits a field.
`default_global_head_dim() = 512` in particular means a model with no
`global_head_dim` key gets 512-wide full-attention heads — and therefore 4×
larger scratch and cache rows on those layers. Silent, load-bearing defaults
are worth reading once.

---

# Part C — Two attention geometries in one model

## C.1 Why a model would do this

Full attention over a long context is expensive in two ways: the score
computation is `O(S)` per query, and the KV cache is `O(S)` in memory. Sliding
window attention caps both at `O(W)`.

But a model made *entirely* of sliding layers cannot move information further
than `depth × W` positions, and it never forms a global summary. So Gemma
interleaves: mostly sliding layers for cheap local processing, with periodic
full-attention layers to mix globally.

The design result is a cost profile like this (E4B, context 8192, W=512):

```text
sliding layer: each query attends ~512 keys        cache rows: 512 (window)
full layer:    each query attends up to 8192 keys  cache rows: 8192
```

If the pattern is 5 sliding : 1 full, the average per-layer attention work is
`(5·512 + 8192)/6 ≈ 1792` keys instead of 8192 — a ~4.6× reduction, while
still having global mixing every sixth layer.

## C.2 The twist: full layers also have wider heads

This is the part that surprises people and breaks ports:

```142:148:src/gemma4_config.rs
    pub fn layer_head_dim(&self, layer_idx: usize) -> usize {
        if self.is_full_attention(layer_idx) {
            self.global_head_dim
        } else {
            self.head_dim
        }
    }
```

`head_dim` is **not** a model constant. It is a per-layer property.

| | Sliding layer | Full layer |
|---|---|---|
| `head_dim` | 128 | 512 |
| Q width (`n_q × hd`) | 20×128 = 2560 | 20×512 = **10240** |
| KV width (`n_kv × hd`) | 4×128 = 512 | 4×512 = **2048** |
| `q_proj` shape | [2560, 2560] | [10240, 2560] |
| KV cache row, Q4_0 | 4 × 4 × 18 = 288 B | 4 × 16 × 18 = **1152 B** |
| RoPE θ | ~10 000 | ~1 000 000 |
| Partial rotary | 1.0 (all channels) | **0.25** (128 of 512) |
| Attends to | last `sliding_window` keys | all keys, causal |

So a full layer costs 4× more in projection weights, 4× more in cache bytes
per position, *and* attends to more positions. Full layers are expensive on
every axis at once.

Three engine consequences follow mechanically:

1. **Kernels are compiled per head_dim.** You will see `_h128`, `_h256`,
   `_h512` suffixes throughout, selected by a `pipeline_for(head_dim)` match.
   Compile-time head_dim lets the shader unroll loops and size registers.
2. **Scratch buffers use `global_head_dim`.** `max_q_out = num_heads *
   max_head_dim` at `gemma4_gpu_model.rs` ~1366. Size scratch to the sliding
   head_dim and full layers overflow — Ch 17 Part A, and the same bug class as
   the MTP draft-head scratch bug in `AGENTS.md` M1.
3. **Cache stride math must use the layer's own head_dim** — and for shared-KV
   layers, the *anchor's* head_dim (Part D.4).

## C.3 Reading the pattern from config

`layer_types[i]` is `"sliding_attention"` or `"full_attention"`:

```138:140:src/gemma4_config.rs
    pub fn is_full_attention(&self, layer_idx: usize) -> bool {
        self.layer_types.get(layer_idx).map_or(false, |t| t == "full_attention")
    }
```

Note `map_or(false, ...)`: an out-of-range index or a missing entry means
*sliding*. Defensive, and worth knowing when a truncated config makes every
layer mysteriously sliding.

From GGUF the pattern arrives as a bool array
(`…attention.sliding_window_pattern`) where **true means SWA**, matching
llama.cpp's `is_swa`. You can see the inversion in the draft-head loader:

```196:197:src/speculative.rs
            // GGUF pattern: true = SWA, false = full (matches llama.cpp is_swa).
            let is_full = !swa_pattern[i];
```

Get that polarity backwards and you build a model where every layer is the
wrong type: wrong head_dim, wrong RoPE θ, wrong masking. It will load, and it
will produce garbage — one of the few Gemma4 mistakes that is *not* subtle.

---

# Part D — Shared KV layers

## D.1 The idea and the saving

The last `num_kv_shared_layers` layers do not have their own K/V at all. They
compute queries and attend against an **earlier** layer's cache.

For E4B: 42 layers, 18 shared → layers 0–23 own KV, layers 24–41 share. The
KV cache therefore stores 24 layers' worth instead of 42:

```text
24/42 ≈ 57% of the cache        →  ~43% memory saved
```

And 18 layers per token skip: K projection, V projection, K-norm, V-norm,
K-RoPE, and the append dispatch. That is a real decode speedup, not just a
memory win.

Why it works at all: adjacent layers' keys are highly redundant. Sharing the
cache is a strong constraint, but the model was *trained* with it, so the
weights are adapted to it.

## D.2 How the engine represents it

```876:885:src/gemma4_gpu_model.rs
    // Layer properties
    pub is_full_attention: bool,
    pub has_kv: bool,           // false for shared KV layers (layers 24-41)
    pub kv_source_layer: usize, // which layer's KV cache to use
    pub head_dim: usize,
    pub q_out_dim: usize,
    pub kv_out_dim: usize,
    pub intermediate_size: usize,
    pub weight_format: WeightFormat,
}
```

Two fields, and they appear in dozens of conditionals. `has_kv` gates
*writing*; `kv_source_layer` redirects *reading*. Note that owning layers set
`kv_source_layer = i` (themselves), so the read path needs no branch at all —
it always indexes `k_cache[layer.kv_source_layer]`. That is why you see that
expression rather than an `if has_kv` at every attention call site.

## D.3 Anchor selection

```1370:1388:src/gemma4_gpu_model.rs
        // Compute kv_source_layer for shared layers
        // For each shared layer, find the last non-shared layer of the same type
        let first_kv_shared = num_layers - config.num_kv_shared_layers;
        for i in first_kv_shared..num_layers {
            let layer_type = &config.layer_types[i];
            // Find the last non-shared layer with the same type
            let mut source = 0;
            for j in (0..first_kv_shared).rev() {
                if &config.layer_types[j] == layer_type {
                    source = j;
                    break;
                }
            }
            layers[i].kv_source_layer = source;
        }
        // Non-shared layers use their own index
        for i in 0..first_kv_shared {
            layers[i].kv_source_layer = i;
        }
```

Read the algorithm: for each shared layer, scan backwards through the
KV-owning layers for the **last one of the same type**. Type matching is not
optional — a sliding layer's cache holds head_dim-128 rows with θ=10000 RoPE
baked in, and a full layer cannot read that. The shapes differ, the positional
encoding differs, and the window semantics differ.

So for E4B all 18 shared sliding layers point at the same anchor (the last
sliding layer below 24), and all shared full layers point at the last full
layer below 24.

**Open question, tracked in `AGENTS.md` #12.** llama.cpp picks the anchor with
`n_layer_kv_from_start - (is_swa ? 2 : 1)`, which is not always the same layer
as "last of the same type." For a standard 5:1 pattern the two rules usually
agree, but they need not. If shared-layer numerics diverge from llama.cpp, this
loop is the first suspect — and it is code you can now read, so check it
against your model's `layer_types` rather than trusting either rule.

## D.4 The invariants shared layers depend on

**1. Ordering.** `kv_source_layer < first_kv_shared`, and layers execute in
increasing index order within a command buffer. So the anchor's K/V for the
current position is written before any sharer reads it. This is not enforced by
a fence; it is enforced by the sequential-dispatch guarantee (Ch 00c Part G)
plus the loop order. Reorder the layer loop and you break it silently.

**2. Stride math uses the anchor's geometry.** A shared layer reading
`k_cache[22]` must use layer 22's `head_dim` and `row_bytes` for address
arithmetic, not its own. Since anchors are the same type, these coincide — but
the code should derive them from the anchor, and `AGENTS.md` #12 mentions
exactly this ("Attention cache reads use anchor layer `row_bytes`").

**3. Append is skipped, not conditional-on-empty.** The buffers for shared
layers are still *allocated* by the pool (Ch 13 Part A.3), they just are never
written. Do not "helpfully" write into them.

## D.5 What shared layers do and do not do

| Step | Owning layer | Shared layer |
|---|---|---|
| input RMSNorm | yes | yes |
| Q projection, Q-norm, Q-RoPE | yes | **yes** |
| K/V projection | yes | no |
| K-norm, K-RoPE, V-norm | yes | no |
| KV append | yes | no |
| attention (read `cache[src]`) | yes | yes |
| O proj, post-attn norm, residual | yes | yes |
| MLP, PLE, layer_scalar | yes | yes |

The row that surprises people: shared layers still do the full Q pipeline
including RoPE. They are not "skip attention" layers; they are "borrow the
keys" layers.

---

# Part E — QK-norm, V-norm, and scale 1.0

Per-layer weights `q_norm_weight` and `k_norm_weight` apply RMSNorm to each
head's `head_dim` slice independently. V is normalized with **no** learned
weight.

Exact order, which must match the reference implementation:

```text
n     = RMSNorm_input(h)                       # whole-vector norm, learned w
Q_raw = W_q · n
K_raw = W_k · n                                # if has_kv
Q     = RoPE( RMSNorm_per_head(Q_raw, q_norm_weight) )
K     = RoPE( RMSNorm_per_head(K_raw, k_norm_weight) )
V     = RMSNorm_per_head_noweight(W_v · n)
attn  = Attention(Q, K, V) with scale = 1.0
```

Ch 00b Part 3.3 derives why the scale is 1.0: QK-norm has already fixed the
magnitude of Q and K with learned gains, so the classic `1/√d` would be a
second, redundant correction. **If you port a kernel that hardcodes `1/√d`, you
must remove it.**

Symptom of getting it wrong at head_dim 512: scores shrunk by ~22×, so the
softmax is nearly uniform and attention returns roughly the mean of all values.
Output stays grammatical and becomes vague — exactly the failure mode that is
hardest to notice by reading.

---

# Part F — Dual RoPE parameters

```150:166:src/gemma4_config.rs
    pub fn sliding_rope_theta(&self) -> f64 {
        self.rope_parameters.as_ref()
            .and_then(|r| r.sliding_attention.as_ref())
            .map_or(10000.0, |c| c.rope_theta)
    }

    pub fn full_rope_theta(&self) -> f64 {
        self.rope_parameters.as_ref()
            .and_then(|r| r.full_attention.as_ref())
            .map_or(1000000.0, |c| c.rope_theta)
    }

    pub fn full_partial_rotary_factor(&self) -> f64 {
        self.rope_parameters.as_ref()
            .and_then(|r| r.full_attention.as_ref())
            .map_or(0.25, |c| c.partial_rotary_factor)
    }
```

Two parameter sets, one per layer type. The reasoning (Ch 00b Part 5.4):
sliding layers only look back `W` positions, so fast frequencies give fine
local resolution; full layers must distinguish offsets in the thousands, so
they need θ = 1e6 to slow every frequency down.

**Partial rotary on full layers.** Only `0.25 × 512 = 128` channels carry
position; the other 384 pass through unrotated. Implemented as `cos=1, sin=0`
in the table, so the consuming kernel has no branch (Ch 00b Part 5.2).

Why would a model rotate only a quarter of a wide head? Because with head_dim
512 there are more channels than position needs — dedicating 384 of them to
content and 128 to position is a better split than smearing position across all
512. It is a capacity allocation decision.

The GGUF path pins `full_partial_rotary_factor = 0.25` to match
HF/llama.cpp's `rope_freqs.weight` mask semantics even when the field is
absent — see the comments in `gemma4_config_from_gguf`. Another load-bearing
default.

---

# Part G — PLE, in outline

Full treatment in Ch 08 Part D; here is what the *architecture* requires.

Each layer receives a small per-layer input built from two sources:

1. **Token identity** — an embedding lookup producing `[num_layers × ple_dim]`
   for the current token, scaled by `√ple_dim`.
2. **Context** — `per_layer_model_projection_weight · h`, sliced per layer,
   RMSNormed, then combined with the token identity (× `1/√2`).

Both are computed in a **pre-pass** before the layer loop (5 dispatches), and
then each layer does:

```text
gate  = W_ple_gate · h                 # hidden → ple_dim (256)
gated = GeLU(gate) ⊙ context[layer]    # elementwise in ple_dim
proj  = W_ple_proj · gated             # ple_dim → hidden
h    ← h + post_per_layer_input_norm(proj)
```

What it is *for*: a per-layer, token-conditioned bias. Rather than requiring
token identity to survive 42 layers of mixing in the residual stream, every
layer gets direct access to a learned function of the token id. Cheap — the
inner dimension is 256, not 10240.

Performance trap worth knowing before you profile: on Q4_K_M GGUFs the PLE
gate/projection tensors are **F32 on disk**, and requantizing them to Q4_0 sent
them down a slow path. Keeping them dense f16 with `mul_mm_f16` cut prefill PLE
time from ~1555 ms to ~230 ms at 4k (`AGENTS.md` E22). Nothing about PLE is
intrinsically slow; the dtype decision was.

---

# Part H — Final norm, lm_head, softcap

```text
h_final = RMSNorm_final(h)
logits  = W_lm · h_final                      # [262144]
logits  = cap · tanh(logits / cap)            # cap = final_logit_softcapping
```

`W_lm` may be tied to `token_embd` (`tie_word_embeddings`), in which case there
is no separate output tensor to load.

The engine has three decode modes, and knowing them explains several
otherwise-odd call sites:

| Mode | Does | Used by |
|---|---|---|
| `Logits` | final norm + lm_head + softcap, read 262144 floats to CPU | server decode, sampling |
| `Sample` | as above but softcap + argmax on GPU, read one int | greedy paths |
| `Advance` | **skips** norm and lm_head entirely | KV-only advance (speculative rewind) |

`Advance` exists because after a speculative rewind you sometimes need the
cache updated for a token whose logits you already have. Skipping the 380 MB
`lm_head` read makes that nearly free.

---

# Part I — Weight formats at layer granularity

`WeightFormat` is `F16 | Q4_0 | Q3_0 | KQuant`, stored per layer.

`KQuant` means "this layer is not uniformly Q4_0 or F16" — the actual per-tensor
type (Q4_K, Q6_K, F32, F16) lives in `BufferView.format`. So there are two
levels of format information, and they answer different questions:

- `layer.weight_format` → which *branch* of the encode cascade to take.
- `view.format` → which *kernel* to dispatch for that tensor.

One global consequence: the fused decode executor requires **all** layers to be
Q4_0 or K-quant. A single F16 layer disables fusion for the whole model
(`fused_decode_eligible`, `decode_fused.rs` ~101). Worth knowing before you
wonder why a model is unexpectedly slow — check the startup log line from
`log_fused_decode_status`.

---

# Part J — How config reaches the engine

```text
GGUF metadata (or HF config.json)
  → gemma4_config_from_gguf  →  Gemma4TextConfig
  → per-layer assembly: BufferViews, head_dim, q_out/kv_out, intermediate_size,
                        is_full_attention, has_kv, weight_format
  → kv_source_layer pass (Part D.3)
  → scratch allocation using max over layers (Ch 17 Part A)
  → KV allocation using KvCacheType::bytes_per_row(layer_head_dim(i))
  → forward_* reads layer fields every token
```

The startup log is your friend here. It prints the KV-sharing split:

```1398:1407:src/gemma4_gpu_model.rs
        if first_kv_shared < num_layers {
            println!(
                "  KV sharing: layers 0-{} have own KV, layers {}-{} share",
                first_kv_shared - 1,
                first_kv_shared,
                num_layers - 1
            );
        } else {
            println!("  KV sharing: all {} layers have own KV (no sharing)", num_layers);
        }
```

## The per-layer debugging protocol

When something is wrong "only on layer 30," print these seven fields for that
layer and its anchor:

```text
layer_types[30]                  sliding or full?
head_dim                         128 or 512?
has_kv                           should be false for 30 in E4B
kv_source_layer                  must be < first_kv_shared
layer_types[kv_source_layer]     MUST equal layer_types[30]
weight_format                    Q4_0 / KQuant / F16
q_out_dim, kv_out_dim            n_q×hd and n_kv×hd for THIS layer
```

Five of the seven have bitten this codebase at least once. The anchor-type
match on line 5 is the highest-yield single check.

---

# Part K — Silent bug catalog for this architecture

| Mistake | Symptom | Why it is silent |
|---|---|---|
| SWA/full polarity inverted | garbage | wrong head_dim → loud, actually |
| `1/√d` kept with QK-norm | vague, hedging text | softmax flattens, still valid |
| Anchor of the wrong type | wrong on shared layers only | shapes may still fit |
| Scratch sized to sliding head_dim | corruption on full layers | may only show at some contexts |
| `layer_scalar` omitted | slow drift over depth | each layer only slightly off |
| PLE branch dropped | slightly worse output | residual identity default (Ch 00b 2.3) |
| Partial rotary applied to all 512 | degradation at long context | rotation is still a rotation |
| Softcap omitted | sampling params behave differently | logits are still ordered the same |

Notice the pattern: almost everything here is a *scale or selection* error, and
almost nothing crashes. That is the argument for reference comparison as a
default habit rather than a last resort.

---

## Exercises

1. Your GGUF has 42 layers, `num_kv_shared_layers = 18`, and a 5-sliding:1-full
   pattern starting with sliding at index 0. List `layer_types` for indices
   20–30, then compute `kv_source_layer` for layers 24 and 29 using the code in
   Part D.3.
2. Compute total Q4_0 KV bytes per position for E4B, accounting for both layer
   types and only the 24 owning layers.
3. A full layer has `q_proj` of shape [10240, 2560]. In Q4_0, how many bytes?
   Compare to a sliding layer's [2560, 2560].
4. `default_global_head_dim()` returns 512. What happens if your model actually
   uses 256 but the GGUF omits the key? Which buffers are over-sized, and which
   kernel variant gets selected?
5. Explain why full layers use θ=1e6 *and* partial rotary 0.25, when either
   alone would extend range.
6. `fused_decode_eligible` requires all layers Q4_0 or K-quant. Devise a
   startup check that would tell a user immediately why fusion is off.

## Checklist

- [ ] Name the five Gemma4-specific features and one engine consequence of each.
- [ ] Draw sliding vs full: head_dim, Q/KV widths, RoPE, cache row bytes.
- [ ] Explain why `head_dim` is a per-layer property and what it breaks.
- [ ] State the shared-KV saving in memory and in skipped work.
- [ ] Reproduce the anchor-selection algorithm and name the `AGENTS.md` #12 risk.
- [ ] List what shared layers still compute.
- [ ] Justify `attention_scale = 1.0` and predict the symptom of keeping `1/√d`.
- [ ] Explain partial rotary as a capacity-allocation decision.
- [ ] Name the two levels of weight-format information and what each selects.
- [ ] Recite the seven-field per-layer debugging print.

**Next:** [03_weights_and_gguf.md](03_weights_and_gguf.md)
