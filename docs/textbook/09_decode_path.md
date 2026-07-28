# 09 — Decode Path: One Token Through the Whole Stack

This chapter is a guided trace of `forward_single_token_inner`. Keep two files
open and scroll along:

```text
src/gemma4_gpu_model.rs   # forward_single_token_inner  ~3364
src/decode_fused.rs       # encode_fused_decode_layer   ~117
```

Prerequisites: [02](02_gemma4_architecture.md) (what a layer must do),
[05](05_quantization_matmul.md) (matvec), [07](07_attention.md) Part D (the
fused attention kernel), [08](08_mlp_norms_ple.md) (MLP/norms/PLE).

---

# Part A — The problem, with a budget

Inputs and outputs:

```text
in:  token_id, kv_seq_len (positions already cached), mode
out: KV cache grown by 1 at every has_kv layer
     Sample  → one token id
     Logits  → 262144 softcapped floats
     Advance → nothing (no final norm, no lm_head)
```

Before tracing code, fix the budget in your head. At 50 tok/s a token costs
**20 ms**. Where can it go?

| Term | Estimate | Source |
|---|---|---|
| weight bytes ÷ bandwidth | ~12.5 ms | 2.5 GB Q4_0 ÷ 200 GB/s (Ch 00c A.1) |
| KV cache reads | ~0.1–1 ms | grows with context (Ch 00b 9.2) |
| dispatch overhead | ~2.4 ms | ~455 dispatches × ~5 µs (`AGENTS.md` #1) |
| CPU: embed gather, sampling, bookkeeping | ~1 ms | Parts B and H |
| measured GPU compute | ~18 ms | `AGENTS.md` #1 |

So decode is a memory-streaming job with a fixed ~2.4 ms of CPU submission
overhead layered on top. Nothing you will read below is compute-limited, and
that is why the trace is mostly about *how many passes over memory* each phase
costs.

---

# Part B — Phase 0: CPU embedding

```3398:3410:src/gemma4_gpu_model.rs
        // CPU embedding lookup (mmap-backed, no GPU gather)
        self.embed_tables
            .decode_embed_into(token_id, hidden_size, &mut self.embed_decode_scratch);
        MetalContext::write_buffer(&self.hidden_buf, &self.embed_decode_scratch);
        if ple_dim > 0 {
            self.embed_tables.decode_ple_into(
                token_id,
                ple_total_dim,
                ple_dim,
                &mut self.ple_decode_scratch,
            );
            MetalContext::write_buffer(&self.ple_token_id_buf, &self.ple_decode_scratch);
        }
```

Two lookups, both on the CPU:

1. **Token embedding** → `hidden_buf`, i.e. `E[token] × √hidden_size`
   (Ch 00b Part 2.2).
2. **PLE token identity** → `ple_token_id_buf`, `[num_layers × ple_dim]` =
   42 × 256 = 10752 floats, scaled by `√ple_dim`.

Why on the CPU: the embedding table is mmap-backed and possibly BF16/F16 on
disk. Gathering one row of 2560 values is ~5 KB of work — a GPU dispatch would
cost more in launch overhead than the copy costs outright. The PLE table gather
is 10752 values (~43 KB), still small.

Two things to notice about the mechanics:

- `write_buffer` is a CPU write into a `StorageModeShared` buffer, and it
  happens **before** any encoding. That satisfies coherence rule 2 from Ch 00c
  Part H: finish CPU writes, then encode, then commit.
- `embed_decode_scratch` is a reused `Vec`, not a fresh allocation. At 50
  tok/s an allocation per token would be harmless, but the pattern matters at
  batch — and it keeps the hot path allocation-free.

State after Phase 0: `hidden_buf` holds the residual stream for this token, the
KV cache does not yet know about it.

---

# Part C — Phase 1: open a command buffer

```3421:3426:src/gemma4_gpu_model.rs
        // `mut` so the phase profiler (or METAL_N_CB>=2) can flush and re-open
        // command buffers mid-token. Default: rope+PLE+layers[0..mid] on CB0,
        // layers[mid..]+head on CB1 — commit CB0 without wait so encode of CB1
        // overlaps GPU execution of CB0.
        let mut cmd = self.ctx.queue.new_command_buffer();
        let mut encoder = cmd.new_compute_command_encoder();
```

Read the comment carefully — it documents a real technique. With
`METAL_N_CB >= 2` the token is split into two command buffers: CB0 is committed
**without waiting**, so while the GPU executes it, the CPU encodes CB1. Since
encoding ~455 dispatches costs ~2.4 ms of CPU time and the GPU work is ~18 ms,
overlapping hides most of that submission cost.

Which is a good moment to internalize the fundamental asymmetry: `encode_*`
calls are CPU work that *writes a script*; nothing executes until `commit()`.
Everything in Parts D through H is script-writing.

---

# Part D — Phase 2: RoPE table fill

```3428:3437:src/gemma4_gpu_model.rs
        // GPU RoPE table fill for all layers (replaces CPU sin/cos + write_buffer).
        self.ctx.encode_rope_fill_decode(
            &encoder,
            &self.decode_rope_cos_packed,
            &self.decode_rope_sin_packed,
            &self.rope_layer_params_buf,
            actual_num_layers as u32,
            self.rope_max_head_dim as u32,
            pos,
        );
```

**One dispatch fills the tables for all 42 layers.** Layout is
`cos_packed[layer * max_head_dim + d]`, so layer `L`'s table starts at
`L * rope_max_head_dim` and attention binds `cos_buf + rope_off(L)`
(`decode_rope_byte_offset`).

Per-layer parameters — θ, `rope_angles`, `head_dim`, `factor` — come from
`rope_layer_params_buf`, a small table uploaded once at load. So the kernel
handles the sliding/full distinction (θ 10 000 vs 1e6, partial rotary 1.0 vs
0.25) by reading data, not by branching on layer index (Ch 00b Part 5.2).

Note `pos` is `self.total_tokens as f32`, not `kv_seq_len`. Those differ after
a sliding-window eviction or an MTP rewind, and RoPE must use the *absolute*
position — Ch 13 Part B.2.

The alternative design — compute sin/cos on the CPU and `write_buffer` — was
replaced. At 42 layers × 512 channels that is ~21k transcendental functions per
token on one core, versus one trivially parallel GPU dispatch.

---

# Part E — Phase 3: PLE pre-pass

```3446:3489:src/gemma4_gpu_model.rs
        if !__ablate.skip_ple() {
            // Step 2a: context_proj = per_layer_model_projection @ embed
            self.ctx.encode_matvec_auto_view(
                encoder,
                &self.per_layer_model_projection_weight,
                &self.hidden_buf,
                &self.ple_context_proj_buf,
                ple_total_dim as u32,
                hidden_size as u32,
            );
            // Step 2b: context_proj *= 1/sqrt(hidden_size)
            self.ctx.encode_vec_scale(
                encoder,
                &self.ple_context_proj_buf,
                &self.ple_combined_buf,
                ple_total_dim as u32,
                context_proj_scale,
            );
            // Step 2c: RMSNorm per layer
            self.ctx.encode_rmsnorm_per_head_view(
                encoder,
                &self.ple_combined_buf,
                &self.per_layer_projection_norm_weight,
                &self.ple_context_proj_buf,
                num_layers as u32,
                ple_dim as u32,
                eps,
            );
            // Step 3: combined = (context_proj + token_identity) * 1/sqrt(2)
            self.ctx.encode_vec_add(
                encoder,
                &self.ple_context_proj_buf,
                &self.ple_token_id_buf,
                &self.ple_combined_buf,
                ple_total_dim as u32,
            );
            self.ctx.encode_vec_scale(
                encoder,
                &self.ple_combined_buf,
                &self.ple_context_proj_buf,
                ple_total_dim as u32,
                ple_input_scale,
            );
        }
```

Five dispatches, and the interesting parts are the details:

**The buffer ping-pong.** Watch the destinations: `ple_context_proj_buf` →
`ple_combined_buf` → `ple_context_proj_buf` → `ple_combined_buf` →
`ple_context_proj_buf`. Two buffers alternate because none of these kernels is
safe in-place (a scale is, a norm is not), and alternating avoids allocating
five. The final result lands in `ple_context_proj_buf`, which is what the layer
loop slices.

**`rmsnorm_per_head_view` is reused for a non-attention purpose.** It normalizes
`num_layers` independent segments of `ple_dim` each — the same kernel shape as
QK-norm's "normalize `n_heads` segments of `head_dim`." One kernel, two
unrelated callers, because the *shape* of the operation is what matters
(Ch 08 Part A.3).

**Two scale factors, applied separately.** `1/√hidden_size` on the projection,
then `1/√2` on the sum with the token identity. The second is the classic
variance-preserving combination of two roughly unit-variance signals. Both are
easy to omit and neither crashes.

**Layout for the layer loop.** Everything is contiguous:
`ple_context_proj_buf[layer * ple_dim ...]`, so a layer reads its slice at byte
offset `layer_idx * ple_dim * 4` with no copy. The comment above the block notes
this replaced 42 per-layer copy-out dispatches — a 42-dispatch saving, about
0.2 ms/token by the Part A model.

`PROFILE_ABLATE`'s `skip_ple` gate lets you measure the whole pre-pass by
deletion (Ch 15 Part B.3).

---

# Part F — Phase 4: the layer loop

## F.1 Eligibility for the fused path

```101:114:src/decode_fused.rs
    pub fn fused_decode_eligible(&self) -> bool {
        if !fused_decode_enabled() {
            return false;
        }
        if self.kv_cache_type != KvCacheType::Q4_0 {
            return false;
        }
        if !self.ctx.use_flash_attention {
            return false;
        }
        self.layers.iter().all(|l| {
            l.weight_format == WeightFormat::Q4_0 || l.weight_format.is_kquant()
        })
    }
```

Four conditions, all model/config properties, evaluated once. Note the last one
is `all()` over layers: **a single F16 layer disables the fused executor for the
entire model.** If a model is mysteriously slow, `log_fused_decode_status`
prints which condition failed — check that before profiling anything.

The non-fused fallback is the older per-op path inline in
`gemma4_gpu_model.rs`. It computes the same thing with more dispatches.
`AGENTS.md` #4 measured both and found the same ~47 tok/s, which is how they
established that the context-scaling problem lived in attention rather than in
dispatch structure.

## F.2 Per-layer setup, and the sliding-window arithmetic

```127:153:src/decode_fused.rs
        let layer = &self.layers[layer_idx];
        let hidden_size = self.config.hidden_size as u32;
        let num_heads = self.config.num_attention_heads as u32;
        let num_kv_heads = self.config.layer_num_kv_heads(layer_idx) as u32;
        let num_kv_groups = (num_heads / num_kv_heads.max(1)) as u32;
        let head_dim = layer.head_dim as u32;
        let q_out = layer.q_out_dim as u32;
        let kv_out = layer.kv_out_dim as u32;
        let intermediate_size = layer.intermediate_size as u32;
        let ple_dim = self.config.hidden_size_per_layer_input as u32;
        let eps = self.config.rms_norm_eps as f32;
        let scale = 1.0f32;
        let rope_off = self.decode_rope_byte_offset(layer_idx);
        let is_full = layer.is_full_attention;
        let attn_kv_seq = kv_seq + 1;
        let effective_kv_seq = if is_full {
            attn_kv_seq
        } else {
            attn_kv_seq.min(self.config.sliding_window as u32)
        };
        let kv_start = if !is_full && attn_kv_seq > self.config.sliding_window as u32 {
            attn_kv_seq - self.config.sliding_window as u32
        } else {
            0u32
        };
        let groups_per_row = head_dim / 32;
        let row_bytes = groups_per_row * 18;
```

This block is worth reading twice, because it is where every Gemma4 quirk from
Ch 02 becomes concrete numbers:

- `num_kv_heads` and `head_dim` come from the **layer**, not the model.
- `scale = 1.0` — QK-norm, hard-coded (Ch 02 Part E).
- `attn_kv_seq = kv_seq + 1`: attention includes the current token, which is not
  yet in the cache. The fused kernel handles it from threadgroup memory
  (Ch 07 Part D.4).
- **The window arithmetic.** For a sliding layer, `effective_kv_seq` is capped
  at `sliding_window` and `kv_start` slides forward once the context exceeds the
  window. So a sliding layer's attention cost stops growing at 512 keys — the
  whole point of SWA.
- `row_bytes = (head_dim/32) × 18` — the Q4_0 KV row stride (Ch 06 Part C).
  Note this is derived from *this layer's* head_dim, which for a shared-KV layer
  must match the anchor's (Ch 02 Part D.4).

Worked example, sliding layer with `sliding_window = 512`:

| `kv_seq` | `attn_kv_seq` | `effective_kv_seq` | `kv_start` |
|---|---|---|---|
| 100 | 101 | 101 | 0 |
| 511 | 512 | 512 | 0 |
| 600 | 601 | 512 | 89 |
| 4000 | 4001 | 512 | 3489 |

And a full layer at `kv_seq = 4000` gets `effective_kv_seq = 4001, kv_start =
0`. That table is the reason a long chat degrades gracefully rather than
linearly: only the full-attention layers pay for the whole context.

One subtlety with consequences for Ch 07: `effective_kv_seq` is also what the
hybrid router compares against its threshold, so sliding layers *never* exceed
512 and therefore may never route to the MWG kernel, while full layers do.

## F.3 The three sub-blocks

```156:157:src/decode_fused.rs
        if !skip_attn {
            n += self.encode_fused_attn_layer(
```

The layer body is three calls plus a scale, each returning its dispatch count
(`n`) so `PROFILE_DISPATCHES` can report per-layer totals:

```text
encode_fused_attn_layer   → Ch 07 (the fusion ladder and the kernel body)
encode_fused_mlp_layer    → Ch 08 Part C (the four-branch choice)
encode_fused_ple_layer    → Ch 08 Part D
encode_vec_scale          → h *= layer_scalar
```

Dispatch budget per layer, from Ch 08 Part F:

| Sub-block | Dispatches (Q4_K, fused path) |
|---|---|
| attention (has_kv): norm+QKV, attention, O proj, residual acc | ~5 |
| MLP: norm+gate‖up+gelu, down, acc | ~4 |
| PLE: gate+gelu, proj, acc | 3–4 |
| layer scalar | 1 |
| **total** | **~13–14** |

Shared-KV layers are cheaper (no K/V projections, no append). Across 42 layers
that lands in the 450–600 range, and `AGENTS.md` #1 measured **455 per token**
on the fused path. When you see that number, you now know where every one of
them comes from.

`skip_attn` / `skip_mlp` / `skip_ple` are the `PROFILE_ABLATE` hooks: run a
real token with one sub-block deleted and diff the wall time. Crude, effective,
and how the prefill phase table in Ch 10 Part H was built.

---

# Part G — What a shared-KV layer does differently

Trace layer 30 (E4B: `has_kv = false`, anchor ≈ 22–23):

```text
same:      input norm → Q projection → Q-norm → Q-RoPE
skipped:   K projection, V projection, K-norm, V-norm, K-RoPE, KV append
different: attention reads k_cache[22] / v_cache[22]
same:      o_proj, post-attn norm + residual, MLP, PLE, layer_scalar
```

Two invariants make this safe, both worth stating precisely:

1. **Anchor is already written.** `kv_source_layer < first_kv_shared` and layers
   execute in increasing order in one command buffer, so layer 22's append for
   *this* position completed before layer 30 reads. Sequential dispatch ordering
   is the only mechanism (Ch 00c Part G) — there is no fence.
2. **Geometry comes from the anchor.** `row_bytes` and `head_dim` must describe
   layer 22's cache. Same type ⇒ same numbers, which is why anchors must match
   type (Ch 02 Part D.3).

And one thing that is *not* skipped: the fused attention kernel for a shared
layer is a different kernel (`attention_flash_decode_qknorm_rope_q4_0`, rung 2)
rather than `full_fused` (rung 1), because there is no K/V to prepare or
append. Rungs, not flags.

---

# Part H — Phase 5: final norm, lm_head, softcap

```text
if mode != Advance:
    normed = rmsnorm(hidden, final_norm_weight)
    logits = lm_head · normed              # [262144]
    logits = cap · tanh(logits / cap)
    Sample → GPU argmax → read back 1 int
    Logits → read back 262144 floats
```

The costs here are worth knowing because they are not small:

- `lm_head` in Q4_0 is `262144 × 2560 × 0.5625 B ≈ 378 MB`. At 200 GB/s that
  is **~1.9 ms — roughly 10% of the token budget in one dispatch.**
- Reading 262144 floats back to the CPU is 1 MB per token, plus a
  `wait_until_completed` sync.

Which explains three design choices you will meet elsewhere:

1. **`DecodeMode::Advance` skips it entirely.** Used when the cache needs
   updating but the logits are already known (speculative rewind, Ch 14).
2. **`Sample` does softcap and argmax on the GPU** and reads back one integer,
   avoiding the 1 MB transfer, for greedy paths.
3. **Prefill sets `want_logits` only on the last chunk row** (Ch 10 Part G),
   because computing `lm_head` for interior prompt positions is pure waste.

If `tie_word_embeddings` is set, `lm_head` *is* the embedding matrix — the same
378 MB you gathered one row from in Phase 0.

---

# Part I — Batch decode

For N concurrent requests the scheduler calls `decode_batch`, which becomes
`forward_decode_batch_with_kv_slots`. Mental model: **N copies of Parts B–H in
one command buffer**, with each copy pointed at its own KV slot.

What batching does and does not buy:

| | Effect |
|---|---|
| Weight reads | **shared** — read once for all N. This is the win. |
| KV reads | N× (different sequences, different lengths) |
| Attention dispatches | N× (one per sequence per layer) |
| Dispatch overhead | amortized across N tokens |
| Sampling (CPU) | N× |

Weight bytes are ~75% of decode traffic, so N=4 does not cost 4× — it costs
maybe 1.5×, which is why continuous batching works at all.

Critically, **batch decode is N independent sequences, not one sequence of N
tokens** (Ch 13 Part D.2). Each has its own `kv_seq_len`, so the attention
kernel is dispatched per sequence and there is no cross-sequence tiling. Do not
confuse this with MTP verify, which *is* one sequence of N tokens and therefore
uses completely different kernels (Ch 14 Part E).

`max_decode_batch_size` chunks larger batches, because scratch buffers are
sized for a maximum.

---

# Part J — Slot binding: do not skim this

In the server path, the model's `k_cache`/`v_cache` fields are made to *point
at* a pool slot's buffers before the forward runs:

```text
forward_single_token_with_kv_slot(token, slot)
  ├─ alias_kv_from_pool(slot)      # k_cache[i] now refers to slot's buffers
  ├─ self.kv_seq_len = slot.seq_len
  ├─ self.total_tokens = slot.total_tokens
  ├─ forward_single_token_inner(...)
  └─ write back seq_len / total_tokens into the slot
```

The forward code is therefore identical for CLI and server paths — it never
knows which it is in. Elegant, and it carries two hazards:

1. **Forget the alias and you write into whichever slot ran last.** Symptom:
   requests contaminating each other's context, worst under concurrency, absent
   in single-request testing.
2. **Forget the counters and RoPE/window arithmetic is wrong.** `kv_seq_len`
   and `total_tokens` are *both* needed and they are not the same number
   (Ch 13 Part B.2).

If you add a forward path, copy an existing wrapper rather than writing the
binding by hand.

---

# Part K — Timing and profiling

Instrumentation available in this very function:

```3369:3377:src/gemma4_gpu_model.rs
        let __profile = std::env::var("PROFILE_DECODE").is_ok();
        // When set, splits the token into per-phase command buffers and times
        // each phase's commit→wait (wall clock). Each phase is its own command
        // buffer dominated by GPU execution, so the numbers isolate where the
        // floor is; they include a small fixed sync per phase (note in output).
        let __pp = std::env::var("PROFILE_PHASES").is_ok();
        let __ablate = ProfileAblate::from_env();
        let layer_count = self.layers.len() as u32;
        let mut __gpu_prof = if !__pp && !__ablate.active() && profile_gpu_enabled() {
```

| Knob | What it does | Distortion to keep in mind |
|---|---|---|
| `PROFILE_DECODE` | CPU wall time per region | includes encode time |
| `PROFILE_PHASES` | one command buffer per phase, timed commit→wait | adds a sync per phase; kills CB overlap |
| `PROFILE_GPU` | GPU timestamps at marks | mutually exclusive with the above |
| `PROFILE_DISPATCHES` | counts dispatches | no timing |
| `PROFILE_ABLATE` | skips sub-blocks | changes what runs, so totals are not additive |

Note the mutual exclusion in the code: `__gpu_prof` is only created when
neither `PROFILE_PHASES` nor ablation is active. Combining them would measure
an artefact. That guard is itself the lesson — **measurement modes that alter
command buffer structure cannot be combined with ones that assume it.**

Expected shape of the numbers at 200-token context (from `AGENTS.md`):

```text
total ≈ 20 ms
  layers  ≈ 18 ms      (of which MLP ~ half, attention ~ a third)
  lm_head ≈ 1.9 ms
  rope + PLE pre-pass  < 0.5 ms
  CPU (embed, sample, encode) ≈ 1–2 ms, mostly overlapped
```

---

# Part L — Debugging protocol

Match the symptom to the phase; do not start by reading kernels.

| Symptom | Likely phase | First check |
|---|---|---|
| First token wrong, rest plausible | B (embedding) or H (softcap) | √hidden scale; cap value |
| Fluent but off-distribution from token 1 | E (PLE), F.2 (scale=1.0), layer_scalar | ablate PLE; print `scale` |
| Correct then garbage after ~128 tokens | F (routing switch) | `needs_explicit_kv_append` (Ch 07 F.5) |
| Correct then garbage after ~512 tokens | F.2 (window arithmetic) | `kv_start` / `effective_kv_seq` table |
| Wrong only in long context | attention scaling | compare `ATTENTION_KERNEL` modes |
| Requests contaminate each other | J (slot binding) | is `alias_kv_from_pool` called? |
| Wrong only on some layers | Ch 02 Part J | the seven-field per-layer print |
| Slow, not wrong | F.1 | `log_fused_decode_status`; then `PROFILE_ABLATE` |

The pattern behind the middle rows: **"correct for N tokens, then wrong" almost
always means a threshold was crossed.** Find the threshold (128 → hybrid
routing, 512 → sliding window, chunk size → prefill boundary) and you have
found the bug's neighbourhood.

---

# Part M — Exercises

1. Count dispatches for one token by hand from Parts D–H (rope + PLE pre-pass +
   42 × per-layer + head). Compare with `PROFILE_DISPATCHES=1`.
2. Fill in the `effective_kv_seq` / `kv_start` table for a sliding layer with
   `sliding_window = 1024` at `kv_seq` = 500, 1023, 1024, 5000.
3. At `kv_seq = 4000`, compute KV bytes read per token for one sliding layer and
   one full layer (Q4_0, `n_kv = 4`, head_dim 128/512). Which dominates?
4. `lm_head` is ~1.9 ms of a 20 ms token. What decode rate would you get if it
   were free? Now explain why MTP's batched `lm_head` (Ch 14 Part D) mattered.
5. Set `METAL_N_CB=1` and `METAL_N_CB=2`, measure with `--bench-decode`, and
   explain the difference using the ~2.4 ms encode cost.
6. Trace what happens if you call `forward_single_token_inner` directly in the
   server path without `alias_kv_from_pool`. At which line does the first wrong
   byte get written?
7. Delete the `1/√2` scale in the PLE pre-pass. Predict the effect on output
   quality, then reason about why no test would fail.
8. `PROFILE_PHASES` reports phase times that sum to more than the unprofiled
   token time. Explain both reasons.
9. The PLE pre-pass uses two buffers in a ping-pong. Identify which of the five
   dispatches could safely run in place, and how many buffers a minimal
   implementation needs.

---

## Checklist

- [ ] I can state the 20 ms budget and its four main terms.
- [ ] I know why the embedding gather is on the CPU.
- [ ] I can explain the `METAL_N_CB` overlap and what it hides.
- [ ] I can describe the packed RoPE table layout and why `pos ≠ kv_seq_len`.
- [ ] I can list the five PLE pre-pass dispatches and the two scale factors.
- [ ] I can recite the four `fused_decode_eligible` conditions.
- [ ] I can derive `effective_kv_seq` and `kv_start` for any layer type and context.
- [ ] I can account for ~455 dispatches per token.
- [ ] I know what a shared-KV layer skips, and the two invariants that allow it.
- [ ] I can explain why `lm_head` is 10% of the budget and how each mode avoids it.
- [ ] I can explain what batch decode shares and what it does not.
- [ ] I can map a symptom to a phase using Part L.

**Next:** [10_prefill_path.md](10_prefill_path.md) — the same model, the other
regime.
