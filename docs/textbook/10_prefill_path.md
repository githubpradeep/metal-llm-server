# 10 — Prefill: Many Tokens at Once

Decode is a bandwidth problem. Prefill is a **compute** problem, and that
single difference changes every kernel choice: matvec becomes matmul,
per-head loops become tiled MMA, and layouts get transposed twice per layer
because attention wants a different memory order than the projections do.

This chapter walks a prefill chunk end to end, in the order the code does it.
Open:

```text
src/gemma4_gpu_model.rs
  PrefillBatchSegment                     ~366
  encode_parallel_prefill_attention_inputs ~6046
  can_use_parallel_prefill_chunk          ~6229
  encode_parallel_prefill_layer_batched   ~6635
  forward_prefill_batch_with_kv_slots     ~8763
  forward_prefill_chunked_with_kv_slot    ~8929
src/shaders/ggml_flash_attn_ext.metal
src/shaders/mul_mm.metal
```

---

## Part A — Why prefill is a different machine

### A.1 Arithmetic intensity flips

Take the MLP gate projection, `[10240, 2560]` in Q4_0 (~14 MB):

| | Weight bytes read | FLOPs | Intensity |
|---|---|---|---|
| Decode (1 token) | ~14 MB | 2·10240·2560 ≈ 52 MFLOP | ~3.7 FLOP/byte |
| Prefill (512 tokens) | ~14 MB | 512× that ≈ 27 GFLOP | ~1900 FLOP/byte |

M1 Pro's ratio of compute to bandwidth is roughly 3.2 TFLOPS ÷ 200 GB/s ≈
16 FLOP/byte. Below that you are bandwidth-bound; above it, compute-bound.
Decode sits at ~4 (bandwidth-bound, hopelessly). Prefill at 512 tokens sits
at ~1900 (compute-bound, comfortably).

**Consequence:** in decode you optimize bytes moved. In prefill you optimize
FLOP efficiency — tiling, MMA utilization, unrolling. Fusing to avoid a
scratch buffer barely matters at 4k; a badly tiled matmul costs you 2×.

This is why the same logical operation has two implementations everywhere in
this codebase: `matvec_*` for decode, `mul_mm*` for prefill.

### A.2 The other difference: causality

Decode attends to all cached tokens; every one is in the past. Prefill has
`seq × seq` interactions where roughly half must be masked. Token 5 in the
chunk may not see token 6. That mask lives **inside** the attention kernel,
not in a separately materialized `[seq, seq]` matrix.

---

## Part B — Prefill scratch and admission checks

`prefill_scratch` is a pre-allocated set of buffers sized for
`max_seq_len` tokens: `hidden_buf`, `normed_buf`, `q_buf`, `k_buf`, `v_buf`,
`q_normed_buf`, `k_normed_buf`, `qkv_stacked_buf`, `attn_out_buf`,
`o_out_buf`, MLP buffers, and so on. Allocated once at model load; a chunk
that does not fit is rejected rather than triggering an allocation on the
hot path:

```6052:6060:src/gemma4_gpu_model.rs
if seq_len == 0 {
    return Err("prefill seq_len must not be empty".to_string());
}
if seq_len > self.prefill_scratch.max_seq_len {
    return Err(format!(
        "prefill chunk has {} tokens, max supported chunk is {}",
        seq_len, self.prefill_scratch.max_seq_len
    ));
}
```

And the parallel path has its own gate:

```6229:6243:src/gemma4_gpu_model.rs
fn can_use_parallel_prefill_chunk(&self, start_pos: usize, seq_len: usize,
                                  kv_pool: &KvCachePool) -> bool {
    if seq_len <= 1 || seq_len > self.prefill_scratch.max_seq_len { return false; }
    if start_pos + seq_len > kv_pool.capacity() as usize { return false; }
    true
}
```

`seq_len <= 1` falls back to the decode path — a one-token "prefill" is just
a decode step, and the decode kernels are better at it. This is the same
threshold idea as `TILED_EXT_MIN_Q`: pick the kernel family that matches the
shape, not the phase name.

The scheduler (Ch 12) is what guarantees chunks fit: it clamps every chunk to
`engine.max_prefill_chunk_tokens()`.

---

## Part C — Attention inputs: the layout journey

`encode_parallel_prefill_attention_inputs` is where most of the layout work
happens. Follow the shapes.

### C.1 Norm, then QKV as a real matmul

```6072:6087:src/gemma4_gpu_model.rs
self.ctx.encode_rmsnorm_batch_view(
    encoder,
    &self.prefill_scratch.hidden_buf,
    &layer.input_layernorm_weight,
    &self.prefill_scratch.normed_buf,
    hidden_size as u32, eps, seq_len as u32,
);

self.encode_prefill_attention_qkv(encoder, layer, seq_len as u32, hidden_size as u32);
```

`rmsnorm_batch` is the batched sibling of `rmsnorm`: **one threadgroup per
token row**, `tgid` selects `row_offset = tgid * dim` (see
`rmsnorm_acc_batch` in `llama.metal` ~1748 for the exact pattern). Same three
phases as Ch 08, now `seq_len` times in parallel — which is exactly why
prefill saturates the GPU and decode does not.

`encode_prefill_attention_qkv` routes to `mul_mm` (or the stacked variant if
`use_prefill_qkv_stacked(layer)`), producing:

```text
normed_buf:      [seq, 2560]
q_buf:           [seq, q_out]     q_out  = num_heads    * head_dim
k_buf, v_buf:    [seq, kv_out]    kv_out = num_kv_heads * head_dim
```

Or, when stacked, one `qkv_stacked_buf` of `[seq, q_out + 2*kv_out]` that a
split kernel later carves up. Stacking exists so one `mul_mm` reads `normed`
once for all three projections instead of three times.

### C.2 The two post-projection paths

There is a fused path and a decomposed path. They compute the same thing;
read the decomposed one first because it names every step:

```6116:6186:src/gemma4_gpu_model.rs
if stacked {
    self.ctx.encode_qkv_split_stacked_batch(...);        // 1. split Q|K|V
}
self.ctx.encode_rmsnorm_batch_view(
    encoder, &self.prefill_scratch.q_buf, &layer.q_norm_weight,
    &self.prefill_scratch.q_normed_buf,
    head_dim as u32, eps, (seq_len * num_heads) as u32);  // 2. Q-norm, N=head_dim
if layer.has_kv {
    self.ctx.encode_rmsnorm_batch_view(... k_buf → k_normed_buf,
        head_dim, eps, (seq_len * num_kv_heads) as u32);  // 3. K-norm
}
self.ctx.encode_transpose_shd(
    encoder, &self.prefill_scratch.q_normed_buf, &self.prefill_scratch.q_buf,
    seq_len as u32, num_heads as u32, head_dim as u32);   // 4. [s,h,d] → [h,s,d]
if layer.has_kv {
    self.ctx.encode_transpose_shd(... k_normed_buf → k_buf ...);
    self.ctx.encode_rmsnorm_noweight_batch(... v_buf → k_normed_buf,
        head_dim, eps, (seq_len * num_kv_heads) as u32);  // 5. V-norm (no weight)
    self.ctx.encode_transpose_shd(... k_normed_buf → v_buf ...);
}
```

Five ideas in there, all worth naming:

1. **QK-norm uses `N = head_dim`, and the "batch" count is
   `seq_len * num_heads`.** Every (token, head) pair is an independent
   reduction over 128 or 256 values. Get this count wrong and you normalize
   across head boundaries — plausible-looking output, wrong model.
2. **V-norm is `noweight`.** Gemma4 normalizes V with no learned scale.
3. **`transpose_shd` converts `[seq, heads, dim]` → `[heads, seq, dim]`.**
   Projections naturally produce token-major (each `mul_mm` output row is one
   token). Attention wants head-major, because one threadgroup owns one head
   and wants its `seq` tokens contiguous.
4. **Buffers are reused as ping-pong scratch.** Look closely: step 5 writes
   V-norm output into `k_normed_buf`, then transposes it back into `v_buf`.
   `k_normed_buf` is dead at that point (its contents were already
   transposed into `k_buf`). Cheap, and a trap if you insert a step between
   them.
5. **`has_kv` guards everything K/V.** Shared-KV layers project only Q; K/V
   come from the anchor layer's cache (Ch 02, Ch 06).

The fused path replaces steps 1–5 with one kernel:

```6090:6114:src/gemma4_gpu_model.rs
if crate::gpu::prefill_qkv_hsd_enabled() {
    self.ctx.encode_prefill_qkv_postproj_hsd(
        encoder,
        if stacked { Some(&self.prefill_scratch.qkv_stacked_buf) } else { None },
        &self.prefill_scratch.q_buf, &self.prefill_scratch.k_buf, &self.prefill_scratch.v_buf,
        &self.prefill_scratch.q_normed_buf, &self.prefill_scratch.k_normed_buf,
        &layer.q_norm_weight, &layer.k_norm_weight,
        layer.q_out_dim as u32, layer.kv_out_dim as u32,
        num_heads as u32, num_kv_heads as u32, head_dim as u32,
        seq_len as u32, eps, stacked, layer.has_kv,
    );
}
```

Split + Q-norm + K-norm + V-norm + all three transposes, in one dispatch that
writes directly in HSD order. Six or seven dispatches saved per layer per
chunk, and — more importantly at 4k — several full passes over
`[seq, q_out]` scratch avoided. Keep the decomposed path around: it is your
A/B reference when the fused kernel produces something suspicious.

### C.3 RoPE, from precomputed tables

```6189:6199:src/gemma4_gpu_model.rs
self.ctx.encode_rotary_batch(
    encoder,
    &self.prefill_scratch.q_buf, &self.prefill_scratch.k_buf,
    &self.per_layer_prefill_cos_bufs[layer_idx],
    &self.per_layer_prefill_sin_bufs[layer_idx],
    num_heads as u32,
    if layer.has_kv { num_kv_heads as u32 } else { 0 },
    head_dim as u32, seq_len as u32,
);
```

Note `num_kv_heads` is passed as **0** for shared-KV layers — the kernel then
rotates Q only. Passing the real count there would rotate a `k_buf` whose
contents are stale garbage for that layer.

The tables are filled on GPU once per chunk, per layer, before the layer loop:

```6215:6224:src/gemma4_gpu_model.rs
self.ctx.encode_rope_fill_prefill_batch(
    encoder,
    &self.per_layer_prefill_cos_bufs[layer_idx],
    &self.per_layer_prefill_sin_bufs[layer_idx],
    &self.rope_layer_params_buf,
    layer_idx as u32, start_pos as u32, seq_len as u32, layer.head_dim as u32,
);
```

Per-layer tables because SWA and full layers use different θ (`freq_base_swa`
vs `freq_base`) and possibly different rotary dims (Ch 02). Table shape is
`[seq, head_dim/2]`, indexed by `start_pos + i`. In decode, the same math
fills a `[1, head_dim/2]` slice for a single position (Ch 09).

---

## Part D — Segments: batching multiple requests through one forward

`PrefillBatchSegment` is how several requests share one prefill forward:

```366:371:src/gemma4_gpu_model.rs
struct PrefillBatchSegment {
    slot: KvSlot,
    row_start: usize,
    token_count: usize,
    start_pos: usize,
}
```

Rows of every scratch buffer are concatenated across requests:

```text
total_seq_len = 5
scratch rows:  [ A0 A1 A2 | B0 B1 ]
segments: { slot=A, row_start=0, token_count=3, start_pos=0   }
          { slot=B, row_start=3, token_count=2, start_pos=120 }
```

Two coordinate systems, and confusing them is the classic bug here:

- **`row_start` / `total_seq_len`** — position in the *scratch* buffers.
- **`start_pos`** — position in *that request's* KV cache and RoPE sequence.

Request B is 120 tokens into its conversation while sitting at scratch row 3.
Every strided kernel takes both.

The matmuls (`mul_mm`, MLP, projections) do not care about segments at all:
they see `total_seq_len` independent rows. Only two things are
segment-aware: **KV append** and **attention** — the two operations with
per-request state.

### D.1 Strided KV append

```6678:6689:src/gemma4_gpu_model.rs
self.ctx.encode_kv_batch_append_strided_f16(
    encoder,
    &self.prefill_scratch.k_buf,
    k_cache,
    num_kv_heads as u32,
    head_dim as u32,
    kv_pool.capacity(),        // stride between heads in the cache
    segment.start_pos as u32,  // destination position in this slot
    segment.token_count as u32,
    total_seq_len as u32,      // source row stride (all segments)
    segment.row_start as u32,  // source row offset for this segment
);
```

One dispatch per segment per tensor (K and V), inside the same encoder. The
Q8_0 and Q4_0 variants are identical in structure — the kernel quantizes as
it writes (Ch 06). The loop runs only `if layer.has_kv`: shared layers write
nothing.

### D.2 Strided causal attention

```6813:6837:src/gemma4_gpu_model.rs
KvCacheType::Q4_0 => {
    let groups_per_row = (head_dim / 32) as u32;
    let row_bytes = groups_per_row * 18;
    self.ctx.encode_attention_causal_strided_q4_0(
        encoder,
        &self.prefill_scratch.q_buf, k_cache, v_cache,
        &self.prefill_scratch.attn_out_buf,
        num_heads as u32, num_kv_heads as u32, num_kv_groups, head_dim as u32,
        (segment.start_pos + segment.token_count) as u32,   // kv_seq visible
        kv_pool.capacity(), scale,
        segment.token_count as u32, segment.start_pos as u32,
        attention_window,                                    // 0 = full attention
        total_seq_len as u32, segment.row_start as u32,
        groups_per_row, row_bytes,
    );
}
```

Six things to notice:

- `kv_seq = start_pos + token_count`: this segment's tokens can attend to
  everything already in its cache **plus** the chunk just appended.
- `attention_window` is `0` for full-attention layers, else
  `config.sliding_window`. The mask is computed in-kernel from
  `start_pos + local_row` and the window.
- `row_bytes = (head_dim/32) * 18` for Q4_0 (`* 34` for Q8_0) — the
  quantized row stride. Hardcoded per type at the call site, so a KV layout
  change must be edited in several places (grep `row_bytes`).
- `scale` is `1.0f32` here: Gemma4's query scaling is folded elsewhere
  (query_pre_attn_scalar / RoPE-time scaling), not applied as the classic
  `1/√head_dim` in the kernel. Do not "fix" this by adding it back.
- The cache read uses `layer.kv_source_layer`, not `layer_idx` — shared
  layers read their anchor's cache.
- One dispatch per segment: different requests have different `kv_seq` and
  different windows, so they cannot share a dispatch.

### D.3 Which attention kernel actually runs

Two families live behind these calls:

- **`attention_causal_*`** — the straightforward per-row kernel.
- **`flash_attn_ext`** (`ggml_flash_attn_ext.metal`) — llama.cpp's tiled
  kernel: one threadgroup owns a tile of query rows, streams K/V tiles into
  threadgroup memory, uses simdgroup matrix ops, keeps online-softmax state
  in registers.

The switch is `TILED_EXT_MIN_Q` (default **2**). Its story is in `AGENTS.md`
M4 and it is one of the best lessons in the log: llama.cpp gates tiled vs vec
at `q_len ≥ 20`, and this repo copied that gate. But *our* sub-20 fallback is
much worse than llama.cpp's vec kernel, so inheriting their threshold
inherited a cliff we did not have to have. Lowering it to 2 took MTP verify
from 44 ms to 36 ms and e2e 37.7 → 42.4 tok/s.

**Generalized:** a tuning constant copied from another codebase is only valid
together with that codebase's alternative path. Re-measure every threshold
you port.

For h256 heads, `NSG` was also raised 4 → 8 (24 KB threadgroup memory, near
the 32 KB ceiling), which closed the remaining prefill gap: 581–591 tok/s at
pp4096 versus llama.cpp's 593.9 (AGENTS E23). h512 stays at NSG=4 — it would
not fit.

---

## Part E — After attention: back to token-major

```6842:6867:src/gemma4_gpu_model.rs
self.ctx.encode_transpose_hsd(
    encoder, &self.prefill_scratch.attn_out_buf, &self.prefill_scratch.q_normed_buf,
    total_seq_len as u32, num_heads as u32, head_dim as u32);      // [h,s,d] → [s,h,d]
self.ctx.encode_prefill_projection_auto_batch_view(
    encoder, &layer.o_proj,
    &self.prefill_scratch.q_normed_buf, &self.prefill_scratch.o_out_buf,
    hidden_size as u32, q_out as u32, total_seq_len as u32);
self.ctx.encode_rmsnorm_acc_batch_view(
    encoder, &self.prefill_scratch.hidden_buf, &self.prefill_scratch.o_out_buf,
    &layer.post_attention_layernorm_weight, hidden_size as u32, eps, total_seq_len as u32);
```

Attention output is head-major; `o_proj` is a matmul over `[seq, q_out]`, so
transpose back. **Two transposes of the whole activation tensor per layer**
is the price of letting each kernel family have its preferred layout. At 4k ×
2560 floats that is ~40 MB touched per transpose — measurable but far cheaper
than running attention in the wrong layout.

`q_normed_buf` is reused again here as the transpose destination. Prefill
scratch reuse is aggressive; when you add a step, check what is still live.

Then MLP and PLE, in batched form: `encode_prefill_mlp_gate_up` (Ch 05 Part
H / Ch 08), `gelu_mul_stacked_batch` when gate‖up came out stacked,
`rmsnorm_acc_batch` for the residual epilogues. Same math as decode, batched
kernels throughout.

---

## Part F — Command buffer structure

```6642:6644:src/gemma4_gpu_model.rs
let cmd = self.ctx.queue.new_command_buffer();
let encoder = cmd.new_compute_command_encoder();
self.encode_parallel_prefill_attention_inputs(encoder, layer_idx, total_seq_len)?;
```

**One command buffer per layer** in prefill, versus one (or two) per *token*
in decode. Why the difference: prefill dispatches are long (milliseconds
each), so per-CB overhead is noise, and per-layer CBs bound peak scratch
lifetime and give the driver natural sync points. In decode, dispatches are
microseconds, so CB overhead is a real fraction and you want as few as
possible.

**Rule of thumb:** batch command buffers when work per dispatch is small;
split them when work per dispatch is large.

---

## Part G — Chunking and `want_logits`

`forward_prefill_chunked_with_kv_slot` loops chunks of at most
`max_prefill_chunk_tokens`, advancing `start_pos` each time. State carried
between chunks lives entirely in the KV cache and `start_pos`.

`want_logits` (Ch 12 E.3) is false for all but the last chunk, skipping
`lm_head` — a `[262144, 2560]` matmul, ~440 MB of weights. For a 4k prompt in
1k chunks that is three avoided lm_head passes.

Prefill must also stash the final hidden state where the caller expects it,
because MTP needs it (`prefill_hidden_activation_at`, and the last-row fix
noted in AGENTS M5). For a multi-token chunk the relevant row is the **last**
one, not row 0 — that off-by-one cost real accept-rate before it was found.

---

## Part H — Where prefill time actually goes

From `benchmarks/prefill_phase_4k.txt` (E2B Q4_K_M, `PROFILE_ABLATE`, 4k):

| Bucket | Δms @4k | Share |
|--------|---------|-------|
| MLP | 4488 | 54% |
| Attention | 2930 | 35% |
| PLE | 1555 → ~230 after E22 | 19% → ~3% |
| lm_head | 402 | 5% |
| CB/embed/rope | 135 | 1.6% |

Three lessons live in that table:

1. **MLP dominates.** `gate∥up` alone is ~3.1 s and already at ~peak FLOPS
   for M1 Pro (~2.64 s theoretical at 3.22 TFLOPS). There is nothing left to
   win there without changing the math.
2. **PLE was a dtype bug, not a kernel problem** — F32 tensors requantized to
   Q4_0 landed on a slow path. Keeping them f16 with `mul_mm_f16` cut 1.3 s
   (E22).
3. **Tile alignment was a wash.** Padding 4096 → 4112 changed nothing.
   Plausible-sounding optimizations that measure as noise are the norm; this
   is what `AGENTS.md` is for.

Ablation is the tool: `PROFILE_ABLATE=SKIP_mlp`, `SKIP_ple`, `SKIP_all`, etc.
`SKIP_all` gives you the floor (135 ms of command-buffer plumbing) so you
know what the buckets are measured against.

---

## Part I — Exercises

1. For a 3-request batch with token counts 100/50/200 and start_pos
   0/500/0, write the segment table and the arguments to each strided
   attention call.

2. `transpose_shd` and `transpose_hsd` — state the input and output layout of
   each, then say which one runs *before* attention and which *after*, and why.

3. Trace `k_normed_buf` through `encode_parallel_prefill_attention_inputs`.
   How many times is it written? Is any value read after being overwritten?

4. Why is `num_kv_heads` passed as 0 to `encode_rotary_batch` on shared-KV
   layers? What would go wrong otherwise?

5. Compute `row_bytes` for head_dim=256 with Q4_0 and Q8_0 KV. Now grep for
   every place that constant is recomputed.

6. Prefill uses one CB per layer, decode one per token. Give the arithmetic
   (dispatch count × overhead vs GPU time) that justifies each.

7. Estimate arithmetic intensity for `o_proj` at seq=1, 8, 64, 512. At which
   size does it cross M1 Pro's ~16 FLOP/byte ridge?

8. Run `PROFILE_ABLATE=SKIP_mlp` at 4k. Does the MLP share match the table?
   If not, what differs in your build?

---

## Checklist

- [ ] I can explain why prefill is compute-bound and decode is not, with numbers.
- [ ] I can list the layout transitions in one prefill layer, in order.
- [ ] I know why QK-norm's batch count is `seq_len * num_heads`.
- [ ] I can explain `row_start`/`total_seq_len` vs `start_pos` without hesitating.
- [ ] I know which two operations are segment-aware and why only those.
- [ ] I can tell the `TILED_EXT_MIN_Q` story and state its general lesson.
- [ ] I know why prefill uses one command buffer per layer.
- [ ] I know the top two prefill time buckets and what was already tried.

**Next:** [11_server_api.md](11_server_api.md) — how a chunk request gets
here from HTTP.
