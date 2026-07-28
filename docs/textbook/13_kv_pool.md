# 13 — The KV Pool and BatchEngine

The KV cache is the largest allocation in the process and the resource that
decides how many concurrent requests you can serve. This chapter is about the
allocator that owns it (`kv_pool.rs`) and the thin facade the scheduler talks
to (`batch_engine.rs`).

Both files are short. Read them fully; then read this chapter for the *why*.

---

## Part A — Sizing: where all the memory goes

```95:132:src/kv_pool.rs
pub fn new(ctx: &MetalContext, config: &Gemma4TextConfig, num_slots: usize,
           max_seq_len: u32, kv_cache_type: KvCacheType) -> Self {
    let num_layers = config.num_hidden_layers;
    let num_kv_heads = config.num_key_value_heads;

    let mut slots = Vec::with_capacity(num_slots);
    for _ in 0..num_slots {
        let mut k_cache = Vec::with_capacity(num_layers);
        let mut v_cache = Vec::with_capacity(num_layers);

        for layer_idx in 0..num_layers {
            let head_dim = config.layer_head_dim(layer_idx);
            assert!(head_dim % 32 == 0,
                "head_dim must be multiple of 32 for quantized KV cache");
            let bytes_per_row = kv_cache_type.bytes_per_row(head_dim);
            let byte_len = (num_kv_heads * max_seq_len as usize * bytes_per_row) as u64;
            k_cache.push(ctx.device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared));
            v_cache.push(ctx.device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared));
        }

        slots.push(KvCacheSlot { k_cache, v_cache, seq_len: 0, total_tokens: 0, in_use: false });
    }

    let free_slots = (0..num_slots).rev().collect();
    ...
}
```

### A.1 The allocation shape

`num_slots × num_layers × 2` separate `MTLBuffer`s. For E4B at 4 slots and
42 layers that is 336 buffers. Each is
`num_kv_heads * max_seq_len * bytes_per_row` bytes.

Why one buffer per layer per tensor rather than one giant buffer with
offsets? Because kernels take a buffer + head/position arithmetic, and
per-layer buffers keep that arithmetic to two terms
(`head * capacity * bytes_per_row + pos * bytes_per_row`). A single mega
buffer would add a layer stride to every address computation in every KV
kernel, for no benefit — Metal buffer handles are cheap.

### A.2 Do the arithmetic once, in full

Note `head_dim = config.layer_head_dim(layer_idx)` — **per layer**. Full
attention layers use `global_head_dim` (512 on E4B), sliding layers use
`head_dim` (128). So the loop allocates two different sizes, and a table that
assumes one head_dim will be badly wrong (Ch 17 Part A).

E4B, 4 slots, `max_seq_len = 8192`, `num_kv_heads = 4`, with the common
5-sliding-to-1-full interleave (≈35 sliding + ≈7 full of 42 — check your
`layer_types`):

| KV type | Sliding layer (hd 128) | Full layer (hd 512) | Per slot (K+V, all layers) | × 4 slots |
|---|---|---|---|---|
| F16 | 4·8192·256 = 8.4 MB | 4·8192·1024 = 33.6 MB | ≈ 1.06 GB | **≈ 4.2 GB** |
| Q8_0 | 4·8192·136 = 4.5 MB | 4·8192·544 = 17.8 MB | ≈ 0.56 GB | **≈ 2.3 GB** |
| Q4_0 | 4·8192·72 = 2.4 MB | 4·8192·288 = 9.4 MB | ≈ 0.30 GB | **≈ 1.2 GB** |

(Per slot = `35·2·sliding + 7·2·full`.)

On a 16 GB M1 Pro with ~4 GB of weights, F16 KV at 4 slots takes another
~4.2 GB and puts you near the edge; Q4_0 KV takes ~1.2 GB and leaves room to
spare. **This table is the reason the Q4_0 KV cache exists**, and why
`AGENTS.md`'s "current best config" includes it. Quantized KV is not a
micro-optimization; it is what makes multi-slot serving fit at all.

One more thing the loop tells you: it runs `for layer_idx in 0..num_layers`
**unconditionally**, with no `has_kv` check. Shared-KV layers therefore get
buffers allocated that nothing ever writes, because attention on those layers
reads `layer.kv_source_layer`'s buffers instead (Ch 06 Part F). On E4B that is
18 of 42 layers — a meaningful chunk of the numbers above sitting idle. Not a
bug, but a straightforward saving available to anyone who wants to thread
`has_kv` into the allocator.

Trade-off: Q4_0 KV loses precision on stored keys/values. In practice quality
holds up well because attention is a weighted average — errors partially
cancel. Verify on your own workload with a needle test rather than trusting
that.

### A.3 The assertion that shapes the design

```rust
assert!(head_dim % 32 == 0, "head_dim must be multiple of 32 for quantized KV cache");
```

Q4_0 and Q8_0 pack 32 values per block. A head_dim that is not a multiple of
32 would need partial blocks, and every KV kernel would need a tail path.
Rather than support that, the pool refuses at construction. E4B (128) and
E2B (256) both satisfy it.

`StorageModeShared` means CPU and GPU see the same pages — no explicit
copies, but you must respect completion boundaries before reading on CPU
(Ch 00c Part H).

---

## Part B — Slot allocation

```145:170:src/kv_pool.rs
pub fn allocate(&mut self) -> Option<KvSlot> {
    let slot_idx = self.free_slots.pop()?;
    let slot = &mut self.slots[slot_idx];
    slot.in_use = true;
    slot.seq_len = 0;
    slot.total_tokens = 0;
    Some(KvSlot(slot_idx))
}

pub fn release(&mut self, slot: KvSlot) -> Result<(), KvPoolError> {
    let slot_idx = slot.index();
    let slot = self.slots.get_mut(slot_idx).ok_or(KvPoolError::InvalidSlot(slot_idx))?;
    if !slot.in_use { return Err(KvPoolError::SlotNotAllocated(slot_idx)); }
    slot.in_use = false;
    slot.seq_len = 0;
    slot.total_tokens = 0;
    self.free_slots.push(slot_idx);
    Ok(())
}
```

A free-list stack. `Option` return, no waiting, no growth: the pool is fixed
at startup and `None` becomes the scheduler's "KV cache pool is full" error
(Ch 12 Part B.1). Backpressure is explicit.

**Buffers are never zeroed.** Only the *metadata* is reset. A newly allocated
slot contains the previous request's key/value bytes — and that is fine,
because `seq_len = 0` means no kernel will ever read them: every attention
dispatch is bounded by `kv_seq`. Zeroing 200 MB per admission would be pure
waste.

This is a good pattern to recognize generally: **bounds-based correctness
instead of clearing.** It is fast, and it is only safe as long as every reader
honors the bound. If you ever see stale text from a previous conversation
leak into a new one, that is your prime suspect: some path read past
`seq_len`.

`release` returning `Err` on a double-release is deliberate — it turns a
would-be silent free-list corruption (the same index pushed twice, then
handed to two requests) into a visible error.

### B.1 Two counters, not one

```45:59:src/kv_pool.rs
pub struct KvCacheSlot {
    pub k_cache: Vec<Buffer>,
    pub v_cache: Vec<Buffer>,
    pub seq_len: u32,          // rows currently valid in the cache
    pub total_tokens: usize,   // absolute tokens processed
    in_use: bool,
}
```

They differ once sliding-window attention starts evicting: `seq_len` is
bounded by the window and by `capacity`, while `total_tokens` keeps counting.
`total_tokens` is what RoPE positions come from — position must keep
increasing even when the cache has wrapped or been trimmed. Using `seq_len`
for RoPE is a subtle long-context bug: text stays fluent while positional
relationships quietly go wrong.

`KvSlotView` is the read-only snapshot handed to the model per forward:

```54:59:src/kv_pool.rs
pub struct KvSlotView {
    pub slot: KvSlot,
    pub slot_index: usize,
    pub seq_len: u32,
    pub total_tokens: usize,
}
```

`Copy`, no borrow of the pool. That matters: the model needs slot metadata for
all batch rows *and* `&mut` access to append to caches. Copying the metadata
out first sidesteps the borrow conflict without `RefCell` or cloning buffers.

### B.2 Aliasing an existing cache: `from_existing`

```62:93:src/kv_pool.rs
/// Create a one-slot adapter over an existing model-owned KV cache.
///
/// Metal buffers are reference-counted handles, so cloning them aliases the
/// same storage. Used by MTP verify to run the parallel prefill kernels
/// against the live decode cache without a full copy.
pub(crate) fn from_existing(
    k_cache: &[Buffer], v_cache: &[Buffer],
    seq_len: u32, total_tokens: usize, max_seq_len: u32, kv_cache_type: KvCacheType,
) -> (Self, KvSlot) { ... }
```

This is a small piece of design worth studying. MTP verify wants to run the
**prefill** kernels — which are written against `KvCachePool` — over the
**single-sequence decode** cache that the model owns directly. Options were:

1. Copy the cache into a pool slot (hundreds of MB per verify — absurd).
2. Duplicate every prefill kernel with a non-pool signature (code bloat).
3. Wrap the existing buffers in a one-slot pool. `Buffer` is a
   reference-counted handle, so `to_vec()` clones handles, not storage.

Option 3, in ~20 lines. The lesson: when an API mismatch blocks reuse, an
adapter over shared handles is often cheaper than either copying data or
forking the API.

The hazard is aliasing: two views of the same storage. Writes through the
adapter are visible to the model, which is exactly what verify wants — and
exactly what would be a disaster if someone assumed the pool were private.
Hence `pub(crate)` and the explanatory comment.

---

## Part C — BatchEngine: a facade with three real decisions

```6:27:src/batch_engine.rs
pub struct BatchEngine { /* model + kv_pool */ }
pub struct TimedForward { pub logits: Vec<f32>, pub latency: Duration }
pub struct DecodeInput  { pub slot: KvSlot, pub token_id: usize }
pub struct PrefillInput { pub slot: KvSlot, pub token_ids: Vec<usize>, pub want_logits: bool }
```

Mostly delegation, but three behaviors matter.

### C.1 Single-input fast path in `prefill_batch`

```77:84:src/batch_engine.rs
if inputs.len() == 1 {
    let input = &inputs[0];
    return vec![self.prefill_chunk(&input.token_ids, input.slot, input.want_logits)];
}
```

Batched prefill has real setup cost: build segments, dispatch per-segment KV
appends and attention, thread `row_start`/`total_seq_len` everywhere (Ch 10
Part D). For one request none of that buys anything, so it takes the simpler
single-slot path. Single-request serving is the common case, so this fast
path is worth its four lines.

### C.2 Chunking by `max_decode_batch_size`

```125:142:src/batch_engine.rs
for chunk in inputs.chunks(self.max_decode_batch_size()) {
    let started_at = Instant::now();
    let model_inputs: Vec<(KvSlot, usize)> = chunk.iter()
        .map(|input| (input.slot, input.token_id)).collect();
    let outputs = self.model.forward_decode_batch_with_kv_slots(&model_inputs, &mut self.kv_pool);
    let per_item_latency = started_at.elapsed() / chunk.len() as u32;
    results.extend(...);
}
```

Decode scratch buffers are sized for `max_batch_size` at load. Ten decoding
requests with a max batch of 4 become three GPU calls, transparently. Output
order is preserved across chunks — the scheduler zips results back by
position (Ch 12 Part C.2), so this contract is load-bearing.

### C.3 Latency attribution is an approximation

```134:134:src/batch_engine.rs
let per_item_latency = started_at.elapsed() / chunk.len() as u32;
```

A batched forward has one wall-clock time. Dividing it evenly is a
convention, not a measurement — no per-row time exists to report, since all
rows share the same weight reads. Read metrics accordingly: **per-request
decode latency in a batched run is amortized, not observed.** For real
kernel-level numbers use `PROFILE_PHASES` / `PROFILE_DISPATCHES`, not
`TimedForward`.

---

## Part D — What "batch decode" means here

Not "one kernel handles N tokens of one sequence." It means **N independent
sequences advance one token each in one GPU forward**:

```text
row 0: slot 2, token 4711, kv_seq 138     (SWA layers window-limited)
row 1: slot 0, token  103, kv_seq 4096
row 2: slot 3, token 9902, kv_seq  27
```

Each row has its own slot buffers, its own `kv_seq`, its own RoPE position.
What they share is the **weights** — the reason batching pays at all. Weight
bytes are the decode bottleneck (Ch 05 Part A), and they are read once per
forward regardless of batch size.

Consequences worth holding onto:

- Batch-2 decode is much cheaper than 2× batch-1, but **not** free: activation
  work, KV reads, and attention all scale with rows. The MTP verify numbers in
  `AGENTS.md` quantify it — seq=3 verify costs ~1.6× a single decode, where a
  pure weight-bandwidth model predicts ~1.1×. That gap (occupancy, three
  weight streams in the MLP, per-row KV traffic) is still open.
- Rows can take **different attention paths**: with `ATTENTION_KERNEL=auto`,
  a row at `kv_seq = 27` uses fused flash while a row at `kv_seq = 4096` uses
  ggml MWG. Per-row routing means per-row KV-append semantics too — which is
  exactly the class of bug in `AGENTS.md` #15 (Ch 07 Part F). When you touch
  batched decode, re-check `needs_explicit_kv_append` for every row.

---

---

## Part E — Capacity planning, and what this design refuses to do

### E.1 The one equation that decides your deployment

```text
KV bytes = slots × Σ_layers[owning] ( 2 × nkv(l) × capacity × bytes_per_row(hd(l)) )
```

Everything else about serving capacity follows from it, because weights are
fixed and activations are negligible. Rearranged for the question you actually
have:

```text
slots ≤ (VRAM_budget − weights) / KV_bytes_per_slot
```

E4B on a 16 GB machine, leaving ~2 GB for the OS and framebuffer:

| KV type | ctx 4096 | ctx 8192 | ctx 16384 | slots at 11.5 GB free (16k ctx) |
|---|---|---|---|---|
| F16 | ~0.27 GB | ~0.53 GB | ~1.06 GB | 10 |
| Q8_0 | ~0.14 GB | ~0.28 GB | ~0.56 GB | 20 |
| Q4_0 | ~0.08 GB | ~0.15 GB | ~0.30 GB | 38 |

(Weights ~2.5 GB in Q4_0; the free figure is `16 − 2 − 2.5`.)

Two lessons. First, `LLAMA_KV_CACHE_TYPE=q4_0` is a **concurrency** decision as
much as a speed one — it roughly triples the slots you can afford. Second,
`LLAMA_CTX_SIZE` is charged per slot whether or not requests use it, because
`capacity` is fixed at allocation (Part A.1). Setting 200 000 context "just in
case" costs you every slot.

### E.2 What happens at the boundary

Allocation returns `Option`, and the scheduler surfaces the `None` as an error
rather than blocking. So the pool never overcommits, and there is no swapping,
no eviction, and no preemption. A request that gets a slot keeps it until it
finishes.

That is a deliberate refusal, and it is worth understanding what it buys and
costs:

| | |
|---|---|
| Buys | no page tables, no copy-out, no rescheduling logic, no fragmentation, latency of an admitted request is predictable |
| Costs | one long request can hold a slot for minutes; no priority; no way to admit a burst by shrinking existing contexts |

The industrial alternative is paged KV (vLLM-style): allocate fixed-size blocks
on demand, so a request only holds memory proportional to its *actual* length,
and preempt by copying blocks out. That converts the `capacity`-per-slot waste
in E.1 into near-perfect utilization, at the cost of an indirection table in
every attention kernel and a block manager in the scheduler.

This engine took the other branch: contiguous per-slot buffers, so attention
address math is `h × capacity × row_bytes + pos × row_bytes` (Ch 06 Part C) with
no indirection. Given that the kernels are hand-written per head_dim and per KV
type, that simplicity is worth real money — but it does mean the memory table
above is the honest ceiling, not a soft target.

### E.3 The failure signature of a slot leak

Slots are returned by `release_slot` in the scheduler's reap path (Ch 12
Part C.3). If any exit route misses it — an early `return` on an error, a
cancellation branch, a panic caught upstream — the slot is gone until restart.

Symptom sequence: throughput normal, then after N requests everything 503s or
queues forever while the GPU sits idle. Diagnostic: count admitted minus
completed requests, or expose `free_slots` in `/metrics`. Structural fix: make
release a `Drop` impl on the handle rather than an explicit call, so no exit
path can skip it.

## Part F — Exercises

1. Compute total KV bytes for E2B at 8 slots and 8192 context for F16 / Q8_0 /
   Q4_0. Use the real per-layer head_dims (sliding vs full) and your
   checkpoint's `layer_types`. Which configurations fit in 16 GB alongside
   ~3 GB of weights?

2. Why is not zeroing buffers on `allocate` safe? Name the invariant, and one
   concrete code change that would break it.

3. Construct a scenario where `seq_len != total_tokens`. Which one feeds RoPE?
   What is the symptom of using the other?

4. Why does `KvSlotView` exist instead of passing `&KvCacheSlot`? Write the
   borrow-checker error you would get without it.

5. `release` errors on double-release. Describe the corruption it prevents,
   step by step, in terms of the free list.

6. Ten decoding requests, `max_decode_batch_size = 4`. How many
   `forward_decode_batch_with_kv_slots` calls? What does each reported
   `latency` mean?

7. `from_existing` aliases buffers. List every invariant a caller must respect
   for that to be safe.

---

## Checklist

- [ ] I can compute KV memory for a config/KV-type/slot-count from scratch.
- [ ] I know why quantized KV is a serving requirement, not a nicety.
- [ ] I can explain why buffers are not zeroed and what makes that safe.
- [ ] I know the difference between `seq_len` and `total_tokens` and which
      one RoPE uses.
- [ ] I can explain what `from_existing` solves and what it risks.
- [ ] I know what "batch decode" means here, and why it is not linear speedup.
- [ ] I know that per-item latency in a batch is amortized, not measured.

**Next:** [14_mtp.md](14_mtp.md) — speculative decoding, which leans on both
the aliasing trick and the batched forward.

7. Your machine has 24 GB and you want 8 concurrent requests at 32 768 context
   on E4B. Which KV types can do it? Show the arithmetic.
8. A slot is allocated at `capacity = 16384` but the request only ever reaches
   900 tokens. How many bytes were wasted, in F16 and Q4_0? What would paged KV
   have saved?
9. Sketch the smallest change that would give this pool priority admission
   (two classes, high and low). What invariant from Part B does it threaten?
