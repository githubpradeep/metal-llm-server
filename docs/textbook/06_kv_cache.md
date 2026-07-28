# 06 — The KV Cache: Layout, Append, and the Bugs That Live There

Attention over a growing sequence would be quadratic if you recomputed keys
and values every step. The KV cache turns it linear: compute each token's K
and V once, store them, reuse forever.

That is the easy part. The hard part is that **the cache is a raw byte buffer
with an address convention that must be identical in five different places** —
the decode append kernel, the prefill batch append kernel, the fused attention
kernel, the MWG attention kernel, and the host code that computes strides. Get
one wrong and you do not get a crash. You get *fluent nonsense*, which is the
most expensive kind of bug there is.

This chapter makes the layout concrete enough to derive from memory.

Prerequisites: [00b](00b_transformers_first_principles.md) §9,
[02](02_gemma4_architecture.md).

---

## Part A — What is stored, and by whom

For each layer that owns KV (`has_kv == true`), and each position `t`:

```text
K[t]: [num_kv_heads, head_dim]   # already RoPE'd and K-norm'd at write time
V[t]: [num_kv_heads, head_dim]   # already V-norm'd
```

**"Already" is doing a lot of work in those comments.** The cache stores
*post-processed* K and V. RoPE is applied before the write, not on read. This
is a real design decision with consequences:

- **Pro:** each key is rotated once, ever. Reading 4000 cached keys costs zero
  RoPE work.
- **Con:** the rotation is baked in at the position it was written. You cannot
  shift a sequence's positions after the fact, which is why prefix-sharing
  tricks that re-position a cached prefix do not work here.

Shared-KV layers (`has_kv == false`) store nothing. They read
`layer.kv_source_layer`'s buffers (Ch 02).

Three runtime containers exist — do not conflate them:

| Container | File | Role |
|---|---|---|
| Model-owned `k_cache`/`v_cache: Vec<Buffer>` | `gemma4_gpu_model.rs` | CLI, single-session, MTP |
| `KvCachePool` slots | `kv_pool.rs` | Server continuous batching (Ch 13) |
| `StreamingKVCache` | `cache.rs` | **CPU** legacy/alternate — not the Metal hot path |

Server decode temporarily `mem::swap`s a pool slot's buffers into the model
(`forward_single_token_with_kv_slot`), runs, and swaps back. MTP instead
*aliases* with `KvCachePool::from_existing` (Ch 13 Part B.2) — refcounted
handles, no copy. Two different tricks for the same impedance mismatch; know
which one you are looking at, because a swap that is not restored leaves the
model pointing at a released slot.

---

## Part B — Row packing by KV type

```86:93:src/gemma4_config.rs
pub fn bytes_per_row(&self, head_dim: usize) -> usize {
    assert!(head_dim % 32 == 0, ...);
    match self {
        KvCacheType::F16 => head_dim * 2,
        KvCacheType::Q8_0 => (head_dim / 32) * 34,
        KvCacheType::Q4_0 => (head_dim / 32) * 18,
    }
}
```

Same block formats as weights (Ch 05): 32 values per block, 18 bytes for Q4_0
(f16 scale + 16 packed nibble bytes), 34 for Q8_0 (f16 scale + 32 int8).

Worked numbers, because you will need these constantly:

| head_dim | F16 | Q8_0 | Q4_0 |
|---|---|---|---|
| 128 | 256 B | 4·34 = 136 B | 4·18 = 72 B |
| 256 | 512 B | 8·34 = 272 B | 8·18 = 144 B |
| 512 | 1024 B | 16·34 = 544 B | 16·18 = 288 B |

Env: `LLAMA_KV_CACHE_TYPE` (default **F16**). Every performance run in
`AGENTS.md` uses `q4_0`; Ch 13 Part A.2 has the total-memory table that
explains why.

---

## Part C — The address convention, derived

Every KV kernel computes the same address. Here it is for Q4_0, taken from the
decode append kernel:

```text
groups_per_row = head_dim / 32
row_bytes      = groups_per_row * 18

byte address of group g, head h, position p:
    h * capacity * row_bytes      (skip whole heads)
  + p * row_bytes                 (skip whole positions within this head)
  + g * 18                        (skip whole blocks within this position)
```

```3471:3509:src/shaders/llama.metal
kernel void kv_cache_append_q4_0(...) {
    // gid indexes (head, group) pairs — one Q4_0 block per thread
    uint base_offset = h * capacity * row_bytes + cur_seq * row_bytes + g * 18;
    // write f16 scale, then 16 bytes of nibble-packed codes
}
```

So the layout is **head-major, then position, then block**:

```text
K buffer for head_dim=128, num_kv_heads=4, capacity=C  (Q4_0, 72 B/row):

┌─ head 0 ────────────────────────────────────────────────┐
│ pos 0 (72 B) │ pos 1 (72 B) │ … │ pos C−1 (72 B)        │
├─ head 1 ────────────────────────────────────────────────┤
│ pos 0        │ pos 1        │ … │ pos C−1               │
├─ head 2, head 3 … ──────────────────────────────────────┤
└─────────────────────────────────────────────────────────┘

Inside one position's 72 bytes:
[ scale|qs (dims 0–31) ][ dims 32–63 ][ dims 64–95 ][ dims 96–127 ]
   18 B                    18 B           18 B          18 B
```

### C.1 The single most important sentence in this chapter

> **`capacity`, not `seq_len`, is the stride between heads.**

Heads are allocated at full capacity so that appending a position never has
to move anything. A reader that strides by the *current* `seq_len` will index
into the middle of head 0's region and read another position's data as if it
were another head's. Symptom: output is fine early (when `seq_len ≈ capacity`
is false but the error is small) and degrades as the sequence grows. This is
the archetypal fluent-nonsense bug.

Derivation drill: head 2, position 100, group 3, head_dim=128, capacity=8192,
Q4_0.

```text
row_bytes = 4 * 18 = 72
addr = 2 * 8192 * 72  +  100 * 72  +  3 * 18
     = 1,179,648      +  7,200     +  54
     = 1,186,902
```

Do that by hand once. It is the fastest way to make the convention permanent.

### C.2 Quantization happens inside the append kernel

The append kernel receives f32 K/V and quantizes as it writes:

```text
scale = max_abs / 7
code  = clamp(round(v / scale) + 8, 0, 15)
```

Identical to weight Q4_0 (Ch 05 Part B), which is why the same
`q4_0_read`-style dequant code in the attention kernels works on both. Note
the cost: quantization is per-position-per-head, on the write path, every
token. It is cheap relative to what it saves in read bandwidth, but it is not
free — it is one of the open suspects in `AGENTS.md`'s unresolved hypotheses.

**Invariant:** the reader's `head_dim` and KV type must match the **writer's**.
For shared-KV layers, the writer is the anchor layer (`kv_source_layer`), so
attention on a shared layer must compute `row_bytes` from the **anchor's**
head_dim, not its own. That is `AGENTS.md` #12.

---

## Part C.3 — The append kernel, line by line

Everything in Parts B and C is enforced by one small kernel. Read it whole; it
is the cache's entire write path for Q4_0.

```3471:3510:src/shaders/llama.metal
kernel void kv_cache_append_q4_0(
    device const float* new_data [[buffer(0)]],
    device uchar* cache [[buffer(1)]],
    constant uint& num_kv_heads [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],
    constant uint& cur_seq [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint groups_per_row = head_dim / 32;
    uint total_groups = num_kv_heads * groups_per_row;
    if (gid >= total_groups) return;

    uint h = gid / groups_per_row;
    uint g = gid % groups_per_row;
    uint row_bytes = groups_per_row * 18;

    float max_abs = 0.0f;
    for (uint d = 0; d < 32; d++) {
        float val = new_data[h * head_dim + g * 32 + d];
        float a = fabs(val);
        if (a > max_abs) max_abs = a;
    }

    float scale = max_abs / 7.0f;
    if (max_abs == 0.0f) scale = 1.0f;
    float inv_scale = 1.0f / scale;

    half scale_h = half(scale);
    uint base_offset = h * capacity * row_bytes + cur_seq * row_bytes + g * 18;
    *reinterpret_cast<device half*>(&cache[base_offset]) = scale_h;

    for (uint i = 0; i < 16; i++) {
        float v_lo = new_data[h * head_dim + g * 32 + i];
        float v_hi = new_data[h * head_dim + g * 32 + i + 16];
        int q_lo = clamp(int(round(v_lo * inv_scale)) + 8, 0, 15);
        int q_hi = clamp(int(round(v_hi * inv_scale)) + 8, 0, 15);
        cache[base_offset + 2 + i] = uchar(q_lo | (q_hi << 4));
    }
}
```

Walk it:

**One thread per 32-value group.** `total_groups = num_kv_heads ×
groups_per_row` — for E4B sliding that is `4 × 4 = 16` threads for the entire
append. Sixteen threads is nothing, which tells you this dispatch is pure
latency, not throughput: ~5 µs of launch overhead to move 288 bytes. That is
precisely why fusing the append into the attention kernel (rung 1 of Ch 07
Part G) is worth it — not for the bytes, for the dispatch.

**`if (gid >= total_groups) return;`** — the mandatory guard, because the host
uses `dispatch_threads` (Ch 00c Part D.2).

**Index decomposition.** `h = gid / groups_per_row`, `g = gid % groups_per_row`.
One flat thread id becomes (head, group). This is the standard trick and it is
worth noticing that the *divisor* is a runtime value here — a small cost the
compiler cannot remove, acceptable in a 16-thread kernel.

**Quantization is computed here, not by the caller.** The thread scans its 32
values for `max_abs`, derives `scale = max_abs / 7`, and writes the f16 scale
followed by 16 packed bytes. So the K/V *scratch* buffers stay f32 and only the
cache is quantized (Ch 06 Part A). Note `if (max_abs == 0.0f) scale = 1.0f` —
without it, an all-zero group produces `scale = 0`, then `inv_scale = inf`, then
`round(0 × inf) = NaN`. All-zero groups do occur (padding, degenerate heads), so
this line is load-bearing rather than defensive.

**The address, exactly as derived in Part C:**

```text
base_offset = h * capacity * row_bytes     # head stride uses CAPACITY
            + cur_seq * row_bytes          # position within the head
            + g * 18                       # which 32-value block
```

`capacity`, not `seq_len` — the one line that Part C.1 warns about. If you pass
`seq_len` here, early tokens land in plausible-looking places and later ones
walk into the next head's region. Fluent nonsense, growing worse with context.

**Nibble packing matches Q4_0 exactly:** value `i` in the low nibble, value
`i + 16` in the high nibble, zero-point 8 (Ch 05 Part B). The attention kernel's
`q4_0_read` inverts precisely this, so any change here must be mirrored there —
they are two halves of one format contract.

**What the kernel does not do:** it does not touch `seq_len`, `total_tokens`, or
any metadata. Physical rows and logical counters are updated by different code,
which is why Part D.3 exists.

---

## Part D — Append paths

Kernels: `kv_cache_append_{f16,q8_0,q4_0}` for decode, plus
`kv_batch_append_strided_{f16,q8_0,q4_0}` for prefill.

### D.1 Decode: fused or explicit, never both, never neither

```text
if needs_explicit_kv_append(has_kv, effective_kv_seq):
    encode_kv_append_*(...)    # f32 K/V scratch → quantized cache row
# else: the fused attention kernel writes the cache itself
```

The fused attention kernel (Ch 07 Part D) attends using the current token's
**f32** K/V from registers/threadgroup memory, and packs them into the cache
as an epilogue. The decomposed path (norm/RoPE separate, ggml MWG attention)
cannot do that — MWG reads only from the cache — so the current token must be
appended *before* attention runs.

```text
Full fused:  attend with f32 K_new/V_new + cached history  →  then pack & append
Explicit:    append first (so MWG sees the current token)  →  then attend
```

Two failure modes, both real:

- **Neither appends** → attention never sees the current token. This is
  `AGENTS.md` #15: at `kv_seq ≥ 128` the `auto` router switched to MWG, but
  `fused_kv_attention_enabled()` was still true, so the explicit append was
  skipped. Result: a coherent opening, then "benefits a powerful benefits…"
  and endless `###`. Fixed by `needs_explicit_kv_append(has_kv,
  effective_kv_seq)` returning true whenever ggml is active.
- **Both append** → the position is written twice; harmless if identical,
  corrupting if the fused epilogue writes at a stale `cur_seq`.

**Rule to carry:** *any* change to attention kernel routing must reconcile KV
append semantics in the same commit. Write the invariant into a function name
(as `needs_explicit_kv_append` does) rather than leaving it implicit in two
`if`s.

### D.2 Prefill: strided batch append

Prefill writes many positions for many requests, from concatenated scratch
buffers (Ch 10 Part D):

```6678:6689:src/gemma4_gpu_model.rs
self.ctx.encode_kv_batch_append_strided_f16(
    encoder,
    &self.prefill_scratch.k_buf,
    k_cache,
    num_kv_heads as u32,
    head_dim as u32,
    kv_pool.capacity(),        // destination: stride between heads
    segment.start_pos as u32,  // destination: first position for this segment
    segment.token_count as u32,
    total_seq_len as u32,      // source: row stride across all segments
    segment.row_start as u32,  // source: this segment's first row
);
```

Two coordinate systems in one call — source rows in scratch
(`row_start`/`total_seq_len`) and destination positions in the cache
(`start_pos`). The kernel indexes `gid` over `(head, seq_in_chunk, group)` and
writes at `start_pos + s`. One dispatch per segment per tensor.

### D.3 Metadata must match physical rows

`slot.seq_len` and `slot.total_tokens` are host-side beliefs about the buffer.
Nothing enforces them; every attention dispatch simply trusts them.

- Append 5 rows but bump `seq_len` by 4 → the 5th is invisible.
- Bump by 6 → attention reads one row of stale bytes from a previous request.

MTP is where this gets exercised hardest: verify writes rows for *all* draft
positions, then `truncate_kv(rewind)` rolls the metadata back for rejected
ones (Ch 14 Part D). The stale bytes remain in the buffer, and that is fine —
correctness comes from the bound, not from clearing (Ch 13 Part B).

---

## Part E — Sliding window

For sliding layers, the host clamps how much of the cache is visible:

```text
effective_kv_seq = min(kv_seq, sliding_window)
kv_start         = kv_seq - effective_kv_seq
```

Attention reads `[kv_start, kv_seq)`. Full-attention layers read the whole
causal prefix; in the prefill API `attention_window = 0` means "no SWA clamp."

Two consequences that trip people up:

1. **The hybrid `auto` threshold uses `effective_kv_seq`, not `kv_seq`.** A
   sliding layer at `kv_seq = 4000` with a 512 window has
   `effective_kv_seq = 512`, and that is the number that decides fused vs MWG.
   Correct — the kernel's actual work is bounded by the window.
2. **Bytes are not reclaimed.** Evicted positions still occupy their rows; the
   window is a read bound, not a deallocation. So SWA saves *attention time*,
   not memory. Memory is `num_kv_heads × capacity × bytes_per_row` regardless.

---

## Part F — Shared KV: the operational checklist

When reading or writing attention code for layer `i`:

```text
src = layers[i].kv_source_layer
read  k_cache[src], v_cache[src]
row_bytes / head_dim  ← from the ANCHOR layer (src), not i
never call append when !layers[i].has_kv
still compute Q for layer i (Q is always per-layer)
```

The anchor choice must match llama.cpp's rule (`AGENTS.md` #12):
`n_layer_kv_from_start − (is_swa ? 2 : 1)` — for E4B that is anchors 22/23,
**not** a nearest-same-type scan. A plausible-looking scan gives you a valid
cache from the wrong layer: coherent output, wrong model. Only long-range
tests (needle-in-haystack) catch it.

---

## Part G — Failure modes worth memorizing

| Symptom | Likely cause |
|---|---|
| Fine then garbage past ~128 tokens with `ATTENTION_KERNEL=auto` | Missing explicit append at the routing switch (#15) |
| Wrong only on late layers | Bad `kv_source_layer`, or `row_bytes` from the wrong layer (#12) |
| Degrades gradually as context grows | Head stride uses `seq_len` instead of `capacity` |
| Wrong only on sliding layers | `kv_start` / window not applied |
| MTP accepts then corrupts | Missing `truncate_kv` after partial accept |
| Stale text from a previous conversation | Something read past `seq_len` |
| Works in CLI, breaks in server | Swap-vs-alias confusion; slot restore left dangling |

Notice the pattern: **almost every entry is "output is grammatical but
wrong."** That is what makes KV bugs expensive, and why the fastest debugging
tool here is a needle test (`ZEBRA42` planted mid-context) rather than reading
sample text for vibes.

---

## Part H — Exercises

1. Compute the byte address of head 3, position 1000, group 7 for head_dim=256,
   capacity=8192, Q4_0. Then for Q8_0. Then for F16.

2. One position of K for `num_kv_heads=4`, head_dim=128: how many bytes in
   each of the three KV types? Now for head_dim=512.

3. A reader strides heads by `seq_len` instead of `capacity`, with
   `seq_len=100`, `capacity=8192`. Which position does it read when it wants
   head 1, position 0? Explain why the error grows with context.

4. Write out both orderings (fused append vs explicit append) as numbered
   steps. For each, say what attention sees for the current token.

5. In the strided prefill append call, which arguments are source coordinates
   and which are destination? What breaks if you swap `total_seq_len` and
   `capacity`?

6. Layer 30 is a shared-KV sliding layer whose anchor is layer 22. List every
   value attention must take from layer 22 rather than layer 30.

7. SWA with window 512 at `kv_seq = 4000`: how many bytes are allocated, how
   many are read per attention dispatch, and how many are dead?

8. `truncate_kv` only changes metadata. Explain why that is sufficient, and
   name the one property of every attention dispatch that makes it safe.

---

## Checklist

- [ ] I can derive the Q4_0 cache address for any (head, position, group).
- [ ] I can explain why `capacity` is the head stride and what breaks otherwise.
- [ ] I know that K is stored post-RoPE and what that forecloses.
- [ ] I can state when append is fused vs explicit, and the #15 failure.
- [ ] I can distinguish source and destination coordinates in strided append.
- [ ] I know why metadata (`seq_len`) is the only thing `truncate_kv` touches.
- [ ] I know SWA bounds reads, not allocation.
- [ ] I can name which values a shared-KV layer must take from its anchor.

**Next:** [07_attention.md](07_attention.md) — the kernels that read this
layout.
