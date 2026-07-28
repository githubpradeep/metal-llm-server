# 07 — Attention Kernels: A Full Course

This chapter is not a map of file names. By the end you should be able to sit
with `llama.metal` open at `flash_decode_full_fused_q4_0_hd_body` and explain
**every line**: what each thread is doing, what memory it touches, why the
barriers are there, and what breaks if you reorder the append.

Prerequisites: [00b](00b_transformers_first_principles.md) (attention math,
online softmax numeric proof, RoPE), [00c](00c_gpu_and_metal_fundamentals.md)
(TG / simdgroup / barriers), [06](06_kv_cache.md) (Q4_0 row address formula).

Open these while reading:

```text
src/shaders/llama.metal          # ~4571–6550  flash helpers + full fused body
src/gpu.rs                       # encode_attention_full_fused_q4_0 ~5819
src/decode_fused.rs              # encode_fused_attn_layer ~215
src/shaders/ggml_flash_attn.metal  # MWG path
```

---

## Part A — What decode attention is, slowly

### A.1 The math again, with shapes for one head

You have already computed, for the current token, a query vector
`Q_h ∈ R^{d}` (and, if this layer owns KV, `K_h, V_h ∈ R^{d}`). The cache
already holds past keys/values for this KV head. Let:

```text
d        = head_dim          # 128 sliding, 512 full on E4B
H_q      = num_attention_heads   # 20
H_kv     = num_key_value_heads   # 4
G        = H_q / H_kv            # 5  (GQA groups)
kv_h     = h / G                 # which KV head query head h reads
S_eff    = effective KV length after sliding-window clamp
kv_start = start index into the cache for this attend
cur_seq  = position where the NEW token's K/V will live
scale    = 1.0 for Gemma4 (QK-norm already scaled magnitudes)
```

Attention for query head `h`:

```text
for t in [kv_start, kv_start + S_eff):
    score[t] = scale * ⟨ Q_h , K_{kv_h}[t] ⟩
α = softmax(score)          # over those S_eff positions only
out_h = Σ_t α[t] * V_{kv_h}[t]
```

After all heads: concatenate `out_0 … out_{H_q-1}` and multiply by `W_O`.

That is the entire contract. Everything below is **how to evaluate this on
an Apple GPU without materializing `score[]` in DRAM and without rereading
weights we do not have** (K/V live in a Q4_0 cache).

### A.2 Why you cannot just write a nested for-loop on the GPU

Naive CPU code:

```text
for t in range(S):
    scores[t] = dot(Q, K[t])
α = softmax(scores)
out = zeros(d)
for t in range(S):
    out += α[t] * V[t]
```

Problems on GPU decode:

1. **`scores` is length S.** At S=4096 that is 16 KB per head just for
   scores, times many heads, and you still need V. FlashAttention exists
   so you never store the full score vector.
2. **K/V are quantized.** Every `dot(Q, K[t])` must dequant on the fly.
3. **The new token's K/V are still f32** in scratch when a fused kernel
   runs — they are not in the Q4 cache yet. The kernel must attend using
   f32 for `t == cur_seq` and Q4 for `t < cur_seq`, then append.
4. **GQA.** Twenty query heads share four KV heads. Dispatch geometry
   must not invent a fifth KV head or always read head 0.

### A.3 Online softmax — the algebra this kernel implements

From 00b you know: keep `(m, ℓ)` and an accumulator, rescale when `m`
grows. This engine's twist (read carefully): after every tile it stores a
**fully normalized** running output, not an unnormalized accumulator.

Define after processing some prefix of scores:

```text
m  = max score so far
ℓ  = Σ exp(score − m) over positions so far
out = (1/ℓ) * Σ exp(score − m) * V     # already normalized
```

New tile with scores `s_0…s_{T−1}` and values `V_0…V_{T−1}`:

```text
m'     = max(m, max s_i)
ℓ'     = ℓ * exp(m − m') + Σ_i exp(s_i − m')
old_f  = (ℓ * exp(m − m')) / ℓ'          # how much old out should shrink
inv_ℓ' = 1 / ℓ'
out'   = out * old_f + inv_ℓ' * Σ_i exp(s_i − m') * V_i
```

Convince yourself: if the new tile does not raise the max (`m'=m`), then
`old_f = ℓ/ℓ'` and the new contribution is weighted by `1/ℓ'`, which is
exactly regenerating the combined softmax. If the new tile raises the max,
`exp(m−m')` shrinks the old mass correctly.

**This is exactly what `flash_softmax_tile` + `flash_accum_v_*` do.**

---

## Part B — Host side: how a dispatch is launched

Before any Metal runs, Rust decides *which* kernel and binds buffers.

### B.1 `encode_attention_full_fused_q4_0` (`gpu.rs`)

```5819:5871:src/gpu.rs
pub fn encode_attention_full_fused_q4_0(...) {
    encoder.set_compute_pipeline_state(
        self.attention_full_fused_q4_0_pipeline_for(head_dim));
    encoder.set_buffer(0, Some(q_raw_buf), 0);            // Q before Q-norm
    encoder.set_buffer(1, Some(&q_norm_weight.buffer), …); // per-head weights
    encoder.set_buffer(2, Some(cos_buf), cos_offset);      // RoPE cos (this layer)
    encoder.set_buffer(3, Some(sin_buf), sin_offset);
    encoder.set_buffer(4, Some(k_raw_buf), 0);             // K before K-norm
    encoder.set_buffer(5, Some(&k_norm_weight.buffer), …);
    encoder.set_buffer(6, Some(v_raw_buf), 0);             // V before V-norm
    encoder.set_buffer(7, Some(out_buf), 0);               // attn output
    encoder.set_buffer(8, Some(k_cache_buf), 0);           // Q4_0 K cache
    encoder.set_buffer(9, Some(v_cache_buf), 0);           // Q4_0 V cache
    // bytes 10..21: num_heads, num_kv_heads, num_kv_groups, head_dim,
    //               kv_seq, capacity, scale, kv_start, groups_per_row,
    //               row_bytes, cur_seq, eps
    encoder.dispatch_thread_groups(
        MTLSize::new(num_heads as u64, 1, 1),  // ONE threadgroup per Q head
        tg_size);                               // typically 256 threads
}
```

**Memorize the grid:** `threadgroups = num_heads`. Threadgroup `tgid`
**is** query head index `h`. Inside that TG, 256 threads cooperate on that
one head's attention.

`pipeline_for(head_dim)` picks `…_h128` / `…_h256` / `…_h512` — three
separately compiled specializations so `HEAD_DIM` is a compile-time
constant inside the shader (better codegen, fixed-size threadgroup arrays).

### B.2 Where this is called from fused decode

In `decode_fused.rs` `encode_fused_attn_layer`, after QKV matvecs have
filled `scratch.q/k/v`:

```text
if has_kv && head_dim in {128,256,512}
   && !attention_use_ggml_for_layer_kv(...):
    encode_attention_full_fused_q4_0(
        scratch.q, q_norm, cos[rope_off], sin[rope_off],
        scratch.k, k_norm, scratch.v, scratch.attn_out,
        k_cache[kv_source_layer], v_cache[kv_source_layer],  // NOTE: source layer
        …)
```

Two load-bearing details:

1. Caches are `kv_source_layer`, not `layer_idx` — shared-KV layers read
   the anchor.
2. If ggml routing is active, this block is **skipped** and a different
   encode runs — and then append semantics change (Part F).

---

## Part C — The kernel entry and threadgroup memory

Look at the h256 entry point:

```6505:6548:src/shaders/llama.metal
kernel void attention_flash_decode_full_fused_q4_0_h256(
    device const float* Q_raw [[buffer(0)]],
    ...
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float shared_q[256];
    threadgroup float shared_k[256];
    threadgroup float shared_v[256];
    threadgroup float shared_scores[256];
    threadgroup float shared_exp[256];
    threadgroup float shared_update[4];
    flash_decode_full_fused_q4_0_hd_body<256, 256>(..., tgid, ...);
}
```

| Array | Size | Role |
|-------|------|------|
| `shared_q` | HEAD_DIM | Q after Q-norm + RoPE (lives in fast TG memory for the whole kernel) |
| `shared_k` | HEAD_DIM | new token's K after K-norm + RoPE (f32) |
| `shared_v` | HEAD_DIM | new token's V after V-norm (f32) |
| `shared_scores` | up to TILE | raw Q·K scores for current tile (also reused as reduction scratch during norm!) |
| `shared_exp` | up to TILE | `exp(score − m')` for current tile |
| `shared_update` | 4 | `(m, ℓ, old_factor, inv_ℓ)` |

For h256, `TILE_KV` template arg is **256** in the full-fused body
instantiation (`<256, 256>`). That means one tile can cover up to 256 KV
positions — at short context the whole cache fits in one tile. (Other flash
kernels use `FLASH_TILE_KV=32`; full-fused specializes differently per
head dim. Read the template args at the call site, do not assume 32
everywhere.)

Thread indices:

| Name | Range | Meaning |
|------|-------|---------|
| `tid` | 0..255 | thread in TG |
| `sgid` | 0..7 | which simdgroup (256/32) |
| `lane` | 0..31 | lane within simdgroup |
| `tgid` | 0..H_q−1 | query head `h` |

---

## Part D — Line-by-line: `flash_decode_full_fused_q4_0_hd_body`

This is the heart of the chapter. Body starts ~6403 in `llama.metal`.

### D.1 Setup and GQA map

```metal
if (h >= num_heads) return;

uint kv_h = h / num_kv_groups;          // GQA: query head → KV head
uint q_offset = h * HEAD_DIM;           // where this head's Q lives in Q_raw / output
uint kv_offset = kv_h * HEAD_DIM;       // where this KV head's K/V live in scratch
uint k_head_base = kv_h * capacity * row_bytes;  // byte offset of this KV head in cache
uint v_head_base = kv_h * capacity * row_bytes;
uint num_simds = FLASH_TG_SIZE / SIMD_SIZE;      // 8
```

**Worked GQA:** `h=7`, `G=5` → `kv_h = 7/5 = 1`. Heads 5,6,7,8,9 all read
KV head 1. If you wrote `kv_h = h % num_kv_heads` you would be wrong
(that is a different grouping convention).

`k_head_base` uses **capacity**, not `kv_seq` — same invariant as Ch 06.

### D.2 Initialize online-softmax state and zero the output

```metal
if (tid == 0) {
    shared_update[0] = -INFINITY;  // m
    shared_update[1] = 0.0f;       // ℓ
}
flash_zero_output_hd<HEAD_DIM>(output, q_offset, tid, FLASH_TG_SIZE);
threadgroup_barrier(...);
```

Only thread 0 writes the stats. Everyone cooperates zeroing
`output[q_offset .. q_offset+HEAD_DIM)`. Barrier: later reads of
`shared_update` / `output` must see these writes.

### D.3 Prepare Q: RMSNorm + RoPE into `shared_q`

```metal
flash_load_q_qknorm_rope_hd<HEAD_DIM>(
    Q_raw, q_norm_weight, cos_buf, sin_buf, eps, q_offset,
    shared_q, shared_scores, tid, FLASH_TG_SIZE);
```

Expand that helper (you should be able to rewrite it from memory):

**Step 1 — energy.** Each thread sums `Q_raw[q_offset+i]²` for
`i = tid, tid+tg_size, …`. Writes `shared_tmp[tid] = partial`
(here `shared_scores` is borrowed as scratch — that is why scores must
be filled later, after this phase).

**Step 2 — tree reduce.** Classic parallel reduction:

```text
for stride = tg_size/2; stride > 0; stride /= 2:
    if tid < stride: shared_tmp[tid] += shared_tmp[tid+stride]
    barrier
```

After this, `shared_tmp[0] = Σ Q²`.

**Step 3 — inv_rms.**

```text
inv_rms = rsqrt(shared_tmp[0] / HEAD_DIM + eps)
```

**Step 4 — scale by norm weights.**

```text
shared_q[d] = Q_raw[q_offset+d] * inv_rms * q_norm_weight[d]
```

Note: `q_norm_weight` is indexed by `d` within the head — the host binds
the per-head weight slice (or the kernel assumes contiguous per-head
layout matching how weights were uploaded). Wrong offset ⇒ wrong norm.

**Step 5 — Neox RoPE.**

```text
half = HEAD_DIM/2
for d in 0..half:
    q1, q2 = shared_q[d], shared_q[d+half]
    shared_q[d]      = q1*cos[d] - q2*sin[d]
    shared_q[d+half] = q2*cos[d] + q1*sin[d]
```

`cos_buf`/`sin_buf` already contain the partial-rotary identity (cos=1,
sin=0) for non-rotated dims — filled by `rope_fill_decode`.

Barrier after the whole helper returns before anyone uses `shared_q` as
the finished query.

### D.4 Prepare K and V of the *current* token into TG memory

```metal
flash_prepare_k_norm_rope_hd<HEAD_DIM>(K_raw, k_norm_weight, cos, sin, eps,
    kv_offset, shared_k, shared_scores, tid, FLASH_TG_SIZE);
flash_prepare_v_norm_hd<HEAD_DIM>(V_raw, eps, kv_offset,
    shared_v, shared_scores, tid, FLASH_TG_SIZE);
```

K path = same as Q (norm with weights + RoPE).
V path = RMSNorm **without** weights (`shared_v[d] = V_raw[d] * inv_rms`).

After this, the TG holds the current token's K/V in f32 in fast memory.
They are **not yet** in the Q4 cache.

### D.5 The tile loop — score

```metal
for (uint kv_tile = 0; kv_tile < kv_seq; kv_tile += TILE_KV) {
    uint tile_count = min(TILE_KV, kv_seq - kv_tile);

    for (uint wave = 0; wave < tile_count; wave += num_simds) {
        uint kv_pos = wave + sgid;          // each simdgroup owns one position
        if (kv_pos < tile_count) {
            uint actual_pos = kv_start + kv_tile + kv_pos;
            float partial = flash_dot_k_shared_hd<HEAD_DIM>(
                K_cache, k_head_base, actual_pos, row_bytes,
                shared_k, cur_seq, shared_q, lane);
            partial = simd_sum(partial);
            if (lane == 0) {
                shared_scores[kv_pos] = partial * scale;
            }
        }
    }
    threadgroup_barrier(...);
```

**Parallelism inside a tile:**

- 8 simdgroups (`num_simds`)
- Each simdgroup scores **one** KV position per wave
- Within a simdgroup, 32 lanes split the `HEAD_DIM` dimensions of the dot
  product, then `simd_sum` combines them
- Lane 0 of that simdgroup writes `shared_scores[kv_pos]`

If `tile_count=100` and `num_simds=8`, you need `ceil(100/8)=13` waves.

### D.6 The critical branch inside the dot product

```5143:5166:src/shaders/llama.metal
inline float flash_dot_k_shared_hd(...) {
    if (pos == cur_seq) {
        // dot(shared_q, shared_k) in f32 — current token
        ...
        return partial;
    }
    return flash_dot_q4_k_hd<HEAD_DIM>(...);  // dequant from Q4 cache
}
```

**This is the fused-append design in one `if`:**

- Past positions: read Q4_0 from `K_cache` via `q4_0_read4`
- Current position (`pos == cur_seq`): read f32 from `shared_k`

If you appended *before* attention and the ggml path forgot to append,
then `pos == cur_seq` is not in the cache and this fused kernel was not
running — you get the hybrid bug (Part F).

### D.7 How Q4_0 dequant works in the score

```3708:3719:src/shaders/llama.metal
inline float q4_0_read(...) {
    uint g = d / 32;
    uint e = d % 32;
    uint offset = head_base + pos * row_bytes + g * 18;
    float scale = float(*reinterpret_cast<device const half*>(&cache[offset]));
    ...
    return float(int(nibble) - 8) * scale;
}
```

Same formula as weight Q4_0 and as `kv_cache_append_q4_0`. Lanes call
`q4_0_read4` to pull 4 dims at a time and `dot` with a `float4` of Q.

### D.8 Softmax update for the tile (single-threaded in this design)

```metal
if (tid == 0) {
    flash_softmax_tile(shared_scores, shared_exp, shared_update, tile_count);
}
barrier;
```

Only thread 0 runs the softmax update (tile_count is small relative to
head work). It fills `shared_exp[i]` and updates `shared_update[0..3]`
as in Part A.3.

### D.9 Accumulate V into the running normalized output

```metal
flash_accum_v_q4_shared_fused_hd<HEAD_DIM, TILE_KV>(
    output, q_offset, V_cache, v_head_base,
    kv_start, kv_tile, tile_count, row_bytes,
    shared_v, cur_seq,
    shared_exp, shared_update[2], shared_update[3],
    tid, FLASH_TG_SIZE);
```

For each dimension `d` owned by `tid` (strided by `tg_size`, vectorized
by 4):

```text
acc = Σ_i shared_exp[i] * V[actual_pos_i][d]
      where V comes from shared_v if pos==cur_seq else q4_0_read4(cache)
output[d] = output[d] * old_factor + acc * inv_ℓ
```

After the last tile, `output[q_offset…]` is the final attention result for
head `h`.

### D.10 Append K/V into the Q4 cache — once per KV head

```metal
if ((h % num_kv_groups) == 0) {
    for (uint g = tid; g < groups_per_row; g += FLASH_TG_SIZE) {
        q4_0_append_group_tg(shared_k, 0, K_cache, k_head_base, cur_seq, row_bytes, g);
        q4_0_append_group_tg(shared_v, 0, V_cache, v_head_base, cur_seq, row_bytes, g);
    }
}
```

**Why `h % num_kv_groups == 0`?** Five query heads share one KV head. All
five TGs computed the same `shared_k`/`shared_v` for that KV head (from
the same `K_raw`/`V_raw`). Only one of them must write the cache, or you
race. Picking the first query head in the group (`h % G == 0`) is the
arbitration.

`q4_0_append_group_tg` quantizes 32 f32 values into one 18-byte block at
`head_base + cur_seq * row_bytes + g*18` — identical packing to the
standalone append kernel.

**Order is attend-then-append**, using f32 for the current token during
attend. That is why fused paths must **not** also call
`encode_kv_append` (double-write is mostly harmless, but the hybrid bug
was the opposite: ggml path needs explicit append and did not get it).

---

## Part E — Walk a concrete numeric micro-example through the kernel

Toy settings so you can simulate on paper:

```text
HEAD_DIM = 4          # real code uses 128/256/512; math is the same
H_q = 2, H_kv = 1, G = 2
kv_seq = 2, kv_start = 0, cur_seq = 1, scale = 1
TILE covers both positions
```

Cache already has pos 0 as Q4. Current token K/V are in scratch.

**TG 0** (`h=0`, `kv_h=0`):

1. Load Q_0, norm+rope → `shared_q`
2. Load K_0, V_0 norm(+rope for K) → `shared_k`, `shared_v`
3. Score pos 0: dequant K_cache[0], dot with shared_q → s0
4. Score pos 1: dot(shared_q, shared_k) → s1
5. Softmax online over [s0,s1] → α0, α1
6. out = α0*V_cache[0] + α1*shared_v
7. Since `0 % 2 == 0`, append shared_k/v to cache at cur_seq=1

**TG 1** (`h=1`, `kv_h=0`):

Same KV head. Steps 1–6 with Q_1. Step 7 **skipped** (`1 % 2 != 0`).

If step 7 ran in both TGs without synchronization, you would corrupt the
cache. The modulo check is not a style choice; it is correctness.

---

## Part F — The other attention kernel: ggml multi-workgroup (MWG)

### F.1 Why a second kernel exists at all

The fused kernel of Part D dispatches **one threadgroup per query head** — at
E4B, 20 threadgroups of 256 threads. That is fine when the work per head is
small, but the work per head is `O(kv_seq)` while the number of threadgroups
never grows. At kv_seq 25 the fixed costs dominate and fused wins; at kv_seq
2000 those same 20 threadgroups each grind serially through 2000 keys.

llama.cpp's answer is to parallelize along the KV axis: split the cache across
`NWG` workgroups, each running an independent online softmax over its slice,
then merge the partial results. This repo ports that path into
`src/shaders/ggml_flash_attn.metal`, selected by `ATTENTION_KERNEL=ggml` (or
`auto` above a threshold).

### F.2 The partitioning, from the source

Each workgroup takes a strided subset of the KV tiles:

```125:125:src/shaders/ggml_flash_attn.metal
 for (int ic0 = iwg*NSG + sgitg; ; ic0 += NWG*NSG) {
```

Workgroup `iwg` starts at tile `iwg*NSG + sgitg` and strides by `NWG*NSG`, so
the `NWG` workgroups interleave tiles rather than taking contiguous ranges.
Interleaving keeps their finish times close, which matters because the reduce
step waits for all of them.

Each workgroup then writes a **partial** result plus its softmax state:

```316:327:src/shaders/ggml_flash_attn.metal
 device float  * dst1 = (device float  *) dst + nrows*DV*NWG;

 const float S = NWG == 1 ? (ss[0] == 0.0f ? 0.0f : 1.0f/ss[0]) : 1.0f;

 ...
 dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) so4[i]*S;

 ...
 if (NWG > 1) {
 ...
 dst1[rid*(2*NWG) + 2*iwg + 0] = ss[0];
 dst1[rid*(2*NWG) + 2*iwg + 1] = ss[1];
```

Read the two branches on `NWG`:

- **`NWG == 1`:** this workgroup saw all the keys, so it normalizes on the spot
  (`S = 1/ℓ`) and writes the final answer. No reduce needed.
- **`NWG > 1`:** it writes the *unnormalized* accumulator (`S = 1`) plus its
  own `(ℓ, m)` into a separate region of the temp buffer (`dst1`, offset
  `nrows*DV*NWG` floats past the outputs). Normalizing early would be wrong —
  each partial has a different `m`, and only the reducer knows the global one.

That offset arithmetic is where `AGENTS.md` #9 went wrong: using `DV4`
(head_dim/4) instead of `DV` (head_dim) for the state region made `(ℓ, m)`
overlap the output data. Nothing crashes when two regions of one buffer
overlap; you just get garbage that looks like a model bug.

### F.3 The reduce kernel

```352:367:src/shaders/ggml_flash_attn.metal
    device const float * ss = (device const float *) htmp + (uint64_t)args.nrows*DV*NWG;

    ...
    float S = ss[rid*(2*NWG) + 2*iwg + 0];
    float M = ss[rid*(2*NWG) + 2*iwg + 1];

    ...
    device const float4 * htmp4 = (device const float4 *) htmp + rid*DV4*NWG;

    for (short i = sgitg; i < DV4; i += NWG) {
        const float4 v = simd_sum(htmp4[i*NWG + iwg]*ms);
```

This is the online-softmax merge of Ch 00b Part 4.5, applied across
workgroups instead of across tiles: take the max of the `NWG` maxima, rescale
each partial by `exp(m_i − m_global)`, sum the rescaled `ℓ` and accumulators,
divide once. Same algebra, different axis.

It has to be a **separate dispatch** because threadgroups cannot synchronize
(Ch 00c Part G). And note `AGENTS.md` #10: the reduce implementation was first
written as a `kernel void` template called from the entry points — illegal in
Metal, since a kernel function cannot call another kernel function. Changing it
to plain `void` fixed it. Two of the eleven numbered MWG problems in the log are
this kind of plumbing, which is what porting a kernel actually costs.

### F.4 When MWG wins and when it loses

| Situation | Winner | Reason |
|---|---|---|
| kv_seq 25 | fused | MWG's 32 dispatches + reduce dominate the tiny work |
| kv_seq 441, NWG=32 | fused | `ceil(441/1024) = 1` iteration per WG — pure overhead (`AGENTS.md` #8: 40.4 tok/s) |
| kv_seq 200–500, tuned | roughly even | ggml is flat ~49 tok/s at any context (`AGENTS.md` #13) |
| kv_seq ≫ NWG × C | MWG | actual KV parallelism to exploit |

The key arithmetic from `AGENTS.md` #8: with `NWG=32` and `C=32` keys per
iteration, one pass across all workgroups covers `NWG × C = 1024` tokens. Below
that, most workgroups do a single iteration or none, and you have paid 32
dispatches plus a reduce for one tile's worth of work. **MWG only pays off when
`kv_seq ≫ 1024`.**

That is the whole justification for the hybrid: fused below the threshold, MWG
above (`AGENTS.md` #14, ~53.5 tok/s at 25 tokens and ~50.0 at 200, versus
54.7/47.9 fused-only and 48.9/49.1 ggml-only).

### F.5 MWG does not append KV — and the bug that followed

**MWG only reads the cache.** It has no append phase, because a workgroup that
sees a slice of the KV axis has no business writing the current token's row.

So the host must append explicitly before dispatching it. Here is the state
that produced `AGENTS.md` #15:

```text
fused_kv_attention_enabled() == true     # global policy, still true in auto mode
attention_use_ggml_for_layer_kv(...) == true   # because kv_seq >= 128

Host reasoned: "fused KV append is enabled → the attention kernel appends"
Reality:        the ggml kernel was selected, and it never appends
Result:         attention over a cache missing the current token
Symptom:        coherent output for ~128 tokens, then "benefits a powerful
                benefits…" and endless ### blocks
```

Note *why* the symptom appears where it does: below 128 the fused kernel runs
and appends, so the beginning of the essay is correct. The failure switches on
exactly when the routing switches.

The fix derives the append decision from the **same** predicate as kernel
selection:

```243:251:src/gpu.rs
pub fn needs_explicit_kv_append(has_kv: bool, effective_kv_seq: u32) -> bool {
    if !has_kv {
        return false;
    }
    if attention_use_ggml_for_layer_kv(has_kv, effective_kv_seq) {
        return true;
    }
    !fused_kv_attention_enabled()
}
```

**The invariant, stated generally:**

> Kernel choice and KV-append side effects are one decision, not two. Any new
> attention kernel must be added to `needs_explicit_kv_append` in the same
> commit.

---

## Part G — The fusion ladder, concretely

Now that you have read one fused kernel end to end, the ladder is legible. Each
rung moves more work inside the Metal kernel and removes a dispatch plus a
round trip through device memory.

| Rung | Encode helper | Inside the kernel | KV append |
|---|---|---|---|
| 1 | `encode_attention_full_fused_q4_0` | Q-norm+RoPE, K-norm+RoPE, V-norm, flash attention, Q4_0 append | **inside** |
| 2 | `attention_flash_decode_qknorm_rope_q4_0` | Q-norm+RoPE + flash; K/V prepared outside | depends on policy |
| 3 | `attention_flash_decode_q4_0` | flash only; norms and RoPE are separate dispatches | explicit |
| 4 | `encode_attention_ggml_q4_0` (MWG) | vec attention + separate reduce | **always explicit** |
| 5 | `encode_attention_with_offset_f16/q8_0` | non-flash paths for other KV types | explicit |

The gate for rung 1–2 (from the decode path, `gemma4_gpu_model.rs` ~3736):

```text
use_fused_q_attn = weights quantized
                && kv_cache_type == Q4_0
                && ctx.use_flash_attention
                && fused_q_attn_enabled()
                && !attention_use_ggml_for_layer_kv(has_kv, kv_seq + 1)
                && head_dim ∈ {128, 256, 512}

use_fused_k_attn = use_fused_q_attn && has_kv && fused_k_attn_enabled()
```

Five conditions, and every one is a fact about the *model or config*, not about
the request: format, KV type, flash availability, routing, head_dim. So the rung
is effectively fixed at load time per layer — which is what makes it safe to
reason about statically.

**Why the ladder is a numerical contract, not just a speed knob.**
`AGENTS.md` #11's follow-up is the cautionary tale. Someone built a
"decomposed but equivalent" path for GQA — separate `rmsnorm`, separate
`apply_rotary`, then a GQA attention kernel — and it produced garbage at a
*higher* tok/s. The individual pieces were each correct. What differed was
where values were rounded and in what order they were combined; the fused
kernel keeps Q in threadgroup memory in a particular precision and applies norm
and RoPE in a particular order, and the decomposed chain did not reproduce that
bit-for-bit.

The resolution was not to debug the decomposition but to write a *fused* GQA
kernel — `attention_flash_decode_qknorm_rope_q4_0_gqa_{h128,h256,h512}` — that
keeps the same `flash_load_q_qknorm_rope_hd` prologue and only changes the
threadgroup-to-head mapping. Change one axis at a time, even inside a kernel.

---

## Part H — Prefill attention

Decode attention and prefill attention share the flash-attention idea and
almost nothing else. This section is the contrast; Ch 10 Part D has the host
plumbing.

### H.1 What changes

| | Decode | Prefill |
|---|---|---|
| Query rows | 1 | 8 per threadgroup (`NQPTG`), many TGs |
| Mask | host passes `kv_start`, `kv_seq` | causal mask computed **inside** the kernel |
| KV of new tokens | one row, kept in threadgroup memory as f32 | whole chunk batch-appended before attention |
| Kernel family | `flash_decode_*` (one TG per head) | `flash_attn_ext_*` (tiled, MMA) or `attention_causal_strided_*` |
| Regime | bandwidth-bound | compute-bound (Ch 00c Part A.3) |
| Uses simdgroup matrices | no | yes |

The mask difference is the structural one. In decode there is exactly one query,
so "which keys may I see" is a pair of integers the host can compute. In
prefill each of the 8 query rows in a tile has a *different* allowed range, so
the kernel must derive it from indices — and for sliding layers, both a lower
and an upper bound.

### H.2 Tile geometry

```16:16:src/shaders/ggml_flash_attn_ext.metal
#define OP_FLASH_ATTN_EXT_NCPSG 64
```

```146:147:src/shaders/ggml_flash_attn_ext.metal
constant int32_t FC_flash_attn_ext_blk_nqptg = 8;
constant int32_t FC_flash_attn_ext_blk_ncpsg = 64;
```

```223:223:src/shaders/ggml_flash_attn_ext.metal
constant int32_t FC_flash_attn_ext_nsg = 4;
```

So the working tile is `8 query rows × 64 keys`, with 4 simdgroups per
threadgroup by default. Each threadgroup:

```text
load 8 query rows into threadgroup memory (as half, for MMA)
for each 64-key chunk in [0, kv_seq):
    compute the 8×64 score block with simdgroup matrix multiplies
    apply causal (and window) masking from indices
    online-softmax update for all 8 rows
    accumulate 8×head_dim outputs via more MMAs
write 8 output rows
```

Compare with Part D: the decode kernel's inner loop is a dot product per key
and a single-threaded softmax update; here everything is an 8×8 matrix
operation. Same algorithm (online softmax), completely different machine
utilization — because at 8 query rows there is enough reuse to feed the matrix
units.

### H.3 The NSG story (`AGENTS.md` E23)

Raising `NSG` from 4 to 8 for `head_dim=256` gave prefill 581–591 tok/s against
llama.cpp's 593.9, cutting the flash phase from ~2480 ms to ~1900 ms at 4k.

More simdgroups per threadgroup means more of the tile in flight, so latency is
better hidden. But threadgroup memory scales with `NSG × head_dim`: h256 at
NSG=8 needs ~24 KB, inside the ~32 KB budget; h512 at NSG=8 would need ~48 KB,
which is not available. So h512 stays at NSG=4 and the host must pick per
head_dim (Ch 00c Part B.4).

This is the cleanest example in the repo of an optimization that is
*head_dim-conditional*. There is no single best NSG.

### H.4 Function constants trim the edges

```214:215:src/shaders/ggml_flash_attn_ext.metal
constant bool FC_flash_attn_ext_has_kvpad [[function_constant(0)]];
constant bool FC_flash_attn_ext_bc_mask [[function_constant(1)]];
```

`has_kvpad` (is `kv_seq` a multiple of the chunk?) and `bc_mask` (does the mask
need bounds checking?) are compiled away. The host picks the variant from the
shapes it is about to dispatch, so the common aligned case runs branch-free.

`AGENTS.md` E16 tested whether *padding the KV length* to help alignment was
worth it and found it a wash (4096 ≈ 4112). Consistent with the design: the
aligned variant was already selected, and one ragged tile out of 64 is noise.

### H.5 `TILED_EXT_MIN_Q=2` — the MTP verify win

llama.cpp switches from its vec kernel to the tiled ext kernel at `q_len ≥ 20`.
This repo's default is **2**, and that difference bought the largest single MTP
gain in the log (`AGENTS.md` M4: verify GPU time 44 → 36 ms, end-to-end
37.7 → 42.4 tok/s).

Why the thresholds differ: llama.cpp's sub-20 fallback is a good vec kernel,
so it has something better to fall back *to*. This engine's small-q fallback is
per-row causal attention — one dispatch per query row, each re-reading the KV —
which is much worse than a tiled kernel running at 3/8 occupancy. The right
threshold depends on the quality of your alternative, not on a property of the
tiled kernel.

Lesson worth generalizing: **an imported constant encodes the source
codebase's trade-offs.** Re-derive it for yours.

---

## Part I — Attention performance map

Every attention experiment in `AGENTS.md`, with the reasoning that made it
predictable in hindsight:

| Experiment | Result | Why |
|---|---|---|
| `fastMathEnabled` (#2) | no change | bandwidth-bound; ALUs idle (Ch 00c A.2) |
| 32-thread vs 256-thread kernel (#7) | identical | same memory path saturated either way |
| `KQ_NR0 = 2` (#3) | 42.8 vs 46.1 | less activation reuse per weight byte |
| non-flash single-pass (#6) | same | tiling overhead is negligible at kv_seq 200 |
| V stride fix (#5) | within noise | not a bank-conflict problem |
| MWG NWG=32 at kv 441 (#8) | 40.4, worse | 1 iteration per WG; dispatch cost dominates |
| ggml MWG always (#13) | flat ~49 | context-stable, no short-ctx win |
| hybrid auto (#14) | 53.5 / 50.0 | best of both, by routing on kv_seq |
| GQA tiled default-on (#11) | garbage at 59.6 | fusion boundary changed append order |
| h256 NSG 4→8 (E23) | prefill 533 → 585 | latency hiding within the smem budget |
| `TILED_EXT_MIN_Q` 20→2 (M4) | verify 44 → 36 ms | our small-q fallback was the weak part |

Two patterns. First, **everything that touched compute did nothing** and
everything that touched bytes-moved or parallelism-structure mattered. Second,
the two garbage-output results were both *fusion boundary* changes, not
arithmetic changes — which is why Part G calls the ladder a contract.

---

## Part J — Exercises that prove you learned this

Do these with the code open. Write answers in your own notes.

1. For `h=13`, `G=5`, what is `kv_h`? Which query heads share that KV head?
   Which of them performs the Q4 append in full fused?

2. Simulate `flash_softmax_tile` by hand with scores `[0, 2, -1]` starting
   from `m=-∞, ℓ=0`. Give `shared_update` after the tile and the `α` values
   implied by `shared_exp * inv_ℓ`.

3. Explain why `shared_scores` can be reused as reduction scratch during
   Q-load without corrupting attention.

4. Trace one wrong scenario: host calls full_fused AND
   `encode_kv_append_q4_0` for the same token. What happens to the cache?
   (Usually overwrite with same data — but say when it would differ.)

5. Trace the hybrid bug scenario at `kv_seq=200` with `ATTENTION_KERNEL=auto`
   if `needs_explicit_kv_append` is deleted. Which token is missing from
   attention, and why is the *first* part of the output still fine?

6. Open `flash_dot_k_shared_hd`. Why must `pos == cur_seq` use `shared_k`
   rather than reading the cache even after a correct append in a
   *different* kernel ordering?

7. MWG with `NWG = 32`, `C = 32`. At what `kv_seq` does each workgroup get at
   least 4 loop iterations? Compare with the 441-token measurement in #8 and
   propose a better `NWG` for kv_seq ≈ 500.

8. In the MWG partial write, explain why `NWG > 1` must **not** normalize by
   `1/ℓ` locally. Construct a two-workgroup example where doing so gives the
   wrong answer.

9. Prefill tile is 8 queries × 64 keys. For a chunk of 512 tokens at kv_seq
   512, how many threadgroups and how many score blocks total? How many of
   those blocks are fully masked out by causality, and what does that imply
   about a causal-aware tile skip?

10. h512 cannot use NSG=8. Compute the threadgroup memory at h256/NSG=8 and
    h512/NSG=8 from the tile shapes, and confirm the 32 KB argument.

11. Design an experiment that would distinguish "our attention kernel scales
    worse than llama.cpp's" from "our dispatch overhead is higher." Which
    `AGENTS.md` entry already did this, and what was the argument?

---

## Part K — Study checklist

- [ ] I can explain the full-fused kernel phases in order without looking.
- [ ] I can derive `old_factor` and `inv_ℓ` from online softmax algebra.
- [ ] I know why append is gated on `h % G == 0`.
- [ ] I know why current-token K/V are f32 in TG memory during attend.
- [ ] I can explain MWG's partitioning, its partial-state layout, and why the
      reduce must be a separate dispatch.
- [ ] I can state at what `kv_seq` MWG starts to make sense, with arithmetic.
- [ ] I can explain the hybrid append bug as a split predicate, and name the fix.
- [ ] I can list the five rungs of the fusion ladder and their append semantics.
- [ ] I can explain why a "decomposed but equivalent" path was not equivalent.
- [ ] I can contrast decode and prefill attention on six axes.
- [ ] I can explain why `TILED_EXT_MIN_Q` differs from llama.cpp's threshold.
- [ ] I can match every `[[buffer(i)]]` in the h256 entry to the encode site.

**Next:** [08_mlp_norms_ple.md](08_mlp_norms_ple.md) for the other half of
the layer — then come back and re-read Part D once more. The second read
is where it sticks.
