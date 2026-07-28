# 00c — GPU Execution and Metal: A Working Mental Model

"The GPU runs my kernel" is not a model you can debug with. This chapter
builds the execution model from the hardware constraint upward, then walks the
exact Metal objects and dispatch geometry this codebase uses, with real
kernels and real encode calls.

By the end you should be able to look at any `kernel void` in
`src/shaders/` and answer: how many threadgroups run this, what does one
threadgroup own, where does it synchronize, and is it bandwidth- or
compute-limited.

---

# Part A — The roofline, derived

## A.1 Two numbers describe a machine

```text
Peak compute:    ~3.2 TFLOPS   (M1 Pro, f32 throughput, order of magnitude)
Peak bandwidth:  ~200 GB/s
```

Any kernel moves `B` bytes and does `F` floating-point operations. Its
**arithmetic intensity** is `I = F / B` FLOP per byte.

Time is bounded below by both limits:

```text
t ≥ B / bandwidth        (memory-bound floor)
t ≥ F / compute          (compute-bound floor)
```

The kernel is memory-bound when `B/bandwidth > F/compute`, i.e. when

```text
I  <  compute / bandwidth  =  3.2e12 / 200e9  ≈  16 FLOP/byte
```

**That single number — ~16 — is the ridge point of this machine.** Below it you
are waiting for memory; above it you are waiting for the ALUs. Memorize it and
most performance questions answer themselves before you write any code.

## A.2 Where our kernels sit

Take one matvec against a `[10240, 2560]` weight matrix (an MLP gate
projection).

FLOPs are fixed by the math: `2 × 10240 × 2560 ≈ 52 MFLOP` (one multiply and
one add per weight). Bytes depend on the format:

| Format | Bytes/weight | Bytes moved | Intensity | Verdict |
|---|---|---|---|---|
| F32 | 4 | 105 MB | 0.5 | memory-bound by 32× |
| F16 | 2 | 52 MB | 1.0 | memory-bound by 16× |
| Q8_0 | 1.06 | 28 MB | 1.9 | memory-bound by 8× |
| Q4_0 | 0.56 | 15 MB | 3.5 | memory-bound by ~4.5× |

Every decode kernel in this repo is memory-bound, and not marginally —
by nearly an order of magnitude. Three consequences follow directly, and all
three appear in `AGENTS.md`:

1. **Compute optimizations do nothing.** `fastMathEnabled = true` (experiment
   #2) changed throughput by zero, because the ALUs already idle most of the
   time. Predicted by the table above, no measurement needed.
2. **Thread count does not matter much.** A 32-thread kernel and a 256-thread
   kernel gave identical throughput (#7). Both saturate the same memory path.
3. **Reducing bytes always helps.** Quantization is the only lever with
   guaranteed payoff, which is why Ch 05 is the longest GPU chapter.

## A.3 Prefill crosses the ridge

Now do the same matvec for `S` tokens at once. Weight bytes are unchanged
(read once, reused for every token); FLOPs multiply by `S`:

```text
I(S) = S × 52 MFLOP / 15 MB ≈ 3.5 × S     (Q4_0)
```

| S | Intensity | Regime |
|---|---|---|
| 1 | 3.5 | memory-bound |
| 4 | 14 | just below the ridge |
| **~5** | **~16** | **crossover** |
| 64 | 224 | compute-bound |
| 512 | 1800 | deeply compute-bound |

So the *same operation* changes character somewhere around a batch of 5 rows.
This is not a rule of thumb; it is the ridge point divided by 3.5.

And it explains the awkward regime MTP verify lives in: `seq` 2–8 straddles the
crossover, too small for matmul tiling and too large to waste as separate
matvecs. That is exactly why the `mul_mv_ext` kernels exist (Ch 05 Part G,
`AGENTS.md` M2) and why forcing `mul_mm` there measured **20 tok/s**.

### Exercises A

1. Compute intensity for the `lm_head` matvec (`[262144, 2560]`, Q4_0) at
   batch 1 and batch 8. Which side of the ridge is each?
2. A machine has 400 GB/s and 3.2 TFLOPS. What is its ridge point? Does Q4_0
   decode get better or worse relative to the ridge?
3. Reading a Q4_0 KV cache row of head_dim 128 (72 bytes) and dotting it with a
   query is how many FLOPs? What is the intensity? What does that tell you
   about attention at long context?

---

# Part B — The execution hierarchy

## B.1 Nested teams, not a grid

```text
Device                        one GPU
  Command Queue               ordered submission mailbox
    Command Buffer            one recorded batch of work; commit → runs
      Compute Encoder         the API you append dispatches to
        Dispatch (grid)       N threadgroups running the same kernel
          Threadgroup         64–1024 threads; owns threadgroup memory
            Simdgroup         exactly 32 threads, lockstep
              Thread          one lane; private registers
```

| Level | Shared state | Cost intuition |
|---|---|---|
| Thread | private registers | free, but limited; spilling is a cliff |
| Simdgroup (32) | implicit, via `simd_*` ops | a few cycles to reduce across all 32 |
| Threadgroup | explicit `threadgroup T x[]` | fast SRAM, **max ~32 KB per TG** |
| Device | `device T*` pointers | DRAM, ~200 GB/s, hundreds of cycles latency |

Two rules govern everything else:

- **A threadgroup is the unit of cooperation.** Threads in one TG can share
  memory and synchronize. Threads in different TGs effectively cannot.
- **A simdgroup is the unit of execution.** 32 lanes advance together. If they
  take different branches, both paths execute with lanes masked off
  (divergence).

## B.2 Simdgroup = 32, always

On Apple GPUs the simd width is 32. `thread_index_in_simdgroup ∈ [0, 32)`.
This is why you see `32`, `64`, `256` everywhere and never `48`.

The useful primitives:

```metal
float s = simd_sum(x);      // sum x across the 32 lanes, broadcast to all
float m = simd_max(x);      // same for max
```

`simd_sum` is a register-level reduction — no memory, no barrier, a handful of
cycles. It is the reason the ggml matvec kernels assign one output row per
simdgroup: 32 lanes each accumulate a partial dot product, then one `simd_sum`
finishes it (Ch 05 Part D).

**Different simdgroups in the same threadgroup are not in lockstep.** To
combine across them you need threadgroup memory plus a barrier.

## B.3 Barriers: what they do and when you need one

```metal
shared_tmp[tid] = partial;                        // each thread writes ITS slot
threadgroup_barrier(mem_flags::mem_threadgroup);  // wait for all writes
float v = shared_tmp[0];                          // now safe to read others'
```

A barrier does two things: it blocks until every thread in the threadgroup
reaches it, and it makes prior threadgroup-memory writes visible.

Real example — the reduction loop in `rmsnorm`:

```1605:1610:src/shaders/llama.metal
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
```

Why the barrier is **inside** the loop: at `stride = 1`, thread 0 reads
`shared_sum[1]`, which thread 1 wrote at `stride = 2`. With 256 threads that is
8 different simdgroups, with no ordering guarantee between them. Remove the
barrier and you get a race whose signature is maddening: correct for
`tg_size ≤ 32` (one simdgroup, accidentally lockstep) and wrong above it.
"Works at head_dim 128, fails at 512" is almost always a missing barrier.

Note also that `if (tid < stride)` means most threads are idle in late
iterations. That is fine — the alternative (fewer threads, more serial work) is
worse, and the reduction is `log₂(256) = 8` steps regardless.

## B.4 Occupancy and the threadgroup-memory budget

The GPU hides memory latency by keeping many threadgroups resident per core.
The limits on residency are registers per thread and **threadgroup memory per
TG**. With ~32 KB available, a kernel that uses 30 KB gets one resident TG per
core and cannot hide latency behind anything.

This is a live constraint in this repo, not a theoretical one. From
`AGENTS.md` E23:

> Raised Metal/host NSG for h256 **4 → 8** (24 KB smem; h512 stays 4).

Work out why h512 cannot follow: threadgroup memory for the flash attention
tiles scales with `NSG × head_dim`, so doubling NSG at head_dim 512 would want
~48 KB — over the limit. It would not compile, or would silently reduce
occupancy to nothing. So the same optimization is right for one head_dim and
impossible for another, and the host code has to know that
(`pipeline_for(head_dim)`).

**Budget your threadgroup memory explicitly.** Count the arrays, multiply by
element size, compare against 32 KB, and expect a cliff, not a slope, when you
cross it.

### Exercises B

1. A kernel declares `threadgroup float tile[8][256];`. How many bytes? How
   many such TGs can be resident per core with a 32 KB budget?
2. Why is `simd_sum` cheaper than a threadgroup-memory reduction? When can you
   *not* use it?
3. In the `rmsnorm` reduction, how many steps for `tg_size = 1024`? How many
   threads are active in the last step?
4. Construct the exact interleaving that makes a barrier-free reduction produce
   a wrong answer with 64 threads.

---

# Part C — Metal objects, as this code uses them

## C.1 The cast

| Metal type | Role here |
|---|---|
| `MTLDevice` | the GPU |
| `MTLCommandQueue` | creates command buffers; one per `MetalContext` |
| `MTLLibrary` | compiled `.metal` source (all shader files concatenated) |
| `MTLFunction` | one `kernel void` looked up by name |
| `MTLComputePipelineState` | executable form of that function (~150 of them) |
| `MTLBuffer` | device memory; `StorageModeShared` here |
| `MTLCommandBuffer` | a recorded batch; `commit()` submits it |
| `MTLComputeCommandEncoder` | `set_*` / `dispatch_*` recording API |

## C.2 The ownership split that keeps this sane

```text
MetalContext (gpu.rs)
  owns: device, queue, ~150 pipelines
  does: encode_*(encoder, ...) — appends to an encoder someone else opened
  does NOT: create command buffers on the hot path, commit, or wait

Gemma4GpuModel / decode_fused (gemma4_gpu_model.rs, decode_fused.rs)
  does: new_command_buffer(), new_compute_command_encoder()
  does: call many encode_*
  does: end_encoding(), commit(), wait_until_completed()
```

Grep `new_command_buffer` in `gpu.rs` and you will barely find it in hot paths
— by design. The consequence is that `encode_*` helpers compose freely: a
caller can put 600 dispatches in one command buffer or split them across two
(`METAL_N_CB`), and no helper needs to know.

## C.3 `BufferView`: a typed window

```18:24:src/gpu.rs
pub struct BufferView {
    pub buffer: Buffer,
    pub offset: u64,
    pub length: u64,
    pub format: u8,   // F16 / Q4_0 / Q4_K / …
}
```

Weights are suballocated from large buffers, so a tensor is a
`(buffer, offset, length)` triple. The `format` tag is the part that is not
mere bookkeeping: a Q4_0 and a Q4_K tensor of the same shape have **identical
byte length** (Ch 03 Part A.2), so nothing about the bytes reveals which
dequantization formula is correct. `format` is a correctness invariant, and
every kernel-selection branch in `decode_fused.rs` reads it.

Offsets go into `set_buffer(index, Some(buf), offset)`, so the kernel always
indexes from zero. That is why so many helpers exist in `_view` / `_at` /
`_at_view` variants — same kernel, different addressing.

## C.4 Compilation: eager, lazy, and specialized

`MetalContext::new` concatenates the shader sources, builds one library, then
resolves each kernel by name into a pipeline:

```text
1. read/concat .metal sources
2. device.new_library_with_source(...)
3. get_fn(name) → library.get_function → new_compute_pipeline_state
   (panics if the name is missing — a typo fails at startup, loudly)
4. store each pipeline in a named struct field
5. resolve env policy once (attention mode, matvec kernel, KV type, …)
```

Three refinements:

- **Eager for decode, lazy for prefill.** `mul_mm_*` pipelines sit behind
  `OnceLock` with a separate library, so a chat-only session never pays that
  compile time.
- **Head-dim specialization.** Separate pipeline fields for `h128` / `h256` /
  `h512`, chosen by a small `pipeline_for(head_dim)` match at encode time.
  Compile-time constant head_dim lets the compiler unroll the inner loops and
  size registers exactly.
- **Function constants.** `flash_attn_ext` builds variants via
  `FunctionConstantValues` (`has_kvpad`, `bc_mask`), so boundary checks compile
  away in the common case. The host picks the variant by shape.

All three are the same idea: **move decisions from run time to compile time
when the decision is known and the code is hot.**

---

# Part D — Dispatch geometry

This is where most beginner bugs live, so we do it concretely, with three real
kernels from this repo that use three different geometries.

## D.1 The two dispatch calls

```text
dispatch_threads(total_threads, threads_per_tg)
    → Metal computes the TG count, rounding up. Threads past your data DO run.

dispatch_thread_groups(tg_count, threads_per_tg)
    → you compute the TG count. Nothing is rounded for you.
```

Use `dispatch_threads` for flat elementwise maps where "thread" is the natural
unit. Use `dispatch_thread_groups` when a threadgroup *owns* something — a row,
a head, a token — because then the count is part of your algorithm and should
be written explicitly.

## D.2 Geometry 1 — elementwise: one thread per element

```2696:2705:src/shaders/llama.metal
kernel void gelu_mul(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = gelu_pytorch_tanh(gate[gid]) * up[gid];
}
```

and the host side:

```4781:4786:src/gpu.rs
        encoder.set_compute_pipeline_state(&self.gelu_mul_pipeline);
        encoder.set_buffer(0, Some(gate_buf), gate_offset);
        encoder.set_buffer(1, Some(up_buf), up_offset);
        encoder.set_buffer(2, Some(out_buf), out_offset);
        encoder.set_bytes(3, 4, &n as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
```

Read the correspondence carefully, because this is the pattern every kernel
follows:

- `[[buffer(0..2)]]` ← `set_buffer(0..2, ...)`, arrays, `device` pointers.
- `[[buffer(3)]]` ← `set_bytes(3, 4, ...)`, a scalar, `constant uint&`.
- `[[thread_position_in_grid]]` is filled by the hardware, not bound.
- `n = 10240`, `tg = 256` → Metal launches `ceil(10240/256) = 40` threadgroups.
- `if (gid >= n) return;` is **mandatory**. With `n = 10241` Metal would launch
  41 TGs = 10496 threads, and 255 of them would write out of bounds.

That guard is the single most common new-kernel bug. It is one line and it is
not optional.

## D.3 Geometry 2 — one threadgroup per row (reduction)

`rmsnorm` (Ch 08 Part A.2) has a completely different shape: one threadgroup,
256 threads, cooperating over one vector of `dim` values. The thread loop is
strided:

```1595:1600:src/shaders/llama.metal
    float partial_sum = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = x[i];
        partial_sum += val * val;
    }
    shared_sum[tid] = partial_sum;
```

Why `i += tg_size` rather than giving each thread a contiguous block? **Memory
coalescing.** At any instant the 256 threads are reading 256 *adjacent* floats,
which the hardware merges into a few wide transactions. If thread 0 read
`[0..10)` and thread 1 read `[10..20)`, each instant would touch 256 addresses
scattered 40 bytes apart — many more transactions for the same data.

**Rule: consecutive threads should touch consecutive addresses.** This one
rule explains the loop structure of nearly every kernel in the repo.

The batched sibling adds an outer dimension via the threadgroup id:

```1756:1763:src/shaders/llama.metal
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint row_offset = tgid * dim;
    threadgroup float shared_sum[256];

    float partial_sum = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = x[row_offset + i];
```

`tgid` selects the token row; the host dispatches `seq_len` threadgroups. One
line of change, and a decode kernel becomes a prefill kernel. This
"threadgroup owns row `tgid`" pattern is everywhere in the batched paths
(Ch 10).

## D.4 Geometry 3 — one threadgroup per attention head

The fused decode attention kernel dispatches **one threadgroup per query
head**, with `FLASH_TG_SIZE = 256` threads each:

```text
tg_count = num_heads            (or num_kv_heads for the GQA-tiled variants)
tg_size  = 256                  (8 simdgroups)
```

What one TG owns: one head's entire attention — load Q into threadgroup
memory, QK-norm it, RoPE it, then loop over KV tiles maintaining online-softmax
state in threadgroup memory, accumulating the output. Ch 07 walks it line by
line.

Why per-head rather than per-token-per-head: in decode there *is* only one
token, so heads are the only available parallelism. 20 heads × 256 threads =
5120 threads — modest, which is part of why decode cannot saturate the machine
and why `AGENTS.md` #7 found thread count irrelevant.

The GQA variants change the geometry to one TG per *KV* head so that the 5
query heads sharing that KV can read the tile once. Better bandwidth, and a
different sharing protocol that must match the cache layout — the source of
`AGENTS.md` #11.

## D.5 Grid math is correctness, not tuning

Three ways to get it wrong, with their symptoms:

| Mistake | Symptom |
|---|---|
| TG count too low | tail of the output never written (stale/zero) |
| TG count too high, no guard | out-of-bounds writes corrupting other buffers |
| `tg_size` mismatched with a `threadgroup` array size | race or overflow inside the kernel |

Note the third: `rmsnorm` declares `shared_sum[256]` and indexes it by `tid`.
Dispatch it with 512 threads per TG and you have a buffer overflow *inside*
threadgroup memory. The kernel does not check; the host must.

### Exercises D

1. `n = 5000`, `tg = 256`, `dispatch_threads`. How many TGs? How many threads
   run past `n`?
2. Rewrite `gelu_mul` for `dispatch_thread_groups`. What TG count do you pass,
   and what must you add inside the kernel?
3. `rmsnorm` with `dim = 2560`, `tg_size = 256`: how many elements per thread
   in phase 1? Which addresses does thread 5 touch?
4. Explain, in terms of memory transactions, why the strided loop beats a
   blocked loop. Then explain when blocked would be *better*.
5. Decode attention: 20 heads, 256 threads/TG. Total threads? Now for prefill
   at S=512 with the tiled kernel — roughly how much more parallelism?

---

# Part E — Memory access patterns

Since everything in decode is bandwidth-bound, how you touch memory *is* the
performance.

**1. Coalesce.** Consecutive lanes → consecutive addresses (Part D.3).

**2. Use wide loads.** `float4` / `half4` moves 16 or 8 bytes in one
instruction. You will see `float4` casts throughout the flash attention and
matvec kernels; they cut instruction count and improve transaction width.

**3. Read weights once.** Fusion's real payoff is not saving dispatches, it is
avoiding a second pass over a big tensor. Note the corollary, which
`AGENTS.md` M7 proves: fusing two kernels that read the *same* weights twice
either way saves almost nothing. Before fusing, ask "which bytes does this stop
reading?" If the answer is "only activation scratch," expect noise.

**4. Prefer streaming to random access.** The KV cache layout (head-major, then
position, then block — Ch 06 Part C) exists so an attention kernel walking
positions reads a contiguous run.

**5. Threadgroup memory is for reuse, not for staging.** Copying device →
threadgroup → registers is only worth it if you read the data more than once.
Loading a KV tile that 5 query heads will use: worth it. Loading each element
once: pure overhead.

---

# Part F — Three axes of slowness

When something is slow, it is one of three things. Diagnose before optimizing.

| Axis | Test | Fix | `AGENTS.md` |
|---|---|---|---|
| **Bandwidth** | bytes moved ÷ 200 GB/s ≈ measured time? | quantize, fuse to skip passes, share tiles | Q4_0 weights, Q4_0 KV |
| **Tiling / occupancy** | ALUs idle but bandwidth not saturated? | tile sizes, NSG, threadgroup memory budget | E23 NSG 4→8, M4 tiled ext |
| **Dispatch / launch** | many tiny kernels? | fuse, mega-kernel, fewer command buffers | #1 (2.4 ms/token, real but not the gap) |

The discipline that makes this work is checking whether the suspect's magnitude
*scales the way the symptom scales*. `AGENTS.md` #1 is the model: dispatch
overhead is ~2.4 ms/token and constant, while the gap versus llama.cpp grows
with context. Therefore dispatch overhead is not the gap — and that single
argument closed four more experiments (#2, #4, #6, #7) without running them.

---

# Part G — Synchronization, and what Metal will not give you

**Within a threadgroup:** `threadgroup_barrier`. Cheap, reliable.

**Within a simdgroup:** implicit; `simd_*` ops need no barrier.

**Between threadgroups in one dispatch: nothing.** No ordering, no reliable
cross-TG atomics for this kind of reduction. If TG 3 needs TG 7's result, you
need two dispatches.

**Between dispatches in one command buffer:** sequential by default. Dispatch
`k+1` sees all of dispatch `k`'s writes. This is what makes the long chains in
`decode_fused.rs` correct without explicit fences.

**Between command buffers:** ordered by submission to the queue.

Two consequences you will meet in the code:

1. **The MWG attention path needs a reduce kernel.** Partitioning the KV cache
   across 32 workgroups means 32 partial results that must be combined — and
   since TGs cannot cooperate, that combination is a *second dispatch*
   (`flash_attn_ext_vec_reduce`). Ch 07 Part F. That extra dispatch is part of
   why MWG loses at short context (#8).
2. **`decode_mega.metal` is serial inside one threadgroup.** A "run the whole
   token in one kernel" design cannot spread across TGs, because layer `k+1`
   depends on all of layer `k` and there is no cross-TG barrier. So it runs one
   TG and accepts low parallelism in exchange for zero dispatch overhead — a
   deliberate trade, and the reason it is an experiment rather than the default.

Also note the ordering subtlety that catches people: **encoding order is not
execution time.** All the `encode_*` calls happen on the CPU first; the GPU
runs later, at `commit()`. So a buffer you overwrite later in the *encode*
sequence is overwritten for *all* dispatches that read it at execute time,
including earlier ones. That is the shape of the RoPE-table reuse bug mentioned
in Ch 00b Part 5.5.

---

# Part H — Unified memory coherence

`MTLResourceStorageModeShared`: CPU and GPU share physical pages. No PCIe, no
explicit copies. But "shared" is not "synchronized":

1. **Do not read GPU output on the CPU before completion.** You need
   `wait_until_completed()` (or a completion handler). Reading early gives you
   whatever was there — often the previous token's values, which looks like a
   subtle model bug rather than a race.
2. **Finish CPU writes before the GPU reads them.** Write buffer → encode →
   commit, on one thread, is safe. Writing after commit is not.
3. **Access patterns still matter.** Unified memory removes the copy, not the
   bandwidth limit or the coalescing requirement.

Practical rule for this codebase: the only places that call
`wait_until_completed` are the forward-path functions in
`gemma4_gpu_model.rs`, and they do it immediately before reading logits or a
sampled token back. If you find yourself wanting to read a buffer from
somewhere else, you are probably in the wrong layer.

---

# Part I — Reading a Metal kernel signature

You will read hundreds of these. The attributes are the whole interface:

| Attribute | Type | Meaning |
|---|---|---|
| `[[buffer(i)]]` | `device T*` / `constant T&` | bound by `set_buffer` / `set_bytes` at index `i` |
| `[[thread_position_in_grid]]` | `uint` | global thread id (flat dispatches) |
| `[[thread_index_in_threadgroup]]` | `uint` | `tid` within the TG |
| `[[threads_per_threadgroup]]` | `uint` | TG size, as dispatched |
| `[[threadgroup_position_in_grid]]` | `uint` | which TG this is — the "row/head I own" |
| `[[simdgroup_index_in_threadgroup]]` | `uint` | which simdgroup (`sgid`) |
| `[[thread_index_in_simdgroup]]` | `uint` | lane, 0–31 |
| `threadgroup T x[N]` | declaration | TG-shared array, counts against 32 KB |
| `device` / `constant` | address space | writable device memory / read-only uniform |

Reading protocol when a kernel misbehaves: put the Metal signature and the
encode site side by side and walk them top to bottom. Is `set_buffer(0, ...)`
the same tensor as `[[buffer(0)]]`? Is every `constant T&` matched by a
`set_bytes` with the right size? Nine times out of ten the bug is visible in
30 seconds this way, and invisible for an hour any other way.

---

# Part J — Labs

Do these with the repo checked out. They take minutes and they lock in the
model.

**Lab 1 — find the ridge point empirically.** Run `--bench-matvec` and
`--bench-mul-mm`. Compute achieved GB/s for the matvec and achieved GFLOPS for
the matmul. Which one is near its respective peak?

**Lab 2 — count dispatches.** `PROFILE_DISPATCHES=1` on one decode token.
Compare against the ~13–14 per layer estimate in Ch 08 Part F. Multiply by
~5 µs and compare with the ~2.4 ms in `AGENTS.md` #1.

**Lab 3 — break a barrier.** Copy `rmsnorm` to `rmsnorm_broken`, delete the
barrier inside the reduction loop, dispatch it with 256 threads, and compare
output against the original. Then try 32 threads and explain why it "works."

**Lab 4 — break the bounds guard.** Copy `gelu_mul`, remove `if (gid >= n)
return;`, dispatch with `n = 100`. What gets corrupted? Why does the corruption
location depend on `tg_size`?

**Lab 5 — budget threadgroup memory.** Find the flash attention kernel's
`threadgroup` declarations for h256. Add up the bytes at NSG=4 and NSG=8.
Confirm the 24 KB figure from `AGENTS.md` E23, then compute what h512 would
need at NSG=8.

---

## Checklist

- [ ] I can derive the ~16 FLOP/byte ridge point and use it to classify a kernel.
- [ ] I can compute arithmetic intensity for a matvec in any weight format.
- [ ] I know at roughly what batch size a projection crosses the ridge.
- [ ] I can name the four levels of the hierarchy and what is shared at each.
- [ ] I can explain why the barrier is inside the reduction loop.
- [ ] I know the 32 KB threadgroup-memory budget and why h512 cannot use NSG=8.
- [ ] I can map every `[[buffer(i)]]` to its `set_buffer`/`set_bytes` call.
- [ ] I know when to use `dispatch_threads` vs `dispatch_thread_groups`.
- [ ] I can explain why consecutive threads must touch consecutive addresses.
- [ ] I know that TGs cannot synchronize, and the two design consequences.
- [ ] I know the three coherence rules for `StorageModeShared`.

**Next:** [01_system_map.md](01_system_map.md), then the GPU chapters.
