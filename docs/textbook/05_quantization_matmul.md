# 05 — Quantization & Matmul: How Decode Actually Burns Bandwidth

If you only remember “we use Q4,” you do not understand this engine. Decode
throughput is almost entirely “how many bytes of weights can we stream and
turn into dots per second.” This chapter builds Q4_0 from first principles,
then walks the production GEMV kernel lane-by-lane, then shows why prefill
switches to `mul_mm` and why MTP verify needs `ext` matvec.

Prerequisites: [00c](00c_gpu_and_metal_fundamentals.md) (bandwidth, TG geometry).
Open while reading:

```text
src/shaders/ggml_mul_mv_q4.metal   # block_q_n_dot_y, mul_vec_q_n_f32_impl
src/gguf.rs                        # CPU dequant reference
src/gpu.rs                         # quantize_q4_0, encode_matvec_*, BufferView
src/ggml_gemv.rs                   # GgmlMulMvArgs layout, dispatch helpers
```

---

## Part A — Why quantization exists here (do the arithmetic)

### A.1 One MLP gate matvec on E4B

```text
W_gate ∈ R^{10240 × 2560}
x      ∈ R^{2560}
y      = W_gate x ∈ R^{10240}
```

FLOPs: `2 * 10240 * 2560 ≈ 52.4e6` (mul+add).

Bytes if weights are f16: `10240 * 2560 * 2 ≈ 52.4 MB`.

Arithmetic intensity: `52.4e6 / 52.4e6 ≈ 1 FLOP/byte`.

M1 Pro class GPU: ~200 GB/s memory, a few TFLOPS peak for this work.
At 1 FLOP/byte you need only ~200 GFLOPS to saturate bandwidth — you are
**memory bound**. Making ALUs faster does nothing (AGENTS: `fastMathEnabled`
was a wash).

Q4_0 stores ~0.5625 bytes/weight → ~14.7 MB for the same matrix → intensity
~3.5 FLOP/byte. Still memory bound, but you move ~3.5× less data.

### A.2 Decode does this over and over

Every token, every layer: Q, K, V, O, gate, up, down, plus PLE and lm_head.
The working set is essentially **the whole model weights** streamed from
DRAM each token. That is why tok/s tracks memory bandwidth and quant format,
not “clever ALU tricks,” until you change bytes or reuse (prefill, ext batch).

---

## Part B — Q4_0 block format until you can write it from memory

### B.1 Layout

32 weights compress to 18 bytes:

```text
byte:  0  1  |  2  3  4  …  17
       d_lo d_hi | qs[0] … qs[15]

d = f16 scale at bytes 0..1
qs[i] packs two codes:
  low  nibble  = code for weight i
  high nibble  = code for weight i+16
```

### B.2 Encode (quantization)

From `gpu.rs::quantize_q4_0` / the KV append kernel — same rule:

```text
max_abs = max_i |w_i|
d       = max_abs / 7.0          # not /8 — keeps symmetric range around 0
code_i  = clamp(round(w_i / d) + 8, 0, 15)
```

Zero-point is **8**. Code 8 ≈ 0. Codes 0 and 15 are the extremes ±7d.

### B.3 Decode (dequantization)

```text
w_i = (code_i - 8) * d
```

CPU reference (`src/gguf.rs`, the `ggml_type::Q4_0` arm of the dequantizer,
condensed):

```rust
let d = f16_to_f32(...);
for i in 0..16 {
    let q_lo = (qs[i] & 0x0F) as i32 - 8;
    let q_hi = (qs[i] >> 4) as i32 - 8;
    out[i] = q_lo as f32 * d;
    out[i + 16] = q_hi as f32 * d;
}
```

### B.4 Worked numbers

Weights: `[0.0, 0.7, -0.7, 1.4, ...]` suppose `max_abs=1.4` → `d=0.2`.

```text
0.0  → round(0)+8 = 8  → dequant (8-8)*0.2 = 0
0.7  → round(3.5)+8 = 12 → (12-8)*0.2 = 0.8   (quantization error!)
-0.7 → round(-3.5)+8 = 4 → (4-8)*0.2 = -0.8
1.4  → round(7)+8 = 15 → (15-8)*0.2 = 1.4
```

Q4_0 is lossy. The engine accepts that for bandwidth. Sensitive tensors
(norms, some PLE) stay f16.

### B.5 Row layout for a matrix

Weight matrix `[M, K]` with `K % 32 == 0`, row-major blocks:

```text
bytes_per_row = (K / 32) * 18
total_bytes   = M * bytes_per_row
```

Row `r` starts at byte `r * bytes_per_row`. The GEMV kernel walks blocks
along the row while streaming activation tiles.

---

## Part C — The algebraic trick that makes the Metal kernel look weird

Naive inner product against one Q4_0 block:

```text
Σ_i (q_i - 8) * d * y_i  =  d * Σ_i q_i y_i  -  8 d Σ_i y_i
```

Compute `sumy = Σ y_i` once, then:

```text
result = d * (Σ q_i y_i_scaled_masks  +  sumy * (-8))
```

That is `block_q_n_dot_y`:

```62:76:src/shaders/ggml_mul_mv_q4.metal
inline float block_q_n_dot_y(device const block_q4_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float2 acc = 0.f;
    device const uint16_t * qs = ((device const uint16_t *)qb_curr + 1 + il/2);
    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F)
                + yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[1] += yl[i + 8] * (qs[i / 2] & 0x00F0)
                + yl[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (sumy * -8.f + acc[0] + acc[1]);
}
```

### C.1 Why `yl` is pre-scaled by 1, 1/256, 1/16, 1/4096

The kernel reads nibbles with masks **without shifting them down to 0..15**.
For example `& 0x0F00` leaves the nibble in bits 8..11, numerically worth
`nibble * 256` if interpreted as an integer in a float multiply. So the
activation that multiplies that mask is stored as `y / 256` in `yl`.

Same idea for `& 0x00F0` (×16) and `& 0xF000` (×4096).

This is not mysticism — it is “fold the shift into the activation” so the
inner loop is mask + multiply + add with no extra shifts. When you port or
“simplify” this and remove the pre-scaling, your matvec is silently wrong
by factors of 16/256/4096.

### C.2 Prove it for one nibble

Suppose low nibble of a byte is code `c=10`, and the mask path uses
`& 0x000F` (already in place). Contribution should be `c * y`.

High nibble path using `& 0x0F00`: the uint16 value contributes
`c * 256` when used as a float factor, so we need `yl = y/256` to get
`c * y`.

---

## Part D — Lane-by-lane: `mul_vec_q_n_f32_impl` (production decode GEMV)

Entry: `matvec_ggml_q4_0` → template `<block_q4_0, NR0=4, NSG=2, NW=32>`.

### D.1 Geometry

```text
NSG = 2 simdgroups per threadgroup
NW  = 32 lanes per simdgroup
NR0 = 4 output rows per simdgroup

threads per TG = 64
rows per TG    = 8
TG grid X      = ceil(M / 8)   for M output rows
```

Inside the TG:

```text
first_row = (tgpig.x * NSG + sgitg) * NR0
```

**Worked:** `tgpig.x=3`, `sgitg=1` → `first_row = (3*2+1)*4 = 28`.
That simdgroup owns output rows 28,29,30,31.

### D.2 Who loads which activations

```metal
const int ix = (tiisg/2);           // which Q4 block along K, among 16 "slots"
const int il = (tiisg%2)*8;         // which half of the 32-weight block
device const float * yb = y + ix * QK4_0 + il;
```

32 lanes split the K axis: each lane repeatedly jumps by `NW/2 = 16`
blocks. Lanes cooperate so that across the simdgroup, the full row of
blocks is covered.

For each block visit, the lane packs 16 floats into `yl[16]` with the
pre-scaling described above, computes `sumy`, then for each of the `NR0`
rows:

```metal
sumf[row] += block_q_n_dot_y(x + ib + row*nb, sumy, yl, il);
```

Note `row*nb`: consecutive output rows' weight blocks are `nb` blocks apart
(`nb = K/32` = blocks per row). The same activation tile dots against
`NR0` different weight rows — **activation reuse**, the small cousin of
what `ext` matvec does for batch>1.

### D.3 Reduction and write

```metal
for each row:
    tot = simd_sum(sumf[row])
    if tiisg == 0 && first_row+row < M:
        dst[...] = tot
```

Only lane 0 writes. That is why wrong `simd_sum` width or writing from all
lanes would trash memory.

### D.4 Host args must match the Metal struct bit-for-bit

```metal
struct ggml_mul_mv_args { int32_t ne00; int32_t ne01; ... };
```

Rust `GgmlMulMvArgs` in `ggml_gemv.rs` must have identical layout. This is
the fragile boundary. If you add a field on one side, matvecs become
garbage without a clean crash.

### D.5 Why Auto picks ggml today

Comment in `DecodeMatvecKernel::pick_for_shape`: ggml `block_q_n_dot_y`
beat the hand-written `fast` kernel on M1 Pro e2e (~27 vs ~25 tok/s). The
`fast` path uses 256-thread TGs and different row tiling — more occupancy
on paper, worse in practice for this shape. **Measure; do not assume.**

---

## Part E — Q4_K and Q6_K: the formats you actually load

A "Q4_K_M" GGUF is not Q4_0. It is mostly **Q4_K** with **Q6_K** for a few
sensitive tensors, and the two dequantize completely differently from Q4_0. If
you learn only Q4_0 you will misread half the kernels in
`ggml_mul_mv_q4.metal`.

### E.1 The Q4_K super-block

```979:984:src/shaders/ggml_mul_mv_q4.metal
struct block_q4_K {
    half     d;
    half     dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};
```

`2 + 2 + 12 + 128 = 144` bytes for **256** weights = 4.5 bits/weight.

The structure is hierarchical, and that is the whole idea:

```text
super-block of 256 weights
  ├─ d     : f16 — scale for the sub-block scales
  ├─ dmin  : f16 — scale for the sub-block minimums
  ├─ 8 sub-blocks of 32 weights, each with:
  │     sc  : 6-bit scale   (quantized, multiplied by d)
  │     m   : 6-bit minimum (quantized, multiplied by dmin)
  └─ qs    : 256 4-bit codes
```

Dequantization is **affine**, not symmetric:

```text
w = (d · sc) · code  −  (dmin · m)
```

Compare with Q4_0's `w = d · (code − 8)`. Two differences that matter:

1. **A per-sub-block minimum.** Q4_0 forces the representable range to be
   symmetric around zero, which wastes codes when a block's weights are, say,
   all in `[0.1, 0.5]`. Q4_K learns an offset per 32 weights, so all 16 codes
   land inside the actual range.
2. **Two levels of scale.** Storing 8 f16 scales per super-block would cost 16
   bytes; storing them as 6-bit ints against a shared f16 `d` costs 6 bytes.
   Same idea for minimums. That is where the byte budget comes from.

The cost is that a nibble alone means nothing — you need the right sub-block's
`(sc, m)` pair, and those are bit-packed across 12 bytes in a layout designed
for extraction speed, not readability:

```1226:1230:src/shaders/ggml_mul_mv_q4.metal
static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                          uchar((q[j+4+k] >>  4) | ((q[j-0+k] & 0xc0) >> 2))};
}
```

Read what this says. The 12 `scales` bytes hold 16 six-bit values (8 scales +
8 mins = 96 bits = 12 bytes exactly). For the first four sub-blocks (`j < 4`)
the 6 bits sit contiguously in the low bits of one byte, so a `& 63` extracts
them. For the last four, the low 4 bits come from one byte and the high 2 bits
are stolen from the top of an earlier byte (`& 0xc0) >> 2`). Nothing is wasted
and nothing is aligned.

**Do not try to derive this from the formula.** Copy it from the reference,
verify with a CPU dequant comparison (`--gguf-kquant-test` in `main.rs` does
exactly that against real tensors), and move on. Bit-packing layouts are
specifications, not mathematics.

### E.2 The dequantizer used by the tiled paths

```1232:1248:src/shaders/ggml_mul_mv_q4.metal
static inline void dequantize_q4_K_f4x4(device const block_q4_K * xb, short il, thread float4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? (float)xb->d : (float)xb->d / 16.f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}
```

Three things to notice, because each is a trick you will see again:

**The `/16.f` on `d`.** When `il >= 2` the kernel reads the *high* nibble with
mask `0xF0`, leaving the code multiplied by 16 (Part C's trick again). Instead
of shifting each of 16 codes right by 4, it divides the scale once by 16.
One division replaces sixteen shifts.

**`il` encodes which 16 of the 256 weights.** `il/4` selects a 32-weight
sub-block pair, `il&1` selects which half, `il&3` selects nibble half and scale
sub-index. A single small integer parameter carries the whole position, so the
caller loops `il` and the dequantizer needs no other state.

**Output is `float4x4`.** 16 values in a register matrix, ready to feed
`simdgroup_multiply_accumulate` or a dot product. The unit of work is chosen to
match the MMA hardware, not the format.

### E.3 Q6_K, briefly

```986:991:src/shaders/ggml_mul_mv_q4.metal
struct block_q6_K {
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    half     d;
};
```

`128 + 64 + 16 + 2 = 210` bytes for 256 weights = 6.5625 bits/weight. Codes are
6 bits, split across a low-nibble array (`ql`) and a 2-bits-per-weight high
array (`qh`), reassembled as:

```1182:1185:src/shaders/ggml_mul_mv_q4.metal
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
```

Note `− 32`: Q6_K is symmetric with zero-point 32 (mid-point of 6 bits), like
Q4_0's 8 — no `dmin` term. Q4_K_M uses Q6_K for tensors where 4 bits hurt,
typically `ffn_down` and the output/embedding matrix.

**Why this matters for the engine:** a single layer can hold Q4_K and Q6_K
tensors simultaneously. That is exactly why `WeightFormat::KQuant` is a
per-layer coarse tag and `BufferView.format` is the per-tensor truth
(Ch 02 Part I), and why the MLP has a "mixed K-quant" branch (Ch 08 Part C.4).

### E.4 The size trap, stated precisely

```text
Q4_0 over K columns: (K/32)  × 18  bytes
Q4_K over K columns: (K/256) × 144 bytes  =  (K/32) × 18 bytes
```

**Identical.** For `K = 2560`: both are 1440 bytes per row. So you cannot infer
the format from `view.length`, and a Q4_0 kernel run on Q4_K data will happily
read 18-byte "blocks" that straddle super-block boundaries, interpret `dmin` as
a scale, and produce numbers of roughly the right magnitude. Output stays
fluent. This is the highest-cost/lowest-visibility bug class in the file, and
the mitigation is entirely upstream: `format` is set at load and never
inferred.

### E.5 K-quant GEMV geometry

```976:977:src/shaders/ggml_mul_mv_q4.metal
#define KQ_NSG 2
#define KQ_NR0 4
```

Same shape as the Q4_0 matvec: 2 simdgroups per threadgroup, 4 output rows per
simdgroup, so 8 rows per TG and 64 threads.

`AGENTS.md` #3 tried `KQ_NR0 = 2` — fewer rows per simdgroup, which means less
activation reuse per weight load — and measured **42.8 vs 46.1 tok/s**. Worse,
as the bandwidth model predicts: halving rows-per-simdgroup doubles the number
of times each activation tile is re-read from threadgroup memory and halves the
work done per weight byte in registers. Do not tune this without a benchmark,
and do not assume "more parallelism" wins in a bandwidth-bound kernel.

---

## Part F — Fusion on matvec: byte accounting

Fusion is popular and frequently disappointing. The way to predict which it
will be, before writing code, is to count the bytes it removes.

| Fusion | Removes | Does **not** remove | Verdict |
|---|---|---|---|
| gate ∥ up in one kernel | one pass over `x` (10 KB), one dispatch | both weight matrices (29 MB) | small |
| gelu_mul into the matvec | two `[10240]` f32 scratch writes + reads (~120 KB), one dispatch | weight bytes | small |
| rmsnorm into the matvec | a `[2560]` roundtrip (~20 KB), one dispatch | weight bytes | small |
| qkv fan-out from one norm | 2 redundant norm passes, 2 dispatches | Q/K/V weight bytes | small |
| **KV append into attention** | a full pass over the KV row + a dispatch | — | real at long ctx |
| Q4_0 instead of F16 weights | **half the weight bytes (14.7 vs 29 MB)** | — | huge |

The pattern is stark: activation-side fusions save kilobytes against tens of
megabytes of weights. They are worth doing — dispatch overhead is ~5 µs each
and there are ~455 per token (`AGENTS.md` #1) — but they are second-order.

`AGENTS.md` M7 is the clean experiment. Fusing gate∥up+GeLU on the ext matvec
path was correct, measured **~0.5–1.5 tok/s** (inside run-to-run noise), and
the log's own explanation is the bandwidth model: "weight bandwidth for gate+up
is unchanged (still read both matrices once); only activation scratch + one
gelu dispatch are saved."

**Rule: before fusing, name the bytes you stop reading. If they are
activations, expect noise.**

---

## Part G — Ext matvec (batch 2–8): the kernel that exists for MTP

### G.1 The regime

MTP verify runs 2–8 rows at once (Ch 14 Part D). Part A.3 of Ch 00c puts the
compute/bandwidth crossover near batch 5, so this range straddles it. Both
obvious options are bad:

- **Per-row matvec, B times:** reads the whole weight matrix B times.
  `B × 14.7 MB` for one MLP gate. At B=3 that is 44 MB where 15 would do.
- **`mul_mm`:** tiles are 64×32; at B=3 you fill 3 of 32 columns, so ~90% of
  every MMA is multiplying padding. `AGENTS.md` M2 measured this: forcing
  `MUL_MM_MIN_SEQ=1` gave **20 tok/s** against 42+ for the ext path.

### G.2 The kernel

```1374:1381:src/shaders/ggml_mul_mv_q4.metal
MV_EXT_KQ_KERNEL(matvec_ggml_ext_q4K_nx8_r2, block_q4_K, dequantize_q4_K_f4x4, 8, 2)
MV_EXT_KQ_KERNEL(matvec_ggml_ext_q4K_nx8_r3, block_q4_K, dequantize_q4_K_f4x4, 8, 3)
MV_EXT_KQ_KERNEL(matvec_ggml_ext_q4K_nx8_r4, block_q4_K, dequantize_q4_K_f4x4, 8, 4)
MV_EXT_KQ_KERNEL(matvec_ggml_ext_q4K_nx8_r5, block_q4_K, dequantize_q4_K_f4x4, 8, 5)
MV_EXT_KQ_KERNEL(matvec_ggml_ext_q6K_nx8_r2, block_q6_K, dequantize_q6_K_f4x4, 8, 2)
MV_EXT_KQ_KERNEL(matvec_ggml_ext_q6K_nx8_r3, block_q6_K, dequantize_q6_K_f4x4, 8, 3)
MV_EXT_KQ_KERNEL(matvec_ggml_ext_q6K_nx8_r4, block_q6_K, dequantize_q6_K_f4x4, 8, 4)
MV_EXT_KQ_KERNEL(matvec_ggml_ext_q6K_nx8_r5, block_q6_K, dequantize_q6_K_f4x4, 8, 5)
```

Eight variants: two formats × four values of `r1ptg` (rows of activation per
threadgroup), all with `nxpsg = 8`. The algorithm:

```text
for each weight block this simdgroup owns:
    dequantize it ONCE into registers (float4x4 via dequantize_q*_K_f4x4)
    for each of r1ptg activation rows:
        accumulate dot(weight_tile, activation_row_tile)
write r1ptg outputs
```

The dequantization — the expensive part for K-quants, with all the bit
unpacking of Part E.1 — is amortized over `r1ptg` rows. Weight bytes are read
once for the whole batch, so bandwidth is `B`-independent while FLOPs scale with
`B`: intensity rises linearly, exactly as Ch 00c Part A.3 wants.

Why four hand-written variants instead of a runtime loop bound: `r1ptg` sets
register allocation. Compile-time `r1ptg` means the accumulator array is
statically sized and fully unrolled. Host-side selection:

```text
mv_ext_kq_r1ptg(batch):  2 → 2,  3 or 6 → 3,  5 → 5,  else → 4
```

The mapping is chosen so `batch` divides evenly into `r1ptg` chunks where
possible (6 = 2×3, 4 = 1×4, 8 = 2×4), avoiding a partially-filled final chunk.

### G.3 What this buys, end to end

From `AGENTS.md` M2–M4, MTP verify at seq=3 went from a sequential
implementation at ~25 tok/s to 42+ with ext matvec, batched `lm_head`, and
tiled attention. Ext matvec alone (with parallel verify) accounted for the jump
to ~34.8.

The remaining puzzle is documented honestly in the log: batch-3 MLP still costs
~1.6× batch-1 where the bandwidth model says ~1.1×. So there is a second effect
— occupancy, or the three separate weight streams — that ext matvec does not
address. Ch 14 Part G.

---

## Part H — Prefill `mul_mm`: a genuinely different algorithm

### H.1 Why the algorithm changes

At sequence length `S`, one weight byte serves `S` activations, so intensity is
`3.5 × S` for Q4 (Ch 00c Part A.3). Past `S ≈ 5` you are compute-bound, and the
right kernel is no longer "stream weights, dot with a vector" but "tile both
operands and feed the matrix units."

### H.2 The tile geometry, from the source

```78:86:src/shaders/ggml_mul_mm_q4.metal
    constexpr short nl = QK_K / 16;  // 16
    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK = 32;
    constexpr short NL0 = NK / 16;  // 2
    constexpr short NL1 = NK / 8;   // 4

    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);
```

Decode this:

```text
One threadgroup computes a [NR0 × NR1] = [64 output rows × 32 batch cols] tile
K is consumed NK = 32 columns at a time
sa = weight tile:     64 × 32 halves = 4096 bytes
sb = activation tile: 32 × 32 halves = 2048 bytes
                                  total 6 KB of threadgroup memory
4 simdgroups per TG (128 threads); each owns a 32 × 16 quarter of the output
```

You can verify the last line from the output address:

```178:180:src/shaders/ggml_mul_mm_q4.metal
    if (!FC_mul_mm_bc_out || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
        device float * C = (device float *)dst + (r0 + 32 * (sgitg & 1))
            + (r1 + 16 * (sgitg >> 1)) * args.ne0;
```

`sgitg & 1` splits the 64 rows into two halves of 32; `sgitg >> 1` splits the 32
columns into two halves of 16. Four simdgroups, `32 × 16` each, tiling
`64 × 32`. That is the entire work decomposition, readable from one address
expression.

### H.3 The inner loop

```160:175:src/shaders/ggml_mul_mm_q4.metal
        FOR_UNROLL(short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL(short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL(short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL(short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
```

Everything here is 8×8 matrices in registers:

- `ma[4]` — four 8×8 half tiles of weights (covering 32 rows).
- `mb[2]` — two 8×8 half tiles of activations (covering 16 columns).
- `mc[8]` — eight 8×8 **float** accumulators = the 32×16 output tile.
- `simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i])` — 8 MMA
  instructions produce all 8 accumulator updates from 4+2 loaded tiles. That
  ratio (6 loads, 8 MMAs, each 8×8×8 = 512 MACs) is the arithmetic-intensity
  win at the register level, and it is why tiling is the whole game in GEMM.

Two details that are not decoration:

**`FOR_UNROLL` is a performance requirement.** The shader's own comment reports
~1.5 vs ~3 TFLOPS for Q4_K with and without full unrolling. Unrolled, the
compiler schedules loads ahead of MMAs and keeps the pipeline full; rolled, each
iteration stalls on its own loads. A factor of two from a pragma.

**Accumulators are f32, operands are f16.** Inputs are cast down for the MMA
hardware; the accumulation stays in f32 so that a 2560-long dot product does not
lose precision. Note the activation cast happens on the way *into* threadgroup
memory (`half2x4(*((device float2x4 *)y))`), so the f16 conversion is paid once
per tile rather than per MMA.

### H.4 Function constants at the edges

```22:23:src/shaders/ggml_mul_mm_q4.metal
constant bool FC_mul_mm_bc_inp [[function_constant(0)]];
constant bool FC_mul_mm_bc_out [[function_constant(1)]];
```

`bc` = bounds check. When `M` and `N` are exact multiples of the tile sizes, the
host selects the variant where both constants are false, and every boundary
branch disappears at compile time — including the slow per-element activation
load path you can see in the `else` of H.2's staging code.

This is the same "move decisions to compile time" theme as head-dim
specialization (Ch 00c Part C.4). It also explains a measurement in
`AGENTS.md` E16: padding the prefill length to align tiles was a **wash**
(4096 vs 4112 identical), because the fast variant was already being selected
at 4096 and the edge cost at 4112 is one partial tile out of many.

---

## Part I — How the host chooses a kernel

The decision tree, in the order the code tests it:

```text
projection of shape [M, K], weight format F, batch/sequence S:

if S ≥ MUL_MM_MIN_SEQ (default ~16) and F ∈ {Q4_K, Q6_K, F16}:
      mul_mm_q4_K_f32 / mul_mm_q6_K_f32 / mul_mm_f16_f32
      → pick bc_inp/bc_out variants from divisibility of M, N

elif 2 ≤ S ≤ 8 and F is K-quant:
      matvec_ggml_ext_q{4,6}K_nx8_r{2..5}      (r from mv_ext_kq_r1ptg(S))
      → and if this is the MLP gate+up: the fused *_gelu_* variant

elif S == 1:                                   # decode
      Q4_0    → matvec_ggml_q4_0  (+ fused dual / gelu / rmsnorm variants)
      K-quant → matvec_ggml_q4_K / q6_K  (+ fused variants)
      F16     → matvec_f16

else:                                          # large S, unsupported format
      batched Q4 projection fallback, per-row
```

Call sites to read: `encode_matvec_*`, `encode_prefill_projection_auto_batch_view`,
`encode_prefill_mlp_gate_up`, `encode_prefill_kquant_projection` in `gpu.rs`,
plus the branch cascades in `decode_fused.rs`.

Three things this tree encodes that are worth stating plainly:

1. **Format decides more than shape does.** A Q4_0 model and a Q4_K_M model
   take different branches at every level.
2. **Small batch is its own regime**, not "prefill with small S." Ch 00c
   Part A.3 says why.
3. **Fused variants are leaves, not a separate axis.** Fusion never changes
   which family is chosen; it only picks a different member of it.

---

## Part J — Exercises

1. Quantize `w = [0, 0.5, −1, 1]` to Q4_0 by hand: `d`, codes, dequantized
   values, per-element error. Then compute the relative error of the largest and
   smallest elements — which suffers more, and why?
2. `M = 10240`, Q4_0 ggml GEMV with 8 rows per TG. How many threadgroups?
   Which rows does `tgpig.x = 10, sgitg = 0` own? Which does `sgitg = 1` own?
3. Explain why `& 0x0F00` in `block_q_n_dot_y` requires `yl = y/256`, and why
   `dequantize_q4_K_f4x4` divides `d` by 16 instead.
4. A `[2560, 2560]` tensor is 1440 bytes per row in both Q4_0 and Q4_K. Write
   the sequence of events if a Q4_0 kernel reads Q4_K bytes: what does it use as
   `d`, and roughly how wrong is the first output element?
5. For Q4_K, count the bits: 8 scales + 8 mins at 6 bits each. Show that they
   fit exactly in 12 bytes, and explain why `get_scale_min_k4_just2` needs two
   different extraction paths.
6. MTP verify at batch 3, MLP gate `[10240, 2560]` Q4_K. Compute weight bytes
   read by (a) three separate matvecs, (b) one ext matvec with `r1ptg = 3`,
   (c) `mul_mm` with a 64×32 tile — and for (c) estimate the fraction of MMA
   work spent on padding.
7. `mul_mm` uses 6 KB of threadgroup memory. How many TGs could be resident per
   core under a 32 KB budget? Compare with the flash attention kernel at h256,
   NSG=8 (24 KB) and comment on why the prefill kernel can afford more
   occupancy.
8. Predict the effect of removing `FOR_UNROLL` from the inner loop, then find
   the shader comment that reports the measurement.

---

## Checklist

- [ ] I can draw a Q4_0 block and dequantize any nibble from memory.
- [ ] I can derive `d·(Σqy − 8Σy)` and point to it in the Metal source.
- [ ] I can draw a Q4_K super-block and write its affine dequant formula.
- [ ] I can explain why Q4_K needs `dmin` and Q6_K does not.
- [ ] I can compute `first_row` for arbitrary `tgpig.x`, `sgitg`.
- [ ] I can state the Q4_0 / Q4_K size coincidence and why `format` is the fix.
- [ ] I can explain the ext matvec's amortization argument with byte counts.
- [ ] I can describe the `mul_mm` tile decomposition and read it from the
      output address expression.
- [ ] I can predict whether a proposed fusion will matter, by naming bytes.
- [ ] I can walk the host decision tree for any (format, S) pair.

**Next:** [06_kv_cache.md](06_kv_cache.md) — same Q4_0 packing, applied to a
tensor that grows every token.
