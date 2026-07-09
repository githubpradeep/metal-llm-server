#include <metal_stdlib>
using namespace metal;

// ─── SIMD-Group Matrix-Vector Multiply (f32 weights) ─────────────────────────
// Computes y = W * x where W is (M, K) f32 and x is (K,) f32, y is (M,) f32
// Uses SIMD groups (32 threads) to parallelize the dot product per row.

constant uint SIMD_SIZE = 32;

kernel void matvec(
    device const float* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (tgid >= M) return;
    
    uint row_offset = tgid * K;
    float acc = 0.0f;
    
    uint k = tid * 4;
    uint stride = SIMD_SIZE * 4;
    
    for (; k + 3 < K; k += stride) {
        float4 w = *reinterpret_cast<device const float4*>(&W[row_offset + k]);
        float4 xv = *reinterpret_cast<device const float4*>(&x[k]);
        acc += dot(w, xv);
    }
    
    for (uint kk = tid + (K / stride) * stride; kk < K; kk += SIMD_SIZE) {
        acc += W[row_offset + kk] * x[kk];
    }
    
    acc = simd_sum(acc);
    if (tid == 0) {
        y[tgid] = acc;
    }
}

// ─── SIMD-Group Matrix-Vector Multiply (f16 weights) ─────────────────────────
// Weights stored as half (2 bytes), activations as float.
// Reads half the memory bandwidth vs f32 weights.
// Accumulates in f32 for precision.

kernel void matvec_f16(
    device const half* W [[buffer(0)]],    // (M, K) row-major, half precision
    device const float* x [[buffer(1)]],   // (K,) f32
    device float* y [[buffer(2)]],         // (M,) f32
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (tgid >= M) return;
    
    uint row_offset = tgid * K;
    float acc = 0.0f;
    
    // Use half4 vectorized loads (8 bytes = 4 halfs at once)
    uint k = tid * 4;
    uint stride = SIMD_SIZE * 4;
    
    for (; k + 3 < K; k += stride) {
        half4 w = *reinterpret_cast<device const half4*>(&W[row_offset + k]);
        float4 xv = *reinterpret_cast<device const float4*>(&x[k]);
        // Convert half4 to float4 and dot
        acc += dot(float4(w), xv);
    }
    
    for (uint kk = tid + (K / stride) * stride; kk < K; kk += SIMD_SIZE) {
        acc += float(W[row_offset + kk]) * x[kk];
    }
    
    acc = simd_sum(acc);
    if (tid == 0) {
        y[tgid] = acc;
    }
}

// ─── Q4_0 Matrix-Vector Multiply ─────────────────────────────────────────────
// Weights quantized to 4-bit with group size 32 (Q4_0 format).
// Each group: [f16 scale][16 bytes packed 4-bit pairs] = 18 bytes per 32 weights.
// Values dequantized as: (nibble - 8) * scale
//
// Optimized: 2 SIMD groups per threadgroup, each handles 2 rows (4 rows total).
// x-vector loaded into registers and reused across rows for better bandwidth.

constant uint Q4_GROUP_SIZE = 32;
constant uint Q4_BLOCK_BYTES = 18;
constant uint N_ROWS_PER_TG = 4;  // rows per threadgroup
constant uint N_SIMDGROUPS = 2;   // SIMD groups per threadgroup

// Helper: compute dot product of one Q4_0 group (32 weights) with an x chunk.
// q points to 16 packed bytes (GGUF layout: byte i = low nibble elem i, high nibble elem i+16).
// The 32 x values are passed as 8 float4s: xv0-3 = elems 0-15, xv4-7 = elems 16-31.
inline float q4_dot_vec(device const uchar* q,
                        float4 xv0, float4 xv1, float4 xv2, float4 xv3,
                        float4 xv4, float4 xv5, float4 xv6, float4 xv7) {
    float local = 0.0f;

    packed_uchar4 q0 = *reinterpret_cast<device const packed_uchar4*>(q + 0);
    packed_uchar4 q1 = *reinterpret_cast<device const packed_uchar4*>(q + 4);
    packed_uchar4 q2 = *reinterpret_cast<device const packed_uchar4*>(q + 8);
    packed_uchar4 q3 = *reinterpret_cast<device const packed_uchar4*>(q + 12);

    local += float(int(q0[0] & 0xF) - 8) * xv0[0] + float(int(q0[0] >> 4) - 8) * xv4[0];
    local += float(int(q0[1] & 0xF) - 8) * xv0[1] + float(int(q0[1] >> 4) - 8) * xv4[1];
    local += float(int(q0[2] & 0xF) - 8) * xv0[2] + float(int(q0[2] >> 4) - 8) * xv4[2];
    local += float(int(q0[3] & 0xF) - 8) * xv0[3] + float(int(q0[3] >> 4) - 8) * xv4[3];

    local += float(int(q1[0] & 0xF) - 8) * xv1[0] + float(int(q1[0] >> 4) - 8) * xv5[0];
    local += float(int(q1[1] & 0xF) - 8) * xv1[1] + float(int(q1[1] >> 4) - 8) * xv5[1];
    local += float(int(q1[2] & 0xF) - 8) * xv1[2] + float(int(q1[2] >> 4) - 8) * xv5[2];
    local += float(int(q1[3] & 0xF) - 8) * xv1[3] + float(int(q1[3] >> 4) - 8) * xv5[3];

    local += float(int(q2[0] & 0xF) - 8) * xv2[0] + float(int(q2[0] >> 4) - 8) * xv6[0];
    local += float(int(q2[1] & 0xF) - 8) * xv2[1] + float(int(q2[1] >> 4) - 8) * xv6[1];
    local += float(int(q2[2] & 0xF) - 8) * xv2[2] + float(int(q2[2] >> 4) - 8) * xv6[2];
    local += float(int(q2[3] & 0xF) - 8) * xv2[3] + float(int(q2[3] >> 4) - 8) * xv6[3];

    local += float(int(q3[0] & 0xF) - 8) * xv3[0] + float(int(q3[0] >> 4) - 8) * xv7[0];
    local += float(int(q3[1] & 0xF) - 8) * xv3[1] + float(int(q3[1] >> 4) - 8) * xv7[1];
    local += float(int(q3[2] & 0xF) - 8) * xv3[2] + float(int(q3[2] >> 4) - 8) * xv7[2];
    local += float(int(q3[3] & 0xF) - 8) * xv3[3] + float(int(q3[3] >> 4) - 8) * xv7[3];

    return local;
}

kernel void matvec_q4(
    device const uchar* W_q4 [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint base_row = tgid * N_ROWS_PER_TG;
    // Each SIMD group handles 2 rows
    uint row0 = base_row + sgid * 2;
    uint row1 = row0 + 1;
    
    uint num_groups = K / Q4_GROUP_SIZE;
    uint row_bytes = num_groups * Q4_BLOCK_BYTES;
    
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    
    bool valid0 = row0 < M;
    bool valid1 = row1 < M;
    
    device const uchar* row0_ptr = W_q4 + row0 * row_bytes;
    device const uchar* row1_ptr = W_q4 + row1 * row_bytes;
    
    // Each lane processes groups strided by SIMD_SIZE
    for (uint g = lane; g < num_groups; g += SIMD_SIZE) {
        uint x_offset = g * Q4_GROUP_SIZE;
        
        // Load x-vector chunk into registers (shared across both rows)
        float4 xv0 = *reinterpret_cast<device const float4*>(&x[x_offset]);
        float4 xv1 = *reinterpret_cast<device const float4*>(&x[x_offset + 4]);
        float4 xv2 = *reinterpret_cast<device const float4*>(&x[x_offset + 8]);
        float4 xv3 = *reinterpret_cast<device const float4*>(&x[x_offset + 12]);
        float4 xv4 = *reinterpret_cast<device const float4*>(&x[x_offset + 16]);
        float4 xv5 = *reinterpret_cast<device const float4*>(&x[x_offset + 20]);
        float4 xv6 = *reinterpret_cast<device const float4*>(&x[x_offset + 24]);
        float4 xv7 = *reinterpret_cast<device const float4*>(&x[x_offset + 28]);
        
        // Process row0
        if (valid0) {
            uint block_offset = g * Q4_BLOCK_BYTES;
            float scale = float(*reinterpret_cast<device const half*>(&row0_ptr[block_offset]));
            device const uchar* q = &row0_ptr[block_offset + 2];
            acc0 += q4_dot_vec(q, xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7) * scale;
        }

        // Process row1 (reuses same x-vector registers)
        if (valid1) {
            uint block_offset = g * Q4_BLOCK_BYTES;
            float scale = float(*reinterpret_cast<device const half*>(&row1_ptr[block_offset]));
            device const uchar* q = &row1_ptr[block_offset + 2];
            acc1 += q4_dot_vec(q, xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7) * scale;
        }
    }

    // Reduce within SIMD group
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    
    if (lane == 0) {
        if (valid0) y[row0] = acc0;
        if (valid1) y[row1] = acc1;
    }
}

// ─── Q4_0 Matrix-Vector Multiply (vectorized dequant, high occupancy) ───────
// Faster replacement for matvec_q4 used on the batch-1 decode path.
//
// Two changes vs matvec_q4:
//   1) Vectorized 4-bit unpack. Instead of 32 scalar nibble extracts per group,
//      each 4-byte chunk (8 weights) is unpacked into two float4 via vector
//      int ops and dotted against deinterleaved x. This cuts the dequant ALU
//      work that was making the kernel compute-bound.
//   2) Larger threadgroups (8 SIMD groups = 256 threads) with each SIMD group
//      owning Q4F_ROWS output rows. More threads/threadgroup keeps more device
//      memory requests in flight (better latency hiding => higher bandwidth),
//      and x is reused from registers across the rows a SIMD group owns.
//
// Q4_0 layout per 32-weight group (GGUF): [f16 scale][16 bytes].
// Byte i: low nibble = elem i, high nibble = elem i+16.

constant uint Q4F_SG   = 8;   // SIMD groups per threadgroup
constant uint Q4F_ROWS = 4;   // output rows per SIMD group

inline float q4_dot_vec_fast(device const uchar* q,
                             float4 xv0, float4 xv1, float4 xv2, float4 xv3,
                             float4 xv4, float4 xv5, float4 xv6, float4 xv7) {
    float4 acc4 = float4(0.0f);

    // chunk 0: bytes 0-3 -> x[0..3], x[16..19]
    {
        packed_uchar4 qp = *reinterpret_cast<device const packed_uchar4*>(q + 0);
        uint4 qi = uint4(qp[0], qp[1], qp[2], qp[3]);
        float4 flo = float4(int4(qi & 0xF) - 8);
        float4 fhi = float4(int4(qi >> 4) - 8);
        acc4 += flo * xv0 + fhi * xv4;
    }
    // chunk 1: bytes 4-7 -> x[4..7], x[20..23]
    {
        packed_uchar4 qp = *reinterpret_cast<device const packed_uchar4*>(q + 4);
        uint4 qi = uint4(qp[0], qp[1], qp[2], qp[3]);
        float4 flo = float4(int4(qi & 0xF) - 8);
        float4 fhi = float4(int4(qi >> 4) - 8);
        acc4 += flo * xv1 + fhi * xv5;
    }
    // chunk 2: bytes 8-11 -> x[8..11], x[24..27]
    {
        packed_uchar4 qp = *reinterpret_cast<device const packed_uchar4*>(q + 8);
        uint4 qi = uint4(qp[0], qp[1], qp[2], qp[3]);
        float4 flo = float4(int4(qi & 0xF) - 8);
        float4 fhi = float4(int4(qi >> 4) - 8);
        acc4 += flo * xv2 + fhi * xv6;
    }
    // chunk 3: bytes 12-15 -> x[12..15], x[28..31]
    {
        packed_uchar4 qp = *reinterpret_cast<device const packed_uchar4*>(q + 12);
        uint4 qi = uint4(qp[0], qp[1], qp[2], qp[3]);
        float4 flo = float4(int4(qi & 0xF) - 8);
        float4 fhi = float4(int4(qi >> 4) - 8);
        acc4 += flo * xv3 + fhi * xv7;
    }

    return acc4.x + acc4.y + acc4.z + acc4.w;
}

// Templated body: each SIMD group owns ROWS output rows; SG_PER_TG SIMD groups
// per threadgroup. Lowering ROWS increases the number of independent row-workers
// (SIMD groups) in flight and cuts per-thread register pressure, which raises
// occupancy and memory-level parallelism for the small/medium M matvecs that
// dominate decode. Higher ROWS amortizes the x loads across rows (cheap, since x
// is tiny and cached) but starves the GPU when M is small. The right value is
// shape-dependent, so we expose ROWS=1/2/4 and select per the MATVEC_ROWS env.
template <uint ROWS>
inline void matvec_q4_fast_body(
    device const uchar* W_q4,
    device const float* x,
    device float* y,
    uint M,
    uint K,
    uint tgid,
    uint sgid,
    uint lane,
    uint sg_per_tg
) {
    uint base_row = tgid * (sg_per_tg * ROWS) + sgid * ROWS;
    uint num_groups = K / Q4_GROUP_SIZE;
    uint row_bytes = num_groups * Q4_BLOCK_BYTES;

    float acc[ROWS];
    bool valid[ROWS];
    device const uchar* rptr[ROWS];
    for (uint r = 0; r < ROWS; ++r) {
        acc[r] = 0.0f;
        uint row = base_row + r;
        valid[r] = row < M;
        rptr[r] = W_q4 + row * row_bytes;
    }

    for (uint g = lane; g < num_groups; g += SIMD_SIZE) {
        uint xo = g * Q4_GROUP_SIZE;
        float4 xv0 = *reinterpret_cast<device const float4*>(&x[xo]);
        float4 xv1 = *reinterpret_cast<device const float4*>(&x[xo + 4]);
        float4 xv2 = *reinterpret_cast<device const float4*>(&x[xo + 8]);
        float4 xv3 = *reinterpret_cast<device const float4*>(&x[xo + 12]);
        float4 xv4 = *reinterpret_cast<device const float4*>(&x[xo + 16]);
        float4 xv5 = *reinterpret_cast<device const float4*>(&x[xo + 20]);
        float4 xv6 = *reinterpret_cast<device const float4*>(&x[xo + 24]);
        float4 xv7 = *reinterpret_cast<device const float4*>(&x[xo + 28]);

        uint bo = g * Q4_BLOCK_BYTES;
        for (uint r = 0; r < ROWS; ++r) {
            if (!valid[r]) continue;
            float scale = float(*reinterpret_cast<device const half*>(&rptr[r][bo]));
            acc[r] += q4_dot_vec_fast(&rptr[r][bo + 2],
                                      xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7) * scale;
        }
    }

    for (uint r = 0; r < ROWS; ++r) {
        float s = simd_sum(acc[r]);
        if (lane == 0 && valid[r]) {
            y[base_row + r] = s;
        }
    }
}

// ROWS=4: 32 rows/threadgroup (current default geometry).
kernel void matvec_q4_fast(
    device const uchar* W_q4 [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    matvec_q4_fast_body<4>(W_q4, x, y, M, K, tgid, sgid, lane, Q4F_SG);
}

// Dual Q4 matvec: y0 = W0 @ x, y1 = W1 @ x with a single load of x per K block.
// Used for gate+up and k+v pairs that share the same input during decode.
template <uint ROWS>
inline void matvec_q4_dual_body(
    device const uchar* W0,
    device const uchar* W1,
    device const float* x,
    device float* y0,
    device float* y1,
    uint M,
    uint K,
    uint tgid,
    uint sgid,
    uint lane,
    uint sg_per_tg
) {
    uint base_row = tgid * (sg_per_tg * ROWS) + sgid * ROWS;
    uint num_groups = K / Q4_GROUP_SIZE;
    uint row_bytes = num_groups * Q4_BLOCK_BYTES;

    float acc0[ROWS];
    float acc1[ROWS];
    bool valid[ROWS];
    device const uchar* rptr0[ROWS];
    device const uchar* rptr1[ROWS];
    for (uint r = 0; r < ROWS; ++r) {
        acc0[r] = 0.0f;
        acc1[r] = 0.0f;
        uint row = base_row + r;
        valid[r] = row < M;
        rptr0[r] = W0 + row * row_bytes;
        rptr1[r] = W1 + row * row_bytes;
    }

    for (uint g = lane; g < num_groups; g += SIMD_SIZE) {
        uint xo = g * Q4_GROUP_SIZE;
        float4 xv0 = *reinterpret_cast<device const float4*>(&x[xo]);
        float4 xv1 = *reinterpret_cast<device const float4*>(&x[xo + 4]);
        float4 xv2 = *reinterpret_cast<device const float4*>(&x[xo + 8]);
        float4 xv3 = *reinterpret_cast<device const float4*>(&x[xo + 12]);
        float4 xv4 = *reinterpret_cast<device const float4*>(&x[xo + 16]);
        float4 xv5 = *reinterpret_cast<device const float4*>(&x[xo + 20]);
        float4 xv6 = *reinterpret_cast<device const float4*>(&x[xo + 24]);
        float4 xv7 = *reinterpret_cast<device const float4*>(&x[xo + 28]);

        uint bo = g * Q4_BLOCK_BYTES;
        for (uint r = 0; r < ROWS; ++r) {
            if (!valid[r]) continue;
            float scale0 = float(*reinterpret_cast<device const half*>(&rptr0[r][bo]));
            acc0[r] += q4_dot_vec_fast(&rptr0[r][bo + 2],
                                         xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7) * scale0;
            float scale1 = float(*reinterpret_cast<device const half*>(&rptr1[r][bo]));
            acc1[r] += q4_dot_vec_fast(&rptr1[r][bo + 2],
                                         xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7) * scale1;
        }
    }

    for (uint r = 0; r < ROWS; ++r) {
        float s0 = simd_sum(acc0[r]);
        float s1 = simd_sum(acc1[r]);
        if (lane == 0 && valid[r]) {
            y0[base_row + r] = s0;
            y1[base_row + r] = s1;
        }
    }
}

kernel void matvec_q4_dual(
    device const uchar* W0 [[buffer(0)]],
    device const uchar* W1 [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* y0 [[buffer(3)]],
    device float* y1 [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    matvec_q4_dual_body<4>(W0, W1, x, y0, y1, M, K, tgid, sgid, lane, Q4F_SG);
}

inline float gelu_pytorch_tanh(float x) {
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    inner = clamp(inner, -10.0f, 10.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}

// gate+up Q4 matvec fused with GeLU(gate)*up — skips gate/up buffer writes.
template <uint ROWS>
inline void matvec_q4_dual_gelu_body(
    device const uchar* W0,
    device const uchar* W1,
    device const float* x,
    device float* y,
    uint M,
    uint K,
    uint tgid,
    uint sgid,
    uint lane,
    uint sg_per_tg
) {
    uint base_row = tgid * (sg_per_tg * ROWS) + sgid * ROWS;
    uint num_groups = K / Q4_GROUP_SIZE;
    uint row_bytes = num_groups * Q4_BLOCK_BYTES;

    float acc0[ROWS];
    float acc1[ROWS];
    bool valid[ROWS];
    device const uchar* rptr0[ROWS];
    device const uchar* rptr1[ROWS];
    for (uint r = 0; r < ROWS; ++r) {
        acc0[r] = 0.0f;
        acc1[r] = 0.0f;
        uint row = base_row + r;
        valid[r] = row < M;
        rptr0[r] = W0 + row * row_bytes;
        rptr1[r] = W1 + row * row_bytes;
    }

    for (uint g = lane; g < num_groups; g += SIMD_SIZE) {
        uint xo = g * Q4_GROUP_SIZE;
        float4 xv0 = *reinterpret_cast<device const float4*>(&x[xo]);
        float4 xv1 = *reinterpret_cast<device const float4*>(&x[xo + 4]);
        float4 xv2 = *reinterpret_cast<device const float4*>(&x[xo + 8]);
        float4 xv3 = *reinterpret_cast<device const float4*>(&x[xo + 12]);
        float4 xv4 = *reinterpret_cast<device const float4*>(&x[xo + 16]);
        float4 xv5 = *reinterpret_cast<device const float4*>(&x[xo + 20]);
        float4 xv6 = *reinterpret_cast<device const float4*>(&x[xo + 24]);
        float4 xv7 = *reinterpret_cast<device const float4*>(&x[xo + 28]);

        uint bo = g * Q4_BLOCK_BYTES;
        for (uint r = 0; r < ROWS; ++r) {
            if (!valid[r]) continue;
            float scale0 = float(*reinterpret_cast<device const half*>(&rptr0[r][bo]));
            acc0[r] += q4_dot_vec_fast(&rptr0[r][bo + 2],
                                         xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7) * scale0;
            float scale1 = float(*reinterpret_cast<device const half*>(&rptr1[r][bo]));
            acc1[r] += q4_dot_vec_fast(&rptr1[r][bo + 2],
                                         xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7) * scale1;
        }
    }

    for (uint r = 0; r < ROWS; ++r) {
        float s0 = simd_sum(acc0[r]);
        float s1 = simd_sum(acc1[r]);
        if (lane == 0 && valid[r]) {
            y[base_row + r] = gelu_pytorch_tanh(s0) * s1;
        }
    }
}

kernel void matvec_q4_dual_gelu(
    device const uchar* W0 [[buffer(0)]],
    device const uchar* W1 [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    matvec_q4_dual_gelu_body<4>(W0, W1, x, y, M, K, tgid, sgid, lane, Q4F_SG);
}

// gate+up rows interleaved as [gate_i][up_i] per intermediate index — better
// cache locality than separate gate/up tensors (uzu-style packed up_projection).
template <uint ROWS>
inline void matvec_q4_interleaved_gelu_body(
    device const uchar* W,
    device const float* x,
    device float* y,
    uint M,
    uint K,
    uint tgid,
    uint sgid,
    uint lane,
    uint sg_per_tg
) {
    uint base_row = tgid * (sg_per_tg * ROWS) + sgid * ROWS;
    uint num_groups = K / Q4_GROUP_SIZE;
    uint row_bytes = num_groups * Q4_BLOCK_BYTES;
    uint pair_bytes = row_bytes * 2;

    float acc0[ROWS];
    float acc1[ROWS];
    bool valid[ROWS];
    device const uchar* gptr[ROWS];
    device const uchar* uptr[ROWS];
    for (uint r = 0; r < ROWS; ++r) {
        acc0[r] = 0.0f;
        acc1[r] = 0.0f;
        uint row = base_row + r;
        valid[r] = row < M;
        device const uchar* pair = W + row * pair_bytes;
        gptr[r] = pair;
        uptr[r] = pair + row_bytes;
    }

    for (uint g = lane; g < num_groups; g += SIMD_SIZE) {
        uint xo = g * Q4_GROUP_SIZE;
        float4 xv0 = *reinterpret_cast<device const float4*>(&x[xo]);
        float4 xv1 = *reinterpret_cast<device const float4*>(&x[xo + 4]);
        float4 xv2 = *reinterpret_cast<device const float4*>(&x[xo + 8]);
        float4 xv3 = *reinterpret_cast<device const float4*>(&x[xo + 12]);
        float4 xv4 = *reinterpret_cast<device const float4*>(&x[xo + 16]);
        float4 xv5 = *reinterpret_cast<device const float4*>(&x[xo + 20]);
        float4 xv6 = *reinterpret_cast<device const float4*>(&x[xo + 24]);
        float4 xv7 = *reinterpret_cast<device const float4*>(&x[xo + 28]);

        uint bo = g * Q4_BLOCK_BYTES;
        for (uint r = 0; r < ROWS; ++r) {
            if (!valid[r]) continue;
            float scale0 = float(*reinterpret_cast<device const half*>(&gptr[r][bo]));
            acc0[r] += q4_dot_vec_fast(&gptr[r][bo + 2],
                                        xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7) * scale0;
            float scale1 = float(*reinterpret_cast<device const half*>(&uptr[r][bo]));
            acc1[r] += q4_dot_vec_fast(&uptr[r][bo + 2],
                                       xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7) * scale1;
        }
    }

    for (uint r = 0; r < ROWS; ++r) {
        float s0 = simd_sum(acc0[r]);
        float s1 = simd_sum(acc1[r]);
        if (lane == 0 && valid[r]) {
            y[base_row + r] = gelu_pytorch_tanh(s0) * s1;
        }
    }
}

kernel void matvec_q4_interleaved_gelu(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    matvec_q4_interleaved_gelu_body<4>(W, x, y, M, K, tgid, sgid, lane, Q4F_SG);
}

// ─── Fused pre-attn RMSNorm + Q4 Q/K/V projections (decode) ─────────────────
// inv_rms is computed once via rmsnorm_inv_rms; these kernels matvec with
// x[i] = hidden[i] * weight[i] * inv_rms without a normed scratch buffer.

template <uint ROWS>
inline void matvec_q4_rmsnorm_hidden_body(
    device const uchar* W_q4,
    device const float* hidden,
    device const float* weight,
    float inv_rms,
    device float* y,
    uint M,
    uint K,
    uint tgid,
    uint sgid,
    uint lane,
    uint sg_per_tg
) {
    uint base_row = tgid * (sg_per_tg * ROWS) + sgid * ROWS;
    uint num_groups = K / Q4_GROUP_SIZE;
    uint row_bytes = num_groups * Q4_BLOCK_BYTES;

    float acc[ROWS];
    bool valid[ROWS];
    device const uchar* rptr[ROWS];
    for (uint r = 0; r < ROWS; ++r) {
        acc[r] = 0.0f;
        uint row = base_row + r;
        valid[r] = row < M;
        rptr[r] = W_q4 + row * row_bytes;
    }

    for (uint g = lane; g < num_groups; g += SIMD_SIZE) {
        uint xo = g * Q4_GROUP_SIZE;
        float4 h0 = *reinterpret_cast<device const float4*>(&hidden[xo]);
        float4 h1 = *reinterpret_cast<device const float4*>(&hidden[xo + 4]);
        float4 h2 = *reinterpret_cast<device const float4*>(&hidden[xo + 8]);
        float4 h3 = *reinterpret_cast<device const float4*>(&hidden[xo + 12]);
        float4 h4 = *reinterpret_cast<device const float4*>(&hidden[xo + 16]);
        float4 h5 = *reinterpret_cast<device const float4*>(&hidden[xo + 20]);
        float4 h6 = *reinterpret_cast<device const float4*>(&hidden[xo + 24]);
        float4 h7 = *reinterpret_cast<device const float4*>(&hidden[xo + 28]);
        float4 w0 = *reinterpret_cast<device const float4*>(&weight[xo]);
        float4 w1 = *reinterpret_cast<device const float4*>(&weight[xo + 4]);
        float4 w2 = *reinterpret_cast<device const float4*>(&weight[xo + 8]);
        float4 w3 = *reinterpret_cast<device const float4*>(&weight[xo + 12]);
        float4 w4 = *reinterpret_cast<device const float4*>(&weight[xo + 16]);
        float4 w5 = *reinterpret_cast<device const float4*>(&weight[xo + 20]);
        float4 w6 = *reinterpret_cast<device const float4*>(&weight[xo + 24]);
        float4 w7 = *reinterpret_cast<device const float4*>(&weight[xo + 28]);
        float4 xv0 = h0 * w0 * inv_rms;
        float4 xv1 = h1 * w1 * inv_rms;
        float4 xv2 = h2 * w2 * inv_rms;
        float4 xv3 = h3 * w3 * inv_rms;
        float4 xv4 = h4 * w4 * inv_rms;
        float4 xv5 = h5 * w5 * inv_rms;
        float4 xv6 = h6 * w6 * inv_rms;
        float4 xv7 = h7 * w7 * inv_rms;

        uint bo = g * Q4_BLOCK_BYTES;
        for (uint r = 0; r < ROWS; ++r) {
            if (!valid[r]) continue;
            float scale = float(*reinterpret_cast<device const half*>(&rptr[r][bo]));
            acc[r] += q4_dot_vec_fast(&rptr[r][bo + 2],
                                      xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7) * scale;
        }
    }

    for (uint r = 0; r < ROWS; ++r) {
        float s = simd_sum(acc[r]);
        if (lane == 0 && valid[r]) {
            y[base_row + r] = s;
        }
    }
}

kernel void matvec_q_rmsnorm_inv_q4(
    device const float* hidden [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* inv_rms_ptr [[buffer(2)]],
    device const uchar* Wq [[buffer(3)]],
    device float* q_out [[buffer(4)]],
    constant uint& M_q [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    float inv_rms = *inv_rms_ptr;
    matvec_q4_rmsnorm_hidden_body<4>(
        Wq, hidden, weight, inv_rms, q_out, M_q, K, tgid, sgid, lane, Q4F_SG);
}

kernel void matvec_qkv_rmsnorm_inv_q4(
    device const float* hidden [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* inv_rms_ptr [[buffer(2)]],
    device const uchar* Wq [[buffer(3)]],
    device const uchar* Wk [[buffer(4)]],
    device const uchar* Wv [[buffer(5)]],
    device float* q_out [[buffer(6)]],
    device float* k_out [[buffer(7)]],
    device float* v_out [[buffer(8)]],
    constant uint& M_q [[buffer(9)]],
    constant uint& M_kv [[buffer(10)]],
    constant uint& K [[buffer(11)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    float inv_rms = *inv_rms_ptr;
    uint rows_per_tg = Q4F_SG * 4u;
    uint q_tgs = (M_q + rows_per_tg - 1u) / rows_per_tg;
    uint kv_tgs = (M_kv + rows_per_tg - 1u) / rows_per_tg;

    if (tgid < q_tgs) {
        matvec_q4_rmsnorm_hidden_body<4>(
            Wq, hidden, weight, inv_rms, q_out, M_q, K, tgid, sgid, lane, Q4F_SG);
    } else if (tgid < q_tgs + kv_tgs) {
        matvec_q4_rmsnorm_hidden_body<4>(
            Wk, hidden, weight, inv_rms, k_out, M_kv, K, tgid - q_tgs, sgid, lane, Q4F_SG);
    } else if (tgid < q_tgs + 2u * kv_tgs) {
        matvec_q4_rmsnorm_hidden_body<4>(
            Wv, hidden, weight, inv_rms, v_out, M_kv, K, tgid - q_tgs - kv_tgs, sgid, lane, Q4F_SG);
    }
}

// Writes only 1/inv_rms — avoids materializing the full normed hidden vector.
kernel void rmsnorm_inv_rms(
    device const float* x [[buffer(0)]],
    device float* inv_rms_out [[buffer(1)]],
    constant uint& dim [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];

    float partial_sum = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = x[i];
        partial_sum += val * val;
    }
    shared_sum[tid] = partial_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        *inv_rms_out = rsqrt(shared_sum[0] / float(dim) + eps);
    }
}

// Interleaved gate∥up+GeLU reading hidden + RMS weight + inv_rms (no normed scratch).
template <uint ROWS>
inline void matvec_q4_interleaved_gelu_hidden_body(
    device const uchar* W,
    device const float* hidden,
    device const float* weight,
    device const float* inv_rms_ptr,
    device float* y,
    uint M,
    uint K,
    uint tgid,
    uint sgid,
    uint lane,
    uint sg_per_tg
) {
    float inv_rms = *inv_rms_ptr;
    uint base_row = tgid * (sg_per_tg * ROWS) + sgid * ROWS;
    uint num_groups = K / Q4_GROUP_SIZE;
    uint row_bytes = num_groups * Q4_BLOCK_BYTES;
    uint pair_bytes = row_bytes * 2;

    float acc0[ROWS];
    float acc1[ROWS];
    bool valid[ROWS];
    device const uchar* gptr[ROWS];
    device const uchar* uptr[ROWS];
    for (uint r = 0; r < ROWS; ++r) {
        acc0[r] = 0.0f;
        acc1[r] = 0.0f;
        uint row = base_row + r;
        valid[r] = row < M;
        device const uchar* pair = W + row * pair_bytes;
        gptr[r] = pair;
        uptr[r] = pair + row_bytes;
    }

    for (uint g = lane; g < num_groups; g += SIMD_SIZE) {
        uint xo = g * Q4_GROUP_SIZE;
        float4 h0 = *reinterpret_cast<device const float4*>(&hidden[xo]);
        float4 h1 = *reinterpret_cast<device const float4*>(&hidden[xo + 4]);
        float4 h2 = *reinterpret_cast<device const float4*>(&hidden[xo + 8]);
        float4 h3 = *reinterpret_cast<device const float4*>(&hidden[xo + 12]);
        float4 h4 = *reinterpret_cast<device const float4*>(&hidden[xo + 16]);
        float4 h5 = *reinterpret_cast<device const float4*>(&hidden[xo + 20]);
        float4 h6 = *reinterpret_cast<device const float4*>(&hidden[xo + 24]);
        float4 h7 = *reinterpret_cast<device const float4*>(&hidden[xo + 28]);
        float4 w0 = *reinterpret_cast<device const float4*>(&weight[xo]);
        float4 w1 = *reinterpret_cast<device const float4*>(&weight[xo + 4]);
        float4 w2 = *reinterpret_cast<device const float4*>(&weight[xo + 8]);
        float4 w3 = *reinterpret_cast<device const float4*>(&weight[xo + 12]);
        float4 w4 = *reinterpret_cast<device const float4*>(&weight[xo + 16]);
        float4 w5 = *reinterpret_cast<device const float4*>(&weight[xo + 20]);
        float4 w6 = *reinterpret_cast<device const float4*>(&weight[xo + 24]);
        float4 w7 = *reinterpret_cast<device const float4*>(&weight[xo + 28]);
        float4 xv0 = h0 * w0 * inv_rms;
        float4 xv1 = h1 * w1 * inv_rms;
        float4 xv2 = h2 * w2 * inv_rms;
        float4 xv3 = h3 * w3 * inv_rms;
        float4 xv4 = h4 * w4 * inv_rms;
        float4 xv5 = h5 * w5 * inv_rms;
        float4 xv6 = h6 * w6 * inv_rms;
        float4 xv7 = h7 * w7 * inv_rms;

        uint bo = g * Q4_BLOCK_BYTES;
        for (uint r = 0; r < ROWS; ++r) {
            if (!valid[r]) continue;
            float scale0 = float(*reinterpret_cast<device const half*>(&gptr[r][bo]));
            acc0[r] += q4_dot_vec_fast(&gptr[r][bo + 2],
                                        xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7) * scale0;
            float scale1 = float(*reinterpret_cast<device const half*>(&uptr[r][bo]));
            acc1[r] += q4_dot_vec_fast(&uptr[r][bo + 2],
                                       xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7) * scale1;
        }
    }

    for (uint r = 0; r < ROWS; ++r) {
        float s0 = simd_sum(acc0[r]);
        float s1 = simd_sum(acc1[r]);
        if (lane == 0 && valid[r]) {
            y[base_row + r] = gelu_pytorch_tanh(s0) * s1;
        }
    }
}

kernel void matvec_q4_interleaved_gelu_hidden(
    device const uchar* W [[buffer(0)]],
    device const float* hidden [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const float* inv_rms [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    matvec_q4_interleaved_gelu_hidden_body<4>(
        W, hidden, weight, inv_rms, y, M, K, tgid, sgid, lane, Q4F_SG);
}

// gate∥up+GeLU → f16 scratch (half activation bandwidth gate→down).
template <uint ROWS>
inline void matvec_q4_interleaved_gelu_f16_body(
    device const uchar* W,
    device const float* x,
    device half* y,
    uint M,
    uint K,
    uint tgid,
    uint sgid,
    uint lane,
    uint sg_per_tg
) {
    uint base_row = tgid * (sg_per_tg * ROWS) + sgid * ROWS;
    uint num_groups = K / Q4_GROUP_SIZE;
    uint row_bytes = num_groups * Q4_BLOCK_BYTES;
    uint pair_bytes = row_bytes * 2;

    float acc0[ROWS];
    float acc1[ROWS];
    bool valid[ROWS];
    device const uchar* gptr[ROWS];
    device const uchar* uptr[ROWS];
    for (uint r = 0; r < ROWS; ++r) {
        acc0[r] = 0.0f;
        acc1[r] = 0.0f;
        uint row = base_row + r;
        valid[r] = row < M;
        device const uchar* pair = W + row * pair_bytes;
        gptr[r] = pair;
        uptr[r] = pair + row_bytes;
    }

    for (uint g = lane; g < num_groups; g += SIMD_SIZE) {
        uint xo = g * Q4_GROUP_SIZE;
        float4 xv0 = *reinterpret_cast<device const float4*>(&x[xo]);
        float4 xv1 = *reinterpret_cast<device const float4*>(&x[xo + 4]);
        float4 xv2 = *reinterpret_cast<device const float4*>(&x[xo + 8]);
        float4 xv3 = *reinterpret_cast<device const float4*>(&x[xo + 12]);
        float4 xv4 = *reinterpret_cast<device const float4*>(&x[xo + 16]);
        float4 xv5 = *reinterpret_cast<device const float4*>(&x[xo + 20]);
        float4 xv6 = *reinterpret_cast<device const float4*>(&x[xo + 24]);
        float4 xv7 = *reinterpret_cast<device const float4*>(&x[xo + 28]);

        uint bo = g * Q4_BLOCK_BYTES;
        for (uint r = 0; r < ROWS; ++r) {
            if (!valid[r]) continue;
            float scale0 = float(*reinterpret_cast<device const half*>(&gptr[r][bo]));
            acc0[r] += q4_dot_vec_fast(&gptr[r][bo + 2],
                                        xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7) * scale0;
            float scale1 = float(*reinterpret_cast<device const half*>(&uptr[r][bo]));
            acc1[r] += q4_dot_vec_fast(&uptr[r][bo + 2],
                                       xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7) * scale1;
        }
    }

    for (uint r = 0; r < ROWS; ++r) {
        float s0 = simd_sum(acc0[r]);
        float s1 = simd_sum(acc1[r]);
        if (lane == 0 && valid[r]) {
            y[base_row + r] = half(gelu_pytorch_tanh(s0) * s1);
        }
    }
}

kernel void matvec_q4_interleaved_gelu_f16(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device half* y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    matvec_q4_interleaved_gelu_f16_body<4>(W, x, y, M, K, tgid, sgid, lane, Q4F_SG);
}

// PLE decode: gate Q4 matvec + GeLU(gate)*context slice in one dispatch.
template <uint ROWS>
inline void matvec_q4_gelu_mul_body(
    device const uchar* W,
    device const float* x,
    device const float* context,
    device float* y,
    uint M,
    uint K,
    uint tgid,
    uint sgid,
    uint lane,
    uint sg_per_tg
) {
    uint base_row = tgid * (sg_per_tg * ROWS) + sgid * ROWS;
    uint num_groups = K / Q4_GROUP_SIZE;
    uint row_bytes = num_groups * Q4_BLOCK_BYTES;

    float acc[ROWS];
    bool valid[ROWS];
    device const uchar* rptr[ROWS];
    for (uint r = 0; r < ROWS; ++r) {
        acc[r] = 0.0f;
        uint row = base_row + r;
        valid[r] = row < M;
        rptr[r] = W + row * row_bytes;
    }

    for (uint g = lane; g < num_groups; g += SIMD_SIZE) {
        uint xo = g * Q4_GROUP_SIZE;
        float4 xv0 = *reinterpret_cast<device const float4*>(&x[xo]);
        float4 xv1 = *reinterpret_cast<device const float4*>(&x[xo + 4]);
        float4 xv2 = *reinterpret_cast<device const float4*>(&x[xo + 8]);
        float4 xv3 = *reinterpret_cast<device const float4*>(&x[xo + 12]);
        float4 xv4 = *reinterpret_cast<device const float4*>(&x[xo + 16]);
        float4 xv5 = *reinterpret_cast<device const float4*>(&x[xo + 20]);
        float4 xv6 = *reinterpret_cast<device const float4*>(&x[xo + 24]);
        float4 xv7 = *reinterpret_cast<device const float4*>(&x[xo + 28]);

        uint bo = g * Q4_BLOCK_BYTES;
        for (uint r = 0; r < ROWS; ++r) {
            if (!valid[r]) continue;
            float scale = float(*reinterpret_cast<device const half*>(&rptr[r][bo]));
            acc[r] += q4_dot_vec_fast(&rptr[r][bo + 2],
                                      xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7) * scale;
        }
    }

    for (uint r = 0; r < ROWS; ++r) {
        float s = simd_sum(acc[r]);
        if (lane == 0 && valid[r]) {
            uint row = base_row + r;
            y[row] = gelu_pytorch_tanh(s) * context[row];
        }
    }
}

kernel void ple_matvec_gelu_q4(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* context [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    matvec_q4_gelu_mul_body<4>(W, x, context, y, M, K, tgid, sgid, lane, Q4F_SG);
}

// ROWS=1: 8 rows/threadgroup → 4× more SIMD groups in flight than ROWS=4.
kernel void matvec_q4_r1(
    device const uchar* W_q4 [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    matvec_q4_fast_body<1>(W_q4, x, y, M, K, tgid, sgid, lane, Q4F_SG);
}

// ROWS=2: 16 rows/threadgroup → 2× more SIMD groups in flight than ROWS=4.
kernel void matvec_q4_r2(
    device const uchar* W_q4 [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    matvec_q4_fast_body<2>(W_q4, x, y, M, K, tgid, sgid, lane, Q4F_SG);
}

// ROWS=8: 64 rows/threadgroup. Each thread carries 8 independent row
// accumulators → more in-flight loads per thread (ILP) to hide memory latency,
// which is the limiter here (kernel sits at ~67% of M1 Pro's 200 GB/s peak and
// scales up with ROWS). Costs registers; benchmark to confirm it doesn't spill.
kernel void matvec_q4_r8(
    device const uchar* W_q4 [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    matvec_q4_fast_body<8>(W_q4, x, y, M, K, tgid, sgid, lane, Q4F_SG);
}

// ─── llama.cpp / ane-infer Q4 GEMV (GGUF layout) ────────────────────────────

constant uint LC4_SG = 4u;
constant uint LC4_NR0 = 2u;
constant uint LC4_NQ = 8u;

inline float q4_dequant_elem(device const uchar* qs, int elem) {
    int byte_idx = elem & 15;
    int is_high = (elem >> 4) & 1;
    uchar byte = qs[byte_idx];
    int nibble = is_high ? (byte >> 4) : (byte & 0x0F);
    return float(nibble) - 8.0f;
}

kernel void matvec_q4_lc(
    device const uchar* W_q4 [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int nb = int(K / Q4_GROUP_SIZE);
    const int r0 = int(tg_id) * int(LC4_NR0);
    if (r0 >= int(M)) return;

    float sumf[LC4_NR0] = { 0.0f, 0.0f };
    const short ix = tiisg / (32 / LC4_NQ);
    const short il = tiisg % (32 / LC4_NQ);
    const int ib0 = int(sgitg) * int(LC4_NQ) + int(ix);
    const uint row_bytes = uint(nb) * Q4_BLOCK_BYTES;

    device const uchar* w_rows[LC4_NR0];
    for (short row = 0; row < LC4_NR0; ++row) {
        const int row_idx = r0 + row;
        if (row_idx < int(M)) {
            w_rows[row] = W_q4 + row_idx * row_bytes;
        }
    }

    float yl[LC4_NQ];
    for (int ib = ib0; ib < nb; ib += int(LC4_SG) * int(LC4_NQ)) {
        device const float* yb = x + ib * Q4_GROUP_SIZE + il * LC4_NQ;
        for (short i = 0; i < LC4_NQ; ++i) {
            yl[i] = yb[i];
        }
        for (short row = 0; row < LC4_NR0; ++row) {
            if (r0 + row >= int(M)) continue;
            const uint bo = uint(ib) * Q4_BLOCK_BYTES;
            const float scale = float(*reinterpret_cast<device const half*>(&w_rows[row][bo]));
            device const uchar* qs = w_rows[row] + bo + 2;
            float sumq = 0.0f;
            for (short i = 0; i < LC4_NQ; ++i) {
                const int elem = il * LC4_NQ + i;
                sumq += q4_dequant_elem(qs, elem) * yl[i];
            }
            sumf[row] += sumq * scale;
        }
    }

    threadgroup float shmem[LC4_NR0][LC4_SG];
    for (short row = 0; row < LC4_NR0; ++row) {
        sumf[row] = simd_sum(sumf[row]);
        if (tiisg == 0) {
            shmem[row][sgitg] = sumf[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg == 0) {
        for (short row = 0; row < LC4_NR0; ++row) {
            if (r0 + row >= int(M)) continue;
            float total = 0.0f;
            for (short s = 0; s < LC4_SG; ++s) {
                total += shmem[row][s];
            }
            y[r0 + row] = total;
        }
    }
}

// ─── Split-K Q4_0 Matrix-Vector Multiply ────────────────────────────────────
// One threadgroup (Q4F_SG SIMD groups = 256 threads) cooperates on a SINGLE
// output row, splitting the K reduction across all 256 threads and combining
// the partials in threadgroup memory. This keeps the GPU full when M is small
// but K is large (e.g. down_proj: M=hidden, K=intermediate), where the
// row-parallel kernels above launch too few threadgroups. Useless for small K
// (most threads idle), so it is selected per-shape, not unconditionally.
kernel void matvec_q4_splitk(
    device const uchar* W_q4 [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint nthreads [[threads_per_threadgroup]]
) {
    uint row = tgid;
    if (row >= M) return;

    uint num_groups = K / Q4_GROUP_SIZE;
    uint row_bytes = num_groups * Q4_BLOCK_BYTES;
    device const uchar* rptr = W_q4 + row * row_bytes;

    float acc = 0.0f;
    // Each thread processes a strided set of 32-weight groups across the row.
    for (uint g = tid; g < num_groups; g += nthreads) {
        uint xo = g * Q4_GROUP_SIZE;
        float4 xv0 = *reinterpret_cast<device const float4*>(&x[xo]);
        float4 xv1 = *reinterpret_cast<device const float4*>(&x[xo + 4]);
        float4 xv2 = *reinterpret_cast<device const float4*>(&x[xo + 8]);
        float4 xv3 = *reinterpret_cast<device const float4*>(&x[xo + 12]);
        float4 xv4 = *reinterpret_cast<device const float4*>(&x[xo + 16]);
        float4 xv5 = *reinterpret_cast<device const float4*>(&x[xo + 20]);
        float4 xv6 = *reinterpret_cast<device const float4*>(&x[xo + 24]);
        float4 xv7 = *reinterpret_cast<device const float4*>(&x[xo + 28]);

        uint bo = g * Q4_BLOCK_BYTES;
        float scale = float(*reinterpret_cast<device const half*>(&rptr[bo]));
        acc += q4_dot_vec_fast(&rptr[bo + 2],
                               xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7) * scale;
    }

    // Reduce 256 partials → 1: simd_sum within each SIMD group, then combine
    // the per-SIMD-group results (≤ 8) through threadgroup memory.
    threadgroup float partials[Q4F_SG];
    float s = simd_sum(acc);
    if (lane == 0) {
        partials[sgid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        uint groups = nthreads / SIMD_SIZE;
        for (uint i = 0; i < groups; ++i) {
            total += partials[i];
        }
        y[row] = total;
    }
}

// ─── Batched f16 Projection for Prefill ─────────────────────────────────────
// Computes Y = X * W^T where X is (S, K), W is (M, K), Y is (S, M).
// Each threadgroup handles four output rows for one sequence position.

kernel void projection_f16_batch(
    device const half* W [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& S [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint s = tgid.y;
    if (s >= S) return;

    uint base_row = tgid.x * N_ROWS_PER_TG;
    uint row0 = base_row + sgid * 2;
    uint row1 = row0 + 1;
    bool valid0 = row0 < M;
    bool valid1 = row1 < M;

    uint x_base = s * K;
    uint y_base = s * M;
    uint row0_offset = row0 * K;
    uint row1_offset = row1 * K;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    uint k = lane * 4;
    uint stride = SIMD_SIZE * 4;
    for (; k + 3 < K; k += stride) {
        float4 xv = *reinterpret_cast<device const float4*>(&X[x_base + k]);
        if (valid0) {
            half4 w = *reinterpret_cast<device const half4*>(&W[row0_offset + k]);
            acc0 += dot(float4(w), xv);
        }
        if (valid1) {
            half4 w = *reinterpret_cast<device const half4*>(&W[row1_offset + k]);
            acc1 += dot(float4(w), xv);
        }
    }

    for (uint kk = lane + (K / stride) * stride; kk < K; kk += SIMD_SIZE) {
        float xv = X[x_base + kk];
        if (valid0) acc0 += float(W[row0_offset + kk]) * xv;
        if (valid1) acc1 += float(W[row1_offset + kk]) * xv;
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (lane == 0) {
        if (valid0) Y[y_base + row0] = acc0;
        if (valid1) Y[y_base + row1] = acc1;
    }
}

// ─── Batched Q4_0 Projection for Prefill ────────────────────────────────────
// Computes Y = X * W_q4^T where X is (S, K), W_q4 is (M, K), Y is (S, M).
// Same Q4_0 layout as matvec_q4. Each threadgroup handles four output rows
// for one sequence position.

kernel void projection_q4_batch(
    device const uchar* W_q4 [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& S [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint s = tgid.y;
    if (s >= S) return;

    uint base_row = tgid.x * N_ROWS_PER_TG;
    uint row0 = base_row + sgid * 2;
    uint row1 = row0 + 1;
    bool valid0 = row0 < M;
    bool valid1 = row1 < M;

    uint num_groups = K / Q4_GROUP_SIZE;
    uint row_bytes = num_groups * Q4_BLOCK_BYTES;
    uint x_base = s * K;
    uint y_base = s * M;
    device const uchar* row0_ptr = W_q4 + row0 * row_bytes;
    device const uchar* row1_ptr = W_q4 + row1 * row_bytes;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint g = lane; g < num_groups; g += SIMD_SIZE) {
        uint x_offset = x_base + g * Q4_GROUP_SIZE;

        float4 xv0 = *reinterpret_cast<device const float4*>(&X[x_offset]);
        float4 xv1 = *reinterpret_cast<device const float4*>(&X[x_offset + 4]);
        float4 xv2 = *reinterpret_cast<device const float4*>(&X[x_offset + 8]);
        float4 xv3 = *reinterpret_cast<device const float4*>(&X[x_offset + 12]);
        float4 xv4 = *reinterpret_cast<device const float4*>(&X[x_offset + 16]);
        float4 xv5 = *reinterpret_cast<device const float4*>(&X[x_offset + 20]);
        float4 xv6 = *reinterpret_cast<device const float4*>(&X[x_offset + 24]);
        float4 xv7 = *reinterpret_cast<device const float4*>(&X[x_offset + 28]);

        if (valid0) {
            uint block_offset = g * Q4_BLOCK_BYTES;
            float scale = float(*reinterpret_cast<device const half*>(&row0_ptr[block_offset]));
            device const uchar* q = &row0_ptr[block_offset + 2];
            acc0 += q4_dot_vec(q, xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7) * scale;
        }

        if (valid1) {
            uint block_offset = g * Q4_BLOCK_BYTES;
            float scale = float(*reinterpret_cast<device const half*>(&row1_ptr[block_offset]));
            device const uchar* q = &row1_ptr[block_offset + 2];
            acc1 += q4_dot_vec(q, xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7) * scale;
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (lane == 0) {
        if (valid0) Y[y_base + row0] = acc0;
        if (valid1) Y[y_base + row1] = acc1;
    }
}

// ─── SIMD-Group Batched Matrix Multiply (for prefill) ────────────────────────
// C = A * B^T where A is (M, K), B is (N, K), C is (M, N)
// Each SIMD group computes one element of C (dot product of row A and row B).
// For small M (prefill), this gives good parallelism across N.

kernel void matmul(
    device const float* A [[buffer(0)]],   // (M, K)
    device const float* B [[buffer(1)]],   // (N, K) — we compute A @ B^T
    device float* C [[buffer(2)]],         // (M, N)
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;  // M dimension
    uint col = gid.x;  // N dimension
    if (row >= M || col >= N) return;
    
    float acc = 0.0f;
    uint a_offset = row * K;
    uint b_offset = col * K;
    
    for (uint k = 0; k < K; k += 4) {
        if (k + 4 <= K) {
            float4 av = float4(A[a_offset + k], A[a_offset + k + 1],
                              A[a_offset + k + 2], A[a_offset + k + 3]);
            float4 bv = float4(B[b_offset + k], B[b_offset + k + 1],
                              B[b_offset + k + 2], B[b_offset + k + 3]);
            acc += dot(av, bv);
        } else {
            for (uint kk = k; kk < K; kk++) {
                acc += A[a_offset + kk] * B[b_offset + kk];
            }
        }
    }
    
    C[row * N + col] = acc;
}

// ─── Tiled Q4 Batch Projection (Optimized) ──────────────────────────────────
// Computes Y = X * W_q4^T where X is (S, K), W_q4 is (M, K), Y is (S, M).
// KEY OPTIMIZATION: Uses threadgroup shared memory to load weight tiles ONCE
// and reuse across multiple sequence positions (TILE_S).
//
// Tiling strategy:
//   - Each threadgroup handles TILE_M output rows × TILE_S sequence positions
//   - Weights are loaded into shared memory in tiles of [TILE_M × TILE_K]
//   - Input X tiles [TILE_S × TILE_K] are loaded into shared memory
//   - Inner loop: accumulate partial products from K-dimension tiles
//
// Grid: (ceil(M/TILE_M), ceil(S/TILE_S), 1)
// Threadgroup: (TILE_M * TILE_S / work_per_thread, 1, 1) — but we use SIMD groups

constant uint TILE_M_Q4 = 8;     // output rows per threadgroup
constant uint TILE_S_Q4 = 8;     // sequence positions per threadgroup
constant uint TILE_K_Q4 = 32;    // K elements per tile iteration (= one Q4 group)
constant uint TG_SIZE_Q4 = 64;   // threads per threadgroup

kernel void projection_q4_batch_tiled(
    device const uchar* W_q4 [[buffer(0)]],
    device const float* X [[buffer(1)]],    // (S, K) row-major
    device float* Y [[buffer(2)]],          // (S, M) row-major
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& S [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    // This threadgroup computes output tile [row_start..row_start+TILE_M, s_start..s_start+TILE_S]
    uint row_start = tgid.x * TILE_M_Q4;
    uint s_start = tgid.y * TILE_S_Q4;

    // Each thread accumulates results for one (row, seq_pos) pair
    // With 64 threads and 8×8=64 outputs, each thread handles exactly one output
    uint local_row = tid / TILE_S_Q4;     // 0..7
    uint local_s = tid % TILE_S_Q4;       // 0..7

    uint global_row = row_start + local_row;
    uint global_s = s_start + local_s;

    if (global_row >= M || global_s >= S) return;

    uint num_groups = K / Q4_GROUP_SIZE;
    uint row_bytes = num_groups * Q4_BLOCK_BYTES;
    device const uchar* row_ptr = W_q4 + global_row * row_bytes;
    uint x_base = global_s * K;

    float acc = 0.0f;

    for (uint g = 0; g < num_groups; g++) {
        uint x_offset = x_base + g * Q4_GROUP_SIZE;
        uint block_offset = g * Q4_BLOCK_BYTES;

        float4 xv0 = *reinterpret_cast<device const float4*>(&X[x_offset]);
        float4 xv1 = *reinterpret_cast<device const float4*>(&X[x_offset + 4]);
        float4 xv2 = *reinterpret_cast<device const float4*>(&X[x_offset + 8]);
        float4 xv3 = *reinterpret_cast<device const float4*>(&X[x_offset + 12]);
        float4 xv4 = *reinterpret_cast<device const float4*>(&X[x_offset + 16]);
        float4 xv5 = *reinterpret_cast<device const float4*>(&X[x_offset + 20]);
        float4 xv6 = *reinterpret_cast<device const float4*>(&X[x_offset + 24]);
        float4 xv7 = *reinterpret_cast<device const float4*>(&X[x_offset + 28]);

        float scale = float(*reinterpret_cast<device const half*>(&row_ptr[block_offset]));
        device const uchar* q = &row_ptr[block_offset + 2];

        acc += q4_dot_vec(q, xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7) * scale;
    }

    Y[global_s * M + global_row] = acc;
}

// ─── Tiled f16 Batch Projection (Optimized) ─────────────────────────────────
// Same tiling as Q4 but for f16 weights. Each thread handles one (row, seq_pos).

kernel void projection_f16_batch_tiled(
    device const half* W [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& S [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint row_start = tgid.x * TILE_M_Q4;
    uint s_start = tgid.y * TILE_S_Q4;

    uint local_row = tid / TILE_S_Q4;
    uint local_s = tid % TILE_S_Q4;

    uint global_row = row_start + local_row;
    uint global_s = s_start + local_s;

    if (global_row >= M || global_s >= S) return;

    uint row_offset = global_row * K;
    uint x_base = global_s * K;

    float acc = 0.0f;

    for (uint k = 0; k + 3 < K; k += 4) {
        half4 w = *reinterpret_cast<device const half4*>(&W[row_offset + k]);
        float4 xv = *reinterpret_cast<device const float4*>(&X[x_base + k]);
        acc += dot(float4(w), xv);
    }
    // Handle remainder
    for (uint k = (K / 4) * 4; k < K; k++) {
        acc += float(W[row_offset + k]) * X[x_base + k];
    }

    Y[global_s * M + global_row] = acc;
}

// ─── RMS Norm ────────────────────────────────────────────────────────────────
// Computes: out[i] = (x[i] / rms) * weight[i]
// where rms = sqrt(mean(x^2) + eps)
// Single threadgroup reduces to compute rms, then all threads normalize.

kernel void rmsnorm(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Shared memory for reduction
    threadgroup float shared_sum[256];
    
    // Each thread accumulates partial sum of squares
    float partial_sum = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = x[i];
        partial_sum += val * val;
    }
    shared_sum[tid] = partial_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float inv_rms = rsqrt(shared_sum[0] / float(dim) + eps);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Normalize
    for (uint i = tid; i < dim; i += tg_size) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

// ─── Fused RMSNorm + Residual Add ────────────────────────────────────────────
// Computes: out[i] = ((a[i] + b[i]) / rms) * weight[i]
// where rms = sqrt(mean((a + b)^2) + eps)
// Saves one full memory pass vs separate vec_add + rmsnorm.

kernel void rmsnorm_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];

    float partial_sum = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = a[i] + b[i];
        partial_sum += val * val;
    }
    shared_sum[tid] = partial_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt(shared_sum[0] / float(dim) + eps);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < dim; i += tg_size) {
        out[i] = (a[i] + b[i]) * inv_rms * weight[i];
    }
}

// ─── Fused RMSNorm + Residual Add with Residual Save ─────────────────────────
// Computes: out[i] = ((a[i] + b[i]) / rms) * weight[i]
//           residual_out[i] = a[i] + b[i]
// This variant also writes the un-normalized sum back to a buffer so it can
// be reused as the next residual. a and residual_out may alias safely because
// each thread only reads/writes its own indices.

kernel void rmsnorm_add_save_residual(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* out [[buffer(3)]],
    device float* residual_out [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];

    float partial_sum = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = a[i] + b[i];
        residual_out[i] = val;
        partial_sum += val * val;
    }
    shared_sum[tid] = partial_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt(shared_sum[0] / float(dim) + eps);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < dim; i += tg_size) {
        out[i] = residual_out[i] * inv_rms * weight[i];
    }
}

// Gemma4 post-norm residual: acc[i] += (x[i] / rms) * weight[i]
kernel void rmsnorm_acc(
    device float* acc [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];

    float partial_sum = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = x[i];
        partial_sum += val * val;
    }
    shared_sum[tid] = partial_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt(shared_sum[0] / float(dim) + eps);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < dim; i += tg_size) {
        acc[i] += x[i] * inv_rms * weight[i];
    }
}

// acc/out split variant for batch decode (residual_buf + normed -> hidden_buf).
kernel void rmsnorm_acc_out(
    device const float* acc [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];

    float partial_sum = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = x[i];
        partial_sum += val * val;
    }
    shared_sum[tid] = partial_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt(shared_sum[0] / float(dim) + eps);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < dim; i += tg_size) {
        out[i] = acc[i] + x[i] * inv_rms * weight[i];
    }
}

// ─── SiLU + Element-wise Multiply (fused gate activation) ───────────────────
// out[i] = silu(gate[i]) * up[i]
// where silu(x) = x / (1 + exp(-x))

kernel void silu_mul(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float g = gate[gid];
    float s = g / (1.0f + exp(-g));
    out[gid] = s * up[gid];
}

// ─── Fused Single-Token Attention (one head) ─────────────────────────────────
// For a single query vector, computes:
//   scores = Q @ K^T * scale
//   weights = softmax(scores)
//   output = weights @ V
// Q: (head_dim,), K: (kv_len, head_dim), V: (kv_len, head_dim)
// Output: (head_dim,)
// Each threadgroup handles one head.

kernel void attention_single_token(
    device const float* Q [[buffer(0)]],       // (num_heads * head_dim)
    device const float* K_cache [[buffer(1)]], // flat cache buffer
    device const float* V_cache [[buffer(2)]], // flat cache buffer
    device float* output [[buffer(3)]],        // (num_heads * head_dim)
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& k_cap [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint h = tgid;  // One threadgroup per head
    if (h >= num_heads) return;
    
    uint kv_h = h / num_kv_groups;
    uint q_offset = h * head_dim;
    uint k_head_base = kv_h * k_cap * head_dim;
    uint v_head_base = kv_h * k_cap * head_dim;  // same layout
    
    threadgroup float shared_dot[256];
    threadgroup float shared_update[4]; // m, l, old output factor, new value factor

    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    for (uint d = tid; d < head_dim; d += tg_size) {
        output[q_offset + d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv = 0; kv < kv_seq; kv++) {
        float partial_dot = 0.0f;
        uint k_offset = k_head_base + kv * head_dim;
        for (uint d = tid; d < head_dim; d += tg_size) {
            partial_dot += Q[q_offset + d] * K_cache[k_offset + d];
        }
        shared_dot[tid] = partial_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_dot[tid] += shared_dot[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            float m = shared_update[0];
            float l = shared_update[1];
            float score = shared_dot[0] * scale;
            float new_m = max(m, score);
            float alpha = exp(m - new_m);
            float beta = exp(score - new_m);
            float new_l = l * alpha + beta;
            shared_update[0] = new_m;
            shared_update[1] = new_l;
            shared_update[2] = new_l > 0.0f ? (l * alpha) / new_l : 0.0f;
            shared_update[3] = new_l > 0.0f ? beta / new_l : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float old_factor = shared_update[2];
        float new_factor = shared_update[3];
        for (uint d = tid; d < head_dim; d += tg_size) {
            uint out_idx = q_offset + d;
            output[out_idx] = output[out_idx] * old_factor
                + new_factor * V_cache[v_head_base + kv * head_dim + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ─── Rotary Position Embedding ───────────────────────────────────────────────
// Apply rotary embeddings in-place to Q and K buffers.

// Per-layer proportional RoPE parameters (uploaded once at model load).
struct RopeLayerParams {
    float theta;
    float factor;
    uint  head_dim;
    uint  rope_angles;
};

// Fill packed cos/sin for all layers at decode position `pos` (one thread per
// layer × half-dimension; replaces CPU sin/cos + host write_buffer per token).
kernel void rope_fill_decode(
    device float* cos_packed [[buffer(0)]],
    device float* sin_packed [[buffer(1)]],
    constant RopeLayerParams* layers [[buffer(2)]],
    constant uint& max_head_dim [[buffer(3)]],
    constant float& pos [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint layer = gid.y;
    uint d = gid.x;
    RopeLayerParams p = layers[layer];
    uint half_dim = p.head_dim / 2u;
    if (d >= half_dim) return;

    uint base = layer * max_head_dim;
    if (d < p.rope_angles) {
        float inv_freq = 1.0f / (pow(p.theta, 2.0f * float(d) / float(p.head_dim))) / p.factor;
        float angle = pos * inv_freq;
        float c = cos(angle);
        float s = sin(angle);
        cos_packed[base + d] = c;
        cos_packed[base + d + half_dim] = c;
        sin_packed[base + d] = s;
        sin_packed[base + d + half_dim] = s;
    } else {
        cos_packed[base + d] = 1.0f;
        cos_packed[base + d + half_dim] = 1.0f;
        sin_packed[base + d] = 0.0f;
        sin_packed[base + d + half_dim] = 0.0f;
    }
}

kernel void apply_rotary(
    device float* q [[buffer(0)]],         // (num_heads * head_dim)
    device float* k [[buffer(1)]],         // (num_kv_heads * head_dim)
    device const float* cos_buf [[buffer(2)]],  // (head_dim)
    device const float* sin_buf [[buffer(3)]],  // (head_dim)
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint half_dim = head_dim / 2;
    uint total_q = num_heads * half_dim;
    uint total_k = num_kv_heads * half_dim;
    
    if (gid < total_q) {
        // Q rotation
        uint h = gid / half_dim;
        uint d = gid % half_dim;
        uint base = h * head_dim;
        float q1 = q[base + d];
        float q2 = q[base + d + half_dim];
        float c = cos_buf[d];
        float s = sin_buf[d];
        q[base + d] = q1 * c - q2 * s;
        q[base + d + half_dim] = q2 * c + q1 * s;
    } else if (gid < total_q + total_k) {
        // K rotation
        uint idx = gid - total_q;
        uint h = idx / half_dim;
        uint d = idx % half_dim;
        uint base = h * head_dim;
        float k1 = k[base + d];
        float k2 = k[base + d + half_dim];
        float c = cos_buf[d];
        float s = sin_buf[d];
        k[base + d] = k1 * c - k2 * s;
        k[base + d + half_dim] = k2 * c + k1 * s;
    }
}

// ─── Element-wise Add ────────────────────────────────────────────────────────

kernel void vec_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    c[gid] = a[gid] + b[gid];
}

// ─── Embedding Gather (bf16 table → f32 output) ─────────────────────────────
// Table rows are stored as raw bf16 bits (ushort). Gemma scales by sqrt(row_width).

inline float bf16_bits_to_float(ushort bits) {
    return as_type<float>(bits << 16);
}

kernel void embed_gather_bf16(
    device const ushort* table [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& token_id [[buffer(2)]],
    constant uint& row_width [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= row_width) return;
    ushort bits = table[token_id * row_width + gid];
    out[gid] = bf16_bits_to_float(bits) * scale;
}

kernel void embed_gather_bf16_batch(
    device const ushort* table [[buffer(0)]],
    device const uint* token_ids [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& row_width [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = batch_size * row_width;
    if (gid >= total) return;
    uint batch_idx = gid / row_width;
    uint col = gid % row_width;
    uint token_id = token_ids[batch_idx];
    ushort bits = table[token_id * row_width + col];
    out[gid] = bf16_bits_to_float(bits) * scale;
}

// ─── Buffer Copy ─────────────────────────────────────────────────────────────
// Copy n floats from src to dst (used for residual save)

kernel void buf_copy(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    dst[gid] = src[gid];
}

// ─── KV Cache Append ─────────────────────────────────────────────────────────
// Append a single token's K or V (num_kv_heads * head_dim) into the cache buffer.
// Cache layout: (num_kv_heads, capacity, head_dim)
// New data layout: (num_kv_heads, head_dim) — one token

kernel void kv_cache_append(
    device const float* new_data [[buffer(0)]],  // (num_kv_heads * head_dim)
    device float* cache [[buffer(1)]],           // (num_kv_heads * capacity * head_dim)
    constant uint& num_kv_heads [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],
    constant uint& cur_seq [[buffer(5)]],        // current seq length (write position)
    uint gid [[thread_position_in_grid]]
) {
    uint total = num_kv_heads * head_dim;
    if (gid >= total) return;
    
    uint h = gid / head_dim;
    uint d = gid % head_dim;
    
    uint src_offset = h * head_dim + d;
    uint dst_offset = h * capacity * head_dim + cur_seq * head_dim + d;
    
    cache[dst_offset] = new_data[src_offset];
}

// ─── KV Cache Append (f16 cache) ────────────────────────────────────────────
// Converts f32 input to f16 and appends to half-precision cache.

kernel void kv_cache_append_f16(
    device const float* new_data [[buffer(0)]],  // (num_kv_heads * head_dim) f32
    device half* cache [[buffer(1)]],            // (num_kv_heads * capacity * head_dim) f16
    constant uint& num_kv_heads [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],
    constant uint& cur_seq [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = num_kv_heads * head_dim;
    if (gid >= total) return;
    
    uint h = gid / head_dim;
    uint d = gid % head_dim;
    
    uint src_offset = h * head_dim + d;
    uint dst_offset = h * capacity * head_dim + cur_seq * head_dim + d;
    
    cache[dst_offset] = half(new_data[src_offset]);
}

// ─── KV Cache Batch Append ───────────────────────────────────────────────────
// Append seq_len tokens of K or V into the cache.
// new_data layout: (num_kv_heads, seq_len, head_dim)
// cache layout: (num_kv_heads, capacity, head_dim)

kernel void kv_cache_batch_append(
    device const float* new_data [[buffer(0)]],
    device float* cache [[buffer(1)]],
    constant uint& num_kv_heads [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],
    constant uint& cur_seq [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = num_kv_heads * seq_len * head_dim;
    if (gid >= total) return;
    
    uint h = gid / (seq_len * head_dim);
    uint remainder = gid % (seq_len * head_dim);
    uint s = remainder / head_dim;
    uint d = remainder % head_dim;
    
    uint src_offset = h * seq_len * head_dim + s * head_dim + d;
    uint dst_offset = h * capacity * head_dim + (cur_seq + s) * head_dim + d;
    
    cache[dst_offset] = new_data[src_offset];
}

// ─── KV Cache Batch Append (f16 cache) ───────────────────────────────────────
// Append seq_len tokens of K or V into a half-precision cache.
// new_data layout: (num_kv_heads, seq_len, head_dim) f32
// cache layout: (num_kv_heads, capacity, head_dim) f16

kernel void kv_cache_batch_append_f16(
    device const float* new_data [[buffer(0)]],
    device half* cache [[buffer(1)]],
    constant uint& num_kv_heads [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],
    constant uint& cur_seq [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = num_kv_heads * seq_len * head_dim;
    if (gid >= total) return;

    uint h = gid / (seq_len * head_dim);
    uint remainder = gid % (seq_len * head_dim);
    uint s = remainder / head_dim;
    uint d = remainder % head_dim;

    uint src_offset = h * seq_len * head_dim + s * head_dim + d;
    uint dst_offset = h * capacity * head_dim + (cur_seq + s) * head_dim + d;

    cache[dst_offset] = half(new_data[src_offset]);
}

// Append a segment from a larger batched prefill tensor into one cache slot.
// new_data layout: (num_kv_heads, source_seq_stride, head_dim) f32
// cache layout: (num_kv_heads, capacity, head_dim) f16

kernel void kv_cache_batch_append_strided_f16(
    device const float* new_data [[buffer(0)]],
    device half* cache [[buffer(1)]],
    constant uint& num_kv_heads [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],
    constant uint& cur_seq [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    constant uint& source_seq_stride [[buffer(7)]],
    constant uint& source_start [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = num_kv_heads * seq_len * head_dim;
    if (gid >= total) return;

    uint h = gid / (seq_len * head_dim);
    uint remainder = gid % (seq_len * head_dim);
    uint s = remainder / head_dim;
    uint d = remainder % head_dim;

    uint src_offset = h * source_seq_stride * head_dim + (source_start + s) * head_dim + d;
    uint dst_offset = h * capacity * head_dim + (cur_seq + s) * head_dim + d;

    cache[dst_offset] = half(new_data[src_offset]);
}

// ─── Batched RMS Norm ────────────────────────────────────────────────────────
// Normalize each row of a (seq_len, dim) matrix independently.
// One threadgroup per row.

kernel void rmsnorm_batch(
    device const float* x [[buffer(0)]],       // (seq_len * dim)
    device const float* weight [[buffer(1)]],  // (dim)
    device float* out [[buffer(2)]],           // (seq_len * dim)
    constant uint& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint row = tgid;
    uint row_offset = row * dim;
    
    threadgroup float shared_sum[256];
    
    float partial_sum = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = x[row_offset + i];
        partial_sum += val * val;
    }
    shared_sum[tid] = partial_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float inv_rms = rsqrt(shared_sum[0] / float(dim) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint i = tid; i < dim; i += tg_size) {
        out[row_offset + i] = x[row_offset + i] * inv_rms * weight[i];
    }
}

// ─── Batched RMS Norm without Weight ────────────────────────────────────────
// Normalize each row of a (num_rows, dim) matrix independently.

kernel void rmsnorm_noweight_batch(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& dim [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint row = tgid;
    uint row_offset = row * dim;

    threadgroup float shared_sum[256];

    float partial_sum = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = x[row_offset + i];
        partial_sum += val * val;
    }
    shared_sum[tid] = partial_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt(shared_sum[0] / float(dim) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < dim; i += tg_size) {
        out[row_offset + i] = x[row_offset + i] * inv_rms;
    }
}

// ─── Batched SiLU * Up ──────────────────────────────────────────────────────

kernel void silu_mul_batch(
    device const float* gate [[buffer(0)]],  // (seq_len * intermediate)
    device const float* up [[buffer(1)]],    // (seq_len * intermediate)
    device float* out [[buffer(2)]],         // (seq_len * intermediate)
    constant uint& n [[buffer(3)]],          // total elements = seq_len * intermediate
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float g = gate[gid];
    out[gid] = (g / (1.0f + exp(-g))) * up[gid];
}

// ─── Batched Vec Add ─────────────────────────────────────────────────────────

kernel void vec_add_batch(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    c[gid] = a[gid] + b[gid];
}

// ─── Batched Rotary Embedding ────────────────────────────────────────────────
// Apply rotary to Q: (num_heads, seq_len, head_dim) and K: (num_kv_heads, seq_len, head_dim)
// cos/sin: (seq_len, head_dim)

kernel void apply_rotary_batch(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    device const float* cos_buf [[buffer(2)]],
    device const float* sin_buf [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint half_dim = head_dim / 2;
    uint total_q = num_heads * seq_len * half_dim;
    uint total_k = num_kv_heads * seq_len * half_dim;
    
    if (gid < total_q) {
        uint h = gid / (seq_len * half_dim);
        uint remainder = gid % (seq_len * half_dim);
        uint s = remainder / half_dim;
        uint d = remainder % half_dim;
        
        uint base = (h * seq_len + s) * head_dim;
        uint cs_offset = s * head_dim;
        
        float q1 = q[base + d];
        float q2 = q[base + d + half_dim];
        float c = cos_buf[cs_offset + d];
        float sv = sin_buf[cs_offset + d];
        q[base + d] = q1 * c - q2 * sv;
        q[base + d + half_dim] = q2 * c + q1 * sv;
    } else if (gid < total_q + total_k) {
        uint idx = gid - total_q;
        uint h = idx / (seq_len * half_dim);
        uint remainder = idx % (seq_len * half_dim);
        uint s = remainder / half_dim;
        uint d = remainder % half_dim;
        
        uint base = (h * seq_len + s) * head_dim;
        uint cs_offset = s * head_dim;
        
        float k1 = k[base + d];
        float k2 = k[base + d + half_dim];
        float c = cos_buf[cs_offset + d];
        float sv = sin_buf[cs_offset + d];
        k[base + d] = k1 * c - k2 * sv;
        k[base + d + half_dim] = k2 * c + k1 * sv;
    }
}

// ─── Causal Multi-Token Attention ────────────────────────────────────────────
// Q: (num_heads, q_len, head_dim)
// K_cache: (num_kv_heads, capacity, head_dim) — first kv_seq positions valid
// V_cache: same layout
// Output: (num_heads, q_len, head_dim)
// One threadgroup per (head, query_position) pair.

kernel void attention_causal(
    device const float* Q [[buffer(0)]],
    device const float* K_cache [[buffer(1)]],
    device const float* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& k_cap [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& q_len [[buffer(11)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint h = tgid / q_len;
    uint qi = tgid % q_len;
    if (h >= num_heads) return;
    
    uint kv_h = h / num_kv_groups;
    uint q_offset = (h * q_len + qi) * head_dim;
    uint k_head_base = kv_h * k_cap * head_dim;
    uint v_head_base = kv_h * k_cap * head_dim;
    
    // Causal mask: this query at position qi can attend to positions 0..qi+1
    // (since during prefill, kv_seq == q_len and positions are 0-indexed)
    uint attend_len = qi + 1;
    
    threadgroup float shared_dot[256];
    threadgroup float shared_update[4]; // m, l, old output factor, new value factor

    uint out_offset = (h * q_len + qi) * head_dim;
    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    for (uint d = tid; d < head_dim; d += tg_size) {
        output[out_offset + d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv = 0; kv < attend_len; kv++) {
        uint k_offset = k_head_base + kv * head_dim;
        float partial_dot = 0.0f;
        for (uint d = tid; d < head_dim; d += tg_size) {
            partial_dot += Q[q_offset + d] * K_cache[k_offset + d];
        }
        shared_dot[tid] = partial_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_dot[tid] += shared_dot[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            float m = shared_update[0];
            float l = shared_update[1];
            float score = shared_dot[0] * scale;
            float new_m = max(m, score);
            float alpha = exp(m - new_m);
            float beta = exp(score - new_m);
            float new_l = l * alpha + beta;
            shared_update[0] = new_m;
            shared_update[1] = new_l;
            shared_update[2] = new_l > 0.0f ? (l * alpha) / new_l : 0.0f;
            shared_update[3] = new_l > 0.0f ? beta / new_l : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float old_factor = shared_update[2];
        float new_factor = shared_update[3];
        for (uint d = tid; d < head_dim; d += tg_size) {
            uint out_idx = out_offset + d;
            output[out_idx] = output[out_idx] * old_factor
                + new_factor * V_cache[v_head_base + kv * head_dim + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ─── Causal Multi-Token Attention with f16 KV cache ─────────────────────────
// Same as attention_causal but reads K/V from half-precision cache buffers.

kernel void attention_causal_f16(
    device const float* Q [[buffer(0)]],
    device const half* K_cache [[buffer(1)]],
    device const half* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& k_cap [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& q_len [[buffer(11)]],
    constant uint& q_start [[buffer(12)]],
    constant uint& attention_window [[buffer(13)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint h = tgid / q_len;
    uint qi = tgid % q_len;
    if (h >= num_heads) return;
    
    uint kv_h = h / num_kv_groups;
    uint q_offset = (h * q_len + qi) * head_dim;
    uint k_head_base = kv_h * k_cap * head_dim;
    uint v_head_base = kv_h * k_cap * head_dim;
    uint attend_len = min(q_start + qi + 1, kv_seq);
    uint attend_start = 0;
    if (attention_window > 0 && attend_len > attention_window) {
        attend_start = attend_len - attention_window;
    }
    
    threadgroup float shared_dot[256];
    threadgroup float shared_update[4]; // m, l, old output factor, new value factor

    uint out_offset = (h * q_len + qi) * head_dim;
    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    for (uint d = tid; d < head_dim; d += tg_size) {
        output[out_offset + d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv = attend_start; kv < attend_len; kv++) {
        uint k_offset = k_head_base + kv * head_dim;
        float partial_dot = 0.0f;
        for (uint d = tid; d < head_dim; d += tg_size) {
            partial_dot += Q[q_offset + d] * float(K_cache[k_offset + d]);
        }
        shared_dot[tid] = partial_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_dot[tid] += shared_dot[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            float m = shared_update[0];
            float l = shared_update[1];
            float score = shared_dot[0] * scale;
            float new_m = max(m, score);
            float alpha = exp(m - new_m);
            float beta = exp(score - new_m);
            float new_l = l * alpha + beta;
            shared_update[0] = new_m;
            shared_update[1] = new_l;
            shared_update[2] = new_l > 0.0f ? (l * alpha) / new_l : 0.0f;
            shared_update[3] = new_l > 0.0f ? beta / new_l : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float old_factor = shared_update[2];
        float new_factor = shared_update[3];
        for (uint d = tid; d < head_dim; d += tg_size) {
            uint out_idx = out_offset + d;
            output[out_idx] = output[out_idx] * old_factor
                + new_factor * float(V_cache[v_head_base + kv * head_dim + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Segmented causal attention for a prefill sub-range inside a larger packed
// request batch. Q/output layout: (num_heads, q_stride, head_dim).

kernel void attention_causal_strided_f16(
    device const float* Q [[buffer(0)]],
    device const half* K_cache [[buffer(1)]],
    device const half* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& k_cap [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& q_len [[buffer(11)]],
    constant uint& q_pos_start [[buffer(12)]],
    constant uint& attention_window [[buffer(13)]],
    constant uint& q_stride [[buffer(14)]],
    constant uint& q_start_row [[buffer(15)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint h = tgid / q_len;
    uint qi = tgid % q_len;
    if (h >= num_heads) return;

    uint kv_h = h / num_kv_groups;
    uint q_row = q_start_row + qi;
    uint q_offset = (h * q_stride + q_row) * head_dim;
    uint k_head_base = kv_h * k_cap * head_dim;
    uint v_head_base = kv_h * k_cap * head_dim;
    uint attend_len = min(q_pos_start + qi + 1, kv_seq);
    uint attend_start = 0;
    if (attention_window > 0 && attend_len > attention_window) {
        attend_start = attend_len - attention_window;
    }

    threadgroup float shared_dot[256];
    threadgroup float shared_update[4];

    uint out_offset = (h * q_stride + q_row) * head_dim;
    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    for (uint d = tid; d < head_dim; d += tg_size) {
        output[out_offset + d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv = attend_start; kv < attend_len; kv++) {
        uint k_offset = k_head_base + kv * head_dim;
        float partial_dot = 0.0f;
        for (uint d = tid; d < head_dim; d += tg_size) {
            partial_dot += Q[q_offset + d] * float(K_cache[k_offset + d]);
        }
        shared_dot[tid] = partial_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_dot[tid] += shared_dot[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            float m = shared_update[0];
            float l = shared_update[1];
            float score = shared_dot[0] * scale;
            float new_m = max(m, score);
            float alpha = exp(m - new_m);
            float beta = exp(score - new_m);
            float new_l = l * alpha + beta;
            shared_update[0] = new_m;
            shared_update[1] = new_l;
            shared_update[2] = new_l > 0.0f ? (l * alpha) / new_l : 0.0f;
            shared_update[3] = new_l > 0.0f ? beta / new_l : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float old_factor = shared_update[2];
        float new_factor = shared_update[3];
        for (uint d = tid; d < head_dim; d += tg_size) {
            uint out_idx = out_offset + d;
            output[out_idx] = output[out_idx] * old_factor
                + new_factor * float(V_cache[v_head_base + kv * head_dim + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ─── GeLU (PyTorch tanh approximation) ───────────────────────────────────────
// out[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// This is the "gelu_pytorch_tanh" variant used in Gemma models.

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

kernel void gelu_mul_f16(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = half(gelu_pytorch_tanh(gate[gid]) * up[gid]);
}

// ─── Batched PLE GeLU * Context ─────────────────────────────────────────────
// gate: (S, ple_dim), context: (S, num_layers, ple_dim), out: (S, ple_dim)

kernel void ple_gelu_mul_batch(
    device const float* gate [[buffer(0)]],
    device const float* context [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& layer_idx [[buffer(3)]],
    constant uint& num_layers [[buffer(4)]],
    constant uint& ple_dim [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = seq_len * ple_dim;
    if (gid >= total) return;

    uint s = gid / ple_dim;
    uint d = gid % ple_dim;
    uint context_offset = (s * num_layers + layer_idx) * ple_dim + d;

    float x = gate[gid];
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    inner = clamp(inner, -10.0f, 10.0f);
    float gelu = 0.5f * x * (1.0f + tanh(inner));
    out[gid] = gelu * context[context_offset];
}

// ─── Element-wise Multiply ───────────────────────────────────────────────────
// out[i] = a[i] * b[i]

kernel void vec_mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = a[gid] * b[gid];
}

// ─── Scaled Vector Add ───────────────────────────────────────────────────────
// out[i] = a[i] + scale * b[i]

kernel void vec_add_scaled(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = a[gid] + scale * b[gid];
}

// ─── Vector Scale (in-place safe: reads from src, writes to dst) ─────────────
// dst[i] = scale * src[i]

kernel void vec_scale(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    dst[gid] = scale * src[gid];
}

// ─── Per-Head RMS Norm ───────────────────────────────────────────────────────
// Apply RMSNorm independently to each head in a [num_heads * head_dim] buffer.
// weight is [head_dim] and is shared across all heads.
// One threadgroup per head.

kernel void rmsnorm_per_head(
    device const float* x [[buffer(0)]],       // (num_heads * head_dim)
    device const float* weight [[buffer(1)]],  // (head_dim)
    device float* out [[buffer(2)]],           // (num_heads * head_dim)
    constant uint& num_heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint h = tgid;
    if (h >= num_heads) return;
    uint base = h * head_dim;

    threadgroup float shared_sum[256];

    float partial_sum = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_size) {
        float val = x[base + i];
        partial_sum += val * val;
    }
    shared_sum[tid] = partial_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt(shared_sum[0] / float(head_dim) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < head_dim; i += tg_size) {
        out[base + i] = x[base + i] * inv_rms * weight[i];
    }
}

// ─── Per-Head RMS Norm (no weight) ──────────────────────────────────────────
// Same as above but without weight multiplication (for V norm in Gemma4).

kernel void rmsnorm_per_head_noweight(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& num_heads [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint h = tgid;
    if (h >= num_heads) return;
    uint base = h * head_dim;

    threadgroup float shared_sum[256];

    float partial_sum = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_size) {
        float val = x[base + i];
        partial_sum += val * val;
    }
    shared_sum[tid] = partial_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt(shared_sum[0] / float(head_dim) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < head_dim; i += tg_size) {
        out[base + i] = x[base + i] * inv_rms;
    }
}

// ─── Partial Rotary Position Embedding ───────────────────────────────────────
// Apply rotary only to the first rotary_dim elements of each head.
// The remaining elements are copied unchanged.

kernel void apply_rotary_partial(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    device const float* cos_buf [[buffer(2)]],
    device const float* sin_buf [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& rotary_dim [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint half_rot = rotary_dim / 2;
    uint total_q = num_heads * half_rot;
    uint total_k = num_kv_heads * half_rot;

    if (gid < total_q) {
        uint h = gid / half_rot;
        uint d = gid % half_rot;
        uint base = h * head_dim;
        float q1 = q[base + d];
        float q2 = q[base + d + half_rot];
        float c = cos_buf[d];
        float s = sin_buf[d];
        q[base + d] = q1 * c - q2 * s;
        q[base + d + half_rot] = q2 * c + q1 * s;
    } else if (gid < total_q + total_k) {
        uint idx = gid - total_q;
        uint h = idx / half_rot;
        uint d = idx % half_rot;
        uint base = h * head_dim;
        float k1 = k[base + d];
        float k2 = k[base + d + half_rot];
        float c = cos_buf[d];
        float s = sin_buf[d];
        k[base + d] = k1 * c - k2 * s;
        k[base + d + half_rot] = k2 * c + k1 * s;
    }
}

// ─── Attention with KV offset (for sliding window) ──────────────────────────
// Same as attention_single_token but starts reading from kv_start position.

kernel void attention_single_token_offset(
    device const float* Q [[buffer(0)]],
    device const float* K_cache [[buffer(1)]],
    device const float* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& k_cap [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& kv_start [[buffer(11)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint h = tgid;
    if (h >= num_heads) return;

    uint kv_h = h / num_kv_groups;
    uint q_offset = h * head_dim;
    uint k_head_base = kv_h * k_cap * head_dim;
    uint v_head_base = kv_h * k_cap * head_dim;

    threadgroup float shared_dot[256];
    threadgroup float shared_update[4]; // m, l, old output factor, new value factor

    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    for (uint d = tid; d < head_dim; d += tg_size) {
        output[q_offset + d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv = 0; kv < kv_seq; kv++) {
        uint actual_pos = kv_start + kv;
        uint k_offset = k_head_base + actual_pos * head_dim;
        float partial_dot = 0.0f;
        for (uint d = tid; d < head_dim; d += tg_size) {
            partial_dot += Q[q_offset + d] * K_cache[k_offset + d];
        }
        shared_dot[tid] = partial_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_dot[tid] += shared_dot[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            float m = shared_update[0];
            float l = shared_update[1];
            float score = shared_dot[0] * scale;
            float new_m = max(m, score);
            float alpha = exp(m - new_m);
            float beta = exp(score - new_m);
            float new_l = l * alpha + beta;
            shared_update[0] = new_m;
            shared_update[1] = new_l;
            shared_update[2] = new_l > 0.0f ? (l * alpha) / new_l : 0.0f;
            shared_update[3] = new_l > 0.0f ? beta / new_l : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float old_factor = shared_update[2];
        float new_factor = shared_update[3];
        for (uint d = tid; d < head_dim; d += tg_size) {
            uint out_idx = q_offset + d;
            output[out_idx] = output[out_idx] * old_factor
                + new_factor * V_cache[v_head_base + actual_pos * head_dim + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ─── Attention with f16 KV cache ─────────────────────────────────────────────
// Same as attention_single_token_offset but reads from half-precision KV cache.

kernel void attention_single_token_offset_f16(
    device const float* Q [[buffer(0)]],
    device const half* K_cache [[buffer(1)]],
    device const half* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& k_cap [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& kv_start [[buffer(11)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint h = tgid;
    if (h >= num_heads) return;

    uint kv_h = h / num_kv_groups;
    uint q_offset = h * head_dim;
    uint k_head_base = kv_h * k_cap * head_dim;
    uint v_head_base = kv_h * k_cap * head_dim;

    threadgroup float shared_dot[256];
    threadgroup float shared_update[4]; // m, l, old output factor, new value factor

    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    for (uint d = tid; d < head_dim; d += tg_size) {
        output[q_offset + d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv = 0; kv < kv_seq; kv++) {
        uint actual_pos = kv_start + kv;
        uint k_offset = k_head_base + actual_pos * head_dim;
        float partial_dot = 0.0f;
        for (uint d = tid; d < head_dim; d += tg_size) {
            partial_dot += Q[q_offset + d] * float(K_cache[k_offset + d]);
        }
        shared_dot[tid] = partial_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_dot[tid] += shared_dot[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            float m = shared_update[0];
            float l = shared_update[1];
            float score = shared_dot[0] * scale;
            float new_m = max(m, score);
            float alpha = exp(m - new_m);
            float beta = exp(score - new_m);
            float new_l = l * alpha + beta;
            shared_update[0] = new_m;
            shared_update[1] = new_l;
            shared_update[2] = new_l > 0.0f ? (l * alpha) / new_l : 0.0f;
            shared_update[3] = new_l > 0.0f ? beta / new_l : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float old_factor = shared_update[2];
        float new_factor = shared_update[3];
        for (uint d = tid; d < head_dim; d += tg_size) {
            uint out_idx = q_offset + d;
            output[out_idx] = output[out_idx] * old_factor
                + new_factor * float(V_cache[v_head_base + actual_pos * head_dim + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ─── GQA-aware decode attention for f16 KV cache ─────────────────────────────
// One threadgroup per KV head, one simdgroup (32 lanes) per query head in the
// group. Each KV cache row is loaded once and reused for all query heads that
// share it, cutting KV bandwidth by num_kv_groups. The per-head dot product is
// reduced with simd_sum instead of a shared-memory tree.
//
// Requires num_kv_groups * 32 threads per threadgroup (max 4*32 = 128 for E4B)
// and head_dim divisible by 32 and <= 512.

kernel void attention_single_token_offset_f16_gqa(
    device const float* Q [[buffer(0)]],
    device const half* K_cache [[buffer(1)]],
    device const half* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& k_cap [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& kv_start [[buffer(11)]],
    ushort tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    const uint kv_h = tgid;
    if (kv_h >= num_kv_heads) return;

    const ushort lane = tid & 31u;
    const ushort group = tid >> 5u;
    if (group >= num_kv_groups) return;

    const uint h = kv_h * num_kv_groups + group;
    if (h >= num_heads) return;

    const uint q_offset = h * head_dim;
    const uint k_head_base = kv_h * k_cap * head_dim;
    const uint v_head_base = kv_h * k_cap * head_dim;

    // Per-lane accumulator for the dimensions this lane owns (head_dim/32 max 16).
    float acc[16];
    for (uint i = 0; i < head_dim / 32u; ++i) {
        acc[i] = 0.0f;
    }

    float m = -INFINITY;
    float l = 0.0f;

    for (uint kv = 0; kv < kv_seq; ++kv) {
        const uint actual_pos = kv_start + kv;
        const uint k_offset = k_head_base + actual_pos * head_dim;

        float dot = 0.0f;
        for (uint d = lane; d < head_dim; d += 32u) {
            dot += Q[q_offset + d] * float(K_cache[k_offset + d]);
        }
        dot = simd_sum(dot);

        const float score = dot * scale;
        const float new_m = max(m, score);
        const float alpha = exp(m - new_m);
        const float beta = exp(score - new_m);
        const float new_l = l * alpha + beta;
        const float old_factor = new_l > 0.0f ? (l * alpha) / new_l : 0.0f;
        const float new_factor = new_l > 0.0f ? beta / new_l : 0.0f;
        m = new_m;
        l = new_l;

        const uint v_offset = v_head_base + actual_pos * head_dim;
        for (uint d = lane; d < head_dim; d += 32u) {
            const uint idx = d / 32u;
            acc[idx] = acc[idx] * old_factor + new_factor * float(V_cache[v_offset + d]);
        }
    }

    for (uint d = lane; d < head_dim; d += 32u) {
        output[q_offset + d] = acc[d / 32u];
    }
}

// ─── Q8_0 KV Cache Append ────────────────────────────────────────────────────
// Quantizes f32 K/V data to Q8_0 format and appends to quantized cache.
// Q8_0 layout per group of 32: [2-byte f16 scale][32 int8 values] = 34 bytes per group
// Cache layout: (num_kv_heads, capacity, groups_per_row * 34)

kernel void kv_cache_append_q8_0(
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
    uint row_bytes = groups_per_row * 34;

    // Find max abs value in this group
    float max_abs = 0.0f;
    for (uint d = 0; d < 32; d++) {
        float val = new_data[h * head_dim + g * 32 + d];
        float a = fabs(val);
        if (a > max_abs) max_abs = a;
    }

    float scale = max_abs / 127.0f;
    if (max_abs == 0.0f) scale = 1.0f;
    float inv_scale = 1.0f / scale;

    // Store scale as f16
    half scale_h = half(scale);
    uint base_offset = h * capacity * row_bytes + cur_seq * row_bytes + g * 34;
    *reinterpret_cast<device half*>(&cache[base_offset]) = scale_h;

    // Store int8 quantized values (as unsigned bytes with two's complement)
    for (uint d = 0; d < 32; d++) {
        float val = new_data[h * head_dim + g * 32 + d];
        int q = clamp(int(round(val * inv_scale)), -128, 127);
        cache[base_offset + 2 + d] = uchar(q);
    }
}

// ─── Q4_0 KV Cache Append ────────────────────────────────────────────────────
// Quantizes f32 K/V data to Q4_0 format and appends to quantized cache.
// Q4_0 layout per group of 32: [2-byte f16 scale][16 packed bytes] = 18 bytes per group
// Cache layout: (num_kv_heads, capacity, groups_per_row * 18)

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

// ─── Q8_0 KV Cache Batch Append ──────────────────────────────────────────────

kernel void kv_cache_batch_append_q8_0(
    device const float* new_data [[buffer(0)]],
    device uchar* cache [[buffer(1)]],
    constant uint& num_kv_heads [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],
    constant uint& cur_seq [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint groups_per_row = head_dim / 32;
    uint total = num_kv_heads * seq_len * groups_per_row;
    if (gid >= total) return;

    uint h = gid / (seq_len * groups_per_row);
    uint remainder = gid % (seq_len * groups_per_row);
    uint s = remainder / groups_per_row;
    uint g = remainder % groups_per_row;
    uint row_bytes = groups_per_row * 34;

    float max_abs = 0.0f;
    for (uint d = 0; d < 32; d++) {
        float val = new_data[h * seq_len * head_dim + s * head_dim + g * 32 + d];
        float a = fabs(val);
        if (a > max_abs) max_abs = a;
    }

    float scale = max_abs / 127.0f;
    if (max_abs == 0.0f) scale = 1.0f;
    float inv_scale = 1.0f / scale;

    half scale_h = half(scale);
    uint base_offset = h * capacity * row_bytes + (cur_seq + s) * row_bytes + g * 34;
    *reinterpret_cast<device half*>(&cache[base_offset]) = scale_h;

    for (uint d = 0; d < 32; d++) {
        float val = new_data[h * seq_len * head_dim + s * head_dim + g * 32 + d];
        int q = clamp(int(round(val * inv_scale)), -128, 127);
        cache[base_offset + 2 + d] = uchar(q);
    }
}

// ─── Q8_0 KV Cache Batch Append Strided ──────────────────────────────────────

kernel void kv_cache_batch_append_strided_q8_0(
    device const float* new_data [[buffer(0)]],
    device uchar* cache [[buffer(1)]],
    constant uint& num_kv_heads [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],
    constant uint& cur_seq [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    constant uint& source_seq_stride [[buffer(7)]],
    constant uint& source_start [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    uint groups_per_row = head_dim / 32;
    uint total = num_kv_heads * seq_len * groups_per_row;
    if (gid >= total) return;

    uint h = gid / (seq_len * groups_per_row);
    uint remainder = gid % (seq_len * groups_per_row);
    uint s = remainder / groups_per_row;
    uint g = remainder % groups_per_row;
    uint row_bytes = groups_per_row * 34;

    float max_abs = 0.0f;
    for (uint d = 0; d < 32; d++) {
        float val = new_data[h * source_seq_stride * head_dim + (source_start + s) * head_dim + g * 32 + d];
        float a = fabs(val);
        if (a > max_abs) max_abs = a;
    }

    float scale = max_abs / 127.0f;
    if (max_abs == 0.0f) scale = 1.0f;
    float inv_scale = 1.0f / scale;

    half scale_h = half(scale);
    uint base_offset = h * capacity * row_bytes + (cur_seq + s) * row_bytes + g * 34;
    *reinterpret_cast<device half*>(&cache[base_offset]) = scale_h;

    for (uint d = 0; d < 32; d++) {
        float val = new_data[h * source_seq_stride * head_dim + (source_start + s) * head_dim + g * 32 + d];
        int q = clamp(int(round(val * inv_scale)), -128, 127);
        cache[base_offset + 2 + d] = uchar(q);
    }
}

// ─── Q4_0 KV Cache Batch Append ──────────────────────────────────────────────

kernel void kv_cache_batch_append_q4_0(
    device const float* new_data [[buffer(0)]],
    device uchar* cache [[buffer(1)]],
    constant uint& num_kv_heads [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],
    constant uint& cur_seq [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint groups_per_row = head_dim / 32;
    uint total = num_kv_heads * seq_len * groups_per_row;
    if (gid >= total) return;

    uint h = gid / (seq_len * groups_per_row);
    uint remainder = gid % (seq_len * groups_per_row);
    uint s = remainder / groups_per_row;
    uint g = remainder % groups_per_row;
    uint row_bytes = groups_per_row * 18;

    float max_abs = 0.0f;
    for (uint d = 0; d < 32; d++) {
        float val = new_data[h * seq_len * head_dim + s * head_dim + g * 32 + d];
        float a = fabs(val);
        if (a > max_abs) max_abs = a;
    }

    float scale = max_abs / 7.0f;
    if (max_abs == 0.0f) scale = 1.0f;
    float inv_scale = 1.0f / scale;

    half scale_h = half(scale);
    uint base_offset = h * capacity * row_bytes + (cur_seq + s) * row_bytes + g * 18;
    *reinterpret_cast<device half*>(&cache[base_offset]) = scale_h;

    for (uint i = 0; i < 16; i++) {
        float v_lo = new_data[h * seq_len * head_dim + s * head_dim + g * 32 + i];
        float v_hi = new_data[h * seq_len * head_dim + s * head_dim + g * 32 + i + 16];
        int q_lo = clamp(int(round(v_lo * inv_scale)) + 8, 0, 15);
        int q_hi = clamp(int(round(v_hi * inv_scale)) + 8, 0, 15);
        cache[base_offset + 2 + i] = uchar(q_lo | (q_hi << 4));
    }
}

// ─── Q4_0 KV Cache Batch Append Strided ──────────────────────────────────────

kernel void kv_cache_batch_append_strided_q4_0(
    device const float* new_data [[buffer(0)]],
    device uchar* cache [[buffer(1)]],
    constant uint& num_kv_heads [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],
    constant uint& cur_seq [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    constant uint& source_seq_stride [[buffer(7)]],
    constant uint& source_start [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    uint groups_per_row = head_dim / 32;
    uint total = num_kv_heads * seq_len * groups_per_row;
    if (gid >= total) return;

    uint h = gid / (seq_len * groups_per_row);
    uint remainder = gid % (seq_len * groups_per_row);
    uint s = remainder / groups_per_row;
    uint g = remainder % groups_per_row;
    uint row_bytes = groups_per_row * 18;

    float max_abs = 0.0f;
    for (uint d = 0; d < 32; d++) {
        float val = new_data[h * source_seq_stride * head_dim + (source_start + s) * head_dim + g * 32 + d];
        float a = fabs(val);
        if (a > max_abs) max_abs = a;
    }

    float scale = max_abs / 7.0f;
    if (max_abs == 0.0f) scale = 1.0f;
    float inv_scale = 1.0f / scale;

    half scale_h = half(scale);
    uint base_offset = h * capacity * row_bytes + (cur_seq + s) * row_bytes + g * 18;
    *reinterpret_cast<device half*>(&cache[base_offset]) = scale_h;

    for (uint i = 0; i < 16; i++) {
        float v_lo = new_data[h * source_seq_stride * head_dim + (source_start + s) * head_dim + g * 32 + i];
        float v_hi = new_data[h * source_seq_stride * head_dim + (source_start + s) * head_dim + g * 32 + i + 16];
        int q_lo = clamp(int(round(v_lo * inv_scale)) + 8, 0, 15);
        int q_hi = clamp(int(round(v_hi * inv_scale)) + 8, 0, 15);
        cache[base_offset + 2 + i] = uchar(q_lo | (q_hi << 4));
    }
}

// ─── Q8_0 Dequantization Helper ──────────────────────────────────────────────

inline float q8_0_read(device const uchar* cache, uint head_base, uint pos, uint row_bytes, uint d) {
    uint g = d / 32;
    uint d_in_group = d % 32;
    uint offset = head_base + pos * row_bytes + g * 34;
    float scale = float(*reinterpret_cast<device const half*>(&cache[offset]));
    return float(reinterpret_cast<device const int8_t*>(&cache[offset + 2])[d_in_group]) * scale;
}

// ─── Q4_0 Dequantization Helper ──────────────────────────────────────────────

inline float q4_0_read(device const uchar* cache, uint head_base, uint pos, uint row_bytes, uint d) {
    uint g = d / 32;
    uint e = d % 32;
    uint offset = head_base + pos * row_bytes + g * 18;
    float scale = float(*reinterpret_cast<device const half*>(&cache[offset]));
    device const uchar* qs = cache + offset + 2;
    if (e < 16) {
        return float(int(qs[e] & 0xF) - 8) * scale;
    } else {
        return float(int(qs[e - 16] >> 4) - 8) * scale;
    }
}

// Vectorized variant: reads 4 consecutive Q4_0 values starting at d (d must be a multiple of 4).
inline float4 q4_0_read4(device const uchar* cache, uint head_base, uint pos, uint row_bytes, uint d) {
    uint g = d / 32;
    uint e = d % 32;
    uint offset = head_base + pos * row_bytes + g * 18;
    float scale = float(*reinterpret_cast<device const half*>(&cache[offset]));
    device const uchar* qs = cache + offset + 2;
    if (e + 3 < 16) {
        return float4(
            float(int(qs[e + 0] & 0xF) - 8) * scale,
            float(int(qs[e + 1] & 0xF) - 8) * scale,
            float(int(qs[e + 2] & 0xF) - 8) * scale,
            float(int(qs[e + 3] & 0xF) - 8) * scale
        );
    }
    if (e >= 16 && e + 3 < 32) {
        uint b = e - 16;
        return float4(
            float(int(qs[b + 0] >> 4) - 8) * scale,
            float(int(qs[b + 1] >> 4) - 8) * scale,
            float(int(qs[b + 2] >> 4) - 8) * scale,
            float(int(qs[b + 3] >> 4) - 8) * scale
        );
    }
    return float4(
        q4_0_read(cache, head_base, pos, row_bytes, d),
        q4_0_read(cache, head_base, pos, row_bytes, d + 1),
        q4_0_read(cache, head_base, pos, row_bytes, d + 2),
        q4_0_read(cache, head_base, pos, row_bytes, d + 3)
    );
}

// ─── Attention with Q8_0 KV cache (single token offset) ──────────────────────

kernel void attention_single_token_offset_q8_0(
    device const float* Q [[buffer(0)]],
    device const uchar* K_cache [[buffer(1)]],
    device const uchar* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& capacity [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& kv_start [[buffer(11)]],
    constant uint& groups_per_row [[buffer(12)]],
    constant uint& row_bytes [[buffer(13)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint h = tgid;
    if (h >= num_heads) return;

    uint kv_h = h / num_kv_groups;
    uint q_offset = h * head_dim;
    uint k_head_base = kv_h * capacity * row_bytes;
    uint v_head_base = kv_h * capacity * row_bytes;

    threadgroup float shared_dot[256];
    threadgroup float shared_update[4];

    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    for (uint d = tid; d < head_dim; d += tg_size) {
        output[q_offset + d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv = 0; kv < kv_seq; kv++) {
        uint actual_pos = kv_start + kv;
        float partial_dot = 0.0f;
        for (uint d = tid; d < head_dim; d += tg_size) {
            partial_dot += Q[q_offset + d] * q8_0_read(K_cache, k_head_base, actual_pos, row_bytes, d);
        }
        shared_dot[tid] = partial_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_dot[tid] += shared_dot[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            float m = shared_update[0];
            float l = shared_update[1];
            float score = shared_dot[0] * scale;
            float new_m = max(m, score);
            float alpha = exp(m - new_m);
            float beta = exp(score - new_m);
            float new_l = l * alpha + beta;
            shared_update[0] = new_m;
            shared_update[1] = new_l;
            shared_update[2] = new_l > 0.0f ? (l * alpha) / new_l : 0.0f;
            shared_update[3] = new_l > 0.0f ? beta / new_l : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float old_factor = shared_update[2];
        float new_factor = shared_update[3];
        for (uint d = tid; d < head_dim; d += tg_size) {
            uint out_idx = q_offset + d;
            output[out_idx] = output[out_idx] * old_factor
                + new_factor * q8_0_read(V_cache, v_head_base, actual_pos, row_bytes, d);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ─── Causal Multi-Token Attention with Q8_0 KV cache ─────────────────────────

kernel void attention_causal_q8_0(
    device const float* Q [[buffer(0)]],
    device const uchar* K_cache [[buffer(1)]],
    device const uchar* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& capacity [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& q_len [[buffer(11)]],
    constant uint& q_start [[buffer(12)]],
    constant uint& attention_window [[buffer(13)]],
    constant uint& groups_per_row [[buffer(14)]],
    constant uint& row_bytes [[buffer(15)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint h = tgid / q_len;
    uint qi = tgid % q_len;
    if (h >= num_heads) return;

    uint kv_h = h / num_kv_groups;
    uint q_offset = (h * q_len + qi) * head_dim;
    uint k_head_base = kv_h * capacity * row_bytes;
    uint v_head_base = kv_h * capacity * row_bytes;
    uint attend_len = min(q_start + qi + 1, kv_seq);
    uint attend_start = 0;
    if (attention_window > 0 && attend_len > attention_window) {
        attend_start = attend_len - attention_window;
    }

    threadgroup float shared_dot[256];
    threadgroup float shared_update[4];

    uint out_offset = (h * q_len + qi) * head_dim;
    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    for (uint d = tid; d < head_dim; d += tg_size) {
        output[out_offset + d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv = attend_start; kv < attend_len; kv++) {
        float partial_dot = 0.0f;
        for (uint d = tid; d < head_dim; d += tg_size) {
            partial_dot += Q[q_offset + d] * q8_0_read(K_cache, k_head_base, kv, row_bytes, d);
        }
        shared_dot[tid] = partial_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_dot[tid] += shared_dot[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            float m = shared_update[0];
            float l = shared_update[1];
            float score = shared_dot[0] * scale;
            float new_m = max(m, score);
            float alpha = exp(m - new_m);
            float beta = exp(score - new_m);
            float new_l = l * alpha + beta;
            shared_update[0] = new_m;
            shared_update[1] = new_l;
            shared_update[2] = new_l > 0.0f ? (l * alpha) / new_l : 0.0f;
            shared_update[3] = new_l > 0.0f ? beta / new_l : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float old_factor = shared_update[2];
        float new_factor = shared_update[3];
        for (uint d = tid; d < head_dim; d += tg_size) {
            uint out_idx = out_offset + d;
            output[out_idx] = output[out_idx] * old_factor
                + new_factor * q8_0_read(V_cache, v_head_base, kv, row_bytes, d);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ─── Strided Causal Attention with Q8_0 KV cache ────────────────────────────

kernel void attention_causal_strided_q8_0(
    device const float* Q [[buffer(0)]],
    device const uchar* K_cache [[buffer(1)]],
    device const uchar* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& capacity [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& q_len [[buffer(11)]],
    constant uint& q_pos_start [[buffer(12)]],
    constant uint& attention_window [[buffer(13)]],
    constant uint& q_stride [[buffer(14)]],
    constant uint& q_start_row [[buffer(15)]],
    constant uint& groups_per_row [[buffer(16)]],
    constant uint& row_bytes [[buffer(17)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint h = tgid / q_len;
    uint qi = tgid % q_len;
    if (h >= num_heads) return;

    uint kv_h = h / num_kv_groups;
    uint q_row = q_start_row + qi;
    uint q_offset = (h * q_stride + q_row) * head_dim;
    uint k_head_base = kv_h * capacity * row_bytes;
    uint v_head_base = kv_h * capacity * row_bytes;
    uint attend_len = min(q_pos_start + qi + 1, kv_seq);
    uint attend_start = 0;
    if (attention_window > 0 && attend_len > attention_window) {
        attend_start = attend_len - attention_window;
    }

    threadgroup float shared_dot[256];
    threadgroup float shared_update[4];

    uint out_offset = (h * q_stride + q_row) * head_dim;
    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    for (uint d = tid; d < head_dim; d += tg_size) {
        output[out_offset + d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv = attend_start; kv < attend_len; kv++) {
        float partial_dot = 0.0f;
        for (uint d = tid; d < head_dim; d += tg_size) {
            partial_dot += Q[q_offset + d] * q8_0_read(K_cache, k_head_base, kv, row_bytes, d);
        }
        shared_dot[tid] = partial_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_dot[tid] += shared_dot[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            float m = shared_update[0];
            float l = shared_update[1];
            float score = shared_dot[0] * scale;
            float new_m = max(m, score);
            float alpha = exp(m - new_m);
            float beta = exp(score - new_m);
            float new_l = l * alpha + beta;
            shared_update[0] = new_m;
            shared_update[1] = new_l;
            shared_update[2] = new_l > 0.0f ? (l * alpha) / new_l : 0.0f;
            shared_update[3] = new_l > 0.0f ? beta / new_l : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float old_factor = shared_update[2];
        float new_factor = shared_update[3];
        for (uint d = tid; d < head_dim; d += tg_size) {
            uint out_idx = out_offset + d;
            output[out_idx] = output[out_idx] * old_factor
                + new_factor * q8_0_read(V_cache, v_head_base, kv, row_bytes, d);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ─── Attention with Q4_0 KV cache (single token offset) ──────────────────────

kernel void attention_single_token_offset_q4_0(
    device const float* Q [[buffer(0)]],
    device const uchar* K_cache [[buffer(1)]],
    device const uchar* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& capacity [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& kv_start [[buffer(11)]],
    constant uint& groups_per_row [[buffer(12)]],
    constant uint& row_bytes [[buffer(13)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint h = tgid;
    if (h >= num_heads) return;

    const uint TILE_KV = 4;
    uint kv_h = h / num_kv_groups;
    uint q_offset = h * head_dim;
    uint k_head_base = kv_h * capacity * row_bytes;
    uint v_head_base = kv_h * capacity * row_bytes;

    uint simd_id = tid / SIMD_SIZE;
    uint lane = tid % SIMD_SIZE;
    uint num_simds = tg_size / SIMD_SIZE;

    threadgroup float shared_scores[TILE_KV * 4];
    threadgroup float shared_exp[TILE_KV];
    threadgroup float shared_update[4];

    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    for (uint d = tid * 4; d + 3 < head_dim; d += tg_size * 4) {
        *reinterpret_cast<device float4*>(&output[q_offset + d]) = float4(0.0f);
    }
    for (uint d = (head_dim / 4) * 4 + tid; d < head_dim; d += tg_size) {
        output[q_offset + d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv_tile = 0; kv_tile < kv_seq; kv_tile += TILE_KV) {
        uint tile_end = min(kv_tile + TILE_KV, kv_seq);
        uint tile_count = tile_end - kv_tile;

        // Compute scores for all KV positions in this tile.
        for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
            uint actual_pos = kv_start + kv_tile + kv_offset;
            float partial_dot = 0.0f;
            for (uint d = tid * 4; d + 3 < head_dim; d += tg_size * 4) {
                float4 qv = *reinterpret_cast<device const float4*>(&Q[q_offset + d]);
                float4 k_vals = q4_0_read4(K_cache, k_head_base, actual_pos, row_bytes, d);
                partial_dot += dot(qv, k_vals);
            }
            for (uint d = (head_dim / 4) * 4 + tid; d < head_dim; d += tg_size) {
                partial_dot += Q[q_offset + d] * q4_0_read(K_cache, k_head_base, actual_pos, row_bytes, d);
            }
            partial_dot = simd_sum(partial_dot);
            if (lane == 0) {
                shared_scores[kv_offset * num_simds + simd_id] = partial_dot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Update running softmax statistics for the tile.
        if (tid == 0) {
            float m_old = shared_update[0];
            float l_old = shared_update[1];
            float m_new = m_old;

            float tile_scores[TILE_KV];
            for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                float s = 0.0f;
                for (uint s_id = 0; s_id < num_simds; s_id++) {
                    s += shared_scores[kv_offset * num_simds + s_id];
                }
                tile_scores[kv_offset] = s * scale;
                m_new = max(m_new, tile_scores[kv_offset]);
            }

            float tile_sum = 0.0f;
            for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                float e = exp(tile_scores[kv_offset] - m_new);
                shared_exp[kv_offset] = e;
                tile_sum += e;
            }

            float l_new = l_old * exp(m_old - m_new) + tile_sum;
            shared_update[0] = m_new;
            shared_update[1] = l_new;
            shared_update[2] = l_new > 0.0f ? (l_old * exp(m_old - m_new)) / l_new : 0.0f;
            shared_update[3] = l_new > 0.0f ? 1.0f / l_new : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float old_factor = shared_update[2];
        float inv_l_new = shared_update[3];

        for (uint d = tid * 4; d + 3 < head_dim; d += tg_size * 4) {
            uint out_idx = q_offset + d;
            float4 ov = *reinterpret_cast<device float4*>(&output[out_idx]);
            float4 acc = float4(0.0f);
            for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                uint actual_pos = kv_start + kv_tile + kv_offset;
                acc += shared_exp[kv_offset] * q4_0_read4(V_cache, v_head_base, actual_pos, row_bytes, d);
            }
            ov = ov * old_factor + acc * inv_l_new;
            *reinterpret_cast<device float4*>(&output[out_idx]) = ov;
        }
        for (uint d = (head_dim / 4) * 4 + tid; d < head_dim; d += tg_size) {
            uint out_idx = q_offset + d;
            float acc = 0.0f;
            for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                uint actual_pos = kv_start + kv_tile + kv_offset;
                acc += shared_exp[kv_offset] * q4_0_read(V_cache, v_head_base, actual_pos, row_bytes, d);
            }
            output[out_idx] = output[out_idx] * old_factor + acc * inv_l_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ─── Causal Multi-Token Attention with Q4_0 KV cache ─────────────────────────

kernel void attention_causal_q4_0(
    device const float* Q [[buffer(0)]],
    device const uchar* K_cache [[buffer(1)]],
    device const uchar* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& capacity [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& q_len [[buffer(11)]],
    constant uint& q_start [[buffer(12)]],
    constant uint& attention_window [[buffer(13)]],
    constant uint& groups_per_row [[buffer(14)]],
    constant uint& row_bytes [[buffer(15)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint h = tgid / q_len;
    uint qi = tgid % q_len;
    if (h >= num_heads) return;

    const uint TILE_KV = 4;
    uint kv_h = h / num_kv_groups;
    uint q_offset = (h * q_len + qi) * head_dim;
    uint k_head_base = kv_h * capacity * row_bytes;
    uint v_head_base = kv_h * capacity * row_bytes;
    uint attend_len = min(q_start + qi + 1, kv_seq);
    uint attend_start = 0;
    if (attention_window > 0 && attend_len > attention_window) {
        attend_start = attend_len - attention_window;
    }

    uint simd_id = tid / SIMD_SIZE;
    uint lane = tid % SIMD_SIZE;
    uint num_simds = tg_size / SIMD_SIZE;

    threadgroup float shared_scores[TILE_KV * 4];
    threadgroup float shared_exp[TILE_KV];
    threadgroup float shared_update[4];

    uint out_offset = (h * q_len + qi) * head_dim;
    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    for (uint d = tid * 4; d + 3 < head_dim; d += tg_size * 4) {
        *reinterpret_cast<device float4*>(&output[out_offset + d]) = float4(0.0f);
    }
    for (uint d = (head_dim / 4) * 4 + tid; d < head_dim; d += tg_size) {
        output[out_offset + d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv_tile = attend_start; kv_tile < attend_len; kv_tile += TILE_KV) {
        uint tile_end = min(kv_tile + TILE_KV, attend_len);
        uint tile_count = tile_end - kv_tile;

        for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
            uint kv = kv_tile + kv_offset;
            float partial_dot = 0.0f;
            for (uint d = tid * 4; d + 3 < head_dim; d += tg_size * 4) {
                float4 qv = *reinterpret_cast<device const float4*>(&Q[q_offset + d]);
                float4 k_vals = q4_0_read4(K_cache, k_head_base, kv, row_bytes, d);
                partial_dot += dot(qv, k_vals);
            }
            for (uint d = (head_dim / 4) * 4 + tid; d < head_dim; d += tg_size) {
                partial_dot += Q[q_offset + d] * q4_0_read(K_cache, k_head_base, kv, row_bytes, d);
            }
            partial_dot = simd_sum(partial_dot);
            if (lane == 0) {
                shared_scores[kv_offset * num_simds + simd_id] = partial_dot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float m_old = shared_update[0];
            float l_old = shared_update[1];
            float m_new = m_old;

            float tile_scores[TILE_KV];
            for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                float s = 0.0f;
                for (uint s_id = 0; s_id < num_simds; s_id++) {
                    s += shared_scores[kv_offset * num_simds + s_id];
                }
                tile_scores[kv_offset] = s * scale;
                m_new = max(m_new, tile_scores[kv_offset]);
            }

            float tile_sum = 0.0f;
            for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                float e = exp(tile_scores[kv_offset] - m_new);
                shared_exp[kv_offset] = e;
                tile_sum += e;
            }

            float l_new = l_old * exp(m_old - m_new) + tile_sum;
            shared_update[0] = m_new;
            shared_update[1] = l_new;
            shared_update[2] = l_new > 0.0f ? (l_old * exp(m_old - m_new)) / l_new : 0.0f;
            shared_update[3] = l_new > 0.0f ? 1.0f / l_new : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float old_factor = shared_update[2];
        float inv_l_new = shared_update[3];

        for (uint d = tid * 4; d + 3 < head_dim; d += tg_size * 4) {
            uint out_idx = out_offset + d;
            float4 ov = *reinterpret_cast<device float4*>(&output[out_idx]);
            float4 acc = float4(0.0f);
            for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                uint kv = kv_tile + kv_offset;
                acc += shared_exp[kv_offset] * q4_0_read4(V_cache, v_head_base, kv, row_bytes, d);
            }
            ov = ov * old_factor + acc * inv_l_new;
            *reinterpret_cast<device float4*>(&output[out_idx]) = ov;
        }
        for (uint d = (head_dim / 4) * 4 + tid; d < head_dim; d += tg_size) {
            uint out_idx = out_offset + d;
            float acc = 0.0f;
            for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                uint kv = kv_tile + kv_offset;
                acc += shared_exp[kv_offset] * q4_0_read(V_cache, v_head_base, kv, row_bytes, d);
            }
            output[out_idx] = output[out_idx] * old_factor + acc * inv_l_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ─── Strided Causal Attention with Q4_0 KV cache ─────────────────────────────

kernel void attention_causal_strided_q4_0(
    device const float* Q [[buffer(0)]],
    device const uchar* K_cache [[buffer(1)]],
    device const uchar* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& capacity [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& q_len [[buffer(11)]],
    constant uint& q_pos_start [[buffer(12)]],
    constant uint& attention_window [[buffer(13)]],
    constant uint& q_stride [[buffer(14)]],
    constant uint& q_start_row [[buffer(15)]],
    constant uint& groups_per_row [[buffer(16)]],
    constant uint& row_bytes [[buffer(17)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint h = tgid / q_len;
    uint qi = tgid % q_len;
    if (h >= num_heads) return;

    const uint TILE_KV = 4;
    uint kv_h = h / num_kv_groups;
    uint q_row = q_start_row + qi;
    uint q_offset = (h * q_stride + q_row) * head_dim;
    uint k_head_base = kv_h * capacity * row_bytes;
    uint v_head_base = kv_h * capacity * row_bytes;
    uint attend_len = min(q_pos_start + qi + 1, kv_seq);
    uint attend_start = 0;
    if (attention_window > 0 && attend_len > attention_window) {
        attend_start = attend_len - attention_window;
    }

    uint simd_id = tid / SIMD_SIZE;
    uint lane = tid % SIMD_SIZE;
    uint num_simds = tg_size / SIMD_SIZE;

    threadgroup float shared_scores[TILE_KV * 4];
    threadgroup float shared_exp[TILE_KV];
    threadgroup float shared_update[4];

    uint out_offset = (h * q_stride + q_row) * head_dim;
    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    for (uint d = tid * 4; d + 3 < head_dim; d += tg_size * 4) {
        *reinterpret_cast<device float4*>(&output[out_offset + d]) = float4(0.0f);
    }
    for (uint d = (head_dim / 4) * 4 + tid; d < head_dim; d += tg_size) {
        output[out_offset + d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv_tile = attend_start; kv_tile < attend_len; kv_tile += TILE_KV) {
        uint tile_end = min(kv_tile + TILE_KV, attend_len);
        uint tile_count = tile_end - kv_tile;

        for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
            uint kv = kv_tile + kv_offset;
            float partial_dot = 0.0f;
            for (uint d = tid * 4; d + 3 < head_dim; d += tg_size * 4) {
                float4 qv = *reinterpret_cast<device const float4*>(&Q[q_offset + d]);
                float4 k_vals = q4_0_read4(K_cache, k_head_base, kv, row_bytes, d);
                partial_dot += dot(qv, k_vals);
            }
            for (uint d = (head_dim / 4) * 4 + tid; d < head_dim; d += tg_size) {
                partial_dot += Q[q_offset + d] * q4_0_read(K_cache, k_head_base, kv, row_bytes, d);
            }
            partial_dot = simd_sum(partial_dot);
            if (lane == 0) {
                shared_scores[kv_offset * num_simds + simd_id] = partial_dot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float m_old = shared_update[0];
            float l_old = shared_update[1];
            float m_new = m_old;

            float tile_scores[TILE_KV];
            for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                float s = 0.0f;
                for (uint s_id = 0; s_id < num_simds; s_id++) {
                    s += shared_scores[kv_offset * num_simds + s_id];
                }
                tile_scores[kv_offset] = s * scale;
                m_new = max(m_new, tile_scores[kv_offset]);
            }

            float tile_sum = 0.0f;
            for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                float e = exp(tile_scores[kv_offset] - m_new);
                shared_exp[kv_offset] = e;
                tile_sum += e;
            }

            float l_new = l_old * exp(m_old - m_new) + tile_sum;
            shared_update[0] = m_new;
            shared_update[1] = l_new;
            shared_update[2] = l_new > 0.0f ? (l_old * exp(m_old - m_new)) / l_new : 0.0f;
            shared_update[3] = l_new > 0.0f ? 1.0f / l_new : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float old_factor = shared_update[2];
        float inv_l_new = shared_update[3];

        for (uint d = tid * 4; d + 3 < head_dim; d += tg_size * 4) {
            uint out_idx = out_offset + d;
            float4 ov = *reinterpret_cast<device float4*>(&output[out_idx]);
            float4 acc = float4(0.0f);
            for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                uint kv = kv_tile + kv_offset;
                acc += shared_exp[kv_offset] * q4_0_read4(V_cache, v_head_base, kv, row_bytes, d);
            }
            ov = ov * old_factor + acc * inv_l_new;
            *reinterpret_cast<device float4*>(&output[out_idx]) = ov;
        }
        for (uint d = (head_dim / 4) * 4 + tid; d < head_dim; d += tg_size) {
            uint out_idx = out_offset + d;
            float acc = 0.0f;
            for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                uint kv = kv_tile + kv_offset;
                acc += shared_exp[kv_offset] * q4_0_read(V_cache, v_head_base, kv, row_bytes, d);
            }
            output[out_idx] = output[out_idx] * old_factor + acc * inv_l_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ─── Transpose: (seq, heads, head_dim) → (heads, seq, head_dim) ─────────────

kernel void transpose_shd_to_hsd(
    device const float* input [[buffer(0)]],   // (seq_len, num_heads, head_dim)
    device float* output [[buffer(1)]],        // (num_heads, seq_len, head_dim)
    constant uint& seq_len [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = seq_len * num_heads * head_dim;
    if (gid >= total) return;
    
    // Input index: (s, h, d)
    uint s = gid / (num_heads * head_dim);
    uint remainder = gid % (num_heads * head_dim);
    uint h = remainder / head_dim;
    uint d = remainder % head_dim;
    
    // Output index: (h, s, d)
    uint out_idx = (h * seq_len + s) * head_dim + d;
    output[out_idx] = input[gid];
}

// ─── Transpose: (heads, seq, head_dim) → (seq, heads, head_dim) ─────────────

kernel void transpose_hsd_to_shd(
    device const float* input [[buffer(0)]],   // (num_heads, seq_len, head_dim)
    device float* output [[buffer(1)]],        // (seq_len, num_heads, head_dim)
    constant uint& seq_len [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = seq_len * num_heads * head_dim;
    if (gid >= total) return;
    
    // Input index: (h, s, d)
    uint h = gid / (seq_len * head_dim);
    uint remainder = gid % (seq_len * head_dim);
    uint s = remainder / head_dim;
    uint d = remainder % head_dim;
    
    // Output index: (s, h, d)
    uint out_idx = (s * num_heads + h) * head_dim + d;
    output[out_idx] = input[gid];
}

// ─── Fused logit softcap + min-p sampling (single threadgroup) ──────────────
// Applies final logit softcapping, temperature, and min-p filtering, then draws
// a sample using the Gumbel-max trick (argmax of e_i + g_i). This keeps the
// sampled token on the GPU so the CPU only reads back 4 bytes instead of the
// full vocab logits, and avoids a CPU-side softmax over the vocabulary.
//
// Gumbel-max: sampling i ~ softmax(e) is equivalent to argmax_i(e_i + g_i),
// g_i = -log(-log(u_i)). min-p keeps tokens with prob >= min_p * max_prob,
// which in shifted-logit space is e_i >= log(min_p) (e_max = 0).

#define SAMPLE_TG 256

inline float sample_rng_uniform(uint seed, uint idx) {
    // Cheap hash → uniform in the open interval (0, 1).
    uint h = seed ^ (idx * 2654435761u);
    h ^= h >> 16; h *= 0x7feb352du;
    h ^= h >> 15; h *= 0x846ca68bu;
    h ^= h >> 16;
    return (float(h) + 0.5f) / 4294967296.0f;
}

kernel void sample_token(
    device const float* logits [[buffer(0)]],
    device uint* out_token [[buffer(1)]],
    constant uint& V [[buffer(2)]],
    constant float& cap [[buffer(3)]],
    constant float& temperature [[buffer(4)]],
    constant float& min_p [[buffer(5)]],
    constant uint& seed [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint nthreads [[threads_per_threadgroup]]
) {
    threadgroup float s_max[SAMPLE_TG];
    threadgroup float s_key[SAMPLE_TG];
    threadgroup uint  s_idx[SAMPLE_TG];

    bool greedy = temperature < 1e-6f;
    float inv_temp = greedy ? 1.0f : (1.0f / temperature);

    // Phase 1: maximum softcapped logit (for numerical stability + min-p ref).
    float local_max = -INFINITY;
    for (uint i = tid; i < V; i += nthreads) {
        float x = clamp(logits[i] / cap, -10.0f, 10.0f);
        local_max = max(local_max, cap * tanh(x));
    }
    s_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) s_max[tid] = max(s_max[tid], s_max[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float gmax = s_max[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Gumbel-max over the min-p-filtered set.
    float log_min_p = (min_p > 0.0f) ? log(min_p) : -INFINITY;
    float best_key = -INFINITY;
    uint best_idx = 0;
    for (uint i = tid; i < V; i += nthreads) {
        float x = clamp(logits[i] / cap, -10.0f, 10.0f);
        float sl = cap * tanh(x);
        float key;
        if (greedy) {
            key = sl;
        } else {
            float e = (sl - gmax) * inv_temp;
            if (e < log_min_p) continue;
            float u = sample_rng_uniform(seed, i);
            float g = -log(-log(u));
            key = e + g;
        }
        if (key > best_key) {
            best_key = key;
            best_idx = i;
        }
    }
    s_key[tid] = best_key;
    s_idx[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_key[tid + s] > s_key[tid]) {
                s_key[tid] = s_key[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out_token[0] = s_idx[0];
}

// ─── FlashAttention-style tiled decode + causal attention ─────────────────────
// Online softmax over KV tiles (no materialized S×S matrix). vs legacy kernels:
//   • Q vector cached in threadgroup memory (read once per head)
//   • TILE_KV=32 positions per tile (fewer barriers than TILE_KV=4 or per-token)
//   • 8 SIMD groups score up to 8 KV positions in parallel per wave
//   • 256-thread threadgroups (legacy decode used 64 threads)

constant uint FLASH_TILE_KV  = 32;
constant uint FLASH_TG_SIZE  = 256;
constant uint FLASH_MAX_HEAD = 512;

inline void flash_zero_output(
    device float* output,
    uint out_base,
    uint head_dim,
    uint tid,
    uint tg_size
) {
    for (uint d = tid * 4; d + 3 < head_dim; d += tg_size * 4) {
        *reinterpret_cast<device float4*>(&output[out_base + d]) = float4(0.0f);
    }
    for (uint d = (head_dim / 4) * 4 + tid; d < head_dim; d += tg_size) {
        output[out_base + d] = 0.0f;
    }
}

inline void flash_load_q(
    device const float* Q,
    uint q_offset,
    threadgroup float* shared_q,
    uint head_dim,
    uint tid,
    uint tg_size
) {
    for (uint d = tid; d < head_dim; d += tg_size) {
        shared_q[d] = Q[q_offset + d];
    }
}

inline float flash_dot_q4_k_cached(
    device const uchar* K_cache,
    uint k_head_base,
    uint pos,
    uint row_bytes,
    uint head_dim,
    threadgroup float* shared_q,
    uint lane
) {
    float partial = 0.0f;
    for (uint d = lane * 4; d + 3 < head_dim; d += SIMD_SIZE * 4) {
        float4 qv = float4(shared_q[d], shared_q[d + 1], shared_q[d + 2], shared_q[d + 3]);
        float4 kv = q4_0_read4(K_cache, k_head_base, pos, row_bytes, d);
        partial += dot(qv, kv);
    }
    for (uint d = (head_dim / 4) * 4 + lane; d < head_dim; d += SIMD_SIZE) {
        partial += shared_q[d] * q4_0_read(K_cache, k_head_base, pos, row_bytes, d);
    }
    return partial;
}

inline float flash_dot_f16_k_cached(
    device const half* K_row,
    uint head_dim,
    threadgroup float* shared_q,
    uint lane
) {
    float partial = 0.0f;
    for (uint d = lane; d < head_dim; d += SIMD_SIZE) {
        partial += shared_q[d] * float(K_row[d]);
    }
    return partial;
}

inline void flash_softmax_tile(
    threadgroup float* shared_scores,
    threadgroup float* shared_exp,
    threadgroup float* shared_update,
    uint tile_count
) {
    float m_old = shared_update[0];
    float l_old = shared_update[1];
    float m_new = m_old;
    for (uint i = 0; i < tile_count; i++) {
        m_new = max(m_new, shared_scores[i]);
    }
    float tile_sum = 0.0f;
    for (uint i = 0; i < tile_count; i++) {
        float e = exp(shared_scores[i] - m_new);
        shared_exp[i] = e;
        tile_sum += e;
    }
    float l_new = l_old * exp(m_old - m_new) + tile_sum;
    shared_update[0] = m_new;
    shared_update[1] = l_new;
    shared_update[2] = l_new > 0.0f ? (l_old * exp(m_old - m_new)) / l_new : 0.0f;
    shared_update[3] = l_new > 0.0f ? 1.0f / l_new : 0.0f;
}

inline void flash_accum_v_q4(
    device float* output,
    uint out_base,
    device const uchar* V_cache,
    uint v_head_base,
    uint kv_start,
    uint kv_tile,
    uint tile_count,
    uint row_bytes,
    uint head_dim,
    threadgroup float* shared_exp,
    float old_factor,
    float inv_l,
    uint tid,
    uint tg_size
) {
    for (uint d = tid * 4; d + 3 < head_dim; d += tg_size * 4) {
        float4 ov = *reinterpret_cast<device float4*>(&output[out_base + d]);
        float4 acc = float4(0.0f);
        for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
            uint actual_pos = kv_start + kv_tile + kv_offset;
            acc += shared_exp[kv_offset]
                * q4_0_read4(V_cache, v_head_base, actual_pos, row_bytes, d);
        }
        ov = ov * old_factor + acc * inv_l;
        *reinterpret_cast<device float4*>(&output[out_base + d]) = ov;
    }
    for (uint d = (head_dim / 4) * 4 + tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
            uint actual_pos = kv_start + kv_tile + kv_offset;
            acc += shared_exp[kv_offset]
                * q4_0_read(V_cache, v_head_base, actual_pos, row_bytes, d);
        }
        output[out_base + d] = output[out_base + d] * old_factor + acc * inv_l;
    }
}

inline void flash_accum_v_f16(
    device float* output,
    uint out_base,
    device const half* V_cache,
    uint v_head_base,
    uint head_dim,
    uint kv_start,
    uint kv_tile,
    uint tile_count,
    threadgroup float* shared_exp,
    float old_factor,
    float inv_l,
    uint tid,
    uint tg_size
) {
    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
            uint actual_pos = kv_start + kv_tile + kv_offset;
            acc += shared_exp[kv_offset]
                * float(V_cache[v_head_base + actual_pos * head_dim + d]);
        }
        output[out_base + d] = output[out_base + d] * old_factor + acc * inv_l;
    }
}

// ─── Fused KV append + flash decode helpers (Q4_0) ───────────────────────────
// Token at cur_seq is read from f32 K/V; prior tokens from Q4 cache. Quantize
// and write the new K/V row once per kv head at kernel end.

inline void q4_0_append_group_tg(
    threadgroup const float* src,
    uint src_base,
    device uchar* cache,
    uint head_base,
    uint pos,
    uint row_bytes,
    uint g
) {
    uint base_offset = head_base + pos * row_bytes + g * 18;
    float max_abs = 0.0f;
    for (uint d = 0; d < 32; d++) {
        float a = fabs(src[src_base + g * 32 + d]);
        if (a > max_abs) max_abs = a;
    }
    float scale = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
    float inv_scale = 1.0f / scale;
    *reinterpret_cast<device half*>(&cache[base_offset]) = half(scale);
    for (uint i = 0; i < 16; i++) {
        int q_lo = clamp(int(round(src[src_base + g * 32 + i] * inv_scale)) + 8, 0, 15);
        int q_hi = clamp(int(round(src[src_base + g * 32 + i + 16] * inv_scale)) + 8, 0, 15);
        cache[base_offset + 2 + i] = uchar(q_lo | (q_hi << 4));
    }
}

inline void q4_0_append_group_f32(
    device const float* src,
    uint src_base,
    device uchar* cache,
    uint head_base,
    uint pos,
    uint row_bytes,
    uint g
) {
    uint base_offset = head_base + pos * row_bytes + g * 18;
    float max_abs = 0.0f;
    for (uint d = 0; d < 32; d++) {
        float a = fabs(src[src_base + g * 32 + d]);
        if (a > max_abs) max_abs = a;
    }
    float scale = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
    float inv_scale = 1.0f / scale;
    *reinterpret_cast<device half*>(&cache[base_offset]) = half(scale);
    for (uint i = 0; i < 16; i++) {
        int q_lo = clamp(int(round(src[src_base + g * 32 + i] * inv_scale)) + 8, 0, 15);
        int q_hi = clamp(int(round(src[src_base + g * 32 + i + 16] * inv_scale)) + 8, 0, 15);
        cache[base_offset + 2 + i] = uchar(q_lo | (q_hi << 4));
    }
}

inline float flash_dot_k_fused(
    device const uchar* K_cache,
    uint k_head_base,
    uint pos,
    uint row_bytes,
    uint head_dim,
    device const float* K_f32,
    uint kv_h,
    uint cur_seq,
    threadgroup float* shared_q,
    uint lane
) {
    if (pos == cur_seq) {
        float partial = 0.0f;
        uint kb = kv_h * head_dim;
        for (uint d = lane * 4; d + 3 < head_dim; d += SIMD_SIZE * 4) {
            float4 qv = float4(shared_q[d], shared_q[d + 1], shared_q[d + 2], shared_q[d + 3]);
            float4 kv = float4(K_f32[kb + d], K_f32[kb + d + 1], K_f32[kb + d + 2], K_f32[kb + d + 3]);
            partial += dot(qv, kv);
        }
        for (uint d = (head_dim / 4) * 4 + lane; d < head_dim; d += SIMD_SIZE) {
            partial += shared_q[d] * K_f32[kb + d];
        }
        return partial;
    }
    return flash_dot_q4_k_cached(K_cache, k_head_base, pos, row_bytes, head_dim, shared_q, lane);
}

inline void flash_accum_v_q4_fused(
    device float* output,
    uint out_base,
    device const uchar* V_cache,
    uint v_head_base,
    uint kv_start,
    uint kv_tile,
    uint tile_count,
    uint row_bytes,
    uint head_dim,
    device const float* V_f32,
    uint kv_h,
    uint cur_seq,
    threadgroup float* shared_exp,
    float old_factor,
    float inv_l,
    uint tid,
    uint tg_size
) {
    uint vb = kv_h * head_dim;
    for (uint d = tid * 4; d + 3 < head_dim; d += tg_size * 4) {
        float4 ov = *reinterpret_cast<device float4*>(&output[out_base + d]);
        float4 acc = float4(0.0f);
        for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
            uint actual_pos = kv_start + kv_tile + kv_offset;
            float4 vv = (actual_pos == cur_seq)
                ? float4(V_f32[vb + d], V_f32[vb + d + 1], V_f32[vb + d + 2], V_f32[vb + d + 3])
                : q4_0_read4(V_cache, v_head_base, actual_pos, row_bytes, d);
            acc += shared_exp[kv_offset] * vv;
        }
        ov = ov * old_factor + acc * inv_l;
        *reinterpret_cast<device float4*>(&output[out_base + d]) = ov;
    }
    for (uint d = (head_dim / 4) * 4 + tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
            uint actual_pos = kv_start + kv_tile + kv_offset;
            float vv = (actual_pos == cur_seq)
                ? V_f32[vb + d]
                : q4_0_read(V_cache, v_head_base, actual_pos, row_bytes, d);
            acc += shared_exp[kv_offset] * vv;
        }
        output[out_base + d] = output[out_base + d] * old_factor + acc * inv_l;
    }
}

// ─── Flash decode: single query token vs KV cache (Q4_0) ─────────────────────

kernel void attention_flash_decode_q4_0(
    device const float* Q [[buffer(0)]],
    device const uchar* K_cache [[buffer(1)]],
    device const uchar* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& capacity [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& kv_start [[buffer(11)]],
    constant uint& groups_per_row [[buffer(12)]],
    constant uint& row_bytes [[buffer(13)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)groups_per_row;
    uint h = tgid;
    if (h >= num_heads) return;

    uint kv_h = h / num_kv_groups;
    uint q_offset = h * head_dim;
    uint k_head_base = kv_h * capacity * row_bytes;
    uint v_head_base = kv_h * capacity * row_bytes;
    uint num_simds = FLASH_TG_SIZE / SIMD_SIZE;

    threadgroup float shared_q[FLASH_MAX_HEAD];
    threadgroup float shared_scores[FLASH_TILE_KV];
    threadgroup float shared_exp[FLASH_TILE_KV];
    threadgroup float shared_update[4];

    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    flash_zero_output(output, q_offset, head_dim, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    flash_load_q(Q, q_offset, shared_q, head_dim, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv_tile = 0; kv_tile < kv_seq; kv_tile += FLASH_TILE_KV) {
        uint tile_count = min(FLASH_TILE_KV, kv_seq - kv_tile);

        for (uint wave = 0; wave < tile_count; wave += num_simds) {
            uint kv_offset = wave + sgid;
            if (kv_offset < tile_count) {
                uint actual_pos = kv_start + kv_tile + kv_offset;
                float partial = flash_dot_q4_k_cached(
                    K_cache, k_head_base, actual_pos, row_bytes, head_dim, shared_q, lane);
                partial = simd_sum(partial);
                if (lane == 0) {
                    shared_scores[kv_offset] = partial * scale;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            flash_softmax_tile(shared_scores, shared_exp, shared_update, tile_count);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        flash_accum_v_q4(
            output, q_offset, V_cache, v_head_base,
            kv_start, kv_tile, tile_count, row_bytes, head_dim,
            shared_exp, shared_update[2], shared_update[3],
            tid, FLASH_TG_SIZE);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ─── Flash decode Q4_0: head-dim specialized (Gemma4 sliding h256 / global h512) ─
// h256 uses TILE_KV=256 so the sliding window (≤512) fits in two tiles.

template<uint HEAD_DIM>
inline void flash_zero_output_hd(
    device float* output,
    uint out_base,
    uint tid,
    uint tg_size
) {
    for (uint d = tid * 4; d + 3 < HEAD_DIM; d += tg_size * 4) {
        *reinterpret_cast<device float4*>(&output[out_base + d]) = float4(0.0f);
    }
    for (uint d = (HEAD_DIM / 4) * 4 + tid; d < HEAD_DIM; d += tg_size) {
        output[out_base + d] = 0.0f;
    }
}

template<uint HEAD_DIM>
inline void flash_load_q_hd(
    device const float* Q,
    uint q_offset,
    threadgroup float* shared_q,
    uint tid,
    uint tg_size
) {
    for (uint d = tid; d < HEAD_DIM; d += tg_size) {
        shared_q[d] = Q[q_offset + d];
    }
}

// QK-norm + RoPE fused into flash Q load (one head per threadgroup).
template<uint HEAD_DIM>
inline void flash_load_q_qknorm_rope_hd(
    device const float* Q_raw,
    device const float* q_norm_weight,
    device const float* cos_buf,
    device const float* sin_buf,
    float eps,
    uint q_offset,
    threadgroup float* shared_q,
    threadgroup float* shared_tmp,
    uint tid,
    uint tg_size
) {
    float partial = 0.0f;
    for (uint i = tid; i < HEAD_DIM; i += tg_size) {
        float val = Q_raw[q_offset + i];
        partial += val * val;
    }
    shared_tmp[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_tmp[tid] += shared_tmp[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shared_tmp[0] / float(HEAD_DIM) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < HEAD_DIM; d += tg_size) {
        shared_q[d] = Q_raw[q_offset + d] * inv_rms * q_norm_weight[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint half_dim = HEAD_DIM / 2u;
    for (uint d = tid; d < half_dim; d += tg_size) {
        float q1 = shared_q[d];
        float q2 = shared_q[d + half_dim];
        float c = cos_buf[d];
        float s = sin_buf[d];
        shared_q[d] = q1 * c - q2 * s;
        shared_q[d + half_dim] = q2 * c + q1 * s;
    }
}

// K-norm + RoPE fused into threadgroup K scratch (one kv head per threadgroup).
template<uint HEAD_DIM>
inline void flash_prepare_k_norm_rope_hd(
    device const float* K_raw,
    device const float* k_norm_weight,
    device const float* cos_buf,
    device const float* sin_buf,
    float eps,
    uint k_offset,
    threadgroup float* shared_k,
    threadgroup float* shared_tmp,
    uint tid,
    uint tg_size
) {
    float partial = 0.0f;
    for (uint i = tid; i < HEAD_DIM; i += tg_size) {
        float val = K_raw[k_offset + i];
        partial += val * val;
    }
    shared_tmp[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_tmp[tid] += shared_tmp[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shared_tmp[0] / float(HEAD_DIM) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < HEAD_DIM; d += tg_size) {
        shared_k[d] = K_raw[k_offset + d] * inv_rms * k_norm_weight[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint half_dim = HEAD_DIM / 2u;
    for (uint d = tid; d < half_dim; d += tg_size) {
        float k1 = shared_k[d];
        float k2 = shared_k[d + half_dim];
        float c = cos_buf[d];
        float s = sin_buf[d];
        shared_k[d] = k1 * c - k2 * s;
        shared_k[d + half_dim] = k2 * c + k1 * s;
    }
}

// V-norm (no weight) into threadgroup V scratch.
template<uint HEAD_DIM>
inline void flash_prepare_v_norm_hd(
    device const float* V_raw,
    float eps,
    uint v_offset,
    threadgroup float* shared_v,
    threadgroup float* shared_tmp,
    uint tid,
    uint tg_size
) {
    float partial = 0.0f;
    for (uint i = tid; i < HEAD_DIM; i += tg_size) {
        float val = V_raw[v_offset + i];
        partial += val * val;
    }
    shared_tmp[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_tmp[tid] += shared_tmp[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shared_tmp[0] / float(HEAD_DIM) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < HEAD_DIM; d += tg_size) {
        shared_v[d] = V_raw[v_offset + d] * inv_rms;
    }
}

template<uint HEAD_DIM>
inline float flash_dot_q4_k_hd(
    device const uchar* K_cache,
    uint k_head_base,
    uint pos,
    uint row_bytes,
    threadgroup float* shared_q,
    uint lane
) {
    float partial = 0.0f;
    for (uint d = lane * 4; d + 3 < HEAD_DIM; d += SIMD_SIZE * 4) {
        float4 qv = float4(shared_q[d], shared_q[d + 1], shared_q[d + 2], shared_q[d + 3]);
        float4 kv = q4_0_read4(K_cache, k_head_base, pos, row_bytes, d);
        partial += dot(qv, kv);
    }
    for (uint d = (HEAD_DIM / 4) * 4 + lane; d < HEAD_DIM; d += SIMD_SIZE) {
        partial += shared_q[d] * q4_0_read(K_cache, k_head_base, pos, row_bytes, d);
    }
    return partial;
}

template<uint HEAD_DIM>
inline float flash_dot_k_fused_hd(
    device const uchar* K_cache,
    uint k_head_base,
    uint pos,
    uint row_bytes,
    device const float* K_f32,
    uint kv_h,
    uint cur_seq,
    threadgroup float* shared_q,
    uint lane
) {
    if (pos == cur_seq) {
        float partial = 0.0f;
        uint kb = kv_h * HEAD_DIM;
        for (uint d = lane * 4; d + 3 < HEAD_DIM; d += SIMD_SIZE * 4) {
            float4 qv = float4(shared_q[d], shared_q[d + 1], shared_q[d + 2], shared_q[d + 3]);
            float4 kv = float4(K_f32[kb + d], K_f32[kb + d + 1], K_f32[kb + d + 2], K_f32[kb + d + 3]);
            partial += dot(qv, kv);
        }
        for (uint d = (HEAD_DIM / 4) * 4 + lane; d < HEAD_DIM; d += SIMD_SIZE) {
            partial += shared_q[d] * K_f32[kb + d];
        }
        return partial;
    }
    return flash_dot_q4_k_hd<HEAD_DIM>(K_cache, k_head_base, pos, row_bytes, shared_q, lane);
}

template<uint HEAD_DIM>
inline float flash_dot_k_shared_hd(
    device const uchar* K_cache,
    uint k_head_base,
    uint pos,
    uint row_bytes,
    threadgroup float* shared_k,
    uint cur_seq,
    threadgroup float* shared_q,
    uint lane
) {
    if (pos == cur_seq) {
        float partial = 0.0f;
        for (uint d = lane * 4; d + 3 < HEAD_DIM; d += SIMD_SIZE * 4) {
            float4 qv = float4(shared_q[d], shared_q[d + 1], shared_q[d + 2], shared_q[d + 3]);
            float4 kv = float4(shared_k[d], shared_k[d + 1], shared_k[d + 2], shared_k[d + 3]);
            partial += dot(qv, kv);
        }
        for (uint d = (HEAD_DIM / 4) * 4 + lane; d < HEAD_DIM; d += SIMD_SIZE) {
            partial += shared_q[d] * shared_k[d];
        }
        return partial;
    }
    return flash_dot_q4_k_hd<HEAD_DIM>(K_cache, k_head_base, pos, row_bytes, shared_q, lane);
}

template<uint HEAD_DIM, uint TILE_KV>
inline void flash_accum_v_q4_fused_hd(
    device float* output,
    uint out_base,
    device const uchar* V_cache,
    uint v_head_base,
    uint kv_start,
    uint kv_tile,
    uint tile_count,
    uint row_bytes,
    device const float* V_f32,
    uint kv_h,
    uint cur_seq,
    threadgroup float* shared_exp,
    float old_factor,
    float inv_l,
    uint tid,
    uint tg_size
) {
    uint vb = kv_h * HEAD_DIM;
    for (uint d = tid * 4; d + 3 < HEAD_DIM; d += tg_size * 4) {
        float4 ov = *reinterpret_cast<device float4*>(&output[out_base + d]);
        float4 acc = float4(0.0f);
        for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
            uint actual_pos = kv_start + kv_tile + kv_offset;
            float4 vv = (actual_pos == cur_seq)
                ? float4(V_f32[vb + d], V_f32[vb + d + 1], V_f32[vb + d + 2], V_f32[vb + d + 3])
                : q4_0_read4(V_cache, v_head_base, actual_pos, row_bytes, d);
            acc += shared_exp[kv_offset] * vv;
        }
        ov = ov * old_factor + acc * inv_l;
        *reinterpret_cast<device float4*>(&output[out_base + d]) = ov;
    }
    for (uint d = (HEAD_DIM / 4) * 4 + tid; d < HEAD_DIM; d += tg_size) {
        float acc = 0.0f;
        for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
            uint actual_pos = kv_start + kv_tile + kv_offset;
            float vv = (actual_pos == cur_seq)
                ? V_f32[vb + d]
                : q4_0_read(V_cache, v_head_base, actual_pos, row_bytes, d);
            acc += shared_exp[kv_offset] * vv;
        }
        output[out_base + d] = output[out_base + d] * old_factor + acc * inv_l;
    }
}

template<uint HEAD_DIM, uint TILE_KV>
inline void flash_accum_v_q4_shared_fused_hd(
    device float* output,
    uint out_base,
    device const uchar* V_cache,
    uint v_head_base,
    uint kv_start,
    uint kv_tile,
    uint tile_count,
    uint row_bytes,
    threadgroup float* shared_v,
    uint cur_seq,
    threadgroup float* shared_exp,
    float old_factor,
    float inv_l,
    uint tid,
    uint tg_size
) {
    for (uint d = tid * 4; d + 3 < HEAD_DIM; d += tg_size * 4) {
        float4 ov = *reinterpret_cast<device float4*>(&output[out_base + d]);
        float4 acc = float4(0.0f);
        for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
            uint actual_pos = kv_start + kv_tile + kv_offset;
            float4 vv = (actual_pos == cur_seq)
                ? float4(shared_v[d], shared_v[d + 1], shared_v[d + 2], shared_v[d + 3])
                : q4_0_read4(V_cache, v_head_base, actual_pos, row_bytes, d);
            acc += shared_exp[kv_offset] * vv;
        }
        ov = ov * old_factor + acc * inv_l;
        *reinterpret_cast<device float4*>(&output[out_base + d]) = ov;
    }
    for (uint d = (HEAD_DIM / 4) * 4 + tid; d < HEAD_DIM; d += tg_size) {
        float acc = 0.0f;
        for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
            uint actual_pos = kv_start + kv_tile + kv_offset;
            float vv = (actual_pos == cur_seq)
                ? shared_v[d]
                : q4_0_read(V_cache, v_head_base, actual_pos, row_bytes, d);
            acc += shared_exp[kv_offset] * vv;
        }
        output[out_base + d] = output[out_base + d] * old_factor + acc * inv_l;
    }
}

template<uint HEAD_DIM, uint TILE_KV>
inline void flash_accum_v_q4_hd(
    device float* output,
    uint out_base,
    device const uchar* V_cache,
    uint v_head_base,
    uint kv_start,
    uint kv_tile,
    uint tile_count,
    uint row_bytes,
    threadgroup float* shared_exp,
    float old_factor,
    float inv_l,
    uint tid,
    uint tg_size
) {
    for (uint d = tid * 4; d + 3 < HEAD_DIM; d += tg_size * 4) {
        float4 ov = *reinterpret_cast<device float4*>(&output[out_base + d]);
        float4 acc = float4(0.0f);
        for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
            uint actual_pos = kv_start + kv_tile + kv_offset;
            acc += shared_exp[kv_offset]
                * q4_0_read4(V_cache, v_head_base, actual_pos, row_bytes, d);
        }
        ov = ov * old_factor + acc * inv_l;
        *reinterpret_cast<device float4*>(&output[out_base + d]) = ov;
    }
    for (uint d = (HEAD_DIM / 4) * 4 + tid; d < HEAD_DIM; d += tg_size) {
        float acc = 0.0f;
        for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
            uint actual_pos = kv_start + kv_tile + kv_offset;
            acc += shared_exp[kv_offset]
                * q4_0_read(V_cache, v_head_base, actual_pos, row_bytes, d);
        }
        output[out_base + d] = output[out_base + d] * old_factor + acc * inv_l;
    }
}

template<uint HEAD_DIM, uint TILE_KV>
void flash_decode_q4_0_hd_body(
    device const float* Q,
    device const uchar* K_cache,
    device const uchar* V_cache,
    device float* output,
    uint h,
    uint num_heads,
    uint num_kv_groups,
    uint capacity,
    uint row_bytes,
    uint kv_seq,
    uint kv_start,
    float scale,
    uint tid,
    uint sgid,
    uint lane,
    threadgroup float* shared_q,
    threadgroup float* shared_scores,
    threadgroup float* shared_exp,
    threadgroup float* shared_update
) {
    if (h >= num_heads) return;

    uint kv_h = h / num_kv_groups;
    uint q_offset = h * HEAD_DIM;
    uint k_head_base = kv_h * capacity * row_bytes;
    uint v_head_base = kv_h * capacity * row_bytes;
    uint num_simds = FLASH_TG_SIZE / SIMD_SIZE;

    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    flash_zero_output_hd<HEAD_DIM>(output, q_offset, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    flash_load_q_hd<HEAD_DIM>(Q, q_offset, shared_q, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv_tile = 0; kv_tile < kv_seq; kv_tile += TILE_KV) {
        uint tile_count = min(TILE_KV, kv_seq - kv_tile);

        for (uint wave = 0; wave < tile_count; wave += num_simds) {
            uint kv_offset = wave + sgid;
            if (kv_offset < tile_count) {
                uint actual_pos = kv_start + kv_tile + kv_offset;
                float partial = flash_dot_q4_k_hd<HEAD_DIM>(
                    K_cache, k_head_base, actual_pos, row_bytes, shared_q, lane);
                partial = simd_sum(partial);
                if (lane == 0) {
                    shared_scores[kv_offset] = partial * scale;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            flash_softmax_tile(shared_scores, shared_exp, shared_update, tile_count);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        flash_accum_v_q4_hd<HEAD_DIM, TILE_KV>(
            output, q_offset, V_cache, v_head_base,
            kv_start, kv_tile, tile_count, row_bytes,
            shared_exp, shared_update[2], shared_update[3],
            tid, FLASH_TG_SIZE);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void attention_flash_decode_q4_0_h128(
    device const float* Q [[buffer(0)]],
    device const uchar* K_cache [[buffer(1)]],
    device const uchar* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& capacity [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& kv_start [[buffer(11)]],
    constant uint& groups_per_row [[buffer(12)]],
    constant uint& row_bytes [[buffer(13)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)num_kv_heads;
    (void)head_dim;
    (void)groups_per_row;
    threadgroup float shared_q[128];
    threadgroup float shared_scores[128];
    threadgroup float shared_exp[128];
    threadgroup float shared_update[4];
    flash_decode_q4_0_hd_body<128, 128>(
        Q, K_cache, V_cache, output, tgid, num_heads, num_kv_groups, capacity, row_bytes,
        kv_seq, kv_start, scale, tid, sgid, lane,
        shared_q, shared_scores, shared_exp, shared_update);
}

kernel void attention_flash_decode_q4_0_h256(
    device const float* Q [[buffer(0)]],
    device const uchar* K_cache [[buffer(1)]],
    device const uchar* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& capacity [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& kv_start [[buffer(11)]],
    constant uint& groups_per_row [[buffer(12)]],
    constant uint& row_bytes [[buffer(13)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)num_kv_heads;
    (void)head_dim;
    (void)groups_per_row;
    threadgroup float shared_q[256];
    threadgroup float shared_scores[256];
    threadgroup float shared_exp[256];
    threadgroup float shared_update[4];
    flash_decode_q4_0_hd_body<256, 256>(
        Q, K_cache, V_cache, output, tgid, num_heads, num_kv_groups, capacity, row_bytes,
        kv_seq, kv_start, scale, tid, sgid, lane,
        shared_q, shared_scores, shared_exp, shared_update);
}

kernel void attention_flash_decode_q4_0_h512(
    device const float* Q [[buffer(0)]],
    device const uchar* K_cache [[buffer(1)]],
    device const uchar* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& capacity [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& kv_start [[buffer(11)]],
    constant uint& groups_per_row [[buffer(12)]],
    constant uint& row_bytes [[buffer(13)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)num_kv_heads;
    (void)head_dim;
    (void)groups_per_row;
    threadgroup float shared_q[512];
    threadgroup float shared_scores[256];
    threadgroup float shared_exp[256];
    threadgroup float shared_update[4];
    flash_decode_q4_0_hd_body<512, 256>(
        Q, K_cache, V_cache, output, tgid, num_heads, num_kv_groups, capacity, row_bytes,
        kv_seq, kv_start, scale, tid, sgid, lane,
        shared_q, shared_scores, shared_exp, shared_update);
}

// ─── Tiled GQA-aware flash decode for Q4_0 KV cache ─────────────────────────
// One threadgroup per KV head; all query heads sharing that KV head are
// processed together.  Each query head gets (FLASH_TG_SIZE / SIMD_SIZE) /
// num_kv_groups simdgroups.  The quantized K tile and V tile are each loaded
// into threadgroup memory once per tile and reused by all query heads,
// cutting KV cache device reads by num_kv_groups.

constant uint GQA_MAX_GROUPS = 4;
constant uint GQA_TILE_KV = 32;
constant uint GQA_MAX_ROW_BYTES = (FLASH_MAX_HEAD / 32) * 18;

inline void flash_zero_output_gqa(
    device float* output,
    uint out_base,
    uint head_dim,
    uint local_tid,
    uint threads_per_head
) {
    for (uint d = local_tid * 4; d + 3 < head_dim; d += threads_per_head * 4) {
        *reinterpret_cast<device float4*>(&output[out_base + d]) = float4(0.0f);
    }
    for (uint d = (head_dim / 4) * 4 + local_tid; d < head_dim; d += threads_per_head) {
        output[out_base + d] = 0.0f;
    }
}

inline void flash_load_q_gqa(
    device const float* Q,
    uint q_offset,
    threadgroup float* shared_q,
    uint head_dim,
    uint local_tid,
    uint threads_per_head
) {
    for (uint d = local_tid; d < head_dim; d += threads_per_head) {
        shared_q[d] = Q[q_offset + d];
    }
}

inline float q4_0_read_tg(threadgroup const uchar* qs, uint d) {
    uint g = d / 32;
    uint e = d % 32;
    uint offset = g * 18;
    float scale = float(*reinterpret_cast<threadgroup const half*>(&qs[offset]));
    if (e < 16) {
        return float(int(qs[offset + 2 + e] & 0xF) - 8) * scale;
    } else {
        return float(int(qs[offset + 2 + e - 16] >> 4) - 8) * scale;
    }
}

inline float4 q4_0_read4_tg(threadgroup const uchar* qs, uint d) {
    uint g = d / 32;
    uint e = d % 32;
    uint offset = g * 18;
    float scale = float(*reinterpret_cast<threadgroup const half*>(&qs[offset]));
    if (e + 3 < 16) {
        return float4(
            float(int(qs[offset + 2 + e + 0] & 0xF) - 8) * scale,
            float(int(qs[offset + 2 + e + 1] & 0xF) - 8) * scale,
            float(int(qs[offset + 2 + e + 2] & 0xF) - 8) * scale,
            float(int(qs[offset + 2 + e + 3] & 0xF) - 8) * scale
        );
    }
    if (e >= 16 && e + 3 < 32) {
        uint b = e - 16;
        return float4(
            float(int(qs[offset + 2 + b + 0] >> 4) - 8) * scale,
            float(int(qs[offset + 2 + b + 1] >> 4) - 8) * scale,
            float(int(qs[offset + 2 + b + 2] >> 4) - 8) * scale,
            float(int(qs[offset + 2 + b + 3] >> 4) - 8) * scale
        );
    }
    return float4(
        q4_0_read_tg(qs, d),
        q4_0_read_tg(qs, d + 1),
        q4_0_read_tg(qs, d + 2),
        q4_0_read_tg(qs, d + 3)
    );
}

inline float flash_dot_q4_k_tg(
    threadgroup const uchar* k_row,
    uint head_dim,
    threadgroup float* shared_q,
    uint lane
) {
    float partial = 0.0f;
    for (uint d = lane * 4; d + 3 < head_dim; d += SIMD_SIZE * 4) {
        float4 qv = float4(shared_q[d], shared_q[d + 1], shared_q[d + 2], shared_q[d + 3]);
        float4 kv = q4_0_read4_tg(k_row, d);
        partial += dot(qv, kv);
    }
    for (uint d = (head_dim / 4) * 4 + lane; d < head_dim; d += SIMD_SIZE) {
        partial += shared_q[d] * q4_0_read_tg(k_row, d);
    }
    return partial;
}

inline void flash_load_kv_tile_q4(
    device const uchar* cache,
    uint head_base,
    uint row_start,
    uint tile_count,
    uint row_bytes,
    uint tid,
    uint tg_size,
    threadgroup uchar* shared_tile
) {
    uint total_bytes = tile_count * row_bytes;
    uint base = head_base + row_start * row_bytes;
    for (uint i = tid * 4; i + 3 < total_bytes; i += tg_size * 4) {
        *reinterpret_cast<threadgroup uint*>(&shared_tile[i]) =
            *reinterpret_cast<device const uint*>(&cache[base + i]);
    }
    for (uint i = (total_bytes / 4) * 4 + tid; i < total_bytes; i += tg_size) {
        shared_tile[i] = cache[base + i];
    }
}

inline void flash_accum_v_q4_tg(
    device float* output,
    uint out_base,
    threadgroup const uchar* v_tile,
    uint row_bytes,
    uint head_dim,
    uint tile_count,
    threadgroup float* shared_exp,
    float old_factor,
    float inv_l,
    uint local_tid,
    uint threads_per_head
) {
    for (uint d = local_tid * 4; d + 3 < head_dim; d += threads_per_head * 4) {
        uint out_idx = out_base + d;
        float4 ov = *reinterpret_cast<device float4*>(&output[out_idx]);
        float4 acc = float4(0.0f);
        for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
            acc += shared_exp[kv_offset] * q4_0_read4_tg(v_tile + kv_offset * row_bytes, d);
        }
        ov = ov * old_factor + acc * inv_l;
        *reinterpret_cast<device float4*>(&output[out_idx]) = ov;
    }
    for (uint d = (head_dim / 4) * 4 + local_tid; d < head_dim; d += threads_per_head) {
        uint out_idx = out_base + d;
        float acc = 0.0f;
        for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
            acc += shared_exp[kv_offset] * q4_0_read_tg(v_tile + kv_offset * row_bytes, d);
        }
        output[out_idx] = output[out_idx] * old_factor + acc * inv_l;
    }
}

kernel void attention_flash_decode_q4_0_gqa(
    device const float* Q [[buffer(0)]],
    device const uchar* K_cache [[buffer(1)]],
    device const uchar* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& capacity [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& kv_start [[buffer(11)]],
    constant uint& groups_per_row [[buffer(12)]],
    constant uint& row_bytes [[buffer(13)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)groups_per_row;
    const uint kv_h = tgid;
    if (kv_h >= num_kv_heads) return;
    if (num_kv_groups == 0 || num_kv_groups > GQA_MAX_GROUPS) return;

    const uint num_simds_total = FLASH_TG_SIZE / SIMD_SIZE;
    if (num_kv_groups > num_simds_total) return;
    if (num_simds_total % num_kv_groups != 0) return;

    const uint simds_per_head = num_simds_total / num_kv_groups;
    const uint threads_per_head = simds_per_head * SIMD_SIZE;
    const uint group = sgid / simds_per_head;
    const uint local_sgid = sgid % simds_per_head;
    const uint local_tid = local_sgid * SIMD_SIZE + lane;
    const uint h = kv_h * num_kv_groups + group;
    if (h >= num_heads) return;

    const uint q_offset = h * head_dim;
    const uint k_head_base = kv_h * capacity * row_bytes;
    const uint v_head_base = kv_h * capacity * row_bytes;

    threadgroup float shared_q[GQA_MAX_GROUPS * FLASH_MAX_HEAD];
    threadgroup float shared_scores[GQA_MAX_GROUPS * GQA_TILE_KV];
    threadgroup float shared_exp[GQA_TILE_KV];
    threadgroup float shared_update[GQA_MAX_GROUPS * 4];
    threadgroup uchar shared_kv_tile[GQA_TILE_KV * GQA_MAX_ROW_BYTES];

    threadgroup float* q_head = shared_q + group * FLASH_MAX_HEAD;
    threadgroup float* scores_head = shared_scores + group * GQA_TILE_KV;
    threadgroup float* update_head = shared_update + group * 4;

    if (local_tid == 0) {
        update_head[0] = -INFINITY;
        update_head[1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    flash_zero_output_gqa(output, q_offset, head_dim, local_tid, threads_per_head);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    flash_load_q_gqa(Q, q_offset, q_head, head_dim, local_tid, threads_per_head);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv_tile = 0; kv_tile < kv_seq; kv_tile += GQA_TILE_KV) {
        uint tile_count = min(GQA_TILE_KV, kv_seq - kv_tile);
        uint row_start = kv_start + kv_tile;

        flash_load_kv_tile_q4(
            K_cache, k_head_base, row_start, tile_count, row_bytes,
            tid, FLASH_TG_SIZE, shared_kv_tile);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint wave = local_sgid; wave < tile_count; wave += simds_per_head) {
            uint kv_offset = wave;
            float partial = flash_dot_q4_k_tg(
                shared_kv_tile + kv_offset * row_bytes, head_dim, q_head, lane);
            partial = simd_sum(partial);
            if (lane == 0) {
                scores_head[kv_offset] = partial * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (local_tid == 0) {
            flash_softmax_tile(scores_head, shared_exp, update_head, tile_count);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        flash_load_kv_tile_q4(
            V_cache, v_head_base, row_start, tile_count, row_bytes,
            tid, FLASH_TG_SIZE, shared_kv_tile);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float old_factor = update_head[2];
        float inv_l = update_head[3];
        flash_accum_v_q4_tg(
            output, q_offset, shared_kv_tile, row_bytes, head_dim,
            tile_count, shared_exp, old_factor, inv_l, local_tid, threads_per_head);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template<uint HEAD_DIM, uint TILE_KV>
void flash_decode_fused_q4_0_hd_body(
    device const float* Q,
    device const float* K_f32,
    device const float* V_f32,
    device uchar* K_cache,
    device uchar* V_cache,
    device float* output,
    uint h,
    uint num_heads,
    uint num_kv_groups,
    uint capacity,
    uint row_bytes,
    uint groups_per_row,
    uint kv_seq,
    uint kv_start,
    uint cur_seq,
    float scale,
    uint tid,
    uint sgid,
    uint lane,
    threadgroup float* shared_q,
    threadgroup float* shared_scores,
    threadgroup float* shared_exp,
    threadgroup float* shared_update
) {
    if (h >= num_heads) return;

    uint kv_h = h / num_kv_groups;
    uint q_offset = h * HEAD_DIM;
    uint k_head_base = kv_h * capacity * row_bytes;
    uint v_head_base = kv_h * capacity * row_bytes;
    uint num_simds = FLASH_TG_SIZE / SIMD_SIZE;

    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    flash_zero_output_hd<HEAD_DIM>(output, q_offset, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    flash_load_q_hd<HEAD_DIM>(Q, q_offset, shared_q, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv_tile = 0; kv_tile < kv_seq; kv_tile += TILE_KV) {
        uint tile_count = min(TILE_KV, kv_seq - kv_tile);

        for (uint wave = 0; wave < tile_count; wave += num_simds) {
            uint kv_offset = wave + sgid;
            if (kv_offset < tile_count) {
                uint actual_pos = kv_start + kv_tile + kv_offset;
                float partial = flash_dot_k_fused_hd<HEAD_DIM>(
                    K_cache, k_head_base, actual_pos, row_bytes,
                    K_f32, kv_h, cur_seq, shared_q, lane);
                partial = simd_sum(partial);
                if (lane == 0) {
                    shared_scores[kv_offset] = partial * scale;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            flash_softmax_tile(shared_scores, shared_exp, shared_update, tile_count);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        flash_accum_v_q4_fused_hd<HEAD_DIM, TILE_KV>(
            output, q_offset, V_cache, v_head_base,
            kv_start, kv_tile, tile_count, row_bytes,
            V_f32, kv_h, cur_seq,
            shared_exp, shared_update[2], shared_update[3],
            tid, FLASH_TG_SIZE);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if ((h % num_kv_groups) == 0) {
        uint src_base = kv_h * HEAD_DIM;
        for (uint g = tid; g < groups_per_row; g += FLASH_TG_SIZE) {
            q4_0_append_group_f32(K_f32, src_base, K_cache, k_head_base, cur_seq, row_bytes, g);
            q4_0_append_group_f32(V_f32, src_base, V_cache, v_head_base, cur_seq, row_bytes, g);
        }
    }
}

kernel void attention_flash_decode_fused_q4_0(
    device const float* Q [[buffer(0)]],
    device const float* K_f32 [[buffer(1)]],
    device const float* V_f32 [[buffer(2)]],
    device float* output [[buffer(3)]],
    device uchar* K_cache [[buffer(4)]],
    device uchar* V_cache [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& num_kv_heads [[buffer(7)]],
    constant uint& num_kv_groups [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& kv_seq [[buffer(10)]],
    constant uint& capacity [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    constant uint& kv_start [[buffer(13)]],
    constant uint& groups_per_row [[buffer(14)]],
    constant uint& row_bytes [[buffer(15)]],
    constant uint& cur_seq [[buffer(16)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)num_kv_heads;
    uint h = tgid;
    if (h >= num_heads) return;

    uint kv_h = h / num_kv_groups;
    uint q_offset = h * head_dim;
    uint k_head_base = kv_h * capacity * row_bytes;
    uint v_head_base = kv_h * capacity * row_bytes;
    uint num_simds = FLASH_TG_SIZE / SIMD_SIZE;

    threadgroup float shared_q[FLASH_MAX_HEAD];
    threadgroup float shared_scores[FLASH_TILE_KV];
    threadgroup float shared_exp[FLASH_TILE_KV];
    threadgroup float shared_update[4];

    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    flash_zero_output(output, q_offset, head_dim, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    flash_load_q(Q, q_offset, shared_q, head_dim, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv_tile = 0; kv_tile < kv_seq; kv_tile += FLASH_TILE_KV) {
        uint tile_count = min(FLASH_TILE_KV, kv_seq - kv_tile);

        for (uint wave = 0; wave < tile_count; wave += num_simds) {
            uint kv_offset = wave + sgid;
            if (kv_offset < tile_count) {
                uint actual_pos = kv_start + kv_tile + kv_offset;
                float partial = flash_dot_k_fused(
                    K_cache, k_head_base, actual_pos, row_bytes, head_dim,
                    K_f32, kv_h, cur_seq, shared_q, lane);
                partial = simd_sum(partial);
                if (lane == 0) {
                    shared_scores[kv_offset] = partial * scale;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            flash_softmax_tile(shared_scores, shared_exp, shared_update, tile_count);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        flash_accum_v_q4_fused(
            output, q_offset, V_cache, v_head_base,
            kv_start, kv_tile, tile_count, row_bytes, head_dim,
            V_f32, kv_h, cur_seq,
            shared_exp, shared_update[2], shared_update[3],
            tid, FLASH_TG_SIZE);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if ((h % num_kv_groups) == 0) {
        uint src_base = kv_h * head_dim;
        for (uint g = tid; g < groups_per_row; g += FLASH_TG_SIZE) {
            q4_0_append_group_f32(K_f32, src_base, K_cache, k_head_base, cur_seq, row_bytes, g);
            q4_0_append_group_f32(V_f32, src_base, V_cache, v_head_base, cur_seq, row_bytes, g);
        }
    }
}

kernel void attention_flash_decode_fused_q4_0_h128(
    device const float* Q [[buffer(0)]],
    device const float* K_f32 [[buffer(1)]],
    device const float* V_f32 [[buffer(2)]],
    device float* output [[buffer(3)]],
    device uchar* K_cache [[buffer(4)]],
    device uchar* V_cache [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& num_kv_heads [[buffer(7)]],
    constant uint& num_kv_groups [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& kv_seq [[buffer(10)]],
    constant uint& capacity [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    constant uint& kv_start [[buffer(13)]],
    constant uint& groups_per_row [[buffer(14)]],
    constant uint& row_bytes [[buffer(15)]],
    constant uint& cur_seq [[buffer(16)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)num_kv_heads;
    (void)head_dim;
    threadgroup float shared_q[128];
    threadgroup float shared_scores[128];
    threadgroup float shared_exp[128];
    threadgroup float shared_update[4];
    flash_decode_fused_q4_0_hd_body<128, 128>(
        Q, K_f32, V_f32, K_cache, V_cache, output, tgid, num_heads, num_kv_groups,
        capacity, row_bytes, groups_per_row, kv_seq, kv_start, cur_seq, scale,
        tid, sgid, lane, shared_q, shared_scores, shared_exp, shared_update);
}

kernel void attention_flash_decode_fused_q4_0_h256(
    device const float* Q [[buffer(0)]],
    device const float* K_f32 [[buffer(1)]],
    device const float* V_f32 [[buffer(2)]],
    device float* output [[buffer(3)]],
    device uchar* K_cache [[buffer(4)]],
    device uchar* V_cache [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& num_kv_heads [[buffer(7)]],
    constant uint& num_kv_groups [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& kv_seq [[buffer(10)]],
    constant uint& capacity [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    constant uint& kv_start [[buffer(13)]],
    constant uint& groups_per_row [[buffer(14)]],
    constant uint& row_bytes [[buffer(15)]],
    constant uint& cur_seq [[buffer(16)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)num_kv_heads;
    (void)head_dim;
    threadgroup float shared_q[256];
    threadgroup float shared_scores[256];
    threadgroup float shared_exp[256];
    threadgroup float shared_update[4];
    flash_decode_fused_q4_0_hd_body<256, 256>(
        Q, K_f32, V_f32, K_cache, V_cache, output, tgid, num_heads, num_kv_groups,
        capacity, row_bytes, groups_per_row, kv_seq, kv_start, cur_seq, scale,
        tid, sgid, lane, shared_q, shared_scores, shared_exp, shared_update);
}

kernel void attention_flash_decode_fused_q4_0_h512(
    device const float* Q [[buffer(0)]],
    device const float* K_f32 [[buffer(1)]],
    device const float* V_f32 [[buffer(2)]],
    device float* output [[buffer(3)]],
    device uchar* K_cache [[buffer(4)]],
    device uchar* V_cache [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& num_kv_heads [[buffer(7)]],
    constant uint& num_kv_groups [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& kv_seq [[buffer(10)]],
    constant uint& capacity [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    constant uint& kv_start [[buffer(13)]],
    constant uint& groups_per_row [[buffer(14)]],
    constant uint& row_bytes [[buffer(15)]],
    constant uint& cur_seq [[buffer(16)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)num_kv_heads;
    (void)head_dim;
    threadgroup float shared_q[512];
    threadgroup float shared_scores[256];
    threadgroup float shared_exp[256];
    threadgroup float shared_update[4];
    flash_decode_fused_q4_0_hd_body<512, 256>(
        Q, K_f32, V_f32, K_cache, V_cache, output, tgid, num_heads, num_kv_groups,
        capacity, row_bytes, groups_per_row, kv_seq, kv_start, cur_seq, scale,
        tid, sgid, lane, shared_q, shared_scores, shared_exp, shared_update);
}

template<uint HEAD_DIM, uint TILE_KV>
void flash_decode_q4_0_qknorm_rope_hd_body(
    device const float* Q_raw,
    device const float* q_norm_weight,
    device const float* cos_buf,
    device const float* sin_buf,
    float eps,
    device const uchar* K_cache,
    device const uchar* V_cache,
    device float* output,
    uint h,
    uint num_heads,
    uint num_kv_groups,
    uint capacity,
    uint row_bytes,
    uint kv_seq,
    uint kv_start,
    float scale,
    uint tid,
    uint sgid,
    uint lane,
    threadgroup float* shared_q,
    threadgroup float* shared_scores,
    threadgroup float* shared_exp,
    threadgroup float* shared_update
) {
    if (h >= num_heads) return;

    uint kv_h = h / num_kv_groups;
    uint q_offset = h * HEAD_DIM;
    uint k_head_base = kv_h * capacity * row_bytes;
    uint v_head_base = kv_h * capacity * row_bytes;
    uint num_simds = FLASH_TG_SIZE / SIMD_SIZE;

    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    flash_zero_output_hd<HEAD_DIM>(output, q_offset, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    flash_load_q_qknorm_rope_hd<HEAD_DIM>(
        Q_raw, q_norm_weight, cos_buf, sin_buf, eps, q_offset,
        shared_q, shared_scores, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv_tile = 0; kv_tile < kv_seq; kv_tile += TILE_KV) {
        uint tile_count = min(TILE_KV, kv_seq - kv_tile);

        for (uint wave = 0; wave < tile_count; wave += num_simds) {
            uint kv_offset = wave + sgid;
            if (kv_offset < tile_count) {
                uint actual_pos = kv_start + kv_tile + kv_offset;
                float partial = flash_dot_q4_k_hd<HEAD_DIM>(
                    K_cache, k_head_base, actual_pos, row_bytes, shared_q, lane);
                partial = simd_sum(partial);
                if (lane == 0) {
                    shared_scores[kv_offset] = partial * scale;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            flash_softmax_tile(shared_scores, shared_exp, shared_update, tile_count);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        flash_accum_v_q4_hd<HEAD_DIM, TILE_KV>(
            output, q_offset, V_cache, v_head_base,
            kv_start, kv_tile, tile_count, row_bytes,
            shared_exp, shared_update[2], shared_update[3],
            tid, FLASH_TG_SIZE);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void attention_flash_decode_qknorm_rope_q4_0_h256(
    device const float* Q_raw [[buffer(0)]],
    device const float* q_norm_weight [[buffer(1)]],
    device const float* cos_buf [[buffer(2)]],
    device const float* sin_buf [[buffer(3)]],
    device const uchar* K_cache [[buffer(4)]],
    device const uchar* V_cache [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& num_heads [[buffer(7)]],
    constant uint& num_kv_heads [[buffer(8)]],
    constant uint& num_kv_groups [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    constant uint& kv_seq [[buffer(11)]],
    constant uint& capacity [[buffer(12)]],
    constant float& scale [[buffer(13)]],
    constant uint& kv_start [[buffer(14)]],
    constant uint& groups_per_row [[buffer(15)]],
    constant uint& row_bytes [[buffer(16)]],
    constant float& eps [[buffer(17)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)num_kv_heads;
    (void)head_dim;
    (void)groups_per_row;
    threadgroup float shared_q[256];
    threadgroup float shared_scores[256];
    threadgroup float shared_exp[256];
    threadgroup float shared_update[4];
    flash_decode_q4_0_qknorm_rope_hd_body<256, 256>(
        Q_raw, q_norm_weight, cos_buf, sin_buf, eps,
        K_cache, V_cache, output, tgid, num_heads, num_kv_groups, capacity, row_bytes,
        kv_seq, kv_start, scale, tid, sgid, lane,
        shared_q, shared_scores, shared_exp, shared_update);
}

kernel void attention_flash_decode_qknorm_rope_q4_0_h128(
    device const float* Q_raw [[buffer(0)]],
    device const float* q_norm_weight [[buffer(1)]],
    device const float* cos_buf [[buffer(2)]],
    device const float* sin_buf [[buffer(3)]],
    device const uchar* K_cache [[buffer(4)]],
    device const uchar* V_cache [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& num_heads [[buffer(7)]],
    constant uint& num_kv_heads [[buffer(8)]],
    constant uint& num_kv_groups [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    constant uint& kv_seq [[buffer(11)]],
    constant uint& capacity [[buffer(12)]],
    constant float& scale [[buffer(13)]],
    constant uint& kv_start [[buffer(14)]],
    constant uint& groups_per_row [[buffer(15)]],
    constant uint& row_bytes [[buffer(16)]],
    constant float& eps [[buffer(17)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)num_kv_heads;
    (void)head_dim;
    (void)groups_per_row;
    threadgroup float shared_q[128];
    threadgroup float shared_scores[128];
    threadgroup float shared_exp[128];
    threadgroup float shared_update[4];
    flash_decode_q4_0_qknorm_rope_hd_body<128, 128>(
        Q_raw, q_norm_weight, cos_buf, sin_buf, eps,
        K_cache, V_cache, output, tgid, num_heads, num_kv_groups, capacity, row_bytes,
        kv_seq, kv_start, scale, tid, sgid, lane,
        shared_q, shared_scores, shared_exp, shared_update);
}

kernel void attention_flash_decode_qknorm_rope_q4_0_h512(
    device const float* Q_raw [[buffer(0)]],
    device const float* q_norm_weight [[buffer(1)]],
    device const float* cos_buf [[buffer(2)]],
    device const float* sin_buf [[buffer(3)]],
    device const uchar* K_cache [[buffer(4)]],
    device const uchar* V_cache [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& num_heads [[buffer(7)]],
    constant uint& num_kv_heads [[buffer(8)]],
    constant uint& num_kv_groups [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    constant uint& kv_seq [[buffer(11)]],
    constant uint& capacity [[buffer(12)]],
    constant float& scale [[buffer(13)]],
    constant uint& kv_start [[buffer(14)]],
    constant uint& groups_per_row [[buffer(15)]],
    constant uint& row_bytes [[buffer(16)]],
    constant float& eps [[buffer(17)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)num_kv_heads;
    (void)head_dim;
    (void)groups_per_row;
    threadgroup float shared_q[512];
    threadgroup float shared_scores[256];
    threadgroup float shared_exp[256];
    threadgroup float shared_update[4];
    flash_decode_q4_0_qknorm_rope_hd_body<512, 256>(
        Q_raw, q_norm_weight, cos_buf, sin_buf, eps,
        K_cache, V_cache, output, tgid, num_heads, num_kv_groups, capacity, row_bytes,
        kv_seq, kv_start, scale, tid, sgid, lane,
        shared_q, shared_scores, shared_exp, shared_update);
}

template<uint HEAD_DIM, uint TILE_KV>
void flash_decode_fused_q4_0_qknorm_rope_hd_body(
    device const float* Q_raw,
    device const float* q_norm_weight,
    device const float* cos_buf,
    device const float* sin_buf,
    float eps,
    device const float* K_f32,
    device const float* V_f32,
    device uchar* K_cache,
    device uchar* V_cache,
    device float* output,
    uint h,
    uint num_heads,
    uint num_kv_groups,
    uint capacity,
    uint row_bytes,
    uint groups_per_row,
    uint kv_seq,
    uint kv_start,
    uint cur_seq,
    float scale,
    uint tid,
    uint sgid,
    uint lane,
    threadgroup float* shared_q,
    threadgroup float* shared_scores,
    threadgroup float* shared_exp,
    threadgroup float* shared_update
) {
    if (h >= num_heads) return;

    uint kv_h = h / num_kv_groups;
    uint q_offset = h * HEAD_DIM;
    uint k_head_base = kv_h * capacity * row_bytes;
    uint v_head_base = kv_h * capacity * row_bytes;
    uint num_simds = FLASH_TG_SIZE / SIMD_SIZE;

    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    flash_zero_output_hd<HEAD_DIM>(output, q_offset, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    flash_load_q_qknorm_rope_hd<HEAD_DIM>(
        Q_raw, q_norm_weight, cos_buf, sin_buf, eps, q_offset,
        shared_q, shared_scores, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv_tile = 0; kv_tile < kv_seq; kv_tile += TILE_KV) {
        uint tile_count = min(TILE_KV, kv_seq - kv_tile);

        for (uint wave = 0; wave < tile_count; wave += num_simds) {
            uint kv_offset = wave + sgid;
            if (kv_offset < tile_count) {
                uint actual_pos = kv_start + kv_tile + kv_offset;
                float partial = flash_dot_k_fused_hd<HEAD_DIM>(
                    K_cache, k_head_base, actual_pos, row_bytes,
                    K_f32, kv_h, cur_seq, shared_q, lane);
                partial = simd_sum(partial);
                if (lane == 0) {
                    shared_scores[kv_offset] = partial * scale;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            flash_softmax_tile(shared_scores, shared_exp, shared_update, tile_count);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        flash_accum_v_q4_fused_hd<HEAD_DIM, TILE_KV>(
            output, q_offset, V_cache, v_head_base,
            kv_start, kv_tile, tile_count, row_bytes,
            V_f32, kv_h, cur_seq,
            shared_exp, shared_update[2], shared_update[3],
            tid, FLASH_TG_SIZE);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if ((h % num_kv_groups) == 0) {
        uint src_base = kv_h * HEAD_DIM;
        for (uint g = tid; g < groups_per_row; g += FLASH_TG_SIZE) {
            q4_0_append_group_f32(K_f32, src_base, K_cache, k_head_base, cur_seq, row_bytes, g);
            q4_0_append_group_f32(V_f32, src_base, V_cache, v_head_base, cur_seq, row_bytes, g);
        }
    }
}

kernel void attention_flash_decode_fused_qknorm_rope_q4_0_h256(
    device const float* Q_raw [[buffer(0)]],
    device const float* q_norm_weight [[buffer(1)]],
    device const float* cos_buf [[buffer(2)]],
    device const float* sin_buf [[buffer(3)]],
    device const float* K_f32 [[buffer(4)]],
    device const float* V_f32 [[buffer(5)]],
    device float* output [[buffer(6)]],
    device uchar* K_cache [[buffer(7)]],
    device uchar* V_cache [[buffer(8)]],
    constant uint& num_heads [[buffer(9)]],
    constant uint& num_kv_heads [[buffer(10)]],
    constant uint& num_kv_groups [[buffer(11)]],
    constant uint& head_dim [[buffer(12)]],
    constant uint& kv_seq [[buffer(13)]],
    constant uint& capacity [[buffer(14)]],
    constant float& scale [[buffer(15)]],
    constant uint& kv_start [[buffer(16)]],
    constant uint& groups_per_row [[buffer(17)]],
    constant uint& row_bytes [[buffer(18)]],
    constant uint& cur_seq [[buffer(19)]],
    constant float& eps [[buffer(20)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)num_kv_heads;
    (void)head_dim;
    threadgroup float shared_q[256];
    threadgroup float shared_scores[256];
    threadgroup float shared_exp[256];
    threadgroup float shared_update[4];
    flash_decode_fused_q4_0_qknorm_rope_hd_body<256, 256>(
        Q_raw, q_norm_weight, cos_buf, sin_buf, eps,
        K_f32, V_f32, K_cache, V_cache, output, tgid, num_heads, num_kv_groups,
        capacity, row_bytes, groups_per_row, kv_seq, kv_start, cur_seq, scale,
        tid, sgid, lane, shared_q, shared_scores, shared_exp, shared_update);
}

kernel void attention_flash_decode_fused_qknorm_rope_q4_0_h128(
    device const float* Q_raw [[buffer(0)]],
    device const float* q_norm_weight [[buffer(1)]],
    device const float* cos_buf [[buffer(2)]],
    device const float* sin_buf [[buffer(3)]],
    device const float* K_f32 [[buffer(4)]],
    device const float* V_f32 [[buffer(5)]],
    device float* output [[buffer(6)]],
    device uchar* K_cache [[buffer(7)]],
    device uchar* V_cache [[buffer(8)]],
    constant uint& num_heads [[buffer(9)]],
    constant uint& num_kv_heads [[buffer(10)]],
    constant uint& num_kv_groups [[buffer(11)]],
    constant uint& head_dim [[buffer(12)]],
    constant uint& kv_seq [[buffer(13)]],
    constant uint& capacity [[buffer(14)]],
    constant float& scale [[buffer(15)]],
    constant uint& kv_start [[buffer(16)]],
    constant uint& groups_per_row [[buffer(17)]],
    constant uint& row_bytes [[buffer(18)]],
    constant uint& cur_seq [[buffer(19)]],
    constant float& eps [[buffer(20)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)num_kv_heads;
    (void)head_dim;
    threadgroup float shared_q[128];
    threadgroup float shared_scores[128];
    threadgroup float shared_exp[128];
    threadgroup float shared_update[4];
    flash_decode_fused_q4_0_qknorm_rope_hd_body<128, 128>(
        Q_raw, q_norm_weight, cos_buf, sin_buf, eps,
        K_f32, V_f32, K_cache, V_cache, output, tgid, num_heads, num_kv_groups,
        capacity, row_bytes, groups_per_row, kv_seq, kv_start, cur_seq, scale,
        tid, sgid, lane, shared_q, shared_scores, shared_exp, shared_update);
}

kernel void attention_flash_decode_fused_qknorm_rope_q4_0_h512(
    device const float* Q_raw [[buffer(0)]],
    device const float* q_norm_weight [[buffer(1)]],
    device const float* cos_buf [[buffer(2)]],
    device const float* sin_buf [[buffer(3)]],
    device const float* K_f32 [[buffer(4)]],
    device const float* V_f32 [[buffer(5)]],
    device float* output [[buffer(6)]],
    device uchar* K_cache [[buffer(7)]],
    device uchar* V_cache [[buffer(8)]],
    constant uint& num_heads [[buffer(9)]],
    constant uint& num_kv_heads [[buffer(10)]],
    constant uint& num_kv_groups [[buffer(11)]],
    constant uint& head_dim [[buffer(12)]],
    constant uint& kv_seq [[buffer(13)]],
    constant uint& capacity [[buffer(14)]],
    constant float& scale [[buffer(15)]],
    constant uint& kv_start [[buffer(16)]],
    constant uint& groups_per_row [[buffer(17)]],
    constant uint& row_bytes [[buffer(18)]],
    constant uint& cur_seq [[buffer(19)]],
    constant float& eps [[buffer(20)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)num_kv_heads;
    (void)head_dim;
    threadgroup float shared_q[512];
    threadgroup float shared_scores[256];
    threadgroup float shared_exp[256];
    threadgroup float shared_update[4];
    flash_decode_fused_q4_0_qknorm_rope_hd_body<512, 256>(
        Q_raw, q_norm_weight, cos_buf, sin_buf, eps,
        K_f32, V_f32, K_cache, V_cache, output, tgid, num_heads, num_kv_groups,
        capacity, row_bytes, groups_per_row, kv_seq, kv_start, cur_seq, scale,
        tid, sgid, lane, shared_q, shared_scores, shared_exp, shared_update);
}

// Full decode fusion: QK-norm+RoPE + K-norm+RoPE + V-norm + KV append + flash attention.
template<uint HEAD_DIM, uint TILE_KV>
void flash_decode_full_fused_q4_0_hd_body(
    device const float* Q_raw,
    device const float* q_norm_weight,
    device const float* cos_buf,
    device const float* sin_buf,
    device const float* K_raw,
    device const float* k_norm_weight,
    device const float* V_raw,
    float eps,
    device uchar* K_cache,
    device uchar* V_cache,
    device float* output,
    uint h,
    uint num_heads,
    uint num_kv_groups,
    uint capacity,
    uint row_bytes,
    uint groups_per_row,
    uint kv_seq,
    uint kv_start,
    uint cur_seq,
    float scale,
    uint tid,
    uint sgid,
    uint lane,
    threadgroup float* shared_q,
    threadgroup float* shared_k,
    threadgroup float* shared_v,
    threadgroup float* shared_scores,
    threadgroup float* shared_exp,
    threadgroup float* shared_update
) {
    if (h >= num_heads) return;

    uint kv_h = h / num_kv_groups;
    uint q_offset = h * HEAD_DIM;
    uint kv_offset = kv_h * HEAD_DIM;
    uint k_head_base = kv_h * capacity * row_bytes;
    uint v_head_base = kv_h * capacity * row_bytes;
    uint num_simds = FLASH_TG_SIZE / SIMD_SIZE;

    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    flash_zero_output_hd<HEAD_DIM>(output, q_offset, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    flash_load_q_qknorm_rope_hd<HEAD_DIM>(
        Q_raw, q_norm_weight, cos_buf, sin_buf, eps, q_offset,
        shared_q, shared_scores, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    flash_prepare_k_norm_rope_hd<HEAD_DIM>(
        K_raw, k_norm_weight, cos_buf, sin_buf, eps, kv_offset,
        shared_k, shared_scores, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    flash_prepare_v_norm_hd<HEAD_DIM>(
        V_raw, eps, kv_offset, shared_v, shared_scores, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv_tile = 0; kv_tile < kv_seq; kv_tile += TILE_KV) {
        uint tile_count = min(TILE_KV, kv_seq - kv_tile);

        for (uint wave = 0; wave < tile_count; wave += num_simds) {
            uint kv_pos = wave + sgid;
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
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            flash_softmax_tile(shared_scores, shared_exp, shared_update, tile_count);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        flash_accum_v_q4_shared_fused_hd<HEAD_DIM, TILE_KV>(
            output, q_offset, V_cache, v_head_base,
            kv_start, kv_tile, tile_count, row_bytes,
            shared_v, cur_seq,
            shared_exp, shared_update[2], shared_update[3],
            tid, FLASH_TG_SIZE);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if ((h % num_kv_groups) == 0) {
        for (uint g = tid; g < groups_per_row; g += FLASH_TG_SIZE) {
            q4_0_append_group_tg(shared_k, 0, K_cache, k_head_base, cur_seq, row_bytes, g);
            q4_0_append_group_tg(shared_v, 0, V_cache, v_head_base, cur_seq, row_bytes, g);
        }
    }
}

kernel void attention_flash_decode_full_fused_q4_0_h256(
    device const float* Q_raw [[buffer(0)]],
    device const float* q_norm_weight [[buffer(1)]],
    device const float* cos_buf [[buffer(2)]],
    device const float* sin_buf [[buffer(3)]],
    device const float* K_raw [[buffer(4)]],
    device const float* k_norm_weight [[buffer(5)]],
    device const float* V_raw [[buffer(6)]],
    device float* output [[buffer(7)]],
    device uchar* K_cache [[buffer(8)]],
    device uchar* V_cache [[buffer(9)]],
    constant uint& num_heads [[buffer(10)]],
    constant uint& num_kv_heads [[buffer(11)]],
    constant uint& num_kv_groups [[buffer(12)]],
    constant uint& head_dim [[buffer(13)]],
    constant uint& kv_seq [[buffer(14)]],
    constant uint& capacity [[buffer(15)]],
    constant float& scale [[buffer(16)]],
    constant uint& kv_start [[buffer(17)]],
    constant uint& groups_per_row [[buffer(18)]],
    constant uint& row_bytes [[buffer(19)]],
    constant uint& cur_seq [[buffer(20)]],
    constant float& eps [[buffer(21)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)num_kv_heads;
    (void)head_dim;
    threadgroup float shared_q[256];
    threadgroup float shared_k[256];
    threadgroup float shared_v[256];
    threadgroup float shared_scores[256];
    threadgroup float shared_exp[256];
    threadgroup float shared_update[4];
    flash_decode_full_fused_q4_0_hd_body<256, 256>(
        Q_raw, q_norm_weight, cos_buf, sin_buf,
        K_raw, k_norm_weight, V_raw, eps,
        K_cache, V_cache, output, tgid, num_heads, num_kv_groups,
        capacity, row_bytes, groups_per_row, kv_seq, kv_start, cur_seq, scale,
        tid, sgid, lane, shared_q, shared_k, shared_v,
        shared_scores, shared_exp, shared_update);
}

kernel void attention_flash_decode_full_fused_q4_0_h128(
    device const float* Q_raw [[buffer(0)]],
    device const float* q_norm_weight [[buffer(1)]],
    device const float* cos_buf [[buffer(2)]],
    device const float* sin_buf [[buffer(3)]],
    device const float* K_raw [[buffer(4)]],
    device const float* k_norm_weight [[buffer(5)]],
    device const float* V_raw [[buffer(6)]],
    device float* output [[buffer(7)]],
    device uchar* K_cache [[buffer(8)]],
    device uchar* V_cache [[buffer(9)]],
    constant uint& num_heads [[buffer(10)]],
    constant uint& num_kv_heads [[buffer(11)]],
    constant uint& num_kv_groups [[buffer(12)]],
    constant uint& head_dim [[buffer(13)]],
    constant uint& kv_seq [[buffer(14)]],
    constant uint& capacity [[buffer(15)]],
    constant float& scale [[buffer(16)]],
    constant uint& kv_start [[buffer(17)]],
    constant uint& groups_per_row [[buffer(18)]],
    constant uint& row_bytes [[buffer(19)]],
    constant uint& cur_seq [[buffer(20)]],
    constant float& eps [[buffer(21)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)num_kv_heads;
    (void)head_dim;
    threadgroup float shared_q[128];
    threadgroup float shared_k[128];
    threadgroup float shared_v[128];
    threadgroup float shared_scores[128];
    threadgroup float shared_exp[128];
    threadgroup float shared_update[4];
    flash_decode_full_fused_q4_0_hd_body<128, 128>(
        Q_raw, q_norm_weight, cos_buf, sin_buf,
        K_raw, k_norm_weight, V_raw, eps,
        K_cache, V_cache, output, tgid, num_heads, num_kv_groups,
        capacity, row_bytes, groups_per_row, kv_seq, kv_start, cur_seq, scale,
        tid, sgid, lane, shared_q, shared_k, shared_v,
        shared_scores, shared_exp, shared_update);
}

kernel void attention_flash_decode_full_fused_q4_0_h512(
    device const float* Q_raw [[buffer(0)]],
    device const float* q_norm_weight [[buffer(1)]],
    device const float* cos_buf [[buffer(2)]],
    device const float* sin_buf [[buffer(3)]],
    device const float* K_raw [[buffer(4)]],
    device const float* k_norm_weight [[buffer(5)]],
    device const float* V_raw [[buffer(6)]],
    device float* output [[buffer(7)]],
    device uchar* K_cache [[buffer(8)]],
    device uchar* V_cache [[buffer(9)]],
    constant uint& num_heads [[buffer(10)]],
    constant uint& num_kv_heads [[buffer(11)]],
    constant uint& num_kv_groups [[buffer(12)]],
    constant uint& head_dim [[buffer(13)]],
    constant uint& kv_seq [[buffer(14)]],
    constant uint& capacity [[buffer(15)]],
    constant float& scale [[buffer(16)]],
    constant uint& kv_start [[buffer(17)]],
    constant uint& groups_per_row [[buffer(18)]],
    constant uint& row_bytes [[buffer(19)]],
    constant uint& cur_seq [[buffer(20)]],
    constant float& eps [[buffer(21)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)num_kv_heads;
    (void)head_dim;
    threadgroup float shared_q[512];
    threadgroup float shared_k[512];
    threadgroup float shared_v[512];
    threadgroup float shared_scores[256];
    threadgroup float shared_exp[256];
    threadgroup float shared_update[4];
    flash_decode_full_fused_q4_0_hd_body<512, 256>(
        Q_raw, q_norm_weight, cos_buf, sin_buf,
        K_raw, k_norm_weight, V_raw, eps,
        K_cache, V_cache, output, tgid, num_heads, num_kv_groups,
        capacity, row_bytes, groups_per_row, kv_seq, kv_start, cur_seq, scale,
        tid, sgid, lane, shared_q, shared_k, shared_v,
        shared_scores, shared_exp, shared_update);
}

// ─── Flash decode: single query token vs KV cache (f16) ──────────────────────

kernel void attention_flash_decode_f16(
    device const float* Q [[buffer(0)]],
    device const half* K_cache [[buffer(1)]],
    device const half* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& k_cap [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& kv_start [[buffer(11)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint h = tgid;
    if (h >= num_heads) return;

    uint kv_h = h / num_kv_groups;
    uint q_offset = h * head_dim;
    uint k_head_base = kv_h * k_cap * head_dim;
    uint v_head_base = kv_h * k_cap * head_dim;
    uint num_simds = FLASH_TG_SIZE / SIMD_SIZE;

    threadgroup float shared_q[FLASH_MAX_HEAD];
    threadgroup float shared_scores[FLASH_TILE_KV];
    threadgroup float shared_exp[FLASH_TILE_KV];
    threadgroup float shared_update[4];

    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    flash_zero_output(output, q_offset, head_dim, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    flash_load_q(Q, q_offset, shared_q, head_dim, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv_tile = 0; kv_tile < kv_seq; kv_tile += FLASH_TILE_KV) {
        uint tile_count = min(FLASH_TILE_KV, kv_seq - kv_tile);

        for (uint wave = 0; wave < tile_count; wave += num_simds) {
            uint kv_offset = wave + sgid;
            if (kv_offset < tile_count) {
                uint actual_pos = kv_start + kv_tile + kv_offset;
                uint k_off = k_head_base + actual_pos * head_dim;
                float partial = flash_dot_f16_k_cached(
                    &K_cache[k_off], head_dim, shared_q, lane);
                partial = simd_sum(partial);
                if (lane == 0) {
                    shared_scores[kv_offset] = partial * scale;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            flash_softmax_tile(shared_scores, shared_exp, shared_update, tile_count);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        flash_accum_v_f16(
            output, q_offset, V_cache, v_head_base, head_dim,
            kv_start, kv_tile, tile_count,
            shared_exp, shared_update[2], shared_update[3],
            tid, FLASH_TG_SIZE);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ─── Flash causal prefill (Q4_0 KV) ────────────────────────────────────────

kernel void attention_flash_causal_q4_0(
    device const float* Q [[buffer(0)]],
    device const uchar* K_cache [[buffer(1)]],
    device const uchar* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& capacity [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& q_len [[buffer(11)]],
    constant uint& q_start [[buffer(12)]],
    constant uint& attention_window [[buffer(13)]],
    constant uint& groups_per_row [[buffer(14)]],
    constant uint& row_bytes [[buffer(15)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)groups_per_row;
    uint h = tgid / q_len;
    uint qi = tgid % q_len;
    if (h >= num_heads) return;

    uint kv_h = h / num_kv_groups;
    uint q_offset = (h * q_len + qi) * head_dim;
    uint out_offset = q_offset;
    uint k_head_base = kv_h * capacity * row_bytes;
    uint v_head_base = kv_h * capacity * row_bytes;
    uint attend_len = min(q_start + qi + 1, kv_seq);
    uint attend_start = 0u;
    if (attention_window > 0u && attend_len > attention_window) {
        attend_start = attend_len - attention_window;
    }
    uint num_simds = FLASH_TG_SIZE / SIMD_SIZE;

    threadgroup float shared_q[FLASH_MAX_HEAD];
    threadgroup float shared_scores[FLASH_TILE_KV];
    threadgroup float shared_exp[FLASH_TILE_KV];
    threadgroup float shared_update[4];

    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    flash_zero_output(output, out_offset, head_dim, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    flash_load_q(Q, q_offset, shared_q, head_dim, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv_tile = attend_start; kv_tile < attend_len; kv_tile += FLASH_TILE_KV) {
        uint tile_end = min(kv_tile + FLASH_TILE_KV, attend_len);
        uint tile_count = tile_end - kv_tile;

        for (uint wave = 0; wave < tile_count; wave += num_simds) {
            uint kv_offset = wave + sgid;
            if (kv_offset < tile_count) {
                uint kv = kv_tile + kv_offset;
                float partial = flash_dot_q4_k_cached(
                    K_cache, k_head_base, kv, row_bytes, head_dim, shared_q, lane);
                partial = simd_sum(partial);
                if (lane == 0) {
                    shared_scores[kv_offset] = partial * scale;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            flash_softmax_tile(shared_scores, shared_exp, shared_update, tile_count);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        flash_accum_v_q4(
            output, out_offset, V_cache, v_head_base,
            0, kv_tile, tile_count, row_bytes, head_dim,
            shared_exp, shared_update[2], shared_update[3],
            tid, FLASH_TG_SIZE);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ─── Flash causal prefill (f16 KV) ─────────────────────────────────────────

kernel void attention_flash_causal_f16(
    device const float* Q [[buffer(0)]],
    device const half* K_cache [[buffer(1)]],
    device const half* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& k_cap [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& q_len [[buffer(11)]],
    constant uint& q_start [[buffer(12)]],
    constant uint& attention_window [[buffer(13)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint h = tgid / q_len;
    uint qi = tgid % q_len;
    if (h >= num_heads) return;

    uint kv_h = h / num_kv_groups;
    uint q_offset = (h * q_len + qi) * head_dim;
    uint out_offset = q_offset;
    uint k_head_base = kv_h * k_cap * head_dim;
    uint v_head_base = kv_h * k_cap * head_dim;
    uint attend_len = min(q_start + qi + 1, kv_seq);
    uint attend_start = 0u;
    if (attention_window > 0u && attend_len > attention_window) {
        attend_start = attend_len - attention_window;
    }
    uint num_simds = FLASH_TG_SIZE / SIMD_SIZE;

    threadgroup float shared_q[FLASH_MAX_HEAD];
    threadgroup float shared_scores[FLASH_TILE_KV];
    threadgroup float shared_exp[FLASH_TILE_KV];
    threadgroup float shared_update[4];

    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    flash_zero_output(output, out_offset, head_dim, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    flash_load_q(Q, q_offset, shared_q, head_dim, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv_tile = attend_start; kv_tile < attend_len; kv_tile += FLASH_TILE_KV) {
        uint tile_end = min(kv_tile + FLASH_TILE_KV, attend_len);
        uint tile_count = tile_end - kv_tile;

        for (uint wave = 0; wave < tile_count; wave += num_simds) {
            uint kv_offset = wave + sgid;
            if (kv_offset < tile_count) {
                uint kv = kv_tile + kv_offset;
                uint k_off = k_head_base + kv * head_dim;
                float partial = flash_dot_f16_k_cached(
                    &K_cache[k_off], head_dim, shared_q, lane);
                partial = simd_sum(partial);
                if (lane == 0) {
                    shared_scores[kv_offset] = partial * scale;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            flash_softmax_tile(shared_scores, shared_exp, shared_update, tile_count);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        flash_accum_v_f16(
            output, out_offset, V_cache, v_head_base, head_dim,
            0, kv_tile, tile_count,
            shared_exp, shared_update[2], shared_update[3],
            tid, FLASH_TG_SIZE);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ─── Flash causal prefill strided Q layout (Q4_0 KV) ───────────────────────

kernel void attention_flash_causal_strided_q4_0(
    device const float* Q [[buffer(0)]],
    device const uchar* K_cache [[buffer(1)]],
    device const uchar* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& capacity [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& q_len [[buffer(11)]],
    constant uint& q_pos_start [[buffer(12)]],
    constant uint& attention_window [[buffer(13)]],
    constant uint& q_stride [[buffer(14)]],
    constant uint& q_start_row [[buffer(15)]],
    constant uint& groups_per_row [[buffer(16)]],
    constant uint& row_bytes [[buffer(17)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)groups_per_row;
    uint h = tgid / q_len;
    uint qi = tgid % q_len;
    if (h >= num_heads) return;

    uint kv_h = h / num_kv_groups;
    uint q_row = q_start_row + qi;
    uint q_offset = (h * q_stride + q_row) * head_dim;
    uint out_offset = q_offset;
    uint k_head_base = kv_h * capacity * row_bytes;
    uint v_head_base = kv_h * capacity * row_bytes;
    uint attend_len = min(q_pos_start + qi + 1, kv_seq);
    uint attend_start = 0u;
    if (attention_window > 0u && attend_len > attention_window) {
        attend_start = attend_len - attention_window;
    }
    uint num_simds = FLASH_TG_SIZE / SIMD_SIZE;

    threadgroup float shared_q[FLASH_MAX_HEAD];
    threadgroup float shared_scores[FLASH_TILE_KV];
    threadgroup float shared_exp[FLASH_TILE_KV];
    threadgroup float shared_update[4];

    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    flash_zero_output(output, out_offset, head_dim, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    flash_load_q(Q, q_offset, shared_q, head_dim, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv_tile = attend_start; kv_tile < attend_len; kv_tile += FLASH_TILE_KV) {
        uint tile_end = min(kv_tile + FLASH_TILE_KV, attend_len);
        uint tile_count = tile_end - kv_tile;

        for (uint wave = 0; wave < tile_count; wave += num_simds) {
            uint kv_offset = wave + sgid;
            if (kv_offset < tile_count) {
                uint kv = kv_tile + kv_offset;
                float partial = flash_dot_q4_k_cached(
                    K_cache, k_head_base, kv, row_bytes, head_dim, shared_q, lane);
                partial = simd_sum(partial);
                if (lane == 0) {
                    shared_scores[kv_offset] = partial * scale;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            flash_softmax_tile(shared_scores, shared_exp, shared_update, tile_count);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        flash_accum_v_q4(
            output, out_offset, V_cache, v_head_base,
            0, kv_tile, tile_count, row_bytes, head_dim,
            shared_exp, shared_update[2], shared_update[3],
            tid, FLASH_TG_SIZE);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ─── Flash causal prefill strided Q layout (f16 KV) ────────────────────────

kernel void attention_flash_causal_strided_f16(
    device const float* Q [[buffer(0)]],
    device const half* K_cache [[buffer(1)]],
    device const half* V_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& num_kv_groups [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_seq [[buffer(8)]],
    constant uint& k_cap [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant uint& q_len [[buffer(11)]],
    constant uint& q_pos_start [[buffer(12)]],
    constant uint& attention_window [[buffer(13)]],
    constant uint& q_stride [[buffer(14)]],
    constant uint& q_start_row [[buffer(15)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint h = tgid / q_len;
    uint qi = tgid % q_len;
    if (h >= num_heads) return;

    uint kv_h = h / num_kv_groups;
    uint q_row = q_start_row + qi;
    uint q_offset = (h * q_stride + q_row) * head_dim;
    uint out_offset = q_offset;
    uint k_head_base = kv_h * k_cap * head_dim;
    uint v_head_base = kv_h * k_cap * head_dim;
    uint attend_len = min(q_pos_start + qi + 1, kv_seq);
    uint attend_start = 0u;
    if (attention_window > 0u && attend_len > attention_window) {
        attend_start = attend_len - attention_window;
    }
    uint num_simds = FLASH_TG_SIZE / SIMD_SIZE;

    threadgroup float shared_q[FLASH_MAX_HEAD];
    threadgroup float shared_scores[FLASH_TILE_KV];
    threadgroup float shared_exp[FLASH_TILE_KV];
    threadgroup float shared_update[4];

    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    flash_zero_output(output, out_offset, head_dim, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    flash_load_q(Q, q_offset, shared_q, head_dim, tid, FLASH_TG_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv_tile = attend_start; kv_tile < attend_len; kv_tile += FLASH_TILE_KV) {
        uint tile_end = min(kv_tile + FLASH_TILE_KV, attend_len);
        uint tile_count = tile_end - kv_tile;

        for (uint wave = 0; wave < tile_count; wave += num_simds) {
            uint kv_offset = wave + sgid;
            if (kv_offset < tile_count) {
                uint kv = kv_tile + kv_offset;
                uint k_off = k_head_base + kv * head_dim;
                float partial = flash_dot_f16_k_cached(
                    &K_cache[k_off], head_dim, shared_q, lane);
                partial = simd_sum(partial);
                if (lane == 0) {
                    shared_scores[kv_offset] = partial * scale;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            flash_softmax_tile(shared_scores, shared_exp, shared_update, tile_count);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        flash_accum_v_f16(
            output, out_offset, V_cache, v_head_base, head_dim,
            0, kv_tile, tile_count,
            shared_exp, shared_update[2], shared_update[3],
            tid, FLASH_TG_SIZE);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ─── GQA tiled causal prefill (flash_attn_ext-style, Q4_0 KV) ───────────────
// One threadgroup per (KV head, query tile).  Processes Q_TILE queries per TG
// using the GQA decode flash path (shared KV tile loads across query heads).

template<uint HEAD_DIM, uint Q_TILE>
void flash_causal_gqa_q4_0_hd_body(
    device const float* Q,
    device const uchar* K_cache,
    device const uchar* V_cache,
    device float* output,
    uint kv_h,
    uint qi_base,
    uint qi_count,
    uint num_heads,
    uint num_kv_heads,
    uint num_kv_groups,
    uint capacity,
    uint row_bytes,
    uint kv_seq,
    uint q_len,
    uint q_start,
    uint attention_window,
    float scale,
    uint tid,
    uint sgid,
    uint lane,
    threadgroup float* shared_q,
    threadgroup float* shared_scores,
    threadgroup float* shared_update,
    threadgroup float* shared_exp,
    threadgroup uchar* shared_kv_tile
) {
    (void)num_kv_heads;
    (void)Q_TILE;
    if (kv_h >= num_kv_heads) return;
    if (num_kv_groups == 0 || num_kv_groups > GQA_MAX_GROUPS) return;

    const uint num_simds_total = FLASH_TG_SIZE / SIMD_SIZE;
    if (num_kv_groups > num_simds_total) return;
    if (num_simds_total % num_kv_groups != 0) return;

    const uint simds_per_head = num_simds_total / num_kv_groups;
    const uint threads_per_head = simds_per_head * SIMD_SIZE;
    const uint k_head_base = kv_h * capacity * row_bytes;
    const uint v_head_base = kv_h * capacity * row_bytes;

    for (uint qix = 0; qix < qi_count; qix++) {
        const uint qi = qi_base + qix;
        const uint global_pos = q_start + qi;
        uint attend_len = min(global_pos + 1u, kv_seq);
        uint attend_start = 0u;
        if (attention_window > 0u && attend_len > attention_window) {
            attend_start = attend_len - attention_window;
        }

        const uint group = sgid / simds_per_head;
        const uint local_sgid = sgid % simds_per_head;
        const uint local_tid = local_sgid * SIMD_SIZE + lane;
        const uint h = kv_h * num_kv_groups + group;
        if (h >= num_heads) continue;

        const uint q_offset = (h * q_len + qi) * HEAD_DIM;
        const uint out_offset = q_offset;
        threadgroup float* q_head = shared_q + group * HEAD_DIM;
        threadgroup float* scores_head = shared_scores + group * GQA_TILE_KV;
        threadgroup float* update_head = shared_update + group * 4;

        if (local_tid == 0) {
            update_head[0] = -INFINITY;
            update_head[1] = 0.0f;
        }
        flash_zero_output_gqa(output, out_offset, HEAD_DIM, local_tid, threads_per_head);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        flash_load_q_gqa(Q, q_offset, q_head, HEAD_DIM, local_tid, threads_per_head);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kv_tile = attend_start; kv_tile < attend_len; kv_tile += GQA_TILE_KV) {
            uint tile_count = min(GQA_TILE_KV, attend_len - kv_tile);

            flash_load_kv_tile_q4(
                K_cache, k_head_base, kv_tile, tile_count, row_bytes,
                tid, FLASH_TG_SIZE, shared_kv_tile);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint wave = local_sgid; wave < tile_count; wave += simds_per_head) {
                uint kv_offset = wave;
                float partial = flash_dot_q4_k_tg(
                    shared_kv_tile + kv_offset * row_bytes, HEAD_DIM, q_head, lane);
                partial = simd_sum(partial);
                if (lane == 0) {
                    scores_head[kv_offset] = partial * scale;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (local_tid == 0) {
                flash_softmax_tile(scores_head, shared_exp, update_head, tile_count);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            flash_load_kv_tile_q4(
                V_cache, v_head_base, kv_tile, tile_count, row_bytes,
                tid, FLASH_TG_SIZE, shared_kv_tile);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float old_factor = update_head[2];
            float inv_l = update_head[3];
            flash_accum_v_q4_tg(
                output, out_offset, shared_kv_tile, row_bytes, HEAD_DIM,
                tile_count, shared_exp, old_factor, inv_l, local_tid, threads_per_head);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

#define FLASH_CAUSAL_GQA_Q4_0_KERNEL(HD, QT) \
kernel void attention_flash_causal_q4_0_gqa_h##HD( \
    device const float* Q [[buffer(0)]], \
    device const uchar* K_cache [[buffer(1)]], \
    device const uchar* V_cache [[buffer(2)]], \
    device float* output [[buffer(3)]], \
    constant uint& num_heads [[buffer(4)]], \
    constant uint& num_kv_heads [[buffer(5)]], \
    constant uint& num_kv_groups [[buffer(6)]], \
    constant uint& head_dim [[buffer(7)]], \
    constant uint& kv_seq [[buffer(8)]], \
    constant uint& capacity [[buffer(9)]], \
    constant float& scale [[buffer(10)]], \
    constant uint& q_len [[buffer(11)]], \
    constant uint& q_start [[buffer(12)]], \
    constant uint& attention_window [[buffer(13)]], \
    constant uint& row_bytes [[buffer(14)]], \
    uint tid [[thread_index_in_threadgroup]], \
    uint2 tgid [[threadgroup_position_in_grid]], \
    uint sgid [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
    (void)head_dim; \
    const uint kv_h = tgid.x; \
    const uint qi_tile = tgid.y; \
    const uint qi_base = qi_tile * QT; \
    if (kv_h >= num_kv_heads || qi_base >= q_len) return; \
    const uint qi_count = min((uint)QT, q_len - qi_base); \
    if (num_kv_groups == 0 || num_kv_groups > GQA_MAX_GROUPS) return; \
    constexpr uint MAX_SLOTS = GQA_MAX_GROUPS; \
    threadgroup float shared_q[MAX_SLOTS * HD]; \
    threadgroup float shared_scores[MAX_SLOTS * GQA_TILE_KV]; \
    threadgroup float shared_update[MAX_SLOTS * 4]; \
    threadgroup float shared_exp[GQA_TILE_KV]; \
    threadgroup uchar shared_kv_tile[GQA_TILE_KV * GQA_MAX_ROW_BYTES]; \
    flash_causal_gqa_q4_0_hd_body<HD, QT>( \
        Q, K_cache, V_cache, output, kv_h, qi_base, qi_count, \
        num_heads, num_kv_heads, num_kv_groups, capacity, row_bytes, \
        kv_seq, q_len, q_start, attention_window, scale, \
        tid, sgid, lane, shared_q, shared_scores, shared_update, shared_exp, \
        shared_kv_tile); \
}

FLASH_CAUSAL_GQA_Q4_0_KERNEL(256, 4)
FLASH_CAUSAL_GQA_Q4_0_KERNEL(512, 2)
