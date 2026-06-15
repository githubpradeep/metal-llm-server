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
// q points to 16 packed bytes. The 32 x values are passed as 8 float4s.
// Uses packed_uchar4 loads so the 16 byte reads collapse to 4 vector loads.
inline float q4_dot_vec(device const uchar* q,
                        float4 xv0, float4 xv1, float4 xv2, float4 xv3,
                        float4 xv4, float4 xv5, float4 xv6, float4 xv7) {
    float local = 0.0f;

    packed_uchar4 q0 = *reinterpret_cast<device const packed_uchar4*>(q + 0);
    packed_uchar4 q1 = *reinterpret_cast<device const packed_uchar4*>(q + 4);
    packed_uchar4 q2 = *reinterpret_cast<device const packed_uchar4*>(q + 8);
    packed_uchar4 q3 = *reinterpret_cast<device const packed_uchar4*>(q + 12);

    local += float(int(q0[0] & 0xF) - 8) * xv0[0] + float(int(q0[0] >> 4) - 8) * xv0[1];
    local += float(int(q0[1] & 0xF) - 8) * xv0[2] + float(int(q0[1] >> 4) - 8) * xv0[3];
    local += float(int(q0[2] & 0xF) - 8) * xv1[0] + float(int(q0[2] >> 4) - 8) * xv1[1];
    local += float(int(q0[3] & 0xF) - 8) * xv1[2] + float(int(q0[3] >> 4) - 8) * xv1[3];

    local += float(int(q1[0] & 0xF) - 8) * xv2[0] + float(int(q1[0] >> 4) - 8) * xv2[1];
    local += float(int(q1[1] & 0xF) - 8) * xv2[2] + float(int(q1[1] >> 4) - 8) * xv2[3];
    local += float(int(q1[2] & 0xF) - 8) * xv3[0] + float(int(q1[2] >> 4) - 8) * xv3[1];
    local += float(int(q1[3] & 0xF) - 8) * xv3[2] + float(int(q1[3] >> 4) - 8) * xv3[3];

    local += float(int(q2[0] & 0xF) - 8) * xv4[0] + float(int(q2[0] >> 4) - 8) * xv4[1];
    local += float(int(q2[1] & 0xF) - 8) * xv4[2] + float(int(q2[1] >> 4) - 8) * xv4[3];
    local += float(int(q2[2] & 0xF) - 8) * xv5[0] + float(int(q2[2] >> 4) - 8) * xv5[1];
    local += float(int(q2[3] & 0xF) - 8) * xv5[2] + float(int(q2[3] >> 4) - 8) * xv5[3];

    local += float(int(q3[0] & 0xF) - 8) * xv6[0] + float(int(q3[0] >> 4) - 8) * xv6[1];
    local += float(int(q3[1] & 0xF) - 8) * xv6[2] + float(int(q3[1] >> 4) - 8) * xv6[3];
    local += float(int(q3[2] & 0xF) - 8) * xv7[0] + float(int(q3[2] >> 4) - 8) * xv7[1];
    local += float(int(q3[3] & 0xF) - 8) * xv7[2] + float(int(q3[3] >> 4) - 8) * xv7[3];

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
    float x = gate[gid];
    // sqrt(2/pi) ≈ 0.7978845608
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    // Clamp to prevent tanh overflow (tanh saturates at ±1 for |x| > ~10)
    inner = clamp(inner, -10.0f, 10.0f);
    float gelu = 0.5f * x * (1.0f + tanh(inner));
    out[gid] = gelu * up[gid];
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
        float v0 = new_data[h * head_dim + g * 32 + i * 2];
        float v1 = new_data[h * head_dim + g * 32 + i * 2 + 1];
        int q0 = clamp(int(round(v0 * inv_scale)) + 8, 0, 15);
        int q1 = clamp(int(round(v1 * inv_scale)) + 8, 0, 15);
        cache[base_offset + 2 + i] = uchar(q0 | (q1 << 4));
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
        float v0 = new_data[h * seq_len * head_dim + s * head_dim + g * 32 + i * 2];
        float v1 = new_data[h * seq_len * head_dim + s * head_dim + g * 32 + i * 2 + 1];
        int q0 = clamp(int(round(v0 * inv_scale)) + 8, 0, 15);
        int q1 = clamp(int(round(v1 * inv_scale)) + 8, 0, 15);
        cache[base_offset + 2 + i] = uchar(q0 | (q1 << 4));
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
        float v0 = new_data[h * source_seq_stride * head_dim + (source_start + s) * head_dim + g * 32 + i * 2];
        float v1 = new_data[h * source_seq_stride * head_dim + (source_start + s) * head_dim + g * 32 + i * 2 + 1];
        int q0 = clamp(int(round(v0 * inv_scale)) + 8, 0, 15);
        int q1 = clamp(int(round(v1 * inv_scale)) + 8, 0, 15);
        cache[base_offset + 2 + i] = uchar(q0 | (q1 << 4));
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
    uint d_in_group = d % 32;
    uint offset = head_base + pos * row_bytes + g * 18;
    float scale = float(*reinterpret_cast<device const half*>(&cache[offset]));
    uint byte_idx = d_in_group / 2;
    uchar packed = cache[offset + 2 + byte_idx];
    if (d_in_group & 1) {
        return float(int(packed >> 4) - 8) * scale;
    } else {
        return float(int(packed & 0xF) - 8) * scale;
    }
}

// Vectorized variant: reads 4 consecutive Q4_0 values starting at d (d must be a multiple of 4).
// Each 32-element group is stored as 16 packed bytes, so 4 values span exactly 2 bytes.
inline float4 q4_0_read4(device const uchar* cache, uint head_base, uint pos, uint row_bytes, uint d) {
    uint g = d / 32;
    uint d_in_group = d % 32;
    uint offset = head_base + pos * row_bytes + g * 18;
    float scale = float(*reinterpret_cast<device const half*>(&cache[offset]));
    uint byte_idx = d_in_group / 2;
    uchar b0 = cache[offset + 2 + byte_idx];
    uchar b1 = cache[offset + 2 + byte_idx + 1];
    return float4(
        float(int(b0 & 0xF) - 8) * scale,
        float(int(b0 >> 4) - 8) * scale,
        float(int(b1 & 0xF) - 8) * scale,
        float(int(b1 >> 4) - 8) * scale
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
