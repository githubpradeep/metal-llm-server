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
// Each SIMD group (32 threads) handles one row.
// Each thread processes groups strided by SIMD_SIZE.
// Inner loop unrolled 4 bytes at a time for better ILP.

constant uint Q4_GROUP_SIZE = 32;
constant uint Q4_BLOCK_BYTES = 18;

kernel void matvec_q4(
    device const uchar* W_q4 [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (tgid >= M) return;
    
    uint num_groups = K / Q4_GROUP_SIZE;
    uint row_byte_offset = tgid * num_groups * Q4_BLOCK_BYTES;
    
    float acc = 0.0f;
    
    // Each thread handles groups strided by SIMD_SIZE
    for (uint g = tid; g < num_groups; g += SIMD_SIZE) {
        uint block_offset = row_byte_offset + g * Q4_BLOCK_BYTES;
        
        // Read scale
        half scale_h = *reinterpret_cast<device const half*>(&W_q4[block_offset]);
        float scale = float(scale_h);
        
        // Read packed quants
        device const uchar* quants = &W_q4[block_offset + 2];
        uint x_offset = g * Q4_GROUP_SIZE;
        
        // Unrolled: process 4 bytes (8 weights) per iteration
        float local_acc = 0.0f;
        for (uint i = 0; i < 16; i += 4) {
            uchar p0 = quants[i];
            uchar p1 = quants[i + 1];
            uchar p2 = quants[i + 2];
            uchar p3 = quants[i + 3];
            
            uint base = x_offset + i * 2;
            local_acc += float(int(p0 & 0x0F) - 8) * x[base];
            local_acc += float(int(p0 >> 4) - 8) * x[base + 1];
            local_acc += float(int(p1 & 0x0F) - 8) * x[base + 2];
            local_acc += float(int(p1 >> 4) - 8) * x[base + 3];
            local_acc += float(int(p2 & 0x0F) - 8) * x[base + 4];
            local_acc += float(int(p2 >> 4) - 8) * x[base + 5];
            local_acc += float(int(p3 & 0x0F) - 8) * x[base + 6];
            local_acc += float(int(p3 >> 4) - 8) * x[base + 7];
        }
        acc += local_acc * scale;
    }
    
    acc = simd_sum(acc);
    if (tid == 0) {
        y[tgid] = acc;
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
    
    // Shared memory for scores and reduction
    threadgroup float scores[2560];  // max kv_seq
    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];
    
    // Initialize shared memory to safe defaults
    shared_max[tid] = -INFINITY;
    shared_sum[tid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 1: Compute Q @ K^T scores (distributed across threads)
    float local_max = -INFINITY;
    for (uint kv = tid; kv < kv_seq; kv += tg_size) {
        float dot = 0.0f;
        uint k_offset = k_head_base + kv * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            dot += Q[q_offset + d] * K_cache[k_offset + d];
        }
        float s = dot * scale;
        scores[kv] = s;
        local_max = max(local_max, s);
    }
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce max
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_score = shared_max[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 2: Softmax (exp and sum)
    float local_sum = 0.0f;
    for (uint kv = tid; kv < kv_seq; kv += tg_size) {
        float e = exp(scores[kv] - max_score);
        scores[kv] = e;
        local_sum += e;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared_sum[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Normalize scores
    for (uint kv = tid; kv < kv_seq; kv += tg_size) {
        scores[kv] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 3: Weighted sum of V (each thread handles a subset of head_dim)
    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint kv = 0; kv < kv_seq; kv++) {
            acc += scores[kv] * V_cache[v_head_base + kv * head_dim + d];
        }
        output[q_offset + d] = acc;
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
    
    threadgroup float scores[2560];
    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];
    
    // Compute scores
    float local_max = -INFINITY;
    for (uint kv = tid; kv < attend_len; kv += tg_size) {
        float dot = 0.0f;
        uint k_offset = k_head_base + kv * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            dot += Q[q_offset + d] * K_cache[k_offset + d];
        }
        float s = dot * scale;
        scores[kv] = s;
        local_max = max(local_max, s);
    }
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_score = shared_max[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Softmax
    float local_sum = 0.0f;
    for (uint kv = tid; kv < attend_len; kv += tg_size) {
        float e = exp(scores[kv] - max_score);
        scores[kv] = e;
        local_sum += e;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared_sum[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint kv = tid; kv < attend_len; kv += tg_size) {
        scores[kv] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Weighted sum of V
    uint out_offset = (h * q_len + qi) * head_dim;
    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint kv = 0; kv < attend_len; kv++) {
            acc += scores[kv] * V_cache[v_head_base + kv * head_dim + d];
        }
        output[out_offset + d] = acc;
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
    float gelu = 0.5f * x * (1.0f + tanh(inner));
    out[gid] = gelu * up[gid];
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

    threadgroup float scores[2560];
    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];

    // Initialize shared memory to safe defaults
    shared_max[tid] = -INFINITY;
    shared_sum[tid] = 0.0f;
    for (uint i = tid; i < kv_seq; i += tg_size) {
        scores[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = -INFINITY;
    for (uint kv = tid; kv < kv_seq; kv += tg_size) {
        float dot = 0.0f;
        uint actual_pos = kv_start + kv;
        uint k_offset = k_head_base + actual_pos * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            dot += Q[q_offset + d] * K_cache[k_offset + d];
        }
        float s = dot * scale;
        scores[kv] = s;
        local_max = max(local_max, s);
    }
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_score = shared_max[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint kv = tid; kv < kv_seq; kv += tg_size) {
        float e = exp(scores[kv] - max_score);
        scores[kv] = e;
        local_sum += e;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared_sum[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv = tid; kv < kv_seq; kv += tg_size) {
        scores[kv] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint kv = 0; kv < kv_seq; kv++) {
            uint actual_pos = kv_start + kv;
            acc += scores[kv] * V_cache[v_head_base + actual_pos * head_dim + d];
        }
        output[q_offset + d] = acc;
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
