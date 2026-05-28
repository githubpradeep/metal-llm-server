#include <metal_stdlib>
using namespace metal;

// ─── Matrix-Vector Multiply ──────────────────────────────────────────────────
// Computes y = W * x where W is (M, K) and x is (K,), y is (M,)
// Each thread computes one output element.
// Dispatched with threadgroups covering M rows.

kernel void matvec(
    device const float* W [[buffer(0)]],   // (M, K) row-major
    device const float* x [[buffer(1)]],   // (K,)
    device float* y [[buffer(2)]],         // (M,)
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= M) return;
    
    uint row_offset = gid * K;
    float acc = 0.0f;
    
    // Vectorized accumulation (4-wide)
    uint k = 0;
    for (; k + 4 <= K; k += 4) {
        float4 w = float4(W[row_offset + k], W[row_offset + k + 1],
                          W[row_offset + k + 2], W[row_offset + k + 3]);
        float4 xv = float4(x[k], x[k + 1], x[k + 2], x[k + 3]);
        acc += dot(w, xv);
    }
    for (; k < K; k++) {
        acc += W[row_offset + k] * x[k];
    }
    
    y[gid] = acc;
}

// ─── Batched Matrix Multiply (for prefill) ───────────────────────────────────
// C = A * B^T where A is (M, K), B is (N, K), C is (M, N)
// Each thread computes one element of C.

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
    threadgroup float scores[1024];  // max kv_seq
    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];
    
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
