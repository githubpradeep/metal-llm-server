#include <metal_stdlib>
using namespace metal;

// ─── Mega decode kernel ──────────────────────────────────────────────────────
// One GPU dispatch per decode token. A host-built op list runs sequentially
// inside a single threadgroup (256 threads). Cross-TG device atomics hang on
// Metal, so matvecs run serially within the TG until parallel sync is fixed.

constant uint MEGA_TG_THREADS = 256;
constant uint MEGA_SINGLE_TG = 1u;
constant uint MEGA_ATTN_TILE = 4u;
// TILE * (MEGA_TG_THREADS / SIMD_SIZE) partial dots per attention tile
constant uint MEGA_ATTN_SCORES = MEGA_ATTN_TILE * (MEGA_TG_THREADS / SIMD_SIZE);

// Scratch buffer indices (must match mega_decode.rs MegaBuf)
constant uint MB_HIDDEN   = 0;
constant uint MB_NORMED   = 1;
constant uint MB_Q        = 2;
constant uint MB_K        = 3;
constant uint MB_V        = 4;
constant uint MB_QN       = 5;
constant uint MB_KN       = 6;
constant uint MB_VN       = 7;
constant uint MB_ATTN     = 8;
constant uint MB_O        = 9;
constant uint MB_GATE     = 10;
constant uint MB_UP       = 11;
constant uint MB_GELU     = 12;
constant uint MB_DOWN     = 13;
constant uint MB_PLE_CTX  = 14;
constant uint MB_PLE_TMP  = 15;
constant uint MB_PLE_TOK  = 16;
constant uint MB_LOGITS   = 17;

enum MegaOpType : uint {
    OpRmsNorm = 0,
    OpRmsNormPerHead = 1,
    OpRmsNormPerHeadNoweight = 2,
    OpRmsNormAddSaveRes = 3,
    OpMatvecQ4 = 4,
    OpMatvecF16 = 5,
    OpRotaryQ = 6,
    OpRotaryK = 7,
    OpKvAppendQ4 = 8,
    OpAttentionQ4 = 9,
    OpVecAdd = 10,
    OpVecScale = 11,
    OpGeluMul = 12,
    OpGeluMulAt = 13,
    OpVecAddScaled = 14,
};

struct MegaParams {
    uint num_ops;
    uint hidden_size;
    uint q_out;
    uint kv_out;
    uint intermediate_size;
    uint vocab_size;
    uint num_heads;
    uint num_kv_heads;
    uint num_kv_groups;
    uint head_dim;
    uint kv_capacity;
    uint kv_seq;
    uint kv_cache_type; // 2 = q4_0
    uint groups_per_row;
    uint row_bytes;
    float eps;
    float attn_scale;
    float ple_input_scale;
    float context_proj_scale;
    float final_logit_cap;
};

struct MegaOpDesc {
    uint op_type;
    uint arg0;
    uint arg1;
    uint arg2;
    uint arg3;
    uint num_tgs;
    uint in_buf;
    uint out_buf;
    uint aux_buf;
    uint weight_buf_idx;
};

inline device float* mega_buf(device float* bufs[18], uint idx) {
    return bufs[idx];
}

inline void mega_rmsnorm(
    device float* x,
    device const float* weight,
    device float* out,
    uint dim,
    float eps,
    threadgroup float* shared_sum,
    uint tid,
    uint tg_size
) {
    float partial = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float v = x[i];
        partial += v * v;
    }
    shared_sum[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shared_sum[0] / float(dim) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < dim; i += tg_size) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

inline void mega_rmsnorm_per_head(
    device float* x,
    device const float* weight,
    device float* out,
    uint num_heads,
    uint head_dim,
    float eps,
    uint tid,
    uint tg_size
) {
    for (uint h = tid; h < num_heads; h += tg_size) {
        uint base = h * head_dim;
        float sum = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            float v = x[base + d];
            sum += v * v;
        }
        float inv = rsqrt(sum / float(head_dim) + eps);
        for (uint d = 0; d < head_dim; d++) {
            out[base + d] = x[base + d] * inv * weight[d];
        }
    }
}

inline void mega_rmsnorm_per_head_noweight(
    device float* x,
    device float* out,
    uint num_heads,
    uint head_dim,
    float eps,
    uint tid,
    uint tg_size
) {
    for (uint h = tid; h < num_heads; h += tg_size) {
        uint base = h * head_dim;
        float sum = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            float v = x[base + d];
            sum += v * v;
        }
        float inv = rsqrt(sum / float(head_dim) + eps);
        for (uint d = 0; d < head_dim; d++) {
            out[base + d] = x[base + d] * inv;
        }
    }
}

inline void mega_rmsnorm_add_save(
    device float* a,
    device const float* b,
    device const float* weight,
    device float* out,
    device float* residual_out,
    uint dim,
    float eps,
    threadgroup float* shared_sum,
    uint tid,
    uint tg_size
) {
    float partial = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float v = a[i] + b[i];
        residual_out[i] = v;
        partial += v * v;
    }
    shared_sum[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shared_sum[0] / float(dim) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < dim; i += tg_size) {
        out[i] = residual_out[i] * inv_rms * weight[i];
    }
}

inline void mega_matvec_q4_tg(
    device const uchar* W,
    device const float* x,
    device float* y,
    uint M,
    uint K,
    uint mega_tg,
    uint num_tgs,
    uint sgid,
    uint lane
) {
    uint rows_per_tg = Q4F_SG * 4u;
    uint total_tgs = (M + rows_per_tg - 1u) / rows_per_tg;
    for (uint vtgid = mega_tg; vtgid < total_tgs; vtgid += num_tgs) {
        matvec_q4_fast_body<4>(W, x, y, M, K, vtgid, sgid, lane, Q4F_SG);
    }
}

inline void mega_matvec_f16_tg(
    device const half* W,
    device const float* x,
    device float* y,
    uint M,
    uint K,
    uint mega_tg,
    uint num_tgs,
    uint tid
) {
    for (uint row = mega_tg; row < M; row += num_tgs) {
        uint row_offset = row * K;
        float acc = 0.0f;
        uint k = tid * 4;
        uint stride = SIMD_SIZE * 4;
        for (; k + 3 < K; k += stride) {
            half4 w = *reinterpret_cast<device const half4*>(&W[row_offset + k]);
            float4 xv = *reinterpret_cast<device const float4*>(&x[k]);
            acc += dot(float4(w), xv);
        }
        for (uint kk = tid + (K / stride) * stride; kk < K; kk += SIMD_SIZE) {
            acc += float(W[row_offset + kk]) * x[kk];
        }
        acc = simd_sum(acc);
        if (tid == 0) y[row] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

inline void mega_rotary_q(
    device float* q,
    device const float* cos_buf,
    device const float* sin_buf,
    uint num_heads,
    uint head_dim,
    uint tid,
    uint tg_size
) {
    uint half_dim = head_dim / 2;
    uint total = num_heads * half_dim;
    for (uint gid = tid; gid < total; gid += tg_size) {
        uint h = gid / half_dim;
        uint d = gid % half_dim;
        uint base = h * head_dim;
        float q1 = q[base + d];
        float q2 = q[base + d + half_dim];
        float c = cos_buf[d];
        float s = sin_buf[d];
        q[base + d] = q1 * c - q2 * s;
        q[base + d + half_dim] = q2 * c + q1 * s;
    }
}

inline void mega_rotary_k(
    device float* k,
    device const float* cos_buf,
    device const float* sin_buf,
    uint num_kv_heads,
    uint head_dim,
    uint tid,
    uint tg_size
) {
    uint half_dim = head_dim / 2;
    uint total = num_kv_heads * half_dim;
    for (uint gid = tid; gid < total; gid += tg_size) {
        uint h = gid / half_dim;
        uint d = gid % half_dim;
        uint base = h * head_dim;
        float k1 = k[base + d];
        float k2 = k[base + d + half_dim];
        float c = cos_buf[d];
        float s = sin_buf[d];
        k[base + d] = k1 * c - k2 * s;
        k[base + d + half_dim] = k2 * c + k1 * s;
    }
}

inline void mega_kv_append_q4(
    device const float* new_data,
    device uchar* cache,
    uint num_kv_heads,
    uint head_dim,
    uint capacity,
    uint cur_seq,
    uint groups_per_row,
    uint row_bytes,
    uint tid,
    uint tg_size
) {
    uint total_groups = num_kv_heads * groups_per_row;
    for (uint g = tid; g < total_groups; g += tg_size) {
        uint h = g / groups_per_row;
        uint gi = g % groups_per_row;
        uint d0 = gi * 32;
        uint base = h * head_dim + d0;
        uint dst = h * capacity * row_bytes + cur_seq * row_bytes + gi * 18;
        float vals[32];
        for (uint i = 0; i < 32; i++) vals[i] = new_data[base + i];
        float max_abs = 0.0f;
        for (uint i = 0; i < 32; i++) max_abs = max(max_abs, abs(vals[i]));
        float scale = max_abs / 7.0f;
        if (max_abs == 0.0f) scale = 1.0f;
        float inv_scale = 1.0f / scale;
        half scale_h = half(scale);
        *reinterpret_cast<device half*>(&cache[dst]) = scale_h;
        for (uint i = 0; i < 16; i++) {
            float v0 = vals[i * 2];
            float v1 = vals[i * 2 + 1];
            int q0 = clamp(int(round(v0 * inv_scale)) + 8, 0, 15);
            int q1 = clamp(int(round(v1 * inv_scale)) + 8, 0, 15);
            cache[dst + 2 + i] = uchar(q0 | (q1 << 4));
        }
    }
}

inline void mega_attention_q4_head(
    device const float* Q,
    device const uchar* K_cache,
    device const uchar* V_cache,
    device float* output,
    uint h,
    uint num_kv_groups,
    uint head_dim,
    uint kv_seq,
    uint kv_start,
    uint capacity,
    uint row_bytes,
    float scale,
    threadgroup float* shared_scores,
    threadgroup float* shared_exp,
    threadgroup float* shared_update,
    uint tid,
    uint tg_size
) {
    uint kv_h = h / num_kv_groups;
    uint q_offset = h * head_dim;
    uint k_head_base = kv_h * capacity * row_bytes;
    uint v_head_base = k_head_base;
    const uint TILE_KV = MEGA_ATTN_TILE;
    uint simd_id = tid / SIMD_SIZE;
    uint num_simds = tg_size / SIMD_SIZE;
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
        for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
            uint actual_pos = kv_start + kv_tile + kv_offset;
            float partial_dot = 0.0f;
            for (uint d = tid * 4; d + 3 < head_dim; d += tg_size * 4) {
                float4 qv = *reinterpret_cast<device const float4*>(&Q[q_offset + d]);
                float4 kv = q4_0_read4(K_cache, k_head_base, actual_pos, row_bytes, d);
                partial_dot += dot(qv, kv);
            }
            for (uint d = (head_dim / 4) * 4 + tid; d < head_dim; d += tg_size) {
                partial_dot += Q[q_offset + d]
                    * q4_0_read(K_cache, k_head_base, actual_pos, row_bytes, d);
            }
            shared_scores[kv_offset * num_simds + simd_id] = partial_dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = num_simds / 2; stride > 0; stride >>= 1) {
                if (simd_id < stride) {
                    shared_scores[kv_offset * num_simds + simd_id]
                        += shared_scores[kv_offset * num_simds + simd_id + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) shared_scores[kv_offset] = shared_scores[kv_offset] * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
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
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float old_factor = shared_update[2];
        float inv_l = shared_update[3];
        for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
            uint actual_pos = kv_start + kv_tile + kv_offset;
            for (uint d = tid * 4; d + 3 < head_dim; d += tg_size * 4) {
                float4 acc = shared_exp[kv_offset]
                    * q4_0_read4(V_cache, v_head_base, actual_pos, row_bytes, d);
                uint out_idx = q_offset + d;
                float4 ov = *reinterpret_cast<device float4*>(&output[out_idx]);
                ov = ov * old_factor + acc * inv_l;
                *reinterpret_cast<device float4*>(&output[out_idx]) = ov;
            }
            for (uint d = (head_dim / 4) * 4 + tid; d < head_dim; d += tg_size) {
                float acc = shared_exp[kv_offset]
                    * q4_0_read(V_cache, v_head_base, actual_pos, row_bytes, d);
                uint out_idx = q_offset + d;
                output[out_idx] = output[out_idx] * old_factor + acc * inv_l;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void decode_mega_gemma4_q4_0(
    constant MegaParams& P [[buffer(0)]],
    constant MegaOpDesc* ops [[buffer(1)]],
    device float* buf_hidden [[buffer(2)]],
    device float* buf_normed [[buffer(3)]],
    device float* buf_q [[buffer(4)]],
    device float* buf_k [[buffer(5)]],
    device float* buf_v [[buffer(6)]],
    device float* buf_qn [[buffer(7)]],
    device float* buf_kn [[buffer(8)]],
    device float* buf_vn [[buffer(9)]],
    device float* buf_attn [[buffer(10)]],
    device float* buf_o [[buffer(11)]],
    device float* buf_gate [[buffer(12)]],
    device float* buf_up [[buffer(13)]],
    device float* buf_gelu [[buffer(14)]],
    device float* buf_down [[buffer(15)]],
    device float* buf_ple_ctx [[buffer(16)]],
    device float* buf_ple_tmp [[buffer(17)]],
    device float* buf_ple_tok [[buffer(18)]],
    device float* buf_logits [[buffer(19)]],
    device const uint64_t* weight_q4 [[buffer(20)]],
    device const uint64_t* weight_f16 [[buffer(21)]],
    device const uint64_t* weight_f32 [[buffer(22)]],
    device const uint64_t* k_cache_table [[buffer(23)]],
    device const uint64_t* v_cache_table [[buffer(24)]],
    device const uint64_t* layer_cos [[buffer(25)]],
    device const uint64_t* layer_sin [[buffer(26)]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_shared_rms[256];
    threadgroup float tg_attn_scores[MEGA_ATTN_SCORES];
    threadgroup float tg_attn_exp[MEGA_ATTN_TILE];
    threadgroup float tg_attn_update[4];

    device float* bufs[18] = {
        buf_hidden, buf_normed, buf_q, buf_k, buf_v, buf_qn, buf_kn, buf_vn,
        buf_attn, buf_o, buf_gate, buf_up, buf_gelu, buf_down,
        buf_ple_ctx, buf_ple_tmp, buf_ple_tok, buf_logits
    };

    for (uint opi = 0; opi < P.num_ops; opi++) {
        constant MegaOpDesc& op = ops[opi];
        device float* in = bufs[op.in_buf];
        device float* out = bufs[op.out_buf];
        device float* aux = bufs[op.aux_buf];

        switch (op.op_type) {
        case OpRmsNorm: {
            device const float* w = (device const float*)weight_f32[op.weight_buf_idx];
            mega_rmsnorm(in, w, out, op.arg0, P.eps, tg_shared_rms, tid, MEGA_TG_THREADS);
            break;
        }
        case OpRmsNormPerHead: {
            device const float* w = (device const float*)weight_f32[op.weight_buf_idx];
            mega_rmsnorm_per_head(in, w, out, op.arg0, op.arg1, P.eps, tid, MEGA_TG_THREADS);
            break;
        }
        case OpRmsNormPerHeadNoweight:
            mega_rmsnorm_per_head_noweight(in, out, op.arg0, op.arg1, P.eps,
                                           tid, MEGA_TG_THREADS);
            break;
        case OpRmsNormAddSaveRes: {
            device const float* w = (device const float*)weight_f32[op.weight_buf_idx];
            mega_rmsnorm_add_save(in, aux, w, out, in, op.arg0, P.eps, tg_shared_rms,
                                  tid, MEGA_TG_THREADS);
            break;
        }
        case OpMatvecQ4: {
            device const uchar* w = (device const uchar*)weight_q4[op.weight_buf_idx];
            mega_matvec_q4_tg(w, in, out, op.arg0, op.arg1, 0u, MEGA_SINGLE_TG,
                              sgid, lane);
            break;
        }
        case OpMatvecF16: {
            device const half* w = (device const half*)weight_f16[op.weight_buf_idx];
            mega_matvec_f16_tg(w, in, out, op.arg0, op.arg1, 0u, MEGA_SINGLE_TG, tid);
            break;
        }
        case OpRotaryQ: {
            device const float* cos_b = (device const float*)layer_cos[op.arg0];
            device const float* sin_b = (device const float*)layer_sin[op.arg0];
            mega_rotary_q(in, cos_b, sin_b, P.num_heads, P.head_dim, tid, MEGA_TG_THREADS);
            break;
        }
        case OpRotaryK: {
            device const float* cos_b = (device const float*)layer_cos[op.arg0];
            device const float* sin_b = (device const float*)layer_sin[op.arg0];
            mega_rotary_k(in, cos_b, sin_b, P.num_kv_heads, P.head_dim, tid, MEGA_TG_THREADS);
            break;
        }
        case OpKvAppendQ4: {
            device uchar* cache = (op.arg1 == 0u)
                ? (device uchar*)k_cache_table[op.arg0]
                : (device uchar*)v_cache_table[op.arg0];
            mega_kv_append_q4(in, cache, P.num_kv_heads, P.head_dim,
                              P.kv_capacity, P.kv_seq, P.groups_per_row,
                              P.row_bytes, tid, MEGA_TG_THREADS);
            break;
        }
        case OpAttentionQ4: {
            device const uchar* kcache = (device const uchar*)k_cache_table[op.arg0];
            device const uchar* vcache = (device const uchar*)v_cache_table[op.arg0];
            for (uint h = 0; h < P.num_heads; h++) {
                mega_attention_q4_head(in, kcache, vcache, out,
                                       h, P.num_kv_groups, P.head_dim,
                                       op.arg1, op.arg2, P.kv_capacity, P.row_bytes,
                                       P.attn_scale,
                                       tg_attn_scores, tg_attn_exp, tg_attn_update,
                                       tid, MEGA_TG_THREADS);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            break;
        }
        case OpVecAdd:
            for (uint i = tid; i < op.arg0; i += MEGA_TG_THREADS) {
                out[i] = in[i] + aux[i];
            }
            break;
        case OpVecScale: {
            float s = as_type<float>(op.arg1);
            for (uint i = tid; i < op.arg0; i += MEGA_TG_THREADS) {
                out[i] = in[i] * s;
            }
            break;
        }
        case OpGeluMul:
            for (uint i = tid; i < op.arg0; i += MEGA_TG_THREADS) {
                float x = in[i];
                float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
                inner = clamp(inner, -10.0f, 10.0f);
                float gelu = 0.5f * x * (1.0f + tanh(inner));
                out[i] = gelu * aux[i];
            }
            break;
        case OpGeluMulAt: {
            uint ple_dim = op.arg1;
            uint layer = op.arg2;
            uint off = layer * ple_dim;
            for (uint i = tid; i < ple_dim; i += MEGA_TG_THREADS) {
                float x = in[i];
                float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
                inner = clamp(inner, -10.0f, 10.0f);
                float gelu = 0.5f * x * (1.0f + tanh(inner));
                out[i] = gelu * aux[off + i];
            }
            break;
        }
        case OpVecAddScaled: {
            float s = as_type<float>(op.arg1);
            for (uint i = tid; i < op.arg0; i += MEGA_TG_THREADS) {
                out[i] = in[i] + aux[i] * s;
            }
            break;
        }
        default:
            break;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
