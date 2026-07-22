#include <metal_stdlib>
using namespace metal;

// TurboQuant rotation: Y = X @ M
//
// X : [rows, dim]   row-major f32   (each row is one head's vector for one token)
// Y : [rows, dim]   row-major f32   (must not alias X)
// M : [dim, dim]    row-major f32   (M[k*dim + c]) — Rᵀ for forward rotation,
//                                    R for the inverse (un-rotation)
//
// One threadgroup per row. The row of X is staged in threadgroup memory and each
// thread strides over the output columns computing a full dot product against a
// column of M. `dim` is bounded by the model's max head dimension (<= 512), so the
// staging buffer is a fixed 512 floats (2 KiB).
kernel void turboquant_rotate(
    device const float* X       [[buffer(0)]],
    device float*       Y       [[buffer(1)]],
    device const float* M       [[buffer(2)]],
    constant uint&      dim     [[buffer(3)]],
    uint  row     [[threadgroup_position_in_grid]],
    uint  tid     [[thread_position_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]])
{
    threadgroup float xs[512];

    const uint base = row * dim;

    // Stage the input row.
    for (uint k = tid; k < dim; k += tg_size) {
        xs[k] = X[base + k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Y[row, c] = sum_k xs[k] * M[k*dim + c]
    for (uint c = tid; c < dim; c += tg_size) {
        float acc = 0.0f;
        uint mi = c;                 // M[0*dim + c]
        for (uint k = 0; k < dim; k++) {
            acc += xs[k] * M[mi];
            mi += dim;               // step to M[(k+1)*dim + c]
        }
        Y[base + c] = acc;
    }
}

// ─── TurboQuant V3 (2/3-bit Lloyd–Max) ───────────────────────────────────────
//
// Row layout in the cache (one row per (kv_head, token)):
//   [ fp16 L2 norm | bit-packed per-coordinate codebook indices ]
// Indices are packed LSB-first: coordinate d occupies bits [d*bits, d*bits+bits).
// row_bytes = 2 + ceil(head_dim*bits/8) (always exact for head_dim % 32 == 0).

// Scores live in device memory (not threadgroup): M1 Pro tg smem is ~32 KB, and
// float scores[8192] alone is 32 KB. Device buffer supports full ctx (≤16384).
#define TQ_MAX_KV 16384

// Read the `bits`-wide packed index for coordinate `d` from a row's index bytes.
inline uint tq_unpack(device const uchar* qs, uint d, uint bits, uint mask) {
    uint bitpos = d * bits;
    uint bi = bitpos >> 3;
    uint wi = bitpos & 7u;
    uint lo = qs[bi];
    uint hi = (wi + bits > 8u) ? uint(qs[bi + 1]) : 0u;
    return ((lo | (hi << 8)) >> wi) & mask;
}

// Quantize already-rotated vectors into the V3 layout. One thread per row
// (= per kv head); decode only ever appends a single token, so rows is tiny.
kernel void turboquant_quant_v3(
    device const float* in        [[buffer(0)]],  // [rows, head_dim] rotated f32
    device uchar*       cache     [[buffer(1)]],
    device const float* centroids [[buffer(2)]],  // [2^bits]
    constant uint&      head_dim  [[buffer(3)]],
    constant uint&      capacity  [[buffer(4)]],
    constant uint&      cur_seq   [[buffer(5)]],
    constant uint&      bits      [[buffer(6)]],
    constant uint&      row_bytes [[buffer(7)]],
    constant uint&      rows      [[buffer(8)]],
    device float*       window    [[buffer(9)]],  // [rows, rw, head_dim] f32 ring, or null
    constant uint&      rw        [[buffer(10)]], // residual window length (0 = disabled)
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    uint n_levels = 1u << bits;
    uint base_in = gid * head_dim;

    // Residual window: store this token's full-precision rotated vector in the ring
    // slot for its absolute position. Read back exactly during recent-token attention.
    if (rw > 0u) {
        uint slot = cur_seq % rw;
        device float* w = window + (gid * rw + slot) * head_dim;
        for (uint d = 0; d < head_dim; d++) w[d] = in[base_in + d];
    }

    // L2 norm (rotation-invariant, so equals the original vector's norm).
    float ss = 0.0f;
    for (uint d = 0; d < head_dim; d++) {
        float v = in[base_in + d];
        ss += v * v;
    }
    float norm = sqrt(ss);
    float inv = norm > 0.0f ? 1.0f / norm : 0.0f;

    uint row_off = gid * capacity * row_bytes + cur_seq * row_bytes;
    *reinterpret_cast<device half*>(&cache[row_off]) = half(norm);
    device uchar* qs = cache + row_off + 2;

    uint idx_bytes = row_bytes - 2;
    for (uint i = 0; i < idx_bytes; i++) qs[i] = 0;

    for (uint d = 0; d < head_dim; d++) {
        float u = in[base_in + d] * inv;
        // Nearest centroid (n_levels <= 16).
        uint best = 0;
        float bd = INFINITY;
        for (uint i = 0; i < n_levels; i++) {
            float dd = fabs(u - centroids[i]);
            if (dd < bd) { bd = dd; best = i; }
        }
        uint bitpos = d * bits;
        uint bi = bitpos >> 3;
        uint wi = bitpos & 7u;
        qs[bi] |= uchar((best << wi) & 0xFFu);
        if (wi + bits > 8u) {
            qs[bi + 1] |= uchar(best >> (8u - wi));
        }
    }
}

// Threadgroup reduction helpers (one partial per simdgroup, combined by lane 0).
// `red` must hold at least ceil(tg_size/32) floats.
inline float tq_tg_max(float v, threadgroup float* red, uint tid, uint tg_size,
                        uint sg_id, uint sg_lane) {
    float sg = simd_max(v);
    if (sg_lane == 0) red[sg_id] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = -INFINITY;
    uint nsg = (tg_size + 31u) / 32u;
    for (uint i = 0; i < nsg; i++) m = max(m, red[i]);
    return m;
}
inline float tq_tg_sum(float v, threadgroup float* red, uint tid, uint tg_size,
                       uint sg_id, uint sg_lane) {
    float sg = simd_sum(v);
    if (sg_lane == 0) red[sg_id] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    uint nsg = (tg_size + 31u) / 32u;
    for (uint i = 0; i < nsg; i++) s += red[i];
    return s;
}

// Fused rotate + quantize + residual-window store. One threadgroup per row
// (= per kv head). Reads the *un-rotated* (qk-normed / RoPE'd) vector directly,
// rotates it into the Haar frame in threadgroup memory, then computes the L2
// norm, stores the full-precision rotated vector into the ring window, and packs
// the Lloyd–Max indices — all without any intermediate global buffer.
kernel void turboquant_rotate_quant_v3(
    device const float* in_normed [[buffer(0)]],  // [rows, seq_stride, head_dim] un-rotated
    device uchar*       cache     [[buffer(1)]],
    device const float* centroids [[buffer(2)]],  // [2^bits]
    constant uint&      head_dim  [[buffer(3)]],
    constant uint&      capacity  [[buffer(4)]],
    constant uint&      cur_seq   [[buffer(5)]],
    constant uint&      bits      [[buffer(6)]],
    constant uint&      row_bytes [[buffer(7)]],
    constant uint&      rows      [[buffer(8)]],
    device float*       window    [[buffer(9)]],   // [rows, rw, head_dim] f32 ring, or null
    constant uint&      rw        [[buffer(10)]],
    device const float* fwd       [[buffer(11)]],  // Rᵀ rotation matrix [head_dim, head_dim]
    constant uint&      seq_stride [[buffer(12)]], // 1 = decode contiguous; prefill = seq_len
    constant uint&      token_index [[buffer(13)]], // which token along seq_stride
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint row     [[threadgroup_position_in_grid]])
{
    if (row >= rows) return;
    uint n_levels = 1u << bits;
    // Prefill layout is (num_kv_heads, seq_stride, head_dim); decode uses stride=1.
    uint base_in = row * seq_stride * head_dim + token_index * head_dim;

    threadgroup float xs[512];     // staged input row
    threadgroup float rot[512];    // rotated vector
    threadgroup uint  idxs[512];   // per-coordinate centroid index
    threadgroup float red[32];
    threadgroup float tg_norm = 0.0f;

    for (uint k = tid; k < head_dim; k += tg_size) xs[k] = in_normed[base_in + k];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // rot[c] = sum_k xs[k] * fwd[k*dim + c]  (coalesced fwd reads across threads)
    for (uint c = tid; c < head_dim; c += tg_size) {
        float acc = 0.0f;
        uint mi = c;
        for (uint k = 0; k < head_dim; k++) { acc += xs[k] * fwd[mi]; mi += head_dim; }
        rot[c] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // L2 norm via parallel reduction.
    float local_ss = 0.0f;
    for (uint d = tid; d < head_dim; d += tg_size) local_ss += rot[d] * rot[d];
    float ss = tq_tg_sum(local_ss, red, tid, tg_size, sg_id, sg_lane);
    if (tid == 0) tg_norm = sqrt(ss);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float norm = tg_norm;
    float invn = norm > 0.0f ? 1.0f / norm : 0.0f;

    // Full-precision residual window store (rotated vector).
    if (rw > 0u) {
        uint slot = cur_seq % rw;
        device float* w = window + (row * rw + slot) * head_dim;
        for (uint d = tid; d < head_dim; d += tg_size) w[d] = rot[d];
    }

    // Nearest centroid for each coordinate of the unit-normalized rotated vector.
    for (uint d = tid; d < head_dim; d += tg_size) {
        float u = rot[d] * invn;
        uint best = 0;
        float bd = INFINITY;
        for (uint i = 0; i < n_levels; i++) {
            float dd = fabs(u - centroids[i]);
            if (dd < bd) { bd = dd; best = i; }
        }
        idxs[d] = best;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 writes the norm and bit-packs the indices (adjacent coords share a
    // byte, so packing is serialized to avoid read-modify-write races).
    uint row_off = row * capacity * row_bytes + cur_seq * row_bytes;
    if (tid == 0) {
        *reinterpret_cast<device half*>(&cache[row_off]) = half(norm);
        device uchar* qs = cache + row_off + 2;
        uint idx_bytes = row_bytes - 2;
        for (uint i = 0; i < idx_bytes; i++) qs[i] = 0;
        for (uint d = 0; d < head_dim; d++) {
            uint best = idxs[d];
            uint bitpos = d * bits;
            uint bi = bitpos >> 3;
            uint wi = bitpos & 7u;
            qs[bi] |= uchar((best << wi) & 0xFFu);
            if (wi + bits > 8u) qs[bi + 1] |= uchar(best >> (8u - wi));
        }
    }
}

// Spill: convert a contiguous range of model-frame Q4_0 hot-ring rows into TQ V3
// (Haar rotate + Lloyd–Max pack). One threadgroup per (kv_head, token).
// Used once when decode first exceeds TURBOQUANT_HOT_WINDOW.
kernel void turboquant_spill_q4_to_v3(
    device const uchar* q4_cache  [[buffer(0)]],  // [rows, hot_cap, q4_row_bytes]
    device uchar*       tq_cache  [[buffer(1)]],  // [rows, tq_cap, tq_row_bytes]
    device const float* centroids [[buffer(2)]],
    device const float* fwd       [[buffer(3)]],  // Rᵀ
    device float*       window    [[buffer(4)]],  // residual ring or null
    constant uint&      head_dim  [[buffer(5)]],
    constant uint&      hot_cap   [[buffer(6)]],
    constant uint&      tq_cap    [[buffer(7)]],
    constant uint&      n_tokens  [[buffer(8)]],  // spill positions [0, n_tokens)
    constant uint&      bits      [[buffer(9)]],
    constant uint&      q4_row_bytes [[buffer(10)]],
    constant uint&      tq_row_bytes [[buffer(11)]],
    constant uint&      rows      [[buffer(12)]],
    constant uint&      rw        [[buffer(13)]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint tgid    [[threadgroup_position_in_grid]])
{
    // Linear TG id: row-major over (kv_head, token).
    uint row = tgid / max(n_tokens, 1u);
    uint pos = tgid % max(n_tokens, 1u);
    if (row >= rows || pos >= n_tokens) return;

    uint n_levels = 1u << bits;
    threadgroup float xs[512];
    threadgroup float rot[512];
    threadgroup uint  idxs[512];
    threadgroup float red[32];
    threadgroup float tg_norm = 0.0f;

    // Dequant one Q4_0 row into xs[].
    uint q4_base = row * hot_cap * q4_row_bytes + pos * q4_row_bytes;
    for (uint d = tid; d < head_dim; d += tg_size) {
        uint g = d / 32;
        uint e = d % 32;
        uint offset = q4_base + g * 18;
        float scale = float(*reinterpret_cast<device const half*>(&q4_cache[offset]));
        device const uchar* qs = q4_cache + offset + 2;
        float v;
        if (e < 16) {
            v = float(int(qs[e] & 0xF) - 8) * scale;
        } else {
            v = float(int(qs[e - 16] >> 4) - 8) * scale;
        }
        xs[d] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Rotate into Haar frame.
    for (uint c = tid; c < head_dim; c += tg_size) {
        float acc = 0.0f;
        uint mi = c;
        for (uint k = 0; k < head_dim; k++) { acc += xs[k] * fwd[mi]; mi += head_dim; }
        rot[c] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_ss = 0.0f;
    for (uint d = tid; d < head_dim; d += tg_size) local_ss += rot[d] * rot[d];
    float ss = tq_tg_sum(local_ss, red, tid, tg_size, sg_id, sg_lane);
    if (tid == 0) tg_norm = sqrt(ss);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float norm = tg_norm;
    float invn = norm > 0.0f ? 1.0f / norm : 0.0f;

    if (rw > 0u) {
        uint slot = pos % rw;
        device float* w = window + (row * rw + slot) * head_dim;
        for (uint d = tid; d < head_dim; d += tg_size) w[d] = rot[d];
    }

    for (uint d = tid; d < head_dim; d += tg_size) {
        float u = rot[d] * invn;
        uint best = 0;
        float bd = INFINITY;
        for (uint i = 0; i < n_levels; i++) {
            float dd = fabs(u - centroids[i]);
            if (dd < bd) { bd = dd; best = i; }
        }
        idxs[d] = best;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row_off = row * tq_cap * tq_row_bytes + pos * tq_row_bytes;
    if (tid == 0) {
        *reinterpret_cast<device half*>(&tq_cache[row_off]) = half(norm);
        device uchar* qs = tq_cache + row_off + 2;
        uint idx_bytes = tq_row_bytes - 2;
        for (uint i = 0; i < idx_bytes; i++) qs[i] = 0;
        for (uint d = 0; d < head_dim; d++) {
            uint best = idxs[d];
            uint bitpos = d * bits;
            uint bi = bitpos >> 3;
            uint wi = bitpos & 7u;
            qs[bi] |= uchar((best << wi) & 0xFFu);
            if (wi + bits > 8u) qs[bi + 1] |= uchar(best >> (8u - wi));
        }
    }
}

// Single-token (decode) attention over the V3 cache, fully fused: one threadgroup
// per query head rotates Q into the Haar frame, runs the attention (with parallel
// softmax reductions), then un-rotates the output back to the model frame — no
// intermediate global buffers. Q_in and `output` are both in the model frame.
//
// Perf notes vs the original scalar kernel:
//   - Unpacks 2/3-bit indices with wider byte loads + float4 dots where possible
//   - V accumulation tiles over positions (reuses score weights in registers)
kernel void turboquant_attn_v3(
    device const float* Q_in      [[buffer(0)]],  // [num_heads, head_dim] un-rotated
    device const uchar* K_cache   [[buffer(1)]],
    device const uchar* V_cache   [[buffer(2)]],
    device float*       output    [[buffer(3)]],  // [num_heads, head_dim] model frame
    constant uint&      num_heads     [[buffer(4)]],
    constant uint&      num_kv_heads  [[buffer(5)]],
    constant uint&      num_kv_groups [[buffer(6)]],
    constant uint&      head_dim      [[buffer(7)]],
    constant uint&      kv_seq        [[buffer(8)]],
    constant uint&      capacity      [[buffer(9)]],
    constant float&     scale         [[buffer(10)]],
    constant uint&      kv_start      [[buffer(11)]],
    constant uint&      k_bits        [[buffer(12)]],
    constant uint&      k_row_bytes   [[buffer(13)]],
    device const float* k_centroids   [[buffer(14)]],
    device const float* Kwin          [[buffer(15)]],  // [num_kv_heads, rw, head_dim] f32
    device const float* Vwin          [[buffer(16)]],
    constant uint&      rw            [[buffer(17)]],   // residual window length (0 = disabled)
    constant uint&      window_lo     [[buffer(18)]],   // smallest absolute pos served from window
    device const float* fwd           [[buffer(19)]],   // Rᵀ — rotate Q into Haar frame
    device const float* inv           [[buffer(20)]],   // R  — un-rotate output
    constant uint&      v_bits        [[buffer(21)]],
    constant uint&      v_row_bytes   [[buffer(22)]],
    device const float* v_centroids   [[buffer(23)]],
    device float*       scores_all    [[buffer(24)]],  // [num_heads, capacity] device scores
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint tgid    [[threadgroup_position_in_grid]])
{
    uint h = tgid;
    if (h >= num_heads) return;

    uint kv_h = h / num_kv_groups;
    uint q_off = h * head_dim;
    uint kv_base_k = kv_h * capacity * k_row_bytes;   // K and V rows may differ in size
    uint kv_base_v = kv_h * capacity * v_row_bytes;
    uint k_mask = (1u << k_bits) - 1u;
    uint v_mask = (1u << v_bits) - 1u;
    device float* scores = scores_all + h * capacity;

    threadgroup float cen_k[16];   // key codebook (2^k_bits)
    threadgroup float cen_v[16];   // value codebook (2^v_bits)
    threadgroup float qrot[512];   // rotated query
    threadgroup float orot[512];   // staging for Q row, then rotated output
    threadgroup float red[32];
    threadgroup float tg_m;
    threadgroup float tg_inv_l;

    for (uint i = tid; i < (1u << k_bits); i += tg_size) cen_k[i] = k_centroids[i];
    for (uint i = tid; i < (1u << v_bits); i += tg_size) cen_v[i] = v_centroids[i];

    // Rotate Q into the Haar frame: qrot[c] = sum_k Q_in[k] * fwd[k*dim + c].
    for (uint k = tid; k < head_dim; k += tg_size) orot[k] = Q_in[q_off + k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = tid; c < head_dim; c += tg_size) {
        float acc = 0.0f;
        uint mi = c;
        for (uint k = 0; k < head_dim; k++) { acc += orot[k] * fwd[mi]; mi += head_dim; }
        qrot[c] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n_pos = (kv_seq > kv_start) ? (kv_seq - kv_start) : 0u;
    // Soft clamp only if caller undersized the scores buffer; prefer capacity.
    n_pos = min(n_pos, capacity);
    n_pos = min(n_pos, (uint)TQ_MAX_KV);

    // Pass 1: scores = scale * norm_k * <q_rot, dequant(k_rot)>. Recent positions
    // (pos >= window_lo) are served from the full-precision residual window instead.
    for (uint j = tid; j < n_pos; j += tg_size) {
        uint pos = kv_start + j;
        float sc;
        if (rw > 0u && pos >= window_lo) {
            device const float* kw = Kwin + (kv_h * rw + (pos % rw)) * head_dim;
            float score_dot = 0.0f;
            for (uint d = 0; d + 3 < head_dim; d += 4) {
                float4 qv = float4(qrot[d], qrot[d+1], qrot[d+2], qrot[d+3]);
                float4 kv = float4(kw[d], kw[d+1], kw[d+2], kw[d+3]);
                score_dot += metal::dot(qv, kv);
            }
            for (uint d = (head_dim & ~3u); d < head_dim; d++) score_dot += qrot[d] * kw[d];
            sc = score_dot * scale;  // window stores full-magnitude rotated K (no separate norm)
        } else {
            uint row = kv_base_k + pos * k_row_bytes;
            float norm = float(*reinterpret_cast<device const half*>(&K_cache[row]));
            device const uchar* qs = K_cache + row + 2;
            float score_dot = 0.0f;
            // Unroll by 4: pack 4 centroid lookups. Bit unpack still scalar per coord
            // but float4 MAC amortizes the arithmetic.
            for (uint d = 0; d + 3 < head_dim; d += 4) {
                float4 qv = float4(qrot[d], qrot[d+1], qrot[d+2], qrot[d+3]);
                float4 kv = float4(
                    cen_k[tq_unpack(qs, d + 0, k_bits, k_mask)],
                    cen_k[tq_unpack(qs, d + 1, k_bits, k_mask)],
                    cen_k[tq_unpack(qs, d + 2, k_bits, k_mask)],
                    cen_k[tq_unpack(qs, d + 3, k_bits, k_mask)]);
                score_dot += metal::dot(qv, kv);
            }
            for (uint d = (head_dim & ~3u); d < head_dim; d++) {
                score_dot += qrot[d] * cen_k[tq_unpack(qs, d, k_bits, k_mask)];
            }
            sc = score_dot * scale * norm;
        }
        scores[j] = sc;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Softmax max (parallel simdgroup reduction).
    float local_m = -INFINITY;
    for (uint j = tid; j < n_pos; j += tg_size) local_m = max(local_m, scores[j]);
    float m = tq_tg_max(local_m, red, tid, tg_size, sg_id, sg_lane);
    if (tid == 0) tg_m = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = tg_m;

    for (uint j = tid; j < n_pos; j += tg_size) scores[j] = exp(scores[j] - m);
    threadgroup_barrier(mem_flags::mem_device);

    float local_l = 0.0f;
    for (uint j = tid; j < n_pos; j += tg_size) local_l += scores[j];
    float l = tq_tg_sum(local_l, red, tid, tg_size, sg_id, sg_lane);
    if (tid == 0) tg_inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_l = tg_inv_l;

    // Pass 2: orot[d] = (sum_j w_j * v_j[d]) / l.
    // Outer loop over output dims (parallel across threads); inner tiles over j
    // so each thread streams scores sequentially with good locality.
    constexpr uint J_TILE = 16;
    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint j0 = 0; j0 < n_pos; j0 += J_TILE) {
            uint j_end = min(j0 + J_TILE, n_pos);
            for (uint j = j0; j < j_end; j++) {
                uint pos = kv_start + j;
                float vv;
                if (rw > 0u && pos >= window_lo) {
                    device const float* vw = Vwin + (kv_h * rw + (pos % rw)) * head_dim;
                    vv = vw[d];
                } else {
                    uint row = kv_base_v + pos * v_row_bytes;
                    float norm = float(*reinterpret_cast<device const half*>(&V_cache[row]));
                    device const uchar* qs = V_cache + row + 2;
                    uint idx = tq_unpack(qs, d, v_bits, v_mask);
                    vv = norm * cen_v[idx];
                }
                acc += scores[j] * vv;
            }
        }
        orot[d] = acc * inv_l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Un-rotate the output back to the model frame: out[c] = sum_k orot[k]*inv[k*dim+c].
    for (uint c = tid; c < head_dim; c += tg_size) {
        float acc = 0.0f;
        uint mi = c;
        for (uint k = 0; k < head_dim; k++) { acc += orot[k] * inv[mi]; mi += head_dim; }
        output[q_off + c] = acc;
    }
}
