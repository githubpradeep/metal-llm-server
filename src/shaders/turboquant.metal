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

// Quantize an already Haar-rotated HSD prefill chunk in one dispatch.
// Grid: rows * seq_len threadgroups, one TG per (KV head, token).
kernel void turboquant_quant_v3_batch(
    device const float* rotated   [[buffer(0)]], // [rows, seq_len, head_dim]
    device uchar*       cache     [[buffer(1)]],
    device const float* centroids [[buffer(2)]],
    constant uint&      head_dim  [[buffer(3)]],
    constant uint&      capacity  [[buffer(4)]],
    constant uint&      start_pos [[buffer(5)]],
    constant uint&      bits      [[buffer(6)]],
    constant uint&      row_bytes [[buffer(7)]],
    constant uint&      rows      [[buffer(8)]],
    device float*       window    [[buffer(9)]],
    constant uint&      rw        [[buffer(10)]],
    constant uint&      seq_len   [[buffer(11)]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint tgid    [[threadgroup_position_in_grid]])
{
    uint row = tgid / seq_len;
    uint token = tgid - row * seq_len;
    if (row >= rows || token >= seq_len) return;

    uint cur_seq = start_pos + token;
    uint n_levels = 1u << bits;
    uint base_in = (row * seq_len + token) * head_dim;

    threadgroup float values[512];
    threadgroup uint idxs[512];
    threadgroup float red[32];
    threadgroup float tg_norm;

    for (uint d = tid; d < head_dim; d += tg_size) {
        values[d] = rotated[base_in + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_ss = 0.0f;
    for (uint d = tid; d < head_dim; d += tg_size) {
        local_ss += values[d] * values[d];
    }
    float ss = tq_tg_sum(local_ss, red, tid, tg_size, sg_id, sg_lane);
    if (tid == 0) tg_norm = sqrt(ss);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float norm = tg_norm;
    float invn = norm > 0.0f ? 1.0f / norm : 0.0f;

    if (rw > 0u) {
        uint slot = cur_seq % rw;
        device float* w = window + (row * rw + slot) * head_dim;
        for (uint d = tid; d < head_dim; d += tg_size) w[d] = values[d];
    }

    for (uint d = tid; d < head_dim; d += tg_size) {
        float u = values[d] * invn;
        uint best = 0;
        float bd = INFINITY;
        for (uint i = 0; i < n_levels; i++) {
            float dd = fabs(u - centroids[i]);
            if (dd < bd) {
                bd = dd;
                best = i;
            }
        }
        idxs[d] = best;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

// Single-token (decode) TQ attention — flash online softmax (Q4 parity).
// One TG per query head: rotate Q, tile over KV with online m/l, un-rotate O.
// `scores_all` is unused (kept for ABI); residual window still supported.
kernel void turboquant_attn_v3(
    device const float* Q_in      [[buffer(0)]],
    device const uchar* K_cache   [[buffer(1)]],
    device const uchar* V_cache   [[buffer(2)]],
    device float*       output    [[buffer(3)]],
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
    device const float* Kwin          [[buffer(15)]],
    device const float* Vwin          [[buffer(16)]],
    constant uint&      rw            [[buffer(17)]],
    constant uint&      window_lo     [[buffer(18)]],
    device const float* fwd           [[buffer(19)]],
    device const float* inv           [[buffer(20)]],
    constant uint&      v_bits        [[buffer(21)]],
    constant uint&      v_row_bytes   [[buffer(22)]],
    device const float* v_centroids   [[buffer(23)]],
    device float*       scores_all    [[buffer(24)]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint tgid    [[threadgroup_position_in_grid]])
{
    (void)scores_all;
    (void)num_kv_heads;
    uint h = tgid;
    if (h >= num_heads) return;

    constexpr uint SIMD_SIZE = 32;
    // Cap at 8 so h512 KV tiles stay under M1 Pro threadgroup memory (~32 KB).
    constexpr uint TILE_KV = 8;

    uint kv_h = h / num_kv_groups;
    uint q_off = h * head_dim;
    uint kv_base_k = kv_h * capacity * k_row_bytes;
    uint kv_base_v = kv_h * capacity * v_row_bytes;
    uint k_mask = (1u << k_bits) - 1u;
    uint v_mask = (1u << v_bits) - 1u;

    uint n_pos = (kv_seq > kv_start) ? (kv_seq - kv_start) : 0u;
    n_pos = min(n_pos, capacity);
    n_pos = min(n_pos, (uint)TQ_MAX_KV);

    threadgroup half cen_k[16];
    threadgroup half cen_v[16];
    threadgroup float qrot[512];
    threadgroup float orot[512];
    threadgroup float kv_values[TILE_KV * 512];
    threadgroup float shared_scores[TILE_KV];
    threadgroup float shared_exp[TILE_KV];
    threadgroup float shared_update[4];

    for (uint i = tid; i < (1u << k_bits); i += tg_size) cen_k[i] = half(k_centroids[i]);
    for (uint i = tid; i < (1u << v_bits); i += tg_size) cen_v[i] = half(v_centroids[i]);

    for (uint k = tid; k < head_dim; k += tg_size) orot[k] = Q_in[q_off + k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = tid; c < head_dim; c += tg_size) {
        float acc = 0.0f;
        uint mi = c;
        for (uint k = 0; k < head_dim; k++) { acc += orot[k] * fwd[mi]; mi += head_dim; }
        qrot[c] = acc;
    }
    if (tid == 0) {
        shared_update[0] = -INFINITY;
        shared_update[1] = 0.0f;
    }
    for (uint d = tid; d < head_dim; d += tg_size) orot[d] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simds = tg_size / SIMD_SIZE;
    uint lane = sg_lane;

    for (uint j0 = 0; j0 < n_pos; j0 += TILE_KV) {
        uint tile_count = min(TILE_KV, n_pos - j0);

        for (uint i = tid; i < tile_count * head_dim; i += tg_size) {
            uint kv_offset = i / head_dim;
            uint d = i - kv_offset * head_dim;
            uint pos = kv_start + j0 + kv_offset;
            if (rw > 0u && pos >= window_lo) {
                device const float* kw = Kwin + (kv_h * rw + (pos % rw)) * head_dim;
                kv_values[kv_offset * 512 + d] = kw[d];
            } else {
                uint row = kv_base_k + pos * k_row_bytes;
                float norm = float(*reinterpret_cast<device const half*>(&K_cache[row]));
                device const uchar* qs = K_cache + row + 2;
                kv_values[kv_offset * 512 + d] =
                    norm * float(cen_k[tq_unpack(qs, d, k_bits, k_mask)]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kv_offset = sg_id; kv_offset < tile_count; kv_offset += num_simds) {
            float partial = 0.0f;
            for (uint d = lane * 4; d + 3 < head_dim; d += SIMD_SIZE * 4) {
                float4 qv = float4(qrot[d], qrot[d + 1], qrot[d + 2], qrot[d + 3]);
                float4 kv = float4(
                    kv_values[kv_offset * 512 + d],
                    kv_values[kv_offset * 512 + d + 1],
                    kv_values[kv_offset * 512 + d + 2],
                    kv_values[kv_offset * 512 + d + 3]);
                partial += metal::dot(qv, kv);
            }
            partial = simd_sum(partial);
            if (lane == 0) shared_scores[kv_offset] = partial * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float m_old = shared_update[0];
            float l_old = shared_update[1];
            float m_new = m_old;
            for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                m_new = max(m_new, shared_scores[kv_offset]);
            }
            float tile_sum = 0.0f;
            for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                float e = exp(shared_scores[kv_offset] - m_new);
                shared_exp[kv_offset] = e;
                tile_sum += e;
            }
            float l_new = l_old * exp(m_old - m_new) + tile_sum;
            shared_update[0] = m_new;
            shared_update[1] = l_new;
            shared_update[2] =
                l_new > 0.0f ? (l_old * exp(m_old - m_new)) / l_new : 0.0f;
            shared_update[3] = l_new > 0.0f ? 1.0f / l_new : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < tile_count * head_dim; i += tg_size) {
            uint kv_offset = i / head_dim;
            uint d = i - kv_offset * head_dim;
            uint pos = kv_start + j0 + kv_offset;
            if (rw > 0u && pos >= window_lo) {
                device const float* vw = Vwin + (kv_h * rw + (pos % rw)) * head_dim;
                kv_values[kv_offset * 512 + d] = vw[d];
            } else {
                uint row = kv_base_v + pos * v_row_bytes;
                float norm = float(*reinterpret_cast<device const half*>(&V_cache[row]));
                device const uchar* qs = V_cache + row + 2;
                kv_values[kv_offset * 512 + d] =
                    norm * float(cen_v[tq_unpack(qs, d, v_bits, v_mask)]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float old_factor = shared_update[2];
        float inv_l_new = shared_update[3];
        for (uint d = tid; d < head_dim; d += tg_size) {
            float acc = orot[d] * old_factor;
            for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                acc += shared_exp[kv_offset] * inv_l_new * kv_values[kv_offset * 512 + d];
            }
            orot[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint c = tid; c < head_dim; c += tg_size) {
        float acc = 0.0f;
        uint mi = c;
        for (uint k = 0; k < head_dim; k++) { acc += orot[k] * inv[mi]; mi += head_dim; }
        output[q_off + c] = acc;
    }
}

// GQA decode (h256, up to 8 groups): one TG per KV head.
// E2B uses head_count_kv=1 → 8 groups; h512 stays on per-head flash (smem).
kernel void turboquant_attn_v3_gqa(
    device const float* Q_in      [[buffer(0)]],
    device const uchar* K_cache   [[buffer(1)]],
    device const uchar* V_cache   [[buffer(2)]],
    device float*       output    [[buffer(3)]],
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
    device const float* Kwin          [[buffer(15)]],
    device const float* Vwin          [[buffer(16)]],
    constant uint&      rw            [[buffer(17)]],
    constant uint&      window_lo     [[buffer(18)]],
    device const float* fwd           [[buffer(19)]],
    device const float* inv           [[buffer(20)]],
    constant uint&      v_bits        [[buffer(21)]],
    constant uint&      v_row_bytes   [[buffer(22)]],
    device const float* v_centroids   [[buffer(23)]],
    device float*       scores_all    [[buffer(24)]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint tgid    [[threadgroup_position_in_grid]])
{
    (void)scores_all;
    constexpr uint SIMD_SIZE = 32;
    constexpr uint HEAD_DIM = 256;
    constexpr uint TILE_KV = 8;
    constexpr uint MAX_G = 8;

    if (head_dim != HEAD_DIM) return;
    uint kv_h = tgid;
    if (kv_h >= num_kv_heads) return;
    if (num_kv_groups == 0u || num_kv_groups > MAX_G) return;
    uint num_simds_total = tg_size / SIMD_SIZE;
    if (num_simds_total % num_kv_groups != 0u) return;

    uint simds_per_head = num_simds_total / num_kv_groups;
    uint threads_per_head = simds_per_head * SIMD_SIZE;
    uint group = sg_id / simds_per_head;
    uint local_sgid = sg_id % simds_per_head;
    uint local_tid = local_sgid * SIMD_SIZE + sg_lane;
    uint h = kv_h * num_kv_groups + group;
    if (h >= num_heads) return;

    uint q_off = h * HEAD_DIM;
    uint kv_base_k = kv_h * capacity * k_row_bytes;
    uint kv_base_v = kv_h * capacity * v_row_bytes;
    uint k_mask = (1u << k_bits) - 1u;
    uint v_mask = (1u << v_bits) - 1u;

    uint n_pos = (kv_seq > kv_start) ? (kv_seq - kv_start) : 0u;
    n_pos = min(n_pos, capacity);
    n_pos = min(n_pos, (uint)TQ_MAX_KV);

    threadgroup half cen_k[16];
    threadgroup half cen_v[16];
    threadgroup float qrot[MAX_G * HEAD_DIM];
    threadgroup float orot[MAX_G * HEAD_DIM];
    threadgroup float kv_values[TILE_KV * HEAD_DIM];
    threadgroup float shared_scores[MAX_G * TILE_KV];
    threadgroup float shared_exp[MAX_G * TILE_KV];
    threadgroup float shared_update[MAX_G * 4];

    for (uint i = tid; i < (1u << k_bits); i += tg_size) cen_k[i] = half(k_centroids[i]);
    for (uint i = tid; i < (1u << v_bits); i += tg_size) cen_v[i] = half(v_centroids[i]);

    threadgroup float* q_head = qrot + group * HEAD_DIM;
    threadgroup float* o_head = orot + group * HEAD_DIM;
    threadgroup float* scores_head = shared_scores + group * TILE_KV;
    threadgroup float* exp_head = shared_exp + group * TILE_KV;
    threadgroup float* update_head = shared_update + group * 4;

    for (uint k = local_tid; k < HEAD_DIM; k += threads_per_head) {
        o_head[k] = Q_in[q_off + k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = local_tid; c < HEAD_DIM; c += threads_per_head) {
        float acc = 0.0f;
        uint mi = c;
        for (uint k = 0; k < HEAD_DIM; k++) { acc += o_head[k] * fwd[mi]; mi += HEAD_DIM; }
        q_head[c] = acc;
    }
    if (local_tid == 0) {
        update_head[0] = -INFINITY;
        update_head[1] = 0.0f;
    }
    for (uint d = local_tid; d < HEAD_DIM; d += threads_per_head) o_head[d] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j0 = 0; j0 < n_pos; j0 += TILE_KV) {
        uint tile_count = min(TILE_KV, n_pos - j0);

        for (uint i = tid; i < tile_count * HEAD_DIM; i += tg_size) {
            uint kv_offset = i / HEAD_DIM;
            uint d = i - kv_offset * HEAD_DIM;
            uint pos = kv_start + j0 + kv_offset;
            if (rw > 0u && pos >= window_lo) {
                device const float* kw = Kwin + (kv_h * rw + (pos % rw)) * HEAD_DIM;
                kv_values[kv_offset * HEAD_DIM + d] = kw[d];
            } else {
                uint row = kv_base_k + pos * k_row_bytes;
                float norm = float(*reinterpret_cast<device const half*>(&K_cache[row]));
                device const uchar* qs = K_cache + row + 2;
                kv_values[kv_offset * HEAD_DIM + d] =
                    norm * float(cen_k[tq_unpack(qs, d, k_bits, k_mask)]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kv_offset = local_sgid; kv_offset < tile_count; kv_offset += simds_per_head) {
            float partial = 0.0f;
            for (uint d = sg_lane * 4; d + 3 < HEAD_DIM; d += SIMD_SIZE * 4) {
                float4 qv = float4(q_head[d], q_head[d+1], q_head[d+2], q_head[d+3]);
                float4 kv = float4(
                    kv_values[kv_offset * HEAD_DIM + d],
                    kv_values[kv_offset * HEAD_DIM + d + 1],
                    kv_values[kv_offset * HEAD_DIM + d + 2],
                    kv_values[kv_offset * HEAD_DIM + d + 3]);
                partial += metal::dot(qv, kv);
            }
            partial = simd_sum(partial);
            if (sg_lane == 0) scores_head[kv_offset] = partial * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (local_tid == 0) {
            float m_old = update_head[0];
            float l_old = update_head[1];
            float m_new = m_old;
            for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                m_new = max(m_new, scores_head[kv_offset]);
            }
            float tile_sum = 0.0f;
            for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                float e = exp(scores_head[kv_offset] - m_new);
                exp_head[kv_offset] = e;
                tile_sum += e;
            }
            float l_new = l_old * exp(m_old - m_new) + tile_sum;
            update_head[0] = m_new;
            update_head[1] = l_new;
            update_head[2] =
                l_new > 0.0f ? (l_old * exp(m_old - m_new)) / l_new : 0.0f;
            update_head[3] = l_new > 0.0f ? 1.0f / l_new : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < tile_count * HEAD_DIM; i += tg_size) {
            uint kv_offset = i / HEAD_DIM;
            uint d = i - kv_offset * HEAD_DIM;
            uint pos = kv_start + j0 + kv_offset;
            if (rw > 0u && pos >= window_lo) {
                device const float* vw = Vwin + (kv_h * rw + (pos % rw)) * HEAD_DIM;
                kv_values[kv_offset * HEAD_DIM + d] = vw[d];
            } else {
                uint row = kv_base_v + pos * v_row_bytes;
                float norm = float(*reinterpret_cast<device const half*>(&V_cache[row]));
                device const uchar* qs = V_cache + row + 2;
                kv_values[kv_offset * HEAD_DIM + d] =
                    norm * float(cen_v[tq_unpack(qs, d, v_bits, v_mask)]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float old_factor = update_head[2];
        float inv_l_new = update_head[3];
        for (uint d = local_tid; d < HEAD_DIM; d += threads_per_head) {
            float acc = o_head[d] * old_factor;
            for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                acc += exp_head[kv_offset] * inv_l_new * kv_values[kv_offset * HEAD_DIM + d];
            }
            o_head[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint c = local_tid; c < HEAD_DIM; c += threads_per_head) {
        float acc = 0.0f;
        uint mi = c;
        for (uint k = 0; k < HEAD_DIM; k++) { acc += o_head[k] * inv[mi]; mi += HEAD_DIM; }
        output[q_off + c] = acc;
    }
}

// Multi-query causal TQ attention for past-hot prefill chunks.
// Layout: Q/out are SHD [q_len, num_heads, head_dim].
// One TG handles four adjacent queries and dequantizes each K/V tile once for
// all four. This is the important distinction from dispatch-batched decode:
// packed KV traffic and centroid lookups are shared across query rows.
template <uint HEAD_DIM, uint Q_TILE, uint TILE_KV>
void turboquant_attn_v3_causal_impl(
    device const float* Q_in      [[buffer(0)]],
    device const uchar* K_cache   [[buffer(1)]],
    device const uchar* V_cache   [[buffer(2)]],
    device float*       output    [[buffer(3)]],
    constant uint&      num_heads     [[buffer(4)]],
    constant uint&      num_kv_heads  [[buffer(5)]],
    constant uint&      num_kv_groups [[buffer(6)]],
    constant uint&      head_dim      [[buffer(7)]],
    constant uint&      kv_seq        [[buffer(8)]],  // absolute end pos after chunk
    constant uint&      capacity      [[buffer(9)]],
    constant float&     scale         [[buffer(10)]],
    constant uint&      q_len         [[buffer(11)]],
    constant uint&      q_start       [[buffer(12)]],
    constant uint&      attention_window [[buffer(13)]], // 0 = full
    constant uint&      k_bits        [[buffer(14)]],
    constant uint&      k_row_bytes   [[buffer(15)]],
    device const float* k_centroids   [[buffer(16)]],
    constant uint&      v_bits        [[buffer(17)]],
    constant uint&      v_row_bytes   [[buffer(18)]],
    device const float* v_centroids   [[buffer(19)]],
    device const float* fwd           [[buffer(20)]], // unused: Q is pre-rotated
    device const float* inv           [[buffer(21)]], // unused: output stays rotated
    threadgroup half* cen_k,
    threadgroup half* cen_v,
    threadgroup float* orot,
    threadgroup float* kv_values,
    threadgroup float* shared_scores,
    threadgroup float* shared_exp,
    threadgroup float* shared_update,
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tgid    [[threadgroup_position_in_grid]])
{
    constexpr uint SIMD_SIZE = 32;

    uint q_tiles = (q_len + Q_TILE - 1u) / Q_TILE;
    uint h = tgid / q_tiles;
    uint q_base = (tgid % q_tiles) * Q_TILE;
    uint q_count = min(Q_TILE, q_len - q_base);
    if (h >= num_heads || q_base >= q_len) return;

    uint kv_h = h / num_kv_groups;
    uint kv_base_k = kv_h * capacity * k_row_bytes;
    uint kv_base_v = kv_h * capacity * v_row_bytes;
    uint k_mask = (1u << k_bits) - 1u;
    uint v_mask = (1u << v_bits) - 1u;

    uint attend_len[Q_TILE];
    uint attend_start[Q_TILE];
    uint tile_start = capacity;
    uint tile_end_all = 0u;
    for (uint q = 0; q < q_count; q++) {
        uint qi = q_base + q;
        uint end = min(q_start + qi + 1u, kv_seq);
        end = min(end, capacity);
        end = min(end, (uint)TQ_MAX_KV);
        uint start = 0u;
        if (attention_window > 0u && end > attention_window) {
            start = end - attention_window;
        }
        attend_len[q] = end;
        attend_start[q] = start;
        tile_start = min(tile_start, start);
        tile_end_all = max(tile_end_all, end);
    }

    for (uint i = tid; i < (1u << k_bits); i += tg_size) cen_k[i] = half(k_centroids[i]);
    for (uint i = tid; i < (1u << v_bits); i += tg_size) cen_v[i] = half(v_centroids[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint simd_id = tid / SIMD_SIZE;
    uint lane = tid % SIMD_SIZE;
    uint num_simds = tg_size / SIMD_SIZE;

    if (tid < q_count) {
        shared_update[tid * 4 + 0] = -INFINITY;
        shared_update[tid * 4 + 1] = 0.0f;
    }
    for (uint q = 0; q < q_count; q++) {
        uint q_tg = q * HEAD_DIM;
        for (uint d = tid; d < head_dim; d += tg_size) orot[q_tg + d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv_tile = tile_start; kv_tile < tile_end_all; kv_tile += TILE_KV) {
        uint tile_end = min(kv_tile + TILE_KV, tile_end_all);
        uint tile_count = tile_end - kv_tile;

        // Dequantize each K coordinate once, then reuse it for all query rows.
        for (uint i = tid; i < tile_count * head_dim; i += tg_size) {
            uint kv_offset = i / head_dim;
            uint d = i - kv_offset * head_dim;
            uint pos = kv_tile + kv_offset;
            uint row = kv_base_k + pos * k_row_bytes;
            float norm = float(*reinterpret_cast<device const half*>(&K_cache[row]));
            device const uchar* qs = K_cache + row + 2;
            kv_values[kv_offset * HEAD_DIM + d] =
                norm * float(cen_k[tq_unpack(qs, d, k_bits, k_mask)]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Assign independent (query, KV) score pairs to SIMD groups. Previously
        // all 8 groups cooperated on one dot, leaving 6 groups idle for h256.
        uint score_pairs = q_count * tile_count;
        for (uint pair = simd_id; pair < score_pairs; pair += num_simds) {
            uint q = pair / tile_count;
            uint kv_offset = pair - q * tile_count;
            uint q_off = ((q_base + q) * num_heads + h) * head_dim;
            uint pos = kv_tile + kv_offset;
            float partial = 0.0f;
            for (uint d = lane * 4; d + 3 < head_dim; d += SIMD_SIZE * 4) {
                float4 qv = float4(
                    Q_in[q_off + d], Q_in[q_off + d + 1],
                    Q_in[q_off + d + 2], Q_in[q_off + d + 3]);
                float4 kv = float4(
                    kv_values[kv_offset * HEAD_DIM + d],
                    kv_values[kv_offset * HEAD_DIM + d + 1],
                    kv_values[kv_offset * HEAD_DIM + d + 2],
                    kv_values[kv_offset * HEAD_DIM + d + 3]);
                partial += metal::dot(qv, kv);
            }
            partial = simd_sum(partial);
            if (lane == 0) {
                shared_scores[q * TILE_KV + kv_offset] =
                    (pos >= attend_start[q] && pos < attend_len[q])
                        ? partial * scale
                        : -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < q_count) {
            uint q = tid;
            uint ui = q * 4;
            float m_old = shared_update[ui + 0];
            float l_old = shared_update[ui + 1];
            float m_new = m_old;
            float tile_scores[TILE_KV];
            for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                float s = shared_scores[q * TILE_KV + kv_offset];
                tile_scores[kv_offset] = s;
                m_new = max(m_new, s);
            }
            float tile_sum = 0.0f;
            for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                float e = exp(tile_scores[kv_offset] - m_new);
                shared_exp[q * TILE_KV + kv_offset] = e;
                tile_sum += e;
            }
            float l_new = l_old * exp(m_old - m_new) + tile_sum;
            shared_update[ui + 0] = m_new;
            shared_update[ui + 1] = l_new;
            shared_update[ui + 2] =
                l_new > 0.0f ? (l_old * exp(m_old - m_new)) / l_new : 0.0f;
            shared_update[ui + 3] = l_new > 0.0f ? 1.0f / l_new : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Dequantize V once for this tile and reuse it across all queries.
        for (uint i = tid; i < tile_count * head_dim; i += tg_size) {
            uint kv_offset = i / head_dim;
            uint d = i - kv_offset * head_dim;
            uint pos = kv_tile + kv_offset;
            uint row = kv_base_v + pos * v_row_bytes;
            float norm = float(*reinterpret_cast<device const half*>(&V_cache[row]));
            device const uchar* qs = V_cache + row + 2;
            kv_values[kv_offset * HEAD_DIM + d] =
                norm * float(cen_v[tq_unpack(qs, d, v_bits, v_mask)]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint q = 0; q < q_count; q++) {
            uint q_tg = q * HEAD_DIM;
            float old_factor = shared_update[q * 4 + 2];
            float inv_l_new = shared_update[q * 4 + 3];
            for (uint d = tid; d < head_dim; d += tg_size) {
                float acc = orot[q_tg + d] * old_factor;
                for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                    acc += shared_exp[q * TILE_KV + kv_offset] * inv_l_new
                        * kv_values[kv_offset * HEAD_DIM + d];
                }
                orot[q_tg + d] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Leave outputs in the Haar frame; one batched inverse matmul follows.
    for (uint q = 0; q < q_count; q++) {
        uint q_tg = q * HEAD_DIM;
        uint out_off = ((q_base + q) * num_heads + h) * head_dim;
        for (uint c = tid; c < head_dim; c += tg_size) {
            output[out_off + c] = orot[q_tg + c];
        }
    }
}

#define TQ_CAUSAL_ENTRY(NAME, HD, QT, KT) \
kernel void NAME( \
    device const float* Q_in [[buffer(0)]], \
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
    constant uint& k_bits [[buffer(14)]], \
    constant uint& k_row_bytes [[buffer(15)]], \
    device const float* k_centroids [[buffer(16)]], \
    constant uint& v_bits [[buffer(17)]], \
    constant uint& v_row_bytes [[buffer(18)]], \
    device const float* v_centroids [[buffer(19)]], \
    device const float* fwd [[buffer(20)]], \
    device const float* inv [[buffer(21)]], \
    uint tid [[thread_index_in_threadgroup]], \
    uint tg_size [[threads_per_threadgroup]], \
    uint tgid [[threadgroup_position_in_grid]]) { \
    threadgroup half cen_k[16]; \
    threadgroup half cen_v[16]; \
    threadgroup float orot[QT * HD]; \
    threadgroup float kv_values[KT * HD]; \
    threadgroup float shared_scores[QT * KT]; \
    threadgroup float shared_exp[QT * KT]; \
    threadgroup float shared_update[QT * 4]; \
    turboquant_attn_v3_causal_impl<HD, QT, KT>( \
        Q_in, K_cache, V_cache, output, num_heads, num_kv_heads, \
        num_kv_groups, head_dim, kv_seq, capacity, scale, q_len, q_start, \
        attention_window, k_bits, k_row_bytes, k_centroids, v_bits, \
        v_row_bytes, v_centroids, fwd, inv, cen_k, cen_v, orot, kv_values, \
        shared_scores, shared_exp, shared_update, tid, tg_size, tgid); \
}

// h256: larger KV tiles (8) amortize dequant. h512 stays at 4 for smem.
TQ_CAUSAL_ENTRY(turboquant_attn_v3_causal_h256, 256, 4, 8)
TQ_CAUSAL_ENTRY(turboquant_attn_v3_causal_h512, 512, 4, 4)

// Prefill GQA causal (h256): one TG per (KV head, query tile).
// All query heads that share the KV head reuse one K/V dequant per tile.
kernel void turboquant_attn_v3_causal_gqa_h256(
    device const float* Q_in [[buffer(0)]],
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
    constant uint& k_bits [[buffer(14)]],
    constant uint& k_row_bytes [[buffer(15)]],
    device const float* k_centroids [[buffer(16)]],
    constant uint& v_bits [[buffer(17)]],
    constant uint& v_row_bytes [[buffer(18)]],
    device const float* v_centroids [[buffer(19)]],
    device const float* fwd [[buffer(20)]],
    device const float* inv [[buffer(21)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    (void)fwd; (void)inv;
    constexpr uint SIMD_SIZE = 32;
    constexpr uint HEAD_DIM = 256;
    constexpr uint Q_TILE = 2;
    constexpr uint TILE_KV = 8;
    constexpr uint MAX_G = 8;

    if (head_dim != HEAD_DIM) return;
    if (num_kv_groups == 0u || num_kv_groups > MAX_G) return;

    uint q_tiles = (q_len + Q_TILE - 1u) / Q_TILE;
    uint kv_h = tgid / q_tiles;
    uint q_base = (tgid % q_tiles) * Q_TILE;
    uint q_count = min(Q_TILE, q_len - q_base);
    if (kv_h >= num_kv_heads || q_base >= q_len) return;

    uint kv_base_k = kv_h * capacity * k_row_bytes;
    uint kv_base_v = kv_h * capacity * v_row_bytes;
    uint k_mask = (1u << k_bits) - 1u;
    uint v_mask = (1u << v_bits) - 1u;

    uint attend_len[Q_TILE];
    uint attend_start[Q_TILE];
    uint tile_start = capacity;
    uint tile_end_all = 0u;
    for (uint q = 0; q < q_count; q++) {
        uint qi = q_base + q;
        uint end = min(q_start + qi + 1u, kv_seq);
        end = min(end, capacity);
        end = min(end, (uint)TQ_MAX_KV);
        uint start = 0u;
        if (attention_window > 0u && end > attention_window) {
            start = end - attention_window;
        }
        attend_len[q] = end;
        attend_start[q] = start;
        tile_start = min(tile_start, start);
        tile_end_all = max(tile_end_all, end);
    }

    threadgroup half cen_k[16];
    threadgroup half cen_v[16];
    threadgroup float orot[MAX_G * Q_TILE * HEAD_DIM];
    threadgroup float kv_values[TILE_KV * HEAD_DIM];
    threadgroup float shared_scores[MAX_G * Q_TILE * TILE_KV];
    threadgroup float shared_exp[MAX_G * Q_TILE * TILE_KV];
    threadgroup float shared_update[MAX_G * Q_TILE * 4];

    for (uint i = tid; i < (1u << k_bits); i += tg_size) cen_k[i] = half(k_centroids[i]);
    for (uint i = tid; i < (1u << v_bits); i += tg_size) cen_v[i] = half(v_centroids[i]);

    uint simd_id = tid / SIMD_SIZE;
    uint lane = tid % SIMD_SIZE;
    uint num_simds = tg_size / SIMD_SIZE;
    uint n_q = num_kv_groups * q_count;

    for (uint gq = tid; gq < n_q; gq += tg_size) {
        shared_update[gq * 4 + 0] = -INFINITY;
        shared_update[gq * 4 + 1] = 0.0f;
    }
    for (uint i = tid; i < n_q * HEAD_DIM; i += tg_size) orot[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv_tile = tile_start; kv_tile < tile_end_all; kv_tile += TILE_KV) {
        uint tile_end = min(kv_tile + TILE_KV, tile_end_all);
        uint tile_count = tile_end - kv_tile;

        for (uint i = tid; i < tile_count * HEAD_DIM; i += tg_size) {
            uint kv_offset = i / HEAD_DIM;
            uint d = i - kv_offset * HEAD_DIM;
            uint pos = kv_tile + kv_offset;
            uint row = kv_base_k + pos * k_row_bytes;
            float norm = float(*reinterpret_cast<device const half*>(&K_cache[row]));
            device const uchar* qs = K_cache + row + 2;
            kv_values[kv_offset * HEAD_DIM + d] =
                norm * float(cen_k[tq_unpack(qs, d, k_bits, k_mask)]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint score_pairs = n_q * tile_count;
        for (uint pair = simd_id; pair < score_pairs; pair += num_simds) {
            uint gq = pair / tile_count;
            uint kv_offset = pair - gq * tile_count;
            uint g = gq / q_count;
            uint q = gq - g * q_count;
            uint h = kv_h * num_kv_groups + g;
            uint q_off = ((q_base + q) * num_heads + h) * HEAD_DIM;
            uint pos = kv_tile + kv_offset;
            float partial = 0.0f;
            for (uint d = lane * 4; d + 3 < HEAD_DIM; d += SIMD_SIZE * 4) {
                float4 qv = float4(
                    Q_in[q_off + d], Q_in[q_off + d + 1],
                    Q_in[q_off + d + 2], Q_in[q_off + d + 3]);
                float4 kv = float4(
                    kv_values[kv_offset * HEAD_DIM + d],
                    kv_values[kv_offset * HEAD_DIM + d + 1],
                    kv_values[kv_offset * HEAD_DIM + d + 2],
                    kv_values[kv_offset * HEAD_DIM + d + 3]);
                partial += metal::dot(qv, kv);
            }
            partial = simd_sum(partial);
            if (lane == 0) {
                shared_scores[gq * TILE_KV + kv_offset] =
                    (h < num_heads && pos >= attend_start[q] && pos < attend_len[q])
                        ? partial * scale
                        : -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint gq = tid; gq < n_q; gq += tg_size) {
            uint ui = gq * 4;
            float m_old = shared_update[ui + 0];
            float l_old = shared_update[ui + 1];
            float m_new = m_old;
            float tile_scores[TILE_KV];
            for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                float s = shared_scores[gq * TILE_KV + kv_offset];
                tile_scores[kv_offset] = s;
                m_new = max(m_new, s);
            }
            float tile_sum = 0.0f;
            for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                float e = exp(tile_scores[kv_offset] - m_new);
                shared_exp[gq * TILE_KV + kv_offset] = e;
                tile_sum += e;
            }
            float l_new = l_old * exp(m_old - m_new) + tile_sum;
            shared_update[ui + 0] = m_new;
            shared_update[ui + 1] = l_new;
            shared_update[ui + 2] =
                l_new > 0.0f ? (l_old * exp(m_old - m_new)) / l_new : 0.0f;
            shared_update[ui + 3] = l_new > 0.0f ? 1.0f / l_new : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < tile_count * HEAD_DIM; i += tg_size) {
            uint kv_offset = i / HEAD_DIM;
            uint d = i - kv_offset * HEAD_DIM;
            uint pos = kv_tile + kv_offset;
            uint row = kv_base_v + pos * v_row_bytes;
            float norm = float(*reinterpret_cast<device const half*>(&V_cache[row]));
            device const uchar* qs = V_cache + row + 2;
            kv_values[kv_offset * HEAD_DIM + d] =
                norm * float(cen_v[tq_unpack(qs, d, v_bits, v_mask)]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint gq = 0; gq < n_q; gq++) {
            uint q_tg = gq * HEAD_DIM;
            float old_factor = shared_update[gq * 4 + 2];
            float inv_l_new = shared_update[gq * 4 + 3];
            for (uint d = tid; d < HEAD_DIM; d += tg_size) {
                float acc = orot[q_tg + d] * old_factor;
                for (uint kv_offset = 0; kv_offset < tile_count; kv_offset++) {
                    acc += shared_exp[gq * TILE_KV + kv_offset] * inv_l_new
                        * kv_values[kv_offset * HEAD_DIM + d];
                }
                orot[q_tg + d] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint gq = 0; gq < n_q; gq++) {
        uint g = gq / q_count;
        uint q = gq - g * q_count;
        uint h = kv_h * num_kv_groups + g;
        if (h >= num_heads) continue;
        uint q_tg = gq * HEAD_DIM;
        uint out_off = ((q_base + q) * num_heads + h) * HEAD_DIM;
        for (uint c = tid; c < HEAD_DIM; c += tg_size) {
            output[out_off + c] = orot[q_tg + c];
        }
    }
}
