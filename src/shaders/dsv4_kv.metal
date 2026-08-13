// DeepSeek-V4 Flash — KV helpers (SWA store/shift + compress + indexer scores).

#include <metal_stdlib>
using namespace metal;

// Copy one latent row into cache[row * head_dim].
kernel void dsv4_kv_store_row(
    device const float *src [[buffer(0)]],
    device float *cache [[buffer(1)]],
    constant int &head_dim [[buffer(2)]],
    constant int &row [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int d = (int)gid;
    if (d >= head_dim) return;
    cache[(ulong)row * head_dim + d] = src[d];
}

// Shift SWA ring left by one row (drop oldest); capacity = swa rows.
kernel void dsv4_kv_shift_left(
    device float *cache [[buffer(0)]],
    constant int &head_dim [[buffer(1)]],
    constant int &swa [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    int d = (int)gid;
    if (d >= head_dim) return;
    for (int i = 1; i < swa; ++i) {
        cache[(ulong)(i - 1) * head_dim + d] = cache[(ulong)i * head_dim + d];
    }
}

kernel void dsv4_compress_mean_pool(
    device const float *k_raw [[buffer(0)]],
    device const float *v_raw [[buffer(1)]],
    device float *k_out [[buffer(2)]],
    device float *v_out [[buffer(3)]],
    constant int &head_dim [[buffer(4)]],
    constant int &ratio [[buffer(5)]],
    constant int &start [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int d = (int)gid;
    if (d >= head_dim) return;
    float ks = 0.0f;
    float vs = 0.0f;
    for (int t = 0; t < ratio; ++t) {
        int off = (start + t) * head_dim + d;
        ks += k_raw[off];
        vs += v_raw[off];
    }
    float inv = 1.0f / (float)ratio;
    k_out[d] = ks * inv;
    v_out[d] = vs * inv;
}

kernel void dsv4_indexer_scores(
    device const float *q_idx [[buffer(0)]],
    device const float *k_idx [[buffer(1)]],
    device float *scores [[buffer(2)]],
    constant int &n_heads [[buffer(3)]],
    constant int &head_dim [[buffer(4)]],
    constant int &n_rows [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int row = (int)gid;
    if (row >= n_rows) return;
    float acc = 0.0f;
    // Average score across indexer heads for selection.
    for (int h = 0; h < n_heads; ++h) {
        device const float *qh = q_idx + (ulong)h * head_dim;
        device const float *kh = k_idx + ((ulong)row * n_heads + h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * kh[d];
        acc += dot;
    }
    scores[row] = acc / (float)n_heads;
}

// CSA: 1 simdgroup per compressed row (32-lane Q·K, ReLU-sum across heads).
kernel void dsv4_select_comp_score(
    device const float *q [[buffer(0)]],
    device const float *k_comp [[buffer(1)]],
    device float *scores [[buffer(2)]],
    constant int &n_head [[buffer(3)]],
    constant int &head_dim [[buffer(4)]],
    constant int &n_comp [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    int ci = (int)tgpig.x;
    if (ci >= n_comp) return;
    const uint lane = (uint)tiisg;
    device const float *kt = k_comp + (ulong)ci * head_dim;
    float s = 0.0f;
    for (int h = 0; h < n_head; ++h) {
        device const float *qh = q + (ulong)h * head_dim;
        float qk = 0.0f;
        for (int d = (int)lane; d < head_dim; d += 32) qk += qh[d] * kt[d];
        qk = simd_sum(qk);
        s += max(qk, 0.0f);
    }
    if (lane == 0) scores[ci] = s;
}

// Top-k from `scores` → sorted ascending indices. Single-thread (k ≤ 512, n ≤ 2048).
kernel void dsv4_select_comp_topk(
    device const float *scores [[buffer(0)]],
    device int *out_idx [[buffer(1)]],
    constant int &n_comp [[buffer(2)]],
    constant int &top_k [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    if (n_comp <= 0) return;
    int k = min(top_k, n_comp);
    if (k >= n_comp) {
        for (int t = 0; t < n_comp; ++t) out_idx[t] = t;
        return;
    }
    constexpr int MAX_N = 2048;
    int n = min(n_comp, MAX_N);
    thread int alive[MAX_N];
    for (int i = 0; i < n; ++i) alive[i] = 1;
    thread int picked[512];
    for (int t = 0; t < k; ++t) {
        int best = -1;
        float best_s = -1.0e30f;
        for (int i = 0; i < n; ++i) {
            if (!alive[i]) continue;
            if (scores[i] > best_s) {
                best_s = scores[i];
                best = i;
            }
        }
        alive[best] = 0;
        picked[t] = best;
    }
    for (int i = 0; i < k; ++i) {
        for (int j = i + 1; j < k; ++j) {
            if (picked[j] < picked[i]) {
                int tmp = picked[i];
                picked[i] = picked[j];
                picked[j] = tmp;
            }
        }
    }
    for (int t = 0; t < k; ++t) out_idx[t] = picked[t];
}

kernel void dsv4_iota_i32(
    device int *out [[buffer(0)]],
    constant int &n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    int i = (int)gid;
    if (i >= n) return;
    out[i] = i;
}

// --- Compressor stages (parallel; matches CPU `compressor_step_from_proj`) ---
// ape is F16 [ratio, width] row-major (pos_mod * width + j).

// Stage 1: write kv_cur / sc_cur+ape into state row. gid over width.
kernel void dsv4_compressor_store_row(
    device const float *kv_cur [[buffer(0)]],
    device const float *sc_cur [[buffer(1)]],
    device const half *ape [[buffer(2)]],
    device float *state_kv [[buffer(3)]],
    device float *state_score [[buffer(4)]],
    constant int &ratio [[buffer(5)]],
    constant int &width [[buffer(6)]],
    constant int &pos [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    int j = (int)gid;
    if (j >= width) return;
    int pos_mod = pos % ratio;
    int row = (ratio == 4) ? (ratio + pos_mod) : pos_mod;
    state_kv[(ulong)row * width + j] = kv_cur[j];
    state_score[(ulong)row * width + j] = sc_cur[j] + (float)ape[(ulong)pos_mod * width + j];
}

// Stage 2: pool one head_dim channel into out_row[j]. gid over head_dim.
kernel void dsv4_compressor_pool(
    device const float *state_kv [[buffer(0)]],
    device const float *state_score [[buffer(1)]],
    device float *out_row [[buffer(2)]],
    constant int &ratio [[buffer(3)]],
    constant int &width [[buffer(4)]],
    constant int &head_dim [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    constexpr float NEG_INF = -1.0e30f;
    int j = (int)gid;
    if (j >= head_dim) return;
    float max_score = NEG_INF;
    if (ratio == 4) {
        for (int r = 0; r < ratio; ++r) {
            float sp = state_score[r * width + j];
            float sc = state_score[(ratio + r) * width + head_dim + j];
            max_score = max(max_score, max(sp, sc));
        }
    } else {
        for (int r = 0; r < ratio; ++r) {
            max_score = max(max_score, state_score[r * width + j]);
        }
    }
    if (max_score <= NEG_INF * 0.5f) {
        out_row[j] = 0.0f;
        return;
    }
    float denom = 0.0f;
    float sum = 0.0f;
    if (ratio == 4) {
        for (int r = 0; r < ratio; ++r) {
            float wp = exp(state_score[r * width + j] - max_score);
            float wc = exp(state_score[(ratio + r) * width + head_dim + j] - max_score);
            denom += wp + wc;
            sum += wp * state_kv[r * width + j];
            sum += wc * state_kv[(ratio + r) * width + head_dim + j];
        }
    } else {
        for (int r = 0; r < ratio; ++r) {
            float w = exp(state_score[r * width + j] - max_score);
            denom += w;
            sum += w * state_kv[r * width + j];
        }
    }
    out_row[j] = (denom > 0.0f) ? (sum / denom) : 0.0f;
}

// Stage 3: RMS + rope + fp8 on out_row (one thread; uses existing helpers).
kernel void dsv4_compressor_rms_rope_fp8(
    device float *out_row [[buffer(0)]],
    device const float *norm [[buffer(1)]],
    constant int &head_dim [[buffer(2)]],
    constant int &n_rot [[buffer(3)]],
    constant int &pos [[buffer(4)]],
    constant int &ratio [[buffer(5)]],
    constant float &rms_eps [[buffer(6)]],
    constant int &use_compress_rope [[buffer(7)]],
    constant float &rope_freq [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    float ss = 0.0f;
    for (int j = 0; j < head_dim; ++j) ss += out_row[j] * out_row[j];
    float scale = rsqrt(ss / (float)head_dim + rms_eps);
    for (int j = 0; j < head_dim; ++j) {
        out_row[j] = out_row[j] * scale * norm[j];
    }
    int comp_pos = pos + 1 - ratio;
    if (use_compress_rope != 0) {
        float freq_scale = 1.0f / 16.0f;
        float attn_factor = 1.0f / (1.0f + 0.1f * log(1.0f / freq_scale));
        dsv4_rope_tail_ext_one(
            out_row, head_dim, n_rot, comp_pos, 160000.0f, freq_scale, 1.0f, attn_factor,
            32.0f, 1.0f, 65536u, 0);
    } else {
        dsv4_rope_tail_ext_one(
            out_row, head_dim, n_rot, comp_pos, rope_freq, 1.0f, 0.0f, 1.0f,
            32.0f, 1.0f, 65536u, 0);
    }
    dsv4_fp8_e4m3fn_nope_inplace(out_row, head_dim, n_rot);
}

// Stage 4: CSA copy_within shuffle. pass=0: second→first; pass=1: first→second.
// gid over width.
kernel void dsv4_compressor_csa_shuffle(
    device float *state_kv [[buffer(0)]],
    device float *state_score [[buffer(1)]],
    constant int &ratio [[buffer(2)]],
    constant int &width [[buffer(3)]],
    constant int &pass [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int j = (int)gid;
    if (j >= width) return;
    if (ratio != 4) return;
    if (pass == 0) {
        for (int r = 0; r < ratio; ++r) {
            int src = (ratio + r) * width + j;
            int dst = r * width + j;
            state_kv[dst] = state_kv[src];
            state_score[dst] = state_score[src];
        }
    } else {
        for (int r = 0; r < ratio; ++r) {
            int src = r * width + j;
            int dst = (ratio + r) * width + j;
            state_kv[dst] = state_kv[src];
            state_score[dst] = state_score[src];
        }
    }
}
