// DeepSeek-V4 Flash — tail RoPE (last n_rot dims of head_dim).
// Matches CPU `rope_tail_inplace` for SWA (freq_scale=1, no YaRN).

#include <metal_stdlib>
using namespace metal;

static inline float dsv4_yarn_corr_dim(int n_dims, uint n_ctx_orig, float n_rot, float base) {
    return (float)n_dims * log((float)n_ctx_orig / (n_rot * 2.0f * M_PI_F)) / (2.0f * log(base));
}

static inline void dsv4_yarn_corr_dims(
    int n_dims, uint n_ctx_orig, float freq_base, float beta_fast, float beta_slow,
    thread float *corr)
{
    float start = floor(dsv4_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end = ceil(dsv4_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    corr[0] = max(0.0f, start);
    corr[1] = min((float)(n_dims - 1), end);
}

static inline float dsv4_yarn_ramp(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / max(high - low, 0.001f);
    return 1.0f - clamp(y, 0.0f, 1.0f);
}

// Matches CPU `rope_tail_ext_inplace` (NoPE prefix untouched).
static inline void dsv4_rope_tail_ext_one(
    device float *xh,
    int head_dim,
    int n_rot,
    int pos,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow,
    uint n_ctx_orig,
    int inverse)
{
    int n_nope = head_dim - n_rot;
    float theta_scale = pow(freq_base, -2.0f / (float)n_rot);
    float sin_sign = (inverse != 0) ? -1.0f : 1.0f;
    float corr[2] = {0.0f, 0.0f};
    if (ext_factor != 0.0f) {
        dsv4_yarn_corr_dims(n_rot, n_ctx_orig, freq_base, beta_fast, beta_slow, corr);
    }
    float theta_extrap = (float)pos;
    for (int i = 0; i < n_rot; i += 2) {
        float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float ramp_mix = dsv4_yarn_ramp(corr[0], corr[1], i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
        }
        float c = cos(theta) * mscale;
        float s = sin_sign * sin(theta) * mscale;
        float x0 = xh[n_nope + i];
        float x1 = xh[n_nope + i + 1];
        xh[n_nope + i] = x0 * c - x1 * s;
        xh[n_nope + i + 1] = x0 * s + x1 * c;
        theta_extrap *= theta_scale;
    }
}

// 32-lane RoPE: lane p applies pair p (theta via the same multiply chain as serial).
static inline void dsv4_rope_tail_ext_lanes(
    device float *xh,
    int head_dim,
    int n_rot,
    int pos,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow,
    uint n_ctx_orig,
    int inverse,
    uint lane)
{
    int n_nope = head_dim - n_rot;
    int n_pairs = n_rot / 2;
    float theta_scale = pow(freq_base, -2.0f / (float)n_rot);
    float sin_sign = (inverse != 0) ? -1.0f : 1.0f;
    float corr[2] = {0.0f, 0.0f};
    if (ext_factor != 0.0f) {
        dsv4_yarn_corr_dims(n_rot, n_ctx_orig, freq_base, beta_fast, beta_slow, corr);
    }
    for (int p = (int)lane; p < n_pairs; p += 32) {
        float theta_extrap = (float)pos;
        for (int j = 0; j < p; ++j) theta_extrap *= theta_scale;
        float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float ramp_mix = dsv4_yarn_ramp(corr[0], corr[1], p * 2) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
        }
        float c = cos(theta) * mscale;
        float s = sin_sign * sin(theta) * mscale;
        int i = p * 2;
        float x0 = xh[n_nope + i];
        float x1 = xh[n_nope + i + 1];
        xh[n_nope + i] = x0 * c - x1 * s;
        xh[n_nope + i + 1] = x0 * s + x1 * c;
    }
}

kernel void dsv4_rope_tail(
    device float *x [[buffer(0)]],
    constant int &n_heads [[buffer(1)]],
    constant int &head_dim [[buffer(2)]],
    constant int &n_rot [[buffer(3)]],
    constant int &pos [[buffer(4)]],
    constant float &freq_base [[buffer(5)]],
    constant int &inverse [[buffer(6)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    int h = (int)tgpig.x;
    if (h >= n_heads) return;
    device float *xh = x + (ulong)h * head_dim;
    dsv4_rope_tail_ext_lanes(
        xh, head_dim, n_rot, pos, freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 65536u,
        inverse, (uint)tiisg);
}

// Full YaRN / compress RoPE (Flash compress: base 160000, scale 1/16, yarn on).
kernel void dsv4_rope_tail_ext(
    device float *x [[buffer(0)]],
    constant int &n_heads [[buffer(1)]],
    constant int &head_dim [[buffer(2)]],
    constant int &n_rot [[buffer(3)]],
    constant int &pos [[buffer(4)]],
    constant float &freq_base [[buffer(5)]],
    constant float &freq_scale [[buffer(6)]],
    constant float &ext_factor [[buffer(7)]],
    constant float &attn_factor [[buffer(8)]],
    constant float &beta_fast [[buffer(9)]],
    constant float &beta_slow [[buffer(10)]],
    constant uint &n_ctx_orig [[buffer(11)]],
    constant int &inverse [[buffer(12)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    int h = (int)tgpig.x;
    if (h >= n_heads) return;
    device float *xh = x + (ulong)h * head_dim;
    dsv4_rope_tail_ext_lanes(
        xh, head_dim, n_rot, pos, freq_base, freq_scale, ext_factor, attn_factor,
        beta_fast, beta_slow, n_ctx_orig, inverse, (uint)tiisg);
}

// Per-row RMS (Q heads): out[h,:] = rms(x[h,:]) [* weight].
kernel void dsv4_rms_norm_rows(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant int &n_rows [[buffer(3)]],
    constant int &row_dim [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    constant int &has_weight [[buffer(6)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    int h = (int)tgpig.x;
    if (h >= n_rows) return;
    const uint lane = (uint)tiisg;
    device const float *xh = x + (ulong)h * row_dim;
    device float *oh = out + (ulong)h * row_dim;
    float ss = 0.0f;
    for (int d = (int)lane; d < row_dim; d += 32) {
        float v = xh[d];
        ss += v * v;
    }
    ss = simd_sum(ss);
    float scale = rsqrt(ss / (float)row_dim + eps);
    for (int d = (int)lane; d < row_dim; d += 32) {
        float v = xh[d] * scale;
        if (has_weight != 0) v *= weight[d];
        oh[d] = v;
    }
}

// E4M3FN NoPE round-trip on first (head_dim - n_rot) dims, groups of 64.
// Matches CPU `fp8_e4m3fn_nope_inplace`.
static inline float dsv4_e4m3fn_value(int i) {
    constexpr float EXP_SCALE[16] = {
        0.0f, 0.015625f, 0.03125f, 0.0625f, 0.125f, 0.25f, 0.5f, 1.0f,
        2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f, 256.0f};
    int exp = (i >> 3) & 0x0f;
    int mant = i & 0x07;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * EXP_SCALE[exp];
}

static inline float dsv4_e4m3fn_round_trip(float x) {
    float sign = (x < 0.0f) ? -1.0f : 1.0f;
    float ax = min(fabs(x), 448.0f);
    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        float best_diff = fabs(ax - dsv4_e4m3fn_value(best));
        float next_diff = fabs(ax - dsv4_e4m3fn_value(best + 1));
        if (next_diff < best_diff
            || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {
            best += 1;
        }
    }
    return sign * dsv4_e4m3fn_value(best);
}

static inline void dsv4_fp8_e4m3fn_nope_inplace(device float *kv, int head_dim, int n_rot) {
    int n_nope = head_dim - n_rot;
    for (int off = 0; off < n_nope; off += 64) {
        int end = min(off + 64, n_nope);
        float amax = 0.0f;
        for (int i = off; i < end; ++i) amax = max(amax, fabs(kv[i]));
        if (amax < 1e-4f) amax = 1e-4f;
        float scale = pow(2.0f, ceil(log2(amax / 448.0f)));
        for (int i = off; i < end; ++i) {
            float v = clamp(kv[i] / scale, -448.0f, 448.0f);
            kv[i] = dsv4_e4m3fn_round_trip(v) * scale;
        }
    }
}

// One simdgroup per 64-wide NoPE group (Flash: 7 groups × 32 lanes).
kernel void dsv4_fp8_e4m3fn_nope(
    device float *kv [[buffer(0)]],
    constant int &head_dim [[buffer(1)]],
    constant int &n_rot [[buffer(2)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    int n_nope = head_dim - n_rot;
    int off = (int)tgpig.x * 64;
    if (off >= n_nope) return;
    int end = min(off + 64, n_nope);
    const uint lane = (uint)tiisg;
    int i0 = off + (int)lane;
    int i1 = off + (int)lane + 32;
    float v0 = (i0 < end) ? kv[i0] : 0.0f;
    float v1 = (i1 < end) ? kv[i1] : 0.0f;
    float amax = max(fabs(v0), fabs(v1));
    amax = simd_max(amax);
    if (amax < 1e-4f) amax = 1e-4f;
    float scale = pow(2.0f, ceil(log2(amax / 448.0f)));
    if (i0 < end) {
        float v = clamp(v0 / scale, -448.0f, 448.0f);
        kv[i0] = dsv4_e4m3fn_round_trip(v) * scale;
    }
    if (i1 < end) {
        float v = clamp(v1 / scale, -448.0f, 448.0f);
        kv[i1] = dsv4_e4m3fn_round_trip(v) * scale;
    }
}

kernel void dsv4_fp8_store(
    device const float *src [[buffer(0)]],
    device uchar *dst [[buffer(1)]],
    constant int &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    int i = (int)gid;
    if (i >= n) return;
    float x = clamp(src[i], -57344.0f, 57344.0f);
    if (x == 0.0f) { dst[i] = 0; return; }
    uint bits = as_type<uint>(x);
    uchar sign = (uchar)((bits >> 31) << 7);
    int exp = (int)((bits >> 23) & 0xff);
    uint mant = bits & 0x7fffffu;
    int e = exp - 127 + 15;
    if (e <= 0) { dst[i] = sign; return; }
    if (e >= 31) { dst[i] = sign | 0x7c; return; }
    uchar m = (uchar)(mant >> 21);
    dst[i] = sign | (uchar)(e << 2) | m;
}
