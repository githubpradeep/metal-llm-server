// DeepSeek-V4 Flash — attention helpers (SWA + mixed + sinks).
// 1 simdgroup (32 threads) per head; two-pass softmax; sink = denom-only.
// Scores live in threadgroup memory (not thread arrays — avoids spill).
// Flash head_dim=512: each lane owns 16 strided dims.

#include <metal_stdlib>
using namespace metal;

kernel void dsv4_attn_swa_mqa(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device const float *sinks [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant int &n_head [[buffer(5)]],
    constant int &head_dim [[buffer(6)]],
    constant int &n_kv [[buffer(7)]],
    constant int &has_sinks [[buffer(8)]],
    constant float &scale [[buffer(9)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    threadgroup float *tg [[threadgroup(0)]])
{
    const int h = (int)tgpig.x;
    if (h >= n_head) return;
    const uint lane = (uint)tiisg;
    const uint hd = (uint)head_dim;
    device const float *qh = q + (ulong)h * hd;

    float q0 = qh[lane];
    float q1 = qh[lane + 32];
    float q2 = qh[lane + 64];
    float q3 = qh[lane + 96];
    float q4 = qh[lane + 128];
    float q5 = qh[lane + 160];
    float q6 = qh[lane + 192];
    float q7 = qh[lane + 224];
    float q8 = qh[lane + 256];
    float q9 = qh[lane + 288];
    float q10 = qh[lane + 320];
    float q11 = qh[lane + 352];
    float q12 = qh[lane + 384];
    float q13 = qh[lane + 416];
    float q14 = qh[lane + 448];
    float q15 = qh[lane + 480];

    const float sink = (has_sinks != 0) ? sinks[h] : -1.0e30f;
    float max_s = sink;
    const int nkv = metal::min(n_kv, 128);

    for (int t = 0; t < nkv; ++t) {
        device const float *kt = k + (ulong)t * hd;
        float qk = q0 * kt[lane] + q1 * kt[lane + 32] + q2 * kt[lane + 64]
            + q3 * kt[lane + 96] + q4 * kt[lane + 128] + q5 * kt[lane + 160]
            + q6 * kt[lane + 192] + q7 * kt[lane + 224] + q8 * kt[lane + 256]
            + q9 * kt[lane + 288] + q10 * kt[lane + 320] + q11 * kt[lane + 352]
            + q12 * kt[lane + 384] + q13 * kt[lane + 416] + q14 * kt[lane + 448]
            + q15 * kt[lane + 480];
        qk = simd_sum(qk);
        const float score = qk * scale;
        if (lane == 0) tg[t] = score;
        max_s = metal::max(max_s, score);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float denom = exp(sink - max_s);
        for (int t = 0; t < nkv; ++t) {
            tg[t] = exp(tg[t] - max_s);
            denom += tg[t];
        }
        tg[nkv] = 1.0f / denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = tg[nkv];

    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    float a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
    float a8 = 0.f, a9 = 0.f, a10 = 0.f, a11 = 0.f;
    float a12 = 0.f, a13 = 0.f, a14 = 0.f, a15 = 0.f;
    for (int t = 0; t < nkv; ++t) {
        const float w = tg[t] * inv;
        device const float *vt = v + (ulong)t * hd;
        a0 += w * vt[lane];
        a1 += w * vt[lane + 32];
        a2 += w * vt[lane + 64];
        a3 += w * vt[lane + 96];
        a4 += w * vt[lane + 128];
        a5 += w * vt[lane + 160];
        a6 += w * vt[lane + 192];
        a7 += w * vt[lane + 224];
        a8 += w * vt[lane + 256];
        a9 += w * vt[lane + 288];
        a10 += w * vt[lane + 320];
        a11 += w * vt[lane + 352];
        a12 += w * vt[lane + 384];
        a13 += w * vt[lane + 416];
        a14 += w * vt[lane + 448];
        a15 += w * vt[lane + 480];
    }
    device float *oh = out + (ulong)h * hd;
    oh[lane] = a0;
    oh[lane + 32] = a1;
    oh[lane + 64] = a2;
    oh[lane + 96] = a3;
    oh[lane + 128] = a4;
    oh[lane + 160] = a5;
    oh[lane + 192] = a6;
    oh[lane + 224] = a7;
    oh[lane + 256] = a8;
    oh[lane + 288] = a9;
    oh[lane + 320] = a10;
    oh[lane + 352] = a11;
    oh[lane + 384] = a12;
    oh[lane + 416] = a13;
    oh[lane + 448] = a14;
    oh[lane + 480] = a15;
}

kernel void dsv4_attn_mixed_mqa(
    device const float *q [[buffer(0)]],
    device const float *k_raw [[buffer(1)]],
    device const float *v_raw [[buffer(2)]],
    device const float *k_comp [[buffer(3)]],
    device const float *v_comp [[buffer(4)]],
    device const int *comp_idx [[buffer(5)]],
    device const float *sinks [[buffer(6)]],
    device float *out [[buffer(7)]],
    constant int &n_head [[buffer(8)]],
    constant int &head_dim [[buffer(9)]],
    constant int &n_raw [[buffer(10)]],
    constant int &n_comp_sel [[buffer(11)]],
    constant int &has_sinks [[buffer(12)]],
    constant float &scale [[buffer(13)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    threadgroup float *tg [[threadgroup(0)]])
{
    const int h = (int)tgpig.x;
    if (h >= n_head) return;
    const uint lane = (uint)tiisg;
    const uint hd = (uint)head_dim;
    device const float *qh = q + (ulong)h * hd;

    float q0 = qh[lane];
    float q1 = qh[lane + 32];
    float q2 = qh[lane + 64];
    float q3 = qh[lane + 96];
    float q4 = qh[lane + 128];
    float q5 = qh[lane + 160];
    float q6 = qh[lane + 192];
    float q7 = qh[lane + 224];
    float q8 = qh[lane + 256];
    float q9 = qh[lane + 288];
    float q10 = qh[lane + 320];
    float q11 = qh[lane + 352];
    float q12 = qh[lane + 384];
    float q13 = qh[lane + 416];
    float q14 = qh[lane + 448];
    float q15 = qh[lane + 480];

    const float sink = (has_sinks != 0) ? sinks[h] : -1.0e30f;
    float max_s = sink;
    const int n_total = metal::min(n_raw + n_comp_sel, 640);

    for (int t = 0; t < n_total; ++t) {
        device const float *kt = (t < n_raw)
            ? (k_raw + (ulong)t * hd)
            : (k_comp + (ulong)comp_idx[t - n_raw] * hd);
        float qk = q0 * kt[lane] + q1 * kt[lane + 32] + q2 * kt[lane + 64]
            + q3 * kt[lane + 96] + q4 * kt[lane + 128] + q5 * kt[lane + 160]
            + q6 * kt[lane + 192] + q7 * kt[lane + 224] + q8 * kt[lane + 256]
            + q9 * kt[lane + 288] + q10 * kt[lane + 320] + q11 * kt[lane + 352]
            + q12 * kt[lane + 384] + q13 * kt[lane + 416] + q14 * kt[lane + 448]
            + q15 * kt[lane + 480];
        qk = simd_sum(qk);
        const float score = qk * scale;
        if (lane == 0) tg[t] = score;
        max_s = metal::max(max_s, score);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float denom = exp(sink - max_s);
        for (int t = 0; t < n_total; ++t) {
            tg[t] = exp(tg[t] - max_s);
            denom += tg[t];
        }
        tg[n_total] = 1.0f / denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = tg[n_total];

    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    float a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
    float a8 = 0.f, a9 = 0.f, a10 = 0.f, a11 = 0.f;
    float a12 = 0.f, a13 = 0.f, a14 = 0.f, a15 = 0.f;
    for (int t = 0; t < n_total; ++t) {
        const float w = tg[t] * inv;
        device const float *vt = (t < n_raw)
            ? (v_raw + (ulong)t * hd)
            : (v_comp + (ulong)comp_idx[t - n_raw] * hd);
        a0 += w * vt[lane];
        a1 += w * vt[lane + 32];
        a2 += w * vt[lane + 64];
        a3 += w * vt[lane + 96];
        a4 += w * vt[lane + 128];
        a5 += w * vt[lane + 160];
        a6 += w * vt[lane + 192];
        a7 += w * vt[lane + 224];
        a8 += w * vt[lane + 256];
        a9 += w * vt[lane + 288];
        a10 += w * vt[lane + 320];
        a11 += w * vt[lane + 352];
        a12 += w * vt[lane + 384];
        a13 += w * vt[lane + 416];
        a14 += w * vt[lane + 448];
        a15 += w * vt[lane + 480];
    }
    device float *oh = out + (ulong)h * hd;
    oh[lane] = a0;
    oh[lane + 32] = a1;
    oh[lane + 64] = a2;
    oh[lane + 96] = a3;
    oh[lane + 128] = a4;
    oh[lane + 160] = a5;
    oh[lane + 192] = a6;
    oh[lane + 224] = a7;
    oh[lane + 256] = a8;
    oh[lane + 288] = a9;
    oh[lane + 320] = a10;
    oh[lane + 352] = a11;
    oh[lane + 384] = a12;
    oh[lane + 416] = a13;
    oh[lane + 448] = a14;
    oh[lane + 480] = a15;
}
