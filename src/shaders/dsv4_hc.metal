// DeepSeek-V4 Flash — mHC kernels (our implementation).
// Math: Sinkhorn split of HC control vector + weighted sum / expand.

#include <metal_stdlib>
using namespace metal;

inline float dsv4_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// mix/base: [n_hc + n_hc + n_hc*n_hc] = pre | post | comb
// scale: [3] = pre_scale, post_scale, comb_scale
// out: same layout as mix
//
// Flash uses n_hc=4: specialize the 4×4 Sinkhorn with float4 row/col
// normalizes and early-exit when both margins are within eps.
kernel void dsv4_hc_split_sinkhorn(
    device const float *mix [[buffer(0)]],
    device const float *scale [[buffer(1)]],
    device const float *base [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant int &n_hc [[buffer(4)]],
    constant int &iters [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    // One thread / one token row. gid = row index.
    const int row = (int)gid;
    const int mix_len = 2 * n_hc + n_hc * n_hc;
    const device float *m = mix + (ulong)row * mix_len;
    const device float *b = base; // shared base
    device float *o = out + (ulong)row * mix_len;

    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    if (n_hc == 4) {
        float4 pre_z = float4(m[0], m[1], m[2], m[3]) * pre_scale
            + float4(b[0], b[1], b[2], b[3]);
        float4 post_z = float4(m[4], m[5], m[6], m[7]) * post_scale
            + float4(b[4], b[5], b[6], b[7]);
        float4 pre = 1.0f / (1.0f + exp(-pre_z)) + eps;
        float4 post = 2.0f / (1.0f + exp(-post_z));
        o[0] = pre.x; o[1] = pre.y; o[2] = pre.z; o[3] = pre.w;
        o[4] = post.x; o[5] = post.y; o[6] = post.z; o[7] = post.w;

        // c rows = dest, cols = src (layout c[src + dst*4])
        float4 r0, r1, r2, r3;
        {
            float4 v = float4(m[8], m[9], m[10], m[11]) * comb_scale
                + float4(b[8], b[9], b[10], b[11]);
            float rm = max(max(v.x, v.y), max(v.z, v.w));
            r0 = exp(v - rm);
            r0 = r0 / (r0.x + r0.y + r0.z + r0.w) + eps;
        }
        {
            float4 v = float4(m[12], m[13], m[14], m[15]) * comb_scale
                + float4(b[12], b[13], b[14], b[15]);
            float rm = max(max(v.x, v.y), max(v.z, v.w));
            r1 = exp(v - rm);
            r1 = r1 / (r1.x + r1.y + r1.z + r1.w) + eps;
        }
        {
            float4 v = float4(m[16], m[17], m[18], m[19]) * comb_scale
                + float4(b[16], b[17], b[18], b[19]);
            float rm = max(max(v.x, v.y), max(v.z, v.w));
            r2 = exp(v - rm);
            r2 = r2 / (r2.x + r2.y + r2.z + r2.w) + eps;
        }
        {
            float4 v = float4(m[20], m[21], m[22], m[23]) * comb_scale
                + float4(b[20], b[21], b[22], b[23]);
            float rm = max(max(v.x, v.y), max(v.z, v.w));
            r3 = exp(v - rm);
            r3 = r3 / (r3.x + r3.y + r3.z + r3.w) + eps;
        }
        // First col normalize (iter 0 col pass)
        {
            float4 col = float4(r0.x, r1.x, r2.x, r3.x);
            float inv = 1.0f / (col.x + col.y + col.z + col.w + eps);
            r0.x *= inv; r1.x *= inv; r2.x *= inv; r3.x *= inv;
            col = float4(r0.y, r1.y, r2.y, r3.y);
            inv = 1.0f / (col.x + col.y + col.z + col.w + eps);
            r0.y *= inv; r1.y *= inv; r2.y *= inv; r3.y *= inv;
            col = float4(r0.z, r1.z, r2.z, r3.z);
            inv = 1.0f / (col.x + col.y + col.z + col.w + eps);
            r0.z *= inv; r1.z *= inv; r2.z *= inv; r3.z *= inv;
            col = float4(r0.w, r1.w, r2.w, r3.w);
            inv = 1.0f / (col.x + col.y + col.z + col.w + eps);
            r0.w *= inv; r1.w *= inv; r2.w *= inv; r3.w *= inv;
        }
        const int n_iter = max(iters, 1);
        for (int iter = 1; iter < n_iter; ++iter) {
            // row normalize
            r0 = r0 / (r0.x + r0.y + r0.z + r0.w + eps);
            r1 = r1 / (r1.x + r1.y + r1.z + r1.w + eps);
            r2 = r2 / (r2.x + r2.y + r2.z + r2.w + eps);
            r3 = r3 / (r3.x + r3.y + r3.z + r3.w + eps);
            // col normalize
            float4 c0 = float4(r0.x, r1.x, r2.x, r3.x);
            float4 c1 = float4(r0.y, r1.y, r2.y, r3.y);
            float4 c2 = float4(r0.z, r1.z, r2.z, r3.z);
            float4 c3 = float4(r0.w, r1.w, r2.w, r3.w);
            float inv0 = 1.0f / (c0.x + c0.y + c0.z + c0.w + eps);
            float inv1 = 1.0f / (c1.x + c1.y + c1.z + c1.w + eps);
            float inv2 = 1.0f / (c2.x + c2.y + c2.z + c2.w + eps);
            float inv3 = 1.0f / (c3.x + c3.y + c3.z + c3.w + eps);
            r0.x *= inv0; r1.x *= inv0; r2.x *= inv0; r3.x *= inv0;
            r0.y *= inv1; r1.y *= inv1; r2.y *= inv1; r3.y *= inv1;
            r0.z *= inv2; r1.z *= inv2; r2.z *= inv2; r3.z *= inv2;
            r0.w *= inv3; r1.w *= inv3; r2.w *= inv3; r3.w *= inv3;
            // Early exit when row+col margins ≈ 1
            float4 rs = float4(
                r0.x + r0.y + r0.z + r0.w,
                r1.x + r1.y + r1.z + r1.w,
                r2.x + r2.y + r2.z + r2.w,
                r3.x + r3.y + r3.z + r3.w);
            float4 cs = float4(
                r0.x + r1.x + r2.x + r3.x,
                r0.y + r1.y + r2.y + r3.y,
                r0.z + r1.z + r2.z + r3.z,
                r0.w + r1.w + r2.w + r3.w);
            float4 err = max(fabs(rs - 1.0f), fabs(cs - 1.0f));
            if (err.x < eps && err.y < eps && err.z < eps && err.w < eps) break;
        }
        o[8] = r0.x; o[9] = r0.y; o[10] = r0.z; o[11] = r0.w;
        o[12] = r1.x; o[13] = r1.y; o[14] = r1.z; o[15] = r1.w;
        o[16] = r2.x; o[17] = r2.y; o[18] = r2.z; o[19] = r2.w;
        o[20] = r3.x; o[21] = r3.y; o[22] = r3.z; o[23] = r3.w;
        return;
    }

    for (int i = 0; i < n_hc; ++i) {
        float z = m[i] * pre_scale + b[i];
        o[i] = dsv4_sigmoid(z) + eps;
    }
    for (int i = 0; i < n_hc; ++i) {
        int off = n_hc + i;
        float z = m[off] * post_scale + b[off];
        o[off] = 2.0f * dsv4_sigmoid(z);
    }

    thread float c[16 * 16];
    for (int dst = 0; dst < n_hc; ++dst) {
        float row_max = -1.0e30f;
        for (int src = 0; src < n_hc; ++src) {
            int idx = src + dst * n_hc;
            int off = 2 * n_hc + idx;
            float v = m[off] * comb_scale + b[off];
            c[idx] = v;
            row_max = max(row_max, v);
        }
        float row_sum = 0.0f;
        for (int src = 0; src < n_hc; ++src) {
            int idx = src + dst * n_hc;
            float v = exp(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }
        float inv = 1.0f / row_sum;
        for (int src = 0; src < n_hc; ++src) {
            int idx = src + dst * n_hc;
            c[idx] = c[idx] * inv + eps;
        }
    }
    for (int src = 0; src < n_hc; ++src) {
        float sum = 0.0f;
        for (int dst = 0; dst < n_hc; ++dst) sum += c[src + dst * n_hc];
        float inv = 1.0f / (sum + eps);
        for (int dst = 0; dst < n_hc; ++dst) c[src + dst * n_hc] *= inv;
    }
    for (int iter = 1; iter < iters; ++iter) {
        for (int dst = 0; dst < n_hc; ++dst) {
            float sum = 0.0f;
            for (int src = 0; src < n_hc; ++src) sum += c[src + dst * n_hc];
            float inv = 1.0f / (sum + eps);
            for (int src = 0; src < n_hc; ++src) c[src + dst * n_hc] *= inv;
        }
        for (int src = 0; src < n_hc; ++src) {
            float sum = 0.0f;
            for (int dst = 0; dst < n_hc; ++dst) sum += c[src + dst * n_hc];
            float inv = 1.0f / (sum + eps);
            for (int dst = 0; dst < n_hc; ++dst) c[src + dst * n_hc] *= inv;
        }
    }
    for (int i = 0; i < n_hc * n_hc; ++i) o[2 * n_hc + i] = c[i];
}

kernel void dsv4_hc_weighted_sum(
    device const float *x_hc [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant int &n_embd [[buffer(3)]],
    constant int &n_hc [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int d = (int)gid;
    if (d >= n_embd) return;
    float acc = 0.0f;
    for (int h = 0; h < n_hc; ++h) {
        acc += x_hc[(ulong)h * n_embd + d] * weights[h];
    }
    out[d] = acc;
}

kernel void dsv4_hc_expand_post(
    device const float *block [[buffer(0)]],
    device const float *add_hc [[buffer(1)]],
    device const float *post [[buffer(2)]],
    device const float *comb [[buffer(3)]],
    device float *hc [[buffer(4)]],
    constant int &n_embd [[buffer(5)]],
    constant int &n_hc [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int d = (int)gid;
    if (d >= n_embd) return;
    for (int h = 0; h < n_hc; ++h) {
        float v = post[h] * block[d];
        for (int s = 0; s < n_hc; ++s) {
            v += comb[h + s * n_hc] * add_hc[(ulong)s * n_embd + d];
        }
        hc[(ulong)h * n_embd + d] = v;
    }
}

kernel void dsv4_rms_norm(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant int &n [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    constant int &has_weight [[buffer(5)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]])
{
    threadgroup float partial[256];
    float local = 0.0f;
    for (uint i = lid; i < (uint)n; i += tg) {
        float v = x[i];
        local += v * v;
    }
    partial[lid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg / 2; stride > 0; stride >>= 1) {
        if (lid < stride) partial[lid] += partial[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = rsqrt(partial[0] / (float)n + eps);
    for (uint i = lid; i < (uint)n; i += tg) {
        float v = x[i] * scale;
        if (has_weight != 0) v *= weight[i];
        out[i] = v;
    }
}
