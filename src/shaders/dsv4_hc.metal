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
    // One threadgroup / one token row. gid = row index.
    const int row = (int)gid;
    const int mix_len = 2 * n_hc + n_hc * n_hc;
    const device float *m = mix + (ulong)row * mix_len;
    const device float *b = base; // shared base
    device float *o = out + (ulong)row * mix_len;

    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

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
