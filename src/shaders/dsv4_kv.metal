// DeepSeek-V4 Flash — KV helpers (compress mean-pool placeholder + indexer scores).

#include <metal_stdlib>
using namespace metal;

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
