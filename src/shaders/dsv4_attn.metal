// DeepSeek-V4 Flash — attention helpers (SWA + mixed + sinks).

#include <metal_stdlib>
using namespace metal;

// Softmax attention over concatenated raw SWA rows (+ optional sinks as extra keys).
// Q: [n_head, head_dim], K/V: [n_kv, head_dim] (MQA broadcasts K/V across heads).
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
    uint gid [[thread_position_in_grid]])
{
    int h = (int)gid;
    if (h >= n_head) return;
    device const float *qh = q + (ulong)h * head_dim;
    // scores over n_kv (+1 sink)
    int n_scores = n_kv + (has_sinks ? 1 : 0);
    thread float scores[512];
    float max_s = -1.0e30f;
    for (int t = 0; t < n_kv; ++t) {
        device const float *kt = k + (ulong)t * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * kt[d];
        scores[t] = dot * scale;
        max_s = max(max_s, scores[t]);
    }
    if (has_sinks != 0) {
        scores[n_kv] = sinks[h];
        max_s = max(max_s, scores[n_kv]);
    }
    float sum = 0.0f;
    for (int t = 0; t < n_scores; ++t) {
        scores[t] = exp(scores[t] - max_s);
        sum += scores[t];
    }
    float inv = 1.0f / sum;
    device float *oh = out + (ulong)h * head_dim;
    for (int d = 0; d < head_dim; ++d) oh[d] = 0.0f;
    for (int t = 0; t < n_kv; ++t) {
        float w = scores[t] * inv;
        device const float *vt = v + (ulong)t * head_dim;
        for (int d = 0; d < head_dim; ++d) oh[d] += w * vt[d];
    }
    // sink contributes 0 value (learned bias on logits only)
}

// Mixed attention: raw KV then selected compressed rows (indices in idx buffer).
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
    uint gid [[thread_position_in_grid]])
{
    int h = (int)gid;
    if (h >= n_head) return;
    device const float *qh = q + (ulong)h * head_dim;
    int n_scores = n_raw + n_comp_sel + (has_sinks ? 1 : 0);
    // Cap thread-local scores; Flash indexer top-k <= 512 + swa 128.
    thread float scores[768];
    float max_s = -1.0e30f;
    int t = 0;
    for (int i = 0; i < n_raw; ++i, ++t) {
        device const float *kt = k_raw + (ulong)i * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * kt[d];
        scores[t] = dot * scale;
        max_s = max(max_s, scores[t]);
    }
    for (int i = 0; i < n_comp_sel; ++i, ++t) {
        int ci = comp_idx[i];
        device const float *kt = k_comp + (ulong)ci * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * kt[d];
        scores[t] = dot * scale;
        max_s = max(max_s, scores[t]);
    }
    if (has_sinks != 0) {
        scores[t] = sinks[h];
        max_s = max(max_s, scores[t]);
        ++t;
    }
    float sum = 0.0f;
    for (int i = 0; i < n_scores; ++i) {
        scores[i] = exp(scores[i] - max_s);
        sum += scores[i];
    }
    float inv = 1.0f / sum;
    device float *oh = out + (ulong)h * head_dim;
    for (int d = 0; d < head_dim; ++d) oh[d] = 0.0f;
    t = 0;
    for (int i = 0; i < n_raw; ++i, ++t) {
        float w = scores[t] * inv;
        device const float *vt = v_raw + (ulong)i * head_dim;
        for (int d = 0; d < head_dim; ++d) oh[d] += w * vt[d];
    }
    for (int i = 0; i < n_comp_sel; ++i, ++t) {
        float w = scores[t] * inv;
        int ci = comp_idx[i];
        device const float *vt = v_comp + (ulong)ci * head_dim;
        for (int d = 0; d < head_dim; ++d) oh[d] += w * vt[d];
    }
}
