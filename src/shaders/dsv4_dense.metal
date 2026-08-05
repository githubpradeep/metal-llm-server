// DeepSeek-V4 Flash — dense F16 / Q8_0 / F32 matvec (clean-room).

#include <metal_stdlib>
using namespace metal;

// One simdgroup (32 threads) per output row.
kernel void dsv4_matvec_f16(
    device const half *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant int &n_out [[buffer(3)]],
    constant int &n_in [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if ((int)tgid >= n_out) return;
    device const half *row = W + (ulong)tgid * (ulong)n_in;
    float acc = 0.0f;
    for (int k = (int)tid; k < n_in; k += 32) {
        acc += (float)row[k] * x[k];
    }
    acc = simd_sum(acc);
    if (tid == 0) y[tgid] = acc;
}

kernel void dsv4_matvec_f32(
    device const float *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant int &n_out [[buffer(3)]],
    constant int &n_in [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if ((int)tgid >= n_out) return;
    device const float *row = W + (ulong)tgid * (ulong)n_in;
    float acc = 0.0f;
    for (int k = (int)tid; k < n_in; k += 32) {
        acc += row[k] * x[k];
    }
    acc = simd_sum(acc);
    if (tid == 0) y[tgid] = acc;
}

struct block_q8_0 {
    half d;
    char qs[32];
};

kernel void dsv4_matvec_q8_0(
    device const uchar *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant int &n_out [[buffer(3)]],
    constant int &n_in [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if ((int)tgid >= n_out) return;
    int nb = n_in / 32;
    int row_bytes = nb * 34;
    device const uchar *wrow = W + (ulong)tgid * (ulong)row_bytes;
    float acc = 0.0f;
    for (int b = (int)tid; b < nb; b += 32) {
        device const block_q8_0 *blk = (device const block_q8_0 *)(wrow + b * 34);
        float d = (float)blk->d;
        device const float *xb = x + b * 32;
        float local = 0.0f;
        for (int i = 0; i < 32; ++i) {
            local += (float)blk->qs[i] * xb[i];
        }
        acc += local * d;
    }
    acc = simd_sum(acc);
    if (tid == 0) y[tgid] = acc;
}

// y[i] += scale * x[i]
kernel void dsv4_axpy(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant float &scale [[buffer(2)]],
    constant int &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int i = (int)gid;
    if (i >= n) return;
    y[i] += scale * x[i];
}

// y = a + b
kernel void dsv4_add(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant int &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int i = (int)gid;
    if (i >= n) return;
    y[i] = a[i] + b[i];
}

// Zero a float buffer.
kernel void dsv4_zero(
    device float *y [[buffer(0)]],
    constant int &n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    int i = (int)gid;
    if (i >= n) return;
    y[i] = 0.0f;
}
