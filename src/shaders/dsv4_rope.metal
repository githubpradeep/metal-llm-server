// DeepSeek-V4 Flash — tail RoPE (last n_rot dims of head_dim).

#include <metal_stdlib>
using namespace metal;

kernel void dsv4_rope_tail(
    device float *x [[buffer(0)]],
    constant int &n_heads [[buffer(1)]],
    constant int &head_dim [[buffer(2)]],
    constant int &n_rot [[buffer(3)]],
    constant int &pos [[buffer(4)]],
    constant float &freq_base [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int h = (int)gid;
    if (h >= n_heads) return;
    device float *xh = x + (ulong)h * head_dim;
    int rope_off = head_dim - n_rot;
    int pairs = n_rot / 2;
    for (int i = 0; i < pairs; ++i) {
        float freq = pow(freq_base, -((float)i) / (float)pairs);
        float angle = (float)pos * freq;
        float c = cos(angle);
        float s = sin(angle);
        int i0 = rope_off + i;
        int i1 = rope_off + i + pairs;
        // Flash uses interleaved pairs in the rotary tail; match even/odd within tail.
        i0 = rope_off + 2 * i;
        i1 = rope_off + 2 * i + 1;
        float x0 = xh[i0];
        float x1 = xh[i1];
        xh[i0] = x0 * c - x1 * s;
        xh[i1] = x0 * s + x1 * c;
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
