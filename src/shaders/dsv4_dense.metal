// DeepSeek-V4 Flash — dense F16 / Q8_0 / F32 matvec (clean-room).
// Geometry matches ds4: F16 NR0=2 + NSG=min(8,ceil(K/128)); Q8 NR0=2 + NSG=4.

#include <metal_stdlib>
using namespace metal;

constant int DSV4_DENSE_NR0 = 2;
constant int DSV4_NW = 32;

static inline void dsv4_dense_reduce_write(
    device float *y,
    thread float *sumf,
    int r0,
    int n_out,
    ushort tiisg,
    ushort sgitg,
    threadgroup float *shmem)
{
    threadgroup float *sh[DSV4_DENSE_NR0];
    for (int row = 0; row < DSV4_DENSE_NR0; ++row) {
        sh[row] = shmem + DSV4_NW * row;
        if (sgitg == 0) {
            sh[row][tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int row = 0; row < DSV4_DENSE_NR0; ++row) {
        if (tiisg == 0) {
            sh[row][sgitg] = sumf[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int row = 0; row < DSV4_DENSE_NR0 && r0 + row < n_out; ++row) {
        const float tot = simd_sum(sh[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            y[r0 + row] = tot;
        }
    }
}

kernel void dsv4_matvec_f16(
    device const half *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant int &n_out [[buffer(3)]],
    constant int &n_in [[buffer(4)]],
    constant int &nsg [[buffer(5)]],
    threadgroup float *shmem [[threadgroup(0)]],
    uint tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int r0 = (int)tgpig * DSV4_DENSE_NR0;
    if (r0 >= n_out) return;
    const int nsg_use = max(nsg, 1);

    constexpr int NB = 32;
    constexpr int NF = 16;
    constexpr int NF4 = NF / 4;
    const int nb = n_in / NB;
    float sumf[DSV4_DENSE_NR0] = {0.f, 0.f};

    const short ix = (short)(tiisg / (DSV4_NW / NF));
    const short il = (short)(tiisg % (DSV4_NW / NF));
    const int ib0 = (int)sgitg * NF + ix;

    device const float4 *y4 = (device const float4 *)(x + ib0 * NB + il * NF);
    float4 yl4[NF4];

    for (int ib = ib0; ib < nb; ib += nsg_use * NF) {
        for (short i = 0; i < NF4; ++i) {
            yl4[i] = y4[i];
        }
        for (int row = 0; row < DSV4_DENSE_NR0; ++row) {
            if (r0 + row >= n_out) break;
            device const half4 *xb4 =
                (device const half4 *)(W + (ulong)(r0 + row) * (ulong)n_in + ib * NB + il * NF);
            float sumq = 0.f;
            for (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), yl4[i]);
            }
            sumf[row] += sumq;
        }
        y4 += (nsg_use * NF * DSV4_NW) / 4;
    }
    for (int i = nb * NB + (int)sgitg * DSV4_NW + (int)tiisg; i < n_in; i += DSV4_NW * nsg_use) {
        for (int row = 0; row < DSV4_DENSE_NR0; ++row) {
            if (r0 + row >= n_out) break;
            sumf[row] += (float)W[(ulong)(r0 + row) * (ulong)n_in + i] * x[i];
        }
    }
    dsv4_dense_reduce_write(y, sumf, r0, n_out, tiisg, sgitg, shmem);
}

// Grouped LoRA-O A: y[g*rank+r] = W[g][r] · x[g*group_dim : (g+1)*group_dim]
// One dispatch for all groups (tgpig.y = group). F16 weights, contiguous groups.
kernel void dsv4_matvec_f16_lora_groups(
    device const half *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant int &n_groups [[buffer(3)]],
    constant int &rank [[buffer(4)]],
    constant int &group_dim [[buffer(5)]],
    constant int &nsg [[buffer(6)]],
    threadgroup float *shmem [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int g = (int)tgpig.y;
    if (g >= n_groups) return;
    const int r0 = (int)tgpig.x * DSV4_DENSE_NR0;
    if (r0 >= rank) return;
    const int nsg_use = max(nsg, 1);
    const int n_in = group_dim;

    constexpr int NB = 32;
    constexpr int NF = 16;
    constexpr int NF4 = NF / 4;
    const int nb = n_in / NB;
    float sumf[DSV4_DENSE_NR0] = {0.f, 0.f};

    const short ix = (short)(tiisg / (DSV4_NW / NF));
    const short il = (short)(tiisg % (DSV4_NW / NF));
    const int ib0 = (int)sgitg * NF + ix;

    device const float *xg = x + (ulong)g * (ulong)group_dim;
    device const float4 *y4 = (device const float4 *)(xg + ib0 * NB + il * NF);
    float4 yl4[NF4];
    device const half *Wg =
        W + (ulong)g * (ulong)rank * (ulong)group_dim;

    for (int ib = ib0; ib < nb; ib += nsg_use * NF) {
        for (short i = 0; i < NF4; ++i) {
            yl4[i] = y4[i];
        }
        for (int row = 0; row < DSV4_DENSE_NR0; ++row) {
            if (r0 + row >= rank) break;
            device const half4 *xb4 =
                (device const half4 *)(Wg + (ulong)(r0 + row) * (ulong)n_in + ib * NB + il * NF);
            float sumq = 0.f;
            for (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), yl4[i]);
            }
            sumf[row] += sumq;
        }
        y4 += (nsg_use * NF * DSV4_NW) / 4;
    }
    for (int i = nb * NB + (int)sgitg * DSV4_NW + (int)tiisg; i < n_in; i += DSV4_NW * nsg_use) {
        for (int row = 0; row < DSV4_DENSE_NR0; ++row) {
            if (r0 + row >= rank) break;
            sumf[row] += (float)Wg[(ulong)(r0 + row) * (ulong)n_in + i] * xg[i];
        }
    }
    device float *yg = y + (ulong)g * (ulong)rank;
    dsv4_dense_reduce_write(yg, sumf, r0, rank, tiisg, sgitg, shmem);
}

kernel void dsv4_matvec_f32(
    device const float *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant int &n_out [[buffer(3)]],
    constant int &n_in [[buffer(4)]],
    constant int &nsg [[buffer(5)]],
    threadgroup float *shmem [[threadgroup(0)]],
    uint tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int r0 = (int)tgpig * DSV4_DENSE_NR0;
    if (r0 >= n_out) return;
    const int nsg_use = max(nsg, 1);

    constexpr int NB = 32;
    constexpr int NF = 16;
    constexpr int NF4 = NF / 4;
    const int nb = n_in / NB;
    float sumf[DSV4_DENSE_NR0] = {0.f, 0.f};

    const short ix = (short)(tiisg / (DSV4_NW / NF));
    const short il = (short)(tiisg % (DSV4_NW / NF));
    const int ib0 = (int)sgitg * NF + ix;

    device const float4 *y4 = (device const float4 *)(x + ib0 * NB + il * NF);
    float4 yl4[NF4];

    for (int ib = ib0; ib < nb; ib += nsg_use * NF) {
        for (short i = 0; i < NF4; ++i) {
            yl4[i] = y4[i];
        }
        for (int row = 0; row < DSV4_DENSE_NR0; ++row) {
            if (r0 + row >= n_out) break;
            device const float4 *xb4 =
                (device const float4 *)(W + (ulong)(r0 + row) * (ulong)n_in + ib * NB + il * NF);
            float sumq = 0.f;
            for (short i = 0; i < NF4; ++i) {
                sumq += dot(xb4[i], yl4[i]);
            }
            sumf[row] += sumq;
        }
        y4 += (nsg_use * NF * DSV4_NW) / 4;
    }
    for (int i = nb * NB + (int)sgitg * DSV4_NW + (int)tiisg; i < n_in; i += DSV4_NW * nsg_use) {
        for (int row = 0; row < DSV4_DENSE_NR0; ++row) {
            if (r0 + row >= n_out) break;
            sumf[row] += W[(ulong)(r0 + row) * (ulong)n_in + i] * x[i];
        }
    }
    dsv4_dense_reduce_write(y, sumf, r0, n_out, tiisg, sgitg, shmem);
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
    constant int &nsg [[buffer(5)]],
    threadgroup float *shmem [[threadgroup(0)]],
    uint tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int r0 = (int)tgpig * DSV4_DENSE_NR0;
    if (r0 >= n_out) return;
    const int nsg_use = max(nsg, 1);

    constexpr int NQ = 8;
    constexpr int QK = 32;
    const int nb = n_in / QK;
    const int row_bytes = nb * 34;
    float sumf[DSV4_DENSE_NR0] = {0.f, 0.f};

    device const block_q8_0 *ax[DSV4_DENSE_NR0];
    for (int row = 0; row < DSV4_DENSE_NR0; ++row) {
        ax[row] = (device const block_q8_0 *)(W + (ulong)(r0 + row) * (ulong)row_bytes);
    }

    const short ix = (short)(tiisg / (DSV4_NW / NQ));
    const short il = (short)(tiisg % (DSV4_NW / NQ));
    const int ib0 = (int)sgitg * NQ + ix;
    float yl[NQ];
    device const float *yb = x + ib0 * QK + il * NQ;

    for (int ib = ib0; ib < nb; ib += nsg_use * NQ) {
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }
        for (int row = 0; row < DSV4_DENSE_NR0; ++row) {
            if (r0 + row >= n_out) break;
            device const char *qs = ax[row][ib].qs + il * NQ;
            float sumq = 0.f;
            for (short i = 0; i < NQ; ++i) {
                sumq += (float)qs[i] * yl[i];
            }
            sumf[row] += sumq * (float)ax[row][ib].d;
        }
        yb += nsg_use * NQ * QK;
    }
    dsv4_dense_reduce_write(y, sumf, r0, n_out, tiisg, sgitg, shmem);
}

// Grouped LoRA-O A (Q8_0): y[g*rank+r] = W[g][r] · x[g*group_dim:(g+1)*group_dim]
// One dispatch for all groups (tgpig.y = group). Flash uses Q8 here, not F16.
kernel void dsv4_matvec_q8_0_lora_groups(
    device const uchar *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant int &n_groups [[buffer(3)]],
    constant int &rank [[buffer(4)]],
    constant int &group_dim [[buffer(5)]],
    constant int &nsg [[buffer(6)]],
    threadgroup float *shmem [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int g = (int)tgpig.y;
    if (g >= n_groups) return;
    const int r0 = (int)tgpig.x * DSV4_DENSE_NR0;
    if (r0 >= rank) return;
    const int nsg_use = max(nsg, 1);

    constexpr int NQ = 8;
    constexpr int QK = 32;
    const int n_in = group_dim;
    const int nb = n_in / QK;
    const int row_bytes = nb * 34;
    float sumf[DSV4_DENSE_NR0] = {0.f, 0.f};

    device const uchar *Wg =
        W + (ulong)g * (ulong)rank * (ulong)row_bytes;
    device const float *xg = x + (ulong)g * (ulong)group_dim;

    device const block_q8_0 *ax[DSV4_DENSE_NR0];
    for (int row = 0; row < DSV4_DENSE_NR0; ++row) {
        ax[row] = (device const block_q8_0 *)(Wg + (ulong)(r0 + row) * (ulong)row_bytes);
    }

    const short ix = (short)(tiisg / (DSV4_NW / NQ));
    const short il = (short)(tiisg % (DSV4_NW / NQ));
    const int ib0 = (int)sgitg * NQ + ix;
    float yl[NQ];
    device const float *yb = xg + ib0 * QK + il * NQ;

    for (int ib = ib0; ib < nb; ib += nsg_use * NQ) {
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }
        for (int row = 0; row < DSV4_DENSE_NR0; ++row) {
            if (r0 + row >= rank) break;
            device const char *qs = ax[row][ib].qs + il * NQ;
            float sumq = 0.f;
            for (short i = 0; i < NQ; ++i) {
                sumq += (float)qs[i] * yl[i];
            }
            sumf[row] += sumq * (float)ax[row][ib].d;
        }
        yb += nsg_use * NQ * QK;
    }
    device float *yg = y + (ulong)g * (ulong)rank;
    dsv4_dense_reduce_write(yg, sumf, r0, rank, tiisg, sgitg, shmem);
}

kernel void dsv4_matvec_q8_0_pair_swiglu(
    device const uchar *gate [[buffer(0)]],
    device const uchar *up [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *mid [[buffer(3)]],
    constant int &n_out [[buffer(4)]],
    constant int &n_in [[buffer(5)]],
    constant float &clampv [[buffer(6)]],
    constant int &nsg [[buffer(7)]],
    threadgroup float *shmem [[threadgroup(0)]],
    uint tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int r0 = (int)tgpig * DSV4_DENSE_NR0;
    if (r0 >= n_out) return;
    const int nsg_use = max(nsg, 1);

    constexpr int NQ = 8;
    constexpr int QK = 32;
    const int nb = n_in / QK;
    const int row_bytes = nb * 34;

    device const block_q8_0 *ag[DSV4_DENSE_NR0];
    device const block_q8_0 *au[DSV4_DENSE_NR0];
    for (int row = 0; row < DSV4_DENSE_NR0; ++row) {
        ag[row] = (device const block_q8_0 *)(gate + (ulong)(r0 + row) * (ulong)row_bytes);
        au[row] = (device const block_q8_0 *)(up + (ulong)(r0 + row) * (ulong)row_bytes);
    }

    float sumg[DSV4_DENSE_NR0] = {0.f, 0.f};
    float sumu[DSV4_DENSE_NR0] = {0.f, 0.f};

    const short ix = (short)(tiisg / (DSV4_NW / NQ));
    const short il = (short)(tiisg % (DSV4_NW / NQ));
    const int ib0 = (int)sgitg * NQ + ix;
    float yl[NQ];
    device const float *yb = x + ib0 * QK + il * NQ;

    for (int ib = ib0; ib < nb; ib += nsg_use * NQ) {
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }
        for (int row = 0; row < DSV4_DENSE_NR0; ++row) {
            if (r0 + row >= n_out) break;
            device const char *qg = ag[row][ib].qs + il * NQ;
            device const char *qu = au[row][ib].qs + il * NQ;
            float sg = 0.f;
            float su = 0.f;
            for (short i = 0; i < NQ; ++i) {
                sg += (float)qg[i] * yl[i];
                su += (float)qu[i] * yl[i];
            }
            sumg[row] += sg * (float)ag[row][ib].d;
            sumu[row] += su * (float)au[row][ib].d;
        }
        yb += nsg_use * NQ * QK;
    }

    threadgroup float *sh_gate[DSV4_DENSE_NR0];
    threadgroup float *sh_up[DSV4_DENSE_NR0];
    for (int row = 0; row < DSV4_DENSE_NR0; ++row) {
        sh_gate[row] = shmem + DSV4_NW * row;
        sh_up[row] = shmem + DSV4_NW * (DSV4_DENSE_NR0 + row);
        if (sgitg == 0) {
            sh_gate[row][tiisg] = 0.0f;
            sh_up[row][tiisg] = 0.0f;
        }
        sumg[row] = simd_sum(sumg[row]);
        sumu[row] = simd_sum(sumu[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int row = 0; row < DSV4_DENSE_NR0; ++row) {
        if (tiisg == 0) {
            sh_gate[row][sgitg] = sumg[row];
            sh_up[row][sgitg] = sumu[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int row = 0; row < DSV4_DENSE_NR0 && r0 + row < n_out; ++row) {
        float g = simd_sum(sh_gate[row][tiisg]);
        float u = simd_sum(sh_up[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            if (clampv > 1.0e-6f) {
                g = min(g, clampv);
                u = clamp(u, -clampv, clampv);
            }
            mid[r0 + row] = (g / (1.0f + exp(-g))) * u;
        }
    }
}

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

// y += scales[scale_idx] * x  (MoE route weight stays on GPU).
kernel void dsv4_axpy_w(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    device const float *scales [[buffer(2)]],
    constant int &scale_idx [[buffer(3)]],
    constant int &n [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int i = (int)gid;
    if (i >= n) return;
    y[i] += scales[scale_idx] * x[i];
}

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

kernel void dsv4_zero(
    device float *y [[buffer(0)]],
    constant int &n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    int i = (int)gid;
    if (i >= n) return;
    y[i] = 0.0f;
}

kernel void dsv4_copy(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant int &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    int i = (int)gid;
    if (i >= n) return;
    y[i] = x[i];
}
