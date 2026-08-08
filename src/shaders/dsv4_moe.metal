// DeepSeek-V4 Flash — IQ2_XXS / Q2_K matvec (our implementation).
// Quant format tables are ggml IQ2_XXS constants.
//
// IQ2 matvec layout (ggml / Metal convention): each lane owns a full 32-wide
// activation tile; grid/sign LUTs live in threadgroup memory. Pair+SiLU fuses
// gate∥up so x is read once and mid is written without materializing gate/up.

#include <metal_stdlib>
using namespace metal;

constant uchar KMASK_IQ2XS[8] = {1, 2, 4, 8, 16, 32, 64, 128};

constant uchar KSIGNS_IQ2XS[128] = {
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255
};

constant ulong IQ2XXS_GRID[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908
};

struct block_iq2_xxs {
    half d;
    ushort qs[32];
};

struct block_q2_k {
    uchar scales[16];
    uchar qs[64];
    half d;
    half dmin;
};

constant int DSV4_NR0 = 4;
constant int DSV4_NSG = 2;
// 256×u64 grid + 128×u8 signs
constant uint DSV4_IQ2_TG_BYTES = 256 * 8 + 128;

static inline void dsv4_iq2_load_tables(
    threadgroup ulong *svalues,
    threadgroup uchar *ssigns,
    ushort tiisg,
    ushort sgitg)
{
    const int tid = (int)(32 * sgitg + tiisg);
    const int nthreads = 32 * DSV4_NSG;
    // Fill 256 grid entries across the threadgroup.
    {
        const int nval = (256 + nthreads - 1) / nthreads;
        for (int i = 0; i < nval; ++i) {
            const int pos = tid + i * nthreads;
            if (pos < 256) svalues[pos] = IQ2XXS_GRID[pos];
        }
    }
    // Fill 128 sign entries.
    {
        const int nval = (128 + nthreads - 1) / nthreads;
        for (int i = 0; i < nval; ++i) {
            const int pos = tid + i * nthreads;
            if (pos < 128) ssigns[pos] = KSIGNS_IQ2XS[pos];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Single-stream IQ2_XXS accumulation (raw dots; caller applies ×0.25).
static inline void dsv4_iq2_accum(
    device const block_iq2_xxs *x0,
    device const float *x,
    int n_in,
    int n_out,
    int first_row,
    int row_bytes,
    threadgroup ulong *svalues,
    threadgroup uchar *ssigns,
    thread float *sumf,
    ushort tiisg)
{
    const int nb = n_in / 256;
    const int nb32 = nb * 8;
    const int row_stride_u16 = row_bytes / 2;

    float yl[32];
    device const float *y4 = x + 32 * (int)tiisg;

    for (int ib32 = (int)tiisg; ib32 < nb32; ib32 += 32) {
        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / 8;
        const int ib = ib32 % 8;

        device const block_iq2_xxs *xr = x0 + ibl;
        device const ushort *qs = xr->qs + 4 * ib;
        device const half *dh = &xr->d;

        for (int row = 0; row < DSV4_NR0; ++row) {
            if (first_row + row >= n_out) break;
            device const uchar *aux8 = (device const uchar *)qs;
            const uint aux32 = (uint)qs[2] | ((uint)qs[3] << 16);
            const float dscale = (float)dh[0] * (0.5f + (float)(aux32 >> 28));

            float s = 0.f;
            for (int l = 0; l < 4; ++l) {
                threadgroup const uchar *grid =
                    (threadgroup const uchar *)(svalues + aux8[l]);
                const uchar signs = ssigns[(aux32 >> (7 * l)) & 127u];
                for (int j = 0; j < 8; ++j) {
                    const float v = yl[8 * l + j];
                    s += v * (float)grid[j] * (signs & KMASK_IQ2XS[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += dscale * s;

            dh += row_stride_u16;
            qs += row_stride_u16;
        }

        y4 += 32 * 32;
    }
}

// Dual-stream gate∥up accumulation (raw dots; caller applies ×0.25).
static inline void dsv4_iq2_pair_accum(
    device const block_iq2_xxs *xg0,
    device const block_iq2_xxs *xu0,
    device const float *x,
    int n_in,
    int n_out,
    int first_row,
    int row_bytes,
    threadgroup ulong *svalues,
    threadgroup uchar *ssigns,
    thread float *sumg,
    thread float *sumu,
    ushort tiisg)
{
    const int nb = n_in / 256;
    const int nb32 = nb * 8;
    const int row_stride_u16 = row_bytes / 2;

    float yl[32];
    device const float *y4 = x + 32 * (int)tiisg;

    for (int ib32 = (int)tiisg; ib32 < nb32; ib32 += 32) {
        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / 8;
        const int ib = ib32 % 8;

        device const block_iq2_xxs *xgr = xg0 + ibl;
        device const block_iq2_xxs *xur = xu0 + ibl;
        device const ushort *qg = xgr->qs + 4 * ib;
        device const ushort *qu = xur->qs + 4 * ib;
        device const half *dhg = &xgr->d;
        device const half *dhu = &xur->d;

        for (int row = 0; row < DSV4_NR0; ++row) {
            if (first_row + row >= n_out) break;
            device const uchar *aux8g = (device const uchar *)qg;
            device const uchar *aux8u = (device const uchar *)qu;
            const uint aux32g = (uint)qg[2] | ((uint)qg[3] << 16);
            const uint aux32u = (uint)qu[2] | ((uint)qu[3] << 16);
            const float dg = (float)dhg[0] * (0.5f + (float)(aux32g >> 28));
            const float du = (float)dhu[0] * (0.5f + (float)(aux32u >> 28));

            float sg = 0.f;
            float su = 0.f;
            for (int l = 0; l < 4; ++l) {
                threadgroup const uchar *gridg =
                    (threadgroup const uchar *)(svalues + aux8g[l]);
                threadgroup const uchar *gridu =
                    (threadgroup const uchar *)(svalues + aux8u[l]);
                const uchar signg = ssigns[(aux32g >> (7 * l)) & 127u];
                const uchar signu = ssigns[(aux32u >> (7 * l)) & 127u];
                for (int j = 0; j < 8; ++j) {
                    const float v = yl[8 * l + j];
                    sg += v * (float)gridg[j] * (signg & KMASK_IQ2XS[j] ? -1.f : 1.f);
                    su += v * (float)gridu[j] * (signu & KMASK_IQ2XS[j] ? -1.f : 1.f);
                }
            }
            sumg[row] += dg * sg;
            sumu[row] += du * su;

            dhg += row_stride_u16;
            dhu += row_stride_u16;
            qg += row_stride_u16;
            qu += row_stride_u16;
        }

        y4 += 32 * 32;
    }
}

kernel void dsv4_matvec_iq2_xxs(
    device const uchar *weight [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant int &n_out [[buffer(3)]],
    constant int &n_in [[buffer(4)]],
    threadgroup uchar *shmem [[threadgroup(0)]],
    uint tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int first_row = ((int)tgpig * DSV4_NSG + (int)sgitg) * DSV4_NR0;
    if (first_row >= n_out) return;

    const int nblocks = n_in / 256;
    const int row_bytes = nblocks * 66;

    threadgroup ulong *svalues = (threadgroup ulong *)shmem;
    threadgroup uchar *ssigns = (threadgroup uchar *)(svalues + 256);
    dsv4_iq2_load_tables(svalues, ssigns, tiisg, sgitg);

    float sumf[DSV4_NR0] = {0.f, 0.f, 0.f, 0.f};
    device const block_iq2_xxs *x0 =
        (device const block_iq2_xxs *)(weight + (ulong)first_row * (ulong)row_bytes);
    dsv4_iq2_accum(
        x0, x, n_in, n_out, first_row, row_bytes, svalues, ssigns, sumf, tiisg);

    for (int row = 0; row < DSV4_NR0; ++row) {
        const float s = simd_sum(sumf[row]) * 0.25f;
        if (tiisg == 0) {
            const int r = first_row + row;
            if (r < n_out) out[r] = s;
        }
    }
}

kernel void dsv4_matvec_iq2_xxs_pair_swiglu(
    device const uchar *gate [[buffer(0)]],
    device const uchar *up [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *mid [[buffer(3)]],
    constant int &n_out [[buffer(4)]],
    constant int &n_in [[buffer(5)]],
    constant float &clampv [[buffer(6)]],
    threadgroup uchar *shmem [[threadgroup(0)]],
    uint tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int first_row = ((int)tgpig * DSV4_NSG + (int)sgitg) * DSV4_NR0;
    if (first_row >= n_out) return;

    const int nblocks = n_in / 256;
    const int row_bytes = nblocks * 66;

    threadgroup ulong *svalues = (threadgroup ulong *)shmem;
    threadgroup uchar *ssigns = (threadgroup uchar *)(svalues + 256);
    dsv4_iq2_load_tables(svalues, ssigns, tiisg, sgitg);

    float sumg[DSV4_NR0] = {0.f, 0.f, 0.f, 0.f};
    float sumu[DSV4_NR0] = {0.f, 0.f, 0.f, 0.f};
    device const block_iq2_xxs *xg0 =
        (device const block_iq2_xxs *)(gate + (ulong)first_row * (ulong)row_bytes);
    device const block_iq2_xxs *xu0 =
        (device const block_iq2_xxs *)(up + (ulong)first_row * (ulong)row_bytes);
    dsv4_iq2_pair_accum(
        xg0, xu0, x, n_in, n_out, first_row, row_bytes,
        svalues, ssigns, sumg, sumu, tiisg);

    for (int row = 0; row < DSV4_NR0; ++row) {
        float g = simd_sum(sumg[row]) * 0.25f;
        float u = simd_sum(sumu[row]) * 0.25f;
        if (tiisg == 0) {
            const int r = first_row + row;
            if (r < n_out) {
                if (clampv > 1.0e-6f) {
                    g = min(g, clampv);
                    u = clamp(u, -clampv, clampv);
                }
                mid[r] = (g / (1.0f + exp(-g))) * u;
            }
        }
    }
}

// Flash top-6: one dispatch, tgpig.z selects expert slot. mid[e*n_out+r] gets
// silu(g)*u*route_w[e] (weight folded in so sum6 needs no axpy).
kernel void dsv4_slots6_iq2_pair_swiglu(
    device const uchar *gate0 [[buffer(0)]],
    device const uchar *gate1 [[buffer(1)]],
    device const uchar *gate2 [[buffer(2)]],
    device const uchar *gate3 [[buffer(3)]],
    device const uchar *gate4 [[buffer(4)]],
    device const uchar *gate5 [[buffer(5)]],
    device const uchar *up0 [[buffer(6)]],
    device const uchar *up1 [[buffer(7)]],
    device const uchar *up2 [[buffer(8)]],
    device const uchar *up3 [[buffer(9)]],
    device const uchar *up4 [[buffer(10)]],
    device const uchar *up5 [[buffer(11)]],
    device const float *x [[buffer(12)]],
    device float *mid [[buffer(13)]],
    device const float *weights [[buffer(14)]],
    constant int &n_out [[buffer(15)]],
    constant int &n_in [[buffer(16)]],
    constant float &clampv [[buffer(17)]],
    threadgroup uchar *shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int expert = (int)tgpig.z;
    if (expert < 0 || expert >= 6) return;
    const int first_row = ((int)tgpig.x * DSV4_NSG + (int)sgitg) * DSV4_NR0;
    if (first_row >= n_out) return;

    device const uchar *gate = gate0;
    device const uchar *up = up0;
    switch (expert) {
    case 1: gate = gate1; up = up1; break;
    case 2: gate = gate2; up = up2; break;
    case 3: gate = gate3; up = up3; break;
    case 4: gate = gate4; up = up4; break;
    case 5: gate = gate5; up = up5; break;
    default: break;
    }

    const int nblocks = n_in / 256;
    const int row_bytes = nblocks * 66;
    threadgroup ulong *svalues = (threadgroup ulong *)shmem;
    threadgroup uchar *ssigns = (threadgroup uchar *)(svalues + 256);
    dsv4_iq2_load_tables(svalues, ssigns, tiisg, sgitg);

    float sumg[DSV4_NR0] = {0.f, 0.f, 0.f, 0.f};
    float sumu[DSV4_NR0] = {0.f, 0.f, 0.f, 0.f};
    device const block_iq2_xxs *xg0 =
        (device const block_iq2_xxs *)(gate + (ulong)first_row * (ulong)row_bytes);
    device const block_iq2_xxs *xu0 =
        (device const block_iq2_xxs *)(up + (ulong)first_row * (ulong)row_bytes);
    dsv4_iq2_pair_accum(
        xg0, xu0, x, n_in, n_out, first_row, row_bytes,
        svalues, ssigns, sumg, sumu, tiisg);

    const float rw = weights[expert];
    device float *mid_e = mid + (ulong)expert * (ulong)n_out;
    for (int row = 0; row < DSV4_NR0; ++row) {
        float g = simd_sum(sumg[row]) * 0.25f;
        float u = simd_sum(sumu[row]) * 0.25f;
        if (tiisg == 0) {
            const int r = first_row + row;
            if (r < n_out) {
                if (clampv > 1.0e-6f) {
                    g = min(g, clampv);
                    u = clamp(u, -clampv, clampv);
                }
                mid_e[r] = (g / (1.0f + exp(-g))) * u * rw;
            }
        }
    }
}

// One dispatch: for each output row, sum Q2_K downs of all 6 weighted mids.
kernel void dsv4_slots6_q2k_sum6(
    device const uchar *down0 [[buffer(0)]],
    device const uchar *down1 [[buffer(1)]],
    device const uchar *down2 [[buffer(2)]],
    device const uchar *down3 [[buffer(3)]],
    device const uchar *down4 [[buffer(4)]],
    device const uchar *down5 [[buffer(5)]],
    device const float *mid [[buffer(6)]],
    device float *out [[buffer(7)]],
    constant int &n_out [[buffer(8)]],
    constant int &n_in [[buffer(9)]],
    uint tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int first_row = ((int)tgpig * DSV4_NSG + (int)sgitg) * DSV4_NR0;
    if (first_row >= n_out) return;

    const int nblocks = n_in / 256;
    const int row_bytes = nblocks * 84;
    float sumf[DSV4_NR0] = {0.f, 0.f, 0.f, 0.f};

    const short ix = (short)(tiisg / 8);
    const short it = (short)(tiisg % 8);
    const short iq = (short)(it / 4);
    const short ir = (short)(it % 4);
    const short is = (short)((8 * ir) / 16);

    for (int expert = 0; expert < 6; ++expert) {
        device const uchar *weight = down0;
        switch (expert) {
        case 1: weight = down1; break;
        case 2: weight = down2; break;
        case 3: weight = down3; break;
        case 4: weight = down4; break;
        case 5: weight = down5; break;
        default: break;
        }
        device const block_q2_k *x0 =
            (device const block_q2_k *)(weight + (ulong)first_row * (ulong)row_bytes);
        device const float *y = mid + (ulong)expert * (ulong)n_in;
        device const float *y4 = y + ix * 256 + 128 * iq + 8 * ir;

        float yl[32];
        for (int ib = ix; ib < nblocks; ib += 4) {
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i + 0];
                sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i + 32];
                sumy[1] += yl[i + 8];
                yl[i + 16] = y4[i + 64];
                sumy[2] += yl[i + 16];
                yl[i + 24] = y4[i + 96];
                sumy[3] += yl[i + 24];
            }

            device const uchar *sc = (device const uchar *)x0[ib].scales + 8 * iq + is;
            device const ushort *qs = (device const ushort *)x0[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x0[ib].d;

            for (int row = 0; row < DSV4_NR0; ++row) {
                if (first_row + row >= n_out) break;
                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                for (int i = 0; i < 8; i += 2) {
                    acc1[0] += yl[i + 0] * (float)(qs[i / 2] & 0x0003);
                    acc2[0] += yl[i + 1] * (float)(qs[i / 2] & 0x0300);
                    acc1[1] += yl[i + 8] * (float)(qs[i / 2] & 0x000c);
                    acc2[1] += yl[i + 9] * (float)(qs[i / 2] & 0x0c00);
                    acc1[2] += yl[i + 16] * (float)(qs[i / 2] & 0x0030);
                    acc2[2] += yl[i + 17] * (float)(qs[i / 2] & 0x3000);
                    acc1[3] += yl[i + 24] * (float)(qs[i / 2] & 0x00c0);
                    acc2[3] += yl[i + 25] * (float)(qs[i / 2] & 0xc000);
                }
                const float d = (float)dh[0];
                const float m = (float)dh[1] * (1.f / 16.f);
                sumf[row] +=
                    d * ((acc1[0] + (1.f / 256.f) * acc2[0]) * (float)(sc[0] & 0xF) * (1.f / 1.f) +
                         (acc1[1] + (1.f / 256.f) * acc2[1]) * (float)(sc[2] & 0xF) * (1.f / 4.f) +
                         (acc1[2] + (1.f / 256.f) * acc2[2]) * (float)(sc[4] & 0xF) * (1.f / 16.f) +
                         (acc1[3] + (1.f / 256.f) * acc2[3]) * (float)(sc[6] & 0xF) * (1.f / 64.f)) -
                    m * (sumy[0] * (float)(sc[0] & 0xF0) + sumy[1] * (float)(sc[2] & 0xF0) +
                         sumy[2] * (float)(sc[4] & 0xF0) + sumy[3] * (float)(sc[6] & 0xF0));

                qs += row_bytes / 2;
                sc += row_bytes;
                dh += row_bytes / 2;
            }
            y4 += 4 * 256;
        }
    }

    for (int row = 0; row < DSV4_NR0; ++row) {
        const float s = simd_sum(sumf[row]);
        if (tiisg == 0) {
            const int r = first_row + row;
            if (r < n_out) out[r] = s;
        }
    }
}

kernel void dsv4_matvec_q2_k(
    device const uchar *weight [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant int &n_out [[buffer(3)]],
    constant int &n_in [[buffer(4)]],
    uint tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int first_row = ((int)tgpig * DSV4_NSG + (int)sgitg) * DSV4_NR0;
    if (first_row >= n_out) return;

    const int nblocks = n_in / 256;
    const int row_bytes = nblocks * 84;
    float sumf[DSV4_NR0] = {0.f, 0.f, 0.f, 0.f};

    // Bit-parallel Q2_K: each lane owns 8 consecutive values in four 64-wide
    // quadrants of a 256-block (ggml Metal mul_mv_q2_K pattern).
    const short ix = (short)(tiisg / 8);   // 0..3  → which of 4 block-chunks
    const short it = (short)(tiisg % 8);   // 0..7
    const short iq = (short)(it / 4);      // 0..1
    const short ir = (short)(it % 4);      // 0..3
    const short is = (short)((8 * ir) / 16);

    device const block_q2_k *x0 =
        (device const block_q2_k *)(weight + (ulong)first_row * (ulong)row_bytes);
    device const float *y = x;
    device const float *y4 = y + ix * 256 + 128 * iq + 8 * ir;

    float yl[32];
    for (int ib = ix; ib < nblocks; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i + 0];
            sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i + 32];
            sumy[1] += yl[i + 8];
            yl[i + 16] = y4[i + 64];
            sumy[2] += yl[i + 16];
            yl[i + 24] = y4[i + 96];
            sumy[3] += yl[i + 24];
        }

        device const uchar *sc = (device const uchar *)x0[ib].scales + 8 * iq + is;
        device const ushort *qs = (device const ushort *)x0[ib].qs + 16 * iq + 4 * ir;
        device const half *dh = &x0[ib].d;

        for (int row = 0; row < DSV4_NR0; ++row) {
            if (first_row + row >= n_out) break;
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i + 0] * (float)(qs[i / 2] & 0x0003);
                acc2[0] += yl[i + 1] * (float)(qs[i / 2] & 0x0300);
                acc1[1] += yl[i + 8] * (float)(qs[i / 2] & 0x000c);
                acc2[1] += yl[i + 9] * (float)(qs[i / 2] & 0x0c00);
                acc1[2] += yl[i + 16] * (float)(qs[i / 2] & 0x0030);
                acc2[2] += yl[i + 17] * (float)(qs[i / 2] & 0x3000);
                acc1[3] += yl[i + 24] * (float)(qs[i / 2] & 0x00c0);
                acc2[3] += yl[i + 25] * (float)(qs[i / 2] & 0xc000);
            }
            const float d = (float)dh[0];
            const float m = (float)dh[1] * (1.f / 16.f);
            sumf[row] +=
                d * ((acc1[0] + (1.f / 256.f) * acc2[0]) * (float)(sc[0] & 0xF) * (1.f / 1.f) +
                     (acc1[1] + (1.f / 256.f) * acc2[1]) * (float)(sc[2] & 0xF) * (1.f / 4.f) +
                     (acc1[2] + (1.f / 256.f) * acc2[2]) * (float)(sc[4] & 0xF) * (1.f / 16.f) +
                     (acc1[3] + (1.f / 256.f) * acc2[3]) * (float)(sc[6] & 0xF) * (1.f / 64.f)) -
                m * (sumy[0] * (float)(sc[0] & 0xF0) + sumy[1] * (float)(sc[2] & 0xF0) +
                     sumy[2] * (float)(sc[4] & 0xF0) + sumy[3] * (float)(sc[6] & 0xF0));

            qs += row_bytes / 2;
            sc += row_bytes;
            dh += row_bytes / 2;
        }
        y4 += 4 * 256;
    }

    for (int row = 0; row < DSV4_NR0; ++row) {
        const float s = simd_sum(sumf[row]);
        if (tiisg == 0) {
            const int r = first_row + row;
            if (r < n_out) out[r] = s;
        }
    }
}

kernel void dsv4_swiglu(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant int &n [[buffer(3)]],
    constant float &clampv [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int i = (int)gid;
    if (i >= n) return;
    float g = gate[i];
    float u = up[i];
    if (clampv > 1.0e-6f) {
        g = min(g, clampv);
        u = clamp(u, -clampv, clampv);
    }
    out[i] = (g / (1.0f + exp(-g))) * u;
}

kernel void dsv4_router_sqrt_softplus(
    device const float *logits [[buffer(0)]],
    device float *probs [[buffer(1)]],
    constant int &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    int i = (int)gid;
    if (i >= n) return;
    float x = logits[i];
    float sp;
    if (x > 20.f) {
        sp = x;
    } else if (x < -20.f) {
        sp = exp(x);
    } else {
        sp = log(1.f + exp(x));
    }
    probs[i] = sqrt(sp);
}

// Map top-k expert ids → LRU slot indices via host-maintained map[-1=miss].
// Also mirrors route_ids into hist at hist_off for end-of-token refresh.
kernel void dsv4_map_route_slots(
    device const int *route_ids [[buffer(0)]],
    device const int *slot_map [[buffer(1)]],
    device int *out_slots [[buffer(2)]],
    device int *miss_flag [[buffer(3)]],
    device int *hist [[buffer(4)]],
    constant int &k [[buffer(5)]],
    constant int &hist_off [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    int miss = 0;
    for (int t = 0; t < k; ++t) {
        int id = route_ids[t];
        int s = (id >= 0) ? slot_map[id] : -1;
        out_slots[t] = s;
        hist[hist_off + t] = id;
        if (s < 0) miss = 1;
    }
    miss_flag[0] = miss;
}

// Packed top-6 IQ2: gate/up are contiguous LRU packs; slots[e] selects offset.
kernel void dsv4_slots6_iq2_pair_swiglu_packed(
    device const uchar *gate_pack [[buffer(0)]],
    device const uchar *up_pack [[buffer(1)]],
    device const int *slots [[buffer(2)]],
    device const float *x [[buffer(3)]],
    device float *mid [[buffer(4)]],
    device const float *weights [[buffer(5)]],
    constant int &n_out [[buffer(6)]],
    constant int &n_in [[buffer(7)]],
    constant float &clampv [[buffer(8)]],
    constant uint &gate_stride [[buffer(9)]],
    constant uint &up_stride [[buffer(10)]],
    threadgroup uchar *shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int expert = (int)tgpig.z;
    if (expert < 0 || expert >= 6) return;
    const int si = slots[expert];
    if (si < 0) {
        // Miss: zero this expert's mid contribution.
        const int first_row = ((int)tgpig.x * DSV4_NSG + (int)sgitg) * DSV4_NR0;
        if (first_row >= n_out) return;
        device float *mid_e = mid + (ulong)expert * (ulong)n_out;
        for (int row = 0; row < DSV4_NR0; ++row) {
            const int r = first_row + row;
            if (r < n_out && tiisg == 0) mid_e[r] = 0.f;
        }
        return;
    }
    const int first_row = ((int)tgpig.x * DSV4_NSG + (int)sgitg) * DSV4_NR0;
    if (first_row >= n_out) return;

    device const uchar *gate = gate_pack + (ulong)si * (ulong)gate_stride;
    device const uchar *up = up_pack + (ulong)si * (ulong)up_stride;

    const int nblocks = n_in / 256;
    const int row_bytes = nblocks * 66;
    threadgroup ulong *svalues = (threadgroup ulong *)shmem;
    threadgroup uchar *ssigns = (threadgroup uchar *)(svalues + 256);
    dsv4_iq2_load_tables(svalues, ssigns, tiisg, sgitg);

    float sumg[DSV4_NR0] = {0.f, 0.f, 0.f, 0.f};
    float sumu[DSV4_NR0] = {0.f, 0.f, 0.f, 0.f};
    device const block_iq2_xxs *xg0 =
        (device const block_iq2_xxs *)(gate + (ulong)first_row * (ulong)row_bytes);
    device const block_iq2_xxs *xu0 =
        (device const block_iq2_xxs *)(up + (ulong)first_row * (ulong)row_bytes);
    dsv4_iq2_pair_accum(
        xg0, xu0, x, n_in, n_out, first_row, row_bytes,
        svalues, ssigns, sumg, sumu, tiisg);

    const float rw = weights[expert];
    device float *mid_e = mid + (ulong)expert * (ulong)n_out;
    for (int row = 0; row < DSV4_NR0; ++row) {
        float g = simd_sum(sumg[row]) * 0.25f;
        float u = simd_sum(sumu[row]) * 0.25f;
        if (tiisg == 0) {
            const int r = first_row + row;
            if (r < n_out) {
                if (clampv > 1.0e-6f) {
                    g = min(g, clampv);
                    u = clamp(u, -clampv, clampv);
                }
                mid_e[r] = (g / (1.0f + exp(-g))) * u * rw;
            }
        }
    }
}

kernel void dsv4_slots6_q2k_sum6_packed(
    device const uchar *down_pack [[buffer(0)]],
    device const int *slots [[buffer(1)]],
    device const float *mid [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant int &n_out [[buffer(4)]],
    constant int &n_in [[buffer(5)]],
    constant uint &down_stride [[buffer(6)]],
    uint tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int first_row = ((int)tgpig * DSV4_NSG + (int)sgitg) * DSV4_NR0;
    if (first_row >= n_out) return;

    const int nblocks = n_in / 256;
    const int row_bytes = nblocks * 84;
    float sumf[DSV4_NR0] = {0.f, 0.f, 0.f, 0.f};

    const short ix = (short)(tiisg / 8);
    const short it = (short)(tiisg % 8);
    const short iq = (short)(it / 4);
    const short ir = (short)(it % 4);
    const short is = (short)((8 * ir) / 16);

    for (int expert = 0; expert < 6; ++expert) {
        const int si = slots[expert];
        if (si < 0) continue;
        device const uchar *weight = down_pack + (ulong)si * (ulong)down_stride;
        device const block_q2_k *x0 =
            (device const block_q2_k *)(weight + (ulong)first_row * (ulong)row_bytes);
        device const float *y = mid + (ulong)expert * (ulong)n_in;
        device const float *y4 = y + ix * 256 + 128 * iq + 8 * ir;

        float yl[32];
        for (int ib = ix; ib < nblocks; ib += 4) {
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i + 0];
                sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i + 32];
                sumy[1] += yl[i + 8];
                yl[i + 16] = y4[i + 64];
                sumy[2] += yl[i + 16];
                yl[i + 24] = y4[i + 96];
                sumy[3] += yl[i + 24];
            }

            device const uchar *sc = (device const uchar *)x0[ib].scales + 8 * iq + is;
            device const ushort *qs = (device const ushort *)x0[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x0[ib].d;

            for (int row = 0; row < DSV4_NR0; ++row) {
                if (first_row + row >= n_out) break;
                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                for (int i = 0; i < 8; i += 2) {
                    acc1[0] += yl[i + 0] * (float)(qs[i / 2] & 0x0003);
                    acc2[0] += yl[i + 1] * (float)(qs[i / 2] & 0x0300);
                    acc1[1] += yl[i + 8] * (float)(qs[i / 2] & 0x000c);
                    acc2[1] += yl[i + 9] * (float)(qs[i / 2] & 0x0c00);
                    acc1[2] += yl[i + 16] * (float)(qs[i / 2] & 0x0030);
                    acc2[2] += yl[i + 17] * (float)(qs[i / 2] & 0x3000);
                    acc1[3] += yl[i + 24] * (float)(qs[i / 2] & 0x00c0);
                    acc2[3] += yl[i + 25] * (float)(qs[i / 2] & 0xc000);
                }
                const float d = (float)dh[0];
                const float m = (float)dh[1] * (1.f / 16.f);
                sumf[row] +=
                    d * ((acc1[0] + (1.f / 256.f) * acc2[0]) * (float)(sc[0] & 0xF) * (1.f / 1.f) +
                         (acc1[1] + (1.f / 256.f) * acc2[1]) * (float)(sc[2] & 0xF) * (1.f / 4.f) +
                         (acc1[2] + (1.f / 256.f) * acc2[2]) * (float)(sc[4] & 0xF) * (1.f / 16.f) +
                         (acc1[3] + (1.f / 256.f) * acc2[3]) * (float)(sc[6] & 0xF) * (1.f / 64.f)) -
                    m * (sumy[0] * (float)(sc[0] & 0xF0) + sumy[1] * (float)(sc[2] & 0xF0) +
                         sumy[2] * (float)(sc[4] & 0xF0) + sumy[3] * (float)(sc[6] & 0xF0));

                qs += row_bytes / 2;
                sc += row_bytes;
                dh += row_bytes / 2;
            }
            y4 += 4 * 256;
        }
    }

    for (int row = 0; row < DSV4_NR0; ++row) {
        const float s = simd_sum(sumf[row]);
        if (tiisg == 0) {
            const int r = first_row + row;
            if (r < n_out) out[r] = s;
        }
    }
}

// Top-k router select (n≤256, k≤8). Matches CPU select_topk_experts.
kernel void dsv4_router_topk(
    device const float *probs [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    device int *out_ids [[buffer(2)]],
    device float *out_w [[buffer(3)]],
    constant int &n [[buffer(4)]],
    constant int &k [[buffer(5)]],
    constant float &weight_scale [[buffer(6)]],
    constant int &has_bias [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    // Selection scores + unbiased probs.
    thread float sel[256];
    thread float pcopy[256];
    thread int alive[256];
    for (int i = 0; i < n; ++i) {
        pcopy[i] = probs[i];
        sel[i] = pcopy[i] + ((has_bias != 0) ? bias[i] : 0.0f);
        alive[i] = 1;
    }
    float sum = 0.0f;
    for (int t = 0; t < k; ++t) {
        int best = -1;
        float best_s = -1.0e30f;
        for (int i = 0; i < n; ++i) {
            if (!alive[i]) continue;
            if (sel[i] > best_s) {
                best_s = sel[i];
                best = i;
            }
        }
        alive[best] = 0;
        out_ids[t] = best;
        out_w[t] = pcopy[best];
        sum += pcopy[best];
    }
    float inv = weight_scale / max(sum, 6.103515625e-5f);
    for (int t = 0; t < k; ++t) out_w[t] *= inv;
}

// Hash-MoE: renorm probs at fixed expert ids for this token.
kernel void dsv4_router_hash_select(
    device const float *probs [[buffer(0)]],
    device const int *eids [[buffer(1)]],
    device int *out_ids [[buffer(2)]],
    device float *out_w [[buffer(3)]],
    constant int &k [[buffer(4)]],
    constant float &weight_scale [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    float sum = 0.0f;
    for (int t = 0; t < k; ++t) {
        int id = eids[t];
        out_ids[t] = id;
        float p = probs[id];
        out_w[t] = p;
        sum += p;
    }
    float inv = weight_scale / max(sum, 6.103515625e-5f);
    for (int t = 0; t < k; ++t) out_w[t] *= inv;
}
