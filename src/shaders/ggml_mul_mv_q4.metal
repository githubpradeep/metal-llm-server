// ggml-metal Q4_0 GEMV (GGUF layout) — block_q_n_dot_y + mul_vec_q_n_f32_impl.
// Ported from ggml-org/llama.cpp ggml-metal.metal (N_R0=4, N_SG=2, N_SIMDWIDTH=32).

#include <metal_stdlib>
using namespace metal;

#define QK4_0 32
#define N_SIMDWIDTH 32

// Must match GgmlMulMvArgs in ggml_gemv.rs
struct ggml_mul_mv_args {
    int32_t ne00;
    int32_t ne01;
    int32_t ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne10;
    int32_t ne11;
    int32_t ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne0;
    int32_t ne1;
    int32_t nr0;
    int16_t r2;
    int16_t r3;
};

// Must match ggml_mul_mv_ext_args in ggml_gemv.rs (ext kernels only).
struct ggml_mul_mv_ext_args {
    int32_t ne00;
    int32_t ne01;
    int32_t ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne10;
    int32_t ne11;
    int32_t ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne0;
    int32_t ne1;
    int16_t r2;
    int16_t r3;
    int16_t nsg;
    int16_t nxpsg;
};

struct block_q4_0 {
    half d;
    uint8_t qs[16];
};

inline float block_q_n_dot_y(device const block_q4_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float2 acc = 0.f;

    device const uint16_t * qs = ((device const uint16_t *)qb_curr + 1 + il/2);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F)
                + yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[1] += yl[i + 8] * (qs[i / 2] & 0x00F0)
                + yl[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (sumy * -8.f + acc[0] + acc[1]);
}

template<typename block_q_type, short NR0, short NSG, short NW>
void mul_vec_q_n_f32_impl(
        device const void  * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
                   uint3 tgpig, uint tiisg, uint sgitg) {
    const int nb = ne00/QK4_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q_type * x = (device const block_q_type *) src0 + offset0;
    device const float        * y = (device const float        *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[16];
    float sumf[NR0];
    for (short row = 0; row < NR0; ++row) sumf[row] = 0.f;

    const int ix = (tiisg/2);
    const int il = (tiisg%2)*8;

    device const float * yb = y + ix * QK4_0 + il;

    for (int ib = ix; ib < nb; ib += NW/2) {
        float sumy = 0;
        for (int i = 0; i < 8; i += 2) {
            sumy += yb[i] + yb[i+1];
            yl[i+0] = yb[i+ 0];
            yl[i+1] = yb[i+ 1]/256.f;

            sumy += yb[i+16] + yb[i+17];
            yl[i+8] = yb[i+16]/16.f;
            yl[i+9] = yb[i+17]/4096.f;
        }

        for (int row = 0; row < NR0; row++) {
            sumf[row] += block_q_n_dot_y(x+ib+row*nb, sumy, yl, il);
        }

        yb += QK4_0 * 16;
    }

    for (int row = 0; row < NR0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < ne01) {
            dst[im*ne0*ne1 + r1*ne0 + first_row + row] = tot;
        }
    }
}

constant uint GGML_N_SG_Q4_0 = 2;
constant uint GGML_N_R0_Q4_0 = 4;

kernel void matvec_ggml_q4_0(
    device const char * W [[buffer(0)]],
    device const char * x [[buffer(1)]],
    device char * y [[buffer(2)]],
    constant ggml_mul_mv_args& args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    mul_vec_q_n_f32_impl<block_q4_0, 4, 2, 32>(
        W, (device const float *)x, (device float *)y,
        args.ne00, args.ne01, args.ne02,
        args.ne10, args.ne12,
        args.ne0, args.ne1,
        uint(args.r2), uint(args.r3),
        tgpig, tiisg, sgitg);
}

template<typename block_q_type, short NR0, short NSG, short NW>
void mul_vec_q_n_f16x_impl(
        device const void  * src0,
        device const half  * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
                   uint3 tgpig, uint tiisg, uint sgitg) {
    const int nb = ne00/QK4_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q_type * x = (device const block_q_type *) src0 + offset0;
    device const half         * y = (device const half         *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[16];
    float sumf[NR0];
    for (short row = 0; row < NR0; ++row) sumf[row] = 0.f;

    const int ix = (tiisg/2);
    const int il = (tiisg%2)*8;

    device const half * yb = y + ix * QK4_0 + il;

    for (int ib = ix; ib < nb; ib += NW/2) {
        float sumy = 0;
        for (int i = 0; i < 8; i += 2) {
            sumy += float(yb[i]) + float(yb[i+1]);
            yl[i+0] = float(yb[i+ 0]);
            yl[i+1] = float(yb[i+ 1])/256.f;

            sumy += float(yb[i+16]) + float(yb[i+17]);
            yl[i+8] = float(yb[i+16])/16.f;
            yl[i+9] = float(yb[i+17])/4096.f;
        }

        for (int row = 0; row < NR0; row++) {
            sumf[row] += block_q_n_dot_y(x+ib+row*nb, sumy, yl, il);
        }

        yb += QK4_0 * 16;
    }

    for (int row = 0; row < NR0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < ne01) {
            dst[im*ne0*ne1 + r1*ne0 + first_row + row] = tot;
        }
    }
}

kernel void matvec_ggml_q4_0_f16x(
    device const char * W [[buffer(0)]],
    device const char * x [[buffer(1)]],
    device char * y [[buffer(2)]],
    constant ggml_mul_mv_args& args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    mul_vec_q_n_f16x_impl<block_q4_0, 4, 2, 32>(
        W, (device const half *)x, (device float *)y,
        args.ne00, args.ne01, args.ne02,
        args.ne10, args.ne12,
        args.ne0, args.ne1,
        uint(args.r2), uint(args.r3),
        tgpig, tiisg, sgitg);
}

// ─── mul_mv_ext (batch matvec) ───────────────────────────────────────────────

template <typename type4>
void dequantize_q4_0_t4(device const block_q4_0 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = (il/4) ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + md;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + md;
    }
}

template<short nxpsg, short r1ptg>
void mul_mv_ext_q4_f32_impl(
        constant ggml_mul_mv_ext_args& args,
        device const char * src0,
        device const char * src1,
        device char * dst,
        uint3 tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short chpt = 4;
    const short nypsg = 32 / nxpsg;
    const short tx = tiisg % nxpsg;
    const short ty = tiisg / nxpsg;
    const short chpb = QK4_0 / 4;

    const int i01 = tgpig.x * (nypsg * args.nsg) + nypsg * sgitg + ty;
    const int i11 = tgpig.y * r1ptg;
    const int i1m = tgpig.z;

    const int i12 = i1m % args.ne12;
    const int i13 = i1m / args.ne12;

    const uint64_t offset0 = i01 * args.nb01 + (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;
    const uint64_t offset1 = i11 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_q4_0 * xq = (i01 < args.ne01)
        ? (device const block_q4_0 *)(src0 + offset0) + tx / chpb
        : (device const block_q4_0 *)src0;

    device const float4 * y4[r1ptg];
    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4[ir1] = (i11 + ir1 < args.ne11)
            ? (device const float4 *)(src1 + offset1 + ir1 * args.nb11) + tx
            : (device const float4 *)src1;
    }

    float sumf[r1ptg];
    for (int ir1 = 0; ir1 < r1ptg; ++ir1) sumf[ir1] = 0.0f;

    short cch = tx % chpb;
    for (int ich = tx; 4 * ich < args.ne00; ich += chpt * nxpsg) {
        float4 lx[chpt];
#pragma unroll
        for (short ch = 0; ch < chpt; ++ch) {
            dequantize_q4_0_t4<float4>(xq, cch, lx[ch]);
            cch += nxpsg;
            if (cch >= chpb) {
                xq += cch / chpb;
                cch %= chpb;
            }
        }
#pragma unroll
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                sumf[ir1] += dot(lx[ch], y4[ir1][ch * nxpsg]);
            }
        }
#pragma unroll
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4[ir1] += chpt * nxpsg;
        }
    }

    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
        if (nxpsg >= 16) sumf[ir1] += simd_shuffle_down(sumf[ir1], 8);
        if (nxpsg >= 8)  sumf[ir1] += simd_shuffle_down(sumf[ir1], 4);
        if (nxpsg >= 4)  sumf[ir1] += simd_shuffle_down(sumf[ir1], 2);
        if (nxpsg >= 2)  sumf[ir1] += simd_shuffle_down(sumf[ir1], 1);
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *)dst
                + (uint64_t)i1m * args.ne0 * args.ne1 + (uint64_t)(i11 + ir1) * args.ne0;
            if (i01 < args.ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

kernel void matvec_ggml_ext_q4_nx4_r4(
    device const char * W [[buffer(0)]],
    device const char * x [[buffer(1)]],
    device char * y [[buffer(2)]],
    constant ggml_mul_mv_ext_args& args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    mul_mv_ext_q4_f32_impl<4, 4>(args, W, x, y, tgpig, tiisg, sgitg);
}

kernel void matvec_ggml_ext_q4_nx8_r4(
    device const char * W [[buffer(0)]],
    device const char * x [[buffer(1)]],
    device char * y [[buffer(2)]],
    constant ggml_mul_mv_ext_args& args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    mul_mv_ext_q4_f32_impl<8, 4>(args, W, x, y, tgpig, tiisg, sgitg);
}

kernel void matvec_ggml_ext_q4_nx16_r4(
    device const char * W [[buffer(0)]],
    device const char * x [[buffer(1)]],
    device char * y [[buffer(2)]],
    constant ggml_mul_mv_ext_args& args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    mul_mv_ext_q4_f32_impl<16, 4>(args, W, x, y, tgpig, tiisg, sgitg);
}

// ─── Fused gate+up Q4_0 GEMV (two weight matrices, shared x loads) ───────────
// One dispatch computes gate = W_gate @ x and up = W_up @ x. The x vector is
// loaded once and reused for both dot products, cutting dispatch bubbles
// compared to two separate matvec_ggml_q4_0 dispatches.

template<typename block_q_type, short NR0, short NSG, short NW>
void mul_vec_dual_q_n_f32_impl(
        device const void  * src0_a,
        device const void  * src0_b,
        device const float * src1,
        device       float * dst_a,
        device       float * dst_b,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
                   uint3 tgpig, uint tiisg, uint sgitg) {
    const int nb = ne00/QK4_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q_type * x_a = (device const block_q_type *) src0_a + offset0;
    device const block_q_type * x_b = (device const block_q_type *) src0_b + offset0;
    device const float        * y   = (device const float        *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[16];
    float sumf_a[NR0];
    float sumf_b[NR0];
    for (short row = 0; row < NR0; ++row) {
        sumf_a[row] = 0.f;
        sumf_b[row] = 0.f;
    }

    const int ix = (tiisg/2);
    const int il = (tiisg%2)*8;

    device const float * yb = y + ix * QK4_0 + il;

    for (int ib = ix; ib < nb; ib += NW/2) {
        float sumy = 0;
        for (int i = 0; i < 8; i += 2) {
            sumy += yb[i] + yb[i+1];
            yl[i+0] = yb[i+ 0];
            yl[i+1] = yb[i+ 1]/256.f;

            sumy += yb[i+16] + yb[i+17];
            yl[i+8] = yb[i+16]/16.f;
            yl[i+9] = yb[i+17]/4096.f;
        }

        for (int row = 0; row < NR0; row++) {
            sumf_a[row] += block_q_n_dot_y(x_a+ib+row*nb, sumy, yl, il);
            sumf_b[row] += block_q_n_dot_y(x_b+ib+row*nb, sumy, yl, il);
        }

        yb += QK4_0 * 16;
    }

    for (int row = 0; row < NR0; ++row) {
        const int row_idx = first_row + row;
        const bool valid = row_idx < ne01;
        const float tot_a = simd_sum(sumf_a[row]);
        const float tot_b = simd_sum(sumf_b[row]);
        if (tiisg == 0 && valid) {
            const uint64_t dst_off = (uint64_t)im*ne0*ne1 + (uint64_t)r1*ne0 + (uint64_t)row_idx;
            dst_a[dst_off] = tot_a;
            dst_b[dst_off] = tot_b;
        }
    }
}

kernel void matvec_ggml_q4_0_dual(
    device const char * W_gate [[buffer(0)]],
    device const char * W_up   [[buffer(1)]],
    device const char * x      [[buffer(2)]],
    device char * gate [[buffer(3)]],
    device char * up   [[buffer(4)]],
    constant ggml_mul_mv_args& args [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    mul_vec_dual_q_n_f32_impl<block_q4_0, 4, 2, 32>(
        W_gate, W_up, (device const float *)x,
        (device float *)gate, (device float *)up,
        args.ne00, args.ne01, args.ne02,
        args.ne10, args.ne12,
        args.ne0, args.ne1,
        uint(args.r2), uint(args.r3),
        tgpig, tiisg, sgitg);
}

// ─── Fused gate+up Q4_0 GEMV with GeLU(gate)*up ─────────────────────────────
// One dispatch computes gelu = GeLU(W_gate @ x) * (W_up @ x). The x vector is
// loaded once and reused for both dot products, then the gate result is run
// through PyTorch tanh-GeLU and multiplied with the up result before writing
// a single gelu scratch vector. This avoids writing gate/up scratch buffers
// and a separate GeLU multiply dispatch.

inline float gelu_pytorch_tanh_q4(float x) {
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    inner = clamp(inner, -10.0f, 10.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}

template<typename block_q_type, short NR0, short NSG, short NW>
void mul_vec_gelu_q_n_f32_impl(
        device const void  * src0_gate,
        device const void  * src0_up,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
                   uint3 tgpig, uint tiisg, uint sgitg) {
    const int nb = ne00/QK4_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q_type * x_gate = (device const block_q_type *) src0_gate + offset0;
    device const block_q_type * x_up   = (device const block_q_type *) src0_up   + offset0;
    device const float        * y      = (device const float        *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[16];
    float sumf_gate[NR0];
    float sumf_up[NR0];
    for (short row = 0; row < NR0; ++row) {
        sumf_gate[row] = 0.f;
        sumf_up[row]   = 0.f;
    }

    const int ix = (tiisg/2);
    const int il = (tiisg%2)*8;

    device const float * yb = y + ix * QK4_0 + il;

    for (int ib = ix; ib < nb; ib += NW/2) {
        float sumy = 0;
        for (int i = 0; i < 8; i += 2) {
            sumy += yb[i] + yb[i+1];
            yl[i+0] = yb[i+ 0];
            yl[i+1] = yb[i+ 1]/256.f;

            sumy += yb[i+16] + yb[i+17];
            yl[i+8] = yb[i+16]/16.f;
            yl[i+9] = yb[i+17]/4096.f;
        }

        for (int row = 0; row < NR0; row++) {
            sumf_gate[row] += block_q_n_dot_y(x_gate+ib+row*nb, sumy, yl, il);
            sumf_up[row]   += block_q_n_dot_y(x_up  +ib+row*nb, sumy, yl, il);
        }

        yb += QK4_0 * 16;
    }

    for (int row = 0; row < NR0; ++row) {
        const int row_idx = first_row + row;
        const bool valid = row_idx < ne01;
        const float gate = simd_sum(sumf_gate[row]);
        const float up   = simd_sum(sumf_up[row]);
        // Compute GeLU on every SIMD lane so the ALUs stay active; only lane 0
        // actually writes. simd_sum returns the reduced value to all lanes.
        const float gelu = valid ? gelu_pytorch_tanh_q4(gate) * up : 0.0f;
        if (tiisg == 0 && valid) {
            const uint64_t dst_off = (uint64_t)im*ne0*ne1 + (uint64_t)r1*ne0 + (uint64_t)row_idx;
            dst[dst_off] = gelu;
        }
    }
}

kernel void matvec_ggml_q4_0_gelu_mul(
    device const char * W_gate [[buffer(0)]],
    device const char * W_up   [[buffer(1)]],
    device const char * x      [[buffer(2)]],
    device char * y            [[buffer(3)]],
    constant ggml_mul_mv_args& args [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    mul_vec_gelu_q_n_f32_impl<block_q4_0, 4, 2, 32>(
        W_gate, W_up, (device const float *)x, (device float *)y,
        args.ne00, args.ne01, args.ne02,
        args.ne10, args.ne12,
        args.ne0, args.ne1,
        uint(args.r2), uint(args.r3),
        tgpig, tiisg, sgitg);
}

// Lower-register-pressure variant: 2 rows per simdgroup, 4 simdgroups per
// threadgroup. Uses the same 8-row-per-TG dispatch as the standard ggml matvec
// but lowers register pressure for the fused gate+up path.
kernel void matvec_ggml_q4_0_gelu_mul_r2s4(
    device const char * W_gate [[buffer(0)]],
    device const char * W_up   [[buffer(1)]],
    device const char * x      [[buffer(2)]],
    device char * y            [[buffer(3)]],
    constant ggml_mul_mv_args& args [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    mul_vec_gelu_q_n_f32_impl<block_q4_0, 2, 4, 32>(
        W_gate, W_up, (device const float *)x, (device float *)y,
        args.ne00, args.ne01, args.ne02,
        args.ne10, args.ne12,
        args.ne0, args.ne1,
        uint(args.r2), uint(args.r3),
        tgpig, tiisg, sgitg);
}
