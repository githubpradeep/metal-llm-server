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

// ─── Q3_0 (3-bit symmetric) kernels ─────────────────────────────────────────

#define QK3_0 32

// Forward declaration — defined later in this file, used by Q3 gelu template.
inline float gelu_pytorch_tanh_q4(float x);

struct block_q3_0 {
    half    d;
    uint8_t qs_low[8];   // low 2 bits: 4 weights per byte
    uint8_t qs_high[4];  // high 1 bit: 8 weights per byte
};

inline float block_q3_n_dot_y(device const block_q3_0 * qb, thread const float * yl, int il) {
    float d = qb->d;
    // Process 16 weights: acc[0] for first 8 (yl[0..7]), acc[1] for second 8 (yl[8..15])
    float2 acc = 0.f;

    // il is 0 or 8 → selects which 16-weight subset (two 8-weight groups)
    // For il=0:  weights 0-7 → qs_low[0..1] + qs_high[0]; weights 16-23 → qs_low[4..5] + qs_high[2]
    // For il=8:  weights 8-15 → qs_low[2..3] + qs_high[1]; weights 24-31 → qs_low[6..7] + qs_high[3]
    device const uint8_t * ql0 = qb->qs_low + il/4;      // first group (weights il .. il+7)
    device const uint8_t * ql1 = qb->qs_low + il/4 + 4;  // second group (weights il+16 .. il+23)
    device const uint8_t * qh0 = qb->qs_high + il/8;     // first group high bits
    device const uint8_t * qh1 = qb->qs_high + il/8 + 2; // second group high bits

    for (int i = 0; i < 4; i++) {
        // First 4 weights of first group
        int low0  = (ql0[0] >> (2*i)) & 0x3;
        int high0 = (qh0[0] >> i) & 0x1;
        // Next 4 weights of first group
        int low1  = (ql0[1] >> (2*i)) & 0x3;
        int high1 = (qh0[0] >> (i+4)) & 0x1;
        int q0 = (low0  | (high0 << 2)) - 4;
        int q1 = (low1  | (high1 << 2)) - 4;
        acc[0] += yl[i]   * q0 + yl[i+4] * q1;

        // First 4 weights of second group
        int low2  = (ql1[0] >> (2*i)) & 0x3;
        int high2 = (qh1[0] >> i) & 0x1;
        // Next 4 weights of second group
        int low3  = (ql1[1] >> (2*i)) & 0x3;
        int high3 = (qh1[0] >> (i+4)) & 0x1;
        int q2 = (low2  | (high2 << 2)) - 4;
        int q3 = (low3  | (high3 << 2)) - 4;
        acc[1] += yl[i+8] * q2 + yl[i+12] * q3;
    }
    return d * (acc[0] + acc[1]);
}

template<typename block_q_type, short NR0, short NSG, short NW>
void mul_vec_q3_f32_impl(
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
    const int nb = ne00/QK3_0;

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

    device const float * yb = y + ix * QK3_0 + il;

    for (int ib = ix; ib < nb; ib += NW/2) {
        for (int i = 0; i < 8; i += 2) {
            yl[i+0] = yb[i+ 0];
            yl[i+1] = yb[i+ 1];
            yl[i+8] = yb[i+16];
            yl[i+9] = yb[i+17];
        }

        for (int row = 0; row < NR0; row++) {
            sumf[row] += block_q3_n_dot_y(x+ib+row*nb, yl, il);
        }

        yb += QK3_0 * 16;
    }

    for (int row = 0; row < NR0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < ne01) {
            dst[im*ne0*ne1 + r1*ne0 + first_row + row] = tot;
        }
    }
}

template<typename block_q_type, short NR0, short NSG, short NW>
void mul_vec_dual_q3_f32_impl(
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
    const int nb = ne00/QK3_0;

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

    device const float * yb = y + ix * QK3_0 + il;

    for (int ib = ix; ib < nb; ib += NW/2) {
        for (int i = 0; i < 8; i += 2) {
            yl[i+0] = yb[i+ 0];
            yl[i+1] = yb[i+ 1];
            yl[i+8] = yb[i+16];
            yl[i+9] = yb[i+17];
        }

        for (int row = 0; row < NR0; row++) {
            sumf_a[row] += block_q3_n_dot_y(x_a+ib+row*nb, yl, il);
            sumf_b[row] += block_q3_n_dot_y(x_b+ib+row*nb, yl, il);
        }

        yb += QK3_0 * 16;
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

template<typename block_q_type, short NR0, short NSG, short NW>
void mul_vec_gelu_q3_f32_impl(
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
    const int nb = ne00/QK3_0;

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

    device const float * yb = y + ix * QK3_0 + il;

    for (int ib = ix; ib < nb; ib += NW/2) {
        for (int i = 0; i < 8; i += 2) {
            yl[i+0] = yb[i+ 0];
            yl[i+1] = yb[i+ 1];
            yl[i+8] = yb[i+16];
            yl[i+9] = yb[i+17];
        }

        for (int row = 0; row < NR0; row++) {
            sumf_gate[row] += block_q3_n_dot_y(x_gate+ib+row*nb, yl, il);
            sumf_up[row]   += block_q3_n_dot_y(x_up  +ib+row*nb, yl, il);
        }

        yb += QK3_0 * 16;
    }

    for (int row = 0; row < NR0; ++row) {
        const int row_idx = first_row + row;
        const bool valid = row_idx < ne01;
        const float gate = simd_sum(sumf_gate[row]);
        const float up   = simd_sum(sumf_up[row]);
        const float gelu = valid ? gelu_pytorch_tanh_q4(gate) * up : 0.0f;
        if (tiisg == 0 && valid) {
            const uint64_t dst_off = (uint64_t)im*ne0*ne1 + (uint64_t)r1*ne0 + (uint64_t)row_idx;
            dst[dst_off] = gelu;
        }
    }
}

// ─── Q3_0 kernel entry points ────────────────────────────────────────────────

kernel void matvec_ggml_q3_0(
    device const char * W [[buffer(0)]],
    device const char * x [[buffer(1)]],
    device char * y [[buffer(2)]],
    constant ggml_mul_mv_args& args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    mul_vec_q3_f32_impl<block_q3_0, 4, 2, 32>(
        W, (device const float *)x, (device float *)y,
        args.ne00, args.ne01, args.ne02,
        args.ne10, args.ne12,
        args.ne0, args.ne1,
        uint(args.r2), uint(args.r3),
        tgpig, tiisg, sgitg);
}

kernel void matvec_ggml_q3_0_dual(
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
    mul_vec_dual_q3_f32_impl<block_q3_0, 4, 2, 32>(
        W_gate, W_up, (device const float *)x,
        (device float *)gate, (device float *)up,
        args.ne00, args.ne01, args.ne02,
        args.ne10, args.ne12,
        args.ne0, args.ne1,
        uint(args.r2), uint(args.r3),
        tgpig, tiisg, sgitg);
}

kernel void matvec_ggml_q3_0_gelu_mul(
    device const char * W_gate [[buffer(0)]],
    device const char * W_up   [[buffer(1)]],
    device const char * x      [[buffer(2)]],
    device char * y            [[buffer(3)]],
    constant ggml_mul_mv_args& args [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    mul_vec_gelu_q3_f32_impl<block_q3_0, 4, 2, 32>(
        W_gate, W_up, (device const float *)x, (device float *)y,
        args.ne00, args.ne01, args.ne02,
        args.ne10, args.ne12,
        args.ne0, args.ne1,
        uint(args.r2), uint(args.r3),
        tgpig, tiisg, sgitg);
}

kernel void matvec_ggml_q3_0_gelu_mul_r2s4(
    device const char * W_gate [[buffer(0)]],
    device const char * W_up   [[buffer(1)]],
    device const char * x      [[buffer(2)]],
    device char * y            [[buffer(3)]],
    constant ggml_mul_mv_args& args [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    mul_vec_gelu_q3_f32_impl<block_q3_0, 2, 4, 32>(
        W_gate, W_up, (device const float *)x, (device float *)y,
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

// ───────────────────────── K-quant matvec (Q4_K / Q6_K) ─────────────────────
//
// Native super-block (QK_K=256) matvec for llama.cpp "_K" weights, used for
// community Q4_K_M GGUFs. Layout is intentionally simple and verifiable rather
// than maximally tuned: one simdgroup computes one output row; the 32 lanes
// stride over the row's super-blocks and a simd_sum reduces the partial dots.
// The batch dimension (tgpig.y in [0, ne11)) lets the SAME kernel serve both
// decode (ne11=1) and prefill (ne11=seq_len). x and y are f32.
//
// Dequant math is byte-for-byte the CPU reference in gguf.rs (get_scale_min_k4
// for Q4_K; the ql/qh/scales unpack for Q6_K).

#define QK_K 256

// K-quant matvec tiling: KQ_NSG simdgroups per threadgroup, each computing
// KQ_NR0 output rows. Rows/threadgroup = KQ_NSG * KQ_NR0. Must match
// `KQ_NSG`/`KQ_NR0` in ggml_gemv.rs.
#define KQ_NSG 2
#define KQ_NR0 2

struct block_q4_K {
    half     d;          // super-block scale for the 6-bit scales
    half     dmin;       // super-block scale for the 6-bit mins
    uint8_t  scales[12]; // 8 sub-block scales + mins, 6-bit packed
    uint8_t  qs[128];    // 4-bit quants
};

struct block_q6_K {
    uint8_t  ql[128];    // lower 4 bits
    uint8_t  qh[64];     // upper 2 bits
    int8_t   scales[16]; // 8-bit sub-block scales
    half     d;          // super-block scale
};

// 6-bit (scale, min) unpack — matches ggml get_scale_min_k4.
inline void kq_scale_min_k4(int j, device const uint8_t * q, thread uint8_t & d, thread uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

// Q4_K matvec: each simdgroup computes KQ_NR0 output rows. All 32 lanes
// cooperate on each 256-weight super-block (8 weights/lane), so every lane
// issues weight loads — keeping the (memory-bound) simdgroup busy even when nb
// is small. Lane layout: 4 lanes per Q4_K sub-block (8 sub-blocks of 32); lane
// t owns sub-block sb=t/4, columns col0=(t%4)*8 .. col0+7. The activation
// (yl) is loaded once per super-block and reused across the KQ_NR0 rows; the
// 8 weight nibbles are read as two 32-bit words instead of scalar bytes.
kernel void matvec_ggml_q4_K(
    device const char * W [[buffer(0)]],
    device const char * x [[buffer(1)]],
    device char       * y [[buffer(2)]],
    constant ggml_mul_mv_args& args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]]
) {
    const int nb        = args.ne00 / QK_K;       // super-blocks per row
    const int first_row = (int)(tgpig.x * KQ_NSG + sgitg) * KQ_NR0;
    if (first_row >= args.ne01) return;
    const int r1 = tgpig.y;                        // batch / token index

    device const float * yr = (device const float *) x + (uint) r1 * args.ne10;

    const int sb   = tiisg >> 2;            // sub-block 0..7
    const int col0 = (int)(tiisg & 3) * 8;  // 0, 8, 16, 24
    const int g    = sb >> 1;               // group 0..3 (64 elems each)
    const int hi   = sb & 1;                // 0 = low nibble, 1 = high nibble

    const int act_off = g * 64 + hi * 32 + col0;

    float sumf[KQ_NR0];
    for (int r = 0; r < KQ_NR0; ++r) sumf[r] = 0.f;

    for (int ib = 0; ib < nb; ++ib) {
        // Activation slice for this lane's 8 columns (shared across rows).
        device const float * yy = yr + ib * QK_K + act_off;
        float yl[8];
        for (int i = 0; i < 8; ++i) yl[i] = yy[i];

        for (int r = 0; r < KQ_NR0; ++r) {
            const int row = first_row + r;
            if (row >= args.ne01) break;
            device const block_q4_K * b = (device const block_q4_K *) W + (uint)row * nb + ib;
            const float d    = (float) b->d;
            const float dmin = (float) b->dmin;
            uint8_t scu, mnu;
            kq_scale_min_k4(sb, b->scales, scu, mnu);
            const float ds = d * scu;
            const float mn = dmin * mnu;
            // 8 nibbles = two aligned 32-bit words (col0 is a multiple of 8).
            device const uint * qw = (device const uint *)(b->qs + g * 32 + col0);
            const uint w0 = qw[0];
            const uint w1 = qw[1];
            float acc = 0.f;
            // Low half (bytes from w0)
            for (int i = 0; i < 4; ++i) {
                const uint byte = (w0 >> (i * 8)) & 0xFF;
                const float nib = hi == 0 ? (float)(byte & 0x0F) : (float)((byte >> 4) & 0x0F);
                acc += (ds * nib - mn) * yl[i];
            }
            // High half (bytes from w1)
            for (int i = 0; i < 4; ++i) {
                const uint byte = (w1 >> (i * 8)) & 0xFF;
                const float nib = hi == 0 ? (float)(byte & 0x0F) : (float)((byte >> 4) & 0x0F);
                acc += (ds * nib - mn) * yl[i + 4];
            }
            sumf[r] += acc;
        }
    }
    for (int r = 0; r < KQ_NR0; ++r) {
        const int row = first_row + r;
        const float tot = simd_sum(sumf[r]);
        if (tiisg == 0 && row < args.ne01) {
            ((device float *) y)[(uint) r1 * args.ne0 + row] = tot;
        }
    }
}

// Q6_K matvec: each simdgroup computes KQ_NR0 rows; all 32 lanes cooperate per
// super-block. Lane layout: 16 lanes per 128-element part; lane t owns part
// p=t/16 and the two l-rows l=(t%16)*2 (+1), each producing 4 outputs (offsets
// 0/32/64/96). The activation is loaded once per super-block and reused.
kernel void matvec_ggml_q6_K(
    device const char * W [[buffer(0)]],
    device const char * x [[buffer(1)]],
    device char       * y [[buffer(2)]],
    constant ggml_mul_mv_args& args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]]
) {
    const int nb        = args.ne00 / QK_K;
    const int first_row = (int)(tgpig.x * KQ_NSG + sgitg) * KQ_NR0;
    if (first_row >= args.ne01) return;
    const int r1 = tgpig.y;

    device const float * yr = (device const float *) x + (uint) r1 * args.ne10;

    const int p  = tiisg >> 4;              // part 0..1 (128 elems each)
    const int lp = (int)(tiisg & 15);       // 0..15 within part

    float sumf[KQ_NR0];
    for (int r = 0; r < KQ_NR0; ++r) sumf[r] = 0.f;

    for (int ib = 0; ib < nb; ++ib) {
        device const float * yy = yr + ib * QK_K + p * 128;
        // Activation for this lane's two l-rows × 4 output slots (shared).
        float yl[2][4];
        for (int li = 0; li < 2; ++li) {
            const int l = lp * 2 + li;
            yl[li][0] = yy[l];
            yl[li][1] = yy[l + 32];
            yl[li][2] = yy[l + 64];
            yl[li][3] = yy[l + 96];
        }
        for (int r = 0; r < KQ_NR0; ++r) {
            const int row = first_row + r;
            if (row >= args.ne01) break;
            device const block_q6_K * b = (device const block_q6_K *) W + (uint)row * nb + ib;
            const float d = (float) b->d;
            device const uint8_t * ql = b->ql + p * 64;
            device const uint8_t * qh = b->qh + p * 32;
            device const int8_t  * sc = b->scales + p * 8;
            // is is identical for li=0/1 (lp*2 is even, never straddles 16).
            const int is = (lp * 2) >> 4;
            const float dsc0 = d * (float) sc[is];
            const float dsc1 = d * (float) sc[is + 2];
            const float dsc2 = d * (float) sc[is + 4];
            const float dsc3 = d * (float) sc[is + 6];
            // Vectorized loads: the two li-rows are at consecutive bytes.
            const ushort qlA = *(device const ushort *)(ql + lp * 2);
            const ushort qlB = *(device const ushort *)(ql + lp * 2 + 32);
            const ushort qhW = *(device const ushort *)(qh + lp * 2);
            float acc = 0.f;
            for (int li = 0; li < 2; ++li) {
                const uint8_t lo  = (li == 0) ? (qlA & 0xFF) : (qlA >> 8);
                const uint8_t lo2 = (li == 0) ? (qlB & 0xFF) : (qlB >> 8);
                const uint8_t h   = (li == 0) ? (qhW & 0xFF) : (qhW >> 8);
                const int q1 = (int)((lo  & 0x0F) | (((h >> 0) & 3) << 4)) - 32;
                const int q2 = (int)((lo2 & 0x0F) | (((h >> 2) & 3) << 4)) - 32;
                const int q3 = (int)((lo  >> 4)   | (((h >> 4) & 3) << 4)) - 32;
                const int q4 = (int)((lo2 >> 4)   | (((h >> 6) & 3) << 4)) - 32;
                acc += dsc0 * (float) q1 * yl[li][0];
                acc += dsc1 * (float) q2 * yl[li][1];
                acc += dsc2 * (float) q3 * yl[li][2];
                acc += dsc3 * (float) q4 * yl[li][3];
            }
            sumf[r] += acc;
        }
    }
    for (int r = 0; r < KQ_NR0; ++r) {
        const int row = first_row + r;
        const float tot = simd_sum(sumf[r]);
        if (tiisg == 0 && row < args.ne01) {
            ((device float *) y)[(uint) r1 * args.ne0 + row] = tot;
        }
    }
}

// Fused gate+up+GeLU for Q4_K weights (Q4_K_M MLP gate/up are both Q4_K).
// One dispatch computes y[row] = GeLU(gate[row]·x) * (up[row]·x), reading the
// (RMSNorm'd) activation once and reusing it for both projections. Mirrors the
// Q4_0 `matvec_q4_dual_gelu` path. W0=gate, W1=up; both share the args shape.
kernel void matvec_ggml_q4_K_gelu_mul(
    device const char * W0 [[buffer(0)]],
    device const char * W1 [[buffer(1)]],
    device const char * x  [[buffer(2)]],
    device char       * y  [[buffer(3)]],
    constant ggml_mul_mv_args& args [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]]
) {
    const int nb        = args.ne00 / QK_K;
    const int first_row = (int)(tgpig.x * KQ_NSG + sgitg) * KQ_NR0;
    if (first_row >= args.ne01) return;
    const int r1 = tgpig.y;

    device const float * yr = (device const float *) x + (uint) r1 * args.ne10;

    const int sb   = tiisg >> 2;
    const int col0 = (int)(tiisg & 3) * 8;
    const int g    = sb >> 1;
    const int hi   = sb & 1;

    const int act_off = g * 64 + hi * 32 + col0;

    float sgate[KQ_NR0];
    float sup[KQ_NR0];
    for (int r = 0; r < KQ_NR0; ++r) { sgate[r] = 0.f; sup[r] = 0.f; }

    for (int ib = 0; ib < nb; ++ib) {
        device const float * yy = yr + ib * QK_K + act_off;
        float yl[8];
        for (int i = 0; i < 8; ++i) yl[i] = yy[i];

        for (int r = 0; r < KQ_NR0; ++r) {
            const int row = first_row + r;
            if (row >= args.ne01) break;
            device const block_q4_K * bg = (device const block_q4_K *) W0 + (uint)row * nb + ib;
            device const block_q4_K * bu = (device const block_q4_K *) W1 + (uint)row * nb + ib;
            uint8_t scu, mnu;
            kq_scale_min_k4(sb, bg->scales, scu, mnu);
            const float dsg = (float) bg->d * scu;
            const float mng = (float) bg->dmin * mnu;
            kq_scale_min_k4(sb, bu->scales, scu, mnu);
            const float dsu = (float) bu->d * scu;
            const float mnu_ = (float) bu->dmin * mnu;
            device const uint * qg = (device const uint *)(bg->qs + g * 32 + col0);
            device const uint * qu = (device const uint *)(bu->qs + g * 32 + col0);
            const uint g0 = qg[0], g1 = qg[1];
            const uint u0 = qu[0], u1 = qu[1];
            float ag = 0.f, au = 0.f;
            // Low half (bytes from g0/u0)
            for (int i = 0; i < 4; ++i) {
                const uint gb = (g0 >> (i * 8)) & 0xFF;
                const uint ub = (u0 >> (i * 8)) & 0xFF;
                const float gn = hi == 0 ? (float)(gb & 0x0F) : (float)((gb >> 4) & 0x0F);
                const float un = hi == 0 ? (float)(ub & 0x0F) : (float)((ub >> 4) & 0x0F);
                ag += (dsg * gn - mng) * yl[i];
                au += (dsu * un - mnu_) * yl[i];
            }
            // High half (bytes from g1/u1)
            for (int i = 0; i < 4; ++i) {
                const uint gb = (g1 >> (i * 8)) & 0xFF;
                const uint ub = (u1 >> (i * 8)) & 0xFF;
                const float gn = hi == 0 ? (float)(gb & 0x0F) : (float)((gb >> 4) & 0x0F);
                const float un = hi == 0 ? (float)(ub & 0x0F) : (float)((ub >> 4) & 0x0F);
                ag += (dsg * gn - mng) * yl[i + 4];
                au += (dsu * un - mnu_) * yl[i + 4];
            }
            sgate[r] += ag;
            sup[r]   += au;
        }
    }
    for (int r = 0; r < KQ_NR0; ++r) {
        const int row = first_row + r;
        const float gtot = simd_sum(sgate[r]);
        const float utot = simd_sum(sup[r]);
        if (tiisg == 0 && row < args.ne01) {
            ((device float *) y)[(uint) r1 * args.ne0 + row] =
                gelu_pytorch_tanh_q4(gtot) * utot;
        }
    }
}
