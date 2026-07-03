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
// Ported from ggml-org/llama.cpp ggml-metal.metal (N_R0=2, N_SG=2).
// Uses the same lane layout, super-block striping, and scale/min folding as
// kernel_mul_mv_q4_K_f32 / kernel_mul_mv_q6_K_f32.

#define QK_K 256

#define KQ_NSG 2
#define KQ_NR0 2

struct block_q4_K {
    half     d;
    half     dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};

struct block_q6_K {
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    half     d;
};

template<short nr0>
void mul_vec_q4_K_f32_impl(
        constant ggml_mul_mv_args& args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = KQ_NSG;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % uint(args.ne12);
    const uint i13 = im / uint(args.ne12);

    const uint64_t offset0 = first_row*args.nb01 + (i12/uint(args.r2))*args.nb02 + (i13/uint(args.r3))*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + i12*args.nb12 + i13*args.nb13;

    device const block_q4_K * x = (device const block_q4_K *)(src0 + offset0);
    device const float      * y = (device const float *)(src1 + offset1);

    float yl[16];
    float yh[16];

    float sumf[nr0]={0.f};

    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            #pragma unroll
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                  (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01/2;
            sc += args.nb01/2;
            dh += args.nb01/2;
        }

        y4 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

kernel void matvec_ggml_q4_K(
    device const char * W [[buffer(0)]],
    device const char * x [[buffer(1)]],
    device char       * y [[buffer(2)]],
    constant ggml_mul_mv_args& args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]]
) {
    mul_vec_q4_K_f32_impl<KQ_NR0>(args, W, x, y, tgpig, tiisg, sgitg);
}

template<short nr0>
void mul_vec_q6_K_f32_impl(
        constant ggml_mul_mv_args& args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = KQ_NSG;

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % uint(args.ne12);
    const uint i13 = im / uint(args.ne12);

    const uint64_t offset0 = first_row*args.nb01 + (i12/uint(args.r2))*args.nb02 + (i13/uint(args.r3))*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + i12*args.nb12 + i13*args.nb13;

    device const block_q6_K * x = (device const block_q6_K *)(src0 + offset0);
    device const float        * yy = (device const float *)(src1 + offset1);

    float sumf[nr0] = { 0.f };

    float yl[16];

    const short tid = tiisg/2;
    const short ix  = tiisg%2;
    const short ip  = tid/8;
    const short il  = tid%8;
    const short l0  = 4*il;
    const short is  = 8*ip + l0/16;

    const short y_offset   = 128*ip + l0;
    const short q_offset_l =  64*ip + l0;
    const short q_offset_h =  32*ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t  * sc = x[i].scales + is;
        device const half    * dh = &x[i].d;

        device const float * y = yy + i * QK_K + y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        for (short row = 0; row < nr0; ++row) {
            float4 sums = {0.f, 0.f, 0.f, 0.f};

            #pragma unroll
            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

            q1 += args.nb01;
            q2 += args.nb01;
            qh += args.nb01;
            sc += args.nb01;
            dh += args.nb01/2;
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

kernel void matvec_ggml_q6_K(
    device const char * W [[buffer(0)]],
    device const char * x [[buffer(1)]],
    device char       * y [[buffer(2)]],
    constant ggml_mul_mv_args& args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]]
) {
    mul_vec_q6_K_f32_impl<KQ_NR0>(args, W, x, y, tgpig, tiisg, sgitg);
}

// Fused gate+up+GeLU for Q4_K weights. Shared activation loads, dual weight streams.
template<short nr0>
void mul_vec_q4_K_gelu_f32_impl(
        constant ggml_mul_mv_args& args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = KQ_NSG;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % uint(args.ne12);
    const uint i13 = im / uint(args.ne12);

    const uint64_t offset0 = first_row*args.nb01 + (i12/uint(args.r2))*args.nb02 + (i13/uint(args.r3))*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + i12*args.nb12 + i13*args.nb13;

    device const block_q4_K * xg = (device const block_q4_K *)(src0_gate + offset0);
    device const block_q4_K * xu = (device const block_q4_K *)(src0_up   + offset0);
    device const float      * y  = (device const float *)(src1 + offset1);

    float yl[16];
    float yh[16];

    float sumf_gate[nr0]={0.f};
    float sumf_up[nr0]={0.f};

    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        device const uint16_t * sc_g = (device const uint16_t *)xg[ib].scales + iq;
        device const uint16_t * q1_g = (device const uint16_t *)xg[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh_g = &xg[ib].d;

        device const uint16_t * sc_u = (device const uint16_t *)xu[ib].scales + iq;
        device const uint16_t * q1_u = (device const uint16_t *)xu[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh_u = &xu[ib].d;

        for (short row = 0; row < nr0; row++) {
            float4 acc1_g = {0.f, 0.f, 0.f, 0.f};
            float4 acc2_g = {0.f, 0.f, 0.f, 0.f};
            float4 acc1_u = {0.f, 0.f, 0.f, 0.f};
            float4 acc2_u = {0.f, 0.f, 0.f, 0.f};

            sc16[0] = sc_g[0] & kmask1;
            sc16[1] = sc_g[2] & kmask1;
            sc16[2] = ((sc_g[4] >> 0) & kmask2) | ((sc_g[0] & kmask3) >> 2);
            sc16[3] = ((sc_g[4] >> 4) & kmask2) | ((sc_g[2] & kmask3) >> 2);

            device const uint16_t * q2_g = q1_g + 32;

            #pragma unroll
            for (short i = 0; i < 4; ++i) {
                acc1_g[0] += yl[2*i + 0] * (q1_g[i] & 0x000F);
                acc1_g[1] += yl[2*i + 1] * (q1_g[i] & 0x0F00);
                acc1_g[2] += yl[2*i + 8] * (q1_g[i] & 0x00F0);
                acc1_g[3] += yl[2*i + 9] * (q1_g[i] & 0xF000);
                acc2_g[0] += yh[2*i + 0] * (q2_g[i] & 0x000F);
                acc2_g[1] += yh[2*i + 1] * (q2_g[i] & 0x0F00);
                acc2_g[2] += yh[2*i + 8] * (q2_g[i] & 0x00F0);
                acc2_g[3] += yh[2*i + 9] * (q2_g[i] & 0xF000);
            }

            sumf_gate[row] += dh_g[0] * ((acc1_g[0] + 1.f/256.f * acc1_g[1]) * sc8[0] +
                                         (acc1_g[2] + 1.f/256.f * acc1_g[3]) * sc8[1] * 1.f/16.f +
                                         (acc2_g[0] + 1.f/256.f * acc2_g[1]) * sc8[4] +
                                         (acc2_g[2] + 1.f/256.f * acc2_g[3]) * sc8[5] * 1.f/16.f) -
                            dh_g[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            sc16[0] = sc_u[0] & kmask1;
            sc16[1] = sc_u[2] & kmask1;
            sc16[2] = ((sc_u[4] >> 0) & kmask2) | ((sc_u[0] & kmask3) >> 2);
            sc16[3] = ((sc_u[4] >> 4) & kmask2) | ((sc_u[2] & kmask3) >> 2);

            device const uint16_t * q2_u = q1_u + 32;

            #pragma unroll
            for (short i = 0; i < 4; ++i) {
                acc1_u[0] += yl[2*i + 0] * (q1_u[i] & 0x000F);
                acc1_u[1] += yl[2*i + 1] * (q1_u[i] & 0x0F00);
                acc1_u[2] += yl[2*i + 8] * (q1_u[i] & 0x00F0);
                acc1_u[3] += yl[2*i + 9] * (q1_u[i] & 0xF000);
                acc2_u[0] += yh[2*i + 0] * (q2_u[i] & 0x000F);
                acc2_u[1] += yh[2*i + 1] * (q2_u[i] & 0x0F00);
                acc2_u[2] += yh[2*i + 8] * (q2_u[i] & 0x00F0);
                acc2_u[3] += yh[2*i + 9] * (q2_u[i] & 0xF000);
            }

            sumf_up[row] += dh_u[0] * ((acc1_u[0] + 1.f/256.f * acc1_u[1]) * sc8[0] +
                                       (acc1_u[2] + 1.f/256.f * acc1_u[3]) * sc8[1] * 1.f/16.f +
                                       (acc2_u[0] + 1.f/256.f * acc2_u[1]) * sc8[4] +
                                       (acc2_u[2] + 1.f/256.f * acc2_u[3]) * sc8[5] * 1.f/16.f) -
                          dh_u[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1_g += args.nb01/2;
            sc_g += args.nb01/2;
            dh_g += args.nb01/2;
            q1_u += args.nb01/2;
            sc_u += args.nb01/2;
            dh_u += args.nb01/2;
        }

        y4 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float gate = simd_sum(sumf_gate[row]);
        const float up   = simd_sum(sumf_up[row]);
        const float gelu = gelu_pytorch_tanh_q4(gate) * up;
        if (tiisg == 0) {
            dst_f32[first_row + row] = gelu;
        }
    }
}

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
    mul_vec_q4_K_gelu_f32_impl<KQ_NR0>(args, W0, W1, x, y, tgpig, tiisg, sgitg);
}
