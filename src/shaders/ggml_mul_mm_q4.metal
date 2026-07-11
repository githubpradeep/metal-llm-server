// Q4_K matrix-matrix multiply for prefill (llama.cpp kernel_mul_mm_q4_K_f32).
//
// C[N, M] = B[N, K] @ A[M, K]^T  with A = Q4_K weights, B/C = f32.
// Layout: input [seq_len, K], output [seq_len, M], weights [M, K] Q4_K.
//
// Grid:     (ceil(N/32), ceil(M/64), 1)
// Threads:  128 = 4 simdgroups × 32
// Shared:   8192 bytes (sa half + sb float; partial-tile reuse)
//
// block_q4_K / QK_K come from ggml_mul_mv_q4.metal (concatenated first).
// Threshold: llama.cpp uses mul_mm when seq_len > 8 (ne11_mm_min).

#include <metal_stdlib>
using namespace metal;

static inline uchar2 mul_mm_get_scale_min_k4(int j, int k, device const uchar * q) {
    return j < 4
        ? uchar2{uchar(q[j + 0 + k] & 63), uchar(q[j + 4 + k] & 63)}
        : uchar2{
              uchar((q[j + 4 + k] & 0xF) | ((q[j - 4 + k] & 0xc0) >> 2)),
              uchar((q[j + 4 + k] >> 4) | ((q[j - 0 + k] & 0xc0) >> 2))};
}

// Same dequant as llama.cpp dequantize_q4_K (half4x4 path).
void mul_mm_dequantize_q4_K(device const block_q4_K * xb, short il, thread half4x4 & reg) {
    device const uchar * q = xb->qs;
    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = mul_mm_get_scale_min_k4(is, il / 2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = dl * (q[i] & mask) - ml;
    }
}

// Must match GgmlMulMmArgs in ggml_gemv.rs / ggml_metal_kargs_mul_mm.
struct ggml_mul_mm_args {
    int32_t ne00;   // K
    int32_t ne02;   // 1
    uint64_t nb01;  // weight row stride (bytes)
    uint64_t nb02;  // weight mat stride (bytes)
    uint64_t nb03;  // 0
    int32_t ne12;   // 1
    uint64_t nb10;  // sizeof(float)
    uint64_t nb11;  // K * sizeof(float)
    uint64_t nb12;  // 0
    uint64_t nb13;  // 0
    int32_t ne0;    // M
    int32_t ne1;    // N (seq_len)
    int16_t r2;     // 1
    int16_t r3;     // 1
};

kernel void mul_mm_q4_K_f32(
    constant ggml_mul_mm_args & args [[buffer(0)]],
    device const char * src0 [[buffer(1)]],  // Q4_K weights [M, K]
    device const char * src1 [[buffer(2)]],  // f32 input [N, K]
    device char * dst [[buffer(3)]],         // f32 output [N, M]
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    constexpr short nl = QK_K / 16;  // 16
    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK = 32;
    constexpr short NL0 = NK / 16;  // 2
    constexpr short NL1 = NK / 8;   // 4

    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup float * sb = (threadgroup float *)(shmem + 4096);

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const short nr0 = (args.ne0 - r0 < NR0) ? short(args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? short(args.ne1 - r1) : NR1;

    const short lr0 = ((short)tiitg / NL0) < nr0 ? ((short)tiitg / NL0) : short(nr0 - 1);
    const short lr1 = ((short)tiitg / NL1) < nr1 ? ((short)tiitg / NL1) : short(nr1 - 1);

    const short il0 = tiitg % NL0;
    short il = il0;

    // offset1 = il0/nl is always 0 for Q4_K (nl=16, il0∈{0,1})
    device const block_q4_K * x =
        (device const block_q4_K *)(src0 + args.nb01 * (r0 + lr0));

    const short iy = 8 * (tiitg % NL1);
    device const float * y =
        (device const float *)(src1 + args.nb11 * (r1 + lr1) + args.nb10 * iy);

    simdgroup_half8x8 ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        {
            half4x4 temp_a;
            mul_mm_dequantize_q4_K(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0) / 8;
                const short lx = (tiitg / NL0) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
            }
        }

        // Bounds-checked input load (handles partial K / edge tiles).
        for (short i = 0; i < 8; ++i) {
            const short sx = (tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short lx = i;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4 * sx + sy;
            *(sb + 64 * ib + 8 * ly + lx) =
                (loop_k + iy + i < args.ne00) ? *(y + i) : 0.f;
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x = (il < 2) ? x + (2 + nl - 1) / nl : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const float * lsmb = sb + 2 * 64 * (sgitg / 2);

        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1) {
        device float * C = (device float *)dst + (r0 + 32 * (sgitg & 1))
            + (r1 + 16 * (sgitg >> 1)) * args.ne0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * args.ne0 * (i / 4), args.ne0, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str =
            ((threadgroup float *)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float * D = (device float *)dst + r0 + (r1 + j) * args.ne0;
                threadgroup float * C = temp_str + (j * NR0);
                int i = 0;
                for (; i < nr0 / 4; i++) {
                    ((device float4 *)D)[i] = ((threadgroup float4 *)C)[i];
                }
                i *= 4;
                for (; i < nr0; i++) {
                    D[i] = C[i];
                }
            }
        }
    }
}

// ─── Q6_K mul_mm (same tiling; llama.cpp kernel_mul_mm_q6_K_f32) ────────────

void mul_mm_dequantize_q6_K(device const block_q6_K * xb, short il, thread half4x4 & reg) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    ql = ql + 32 * (il / 8) + 16 * ((il / 2) & 1) + 8 * (il & 1);
    qh = qh + 16 * (il / 8) + 8 * (il & 1);
    float sc = scales[(il % 2) + 2 * ((il / 2))];
    il = (il / 2) & 3;

    const uint32_t kmask1 = il > 1 ? (il > 2 ? 0xC0C0C0C0 : 0x30303030) : (il > 0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il > 1 ? 0xF0F0F0F0 : 0x0F0F0F0F;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il > 2 ? 2 : 0;
    const uint8_t shl_h = il > 1 ? 0 : (il > 0 ? 2 : 4);
    const uint8_t shr_l = il > 1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t low = (ql[2 * i] | (uint32_t)(ql[2 * i + 1] << 16)) & kmask2;
        const uint32_t high = (qh[2 * i] | (uint32_t)(qh[2 * i + 1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = dl0 * ((half)(q & 0xFF)) - ml;
        reg[i][1] = dl1 * ((float)(q & 0xFF00)) - ml;
        reg[i][2] = dl2 * ((float)(q & 0xFF0000)) - ml;
        reg[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
    }
}

kernel void mul_mm_q6_K_f32(
    constant ggml_mul_mm_args & args [[buffer(0)]],
    device const char * src0 [[buffer(1)]],
    device const char * src1 [[buffer(2)]],
    device char * dst [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    constexpr short nl = QK_K / 16;
    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK = 32;
    constexpr short NL0 = NK / 16;
    constexpr short NL1 = NK / 8;

    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup float * sb = (threadgroup float *)(shmem + 4096);

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const short nr0 = (args.ne0 - r0 < NR0) ? short(args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? short(args.ne1 - r1) : NR1;

    const short lr0 = ((short)tiitg / NL0) < nr0 ? ((short)tiitg / NL0) : short(nr0 - 1);
    const short lr1 = ((short)tiitg / NL1) < nr1 ? ((short)tiitg / NL1) : short(nr1 - 1);

    const short il0 = tiitg % NL0;
    short il = il0;

    device const block_q6_K * x =
        (device const block_q6_K *)(src0 + args.nb01 * (r0 + lr0));

    const short iy = 8 * (tiitg % NL1);
    device const float * y =
        (device const float *)(src1 + args.nb11 * (r1 + lr1) + args.nb10 * iy);

    simdgroup_half8x8 ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        {
            half4x4 temp_a;
            mul_mm_dequantize_q6_K(x, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0) / 8;
                const short lx = (tiitg / NL0) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
            }
        }

        for (short i = 0; i < 8; ++i) {
            const short sx = (tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short lx = i;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4 * sx + sy;
            *(sb + 64 * ib + 8 * ly + lx) =
                (loop_k + iy + i < args.ne00) ? *(y + i) : 0.f;
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x = (il < 2) ? x + (2 + nl - 1) / nl : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const float * lsmb = sb + 2 * 64 * (sgitg / 2);

        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1) {
        device float * C = (device float *)dst + (r0 + 32 * (sgitg & 1))
            + (r1 + 16 * (sgitg >> 1)) * args.ne0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * args.ne0 * (i / 4), args.ne0, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str =
            ((threadgroup float *)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float * D = (device float *)dst + r0 + (r1 + j) * args.ne0;
                threadgroup float * C = temp_str + (j * NR0);
                int i = 0;
                for (; i < nr0 / 4; i++) {
                    ((device float4 *)D)[i] = ((threadgroup float4 *)C)[i];
                }
                i *= 4;
                for (; i < nr0; i++) {
                    D[i] = C[i];
                }
            }
        }
    }
}

// ─── f16 src1 variants (llama.cpp kernel_mul_mm_q*_K_f16) ───────────────────
kernel void mul_mm_q4_K_f16(
    constant ggml_mul_mm_args & args [[buffer(0)]],
    device const char * src0 [[buffer(1)]],  // Q4_K weights [M, K]
    device const char * src1 [[buffer(2)]],  // f16 input [N, K]
    device char * dst [[buffer(3)]],         // f32 output [N, M]
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    constexpr short nl = QK_K / 16;  // 16
    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK = 32;
    constexpr short NL0 = NK / 16;  // 2
    constexpr short NL1 = NK / 8;   // 4

    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const short nr0 = (args.ne0 - r0 < NR0) ? short(args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? short(args.ne1 - r1) : NR1;

    const short lr0 = ((short)tiitg / NL0) < nr0 ? ((short)tiitg / NL0) : short(nr0 - 1);
    const short lr1 = ((short)tiitg / NL1) < nr1 ? ((short)tiitg / NL1) : short(nr1 - 1);

    const short il0 = tiitg % NL0;
    short il = il0;

    // offset1 = il0/nl is always 0 for Q4_K (nl=16, il0∈{0,1})
    device const block_q4_K * x =
        (device const block_q4_K *)(src0 + args.nb01 * (r0 + lr0));

    const short iy = 8 * (tiitg % NL1);
    device const half * y =
        (device const half *)(src1 + args.nb11 * (r1 + lr1) + args.nb10 * iy);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        {
            half4x4 temp_a;
            mul_mm_dequantize_q4_K(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0) / 8;
                const short lx = (tiitg / NL0) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
            }
        }

                // Keep B as half (llama.cpp q*_K_f16); K%32==0.
        {
            const short sx = (tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4 * sx + sy;
            *((threadgroup half2x4 *)(sb + 64 * ib + 8 * ly)) = *((device half2x4 *)y);
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x = (il < 2) ? x + (2 + nl - 1) / nl : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);

        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1) {
        device float * C = (device float *)dst + (r0 + 32 * (sgitg & 1))
            + (r1 + 16 * (sgitg >> 1)) * args.ne0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * args.ne0 * (i / 4), args.ne0, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str =
            ((threadgroup float *)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float * D = (device float *)dst + r0 + (r1 + j) * args.ne0;
                threadgroup float * C = temp_str + (j * NR0);
                int i = 0;
                for (; i < nr0 / 4; i++) {
                    ((device float4 *)D)[i] = ((threadgroup float4 *)C)[i];
                }
                i *= 4;
                for (; i < nr0; i++) {
                    D[i] = C[i];
                }
            }
        }
    }
}

kernel void mul_mm_q6_K_f16(
    constant ggml_mul_mm_args & args [[buffer(0)]],
    device const char * src0 [[buffer(1)]],
    device const char * src1 [[buffer(2)]],
    device char * dst [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    constexpr short nl = QK_K / 16;
    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK = 32;
    constexpr short NL0 = NK / 16;
    constexpr short NL1 = NK / 8;

    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const short nr0 = (args.ne0 - r0 < NR0) ? short(args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? short(args.ne1 - r1) : NR1;

    const short lr0 = ((short)tiitg / NL0) < nr0 ? ((short)tiitg / NL0) : short(nr0 - 1);
    const short lr1 = ((short)tiitg / NL1) < nr1 ? ((short)tiitg / NL1) : short(nr1 - 1);

    const short il0 = tiitg % NL0;
    short il = il0;

    device const block_q6_K * x =
        (device const block_q6_K *)(src0 + args.nb01 * (r0 + lr0));

    const short iy = 8 * (tiitg % NL1);
    device const half * y =
        (device const half *)(src1 + args.nb11 * (r1 + lr1) + args.nb10 * iy);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        {
            half4x4 temp_a;
            mul_mm_dequantize_q6_K(x, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0) / 8;
                const short lx = (tiitg / NL0) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
            }
        }        // Keep B as half (llama.cpp q*_K_f16); K%32==0.
        {
            const short sx = (tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4 * sx + sy;
            *((threadgroup half2x4 *)(sb + 64 * ib + 8 * ly)) = *((device half2x4 *)y);
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x = (il < 2) ? x + (2 + nl - 1) / nl : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);

        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1) {
        device float * C = (device float *)dst + (r0 + 32 * (sgitg & 1))
            + (r1 + 16 * (sgitg >> 1)) * args.ne0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * args.ne0 * (i / 4), args.ne0, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str =
            ((threadgroup float *)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float * D = (device float *)dst + r0 + (r1 + j) * args.ne0;
                threadgroup float * C = temp_str + (j * NR0);
                int i = 0;
                for (; i < nr0 / 4; i++) {
                    ((device float4 *)D)[i] = ((threadgroup float4 *)C)[i];
                }
                i *= 4;
                for (; i < nr0; i++) {
                    D[i] = C[i];
                }
            }
        }
    }
}

// ─── f16 src1 + f16 dst ───────────────────────────────────────────────────
kernel void mul_mm_q4_K_f16_f16(
    constant ggml_mul_mm_args & args [[buffer(0)]],
    device const char * src0 [[buffer(1)]],  // Q4_K weights [M, K]
    device const char * src1 [[buffer(2)]],  // f16 input [N, K]
    device char * dst [[buffer(3)]],         // f16 output [N, M]
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    constexpr short nl = QK_K / 16;  // 16
    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK = 32;
    constexpr short NL0 = NK / 16;  // 2
    constexpr short NL1 = NK / 8;   // 4

    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup float * sb = (threadgroup float *)(shmem + 4096);

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const short nr0 = (args.ne0 - r0 < NR0) ? short(args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? short(args.ne1 - r1) : NR1;

    const short lr0 = ((short)tiitg / NL0) < nr0 ? ((short)tiitg / NL0) : short(nr0 - 1);
    const short lr1 = ((short)tiitg / NL1) < nr1 ? ((short)tiitg / NL1) : short(nr1 - 1);

    const short il0 = tiitg % NL0;
    short il = il0;

    // offset1 = il0/nl is always 0 for Q4_K (nl=16, il0∈{0,1})
    device const block_q4_K * x =
        (device const block_q4_K *)(src0 + args.nb01 * (r0 + lr0));

    const short iy = 8 * (tiitg % NL1);
    device const half * y =
        (device const half *)(src1 + args.nb11 * (r1 + lr1) + args.nb10 * iy);

    simdgroup_half8x8 ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        {
            half4x4 temp_a;
            mul_mm_dequantize_q4_K(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0) / 8;
                const short lx = (tiitg / NL0) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
            }
        }

        // Bounds-checked input load (handles partial K / edge tiles).
        for (short i = 0; i < 8; ++i) {
            const short sx = (tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short lx = i;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4 * sx + sy;
            *(sb + 64 * ib + 8 * ly + lx) =
                (loop_k + iy + i < args.ne00) ? float(*(y + i)) : 0.f;
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x = (il < 2) ? x + (2 + nl - 1) / nl : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const float * lsmb = sb + 2 * 64 * (sgitg / 2);

        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    // Store accumulators as f16 (partial tiles via float TG scratch then convert).
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str =
        ((threadgroup float *)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += NR1) {
            device half * D = (device half *)dst + r0 + (r1 + j) * args.ne0;
            threadgroup float * C = temp_str + (j * NR0);
            int i = 0;
            for (; i + 3 < nr0; i += 4) {
                float4 v = *((threadgroup float4 *)(C + i));
                *((device half4 *)(D + i)) = half4(half(v.x), half(v.y), half(v.z), half(v.w));
            }
            for (; i < nr0; i++) {
                D[i] = half(C[i]);
            }
        }
    }
}

kernel void mul_mm_q6_K_f16_f16(
    constant ggml_mul_mm_args & args [[buffer(0)]],
    device const char * src0 [[buffer(1)]],
    device const char * src1 [[buffer(2)]],
    device char * dst [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    constexpr short nl = QK_K / 16;
    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK = 32;
    constexpr short NL0 = NK / 16;
    constexpr short NL1 = NK / 8;

    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup float * sb = (threadgroup float *)(shmem + 4096);

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const short nr0 = (args.ne0 - r0 < NR0) ? short(args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? short(args.ne1 - r1) : NR1;

    const short lr0 = ((short)tiitg / NL0) < nr0 ? ((short)tiitg / NL0) : short(nr0 - 1);
    const short lr1 = ((short)tiitg / NL1) < nr1 ? ((short)tiitg / NL1) : short(nr1 - 1);

    const short il0 = tiitg % NL0;
    short il = il0;

    device const block_q6_K * x =
        (device const block_q6_K *)(src0 + args.nb01 * (r0 + lr0));

    const short iy = 8 * (tiitg % NL1);
    device const half * y =
        (device const half *)(src1 + args.nb11 * (r1 + lr1) + args.nb10 * iy);

    simdgroup_half8x8 ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        {
            half4x4 temp_a;
            mul_mm_dequantize_q6_K(x, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0) / 8;
                const short lx = (tiitg / NL0) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
            }
        }

        for (short i = 0; i < 8; ++i) {
            const short sx = (tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short lx = i;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4 * sx + sy;
            *(sb + 64 * ib + 8 * ly + lx) =
                (loop_k + iy + i < args.ne00) ? float(*(y + i)) : 0.f;
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x = (il < 2) ? x + (2 + nl - 1) / nl : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const float * lsmb = sb + 2 * 64 * (sgitg / 2);

        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    // Store accumulators as f16 (partial tiles via float TG scratch then convert).
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str =
        ((threadgroup float *)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += NR1) {
            device half * D = (device half *)dst + r0 + (r1 + j) * args.ne0;
            threadgroup float * C = temp_str + (j * NR0);
            int i = 0;
            for (; i + 3 < nr0; i += 4) {
                float4 v = *((threadgroup float4 *)(C + i));
                *((device half4 *)(D + i)) = half4(half(v.x), half(v.y), half(v.z), half(v.w));
            }
            for (; i < nr0; i++) {
                D[i] = half(C[i]);
            }
        }
    }
}
