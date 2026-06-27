// ggml-metal kernel_flash_attn_ext_vec — decode Q4_0 KV (llama.cpp port).
// block_q4_0 + dequantize_q4_0_t4 come from ggml_mul_mv_q4.metal (concatenated first).
#include <metal_stdlib>
using namespace metal;

#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define N_SIMDWIDTH 32

struct ggml_flash_attn_args {
    int32_t ne01; int32_t ne02; int32_t ne03;
    uint64_t nb01; uint64_t nb02; uint64_t nb03;
    int32_t ne11; int32_t ne_12_2; int32_t ne_12_3;
    int32_t ns10; uint64_t nb11; uint64_t nb12; uint64_t nb13;
    int32_t ns20; uint64_t nb21; uint64_t nb22; uint64_t nb23;
    int32_t ne1; int32_t ne2; int32_t ne3;
    float scale;
    float max_bias; float m0; float m1; int32_t n_head_log2; float logit_softcap;
};

template<short DK, short DV, short NE_FA, short NSG_FA, short NS10_FA, short NS20_FA>
void flash_attn_ext_vec_impl(
    constant ggml_flash_attn_args & args,
    device const char * q,
    device const char * k,
    device const char * v,
    device char * dst,
    threadgroup half * shmem_f16,
    uint3 tgpig,
    ushort tiisg,
    ushort sgitg) {
#define NWG 1
#define NSG NSG_FA
#define NS10 NS10_FA
#define NS20 NS20_FA

#define NE NE_FA
#define C 32
#define nl_k 8
#define nl_v 8
typedef half4 q4_t;
typedef half4 k4_t;
typedef half4 v4_t;
typedef float qk_t;
typedef float s_t;
typedef float s4_t;
typedef float4 o4_t;
typedef block_q4_0 kd4_t;
typedef block_q4_0 vd4_t;
 static_assert(DK % 32 == 0, "DK must be divisible by 32");
 static_assert(DV % 32 == 0, "DV must be divisible by 32");

 const short iwg = tgpig[2]%NWG;

 const ushort iq3 = tgpig[2]/NWG;
 const ushort iq2 = tgpig[1];
 const ushort iq1 = tgpig[0];

 constexpr short DK4 = DK/4;
 constexpr short DV4 = DV/4;

 constexpr short PK = PAD2(DK, 128);
 constexpr short PK4 = PK/4;

 constexpr short PV = PAD2(DV, 128);
 constexpr short PV4 = PV/4;

 constexpr short NW = N_SIMDWIDTH;
 constexpr short NL = NW/NE;
 constexpr short SH = 4*C;

 static_assert(DK4 % NL == 0, "DK4 must be divisible by NL");
 static_assert(DV4 % NL == 0, "DV4 must be divisible by NL");

 threadgroup q4_t * sq4 = (threadgroup q4_t *) (shmem_f16 + 0*PK);
 threadgroup s_t * ss = (threadgroup s_t *) (shmem_f16 + sgitg*SH + NSG*PK);
 threadgroup s4_t * ss4 = (threadgroup s4_t *) (shmem_f16 + sgitg*SH + NSG*PK);
 threadgroup o4_t * so4 = (threadgroup o4_t *) (shmem_f16 + 2*sgitg*PV + NSG*PK + NSG*SH);

 so4 += tiisg;

 {
 q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

 const short ikv2 = iq2/(args.ne02/args.ne_12_2);
 const short ikv3 = iq3/(args.ne03/args.ne_12_3);

 k += ikv2*args.nb12 + ikv3*args.nb13;
 v += ikv2*args.nb22 + ikv3*args.nb23;
 }

 device const float4 * q4 = (device const float4 *) ((device const char *) q);

 if (iq1 < args.ne01) {
 for (short i = tiisg; i < PK4; i += NW) {
 if (i < DK4) {
 sq4[i] = (q4_t) q4[i];
 } else {
 sq4[i] = (q4_t) 0.0f;
 }
 }
 }

 for (short i = 0; i < DV4/NL; ++i) {
 so4[i*NL] = (o4_t) 0.0f;
 }

 for (short i = tiisg; i < SH/4; i += NW) {
 ss4[i] = (s4_t) 0.0f;
 }

 threadgroup_barrier(mem_flags::mem_threadgroup);

 {
 float S = 0.0f;
 float M = -FLT_MAX/2;

 const short tx = tiisg%NL;
 const short ty = tiisg/NL;

 for (int ic0 = iwg*NSG + sgitg; ; ic0 += NWG*NSG) {
 int ic = ic0*C;
 if (ic >= args.ne11) {
 break;
 }

 {
 device const k4_t * pk4 = (device const k4_t *) (k + ic*args.nb11);
 threadgroup const q4_t * pq4 = sq4;

 pk4 += ty*NS10/4 + tx;
 pq4 += tx;

 qk_t mqk[C/NE] = { [ 0 ... C/NE - 1] = 0.0f };

 FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
 device const kd4_t * pk = (device const kd4_t *) (k + ((ic + NE*cc + ty)*args.nb11));

 k4_t mk;

 FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
 const short i = ii*NL + tx;

 dequantize_q4_0_t4<half4>(pk + i/nl_k, i%nl_k, mk);

 mqk[cc] += dot((float4) mk, (float4) sq4[i]);
 }

 if (NE == 1) {
 mqk[cc] = simd_sum(mqk[cc]);
 } else {
 if (NE <= 1) {
 mqk[cc] += simd_shuffle_down(mqk[cc], 16);
 }
 if (NE <= 2) {
 mqk[cc] += simd_shuffle_down(mqk[cc], 8);
 }
 if (NE <= 4) {
 mqk[cc] += simd_shuffle_down(mqk[cc], 4);
 }
 if (NE <= 8) {
 mqk[cc] += simd_shuffle_down(mqk[cc], 2);
 }
 if (NE <= 16) {
 mqk[cc] += simd_shuffle_down(mqk[cc], 1);
 }

 mqk[cc] = simd_shuffle(mqk[cc], NL*ty);
 }
 }

 {
 if (ic + NE*tx + ty < args.ne11) {
 ss[NE*tx + ty] = mqk[tx] * args.scale;
 } else {
 ss[NE*tx + ty] = -FLT_MAX / 2.0f;
 }
 }
 }

 simdgroup_barrier(mem_flags::mem_threadgroup);

 {
 const float m = M;
 const float s = ss[tiisg];

 M = simd_max(max(M, s));

 const float ms = exp(m - M);
 const float vs = exp(s - M);

 S = S*ms + simd_sum(vs);

 ss[tiisg] = vs;

 if ((DV4/NL % NW == 0) || ty == 0) {
 FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
 so4[ii*NL] *= ms;
 }
 }
 }

 simdgroup_barrier(mem_flags::mem_threadgroup);

 {
 o4_t lo[DV4/NL];
 FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
 lo[ii] = 0.0f;
 }

 FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
 device const vd4_t * pv4 = (device const vd4_t *) (v + ((ic + NE*cc + ty)*args.nb21));

 FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
 const short i = ii*NL + tx;

 v4_t mv;
 dequantize_q4_0_t4<half4>(pv4 + i/nl_v, i%nl_v, mv);

 lo[ii] += o4_t(float4(mv)*float4(ss[NE*cc + ty]));
 }
 }

 FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
 if (NE > 1) {
 lo[ii][0] += simd_shuffle_down(lo[ii][0], 16);
 lo[ii][1] += simd_shuffle_down(lo[ii][1], 16);
 lo[ii][2] += simd_shuffle_down(lo[ii][2], 16);
 lo[ii][3] += simd_shuffle_down(lo[ii][3], 16);
 }

 if (NE > 2) {
 lo[ii][0] += simd_shuffle_down(lo[ii][0], 8);
 lo[ii][1] += simd_shuffle_down(lo[ii][1], 8);
 lo[ii][2] += simd_shuffle_down(lo[ii][2], 8);
 lo[ii][3] += simd_shuffle_down(lo[ii][3], 8);
 }

 if (NE > 4) {
 lo[ii][0] += simd_shuffle_down(lo[ii][0], 4);
 lo[ii][1] += simd_shuffle_down(lo[ii][1], 4);
 lo[ii][2] += simd_shuffle_down(lo[ii][2], 4);
 lo[ii][3] += simd_shuffle_down(lo[ii][3], 4);
 }

 if (NE > 8) {
 lo[ii][0] += simd_shuffle_down(lo[ii][0], 2);
 lo[ii][1] += simd_shuffle_down(lo[ii][1], 2);
 lo[ii][2] += simd_shuffle_down(lo[ii][2], 2);
 lo[ii][3] += simd_shuffle_down(lo[ii][3], 2);
 }

 if (NE > 16) {
 lo[ii][0] += simd_shuffle_down(lo[ii][0], 1);
 lo[ii][1] += simd_shuffle_down(lo[ii][1], 1);
 lo[ii][2] += simd_shuffle_down(lo[ii][2], 1);
 lo[ii][3] += simd_shuffle_down(lo[ii][3], 1);
 }
 }

 if ((DV4/NL % NW == 0) || ty == 0) {
 FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
 so4[ii*NL] += lo[ii];
 }
 }
 }
 }

 if (tiisg == 0) {
 ss[0] = (s_t) S;
 ss[1] = (s_t) M;
 }
 }

 so4 -= tiisg;

 threadgroup_barrier(mem_flags::mem_threadgroup);

 for (short r = NSG/2; r > 0; r >>= 1) {
 if (sgitg < r) {
 const float S0 = ss[ 0];
 const float S1 = ss[r*(SH/2) + 0];

 const float M0 = ss[ 1];
 const float M1 = ss[r*(SH/2) + 1];

 const float M = max(M0, M1);

 const float ms0 = exp(M0 - M);
 const float ms1 = exp(M1 - M);

 const float S = S0*ms0 + S1*ms1;

 if (tiisg == 0) {
 ss[0] = S;
 ss[1] = M;
 }

 for (short i = tiisg; i < DV4; i += NW) {
 so4[i] = so4[i]*ms0 + so4[i + r*PV4]*ms1;
 }
 }

 threadgroup_barrier(mem_flags::mem_threadgroup);
 }

 if (sgitg == 0) {
 const int64_t nrows = args.ne3*args.ne2*args.ne1;
 const int64_t rid = iq3*args.ne2*args.ne1 + iq2 + iq1*args.ne1;

 device float4 * dst4 = (device float4 *) dst;

 const float S = NWG == 1 ? (ss[0] == 0.0f ? 0.0f : 1.0f/ss[0]) : 1.0f;

 for (short i = tiisg; i < DV4; i += NW) {
 dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) so4[i]*S;
 }
 }

#undef NWG
#undef NSG
#undef NS10
#undef NS20
#undef NE
#undef C
#undef nl_k
#undef nl_v
}

kernel void flash_attn_ggml_q4_0_h256(
    constant ggml_flash_attn_args & args [[buffer(0)]],
    device const char * q [[buffer(1)]],
    device const char * k [[buffer(2)]],
    device const char * v [[buffer(3)]],
    device char * dst [[buffer(4)]],
    threadgroup half * shmem_f16 [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    flash_attn_ext_vec_impl<256, 256, 1, 1, 8, 8>(args, q, k, v, dst, shmem_f16, tgpig, tiisg, sgitg);
}

kernel void flash_attn_ggml_q4_0_h128(
    constant ggml_flash_attn_args & args [[buffer(0)]],
    device const char * q [[buffer(1)]],
    device const char * k [[buffer(2)]],
    device const char * v [[buffer(3)]],
    device char * dst [[buffer(4)]],
    threadgroup half * shmem_f16 [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    flash_attn_ext_vec_impl<128, 128, 1, 1, 4, 4>(args, q, k, v, dst, shmem_f16, tgpig, tiisg, sgitg);
}

kernel void flash_attn_ggml_q4_0_h512(
    constant ggml_flash_attn_args & args [[buffer(0)]],
    device const char * q [[buffer(1)]],
    device const char * k [[buffer(2)]],
    device const char * v [[buffer(3)]],
    device char * dst [[buffer(4)]],
    threadgroup half * shmem_f16 [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    flash_attn_ext_vec_impl<512, 512, 1, 1, 16, 16>(args, q, k, v, dst, shmem_f16, tgpig, tiisg, sgitg);
}
