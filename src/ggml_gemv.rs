//! ggml-metal Q4_0 GEMV dispatch helpers (decode matvec + mul_mv_ext variants).
//!
//! Weights use standard GGUF Q4_0 block layout: 18 bytes / 32 weights.

#![allow(dead_code)]

/// ggml_metal_kargs_mul_mv — must match `ggml_mul_mv_args` in ggml_mul_mv_q4.metal.
#[repr(C)]
pub struct GgmlMulMvArgs {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne10: i32,
    pub ne11: i32,
    pub ne12: i32,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub nr0: i32,
    pub r2: i16,
    pub r3: i16,
}

/// ggml_metal_kargs_mul_mv_ext — must match `ggml_mul_mv_ext_args` in ggml_mul_mv_q4.metal.
#[repr(C)]
pub struct GgmlMulMvExtArgs {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne10: i32,
    pub ne11: i32,
    pub ne12: i32,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub r2: i16,
    pub r3: i16,
    pub nsg: i16,
    pub nxpsg: i16,
}

pub const GGML_NR0_Q4_0: u32 = 4;
pub const GGML_NSG_Q4_0: u32 = 2;

/// Which ggml matvec kernel to launch for decode (batch-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlMatvecKind {
    /// `kernel_mul_mv_q4_0_f32` — llama.cpp decode default (ne11=1).
    MulMv,
    /// `kernel_mul_mv_ext_q4_0_f32_r1_4` with runtime nxpsg.
    ExtNx4,
    ExtNx8,
    ExtNx16,
}

impl GgmlMatvecKind {
    pub fn metal_entry(self) -> &'static str {
        match self {
            GgmlMatvecKind::MulMv => "matvec_ggml_q4_0",
            GgmlMatvecKind::ExtNx4 => "matvec_ggml_ext_q4_nx4_r4",
            GgmlMatvecKind::ExtNx8 => "matvec_ggml_ext_q4_nx8_r4",
            GgmlMatvecKind::ExtNx16 => "matvec_ggml_ext_q4_nx16_r4",
        }
    }

    /// Pick kernel matching ggml-metal heuristics for decode (ne11=1).
    pub fn pick_decode(k: u32) -> Self {
        // mul_mv_ext is for ne11 in [2,8]; decode uses mul_mv_q4_0.
        let _ = k;
        GgmlMatvecKind::MulMv
    }

    /// nxpsg selection from ggml-metal-ops (for ext / bench).
    pub fn pick_ext_nxpsg(k: u32, batch: u32) -> Self {
        if k % 256 == 0 && batch < 3 {
            GgmlMatvecKind::ExtNx16
        } else if k % 128 == 0 {
            GgmlMatvecKind::ExtNx8
        } else {
            GgmlMatvecKind::ExtNx4
        }
    }
}

fn row_bytes(k: u32) -> u64 {
    (k as u64 / 32) * 18
}

pub fn mul_mv_args(m: u32, k: u32) -> GgmlMulMvArgs {
    let rb = row_bytes(k);
    GgmlMulMvArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 18,
        nb01: rb,
        nb02: rb * m as u64,
        nb03: 0,
        ne10: k as i32,
        ne11: 1,
        ne12: 1,
        nb10: 4,
        nb11: (k as u64) * 4,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: 1,
        nr0: GGML_NR0_Q4_0 as i32,
        r2: 1,
        r3: 1,
    }
}

pub fn mul_mv_ext_args(m: u32, k: u32, batch: u32, kind: GgmlMatvecKind) -> GgmlMulMvExtArgs {
    let rb = row_bytes(k);
    let nxpsg = match kind {
        GgmlMatvecKind::ExtNx16 => 16,
        GgmlMatvecKind::ExtNx8 => 8,
        GgmlMatvecKind::ExtNx4 => 4,
        GgmlMatvecKind::MulMv => 8,
    };
    GgmlMulMvExtArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 18,
        nb01: rb,
        nb02: rb * m as u64,
        nb03: 0,
        ne10: k as i32,
        ne11: batch as i32,
        ne12: 1,
        nb10: 16,
        nb11: (k as u64) * 4,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: batch as i32,
        r2: 1,
        r3: 1,
        nsg: GGML_NSG_Q4_0 as i16,
        nxpsg: nxpsg as i16,
    }
}

/// Threadgroups for ggml mul_mv Q4_0 decode dispatch.
pub fn mul_mv_dispatch(m: u32, batch: u32) -> (u64, u64, u64, u64, u64) {
    let rows_per_tg = GGML_NR0_Q4_0 * GGML_NSG_Q4_0;
    let tg_x = ((m + rows_per_tg - 1) / rows_per_tg) as u64;
    let tg_y = batch as u64;
    (tg_x, tg_y, 1, 32, GGML_NSG_Q4_0 as u64)
}

/// K-quant matvec tiling: KQ_NSG simdgroups/threadgroup, KQ_NR0 rows each.
/// Must match `KQ_NSG`/`KQ_NR0` in ggml_mul_mv_q4.metal.
pub const KQ_NSG: u32 = 2;
pub const KQ_NR0: u32 = 4;

/// Args for K-quant matvec (`matvec_ggml_q4_K` / `matvec_ggml_q6_K`).
/// Byte strides match ggml-metal: `nb01` is the row stride in bytes.
/// `block_bytes` is 144 for Q4_K, 210 for Q6_K. `batch` is 1 for decode.
pub fn mul_mv_args_k(m: u32, k: u32, batch: u32, block_bytes: u64) -> GgmlMulMvArgs {
    let nb01 = (k as u64 / 256) * block_bytes;
    GgmlMulMvArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: block_bytes,
        nb01,
        nb02: nb01 * m as u64,
        nb03: 0,
        ne10: k as i32,
        ne11: batch as i32,
        ne12: 1,
        nb10: 4,
        nb11: (k as u64) * 4,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: batch as i32,
        nr0: KQ_NR0 as i32,
        r2: 1,
        r3: 1,
    }
}

pub const Q4_K_BLOCK_BYTES: u64 = 144;
pub const Q6_K_BLOCK_BYTES: u64 = 210;

/// Threadgroups/threads for the K-quant matvec: ceil(m / (KQ_NSG*KQ_NR0))
/// groups in x, `batch` in y; each group is 32 threads × KQ_NSG simdgroups,
/// and each simdgroup computes KQ_NR0 rows.
pub fn mul_mv_k_dispatch(m: u32, batch: u32) -> (u64, u64, u64, u64, u64) {
    let rows_per_tg = KQ_NSG * KQ_NR0;
    let tg_x = ((m + rows_per_tg - 1) / rows_per_tg) as u64;
    (tg_x, batch as u64, 1, 32, KQ_NSG as u64)
}

/// Threadgroups for fused K-quant pre-attn RMSNorm + Q/K/V matvec (decode).
pub fn kquant_fused_qkv_dispatch(m_q: u32, m_kv: u32, has_kv: bool) -> (u64, u64, u64) {
    let rows_per_tg = KQ_NSG * KQ_NR0;
    let q_tgs = (m_q + rows_per_tg - 1) / rows_per_tg;
    let kv_tgs = if has_kv {
        (m_kv + rows_per_tg - 1) / rows_per_tg
    } else {
        0
    };
    let total = q_tgs + 2 * kv_tgs;
    (total as u64, 32, KQ_NSG as u64)
}

/// Threadgroups for mul_mv_ext (r1ptg=4, nsg=2).
pub fn mul_mv_ext_dispatch(m: u32, batch: u32, nxpsg: i16) -> (u64, u64, u64, u64, u64) {
    let nypsg = 32 / nxpsg as u32;
    let r0ptg = nypsg * GGML_NSG_Q4_0;
    let r1ptg = 4u32;
    let tg_x = ((m + r0ptg - 1) / r0ptg) as u64;
    let tg_y = ((batch + r1ptg - 1) / r1ptg) as u64;
    (tg_x, tg_y, 1, 32, GGML_NSG_Q4_0 as u64)
}

// ─── K-quant small-batch ext matvec (MTP verify, batch 2..8) ────────────────

/// nxpsg for the K-quant ext kernels (K dims are all %128==0 → 8, per llama.cpp).
pub const MV_EXT_KQ_NXPSG: u32 = 8;
pub const MV_EXT_KQ_NSG: u32 = 2;

/// Simdgroups per threadgroup for the K-quant ext matvec. llama.cpp uses 2, but
/// on M1 Pro the small-batch (MTP verify) matvec is latency-bound: more
/// simdgroups per TG hide Q4_K/Q6_K dequant load latency. Tunable via
/// `MV_EXT_NSG` for sweeps; the kernel indexes rows by `args.nsg`, so the
/// dispatch and args must agree.
pub fn mv_ext_kq_nsg() -> u32 {
    static NSG: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *NSG.get_or_init(|| {
        std::env::var("MV_EXT_NSG")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|&n| n >= 1 && n <= 8 && 32 % MV_EXT_KQ_NXPSG == 0)
            .unwrap_or(MV_EXT_KQ_NSG)
    })
}

/// llama.cpp r1ptg selection by ne11 (src1 rows per threadgroup).
pub fn mv_ext_kq_r1ptg(batch: u32) -> u32 {
    match batch {
        2 => 2,
        3 | 6 => 3,
        5 => 5,
        _ => 4,
    }
}

/// Args for the K-quant ext matvec (`matvec_ggml_ext_q{4,6}K_nx8_r*`).
pub fn mul_mv_ext_args_k(m: u32, k: u32, batch: u32, block_bytes: u64) -> GgmlMulMvExtArgs {
    let nb01 = (k as u64 / 256) * block_bytes;
    GgmlMulMvExtArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: block_bytes,
        nb01,
        nb02: nb01 * m as u64,
        nb03: 0,
        ne10: k as i32,
        ne11: batch as i32,
        ne12: 1,
        nb10: 16,
        nb11: (k as u64) * 4,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: batch as i32,
        r2: 1,
        r3: 1,
        nsg: mv_ext_kq_nsg() as i16,
        nxpsg: MV_EXT_KQ_NXPSG as i16,
    }
}

/// Threadgroups for the K-quant ext matvec.
pub fn mul_mv_ext_k_dispatch(m: u32, batch: u32, r1ptg: u32) -> (u64, u64, u64, u64, u64) {
    let nsg = mv_ext_kq_nsg();
    let nypsg = 32 / MV_EXT_KQ_NXPSG;
    let r0ptg = nypsg * nsg;
    let tg_x = ((m + r0ptg - 1) / r0ptg) as u64;
    let tg_y = ((batch + r1ptg - 1) / r1ptg) as u64;
    (tg_x, tg_y, 1, 32, nsg as u64)
}

// ─── Q4_K matrix-matrix (prefill) ───────────────────────────────────────────

/// ggml_metal_kargs_mul_mm — must match `ggml_mul_mm_args` in ggml_mul_mm_q4.metal.
#[repr(C)]
pub struct GgmlMulMmArgs {
    pub ne00: i32,
    pub ne02: i32,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne12: i32,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub r2: i16,
    pub r3: i16,
}

/// llama.cpp `ne11_mm_min`: use mul_mm when batch (seq_len) is above this.
pub const MUL_MM_MIN_SEQ: u32 = 8;

/// Runtime override (`MUL_MM_MIN_SEQ`), e.g. lower it so small MTP verify
/// batches use mul_mm (weights read once for all rows) instead of per-row gemv.
fn mul_mm_min_seq() -> u32 {
    static MIN_SEQ: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *MIN_SEQ.get_or_init(|| {
        std::env::var("MUL_MM_MIN_SEQ")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(MUL_MM_MIN_SEQ)
    })
}

pub const MUL_MM_NR0: u32 = 64;
pub const MUL_MM_NR1: u32 = 32;
pub const MUL_MM_NSG: u32 = 4;
/// llama uses 6144 when bc_out=false (sa 4096 + sb 2048); 8192 when bc_out=true.
pub const MUL_MM_SMEM: u64 = 6144;
pub const MUL_MM_SMEM_BC_OUT: u64 = 8192;

/// Prefer simdgroup matmul over matvec for prefill when seq is long enough
/// and K is aligned (llama.cpp: ne00 >= 64 && ne11 > 8).
pub fn should_use_mul_mm(k: u32, seq_len: u32) -> bool {
    k >= 64 && k % 32 == 0 && seq_len > mul_mm_min_seq()
}

pub fn mul_mm_args_k(m: u32, k: u32, seq_len: u32, block_bytes: u64) -> GgmlMulMmArgs {
    let nb01 = (k as u64 / 256) * block_bytes;
    GgmlMulMmArgs {
        ne00: k as i32,
        ne02: 1,
        nb01,
        nb02: nb01 * m as u64,
        nb03: 0,
        ne12: 1,
        nb10: 4,
        nb11: (k as u64) * 4,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: seq_len as i32,
        r2: 1,
        r3: 1,
    }
}

/// Same as `mul_mm_args_k` but src1 is f16 (nb10=2).
pub fn mul_mm_args_k_f16(m: u32, k: u32, seq_len: u32, block_bytes: u64) -> GgmlMulMmArgs {
    let mut args = mul_mm_args_k(m, k, seq_len, block_bytes);
    args.nb10 = 2;
    args.nb11 = (k as u64) * 2;
    args
}

/// Dense f16 weight rows: `nb01 = K * sizeof(half)`.
pub fn mul_mm_args_f16(m: u32, k: u32, seq_len: u32) -> GgmlMulMmArgs {
    let nb01 = (k as u64) * 2;
    GgmlMulMmArgs {
        ne00: k as i32,
        ne02: 1,
        nb01,
        nb02: nb01 * m as u64,
        nb03: 0,
        ne12: 1,
        nb10: 4,
        nb11: (k as u64) * 4,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: seq_len as i32,
        r2: 1,
        r3: 1,
    }
}

pub fn mul_mm_args_q4_k(m: u32, k: u32, seq_len: u32) -> GgmlMulMmArgs {
    mul_mm_args_k(m, k, seq_len, Q4_K_BLOCK_BYTES)
}

pub fn mul_mm_args_q6_k(m: u32, k: u32, seq_len: u32) -> GgmlMulMmArgs {
    mul_mm_args_k(m, k, seq_len, Q6_K_BLOCK_BYTES)
}

/// Dispatch: (ceil(N/NR1), ceil(M/NR0), 1) threadgroups; 32×NSG threads.
pub fn mul_mm_dispatch(m: u32, seq_len: u32) -> (u64, u64, u64, u64, u64) {
    let tg_x = ((seq_len + MUL_MM_NR1 - 1) / MUL_MM_NR1) as u64;
    let tg_y = ((m + MUL_MM_NR0 - 1) / MUL_MM_NR0) as u64;
    (tg_x, tg_y, 1, 32, MUL_MM_NSG as u64)
}
