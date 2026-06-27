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

/// Output rows per threadgroup for the K-quant matvec (one row per simdgroup).
/// Must match `KQ_NSG` in ggml_mul_mv_q4.metal.
pub const KQ_NSG: u32 = 4;

/// Args for the native K-quant matvec (`matvec_ggml_q4_K` / `matvec_ggml_q6_K`).
/// The kernels only read ne00/ne01/ne10/ne0 + the batch (ne11 via tgpig.y), so
/// the byte-stride fields are left zero. `batch` is 1 for decode, seq_len for
/// prefill. `m` = output rows, `k` = reduction dim (multiple of 256).
pub fn mul_mv_args_k(m: u32, k: u32, batch: u32) -> GgmlMulMvArgs {
    GgmlMulMvArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 0,
        nb01: 0,
        nb02: 0,
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
        nr0: 1,
        r2: 1,
        r3: 1,
    }
}

/// Threadgroups/threads for the K-quant matvec: ceil(m/KQ_NSG) groups in x,
/// `batch` in y; each group is 32 threads × KQ_NSG simdgroups.
pub fn mul_mv_k_dispatch(m: u32, batch: u32) -> (u64, u64, u64, u64, u64) {
    let tg_x = ((m + KQ_NSG - 1) / KQ_NSG) as u64;
    (tg_x, batch as u64, 1, 32, KQ_NSG as u64)
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
