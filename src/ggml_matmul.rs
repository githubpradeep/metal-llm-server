//! ggml-metal batched matrix-matrix multiply (prefill) dispatch helpers.
//!
//! Computes Y = X @ W^T where W is [M, K] quantized and X is [S, K] f32 row-major.

#![allow(dead_code)]

/// Must match `ggml_mul_mm_args` in ggml_mul_mm.metal / ggml-metal-impl.h.
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

pub const Q4_K_BLOCK_BYTES: u64 = 144;
pub const Q4_0_BLOCK_BYTES: u64 = 18;

/// Tile geometry for legacy ggml `kernel_mul_mm` (simdgroup matmul path).
pub const MM_NR0: u32 = 64;
pub const MM_NR1: u32 = 32;
pub const MM_NSG: u32 = 2;
pub const MM_THREADS: u32 = 32;
pub const MM_SMEM_BYTES: u64 = 8192;

/// Use true GEMM when batch > 1 and K is large enough (matches ggml-metal heuristic).
pub fn prefill_use_mul_mm(seq_len: u32, k: u32) -> bool {
    seq_len > 1 && k >= 64
}

pub fn mul_mm_disabled() -> bool {
    matches!(
        std::env::var("PREFILL_MATMUL").as_deref(),
        Ok("0") | Ok("false") | Ok("legacy") | Ok("LEGACY")
    )
}

pub fn mul_mm_args(m: u32, k: u32, seq_len: u32, block_bytes: u64) -> GgmlMulMmArgs {
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

pub fn mul_mm_args_q4_0(m: u32, k: u32, seq_len: u32) -> GgmlMulMmArgs {
    let nb01 = (k as u64 / 32) * Q4_0_BLOCK_BYTES;
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

/// Threadgroups for mul_mm: X over seq_len (NR1), Y over M (NR0).
pub fn mul_mm_dispatch(m: u32, seq_len: u32) -> (u64, u64, u64, u64, u64) {
    let tg_x = ((seq_len + MM_NR1 - 1) / MM_NR1) as u64;
    let tg_y = ((m + MM_NR0 - 1) / MM_NR0) as u64;
    (tg_x, tg_y, 1, MM_THREADS as u64, MM_NSG as u64)
}
