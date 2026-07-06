//! ggml-metal flash_attn_ext_vec decode helpers (Q4_0 KV cache).

#![allow(dead_code)]

/// Must match `ggml_flash_attn_args` in ggml_flash_attn.metal.
#[repr(C)]
pub struct GgmlFlashAttnArgs {
    pub ne01: i32,
    pub ne02: i32,
    pub ne03: i32,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne11: i32,
    pub ne_12_2: i32,
    pub ne_12_3: i32,
    pub ns10: i32,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub ns20: i32,
    pub nb21: u64,
    pub nb22: u64,
    pub nb23: u64,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub scale: f32,
    pub max_bias: f32,
    pub m0: f32,
    pub m1: f32,
    pub n_head_log2: i32,
    pub logit_softcap: f32,
}

/// Must match `ggml_flash_attn_reduce_args` in ggml_flash_attn.metal.
#[repr(C)]
pub struct GgmlFlashAttnReduceArgs {
    pub nrows: i32,
}

/// llama.cpp decode default: 32 workgroups partition KV (C=32 tokens each).
pub const NWG: u64 = 32;
const NCPSG: u64 = 32;
const NSG: u64 = 1;

fn pad_to(x: u64, n: u64) -> u64 {
    ((x + n - 1) / n) * n
}

/// Threadgroup shared memory bytes for flash_attn_ext_vec (matches ggml FATTN_SMEM).
pub fn flash_attn_smem_bytes(head_dim: u32) -> u64 {
    let ne00 = head_dim as u64;
    let ne20 = head_dim as u64;
    let inner = (pad_to(ne00, 128) + 4 * NCPSG + 2 * pad_to(ne20, 128)) * NSG * 2;
    pad_to(inner, 16)
}

/// Temp buffer for multi-WG path: ne01_max * ne02 * ne03 * nwg * (head_dim + 2) f32.
pub fn flash_attn_tmp_bytes(num_heads: u32, head_dim: u32) -> u64 {
    let ne01_max = 1u64;
    let ne02 = num_heads as u64;
    let ne03 = 1u64;
    let ne20 = head_dim as u64;
    ne01_max * ne02 * ne03 * NWG * (ne20 + 2) * std::mem::size_of::<f32>() as u64
}

pub fn flash_attn_args(
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_capacity: u32,
    kv_seq: u32,
    row_bytes: u64,
    scale: f32,
) -> GgmlFlashAttnArgs {
    let blocks = (head_dim / 32) as i32;
    let head_stride = head_dim as u64 * 4;
    let kv_stride = kv_capacity as u64 * row_bytes;
    GgmlFlashAttnArgs {
        ne01: 1,
        ne02: num_heads as i32,
        ne03: 1,
        nb01: head_stride,
        nb02: head_stride,
        nb03: 0,
        ne11: kv_seq as i32,
        ne_12_2: num_kv_heads as i32,
        ne_12_3: 1,
        ns10: blocks,
        nb11: row_bytes,
        nb12: kv_stride,
        nb13: 0,
        ns20: blocks,
        nb21: row_bytes,
        nb22: kv_stride,
        nb23: 0,
        ne1: 1,
        ne2: num_heads as i32,
        ne3: 1,
        scale,
        max_bias: 0.0,
        m0: 0.0,
        m1: 0.0,
        n_head_log2: 0,
        logit_softcap: 0.0,
    }
}

/// Main kernel dispatch: one threadgroup per head × NWG KV partitions.
pub fn flash_attn_dispatch(num_heads: u32) -> (u64, u64, u64, u64, u64) {
    (1, num_heads as u64, NWG, 32, NSG)
}

/// Reduce kernel dispatch after multi-WG main pass.
pub fn flash_attn_reduce_dispatch(num_heads: u32) -> (u64, u64, u64, u64, u64, u64) {
    let nrows = num_heads as u64;
    (nrows, 1, 1, 32 * NWG, 1, 1)
}
