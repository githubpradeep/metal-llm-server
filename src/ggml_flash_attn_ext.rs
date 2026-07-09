//! llama.cpp kernel_flash_attn_ext tiled prefill (Q4_0 KV, causal f16 mask).

#![allow(dead_code)]

pub const NQPTG: u32 = 8;
pub const NCPSG: u32 = 64;

#[repr(C)]
pub struct FlashAttnExtPadArgs {
    pub ne11: i32,
    pub ne_12_2: i32,
    pub ne_12_3: i32,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub nb21: u64,
    pub nb22: u64,
    pub nb23: u64,
    pub ne31: i32,
    pub ne32: i32,
    pub ne33: i32,
    pub nb31: u64,
    pub nb32: u64,
    pub nb33: u64,
}

#[repr(C)]
pub struct FlashAttnExtBlkArgs {
    pub ne01: i32,
    pub ne30: i32,
    pub ne31: i32,
    pub ne32: i32,
    pub ne33: i32,
    pub nb31: u64,
    pub nb32: u64,
    pub nb33: u64,
}

#[repr(C)]
pub struct FlashAttnExtArgs {
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
    pub ne31: i32,
    pub ne32: i32,
    pub ne33: i32,
    pub nb31: u64,
    pub nb32: u64,
    pub nb33: u64,
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

fn pad_to(x: u64, n: u64) -> u64 {
    ((x + n - 1) / n) * n
}

/// Use tiled flash_attn_ext when q_len ≥ 20 (matches llama.cpp vec/tiled switch).
pub fn prefill_use_tiled_ext(q_len: u32, head_dim: u32) -> bool {
    q_len >= 20 && head_dim % 32 == 0 && matches!(head_dim, 256 | 512)
}

pub fn mask_bytes(q_len: u32, kv_seq: u32) -> u64 {
    q_len as u64 * kv_seq as u64 * 2
}

pub fn pad_bytes(row_bytes: u64, num_kv_heads: u32, q_len: u32, kv_seq: u32) -> u64 {
    let kv_plane = row_bytes * num_kv_heads as u64;
    let mask_plane = q_len as u64 * kv_seq as u64 * 2;
    NCPSG as u64 * (kv_plane + kv_plane + mask_plane)
}

pub fn blk_bytes(q_len: u32, kv_seq: u32) -> u64 {
    let nblk1 = (q_len + NQPTG - 1) / NQPTG;
    let nblk0 = (kv_seq + NCPSG - 1) / NCPSG;
    pad_to(nblk0 as u64 * nblk1 as u64 * q_len as u64, 32)
}

pub fn scratch_bytes(
    max_q_len: u32,
    kv_capacity: u32,
    num_kv_heads: u32,
    row_bytes: u64,
) -> u64 {
    mask_bytes(max_q_len, kv_capacity)
        + pad_bytes(row_bytes, num_kv_heads, max_q_len, kv_capacity)
        + blk_bytes(max_q_len, kv_capacity)
}

pub fn smem_bytes(head_dim: u32, nsg: u32) -> u64 {
    let ne00 = head_dim as u64;
    let ne20 = head_dim as u64;
    let nqptg = NQPTG as u64;
    let ncpsg = NCPSG as u64;
    let is_q = 1u64;
    let inner = (nqptg * (ne00 + 2 * pad_to(ne20, 64) + 2 * (2 * ncpsg)) + is_q * (16 * 32 * nsg as u64)) * 2;
    pad_to(inner, 16)
}

pub fn nsg_for_head_dim(head_dim: u32) -> u32 {
    if head_dim >= 512 { 8 } else { 4 }
}

pub fn flash_attn_ext_args(
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_seq: u32,
    capacity: u32,
    row_bytes: u64,
    q_len: u32,
    scale: f32,
) -> FlashAttnExtArgs {
    let blocks = (head_dim / 32) as i32;
    let head_stride = head_dim as u64 * 4;
    let kv_stride = capacity as u64 * row_bytes;
    FlashAttnExtArgs {
        ne01: q_len as i32,
        ne02: num_heads as i32,
        ne03: 1,
        nb01: head_stride,
        nb02: head_stride * q_len as u64,
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
        ne31: q_len as i32,
        ne32: 1,
        ne33: 1,
        nb31: kv_seq as u64 * 2,
        nb32: 0,
        nb33: 0,
        ne1: num_heads as i32,
        ne2: q_len as i32,
        ne3: 1,
        scale,
        max_bias: 0.0,
        m0: 0.0,
        m1: 0.0,
        n_head_log2: 0,
        logit_softcap: 0.0,
    }
}

pub fn pad_args(
    num_kv_heads: u32,
    kv_seq: u32,
    row_bytes: u64,
    q_len: u32,
) -> FlashAttnExtPadArgs {
    FlashAttnExtPadArgs {
        ne11: kv_seq as i32,
        ne_12_2: num_kv_heads as i32,
        ne_12_3: 1,
        nb11: row_bytes,
        nb12: row_bytes * kv_seq as u64,
        nb13: 0,
        nb21: row_bytes,
        nb22: row_bytes * kv_seq as u64,
        nb23: 0,
        ne31: q_len as i32,
        ne32: 1,
        ne33: 1,
        nb31: 2,
        nb32: kv_seq as u64 * 2,
        nb33: 0,
    }
}

pub fn blk_args(q_len: u32, kv_seq: u32) -> FlashAttnExtBlkArgs {
    FlashAttnExtBlkArgs {
        ne01: q_len as i32,
        ne30: kv_seq as i32,
        ne31: q_len as i32,
        ne32: 1,
        ne33: 1,
        nb31: kv_seq as u64 * 2,
        nb32: 0,
        nb33: 0,
    }
}

const F16_NEG_INF: u16 = 0xFC00;

/// Fill causal f16 mask: shape [q_len, kv_seq], row-major (query × kv).
pub fn fill_causal_mask(
    out: &mut [u16],
    q_len: u32,
    kv_seq: u32,
    q_start: u32,
    attention_window: u32,
) {
    assert_eq!(out.len(), (q_len as usize) * (kv_seq as usize));
    for qi in 0..q_len {
        let q_pos = q_start + qi;
        let attend_len = (q_pos + 1).min(kv_seq);
        let attend_start = if attention_window > 0 && attend_len > attention_window {
            attend_len - attention_window
        } else {
            0
        };
        for kj in 0..kv_seq {
            let allowed = kj <= q_pos && kj >= attend_start;
            out[qi as usize * kv_seq as usize + kj as usize] = if allowed { 0 } else { F16_NEG_INF };
        }
    }
}

pub struct ScratchLayout {
    pub mask_off: u64,
    pub pad_off: u64,
    pub blk_off: u64,
    pub total: u64,
}

pub fn scratch_layout(
    max_q_len: u32,
    kv_capacity: u32,
    num_kv_heads: u32,
    row_bytes: u64,
) -> ScratchLayout {
    let mask = mask_bytes(max_q_len, kv_capacity);
    let pad = pad_bytes(row_bytes, num_kv_heads, max_q_len, kv_capacity);
    let blk = blk_bytes(max_q_len, kv_capacity);
    ScratchLayout {
        mask_off: 0,
        pad_off: mask,
        blk_off: mask + pad,
        total: mask + pad + blk,
    }
}
