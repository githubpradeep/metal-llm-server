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

/// Max KV length for tiled-prefill mask/blk scratch.
///
/// Scratch is `[q_chunk × kv] f16` × 2 windows. With `LLAMA_CTX_SIZE=200k` and
/// `max_q=4096` that would be multi-GB if sized to full capacity — thrashing
/// even when tiled is unused. Cap here; layers with `kv_seq` above this fall
/// back to legacy causal attention.
pub const MAX_TILED_PREFILL_KV: u32 = 65536;

/// Use tiled flash_attn_ext when q_len ≥ 20 (matches llama.cpp vec/tiled switch).
///
/// Enabled for h256 (SWA) and h512 (full) once pad/mask/NSG fixes landed.
/// Host `nsg_for_head_dim` must match Metal entry NSG (h256=8, h512=4).
pub fn prefill_use_tiled_ext(q_len: u32, head_dim: u32) -> bool {
    q_len >= 20 && matches!(head_dim, 256 | 512)
}

pub fn mask_bytes(q_len: u32, kv_seq: u32) -> u64 {
    q_len as u64 * kv_seq as u64 * 2
}

/// Pad scratch: last partial KV chunk (C rows) + causal mask rows for those C cols.
/// Layout matches llama.cpp/ds4: `C * (2 * row_bytes * n_kv_heads + q_len * sizeof(half))`.
pub fn pad_bytes(row_bytes: u64, num_kv_heads: u32, q_len: u32, _kv_seq: u32) -> u64 {
    NCPSG as u64 * (2 * row_bytes * num_kv_heads as u64 + q_len as u64 * 2)
}

pub fn blk_bytes(q_len: u32, kv_seq: u32) -> u64 {
    // One byte per (KV-tile × Q-tile); matches ds4/llama.cpp:
    //   align_up(nblk0 * nblk1, 32)
    // (Previously multiplied by q_len → ~GB of wasted scratch.)
    let nblk1 = (q_len + NQPTG - 1) / NQPTG;
    let nblk0 = (kv_seq + NCPSG - 1) / NCPSG;
    pad_to(nblk0 as u64 * nblk1 as u64, 32)
}

pub fn scratch_bytes(
    max_q_len: u32,
    kv_capacity: u32,
    num_kv_heads: u32,
    row_bytes: u64,
) -> u64 {
    let mask_kv = kv_capacity.min(MAX_TILED_PREFILL_KV);
    mask_bytes(max_q_len, mask_kv)
        + pad_bytes(row_bytes, num_kv_heads, max_q_len, mask_kv)
        + blk_bytes(max_q_len, mask_kv)
}

/// Host NSG must match Metal entry points:
/// - h256: NSG=8 (24 KB smem; matches llama.cpp for dk<512)
/// - h512: NSG=4 (32 KB; NSG=8 would be 36 KB > Metal limit with Q4)
pub fn nsg_for_head_dim(head_dim: u32) -> u32 {
    if head_dim <= 256 {
        8
    } else {
        4
    }
}

pub fn smem_bytes(head_dim: u32, nsg: u32) -> u64 {
    let ne00 = head_dim as u64;
    let ne20 = head_dim as u64;
    let nqptg = NQPTG as u64;
    let ncpsg = NCPSG as u64;
    let is_q = 1u64;
    let inner = (nqptg * (ne00 + 2 * pad_to(ne20, 64) + 2 * (2 * ncpsg)) + is_q * (16 * 32 * nsg as u64)) * 2;
    let bytes = pad_to(inner, 16);
    debug_assert!(
        bytes <= 32768,
        "flash_attn_ext smem {bytes} exceeds Metal 32KB limit (head_dim={head_dim}, nsg={nsg})"
    );
    bytes
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
    // Prefill flash_attn_ext consumes f32 Q (cast to half inside the kernel, like llama.cpp).
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
    capacity: u32,
    row_bytes: u64,
    q_len: u32,
) -> FlashAttnExtPadArgs {
    // KV cache is [kv_head][capacity][row]; head stride must use capacity, not kv_seq.
    let kv_head_stride = capacity as u64 * row_bytes;
    let mask_row_bytes = kv_seq as u64 * 2;
    FlashAttnExtPadArgs {
        ne11: kv_seq as i32,
        ne_12_2: num_kv_heads as i32,
        ne_12_3: 1,
        nb11: row_bytes,
        nb12: kv_head_stride,
        nb13: 0,
        nb21: row_bytes,
        nb22: kv_head_stride,
        nb23: 0,
        ne31: q_len as i32,
        ne32: 1,
        ne33: 1,
        // Causal mask is [q_len][kv_seq] f16 row-major (same as fill_causal_mask).
        nb31: mask_row_bytes,
        nb32: 0,
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

#[repr(C)]
pub struct FlashAttnExtMaskFillArgs {
    pub q_len: u32,
    pub kv_seq: u32,
    pub q_start: u32,
    pub attention_window: u32,
}

pub fn mask_fill_args(
    q_len: u32,
    kv_seq: u32,
    q_start: u32,
    attention_window: u32,
) -> FlashAttnExtMaskFillArgs {
    FlashAttnExtMaskFillArgs {
        q_len,
        kv_seq,
        q_start,
        attention_window,
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
    // Default deny, then punch allowed [attend_start, q_pos] holes.
    out.fill(F16_NEG_INF);
    for qi in 0..q_len {
        let q_pos = q_start + qi;
        let attend_len = (q_pos + 1).min(kv_seq);
        let attend_start = if attention_window > 0 && attend_len > attention_window {
            attend_len - attention_window
        } else {
            0
        };
        let row = qi as usize * kv_seq as usize;
        for kj in attend_start..attend_len {
            out[row + kj as usize] = 0;
        }
    }
}

pub struct ScratchLayout {
    /// Shared pad region (GPU-ordered per layer; safe to reuse in one CB).
    pub pad_off: u64,
    full_mask_off: u64,
    full_blk_off: u64,
    swa_mask_off: u64,
    swa_blk_off: u64,
    /// Max KV columns the mask/blk planes can hold (may be < model capacity).
    pub mask_kv_capacity: u32,
    pub total: u64,
}

impl ScratchLayout {
    /// Separate CPU-written mask+blk planes for full vs sliding-window layers.
    /// One CB encodes all layers; CPU fills must not overwrite a plane still
    /// referenced by earlier dispatches in that CB.
    pub fn mask_blk_off(&self, attention_window: u32) -> (u64, u64) {
        if attention_window == 0 {
            (self.full_mask_off, self.full_blk_off)
        } else {
            (self.swa_mask_off, self.swa_blk_off)
        }
    }
}

/// Per-window mask+blk readiness inside one prefill chunk CB.
#[derive(Clone, Copy, Default)]
struct PrefillExtMaskSlot {
    q_len: u32,
    kv_seq: u32,
    q_start: u32,
    attention_window: u32,
    ready: bool,
}

impl PrefillExtMaskSlot {
    fn matches(&self, q_len: u32, kv_seq: u32, q_start: u32, attention_window: u32) -> bool {
        self.ready
            && self.q_len == q_len
            && self.kv_seq == kv_seq
            && self.q_start == q_start
            && self.attention_window == attention_window
    }

    fn mark(&mut self, q_len: u32, kv_seq: u32, q_start: u32, attention_window: u32) {
        self.q_len = q_len;
        self.kv_seq = kv_seq;
        self.q_start = q_start;
        self.attention_window = attention_window;
        self.ready = true;
    }
}

/// Cache mask+blk prepass across layers; full and SWA keep independent slots.
#[derive(Clone, Copy, Default)]
pub struct PrefillExtMaskCache {
    full: PrefillExtMaskSlot,
    swa: PrefillExtMaskSlot,
}

impl PrefillExtMaskCache {
    fn slot(&self, attention_window: u32) -> &PrefillExtMaskSlot {
        if attention_window == 0 {
            &self.full
        } else {
            &self.swa
        }
    }

    fn slot_mut(&mut self, attention_window: u32) -> &mut PrefillExtMaskSlot {
        if attention_window == 0 {
            &mut self.full
        } else {
            &mut self.swa
        }
    }

    pub fn matches(&self, q_len: u32, kv_seq: u32, q_start: u32, attention_window: u32) -> bool {
        self.slot(attention_window)
            .matches(q_len, kv_seq, q_start, attention_window)
    }

    pub fn mark(&mut self, q_len: u32, kv_seq: u32, q_start: u32, attention_window: u32) {
        self.slot_mut(attention_window)
            .mark(q_len, kv_seq, q_start, attention_window);
    }
}

pub fn scratch_layout(
    max_q_len: u32,
    kv_capacity: u32,
    num_kv_heads: u32,
    row_bytes: u64,
) -> ScratchLayout {
    let mask_kv = kv_capacity.min(MAX_TILED_PREFILL_KV);
    let mask = mask_bytes(max_q_len, mask_kv);
    let pad = pad_bytes(row_bytes, num_kv_heads, max_q_len, mask_kv);
    let blk = blk_bytes(max_q_len, mask_kv);
    // [full_mask][swa_mask][pad][full_blk][swa_blk]
    let full_mask_off = 0;
    let swa_mask_off = mask;
    let pad_off = mask * 2;
    let full_blk_off = pad_off + pad;
    let swa_blk_off = full_blk_off + blk;
    ScratchLayout {
        pad_off,
        full_mask_off,
        full_blk_off,
        swa_mask_off,
        swa_blk_off,
        mask_kv_capacity: mask_kv,
        total: swa_blk_off + blk,
    }
}
