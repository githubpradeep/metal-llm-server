//! Learned CSA/HCA time-axis compressor (studied from ds4 `compressor_decode_one`).
//! Reimplemented in-tree — not a paste of ds4.

use super::attn::{rope_tail_compress_inplace, rope_tail_inplace};
use super::hc::rms_norm;
use super::model::DenseW;

const NEG_INF: f32 = -1.0e30;

/// Per-layer compressor working state (attn and optional indexer streams).
#[derive(Debug)]
pub struct CompressorState {
    pub ratio: u32,
    pub head_dim: usize,
    /// coff=2 for ratio-4, else 1. width = coff * head_dim.
    pub coff: u32,
    pub width: usize,
    /// Rows: ratio (HCA) or 2*ratio (CSA). Each row is `width` floats.
    pub state_kv: Vec<f32>,
    pub state_score: Vec<f32>,
}

impl CompressorState {
    pub fn new(ratio: u32, head_dim: usize) -> Option<Self> {
        if ratio == 0 {
            return None;
        }
        let coff = if ratio == 4 { 2u32 } else { 1 };
        let width = coff as usize * head_dim;
        let nrows = if ratio == 4 {
            2 * ratio as usize
        } else {
            ratio as usize
        };
        Some(Self {
            ratio,
            head_dim,
            coff,
            width,
            state_kv: vec![0.0; nrows * width],
            state_score: vec![0.0; nrows * width],
        })
    }

    pub fn clear(&mut self) {
        self.state_kv.fill(0.0);
        self.state_score.fill(0.0);
    }
}

fn ape_at(ape: &DenseW, j: usize, pos_mod: usize) -> f32 {
    // ggml: ne0=width, ne1=ratio → element (j, pos_mod)
    let width = ape.dims()[0];
    let idx = pos_mod * width + j;
    match ape {
        DenseW::F16 { data, .. } => crate::gpu::f16_to_f32(data[idx]),
        DenseW::F32 { data, .. } => data[idx],
        DenseW::Q8 { .. } => panic!("ape should be F16/F32"),
    }
}

fn pool_decode_state(out: &mut [f32], state: &CompressorState) {
    let head_dim = state.head_dim;
    let width = state.width;
    let ratio = state.ratio as usize;
    for j in 0..head_dim {
        let mut max_score = NEG_INF;
        if state.ratio == 4 {
            for r in 0..ratio {
                let sp = state.state_score[r * width + j];
                let sc = state.state_score[(ratio + r) * width + head_dim + j];
                max_score = max_score.max(sp).max(sc);
            }
        } else {
            for r in 0..ratio {
                max_score = max_score.max(state.state_score[r * width + j]);
            }
        }
        if max_score <= NEG_INF * 0.5 {
            out[j] = 0.0;
            continue;
        }
        let mut denom = 0.0f32;
        let mut sum = 0.0f32;
        if state.ratio == 4 {
            for r in 0..ratio {
                let wp = (state.state_score[r * width + j] - max_score).exp();
                let wc = (state.state_score[(ratio + r) * width + head_dim + j] - max_score).exp();
                denom += wp + wc;
                sum += wp * state.state_kv[r * width + j];
                sum += wc * state.state_kv[(ratio + r) * width + head_dim + j];
            }
        } else {
            for r in 0..ratio {
                let w = (state.state_score[r * width + j] - max_score).exp();
                denom += w;
                sum += w * state.state_kv[r * width + j];
            }
        }
        out[j] = if denom > 0.0 { sum / denom } else { 0.0 };
    }
}

/// One decode step. Returns `Some(compressed_row)` when a compressed token is emitted.
pub fn compressor_step(
    state: &mut CompressorState,
    x_normed: &[f32],
    wkv: &DenseW,
    wgate: &DenseW,
    ape: &DenseW,
    norm: &[f32],
    pos: usize,
    n_rot: usize,
    layer: usize,
    rope_freq: f32,
    rms_eps: f32,
    use_compress_rope: bool,
) -> Option<Vec<f32>> {
    let ratio = state.ratio as usize;
    let width = state.width;
    let head_dim = state.head_dim;
    let pos_mod = pos % ratio;
    let row = if state.ratio == 4 {
        ratio + pos_mod
    } else {
        pos_mod
    };
    let should_compress = (pos + 1) % ratio == 0;

    let mut kv_cur = vec![0.0f32; width];
    let mut sc_cur = vec![0.0f32; width];
    wkv.matvec_ggml(x_normed, &mut kv_cur);
    wgate.matvec_ggml(x_normed, &mut sc_cur);
    for j in 0..width {
        sc_cur[j] += ape_at(ape, j, pos_mod);
    }
    state.state_kv[row * width..(row + 1) * width].copy_from_slice(&kv_cur);
    state.state_score[row * width..(row + 1) * width].copy_from_slice(&sc_cur);

    if !should_compress {
        return None;
    }

    let mut pooled = vec![0.0f32; head_dim];
    pool_decode_state(&mut pooled, state);

    let mut out = vec![0.0f32; head_dim];
    rms_norm(&mut out, &pooled, norm, rms_eps);

    let comp_pos = pos + 1 - ratio;
    if use_compress_rope {
        rope_tail_compress_inplace(&mut out, head_dim, n_rot, comp_pos, false);
    } else {
        rope_tail_inplace(&mut out, head_dim, n_rot, comp_pos, rope_freq, false);
    }
    fp8_e4m3fn_nope_inplace(&mut out, n_rot);

    if state.ratio == 4 {
        for r in 0..ratio {
            let src = (ratio + r) * width;
            let dst = r * width;
            state.state_kv.copy_within(src..src + width, dst);
            state.state_score.copy_within(src..src + width, dst);
        }
        for r in 0..ratio {
            let src = r * width;
            let dst = (ratio + r) * width;
            state.state_kv.copy_within(src..src + width, dst);
            state.state_score.copy_within(src..src + width, dst);
        }
    }

    let _ = layer;
    Some(out)
}

fn e4m3fn_value(i: i32) -> f32 {
    const EXP_SCALE: [f32; 16] = [
        0.0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0,
        128.0, 256.0,
    ];
    let exp = (i >> 3) & 0x0f;
    let mant = i & 0x07;
    if exp == 0 {
        mant as f32 * 0.001953125
    } else {
        (1.0 + mant as f32 * 0.125) * EXP_SCALE[exp as usize]
    }
}

fn e4m3fn_round_trip(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0f32 } else { 1.0 };
    let ax = x.abs().min(448.0);
    let mut lo = 0i32;
    let mut hi = 126i32;
    while lo < hi {
        let mid = (lo + hi + 1) >> 1;
        if e4m3fn_value(mid) <= ax {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let mut best = lo;
    if best < 126 {
        let best_diff = (ax - e4m3fn_value(best)).abs();
        let next_diff = (ax - e4m3fn_value(best + 1)).abs();
        if next_diff < best_diff
            || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)
        {
            best += 1;
        }
    }
    sign * e4m3fn_value(best)
}

/// E4M3FN NoPE quantize (groups of 64) — ds4 `dsv4_fp8_kv_quantize_row_inplace_cpu`.
pub fn fp8_e4m3fn_nope_inplace(kv: &mut [f32], n_rot: usize) {
    let n_nope = kv.len().saturating_sub(n_rot);
    let mut off = 0;
    while off < n_nope {
        let end = (off + 64).min(n_nope);
        let mut amax = 0.0f32;
        for i in off..end {
            amax = amax.max(kv[i].abs());
        }
        if amax < 1e-4 {
            amax = 1e-4;
        }
        let scale = 2f32.powf((amax / 448.0).log2().ceil());
        for i in off..end {
            let v = (kv[i] / scale).clamp(-448.0, 448.0);
            kv[i] = e4m3fn_round_trip(v) * scale;
        }
        off += 64;
    }
}

/// Legacy alias kept for older call sites.
pub fn fp8_nope_round_inplace(kv: &mut [f32], n_rot: usize) {
    fp8_e4m3fn_nope_inplace(kv, n_rot);
}
