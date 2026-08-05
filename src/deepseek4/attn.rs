//! CPU attention for DeepSeek-V4 Flash (SWA + mixed + sinks + tail RoPE).

use super::config::{
    DEFAULT_COMPRESS_ROPE_FREQ_BASE, DEFAULT_ROPE_FREQ_BASE, DEFAULT_ROPE_ORIG_CTX,
    DEFAULT_ROPE_SCALE_FACTOR, DEFAULT_ROPE_YARN_BETA_FAST, DEFAULT_ROPE_YARN_BETA_SLOW,
};
use super::kv::{indexer_topk, LayerKvState};

/// Tail RoPE matching ds4 `rope_tail_ext_inplace` (NoPE prefix untouched).
pub fn rope_tail_ext_inplace(
    x: &mut [f32],
    head_dim: usize,
    n_rot: usize,
    pos: usize,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    n_ctx_orig: u64,
    inverse: bool,
) {
    let n_nope = head_dim - n_rot;
    let theta_scale = freq_base.powf(-2.0 / n_rot as f32);
    let sin_sign = if inverse { -1.0f32 } else { 1.0 };

    let mut corr = [0.0f32; 2];
    if ext_factor != 0.0 {
        corr = yarn_corr_dims(n_rot as i32, n_ctx_orig, freq_base, beta_fast, beta_slow);
    }

    let tail = &mut x[n_nope..n_nope + n_rot];
    let mut theta_extrap = pos as f32;
    for i in (0..n_rot).step_by(2) {
        let theta_interp = freq_scale * theta_extrap;
        let mut theta = theta_interp;
        let mut mscale = attn_factor;
        if ext_factor != 0.0 {
            let ramp_mix = yarn_ramp(corr[0], corr[1], i as i32) * ext_factor;
            theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0 + 0.1 * (1.0 / freq_scale).ln();
        }
        let c = theta.cos() * mscale;
        let s = sin_sign * theta.sin() * mscale;
        let x0 = tail[i];
        let x1 = tail[i + 1];
        tail[i] = x0 * c - x1 * s;
        tail[i + 1] = x0 * s + x1 * c;
        theta_extrap *= theta_scale;
    }
}

fn yarn_corr_dim(n_dims: i32, n_ctx_orig: u64, n_rot: f32, base: f32) -> f32 {
    (n_dims as f32) * ((n_ctx_orig as f32) / (n_rot * 2.0 * std::f32::consts::PI)).ln()
        / (2.0 * base.ln())
}

fn yarn_corr_dims(
    n_dims: i32,
    n_ctx_orig: u64,
    freq_base: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> [f32; 2] {
    let start = yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base).floor();
    let end = yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base).ceil();
    [start.max(0.0), end.min((n_dims - 1) as f32)]
}

fn yarn_ramp(low: f32, high: f32, i0: i32) -> f32 {
    let y = ((i0 / 2) as f32 - low) / (high - low).max(0.001);
    1.0 - y.clamp(0.0, 1.0)
}

/// Raw-layer RoPE (ratio==0): base 10000, scale 1, no YaRN.
pub fn rope_tail_inplace(
    x: &mut [f32],
    head_dim: usize,
    n_rot: usize,
    pos: usize,
    freq_base: f32,
    inverse: bool,
) {
    rope_tail_ext_inplace(
        x,
        head_dim,
        n_rot,
        pos,
        freq_base,
        1.0,
        0.0,
        1.0,
        DEFAULT_ROPE_YARN_BETA_FAST,
        DEFAULT_ROPE_YARN_BETA_SLOW,
        DEFAULT_ROPE_ORIG_CTX,
        inverse,
    );
}

/// Compressed-layer RoPE: base 160000, scale 1/16, YaRN on; attn_factor cancels mag.
pub fn rope_tail_compress_inplace(
    x: &mut [f32],
    head_dim: usize,
    n_rot: usize,
    pos: usize,
    inverse: bool,
) {
    let freq_scale = 1.0 / DEFAULT_ROPE_SCALE_FACTOR;
    // ds4 cancels the YaRN magnitude term: attn_factor = 1 / (1 + 0.1*ln(1/freq_scale))
    let attn_factor = 1.0 / (1.0 + 0.1 * (1.0 / freq_scale).ln());
    rope_tail_ext_inplace(
        x,
        head_dim,
        n_rot,
        pos,
        DEFAULT_COMPRESS_ROPE_FREQ_BASE,
        freq_scale,
        1.0,
        attn_factor,
        DEFAULT_ROPE_YARN_BETA_FAST,
        DEFAULT_ROPE_YARN_BETA_SLOW,
        DEFAULT_ROPE_ORIG_CTX,
        inverse,
    );
}

pub fn apply_rope_all_heads(
    q: &mut [f32],
    n_head: usize,
    head_dim: usize,
    n_rot: usize,
    pos: usize,
    compress: bool,
    inverse: bool,
) {
    for h in 0..n_head {
        let off = h * head_dim;
        if compress {
            rope_tail_compress_inplace(
                &mut q[off..off + head_dim],
                head_dim,
                n_rot,
                pos,
                inverse,
            );
        } else {
            rope_tail_inplace(
                &mut q[off..off + head_dim],
                head_dim,
                n_rot,
                pos,
                DEFAULT_ROPE_FREQ_BASE,
                inverse,
            );
        }
    }
}

/// MQA attention over raw SWA (+ sinks). `q` is [n_head, head_dim], k/v [n_kv, head_dim].
/// Sinks enter the softmax denom only (no value contribution) — ds4 `layer_attention_rows_one`.
pub fn attn_swa_mqa(
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sinks: Option<&[f32]>,
    n_head: usize,
    head_dim: usize,
    n_kv: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    for h in 0..n_head {
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let sink = sinks.map(|s| s[h]).unwrap_or(f32::NEG_INFINITY);
        let mut scores = Vec::with_capacity(n_kv);
        let mut max_s = sink;
        for t in 0..n_kv {
            let kt = &k[t * head_dim..(t + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += qh[d] * kt[d];
            }
            let s = dot * scale;
            scores.push(s);
            max_s = max_s.max(s);
        }
        let mut denom = (sink - max_s).exp();
        let oh = &mut out[h * head_dim..(h + 1) * head_dim];
        oh.fill(0.0);
        for t in 0..n_kv {
            let w = (scores[t] - max_s).exp();
            denom += w;
            let vt = &v[t * head_dim..(t + 1) * head_dim];
            for d in 0..head_dim {
                oh[d] += w * vt[d];
            }
        }
        let inv = 1.0 / denom;
        for d in 0..head_dim {
            oh[d] *= inv;
        }
    }
}

/// Mixed attention: raw + selected compressed rows (K≡V latent).
pub fn attn_mixed_mqa(
    out: &mut [f32],
    q: &[f32],
    kv: &LayerKvState,
    comp_idx: &[usize],
    sinks: Option<&[f32]>,
    n_head: usize,
    head_dim: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_raw = kv.raw_len;
    for h in 0..n_head {
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let sink = sinks.map(|s| s[h]).unwrap_or(f32::NEG_INFINITY);
        let mut scores = Vec::with_capacity(n_raw + comp_idx.len());
        let mut max_s = sink;
        for t in 0..n_raw {
            let kt = &kv.raw_k[t * head_dim..(t + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += qh[d] * kt[d];
            }
            let s = dot * scale;
            scores.push(s);
            max_s = max_s.max(s);
        }
        for &ci in comp_idx {
            let kt = &kv.comp_k[ci * head_dim..(ci + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += qh[d] * kt[d];
            }
            let s = dot * scale;
            scores.push(s);
            max_s = max_s.max(s);
        }
        let mut denom = (sink - max_s).exp();
        let oh = &mut out[h * head_dim..(h + 1) * head_dim];
        oh.fill(0.0);
        let mut t = 0;
        for i in 0..n_raw {
            let w = (scores[t] - max_s).exp();
            t += 1;
            denom += w;
            let vt = &kv.raw_k[i * head_dim..(i + 1) * head_dim];
            for d in 0..head_dim {
                oh[d] += w * vt[d];
            }
        }
        for &ci in comp_idx {
            let w = (scores[t] - max_s).exp();
            t += 1;
            denom += w;
            let vt = &kv.comp_k[ci * head_dim..(ci + 1) * head_dim];
            for d in 0..head_dim {
                oh[d] += w * vt[d];
            }
        }
        let inv = 1.0 / denom;
        for d in 0..head_dim {
            oh[d] *= inv;
        }
    }
}

/// CSA top-k over compressed rows via mean Q·K (placeholder until full indexer FP4).
pub fn select_compressed_rows(
    q: &[f32],
    kv: &LayerKvState,
    top_k: usize,
    _indexer_head_dim: usize,
) -> Vec<usize> {
    let n = kv.comp_len;
    if n == 0 {
        return vec![];
    }
    let hd = kv.cfg.head_dim;
    let n_head = q.len() / hd;
    let mut scores = vec![0.0f32; n];
    for ci in 0..n {
        let kt = &kv.comp_k[ci * hd..(ci + 1) * hd];
        let mut s = 0.0f32;
        for h in 0..n_head {
            let qh = &q[h * hd..(h + 1) * hd];
            let mut dot = 0.0f32;
            for d in 0..hd {
                dot += qh[d] * kt[d];
            }
            s += dot.max(0.0);
        }
        scores[ci] = s;
    }
    indexer_topk(&scores, top_k)
}
