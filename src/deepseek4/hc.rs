//! Manifold-Constrained Hyper-Connections (mHC) — CPU reference.
//!
//! Algorithm studied from ds4 `hc_split_sinkhorn_one` / `hc_weighted_sum_one` /
//! `hc_pre_from_state_one` / HC expand-post. Reimplemented here; Metal mirrors
//! the same math in `shaders/dsv4_hc.metal`.

use super::config::{DEFAULT_HC_EPS, FLASH_N_HC};

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Sinkhorn split of the HC control vector.
///
/// `mix` is the F16 matvec output of shape `[n_hc + n_hc + n_hc*n_hc]` =
/// pre (n_hc) | post (n_hc) | comb (n_hc²).
/// `scale` / `base` are length-3 / length-`2*n_hc + n_hc²` respectively
/// (Flash: scale has 3 elems pre/post/comb; base matches mix length).
///
/// Writes `out` with the same layout: pre weights, post weights, doubly-stochastic comb.
pub fn hc_split_sinkhorn(
    out: &mut [f32],
    mix: &[f32],
    scale: &[f32],
    base: &[f32],
    n_hc: usize,
    iters: usize,
    eps: f32,
) {
    assert!(n_hc <= 16);
    assert_eq!(mix.len(), 2 * n_hc + n_hc * n_hc);
    assert_eq!(out.len(), mix.len());
    assert!(scale.len() >= 3);
    assert_eq!(base.len(), mix.len());

    let pre_scale = scale[0];
    let post_scale = scale[1];
    let comb_scale = scale[2];

    for i in 0..n_hc {
        let z = mix[i] * pre_scale + base[i];
        out[i] = sigmoid(z) + eps;
    }
    for i in 0..n_hc {
        let off = n_hc + i;
        let z = mix[off] * post_scale + base[off];
        out[off] = 2.0 * sigmoid(z);
    }

    let mut c = [0.0f32; 16 * 16];
    for dst in 0..n_hc {
        let mut row_max = f32::NEG_INFINITY;
        for src in 0..n_hc {
            let idx = src + dst * n_hc;
            let off = 2 * n_hc + idx;
            let v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            if v > row_max {
                row_max = v;
            }
        }
        let mut row_sum = 0.0f32;
        for src in 0..n_hc {
            let idx = src + dst * n_hc;
            let v = (c[idx] - row_max).exp();
            c[idx] = v;
            row_sum += v;
        }
        let inv = 1.0 / row_sum;
        for src in 0..n_hc {
            let idx = src + dst * n_hc;
            c[idx] = c[idx] * inv + eps;
        }
    }

    for src in 0..n_hc {
        let mut sum = 0.0f32;
        for dst in 0..n_hc {
            sum += c[src + dst * n_hc];
        }
        let inv = 1.0 / (sum + eps);
        for dst in 0..n_hc {
            c[src + dst * n_hc] *= inv;
        }
    }

    for _iter in 1..iters {
        for dst in 0..n_hc {
            let mut sum = 0.0f32;
            for src in 0..n_hc {
                sum += c[src + dst * n_hc];
            }
            let inv = 1.0 / (sum + eps);
            for src in 0..n_hc {
                c[src + dst * n_hc] *= inv;
            }
        }
        for src in 0..n_hc {
            let mut sum = 0.0f32;
            for dst in 0..n_hc {
                sum += c[src + dst * n_hc];
            }
            let inv = 1.0 / (sum + eps);
            for dst in 0..n_hc {
                c[src + dst * n_hc] *= inv;
            }
        }
    }

    for i in 0..(n_hc * n_hc) {
        out[2 * n_hc + i] = c[i];
    }
}

/// Reduce 4 HC streams → plain embedding: `out[d] = Σ_h x[h,d] * w[h]`.
pub fn hc_weighted_sum(out: &mut [f32], x_hc: &[f32], weights: &[f32], n_embd: usize, n_hc: usize) {
    assert_eq!(out.len(), n_embd);
    assert_eq!(x_hc.len(), n_embd * n_hc);
    assert_eq!(weights.len(), n_hc);
    for d in 0..n_embd {
        let mut acc = 0.0f32;
        for h in 0..n_hc {
            acc += x_hc[h * n_embd + d] * weights[h];
        }
        out[d] = acc;
    }
}

/// Expand block output back into HC residual using post + comb.
///
/// For each stream h:
///   `hc[h] += post[h] * block + Σ_s comb[s,h] * residual_add[s]`
/// Flash typically uses `residual_add = pre-attn residual streams` (identity
/// residual path). Here `add_hc` is the HC state before the sublayer (n_hc × n_embd)
/// and `block` is the sublayer output (n_embd).
pub fn hc_expand_post(
    hc: &mut [f32],
    block: &[f32],
    add_hc: &[f32],
    post: &[f32],
    comb: &[f32],
    n_embd: usize,
    n_hc: usize,
) {
    assert_eq!(hc.len(), n_embd * n_hc);
    assert_eq!(add_hc.len(), n_embd * n_hc);
    assert_eq!(block.len(), n_embd);
    assert_eq!(post.len(), n_hc);
    assert_eq!(comb.len(), n_hc * n_hc);

    for h in 0..n_hc {
        for d in 0..n_embd {
            let mut v = post[h] * block[d];
            for s in 0..n_hc {
                // ds4 hc_post_one addresses comb as [dst, src] = dst + src*n_hc,
                // even though Sinkhorn fills src + dst*n_hc (intentional layout).
                v += comb[h + s * n_hc] * add_hc[s * n_embd + d];
            }
            hc[h * n_embd + d] = v;
        }
    }
}

/// Seed all HC streams with the same token embedding.
pub fn hc_from_plain_embedding(out_hc: &mut [f32], x: &[f32], n_embd: usize, n_hc: usize) {
    assert_eq!(out_hc.len(), n_embd * n_hc);
    assert_eq!(x.len(), n_embd);
    for h in 0..n_hc {
        out_hc[h * n_embd..(h + 1) * n_embd].copy_from_slice(x);
    }
}

/// RMSNorm without learned weight (used on flattened HC state before HC fn).
pub fn rms_norm_no_weight(out: &mut [f32], x: &[f32], eps: f32) {
    assert_eq!(out.len(), x.len());
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let scale = (ss / x.len() as f32 + eps).sqrt().recip();
    for i in 0..x.len() {
        out[i] = x[i] * scale;
    }
}

/// RMSNorm with weight.
pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    assert_eq!(out.len(), x.len());
    assert_eq!(weight.len(), x.len());
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let scale = (ss / x.len() as f32 + eps).sqrt().recip();
    for i in 0..x.len() {
        out[i] = x[i] * scale * weight[i];
    }
}

/// Output HC collapse: softmax-like post weights over 4 streams → embd, then used before lm_head.
pub fn output_hc_weights(post_scale: f32, mix_post: &[f32], base_post: &[f32], eps: f32) -> [f32; FLASH_N_HC] {
    let mut w = [0.0f32; FLASH_N_HC];
    for i in 0..FLASH_N_HC {
        w[i] = 2.0 * sigmoid(mix_post[i] * post_scale + base_post[i]) + eps;
    }
    let sum: f32 = w.iter().sum::<f32>() + eps;
    for i in 0..FLASH_N_HC {
        w[i] /= sum;
    }
    w
}

pub fn default_eps() -> f32 {
    DEFAULT_HC_EPS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinkhorn_rows_cols_approx_one() {
        let n_hc = 4;
        let mix_len = 2 * n_hc + n_hc * n_hc;
        let mut mix = vec![0.1f32; mix_len];
        for (i, v) in mix.iter_mut().enumerate() {
            *v = (i as f32) * 0.01 - 0.1;
        }
        let scale = [1.0f32, 1.0, 1.0];
        let base = vec![0.0f32; mix_len];
        let mut out = vec![0.0f32; mix_len];
        hc_split_sinkhorn(&mut out, &mix, &scale, &base, n_hc, 20, 1e-6);
        // Columns of comb should sum ~1 after Sinkhorn.
        for dst in 0..n_hc {
            let mut s = 0.0f32;
            for src in 0..n_hc {
                s += out[2 * n_hc + src + dst * n_hc];
            }
            assert!((s - 1.0).abs() < 1e-3, "col {dst} sum={s}");
        }
    }
}
