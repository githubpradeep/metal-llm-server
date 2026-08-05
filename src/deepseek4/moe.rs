//! DeepSeek-V4 MoE routing: √softplus, hash-MoE, top-k selection.
//!
//! Studied from ds4 router path (`softplus_stable` + `sqrtf`, hash `tid2eid`,
//! expert_weight_scale=1.5). Reimplemented for llama-sinks.

use super::config::DEFAULT_EXPERT_WEIGHT_SCALE;

#[inline]
pub fn softplus_stable(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU with optional clamp on gate/up (Flash clamp ≈ 50).
pub fn swiglu(out: &mut [f32], gate: &[f32], up: &[f32], clamp: f32) {
    assert_eq!(out.len(), gate.len());
    assert_eq!(out.len(), up.len());
    let use_clamp = clamp > 1e-6;
    for i in 0..out.len() {
        let mut g = gate[i];
        let mut u = up[i];
        if use_clamp {
            g = g.min(clamp);
            u = u.clamp(-clamp, clamp);
        }
        out[i] = silu(g) * u;
    }
}

/// Convert router logits → √softplus scores (unnormalized).
pub fn router_probs_sqrt_softplus(logits: &[f32], probs: &mut [f32]) {
    assert_eq!(logits.len(), probs.len());
    for i in 0..logits.len() {
        probs[i] = softplus_stable(logits[i]).sqrt();
    }
}

/// Select top-k indices by `scores` (optionally + bias for selection only).
/// Returns (expert_ids, renormed_weights) with weights from unbiased probs × scale.
pub fn select_topk_experts(
    probs: &[f32],
    bias: Option<&[f32]>,
    k: usize,
    weight_scale: f32,
) -> (Vec<u32>, Vec<f32>) {
    let n = probs.len();
    assert!(k <= n);
    let mut scored: Vec<(u32, f32, f32)> = (0..n)
        .map(|i| {
            let p = probs[i];
            let sel = match bias {
                Some(b) => p + b[i],
                None => p,
            };
            (i as u32, sel, p)
        })
        .collect();
    scored.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(k);
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut ids = Vec::with_capacity(k);
    let mut weights = Vec::with_capacity(k);
    let mut sum = 0.0f32;
    for &(id, _, p) in &scored {
        ids.push(id);
        weights.push(p);
        sum += p;
    }
    let inv = weight_scale / sum.max(6.103515625e-5);
    for w in &mut weights {
        *w *= inv;
    }
    (ids, weights)
}

/// Hash-MoE: `ffn_gate_tid2eid` is I32 `[n_expert_used, vocab]` (ne0=k, ne1=vocab).
/// Contiguous layout is `[token][slot]` → index `token * k + slot` (ds4
/// `layer_hash_selected_experts`).
pub fn hash_experts_for_token(
    tid2eid: &[i32],
    vocab: usize,
    token: usize,
    k: usize,
    probs: &[f32],
    weight_scale: f32,
) -> (Vec<u32>, Vec<f32>) {
    assert!(token < vocab);
    assert!(tid2eid.len() >= vocab * k);
    let mut ids = Vec::with_capacity(k);
    let mut weights = Vec::with_capacity(k);
    let mut sum = 0.0f32;
    let row = &tid2eid[token * k..(token + 1) * k];
    for &eid in row {
        assert!(eid >= 0);
        let id = eid as u32;
        let p = probs.get(id as usize).copied().unwrap_or(0.0);
        ids.push(id);
        weights.push(p);
        sum += p;
    }
    // ds4 floors the sum with fp16 min (~6.103515625e-5) before scale.
    let sum = sum.max(6.103515625e-5);
    let inv = weight_scale / sum;
    for w in &mut weights {
        *w *= inv;
    }
    (ids, weights)
}

pub fn default_weight_scale() -> f32 {
    DEFAULT_EXPERT_WEIGHT_SCALE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_renorms_to_scale() {
        let probs = [0.1f32, 0.5, 0.2, 0.05, 0.15];
        let (ids, w) = select_topk_experts(&probs, None, 2, 1.5);
        assert_eq!(ids.len(), 2);
        let s: f32 = w.iter().sum();
        assert!((s - 1.5).abs() < 1e-5, "sum={s}");
    }
}
