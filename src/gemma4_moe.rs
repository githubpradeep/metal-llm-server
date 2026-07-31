//! Gemma 4 MoE helpers (26B-A4B): top-k routing, expert views, CPU cross-check.

use crate::gguf::{self, ggml_type, dequant_row_to_f32};
use crate::gpu::{f16_to_f32, BufferView};

/// Softmax over all experts, then top-k with renormalized weights.
pub fn softmax_topk_renorm(logits: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    assert!(!logits.is_empty());
    let k = k.min(logits.len()).max(1);
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&x| (x - max_l).exp()).collect();
    let sum: f32 = probs.iter().sum();
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for p in &mut probs {
        *p *= inv;
    }
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    let wsum: f32 = idx.iter().map(|&i| probs[i]).sum();
    let inv_w = if wsum > 0.0 { 1.0 / wsum } else { 0.0 };
    let weights: Vec<f32> = idx.iter().map(|&i| probs[i] * inv_w).collect();
    (idx, weights)
}

/// Byte stride of one expert's fused gate∥up Q4_K matrix `[n_embd, 2*n_ff]`.
pub fn gate_up_expert_bytes(n_embd: usize, n_ff_exp: usize) -> u64 {
    let n_ff2 = n_ff_exp * 2;
    let blocks_per_row = n_embd / 256;
    (n_ff2 * blocks_per_row * 144) as u64
}

/// Byte offset of gate (or up) rows within one expert's gate_up blob.
/// Gate is the first `n_ff` rows; up is the second (llama.cpp / conversion).
pub fn gate_up_half_offset(n_embd: usize, n_ff_exp: usize, is_up: bool) -> u64 {
    let blocks_per_row = n_embd / 256;
    let row_bytes = (blocks_per_row * 144) as u64;
    if is_up {
        n_ff_exp as u64 * row_bytes
    } else {
        0
    }
}

/// Byte stride of one expert's down matrix `[n_ff_exp, n_embd]` for the given
/// ggml block size (24 for Q5_1, 34 for Q8_0).
pub fn down_expert_bytes(n_ff_exp: usize, n_embd: usize, block_bytes: usize) -> u64 {
    let epb = 32usize;
    let n = n_ff_exp * n_embd;
    ((n / epb) * block_bytes) as u64
}

fn down_block_bytes(format: u8) -> usize {
    match format {
        crate::gpu::weight_fmt::Q5_1 => 24,
        crate::gpu::weight_fmt::Q8_0 => 34,
        other => panic!("unsupported MoE down format {other}"),
    }
}

/// View into one expert's gate or up half of `gate_up_exps` (Q4_K).
pub fn expert_gate_up_view(
    gate_up_exps: &BufferView,
    expert: usize,
    n_embd: usize,
    n_ff_exp: usize,
    is_up: bool,
) -> BufferView {
    let expert_bytes = gate_up_expert_bytes(n_embd, n_ff_exp);
    let half = gate_up_half_offset(n_embd, n_ff_exp, is_up);
    let half_bytes = gate_up_expert_bytes(n_embd, n_ff_exp) / 2;
    BufferView {
        buffer: gate_up_exps.buffer.clone(),
        offset: gate_up_exps.offset + expert as u64 * expert_bytes + half,
        length: half_bytes,
        format: crate::gpu::weight_fmt::Q4_K,
    }
}

/// View into one expert's down matrix `[n_embd rows × n_ff_exp cols]`.
pub fn expert_down_view(
    down_exps: &BufferView,
    expert: usize,
    n_ff_exp: usize,
    n_embd: usize,
) -> BufferView {
    let bpb = down_block_bytes(down_exps.format);
    let stride = down_expert_bytes(n_ff_exp, n_embd, bpb);
    BufferView {
        buffer: down_exps.buffer.clone(),
        offset: down_exps.offset + expert as u64 * stride,
        length: stride,
        format: down_exps.format,
    }
}

fn expert_gate_up_bytes<'a>(
    gate_up_exps: &'a BufferView,
    expert: usize,
    n_embd: usize,
    n_ff_exp: usize,
    is_up: bool,
) -> &'a [u8] {
    let expert_bytes = gate_up_expert_bytes(n_embd, n_ff_exp) as usize;
    let half = gate_up_half_offset(n_embd, n_ff_exp, is_up) as usize;
    let half_bytes = expert_bytes / 2;
    let all = gate_up_exps.as_bytes();
    let base = expert * expert_bytes + half;
    &all[base..base + half_bytes]
}

fn expert_down_bytes<'a>(
    down_exps: &'a BufferView,
    expert: usize,
    n_ff_exp: usize,
    n_embd: usize,
) -> &'a [u8] {
    let bpb = down_block_bytes(down_exps.format);
    let stride = down_expert_bytes(n_ff_exp, n_embd, bpb) as usize;
    let all = down_exps.as_bytes();
    let base = expert * stride;
    &all[base..base + stride]
}

/// CPU Q4_K gemv: `y[m] = W[m, k] @ x[k]`.
pub fn q4_k_gemv(weight_bytes: &[u8], x: &[f32], y: &mut [f32], m: usize, k: usize) {
    assert_eq!(x.len(), k);
    assert_eq!(y.len(), m);
    assert_eq!(k % 256, 0);
    let blocks_per_row = k / 256;
    let row_bytes = blocks_per_row * 144;
    assert_eq!(weight_bytes.len(), m * row_bytes);
    let mut row_f = vec![0.0f32; k];
    for row in 0..m {
        let bytes = &weight_bytes[row * row_bytes..(row + 1) * row_bytes];
        dequant_row_to_f32(ggml_type::Q4_K, bytes, k, &mut row_f);
        let mut acc = 0.0f32;
        for i in 0..k {
            acc += row_f[i] * x[i];
        }
        y[row] = acc;
    }
}

fn gelu_mul_cpu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_560_8;
    const GELU_COEF_A: f32 = 0.044_715;
    for i in 0..gate.len() {
        let x = gate[i];
        let gelu = 0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)).tanh());
        out[i] = gelu * up[i];
    }
}

/// CPU Q5_1 gemv: `y[n_embd] = scale * W[n_embd, n_ff] @ x[n_ff]` (overwrite).
pub fn q5_1_gemv_expert(
    down_bytes: &[u8],
    x: &[f32],
    y: &mut [f32],
    n_ff_exp: usize,
    n_embd: usize,
    scale: f32,
) {
    assert_eq!(x.len(), n_ff_exp);
    assert_eq!(y.len(), n_embd);
    let expected = down_expert_bytes(n_ff_exp, n_embd, 24) as usize;
    assert_eq!(down_bytes.len(), expected);
    let blocks_per_row = n_ff_exp / 32;
    let row_bytes = blocks_per_row * 24;
    for m in 0..n_embd {
        let row = &down_bytes[m * row_bytes..(m + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let base = b * 24;
            let d = f16_to_f32(u16::from_le_bytes([row[base], row[base + 1]]));
            let minv = f16_to_f32(u16::from_le_bytes([row[base + 2], row[base + 3]]));
            let qh = u32::from_le_bytes([
                row[base + 4],
                row[base + 5],
                row[base + 6],
                row[base + 7],
            ]);
            let qs = &row[base + 8..base + 24];
            let xoff = b * 32;
            for i in 0..16 {
                let xh_0 = (((qh >> (i + 0)) << 4) & 0x10) as f32;
                let xh_1 = ((qh >> (i + 12)) & 0x10) as f32;
                let x0 = (qs[i] & 0x0F) as f32 + xh_0;
                let x1 = (qs[i] >> 4) as f32 + xh_1;
                acc += (x0 * d + minv) * x[xoff + i];
                acc += (x1 * d + minv) * x[xoff + i + 16];
            }
        }
        y[m] = scale * acc;
    }
}

/// CPU Q8_0 gemv: `y[n_embd] = scale * W[n_embd, n_ff] @ x[n_ff]`.
pub fn q8_0_gemv_expert(
    down_bytes: &[u8],
    x: &[f32],
    y: &mut [f32],
    n_ff_exp: usize,
    n_embd: usize,
    scale: f32,
) {
    assert_eq!(x.len(), n_ff_exp);
    assert_eq!(y.len(), n_embd);
    let expected = down_expert_bytes(n_ff_exp, n_embd, 34) as usize;
    assert_eq!(down_bytes.len(), expected);
    let blocks_per_row = n_ff_exp / 32;
    let row_bytes = blocks_per_row * 34;
    for m in 0..n_embd {
        let row = &down_bytes[m * row_bytes..(m + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let base = b * 34;
            let d = f16_to_f32(u16::from_le_bytes([row[base], row[base + 1]]));
            let qs = &row[base + 2..base + 34];
            let xoff = b * 32;
            for i in 0..32 {
                acc += (qs[i] as i8 as f32) * d * x[xoff + i];
            }
        }
        y[m] = scale * acc;
    }
}

/// Full CPU expert FFN (for Metal cross-check only).
pub fn expert_ffn_cpu(
    gate_up_exps: &BufferView,
    down_exps: &BufferView,
    x: &[f32],
    expert: usize,
    n_embd: usize,
    n_ff_exp: usize,
    scale: f32,
    swap_gate_up: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let gate_b = expert_gate_up_bytes(gate_up_exps, expert, n_embd, n_ff_exp, swap_gate_up);
    let up_b = expert_gate_up_bytes(gate_up_exps, expert, n_embd, n_ff_exp, !swap_gate_up);
    let down_b = expert_down_bytes(down_exps, expert, n_ff_exp, n_embd);
    let mut gate = vec![0.0f32; n_ff_exp];
    let mut up = vec![0.0f32; n_ff_exp];
    let mut gelu = vec![0.0f32; n_ff_exp];
    let mut down = vec![0.0f32; n_embd];
    q4_k_gemv(gate_b, x, &mut gate, n_ff_exp, n_embd);
    q4_k_gemv(up_b, x, &mut up, n_ff_exp, n_embd);
    gelu_mul_cpu(&gate, &up, &mut gelu);
    match down_exps.format {
        crate::gpu::weight_fmt::Q5_1 => {
            q5_1_gemv_expert(down_b, &gelu, &mut down, n_ff_exp, n_embd, scale)
        }
        crate::gpu::weight_fmt::Q8_0 => {
            q8_0_gemv_expert(down_b, &gelu, &mut down, n_ff_exp, n_embd, scale)
        }
        other => panic!("unsupported MoE down format {other}"),
    }
    (gate, up, gelu, down)
}

pub fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[allow(dead_code)]
pub fn assert_q5_1_supported() {
    let _ = ggml_type::Q5_1;
    let _ = gguf::ggml_type_name(ggml_type::Q5_1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_topk_renorm_sums_to_one() {
        let logits = vec![1.0f32, 2.0, 0.5, 3.0, -1.0];
        let (idx, w) = softmax_topk_renorm(&logits, 3);
        assert_eq!(idx.len(), 3);
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        assert_eq!(idx[0], 3);
    }
}
