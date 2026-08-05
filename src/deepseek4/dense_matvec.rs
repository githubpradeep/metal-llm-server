//! F16 / Q8 matvec helpers for dense DeepSeek weights kept in native storage.

use crate::gpu::f16_to_f32;

pub fn matvec_f16_weights(weight_f16: &[u16], x: &[f32], n_out: usize, n_in: usize, out: &mut [f32]) {
    // Accept [n_out, n_in] row-major (each output row contiguous of length n_in).
    assert_eq!(
        weight_f16.len(),
        n_out * n_in,
        "f16 weight len {} != {}*{} (out*in)",
        weight_f16.len(),
        n_out,
        n_in
    );
    for r in 0..n_out {
        let mut acc = 0.0f32;
        let row = &weight_f16[r * n_in..(r + 1) * n_in];
        for i in 0..n_in {
            acc += f16_to_f32(row[i]) * x[i];
        }
        out[r] = acc;
    }
}

pub fn bytes_to_f16_vec(bytes: &[u8]) -> Vec<u16> {
    assert_eq!(bytes.len() % 2, 0);
    let mut v = Vec::with_capacity(bytes.len() / 2);
    for i in 0..bytes.len() / 2 {
        v.push(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]));
    }
    v
}

pub fn matvec_q8_0(weight: &[u8], x: &[f32], n_out: usize, n_in: usize, out: &mut [f32]) {
    assert_eq!(n_in % 32, 0, "q8 n_in must be multiple of 32, got {n_in}");
    let blocks_per_row = n_in / 32;
    let row_bytes = blocks_per_row * 34;
    let expected = n_out * row_bytes;
    if weight.len() != expected {
        // Some tensors are stored with the opposite leading dim; try swapped.
        if n_out % 32 == 0 {
            let alt_blocks = n_out / 32;
            let alt_row = alt_blocks * 34;
            let alt = n_in * alt_row;
            if weight.len() == alt {
                // Weight is [n_in, n_out] — do transposed matvec: out[r] = sum_c W[c,r]*x[c]
                // Too slow / different layout; panic with guidance.
                panic!(
                    "q8 weight appears transposed: len={} for out={} in={}; expected row-major out-major",
                    weight.len(),
                    n_out,
                    n_in
                );
            }
        }
        panic!(
            "q8 weight len {} != expected {} (out={} in={} row_bytes={})",
            weight.len(),
            expected,
            n_out,
            n_in,
            row_bytes
        );
    }
    for r in 0..n_out {
        let mut acc = 0.0f32;
        let row = &weight[r * row_bytes..(r + 1) * row_bytes];
        for b in 0..blocks_per_row {
            let base = b * 34;
            let d = f16_to_f32(u16::from_le_bytes([row[base], row[base + 1]]));
            let qs = &row[base + 2..base + 34];
            let xoff = b * 32;
            for i in 0..32 {
                acc += (qs[i] as i8 as f32) * d * x[xoff + i];
            }
        }
        out[r] = acc;
    }
}

/// Infer (n_out, n_in) from a dense weight element count and the activation size.
pub fn infer_out_in(weight_elems: usize, n_in: usize) -> usize {
    assert!(n_in > 0 && weight_elems % n_in == 0);
    weight_elems / n_in
}
