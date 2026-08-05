//! CPU dequant for IQ2_XXS and Q2_K (ggml block layouts).
//!
//! Lookup tables are the public ggml IQ2_XXS grid/sign tables (quant format
//! constants, not a ds4 kernel). Dequant math follows the ggml/IQ2_XXS spec
//! studied via ds4's usage.

use crate::gpu::f16_to_f32;

pub const QK_K: usize = 256;
pub const IQ2_XXS_BLOCK_BYTES: usize = 66; // half d + 32×u16
pub const Q2_K_BLOCK_BYTES: usize = 84; // scales[16] + qs[64] + d + dmin

/// ggml IQ2_XXS sign bit masks (8 lanes).
pub static KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// ggml IQ2_XXS sign table (128 entries).
pub static KSIGNS_IQ2XS: [u8; 128] = [
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149,
    150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170,
    43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212,
    85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232,
    105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123,
    252, 125, 126, 255,
];

/// ggml IQ2_XXS grid: 256 × 8 packed uint8 values as little-endian u64.
pub static IQ2XXS_GRID: [u64; 256] = [
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
];

#[inline]
fn grid_bytes(idx: u8) -> [u8; 8] {
    IQ2XXS_GRID[idx as usize].to_le_bytes()
}

/// Dequantize one IQ2_XXS block (256 values) into `out`.
/// Matches ds4 `ds4_vec_dot_iq2_xxs_f32` / ggml: 8×32 groups, each 4×8 grids.
pub fn dequant_iq2_xxs_block(block: &[u8], out: &mut [f32]) {
    assert!(block.len() >= IQ2_XXS_BLOCK_BYTES);
    assert!(out.len() >= QK_K);
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..2 + 64]; // 32 × u16

    for ib32 in 0..8 {
        let q2 = &qs[ib32 * 8..ib32 * 8 + 8];
        let aux32_g = u32::from_le_bytes([q2[0], q2[1], q2[2], q2[3]]);
        let aux32_s = u32::from_le_bytes([q2[4], q2[5], q2[6], q2[7]]);
        let aux8 = aux32_g.to_le_bytes();
        // scale = 0.125 * d * (2*(aux>>28) + 1) == d*(0.5+(aux>>28))*0.25
        let scale = 0.125 * d * (2.0 * ((aux32_s >> 28) as f32) + 1.0);
        let base = ib32 * 32;
        for l in 0..4 {
            let grid = grid_bytes(aux8[l]);
            let sign_idx = ((aux32_s >> (7 * l)) & 127) as usize;
            let signs = KSIGNS_IQ2XS[sign_idx];
            for j in 0..8 {
                let sign = if signs & KMASK_IQ2XS[j] != 0 {
                    -1.0
                } else {
                    1.0
                };
                out[base + l * 8 + j] = scale * (grid[j] as f32) * sign;
            }
        }
    }
}

/// Dequantize one Q2_K block (256 values).
/// Layout matches ggml `block_q2_K` / ds4 `q2_k_value_f32`:
/// `scales[16] + qs[64] + half d + half dmin`.
pub fn dequant_q2_k_block(block: &[u8], out: &mut [f32]) {
    assert!(block.len() >= Q2_K_BLOCK_BYTES);
    assert!(out.len() >= QK_K);
    let scales = &block[0..16];
    let qs = &block[16..80];
    let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));

    for idx in 0..QK_K {
        let group = idx / 16;
        let l = idx % 16;
        let q_base = 32 * (group / 8) + 16 * (group & 1);
        let shift = ((group / 2) & 3) * 2;
        let q = ((qs[q_base + l] as u32) >> shift) & 0x03;
        let sc = scales[group];
        out[idx] = d * ((sc & 0x0f) as f32) * (q as f32) - dmin * ((sc >> 4) as f32);
    }
}

/// Dot product of a dequantized IQ2_XXS weight row (n elems, n % 256 == 0) with x.
pub fn matvec_iq2_xxs_row(weight: &[u8], x: &[f32], n: usize) -> f32 {
    assert_eq!(n % QK_K, 0);
    assert_eq!(x.len(), n);
    let nblocks = n / QK_K;
    assert_eq!(weight.len(), nblocks * IQ2_XXS_BLOCK_BYTES);
    let mut acc = 0.0f32;
    let mut tmp = [0.0f32; QK_K];
    for b in 0..nblocks {
        let off = b * IQ2_XXS_BLOCK_BYTES;
        dequant_iq2_xxs_block(&weight[off..off + IQ2_XXS_BLOCK_BYTES], &mut tmp);
        let xoff = b * QK_K;
        for i in 0..QK_K {
            acc += tmp[i] * x[xoff + i];
        }
    }
    acc
}

pub fn matvec_q2_k_row(weight: &[u8], x: &[f32], n: usize) -> f32 {
    assert_eq!(n % QK_K, 0);
    assert_eq!(x.len(), n);
    let nblocks = n / QK_K;
    assert_eq!(weight.len(), nblocks * Q2_K_BLOCK_BYTES);
    let mut acc = 0.0f32;
    let mut tmp = [0.0f32; QK_K];
    for b in 0..nblocks {
        let off = b * Q2_K_BLOCK_BYTES;
        dequant_q2_k_block(&weight[off..off + Q2_K_BLOCK_BYTES], &mut tmp);
        let xoff = b * QK_K;
        for i in 0..QK_K {
            acc += tmp[i] * x[xoff + i];
        }
    }
    acc
}

/// Full matvec: weight is `[n_out, n_in]` row-major quantized blocks.
pub fn matvec_iq2_xxs(weight: &[u8], x: &[f32], n_out: usize, n_in: usize, out: &mut [f32]) {
    assert_eq!(out.len(), n_out);
    let row_bytes = (n_in / QK_K) * IQ2_XXS_BLOCK_BYTES;
    for r in 0..n_out {
        out[r] = matvec_iq2_xxs_row(&weight[r * row_bytes..(r + 1) * row_bytes], x, n_in);
    }
}

pub fn matvec_q2_k(weight: &[u8], x: &[f32], n_out: usize, n_in: usize, out: &mut [f32]) {
    assert_eq!(out.len(), n_out);
    let row_bytes = (n_in / QK_K) * Q2_K_BLOCK_BYTES;
    for r in 0..n_out {
        out[r] = matvec_q2_k_row(&weight[r * row_bytes..(r + 1) * row_bytes], x, n_in);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iq2_xxs_block_bytes_match_spec() {
        assert_eq!(IQ2_XXS_BLOCK_BYTES, 66);
        assert_eq!(Q2_K_BLOCK_BYTES, 84);
        assert_eq!(QK_K, 256);
    }

    #[test]
    fn iq2_matvec_zero_input_is_zero() {
        let n_in = 256;
        let n_out = 2;
        let row_bytes = (n_in / QK_K) * IQ2_XXS_BLOCK_BYTES;
        let weight = vec![0u8; n_out * row_bytes];
        let x = vec![0.0f32; n_in];
        let mut out = vec![1.0f32; n_out];
        matvec_iq2_xxs(&weight, &x, n_out, n_in, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }
}
