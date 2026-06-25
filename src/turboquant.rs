//! TurboQuant KV-cache compression.
//!
//! TurboQuant (Google, ICLR 2026 — <https://arxiv.org/abs/2504.19874>) compresses
//! the KV cache by multiplying every key/value vector with a fixed **Haar-random
//! orthogonal rotation** before quantizing it. After the rotation each coordinate
//! is (approximately) i.i.d. Gaussian, which spreads "outlier channels" evenly and
//! makes cheap per-block scalar quantization near information-theoretically optimal.
//!
//! The decisive property is that a rotation preserves inner products:
//!
//! ```text
//!   q · k = (R q) · (R k)        for any orthogonal R
//! ```
//!
//! so we can store the **rotated** key (quantized) and, at attention time, rotate
//! only the query once — the dot products come out identical. For values the
//! output is accumulated in the rotated frame and rotated back once per head.
//!
//! This module implements the *fast* variant used by the reference MLX project
//! ("V2 rotated"): **rotation + Q4_0**. The rotated vectors are stored in the
//! existing Q4_0 block layout (18 bytes / 32 weights), so all of the heavily
//! optimized Q4_0 append + flash-attention kernels are reused unchanged. The only
//! added work is a small `head_dim × head_dim` rotation matmul for Q/K/V and a
//! single un-rotation of the attention output.
//!
//! The studied reference also describes a higher-quality "V3" path (per-coordinate
//! Lloyd–Max codebook instead of affine Q4_0). The Lloyd–Max solver is provided
//! here ([`lloyd_max_centroids`]) for that future path; the runtime currently uses
//! the faster affine-Q4_0 storage.

use metal::{Buffer, Device, MTLResourceOptions};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use std::collections::HashMap;

/// Base RNG seed for rotation-matrix generation. Fixed so a cache produced in one
/// run can be read back identically in another (rotations must match end-to-end).
const ROTATION_SEED: u64 = 0x7401_B0CA_5EED_2026;

/// Generate a `dim × dim` Haar-distributed random orthogonal matrix `R`
/// (row-major), via modified Gram–Schmidt of a Gaussian matrix.
///
/// The columns of a matrix with i.i.d. N(0,1) entries, once orthonormalized, are
/// uniformly (Haar) distributed on the orthogonal group. We orthonormalize the
/// columns and write `R` row-major. The sign of each column is fixed to be
/// deterministic (first non-negligible component positive).
fn generate_rotation(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);

    // Column-major scratch: cols[c] is the c-th column (length `dim`).
    let mut cols: Vec<Vec<f64>> = (0..dim)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    let x: f64 = StandardNormal.sample(&mut rng);
                    x
                })
                .collect()
        })
        .collect();

    // Modified Gram–Schmidt orthonormalization of the columns.
    for c in 0..dim {
        // Subtract projections onto previously finalized columns.
        for p in 0..c {
            let mut dot = 0.0f64;
            for r in 0..dim {
                dot += cols[c][r] * cols[p][r];
            }
            for r in 0..dim {
                cols[c][r] -= dot * cols[p][r];
            }
        }
        // Normalize.
        let mut norm = 0.0f64;
        for r in 0..dim {
            norm += cols[c][r] * cols[c][r];
        }
        norm = norm.sqrt();
        if norm < 1e-12 {
            // Degenerate (astronomically unlikely): fall back to a basis vector.
            for r in 0..dim {
                cols[c][r] = if r == c { 1.0 } else { 0.0 };
            }
        } else {
            let inv = 1.0 / norm;
            for r in 0..dim {
                cols[c][r] *= inv;
            }
        }
        // Deterministic sign fix.
        let mut sign = 1.0f64;
        for r in 0..dim {
            if cols[c][r].abs() > 1e-9 {
                sign = cols[c][r].signum();
                break;
            }
        }
        if sign < 0.0 {
            for r in 0..dim {
                cols[c][r] = -cols[c][r];
            }
        }
    }

    // Emit R row-major: R[i][j] = cols[j][i].
    let mut r = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            r[i * dim + j] = cols[j][i] as f32;
        }
    }
    r
}

/// Per-`head_dim` rotation matrices, resident on the GPU.
///
/// The rotation kernel computes `Y = X @ M` (row-major, `M[k*dim + c]`):
/// * `fwd` = Rᵀ — applies the forward rotation to a **row** vector
///   (`y_row = x_row @ Rᵀ`, i.e. `y = R·x`). Used for Q, K, V before storage.
/// * `inv` = R — un-rotates a row vector (`y_row = x_row @ R`, i.e. `y = Rᵀ·õ`).
///   Used on the attention output to return to the original frame.
struct RotationMatrices {
    fwd: Buffer,
    inv: Buffer,
    /// Lloyd–Max centroids for this head dim at the active bit-width (V3 path).
    /// `2^bits` f32 values scaled for unit-vector coordinates (std 1/sqrt(dim)).
    centroids: Buffer,
}

/// Holds the GPU rotation matrices (and Lloyd–Max codebooks) for every distinct
/// head dimension in the model.
pub struct TurboQuant {
    by_dim: HashMap<usize, RotationMatrices>,
    bits: u8,
}

impl TurboQuant {
    /// Build rotation matrices + Lloyd–Max codebooks for each distinct head
    /// dimension. `bits` is the active KV bit-width (2, 3, or 4).
    pub fn new(device: &Device, head_dims: &[usize], bits: u8) -> Self {
        let mut by_dim = HashMap::new();
        for &dim in head_dims {
            if by_dim.contains_key(&dim) {
                continue;
            }
            // Distinct seed per dimension so different head sizes get independent
            // rotations (avoids accidental structure shared across dims).
            let seed = ROTATION_SEED ^ (dim as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let r = generate_rotation(dim, seed);

            // fwd = Rᵀ : fwd[k*dim + c] = R[c*dim + k]
            let mut fwd = vec![0.0f32; dim * dim];
            for k in 0..dim {
                for c in 0..dim {
                    fwd[k * dim + c] = r[c * dim + k];
                }
            }

            // Lloyd–Max codebook for a unit vector's coordinate after rotation:
            // std = 1/sqrt(dim). Used by the 2/3-bit V3 path.
            let sigma = 1.0 / (dim as f64).sqrt();
            let centroids = lloyd_max_centroids(bits as u32, sigma, 200_000);

            // Rotation matrices are stored as f32. Inner-product preservation
            // (q·k = (Rq)·(Rk)) relies on R being exactly orthogonal; fp16 erodes
            // that by ~sqrt(dim)·1e-3, which is fine for 3/4-bit but tips 2-bit over
            // at long contexts — so we keep full precision for all bit-widths.
            let fwd_buf = Self::upload(device, &fwd);
            let inv_buf = Self::upload(device, &r);
            let cen_buf = Self::upload(device, &centroids);
            by_dim.insert(
                dim,
                RotationMatrices {
                    fwd: fwd_buf,
                    inv: inv_buf,
                    centroids: cen_buf,
                },
            );
        }
        TurboQuant { by_dim, bits }
    }

    /// Active KV bit-width.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Lloyd–Max codebook buffer (`2^bits` f32) for the given head dimension.
    pub fn centroids(&self, head_dim: usize) -> &Buffer {
        &self
            .by_dim
            .get(&head_dim)
            .unwrap_or_else(|| panic!("TurboQuant: no codebook for head_dim={head_dim}"))
            .centroids
    }

    fn upload(device: &Device, data: &[f32]) -> Buffer {
        let byte_len = std::mem::size_of_val(data) as u64;
        device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Forward-rotation matrix (Rᵀ) for the given head dimension. Apply to Q, K, V.
    pub fn fwd(&self, head_dim: usize) -> &Buffer {
        &self
            .by_dim
            .get(&head_dim)
            .unwrap_or_else(|| panic!("TurboQuant: no rotation matrix for head_dim={head_dim}"))
            .fwd
    }

    /// Inverse-rotation matrix (R) for the given head dimension. Apply to attn out.
    pub fn inv(&self, head_dim: usize) -> &Buffer {
        &self
            .by_dim
            .get(&head_dim)
            .unwrap_or_else(|| panic!("TurboQuant: no rotation matrix for head_dim={head_dim}"))
            .inv
    }
}

/// Solve the Lloyd–Max optimal scalar quantizer for a zero-mean unit-variance
/// Gaussian (the per-coordinate distribution after rotating a unit vector), then
/// scale to std `sigma`. Returns the `2^bits` sorted centroids.
///
/// This is the codebook for the TurboQuant "V3" path (non-uniform centroids
/// instead of affine Q4_0), used by the 2/3-bit KV cache to stay coherent at
/// compression ratios Q4_0 cannot reach.
pub fn lloyd_max_centroids(bits: u32, sigma: f64, samples: usize) -> Vec<f32> {
    let n_levels = 1usize << bits;
    // Deterministic Gaussian sample set (k-means in 1-D over the distribution).
    let mut rng = StdRng::seed_from_u64(0x110D_4A0C_0DEB_0042 ^ (bits as u64));
    let mut data: Vec<f64> = (0..samples)
        .map(|_| {
            let x: f64 = StandardNormal.sample(&mut rng);
            x
        })
        .collect();
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Initialize centroids at evenly spaced quantiles.
    let mut centroids: Vec<f64> = (0..n_levels)
        .map(|i| {
            let q = (i as f64 + 0.5) / n_levels as f64;
            let idx = ((q * samples as f64) as usize).min(samples - 1);
            data[idx]
        })
        .collect();

    for _ in 0..100 {
        // Assign + accumulate.
        let mut sums = vec![0.0f64; n_levels];
        let mut counts = vec![0usize; n_levels];
        for &x in &data {
            // Nearest centroid (centroids stay sorted).
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for (i, &c) in centroids.iter().enumerate() {
                let d = (x - c).abs();
                if d < best_d {
                    best_d = d;
                    best = i;
                }
            }
            sums[best] += x;
            counts[best] += 1;
        }
        let mut max_shift = 0.0f64;
        for i in 0..n_levels {
            if counts[i] > 0 {
                let nc = sums[i] / counts[i] as f64;
                max_shift = max_shift.max((nc - centroids[i]).abs());
                centroids[i] = nc;
            }
        }
        if max_shift < 1e-9 {
            break;
        }
    }

    centroids.iter().map(|&c| (c * sigma) as f32).collect()
}
