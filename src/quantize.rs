/// Linear layer using Accelerate BLAS for all matmuls.
/// On Apple Silicon, Accelerate's sgemv/sgemm uses the AMX coprocessor
/// which is faster than any scalar int8 kernel we could write.
/// The model weights stay in f32 for maximum BLAS throughput.

extern "C" {
    fn cblas_sgemv(
        order: i32, trans: i32, m: i32, n: i32,
        alpha: f32, a: *const f32, lda: i32,
        x: *const f32, incx: i32,
        beta: f32, y: *mut f32, incy: i32,
    );

    fn cblas_sgemm(
        order: i32, transa: i32, transb: i32,
        m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
}

/// Weight matrix stored as flat f32 with direct BLAS dispatch.
pub struct QuantizedLinear {
    pub weights: Vec<f32>,  // (out_features, in_features) row-major
    pub out_features: usize,
    pub in_features: usize,
}

impl QuantizedLinear {
    pub fn from_f32(weights: &[f32], out_features: usize, in_features: usize) -> Self {
        QuantizedLinear {
            weights: weights.to_vec(),
            out_features,
            in_features,
        }
    }

    /// Single vector: y = W * x using sgemv.
    #[inline]
    pub fn forward_vec(&self, x: &[f32], y: &mut [f32]) {
        unsafe {
            cblas_sgemv(
                101, 111, // RowMajor, NoTrans
                self.out_features as i32,
                self.in_features as i32,
                1.0,
                self.weights.as_ptr(),
                self.in_features as i32,
                x.as_ptr(), 1,
                0.0,
                y.as_mut_ptr(), 1,
            );
        }
    }

    /// Batch: C = X * W^T using sgemm.
    /// X: (m, in_features), C: (m, out_features)
    /// W: (out_features, in_features) → W^T: (in_features, out_features)
    /// C = X @ W^T = X * W^T
    /// In BLAS terms (row-major): C(m,n) = A(m,k) * B^T(n,k)
    pub fn forward_batch(&self, x: &[f32], m: usize, output: &mut [f32]) {
        if m == 1 {
            self.forward_vec(x, output);
            return;
        }
        unsafe {
            // C = alpha * A * B^T + beta * C
            // A = X: (m, k) row-major, lda = k
            // B = W: (n, k) row-major, we want B^T so transb = Trans(112)
            // C = output: (m, n) row-major, ldc = n
            let k = self.in_features as i32;
            let n = self.out_features as i32;
            cblas_sgemm(
                101,       // CblasRowMajor
                111,       // CblasNoTrans (A)
                112,       // CblasTrans (B) → computes A * B^T
                m as i32,  // M
                n,         // N
                k,         // K
                1.0,       // alpha
                x.as_ptr(),
                k,         // lda
                self.weights.as_ptr(),
                k,         // ldb (B is n×k row-major, transposed)
                0.0,       // beta
                output.as_mut_ptr(),
                n,         // ldc
            );
        }
    }
}
