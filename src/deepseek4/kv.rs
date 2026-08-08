//! DeepSeek-V4 KV caches: raw SWA (128), compressed CSA/HCA rows, indexer stream.

use super::config::Dsv4Config;

/// E5M2-ish FP8 store used for NoPE KV dims in Flash (studied from ds4 FP8 KV path).
/// We use a simple e5m2-compatible pack: store as f16 for correctness-first MVP,
/// with an optional fp8 path behind a flag once parity tests exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvStorage {
    /// Working format for development / CPU path.
    F16,
    /// Packed FP8 (e5m2) matching training graph — Metal path.
    Fp8E5M2,
}

#[derive(Debug, Clone)]
pub struct LayerKvConfig {
    pub compress_ratio: u32,
    pub has_indexer: bool,
    pub head_dim: usize,
    pub n_rot: usize,
    pub swa: usize,
}

impl LayerKvConfig {
    pub fn from_cfg(cfg: &Dsv4Config, layer: usize) -> Self {
        let compress_ratio = cfg.compress_ratio(layer);
        Self {
            compress_ratio,
            has_indexer: compress_ratio == 4,
            head_dim: cfg.head_dim,
            n_rot: cfg.n_rot,
            swa: cfg.n_swa,
        }
    }

    /// Bytes per raw KV row (K+V), f16 storage.
    pub fn raw_row_bytes_f16(&self) -> usize {
        // MQA: 1 KV head, K and V each head_dim f16.
        2 * self.head_dim * 2
    }

    pub fn compressed_capacity(&self, ctx: usize) -> usize {
        if self.compress_ratio == 0 {
            0
        } else {
            (ctx + self.compress_ratio as usize - 1) / self.compress_ratio as usize
        }
    }
}

/// Per-layer KV state (CPU-side for now; Metal buffers allocated in model).
#[derive(Debug)]
pub struct LayerKvState {
    pub cfg: LayerKvConfig,
    /// Ring buffer of raw SWA tokens (up to swa). Each entry: k[head_dim] + v[head_dim] f32 for CPU.
    pub raw_k: Vec<f32>,
    pub raw_v: Vec<f32>,
    pub raw_len: usize,
    pub raw_pos: Vec<usize>,
    /// Compressed K/V rows (f32 CPU).
    pub comp_k: Vec<f32>,
    pub comp_v: Vec<f32>,
    pub comp_len: usize,
    /// Indexer compressed stream (CSA only).
    pub idx_k: Vec<f32>,
    pub idx_v: Vec<f32>,
    pub idx_len: usize,
    /// Attention sinks (learned), length n_head (scores) — stored on model weights.
    pub token_count: usize,
}

impl LayerKvState {
    pub fn new(cfg: LayerKvConfig, max_ctx: usize) -> Self {
        let hd = cfg.head_dim;
        let swa = cfg.swa;
        let comp_cap = cfg.compressed_capacity(max_ctx).max(1);
        Self {
            raw_k: vec![0.0; swa * hd],
            raw_v: vec![0.0; swa * hd],
            raw_len: 0,
            raw_pos: vec![0; swa],
            comp_k: vec![0.0; comp_cap * hd],
            comp_v: vec![0.0; comp_cap * hd],
            comp_len: 0,
            idx_k: vec![0.0; if cfg.has_indexer { comp_cap * hd } else { 0 }],
            idx_v: vec![0.0; if cfg.has_indexer { comp_cap * hd } else { 0 }],
            idx_len: 0,
            token_count: 0,
            cfg,
        }
    }

    pub fn clear(&mut self) {
        self.raw_len = 0;
        self.comp_len = 0;
        self.idx_len = 0;
        self.token_count = 0;
    }

    /// Push one raw KV token into the SWA ring.
    pub fn push_raw(&mut self, k: &[f32], v: &[f32], pos: usize) {
        let hd = self.cfg.head_dim;
        assert_eq!(k.len(), hd);
        assert_eq!(v.len(), hd);
        if self.raw_len >= self.cfg.swa {
            for i in 1..self.cfg.swa {
                let dst = (i - 1) * hd;
                let src = i * hd;
                self.raw_k.copy_within(src..src + hd, dst);
                self.raw_v.copy_within(src..src + hd, dst);
            }
        }
        let slot = self.note_raw_push(pos);
        self.raw_k[slot * hd..(slot + 1) * hd].copy_from_slice(k);
        self.raw_v[slot * hd..(slot + 1) * hd].copy_from_slice(v);
    }

    /// Advance SWA ring metadata without copying K/V (GPU ring is source of truth).
    pub fn note_raw_push(&mut self, pos: usize) -> usize {
        let slot = if self.raw_len < self.cfg.swa {
            let s = self.raw_len;
            self.raw_len += 1;
            s
        } else {
            for i in 1..self.cfg.swa {
                self.raw_pos[i - 1] = self.raw_pos[i];
            }
            self.cfg.swa - 1
        };
        self.raw_pos[slot] = pos;
        self.token_count += 1;
        slot
    }

    /// Push one compressed KV row (K≡V for Flash MQA latent).
    pub fn push_compressed(&mut self, k: &[f32], v: &[f32]) {
        let hd = self.cfg.head_dim;
        assert_eq!(k.len(), hd);
        assert_eq!(v.len(), hd);
        let dst = self.comp_len;
        if (dst + 1) * hd > self.comp_k.len() {
            self.comp_k.resize((dst + 1) * hd, 0.0);
            self.comp_v.resize((dst + 1) * hd, 0.0);
        }
        self.comp_k[dst * hd..(dst + 1) * hd].copy_from_slice(k);
        self.comp_v[dst * hd..(dst + 1) * hd].copy_from_slice(v);
        self.comp_len += 1;
    }

    /// Bump compressed-row length after a GPU emit (no host copy).
    pub fn note_comp_push(&mut self) {
        self.comp_len += 1;
    }

    /// Mean-pool fallback (kept for tests). Production path uses `compressor_step`.
    pub fn maybe_emit_mean_compressed(&mut self) {
        let ratio = self.cfg.compress_ratio as usize;
        if ratio == 0 || self.raw_len < ratio {
            return;
        }
        if self.token_count % ratio != 0 {
            return;
        }
        let hd = self.cfg.head_dim;
        let start = self.raw_len - ratio;
        let mut k = vec![0.0f32; hd];
        let mut v = vec![0.0f32; hd];
        for t in 0..ratio {
            let off = (start + t) * hd;
            for d in 0..hd {
                k[d] += self.raw_k[off + d];
                v[d] += self.raw_v[off + d];
            }
        }
        let inv = 1.0 / ratio as f32;
        for d in 0..hd {
            k[d] *= inv;
            v[d] *= inv;
        }
        self.push_compressed(&k, &v);
    }
}

/// Select top-k compressed row indices by scores (CSA indexer).
pub fn indexer_topk(scores: &[f32], k: usize) -> Vec<usize> {
    let n = scores.len();
    if n == 0 {
        return vec![];
    }
    let k = k.min(n);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.select_nth_unstable_by(k - 1, |&a, &b| {
        scores[b].partial_cmp(&scores[a]).unwrap()
    });
    idx.truncate(k);
    idx.sort_unstable();
    idx
}

/// Pack f32 → e5m2 byte (approx). Used by Metal KV path tests.
pub fn f32_to_fp8_e5m2(x: f32) -> u8 {
    // Clamp to e5m2 range roughly ±57344.
    let x = x.clamp(-57344.0, 57344.0);
    if x == 0.0 {
        return 0;
    }
    let bits = x.to_bits();
    let sign = ((bits >> 31) as u8) << 7;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    let e = exp - 127 + 15;
    if e <= 0 {
        return sign;
    }
    if e >= 31 {
        return sign | 0x7c; // max finite-ish
    }
    let m = (mant >> 21) as u8; // 2 bits
    sign | ((e as u8) << 2) | m
}

pub fn fp8_e5m2_to_f32(b: u8) -> f32 {
    let sign = (b >> 7) & 1;
    let exp = ((b >> 2) & 0x1f) as i32;
    let mant = (b & 0x3) as u32;
    if exp == 0 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }
    let e = (exp - 15 + 127) as u32;
    let bits = ((sign as u32) << 31) | (e << 23) | (mant << 21);
    f32::from_bits(bits)
}
