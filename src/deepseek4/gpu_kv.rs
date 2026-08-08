//! Per-layer GPU KV / compressor residency for DeepSeek-V4 Flash.

use super::compressor::CompressorState;
use super::kv::LayerKvConfig;
use super::metal_ctx::Dsv4Metal;
use metal::Buffer;

pub struct GpuLayerKv {
    pub raw: Buffer,
    pub comp: Buffer,
    pub comp_cap: usize,
    pub state_kv: Option<Buffer>,
    pub state_score: Option<Buffer>,
    pub idx_state_kv: Option<Buffer>,
    pub idx_state_score: Option<Buffer>,
    pub ratio: u32,
    pub width: usize,
    pub idx_width: usize,
}

impl GpuLayerKv {
    pub fn new(
        metal: &Dsv4Metal,
        cfg: &LayerKvConfig,
        indexer_head_dim: usize,
        max_ctx: usize,
    ) -> Self {
        let hd = cfg.head_dim;
        let swa = cfg.swa;
        let comp_cap = cfg.compressed_capacity(max_ctx).max(1);
        let (state_kv, state_score, width, ratio) =
            if let Some(st) = CompressorState::new(cfg.compress_ratio, hd) {
                (
                    Some(metal.buffer_zeros(st.state_kv.len() * 4)),
                    Some(metal.buffer_zeros(st.state_score.len() * 4)),
                    st.width,
                    st.ratio,
                )
            } else {
                (None, None, 0, 0)
            };
        let (idx_state_kv, idx_state_score, idx_width) = if cfg.has_indexer {
            if let Some(st) = CompressorState::new(cfg.compress_ratio, indexer_head_dim) {
                (
                    Some(metal.buffer_zeros(st.state_kv.len() * 4)),
                    Some(metal.buffer_zeros(st.state_score.len() * 4)),
                    st.width,
                )
            } else {
                (None, None, 0)
            }
        } else {
            (None, None, 0)
        };
        Self {
            raw: metal.buffer_zeros(swa * hd * 4),
            comp: metal.buffer_zeros(comp_cap * hd * 4),
            comp_cap,
            state_kv,
            state_score,
            idx_state_kv,
            idx_state_score,
            ratio,
            width,
            idx_width,
        }
    }

    pub fn clear(&self) {
        let zero = |b: &Buffer| {
            let n = b.length() as usize;
            unsafe {
                std::ptr::write_bytes(b.contents() as *mut u8, 0, n);
            }
        };
        zero(&self.raw);
        zero(&self.comp);
        if let Some(b) = self.state_kv.as_ref() {
            zero(b);
        }
        if let Some(b) = self.state_score.as_ref() {
            zero(b);
        }
        if let Some(b) = self.idx_state_kv.as_ref() {
            zero(b);
        }
        if let Some(b) = self.idx_state_score.as_ref() {
            zero(b);
        }
    }
}
