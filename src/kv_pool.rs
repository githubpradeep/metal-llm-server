use metal::{Buffer, MTLResourceOptions};
use std::fmt;

use crate::gemma4_config::{Gemma4TextConfig, KvCacheType};
use crate::gpu::MetalContext;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KvSlot(usize);

impl KvSlot {
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug)]
pub enum KvPoolError {
    InvalidSlot(usize),
    InvalidLayer { layer: usize, num_layers: usize },
    SlotNotAllocated(usize),
}

impl fmt::Display for KvPoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KvPoolError::InvalidSlot(slot) => write!(f, "invalid KV slot: {}", slot),
            KvPoolError::InvalidLayer { layer, num_layers } => {
                write!(f, "invalid KV layer: {} (num_layers={})", layer, num_layers)
            }
            KvPoolError::SlotNotAllocated(slot) => write!(f, "KV slot is not allocated: {}", slot),
        }
    }
}

impl std::error::Error for KvPoolError {}

/// Optional TurboQuant per-slot rings allocated with the pool (serve / multi-request).
#[derive(Clone, Copy, Debug, Default)]
pub struct TqPoolConfig {
    /// Model-frame Q4 hot window length (0 = disabled).
    pub hot_w: u32,
    /// Residual fp32 window length (0 = disabled).
    pub rw: u32,
}

pub struct KvCachePool {
    slots: Vec<KvCacheSlot>,
    free_slots: Vec<usize>,
    max_seq_len: u32,
    num_layers: usize,
    pub kv_cache_type: KvCacheType,
    /// TurboQuant Q4 hot capacity (0 if not allocated).
    pub tq_hot_w: u32,
    /// TurboQuant residual window capacity (0 if not allocated).
    pub tq_rw: u32,
}

pub struct KvCacheSlot {
    pub k_cache: Vec<Buffer>,
    pub v_cache: Vec<Buffer>,
    /// Per-layer Q4 hot rings (empty when TQ hot disabled).
    pub tq_hot_k: Vec<Buffer>,
    pub tq_hot_v: Vec<Buffer>,
    /// Per-layer residual fp32 windows (empty when `tq_rw == 0`).
    pub tq_rw_k: Vec<Buffer>,
    pub tq_rw_v: Vec<Buffer>,
    pub tq_hot_spilled: bool,
    pub seq_len: u32,
    pub total_tokens: usize,
    in_use: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct KvSlotView {
    pub slot: KvSlot,
    pub slot_index: usize,
    pub seq_len: u32,
    pub total_tokens: usize,
}

impl KvCachePool {
    /// Create a one-slot adapter over an existing model-owned KV cache.
    ///
    /// Metal buffers are reference-counted handles, so cloning them aliases the
    /// same storage. Used by MTP verify to run the parallel prefill kernels
    /// against the live decode cache without a full copy.
    ///
    /// Optional `tq_hot` / `tq_rw` alias the model's TurboQuant rings so serve
    /// and CLI verify share the same storage during the swap-free path.
    pub(crate) fn from_existing(
        k_cache: &[Buffer],
        v_cache: &[Buffer],
        seq_len: u32,
        total_tokens: usize,
        max_seq_len: u32,
        kv_cache_type: KvCacheType,
        tq: TqPoolConfig,
        tq_hot: Option<(&[Buffer], &[Buffer])>,
        tq_rw: Option<(&[Buffer], &[Buffer])>,
        tq_hot_spilled: bool,
    ) -> (Self, KvSlot) {
        assert_eq!(k_cache.len(), v_cache.len());
        let (tq_hot_k, tq_hot_v) = match tq_hot {
            Some((k, v)) => {
                assert_eq!(k.len(), v.len());
                assert_eq!(k.len(), k_cache.len());
                (k.to_vec(), v.to_vec())
            }
            None => (Vec::new(), Vec::new()),
        };
        let (tq_rw_k, tq_rw_v) = match tq_rw {
            Some((k, v)) => {
                assert_eq!(k.len(), v.len());
                assert_eq!(k.len(), k_cache.len());
                (k.to_vec(), v.to_vec())
            }
            None => (Vec::new(), Vec::new()),
        };
        let slot = KvSlot(0);
        (
            Self {
                slots: vec![KvCacheSlot {
                    k_cache: k_cache.to_vec(),
                    v_cache: v_cache.to_vec(),
                    tq_hot_k,
                    tq_hot_v,
                    tq_rw_k,
                    tq_rw_v,
                    tq_hot_spilled,
                    seq_len,
                    total_tokens,
                    in_use: true,
                }],
                free_slots: Vec::new(),
                max_seq_len,
                num_layers: k_cache.len(),
                kv_cache_type,
                tq_hot_w: tq.hot_w,
                tq_rw: tq.rw,
            },
            slot,
        )
    }

    pub fn new(
        ctx: &MetalContext,
        config: &Gemma4TextConfig,
        num_slots: usize,
        max_seq_len: u32,
        kv_cache_type: KvCacheType,
        tq: TqPoolConfig,
    ) -> Self {
        let num_layers = config.num_hidden_layers;
        let want_hot = tq.hot_w > 0
            && matches!(kv_cache_type, KvCacheType::TurboQuant { .. })
            && !kv_cache_type.tq_affine();
        let hot_w = if want_hot { tq.hot_w } else { 0 };
        let want_rw = tq.rw > 0
            && matches!(kv_cache_type, KvCacheType::TurboQuant { .. })
            && !kv_cache_type.tq_affine();
        let rw = if want_rw { tq.rw } else { 0 };

        let mut slots = Vec::with_capacity(num_slots);
        for _ in 0..num_slots {
            let mut k_cache = Vec::with_capacity(num_layers);
            let mut v_cache = Vec::with_capacity(num_layers);
            let mut tq_hot_k = Vec::with_capacity(if hot_w > 0 { num_layers } else { 0 });
            let mut tq_hot_v = Vec::with_capacity(if hot_w > 0 { num_layers } else { 0 });
            let mut tq_rw_k = Vec::with_capacity(if rw > 0 { num_layers } else { 0 });
            let mut tq_rw_v = Vec::with_capacity(if rw > 0 { num_layers } else { 0 });

            for layer_idx in 0..num_layers {
                let head_dim = config.layer_head_dim(layer_idx);
                assert!(
                    head_dim % 32 == 0,
                    "head_dim must be multiple of 32 for quantized KV cache"
                );
                let num_kv_heads = config.layer_num_kv_heads(layer_idx);
                let k_byte_len =
                    (num_kv_heads * max_seq_len as usize * kv_cache_type.k_row_bytes(head_dim))
                        as u64;
                let v_byte_len =
                    (num_kv_heads * max_seq_len as usize * kv_cache_type.v_row_bytes(head_dim))
                        as u64;
                k_cache.push(
                    ctx.device
                        .new_buffer(k_byte_len.max(1), MTLResourceOptions::StorageModeShared),
                );
                v_cache.push(
                    ctx.device
                        .new_buffer(v_byte_len.max(1), MTLResourceOptions::StorageModeShared),
                );

                if hot_w > 0 {
                    let q4_row = (head_dim / 32) * 18;
                    let hot_bytes =
                        (num_kv_heads * hot_w as usize * q4_row).max(1) as u64;
                    tq_hot_k.push(
                        ctx.device
                            .new_buffer(hot_bytes, MTLResourceOptions::StorageModeShared),
                    );
                    tq_hot_v.push(
                        ctx.device
                            .new_buffer(hot_bytes, MTLResourceOptions::StorageModeShared),
                    );
                }

                if rw > 0 {
                    let rw_bytes =
                        (num_kv_heads * rw as usize * head_dim * std::mem::size_of::<f32>())
                            .max(1) as u64;
                    tq_rw_k.push(
                        ctx.device
                            .new_buffer(rw_bytes, MTLResourceOptions::StorageModeShared),
                    );
                    tq_rw_v.push(
                        ctx.device
                            .new_buffer(rw_bytes, MTLResourceOptions::StorageModeShared),
                    );
                }
            }

            slots.push(KvCacheSlot {
                k_cache,
                v_cache,
                tq_hot_k,
                tq_hot_v,
                tq_rw_k,
                tq_rw_v,
                tq_hot_spilled: false,
                seq_len: 0,
                total_tokens: 0,
                in_use: false,
            });
        }

        if hot_w > 0 && num_slots > 0 {
            let hot_bytes: u64 = slots[0]
                .tq_hot_k
                .iter()
                .map(|b| b.length())
                .sum::<u64>()
                + slots[0].tq_hot_v.iter().map(|b| b.length()).sum::<u64>();
            println!(
                "  KV pool TurboQuant: {} slots × Q4 hot {} tok (~{:.1} MB/slot)",
                num_slots,
                hot_w,
                hot_bytes as f64 / 1e6
            );
        }

        let free_slots = (0..num_slots).rev().collect();

        Self {
            slots,
            free_slots,
            max_seq_len,
            num_layers,
            kv_cache_type,
            tq_hot_w: hot_w,
            tq_rw: rw,
        }
    }

    pub fn allocate(&mut self) -> Option<KvSlot> {
        let slot_idx = self.free_slots.pop()?;
        let slot = &mut self.slots[slot_idx];
        slot.in_use = true;
        slot.seq_len = 0;
        slot.total_tokens = 0;
        slot.tq_hot_spilled = false;
        Some(KvSlot(slot_idx))
    }

    pub fn release(&mut self, slot: KvSlot) -> Result<(), KvPoolError> {
        let slot_idx = slot.index();
        let slot = self
            .slots
            .get_mut(slot_idx)
            .ok_or(KvPoolError::InvalidSlot(slot_idx))?;

        if !slot.in_use {
            return Err(KvPoolError::SlotNotAllocated(slot_idx));
        }

        slot.in_use = false;
        slot.seq_len = 0;
        slot.total_tokens = 0;
        slot.tq_hot_spilled = false;
        self.free_slots.push(slot_idx);
        Ok(())
    }

    pub fn reset(&mut self, slot: KvSlot) -> Result<(), KvPoolError> {
        let slot = self.slot_mut(slot)?;
        slot.seq_len = 0;
        slot.total_tokens = 0;
        slot.tq_hot_spilled = false;
        Ok(())
    }

    pub fn seq_len(&self, slot: KvSlot) -> Result<u32, KvPoolError> {
        Ok(self.slot(slot)?.seq_len)
    }

    pub fn total_tokens(&self, slot: KvSlot) -> Result<usize, KvPoolError> {
        Ok(self.slot(slot)?.total_tokens)
    }

    pub fn tq_hot_spilled(&self, slot: KvSlot) -> Result<bool, KvPoolError> {
        Ok(self.slot(slot)?.tq_hot_spilled)
    }

    pub fn set_tq_hot_spilled(&mut self, slot: KvSlot, spilled: bool) -> Result<(), KvPoolError> {
        self.slot_mut(slot)?.tq_hot_spilled = spilled;
        Ok(())
    }

    pub fn slot_view(&self, slot: KvSlot) -> Result<KvSlotView, KvPoolError> {
        let slot_state = self.slot(slot)?;
        Ok(KvSlotView {
            slot,
            slot_index: slot.index(),
            seq_len: slot_state.seq_len,
            total_tokens: slot_state.total_tokens,
        })
    }

    pub fn slot_views(&self, slots: &[KvSlot]) -> Result<Vec<KvSlotView>, KvPoolError> {
        slots.iter().map(|&slot| self.slot_view(slot)).collect()
    }

    pub fn slot_buffers(&self, slot: KvSlot) -> Result<(&[Buffer], &[Buffer]), KvPoolError> {
        let slot_state = self.slot(slot)?;
        Ok((&slot_state.k_cache, &slot_state.v_cache))
    }

    pub fn layer_k_cache(
        &self,
        slot: KvSlot,
        layer_idx: usize,
    ) -> Result<&Buffer, KvPoolError> {
        let slot_state = self.slot(slot)?;
        self.layer_buffer(&slot_state.k_cache, layer_idx)
    }

    pub fn layer_v_cache(
        &self,
        slot: KvSlot,
        layer_idx: usize,
    ) -> Result<&Buffer, KvPoolError> {
        let slot_state = self.slot(slot)?;
        self.layer_buffer(&slot_state.v_cache, layer_idx)
    }

    pub fn layer_tq_hot_k(
        &self,
        slot: KvSlot,
        layer_idx: usize,
    ) -> Result<&Buffer, KvPoolError> {
        let slot_state = self.slot(slot)?;
        if slot_state.tq_hot_k.is_empty() {
            return Err(KvPoolError::InvalidLayer {
                layer: layer_idx,
                num_layers: 0,
            });
        }
        self.layer_buffer(&slot_state.tq_hot_k, layer_idx)
    }

    pub fn layer_tq_hot_v(
        &self,
        slot: KvSlot,
        layer_idx: usize,
    ) -> Result<&Buffer, KvPoolError> {
        let slot_state = self.slot(slot)?;
        if slot_state.tq_hot_v.is_empty() {
            return Err(KvPoolError::InvalidLayer {
                layer: layer_idx,
                num_layers: 0,
            });
        }
        self.layer_buffer(&slot_state.tq_hot_v, layer_idx)
    }

    pub fn layer_tq_rw_k(
        &self,
        slot: KvSlot,
        layer_idx: usize,
    ) -> Result<&Buffer, KvPoolError> {
        let slot_state = self.slot(slot)?;
        if slot_state.tq_rw_k.is_empty() {
            return Err(KvPoolError::InvalidLayer {
                layer: layer_idx,
                num_layers: 0,
            });
        }
        self.layer_buffer(&slot_state.tq_rw_k, layer_idx)
    }

    pub fn layer_tq_rw_v(
        &self,
        slot: KvSlot,
        layer_idx: usize,
    ) -> Result<&Buffer, KvPoolError> {
        let slot_state = self.slot(slot)?;
        if slot_state.tq_rw_v.is_empty() {
            return Err(KvPoolError::InvalidLayer {
                layer: layer_idx,
                num_layers: 0,
            });
        }
        self.layer_buffer(&slot_state.tq_rw_v, layer_idx)
    }

    pub fn has_tq_hot(&self, slot: KvSlot) -> Result<bool, KvPoolError> {
        Ok(!self.slot(slot)?.tq_hot_k.is_empty())
    }

    pub fn capacity(&self) -> u32 {
        self.max_seq_len
    }

    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    pub fn available_slots(&self) -> usize {
        self.free_slots.len()
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn with_slot_mut<T>(
        &mut self,
        slot: KvSlot,
        f: impl FnOnce(&mut KvCacheSlot) -> T,
    ) -> Result<T, KvPoolError> {
        let slot = self.slot_mut(slot)?;
        Ok(f(slot))
    }

    fn slot(&self, slot: KvSlot) -> Result<&KvCacheSlot, KvPoolError> {
        let slot_idx = slot.index();
        let slot = self
            .slots
            .get(slot_idx)
            .ok_or(KvPoolError::InvalidSlot(slot_idx))?;

        if !slot.in_use {
            return Err(KvPoolError::SlotNotAllocated(slot_idx));
        }

        Ok(slot)
    }

    fn slot_mut(&mut self, slot: KvSlot) -> Result<&mut KvCacheSlot, KvPoolError> {
        let slot_idx = slot.index();
        let slot = self
            .slots
            .get_mut(slot_idx)
            .ok_or(KvPoolError::InvalidSlot(slot_idx))?;

        if !slot.in_use {
            return Err(KvPoolError::SlotNotAllocated(slot_idx));
        }

        Ok(slot)
    }

    fn layer_buffer<'a>(
        &self,
        buffers: &'a [Buffer],
        layer_idx: usize,
    ) -> Result<&'a Buffer, KvPoolError> {
        buffers.get(layer_idx).ok_or(KvPoolError::InvalidLayer {
            layer: layer_idx,
            num_layers: self.num_layers,
        })
    }
}
