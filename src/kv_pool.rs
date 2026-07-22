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

pub struct KvCachePool {
    slots: Vec<KvCacheSlot>,
    free_slots: Vec<usize>,
    max_seq_len: u32,
    num_layers: usize,
    pub kv_cache_type: KvCacheType,
}

pub struct KvCacheSlot {
    pub k_cache: Vec<Buffer>,
    pub v_cache: Vec<Buffer>,
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
    pub(crate) fn from_existing(
        k_cache: &[Buffer],
        v_cache: &[Buffer],
        seq_len: u32,
        total_tokens: usize,
        max_seq_len: u32,
        kv_cache_type: KvCacheType,
    ) -> (Self, KvSlot) {
        assert_eq!(k_cache.len(), v_cache.len());
        let slot = KvSlot(0);
        (
            Self {
                slots: vec![KvCacheSlot {
                    k_cache: k_cache.to_vec(),
                    v_cache: v_cache.to_vec(),
                    seq_len,
                    total_tokens,
                    in_use: true,
                }],
                free_slots: Vec::new(),
                max_seq_len,
                num_layers: k_cache.len(),
                kv_cache_type,
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
    ) -> Self {
        let num_layers = config.num_hidden_layers;
        let num_kv_heads = config.num_key_value_heads;

        let mut slots = Vec::with_capacity(num_slots);
        for _ in 0..num_slots {
            let mut k_cache = Vec::with_capacity(num_layers);
            let mut v_cache = Vec::with_capacity(num_layers);

            for layer_idx in 0..num_layers {
                let head_dim = config.layer_head_dim(layer_idx);
                assert!(head_dim % 32 == 0, "head_dim must be multiple of 32 for quantized KV cache");
                let k_byte_len = (num_kv_heads * max_seq_len as usize * kv_cache_type.k_row_bytes(head_dim)) as u64;
                let v_byte_len = (num_kv_heads * max_seq_len as usize * kv_cache_type.v_row_bytes(head_dim)) as u64;
                k_cache.push(
                    ctx.device
                        .new_buffer(k_byte_len, MTLResourceOptions::StorageModeShared),
                );
                v_cache.push(
                    ctx.device
                        .new_buffer(v_byte_len, MTLResourceOptions::StorageModeShared),
                );
            }

            slots.push(KvCacheSlot {
                k_cache,
                v_cache,
                seq_len: 0,
                total_tokens: 0,
                in_use: false,
            });
        }

        let free_slots = (0..num_slots).rev().collect();

        Self {
            slots,
            free_slots,
            max_seq_len,
            num_layers,
            kv_cache_type,
        }
    }

    pub fn allocate(&mut self) -> Option<KvSlot> {
        let slot_idx = self.free_slots.pop()?;
        let slot = &mut self.slots[slot_idx];
        slot.in_use = true;
        slot.seq_len = 0;
        slot.total_tokens = 0;
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
        self.free_slots.push(slot_idx);
        Ok(())
    }

    pub fn reset(&mut self, slot: KvSlot) -> Result<(), KvPoolError> {
        let slot = self.slot_mut(slot)?;
        slot.seq_len = 0;
        slot.total_tokens = 0;
        Ok(())
    }

    pub fn seq_len(&self, slot: KvSlot) -> Result<u32, KvPoolError> {
        Ok(self.slot(slot)?.seq_len)
    }

    pub fn total_tokens(&self, slot: KvSlot) -> Result<usize, KvPoolError> {
        Ok(self.slot(slot)?.total_tokens)
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
        buffers
            .get(layer_idx)
            .ok_or(KvPoolError::InvalidLayer {
                layer: layer_idx,
                num_layers: self.num_layers,
            })
    }
}
