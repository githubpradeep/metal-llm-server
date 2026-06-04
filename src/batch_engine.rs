use std::time::{Duration, Instant};

use crate::gemma4_gpu_model::Gemma4GpuModel;
use crate::kv_pool::{KvCachePool, KvPoolError, KvSlot};

pub struct BatchEngine {
    model: Gemma4GpuModel,
    kv_pool: KvCachePool,
}

pub struct TimedForward {
    pub logits: Vec<f32>,
    pub latency: Duration,
}

impl BatchEngine {
    pub fn new(model: Gemma4GpuModel, kv_pool_slots: usize) -> Self {
        let kv_pool = model.create_kv_pool(kv_pool_slots, model.kv_capacity);
        Self { model, kv_pool }
    }

    pub fn allocate_slot(&mut self) -> Option<KvSlot> {
        self.kv_pool.allocate()
    }

    pub fn release_slot(&mut self, slot: KvSlot) -> Result<(), KvPoolError> {
        self.kv_pool.release(slot)
    }

    pub fn available_slots(&self) -> usize {
        self.kv_pool.available_slots()
    }

    pub fn max_prefill_chunk_tokens(&self) -> usize {
        self.model.max_parallel_prefill_seq().max(1)
    }

    pub fn prefill_chunk(&mut self, token_ids: &[usize], slot: KvSlot) -> Result<TimedForward, String> {
        let started_at = Instant::now();
        let logits = self
            .model
            .forward_prefill_chunk_with_kv_slot(token_ids, &mut self.kv_pool, slot)?;
        Ok(TimedForward {
            logits,
            latency: started_at.elapsed(),
        })
    }

    pub fn decode_one(&mut self, token_id: usize, slot: KvSlot) -> Result<TimedForward, String> {
        let started_at = Instant::now();
        let logits = self
            .model
            .forward_single_token_with_kv_slot(token_id, &mut self.kv_pool, slot)
            .map_err(|err| err.to_string())?;
        Ok(TimedForward {
            logits,
            latency: started_at.elapsed(),
        })
    }
}
