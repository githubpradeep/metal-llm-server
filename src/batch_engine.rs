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

pub struct DecodeInput {
    pub slot: KvSlot,
    pub token_id: usize,
}

pub struct PrefillInput {
    pub slot: KvSlot,
    pub token_ids: Vec<usize>,
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

    pub fn max_decode_batch_size(&self) -> usize {
        self.model.max_decode_batch_size().max(1)
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

    pub fn prefill_batch(&mut self, inputs: &[PrefillInput]) -> Vec<Result<TimedForward, String>> {
        inputs
            .iter()
            .map(|input| self.prefill_chunk(&input.token_ids, input.slot))
            .collect()
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

    pub fn decode_batch(&mut self, inputs: &[DecodeInput]) -> Vec<Result<TimedForward, String>> {
        if inputs.is_empty() {
            return Vec::new();
        }

        let mut results = Vec::with_capacity(inputs.len());
        for chunk in inputs.chunks(self.max_decode_batch_size()) {
            let started_at = Instant::now();
            let model_inputs: Vec<(KvSlot, usize)> = chunk
                .iter()
                .map(|input| (input.slot, input.token_id))
                .collect();
            let outputs = self
                .model
                .forward_decode_batch_with_kv_slots(&model_inputs, &mut self.kv_pool);
            let per_item_latency = started_at.elapsed() / chunk.len() as u32;

            results.extend(outputs.into_iter().map(|output| {
                output.map(|logits| TimedForward {
                    logits,
                    latency: per_item_latency,
                })
            }));
        }

        results
    }
}
