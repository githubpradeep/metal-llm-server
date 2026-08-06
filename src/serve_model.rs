//! Shared GPU-model surface for the OpenAI-compatible server / scheduler.

use crate::kv_pool::{KvCachePool, KvPoolError, KvSlot};

/// Methods required by [`crate::batch_engine::BatchEngine`] and the scheduler.
pub trait ServeGpuModel: Send + 'static {
    fn kv_capacity(&self) -> u32;

    fn create_kv_pool(&self, num_slots: usize, max_seq_len: u32) -> KvCachePool;

    fn max_parallel_prefill_seq(&self) -> usize;

    fn max_decode_batch_size(&self) -> usize;

    fn forward_prefill_chunk_with_kv_slot(
        &mut self,
        token_ids: &[usize],
        kv_pool: &mut KvCachePool,
        slot: KvSlot,
        want_logits: bool,
    ) -> Result<Vec<f32>, String>;

    fn forward_prefill_batch_with_kv_slots(
        &mut self,
        inputs: &[(KvSlot, &[usize])],
        kv_pool: &mut KvCachePool,
    ) -> Vec<Result<Vec<f32>, String>>;

    fn forward_single_token_with_kv_slot(
        &mut self,
        token_id: usize,
        kv_pool: &mut KvCachePool,
        slot: KvSlot,
    ) -> Result<Vec<f32>, KvPoolError>;

    fn forward_decode_batch_with_kv_slots(
        &mut self,
        inputs: &[(KvSlot, usize)],
        kv_pool: &mut KvCachePool,
    ) -> Vec<Result<Vec<f32>, String>>;
}
