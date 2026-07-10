use std::sync::{
    atomic::{AtomicU8, Ordering},
    mpsc::{Receiver, SyncSender, TryRecvError},
    Arc,
};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use crate::batch_engine::{BatchEngine, DecodeInput, PrefillInput};
use crate::gemma4_gpu_model::Gemma4GpuModel;
use crate::metrics::Metrics;
use crate::sampling::{self, SamplingParams};

pub const CANCEL_NONE: u8 = 0;
pub const CANCEL_CLIENT: u8 = 1;
pub const CANCEL_STOP: u8 = 2;

#[derive(Clone)]
pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub min_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub eos_token_ids: Vec<usize>,
    pub min_decode_tokens: usize,
    pub request_timeout: Duration,
}

pub struct InferenceRequest {
    pub id: String,
    pub input_ids: Vec<usize>,
    pub params: GenerationParams,
    pub response_tx: mpsc::Sender<StreamEvent>,
    pub cancel: Arc<AtomicU8>,
    pub created_at: Instant,
}

pub enum StreamEvent {
    Token { token_id: usize },
    Done { finish_reason: String },
    Error { message: String },
}

pub struct Scheduler {
    engine: BatchEngine,
    metrics: Arc<Metrics>,
    config: SchedulerConfig,
    next_prefill_index: usize,
}

#[derive(Clone, Default)]
pub struct SchedulerConfig {
    pub max_prefill_tokens_per_tick: Option<usize>,
}

impl Scheduler {
    pub fn new(model: Gemma4GpuModel, kv_pool_slots: usize, metrics: Arc<Metrics>) -> Self {
        Self::new_with_config(model, kv_pool_slots, metrics, SchedulerConfig::default())
    }

    pub fn new_with_config(
        model: Gemma4GpuModel,
        kv_pool_slots: usize,
        metrics: Arc<Metrics>,
        config: SchedulerConfig,
    ) -> Self {
        Self {
            engine: BatchEngine::new(model, kv_pool_slots),
            metrics,
            config,
            next_prefill_index: 0,
        }
    }

    pub fn run(mut self, request_rx: Receiver<InferenceRequest>) {
        let mut active = Vec::new();
        let mut receiver_open = true;

        while receiver_open || !active.is_empty() {
            if active.is_empty() && receiver_open {
                match request_rx.recv() {
                    Ok(request) => self.admit_request(request, &mut active),
                    Err(_) => {
                        receiver_open = false;
                        continue;
                    }
                }
            }

            if receiver_open && self.engine.available_slots() > 0 {
                receiver_open = self.try_admit_available(&request_rx, &mut active);
            }

            if !active.is_empty() {
                self.decode_active_round(&mut active);
                self.prefill_active_round(&mut active);
            }
        }
    }

    fn admit_request(&mut self, request: InferenceRequest, active: &mut Vec<ActiveRequest>) {
        self.metrics.record_dequeue();

        let Some(slot) = self.engine.allocate_slot() else {
            let _ = request.response_tx.blocking_send(StreamEvent::Error {
                message: "KV cache pool is full".to_string(),
            });
            self.finish_request(&request, RequestFinish::new("error_kv_pool_full", 0));
            return;
        };

        if request.created_at.elapsed() >= request.params.request_timeout {
            let _ = request.response_tx.blocking_send(StreamEvent::Done {
                finish_reason: "timeout".to_string(),
            });
            let _ = self.engine.release_slot(slot);
            self.finish_request(&request, RequestFinish::new("timeout", 0));
            return;
        }

        if let Some(reason) = cancellation_finish_reason(&request) {
            let _ = request.response_tx.blocking_send(StreamEvent::Done {
                finish_reason: reason.to_string(),
            });
            let _ = self.engine.release_slot(slot);
            self.finish_request(&request, RequestFinish::new(reason, 0));
            return;
        }

        self.metrics.record_prefill_start();
        active.push(ActiveRequest {
            request,
            slot,
            phase: ActivePhase::Prefilling,
            prefill_cursor: 0,
            logits: Vec::new(),
            generated_tokens: Vec::new(),
            completion_tokens: 0,
            prefill_chunks_done: 0,
            prefill_latency: Duration::ZERO,
            decode_started_at: None,
            decode_compute_latency: Duration::ZERO,
        });
    }

    fn try_admit_available(
        &mut self,
        request_rx: &Receiver<InferenceRequest>,
        active: &mut Vec<ActiveRequest>,
    ) -> bool {
        while self.engine.available_slots() > 0 {
            match request_rx.try_recv() {
                Ok(request) => self.admit_request(request, active),
                Err(TryRecvError::Empty) => return true,
                Err(TryRecvError::Disconnected) => return false,
            }
        }

        true
    }

    fn decode_active_round(&mut self, active: &mut Vec<ActiveRequest>) {
        let round_len = active.len();
        let mut decode_batch = Vec::new();
        let mut finished = Vec::new();

        let mut index = 0;
        while index < round_len && index < active.len() {
            if active[index].phase != ActivePhase::Decoding {
                index += 1;
                continue;
            }

            match prepare_decode_token(&mut active[index]) {
                DecodePreparation::Forward(next_token) => {
                    decode_batch.push(PreparedDecode {
                        active_index: index,
                        input: DecodeInput {
                            slot: active[index].slot,
                            token_id: next_token,
                        },
                    });
                }
                DecodePreparation::Finish(finish) => {
                    finished.push(FinishedRequest::done(index, finish));
                }
            }

            index += 1;
        }

        let inputs: Vec<DecodeInput> = decode_batch
            .iter()
            .map(|prepared| DecodeInput {
                slot: prepared.input.slot,
                token_id: prepared.input.token_id,
            })
            .collect();

        if !inputs.is_empty() {
            self.metrics.record_decode_batch(inputs.len());
        }
        for (prepared, output) in decode_batch
            .into_iter()
            .zip(self.engine.decode_batch(&inputs))
        {
            match output {
                Ok(forward) => {
                    let active_request = &mut active[prepared.active_index];
                    active_request.logits = forward.logits;
                    active_request.decode_compute_latency += forward.latency;
                    self.metrics.record_decode_compute(forward.latency);
                }
                Err(message) => {
                    finished.push(FinishedRequest::error(
                        prepared.active_index,
                        message,
                        active[prepared.active_index].finish("error"),
                    ));
                }
            }
        }

        finished.sort_by_key(|finish| finish.active_index);
        finished.dedup_by_key(|finish| finish.active_index);
        for finish in finished.into_iter().rev() {
            let active_request = active.swap_remove(finish.active_index);
            if let Some(message) = finish.error_message {
                let _ = active_request
                    .request
                    .response_tx
                    .blocking_send(StreamEvent::Error {
                        message: message.clone(),
                    });
                self.finish_request(
                    &active_request.request,
                    active_request.finish(&format!("error: {}", message)),
                );
            } else {
                self.finish_request(&active_request.request, finish.finish);
            }
            let _ = self.engine.release_slot(active_request.slot);
        }
    }

    fn prefill_active_round(&mut self, active: &mut Vec<ActiveRequest>) {
        let mut prefill_batch = Vec::new();
        let mut finished = Vec::new();
        let max_chunk_tokens = self.engine.max_prefill_chunk_tokens();
        let token_budget = self.max_prefill_tokens_per_tick();
        let (prefill_plan, next_prefill_index) = plan_prefill_round(
            active.len(),
            self.next_prefill_index,
            max_chunk_tokens,
            token_budget,
            |index| active[index].phase == ActivePhase::Prefilling,
            |index| {
                active[index]
                    .request
                    .input_ids
                    .len()
                    .saturating_sub(active[index].prefill_cursor)
            },
        );
        self.next_prefill_index = next_prefill_index;

        for plan in prefill_plan {
            match prepare_prefill_chunk(&active[plan.active_index], plan.token_count) {
                PrefillPreparation::Forward(input) => {
                    prefill_batch.push(PreparedPrefill {
                        active_index: plan.active_index,
                        token_count: input.token_ids.len(),
                        input,
                    });
                }
                PrefillPreparation::Finish(finish) => {
                    finished.push(FinishedRequest::done(plan.active_index, finish));
                }
            }
        }

        let inputs: Vec<PrefillInput> = prefill_batch
            .iter()
            .map(|prepared| PrefillInput {
                slot: prepared.input.slot,
                token_ids: prepared.input.token_ids.clone(),
                want_logits: prepared.input.want_logits,
            })
            .collect();

        if !inputs.is_empty() {
            self.metrics.record_prefill_batch(inputs.len());
        }
        for (prepared, output) in prefill_batch
            .into_iter()
            .zip(self.engine.prefill_batch(&inputs))
        {
            match output {
                Ok(forward) => {
                    let active_request = &mut active[prepared.active_index];
                    active_request.logits = forward.logits;
                    active_request.prefill_cursor += prepared.token_count;
                    active_request.prefill_chunks_done += 1;
                    active_request.prefill_latency += forward.latency;
                    self.metrics
                        .record_prefill_chunk(prepared.token_count, forward.latency);

                    if active_request.prefill_cursor >= active_request.request.input_ids.len() {
                        active_request.phase = ActivePhase::Decoding;
                        active_request.decode_started_at = Some(Instant::now());
                        self.metrics.record_prefill_to_decode();
                    }
                }
                Err(message) => {
                    finished.push(FinishedRequest::error(
                        prepared.active_index,
                        message,
                        active[prepared.active_index].finish("error"),
                    ));
                }
            }
        }

        finished.sort_by_key(|finish| finish.active_index);
        finished.dedup_by_key(|finish| finish.active_index);
        for finish in finished.into_iter().rev() {
            let active_request = active.swap_remove(finish.active_index);
            if let Some(message) = finish.error_message {
                let _ = active_request
                    .request
                    .response_tx
                    .blocking_send(StreamEvent::Error {
                        message: message.clone(),
                    });
                self.finish_request(
                    &active_request.request,
                    active_request.finish(&format!("error: {}", message)),
                );
            } else {
                self.finish_request(&active_request.request, finish.finish);
            }
            let _ = self.engine.release_slot(active_request.slot);
        }
    }

    fn max_prefill_tokens_per_tick(&self) -> usize {
        let max_chunk_tokens = self.engine.max_prefill_chunk_tokens();
        self.config
            .max_prefill_tokens_per_tick
            .unwrap_or(max_chunk_tokens)
            .min(max_chunk_tokens)
            .max(1)
    }

    fn finish_request(&self, request: &InferenceRequest, finish: RequestFinish) {
        let latency = request.created_at.elapsed();
        self.metrics.record_finish(
            &finish.reason,
            finish.completion_tokens,
            latency,
            finish.decode_latency,
            finish.was_prefilling,
            finish.was_decoding,
        );

        println!(
            "{}",
            serde_json::json!({
                "event": "request_complete",
                "request_id": request.id,
                "prompt_tokens": request.input_ids.len(),
                "prefill_chunks": finish.prefill_chunks,
                "completion_tokens": finish.completion_tokens,
                "latency_ms": latency.as_millis(),
                "prefill_latency_ms": finish.prefill_latency.as_millis(),
                "decode_latency_ms": finish.decode_latency.as_millis(),
                "decode_compute_latency_ms": finish.decode_compute_latency.as_millis(),
                "finish_reason": finish.reason,
            })
        );
    }
}

#[derive(Clone, Copy, PartialEq)]
enum ActivePhase {
    Prefilling,
    Decoding,
}

struct ActiveRequest {
    request: InferenceRequest,
    slot: crate::kv_pool::KvSlot,
    phase: ActivePhase,
    prefill_cursor: usize,
    logits: Vec<f32>,
    generated_tokens: Vec<usize>,
    completion_tokens: usize,
    prefill_chunks_done: usize,
    prefill_latency: Duration,
    decode_started_at: Option<Instant>,
    decode_compute_latency: Duration,
}

struct PreparedDecode {
    active_index: usize,
    input: DecodeInput,
}

struct PreparedPrefill {
    active_index: usize,
    token_count: usize,
    input: PrefillInput,
}

#[derive(Debug, Eq, PartialEq)]
struct PrefillPlanItem {
    active_index: usize,
    token_count: usize,
}

struct FinishedRequest {
    active_index: usize,
    finish: RequestFinish,
    error_message: Option<String>,
}

impl FinishedRequest {
    fn done(active_index: usize, finish: RequestFinish) -> Self {
        Self {
            active_index,
            finish,
            error_message: None,
        }
    }

    fn error(active_index: usize, message: String, finish: RequestFinish) -> Self {
        Self {
            active_index,
            finish,
            error_message: Some(message),
        }
    }
}

impl ActiveRequest {
    fn finish(&self, reason: &str) -> RequestFinish {
        RequestFinish::with_timings(
            reason,
            self.completion_tokens,
            self.request.input_ids.len(),
            self.prefill_chunks_done,
            self.prefill_latency,
            self.decode_started_at
                .map(|started_at| started_at.elapsed())
                .unwrap_or(Duration::ZERO),
            self.decode_compute_latency,
            self.phase == ActivePhase::Prefilling,
            self.phase == ActivePhase::Decoding,
        )
    }
}

enum DecodePreparation {
    Forward(usize),
    Finish(RequestFinish),
}

enum PrefillPreparation {
    Forward(PrefillInput),
    Finish(RequestFinish),
}

fn plan_prefill_round(
    active_len: usize,
    start_index: usize,
    max_chunk_tokens: usize,
    token_budget: usize,
    mut is_prefilling: impl FnMut(usize) -> bool,
    mut remaining_tokens: impl FnMut(usize) -> usize,
) -> (Vec<PrefillPlanItem>, usize) {
    if active_len == 0 || max_chunk_tokens == 0 || token_budget == 0 {
        return (Vec::new(), 0);
    }

    let start_index = start_index % active_len;
    let candidates: Vec<(usize, usize)> = (0..active_len)
        .filter_map(|offset| {
            let active_index = (start_index + offset) % active_len;
            if !is_prefilling(active_index) {
                return None;
            }

            let remaining = remaining_tokens(active_index).min(max_chunk_tokens);
            if remaining == 0 {
                None
            } else {
                Some((active_index, remaining))
            }
        })
        .collect();

    if candidates.is_empty() {
        return (Vec::new(), start_index);
    }

    let mut allocations = vec![0usize; candidates.len()];
    let mut remaining_budget = token_budget;

    while remaining_budget > 0 {
        let eligible_count = candidates
            .iter()
            .zip(&allocations)
            .filter(|((_, remaining), allocated)| **allocated < *remaining)
            .count();
        if eligible_count == 0 {
            break;
        }

        let quantum = (remaining_budget / eligible_count).max(1);
        let mut made_progress = false;
        for ((_, remaining), allocated) in candidates.iter().zip(&mut allocations) {
            if *allocated >= *remaining {
                continue;
            }

            let take = (*remaining - *allocated).min(quantum).min(remaining_budget);
            if take == 0 {
                continue;
            }
            *allocated += take;
            remaining_budget -= take;
            made_progress = true;

            if remaining_budget == 0 {
                break;
            }
        }

        if !made_progress {
            break;
        }
    }

    let plan: Vec<PrefillPlanItem> = candidates
        .iter()
        .zip(allocations)
        .filter_map(|(&(active_index, _), token_count)| {
            if token_count == 0 {
                None
            } else {
                Some(PrefillPlanItem {
                    active_index,
                    token_count,
                })
            }
        })
        .collect();

    let next_index = plan
        .last()
        .map(|item| (item.active_index + 1) % active_len)
        .unwrap_or(start_index);

    (plan, next_index)
}

fn prepare_prefill_chunk(active: &ActiveRequest, chunk_size: usize) -> PrefillPreparation {
    if active.request.created_at.elapsed() >= active.request.params.request_timeout {
        let _ = active.request.response_tx.blocking_send(StreamEvent::Done {
            finish_reason: "timeout".to_string(),
        });
        return PrefillPreparation::Finish(active.finish("timeout"));
    }

    if let Some(reason) = cancellation_finish_reason(&active.request) {
        let _ = active.request.response_tx.blocking_send(StreamEvent::Done {
            finish_reason: reason.to_string(),
        });
        return PrefillPreparation::Finish(active.finish(reason));
    }

    let chunk_start = active.prefill_cursor;
    let chunk_end = (chunk_start + chunk_size).min(active.request.input_ids.len());
    PrefillPreparation::Forward(PrefillInput {
        slot: active.slot,
        token_ids: active.request.input_ids[chunk_start..chunk_end].to_vec(),
        want_logits: chunk_end >= active.request.input_ids.len(),
    })
}

fn prepare_decode_token(active: &mut ActiveRequest) -> DecodePreparation {
    if active.request.created_at.elapsed() >= active.request.params.request_timeout {
        let _ = active.request.response_tx.blocking_send(StreamEvent::Done {
            finish_reason: "timeout".to_string(),
        });
        return DecodePreparation::Finish(active.finish("timeout"));
    }

    if let Some(reason) = cancellation_finish_reason(&active.request) {
        let _ = active.request.response_tx.blocking_send(StreamEvent::Done {
            finish_reason: reason.to_string(),
        });
        return DecodePreparation::Finish(active.finish(reason));
    }

    if active.completion_tokens >= active.request.params.max_tokens {
        let _ = active.request.response_tx.blocking_send(StreamEvent::Done {
            finish_reason: "stop".to_string(),
        });
        return DecodePreparation::Finish(active.finish("stop"));
    }

    let sampling_params = SamplingParams {
        temperature: active.request.params.temperature,
        min_p: active.request.params.min_p,
        top_k: active.request.params.top_k,
        repetition_penalty: active.request.params.repetition_penalty,
        frequency_penalty: active.request.params.frequency_penalty,
    };

    let mut next_token =
        sampling::sample_with_params(&active.logits, &sampling_params, &active.generated_tokens);

    // Control tokens that should not lead an answer turn.
    // 1=<eos>, 100=<|channel>, 101=<channel|>, 105=<|turn>, 106=<turn|>, 107='\n'
    const FIRST_TOKEN_BLOCKLIST: &[usize] = &[1, 100, 101, 105, 106, 107];

    let mut guard = 0;
    loop {
        let block_eos = active.completion_tokens < active.request.params.min_decode_tokens
            && active.request.params.eos_token_ids.contains(&next_token);
        let block_first =
            active.completion_tokens == 0 && FIRST_TOKEN_BLOCKLIST.contains(&next_token);
        if (!block_eos && !block_first) || guard >= 64 {
            break;
        }
        let mut masked_logits = active.logits.clone();
        if block_first {
            for &blocked in FIRST_TOKEN_BLOCKLIST {
                if blocked < masked_logits.len() {
                    masked_logits[blocked] = f32::NEG_INFINITY;
                }
            }
        }
        for eos in &active.request.params.eos_token_ids {
            if *eos < masked_logits.len() {
                masked_logits[*eos] = f32::NEG_INFINITY;
            }
        }
        next_token = sampling::sample_with_params(
            &masked_logits,
            &sampling_params,
            &active.generated_tokens,
        );
        guard += 1;
    }

    if active.request.params.eos_token_ids.contains(&next_token) {
        let _ = active.request.response_tx.blocking_send(StreamEvent::Done {
            finish_reason: "stop".to_string(),
        });
        return DecodePreparation::Finish(active.finish("stop"));
    }

    active.completion_tokens += 1;
    active.generated_tokens.push(next_token);
    if active
        .request
        .response_tx
        .blocking_send(StreamEvent::Token {
            token_id: next_token,
        })
        .is_err()
    {
        return DecodePreparation::Finish(active.finish("cancelled"));
    }

    if let Some(reason) = cancellation_finish_reason(&active.request) {
        let _ = active.request.response_tx.blocking_send(StreamEvent::Done {
            finish_reason: reason.to_string(),
        });
        return DecodePreparation::Finish(active.finish(reason));
    }

    if active.completion_tokens >= active.request.params.max_tokens {
        let _ = active.request.response_tx.blocking_send(StreamEvent::Done {
            finish_reason: "stop".to_string(),
        });
        return DecodePreparation::Finish(active.finish("stop"));
    }

    DecodePreparation::Forward(next_token)
}

fn cancellation_finish_reason(request: &InferenceRequest) -> Option<&'static str> {
    match request.cancel.load(Ordering::Relaxed) {
        CANCEL_STOP => Some("stop"),
        CANCEL_CLIENT => Some("cancelled"),
        _ => None,
    }
}

struct RequestFinish {
    reason: String,
    completion_tokens: usize,
    prefill_tokens: usize,
    prefill_chunks: usize,
    prefill_latency: Duration,
    decode_latency: Duration,
    decode_compute_latency: Duration,
    was_prefilling: bool,
    was_decoding: bool,
}

impl RequestFinish {
    fn new(reason: &str, completion_tokens: usize) -> Self {
        Self {
            reason: reason.to_string(),
            completion_tokens,
            prefill_tokens: 0,
            prefill_chunks: 0,
            prefill_latency: Duration::ZERO,
            decode_latency: Duration::ZERO,
            decode_compute_latency: Duration::ZERO,
            was_prefilling: false,
            was_decoding: false,
        }
    }

    fn with_timings(
        reason: &str,
        completion_tokens: usize,
        prefill_tokens: usize,
        prefill_chunks: usize,
        prefill_latency: Duration,
        decode_latency: Duration,
        decode_compute_latency: Duration,
        was_prefilling: bool,
        was_decoding: bool,
    ) -> Self {
        Self {
            reason: reason.to_string(),
            completion_tokens,
            prefill_tokens,
            prefill_chunks,
            prefill_latency,
            decode_latency,
            decode_compute_latency,
            was_prefilling,
            was_decoding,
        }
    }
}

pub fn spawn_scheduler(
    model: Gemma4GpuModel,
    queue_depth: usize,
    kv_pool_slots: usize,
    metrics: Arc<Metrics>,
) -> SyncSender<InferenceRequest> {
    spawn_scheduler_with_config(
        model,
        queue_depth,
        kv_pool_slots,
        metrics,
        SchedulerConfig::default(),
    )
}

pub fn spawn_scheduler_with_config(
    model: Gemma4GpuModel,
    queue_depth: usize,
    kv_pool_slots: usize,
    metrics: Arc<Metrics>,
    config: SchedulerConfig,
) -> SyncSender<InferenceRequest> {
    let (request_tx, request_rx) = std::sync::mpsc::sync_channel(queue_depth);
    std::thread::spawn(move || {
        Scheduler::new_with_config(model, kv_pool_slots, metrics, config).run(request_rx)
    });
    request_tx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_plan_spreads_total_tokens_per_tick() {
        let phases = [true, true, true];
        let remaining = [200, 200, 200];

        let (plan, next_index) = plan_prefill_round(
            phases.len(),
            0,
            128,
            128,
            |index| phases[index],
            |index| remaining[index],
        );

        assert_eq!(
            plan,
            vec![
                PrefillPlanItem {
                    active_index: 0,
                    token_count: 43,
                },
                PrefillPlanItem {
                    active_index: 1,
                    token_count: 43,
                },
                PrefillPlanItem {
                    active_index: 2,
                    token_count: 42,
                },
            ]
        );
        assert_eq!(next_index, 0);
        assert_eq!(plan.iter().map(|item| item.token_count).sum::<usize>(), 128);
    }

    #[test]
    fn prefill_plan_redistributes_short_chunks_until_budget_is_spent() {
        let phases = [true, true, true];
        let remaining = [8, 40, 120];

        let (plan, next_index) = plan_prefill_round(
            phases.len(),
            0,
            128,
            64,
            |index| phases[index],
            |index| remaining[index],
        );

        assert_eq!(
            plan,
            vec![
                PrefillPlanItem {
                    active_index: 0,
                    token_count: 8,
                },
                PrefillPlanItem {
                    active_index: 1,
                    token_count: 28,
                },
                PrefillPlanItem {
                    active_index: 2,
                    token_count: 28,
                },
            ]
        );
        assert_eq!(next_index, 0);
        assert_eq!(plan.iter().map(|item| item.token_count).sum::<usize>(), 64);
    }

    #[test]
    fn prefill_plan_round_robins_from_previous_cursor() {
        let phases = [true, false, true, true];
        let remaining = [200, 0, 200, 200];

        let (plan, next_index) = plan_prefill_round(
            phases.len(),
            2,
            128,
            128,
            |index| phases[index],
            |index| remaining[index],
        );

        assert_eq!(
            plan,
            vec![
                PrefillPlanItem {
                    active_index: 2,
                    token_count: 43,
                },
                PrefillPlanItem {
                    active_index: 3,
                    token_count: 43,
                },
                PrefillPlanItem {
                    active_index: 0,
                    token_count: 42,
                },
            ]
        );
        assert_eq!(next_index, 1);
    }

    #[test]
    fn prefill_plan_gives_single_prefill_request_full_budget() {
        let phases = [false, true, false];
        let remaining = [0, 200, 0];

        let (plan, next_index) = plan_prefill_round(
            phases.len(),
            0,
            128,
            128,
            |index| phases[index],
            |index| remaining[index],
        );

        assert_eq!(
            plan,
            vec![PrefillPlanItem {
                active_index: 1,
                token_count: 128,
            }]
        );
        assert_eq!(next_index, 2);
    }
}
