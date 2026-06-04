use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::{Receiver, SyncSender},
    Arc,
};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use crate::gemma4_gpu_model::Gemma4GpuModel;
use crate::metrics::Metrics;
use crate::sampling::{self, SamplingParams};

#[derive(Clone)]
pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub min_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub eos_token_ids: Vec<usize>,
    pub request_timeout: Duration,
}

pub struct InferenceRequest {
    pub id: String,
    pub input_ids: Vec<usize>,
    pub params: GenerationParams,
    pub response_tx: mpsc::Sender<StreamEvent>,
    pub cancel: Arc<AtomicBool>,
    pub created_at: Instant,
}

pub enum StreamEvent {
    Token { token_id: usize },
    Done { finish_reason: String },
    Error { message: String },
}

pub struct Scheduler {
    model: Gemma4GpuModel,
    kv_pool_slots: usize,
    metrics: Arc<Metrics>,
}

impl Scheduler {
    pub fn new(model: Gemma4GpuModel, kv_pool_slots: usize, metrics: Arc<Metrics>) -> Self {
        Self {
            model,
            kv_pool_slots,
            metrics,
        }
    }

    pub fn run(mut self, request_rx: Receiver<InferenceRequest>) {
        let mut kv_pool = self
            .model
            .create_kv_pool(self.kv_pool_slots, self.model.kv_capacity);

        while let Ok(request) = request_rx.recv() {
            self.metrics.record_dequeue();
            self.run_request(request, &mut kv_pool);
        }
    }

    fn run_request(&mut self, request: InferenceRequest, kv_pool: &mut crate::kv_pool::KvCachePool) {
        let Some(slot) = kv_pool.allocate() else {
            let _ = request.response_tx.blocking_send(StreamEvent::Error {
                message: "KV cache pool is full".to_string(),
            });
            self.finish_request(&request, "error_kv_pool_full", 0);
            return;
        };

        let result = self.run_request_in_slot(&request, kv_pool, slot);
        let _ = kv_pool.release(slot);

        match result {
            Ok(finish) => {
                self.finish_request(&request, &finish.reason, finish.completion_tokens);
            }
            Err(message) => {
                let _ = request.response_tx.blocking_send(StreamEvent::Error {
                    message: message.clone(),
                });
                self.finish_request(&request, &format!("error: {}", message), 0);
            }
        }
    }

    fn run_request_in_slot(
        &mut self,
        request: &InferenceRequest,
        kv_pool: &mut crate::kv_pool::KvCachePool,
        slot: crate::kv_pool::KvSlot,
    ) -> Result<RequestFinish, String> {
        if request.created_at.elapsed() >= request.params.request_timeout {
            let _ = request.response_tx.blocking_send(StreamEvent::Done {
                finish_reason: "timeout".to_string(),
            });
            return Ok(RequestFinish::new("timeout", 0));
        }

        if request.cancel.load(Ordering::Relaxed) {
            let _ = request.response_tx.blocking_send(StreamEvent::Done {
                finish_reason: "cancelled".to_string(),
            });
            return Ok(RequestFinish::new("cancelled", 0));
        }

        let mut logits = self
            .model
            .forward_prefill_with_kv_slot(&request.input_ids, kv_pool, slot)
            .map_err(|err| err.to_string())?;

        let mut completion_tokens = 0;
        let mut generated_tokens = Vec::new();
        for _ in 0..request.params.max_tokens {
            if request.created_at.elapsed() >= request.params.request_timeout {
                let _ = request.response_tx.blocking_send(StreamEvent::Done {
                    finish_reason: "timeout".to_string(),
                });
                return Ok(RequestFinish::new("timeout", completion_tokens));
            }

            if request.cancel.load(Ordering::Relaxed) {
                let _ = request.response_tx.blocking_send(StreamEvent::Done {
                    finish_reason: "cancelled".to_string(),
                });
                return Ok(RequestFinish::new("cancelled", completion_tokens));
            }

            let next_token = sampling::sample_with_params(
                &logits,
                &SamplingParams {
                    temperature: request.params.temperature,
                    min_p: request.params.min_p,
                    top_k: request.params.top_k,
                    repetition_penalty: request.params.repetition_penalty,
                    frequency_penalty: request.params.frequency_penalty,
                },
                &generated_tokens,
            );

            if request.params.eos_token_ids.contains(&next_token) {
                break;
            }

            completion_tokens += 1;
            generated_tokens.push(next_token);
            if request
                .response_tx
                .blocking_send(StreamEvent::Token { token_id: next_token })
                .is_err()
            {
                return Ok(RequestFinish::new("client_disconnected", completion_tokens));
            }

            if request.cancel.load(Ordering::Relaxed) {
                let _ = request.response_tx.blocking_send(StreamEvent::Done {
                    finish_reason: "cancelled".to_string(),
                });
                return Ok(RequestFinish::new("cancelled", completion_tokens));
            }

            logits = self
                .model
                .forward_single_token_with_kv_slot(next_token, kv_pool, slot)
                .map_err(|err| err.to_string())?;
        }

        let _ = request.response_tx.blocking_send(StreamEvent::Done {
            finish_reason: "stop".to_string(),
        });
        Ok(RequestFinish::new("stop", completion_tokens))
    }

    fn finish_request(&self, request: &InferenceRequest, finish_reason: &str, completion_tokens: usize) {
        let latency = request.created_at.elapsed();
        self.metrics
            .record_finish(finish_reason, completion_tokens, latency);

        println!(
            "{}",
            serde_json::json!({
                "event": "request_complete",
                "request_id": request.id,
                "prompt_tokens": request.input_ids.len(),
                "completion_tokens": completion_tokens,
                "latency_ms": latency.as_millis(),
                "finish_reason": finish_reason,
            })
        );
    }
}

struct RequestFinish {
    reason: String,
    completion_tokens: usize,
}

impl RequestFinish {
    fn new(reason: &str, completion_tokens: usize) -> Self {
        Self {
            reason: reason.to_string(),
            completion_tokens,
        }
    }
}

pub fn spawn_scheduler(
    model: Gemma4GpuModel,
    queue_depth: usize,
    kv_pool_slots: usize,
    metrics: Arc<Metrics>,
) -> SyncSender<InferenceRequest> {
    let (request_tx, request_rx) = std::sync::mpsc::sync_channel(queue_depth);
    std::thread::spawn(move || Scheduler::new(model, kv_pool_slots, metrics).run(request_rx));
    request_tx
}
