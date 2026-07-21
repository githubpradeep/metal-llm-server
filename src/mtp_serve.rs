//! Serial MTP (draft/verify speculative decode) scheduler for the HTTP server.
//!
//! MTP verify/draft runs on the model's *global* KV state (`self.k_cache`,
//! `self.kv_seq_len`, `self.total_tokens`) and internally re-aliases pool slots,
//! so it is inherently single-sequence. The batched multi-slot scheduler cannot
//! run it across concurrent requests. When `--mtp` is passed with `--serve`, the
//! server therefore uses this serial scheduler: prefill and decode happen one
//! request at a time via the same draft/verify loop as the CLI
//! (`generate_gemma4_gpu_mtp`), streaming accepted tokens as they land.
//!
//! Concurrent requests queue and are served FIFO.

use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::gemma4_gpu_model::Gemma4GpuModel;
use crate::gemma4_mtp::Gemma4MtpAssistant;
use crate::metrics::Metrics;
use crate::sampling;
use crate::scheduler::{
    cancellation_finish_reason, InferenceRequest, StreamEvent, CANCEL_CLIENT, CANCEL_STOP,
};

/// Maximum draft tokens per verify batch. Bounded by the model's MTP verify
/// scratch (`MAX_MTP_VERIFY_SEQ` = 8 → up to 7 tail steps).
const MAX_DRAFT_STEPS: usize = 7;

/// Control tokens that must not lead an answer turn.
/// 1=<eos>, 100/101=channel markers, 105/106=turn markers, 107='\n'.
const FIRST_TOKEN_BLOCKLIST: &[usize] = &[1, 100, 101, 105, 106, 107];

pub struct MtpScheduler {
    model: Gemma4GpuModel,
    assistant: Gemma4MtpAssistant,
    metrics: Arc<Metrics>,
    draft_steps: usize,
    adaptive: bool,
    p_min: f32,
}

impl MtpScheduler {
    pub fn new(
        model: Gemma4GpuModel,
        assistant: Gemma4MtpAssistant,
        metrics: Arc<Metrics>,
    ) -> Self {
        Self {
            model,
            assistant,
            metrics,
            draft_steps: parse_mtp_draft_steps(),
            adaptive: parse_mtp_adaptive(),
            p_min: parse_mtp_p_min(),
        }
    }

    pub fn run(mut self, request_rx: Receiver<InferenceRequest>) {
        while let Ok(request) = request_rx.recv() {
            self.metrics.record_dequeue();
            self.serve_request(request);
        }
    }

    fn serve_request(&mut self, request: InferenceRequest) {
        if request.created_at.elapsed() >= request.params.request_timeout {
            let _ = request.response_tx.blocking_send(StreamEvent::Done {
                finish_reason: "timeout".to_string(),
            });
            self.record_finish(
                &request,
                "timeout",
                0,
                Duration::ZERO,
                Duration::ZERO,
                &MtpDecodeStats::default(),
            );
            return;
        }
        if let Some(reason) = cancellation_finish_reason(&request) {
            let _ = request.response_tx.blocking_send(StreamEvent::Done {
                finish_reason: reason.to_string(),
            });
            self.record_finish(
                &request,
                reason,
                0,
                Duration::ZERO,
                Duration::ZERO,
                &MtpDecodeStats::default(),
            );
            return;
        }

        if request.input_ids.is_empty() {
            let _ = request.response_tx.blocking_send(StreamEvent::Error {
                message: "empty prompt".to_string(),
            });
            self.record_finish(
                &request,
                "error",
                0,
                Duration::ZERO,
                Duration::ZERO,
                &MtpDecodeStats::default(),
            );
            return;
        }

        // ── Prefill into a fresh single-slot pool, then alias into model global
        //    state so the draft/verify loop operates on this request's KV.
        self.metrics.record_prefill_start();
        let prefill_started = Instant::now();
        let mut kv_pool = self.model.create_kv_pool(1, self.model.kv_capacity);
        let slot = match kv_pool.allocate() {
            Some(slot) => slot,
            None => {
                let _ = request.response_tx.blocking_send(StreamEvent::Error {
                    message: "KV cache pool is full".to_string(),
                });
                self.record_finish(
                    &request,
                    "error_kv_pool_full",
                    0,
                    Duration::ZERO,
                    Duration::ZERO,
                    &MtpDecodeStats::default(),
                );
                return;
            }
        };

        let prefill = self
            .model
            .forward_prefill_pool(&request.input_ids, &mut kv_pool, slot)
            .and_then(|logits| {
                self.model
                    .alias_kv_from_pool(&kv_pool, slot)
                    .map(|()| logits)
            });
        let logits = match prefill {
            Ok(logits) => logits,
            Err(message) => {
                let _ = request
                    .response_tx
                    .blocking_send(StreamEvent::Error { message: message.clone() });
                self.record_finish(
                    &request,
                    "error",
                    0,
                    prefill_started.elapsed(),
                    Duration::ZERO,
                    &MtpDecodeStats::default(),
                );
                return;
            }
        };
        let prefill_latency = prefill_started.elapsed();
        self.metrics
            .record_prefill_chunk(request.input_ids.len(), prefill_latency);
        self.metrics.record_prefill_to_decode();

        self.decode_loop(&request, logits, prefill_latency);
    }

    fn decode_loop(
        &mut self,
        request: &InferenceRequest,
        prefill_logits: Vec<f32>,
        prefill_latency: Duration,
    ) {
        let params = &request.params;
        let decode_started = Instant::now();
        let mut decode_compute = Duration::ZERO;
        let mut completion_tokens = 0usize;
        let mut generated: Vec<usize> = Vec::new();
        let mut stats = MtpDecodeStats::default();

        // First token comes straight from the prefill logits (greedy — MTP is a
        // greedy speculative scheme; sampling params other than greedy are not
        // guaranteed to match the verify path, so we decode greedily and honor
        // eos + first-token guarding only).
        //
        // First-token guard: mask control tokens (and eos while below
        // min_decode_tokens) in the prefill logits so `id_last` never *starts*
        // as a stop token. Re-feeding a stop token into the draft head produces
        // more stop tokens and spins the KV forever — the first token must be a
        // real content token or we accept the model's decision to stop.
        let mut masked_prefill = prefill_logits.clone();
        for &blocked in FIRST_TOKEN_BLOCKLIST {
            if blocked < masked_prefill.len() {
                masked_prefill[blocked] = f32::NEG_INFINITY;
            }
        }
        if params.min_decode_tokens > 0 {
            for &eos in &params.eos_token_ids {
                if eos < masked_prefill.len() {
                    masked_prefill[eos] = f32::NEG_INFINITY;
                }
            }
        }
        let first_greedy = sampling::argmax(&prefill_logits);
        // If the model genuinely wants to stop immediately and there is no
        // min-decode floor, honor it.
        let mut id_last = if params.eos_token_ids.contains(&first_greedy)
            && params.min_decode_tokens == 0
        {
            let _ = request.response_tx.blocking_send(StreamEvent::Done {
                finish_reason: "stop".to_string(),
            });
            self.record_finish(
                request,
                "stop",
                0,
                prefill_latency,
                decode_started.elapsed(),
                &stats,
            );
            return;
        } else {
            sampling::argmax(&masked_prefill)
        };
        let mut mtp_hidden = self.model.last_hidden_activation();

        // Emit the first token.
        match self.emit_token(request, id_last, &mut completion_tokens, &mut generated) {
            EmitOutcome::Continue => {}
            EmitOutcome::Stop(reason) => {
                self.record_finish(
                    request,
                    &reason,
                    completion_tokens,
                    prefill_latency,
                    decode_started.elapsed(),
                    &stats,
                );
                return;
            }
        }

        loop {
            if completion_tokens >= params.max_tokens {
                self.finish_stop(
                    request,
                    completion_tokens,
                    prefill_latency,
                    decode_started,
                    &stats,
                );
                return;
            }
            if let Some(reason) = cancellation_finish_reason(request) {
                let _ = request.response_tx.blocking_send(StreamEvent::Done {
                    finish_reason: reason.to_string(),
                });
                self.record_finish(
                    request,
                    reason,
                    completion_tokens,
                    prefill_latency,
                    decode_started.elapsed(),
                    &stats,
                );
                return;
            }
            if request.created_at.elapsed() >= params.request_timeout {
                let _ = request.response_tx.blocking_send(StreamEvent::Done {
                    finish_reason: "timeout".to_string(),
                });
                self.record_finish(
                    request,
                    "timeout",
                    completion_tokens,
                    prefill_latency,
                    decode_started.elapsed(),
                    &stats,
                );
                return;
            }

            let tail_steps = effective_draft_tail_steps(
                &stats.accepted_per_cycle,
                self.draft_steps,
                self.adaptive,
            );
            let n_draft = tail_steps + 1;

            let step_started = Instant::now();
            let drafted = match self
                .assistant
                .draft_chain(id_last, &mtp_hidden, n_draft, &self.model, self.p_min)
            {
                Ok(drafted) => drafted,
                Err(message) => {
                    let _ = request
                        .response_tx
                        .blocking_send(StreamEvent::Error { message: message.clone() });
                    self.record_finish(
                        request,
                        "error",
                        completion_tokens,
                        prefill_latency,
                        decode_started.elapsed(),
                        &stats,
                    );
                    return;
                }
            };

            // Produce the accepted-token list for this cycle.
            let accepted_ids: Vec<usize> = if drafted.is_empty() {
                // Rare fallback: plain single-token decode on global KV.
                let next_logits = self.model.forward_single_token(id_last);
                mtp_hidden = self.model.last_hidden_activation();
                let next = sampling::argmax(&next_logits);
                stats.record_cycle(/*drafted*/ 0, /*accepted*/ 0);
                id_last = next;
                vec![next]
            } else {
                let mut verify_batch = Vec::with_capacity(drafted.len() + 1);
                verify_batch.push(id_last);
                verify_batch.extend_from_slice(&drafted);

                let verify_tokens = match self.model.forward_verify_batch(&verify_batch) {
                    Ok(tokens) => tokens,
                    Err(message) => {
                        let _ = request
                            .response_tx
                            .blocking_send(StreamEvent::Error { message: message.clone() });
                        self.record_finish(
                            request,
                            "error",
                            completion_tokens,
                            prefill_latency,
                            decode_started.elapsed(),
                            &stats,
                        );
                        return;
                    }
                };

                let mut ids: Vec<usize> = Vec::with_capacity(drafted.len() + 1);
                let mut n_accepted = 0usize;
                for i in 0..drafted.len() {
                    let pred = verify_tokens[i];
                    ids.push(pred);
                    if pred != drafted[i] {
                        break;
                    }
                    n_accepted += 1;
                }
                if n_accepted == drafted.len() {
                    ids.push(verify_tokens[drafted.len()]);
                }
                stats.record_cycle(drafted.len(), n_accepted);

                // Roll back KV for rejected drafts.
                let rewind = (drafted.len() - n_accepted) as u32;
                if rewind > 0 {
                    self.model.truncate_kv(rewind);
                }

                let i_h = n_accepted.min(verify_batch.len() - 1);
                mtp_hidden = self.model.prefill_hidden_activation_at(i_h);
                id_last = *ids.last().unwrap();
                ids
            };

            decode_compute += step_started.elapsed();
            self.metrics.record_decode_compute(step_started.elapsed());

            for tok in accepted_ids {
                match self.emit_token(request, tok, &mut completion_tokens, &mut generated) {
                    EmitOutcome::Continue => {}
                    EmitOutcome::Stop(reason) => {
                        self.record_finish(
                            request,
                            &reason,
                            completion_tokens,
                            prefill_latency,
                            decode_started.elapsed(),
                            &stats,
                        );
                        return;
                    }
                }
            }
        }
    }

    /// Emit one token to the stream. Stops unconditionally on eos — MTP decodes
    /// greedily, so once the verifier commits an eos token the turn is over.
    /// The min-decode / first-token guard is applied to the *first* token in
    /// `decode_loop` (by masking prefill logits), not here: re-feeding a stop
    /// token back into the draft head only produces more stop tokens and spins
    /// the KV cache forever.
    fn emit_token(
        &self,
        request: &InferenceRequest,
        token: usize,
        completion_tokens: &mut usize,
        generated: &mut Vec<usize>,
    ) -> EmitOutcome {
        let params = &request.params;

        // eos → end of turn.
        if params.eos_token_ids.contains(&token) {
            let _ = request.response_tx.blocking_send(StreamEvent::Done {
                finish_reason: "stop".to_string(),
            });
            return EmitOutcome::Stop("stop".to_string());
        }

        *completion_tokens += 1;
        generated.push(token);
        if request
            .response_tx
            .blocking_send(StreamEvent::Token { token_id: token })
            .is_err()
        {
            return EmitOutcome::Stop("cancelled".to_string());
        }

        if let Some(reason) = cancellation_finish_reason(request) {
            let _ = request.response_tx.blocking_send(StreamEvent::Done {
                finish_reason: reason.to_string(),
            });
            return EmitOutcome::Stop(reason.to_string());
        }

        if *completion_tokens >= params.max_tokens {
            let _ = request.response_tx.blocking_send(StreamEvent::Done {
                finish_reason: "stop".to_string(),
            });
            return EmitOutcome::Stop("stop".to_string());
        }

        EmitOutcome::Continue
    }

    fn finish_stop(
        &self,
        request: &InferenceRequest,
        completion_tokens: usize,
        prefill_latency: Duration,
        decode_started: Instant,
        stats: &MtpDecodeStats,
    ) {
        let _ = request.response_tx.blocking_send(StreamEvent::Done {
            finish_reason: "stop".to_string(),
        });
        self.record_finish(
            request,
            "stop",
            completion_tokens,
            prefill_latency,
            decode_started.elapsed(),
            stats,
        );
    }

    fn record_finish(
        &self,
        request: &InferenceRequest,
        reason: &str,
        completion_tokens: usize,
        prefill_latency: Duration,
        decode_latency: Duration,
        stats: &MtpDecodeStats,
    ) {
        let latency = request.created_at.elapsed();
        self.metrics.record_finish(
            reason,
            completion_tokens,
            latency,
            decode_latency,
            false,
            completion_tokens > 0,
        );

        let decode_tok_s = if decode_latency.as_secs_f64() > 0.0 {
            completion_tokens as f64 / decode_latency.as_secs_f64()
        } else {
            0.0
        };
        // CLI parity: accepted/drafted (not vs configured draft_steps).
        let accept_rate = if stats.drafted_total > 0 {
            stats.accepted_total as f64 / stats.drafted_total as f64
        } else {
            0.0
        };
        // Emitted tokens per main-model verify/fallback forward (≳2.0 for ~2×).
        let tokens_per_forward = if stats.main_forwards > 0 {
            completion_tokens as f64 / stats.main_forwards as f64
        } else {
            0.0
        };

        println!(
            "{}",
            serde_json::json!({
                "event": "request_complete",
                "request_id": request.id,
                "prompt_tokens": request.input_ids.len(),
                "completion_tokens": completion_tokens,
                "latency_ms": latency.as_millis(),
                "prefill_latency_ms": prefill_latency.as_millis(),
                "decode_latency_ms": decode_latency.as_millis(),
                "decode_tokens_per_s": format!("{:.2}", decode_tok_s),
                "draft_cycles": stats.main_forwards,
                "drafted_total": stats.drafted_total,
                "accepted_total": stats.accepted_total,
                "accept_rate": format!("{:.3}", accept_rate),
                "tokens_per_forward": format!("{:.2}", tokens_per_forward),
                "finish_reason": reason,
                "mode": "mtp",
            })
        );
    }
}

/// Per-request MTP decode counters (CLI-parity accept / tok-per-forward).
#[derive(Default)]
struct MtpDecodeStats {
    /// Accepted draft tokens each cycle (excludes bonus greedy token).
    accepted_per_cycle: Vec<usize>,
    drafted_total: usize,
    accepted_total: usize,
    /// Verify batches + single-token fallbacks (main-model forwards).
    main_forwards: usize,
}

impl MtpDecodeStats {
    fn record_cycle(&mut self, drafted: usize, accepted: usize) {
        self.main_forwards += 1;
        self.drafted_total += drafted;
        self.accepted_total += accepted;
        self.accepted_per_cycle.push(accepted);
    }
}

enum EmitOutcome {
    Continue,
    Stop(String),
}

/// Spawn the serial MTP scheduler thread and return the request sender.
pub fn spawn_mtp_scheduler(
    model: Gemma4GpuModel,
    assistant: Gemma4MtpAssistant,
    queue_depth: usize,
    metrics: Arc<Metrics>,
) -> SyncSender<InferenceRequest> {
    let (request_tx, request_rx) = std::sync::mpsc::sync_channel(queue_depth);
    std::thread::spawn(move || {
        MtpScheduler::new(model, assistant, metrics).run(request_rx);
    });
    request_tx
}

// ── Draft-step heuristics (mirrors the CLI MTP path in main.rs) ──────────────

fn parse_mtp_draft_steps() -> usize {
    std::env::var("LLAMA_MTP_DRAFT_STEPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .map(|value| value.min(MAX_DRAFT_STEPS + 1))
        .unwrap_or(4)
}

fn parse_mtp_adaptive() -> bool {
    std::env::var("LLAMA_MTP_ADAPTIVE")
        .map(|value| value == "1" || value.to_ascii_lowercase() == "true")
        .unwrap_or(false)
}

fn parse_mtp_p_min() -> f32 {
    std::env::var("LLAMA_MTP_P_MIN")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|&v| v > 0.0 && v < 1.0)
        .unwrap_or(0.0)
}

fn adaptive_draft_tail_steps(accept_history: &[usize], max_steps: usize) -> usize {
    if max_steps <= 1 {
        return 0;
    }
    if accept_history.is_empty() {
        return max_steps - 1;
    }
    let window = accept_history.len().min(12);
    let recent = &accept_history[accept_history.len() - window..];
    let avg = recent.iter().sum::<usize>() as f64 / window as f64;
    if avg >= 3.0 {
        max_steps - 1
    } else if avg >= 2.0 {
        (max_steps - 1).min(2)
    } else {
        // Keep one tail as a probe. Zero tails self-locks: cycles can never
        // accept enough to climb the avg threshold again.
        1
    }
}

/// llama.cpp draft-mtp: depth = n_max (or adaptive), no SWA/context clamp.
fn effective_draft_tail_steps(
    accept_history: &[usize],
    max_steps: usize,
    adaptive: bool,
) -> usize {
    let tail = if adaptive {
        adaptive_draft_tail_steps(accept_history, max_steps)
    } else {
        max_steps.saturating_sub(1)
    };
    tail.min(MAX_DRAFT_STEPS)
}

/// Silence unused-import lints on the cancel constants imported for symmetry.
#[allow(dead_code)]
fn _cancel_constants_used() -> (u8, u8) {
    (CANCEL_CLIENT, CANCEL_STOP)
}
