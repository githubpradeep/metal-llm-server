use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

pub struct Metrics {
    requests_total: AtomicU64,
    requests_completed: AtomicU64,
    requests_failed: AtomicU64,
    queue_full_total: AtomicU64,
    prompt_tokens_total: AtomicU64,
    prefill_tokens_total: AtomicU64,
    prefill_chunks_total: AtomicU64,
    prefill_batches_total: AtomicU64,
    prefill_batch_items_total: AtomicU64,
    prefill_batch_items_max: AtomicU64,
    completion_tokens_total: AtomicU64,
    decode_batches_total: AtomicU64,
    decode_batch_items_total: AtomicU64,
    decode_batch_items_max: AtomicU64,
    latency_ms_total: AtomicU64,
    latency_ms_max: AtomicU64,
    prefill_latency_ms_total: AtomicU64,
    prefill_latency_ms_max: AtomicU64,
    decode_latency_ms_total: AtomicU64,
    decode_latency_ms_max: AtomicU64,
    decode_compute_latency_ms_total: AtomicU64,
    decode_compute_latency_ms_max: AtomicU64,
    queued_requests: AtomicUsize,
    active_requests: AtomicUsize,
    prefilling_requests: AtomicUsize,
    decoding_requests: AtomicUsize,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            requests_completed: AtomicU64::new(0),
            requests_failed: AtomicU64::new(0),
            queue_full_total: AtomicU64::new(0),
            prompt_tokens_total: AtomicU64::new(0),
            prefill_tokens_total: AtomicU64::new(0),
            prefill_chunks_total: AtomicU64::new(0),
            prefill_batches_total: AtomicU64::new(0),
            prefill_batch_items_total: AtomicU64::new(0),
            prefill_batch_items_max: AtomicU64::new(0),
            completion_tokens_total: AtomicU64::new(0),
            decode_batches_total: AtomicU64::new(0),
            decode_batch_items_total: AtomicU64::new(0),
            decode_batch_items_max: AtomicU64::new(0),
            latency_ms_total: AtomicU64::new(0),
            latency_ms_max: AtomicU64::new(0),
            prefill_latency_ms_total: AtomicU64::new(0),
            prefill_latency_ms_max: AtomicU64::new(0),
            decode_latency_ms_total: AtomicU64::new(0),
            decode_latency_ms_max: AtomicU64::new(0),
            decode_compute_latency_ms_total: AtomicU64::new(0),
            decode_compute_latency_ms_max: AtomicU64::new(0),
            queued_requests: AtomicUsize::new(0),
            active_requests: AtomicUsize::new(0),
            prefilling_requests: AtomicUsize::new(0),
            decoding_requests: AtomicUsize::new(0),
        }
    }

    pub fn record_enqueue(&self, prompt_tokens: usize) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.prompt_tokens_total
            .fetch_add(prompt_tokens as u64, Ordering::Relaxed);
        self.queued_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_queue_full(&self) {
        self.requests_failed.fetch_add(1, Ordering::Relaxed);
        self.queue_full_total.fetch_add(1, Ordering::Relaxed);
        self.decrement_queued();
    }

    pub fn record_dequeue(&self) {
        self.decrement_queued();
        self.active_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_prefill_start(&self) {
        self.prefilling_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_prefill_chunk(&self, prefill_tokens: usize, prefill_latency: Duration) {
        let latency_ms = prefill_latency.as_millis() as u64;
        self.prefill_tokens_total
            .fetch_add(prefill_tokens as u64, Ordering::Relaxed);
        self.prefill_chunks_total.fetch_add(1, Ordering::Relaxed);
        self.prefill_latency_ms_total
            .fetch_add(latency_ms, Ordering::Relaxed);
        Self::record_max(&self.prefill_latency_ms_max, latency_ms);
    }

    pub fn record_prefill_batch(&self, batch_items: usize) {
        self.prefill_batches_total.fetch_add(1, Ordering::Relaxed);
        self.prefill_batch_items_total
            .fetch_add(batch_items as u64, Ordering::Relaxed);
        Self::record_max(&self.prefill_batch_items_max, batch_items as u64);
    }

    pub fn record_prefill_to_decode(&self) {
        self.decrement_prefilling();
        self.decoding_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_decode_compute(&self, decode_compute_latency: Duration) {
        let latency_ms = decode_compute_latency.as_millis() as u64;
        self.decode_compute_latency_ms_total
            .fetch_add(latency_ms, Ordering::Relaxed);
        Self::record_max(&self.decode_compute_latency_ms_max, latency_ms);
    }

    pub fn record_decode_batch(&self, batch_items: usize) {
        self.decode_batches_total.fetch_add(1, Ordering::Relaxed);
        self.decode_batch_items_total
            .fetch_add(batch_items as u64, Ordering::Relaxed);
        Self::record_max(&self.decode_batch_items_max, batch_items as u64);
    }

    pub fn record_finish(
        &self,
        finish_reason: &str,
        completion_tokens: usize,
        latency: Duration,
        decode_latency: Duration,
        was_prefilling: bool,
        was_decoding: bool,
    ) {
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
        if was_prefilling {
            self.decrement_prefilling();
        }
        if was_decoding {
            self.decrement_decoding();
        }
        self.completion_tokens_total
            .fetch_add(completion_tokens as u64, Ordering::Relaxed);
        let latency_ms = latency.as_millis() as u64;
        let decode_latency_ms = decode_latency.as_millis() as u64;
        self.latency_ms_total
            .fetch_add(latency_ms, Ordering::Relaxed);
        Self::record_max(&self.latency_ms_max, latency_ms);
        self.decode_latency_ms_total
            .fetch_add(decode_latency_ms, Ordering::Relaxed);
        Self::record_max(&self.decode_latency_ms_max, decode_latency_ms);

        if finish_reason.starts_with("error") {
            self.requests_failed.fetch_add(1, Ordering::Relaxed);
        } else {
            self.requests_completed.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn render_prometheus(&self) -> String {
        let requests_total = self.requests_total.load(Ordering::Relaxed);
        let requests_completed = self.requests_completed.load(Ordering::Relaxed);
        let requests_failed = self.requests_failed.load(Ordering::Relaxed);
        let queue_full_total = self.queue_full_total.load(Ordering::Relaxed);
        let prompt_tokens_total = self.prompt_tokens_total.load(Ordering::Relaxed);
        let prefill_tokens_total = self.prefill_tokens_total.load(Ordering::Relaxed);
        let prefill_chunks_total = self.prefill_chunks_total.load(Ordering::Relaxed);
        let prefill_batches_total = self.prefill_batches_total.load(Ordering::Relaxed);
        let prefill_batch_items_total = self.prefill_batch_items_total.load(Ordering::Relaxed);
        let prefill_batch_items_max = self.prefill_batch_items_max.load(Ordering::Relaxed);
        let completion_tokens_total = self.completion_tokens_total.load(Ordering::Relaxed);
        let decode_batches_total = self.decode_batches_total.load(Ordering::Relaxed);
        let decode_batch_items_total = self.decode_batch_items_total.load(Ordering::Relaxed);
        let decode_batch_items_max = self.decode_batch_items_max.load(Ordering::Relaxed);
        let latency_ms_total = self.latency_ms_total.load(Ordering::Relaxed);
        let latency_ms_max = self.latency_ms_max.load(Ordering::Relaxed);
        let prefill_latency_ms_total = self.prefill_latency_ms_total.load(Ordering::Relaxed);
        let prefill_latency_ms_max = self.prefill_latency_ms_max.load(Ordering::Relaxed);
        let decode_latency_ms_total = self.decode_latency_ms_total.load(Ordering::Relaxed);
        let decode_latency_ms_max = self.decode_latency_ms_max.load(Ordering::Relaxed);
        let decode_compute_latency_ms_total =
            self.decode_compute_latency_ms_total.load(Ordering::Relaxed);
        let decode_compute_latency_ms_max =
            self.decode_compute_latency_ms_max.load(Ordering::Relaxed);
        let queued_requests = self.queued_requests.load(Ordering::Relaxed);
        let active_requests = self.active_requests.load(Ordering::Relaxed);
        let prefilling_requests = self.prefilling_requests.load(Ordering::Relaxed);
        let decoding_requests = self.decoding_requests.load(Ordering::Relaxed);
        let requests_finished = requests_completed + requests_failed;
        let prefill_batch_items_avg = Self::avg(prefill_batch_items_total, prefill_batches_total);
        let decode_batch_items_avg = Self::avg(decode_batch_items_total, decode_batches_total);
        let request_latency_ms_avg = Self::avg(latency_ms_total, requests_finished);
        let prefill_latency_ms_avg = Self::avg(prefill_latency_ms_total, prefill_chunks_total);
        let decode_latency_ms_avg = Self::avg(decode_latency_ms_total, requests_finished);
        let decode_compute_latency_ms_avg =
            Self::avg(decode_compute_latency_ms_total, decode_batches_total);

        format!(
            concat!(
                "# HELP llama_requests_total Total accepted inference requests.\n",
                "# TYPE llama_requests_total counter\n",
                "llama_requests_total {}\n",
                "# HELP llama_requests_completed_total Total completed inference requests.\n",
                "# TYPE llama_requests_completed_total counter\n",
                "llama_requests_completed_total {}\n",
                "# HELP llama_requests_failed_total Total failed inference requests.\n",
                "# TYPE llama_requests_failed_total counter\n",
                "llama_requests_failed_total {}\n",
                "# HELP llama_queue_full_total Total requests rejected because the scheduler queue was full.\n",
                "# TYPE llama_queue_full_total counter\n",
                "llama_queue_full_total {}\n",
                "# HELP llama_prompt_tokens_total Total prompt tokens accepted.\n",
                "# TYPE llama_prompt_tokens_total counter\n",
                "llama_prompt_tokens_total {}\n",
                "# HELP llama_prefill_tokens_total Total prompt tokens processed by prefill.\n",
                "# TYPE llama_prefill_tokens_total counter\n",
                "llama_prefill_tokens_total {}\n",
                "# HELP llama_prefill_chunks_total Total prompt chunks processed by prefill.\n",
                "# TYPE llama_prefill_chunks_total counter\n",
                "llama_prefill_chunks_total {}\n",
                "# HELP llama_prefill_batches_total Total scheduler prefill batches submitted.\n",
                "# TYPE llama_prefill_batches_total counter\n",
                "llama_prefill_batches_total {}\n",
                "# HELP llama_prefill_batch_items_total Total prefill requests included in scheduler prefill batches.\n",
                "# TYPE llama_prefill_batch_items_total counter\n",
                "llama_prefill_batch_items_total {}\n",
                "# HELP llama_prefill_batch_items_avg Average requests per scheduler prefill batch.\n",
                "# TYPE llama_prefill_batch_items_avg gauge\n",
                "llama_prefill_batch_items_avg {:.6}\n",
                "# HELP llama_prefill_batch_items_max Largest scheduler prefill batch size observed.\n",
                "# TYPE llama_prefill_batch_items_max gauge\n",
                "llama_prefill_batch_items_max {}\n",
                "# HELP llama_completion_tokens_total Total completion tokens generated.\n",
                "# TYPE llama_completion_tokens_total counter\n",
                "llama_completion_tokens_total {}\n",
                "# HELP llama_decode_batches_total Total scheduler decode batches submitted.\n",
                "# TYPE llama_decode_batches_total counter\n",
                "llama_decode_batches_total {}\n",
                "# HELP llama_decode_batch_items_total Total decode requests included in scheduler decode batches.\n",
                "# TYPE llama_decode_batch_items_total counter\n",
                "llama_decode_batch_items_total {}\n",
                "# HELP llama_decode_batch_items_avg Average requests per scheduler decode batch.\n",
                "# TYPE llama_decode_batch_items_avg gauge\n",
                "llama_decode_batch_items_avg {:.6}\n",
                "# HELP llama_decode_batch_items_max Largest scheduler decode batch size observed.\n",
                "# TYPE llama_decode_batch_items_max gauge\n",
                "llama_decode_batch_items_max {}\n",
                "# HELP llama_request_latency_ms_total Sum of request latency in milliseconds.\n",
                "# TYPE llama_request_latency_ms_total counter\n",
                "llama_request_latency_ms_total {}\n",
                "# HELP llama_request_latency_ms_avg Average completed request latency in milliseconds.\n",
                "# TYPE llama_request_latency_ms_avg gauge\n",
                "llama_request_latency_ms_avg {:.6}\n",
                "# HELP llama_request_latency_ms_max Largest completed request latency in milliseconds.\n",
                "# TYPE llama_request_latency_ms_max gauge\n",
                "llama_request_latency_ms_max {}\n",
                "# HELP llama_prefill_latency_ms_total Sum of prefill latency in milliseconds.\n",
                "# TYPE llama_prefill_latency_ms_total counter\n",
                "llama_prefill_latency_ms_total {}\n",
                "# HELP llama_prefill_latency_ms_avg Average prefill chunk latency in milliseconds.\n",
                "# TYPE llama_prefill_latency_ms_avg gauge\n",
                "llama_prefill_latency_ms_avg {:.6}\n",
                "# HELP llama_prefill_latency_ms_max Largest prefill chunk latency in milliseconds.\n",
                "# TYPE llama_prefill_latency_ms_max gauge\n",
                "llama_prefill_latency_ms_max {}\n",
                "# HELP llama_decode_latency_ms_total Sum of decode latency in milliseconds.\n",
                "# TYPE llama_decode_latency_ms_total counter\n",
                "llama_decode_latency_ms_total {}\n",
                "# HELP llama_decode_latency_ms_avg Average request decode phase latency in milliseconds.\n",
                "# TYPE llama_decode_latency_ms_avg gauge\n",
                "llama_decode_latency_ms_avg {:.6}\n",
                "# HELP llama_decode_latency_ms_max Largest request decode phase latency in milliseconds.\n",
                "# TYPE llama_decode_latency_ms_max gauge\n",
                "llama_decode_latency_ms_max {}\n",
                "# HELP llama_decode_compute_latency_ms_total Sum of decode GPU work latency in milliseconds.\n",
                "# TYPE llama_decode_compute_latency_ms_total counter\n",
                "llama_decode_compute_latency_ms_total {}\n",
                "# HELP llama_decode_compute_latency_ms_avg Average decode batch GPU work latency in milliseconds.\n",
                "# TYPE llama_decode_compute_latency_ms_avg gauge\n",
                "llama_decode_compute_latency_ms_avg {:.6}\n",
                "# HELP llama_decode_compute_latency_ms_max Largest decode batch GPU work latency in milliseconds.\n",
                "# TYPE llama_decode_compute_latency_ms_max gauge\n",
                "llama_decode_compute_latency_ms_max {}\n",
                "# HELP llama_queued_requests Current scheduler queue depth.\n",
                "# TYPE llama_queued_requests gauge\n",
                "llama_queued_requests {}\n",
                "# HELP llama_active_requests Current active scheduler requests.\n",
                "# TYPE llama_active_requests gauge\n",
                "llama_active_requests {}\n",
                "# HELP llama_prefilling_requests Current requests in prefill phase.\n",
                "# TYPE llama_prefilling_requests gauge\n",
                "llama_prefilling_requests {}\n",
                "# HELP llama_decoding_requests Current requests in decode phase.\n",
                "# TYPE llama_decoding_requests gauge\n",
                "llama_decoding_requests {}\n",
            ),
            requests_total,
            requests_completed,
            requests_failed,
            queue_full_total,
            prompt_tokens_total,
            prefill_tokens_total,
            prefill_chunks_total,
            prefill_batches_total,
            prefill_batch_items_total,
            prefill_batch_items_avg,
            prefill_batch_items_max,
            completion_tokens_total,
            decode_batches_total,
            decode_batch_items_total,
            decode_batch_items_avg,
            decode_batch_items_max,
            latency_ms_total,
            request_latency_ms_avg,
            latency_ms_max,
            prefill_latency_ms_total,
            prefill_latency_ms_avg,
            prefill_latency_ms_max,
            decode_latency_ms_total,
            decode_latency_ms_avg,
            decode_latency_ms_max,
            decode_compute_latency_ms_total,
            decode_compute_latency_ms_avg,
            decode_compute_latency_ms_max,
            queued_requests,
            active_requests,
            prefilling_requests,
            decoding_requests,
        )
    }

    fn avg(total: u64, count: u64) -> f64 {
        if count == 0 {
            0.0
        } else {
            total as f64 / count as f64
        }
    }

    fn record_max(metric: &AtomicU64, value: u64) {
        let _ = metric.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            if value > current {
                Some(value)
            } else {
                None
            }
        });
    }

    fn decrement_queued(&self) {
        let _ =
            self.queued_requests
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    current.checked_sub(1)
                });
    }

    fn decrement_prefilling(&self) {
        let _ = self.prefilling_requests.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |current| current.checked_sub(1),
        );
    }

    fn decrement_decoding(&self) {
        let _ =
            self.decoding_requests
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    current.checked_sub(1)
                });
    }
}
