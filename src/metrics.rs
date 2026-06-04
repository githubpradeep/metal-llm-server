use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

pub struct Metrics {
    requests_total: AtomicU64,
    requests_completed: AtomicU64,
    requests_failed: AtomicU64,
    queue_full_total: AtomicU64,
    prompt_tokens_total: AtomicU64,
    completion_tokens_total: AtomicU64,
    latency_ms_total: AtomicU64,
    queued_requests: AtomicUsize,
    active_requests: AtomicUsize,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            requests_completed: AtomicU64::new(0),
            requests_failed: AtomicU64::new(0),
            queue_full_total: AtomicU64::new(0),
            prompt_tokens_total: AtomicU64::new(0),
            completion_tokens_total: AtomicU64::new(0),
            latency_ms_total: AtomicU64::new(0),
            queued_requests: AtomicUsize::new(0),
            active_requests: AtomicUsize::new(0),
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

    pub fn record_finish(&self, finish_reason: &str, completion_tokens: usize, latency: Duration) {
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
        self.completion_tokens_total
            .fetch_add(completion_tokens as u64, Ordering::Relaxed);
        self.latency_ms_total
            .fetch_add(latency.as_millis() as u64, Ordering::Relaxed);

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
        let completion_tokens_total = self.completion_tokens_total.load(Ordering::Relaxed);
        let latency_ms_total = self.latency_ms_total.load(Ordering::Relaxed);
        let queued_requests = self.queued_requests.load(Ordering::Relaxed);
        let active_requests = self.active_requests.load(Ordering::Relaxed);

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
                "# HELP llama_completion_tokens_total Total completion tokens generated.\n",
                "# TYPE llama_completion_tokens_total counter\n",
                "llama_completion_tokens_total {}\n",
                "# HELP llama_request_latency_ms_total Sum of request latency in milliseconds.\n",
                "# TYPE llama_request_latency_ms_total counter\n",
                "llama_request_latency_ms_total {}\n",
                "# HELP llama_queued_requests Current scheduler queue depth.\n",
                "# TYPE llama_queued_requests gauge\n",
                "llama_queued_requests {}\n",
                "# HELP llama_active_requests Current active scheduler requests.\n",
                "# TYPE llama_active_requests gauge\n",
                "llama_active_requests {}\n",
            ),
            requests_total,
            requests_completed,
            requests_failed,
            queue_full_total,
            prompt_tokens_total,
            completion_tokens_total,
            latency_ms_total,
            queued_requests,
            active_requests,
        )
    }

    fn decrement_queued(&self) {
        let _ = self
            .queued_requests
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_sub(1)
            });
    }
}
