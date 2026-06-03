# Production Inference Server — Design

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      HTTP Layer (axum)                        │
│  /v1/chat/completions  /v1/models  /health  /metrics         │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                      Request Scheduler                        │
│  - Accepts requests into priority queue                      │
│  - Manages request lifecycle (pending → prefill → decode)    │
│  - Enforces timeouts, max concurrent, max queue depth        │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                      Batch Engine                             │
│  - Groups decode tokens from active requests into batches    │
│  - Interleaves chunked prefill with decode batches           │
│  - Calls GPU forward pass with batched inputs                │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                      GPU Model (Metal)                        │
│  - Parallel prefill (matmul path)                            │
│  - Batched decode (batched matvec)                           │
│  - f16 KV cache pool with slot management                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Module Design

### Module 1: `scheduler.rs` — Request Scheduler

**Responsibility**: Manage the lifecycle of inference requests.

```rust
pub struct Scheduler {
    pending: VecDeque<InferenceRequest>,  // waiting for GPU
    active: Vec<ActiveRequest>,           // currently generating
    kv_pool: KvCachePool,                 // manages cache slots
    max_active: usize,                    // max concurrent generations (4)
    max_queue: usize,                     // max pending queue depth (32)
    request_timeout: Duration,            // per-request timeout (60s)
}

pub struct InferenceRequest {
    id: String,
    messages: Vec<Message>,
    params: GenerationParams,
    response_tx: tokio::sync::mpsc::Sender<StreamEvent>,
    created_at: Instant,
}

pub struct ActiveRequest {
    id: String,
    kv_slot: usize,                       // assigned KV cache slot
    token_ids: Vec<usize>,                // generated tokens so far
    logits: Vec<f32>,                     // last logits
    params: GenerationParams,
    response_tx: tokio::sync::mpsc::Sender<StreamEvent>,
    state: RequestState,                  // Prefilling | Decoding | Done
}

pub enum RequestState {
    Prefilling { tokens: Vec<usize>, position: usize, chunk_size: usize },
    Decoding,
    Done,
}
```

**Scheduler loop** (runs on a dedicated thread):
```
loop {
    1. Check for timed-out requests → send error, reclaim slot
    2. If active slots available && pending queue non-empty:
       - Dequeue request, assign KV slot, start prefill
    3. Collect one batch:
       - For each active request in Prefilling state: take next chunk (64 tokens)
       - For each active request in Decoding state: take 1 token to generate
    4. Submit batch to GPU engine
    5. Distribute results:
       - Prefill chunks: update position, transition to Decoding when done
       - Decode tokens: sample, check EOS, stream to client
    6. Reclaim slots for Done requests
}
```

---

### Module 2: `kv_pool.rs` — KV Cache Pool

**Responsibility**: Pre-allocate and manage per-request KV cache slots.

```rust
pub struct KvCachePool {
    num_slots: usize,                     // e.g., 4
    max_seq_len: usize,                   // 1024
    k_caches: Vec<Vec<Buffer>>,           // [slot][layer] -> Buffer
    v_caches: Vec<Vec<Buffer>>,           // [slot][layer] -> Buffer
    slot_seq_lens: Vec<u32>,              // current seq len per slot
    free_slots: Vec<usize>,              // available slot indices
}

impl KvCachePool {
    fn allocate(&mut self) -> Option<usize>;        // get a free slot
    fn release(&mut self, slot: usize);             // return slot to pool
    fn reset(&mut self, slot: usize);               // zero out seq_len
    fn seq_len(&self, slot: usize) -> u32;
    fn k_cache(&self, slot: usize, layer: usize) -> &Buffer;
    fn v_cache(&self, slot: usize, layer: usize) -> &Buffer;
}
```

**Memory budget** (M1 Pro 16GB):
- Per slot: 42 layers × 2 heads × 1024 positions × 512 max_head_dim × 2 bytes = ~88 MB
- 4 slots: ~352 MB (fits easily)

---

### Module 3: `batch_engine.rs` — Batch Engine

**Responsibility**: Execute GPU forward passes for batched requests.

```rust
pub struct BatchEngine {
    model: Gemma4GpuModel,                // owns the model weights
    kv_pool: KvCachePool,                 // shared KV cache pool
}

pub struct BatchInput {
    // Prefill chunks: (slot_id, token_ids, start_position)
    prefill_chunks: Vec<(usize, Vec<usize>, usize)>,
    // Decode tokens: (slot_id, last_token_id)
    decode_tokens: Vec<(usize, usize)>,
}

pub struct BatchOutput {
    // Per-slot logits for decode
    decode_logits: Vec<(usize, Vec<f32>)>,
    // Prefill completion status
    prefill_done: Vec<usize>,             // slot_ids that finished prefill
}
```

**Execution strategy**:
1. **Prefill chunks first** (higher priority, unblocks new requests):
   - For each chunk: run parallel prefill (matmul) into the slot's KV cache
   - Use existing `encode_matmul`, `encode_rmsnorm_batch`, etc.
2. **Decode batch**:
   - Collect hidden states from all decode slots
   - Run batched forward: each slot processes 1 token through all layers
   - For single-token decode, this is effectively N independent matvecs (no cross-request interaction)
   - Optimization: batch the matvecs into a matmul when N > 1

---

### Module 4: `prefill.rs` — Parallel Prefill Path

**Responsibility**: Process all prompt tokens in one pass per layer.

**Data flow for seq_len=S tokens**:
```
embed: [S, hidden_size]
  → per layer:
    rmsnorm_batch: [S, hidden_size]
    matmul Q: [S, hidden_size] × [q_out, hidden_size]^T → [S, q_out]
    matmul K: [S, hidden_size] × [kv_out, hidden_size]^T → [S, kv_out]
    matmul V: [S, hidden_size] × [kv_out, hidden_size]^T → [S, kv_out]
    QK norm (per-head per-position)
    rotary_batch (all positions at once)
    KV cache batch append (write S positions)
    causal attention (S queries × S keys, masked)
    matmul O: [S, q_out] × [hidden_size, q_out]^T → [S, hidden_size]
    residual add batch
    rmsnorm_batch
    matmul gate/up: [S, hidden_size] × [inter, hidden_size]^T → [S, inter]
    gelu_mul_batch
    matmul down: [S, inter] × [hidden_size, inter]^T → [S, hidden_size]
    residual add batch
    PLE batch
    layer_scalar batch
  → final_norm_batch
  → lm_head: only last position → [vocab_size] logits
```

**Key changes to existing code**:
- Add `forward_prefill_parallel(token_ids: &[usize], kv_slot: usize) -> Vec<f32>`
- Reuse existing batch kernels: `encode_matmul`, `encode_rmsnorm_batch`, `encode_rotary_batch`, `encode_kv_batch_append`, `encode_attention_causal`
- Need new: `encode_gelu_mul_batch`, per-head norm batch, PLE batch
- Scratch buffers sized for max batch: `[max_seq × hidden_size]` etc.

---

### Module 5: `server.rs` — HTTP Layer (updated)

**Changes from current**:
- Remove `Mutex<Model>` — model owned by BatchEngine on scheduler thread
- Requests submitted via channel to scheduler
- Response streamed back via per-request mpsc channel
- Add CORS middleware
- Add `/metrics` endpoint
- Add error handling middleware (catch panics → 500)

```rust
pub async fn run_server(model: Gemma4GpuModel, tokenizer: Tokenizer, port: u16) {
    let (request_tx, request_rx) = mpsc::channel(64);
    
    // Spawn scheduler on dedicated thread (owns model + GPU)
    std::thread::spawn(move || {
        let mut scheduler = Scheduler::new(model, config);
        scheduler.run(request_rx);  // blocking loop
    });
    
    // HTTP server on tokio runtime
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_handler))
        .route("/v1/models", get(models_handler))
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .layer(CorsLayer::permissive())
        .with_state(AppState { request_tx, tokenizer });
    
    axum::serve(listener, app).await.unwrap();
}
```

---

### Module 6: `sampling.rs` — Enhanced Sampling (updated)

**Add**:
- `repetition_penalty(logits, generated_tokens, penalty)` — penalize repeated tokens
- `frequency_penalty(logits, token_counts, penalty)` — penalize frequent tokens
- `top_k_filter(logits, k)` — zero out all but top-k before softmax
- `check_stop_sequences(text, stop_seqs)` — string-based stop detection

```rust
pub struct GenerationParams {
    pub temperature: f32,
    pub top_k: usize,           // 0 = disabled
    pub min_p: f32,
    pub repetition_penalty: f32, // 1.0 = disabled, 1.1 = typical
    pub max_tokens: usize,
    pub stop_sequences: Vec<String>,
    pub eos_token_ids: Vec<usize>,
}

pub fn sample_with_params(logits: &mut [f32], params: &GenerationParams, history: &[usize]) -> usize {
    apply_repetition_penalty(logits, history, params.repetition_penalty);
    if params.top_k > 0 { apply_top_k(logits, params.top_k); }
    sample(logits, params.temperature, params.min_p)
}
```

---

## Data Flow: Single Request Lifecycle

```
1. HTTP POST /v1/chat/completions
   → Parse JSON, validate, apply chat template
   → Create InferenceRequest, send to scheduler channel

2. Scheduler receives request
   → If slot available: assign KV slot, begin prefill
   → If no slot: queue (or reject if queue full)

3. Prefill (parallel):
   → Process prompt in 64-token chunks via matmul path
   → Each chunk writes to KV cache, advances position
   → Last chunk: extract logits from final position

4. Decode loop:
   → Sample next token from logits
   → Check EOS / stop sequences / max_tokens
   → Stream token to client via SSE
   → Run forward_single_token to get next logits
   → Repeat until done

5. Completion:
   → Send final SSE event with finish_reason
   → Release KV slot back to pool
   → Log request metrics
```

---

---

## Module 7: `flash_attention.rs` — Tiled Attention (Metal)

**Responsibility**: Compute causal self-attention without materializing the full S×S matrix.

**Algorithm (Online Softmax + Tiling)**:
```
For each query tile Q_i (block of Br rows):
    m_i = -inf, l_i = 0, O_i = 0
    For each key tile K_j (block of Bc cols):
        S_ij = Q_i × K_j^T / sqrt(d)          // [Br × Bc] tile in SRAM
        Apply causal mask (set future positions to -inf)
        m_new = max(m_i, rowmax(S_ij))
        P_ij = exp(S_ij - m_new)
        l_new = exp(m_i - m_new) * l_i + rowsum(P_ij)
        O_i = exp(m_i - m_new) * O_i + P_ij × V_j
        m_i = m_new, l_i = l_new
    O_i = O_i / l_i                            // final normalization
```

**Metal kernel design**:
```rust
pub struct FlashAttentionConfig {
    block_size_q: usize,    // Br = 32 or 64 (tuned for M1 threadgroup memory)
    block_size_kv: usize,   // Bc = 32 or 64
    head_dim: usize,        // 128 for Gemma4
    num_heads: usize,
    causal: bool,
}

// Metal shader: one threadgroup per (batch, head, q_block) triplet
// Shared memory: Q tile [Br × d], K tile [Bc × d], V tile [Bc × d], S tile [Br × Bc]
// Output: O [Br × d] written to device memory
```

**Integration**:
- Replace `encode_attention_causal` in prefill path when seq_len > threshold (e.g., 256)
- Decode path (seq_len=1 query against full KV) doesn't benefit — keep existing kernel
- Threadgroup memory budget on M1: 32KB → Br=32, Bc=32, d=128, f16 → fits

**Memory savings** (4096 context):
- Naive: 4096 × 4096 × 2 bytes × 8 heads = 512 MB per layer (impossible)
- Tiled: Br × Bc × 2 bytes × 8 heads in threadgroup memory = 32 KB (trivial)

---

## Module 8: `radix_cache.rs` — Prefix Sharing

**Responsibility**: Deduplicate KV cache storage for shared prompt prefixes across requests.

```rust
pub struct RadixTree {
    root: RadixNode,
}

pub struct RadixNode {
    token_ids: Vec<usize>,              // edge label (token sequence)
    kv_block_ids: Vec<usize>,           // indices into KV block pool
    children: HashMap<usize, RadixNode>, // keyed by first token of child edge
    ref_count: usize,                   // number of active requests using this prefix
    last_access: Instant,               // for LRU eviction
}

pub struct KvBlockPool {
    blocks: Vec<KvBlock>,               // fixed-size blocks (e.g., 64 positions each)
    free_list: Vec<usize>,
}

pub struct KvBlock {
    k_data: Vec<Buffer>,                // [layer] -> f16 buffer, 64 × head_dim × num_kv_heads
    v_data: Vec<Buffer>,
    seq_start: usize,                   // position offset within the full sequence
    len: usize,                         // actual tokens stored (≤ block_size)
}
```

**Lookup flow**:
1. New request arrives with token_ids `[t0, t1, ..., tN]`
2. Walk radix tree matching longest prefix → get existing KV blocks
3. Only prefill the suffix `[t_matched+1, ..., tN]`
4. Append new KV blocks for suffix, insert into tree
5. On request completion: decrement ref_count, evict if ref_count=0 and under memory pressure

**Eviction policy**: LRU among ref_count=0 nodes. Keep high-value prefixes (system prompts, common conversation starters) warm.

---

## Module 9: `speculative.rs` — Speculative Decoding

**Responsibility**: Accelerate decode by proposing multiple tokens with a draft model and verifying in one pass.

```rust
pub struct SpeculativeDecoder {
    draft_model: DraftModel,            // small model (e.g., 2B params)
    main_model: Gemma4GpuModel,         // full model (verification)
    num_speculative: usize,             // N = 4–8 draft tokens per step
}

pub struct DraftModel {
    // Lightweight model sharing tokenizer with main model
    // Could be: distilled head, early-exit from main model, or separate small model
}
```

**Algorithm (per decode step)**:
```
1. Draft phase: run draft model autoregressively for N tokens
   draft_tokens = [d1, d2, ..., dN]
   draft_probs  = [p1, p2, ..., pN]  (draft model's probability for each token)

2. Verify phase: run main model on [current_token, d1, d2, ..., dN] in ONE forward pass
   main_probs = [q1, q2, ..., qN, q_{N+1}]  (main model's probs at each position)

3. Acceptance (rejection sampling):
   For i = 1 to N:
     if random() < min(1, q_i[d_i] / p_i[d_i]):
       accept d_i
     else:
       resample from adjusted distribution: (q_i - p_i)+ normalized
       reject remaining tokens
       break

4. Bonus token: if all N accepted, sample one more from q_{N+1}
```

**Expected speedup**:
- Acceptance rate α ≈ 0.7 for well-matched draft model
- Expected tokens per step: (1 - α^(N+1)) / (1 - α) ≈ 3.0 for N=5, α=0.7
- Verification cost: ~1.2x single decode (N+1 tokens through main model, matmul not matvec)
- Net speedup: ~2.5x decode throughput

**Integration**:
- Works within existing scheduler: speculative decode is an optimization of the decode step
- KV cache must support rollback (discard rejected positions)
- Draft model can run on CPU (small enough) or share GPU

---

## Module 10: `overlap.rs` — Pipeline & Metal Captured Buffers

**Responsibility**: Eliminate scheduling gaps and kernel launch overhead.

**CPU/GPU Overlap**:
```rust
pub struct PipelinedScheduler {
    // Double-buffer pattern:
    // While GPU executes batch N, CPU prepares batch N+1
    current_batch: BatchInput,
    next_batch: Option<BatchInput>,
    
    // Metal event for synchronization
    gpu_done_event: metal::Event,
}
```

**Metal Indirect Command Buffers** (decode path):
```
// Pre-encode the full decode forward pass (42 layers × K kernels) into an ICB
// Since decode always processes exactly 1 token per slot with fixed buffer layouts,
// the command structure is identical every iteration — only the input token changes

pub struct CapturedDecodePass {
    indirect_buffer: metal::IndirectCommandBuffer,
    // Argument buffers that get updated each iteration:
    input_token_buffer: Buffer,        // overwritten each step
    output_logits_buffer: Buffer,      // read after each step
    kv_seq_len_buffer: Buffer,         // incremented each step
}

// Per decode step:
// 1. Write new token_id to input_token_buffer
// 2. Increment seq_len in kv_seq_len_buffer
// 3. Execute ICB (single API call replaces 200+ individual dispatches)
```

**Expected gains**:
- Kernel launch overhead: ~10μs × 200 dispatches = 2ms per decode step
- With ICB: single executeIndirect call ≈ 20μs
- Net saving: ~1.8ms per token → at 14 tok/s that's a 2.5% speedup (more impactful at higher throughput)

---

## Module 11: `tokenizer_pool.rs` — Tokenizer Workers

**Responsibility**: Parallel tokenization without blocking GPU scheduler or async runtime.

```rust
pub struct TokenizerPool {
    pool: rayon::ThreadPool,            // dedicated CPU thread pool
    tokenizer: Arc<Tokenizer>,          // shared tokenizer (Tokenizer is Send+Sync)
}

impl TokenizerPool {
    pub fn new(tokenizer: Tokenizer, num_threads: usize) -> Self {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("tokenizer-{}", i))
            .build()
            .unwrap();
        Self { pool, tokenizer: Arc::new(tokenizer) }
    }

    pub async fn encode(&self, text: &str) -> Vec<usize> {
        let tokenizer = self.tokenizer.clone();
        let text = text.to_string();
        tokio::task::spawn_blocking(move || {
            tokenizer.encode(&text)
        }).await.unwrap()
    }

    pub async fn decode(&self, token_ids: &[usize]) -> String {
        let tokenizer = self.tokenizer.clone();
        let ids = token_ids.to_vec();
        tokio::task::spawn_blocking(move || {
            tokenizer.decode(&ids)
        }).await.unwrap()
    }
}
```

**Integration**: Replace direct `tokenizer.encode()` calls in HTTP handlers with `tokenizer_pool.encode().await`.

---

## Updated File Structure

```
src/
├── main.rs              (entry point, arg parsing)
├── server.rs            (HTTP handlers, CORS, error handling)
├── scheduler.rs         (NEW: request lifecycle, queue, batching loop)
├── kv_pool.rs           (NEW: KV cache slot management)
├── batch_engine.rs      (NEW: batched GPU execution)
├── prefill.rs           (NEW: parallel prefill forward pass)
├── flash_attention.rs   (NEW: tiled attention Metal kernel)
├── radix_cache.rs       (NEW: prefix-sharing KV cache tree)
├── speculative.rs       (NEW: speculative decoding with draft model)
├── overlap.rs           (NEW: pipeline scheduling + Metal ICBs)
├── tokenizer_pool.rs    (NEW: parallel tokenization workers)
├── gemma4_gpu_model.rs  (existing model, add batch methods)
├── gemma4_config.rs     (existing config)
├── gpu.rs               (existing Metal context + kernels)
├── sampling.rs          (enhanced with rep penalty, top-k, stop seqs)
├── shaders/
│   ├── llama.metal      (existing + new batch kernels)
│   └── flash_attn.metal (NEW: tiled attention shader)
└── metrics.rs           (NEW: counters, /metrics endpoint)
```

---

## Updated Implementation Order

| Task | Module | Depends On | Effort |
|------|--------|-----------|--------|
| T1: Parallel prefill for Gemma4 | `prefill.rs`, `gemma4_gpu_model.rs` | — | Large |
| T2: Enhanced sampling | `sampling.rs` | — | Small |
| T3: Server robustness (errors, CORS, timeout, context limit) | `server.rs` | — | Small |
| T4: KV cache pool | `kv_pool.rs` | — | Medium |
| T5: Request scheduler | `scheduler.rs` | T4 | Medium |
| T6: Integrate scheduler with server | `server.rs` | T5 | Medium |
| T7: Continuous batching in decode | `batch_engine.rs` | T5, T4 | Large |
| T8: Chunked prefill | `batch_engine.rs` | T1, T7 | Medium |
| T9: Metrics & logging | `metrics.rs`, `server.rs` | T6 | Small |
| T10: FlashAttention Metal kernel | `flash_attention.rs`, `shaders/flash_attn.metal` | T1 | Large |
| T11: Radix cache / prefix sharing | `radix_cache.rs` | T4 | Large |
| T12: Speculative decoding | `speculative.rs` | T1, T7 | Large |
| T13: Overlap scheduling + Metal ICBs | `overlap.rs` | T7 | Medium |
| T14: Tokenizer workers | `tokenizer_pool.rs` | T6 | Small |

---

## Key Design Decisions

1. **Scheduler on dedicated OS thread** (not tokio task) — GPU work is blocking and shouldn't pollute the async runtime
2. **Channel-based communication** between HTTP layer and scheduler — clean separation, no shared mutable state
3. **KV pool pre-allocated at startup** — no runtime allocation on hot path
4. **Prefill chunks of 64 tokens** — balances latency (not too long blocking) with throughput (matmul efficient at 64+)
5. **Decode "batching" is implicit** — for M1 Pro, true batched matvec has limited benefit since we're memory-bandwidth-bound. The real win is overlapping prefill with decode for latency fairness
6. **Model weights shared (read-only)** — only KV cache is per-request mutable state
7. **FlashAttention threshold at 256 tokens** — below this, naive attention fits in memory cheaply; above, tiled kernel avoids quadratic blowup
8. **Radix cache with block-granular storage** — 64-token blocks balance prefix-sharing granularity with metadata overhead; LRU eviction keeps hot prefixes warm
9. **Speculative decoding as scheduler-transparent optimization** — the scheduler sees it as a single decode step that produces multiple tokens; no architectural coupling
10. **Metal ICBs for decode only** — decode has fixed shapes (1 token per slot); prefill varies per request and can't be pre-captured
11. **Tokenizer on rayon pool, not tokio** — CPU-bound BPE work should not compete with async I/O; spawn_blocking bridges the async/sync boundary
