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

## Module 12: `kv_persist.rs` — On-Disk KV Cache Persistence

**Responsibility**: Save/load full KV cache state to SSD for instant session resume.

```rust
use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::{Read, Write, BufWriter, BufReader};

/// File format:
/// [magic: 4 bytes "KVRS"]
/// [version: u32]
/// [model_id: 32 bytes SHA256 of model config]
/// [quant_profile: u8 (0=q4, 1=f16, 2=fp8)]
/// [num_layers: u32]
/// [kv_seq_len: u32]
/// [token_history_len: u32]
/// [token_history: token_history_len × u32]
/// [per-layer KV data: num_layers × (k_cache_bytes + v_cache_bytes)]

pub struct KvPersistence {
    cache_dir: PathBuf,           // ~/.cache/gemma4-server/sessions/
    model_id: [u8; 32],          // SHA256 of config.json for compatibility check
    quant_profile: u8,
}

pub struct SessionMetadata {
    pub session_id: String,       // SHA1 of first user message
    pub token_count: u32,
    pub created_at: u64,
    pub last_used: u64,
    pub file_size: u64,
}

impl KvPersistence {
    pub fn new(cache_dir: &Path, model_id: [u8; 32], quant_profile: u8) -> Self { ... }

    /// Save current KV state to disk. ~88 MB writes at ~4 GB/s SSD = ~22ms.
    pub fn save_session(
        &self,
        session_id: &str,
        k_caches: &[Buffer],       // GPU buffers — read back to CPU then write
        v_caches: &[Buffer],
        kv_seq_len: u32,
        token_history: &[u32],
    ) -> Result<PathBuf, Error> { ... }

    /// Load KV state from disk. Validates model_id + quant_profile match.
    pub fn load_session(
        &self,
        session_id: &str,
    ) -> Result<(Vec<Vec<u8>>, Vec<Vec<u8>>, u32, Vec<u32>), Error> { ... }

    /// Find best matching session for a given prompt (longest prefix match).
    pub fn find_prefix_match(
        &self,
        token_ids: &[u32],
    ) -> Option<(SessionMetadata, u32)> { ... }  // returns (metadata, common_prefix_len)

    /// List all saved sessions, sorted by last_used.
    pub fn list_sessions(&self) -> Vec<SessionMetadata> { ... }

    /// Evict sessions exceeding budget (LRU).
    pub fn evict_to_budget(&self, budget_bytes: u64) { ... }
}
```

**Integration with scheduler**:
```rust
// On request arrival:
// 1. Tokenize prompt
// 2. Check kv_persist.find_prefix_match(tokens)
// 3. If match found and prefix_len > threshold:
//    - Load session from disk (~20ms)
//    - Write KV buffers to GPU
//    - Only prefill the suffix tokens[prefix_len..]
// 4. On conversation turn completion:
//    - Background-save current KV state to disk
```

**Performance**: M1 Pro SSD writes 88 MB in ~22ms (4 GB/s). Reads same in ~18ms (5 GB/s sequential). Async I/O via `tokio::fs` so it doesn't block the scheduler.

---

## Module 13: `fp8_kv.rs` — FP8 KV Cache

**Responsibility**: Pack/unpack KV values in 8-bit floating point to double context capacity.

```rust
/// FP8 E4M3 format: 1 sign + 4 exponent + 3 mantissa bits
/// Range: ±448, precision: ~0.1% relative error
/// Perfect for attention keys/values which are normalized by QK-norm

pub struct Fp8KvCache {
    k_caches: Vec<Buffer>,    // [layer][num_kv_heads × capacity × head_dim] in FP8
    v_caches: Vec<Buffer>,    // same shape, FP8
    capacity: u32,
    seq_len: u32,
}
```

**Metal kernels needed**:
```metal
// Quantize f32 → FP8 E4M3 during KV append
kernel void kv_append_fp8(
    device const float* new_kv,     // f32 input from projection
    device uchar* cache,            // FP8 packed cache
    constant uint& position,
    constant uint& head_dim,
    uint tid [[thread_position_in_grid]]
) {
    float val = new_kv[tid];
    // Clamp to FP8 E4M3 range [-448, 448]
    val = clamp(val, -448.0f, 448.0f);
    // Pack: sign(1) | exponent(4) | mantissa(3)
    cache[position * head_dim + tid] = float_to_fp8_e4m3(val);
}

// Dequantize FP8 → f32 during attention score computation
// (integrated into attention kernel, not standalone)
```

**Memory savings**:
| Context | f16 KV | FP8 KV | Savings |
|---------|--------|--------|---------|
| 1024 | 88 MB | 44 MB | 44 MB |
| 4096 | 352 MB | 176 MB | 176 MB |
| 8192 | 704 MB | 352 MB | 352 MB |

**Quality validation**: Run full eval suite comparing f16 vs FP8 KV outputs. DeepSeek V4 (ds4) confirms FP8 KV works well in practice.

---

## Module 14: `fused_mlp.metal` — Fused Gate+Up+Activation Kernel

**Responsibility**: Combine gate and up projections with GeLU activation into a single dispatch.

```metal
// Current: 3 dispatches per layer
//   1. matvec_q4(gate_proj, x) → gate_buf
//   2. matvec_q4(up_proj, x)   → up_buf
//   3. gelu_mul(gate_buf, up_buf) → out_buf

// Fused: 1 dispatch per layer
//   matvec_q4_pair_gelu(gate_proj, up_proj, x) → out_buf
kernel void matvec_q4_pair_gelu(
    device const uchar* W_gate [[buffer(0)]],
    device const uchar* W_up   [[buffer(1)]],
    device const float* x      [[buffer(2)]],
    device float* out          [[buffer(3)]],
    constant uint& M           [[buffer(4)]],  // intermediate_size
    constant uint& K           [[buffer(5)]],  // hidden_size
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint row = tgid;

    // Compute gate and up dot products cooperatively
    float gate_acc = 0.0;
    float up_acc = 0.0;

    for (uint g = tid; g < num_groups; g += 32) {
        // ... Q4 dequant for gate weights ...
        gate_acc += local_gate * scale_gate;
        // ... Q4 dequant for up weights ...
        up_acc += local_up * scale_up;
    }

    gate_acc = simd_sum(gate_acc);
    up_acc = simd_sum(up_acc);

    if (tid == 0) {
        // Fused GeLU(gate) * up
        float gelu_gate = gate_acc * 0.5 * (1.0 + tanh(0.7978845608 * (gate_acc + 0.044715 * gate_acc * gate_acc * gate_acc)));
        out[row] = gelu_gate * up_acc;
    }
}
```

**Benefits**:
- Eliminates 2 intermediate buffer writes per layer (gate_buf, up_buf)
- Saves ~84 kernel dispatches per token (42 layers × 2 eliminated)
- Both matvecs read the same input `x` — shared in registers

**Expected speedup**: 5–8% decode throughput.

---

## Module 15: Power Throttling

**Responsibility**: Configurable GPU utilization cap for thermal/noise/battery management.

```rust
pub struct PowerThrottle {
    target_percent: u32,        // 0-100, default 100
    last_token_gpu_time: Duration,
    sleep_per_token: Duration,  // computed from target and measured GPU time
}

impl PowerThrottle {
    pub fn new(target_percent: u32) -> Self { ... }

    /// Call after each decode token. Sleeps if needed to hit target utilization.
    pub fn throttle_after_token(&mut self, gpu_time: Duration) {
        if self.target_percent >= 100 { return; }

        // target_pct = gpu_time / (gpu_time + sleep_time)
        // sleep_time = gpu_time * (100 / target_pct - 1)
        let ratio = (100.0 / self.target_percent as f64) - 1.0;
        let sleep = Duration::from_secs_f64(gpu_time.as_secs_f64() * ratio);
        std::thread::sleep(sleep);
    }

    /// Call between prefill chunks for chunk-level throttling.
    pub fn throttle_after_chunk(&mut self, chunk_gpu_time: Duration) {
        // Same logic, applied per-chunk during prefill
        self.throttle_after_token(chunk_gpu_time);
    }
}
```

**API endpoint**: `POST /v1/power` with `{"percent": 50}` to adjust at runtime.
**CLI**: `--power 70` flag.

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
├── kv_persist.rs        (NEW: on-disk KV cache save/load)
├── fp8_kv.rs            (NEW: FP8 KV cache quantization)
├── power.rs             (NEW: thermal/power throttling)
├── gemma4_gpu_model.rs  (existing model, add batch methods)
├── gemma4_config.rs     (existing config)
├── gpu.rs               (existing Metal context + kernels)
├── sampling.rs          (enhanced with rep penalty, top-k, stop seqs)
├── shaders/
│   ├── llama.metal      (existing + new batch kernels)
│   ├── flash_attn.metal (NEW: tiled attention shader)
│   └── fused_mlp.metal  (NEW: fused gate+up+gelu kernel)
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
| T15: On-disk KV persistence | `kv_persist.rs` | T4 | Medium |
| T16: FP8 KV cache | `fp8_kv.rs`, `shaders/llama.metal` | T4 | Medium |
| T17: Fused MLP kernels | `shaders/fused_mlp.metal`, `gpu.rs` | — | Medium |
| T18: Power throttling | `power.rs`, `server.rs` | — | Small |

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
12. **KV persistence uses raw GPU readback + sequential write** — mmap or async I/O adds complexity; at 88 MB the write completes in ~22ms which is fast enough to do synchronously between turns
13. **FP8 E4M3 chosen over E5M2** — E4M3 has higher precision (3 mantissa bits vs 2) at the cost of smaller range (±448 vs ±57344); KV values are normalized by QK-norm so they stay well within ±448
14. **Fused MLP only for Q4 layers initially** — f16 layers are already fast; the fusion benefit is largest for Q4 where the dequantization cost dominates and sharing the input vector across two matvecs saves register pressure
15. **Power throttling at token granularity** — sleeping between layers would cause pipeline bubbles; sleeping between tokens is clean and predictable
