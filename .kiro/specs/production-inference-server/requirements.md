# Production Inference Server — Requirements

## Context
We have a working Gemma4 E4B inference server (Rust/Metal) with:
- Mixed Q4/f16 quantization, ~14 tok/s decode on M1 Pro
- OpenAI-compatible API (streaming + non-streaming)
- Single-user, sequential token-by-token prefill
- f16 KV cache, 1024 max context

The goal is to make this production-ready for real users.

---

## Requirements

### 1. Parallel Prefill
- **What**: Process all prompt tokens in a single forward pass using matrix-matrix multiply instead of sequential token-by-token
- **Why**: Current prefill of a 50-token prompt takes ~3.5s (50 × matvec). Parallel prefill should take ~0.3s (1 × matmul)
- **Acceptance**: Prefill throughput > 200 tok/s for prompts up to 512 tokens
- **Constraint**: Must produce identical KV cache state as sequential prefill (correctness preserved)

### 2. Request Queue & Concurrency
- **What**: Accept multiple HTTP requests concurrently; queue them and process one at a time on the GPU
- **Why**: Current Mutex causes the second request to block silently; no timeout, no feedback
- **Acceptance**: 10 concurrent requests handled gracefully; pending requests get response within request timeout
- **Constraint**: Single GPU, single model instance; no need for true parallel batching initially

### 3. KV Cache Pool
- **What**: Pre-allocate N KV cache slots; assign one per active request; reclaim on completion
- **Why**: Currently one global cache is reset per request. For future batching, each request needs its own cache
- **Acceptance**: Support at least 4 concurrent cache slots; graceful rejection when pool is full
- **Constraint**: Total KV memory must fit in GPU memory (M1 Pro 16GB shared)

### 4. Continuous Batching
- **What**: Batch decode tokens from multiple active requests into a single GPU forward pass
- **Why**: Increases GPU utilization when serving multiple users; higher aggregate throughput
- **Acceptance**: 2-4 concurrent decode streams batched; aggregate throughput > 1.5x single-user
- **Constraint**: Each request maintains independent generation state (sampling, EOS detection, token count)

### 5. Chunked Prefill
- **What**: Break long prompts into chunks (e.g., 64 tokens) and interleave with decode tokens between chunks
- **Why**: Prevents a single long prompt from blocking all other requests during prefill
- **Acceptance**: A 512-token prefill does not add more than 200ms latency to concurrent decode requests
- **Constraint**: Requires continuous batching to be implemented first

### 6. Sampling & Generation Quality
- **What**: Repetition penalty, frequency penalty, stop sequences from API, top-k filtering
- **Why**: Prevents output loops; clients expect these parameters to work
- **Acceptance**: `repetition_penalty`, `stop`, `top_k` parameters respected in API
- **Constraint**: Must not degrade throughput by more than 5%

### 7. Server Robustness
- **What**: Proper error handling, max context enforcement, request timeout, CORS, graceful shutdown
- **Why**: Production servers can't panic on bad input or hang forever
- **Acceptance**: Bad JSON → 400; prompt too long → 400 with message; generation timeout → partial response; Ctrl+C → finish current then exit
- **Constraint**: No external runtime dependencies beyond what's in Cargo.toml

### 8. Observability
- **What**: Request logging, latency metrics, token throughput counters, queue depth
- **Why**: Can't operate what you can't observe
- **Acceptance**: Structured JSON logs per request; /metrics endpoint with key counters
- **Constraint**: Minimal overhead; no external telemetry services required

---

### 9. FlashAttention / Tiled Attention (Metal)
- **What**: Implement a tiled attention kernel that computes causal self-attention in SRAM tiles, avoiding materializing the full S×S attention matrix in memory
- **Why**: Standard attention is O(S²) in memory. At 4K context, that's 64MB per layer per head just for the attention matrix. Tiled attention reduces HBM reads to O(S) and unlocks longer contexts
- **Acceptance**: Support 4096+ context length; attention memory usage sub-linear in sequence length; identical output to naive attention (within f16 tolerance)
- **Constraint**: Must work on Metal (no CUDA); requires online softmax (streaming max/sum correction across tiles)

### 10. Radix Cache / Prefix Sharing
- **What**: Share KV cache entries across requests that have identical prompt prefixes (e.g., system prompt, shared conversation history)
- **Why**: In multi-turn chat, every request in a session shares the system prompt + prior turns. Recomputing and storing this redundantly wastes both compute and memory
- **Acceptance**: Second request in same conversation skips prefill for shared prefix; KV memory usage for N sessions with shared prefix ≈ 1× prefix + N× unique suffix (not N× full)
- **Constraint**: Must handle invalidation when prefix diverges; tree-based lookup with O(log N) search

### 11. Speculative Decoding
- **What**: Use a small draft model to propose N candidate tokens, then verify all N in a single forward pass of the main model, accepting a prefix of correct tokens
- **Why**: Decode is memory-bandwidth-bound (1 token at a time). Speculative decoding amortizes the weight-loading cost across multiple accepted tokens, giving 2–3x decode speedup
- **Acceptance**: Average acceptance rate > 60% with matched draft model; end-to-end decode throughput > 2x single-token baseline; output distribution identical to standard sampling (rejection sampling guarantees this)
- **Constraint**: Requires a draft model (e.g., Gemma 2B or distilled head); draft model must share tokenizer with main model

### 12. Overlap Scheduling & Metal Captured Buffers
- **What**: Pipeline GPU compute with CPU scheduling so batch N+1 is prepared while batch N executes; use Metal indirect command buffers to eliminate per-iteration kernel launch overhead in the decode loop
- **Why**: Kernel launch overhead (~10–20μs per dispatch) adds up across 42 layers × multiple kernels. CPU/GPU overlap eliminates scheduling stalls between iterations
- **Acceptance**: Decode loop shows < 5% idle gap between GPU commands (measured via Metal System Trace); per-token latency reduced by 10–20% from launch overhead elimination
- **Constraint**: Metal captured buffers require fixed tensor shapes — only applicable to decode (batch size changes require re-capture)

### 13. Tokenizer Workers
- **What**: Off-load tokenization (encode + decode) to a dedicated thread pool separate from the HTTP async runtime and the GPU scheduler thread
- **Why**: Tokenization is CPU-bound and can block the async runtime at high QPS. BPE encoding of long prompts can take 1–5ms
- **Acceptance**: Tokenization latency does not appear in GPU scheduler critical path; supports 50+ concurrent tokenization requests without blocking HTTP handlers
- **Constraint**: Must handle tokenizer thread pool sizing; tokio::spawn_blocking or dedicated rayon pool

### 14. On-Disk KV Cache Persistence
- **What**: Serialize the full KV cache state + token history to SSD after each conversation turn; reload on session resume without re-prefilling
- **Why**: Multi-turn chat re-prefills the entire conversation history on every new message. With M1 Pro SSD at ~5 GB/s, loading an 88 MB KV cache takes ~18ms vs re-prefilling 500 tokens at 14 tok/s = 36s. That's 2000x faster session resume
- **Acceptance**: Saved session resumes in < 100ms regardless of conversation length; KV state is byte-identical to what prefill would have produced; sessions survive server restart
- **Constraint**: File format must include model ID + quant profile to prevent loading incompatible states; must handle graceful invalidation when model weights change

### 15. FP8 KV Cache
- **What**: Store KV cache in 8-bit floating point (E4M3 or E5M2) instead of f16, with on-the-fly quantization during KV write and dequantization during attention read
- **Why**: Halves KV cache memory from 2 bytes/value to 1 byte/value. Doubles effective context capacity: 1024→2048 context at same memory, or fit 8K context where 4K previously maxed out
- **Acceptance**: Context capacity doubled at same memory budget; attention output within 0.5% cosine similarity of f16 KV baseline; no measurable quality degradation in eval scores
- **Constraint**: Requires new Metal kernel for FP8 pack/unpack; must validate quality across all 10 eval categories before shipping

### 16. Fused MLP Kernels
- **What**: Combine gate_proj + up_proj + GeLU activation into a single GPU kernel dispatch, eliminating intermediate buffer writes between the three operations
- **Why**: Currently 3 separate dispatches per layer for MLP front-half (gate matvec, up matvec, gelu_mul). Fusing eliminates 2 intermediate buffer writes per layer × 42 layers = 84 eliminated memory round-trips per token
- **Acceptance**: Decode throughput improves 5–8%; identical output to unfused path (bitwise for Q4, within f16 tolerance for f16 layers)
- **Constraint**: Fused kernel must handle both Q4 and f16 weight variants; threadgroup memory must fit gate+up intermediate values

### 17. Power / Thermal Throttling
- **What**: CLI flag `--power N` (0–100) that inserts configurable sleep between decode iterations to reduce sustained GPU load
- **Why**: Continuous 100% GPU on a MacBook causes thermal throttling, fan noise, and battery drain. A power cap makes the engine usable as a quiet background service
- **Acceptance**: `--power 50` reduces GPU utilization to ~50% with proportionally lower heat/fan noise; power setting adjustable at runtime via API endpoint
- **Constraint**: Must not affect model output (same tokens produced, just slower); sleep granularity should be per-token or per-layer, not coarse

---

## Priority Order (implementation sequence)

| Phase | Items | Milestone |
|-------|-------|-----------|
| **Phase 1** | #1 Parallel Prefill, #6 Sampling, #7 Robustness | Single-user production-ready |
| **Phase 2** | #2 Request Queue, #3 KV Cache Pool, #14 KV Persistence | Multi-user capable + session resume |
| **Phase 3** | #4 Continuous Batching, #5 Chunked Prefill | High-throughput serving |
| **Phase 4** | #8 Observability, #17 Power Throttling | Operations-ready |
| **Phase 5** | #9 FlashAttention, #10 Radix Cache, #15 FP8 KV Cache | Long-context & memory-efficient |
| **Phase 6** | #11 Speculative Decoding, #16 Fused MLP Kernels | Decode throughput 2–3x |
| **Phase 7** | #12 Overlap Scheduling, #13 Tokenizer Workers | Final latency optimizations |

---

## Non-Requirements (explicitly out of scope)
- Multi-GPU / distributed inference
- Model hot-swapping
- LoRA adapter loading
- Vision/audio multimodal input
- Training or fine-tuning
- Authentication / rate limiting (handled by reverse proxy)
- Mixture of Experts routing (Gemma4 E4B is dense; add if MoE model support needed later)
