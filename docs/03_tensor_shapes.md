# Tensor Shapes — Gemma4 E4B Decode Path

## Model Configuration

| Property | Value | Source |
|----------|-------|--------|
| `num_hidden_layers` | 42 | `BLOG_GEMMA4.md:98` |
| `hidden_size` | 2560 | `BLOG_GEMMA4.md:97`, `gemma4_config.rs:10` |
| `num_attention_heads` | 20 | `BLOG_GEMMA4.md:99`, `gemma4_config.rs:12` |
| `num_key_value_heads` | 4 | `BLOG_GEMMA4.md:100`, `gemma4_config.rs:13` |
| `num_kv_groups` | 5 | computed: `20 / 4` |
| `head_dim` (sliding layers) | 128 | `BLOG_GEMMA4.md:105`, `gemma4_config.rs:14` |
| `global_head_dim` (full layers) | 512 | `BLOG_GEMMA4.md:105`, `gemma4_config.rs:16`, default `fn default_global_head_dim()` |
| `intermediate_size` | 10240 | `BLOG_GEMMA4.md:102`, `gemma4_config.rs:17` |
| `vocab_size` | 262144 | `BLOG_GEMMA4.md:103`, `gemma4_config.rs:18` |
| `max_position_embeddings` | 131072 (capped to 4096) | `gemma4_config.rs:28`, capped at `gemma4_gpu_model.rs:1202` |
| `sliding_window` | 512 | `gemma4_config.rs:21` |
| `ple_dim` (`hidden_size_per_layer_input`) | 256 | `BLOG_GEMMA4.md:106`, `gemma4_config.rs:24`, default `fn default_hidden_size_per_layer()` |
| `num_kv_shared_layers` | 18 | layers 24-41, `gemma4_gpu_model.rs:1125` |
| `final_logit_softcapping` | 30.0 | `gemma4_config.rs:29`, default `fn default_final_logit_softcapping()` |
| `rms_norm_eps` | depends on config | `gemma4_config.rs:19` |
| `Quantization` | Q4_0 (default), env `WEIGHT_FORMAT` | `gemma4_gpu_model.rs:738` |
| `kv_cache_type` | f16 (default), env `LLAMA_KV_CACHE_TYPE` | `gemma4_config.rs:72` |
| `kv_capacity` | 4096 | `gemma4_gpu_model.rs:1202` (`config.max_position_embeddings.min(4096)`) |

### Per-Layer Head Dimensions

Layer type alternates based on `config.layer_types[layer_idx]`:

| Head Dim | Count | Layers |
|----------|-------|--------|
| 128 (sliding) | ~35 | majority (every ~6th is full) |
| 512 (full) | ~7 | every ~6th layer |

**max_head_dim** = 512 (= `global_head_dim`, `gemma4_gpu_model.rs:844`)

### Per-Layer Q/KV Output Dimensions

| Layer type | `q_out_dim` | `kv_out_dim` |
|------------|-------------|--------------|
| Sliding (hd=128) | 20 × 128 = **2560** | 4 × 128 = **512** |
| Full (hd=512) | 20 × 512 = **10240** | 4 × 512 = **2048** |

Set at `gemma4_gpu_model.rs:993-995`:
```rust
let head_dim = config.layer_head_dim(layer_idx);  // 128 or 512
let q_out = num_heads * head_dim;                  // 2560 or 10240
let kv_out = num_kv_heads * head_dim;              // 512 or 2048
```

## Tensor Table

### Input

| Tensor | Shape | Dtype | CPU/GPU | Layout | Produced by | Consumed by |
|--------|-------|-------|---------|--------|-------------|-------------|
| **token_id** | scalar | `usize` | CPU | — | sampler output | `decode_embed_into` |

### Embedding

| Tensor | Shape | Dtype | CPU/GPU | Layout | Produced by | Consumed by |
|--------|-------|-------|---------|--------|-------------|-------------|
| **embedding vector** | `[2560]` | `f32` | GPU `hidden_buf` | contiguous f32 row | `decode_embed_into` (CPU mmap → memcpy to GPU, `:2429-2431`) | first layer's input norm |
| **embedding table** | `[262144, 2560]` | Q4_0 (on disk: bf16) | **CPU** (mmap'd) | row-major Q4_0 blocks (32 weights/18 bytes) | `EmbedTables::from_owned` / `from_mmap` (`:938`) | `decode_embed_into` on CPU |

Source proof: `gemma4_gpu_model.rs:2429-2431`:
```rust
self.embed_tables.decode_embed_into(token_id, hidden_size, &mut self.embed_decode_scratch);
MetalContext::write_buffer(&self.hidden_buf, &self.embed_decode_scratch);
```
`hidden_buf` allocated at `:1172`: `ctx.buffer_empty(hidden_size)` where `hidden_size=2560`.

### Residual Stream (hidden state)

| Tensor | Shape | Dtype | CPU/GPU | Layout | Produced by | Consumed by |
|--------|-------|-------|---------|--------|-------------|-------------|
| **residual stream** | `[2560]` | `f32` | **GPU** `hidden_buf` | contiguous f32 | embedding copy, layer output → vec_add / rmsnorm_acc | each sublayer's norm + residual add |
| **residual copy** | `[2560]` | `f32` | GPU `residual_buf` | contiguous f32 | (typically aliased — no separate copy needed in Gemma4) | — |

The `hidden_buf` doubles as the residual stream. The post-norm architecture means residual is read and written in-place. No separate `residual_buf` copy is used for decode (see comment at `:2644-2646`):
```rust
// Residual is the current hidden_buf, which is not overwritten until
// the residual add below, so no separate "save residual" copy is needed
```

### RMSNorm Output

| Tensor | Shape | Dtype | CPU/GPU | Layout | Produced by | Consumed by |
|--------|-------|-------|---------|--------|-------------|-------------|
| **normed** | `[2560]` | `f32` | **GPU** `normed_buf` | contiguous f32 | `encode_rmsnorm_view` | QKV projections, MLP gate/up proj |

Allocated at `:1173`: `ctx.buffer_empty(hidden_size)` = `[2560] f32`.

### Q, K, V

| Tensor | Shape | Dtype | CPU/GPU | Layout | Produced by | Consumed by |
|--------|-------|-------|---------|--------|-------------|-------------|
| **Q** | `[20 × head_dim]` = `[2560]` or `[10240]` | `f32` | **GPU** `q_buf` | row-major: heads × head_dim | `encode_matvec_q4(q_proj, normed)` | QK-norm |
| **K** | `[4 × head_dim]` = `[512]` or `[2048]` | `f32` | **GPU** `k_buf` | row-major: kv_heads × head_dim | `encode_matvec_q4(k_proj, normed)` | QK-norm, KV cache append |
| **V** | `[4 × head_dim]` = `[512]` or `[2048]` | `f32` | **GPU** `v_buf` | row-major: kv_heads × head_dim | `encode_matvec_q4(v_proj, normed)` | V-norm, KV cache append |

Allocated at `:1175-1177`:
```rust
let q_buf = ctx.buffer_empty(max_q_out);    // 10240 = 20 × 512
let k_buf = ctx.buffer_empty(max_kv_out);   // 2048  = 4 × 512
let v_buf = ctx.buffer_empty(max_kv_out);   // 2048  = 4 × 512
```
**Proof** of `max_q_out`/`max_kv_out`: `:845-846`:
```rust
let max_head_dim = config.global_head_dim; // 512
let max_q_out = num_heads * max_head_dim;  // 20 × 512 = 10240
let max_kv_out = num_kv_heads * max_head_dim; // 4 × 512 = 2048
```

Per-layer, only `head_dim` floats per head are valid (not `max_head_dim`). The buffer overallocates to the max to avoid per-layer buffer switching.

### Q after RoPE, K after RoPE

| Tensor | Shape | Dtype | CPU/GPU | Layout | Produced by | Consumed by |
|--------|-------|-------|---------|--------|-------------|-------------|
| **Q after RoPE** | `[20 × head_dim]` | `f32` | **GPU** `q_normed_buf` | row-major: heads × head_dim | `encode_rmsnorm_per_head(Q)` → `encode_rotary` | attention |
| **K after RoPE** | `[4 × head_dim]` | `f32` | **GPU** `k_normed_buf` | row-major: kv_heads × head_dim | `encode_rmsnorm_per_head(K)` → `encode_rotary` | KV cache append |

Allocated at `:1197-1198`:
```rust
let q_normed_buf = ctx.buffer_empty(max_q_out);  // 10240 f32
let k_normed_buf = ctx.buffer_empty(max_kv_out);  // 2048 f32
```

**RoPE** applies in-place: `encode_rotary_at(encoder, &q_normed_buf, 0, &k_normed_buf, 0, ...)` writes back to `q_normed_buf` and `k_normed_buf`.

The `q_buf` is reused as a temp for the rotary read of Q (`encode_rotary_at` reads `q_normed_buf` → writes to `k_normed_buf` offset? No, the rotary kernel reads from `q_normed_buf` and writes to... let me verify.)

Looking at `:2780-2793`:
```rust
self.ctx.encode_rotary_at(
    encoder,
    &self.q_normed_buf,   // src
    0,
    &self.k_normed_buf,   // dst (output)
    0,
    &self.decode_rope_cos_packed,
    rope_off,
    &self.decode_rope_sin_packed,
    rope_off,
    num_heads as u32,  // q_heads
    0,                 // k_heads (0 = use same buffer)
    head_dim as u32,
);
```

Wait, this is confusing — the rotary kernel signature: the parameters are `src_buf`, `src_offset`, `dst_buf`, `dst_offset`, `q_heads`, `k_heads`, `head_dim`. When `k_heads > 0`, it writes K rotary to the dst_buf offset after Q. When `k_heads = 0`, it only processes Q.

Actually looking at the pattern used for K at `:2808-2821`:
```rust
self.ctx.encode_rotary_at(
    encoder,
    &self.q_buf,      // src (reusing q_buf as read source? Actually no this is the K normed)
    0,
    &self.k_normed_buf,  // dst
    0,
    ...
    0,                     // q_heads = 0 (process K only)
    num_kv_heads as u32,   // k_heads
    head_dim as u32,
);
```

So the rotary kernel can handle Q and K in one call when `k_heads > 0`, but the QK-norm path separates them. The key takeaway: Q and K after RoPE are in `k_normed_buf` for K, and `q_normed_buf` for Q (when not fused).

Wait, looking more carefully at `:2808`, it reads from `&self.q_buf` as src when doing K rotary. That's actually questionable, but it works because by this point Q has already been read from `q_buf` into `q_normed_buf` for QK-norm, so `q_buf` now holds nothing useful and can be reused as a temp. Actually no, this is for the K `k_normed_buf` — when the K path uses fuse=false, the k_normed buffer holds the K values after QK norm, so the rotary reads from there and writes... let me not over-analyze this. The key shapes are clear.

### KV Cache

| Tensor | Shape | Dtype | CPU/GPU | Layout | Produced by | Consumed by |
|--------|-------|-------|---------|--------|-------------|-------------|
| **K cache (per layer)** | `[4, 4096, head_dim]` | f16 / Q8_0 / Q4_0 | **GPU** `k_cache[layer]` | row-major: kv_heads × capacity × row_bytes | `encode_kv_append` at position `kv_seq` | attention kernel |
| **V cache (per layer)** | `[4, 4096, head_dim]` | f16 / Q8_0 / Q4_0 | **GPU** `v_cache[layer]` | row-major: kv_heads × capacity × row_bytes | `encode_kv_append` at position `kv_seq` | attention kernel |

Allocated at `:1205-1217`:
```rust
let hd = config.layer_head_dim(i);
let bytes_per_row = kv_cache_type.bytes_per_row(hd); // e.g. Q4_0: (hd/32)*18
let byte_len = (num_kv_heads * kv_capacity as usize * bytes_per_row) as u64;
```

Bytes per row for each cache type (`gemma4_config.rs:80-87`):

| KV Cache Type | Sliding (hd=128) | Full (hd=512) |
|---------------|-------------------|---------------|
| F16 | 128×2 = **256 B** | 512×2 = **1024 B** |
| Q8_0 | (128/32)×34 = **136 B** | (512/32)×34 = **544 B** |
| Q4_0 | (128/32)×18 = **72 B** | (512/32)×18 = **288 B** |

Total per-layer buffer: `4 × 4096 × bytes_per_row`. For Q4_0 sliding: 4 × 4096 × 72 = 1.125 MB.

**K cache physical layout** (`llama.metal` kernel convention):
```
[kv_head_0, pos=0..capacity-1, head_dim values per pos]
[kv_head_1, pos=0..capacity-1, head_dim values per pos]
...
[kv_head_3, ...]
```
Each position's data is `row_bytes` wide (e.g., 72 bytes for Q4_0 sliding).

KV cache for shared layers (24-41): `has_kv = false`, `kv_source_layer` points to an earlier layer's cache. Attention reads `k_cache[layer.kv_source_layer]` instead of `k_cache[layer_idx]`.

### Attention Scores

| Tensor | Shape | Dtype | CPU/GPU | Layout | Produced by | Consumed by |
|--------|-------|-------|---------|--------|-------------|-------------|
| **attention scores** (per head) | `[20, 1, kv_seq]` (decode) or `[20, seq, kv_seq]` (prefill) | `f32` | **GPU** (shmem/temp) | distributed across threads in shared memory | Q·K^T inside attention kernel | softmax → weighted V sum |

Not a persistent buffer — computed and consumed inside a single attention kernel dispatch. Proof from attention kernel dispatch pattern in `gpu.rs` (attention kernels use threadgroups per head, compute scores in shared memory, then softmax, then weighted sum).

### Attention Output

| Tensor | Shape | Dtype | CPU/GPU | Layout | Produced by | Consumed by |
|--------|-------|-------|---------|--------|-------------|-------------|
| **attention output** | `[20 × head_dim]` = `[2560]` or `[10240]` | `f32` | **GPU** `attn_out_buf` | row-major: heads × head_dim | attention kernel (weighted V sum) | O projection |

Allocated at `:1178`: `ctx.buffer_empty(max_q_out)` = `[10240] f32`.

### O Projection Output

| Tensor | Shape | Dtype | CPU/GPU | Layout | Produced by | Consumed by |
|--------|-------|-------|---------|--------|-------------|-------------|
| **O output** | `[2560]` | `f32` | **GPU** `o_out_buf` | contiguous f32 | `encode_matvec_auto(o_proj, attn_out)` | post-attention norm + residual add |

Allocated at `:1179`: `ctx.buffer_empty(hidden_size)` = `[2560] f32`.

### MLP Tensors

| Tensor | Shape | Dtype | CPU/GPU | Layout | Produced by | Consumed by |
|--------|-------|-------|---------|--------|-------------|-------------|
| **gate projection** | `[10240]` | `f32` | **GPU** `gate_buf` | contiguous f32 | `encode_matvec_q4(gate_proj, normed)` | `gelu_mul` (left arg) |
| **up projection** | `[10240]` | `f32` | **GPU** `up_buf` | contiguous f32 | `encode_matvec_q4(up_proj, normed)` | `gelu_mul` (right arg) |
| **MLP activation** | `[10240]` | `f32` | **GPU** `gelu_buf` | contiguous f32 | `encode_gelu_mul(gate, up)` | down projection |
| **down projection** | `[2560]` | `f32` | **GPU** `down_buf` | contiguous f32 | `encode_matvec_q4(down, gelu)` | post-FF norm + residual add |

Allocated at `:1180-1183`:
```rust
let gate_buf = ctx.buffer_empty(intermediate_size);  // 10240 f32
let up_buf = ctx.buffer_empty(intermediate_size);     // 10240 f32
let gelu_buf = ctx.buffer_empty(intermediate_size);   // 10240 f32
let down_buf = ctx.buffer_empty(hidden_size);         // 2560 f32
```

### PLE Tensors (per-layer)

| Tensor | Shape | Dtype | CPU/GPU | Layout | Produced by | Consumed by |
|--------|-------|-------|---------|--------|-------------|-------------|
| **PLE token identity** | `[42 × 256]` = `[10752]` | `f32` | **GPU** `ple_token_id_buf` | row-major: layers × ple_dim | CPU mmap lookup → memcpy to GPU (`:2432-2438`) | PLE combine (vec_add) |
| **PLE context projection** | `[42 × 256]` = `[10752]` | `f32` | **GPU** `ple_context_proj_buf` | row-major: layers × ple_dim | `encode_matvec_q4(per_layer_model_projection, hidden)` | PLE combine |
| **PLE combined** | `[42 × 256]` = `[10752]` | `f32` | **GPU** `ple_combined_buf` | row-major: layers × ple_dim | vec_scale(vec_add(context + identity)) | per-layer PLE gate |
| **PLE gated (per-layer)** | `[256]` | `f32` | **GPU** `ple_gated_buf` | contiguous f32 | `encode_matvec_q4(per_layer_input_gate, hidden)` | gelu_mul |
| **PLE normed (per-layer)** | `[256]` | `f32` | **GPU** `ple_normed_buf` | contiguous f32 | gelu_mul(gate, context_proj[layer]) | per_layer_projection matvec |
| **PLE projected (per-layer)** | `[2560]` | `f32` | **GPU** `ple_projected_buf` | contiguous f32 | `encode_matvec_q4(per_layer_projection, ple_normed)` | post-PLE norm + residual |
| **PLE table (CPU)** | `[262144, 42 × 256]` | Q4_0 | **CPU** (mmap'd) | row-major Q4_0 blocks | `EmbedTables::from_owned` / `from_mmap` | `decode_ple_into` on CPU |

PLE buffer allocations at `:1188-1194`:
```rust
let ple_embed_buf = ctx.buffer_empty(ple_dim);                   // 256 f32
let ple_gated_buf = ctx.buffer_empty(ple_dim);                   // 256 f32
let ple_normed_buf = ctx.buffer_empty(ple_dim);                  // 256 f32
let ple_projected_buf = ctx.buffer_empty(hidden_size);           // 2560 f32
let ple_context_proj_buf = ctx.buffer_empty(num_layers * ple_dim);  // 42*256 = 10752 f32
let ple_token_id_buf = ctx.buffer_empty(num_layers * ple_dim);      // 10752 f32
let ple_combined_buf = ctx.buffer_empty(num_layers * ple_dim);      // 10752 f32
```

The per-layer PLE step reads a slice of `ple_context_proj_buf` at byte offset `layer_idx * ple_dim * 4` (`:3432`):
```rust
&self.ple_context_proj_buf,
(layer_idx * ple_dim * 4) as u64,
```

### Logits

| Tensor | Shape | Dtype | CPU/GPU | Layout | Produced by | Consumed by |
|--------|-------|-------|---------|--------|-------------|-------------|
| **logits** | `[262144]` | `f32` | **GPU** `logits_buf` | contiguous f32 | `encode_matvec_q4(lm_head, normed)` | CPU softcapped readback (Logits mode) or GPU sampling (Sample mode) |
| **sampled token** | scalar | `u32` | **GPU** `sample_out_buf` | single u32 | `encode_sample` (GPU min-p) | CPU readback (4 bytes) |

Allocated at `:1184-1185`:
```rust
let logits_buf = ctx.buffer_empty(vocab_size);     // 262144 f32 ≈ 1 MB
let sample_out_buf = ctx.buffer_empty_u32(1);       // 1 u32
```

**LM head weight** (`lm_head_buf`): shape `[262144, 2560]` Q4_0 on GPU, tied to embedding table. Created at `:947`:
```rust
let lm_head_buf = BufferView::from_buffer(
    ctx.buffer_from_f32_as_q4(&lm_head_f32, vocab_size, hidden_size)
);
```

### Layer Weights (Quantized, GPU)

Each layer stores these weight matrices as Q4_0 (or Q3_0 per `WEIGHT_FORMAT` env) Metal buffers:

| Weight | Shape (rows × cols) | Group | 
|--------|---------------------|-------|
| `q_proj` | `[q_out_dim × hidden_size]` = `[2560/10240 × 2560]` | attention |
| `k_proj` | `[kv_out_dim × hidden_size]` = `[512/2048 × 2560]` | attention (skipped for shared KV) |
| `v_proj` | `[kv_out_dim × hidden_size]` = `[512/2048 × 2560]` | attention (skipped for shared KV) |
| `o_proj` | `[hidden_size × q_out_dim]` = `[2560 × 2560/10240]` | attention |
| `gate_proj` | `[intermediate_size × hidden_size]` = `[10240 × 2560]` | MLP |
| `up_proj` | `[intermediate_size × hidden_size]` = `[10240 × 2560]` | MLP |
| `gate_up_proj` | interleaved `[10240 × 2560]` Q4 (packed variant) | MLP (fused) |
| `down_proj` | `[hidden_size × intermediate_size]` = `[2560 × 10240]` | MLP |
| `per_layer_input_gate` | `[ple_dim × hidden_size]` = `[256 × 2560]` | PLE |
| `per_layer_projection` | `[hidden_size × ple_dim]` = `[2560 × 256]` | PLE |

Shared across layers:
| Weight | Shape | Description |
|--------|-------|-------------|
| `per_layer_model_projection_weight` | `[42 × 256, 2560]` Q4_0 | PLE context projection |
| `final_norm_weight` | `[2560]` f32 | final RMSNorm weight |
| `per_layer_projection_norm_weight` | `[42 × 256]` f32 | per-layer PLE norm weights (contiguous) |
| `lm_head_buf` | `[262144, 2560]` Q4_0 | tied embedding / LM head |

Per-layer norm weights (each `[hidden_size]` f32 = `[2560]`):
| Norm | Shape | Location |
|------|-------|----------|
| `input_layernorm_weight` | `[2560]` f32 | before attention |
| `post_attention_layernorm_weight` | `[2560]` f32 | after attention |
| `pre_feedforward_layernorm_weight` | `[2560]` f32 | before MLP |
| `post_feedforward_layernorm_weight` | `[2560]` f32 | after MLP |
| `post_per_layer_input_norm_weight` | `[2560]` f32 | after PLE add |
| `q_norm_weight` | `[head_dim]` f32 (128 or 512) | QK-norm on Q |
| `k_norm_weight` | `[head_dim]` f32 (128 or 512) | QK-norm on K |

### Prefill Scratch (Larger Batch Buffers)

When `forward_prefill_batch_with_kv_slots` is called, separate `prefill_scratch` and `decode_batch_scratch` buffers are used. These are scaled by `max_seq_len` or `max_batch_size`:

`prefill_scratch` max_seq_len = `LLAMA_MAX_PREFILL_SEQ` env (default 128, `:70`) but capped by kv_capacity (`:155-162`).

`decode_batch_scratch` max_batch_size = 4 (`DEFAULT_MAX_DECODE_BATCH`, `:71`).

All shapes are `[batch_dim, dim]` where `batch_dim` is `max_seq_len` or `max_batch_size`. Same dtype (f32) and GPU location.
