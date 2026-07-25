# Decode Path: One Token → Next Token

## Entry Points

Two paths lead into single-token decode:

**1. CLI interactive** (`main.rs:305`):
```
generate_gemma4_gpu()
  └─ model.forward_single_token_sample(token_id, temperature, min_p, seed)
       └─ forward_single_token_inner(token_id, DecodeMode::Sample(...))
```

**2. Server / scheduler** (`scheduler.rs:207`):
```
batch_engine.decode_batch(&inputs)              ← batch_engine.rs:109
  └─ model.forward_decode_batch_with_kv_slots()  ← gemma4_gpu_model.rs:6584
       └─ (batch size 1 fallback:)
          model.forward_single_token_with_kv_slot(token_id, kv_pool, slot)  ← :6551
            └─ swaps kv_pool cache ↔ self.k_cache/v_cache
            └─ self.forward_single_token(token_id)     ← :2357
                 └─ forward_single_token_inner(token_id, DecodeMode::Logits)
```

## `forward_single_token_inner` — Full Call Graph

> File: `gemma4_gpu_model.rs:2394`

```
forward_single_token_inner(token_id, mode)
│
├── [CPU] Embedding lookup ────────────────────────────────────────── CPU
│   embed_tables.decode_embed_into(token_id, hidden_size, &mut embed_decode_scratch)
│   MetalContext::write_buffer(&hidden_buf, &embed_decode_scratch)     ← memcpy to GPU
│   embed_tables.decode_ple_into(token_id, ple_total_dim, ple_dim, &mut ple_decode_scratch)
│   MetalContext::write_buffer(&ple_token_id_buf, &ple_decode_scratch) ← memcpy to GPU
│
├── [GPU] Create Metal command buffer ───────────────────────────────
│   cmd = ctx.queue.new_command_buffer()
│   encoder = cmd.new_compute_command_encoder()
│
├── [GPU] RoPE table fill ───────────────────────────────────────────
│   encode_rope_fill_decode(encoder, cos_packed, sin_packed, params,
│                           num_layers, max_head_dim, pos)
│   Kernel: rope_fill_decode        ← llama.metal
│
├── [GPU] PLE pre-pass (optional) ───────────────────────────────────
│   encode_matvec_q4_view(encoder, per_layer_model_projection, hidden, ple_context_proj,
│                         ple_total_dim, hidden_size)
│     Kernel: matvec_q4_*           ← llama.metal
│   encode_vec_scale(encoder, ple_context_proj, ple_combined, ple_total_dim, 1/√hidden)
│     Kernel: vec_scale             ← llama.metal
│   encode_rmsnorm_per_head_view(encoder, ple_combined, proj_norm, ple_context_proj,
│                                num_layers, ple_dim, eps)
│     Kernel: rmsnorm_per_head      ← llama.metal
│   encode_vec_add(encoder, ple_context_proj, ple_token_id, ple_combined, ple_total_dim)
│     Kernel: vec_add               ← llama.metal
│   encode_vec_scale(encoder, ple_combined, ple_context_proj, ple_total_dim, 1/√2)
│
├── [GPU] For each of 42 layers ─────────────────────────────────────
│   for layer_idx in 0..actual_num_layers:
│   │
│   ├─ Attention Block ──────────────────────────────────────────────
│   │  Choice A: Fused QKV (Q4 + FUSED_QKV=1) ──────────────────
│   │  │ encode_rmsnorm_qkv_q4_view(encoder, hidden, input_layernorm,
│   │  │   inv_rms, q_proj, k_proj, v_proj, q_buf, k_buf, v_buf,
│   │  │   q_out, kv_out, hidden_size, eps)
│   │  │   Kernel: rmsnorm_qkv_q4_f16   ← llama.metal (fused RMSNorm + 3× matvec)
│   │  │
│   │  Choice B: Unfused ─────────────────────────────────────────
│   │    encode_rmsnorm_view(encoder, hidden, input_ln, normed, hidden, eps)
│   │      Kernel: rmsnorm              ← llama.metal
│   │    encode_matvec_q4_view(encoder, q_proj, normed, q_buf, q_out, hidden)
│   │      Kernel: matvec_q4_*          ← llama.metal
│   │    encode_matvec_q4_view(encoder, k_proj, normed, k_buf, kv_out, hidden)  [if has_kv]
│   │    encode_matvec_q4_view(encoder, v_proj, normed, v_buf, kv_out, hidden)  [if has_kv]
│   │
│   │  QK Norm ───────────────────────────────────────────────────
│   │  (skipped if use_fused_q_attn, which fuses into attention kernel)
│   │    encode_rmsnorm_per_head_view(encoder, q_buf, q_norm, q_normed, num_heads, head_dim, eps)
│   │    encode_rmsnorm_per_head_view(encoder, k_buf, k_norm, k_normed, num_kv_heads, head_dim, eps) [if has_kv]
│   │
│   │  RoPE ──────────────────────────────────────────────────────
│   │  (skipped if use_fused_q_attn)
│   │    encode_rotary_at(encoder, q_normed, 0, k_normed, 0,
│   │      cos_packed, rope_off, sin_packed, rope_off,
│   │      num_heads, 0, head_dim)
│   │      Kernel: rotary              ← llama.metal
│   │    encode_rotary_at(encoder, ..., 0, num_kv_heads, head_dim)  [if has_kv]
│   │
│   │  V Norm ──────────────────────────────────────────────────────
│   │  (skipped if use_fused_k_attn)
│   │    encode_rmsnorm_per_head_noweight(encoder, v_buf, gate_buf, num_kv_heads, head_dim, eps)
│   │
│   │  KV Cache Append ─────────────────────────────────────────────
│   │  (skipped if use_fused_k_attn with Q4_0)
│   │    encode_kv_append_{f16|q8_0|q4_0}(encoder, k_normed, k_cache[layer_idx], ...)
│   │      Kernel: kv_append_{f16|q8_0|q4_0}  ← llama.metal
│   │    encode_kv_append_{f16|q8_0|q4_0}(encoder, v_normed, v_cache[layer_idx], ...)
│   │
│   │  Attention ───────────────────────────────────────────────────
│   │  Many variants selected by kv_cache_type + fusion flags:
│   │    Options for Q4_0 cache:
│   │      attention_full_fused_q4_0       ← fused QK-norm + RoPE + attn + append
│   │      attention_fused_qknorm_rope_q4_0  ← fused QK-norm + RoPE + attn
│   │      attention_qknorm_rope_q4_0        ← fused QK-norm + RoPE + attn (no KV)
│   │      attention_ggml_q4_0             ← llama.cpp flash attn port
│   │      attention_fused_q4_0             ← fused KV norm + attn
│   │      attention_with_offset_q4_0_gqa  ← GQA-aware q4 attn
│   │      attention_with_offset_q4_0       ← fallback Q4 attn
│   │    Options for F16 cache:
│   │      attention_with_offset_f16_gqa
│   │      attention_with_offset_f16
│   │    Options for Q8_0 cache:
│   │      attention_with_offset_q8_0
│   │
│   │  O Projection ────────────────────────────────────────────────
│   │    encode_matvec_auto_view(encoder, o_proj, attn_out, o_out, hidden, q_out)
│   │
│   │  Post-Attention Norm + Residual (3→1) ────────────────────────
│   │    encode_proj_norm_residual(ctx, encoder, hidden, o_out, post_attn_ln, hidden, hidden, eps)
│   │      ├─ encode_rmsnorm_acc_view()    [fused_rmsnorm_acc] → Kernel: rmsnorm_acc
│   │      └─ Unfused: rmsnorm → vec_add
│   │
│   ├─ MLP Block ────────────────────────────────────────────────────
│   │  Many fusion variants by weight_format + env flags.
│   │  Typical (Q4_0, no fusion):
│   │    encode_rmsnorm_view(encoder, hidden, pre_ff_ln, normed, hidden, eps)
│   │    encode_matvec_q4_dual_view(encoder, gate_proj, up_proj, normed, gate, up, inter, hidden)
│   │      Kernel: matvec_q4_dual         ← llama.metal (one dispatch, two matmuls)
│   │    encode_gelu_mul(encoder, gate, up, gelu, inter)
│   │      Kernel: gelu_mul              ← llama.metal
│   │    encode_matvec_auto_view(encoder, down_proj, gelu, down, hidden, inter)
│   │  Fused variant (when enabled):
│   │    encode_rmsnorm_mlp_fused_q4_gelu_down_packed_from_hidden_at_view()
│   │      Kernel: rmsnorm_mlp_gelu_down_q4_packed  ← llama.metal
│   │
│   │  Post-FF Norm + Residual ──────────────────────────────────────
│   │    [same pattern: rmsnorm_acc or rmsnorm → vec_add]
│   │
│   ├─ PLE per-layer ────────────────────────────────────────────────
│   │  encode_matvec_auto_view(encoder, per_layer_input_gate, hidden, ple_gated, ple_dim, hidden)
│   │  encode_gelu_mul_at(encoder, ple_gated, ple_context_proj[layer], ple_normed, ple_dim)
│   │  encode_matvec_auto_view(encoder, per_layer_projection, ple_normed, ple_projected, hidden, ple_dim)
│   │  encode_proj_norm_residual(ctx, encoder, hidden, ple_projected, post_ple_norm, hidden, hidden, eps)
│   │
│   ├─ Layer scalar ─────────────────────────────────────────────────
│   │  encode_vec_scale(encoder, hidden, hidden, hidden_size, layer_scalar)
│   │    Kernel: vec_scale              ← llama.metal
│   │
│   └─ [end for]
│
├── [GPU] Final Norm + LM Head ────────────────────────────────────
│   (skipped in DecodeMode::Advance)
│   encode_rmsnorm_view(encoder, hidden, final_norm, normed, hidden, eps)
│   encode_matvec_q4_view(encoder, lm_head_buf, normed, logits_buf, vocab_size, hidden)
│     Kernel: matvec_q4_lm_head         ← llama.metal
│
├── [GPU] Sampling (Sample mode only) ──────────────────────────────
│   encode_sample(encoder, logits_buf, sample_out_buf, vocab_size, cap, temp, min_p, seed)
│     Kernel: sample_min_p              ← llama.metal (GPU-side min-p sampling)
│
├── Commit + Wait ──────────────────────────────────────────────────
│   encoder.end_encoding()
│   cmd.commit()
│   cmd.wait_until_completed()           ← blocks CPU until GPU finishes
│
├── [CPU] Readback ────────────────────────────────────────────────
│   Sample mode: MetalContext::read_u32(&sample_out_buf)  → 4 bytes (next token id)
│   Logits mode: MetalContext::read_buffer(&logits_buf, vocab_size) → entire vocab
│                + CPU-side logit softcapping: cap * tanh(x/cap)
│   Advance mode: nothing read back
│
└── Update state
    self.total_tokens += 1
    self.kv_seq_len += 1
```

## CPU vs GPU Boundary Summary

| Step | Where | Data moved |
|------|-------|------------|
| Embedding lookup | **CPU** (mmap'd bf16/Q4 table) | nothing to GPU yet |
| Write embed → hidden_buf | **CPU→GPU** (memcpy) | hidden_size × f32 bytes |
| Write PLE → ple_token_id_buf | **CPU→GPU** (memcpy) | num_layers × ple_dim × f32 bytes |
| All other forward pass ops | **GPU** (Metal shaders) | — |
| Sample token readback | **GPU→CPU** (4 bytes) | 1 × u32 |
| Logits readback | **GPU→CPU** (vocab_size × f32 bytes) | 500+ KB (only in Logits mode) |
| CPU sampling (scheduler path) | **CPU** | on already-read-back logits |

Embedding tables live on CPU. The lookup is a CPU-side index+dequant then memcpy to a pre-allocated GPU buffer. The lm_head is on GPU (Q4_0 quantized). PLE token identity is on CPU, PLE context projection is on GPU.

## Metal Kernels Launched Per Token (non-fused, typical config)

Rough count assuming no fusion and Q4_0 KV cache:

| Operation | Count | Kernel |
|-----------|-------|--------|
| rope_fill_decode | 1 | RoPE table fill |
| matvec_q4 (PLE ctx) | 1 | PLE context projection |
| vec_scale (PLE) | 2 | PLE scaling |
| rmsnorm_per_head (PLE) | 1 | PLE norm |
| vec_add (PLE) | 1 | PLE combine |
| **Per layer** (×42) | | |
| rmsnorm (input) | 42 | Pre-attention norm |
| matvec_q4 (Q proj) | 42 | Q projection |
| matvec_q4 (K proj) | 42* | K projection |
| matvec_q4 (V proj) | 42* | V projection |
| rmsnorm_per_head (Q) | 42 | QK norm Q |
| rmsnorm_per_head (K) | 42* | QK norm K |
| rotary (Q) | 42 | RoPE Q |
| rotary (K) | 42* | RoPE K |
| rmsnorm_per_head (V) | 42* | V norm |
| kv_append_q4_0 (K) | 42* | KV cache K |
| kv_append_q4_0 (V) | 42* | KV cache V |
| attention_q4_0 | 42 | Attention |
| matvec_q4 (O proj) | 42 | O projection |
| rmsnorm (post-attn) | 42 | Post-attn norm |
| vec_add (residual) | 42 | Residual add |
| rmsnorm (pre-FF) | 42 | Pre-FF norm |
| matvec_q4 (gate) | 42 | Gate projection |
| matvec_q4 (up) | 42 | Up projection |
| gelu_mul | 42 | SiLU/GeLU activation |
| matvec_q4 (down) | 42 | Down projection |
| rmsnorm (post-FF) | 42 | Post-FF norm |
| vec_add (residual) | 42 | Residual add |
| matvec_q4 (PLE gate) | 42 | PLE gate |
| gelu_mul (PLE) | 42 | PLE activation |
| matvec_q4 (PLE proj) | 42 | PLE project back |
| rmsnorm (post-PLE) | 42 | Post-PLE norm |
| vec_add (PLE residual) | 42 | PLE residual add |
| vec_scale (layer scalar) | 42 | Layer scalar mul |
| rmsnorm (final) | 1 | Final norm |
| matvec_q4 (lm_head) | 1 | LM head |
| sample_min_p | 1* | GPU sampling (Sample mode) |

> **~97 kernels without fusion, ~113 when layers have KV.** With fusion, many of these combine (e.g., rmsnorm_qkv_q4 does 4 ops in 1 kernel; attention_full_fused_q4_0 does QK-norm + RoPE + attention + KV append in 1 kernel).

> *For shared KV layers (layers 24-41, `has_kv=false`), K/V projections, QK/K/V norm, RoPE on K/V, and KV cache append are **skipped** — they reuse the KV source layer's cache.

## Unknown / Unclear Areas

1. **Mega kernel path** (`mega_kernel_enabled`): When `MEGA_KERNEL=1`, the entire per-token forward pass is encoded as a single Metal dispatch via `MegaDecodeGraph::encode()`. The op graph is built once at init time and replayed per token. The per-layer logic lives in `mega_decode.rs` — need to verify it mirrors the standard path exactly.

2. **Batch decode path** (`forward_decode_batch_encoded_with_kv_slots`): Current implementation falls back to single-token decode for batch size 1. The multi-request batch path encodes a different set of kernels (batch-aware projections, batch RoPE precomputed on CPU, batch attention). This is a separate code path not traced here.

3. **Fused kernel selection logic**: The many `if-else` chains (fused_qkv, fused_rmsnorm_acc, fused_mlp_gelu_down, use_fused_q_attn, use_fused_k_attn) create a combinatorial explosion of possible kernel sets per layer. The doc above shows the most common path.

4. **GPU sampling kernel** (`encode_sample` → `sample_min_p`): Does it apply repetition/frequency penalty? The CPU sampler does (`sampling.rs:33-34`), but the GPU path appears to be pure min-p with temperature. Need to verify.

5. **KV cache sharing** for layers 24-41: The code sets `kv_source_layer` for shared layers. The attention kernel reads from `k_cache[layer.kv_source_layer]` and `v_cache[layer.kv_source_layer]`. These layers still project Q (which is needed for the attention query) but skip K/V projection and cache append. The code correctly handles this via `layer.has_kv` checks.
