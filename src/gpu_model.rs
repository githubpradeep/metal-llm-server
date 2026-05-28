use metal::*;
use crate::config::LlamaConfig;
use crate::gpu::MetalContext;
use crate::weights::ModelWeights;

/// GPU-resident model with persistent KV cache on GPU.
/// All operations for one token are encoded into a SINGLE command buffer.
pub struct GpuLlamaModel {
    pub ctx: MetalContext,
    pub config: LlamaConfig,

    pub embed_tokens: Vec<f32>,
    pub layers: Vec<GpuDecoderLayer>,
    pub final_norm_weight: Buffer,
    pub lm_head_weight: Buffer,

    // Pre-allocated scratch buffers (reused every token)
    pub hidden_buf: Buffer,
    pub normed_buf: Buffer,
    pub residual_buf: Buffer,
    pub q_buf: Buffer,
    pub k_buf: Buffer,
    pub v_buf: Buffer,
    pub attn_out_buf: Buffer,
    pub o_out_buf: Buffer,
    pub gate_buf: Buffer,
    pub up_buf: Buffer,
    pub silu_buf: Buffer,
    pub down_buf: Buffer,
    pub logits_buf: Buffer,

    // GPU-resident KV cache per layer
    pub k_cache: Vec<Buffer>,
    pub v_cache: Vec<Buffer>,
    pub kv_seq_lens: Vec<u32>,
    pub kv_capacity: u32,

    // Rotary precomputed
    pub inv_freq: Vec<f32>,
    pub cos_buf: Buffer,
    pub sin_buf: Buffer,

    pub total_tokens: usize,
}

pub struct GpuDecoderLayer {
    pub q_proj: Buffer,
    pub k_proj: Buffer,
    pub v_proj: Buffer,
    pub o_proj: Buffer,
    pub gate_proj: Buffer,
    pub up_proj: Buffer,
    pub down_proj: Buffer,
    pub input_ln_weight: Buffer,
    pub post_ln_weight: Buffer,
}

impl GpuLlamaModel {
    pub fn new(config: &LlamaConfig, weights: &ModelWeights) -> Self {
        let ctx = MetalContext::new();
        let hidden_size = config.hidden_size;
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;

        let embed_tokens = weights.get_1d_raw("model.embed_tokens.weight");

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            println!("  Loading GPU layer {}/{} (Q4_0 quantized)", i + 1, config.num_hidden_layers);
            let hidden = config.hidden_size;
            let q_out = config.num_attention_heads * head_dim;
            let kv_out = config.num_key_value_heads * head_dim;
            let inter = config.intermediate_size;

            layers.push(GpuDecoderLayer {
                q_proj: ctx.buffer_from_f32_as_q4(&weights.get_1d_raw(&format!("model.layers.{}.self_attn.q_proj.weight", i)), q_out, hidden),
                k_proj: ctx.buffer_from_f32_as_q4(&weights.get_1d_raw(&format!("model.layers.{}.self_attn.k_proj.weight", i)), kv_out, hidden),
                v_proj: ctx.buffer_from_f32_as_q4(&weights.get_1d_raw(&format!("model.layers.{}.self_attn.v_proj.weight", i)), kv_out, hidden),
                o_proj: ctx.buffer_from_f32_as_q4(&weights.get_1d_raw(&format!("model.layers.{}.self_attn.o_proj.weight", i)), hidden, q_out),
                gate_proj: ctx.buffer_from_f32_as_q4(&weights.get_1d_raw(&format!("model.layers.{}.mlp.gate_proj.weight", i)), inter, hidden),
                up_proj: ctx.buffer_from_f32_as_q4(&weights.get_1d_raw(&format!("model.layers.{}.mlp.up_proj.weight", i)), inter, hidden),
                down_proj: ctx.buffer_from_f32_as_q4(&weights.get_1d_raw(&format!("model.layers.{}.mlp.down_proj.weight", i)), hidden, inter),
                input_ln_weight: ctx.buffer_from_slice(&weights.get_1d(&format!("model.layers.{}.input_layernorm.weight", i))),
                post_ln_weight: ctx.buffer_from_slice(&weights.get_1d(&format!("model.layers.{}.post_attention_layernorm.weight", i))),
            });
        }

        let final_norm_weight = ctx.buffer_from_slice(&weights.get_1d("model.norm.weight"));
        let lm_head_data = weights.get_1d_raw("lm_head.weight");
        let lm_head_weight = ctx.buffer_from_f32_as_q4(&lm_head_data, config.vocab_size, config.hidden_size);

        // Pre-allocate scratch buffers
        let hidden_buf = ctx.buffer_empty(hidden_size);
        let normed_buf = ctx.buffer_empty(hidden_size);
        let residual_buf = ctx.buffer_empty(hidden_size);
        let q_buf = ctx.buffer_empty(num_heads * head_dim);
        let k_buf = ctx.buffer_empty(num_kv_heads * head_dim);
        let v_buf = ctx.buffer_empty(num_kv_heads * head_dim);
        let attn_out_buf = ctx.buffer_empty(num_heads * head_dim);
        let o_out_buf = ctx.buffer_empty(hidden_size);
        let gate_buf = ctx.buffer_empty(intermediate_size);
        let up_buf = ctx.buffer_empty(intermediate_size);
        let silu_buf = ctx.buffer_empty(intermediate_size);
        let down_buf = ctx.buffer_empty(hidden_size);
        let logits_buf = ctx.buffer_empty(vocab_size);

        // KV cache: pre-allocate for full context window (no eviction)
        let kv_capacity = config.max_position_embeddings as u32; // full context
        let mut k_cache = Vec::with_capacity(config.num_hidden_layers);
        let mut v_cache = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            k_cache.push(ctx.buffer_empty(num_kv_heads * kv_capacity as usize * head_dim));
            v_cache.push(ctx.buffer_empty(num_kv_heads * kv_capacity as usize * head_dim));
        }
        let kv_seq_lens = vec![0u32; config.num_hidden_layers];

        // Rotary
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / config.rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let cos_buf = ctx.buffer_empty(head_dim);
        let sin_buf = ctx.buffer_empty(head_dim);

        GpuLlamaModel {
            ctx, config: config.clone(), embed_tokens, layers,
            final_norm_weight, lm_head_weight,
            hidden_buf, normed_buf, residual_buf,
            q_buf, k_buf, v_buf, attn_out_buf, o_out_buf,
            gate_buf, up_buf, silu_buf, down_buf, logits_buf,
            k_cache, v_cache, kv_seq_lens, kv_capacity,
            inv_freq, cos_buf, sin_buf,
            total_tokens: 0,
        }
    }

    /// Forward one token. ALL GPU work in a SINGLE command buffer submission.
    pub fn forward_single_token(&mut self, token_id: usize) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let head_dim = self.config.head_dim();
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let num_kv_groups = (num_heads / num_kv_heads) as u32;
        let intermediate_size = self.config.intermediate_size;
        let vocab_size = self.config.vocab_size;
        let eps = self.config.rms_norm_eps as f32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Embed token (CPU, trivial)
        let embed_offset = token_id * hidden_size;
        let embed_slice = &self.embed_tokens[embed_offset..embed_offset + hidden_size];
        MetalContext::write_buffer(&self.hidden_buf, embed_slice);

        // Compute rotary cos/sin for current position (CPU, trivial)
        let pos = self.total_tokens as f32;
        let half_dim = head_dim / 2;
        let mut cos_data = vec![0.0f32; head_dim];
        let mut sin_data = vec![0.0f32; head_dim];
        for (i, &freq) in self.inv_freq.iter().enumerate() {
            let angle = pos * freq;
            cos_data[i] = angle.cos();
            cos_data[i + half_dim] = angle.cos();
            sin_data[i] = angle.sin();
            sin_data[i + half_dim] = angle.sin();
        }
        MetalContext::write_buffer(&self.cos_buf, &cos_data);
        MetalContext::write_buffer(&self.sin_buf, &sin_data);

        // Current KV sequence length (same for all layers in streaming mode)
        let kv_seq = self.kv_seq_lens[0];

        // ═══ SINGLE COMMAND BUFFER FOR ENTIRE FORWARD PASS ═══
        let cmd = self.ctx.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Copy hidden → residual
            self.ctx.encode_copy(encoder, &self.hidden_buf, &self.residual_buf, hidden_size as u32);

            // RMS Norm
            self.ctx.encode_rmsnorm(encoder, &self.hidden_buf, &layer.input_ln_weight, &self.normed_buf, hidden_size as u32, eps);

            // Q, K, V projections
            self.ctx.encode_matvec_q4(encoder, &layer.q_proj, &self.normed_buf, &self.q_buf, (num_heads * head_dim) as u32, hidden_size as u32);
            self.ctx.encode_matvec_q4(encoder, &layer.k_proj, &self.normed_buf, &self.k_buf, (num_kv_heads * head_dim) as u32, hidden_size as u32);
            self.ctx.encode_matvec_q4(encoder, &layer.v_proj, &self.normed_buf, &self.v_buf, (num_kv_heads * head_dim) as u32, hidden_size as u32);

            // Rotary embeddings
            self.ctx.encode_rotary(encoder, &self.q_buf, &self.k_buf, &self.cos_buf, &self.sin_buf, num_heads as u32, num_kv_heads as u32, head_dim as u32);

            // Append K, V to GPU cache
            self.ctx.encode_kv_append(encoder, &self.k_buf, &self.k_cache[layer_idx], num_kv_heads as u32, head_dim as u32, self.kv_capacity, kv_seq);
            self.ctx.encode_kv_append(encoder, &self.v_buf, &self.v_cache[layer_idx], num_kv_heads as u32, head_dim as u32, self.kv_capacity, kv_seq);

            // Attention (kv_seq + 1 because we just appended)
            let attn_kv_seq = kv_seq + 1;
            self.ctx.encode_attention(encoder, &self.q_buf, &self.k_cache[layer_idx], &self.v_cache[layer_idx], &self.attn_out_buf,
                num_heads as u32, num_kv_heads as u32, num_kv_groups,
                head_dim as u32, attn_kv_seq, self.kv_capacity, scale);

            // O projection
            self.ctx.encode_matvec_q4(encoder, &layer.o_proj, &self.attn_out_buf, &self.o_out_buf, hidden_size as u32, (num_heads * head_dim) as u32);

            // Residual add → hidden
            self.ctx.encode_vec_add(encoder, &self.residual_buf, &self.o_out_buf, &self.hidden_buf, hidden_size as u32);

            // Copy hidden → residual (for MLP residual)
            self.ctx.encode_copy(encoder, &self.hidden_buf, &self.residual_buf, hidden_size as u32);

            // Post-attention norm
            self.ctx.encode_rmsnorm(encoder, &self.hidden_buf, &layer.post_ln_weight, &self.normed_buf, hidden_size as u32, eps);

            // MLP
            self.ctx.encode_matvec_q4(encoder, &layer.gate_proj, &self.normed_buf, &self.gate_buf, intermediate_size as u32, hidden_size as u32);
            self.ctx.encode_matvec_q4(encoder, &layer.up_proj, &self.normed_buf, &self.up_buf, intermediate_size as u32, hidden_size as u32);
            self.ctx.encode_silu_mul(encoder, &self.gate_buf, &self.up_buf, &self.silu_buf, intermediate_size as u32);
            self.ctx.encode_matvec_q4(encoder, &layer.down_proj, &self.silu_buf, &self.down_buf, hidden_size as u32, intermediate_size as u32);

            // Residual add → hidden
            self.ctx.encode_vec_add(encoder, &self.residual_buf, &self.down_buf, &self.hidden_buf, hidden_size as u32);
        }

        // Final norm
        self.ctx.encode_rmsnorm(encoder, &self.hidden_buf, &self.final_norm_weight, &self.normed_buf, hidden_size as u32, eps);

        // LM head
        self.ctx.encode_matvec_q4(encoder, &self.lm_head_weight, &self.normed_buf, &self.logits_buf, vocab_size as u32, hidden_size as u32);

        // Submit and wait
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // Update KV tracking
        self.total_tokens += 1;
        for seq in self.kv_seq_lens.iter_mut() {
            *seq += 1;
        }

        // Read logits back
        MetalContext::read_buffer(&self.logits_buf, vocab_size)
    }

    pub fn num_items(&self) -> usize {
        self.total_tokens
    }

    /// Batched prefill: process all prompt tokens.
    /// Uses single-token path since weights are f16 and matmul kernel expects f32.
    /// For short prompts this is fast enough.
    pub fn forward_prefill(&mut self, token_ids: &[usize]) -> Vec<f32> {
        let mut logits = Vec::new();
        for &tid in token_ids {
            logits = self.forward_single_token(tid);
        }
        logits
    }
}
