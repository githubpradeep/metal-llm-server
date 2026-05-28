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
            println!("  Loading GPU layer {}/{}", i + 1, config.num_hidden_layers);
            layers.push(GpuDecoderLayer {
                q_proj: ctx.buffer_from_slice(&weights.get_1d_raw(&format!("model.layers.{}.self_attn.q_proj.weight", i))),
                k_proj: ctx.buffer_from_slice(&weights.get_1d_raw(&format!("model.layers.{}.self_attn.k_proj.weight", i))),
                v_proj: ctx.buffer_from_slice(&weights.get_1d_raw(&format!("model.layers.{}.self_attn.v_proj.weight", i))),
                o_proj: ctx.buffer_from_slice(&weights.get_1d_raw(&format!("model.layers.{}.self_attn.o_proj.weight", i))),
                gate_proj: ctx.buffer_from_slice(&weights.get_1d_raw(&format!("model.layers.{}.mlp.gate_proj.weight", i))),
                up_proj: ctx.buffer_from_slice(&weights.get_1d_raw(&format!("model.layers.{}.mlp.up_proj.weight", i))),
                down_proj: ctx.buffer_from_slice(&weights.get_1d_raw(&format!("model.layers.{}.mlp.down_proj.weight", i))),
                input_ln_weight: ctx.buffer_from_slice(&weights.get_1d(&format!("model.layers.{}.input_layernorm.weight", i))),
                post_ln_weight: ctx.buffer_from_slice(&weights.get_1d(&format!("model.layers.{}.post_attention_layernorm.weight", i))),
            });
        }

        let final_norm_weight = ctx.buffer_from_slice(&weights.get_1d("model.norm.weight"));
        let lm_head_weight = ctx.buffer_from_slice(&weights.get_1d_raw("lm_head.weight"));

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

        // KV cache: pre-allocate for sink_size + window_size tokens
        let kv_capacity = 128u32; // sink(4) + window(64) + headroom
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
            self.ctx.encode_matvec(encoder, &layer.q_proj, &self.normed_buf, &self.q_buf, (num_heads * head_dim) as u32, hidden_size as u32);
            self.ctx.encode_matvec(encoder, &layer.k_proj, &self.normed_buf, &self.k_buf, (num_kv_heads * head_dim) as u32, hidden_size as u32);
            self.ctx.encode_matvec(encoder, &layer.v_proj, &self.normed_buf, &self.v_buf, (num_kv_heads * head_dim) as u32, hidden_size as u32);

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
            self.ctx.encode_matvec(encoder, &layer.o_proj, &self.attn_out_buf, &self.o_out_buf, hidden_size as u32, (num_heads * head_dim) as u32);

            // Residual add → hidden
            self.ctx.encode_vec_add(encoder, &self.residual_buf, &self.o_out_buf, &self.hidden_buf, hidden_size as u32);

            // Copy hidden → residual (for MLP residual)
            self.ctx.encode_copy(encoder, &self.hidden_buf, &self.residual_buf, hidden_size as u32);

            // Post-attention norm
            self.ctx.encode_rmsnorm(encoder, &self.hidden_buf, &layer.post_ln_weight, &self.normed_buf, hidden_size as u32, eps);

            // MLP
            self.ctx.encode_matvec(encoder, &layer.gate_proj, &self.normed_buf, &self.gate_buf, intermediate_size as u32, hidden_size as u32);
            self.ctx.encode_matvec(encoder, &layer.up_proj, &self.normed_buf, &self.up_buf, intermediate_size as u32, hidden_size as u32);
            self.ctx.encode_silu_mul(encoder, &self.gate_buf, &self.up_buf, &self.silu_buf, intermediate_size as u32);
            self.ctx.encode_matvec(encoder, &layer.down_proj, &self.silu_buf, &self.down_buf, hidden_size as u32, intermediate_size as u32);

            // Residual add → hidden
            self.ctx.encode_vec_add(encoder, &self.residual_buf, &self.down_buf, &self.hidden_buf, hidden_size as u32);
        }

        // Final norm
        self.ctx.encode_rmsnorm(encoder, &self.hidden_buf, &self.final_norm_weight, &self.normed_buf, hidden_size as u32, eps);

        // LM head
        self.ctx.encode_matvec(encoder, &self.lm_head_weight, &self.normed_buf, &self.logits_buf, vocab_size as u32, hidden_size as u32);

        // Submit and wait
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // Update KV tracking
        self.total_tokens += 1;
        for seq in self.kv_seq_lens.iter_mut() {
            *seq += 1;
        }

        // Evict middle tokens if cache exceeds sink + window budget
        let keep_total = 68u32; // sink(4) + window(64)
        let sink_size = 4u32;
        let window_size = 64u32;
        if self.kv_seq_lens[0] > keep_total {
            self.evict_kv_cache(sink_size, window_size);
        }

        // Read logits back
        MetalContext::read_buffer(&self.logits_buf, vocab_size)
    }

    /// Evict middle tokens from KV cache: keep first sink_size + last window_size.
    /// Done on CPU since it's infrequent and the cache is small.
    fn evict_kv_cache(&mut self, sink_size: u32, window_size: u32) {
        let head_dim = self.config.head_dim();
        let num_kv_heads = self.config.num_key_value_heads;
        let cap = self.kv_capacity as usize;
        let cur_seq = self.kv_seq_lens[0] as usize;
        let keep_total = (sink_size + window_size) as usize;
        let tail_start = cur_seq - window_size as usize;

        for layer_idx in 0..self.config.num_hidden_layers {
            let k_data = MetalContext::read_buffer(&self.k_cache[layer_idx], num_kv_heads * cap * head_dim);
            let v_data = MetalContext::read_buffer(&self.v_cache[layer_idx], num_kv_heads * cap * head_dim);

            let mut new_k = k_data.clone();
            let mut new_v = v_data.clone();

            // For each head: move tail tokens right after sink tokens
            for h in 0..num_kv_heads {
                let base = h * cap * head_dim;
                let sink_end = base + sink_size as usize * head_dim;
                let tail_src = base + tail_start * head_dim;
                let tail_len = window_size as usize * head_dim;

                // Copy tail after sink
                new_k.copy_within(tail_src..tail_src + tail_len, sink_end);
                new_v.copy_within(tail_src..tail_src + tail_len, sink_end);
            }

            MetalContext::write_buffer(&self.k_cache[layer_idx], &new_k);
            MetalContext::write_buffer(&self.v_cache[layer_idx], &new_v);
        }

        for seq in self.kv_seq_lens.iter_mut() {
            *seq = keep_total as u32;
        }
    }

    pub fn num_items(&self) -> usize {
        self.total_tokens
    }

    /// Batched prefill: process all prompt tokens in one GPU submission per layer.
    /// Much faster than processing one token at a time.
    pub fn forward_prefill(&mut self, token_ids: &[usize]) -> Vec<f32> {
        let seq_len = token_ids.len();
        let hidden_size = self.config.hidden_size;
        let head_dim = self.config.head_dim();
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let num_kv_groups = (num_heads / num_kv_heads) as u32;
        let intermediate_size = self.config.intermediate_size;
        let vocab_size = self.config.vocab_size;
        let eps = self.config.rms_norm_eps as f32;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let half_dim = head_dim / 2;

        // Embed all tokens
        let mut embed_data = vec![0.0f32; seq_len * hidden_size];
        for (i, &tid) in token_ids.iter().enumerate() {
            let src = tid * hidden_size;
            embed_data[i * hidden_size..(i + 1) * hidden_size]
                .copy_from_slice(&self.embed_tokens[src..src + hidden_size]);
        }

        // Compute cos/sin for all positions
        let mut cos_data = vec![0.0f32; seq_len * head_dim];
        let mut sin_data = vec![0.0f32; seq_len * head_dim];
        for s in 0..seq_len {
            let pos = (self.total_tokens + s) as f32;
            let offset = s * head_dim;
            for (i, &freq) in self.inv_freq.iter().enumerate() {
                let angle = pos * freq;
                cos_data[offset + i] = angle.cos();
                cos_data[offset + i + half_dim] = angle.cos();
                sin_data[offset + i] = angle.sin();
                sin_data[offset + i + half_dim] = angle.sin();
            }
        }

        // Allocate prefill buffers
        let hidden_buf = self.ctx.buffer_from_slice(&embed_data);
        let cos_buf = self.ctx.buffer_from_slice(&cos_data);
        let sin_buf = self.ctx.buffer_from_slice(&sin_data);
        let normed_buf = self.ctx.buffer_empty(seq_len * hidden_size);
        let residual_buf = self.ctx.buffer_empty(seq_len * hidden_size);
        let q_buf = self.ctx.buffer_empty(seq_len * num_heads * head_dim);
        let k_buf = self.ctx.buffer_empty(seq_len * num_kv_heads * head_dim);
        let v_buf = self.ctx.buffer_empty(seq_len * num_kv_heads * head_dim);
        let q_t_buf = self.ctx.buffer_empty(num_heads * seq_len * head_dim);
        let k_t_buf = self.ctx.buffer_empty(num_kv_heads * seq_len * head_dim);
        let v_t_buf = self.ctx.buffer_empty(num_kv_heads * seq_len * head_dim);
        let attn_out_buf = self.ctx.buffer_empty(num_heads * seq_len * head_dim);
        let attn_out_t_buf = self.ctx.buffer_empty(seq_len * num_heads * head_dim);
        let o_out_buf = self.ctx.buffer_empty(seq_len * hidden_size);
        let gate_buf = self.ctx.buffer_empty(seq_len * intermediate_size);
        let up_buf = self.ctx.buffer_empty(seq_len * intermediate_size);
        let silu_buf = self.ctx.buffer_empty(seq_len * intermediate_size);
        let down_buf = self.ctx.buffer_empty(seq_len * hidden_size);

        let total_hidden = (seq_len * hidden_size) as u32;
        let sl = seq_len as u32;

        // Single command buffer for entire prefill
        let cmd = self.ctx.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Copy hidden → residual
            self.ctx.encode_copy(encoder, &hidden_buf, &residual_buf, total_hidden);

            // Batched RMS Norm
            self.ctx.encode_rmsnorm_batch(encoder, &hidden_buf, &layer.input_ln_weight, &normed_buf, hidden_size as u32, eps, sl);

            // Q, K, V: matmul (seq_len, hidden) × (out, hidden)^T → (seq_len, out)
            self.ctx.encode_matmul(encoder, &normed_buf, &layer.q_proj, &q_buf, sl, (num_heads * head_dim) as u32, hidden_size as u32);
            self.ctx.encode_matmul(encoder, &normed_buf, &layer.k_proj, &k_buf, sl, (num_kv_heads * head_dim) as u32, hidden_size as u32);
            self.ctx.encode_matmul(encoder, &normed_buf, &layer.v_proj, &v_buf, sl, (num_kv_heads * head_dim) as u32, hidden_size as u32);

            // Transpose Q: (seq, heads, hd) → (heads, seq, hd)
            self.ctx.encode_transpose_shd(encoder, &q_buf, &q_t_buf, sl, num_heads as u32, head_dim as u32);
            self.ctx.encode_transpose_shd(encoder, &k_buf, &k_t_buf, sl, num_kv_heads as u32, head_dim as u32);
            self.ctx.encode_transpose_shd(encoder, &v_buf, &v_t_buf, sl, num_kv_heads as u32, head_dim as u32);

            // Rotary (batched)
            self.ctx.encode_rotary_batch(encoder, &q_t_buf, &k_t_buf, &cos_buf, &sin_buf, num_heads as u32, num_kv_heads as u32, head_dim as u32, sl);

            // Append all K, V to cache
            let kv_seq = self.kv_seq_lens[layer_idx];
            self.ctx.encode_kv_batch_append(encoder, &k_t_buf, &self.k_cache[layer_idx], num_kv_heads as u32, head_dim as u32, self.kv_capacity, kv_seq, sl);
            self.ctx.encode_kv_batch_append(encoder, &v_t_buf, &self.v_cache[layer_idx], num_kv_heads as u32, head_dim as u32, self.kv_capacity, kv_seq, sl);

            // Causal attention
            let attn_kv_seq = kv_seq + sl;
            self.ctx.encode_attention_causal(encoder, &q_t_buf, &self.k_cache[layer_idx], &self.v_cache[layer_idx], &attn_out_buf,
                num_heads as u32, num_kv_heads as u32, num_kv_groups,
                head_dim as u32, attn_kv_seq, self.kv_capacity, scale, sl);

            // Transpose back: (heads, seq, hd) → (seq, heads, hd)
            self.ctx.encode_transpose_hsd(encoder, &attn_out_buf, &attn_out_t_buf, sl, num_heads as u32, head_dim as u32);

            // O projection: (seq, hidden) × (hidden, hidden)^T
            self.ctx.encode_matmul(encoder, &attn_out_t_buf, &layer.o_proj, &o_out_buf, sl, hidden_size as u32, (num_heads * head_dim) as u32);

            // Residual add
            self.ctx.encode_vec_add_batch(encoder, &residual_buf, &o_out_buf, &hidden_buf, total_hidden);

            // Copy hidden → residual
            self.ctx.encode_copy(encoder, &hidden_buf, &residual_buf, total_hidden);

            // Post-attention norm
            self.ctx.encode_rmsnorm_batch(encoder, &hidden_buf, &layer.post_ln_weight, &normed_buf, hidden_size as u32, eps, sl);

            // MLP
            self.ctx.encode_matmul(encoder, &normed_buf, &layer.gate_proj, &gate_buf, sl, intermediate_size as u32, hidden_size as u32);
            self.ctx.encode_matmul(encoder, &normed_buf, &layer.up_proj, &up_buf, sl, intermediate_size as u32, hidden_size as u32);
            self.ctx.encode_silu_mul_batch(encoder, &gate_buf, &up_buf, &silu_buf, (seq_len * intermediate_size) as u32);
            self.ctx.encode_matmul(encoder, &silu_buf, &layer.down_proj, &down_buf, sl, hidden_size as u32, intermediate_size as u32);

            // Residual add
            self.ctx.encode_vec_add_batch(encoder, &residual_buf, &down_buf, &hidden_buf, total_hidden);
        }

        // Final norm (only last token needed for logits, but norm all for correctness)
        self.ctx.encode_rmsnorm_batch(encoder, &hidden_buf, &self.final_norm_weight, &normed_buf, hidden_size as u32, eps, sl);

        // LM head on LAST token only (avoid massive seq×vocab matmul)
        // normed_buf has (seq_len, hidden_size) — last token starts at offset (seq_len-1)*hidden_size
        // We use matvec on the last row only by creating a view at the right offset
        let last_token_offset = ((seq_len - 1) * hidden_size * 4) as u64; // byte offset
        let logits_buf = self.ctx.buffer_empty(vocab_size);

        // End current encoder, submit, wait — then do lm_head with offset
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // LM head on last token (separate small command)
        let cmd2 = self.ctx.queue.new_command_buffer();
        let enc2 = cmd2.new_compute_command_encoder();
        enc2.set_compute_pipeline_state(&self.ctx.matvec_pipeline);
        enc2.set_buffer(0, Some(&self.lm_head_weight), 0);
        enc2.set_buffer(1, Some(&normed_buf), last_token_offset);
        enc2.set_buffer(2, Some(&logits_buf), 0);
        let m_val = vocab_size as u32;
        let k_val = hidden_size as u32;
        enc2.set_bytes(3, 4, &m_val as *const u32 as *const _);
        enc2.set_bytes(4, 4, &k_val as *const u32 as *const _);
        // SIMD-group dispatch: one threadgroup per row, 32 threads each
        let num_tgs = MTLSize::new(m_val as u64, 1, 1);
        let tg_size = MTLSize::new(32, 1, 1);
        enc2.dispatch_thread_groups(num_tgs, tg_size);
        enc2.end_encoding();
        cmd2.commit();
        cmd2.wait_until_completed();

        // Update tracking
        self.total_tokens += seq_len;
        for seq in self.kv_seq_lens.iter_mut() {
            *seq += sl;
        }

        // Read logits (just vocab_size floats for last token)
        MetalContext::read_buffer(&logits_buf, vocab_size)
    }
}
