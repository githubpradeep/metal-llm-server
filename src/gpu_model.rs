use metal::*;
use crate::cache::StreamingKVCache;
use crate::config::LlamaConfig;
use crate::gpu::MetalContext;
use crate::weights::ModelWeights;

/// GPU-resident model: all weights stored as Metal buffers in shared memory.
/// Forward pass runs entirely on GPU, only logits are read back to CPU.
pub struct GpuLlamaModel {
    pub ctx: MetalContext,
    pub config: LlamaConfig,

    // Embedding table (vocab_size * hidden_size)
    pub embed_tokens: Vec<f32>, // kept on CPU for token lookup (tiny cost)

    // Per-layer weights as Metal buffers
    pub layers: Vec<GpuDecoderLayer>,

    // Final norm
    pub final_norm_weight: Buffer,

    // LM head
    pub lm_head_weight: Buffer, // (vocab_size, hidden_size)
}

pub struct GpuDecoderLayer {
    pub q_proj: Buffer,    // (num_heads * head_dim, hidden_size)
    pub k_proj: Buffer,    // (num_kv_heads * head_dim, hidden_size)
    pub v_proj: Buffer,    // (num_kv_heads * head_dim, hidden_size)
    pub o_proj: Buffer,    // (hidden_size, num_heads * head_dim)
    pub gate_proj: Buffer, // (intermediate_size, hidden_size)
    pub up_proj: Buffer,   // (intermediate_size, hidden_size)
    pub down_proj: Buffer, // (hidden_size, intermediate_size)
    pub input_ln_weight: Buffer,
    pub post_ln_weight: Buffer,
}

impl GpuLlamaModel {
    pub fn new(config: &LlamaConfig, weights: &ModelWeights) -> Self {
        let ctx = MetalContext::new();

        let embed_tokens = weights.get_1d_raw("model.embed_tokens.weight");

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            println!("  Loading GPU layer {}/{}", i + 1, config.num_hidden_layers);
            let layer = GpuDecoderLayer {
                q_proj: ctx.buffer_from_slice(
                    &weights.get_1d_raw(&format!("model.layers.{}.self_attn.q_proj.weight", i))
                ),
                k_proj: ctx.buffer_from_slice(
                    &weights.get_1d_raw(&format!("model.layers.{}.self_attn.k_proj.weight", i))
                ),
                v_proj: ctx.buffer_from_slice(
                    &weights.get_1d_raw(&format!("model.layers.{}.self_attn.v_proj.weight", i))
                ),
                o_proj: ctx.buffer_from_slice(
                    &weights.get_1d_raw(&format!("model.layers.{}.self_attn.o_proj.weight", i))
                ),
                gate_proj: ctx.buffer_from_slice(
                    &weights.get_1d_raw(&format!("model.layers.{}.mlp.gate_proj.weight", i))
                ),
                up_proj: ctx.buffer_from_slice(
                    &weights.get_1d_raw(&format!("model.layers.{}.mlp.up_proj.weight", i))
                ),
                down_proj: ctx.buffer_from_slice(
                    &weights.get_1d_raw(&format!("model.layers.{}.mlp.down_proj.weight", i))
                ),
                input_ln_weight: ctx.buffer_from_slice(
                    &weights.get_1d(&format!("model.layers.{}.input_layernorm.weight", i))
                ),
                post_ln_weight: ctx.buffer_from_slice(
                    &weights.get_1d(&format!("model.layers.{}.post_attention_layernorm.weight", i))
                ),
            };
            layers.push(layer);
        }

        let final_norm_weight = ctx.buffer_from_slice(&weights.get_1d("model.norm.weight"));

        let lm_head_data = weights.get_1d_raw("lm_head.weight");
        let lm_head_weight = ctx.buffer_from_slice(&lm_head_data);

        GpuLlamaModel {
            ctx,
            config: config.clone(),
            embed_tokens,
            layers,
            final_norm_weight,
            lm_head_weight,
        }
    }

    /// Run a single-token forward pass on GPU.
    /// Returns logits as Vec<f32> on CPU.
    pub fn forward_single_token(
        &self,
        token_id: usize,
        kv_cache: &mut StreamingKVCache,
    ) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let head_dim = self.config.head_dim();
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let intermediate_size = self.config.intermediate_size;
        let vocab_size = self.config.vocab_size;
        let eps = self.config.rms_norm_eps as f32;

        // Embed token (CPU lookup, tiny)
        let embed_offset = token_id * hidden_size;
        let embed_slice = &self.embed_tokens[embed_offset..embed_offset + hidden_size];

        // Compute position embeddings (CPU, tiny)
        let pos = kv_cache.num_items() as f32;
        let half_dim = head_dim / 2;
        let mut cos_data = vec![0.0f32; head_dim];
        let mut sin_data = vec![0.0f32; head_dim];
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / self.config.rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        for (i, &freq) in inv_freq.iter().enumerate() {
            let angle = pos * freq;
            cos_data[i] = angle.cos();
            cos_data[i + half_dim] = angle.cos();
            sin_data[i] = angle.sin();
            sin_data[i + half_dim] = angle.sin();
        }

        // Upload to GPU buffers
        let mut hidden_buf = self.ctx.buffer_from_slice(embed_slice);
        let cos_buf = self.ctx.buffer_from_slice(&cos_data);
        let sin_buf = self.ctx.buffer_from_slice(&sin_data);

        // Scratch buffers
        let normed_buf = self.ctx.buffer_empty(hidden_size);
        let q_buf = self.ctx.buffer_empty(num_heads * head_dim);
        let k_buf = self.ctx.buffer_empty(num_kv_heads * head_dim);
        let v_buf = self.ctx.buffer_empty(num_kv_heads * head_dim);
        let attn_out_buf = self.ctx.buffer_empty(num_heads * head_dim);
        let o_out_buf = self.ctx.buffer_empty(hidden_size);
        let residual_buf = self.ctx.buffer_empty(hidden_size);
        let gate_buf = self.ctx.buffer_empty(intermediate_size);
        let up_buf = self.ctx.buffer_empty(intermediate_size);
        let silu_buf = self.ctx.buffer_empty(intermediate_size);
        let down_buf = self.ctx.buffer_empty(hidden_size);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Save residual (copy hidden → residual)
            // Use vec_add with zero to copy (or just read/write)
            let hidden_data = MetalContext::read_buffer(&hidden_buf, hidden_size);
            MetalContext::write_buffer(&residual_buf, &hidden_data);

            // RMS Norm
            self.ctx.rmsnorm(&hidden_buf, &layer.input_ln_weight, &normed_buf, hidden_size as u32, eps);

            // Q, K, V projections
            self.ctx.matvec(&layer.q_proj, &normed_buf, &q_buf, (num_heads * head_dim) as u32, hidden_size as u32);
            self.ctx.matvec(&layer.k_proj, &normed_buf, &k_buf, (num_kv_heads * head_dim) as u32, hidden_size as u32);
            self.ctx.matvec(&layer.v_proj, &normed_buf, &v_buf, (num_kv_heads * head_dim) as u32, hidden_size as u32);

            // Rotary embeddings
            self.ctx.apply_rotary(&q_buf, &k_buf, &cos_buf, &sin_buf, num_heads as u32, num_kv_heads as u32, head_dim as u32);

            // Update KV cache (CPU side — cache is small)
            let k_data = MetalContext::read_buffer(&k_buf, num_kv_heads * head_dim);
            let v_data = MetalContext::read_buffer(&v_buf, num_kv_heads * head_dim);
            let kv_seq = kv_cache.update(&k_data, &v_data, 1, num_kv_heads, head_dim, layer_idx);

            // Upload KV cache to GPU for attention
            let (k_cache_slice, _, k_cap) = kv_cache.get_key_slice(layer_idx);
            let (v_cache_slice, _, v_cap) = kv_cache.get_value_slice(layer_idx);
            let k_cache_buf = self.ctx.buffer_from_slice(k_cache_slice);
            let v_cache_buf = self.ctx.buffer_from_slice(v_cache_slice);

            // Attention
            let scale = 1.0 / (head_dim as f32).sqrt();
            self.ctx.attention_single_token(
                &q_buf, &k_cache_buf, &v_cache_buf, &attn_out_buf,
                num_heads as u32, num_kv_heads as u32, num_kv_groups as u32,
                head_dim as u32, kv_seq as u32, k_cap as u32, scale,
            );

            // O projection
            self.ctx.matvec(&layer.o_proj, &attn_out_buf, &o_out_buf, hidden_size as u32, (num_heads * head_dim) as u32);

            // Residual add
            self.ctx.vec_add(&residual_buf, &o_out_buf, &hidden_buf, hidden_size as u32);

            // Save residual
            let hidden_data = MetalContext::read_buffer(&hidden_buf, hidden_size);
            MetalContext::write_buffer(&residual_buf, &hidden_data);

            // Post-attention norm
            self.ctx.rmsnorm(&hidden_buf, &layer.post_ln_weight, &normed_buf, hidden_size as u32, eps);

            // MLP: gate and up projections
            self.ctx.matvec(&layer.gate_proj, &normed_buf, &gate_buf, intermediate_size as u32, hidden_size as u32);
            self.ctx.matvec(&layer.up_proj, &normed_buf, &up_buf, intermediate_size as u32, hidden_size as u32);

            // SiLU(gate) * up
            self.ctx.silu_mul(&gate_buf, &up_buf, &silu_buf, intermediate_size as u32);

            // Down projection
            self.ctx.matvec(&layer.down_proj, &silu_buf, &down_buf, hidden_size as u32, intermediate_size as u32);

            // Residual add
            self.ctx.vec_add(&residual_buf, &down_buf, &hidden_buf, hidden_size as u32);
        }

        // Final norm
        self.ctx.rmsnorm(&hidden_buf, &self.final_norm_weight, &normed_buf, hidden_size as u32, eps);

        // LM head
        let logits_buf = self.ctx.buffer_empty(vocab_size);
        self.ctx.matvec(&self.lm_head_weight, &normed_buf, &logits_buf, vocab_size as u32, hidden_size as u32);

        // Read logits back to CPU
        MetalContext::read_buffer(&logits_buf, vocab_size)
    }
}
