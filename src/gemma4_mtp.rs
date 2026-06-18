use metal::*;
use safetensors::SafeTensors;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::gemma4_config::{KvCacheType, RopeParameters};
use crate::gemma4_gpu_model::Gemma4GpuModel;
use crate::gpu::MetalContext;

pub struct Gemma4MtpAssistant {
    config: AssistantTextConfig,
    backbone_hidden_size: usize,
    num_centroids: usize,
    centroid_top_k: usize,
    vocab_size_per_centroid: usize,
    embed_tokens_f32: Vec<f32>,
    centroids_weight_f32: Vec<f32>,
    token_ordering_clusters: Vec<Vec<usize>>,
    pre_projection: Buffer,
    post_projection: Buffer,
    layers: Vec<AssistantLayer>,
    final_norm_weight: Buffer,
    activation_buf: Buffer,
    hidden_buf: Buffer,
    normed_buf: Buffer,
    residual_buf: Buffer,
    q_buf: Buffer,
    q_rotary_buf: Buffer,
    attn_out_buf: Buffer,
    o_out_buf: Buffer,
    gate_buf: Buffer,
    up_buf: Buffer,
    gelu_buf: Buffer,
    projected_activation_buf: Buffer,
    centroids_weight_buf: Buffer,
    centroid_scores_buf: Buffer,
    per_layer_cos_bufs: Vec<Buffer>,
    per_layer_sin_bufs: Vec<Buffer>,
    embed_scale: f32,
    cached_rotary_pos: Option<usize>,
    cached_target_tokens: usize,
    pub gpu_passes: u64,
}

pub struct MtpDraft {
    pub token_id: usize,
    pub projected_activation: Vec<f32>,
}

struct AssistantLayer {
    q_proj: Buffer,
    o_proj: Buffer,
    gate_proj: Buffer,
    up_proj: Buffer,
    down_proj: Buffer,
    input_layernorm_weight: Buffer,
    post_attention_layernorm_weight: Buffer,
    pre_feedforward_layernorm_weight: Buffer,
    post_feedforward_layernorm_weight: Buffer,
    q_norm_weight: Buffer,
    layer_scalar: f32,
    is_full_attention: bool,
    head_dim: usize,
    q_out_dim: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct AssistantOuterConfig {
    backbone_hidden_size: usize,
    #[serde(default = "default_num_centroids")]
    num_centroids: usize,
    #[serde(default = "default_centroid_top_k")]
    centroid_intermediate_top_k: usize,
    #[serde(default)]
    use_ordered_embeddings: bool,
    text_config: AssistantTextConfig,
}

fn default_num_centroids() -> usize {
    2048
}

fn default_centroid_top_k() -> usize {
    32
}

#[derive(Clone, Debug, Deserialize)]
struct AssistantTextConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    #[serde(default = "default_global_head_dim")]
    global_head_dim: usize,
    intermediate_size: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    sliding_window: usize,
    layer_types: Vec<String>,
    #[serde(default)]
    final_logit_softcapping: Option<f32>,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
}

impl AssistantTextConfig {
    fn is_full_attention(&self, layer_idx: usize) -> bool {
        self.layer_types
            .get(layer_idx)
            .map_or(false, |layer_type| layer_type == "full_attention")
    }

    fn layer_head_dim(&self, layer_idx: usize) -> usize {
        if self.is_full_attention(layer_idx) {
            self.global_head_dim
        } else {
            self.head_dim
        }
    }

    fn sliding_rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.sliding_attention.as_ref())
            .map_or(10000.0, |c| c.rope_theta)
    }

    fn full_rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.full_attention.as_ref())
            .map_or(1000000.0, |c| c.rope_theta)
    }

    fn full_partial_rotary_factor(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.full_attention.as_ref())
            .map_or(0.25, |c| c.partial_rotary_factor)
    }

    fn full_rope_factor(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.full_attention.as_ref())
            .map_or(1.0, |c| c.factor)
    }

    fn sliding_rope_factor(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.sliding_attention.as_ref())
            .map_or(1.0, |c| c.factor)
    }
}

fn default_global_head_dim() -> usize {
    512
}

impl Gemma4MtpAssistant {
    pub fn new(model_dir: &str, ctx: &MetalContext, target: &Gemma4GpuModel) -> Self {
        let config_path = Path::new(model_dir).join("config.json");
        let config_str = fs::read_to_string(&config_path).expect("Failed to read assistant config");
        let outer: AssistantOuterConfig =
            serde_json::from_str(&config_str).expect("Failed to parse assistant config");
        let config = outer.text_config;
        let backbone_hidden_size = outer.backbone_hidden_size;
        let num_centroids = outer.num_centroids;
        let centroid_top_k = outer.centroid_intermediate_top_k;
        assert!(
            outer.use_ordered_embeddings,
            "assistant must use ordered embeddings (centroid lm_head)"
        );
        assert_eq!(
            config.vocab_size % num_centroids,
            0,
            "vocab_size must be divisible by num_centroids"
        );
        let vocab_size_per_centroid = config.vocab_size / num_centroids;

        assert_eq!(
            backbone_hidden_size, target.config.hidden_size,
            "assistant backbone_hidden_size must match target hidden_size"
        );
        assert_eq!(
            config.num_key_value_heads, target.config.num_key_value_heads,
            "assistant and target KV head counts must match"
        );
        assert_eq!(
            config.vocab_size, target.config.vocab_size,
            "assistant and target vocab sizes must match"
        );

        let tensors = load_assistant_tensors(model_dir);

        let embed_tokens_f16 = raw_tensor_u16(&tensors, "model.embed_tokens.weight");
        let token_ordering_flat = tensor_i64(&tensors, "masked_embedding.token_ordering")
            .into_iter()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        assert_eq!(
            token_ordering_flat.len(),
            config.vocab_size,
            "assistant token_ordering must match vocab size"
        );
        let mut token_ordering_clusters = vec![Vec::with_capacity(vocab_size_per_centroid); num_centroids];
        for (idx, &token_id) in token_ordering_flat.iter().enumerate() {
            let cluster = idx / vocab_size_per_centroid;
            token_ordering_clusters[cluster].push(token_id);
        }
        for cluster in &token_ordering_clusters {
            assert_eq!(
                cluster.len(),
                vocab_size_per_centroid,
                "each centroid cluster must have vocab_size/num_centroids entries"
            );
        }

        let embed_tokens_f32 = embed_tokens_f16
            .iter()
            .map(|&bits| bf16_to_f32(bits))
            .collect::<Vec<_>>();
        let centroids_weight_f32 = tensor_f32(&tensors, "masked_embedding.centroids.weight");
        let hidden_size = config.hidden_size;
        assert_eq!(
            centroids_weight_f32.len(),
            num_centroids * hidden_size,
            "centroids weight must be [num_centroids, hidden_size]"
        );

        let pre_projection = ctx.buffer_from_f32_as_f16(&tensor_f32(&tensors, "pre_projection.weight"));
        let post_projection = ctx.buffer_from_f32_as_f16(&tensor_f32(&tensors, "post_projection.weight"));
        let final_norm_weight = ctx.buffer_from_slice(&tensor_f32(&tensors, "model.norm.weight"));

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", layer_idx);
            let head_dim = config.layer_head_dim(layer_idx);
            let q_out_dim = config.num_attention_heads * head_dim;
            let layer_scalar = tensor_f32(&tensors, &format!("{}.layer_scalar", prefix))[0];

            layers.push(AssistantLayer {
                q_proj: ctx.buffer_from_f32_as_f16(&tensor_f32(&tensors, &format!("{}.self_attn.q_proj.weight", prefix))),
                o_proj: ctx.buffer_from_f32_as_f16(&tensor_f32(&tensors, &format!("{}.self_attn.o_proj.weight", prefix))),
                gate_proj: ctx.buffer_from_f32_as_f16(&tensor_f32(&tensors, &format!("{}.mlp.gate_proj.weight", prefix))),
                up_proj: ctx.buffer_from_f32_as_f16(&tensor_f32(&tensors, &format!("{}.mlp.up_proj.weight", prefix))),
                down_proj: ctx.buffer_from_f32_as_f16(&tensor_f32(&tensors, &format!("{}.mlp.down_proj.weight", prefix))),
                input_layernorm_weight: ctx.buffer_from_slice(&tensor_f32(&tensors, &format!("{}.input_layernorm.weight", prefix))),
                post_attention_layernorm_weight: ctx.buffer_from_slice(&tensor_f32(&tensors, &format!("{}.post_attention_layernorm.weight", prefix))),
                pre_feedforward_layernorm_weight: ctx.buffer_from_slice(&tensor_f32(&tensors, &format!("{}.pre_feedforward_layernorm.weight", prefix))),
                post_feedforward_layernorm_weight: ctx.buffer_from_slice(&tensor_f32(&tensors, &format!("{}.post_feedforward_layernorm.weight", prefix))),
                q_norm_weight: ctx.buffer_from_slice(&tensor_f32(&tensors, &format!("{}.self_attn.q_norm.weight", prefix))),
                layer_scalar,
                is_full_attention: config.is_full_attention(layer_idx),
                head_dim,
                q_out_dim,
            });
        }

        let max_head_dim = config.global_head_dim;
        let max_q_out = config.num_attention_heads * max_head_dim;
        let activation_buf = ctx.buffer_empty(backbone_hidden_size * 2);
        let hidden_buf = ctx.buffer_empty(config.hidden_size);
        let normed_buf = ctx.buffer_empty(config.hidden_size);
        let residual_buf = ctx.buffer_empty(config.hidden_size);
        let q_buf = ctx.buffer_empty(max_q_out);
        let q_rotary_buf = ctx.buffer_empty(max_q_out);
        let attn_out_buf = ctx.buffer_empty(max_q_out);
        let o_out_buf = ctx.buffer_empty(config.hidden_size);
        let gate_buf = ctx.buffer_empty(config.intermediate_size);
        let up_buf = ctx.buffer_empty(config.intermediate_size);
        let gelu_buf = ctx.buffer_empty(config.intermediate_size);
        let projected_activation_buf = ctx.buffer_empty(backbone_hidden_size);
        let centroids_weight_buf = ctx.buffer_from_f32_as_f16(&centroids_weight_f32);
        let centroid_scores_buf = ctx.buffer_empty(num_centroids);
        let embed_scale = (backbone_hidden_size as f32).sqrt();

        let mut per_layer_cos_bufs = Vec::with_capacity(config.num_hidden_layers);
        let mut per_layer_sin_bufs = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let head_dim = config.layer_head_dim(layer_idx);
            per_layer_cos_bufs.push(ctx.buffer_empty(head_dim));
            per_layer_sin_bufs.push(ctx.buffer_empty(head_dim));
        }

        println!(
            "  Gemma4 MTP assistant: {} layers, hidden={}, backbone_hidden={}, vocab={}",
            config.num_hidden_layers, config.hidden_size, backbone_hidden_size, config.vocab_size
        );
        println!(
            "  Centroid lm_head: {} centroids, top_k={}, {} tokens/cluster",
            num_centroids, centroid_top_k, vocab_size_per_centroid
        );
        if let (Some(sliding), Some(full)) = (
            target.mtp_kv_source_layer(false),
            target.mtp_kv_source_layer(true),
        ) {
            println!("  Shared KV sources: sliding_attention=layer {}, full_attention=layer {}", sliding, full);
        }

        Self {
            config,
            backbone_hidden_size,
            num_centroids,
            centroid_top_k,
            vocab_size_per_centroid,
            embed_tokens_f32,
            centroids_weight_f32,
            token_ordering_clusters,
            pre_projection,
            post_projection,
            layers,
            final_norm_weight,
            activation_buf,
            hidden_buf,
            normed_buf,
            residual_buf,
            q_buf,
            q_rotary_buf,
            attn_out_buf,
            o_out_buf,
            gate_buf,
            up_buf,
            gelu_buf,
            projected_activation_buf,
            centroids_weight_buf,
            centroid_scores_buf,
            per_layer_cos_bufs,
            per_layer_sin_bufs,
            embed_scale,
            cached_rotary_pos: None,
            cached_target_tokens: usize::MAX,
            gpu_passes: 0,
        }
    }

    /// Draft only the first token (cheap probe before main-model verify).
    pub fn draft_first(
        &mut self,
        initial_token: usize,
        initial_activation: &[f32],
        target: &Gemma4GpuModel,
    ) -> Result<usize, String> {
        let tokens = self.draft_chain(initial_token, initial_activation, 1, target)?;
        Ok(tokens[0])
    }

    /// Continue drafting after `draft_first` matched. Reuses GPU projected activation.
    pub fn draft_tail(
        &mut self,
        from_token: usize,
        steps: usize,
        target: &Gemma4GpuModel,
    ) -> Result<Vec<usize>, String> {
        if steps == 0 {
            return Ok(Vec::new());
        }
        if target.kv_seq_len == 0 {
            return Err("target KV cache is empty; prefill or decode one token first".to_string());
        }

        self.ensure_rotary(target);

        let mut draft_token = from_token;
        let mut drafts = Vec::with_capacity(steps);

        for _ in 0..steps {
            self.write_activation_first_half(draft_token, target)?;
            self.encode_draft_forward(target, true)?;
            draft_token = self.centroid_argmax_from_gpu()?;
            drafts.push(draft_token);
        }

        Ok(drafts)
    }

    /// Draft multiple tokens with one RoPE table build and minimal GPU/CPU sync.
    pub fn draft_chain(
        &mut self,
        initial_token: usize,
        initial_activation: &[f32],
        steps: usize,
        target: &Gemma4GpuModel,
    ) -> Result<Vec<usize>, String> {
        if steps == 0 {
            return Ok(Vec::new());
        }
        if initial_activation.len() != self.backbone_hidden_size {
            return Err(format!(
                "target activation has {} values, expected {}",
                initial_activation.len(),
                self.backbone_hidden_size
            ));
        }
        if target.kv_seq_len == 0 {
            return Err("target KV cache is empty; prefill or decode one token first".to_string());
        }

        self.ensure_rotary(target);

        let mut draft_token = initial_token;
        let mut drafts = Vec::with_capacity(steps);

        for step in 0..steps {
            if step == 0 {
                self.write_activation_first_half(draft_token, target)?;
                MetalContext::write_buffer_at(
                    &self.activation_buf,
                    self.backbone_hidden_size,
                    initial_activation,
                );
            } else {
                self.write_activation_first_half(draft_token, target)?;
            }

            self.encode_draft_forward(target, step > 0)?;
            draft_token = self.centroid_argmax_from_gpu()?;
            drafts.push(draft_token);
        }

        Ok(drafts)
    }

    pub fn draft_next(
        &mut self,
        token_id: usize,
        target_activation: &[f32],
        target: &Gemma4GpuModel,
    ) -> Result<MtpDraft, String> {
        let tokens = self.draft_chain(token_id, target_activation, 1, target)?;
        let projected_activation =
            MetalContext::read_buffer(&self.projected_activation_buf, self.backbone_hidden_size);
        Ok(MtpDraft {
            token_id: tokens[0],
            projected_activation,
        })
    }

    fn ensure_rotary(&mut self, target: &Gemma4GpuModel) {
        let position = target.total_tokens.saturating_sub(1);
        if self.cached_rotary_pos == Some(position)
            && self.cached_target_tokens == target.total_tokens
        {
            return;
        }
        self.prepare_rotary(position);
        self.cached_rotary_pos = Some(position);
        self.cached_target_tokens = target.total_tokens;
    }

    fn write_activation_first_half(
        &self,
        token_id: usize,
        target: &Gemma4GpuModel,
    ) -> Result<(), String> {
        let hidden_size = self.backbone_hidden_size;
        let embed_offset = token_id
            .checked_mul(hidden_size)
            .ok_or_else(|| format!("token id {} overflowed embedding offset", token_id))?;
        if embed_offset + hidden_size > target.embed_tokens_f16.len() {
            return Err(format!("token id {} is outside embed_tokens", token_id));
        }

        let ptr = self.activation_buf.contents() as *mut f32;
        let scale = self.embed_scale;
        unsafe {
            for i in 0..hidden_size {
                let bits = target.embed_tokens_f16[embed_offset + i];
                *ptr.add(i) = bf16_to_f32(bits) * scale;
            }
        }
        Ok(())
    }

    fn encode_draft_forward(&mut self, target: &Gemma4GpuModel, chain_from_projected: bool) -> Result<(), String> {
        let ctx = &target.ctx;
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let num_kv_groups = (num_heads / num_kv_heads) as u32;
        let eps = self.config.rms_norm_eps as f32;

        let cmd = ctx.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        if chain_from_projected {
            ctx.encode_copy_at(
                encoder,
                &self.projected_activation_buf,
                0,
                &self.activation_buf,
                (self.backbone_hidden_size as u64) * 4,
                self.backbone_hidden_size as u32,
            );
        }

        ctx.encode_matvec_f16(
            encoder,
            &self.pre_projection,
            &self.activation_buf,
            &self.hidden_buf,
            hidden_size as u32,
            (self.backbone_hidden_size * 2) as u32,
        );

        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            let head_dim = layer.head_dim;
            let q_out = layer.q_out_dim;
            let source_layer = target
                .mtp_kv_source_layer(layer.is_full_attention)
                .ok_or_else(|| {
                    format!(
                        "no target KV source for assistant {} layer {}",
                        if layer.is_full_attention { "full" } else { "sliding" },
                        layer_idx
                    )
                })?;
            let k_cache = target
                .k_cache
                .get(source_layer)
                .ok_or_else(|| format!("target K cache layer {} missing", source_layer))?;
            let v_cache = target
                .v_cache
                .get(source_layer)
                .ok_or_else(|| format!("target V cache layer {} missing", source_layer))?;

            ctx.encode_copy(
                encoder,
                &self.hidden_buf,
                &self.residual_buf,
                hidden_size as u32,
            );
            ctx.encode_rmsnorm(
                encoder,
                &self.hidden_buf,
                &layer.input_layernorm_weight,
                &self.normed_buf,
                hidden_size as u32,
                eps,
            );
            ctx.encode_matvec_f16(
                encoder,
                &layer.q_proj,
                &self.normed_buf,
                &self.q_buf,
                q_out as u32,
                hidden_size as u32,
            );
            ctx.encode_rmsnorm_per_head(
                encoder,
                &self.q_buf,
                &layer.q_norm_weight,
                &self.q_rotary_buf,
                num_heads as u32,
                head_dim as u32,
                eps,
            );
            ctx.encode_rotary(
                encoder,
                &self.q_rotary_buf,
                &self.q_buf,
                &self.per_layer_cos_bufs[layer_idx],
                &self.per_layer_sin_bufs[layer_idx],
                num_heads as u32,
                0,
                head_dim as u32,
            );

            let effective_kv_seq = if layer.is_full_attention {
                target.kv_seq_len
            } else {
                target.kv_seq_len.min(self.config.sliding_window as u32)
            };
            let kv_start = if !layer.is_full_attention
                && target.kv_seq_len > self.config.sliding_window as u32
            {
                target.kv_seq_len - self.config.sliding_window as u32
            } else {
                0
            };

            match target.kv_cache_type {
                KvCacheType::F16 => {
                    ctx.encode_attention_with_offset_f16(
                        encoder,
                        &self.q_rotary_buf,
                        k_cache,
                        v_cache,
                        &self.attn_out_buf,
                        num_heads as u32,
                        num_kv_heads as u32,
                        num_kv_groups,
                        head_dim as u32,
                        effective_kv_seq,
                        target.kv_capacity,
                        1.0,
                        kv_start,
                    );
                }
                KvCacheType::Q8_0 => {
                    let groups_per_row = (head_dim / 32) as u32;
                    let row_bytes = groups_per_row * 34;
                    ctx.encode_attention_with_offset_q8_0(
                        encoder,
                        &self.q_rotary_buf,
                        k_cache,
                        v_cache,
                        &self.attn_out_buf,
                        num_heads as u32,
                        num_kv_heads as u32,
                        num_kv_groups,
                        head_dim as u32,
                        effective_kv_seq,
                        target.kv_capacity,
                        1.0,
                        kv_start,
                        groups_per_row,
                        row_bytes,
                    );
                }
                KvCacheType::Q4_0 => {
                    let groups_per_row = (head_dim / 32) as u32;
                    let row_bytes = groups_per_row * 18;
                    ctx.encode_attention_with_offset_q4_0(
                        encoder,
                        &self.q_rotary_buf,
                        k_cache,
                        v_cache,
                        &self.attn_out_buf,
                        num_heads as u32,
                        num_kv_heads as u32,
                        num_kv_groups,
                        head_dim as u32,
                        effective_kv_seq,
                        target.kv_capacity,
                        1.0,
                        kv_start,
                        groups_per_row,
                        row_bytes,
                    );
                }
            }

            ctx.encode_matvec_f16(
                encoder,
                &layer.o_proj,
                &self.attn_out_buf,
                &self.o_out_buf,
                hidden_size as u32,
                q_out as u32,
            );
            ctx.encode_rmsnorm(
                encoder,
                &self.o_out_buf,
                &layer.post_attention_layernorm_weight,
                &self.normed_buf,
                hidden_size as u32,
                eps,
            );
            ctx.encode_vec_add(
                encoder,
                &self.residual_buf,
                &self.normed_buf,
                &self.hidden_buf,
                hidden_size as u32,
            );
            ctx.encode_copy(
                encoder,
                &self.hidden_buf,
                &self.residual_buf,
                hidden_size as u32,
            );
            ctx.encode_rmsnorm(
                encoder,
                &self.hidden_buf,
                &layer.pre_feedforward_layernorm_weight,
                &self.normed_buf,
                hidden_size as u32,
                eps,
            );
            ctx.encode_matvec_f16(
                encoder,
                &layer.gate_proj,
                &self.normed_buf,
                &self.gate_buf,
                intermediate_size as u32,
                hidden_size as u32,
            );
            ctx.encode_matvec_f16(
                encoder,
                &layer.up_proj,
                &self.normed_buf,
                &self.up_buf,
                intermediate_size as u32,
                hidden_size as u32,
            );
            ctx.encode_gelu_mul(
                encoder,
                &self.gate_buf,
                &self.up_buf,
                &self.gelu_buf,
                intermediate_size as u32,
            );
            ctx.encode_matvec_f16(
                encoder,
                &layer.down_proj,
                &self.gelu_buf,
                &self.o_out_buf,
                hidden_size as u32,
                intermediate_size as u32,
            );
            ctx.encode_rmsnorm(
                encoder,
                &self.o_out_buf,
                &layer.post_feedforward_layernorm_weight,
                &self.normed_buf,
                hidden_size as u32,
                eps,
            );
            ctx.encode_vec_add(
                encoder,
                &self.residual_buf,
                &self.normed_buf,
                &self.hidden_buf,
                hidden_size as u32,
            );
            ctx.encode_vec_scale(
                encoder,
                &self.hidden_buf,
                &self.residual_buf,
                hidden_size as u32,
                layer.layer_scalar,
            );
            ctx.encode_copy(
                encoder,
                &self.residual_buf,
                &self.hidden_buf,
                hidden_size as u32,
            );
        }

        ctx.encode_rmsnorm(
            encoder,
            &self.hidden_buf,
            &self.final_norm_weight,
            &self.normed_buf,
            hidden_size as u32,
            eps,
        );
        ctx.encode_matvec_f16(
            encoder,
            &self.centroids_weight_buf,
            &self.normed_buf,
            &self.centroid_scores_buf,
            self.num_centroids as u32,
            hidden_size as u32,
        );
        ctx.encode_matvec_f16(
            encoder,
            &self.post_projection,
            &self.normed_buf,
            &self.projected_activation_buf,
            self.backbone_hidden_size as u32,
            hidden_size as u32,
        );

        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        self.gpu_passes += 1;
        Ok(())
    }

    /// Top-k centroid sparse argmax over ~4096 candidates (no full-vocab buffer).
    fn centroid_argmax_from_gpu(&self) -> Result<usize, String> {
        let hidden_size = self.config.hidden_size;
        let centroid_scores = unsafe {
            std::slice::from_raw_parts(
                self.centroid_scores_buf.contents() as *const f32,
                self.num_centroids,
            )
        };
        let hidden = unsafe {
            std::slice::from_raw_parts(
                self.normed_buf.contents() as *const f32,
                hidden_size,
            )
        };

        let mut ranked_centroids: Vec<usize> = (0..self.num_centroids).collect();
        ranked_centroids.select_nth_unstable_by(self.centroid_top_k - 1, |&a, &b| {
            centroid_scores[b]
                .partial_cmp(&centroid_scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked_centroids.truncate(self.centroid_top_k);

        let mut best_token = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for &cluster in &ranked_centroids {
            for &token_id in &self.token_ordering_clusters[cluster] {
                let emb_offset = token_id * hidden_size;
                let mut dot = 0.0f32;
                for i in 0..hidden_size {
                    dot += self.embed_tokens_f32[emb_offset + i] * hidden[i];
                }
                if dot > best_score {
                    best_score = dot;
                    best_token = token_id;
                }
            }
        }

        Ok(best_token)
    }

    fn prepare_rotary(&self, position: usize) {
        let pos = position as f32;
        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            let head_dim = layer.head_dim;
            let rope_theta = if layer.is_full_attention {
                self.config.full_rope_theta()
            } else {
                self.config.sliding_rope_theta()
            };
            let rope_factor = if layer.is_full_attention {
                self.config.full_rope_factor()
            } else {
                self.config.sliding_rope_factor()
            };
            let rotary_dim = if layer.is_full_attention {
                (head_dim as f64 * self.config.full_partial_rotary_factor()) as usize
            } else {
                head_dim
            };
            let rope_angles = rotary_dim / 2;
            let half_dim = head_dim / 2;
            let mut cos_data = vec![0.0f32; head_dim];
            let mut sin_data = vec![0.0f32; head_dim];
            for i in 0..rope_angles {
                let inv_freq = 1.0
                    / (rope_theta.powf(i as f64 * 2.0 / head_dim as f64) as f32)
                    / rope_factor as f32;
                let angle = pos * inv_freq;
                cos_data[i] = angle.cos();
                cos_data[i + half_dim] = angle.cos();
                sin_data[i] = angle.sin();
                sin_data[i + half_dim] = angle.sin();
            }
            for i in rope_angles..half_dim {
                cos_data[i] = 1.0;
                cos_data[i + half_dim] = 1.0;
            }
            MetalContext::write_buffer(&self.per_layer_cos_bufs[layer_idx], &cos_data);
            MetalContext::write_buffer(&self.per_layer_sin_bufs[layer_idx], &sin_data);
        }
    }
}

fn load_assistant_tensors(model_dir: &str) -> HashMap<String, TensorData> {
    let model_path = Path::new(model_dir).join("model.safetensors");
    let file = fs::File::open(&model_path).expect("Failed to open assistant model.safetensors");
    let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("Failed to mmap assistant weights");
    let safetensors = SafeTensors::deserialize(&mmap).expect("Failed to read assistant safetensors");

    let mut tensors = HashMap::new();
    for (name, tensor_view) in safetensors.tensors() {
        tensors.insert(
            name.to_string(),
            TensorData {
                dtype: tensor_view.dtype(),
                data: tensor_view.data().to_vec(),
            },
        );
    }
    tensors
}

struct TensorData {
    dtype: safetensors::Dtype,
    data: Vec<u8>,
}

fn raw_tensor_u16(tensors: &HashMap<String, TensorData>, name: &str) -> Vec<u16> {
    let tensor = tensors
        .get(name)
        .unwrap_or_else(|| panic!("assistant tensor missing: {}", name));
    tensor
        .data
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

fn tensor_f32(tensors: &HashMap<String, TensorData>, name: &str) -> Vec<f32> {
    let tensor = tensors
        .get(name)
        .unwrap_or_else(|| panic!("assistant tensor missing: {}", name));
    decode_tensor_to_f32(tensor.dtype, &tensor.data)
}

fn tensor_i64(tensors: &HashMap<String, TensorData>, name: &str) -> Vec<i64> {
    let tensor = tensors
        .get(name)
        .unwrap_or_else(|| panic!("assistant tensor missing: {}", name));
    assert_eq!(
        tensor.dtype,
        safetensors::Dtype::I64,
        "assistant tensor {} must be I64",
        name
    );
    tensor
        .data
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .collect()
}

fn decode_tensor_to_f32(dtype: safetensors::Dtype, raw_data: &[u8]) -> Vec<f32> {
    match dtype {
        safetensors::Dtype::F32 => raw_data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        safetensors::Dtype::F16 => raw_data
            .chunks_exact(2)
            .map(|b| half_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        safetensors::Dtype::BF16 => raw_data
            .chunks_exact(2)
            .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        safetensors::Dtype::I64 => raw_data
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
            .collect(),
        _ => panic!("Unsupported assistant tensor dtype: {:?}", dtype),
    }
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut e = 1u32;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let f_exp = 127 - 15 + 1 - e;
        let f_mant = (m & 0x3FF) << 13;
        f32::from_bits((sign << 31) | (f_exp << 23) | f_mant)
    } else if exp == 31 {
        f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
    } else {
        let f_exp = (exp as i32 - 15 + 127) as u32;
        let f_mant = mant << 13;
        f32::from_bits((sign << 31) | (f_exp << 23) | f_mant)
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
