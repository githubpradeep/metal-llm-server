use metal::*;
use safetensors::SafeTensors;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::gemma4_config::{KvCacheType, RopeParameters};
use crate::gemma4_gpu_model::Gemma4GpuModel;
use crate::gpu::MetalContext;
use crate::sampling;

pub struct Gemma4MtpAssistant {
    ctx: MetalContext,
    config: AssistantTextConfig,
    backbone_hidden_size: usize,
    token_to_ordered_row: Vec<usize>,
    lm_head_buf: Buffer,
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
    logits_buf: Buffer,
    projected_activation_buf: Buffer,
    per_layer_cos_bufs: Vec<Buffer>,
    per_layer_sin_bufs: Vec<Buffer>,
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
    text_config: AssistantTextConfig,
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
    pub fn new(model_dir: &str, target: &Gemma4GpuModel) -> Self {
        let config_path = Path::new(model_dir).join("config.json");
        let config_str = fs::read_to_string(&config_path).expect("Failed to read assistant config");
        let outer: AssistantOuterConfig =
            serde_json::from_str(&config_str).expect("Failed to parse assistant config");
        let config = outer.text_config;
        let backbone_hidden_size = outer.backbone_hidden_size;

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

        let ctx = MetalContext::new();
        let tensors = load_assistant_tensors(model_dir);

        let embed_tokens_f16 = raw_tensor_u16(&tensors, "model.embed_tokens.weight");
        let ordered_row_to_token = tensor_i64(&tensors, "masked_embedding.token_ordering")
            .into_iter()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        assert_eq!(
            ordered_row_to_token.len(),
            config.vocab_size,
            "assistant token_ordering must match vocab size"
        );
        let mut token_to_ordered_row = vec![0usize; ordered_row_to_token.len()];
        for (ordered_row, &token_id) in ordered_row_to_token.iter().enumerate() {
            token_to_ordered_row[token_id] = ordered_row;
        }
        let embed_tokens_f32 = embed_tokens_f16
            .iter()
            .map(|&bits| bf16_to_f32(bits))
            .collect::<Vec<_>>();
        let lm_head_buf = ctx.buffer_from_f32_as_f16(&embed_tokens_f32);

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
        let logits_buf = ctx.buffer_empty(config.vocab_size);
        let projected_activation_buf = ctx.buffer_empty(backbone_hidden_size);

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
        println!("  Ordered embeddings: true");
        if let (Some(sliding), Some(full)) = (
            target.mtp_kv_source_layer(false),
            target.mtp_kv_source_layer(true),
        ) {
            println!("  Shared KV sources: sliding_attention=layer {}, full_attention=layer {}", sliding, full);
        }

        Self {
            ctx,
            config,
            backbone_hidden_size,
            token_to_ordered_row,
            lm_head_buf,
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
            logits_buf,
            projected_activation_buf,
            per_layer_cos_bufs,
            per_layer_sin_bufs,
        }
    }

    pub fn draft_next(
        &mut self,
        token_id: usize,
        target_activation: &[f32],
        target: &Gemma4GpuModel,
    ) -> Result<MtpDraft, String> {
        if target_activation.len() != self.backbone_hidden_size {
            return Err(format!(
                "target activation has {} values, expected {}",
                target_activation.len(),
                self.backbone_hidden_size
            ));
        }
        if target.kv_seq_len == 0 {
            return Err("target KV cache is empty; prefill or decode one token first".to_string());
        }

        self.prepare_activation(token_id, target_activation, target)?;
        self.prepare_rotary(target.total_tokens.saturating_sub(1));

        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let num_kv_groups = (num_heads / num_kv_heads) as u32;
        let eps = self.config.rms_norm_eps as f32;

        let cmd = self.ctx.queue.new_command_buffer();
        {
            let encoder = cmd.new_compute_command_encoder();
            self.ctx.encode_matvec_f16(
                encoder,
                &self.pre_projection,
                &self.activation_buf,
                &self.hidden_buf,
                hidden_size as u32,
                (self.backbone_hidden_size * 2) as u32,
            );
            encoder.end_encoding();
        }

        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            let encoder = cmd.new_compute_command_encoder();
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

            self.ctx.encode_copy(
                encoder,
                &self.hidden_buf,
                &self.residual_buf,
                hidden_size as u32,
            );
            self.ctx.encode_rmsnorm(
                encoder,
                &self.hidden_buf,
                &layer.input_layernorm_weight,
                &self.normed_buf,
                hidden_size as u32,
                eps,
            );
            self.ctx.encode_matvec_f16(
                encoder,
                &layer.q_proj,
                &self.normed_buf,
                &self.q_buf,
                q_out as u32,
                hidden_size as u32,
            );
            self.ctx.encode_rmsnorm_per_head(
                encoder,
                &self.q_buf,
                &layer.q_norm_weight,
                &self.q_rotary_buf,
                num_heads as u32,
                head_dim as u32,
                eps,
            );
            self.ctx.encode_rotary(
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
                    self.ctx.encode_attention_with_offset_f16(
                        encoder,
                        &self.q_buf,
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
                    self.ctx.encode_attention_with_offset_q8_0(
                        encoder,
                        &self.q_buf,
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
                    self.ctx.encode_attention_with_offset_q4_0(
                        encoder,
                        &self.q_buf,
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

            self.ctx.encode_matvec_f16(
                encoder,
                &layer.o_proj,
                &self.attn_out_buf,
                &self.o_out_buf,
                hidden_size as u32,
                q_out as u32,
            );
            self.ctx.encode_rmsnorm(
                encoder,
                &self.o_out_buf,
                &layer.post_attention_layernorm_weight,
                &self.normed_buf,
                hidden_size as u32,
                eps,
            );
            self.ctx.encode_vec_add(
                encoder,
                &self.residual_buf,
                &self.normed_buf,
                &self.hidden_buf,
                hidden_size as u32,
            );

            self.ctx.encode_copy(
                encoder,
                &self.hidden_buf,
                &self.residual_buf,
                hidden_size as u32,
            );
            self.ctx.encode_rmsnorm(
                encoder,
                &self.hidden_buf,
                &layer.pre_feedforward_layernorm_weight,
                &self.normed_buf,
                hidden_size as u32,
                eps,
            );
            self.ctx.encode_matvec_f16(
                encoder,
                &layer.gate_proj,
                &self.normed_buf,
                &self.gate_buf,
                intermediate_size as u32,
                hidden_size as u32,
            );
            self.ctx.encode_matvec_f16(
                encoder,
                &layer.up_proj,
                &self.normed_buf,
                &self.up_buf,
                intermediate_size as u32,
                hidden_size as u32,
            );
            self.ctx.encode_gelu_mul(
                encoder,
                &self.gate_buf,
                &self.up_buf,
                &self.gelu_buf,
                intermediate_size as u32,
            );
            self.ctx.encode_matvec_f16(
                encoder,
                &layer.down_proj,
                &self.gelu_buf,
                &self.o_out_buf,
                hidden_size as u32,
                intermediate_size as u32,
            );
            self.ctx.encode_rmsnorm(
                encoder,
                &self.o_out_buf,
                &layer.post_feedforward_layernorm_weight,
                &self.normed_buf,
                hidden_size as u32,
                eps,
            );
            self.ctx.encode_vec_add(
                encoder,
                &self.residual_buf,
                &self.normed_buf,
                &self.hidden_buf,
                hidden_size as u32,
            );
            self.ctx.encode_vec_scale(
                encoder,
                &self.hidden_buf,
                &self.residual_buf,
                hidden_size as u32,
                layer.layer_scalar,
            );
            self.ctx.encode_copy(
                encoder,
                &self.residual_buf,
                &self.hidden_buf,
                hidden_size as u32,
            );

            encoder.end_encoding();
        }

        {
            let encoder = cmd.new_compute_command_encoder();
            self.ctx.encode_rmsnorm(
                encoder,
                &self.hidden_buf,
                &self.final_norm_weight,
                &self.normed_buf,
                hidden_size as u32,
                eps,
            );
            self.ctx.encode_matvec_f16(
                encoder,
                &self.lm_head_buf,
                &self.normed_buf,
                &self.logits_buf,
                self.config.vocab_size as u32,
                hidden_size as u32,
            );
            self.ctx.encode_matvec_f16(
                encoder,
                &self.post_projection,
                &self.normed_buf,
                &self.projected_activation_buf,
                self.backbone_hidden_size as u32,
                hidden_size as u32,
            );
            encoder.end_encoding();
        }
        cmd.commit();
        cmd.wait_until_completed();

        let mut logits = MetalContext::read_buffer(&self.logits_buf, self.config.vocab_size);
        if let Some(cap) = self.config.final_logit_softcapping {
            for logit in &mut logits {
                let x = (*logit / cap).clamp(-10.0, 10.0);
                *logit = cap * x.tanh();
            }
        }
        let projected_activation =
            MetalContext::read_buffer(&self.projected_activation_buf, self.backbone_hidden_size);
        let token_id = sampling::argmax(&logits);

        Ok(MtpDraft {
            token_id,
            projected_activation,
        })
    }

    fn prepare_activation(
        &self,
        token_id: usize,
        target_activation: &[f32],
        target: &Gemma4GpuModel,
    ) -> Result<(), String> {
        let mut activation = target.token_embedding(token_id)?;
        activation.extend_from_slice(target_activation);
        MetalContext::write_buffer(&self.activation_buf, &activation);
        Ok(())
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
