use metal::*;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use safetensors::SafeTensors;
use serde::Deserialize;

use crate::gemma4_config::Gemma4TextConfig;
use crate::gpu::MetalContext;

/// Gemma 4 E4B GPU-resident model with persistent KV cache on Metal.
/// All operations for one token are encoded into a SINGLE command buffer.
pub struct Gemma4GpuModel {
    pub ctx: MetalContext,
    pub config: Gemma4TextConfig,

    // Embedding tables (kept in CPU memory for lookup)
    pub embed_tokens: Vec<f32>,           // [vocab_size, hidden_size]
    pub embed_tokens_per_layer: Vec<f32>, // [vocab_size, num_layers * ple_dim]

    // Per-layer weights on GPU
    pub layers: Vec<Gemma4GpuLayer>,

    // Shared weights
    pub final_norm_weight: Buffer,
    pub per_layer_projection_norm_weight: Buffer,

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
    pub gelu_buf: Buffer,
    pub down_buf: Buffer,
    pub logits_buf: Buffer,

    // PLE scratch buffers
    pub ple_embed_buf: Buffer,    // [ple_dim] = 256
    pub ple_gated_buf: Buffer,    // [ple_dim]
    pub ple_normed_buf: Buffer,   // [ple_dim]
    pub ple_projected_buf: Buffer, // [hidden_size]

    // QK norm scratch
    pub q_normed_buf: Buffer,
    pub k_normed_buf: Buffer,

    // GPU-resident KV cache per layer
    pub k_cache: Vec<Buffer>,
    pub v_cache: Vec<Buffer>,
    pub kv_seq_len: u32,
    pub kv_capacity: u32,

    // Rotary precomputed buffers (per-layer since sliding/full differ)
    pub cos_buf: Buffer,
    pub sin_buf: Buffer,

    pub total_tokens: usize,
}

pub struct Gemma4GpuLayer {
    pub q_proj: Buffer,
    pub k_proj: Buffer,
    pub v_proj: Buffer,
    pub o_proj: Buffer,
    pub gate_proj: Buffer,
    pub up_proj: Buffer,
    pub down_proj: Buffer,

    // 4 norms per layer (Gemma-style)
    pub input_layernorm_weight: Buffer,
    pub post_attention_layernorm_weight: Buffer,
    pub pre_feedforward_layernorm_weight: Buffer,
    pub post_feedforward_layernorm_weight: Buffer,

    // PLE weights
    pub post_per_layer_input_norm_weight: Buffer,
    pub per_layer_input_gate_weight: Buffer, // Q4: [ple_dim, hidden_size]
    pub per_layer_projection_weight: Buffer, // Q4: [hidden_size, ple_dim]
    pub layer_scalar: f32,

    // QK norm weights
    pub q_norm_weight: Buffer,
    pub k_norm_weight: Buffer,

    // Layer properties
    pub is_full_attention: bool,
    pub head_dim: usize,
    pub q_out_dim: usize,
    pub kv_out_dim: usize,
}

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

impl Gemma4GpuModel {
    /// Load model weights layer-by-layer to stay within 16GB RAM.
    /// Each shard is loaded, relevant tensors extracted and quantized, then dropped.
    pub fn new(model_dir: &str) -> Self {
        let config_path = Path::new(model_dir).join("config.json");
        let config_str = fs::read_to_string(&config_path)
            .expect("Failed to read config.json");

        // Parse the outer config which wraps text_config
        let outer: serde_json::Value = serde_json::from_str(&config_str)
            .expect("Failed to parse config.json");

        let text_config: Gemma4TextConfig = if let Some(tc) = outer.get("text_config") {
            serde_json::from_value(tc.clone()).expect("Failed to parse text_config")
        } else {
            // Flat config (text_config fields at top level)
            serde_json::from_str(&config_str).expect("Failed to parse config as Gemma4TextConfig")
        };

        let config = text_config;
        let ctx = MetalContext::new();

        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let num_layers = config.num_hidden_layers;
        let ple_dim = config.hidden_size_per_layer_input; // 256

        // Determine max head_dim across all layers for buffer allocation
        let max_head_dim = config.global_head_dim; // 512 (full attention layers)
        let max_q_out = num_heads * max_head_dim;
        let max_kv_out = num_kv_heads * max_head_dim;

        println!("  Gemma4 E4B: {} layers, hidden={}, heads={}, kv_heads={}",
                 num_layers, hidden_size, num_heads, num_kv_heads);
        println!("  Sliding head_dim={}, Full head_dim={}, PLE dim={}",
                 config.head_dim, config.global_head_dim, ple_dim);

        // Build shard file list
        let index_path = Path::new(model_dir).join("model.safetensors.index.json");
        let shard_files: Vec<String> = if index_path.exists() {
            let index_str = fs::read_to_string(&index_path).unwrap();
            let index: SafetensorsIndex = serde_json::from_str(&index_str).unwrap();
            let mut files: Vec<String> = index.weight_map.values().cloned().collect();
            files.sort();
            files.dedup();
            files
        } else {
            vec!["model.safetensors".to_string()]
        };

        // We'll use a helper that loads a shard and returns a HashMap of tensors
        let prefix = "model.language_model.";

        println!("  Loading embeddings...");

        // Load embed_tokens and embed_tokens_per_layer from shards
        let mut embed_tokens: Vec<f32> = Vec::new();
        let mut embed_tokens_per_layer: Vec<f32> = Vec::new();
        let mut final_norm_data: Vec<f32> = Vec::new();
        let mut per_layer_proj_norm_data: Vec<f32> = Vec::new();

        // Load global weights from shards
        for shard_file in &shard_files {
            let shard_path = Path::new(model_dir).join(shard_file);
            let data = fs::read(&shard_path)
                .unwrap_or_else(|_| panic!("Failed to read shard: {}", shard_file));
            let safetensors = SafeTensors::deserialize(&data)
                .expect("Failed to deserialize safetensors");

            for (name, tensor_view) in safetensors.tensors() {
                let clean_name = name.strip_prefix(prefix).unwrap_or(&name);

                if clean_name == "embed_tokens.weight" && embed_tokens.is_empty() {
                    embed_tokens = decode_tensor_to_f32(&tensor_view);
                    println!("    embed_tokens: {:?}", tensor_view.shape());
                } else if clean_name == "embed_tokens_per_layer.weight" && embed_tokens_per_layer.is_empty() {
                    embed_tokens_per_layer = decode_tensor_to_f32(&tensor_view);
                    println!("    embed_tokens_per_layer: {:?}", tensor_view.shape());
                } else if clean_name == "model.norm.weight" || clean_name == "norm.weight" {
                    if final_norm_data.is_empty() {
                        final_norm_data = decode_tensor_to_f32(&tensor_view);
                    }
                } else if clean_name == "model.per_layer_projection_norm.weight"
                    || clean_name == "per_layer_projection_norm.weight"
                {
                    if per_layer_proj_norm_data.is_empty() {
                        per_layer_proj_norm_data = decode_tensor_to_f32(&tensor_view);
                    }
                }
            }
            // Drop shard data here
        }

        assert!(!embed_tokens.is_empty(), "embed_tokens not found");
        assert!(!embed_tokens_per_layer.is_empty(), "embed_tokens_per_layer not found");

        let final_norm_weight = ctx.buffer_from_slice(&final_norm_data);
        let per_layer_projection_norm_weight = ctx.buffer_from_slice(&per_layer_proj_norm_data);

        // Now load layers one by one
        println!("  Loading layers (Q4_0 quantized)...");
        let mut layers = Vec::with_capacity(num_layers);

        for layer_idx in 0..num_layers {
            println!("    Layer {}/{}", layer_idx + 1, num_layers);
            let is_full = config.is_full_attention(layer_idx);
            let head_dim = config.layer_head_dim(layer_idx);
            let q_out = num_heads * head_dim;
            let kv_out = num_kv_heads * head_dim;

            // Load this layer's weights from the appropriate shard(s)
            let layer_prefix = format!("layers.{}", layer_idx);
            let mut layer_tensors: HashMap<String, Vec<f32>> = HashMap::new();

            for shard_file in &shard_files {
                let shard_path = Path::new(model_dir).join(shard_file);
                let data = fs::read(&shard_path)
                    .unwrap_or_else(|_| panic!("Failed to read shard: {}", shard_file));
                let safetensors = SafeTensors::deserialize(&data)
                    .expect("Failed to deserialize safetensors");

                for (name, tensor_view) in safetensors.tensors() {
                    let clean_name = name.strip_prefix(prefix).unwrap_or(&name);
                    if clean_name.starts_with(&layer_prefix) {
                        let short_name = clean_name.strip_prefix(&format!("{}.", layer_prefix)).unwrap_or(clean_name);
                        if !layer_tensors.contains_key(short_name) {
                            layer_tensors.insert(short_name.to_string(), decode_tensor_to_f32(&tensor_view));
                        }
                    }
                }
            }

            // Extract layer_scalar
            let layer_scalar = layer_tensors.get("layer_scalar")
                .map(|v| v[0])
                .unwrap_or(1.0);

            // Build GPU buffers for this layer
            let q_proj_data = layer_tensors.remove("self_attn.q_proj.weight").expect("q_proj missing");
            let k_proj_data = layer_tensors.remove("self_attn.k_proj.weight").expect("k_proj missing");
            let v_proj_data = layer_tensors.remove("self_attn.v_proj.weight").expect("v_proj missing");
            let o_proj_data = layer_tensors.remove("self_attn.o_proj.weight").expect("o_proj missing");
            let gate_proj_data = layer_tensors.remove("mlp.gate_proj.weight").expect("gate_proj missing");
            let up_proj_data = layer_tensors.remove("mlp.up_proj.weight").expect("up_proj missing");
            let down_proj_data = layer_tensors.remove("mlp.down_proj.weight").expect("down_proj missing");

            let input_ln = layer_tensors.remove("input_layernorm.weight").expect("input_layernorm missing");
            let post_attn_ln = layer_tensors.remove("post_attention_layernorm.weight").expect("post_attention_layernorm missing");
            let pre_ff_ln = layer_tensors.remove("pre_feedforward_layernorm.weight").expect("pre_feedforward_layernorm missing");
            let post_ff_ln = layer_tensors.remove("post_feedforward_layernorm.weight").expect("post_feedforward_layernorm missing");
            let post_ple_norm = layer_tensors.remove("post_per_layer_input_norm.weight").expect("post_per_layer_input_norm missing");

            let ple_gate_data = layer_tensors.remove("per_layer_input_gate.weight").expect("per_layer_input_gate missing");
            let ple_proj_data = layer_tensors.remove("per_layer_projection.weight").expect("per_layer_projection missing");

            let q_norm_data = layer_tensors.remove("self_attn.q_norm.weight").expect("q_norm missing");
            let k_norm_data = layer_tensors.remove("self_attn.k_norm.weight").expect("k_norm missing");

            let layer = Gemma4GpuLayer {
                q_proj: ctx.buffer_from_f32_as_q4(&q_proj_data, q_out, hidden_size),
                k_proj: ctx.buffer_from_f32_as_q4(&k_proj_data, kv_out, hidden_size),
                v_proj: ctx.buffer_from_f32_as_q4(&v_proj_data, kv_out, hidden_size),
                o_proj: ctx.buffer_from_f32_as_q4(&o_proj_data, hidden_size, q_out),
                gate_proj: ctx.buffer_from_f32_as_q4(&gate_proj_data, intermediate_size, hidden_size),
                up_proj: ctx.buffer_from_f32_as_q4(&up_proj_data, intermediate_size, hidden_size),
                down_proj: ctx.buffer_from_f32_as_q4(&down_proj_data, hidden_size, intermediate_size),

                input_layernorm_weight: ctx.buffer_from_slice(&input_ln),
                post_attention_layernorm_weight: ctx.buffer_from_slice(&post_attn_ln),
                pre_feedforward_layernorm_weight: ctx.buffer_from_slice(&pre_ff_ln),
                post_feedforward_layernorm_weight: ctx.buffer_from_slice(&post_ff_ln),
                post_per_layer_input_norm_weight: ctx.buffer_from_slice(&post_ple_norm),

                per_layer_input_gate_weight: ctx.buffer_from_f32_as_q4(&ple_gate_data, ple_dim, hidden_size),
                per_layer_projection_weight: ctx.buffer_from_f32_as_q4(&ple_proj_data, hidden_size, ple_dim),
                layer_scalar,

                q_norm_weight: ctx.buffer_from_slice(&q_norm_data),
                k_norm_weight: ctx.buffer_from_slice(&k_norm_data),

                is_full_attention: is_full,
                head_dim,
                q_out_dim: q_out,
                kv_out_dim: kv_out,
            };

            layers.push(layer);
            // layer_tensors dropped here, freeing memory
        }

        // Pre-allocate scratch buffers
        let hidden_buf = ctx.buffer_empty(hidden_size);
        let normed_buf = ctx.buffer_empty(hidden_size);
        let residual_buf = ctx.buffer_empty(hidden_size);
        let q_buf = ctx.buffer_empty(max_q_out);
        let k_buf = ctx.buffer_empty(max_kv_out);
        let v_buf = ctx.buffer_empty(max_kv_out);
        let attn_out_buf = ctx.buffer_empty(max_q_out);
        let o_out_buf = ctx.buffer_empty(hidden_size);
        let gate_buf = ctx.buffer_empty(intermediate_size);
        let up_buf = ctx.buffer_empty(intermediate_size);
        let gelu_buf = ctx.buffer_empty(intermediate_size);
        let down_buf = ctx.buffer_empty(hidden_size);
        let logits_buf = ctx.buffer_empty(vocab_size);

        // PLE scratch
        let ple_embed_buf = ctx.buffer_empty(ple_dim);
        let ple_gated_buf = ctx.buffer_empty(ple_dim);
        let ple_normed_buf = ctx.buffer_empty(ple_dim);
        let ple_projected_buf = ctx.buffer_empty(hidden_size);

        // QK norm scratch (max head_dim per head)
        let q_normed_buf = ctx.buffer_empty(max_q_out);
        let k_normed_buf = ctx.buffer_empty(max_kv_out);

        // KV cache: use sliding_window for sliding layers, full context for full layers
        // For simplicity, allocate max capacity (full context) for all layers
        let kv_capacity = config.max_position_embeddings.min(8192) as u32; // cap at 8K for memory
        let mut k_cache = Vec::with_capacity(num_layers);
        let mut v_cache = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let hd = config.layer_head_dim(i);
            k_cache.push(ctx.buffer_empty(num_kv_heads * kv_capacity as usize * hd));
            v_cache.push(ctx.buffer_empty(num_kv_heads * kv_capacity as usize * hd));
        }

        // Rotary buffers (allocate for max head_dim)
        let cos_buf = ctx.buffer_empty(max_head_dim);
        let sin_buf = ctx.buffer_empty(max_head_dim);

        println!("  Gemma4 model loaded successfully (Q4_0 quantized on Metal)");

        Gemma4GpuModel {
            ctx,
            config,
            embed_tokens,
            embed_tokens_per_layer,
            layers,
            final_norm_weight,
            per_layer_projection_norm_weight,
            hidden_buf,
            normed_buf,
            residual_buf,
            q_buf,
            k_buf,
            v_buf,
            attn_out_buf,
            o_out_buf,
            gate_buf,
            up_buf,
            gelu_buf,
            down_buf,
            logits_buf,
            ple_embed_buf,
            ple_gated_buf,
            ple_normed_buf,
            ple_projected_buf,
            q_normed_buf,
            k_normed_buf,
            k_cache,
            v_cache,
            kv_seq_len: 0,
            kv_capacity,
            cos_buf,
            sin_buf,
            total_tokens: 0,
        }
    }

    /// Forward one token through the entire model. ALL GPU work in a SINGLE command buffer.
    pub fn forward_single_token(&mut self, token_id: usize) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let num_kv_groups = (num_heads / num_kv_heads) as u32;
        let intermediate_size = self.config.intermediate_size;
        let vocab_size = self.config.vocab_size;
        let eps = self.config.rms_norm_eps as f32;
        let ple_dim = self.config.hidden_size_per_layer_input;
        let num_layers = self.config.num_hidden_layers;

        // Embed token (CPU)
        let embed_offset = token_id * hidden_size;
        let embed_slice = &self.embed_tokens[embed_offset..embed_offset + hidden_size];
        MetalContext::write_buffer(&self.hidden_buf, embed_slice);

        let kv_seq = self.kv_seq_len;

        // Precompute all rotary cos/sin per layer (CPU, trivial)
        let pos = self.total_tokens as f32;
        let mut rotary_cos_per_layer: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
        let mut rotary_sin_per_layer: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
        let mut rotary_dim_per_layer: Vec<usize> = Vec::with_capacity(num_layers);

        for layer_idx in 0..num_layers {
            let layer = &self.layers[layer_idx];
            let head_dim = layer.head_dim;
            let is_full = layer.is_full_attention;

            let rope_theta = if is_full {
                self.config.full_rope_theta()
            } else {
                self.config.sliding_rope_theta()
            };

            let rotary_dim = if is_full {
                (head_dim as f64 * self.config.full_partial_rotary_factor()) as usize
            } else {
                head_dim
            };

            let half_rot = rotary_dim / 2;
            let mut cos_data = vec![0.0f32; head_dim];
            let mut sin_data = vec![0.0f32; head_dim];
            for i in 0..half_rot {
                let freq = 1.0 / (rope_theta.powf(i as f64 * 2.0 / rotary_dim as f64) as f32);
                let angle = pos * freq;
                cos_data[i] = angle.cos();
                cos_data[i + half_rot] = angle.cos();
                sin_data[i] = angle.sin();
                sin_data[i + half_rot] = angle.sin();
            }
            if rotary_dim < head_dim {
                for i in rotary_dim..head_dim {
                    cos_data[i] = 1.0;
                }
            }

            rotary_cos_per_layer.push(cos_data);
            rotary_sin_per_layer.push(sin_data);
            rotary_dim_per_layer.push(rotary_dim);
        }

        // ═══ SINGLE COMMAND BUFFER FOR ENTIRE FORWARD PASS ═══
        let cmd = self.ctx.queue.new_command_buffer();

        for layer_idx in 0..num_layers {
            let layer = &self.layers[layer_idx];
            let head_dim = layer.head_dim;
            let q_out = layer.q_out_dim;
            let kv_out = layer.kv_out_dim;
            let is_full = layer.is_full_attention;
            let scale = 1.0 / (head_dim as f32).sqrt();
            let rotary_dim = rotary_dim_per_layer[layer_idx];

            // Write PLE embedding and rotary data before encoding this layer's GPU work
            let ple_offset = token_id * (num_layers * ple_dim) + layer_idx * ple_dim;
            let ple_slice = &self.embed_tokens_per_layer[ple_offset..ple_offset + ple_dim];
            MetalContext::write_buffer(&self.ple_embed_buf, ple_slice);
            MetalContext::write_buffer(&self.cos_buf, &rotary_cos_per_layer[layer_idx]);
            MetalContext::write_buffer(&self.sin_buf, &rotary_sin_per_layer[layer_idx]);

            let encoder = cmd.new_compute_command_encoder();

            // ─── Per-Layer Embedding (PLE) ───
            // Gate: ple_gated = per_layer_input_gate @ hidden_state → [ple_dim]
            self.ctx.encode_matvec_q4(encoder, &layer.per_layer_input_gate_weight,
                &self.hidden_buf, &self.ple_gated_buf, ple_dim as u32, hidden_size as u32);

            // Element-wise multiply: ple_gated = ple_embed * ple_gated
            self.ctx.encode_vec_mul(encoder, &self.ple_embed_buf, &self.ple_gated_buf,
                &self.ple_gated_buf, ple_dim as u32);

            // RMSNorm the gated PLE with per_layer_projection_norm
            self.ctx.encode_rmsnorm(encoder, &self.ple_gated_buf,
                &self.per_layer_projection_norm_weight, &self.ple_normed_buf,
                ple_dim as u32, eps);

            // Project: ple_projected = per_layer_projection @ ple_normed → [hidden_size]
            self.ctx.encode_matvec_q4(encoder, &layer.per_layer_projection_weight,
                &self.ple_normed_buf, &self.ple_projected_buf, hidden_size as u32, ple_dim as u32);

            // Post-PLE norm on projected output
            self.ctx.encode_rmsnorm(encoder, &self.ple_projected_buf,
                &layer.post_per_layer_input_norm_weight, &self.ple_projected_buf,
                hidden_size as u32, eps);

            // Add PLE to hidden: hidden = hidden + ple_projected
            self.ctx.encode_vec_add(encoder, &self.hidden_buf, &self.ple_projected_buf,
                &self.hidden_buf, hidden_size as u32);

            // ─── Attention Block ───
            // Save residual
            self.ctx.encode_copy(encoder, &self.hidden_buf, &self.residual_buf, hidden_size as u32);

            // Pre-attention norm
            self.ctx.encode_rmsnorm(encoder, &self.hidden_buf,
                &layer.input_layernorm_weight, &self.normed_buf, hidden_size as u32, eps);

            // Q, K, V projections
            self.ctx.encode_matvec_q4(encoder, &layer.q_proj, &self.normed_buf,
                &self.q_buf, q_out as u32, hidden_size as u32);
            self.ctx.encode_matvec_q4(encoder, &layer.k_proj, &self.normed_buf,
                &self.k_buf, kv_out as u32, hidden_size as u32);
            self.ctx.encode_matvec_q4(encoder, &layer.v_proj, &self.normed_buf,
                &self.v_buf, kv_out as u32, hidden_size as u32);

            // QK Norm: apply RMSNorm per-head to Q and K
            self.ctx.encode_rmsnorm_per_head(encoder, &self.q_buf, &layer.q_norm_weight,
                &self.q_normed_buf, num_heads as u32, head_dim as u32, eps);
            self.ctx.encode_rmsnorm_per_head(encoder, &self.k_buf, &layer.k_norm_weight,
                &self.k_normed_buf, num_kv_heads as u32, head_dim as u32, eps);

            // Apply rotary to Q and K
            if rotary_dim == head_dim {
                self.ctx.encode_rotary(encoder, &self.q_normed_buf, &self.k_normed_buf,
                    &self.cos_buf, &self.sin_buf, num_heads as u32, num_kv_heads as u32, head_dim as u32);
            } else {
                self.ctx.encode_rotary_partial(encoder, &self.q_normed_buf, &self.k_normed_buf,
                    &self.cos_buf, &self.sin_buf, num_heads as u32, num_kv_heads as u32,
                    head_dim as u32, rotary_dim as u32);
            }

            // Append K, V to cache
            self.ctx.encode_kv_append(encoder, &self.k_normed_buf, &self.k_cache[layer_idx],
                num_kv_heads as u32, head_dim as u32, self.kv_capacity, kv_seq);
            self.ctx.encode_kv_append(encoder, &self.v_buf, &self.v_cache[layer_idx],
                num_kv_heads as u32, head_dim as u32, self.kv_capacity, kv_seq);

            // Attention
            let attn_kv_seq = kv_seq + 1;

            // For sliding window layers, limit attention to last sliding_window tokens
            let effective_kv_seq = if !is_full {
                attn_kv_seq.min(self.config.sliding_window as u32)
            } else {
                attn_kv_seq
            };

            // For sliding window, adjust the start position in the cache
            let kv_start = if !is_full && attn_kv_seq > self.config.sliding_window as u32 {
                attn_kv_seq - self.config.sliding_window as u32
            } else {
                0u32
            };

            self.ctx.encode_attention_with_offset(encoder,
                &self.q_normed_buf, &self.k_cache[layer_idx], &self.v_cache[layer_idx],
                &self.attn_out_buf,
                num_heads as u32, num_kv_heads as u32, num_kv_groups,
                head_dim as u32, effective_kv_seq, self.kv_capacity, scale, kv_start);

            // O projection
            self.ctx.encode_matvec_q4(encoder, &layer.o_proj, &self.attn_out_buf,
                &self.o_out_buf, hidden_size as u32, q_out as u32);

            // Post-attention norm
            self.ctx.encode_rmsnorm(encoder, &self.o_out_buf,
                &layer.post_attention_layernorm_weight, &self.o_out_buf,
                hidden_size as u32, eps);

            // Residual add
            self.ctx.encode_vec_add(encoder, &self.residual_buf, &self.o_out_buf,
                &self.hidden_buf, hidden_size as u32);

            // ─── MLP Block ───
            // Save residual
            self.ctx.encode_copy(encoder, &self.hidden_buf, &self.residual_buf, hidden_size as u32);

            // Pre-feedforward norm
            self.ctx.encode_rmsnorm(encoder, &self.hidden_buf,
                &layer.pre_feedforward_layernorm_weight, &self.normed_buf,
                hidden_size as u32, eps);

            // MLP: gate_proj, up_proj, GeLU activation, down_proj
            self.ctx.encode_matvec_q4(encoder, &layer.gate_proj, &self.normed_buf,
                &self.gate_buf, intermediate_size as u32, hidden_size as u32);
            self.ctx.encode_matvec_q4(encoder, &layer.up_proj, &self.normed_buf,
                &self.up_buf, intermediate_size as u32, hidden_size as u32);
            self.ctx.encode_gelu_mul(encoder, &self.gate_buf, &self.up_buf,
                &self.gelu_buf, intermediate_size as u32);
            self.ctx.encode_matvec_q4(encoder, &layer.down_proj, &self.gelu_buf,
                &self.down_buf, hidden_size as u32, intermediate_size as u32);

            // Post-feedforward norm
            self.ctx.encode_rmsnorm(encoder, &self.down_buf,
                &layer.post_feedforward_layernorm_weight, &self.down_buf,
                hidden_size as u32, eps);

            // Residual add + layer_scalar: hidden = residual + layer_scalar * down
            self.ctx.encode_vec_add_scaled(encoder, &self.residual_buf, &self.down_buf,
                &self.hidden_buf, hidden_size as u32, layer.layer_scalar);

            encoder.end_encoding();
        }

        // Final norm (separate encoder)
        let encoder = cmd.new_compute_command_encoder();
        self.ctx.encode_rmsnorm(encoder, &self.hidden_buf,
            &self.final_norm_weight, &self.normed_buf, hidden_size as u32, eps);
        encoder.end_encoding();

        // Submit and wait
        cmd.commit();
        cmd.wait_until_completed();

        // Read normed hidden state and compute logits on CPU (tied embeddings)
        let normed = MetalContext::read_buffer(&self.normed_buf, hidden_size);
        let mut logits = vec![0.0f32; vocab_size];
        // logits = embed_tokens @ normed (embed_tokens is [vocab_size, hidden_size])
        for v in 0..vocab_size {
            let offset = v * hidden_size;
            let mut dot = 0.0f32;
            for d in 0..hidden_size {
                dot += self.embed_tokens[offset + d] * normed[d];
            }
            logits[v] = dot;
        }

        // Logit softcapping: logits = cap * tanh(logits / cap)
        let cap = self.config.final_logit_softcapping;
        for l in logits.iter_mut() {
            *l = cap * (*l / cap).tanh();
        }

        // Update state
        self.total_tokens += 1;
        self.kv_seq_len += 1;

        logits
    }

    pub fn num_items(&self) -> usize {
        self.total_tokens
    }

    /// Batched prefill: process all prompt tokens sequentially.
    pub fn forward_prefill(&mut self, token_ids: &[usize]) -> Vec<f32> {
        let mut logits = Vec::new();
        for &tid in token_ids {
            logits = self.forward_single_token(tid);
        }
        logits
    }
}

/// Decode a safetensors tensor view to Vec<f32>, handling f32/f16/bf16.
fn decode_tensor_to_f32(tensor_view: &safetensors::tensor::TensorView) -> Vec<f32> {
    let dtype = tensor_view.dtype();
    let raw_data = tensor_view.data();

    match dtype {
        safetensors::Dtype::F32 => {
            raw_data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        }
        safetensors::Dtype::F16 => {
            raw_data
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half_to_f32(bits)
                })
                .collect()
        }
        safetensors::Dtype::BF16 => {
            raw_data
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    bf16_to_f32(bits)
                })
                .collect()
        }
        _ => panic!("Unsupported dtype: {:?}", dtype),
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
        let f_exp = (127 - 15 + 1 - e) as u32;
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
