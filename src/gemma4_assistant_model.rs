use metal::*;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use safetensors::SafeTensors;

use crate::gemma4_assistant_config::Gemma4AssistantConfig;
use crate::gemma4_config::KvCacheType;
use crate::gemma4_gpu_model::{bf16_to_f32, decode_tensor_to_f32, half_to_f32, raw_to_u16};
use crate::gpu::MetalContext;
use crate::sampling::argmax;

use crate::gemma4_gpu_model::MainModelKvView;

pub struct AssistantForwardOutput {
    pub logits: Vec<f32>,
    pub projected_hidden_state: Vec<f32>,
}

pub struct Gemma4AssistantGpuLayer {
    pub q_proj: Buffer,
    pub o_proj: Buffer,
    pub gate_proj: Buffer,
    pub up_proj: Buffer,
    pub down_proj: Buffer,

    pub input_layernorm_weight: Buffer,
    pub post_attention_layernorm_weight: Buffer,
    pub pre_feedforward_layernorm_weight: Buffer,
    pub post_feedforward_layernorm_weight: Buffer,

    pub q_norm_weight: Buffer,
    pub layer_scalar: f32,

    pub is_full_attention: bool,
    pub head_dim: usize,
    pub q_out_dim: usize,
}

pub struct Gemma4AssistantGpuModel {
    pub ctx: MetalContext,
    pub config: Gemma4AssistantConfig,

    // Embeddings / head (tied)
    pub embed_tokens_f16: Vec<u16>,
    pub lm_head_buf: Buffer,

    // Projections between backbone and assistant hidden size
    pub pre_projection: Buffer,  // [assistant_hidden, 2*backbone_hidden]
    pub post_projection: Buffer, // [backbone_hidden, assistant_hidden]

    pub final_norm_weight: Buffer,
    pub layers: Vec<Gemma4AssistantGpuLayer>,

    // Masked ordered embedding (optional)
    pub use_ordered_embeddings: bool,
    pub centroids_f32: Option<Vec<f32>>,     // [num_centroids, assistant_hidden]
    pub centroids_buf: Option<Buffer>,       // same centroids as f16 on GPU
    pub token_ordering: Option<Vec<usize>>,  // [vocab_size]

    // Ordered-embedding scratch buffers
    pub centroid_logits_buf: Buffer,         // [num_centroids]
    pub selected_indices_buf: Buffer,        // [top_k * vocab_per_centroid]
    pub selected_logits_buf: Buffer,         // [top_k * vocab_per_centroid]

    // Scratch buffers (single token)
    pub hidden_buf: Buffer,
    pub normed_buf: Buffer,
    pub residual_buf: Buffer,
    pub q_buf: Buffer,
    pub q_normed_buf: Buffer,
    pub attn_out_buf: Buffer,
    pub o_out_buf: Buffer,
    pub gate_buf: Buffer,
    pub up_buf: Buffer,
    pub gelu_buf: Buffer,
    pub down_buf: Buffer,
    pub logits_buf: Buffer,
    pub projected_hidden_buf: Buffer, // [backbone_hidden]
    pub pre_projection_input_buf: Buffer, // [2 * backbone_hidden]

    // Rotary buffers (single token, per layer)
    pub cos_buf: Buffer,
    pub sin_buf: Buffer,
}

impl Gemma4AssistantGpuModel {
    pub fn new(assistant_dir: &str) -> Self {
        let config_path = Path::new(assistant_dir).join("config.json");
        let config_str = fs::read_to_string(&config_path).expect("Failed to read assistant config.json");
        let config: Gemma4AssistantConfig =
            serde_json::from_str(&config_str).expect("Failed to parse assistant config.json");

        let model_path = Path::new(assistant_dir).join("model.safetensors");
        let file = fs::File::open(&model_path)
            .unwrap_or_else(|_| panic!("Failed to open assistant safetensors: {:?}", model_path));
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .unwrap_or_else(|_| panic!("Failed to mmap assistant safetensors: {:?}", model_path));
        let safetensors =
            SafeTensors::deserialize(&mmap).expect("Failed to deserialize assistant safetensors");

        let ctx = MetalContext::new();
        let text_config = &config.text_config;

        let hidden_size = text_config.hidden_size;
        let backbone_hidden = config.backbone_hidden_size;
        let vocab_size = text_config.vocab_size;
        let num_layers = text_config.num_hidden_layers;
        let intermediate_size = text_config.intermediate_size;
        let num_heads = text_config.num_attention_heads;

        println!("Loading Gemma4 assistant (MTP drafter) from: {}", assistant_dir);
        println!("  Assistant: {} layers, hidden={}, heads={}, kv_heads={}",
            num_layers, hidden_size, num_heads, text_config.num_key_value_heads);
        println!("  Backbone hidden size: {}", backbone_hidden);
        println!("  Ordered embeddings: {}", config.use_ordered_embeddings);

        // Load global weights
        let mut embed_tokens_f16: Vec<u16> = Vec::new();
        let mut pre_projection_data: Vec<f32> = Vec::new();
        let mut post_projection_data: Vec<f32> = Vec::new();
        let mut final_norm_data: Vec<f32> = Vec::new();
        let mut centroids_data: Vec<f32> = Vec::new();
        let mut token_ordering: Vec<usize> = Vec::new();

        for (name, tensor_view) in safetensors.tensors() {
            if name == "model.embed_tokens.weight" {
                embed_tokens_f16 = raw_to_u16(tensor_view.data());
                println!("    embed_tokens: {:?} (tied lm_head)", tensor_view.shape());
            } else if name == "pre_projection.weight" {
                pre_projection_data = decode_tensor_to_f32(&tensor_view);
                println!("    pre_projection: {:?}", tensor_view.shape());
            } else if name == "post_projection.weight" {
                post_projection_data = decode_tensor_to_f32(&tensor_view);
                println!("    post_projection: {:?}", tensor_view.shape());
            } else if name == "model.norm.weight" {
                final_norm_data = decode_tensor_to_f32(&tensor_view);
            } else if name == "masked_embedding.centroids.weight" {
                centroids_data = decode_tensor_to_f32(&tensor_view);
                println!("    masked_embedding centroids: {:?}", tensor_view.shape());
            } else if name == "masked_embedding.token_ordering" {
                token_ordering = decode_token_ordering(&tensor_view);
                println!("    masked_embedding token_ordering: {:?} (dtype {:?})", tensor_view.shape(), tensor_view.dtype());
            }
        }

        assert!(!embed_tokens_f16.is_empty(), "assistant embed_tokens not found");
        assert!(!pre_projection_data.is_empty(), "assistant pre_projection not found");
        assert!(!post_projection_data.is_empty(), "assistant post_projection not found");

        // LM head = tied embed_tokens, uploaded as f16 for GPU matvec
        let lm_head_f32: Vec<f32> = embed_tokens_f16.iter().map(|&b| bf16_to_f32(b)).collect();
        let lm_head_buf = ctx.buffer_from_f32_as_f16(&lm_head_f32);

        let pre_projection = ctx.buffer_from_f32_as_f16(&pre_projection_data);
        let post_projection = ctx.buffer_from_f32_as_f16(&post_projection_data);
        let final_norm_weight = ctx.buffer_from_slice(&final_norm_data);

        let (centroids_f32, centroids_buf, token_ordering) = if config.use_ordered_embeddings {
            assert!(
                !centroids_data.is_empty(),
                "masked_embedding.centroids.weight required when use_ordered_embeddings is true"
            );
            assert!(
                !token_ordering.is_empty(),
                "masked_embedding.token_ordering required when use_ordered_embeddings is true"
            );
            assert_eq!(
                token_ordering.len(),
                vocab_size,
                "token_ordering length ({}) must match vocab_size ({})",
                token_ordering.len(),
                vocab_size
            );
            (
                Some(centroids_data.clone()),
                Some(ctx.buffer_from_f32_as_f16(&centroids_data)),
                Some(token_ordering),
            )
        } else {
            (None, None, None)
        };

        // Load layers
        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let layer_prefix = format!("model.layers.{}", layer_idx);
            let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();

            for (name, tensor_view) in safetensors.tensors() {
                if let Some(rest) = name.strip_prefix(&format!("{}.", layer_prefix)) {
                    if !tensors.contains_key(rest) {
                        tensors.insert(rest.to_string(), decode_tensor_to_f32(&tensor_view));
                    }
                }
            }

            let is_full = text_config.is_full_attention(layer_idx);
            let head_dim = text_config.layer_head_dim(layer_idx);
            let q_out = num_heads * head_dim;

            let layer_scalar = tensors
                .get("layer_scalar")
                .map(|v| v[0])
                .unwrap_or(1.0);

            let layer = Gemma4AssistantGpuLayer {
                q_proj: ctx.buffer_from_f32_as_f16(
                    tensors.get("self_attn.q_proj.weight").expect("q_proj missing"),
                ),
                o_proj: ctx.buffer_from_f32_as_f16(
                    tensors.get("self_attn.o_proj.weight").expect("o_proj missing"),
                ),
                gate_proj: ctx.buffer_from_f32_as_f16(
                    tensors.get("mlp.gate_proj.weight").expect("gate_proj missing"),
                ),
                up_proj: ctx.buffer_from_f32_as_f16(
                    tensors.get("mlp.up_proj.weight").expect("up_proj missing"),
                ),
                down_proj: ctx.buffer_from_f32_as_f16(
                    tensors.get("mlp.down_proj.weight").expect("down_proj missing"),
                ),
                input_layernorm_weight: ctx.buffer_from_slice(
                    tensors.get("input_layernorm.weight").expect("input_layernorm missing"),
                ),
                post_attention_layernorm_weight: ctx.buffer_from_slice(
                    tensors
                        .get("post_attention_layernorm.weight")
                        .expect("post_attention_layernorm missing"),
                ),
                pre_feedforward_layernorm_weight: ctx.buffer_from_slice(
                    tensors
                        .get("pre_feedforward_layernorm.weight")
                        .expect("pre_feedforward_layernorm missing"),
                ),
                post_feedforward_layernorm_weight: ctx.buffer_from_slice(
                    tensors
                        .get("post_feedforward_layernorm.weight")
                        .expect("post_feedforward_layernorm missing"),
                ),
                q_norm_weight: ctx.buffer_from_slice(
                    tensors.get("self_attn.q_norm.weight").expect("q_norm missing"),
                ),
                layer_scalar,
                is_full_attention: is_full,
                head_dim,
                q_out_dim: q_out,
            };
            layers.push(layer);
        }

        // Scratch buffers
        let hidden_buf = ctx.buffer_empty(hidden_size);
        let normed_buf = ctx.buffer_empty(hidden_size);
        let residual_buf = ctx.buffer_empty(hidden_size);

        let (centroid_logits_buf, selected_indices_buf, selected_logits_buf) =
            if config.use_ordered_embeddings {
                let num_centroids = config.num_centroids;
                let top_k = config.centroid_intermediate_top_k;
                assert!(
                    num_centroids > 0,
                    "num_centroids must be positive when use_ordered_embeddings is true"
                );
                assert!(
                    top_k > 0,
                    "centroid_intermediate_top_k must be positive when use_ordered_embeddings is true"
                );
                assert_eq!(
                    vocab_size % num_centroids,
                    0,
                    "vocab_size ({}) must be divisible by num_centroids ({})",
                    vocab_size,
                    num_centroids
                );
                let vocab_per_centroid = vocab_size / num_centroids;
                let num_selected = top_k * vocab_per_centroid;
                (
                    ctx.buffer_empty(num_centroids),
                    ctx.buffer_empty_u32(num_selected),
                    ctx.buffer_empty(num_selected),
                )
            } else {
                (
                    ctx.buffer_empty(1),
                    ctx.buffer_empty_u32(1),
                    ctx.buffer_empty(1),
                )
            };

        let max_q_out = layers.iter().map(|l| l.q_out_dim).max().unwrap_or(0);
        let q_buf = ctx.buffer_empty(max_q_out);
        let q_normed_buf = ctx.buffer_empty(max_q_out);
        let attn_out_buf = ctx.buffer_empty(max_q_out);
        let o_out_buf = ctx.buffer_empty(hidden_size);
        let gate_buf = ctx.buffer_empty(intermediate_size);
        let up_buf = ctx.buffer_empty(intermediate_size);
        let gelu_buf = ctx.buffer_empty(intermediate_size);
        let down_buf = ctx.buffer_empty(hidden_size);
        let logits_buf = ctx.buffer_empty(vocab_size);
        let projected_hidden_buf = ctx.buffer_empty(backbone_hidden);
        let pre_projection_input_buf = ctx.buffer_empty(2 * backbone_hidden);

        let max_head_dim = layers.iter().map(|l| l.head_dim).max().unwrap_or(0);
        let cos_buf = ctx.buffer_empty(max_head_dim);
        let sin_buf = ctx.buffer_empty(max_head_dim);

        println!("  Assistant loaded successfully");

        Gemma4AssistantGpuModel {
            ctx,
            config: config.clone(),
            embed_tokens_f16,
            lm_head_buf,
            pre_projection,
            post_projection,
            final_norm_weight,
            layers,
            use_ordered_embeddings: config.use_ordered_embeddings,
            centroids_f32,
            centroids_buf,
            token_ordering,
            centroid_logits_buf,
            selected_indices_buf,
            selected_logits_buf,
            hidden_buf,
            normed_buf,
            residual_buf,
            q_buf,
            q_normed_buf,
            attn_out_buf,
            o_out_buf,
            gate_buf,
            up_buf,
            gelu_buf,
            down_buf,
            logits_buf,
            projected_hidden_buf,
            pre_projection_input_buf,
            cos_buf,
            sin_buf,
        }
    }

    /// Project the concatenation of [backbone_token_embedding, backbone_hidden_state]
    /// down to the assistant's hidden size. Done on CPU for simplicity.
    pub fn prepare_inputs_embeds(
        &self,
        token_id: usize,
        backbone_hidden_state: &[f32],
        main_embed_tokens_f16: &[u16],
        main_hidden_size: usize,
    ) -> Vec<f32> {
        let backbone = self.config.backbone_hidden_size;
        assert_eq!(backbone_hidden_state.len(), backbone);
        assert_eq!(main_hidden_size, backbone);

        // Look up the main model's token embedding and decode from bf16.
        // Gemma scales embeddings by sqrt(hidden_size), same as the main model.
        let embed_offset = token_id * main_hidden_size;
        let embed_scale = (main_hidden_size as f32).sqrt();
        let mut concat = vec![0.0f32; 2 * backbone];
        for i in 0..backbone {
            concat[i] = bf16_to_f32(main_embed_tokens_f16[embed_offset + i]) * embed_scale;
            concat[backbone + i] = backbone_hidden_state[i];
        }

        // Matvec with pre_projection: out[256] = W[256, 5120] @ concat[5120]
        // We keep the weights as f16 on the GPU, so upload concat and use GPU matvec.
        // For a single token the CPU fallback is also fine; use GPU for consistency.
        MetalContext::write_buffer(&self.pre_projection_input_buf, &concat);
        let cmd = self.ctx.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        self.ctx.encode_matvec_f16(
            encoder,
            &self.pre_projection,
            &self.pre_projection_input_buf,
            &self.hidden_buf,
            self.config.text_config.hidden_size as u32,
            (2 * backbone) as u32,
        );
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        MetalContext::read_buffer(&self.hidden_buf, self.config.text_config.hidden_size)
    }

    /// Run one assistant forward step using the main model's KV cache.
    ///
    /// The assistant has no K/V projections; it performs cross-attention where Q is
    /// produced from the assistant hidden state and K/V are read from the target/main
    /// model's shared KV cache. This matches the llama.cpp Gemma4 assistant graph.
    pub fn forward_step(
        &mut self,
        inputs_embeds: &[f32],
        position_id: u32,
        main_kv: &MainModelKvView,
    ) -> AssistantForwardOutput {
        let hidden_size = self.config.text_config.hidden_size;
        let intermediate_size = self.config.text_config.intermediate_size;
        let vocab_size = self.config.text_config.vocab_size;
        let num_heads = self.config.text_config.num_attention_heads;
        let num_kv_heads_main = main_kv.num_kv_heads as usize;
        assert!(
            num_heads % num_kv_heads_main == 0,
            "assistant num_heads ({}) must be divisible by main num_kv_heads ({})",
            num_heads,
            num_kv_heads_main
        );
        let num_kv_groups = (num_heads / num_kv_heads_main) as u32;
        let eps = self.config.text_config.rms_norm_eps as f32;
        let scale = 1.0f32;

        MetalContext::write_buffer(&self.hidden_buf, inputs_embeds);

        let cmd = self.ctx.queue.new_command_buffer();

        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            let head_dim = layer.head_dim;
            let q_out = layer.q_out_dim;
            let is_full = layer.is_full_attention;

            let encoder = cmd.new_compute_command_encoder();

            // Attention block
            self.ctx.encode_copy(encoder, &self.hidden_buf, &self.residual_buf, hidden_size as u32);
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
                &self.q_normed_buf,
                num_heads as u32,
                head_dim as u32,
                eps,
            );

            // Rotary for Q only (single position)
            let rotary_cos = compute_rotary_cos(head_dim, position_id, is_full, &self.config.text_config);
            let rotary_sin = compute_rotary_sin(head_dim, position_id, is_full, &self.config.text_config);
            MetalContext::write_buffer(&self.cos_buf, &rotary_cos);
            MetalContext::write_buffer(&self.sin_buf, &rotary_sin);

            self.ctx.encode_rotary(
                encoder,
                &self.q_normed_buf,
                &self.q_normed_buf, // dummy k buf, not used for k when num_kv_heads=0
                &self.cos_buf,
                &self.sin_buf,
                num_heads as u32,
                0, // no K/V heads in assistant, only Q
                head_dim as u32,
            );

            // Cross-attention using the main model's shared KV cache (one source per
            // attention type, shared by all assistant layers of that type).
            let (k_cache, v_cache) = main_kv.kv_pair(is_full);
            let expected_head_dim = main_kv.expected_head_dim(is_full) as usize;
            assert_eq!(
                head_dim, expected_head_dim,
                "assistant layer {} head_dim ({}) must match main KV source head_dim ({})",
                layer_idx, head_dim, expected_head_dim
            );
            let attn_kv_seq = main_kv.seq_len;
            let effective_kv_seq = if is_full {
                attn_kv_seq
            } else {
                attn_kv_seq.min(self.config.text_config.sliding_window as u32)
            };
            let kv_start = if !is_full && attn_kv_seq > self.config.text_config.sliding_window as u32 {
                attn_kv_seq - self.config.text_config.sliding_window as u32
            } else {
                0
            };

            match main_kv.kv_cache_type {
                KvCacheType::F16 => {
                    self.ctx.encode_attention_with_offset_f16(
                        encoder,
                        &self.q_normed_buf,
                        k_cache,
                        v_cache,
                        &self.attn_out_buf,
                        num_heads as u32,
                        main_kv.num_kv_heads,
                        num_kv_groups,
                        head_dim as u32,
                        effective_kv_seq,
                        main_kv.capacity,
                        scale,
                        kv_start,
                    );
                }
                KvCacheType::Q8_0 => {
                    let groups_per_row = (head_dim / 32) as u32;
                    let row_bytes = groups_per_row * 34;
                    self.ctx.encode_attention_with_offset_q8_0(
                        encoder,
                        &self.q_normed_buf,
                        k_cache,
                        v_cache,
                        &self.attn_out_buf,
                        num_heads as u32,
                        main_kv.num_kv_heads,
                        num_kv_groups,
                        head_dim as u32,
                        effective_kv_seq,
                        main_kv.capacity,
                        scale,
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
                        &self.q_normed_buf,
                        k_cache,
                        v_cache,
                        &self.attn_out_buf,
                        num_heads as u32,
                        main_kv.num_kv_heads,
                        num_kv_groups,
                        head_dim as u32,
                        effective_kv_seq,
                        main_kv.capacity,
                        scale,
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

            // MLP block
            self.ctx.encode_copy(encoder, &self.hidden_buf, &self.residual_buf, hidden_size as u32);
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
                &self.down_buf,
                hidden_size as u32,
                intermediate_size as u32,
            );

            self.ctx.encode_rmsnorm(
                encoder,
                &self.down_buf,
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

        // Final norm
        let encoder = cmd.new_compute_command_encoder();
        self.ctx.encode_rmsnorm(
            encoder,
            &self.hidden_buf,
            &self.final_norm_weight,
            &self.normed_buf,
            hidden_size as u32,
            eps,
        );

        // LM head
        if self.use_ordered_embeddings {
            // Compute centroid logits on the GPU, then pick top-k centroids on the
            // CPU (small vector) and gather/score only the tokens inside those
            // centroids on the GPU. This avoids the expensive CPU sparse matmul
            // and the full [vocab_size] GPU matvec.
            let num_centroids = self.config.num_centroids;
            let top_k = self.config.centroid_intermediate_top_k;
            let vocab_per_centroid = vocab_size / num_centroids;
            let num_selected = top_k * vocab_per_centroid;

            self.ctx.encode_matvec_f16(
                encoder,
                self.centroids_buf.as_ref().unwrap(),
                &self.normed_buf,
                &self.centroid_logits_buf,
                num_centroids as u32,
                hidden_size as u32,
            );
            encoder.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            // Read back centroid scores and pick the top-k centroids on the CPU.
            let centroid_logits =
                MetalContext::read_buffer(&self.centroid_logits_buf, num_centroids);
            let mut indexed: Vec<(usize, f32)> = centroid_logits
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, v))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top_centroids: Vec<usize> =
                indexed.iter().take(top_k).map(|(i, _)| *i).collect();

            // Build the list of canonical token positions whose embeddings we need.
            let ordering = self.token_ordering.as_ref().unwrap();
            let mut selected_indices = vec![0u32; num_selected];
            for (i, &centroid_idx) in top_centroids.iter().enumerate() {
                for offset in 0..vocab_per_centroid {
                    let canonical_pos = ordering[centroid_idx * vocab_per_centroid + offset];
                    selected_indices[i * vocab_per_centroid + offset] = canonical_pos as u32;
                }
            }
            MetalContext::write_u32_buffer(&self.selected_indices_buf, &selected_indices);

            // Second GPU pass: fill full logits with mask value, gather selected
            // logits, scatter them back, and compute the post-projection.
            let cmd2 = self.ctx.queue.new_command_buffer();
            let encoder2 = cmd2.new_compute_command_encoder();

            let mask_value = -1e9f32;
            self.ctx.encode_ordered_embedding_fill(
                encoder2,
                &self.logits_buf,
                vocab_size as u32,
                mask_value,
            );
            self.ctx.encode_ordered_embedding_gather_logits(
                encoder2,
                &self.lm_head_buf,
                &self.normed_buf,
                &self.selected_indices_buf,
                &self.selected_logits_buf,
                hidden_size as u32,
                num_selected as u32,
            );
            self.ctx.encode_ordered_embedding_scatter_logits(
                encoder2,
                &self.logits_buf,
                &self.selected_indices_buf,
                &self.selected_logits_buf,
                num_selected as u32,
            );
            self.ctx.encode_matvec_f16(
                encoder2,
                &self.post_projection,
                &self.normed_buf,
                &self.projected_hidden_buf,
                self.config.backbone_hidden_size as u32,
                hidden_size as u32,
            );

            encoder2.end_encoding();
            cmd2.commit();
            cmd2.wait_until_completed();

            let logits = MetalContext::read_buffer(&self.logits_buf, vocab_size);
            let projected_hidden_state =
                MetalContext::read_buffer(&self.projected_hidden_buf, self.config.backbone_hidden_size);

            return AssistantForwardOutput {
                logits,
                projected_hidden_state,
            };
        }

        self.ctx.encode_matvec_f16(
            encoder,
            &self.lm_head_buf,
            &self.normed_buf,
            &self.logits_buf,
            vocab_size as u32,
            hidden_size as u32,
        );

        // Post-projection into backbone hidden size
        self.ctx.encode_matvec_f16(
            encoder,
            &self.post_projection,
            &self.normed_buf,
            &self.projected_hidden_buf,
            self.config.backbone_hidden_size as u32,
            hidden_size as u32,
        );

        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let logits = MetalContext::read_buffer(&self.logits_buf, vocab_size);
        let projected_hidden_state =
            MetalContext::read_buffer(&self.projected_hidden_buf, self.config.backbone_hidden_size);

        AssistantForwardOutput {
            logits,
            projected_hidden_state,
        }
    }

    /// Draft multiple tokens autoregressively from a single starting position.
    ///
    /// All draft tokens are evaluated at the same `position_id` because the assistant
    /// cross-attends to the target's KV cache; this matches the shared-memory MTP
    /// path in llama.cpp where `is_mem_shared` is true.
    pub fn draft_tokens(
        &mut self,
        start_token: usize,
        start_hidden_state: &[f32],
        main_embed_tokens_f16: &[u16],
        main_hidden_size: usize,
        main_kv: &MainModelKvView,
        position_id: u32,
        max_tokens: usize,
        eos_token_ids: &[usize],
    ) -> Vec<(usize, Vec<f32>)> {
        assert_eq!(
            main_hidden_size,
            self.config.backbone_hidden_size,
            "main model hidden size ({}) must match assistant backbone_hidden_size ({})",
            main_hidden_size,
            self.config.backbone_hidden_size
        );

        let mut drafts = Vec::with_capacity(max_tokens);
        let mut last_token = start_token;
        let mut last_hidden = start_hidden_state.to_vec();
        let debug = Self::mtp_debug_enabled();

        for i in 0..max_tokens {
            let inputs = self.prepare_inputs_embeds(
                last_token,
                &last_hidden,
                main_embed_tokens_f16,
                main_hidden_size,
            );

            // DEBUG: save tensors on the very first draft call (files don't exist yet).
            if debug && i == 0 && !std::path::Path::new("/tmp/assist_debug_concat.bin").exists() {
                let backbone = self.config.backbone_hidden_size;
                let embed_offset = last_token * main_hidden_size;
                let embed_scale = (main_hidden_size as f32).sqrt();
                let mut concat = vec![0.0f32; 2 * backbone];
                for j in 0..backbone {
                    concat[j] = bf16_to_f32(main_embed_tokens_f16[embed_offset + j]) * embed_scale;
                    concat[backbone + j] = last_hidden[j];
                }
                Self::print_vec_stats("input_hidden", &last_hidden);
                Self::print_vec_stats("pre_projection_out", &inputs);
                Self::save_f32_slice("/tmp/assist_debug_concat.bin", &concat);
                Self::save_f32_slice("/tmp/assist_debug_inputs.bin", &inputs);
                Self::save_kv("/tmp/assist_debug_kv", main_kv);
            }

            let out = self.forward_step(&inputs, position_id, main_kv);

            if debug && i == 0 {
                Self::print_vec_stats("projected_hidden", &out.projected_hidden_state);
                Self::print_logits_stats(&out.logits);
                Self::save_f32_slice("/tmp/assist_debug_projected.bin", &out.projected_hidden_state);
                Self::save_f32_slice("/tmp/assist_debug_logits.bin", &out.logits);
            }

            let token = argmax(&out.logits);
            last_hidden = out.projected_hidden_state;
            last_token = token;
            drafts.push((token, out.logits));

            if eos_token_ids.contains(&token) {
                break;
            }
        }

        drafts
    }

    fn mtp_debug_enabled() -> bool {
        match std::env::var("MTP_DEBUG") {
            Ok(v) => !v.is_empty() && v != "0",
            Err(_) => false,
        }
    }

    fn print_vec_stats(name: &str, v: &[f32]) {
        let n = v.len();
        let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = v.iter().sum();
        let mean = sum / n as f32;
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nan_count = v.iter().filter(|&&x| x.is_nan()).count();
        eprintln!(
            "[assist-debug] {} len={} min={:.4} max={:.4} mean={:.4} norm={:.4} nan={} first={:?}",
            name,
            n,
            min,
            max,
            mean,
            norm,
            nan_count,
            &v[..n.min(5)]
        );
    }

    fn print_logits_stats(logits: &[f32]) {
        let n = logits.len();
        let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top5: Vec<(usize, f32)> = indexed.into_iter().take(5).collect();
        let nan_count = logits.iter().filter(|&&x| x.is_nan()).count();
        eprintln!(
            "[assist-debug] logits len={} min={:.4} max={:.4} nan={} top5={:?}",
            n, min, max, nan_count, top5
        );
    }

    fn save_f32_slice(path: &str, data: &[f32]) {
        let bytes: Vec<u8> = data.iter().flat_map(|&x| x.to_le_bytes()).collect();
        let _ = std::fs::write(path, bytes);
    }

    fn save_kv(prefix: &str, main_kv: &MainModelKvView<'_>) {
        use crate::gpu::MetalContext;
        let head_dim_sliding = 256usize;
        let head_dim_full = 512usize;
        let num_kv_heads = 2usize;
        let seq_len = main_kv.seq_len as usize;
        let capacity = main_kv.capacity as usize;

        // The main KV cache is stored as half-precision (f16). read_buffer treats
        // the memory as f32, so read it as raw u16 and convert manually.
        let k_sliding = Self::read_f16_buffer(main_kv.sliding_k, num_kv_heads * capacity * head_dim_sliding);
        let v_sliding = Self::read_f16_buffer(main_kv.sliding_v, num_kv_heads * capacity * head_dim_sliding);
        let k_full = Self::read_f16_buffer(main_kv.full_k, num_kv_heads * capacity * head_dim_full);
        let v_full = Self::read_f16_buffer(main_kv.full_v, num_kv_heads * capacity * head_dim_full);

        // Save as f32 little-endian with shape metadata prefix
        let mut out = Vec::new();
        out.extend_from_slice(&(seq_len as u32).to_le_bytes());
        out.extend_from_slice(&(capacity as u32).to_le_bytes());
        for v in &[k_sliding, v_sliding, k_full, v_full] {
            for &x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        let _ = std::fs::write(format!("{}.bin", prefix), out);
    }

    fn read_f16_buffer(buf: &metal::Buffer, num_elements: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const u16;
        let u16s = unsafe { std::slice::from_raw_parts(ptr, num_elements) };
        u16s.iter().map(|&bits| half_to_f32(bits)).collect()
    }
}

fn compute_rotary_cos(
    head_dim: usize,
    position: u32,
    is_full: bool,
    config: &crate::gemma4_config::Gemma4TextConfig,
) -> Vec<f32> {
    let rope_theta = if is_full {
        config.full_rope_theta()
    } else {
        config.sliding_rope_theta()
    };
    let rope_factor = if is_full {
        config.full_rope_factor()
    } else {
        config.sliding_rope_factor()
    };
    let rotary_dim = if is_full {
        (head_dim as f64 * config.full_partial_rotary_factor()) as usize
    } else {
        head_dim
    };
    let rope_angles = rotary_dim / 2;
    let half_dim = head_dim / 2;
    let pos = position as f32;

    let mut cos = vec![0.0f32; head_dim];
    for i in 0..rope_angles {
        let inv_freq = 1.0
            / (rope_theta.powf(i as f64 * 2.0 / head_dim as f64) as f32)
            / rope_factor as f32;
        let angle = pos * inv_freq;
        cos[i] = angle.cos();
        cos[i + half_dim] = angle.cos();
    }
    for i in rope_angles..half_dim {
        cos[i] = 1.0;
        cos[i + half_dim] = 1.0;
    }
    cos
}

fn compute_rotary_sin(
    head_dim: usize,
    position: u32,
    is_full: bool,
    config: &crate::gemma4_config::Gemma4TextConfig,
) -> Vec<f32> {
    let rope_theta = if is_full {
        config.full_rope_theta()
    } else {
        config.sliding_rope_theta()
    };
    let rope_factor = if is_full {
        config.full_rope_factor()
    } else {
        config.sliding_rope_factor()
    };
    let rotary_dim = if is_full {
        (head_dim as f64 * config.full_partial_rotary_factor()) as usize
    } else {
        head_dim
    };
    let rope_angles = rotary_dim / 2;
    let half_dim = head_dim / 2;
    let pos = position as f32;

    let mut sin = vec![0.0f32; head_dim];
    for i in 0..rope_angles {
        let inv_freq = 1.0
            / (rope_theta.powf(i as f64 * 2.0 / head_dim as f64) as f32)
            / rope_factor as f32;
        let angle = pos * inv_freq;
        sin[i] = angle.sin();
        sin[i + half_dim] = angle.sin();
    }
    sin
}

/// Decode the masked_embedding token_ordering tensor to a Vec<usize>.
/// The tensor is usually int64, but we also tolerate int32 for robustness.
fn decode_token_ordering(tensor_view: &safetensors::tensor::TensorView) -> Vec<usize> {
    let data = tensor_view.data();
    match tensor_view.dtype() {
        safetensors::Dtype::I64 => data
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as usize)
            .collect(),
        safetensors::Dtype::I32 => data
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
            .collect(),
        safetensors::Dtype::U32 => data
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
            .collect(),
        _ => panic!(
            "unsupported dtype for masked_embedding.token_ordering: {:?}",
            tensor_view.dtype()
        ),
    }
}

fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    ((bits >> 16) & 0xFFFF) as u16
}
