use metal::*;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use safetensors::SafeTensors;
use serde::Deserialize;

use crate::gemma4_config::Gemma4TextConfig;
use crate::gpu::MetalContext;
use crate::kv_pool::{KvCachePool, KvPoolError, KvSlot, KvSlotView};

const DEFAULT_MAX_PREFILL_SEQ: usize = 128;
const DEFAULT_MAX_DECODE_BATCH: usize = 4;

fn configured_max_prefill_seq(kv_capacity: u32) -> usize {
    std::env::var("LLAMA_MAX_PREFILL_SEQ")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_MAX_PREFILL_SEQ)
        .min(kv_capacity as usize)
}

/// Gemma 4 E4B GPU-resident model with persistent KV cache on Metal.
/// All operations for one token are encoded into a SINGLE command buffer.
pub struct Gemma4GpuModel {
    pub ctx: MetalContext,
    pub config: Gemma4TextConfig,

    // Embedding table (Q4 quantized for memory efficiency)
    pub embed_tokens_f16: Vec<u16>, // [vocab_size * hidden_size] as bf16 (~1.3GB)
    // Per-layer embedding table (Q4 quantized, ~1.6GB)
    pub embed_tokens_per_layer_f16: Vec<u16>, // [vocab_size * num_layers * ple_dim] as bf16

    // LM head (tied to embed_tokens, stored as f16 Metal buffer for GPU matvec)
    pub lm_head_buf: Buffer,

    // Per-layer weights on GPU
    pub layers: Vec<Gemma4GpuLayer>,

    // Shared weights
    pub final_norm_weight: Buffer,
    pub per_layer_projection_norm_weight: Buffer,
    pub per_layer_model_projection_weight: Buffer, // [num_layers * ple_dim, hidden_size] f16

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
    pub prefill_scratch: PrefillScratch,
    pub decode_batch_scratch: DecodeBatchScratch,

    // PLE scratch buffers
    pub ple_embed_buf: Buffer, // [ple_dim] = 256 (unused now, kept for compat)
    pub ple_gated_buf: Buffer, // [ple_dim]
    pub ple_normed_buf: Buffer, // [ple_dim]
    pub ple_projected_buf: Buffer, // [hidden_size]
    pub ple_context_proj_buf: Buffer, // [num_layers * ple_dim] for context projection
    pub ple_token_id_buf: Buffer, // [num_layers * ple_dim] token identity embedding
    pub ple_combined_buf: Buffer, // [num_layers * ple_dim] combined output

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

    // Per-layer persistent buffers (reused every token, contents overwritten)
    pub per_layer_cos_bufs: Vec<Buffer>,
    pub per_layer_sin_bufs: Vec<Buffer>,
    pub per_layer_prefill_cos_bufs: Vec<Buffer>,
    pub per_layer_prefill_sin_bufs: Vec<Buffer>,
    pub per_layer_decode_batch_cos_bufs: Vec<Buffer>,
    pub per_layer_decode_batch_sin_bufs: Vec<Buffer>,
    pub per_layer_decode_batch_append_pos_bufs: Vec<Buffer>,
    pub per_layer_decode_batch_kv_start_bufs: Vec<Buffer>,
    pub per_layer_decode_batch_kv_len_bufs: Vec<Buffer>,
    pub per_layer_ple_bufs: Vec<Buffer>,

    pub total_tokens: usize,
}

pub struct PrefillScratch {
    pub max_seq_len: usize,
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
    pub ple_context_proj_buf: Buffer,
    pub ple_token_id_buf: Buffer,
    pub ple_combined_buf: Buffer,
    pub q_normed_buf: Buffer,
    pub k_normed_buf: Buffer,
}

pub struct DecodeBatchScratch {
    pub max_batch_size: usize,
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
    pub ple_context_proj_buf: Buffer,
    pub ple_token_id_buf: Buffer,
    pub ple_combined_buf: Buffer,
    pub q_normed_buf: Buffer,
    pub k_normed_buf: Buffer,
}

struct BatchedTokenInputs {
    batch_size: usize,
    hidden: Vec<f32>,
    ple_token_identity: Vec<f32>,
}

struct DecodeBatchRowOffsets {
    hidden: u64,
    q: u64,
    kv: u64,
    intermediate: u64,
    ple_row: u64,
}

struct PrefillRowOffsets {
    hidden: u64,
    q: u64,
    kv: u64,
    intermediate: u64,
    ple_row: u64,
}

#[derive(Clone, Copy)]
struct PrefillBatchSegment {
    slot: KvSlot,
    row_start: usize,
    token_count: usize,
    start_pos: usize,
}

impl DecodeBatchScratch {
    fn new(
        ctx: &MetalContext,
        max_batch_size: usize,
        hidden_size: usize,
        max_q_out: usize,
        max_kv_out: usize,
        intermediate_size: usize,
        vocab_size: usize,
        num_layers: usize,
        ple_dim: usize,
    ) -> Self {
        Self {
            max_batch_size,
            hidden_buf: ctx.buffer_empty(max_batch_size * hidden_size),
            normed_buf: ctx.buffer_empty(max_batch_size * hidden_size),
            residual_buf: ctx.buffer_empty(max_batch_size * hidden_size),
            q_buf: ctx.buffer_empty(max_batch_size * max_q_out),
            k_buf: ctx.buffer_empty(max_batch_size * max_kv_out),
            v_buf: ctx.buffer_empty(max_batch_size * max_kv_out),
            attn_out_buf: ctx.buffer_empty(max_batch_size * max_q_out),
            o_out_buf: ctx.buffer_empty(max_batch_size * hidden_size),
            gate_buf: ctx.buffer_empty(max_batch_size * intermediate_size),
            up_buf: ctx.buffer_empty(max_batch_size * intermediate_size),
            gelu_buf: ctx.buffer_empty(max_batch_size * intermediate_size),
            down_buf: ctx.buffer_empty(max_batch_size * hidden_size),
            logits_buf: ctx.buffer_empty(max_batch_size * vocab_size),
            ple_context_proj_buf: ctx.buffer_empty(max_batch_size * num_layers * ple_dim),
            ple_token_id_buf: ctx.buffer_empty(max_batch_size * num_layers * ple_dim),
            ple_combined_buf: ctx.buffer_empty(max_batch_size * num_layers * ple_dim),
            q_normed_buf: ctx.buffer_empty(max_batch_size * max_q_out),
            k_normed_buf: ctx.buffer_empty(max_batch_size * max_kv_out),
        }
    }
}

impl PrefillScratch {
    fn new(
        ctx: &MetalContext,
        max_seq_len: usize,
        hidden_size: usize,
        max_q_out: usize,
        max_kv_out: usize,
        intermediate_size: usize,
        vocab_size: usize,
        num_layers: usize,
        ple_dim: usize,
    ) -> Self {
        Self {
            max_seq_len,
            hidden_buf: ctx.buffer_empty(max_seq_len * hidden_size),
            normed_buf: ctx.buffer_empty(max_seq_len * hidden_size),
            residual_buf: ctx.buffer_empty(max_seq_len * hidden_size),
            q_buf: ctx.buffer_empty(max_seq_len * max_q_out),
            k_buf: ctx.buffer_empty(max_seq_len * max_kv_out),
            v_buf: ctx.buffer_empty(max_seq_len * max_kv_out),
            attn_out_buf: ctx.buffer_empty(max_seq_len * max_q_out),
            o_out_buf: ctx.buffer_empty(max_seq_len * hidden_size),
            gate_buf: ctx.buffer_empty(max_seq_len * intermediate_size),
            up_buf: ctx.buffer_empty(max_seq_len * intermediate_size),
            gelu_buf: ctx.buffer_empty(max_seq_len * intermediate_size),
            down_buf: ctx.buffer_empty(max_seq_len * hidden_size),
            logits_buf: ctx.buffer_empty(vocab_size),
            ple_context_proj_buf: ctx.buffer_empty(max_seq_len * num_layers * ple_dim),
            ple_token_id_buf: ctx.buffer_empty(max_seq_len * num_layers * ple_dim),
            ple_combined_buf: ctx.buffer_empty(max_seq_len * num_layers * ple_dim),
            q_normed_buf: ctx.buffer_empty(max_seq_len * max_q_out),
            k_normed_buf: ctx.buffer_empty(max_seq_len * max_kv_out),
        }
    }
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
    pub has_kv: bool,           // false for shared KV layers (layers 24-41)
    pub kv_source_layer: usize, // which layer's KV cache to use
    pub head_dim: usize,
    pub q_out_dim: usize,
    pub kv_out_dim: usize,
    pub use_f16: bool, // true = f16 weights (sensitive layers), false = Q4
}

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

impl Gemma4GpuModel {
    /// Load model weights. Uses a Q4 cache file for fast subsequent loads.
    /// First run: loads safetensors, quantizes, saves cache (~116s).
    /// Subsequent runs: loads pre-quantized cache directly (~5-10s).
    pub fn new(model_dir: &str) -> Self {
        let cache_path = Path::new(model_dir).join("model.q4cache");

        if cache_path.exists() {
            println!("  Found Q4 cache, loading pre-quantized weights...");
            return Self::load_from_cache(model_dir, &cache_path);
        }

        println!("  No Q4 cache found, quantizing from safetensors (one-time)...");
        let model = Self::load_from_safetensors(model_dir);

        // Save cache for next time
        println!("  Saving Q4 cache for fast future loads...");
        model.save_cache(&cache_path);

        model
    }

    fn load_from_safetensors(model_dir: &str) -> Self {
        let config_path = Path::new(model_dir).join("config.json");
        let config_str = fs::read_to_string(&config_path).expect("Failed to read config.json");

        // Parse the outer config which wraps text_config
        let outer: serde_json::Value =
            serde_json::from_str(&config_str).expect("Failed to parse config.json");

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

        println!(
            "  Gemma4 E4B: {} layers, hidden={}, heads={}, kv_heads={}",
            num_layers, hidden_size, num_heads, num_kv_heads
        );
        println!(
            "  Sliding head_dim={}, Full head_dim={}, PLE dim={}",
            config.head_dim, config.global_head_dim, ple_dim
        );

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

        println!("  Loading embeddings (f16, memory-efficient)...");

        // embed_tokens: [262144, 2560] in bf16 = 1.34 GB (keep as u16 vec)
        // embed_tokens_per_layer: [262144, 10752] in bf16 = 5.6 GB (keep as u16 vec)
        let mut embed_tokens_f16: Vec<u16> = Vec::new();
        let mut embed_tokens_per_layer_f16: Vec<u16> = Vec::new();
        let mut final_norm_data: Vec<f32> = Vec::new();
        let mut per_layer_proj_norm_data: Vec<f32> = Vec::new();
        let mut per_layer_model_proj_data: Vec<f32> = Vec::new();

        // Load global weights from shards — use memory mapping to avoid loading entire file
        for shard_file in &shard_files {
            let shard_path = Path::new(model_dir).join(shard_file);
            let file = fs::File::open(&shard_path)
                .unwrap_or_else(|_| panic!("Failed to open shard: {}", shard_file));
            let mmap = unsafe { memmap2::Mmap::map(&file) }
                .unwrap_or_else(|_| panic!("Failed to mmap shard: {}", shard_file));
            let safetensors =
                SafeTensors::deserialize(&mmap).expect("Failed to deserialize safetensors");

            for (name, tensor_view) in safetensors.tensors() {
                let clean_name = name.strip_prefix(prefix).unwrap_or(&name);

                if clean_name == "embed_tokens.weight" && embed_tokens_f16.is_empty() {
                    embed_tokens_f16 = raw_to_u16(tensor_view.data());
                    println!(
                        "    embed_tokens: {:?} (kept as f16, {:.1} MB)",
                        tensor_view.shape(),
                        embed_tokens_f16.len() * 2 / 1024 / 1024
                    );
                } else if clean_name == "embed_tokens_per_layer.weight"
                    && embed_tokens_per_layer_f16.is_empty()
                {
                    embed_tokens_per_layer_f16 = raw_to_u16(tensor_view.data());
                    println!(
                        "    embed_tokens_per_layer: {:?} (kept as f16, {:.1} MB)",
                        tensor_view.shape(),
                        embed_tokens_per_layer_f16.len() * 2 / 1024 / 1024
                    );
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
                } else if clean_name == "model.per_layer_model_projection.weight"
                    || clean_name == "per_layer_model_projection.weight"
                {
                    if per_layer_model_proj_data.is_empty() {
                        per_layer_model_proj_data = decode_tensor_to_f32(&tensor_view);
                        println!("    per_layer_model_projection: {:?}", tensor_view.shape());
                    }
                }
            }
            // mmap dropped here
        }

        assert!(!embed_tokens_f16.is_empty(), "embed_tokens not found");
        assert!(
            !embed_tokens_per_layer_f16.is_empty(),
            "embed_tokens_per_layer not found"
        );

        // Create GPU buffer for lm_head (tied embeddings — quantize to Q4_0)
        // Convert bf16 → f32 first, then quantize to Q4_0
        let lm_head_f32: Vec<f32> = embed_tokens_f16
            .iter()
            .map(|&bits| bf16_to_f32(bits))
            .collect();
        let lm_head_buf = ctx.buffer_from_f32_as_q4(&lm_head_f32, vocab_size, hidden_size);
        println!(
            "    lm_head (tied, Q4_0 on GPU): {:.1} MB",
            lm_head_buf.length() as f64 / 1024.0 / 1024.0
        );

        let final_norm_weight = ctx.buffer_from_slice(&final_norm_data);
        let per_layer_projection_norm_weight = ctx.buffer_from_slice(&per_layer_proj_norm_data);
        let per_layer_model_projection_weight = if !per_layer_model_proj_data.is_empty() {
            ctx.buffer_from_f32_as_q4(
                &per_layer_model_proj_data,
                num_layers * ple_dim,
                hidden_size,
            )
        } else {
            // Fallback: create empty buffer (shouldn't happen for E4B)
            println!(
                "  WARNING: per_layer_model_projection not found, PLE context projection disabled"
            );
            ctx.buffer_empty(1)
        };

        // Load all layers
        let num_layers_to_load = num_layers;
        println!(
            "  Loading layers (Q4_0 quantized, {} layers)...",
            num_layers_to_load
        );
        let mut layers = Vec::with_capacity(num_layers_to_load);

        for layer_idx in 0..num_layers_to_load {
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
                let file = fs::File::open(&shard_path)
                    .unwrap_or_else(|_| panic!("Failed to open shard: {}", shard_file));
                let mmap = unsafe { memmap2::Mmap::map(&file) }
                    .unwrap_or_else(|_| panic!("Failed to mmap shard: {}", shard_file));
                let safetensors =
                    SafeTensors::deserialize(&mmap).expect("Failed to deserialize safetensors");

                for (name, tensor_view) in safetensors.tensors() {
                    let clean_name = name.strip_prefix(prefix).unwrap_or(&name);
                    if clean_name.starts_with(&layer_prefix) {
                        let short_name = clean_name
                            .strip_prefix(&format!("{}.", layer_prefix))
                            .unwrap_or(clean_name);
                        if !layer_tensors.contains_key(short_name) {
                            layer_tensors
                                .insert(short_name.to_string(), decode_tensor_to_f32(&tensor_view));
                        }
                    }
                }
            }

            // Extract layer_scalar
            let layer_scalar = layer_tensors
                .get("layer_scalar")
                .map(|v| v[0])
                .unwrap_or(1.0);

            // Build GPU buffers for this layer
            let q_proj_data = layer_tensors
                .remove("self_attn.q_proj.weight")
                .expect("q_proj missing");
            let k_proj_data = layer_tensors
                .remove("self_attn.k_proj.weight")
                .expect("k_proj missing");
            let v_proj_data = layer_tensors
                .remove("self_attn.v_proj.weight")
                .expect("v_proj missing");
            let o_proj_data = layer_tensors
                .remove("self_attn.o_proj.weight")
                .expect("o_proj missing");
            let gate_proj_data = layer_tensors
                .remove("mlp.gate_proj.weight")
                .expect("gate_proj missing");
            let up_proj_data = layer_tensors
                .remove("mlp.up_proj.weight")
                .expect("up_proj missing");
            let down_proj_data = layer_tensors
                .remove("mlp.down_proj.weight")
                .expect("down_proj missing");

            let input_ln = layer_tensors
                .remove("input_layernorm.weight")
                .expect("input_layernorm missing");
            let post_attn_ln = layer_tensors
                .remove("post_attention_layernorm.weight")
                .expect("post_attention_layernorm missing");
            let pre_ff_ln = layer_tensors
                .remove("pre_feedforward_layernorm.weight")
                .expect("pre_feedforward_layernorm missing");
            let post_ff_ln = layer_tensors
                .remove("post_feedforward_layernorm.weight")
                .expect("post_feedforward_layernorm missing");
            let post_ple_norm = layer_tensors
                .remove("post_per_layer_input_norm.weight")
                .expect("post_per_layer_input_norm missing");

            let ple_gate_data = layer_tensors
                .remove("per_layer_input_gate.weight")
                .expect("per_layer_input_gate missing");
            let ple_proj_data = layer_tensors
                .remove("per_layer_projection.weight")
                .expect("per_layer_projection missing");

            let q_norm_data = layer_tensors
                .remove("self_attn.q_norm.weight")
                .expect("q_norm missing");
            let k_norm_data = layer_tensors
                .remove("self_attn.k_norm.weight")
                .expect("k_norm missing");

            // Mixed quantization: first 3 layers, last 3 layers, and attention o_proj use f16
            // Everything else uses Q4_0. This preserves quality on edge cases.
            let is_sensitive_layer = layer_idx < 3 || layer_idx >= (num_layers - 3);

            let layer = Gemma4GpuLayer {
                q_proj: if is_sensitive_layer {
                    ctx.buffer_from_f32_as_f16(&q_proj_data)
                } else {
                    ctx.buffer_from_f32_as_q4(&q_proj_data, q_out, hidden_size)
                },
                k_proj: if is_sensitive_layer {
                    ctx.buffer_from_f32_as_f16(&k_proj_data)
                } else {
                    ctx.buffer_from_f32_as_q4(&k_proj_data, kv_out, hidden_size)
                },
                v_proj: if is_sensitive_layer {
                    ctx.buffer_from_f32_as_f16(&v_proj_data)
                } else {
                    ctx.buffer_from_f32_as_q4(&v_proj_data, kv_out, hidden_size)
                },
                o_proj: ctx.buffer_from_f32_as_f16(&o_proj_data), // always f16 (quality-critical)
                gate_proj: if is_sensitive_layer {
                    ctx.buffer_from_f32_as_f16(&gate_proj_data)
                } else {
                    ctx.buffer_from_f32_as_q4(&gate_proj_data, intermediate_size, hidden_size)
                },
                up_proj: if is_sensitive_layer {
                    ctx.buffer_from_f32_as_f16(&up_proj_data)
                } else {
                    ctx.buffer_from_f32_as_q4(&up_proj_data, intermediate_size, hidden_size)
                },
                down_proj: if is_sensitive_layer {
                    ctx.buffer_from_f32_as_f16(&down_proj_data)
                } else {
                    ctx.buffer_from_f32_as_q4(&down_proj_data, hidden_size, intermediate_size)
                },

                input_layernorm_weight: ctx.buffer_from_slice(&input_ln),
                post_attention_layernorm_weight: ctx.buffer_from_slice(&post_attn_ln),
                pre_feedforward_layernorm_weight: ctx.buffer_from_slice(&pre_ff_ln),
                post_feedforward_layernorm_weight: ctx.buffer_from_slice(&post_ff_ln),
                post_per_layer_input_norm_weight: ctx.buffer_from_slice(&post_ple_norm),

                per_layer_input_gate_weight: ctx.buffer_from_f32_as_f16(&ple_gate_data), // always f16 (small, quality-critical)
                per_layer_projection_weight: ctx.buffer_from_f32_as_f16(&ple_proj_data), // always f16
                layer_scalar,

                q_norm_weight: ctx.buffer_from_slice(&q_norm_data),
                k_norm_weight: ctx.buffer_from_slice(&k_norm_data),

                is_full_attention: is_full,
                has_kv: layer_idx < (num_layers - config.num_kv_shared_layers),
                kv_source_layer: 0,
                head_dim,
                q_out_dim: q_out,
                kv_out_dim: kv_out,
                use_f16: is_sensitive_layer,
            };

            layers.push(layer);
            // layer_tensors dropped here, freeing memory
        }

        // Compute kv_source_layer for shared layers
        // For each shared layer, find the last non-shared layer of the same type
        let first_kv_shared = num_layers - config.num_kv_shared_layers;
        for i in first_kv_shared..num_layers {
            let layer_type = &config.layer_types[i];
            // Find the last non-shared layer with the same type
            let mut source = 0;
            for j in (0..first_kv_shared).rev() {
                if &config.layer_types[j] == layer_type {
                    source = j;
                    break;
                }
            }
            layers[i].kv_source_layer = source;
        }
        // Non-shared layers use their own index
        for i in 0..first_kv_shared {
            layers[i].kv_source_layer = i;
        }

        println!(
            "  KV sharing: layers 0-{} have own KV, layers {}-{} share",
            first_kv_shared - 1,
            first_kv_shared,
            num_layers - 1
        );

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
        let ple_context_proj_buf = ctx.buffer_empty(num_layers * ple_dim);
        let ple_token_id_buf = ctx.buffer_empty(num_layers * ple_dim);
        let ple_combined_buf = ctx.buffer_empty(num_layers * ple_dim);

        // QK norm scratch (max head_dim per head)
        let q_normed_buf = ctx.buffer_empty(max_q_out);
        let k_normed_buf = ctx.buffer_empty(max_kv_out);

        // KV cache: f16 precision to halve memory bandwidth
        let kv_capacity = config.max_position_embeddings.min(1024) as u32;
        let mut k_cache = Vec::with_capacity(num_layers);
        let mut v_cache = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let hd = config.layer_head_dim(i);
            let byte_len = (num_kv_heads * kv_capacity as usize * hd * 2) as u64; // f16 = 2 bytes
            k_cache.push(
                ctx.device
                    .new_buffer(byte_len, MTLResourceOptions::StorageModeShared),
            );
            v_cache.push(
                ctx.device
                    .new_buffer(byte_len, MTLResourceOptions::StorageModeShared),
            );
        }
        let max_prefill_seq = configured_max_prefill_seq(kv_capacity);
        let prefill_scratch = PrefillScratch::new(
            &ctx,
            max_prefill_seq,
            hidden_size,
            max_q_out,
            max_kv_out,
            intermediate_size,
            vocab_size,
            num_layers,
            ple_dim,
        );
        let decode_batch_scratch = DecodeBatchScratch::new(
            &ctx,
            DEFAULT_MAX_DECODE_BATCH,
            hidden_size,
            max_q_out,
            max_kv_out,
            intermediate_size,
            vocab_size,
            num_layers,
            ple_dim,
        );
        // Rotary buffers (allocate for max head_dim)
        let cos_buf = ctx.buffer_empty(max_head_dim);
        let sin_buf = ctx.buffer_empty(max_head_dim);

        // Per-layer persistent buffers for cos/sin/ple (allocated once, overwritten each token)
        let mut per_layer_cos_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_sin_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_prefill_cos_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_prefill_sin_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_decode_batch_cos_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_decode_batch_sin_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_decode_batch_append_pos_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_decode_batch_kv_start_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_decode_batch_kv_len_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_ple_bufs = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let hd = config.layer_head_dim(i);
            per_layer_cos_bufs.push(ctx.buffer_empty(hd));
            per_layer_sin_bufs.push(ctx.buffer_empty(hd));
            per_layer_prefill_cos_bufs.push(ctx.buffer_empty(max_prefill_seq * hd));
            per_layer_prefill_sin_bufs.push(ctx.buffer_empty(max_prefill_seq * hd));
            per_layer_decode_batch_cos_bufs.push(ctx.buffer_empty(DEFAULT_MAX_DECODE_BATCH * hd));
            per_layer_decode_batch_sin_bufs.push(ctx.buffer_empty(DEFAULT_MAX_DECODE_BATCH * hd));
            per_layer_decode_batch_append_pos_bufs
                .push(ctx.buffer_empty_u32(DEFAULT_MAX_DECODE_BATCH));
            per_layer_decode_batch_kv_start_bufs
                .push(ctx.buffer_empty_u32(DEFAULT_MAX_DECODE_BATCH));
            per_layer_decode_batch_kv_len_bufs.push(ctx.buffer_empty_u32(DEFAULT_MAX_DECODE_BATCH));
            per_layer_ple_bufs.push(ctx.buffer_empty(ple_dim));
        }

        println!(
            "  Parallel prefill scratch: max_seq={}",
            prefill_scratch.max_seq_len
        );
        println!(
            "  Decode batch scratch: max_batch={}",
            decode_batch_scratch.max_batch_size
        );
        println!("  Gemma4 model loaded successfully (Q4_0 quantized on Metal)");

        Gemma4GpuModel {
            ctx,
            config,
            embed_tokens_f16,
            embed_tokens_per_layer_f16,
            lm_head_buf,
            layers,
            final_norm_weight,
            per_layer_projection_norm_weight,
            per_layer_model_projection_weight,
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
            prefill_scratch,
            decode_batch_scratch,
            ple_embed_buf,
            ple_gated_buf,
            ple_normed_buf,
            ple_projected_buf,
            ple_context_proj_buf,
            ple_token_id_buf,
            ple_combined_buf,
            q_normed_buf,
            k_normed_buf,
            k_cache,
            v_cache,
            kv_seq_len: 0,
            kv_capacity,
            cos_buf,
            sin_buf,
            per_layer_cos_bufs,
            per_layer_sin_bufs,
            per_layer_prefill_cos_bufs,
            per_layer_prefill_sin_bufs,
            per_layer_decode_batch_cos_bufs,
            per_layer_decode_batch_sin_bufs,
            per_layer_decode_batch_append_pos_bufs,
            per_layer_decode_batch_kv_start_bufs,
            per_layer_decode_batch_kv_len_bufs,
            per_layer_ple_bufs,
            total_tokens: 0,
        }
    }

    /// Save all quantized weights to a binary cache file for fast loading.
    fn save_cache(&self, path: &Path) {
        use std::io::Write;
        let mut file = fs::File::create(path).expect("Failed to create cache file");
        let magic = b"GQ4C"; // Gemma Q4 Cache
        file.write_all(magic).unwrap();

        // Save embeddings (raw bf16)
        let embed_bytes = unsafe {
            std::slice::from_raw_parts(
                self.embed_tokens_f16.as_ptr() as *const u8,
                self.embed_tokens_f16.len() * 2,
            )
        };
        let len = embed_bytes.len() as u64;
        file.write_all(&len.to_le_bytes()).unwrap();
        file.write_all(embed_bytes).unwrap();

        // Save embed_tokens_per_layer (raw bf16)
        let ple_bytes = unsafe {
            std::slice::from_raw_parts(
                self.embed_tokens_per_layer_f16.as_ptr() as *const u8,
                self.embed_tokens_per_layer_f16.len() * 2,
            )
        };
        let len = ple_bytes.len() as u64;
        file.write_all(&len.to_le_bytes()).unwrap();
        file.write_all(ple_bytes).unwrap();

        // Save lm_head (Q4 on GPU)
        let lm_head_len = self.lm_head_buf.length() as u64;
        file.write_all(&lm_head_len.to_le_bytes()).unwrap();
        let lm_head_bytes = unsafe {
            std::slice::from_raw_parts(
                self.lm_head_buf.contents() as *const u8,
                lm_head_len as usize,
            )
        };
        file.write_all(lm_head_bytes).unwrap();

        // Save per_layer_model_projection (Q4 on GPU)
        let proj_len = self.per_layer_model_projection_weight.length() as u64;
        file.write_all(&proj_len.to_le_bytes()).unwrap();
        let proj_bytes = unsafe {
            std::slice::from_raw_parts(
                self.per_layer_model_projection_weight.contents() as *const u8,
                proj_len as usize,
            )
        };
        file.write_all(proj_bytes).unwrap();

        // Save norms
        let norm_len = self.final_norm_weight.length() as u64;
        file.write_all(&norm_len.to_le_bytes()).unwrap();
        let norm_bytes = unsafe {
            std::slice::from_raw_parts(
                self.final_norm_weight.contents() as *const u8,
                norm_len as usize,
            )
        };
        file.write_all(norm_bytes).unwrap();

        let pnorm_len = self.per_layer_projection_norm_weight.length() as u64;
        file.write_all(&pnorm_len.to_le_bytes()).unwrap();
        let pnorm_bytes = unsafe {
            std::slice::from_raw_parts(
                self.per_layer_projection_norm_weight.contents() as *const u8,
                pnorm_len as usize,
            )
        };
        file.write_all(pnorm_bytes).unwrap();

        // Save per-layer weights
        let num_layers = self.layers.len() as u32;
        file.write_all(&num_layers.to_le_bytes()).unwrap();
        for layer in &self.layers {
            // Save all GPU buffers as raw bytes
            let save_buf = |f: &mut fs::File, buf: &Buffer| {
                let len = buf.length() as u64;
                f.write_all(&len.to_le_bytes()).unwrap();
                let bytes = unsafe {
                    std::slice::from_raw_parts(buf.contents() as *const u8, len as usize)
                };
                f.write_all(bytes).unwrap();
            };
            save_buf(&mut file, &layer.q_proj);
            save_buf(&mut file, &layer.k_proj);
            save_buf(&mut file, &layer.v_proj);
            save_buf(&mut file, &layer.o_proj);
            save_buf(&mut file, &layer.gate_proj);
            save_buf(&mut file, &layer.up_proj);
            save_buf(&mut file, &layer.down_proj);
            save_buf(&mut file, &layer.input_layernorm_weight);
            save_buf(&mut file, &layer.post_attention_layernorm_weight);
            save_buf(&mut file, &layer.pre_feedforward_layernorm_weight);
            save_buf(&mut file, &layer.post_feedforward_layernorm_weight);
            save_buf(&mut file, &layer.post_per_layer_input_norm_weight);
            save_buf(&mut file, &layer.per_layer_input_gate_weight);
            save_buf(&mut file, &layer.per_layer_projection_weight);
            save_buf(&mut file, &layer.q_norm_weight);
            save_buf(&mut file, &layer.k_norm_weight);
            file.write_all(&layer.layer_scalar.to_le_bytes()).unwrap();
            file.write_all(&(layer.is_full_attention as u8).to_le_bytes())
                .unwrap();
            file.write_all(&(layer.has_kv as u8).to_le_bytes()).unwrap();
            file.write_all(&(layer.use_f16 as u8).to_le_bytes())
                .unwrap();
            file.write_all(&(layer.kv_source_layer as u32).to_le_bytes())
                .unwrap();
            file.write_all(&(layer.head_dim as u32).to_le_bytes())
                .unwrap();
            file.write_all(&(layer.q_out_dim as u32).to_le_bytes())
                .unwrap();
            file.write_all(&(layer.kv_out_dim as u32).to_le_bytes())
                .unwrap();
        }
        println!(
            "  Cache saved: {:.1} MB",
            file.metadata().map(|m| m.len()).unwrap_or(0) as f64 / 1024.0 / 1024.0
        );
    }

    /// Load model from pre-quantized cache file (fast path).
    fn load_from_cache(model_dir: &str, cache_path: &Path) -> Self {
        use std::io::Read;

        let config_path = Path::new(model_dir).join("config.json");
        let config_str = fs::read_to_string(&config_path).expect("Failed to read config.json");
        let outer: serde_json::Value =
            serde_json::from_str(&config_str).expect("Failed to parse config.json");
        let config: Gemma4TextConfig = if let Some(tc) = outer.get("text_config") {
            serde_json::from_value(tc.clone()).expect("Failed to parse text_config")
        } else {
            serde_json::from_str(&config_str).expect("Failed to parse config")
        };

        let ctx = MetalContext::new();
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let num_layers = config.num_hidden_layers;
        let ple_dim = config.hidden_size_per_layer_input;
        let max_head_dim = config.global_head_dim;
        let max_q_out = num_heads * max_head_dim;
        let max_kv_out = num_kv_heads * max_head_dim;

        let mut file = fs::File::open(cache_path).expect("Failed to open cache");
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).unwrap();
        assert_eq!(&magic, b"GQ4C", "Invalid cache magic");

        let read_len = |f: &mut fs::File| -> u64 {
            let mut buf = [0u8; 8];
            f.read_exact(&mut buf).unwrap();
            u64::from_le_bytes(buf)
        };

        let read_buf = |f: &mut fs::File, device: &Device| -> Buffer {
            let len = read_len(f);
            let mut data = vec![0u8; len as usize];
            f.read_exact(&mut data).unwrap();
            device.new_buffer_with_data(
                data.as_ptr() as *const _,
                len,
                MTLResourceOptions::StorageModeShared,
            )
        };

        // Load embeddings
        let embed_len = read_len(&mut file) as usize;
        let mut embed_bytes = vec![0u8; embed_len];
        file.read_exact(&mut embed_bytes).unwrap();
        let embed_tokens_f16: Vec<u16> = embed_bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        println!(
            "    embed_tokens: {:.1} MB",
            embed_len as f64 / 1024.0 / 1024.0
        );

        // Load embed_tokens_per_layer
        let ple_embed_len = read_len(&mut file) as usize;
        let mut ple_bytes = vec![0u8; ple_embed_len];
        file.read_exact(&mut ple_bytes).unwrap();
        let embed_tokens_per_layer_f16: Vec<u16> = ple_bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        println!(
            "    embed_tokens_per_layer: {:.1} MB",
            ple_embed_len as f64 / 1024.0 / 1024.0
        );

        // Load lm_head, projection, norms
        let lm_head_buf = read_buf(&mut file, &ctx.device);
        let per_layer_model_projection_weight = read_buf(&mut file, &ctx.device);
        let final_norm_weight = read_buf(&mut file, &ctx.device);
        let per_layer_projection_norm_weight = read_buf(&mut file, &ctx.device);

        // Load layers
        let mut nl_buf = [0u8; 4];
        file.read_exact(&mut nl_buf).unwrap();
        let num_layers_in_cache = u32::from_le_bytes(nl_buf) as usize;
        assert_eq!(num_layers_in_cache, num_layers);

        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let q_proj = read_buf(&mut file, &ctx.device);
            let k_proj = read_buf(&mut file, &ctx.device);
            let v_proj = read_buf(&mut file, &ctx.device);
            let o_proj = read_buf(&mut file, &ctx.device);
            let gate_proj = read_buf(&mut file, &ctx.device);
            let up_proj = read_buf(&mut file, &ctx.device);
            let down_proj = read_buf(&mut file, &ctx.device);
            let input_layernorm_weight = read_buf(&mut file, &ctx.device);
            let post_attention_layernorm_weight = read_buf(&mut file, &ctx.device);
            let pre_feedforward_layernorm_weight = read_buf(&mut file, &ctx.device);
            let post_feedforward_layernorm_weight = read_buf(&mut file, &ctx.device);
            let post_per_layer_input_norm_weight = read_buf(&mut file, &ctx.device);
            let per_layer_input_gate_weight = read_buf(&mut file, &ctx.device);
            let per_layer_projection_weight = read_buf(&mut file, &ctx.device);
            let q_norm_weight = read_buf(&mut file, &ctx.device);
            let k_norm_weight = read_buf(&mut file, &ctx.device);

            let mut scalar_buf = [0u8; 4];
            file.read_exact(&mut scalar_buf).unwrap();
            let layer_scalar = f32::from_le_bytes(scalar_buf);

            let mut meta = [0u8; 1];
            file.read_exact(&mut meta).unwrap();
            let is_full_attention = meta[0] != 0;
            file.read_exact(&mut meta).unwrap();
            let has_kv = meta[0] != 0;
            file.read_exact(&mut meta).unwrap();
            let use_f16 = meta[0] != 0;

            let mut u32_buf = [0u8; 4];
            file.read_exact(&mut u32_buf).unwrap();
            let kv_source_layer = u32::from_le_bytes(u32_buf) as usize;
            file.read_exact(&mut u32_buf).unwrap();
            let head_dim = u32::from_le_bytes(u32_buf) as usize;
            file.read_exact(&mut u32_buf).unwrap();
            let q_out_dim = u32::from_le_bytes(u32_buf) as usize;
            file.read_exact(&mut u32_buf).unwrap();
            let kv_out_dim = u32::from_le_bytes(u32_buf) as usize;

            if (layer_idx + 1) % 10 == 0 || layer_idx == num_layers - 1 {
                println!("    Loaded layer {}/{}", layer_idx + 1, num_layers);
            }

            layers.push(Gemma4GpuLayer {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                gate_proj,
                up_proj,
                down_proj,
                input_layernorm_weight,
                post_attention_layernorm_weight,
                pre_feedforward_layernorm_weight,
                post_feedforward_layernorm_weight,
                post_per_layer_input_norm_weight,
                per_layer_input_gate_weight,
                per_layer_projection_weight,
                layer_scalar,
                q_norm_weight,
                k_norm_weight,
                is_full_attention,
                has_kv,
                kv_source_layer,
                head_dim,
                q_out_dim,
                kv_out_dim,
                use_f16,
            });
        }

        // Allocate scratch buffers (same as load_from_safetensors)
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
        let ple_embed_buf = ctx.buffer_empty(ple_dim);
        let ple_gated_buf = ctx.buffer_empty(ple_dim);
        let ple_normed_buf = ctx.buffer_empty(ple_dim);
        let ple_projected_buf = ctx.buffer_empty(hidden_size);
        let ple_context_proj_buf = ctx.buffer_empty(num_layers * ple_dim);
        let ple_token_id_buf = ctx.buffer_empty(num_layers * ple_dim);
        let ple_combined_buf = ctx.buffer_empty(num_layers * ple_dim);
        let q_normed_buf = ctx.buffer_empty(max_q_out);
        let k_normed_buf = ctx.buffer_empty(max_kv_out);

        let kv_capacity = config.max_position_embeddings.min(1024) as u32;
        let mut k_cache = Vec::with_capacity(num_layers);
        let mut v_cache = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let hd = config.layer_head_dim(i);
            let byte_len = (num_kv_heads * kv_capacity as usize * hd * 2) as u64;
            k_cache.push(
                ctx.device
                    .new_buffer(byte_len, MTLResourceOptions::StorageModeShared),
            );
            v_cache.push(
                ctx.device
                    .new_buffer(byte_len, MTLResourceOptions::StorageModeShared),
            );
        }
        let max_prefill_seq = configured_max_prefill_seq(kv_capacity);
        let prefill_scratch = PrefillScratch::new(
            &ctx,
            max_prefill_seq,
            hidden_size,
            max_q_out,
            max_kv_out,
            intermediate_size,
            vocab_size,
            num_layers,
            ple_dim,
        );
        let decode_batch_scratch = DecodeBatchScratch::new(
            &ctx,
            DEFAULT_MAX_DECODE_BATCH,
            hidden_size,
            max_q_out,
            max_kv_out,
            intermediate_size,
            vocab_size,
            num_layers,
            ple_dim,
        );

        let cos_buf = ctx.buffer_empty(max_head_dim);
        let sin_buf = ctx.buffer_empty(max_head_dim);

        let mut per_layer_cos_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_sin_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_prefill_cos_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_prefill_sin_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_decode_batch_cos_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_decode_batch_sin_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_decode_batch_append_pos_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_decode_batch_kv_start_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_decode_batch_kv_len_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_ple_bufs = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let hd = config.layer_head_dim(i);
            per_layer_cos_bufs.push(ctx.buffer_empty(hd));
            per_layer_sin_bufs.push(ctx.buffer_empty(hd));
            per_layer_prefill_cos_bufs.push(ctx.buffer_empty(max_prefill_seq * hd));
            per_layer_prefill_sin_bufs.push(ctx.buffer_empty(max_prefill_seq * hd));
            per_layer_decode_batch_cos_bufs.push(ctx.buffer_empty(DEFAULT_MAX_DECODE_BATCH * hd));
            per_layer_decode_batch_sin_bufs.push(ctx.buffer_empty(DEFAULT_MAX_DECODE_BATCH * hd));
            per_layer_decode_batch_append_pos_bufs
                .push(ctx.buffer_empty_u32(DEFAULT_MAX_DECODE_BATCH));
            per_layer_decode_batch_kv_start_bufs
                .push(ctx.buffer_empty_u32(DEFAULT_MAX_DECODE_BATCH));
            per_layer_decode_batch_kv_len_bufs.push(ctx.buffer_empty_u32(DEFAULT_MAX_DECODE_BATCH));
            per_layer_ple_bufs.push(ctx.buffer_empty(ple_dim));
        }

        println!(
            "  Parallel prefill scratch: max_seq={}",
            prefill_scratch.max_seq_len
        );
        println!(
            "  Decode batch scratch: max_batch={}",
            decode_batch_scratch.max_batch_size
        );
        println!("  Loaded from Q4 cache successfully");

        Gemma4GpuModel {
            ctx,
            config,
            embed_tokens_f16,
            embed_tokens_per_layer_f16,
            lm_head_buf,
            layers,
            final_norm_weight,
            per_layer_projection_norm_weight,
            per_layer_model_projection_weight,
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
            prefill_scratch,
            decode_batch_scratch,
            ple_embed_buf,
            ple_gated_buf,
            ple_normed_buf,
            ple_projected_buf,
            ple_context_proj_buf,
            ple_token_id_buf,
            ple_combined_buf,
            q_normed_buf,
            k_normed_buf,
            k_cache,
            v_cache,
            kv_seq_len: 0,
            kv_capacity,
            cos_buf,
            sin_buf,
            per_layer_cos_bufs,
            per_layer_sin_bufs,
            per_layer_prefill_cos_bufs,
            per_layer_prefill_sin_bufs,
            per_layer_decode_batch_cos_bufs,
            per_layer_decode_batch_sin_bufs,
            per_layer_decode_batch_append_pos_bufs,
            per_layer_decode_batch_kv_start_bufs,
            per_layer_decode_batch_kv_len_bufs,
            per_layer_ple_bufs,
            total_tokens: 0,
        }
    }

    /// Forward one token through the entire model. One command buffer per layer.
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

        // Embed token (CPU — decode from f16 on the fly)
        // Gemma scales embeddings by sqrt(hidden_size)
        let embed_offset = token_id * hidden_size;
        let embed_scale = (hidden_size as f32).sqrt();
        let mut embed_f32 = vec![0.0f32; hidden_size];
        for i in 0..hidden_size {
            embed_f32[i] = bf16_to_f32(self.embed_tokens_f16[embed_offset + i]) * embed_scale;
        }
        MetalContext::write_buffer(&self.hidden_buf, &embed_f32);

        let kv_seq = self.kv_seq_len;

        // ─── PLE: Compute per-layer inputs on GPU ───
        // Reference: Gemma4TextModel.get_per_layer_inputs() + project_per_layer_inputs()
        //
        // 1. Token identity: embed_tokens_per_layer(token_id) * sqrt(ple_dim)
        // 2. Context projection: per_layer_model_projection(embed) * (1/sqrt(hidden_size))
        //    then per_layer_projection_norm (RMSNorm per layer)
        // 3. Combined: (context_proj + token_identity) * (1/sqrt(2))
        let ple_total_dim = num_layers * ple_dim;
        let ple_input_scale = std::f32::consts::FRAC_1_SQRT_2; // 1/sqrt(2)
        let context_proj_scale = 1.0 / (hidden_size as f32).sqrt();

        // Step 1: Token identity from embed_tokens_per_layer (CPU decode bf16, write to GPU)
        let ple_token_offset = token_id * ple_total_dim;
        let ple_scale = (ple_dim as f32).sqrt();
        let mut ple_token_identity = vec![0.0f32; ple_total_dim];
        if ple_token_offset + ple_total_dim <= self.embed_tokens_per_layer_f16.len() {
            for i in 0..ple_total_dim {
                ple_token_identity[i] =
                    bf16_to_f32(self.embed_tokens_per_layer_f16[ple_token_offset + i]) * ple_scale;
            }
        }
        MetalContext::write_buffer(&self.ple_token_id_buf, &ple_token_identity);

        // Steps 2-3: GPU computation (will be part of main command buffer)
        // We'll encode PLE ops at the start of the main command buffer,
        // then reference ple_context_proj_buf with offsets per layer.

        // Precompute all rotary cos/sin per layer (CPU, trivial)
        let pos = self.total_tokens as f32;
        let mut rotary_cos_per_layer: Vec<Vec<f32>> = Vec::with_capacity(self.layers.len());
        let mut rotary_sin_per_layer: Vec<Vec<f32>> = Vec::with_capacity(self.layers.len());

        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            let head_dim = layer.head_dim;
            let is_full = layer.is_full_attention;

            let rope_theta = if is_full {
                self.config.full_rope_theta()
            } else {
                self.config.sliding_rope_theta()
            };

            let rope_factor = if is_full {
                self.config.full_rope_factor()
            } else {
                self.config.sliding_rope_factor()
            };

            let rotary_dim = if is_full {
                (head_dim as f64 * self.config.full_partial_rotary_factor()) as usize
            } else {
                head_dim
            };

            // Proportional RoPE: compute rope_angles frequencies using head_dim as denominator
            // then pad with zeros for non-rotary dimensions.
            // Reference: _compute_proportional_rope_parameters in modeling_rope_utils.py
            let rope_angles = rotary_dim / 2; // = partial_rotary_factor * head_dim / 2
            let half_dim = head_dim / 2;
            let mut cos_data = vec![0.0f32; head_dim];
            let mut sin_data = vec![0.0f32; head_dim];
            for i in 0..rope_angles {
                // inv_freq = 1 / (theta ^ (2i / head_dim)) / factor
                let inv_freq = 1.0
                    / (rope_theta.powf(i as f64 * 2.0 / head_dim as f64) as f32)
                    / rope_factor as f32;
                let angle = pos * inv_freq;
                cos_data[i] = angle.cos();
                cos_data[i + half_dim] = angle.cos();
                sin_data[i] = angle.sin();
                sin_data[i + half_dim] = angle.sin();
            }
            // Non-rotary angles (nope_angles) remain 0 in inv_freq → cos=1, sin=0
            // cos_data[rope_angles..half_dim] and cos_data[half_dim+rope_angles..head_dim] stay 0
            // But for the rotary kernel, we need cos=1 for pass-through dimensions
            for i in rope_angles..half_dim {
                cos_data[i] = 1.0;
                cos_data[i + half_dim] = 1.0;
                // sin stays 0.0 (already initialized)
            }

            rotary_cos_per_layer.push(cos_data);
            rotary_sin_per_layer.push(sin_data);
        }

        // ═══ ENCODE ALL LAYERS INTO A SINGLE COMMAND BUFFER ═══
        // Write per-layer cos/sin data into persistent buffers
        let actual_num_layers = self.layers.len();
        for layer_idx in 0..actual_num_layers {
            MetalContext::write_buffer(
                &self.per_layer_cos_bufs[layer_idx],
                &rotary_cos_per_layer[layer_idx],
            );
            MetalContext::write_buffer(
                &self.per_layer_sin_bufs[layer_idx],
                &rotary_sin_per_layer[layer_idx],
            );
        }

        let cmd = self.ctx.queue.new_command_buffer();

        // ─── PLE pre-pass: encode into the same command buffer ───
        {
            let ple_enc = cmd.new_compute_command_encoder();

            // Step 2a: context_proj = per_layer_model_projection @ embed (GPU matvec)
            self.ctx.encode_matvec_q4(
                ple_enc,
                &self.per_layer_model_projection_weight,
                &self.hidden_buf,
                &self.ple_context_proj_buf,
                ple_total_dim as u32,
                hidden_size as u32,
            );

            // Step 2b: context_proj *= 1/sqrt(hidden_size)
            self.ctx.encode_vec_scale(
                ple_enc,
                &self.ple_context_proj_buf,
                &self.ple_combined_buf,
                ple_total_dim as u32,
                context_proj_scale,
            );

            // Step 2c: RMSNorm per layer (42 chunks of 256)
            self.ctx.encode_rmsnorm_per_head(
                ple_enc,
                &self.ple_combined_buf,
                &self.per_layer_projection_norm_weight,
                &self.ple_context_proj_buf,
                num_layers as u32,
                ple_dim as u32,
                eps,
            );

            // Step 3: combined = (context_proj + token_identity) * 1/sqrt(2)
            self.ctx.encode_vec_add(
                ple_enc,
                &self.ple_context_proj_buf,
                &self.ple_token_id_buf,
                &self.ple_combined_buf,
                ple_total_dim as u32,
            );
            self.ctx.encode_vec_scale(
                ple_enc,
                &self.ple_combined_buf,
                &self.ple_context_proj_buf,
                ple_total_dim as u32,
                ple_input_scale,
            );

            // Copy per-layer slices from ple_context_proj_buf to per_layer_ple_bufs
            // (GPU-to-GPU copy, avoids CPU readback)
            for i in 0..actual_num_layers {
                let offset = (i * ple_dim * 4) as u64; // byte offset
                ple_enc.set_compute_pipeline_state(&self.ctx.buf_copy_pipeline);
                ple_enc.set_buffer(0, Some(&self.ple_context_proj_buf), offset);
                ple_enc.set_buffer(1, Some(&self.per_layer_ple_bufs[i]), 0);
                let n = ple_dim as u32;
                ple_enc.set_bytes(2, 4, &n as *const u32 as *const _);
                ple_enc
                    .dispatch_threads(MTLSize::new(ple_dim as u64, 1, 1), MTLSize::new(256, 1, 1));
            }

            ple_enc.end_encoding();
        }

        for layer_idx in 0..actual_num_layers {
            let layer = &self.layers[layer_idx];
            let head_dim = layer.head_dim;
            let q_out = layer.q_out_dim;
            let kv_out = layer.kv_out_dim;
            let is_full = layer.is_full_attention;
            // Gemma4 uses attention_scale = 1.0 (QK norm handles scaling)
            let scale = 1.0f32;

            let encoder = cmd.new_compute_command_encoder();

            // ─── Attention Block ───
            // Save residual
            self.ctx.encode_copy(
                encoder,
                &self.hidden_buf,
                &self.residual_buf,
                hidden_size as u32,
            );

            // Pre-attention norm
            self.ctx.encode_rmsnorm(
                encoder,
                &self.hidden_buf,
                &layer.input_layernorm_weight,
                &self.normed_buf,
                hidden_size as u32,
                eps,
            );

            // Q projection (always computed)
            if layer.use_f16 {
                self.ctx.encode_matvec_f16(
                    encoder,
                    &layer.q_proj,
                    &self.normed_buf,
                    &self.q_buf,
                    q_out as u32,
                    hidden_size as u32,
                );
            } else {
                self.ctx.encode_matvec_q4(
                    encoder,
                    &layer.q_proj,
                    &self.normed_buf,
                    &self.q_buf,
                    q_out as u32,
                    hidden_size as u32,
                );
            }

            // QK Norm on Q
            self.ctx.encode_rmsnorm_per_head(
                encoder,
                &self.q_buf,
                &layer.q_norm_weight,
                &self.q_normed_buf,
                num_heads as u32,
                head_dim as u32,
                eps,
            );

            // Apply rotary to Q (full head_dim — non-rotary dims have cos=1, sin=0 for pass-through)
            self.ctx.encode_rotary(
                encoder,
                &self.q_normed_buf,
                &self.k_normed_buf,
                &self.per_layer_cos_bufs[layer_idx],
                &self.per_layer_sin_bufs[layer_idx],
                num_heads as u32,
                0,
                head_dim as u32,
            );

            // K, V only for non-shared layers
            if layer.has_kv {
                if layer.use_f16 {
                    self.ctx.encode_matvec_f16(
                        encoder,
                        &layer.k_proj,
                        &self.normed_buf,
                        &self.k_buf,
                        kv_out as u32,
                        hidden_size as u32,
                    );
                    self.ctx.encode_matvec_f16(
                        encoder,
                        &layer.v_proj,
                        &self.normed_buf,
                        &self.v_buf,
                        kv_out as u32,
                        hidden_size as u32,
                    );
                } else {
                    self.ctx.encode_matvec_q4(
                        encoder,
                        &layer.k_proj,
                        &self.normed_buf,
                        &self.k_buf,
                        kv_out as u32,
                        hidden_size as u32,
                    );
                    self.ctx.encode_matvec_q4(
                        encoder,
                        &layer.v_proj,
                        &self.normed_buf,
                        &self.v_buf,
                        kv_out as u32,
                        hidden_size as u32,
                    );
                }

                // K norm + rotary (full head_dim — non-rotary dims pass through)
                self.ctx.encode_rmsnorm_per_head(
                    encoder,
                    &self.k_buf,
                    &layer.k_norm_weight,
                    &self.k_normed_buf,
                    num_kv_heads as u32,
                    head_dim as u32,
                    eps,
                );
                self.ctx.encode_rotary(
                    encoder,
                    &self.q_buf,
                    &self.k_normed_buf,
                    &self.per_layer_cos_bufs[layer_idx],
                    &self.per_layer_sin_bufs[layer_idx],
                    0,
                    num_kv_heads as u32,
                    head_dim as u32,
                );

                // V norm (no weight)
                self.ctx.encode_rmsnorm_per_head_noweight(
                    encoder,
                    &self.v_buf,
                    &self.gate_buf,
                    num_kv_heads as u32,
                    head_dim as u32,
                    eps,
                );

                // Append to this layer's cache (f16)
                self.ctx.encode_kv_append_f16(
                    encoder,
                    &self.k_normed_buf,
                    &self.k_cache[layer_idx],
                    num_kv_heads as u32,
                    head_dim as u32,
                    self.kv_capacity,
                    kv_seq,
                );
                self.ctx.encode_kv_append_f16(
                    encoder,
                    &self.gate_buf,
                    &self.v_cache[layer_idx],
                    num_kv_heads as u32,
                    head_dim as u32,
                    self.kv_capacity,
                    kv_seq,
                );
            }

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

            self.ctx.encode_attention_with_offset_f16(
                encoder,
                &self.q_normed_buf,
                &self.k_cache[layer.kv_source_layer],
                &self.v_cache[layer.kv_source_layer],
                &self.attn_out_buf,
                num_heads as u32,
                num_kv_heads as u32,
                num_kv_groups,
                head_dim as u32,
                effective_kv_seq,
                self.kv_capacity,
                scale,
                kv_start,
            );

            // O projection (always f16 for quality)
            self.ctx.encode_matvec_f16(
                encoder,
                &layer.o_proj,
                &self.attn_out_buf,
                &self.o_out_buf,
                hidden_size as u32,
                q_out as u32,
            );

            // Post-attention norm (cannot be in-place, use normed_buf as temp)
            self.ctx.encode_rmsnorm(
                encoder,
                &self.o_out_buf,
                &layer.post_attention_layernorm_weight,
                &self.normed_buf,
                hidden_size as u32,
                eps,
            );

            // Residual add: hidden = residual + normed_attn_output
            self.ctx.encode_vec_add(
                encoder,
                &self.residual_buf,
                &self.normed_buf,
                &self.hidden_buf,
                hidden_size as u32,
            );

            // ─── MLP Block ───
            // Save residual
            self.ctx.encode_copy(
                encoder,
                &self.hidden_buf,
                &self.residual_buf,
                hidden_size as u32,
            );

            // Pre-feedforward norm
            self.ctx.encode_rmsnorm(
                encoder,
                &self.hidden_buf,
                &layer.pre_feedforward_layernorm_weight,
                &self.normed_buf,
                hidden_size as u32,
                eps,
            );

            // MLP: gate_proj, up_proj, GeLU activation, down_proj
            if layer.use_f16 {
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
            } else {
                self.ctx.encode_matvec_q4(
                    encoder,
                    &layer.gate_proj,
                    &self.normed_buf,
                    &self.gate_buf,
                    intermediate_size as u32,
                    hidden_size as u32,
                );
                self.ctx.encode_matvec_q4(
                    encoder,
                    &layer.up_proj,
                    &self.normed_buf,
                    &self.up_buf,
                    intermediate_size as u32,
                    hidden_size as u32,
                );
            }
            self.ctx.encode_gelu_mul(
                encoder,
                &self.gate_buf,
                &self.up_buf,
                &self.gelu_buf,
                intermediate_size as u32,
            );
            if layer.use_f16 {
                self.ctx.encode_matvec_f16(
                    encoder,
                    &layer.down_proj,
                    &self.gelu_buf,
                    &self.down_buf,
                    hidden_size as u32,
                    intermediate_size as u32,
                );
            } else {
                self.ctx.encode_matvec_q4(
                    encoder,
                    &layer.down_proj,
                    &self.gelu_buf,
                    &self.down_buf,
                    hidden_size as u32,
                    intermediate_size as u32,
                );
            }

            // Post-feedforward norm (cannot be in-place, use normed_buf as temp)
            self.ctx.encode_rmsnorm(
                encoder,
                &self.down_buf,
                &layer.post_feedforward_layernorm_weight,
                &self.normed_buf,
                hidden_size as u32,
                eps,
            );

            // Residual add: hidden = residual + normed_mlp_output
            self.ctx.encode_vec_add(
                encoder,
                &self.residual_buf,
                &self.normed_buf,
                &self.hidden_buf,
                hidden_size as u32,
            );

            // ─── Per-Layer Embedding (PLE) — after MLP, before layer_scalar ───
            // Save residual for PLE
            self.ctx.encode_copy(
                encoder,
                &self.hidden_buf,
                &self.residual_buf,
                hidden_size as u32,
            );
            // Gate: ple_gated = per_layer_input_gate(hidden) → [ple_dim]
            self.ctx.encode_matvec_f16(
                encoder,
                &layer.per_layer_input_gate_weight,
                &self.hidden_buf,
                &self.ple_gated_buf,
                ple_dim as u32,
                hidden_size as u32,
            );
            // GeLU activation (PLE uses GeLU, not the model's hidden_activation)
            // Reuse gelu_mul with up=ple_embed (element-wise multiply after gelu)
            // gelu_mul does: out = gelu(gate) * up
            self.ctx.encode_gelu_mul(
                encoder,
                &self.ple_gated_buf,
                &self.per_layer_ple_bufs[layer_idx],
                &self.ple_normed_buf,
                ple_dim as u32,
            );
            // Project back: ple_projected = per_layer_projection(ple_normed) → [hidden]
            self.ctx.encode_matvec_f16(
                encoder,
                &layer.per_layer_projection_weight,
                &self.ple_normed_buf,
                &self.ple_projected_buf,
                hidden_size as u32,
                ple_dim as u32,
            );
            // Post-PLE norm
            self.ctx.encode_rmsnorm(
                encoder,
                &self.ple_projected_buf,
                &layer.post_per_layer_input_norm_weight,
                &self.o_out_buf,
                hidden_size as u32,
                eps,
            );
            // Residual: hidden = hidden + ple_output
            self.ctx.encode_vec_add(
                encoder,
                &self.residual_buf,
                &self.o_out_buf,
                &self.hidden_buf,
                hidden_size as u32,
            );

            // Layer scalar: hidden *= layer_scalar (prevents signal growth across layers)
            // Use residual_buf as temp to avoid in-place read/write race
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

        // Commit all layer work
        cmd.commit();
        cmd.wait_until_completed();

        // Final norm
        let cmd = self.ctx.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        self.ctx.encode_rmsnorm(
            encoder,
            &self.hidden_buf,
            &self.final_norm_weight,
            &self.normed_buf,
            hidden_size as u32,
            eps,
        );
        encoder.end_encoding();

        // Submit and wait
        cmd.commit();
        cmd.wait_until_completed();

        // Compute logits using GPU (tied embeddings = embed_tokens as lm_head)
        // Use matvec_f16: embed_tokens_f16 is (vocab_size, hidden_size) in bf16
        let cmd2 = self.ctx.queue.new_command_buffer();
        let enc2 = cmd2.new_compute_command_encoder();
        self.ctx.encode_matvec_q4(
            enc2,
            &self.lm_head_buf,
            &self.normed_buf,
            &self.logits_buf,
            vocab_size as u32,
            hidden_size as u32,
        );
        enc2.end_encoding();
        cmd2.commit();
        cmd2.wait_until_completed();

        let mut logits = MetalContext::read_buffer(&self.logits_buf, vocab_size);

        // Logit softcapping: logits = cap * tanh(logits / cap)
        // Clamp input to tanh to prevent NaN from overflow (same issue as GeLU)
        let cap = self.config.final_logit_softcapping;
        for l in logits.iter_mut() {
            let x = (*l / cap).clamp(-10.0, 10.0);
            *l = cap * x.tanh();
        }

        // Update state
        self.total_tokens += 1;
        self.kv_seq_len += 1;

        logits
    }

    pub fn num_items(&self) -> usize {
        self.total_tokens
    }

    pub fn reset_legacy_state(&mut self) {
        self.kv_seq_len = 0;
        self.total_tokens = 0;
    }

    pub fn create_kv_pool(&self, num_slots: usize, max_seq_len: u32) -> KvCachePool {
        let max_seq_len = max_seq_len.min(self.config.max_position_embeddings as u32);
        KvCachePool::new(&self.ctx, &self.config, num_slots, max_seq_len)
    }

    pub fn max_parallel_prefill_seq(&self) -> usize {
        self.prefill_scratch.max_seq_len
    }

    pub fn max_decode_batch_size(&self) -> usize {
        self.decode_batch_scratch.max_batch_size
    }

    pub fn prepare_parallel_prefill_inputs(&mut self, token_ids: &[usize]) -> Result<(), String> {
        if token_ids.is_empty() {
            return Err("prefill token_ids must not be empty".to_string());
        }
        if token_ids.len() > self.prefill_scratch.max_seq_len {
            return Err(format!(
                "prefill chunk has {} tokens, max supported chunk is {}",
                token_ids.len(),
                self.prefill_scratch.max_seq_len
            ));
        }

        let inputs = self.prepare_batched_token_inputs(token_ids)?;
        MetalContext::write_buffer(&self.prefill_scratch.hidden_buf, &inputs.hidden);
        MetalContext::write_buffer(
            &self.prefill_scratch.ple_token_id_buf,
            &inputs.ple_token_identity,
        );
        Ok(())
    }

    fn prepare_parallel_prefill_rotary(
        &mut self,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<(), String> {
        self.prepare_parallel_prefill_rotary_segments(&[(start_pos, seq_len)])
    }

    fn prepare_parallel_prefill_rotary_segments(
        &mut self,
        segments: &[(usize, usize)],
    ) -> Result<(), String> {
        let total_seq_len: usize = segments.iter().map(|(_, seq_len)| *seq_len).sum();
        if total_seq_len == 0 {
            return Err("prefill seq_len must not be empty".to_string());
        }
        if total_seq_len > self.prefill_scratch.max_seq_len {
            return Err(format!(
                "prefill chunk has {} tokens, max supported chunk is {}",
                total_seq_len, self.prefill_scratch.max_seq_len
            ));
        }

        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            let head_dim = layer.head_dim;
            let half_dim = head_dim / 2;
            let is_full = layer.is_full_attention;
            let rope_theta = if is_full {
                self.config.full_rope_theta()
            } else {
                self.config.sliding_rope_theta()
            };
            let rope_factor = if is_full {
                self.config.full_rope_factor()
            } else {
                self.config.sliding_rope_factor()
            };
            let rotary_dim = if is_full {
                (head_dim as f64 * self.config.full_partial_rotary_factor()) as usize
            } else {
                head_dim
            };
            let rope_angles = rotary_dim / 2;

            let mut cos_batch = vec![0.0f32; total_seq_len * head_dim];
            let mut sin_batch = vec![0.0f32; total_seq_len * head_dim];

            let mut row_idx = 0;
            for &(start_pos, seq_len) in segments {
                for token_idx in 0..seq_len {
                    let pos = (start_pos + token_idx) as f32;
                    let token_offset = row_idx * head_dim;

                    for i in 0..rope_angles {
                        let inv_freq = 1.0
                            / (rope_theta.powf(i as f64 * 2.0 / head_dim as f64) as f32)
                            / rope_factor as f32;
                        let angle = pos * inv_freq;
                        cos_batch[token_offset + i] = angle.cos();
                        cos_batch[token_offset + i + half_dim] = angle.cos();
                        sin_batch[token_offset + i] = angle.sin();
                        sin_batch[token_offset + i + half_dim] = angle.sin();
                    }

                    for i in rope_angles..half_dim {
                        cos_batch[token_offset + i] = 1.0;
                        cos_batch[token_offset + i + half_dim] = 1.0;
                    }

                    row_idx += 1;
                }
            }

            MetalContext::write_buffer(&self.per_layer_prefill_cos_bufs[layer_idx], &cos_batch);
            MetalContext::write_buffer(&self.per_layer_prefill_sin_bufs[layer_idx], &sin_batch);
        }

        Ok(())
    }

    pub fn prepare_decode_batch_inputs(&mut self, token_ids: &[usize]) -> Result<(), String> {
        if token_ids.is_empty() {
            return Err("decode batch token_ids must not be empty".to_string());
        }
        if token_ids.len() > self.decode_batch_scratch.max_batch_size {
            return Err(format!(
                "decode batch has {} requests, max supported batch is {}",
                token_ids.len(),
                self.decode_batch_scratch.max_batch_size
            ));
        }

        let inputs = self.prepare_batched_token_inputs(token_ids)?;
        MetalContext::write_buffer(&self.decode_batch_scratch.hidden_buf, &inputs.hidden);
        MetalContext::write_buffer(
            &self.decode_batch_scratch.ple_token_id_buf,
            &inputs.ple_token_identity,
        );
        Ok(())
    }

    fn prepare_decode_batch_rotary(&mut self, slot_views: &[KvSlotView]) -> Result<(), String> {
        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            let head_dim = layer.head_dim;
            let half_dim = head_dim / 2;
            let is_full = layer.is_full_attention;
            let rope_theta = if is_full {
                self.config.full_rope_theta()
            } else {
                self.config.sliding_rope_theta()
            };
            let rope_factor = if is_full {
                self.config.full_rope_factor()
            } else {
                self.config.sliding_rope_factor()
            };
            let rotary_dim = if is_full {
                (head_dim as f64 * self.config.full_partial_rotary_factor()) as usize
            } else {
                head_dim
            };
            let rope_angles = rotary_dim / 2;

            let mut cos_batch = vec![0.0f32; slot_views.len() * head_dim];
            let mut sin_batch = vec![0.0f32; slot_views.len() * head_dim];

            for (batch_idx, slot_view) in slot_views.iter().enumerate() {
                let pos = slot_view.total_tokens as f32;
                let batch_offset = batch_idx * head_dim;

                for i in 0..rope_angles {
                    let inv_freq = 1.0
                        / (rope_theta.powf(i as f64 * 2.0 / head_dim as f64) as f32)
                        / rope_factor as f32;
                    let angle = pos * inv_freq;
                    cos_batch[batch_offset + i] = angle.cos();
                    cos_batch[batch_offset + i + half_dim] = angle.cos();
                    sin_batch[batch_offset + i] = angle.sin();
                    sin_batch[batch_offset + i + half_dim] = angle.sin();
                }

                for i in rope_angles..half_dim {
                    cos_batch[batch_offset + i] = 1.0;
                    cos_batch[batch_offset + i + half_dim] = 1.0;
                }
            }

            MetalContext::write_buffer(
                &self.per_layer_decode_batch_cos_bufs[layer_idx],
                &cos_batch,
            );
            MetalContext::write_buffer(
                &self.per_layer_decode_batch_sin_bufs[layer_idx],
                &sin_batch,
            );
        }

        Ok(())
    }

    fn prepare_decode_batch_attention_metadata(
        &mut self,
        slot_views: &[KvSlotView],
    ) -> Result<(), String> {
        let mut append_positions = vec![0u32; slot_views.len()];
        for (batch_idx, slot_view) in slot_views.iter().enumerate() {
            append_positions[batch_idx] = slot_view.seq_len;
        }

        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            let mut kv_starts = vec![0u32; slot_views.len()];
            let mut kv_lens = vec![0u32; slot_views.len()];

            for (batch_idx, &append_pos) in append_positions.iter().enumerate() {
                let attn_kv_seq = append_pos + 1;
                if layer.is_full_attention {
                    kv_starts[batch_idx] = 0;
                    kv_lens[batch_idx] = attn_kv_seq;
                } else {
                    let window = self.config.sliding_window as u32;
                    kv_lens[batch_idx] = attn_kv_seq.min(window);
                    kv_starts[batch_idx] = attn_kv_seq.saturating_sub(window);
                }
            }

            MetalContext::write_u32_buffer(
                &self.per_layer_decode_batch_append_pos_bufs[layer_idx],
                &append_positions,
            );
            MetalContext::write_u32_buffer(
                &self.per_layer_decode_batch_kv_start_bufs[layer_idx],
                &kv_starts,
            );
            MetalContext::write_u32_buffer(
                &self.per_layer_decode_batch_kv_len_bufs[layer_idx],
                &kv_lens,
            );
        }

        Ok(())
    }

    fn prepare_batched_token_inputs(
        &self,
        token_ids: &[usize],
    ) -> Result<BatchedTokenInputs, String> {
        let hidden_size = self.config.hidden_size;
        let num_layers = self.config.num_hidden_layers;
        let ple_dim = self.config.hidden_size_per_layer_input;
        let ple_total_dim = num_layers * ple_dim;
        let embed_scale = (hidden_size as f32).sqrt();
        let ple_scale = (ple_dim as f32).sqrt();

        let mut hidden = vec![0.0f32; token_ids.len() * hidden_size];
        let mut ple_token_identity = vec![0.0f32; token_ids.len() * ple_total_dim];

        for (pos, &token_id) in token_ids.iter().enumerate() {
            let embed_offset = token_id
                .checked_mul(hidden_size)
                .ok_or_else(|| format!("token id {} overflowed embedding offset", token_id))?;
            if embed_offset + hidden_size > self.embed_tokens_f16.len() {
                return Err(format!("token id {} is outside embed_tokens", token_id));
            }

            let hidden_offset = pos * hidden_size;
            for i in 0..hidden_size {
                hidden[hidden_offset + i] =
                    bf16_to_f32(self.embed_tokens_f16[embed_offset + i]) * embed_scale;
            }

            let ple_token_offset = token_id
                .checked_mul(ple_total_dim)
                .ok_or_else(|| format!("token id {} overflowed PLE offset", token_id))?;
            if ple_token_offset + ple_total_dim > self.embed_tokens_per_layer_f16.len() {
                return Err(format!(
                    "token id {} is outside embed_tokens_per_layer",
                    token_id
                ));
            }

            let ple_out_offset = pos * ple_total_dim;
            for i in 0..ple_total_dim {
                ple_token_identity[ple_out_offset + i] =
                    bf16_to_f32(self.embed_tokens_per_layer_f16[ple_token_offset + i]) * ple_scale;
            }
        }

        Ok(BatchedTokenInputs {
            batch_size: token_ids.len(),
            hidden,
            ple_token_identity,
        })
    }

    fn f32_byte_offset(elements: usize) -> u64 {
        (elements * std::mem::size_of::<f32>()) as u64
    }

    fn decode_batch_row_offsets(&self, batch_idx: usize) -> DecodeBatchRowOffsets {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let ple_dim = self.config.hidden_size_per_layer_input;
        let ple_total_dim = self.config.num_hidden_layers * ple_dim;
        let max_q_out = self
            .layers
            .iter()
            .map(|layer| layer.q_out_dim)
            .max()
            .unwrap_or(0);
        let max_kv_out = self
            .layers
            .iter()
            .map(|layer| layer.kv_out_dim)
            .max()
            .unwrap_or(0);

        DecodeBatchRowOffsets {
            hidden: Self::f32_byte_offset(batch_idx * hidden_size),
            q: Self::f32_byte_offset(batch_idx * max_q_out),
            kv: Self::f32_byte_offset(batch_idx * max_kv_out),
            intermediate: Self::f32_byte_offset(batch_idx * intermediate_size),
            ple_row: Self::f32_byte_offset(batch_idx * ple_total_dim),
        }
    }

    fn decode_batch_layer_ple_offset(&self, batch_idx: usize, layer_idx: usize) -> u64 {
        let ple_dim = self.config.hidden_size_per_layer_input;
        let ple_total_dim = self.config.num_hidden_layers * ple_dim;
        Self::f32_byte_offset(batch_idx * ple_total_dim + layer_idx * ple_dim)
    }

    fn prefill_row_offsets(&self, token_idx: usize) -> PrefillRowOffsets {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let ple_dim = self.config.hidden_size_per_layer_input;
        let ple_total_dim = self.config.num_hidden_layers * ple_dim;
        let max_q_out = self
            .layers
            .iter()
            .map(|layer| layer.q_out_dim)
            .max()
            .unwrap_or(0);
        let max_kv_out = self
            .layers
            .iter()
            .map(|layer| layer.kv_out_dim)
            .max()
            .unwrap_or(0);

        PrefillRowOffsets {
            hidden: Self::f32_byte_offset(token_idx * hidden_size),
            q: Self::f32_byte_offset(token_idx * max_q_out),
            kv: Self::f32_byte_offset(token_idx * max_kv_out),
            intermediate: Self::f32_byte_offset(token_idx * intermediate_size),
            ple_row: Self::f32_byte_offset(token_idx * ple_total_dim),
        }
    }

    fn encode_parallel_prefill_ple_context(&mut self, seq_len: usize) -> Result<(), String> {
        if seq_len == 0 {
            return Err("prefill seq_len must not be empty".to_string());
        }
        if seq_len > self.prefill_scratch.max_seq_len {
            return Err(format!(
                "prefill chunk has {} tokens, max supported chunk is {}",
                seq_len, self.prefill_scratch.max_seq_len
            ));
        }

        let hidden_size = self.config.hidden_size;
        let num_layers = self.config.num_hidden_layers;
        let ple_dim = self.config.hidden_size_per_layer_input;
        let ple_total_dim = num_layers * ple_dim;
        let eps = self.config.rms_norm_eps as f32;
        let context_proj_scale = 1.0f32 / (hidden_size as f32).sqrt();
        let ple_input_scale = 1.0f32 / 2.0f32.sqrt();

        let cmd = self.ctx.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        let total_ple = (seq_len * ple_total_dim) as u32;

        self.ctx.encode_projection_q4_batch(
            encoder,
            &self.per_layer_model_projection_weight,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.ple_context_proj_buf,
            ple_total_dim as u32,
            hidden_size as u32,
            seq_len as u32,
        );
        self.ctx.encode_vec_scale(
            encoder,
            &self.prefill_scratch.ple_context_proj_buf,
            &self.prefill_scratch.ple_combined_buf,
            total_ple,
            context_proj_scale,
        );
        self.ctx.encode_rmsnorm_batch(
            encoder,
            &self.prefill_scratch.ple_combined_buf,
            &self.per_layer_projection_norm_weight,
            &self.prefill_scratch.ple_context_proj_buf,
            ple_dim as u32,
            eps,
            (seq_len * num_layers) as u32,
        );
        self.ctx.encode_vec_add_batch(
            encoder,
            &self.prefill_scratch.ple_context_proj_buf,
            &self.prefill_scratch.ple_token_id_buf,
            &self.prefill_scratch.ple_combined_buf,
            total_ple,
        );
        self.ctx.encode_vec_scale(
            encoder,
            &self.prefill_scratch.ple_combined_buf,
            &self.prefill_scratch.ple_context_proj_buf,
            total_ple,
            ple_input_scale,
        );

        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    fn encode_parallel_prefill_attention_inputs(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer_idx: usize,
        seq_len: usize,
    ) -> Result<(), String> {
        if seq_len == 0 {
            return Err("prefill seq_len must not be empty".to_string());
        }
        if seq_len > self.prefill_scratch.max_seq_len {
            return Err(format!(
                "prefill chunk has {} tokens, max supported chunk is {}",
                seq_len, self.prefill_scratch.max_seq_len
            ));
        }

        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| format!("invalid layer index {}", layer_idx))?;
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = layer.head_dim;
        let q_out = layer.q_out_dim;
        let kv_out = layer.kv_out_dim;
        let eps = self.config.rms_norm_eps as f32;

        self.ctx.encode_rmsnorm_batch(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &layer.input_layernorm_weight,
            &self.prefill_scratch.normed_buf,
            hidden_size as u32,
            eps,
            seq_len as u32,
        );

        if layer.use_f16 {
            self.ctx.encode_projection_f16_batch(
                encoder,
                &layer.q_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.q_buf,
                q_out as u32,
                hidden_size as u32,
                seq_len as u32,
            );
            if layer.has_kv {
                self.ctx.encode_projection_f16_batch(
                    encoder,
                    &layer.k_proj,
                    &self.prefill_scratch.normed_buf,
                    &self.prefill_scratch.k_buf,
                    kv_out as u32,
                    hidden_size as u32,
                    seq_len as u32,
                );
                self.ctx.encode_projection_f16_batch(
                    encoder,
                    &layer.v_proj,
                    &self.prefill_scratch.normed_buf,
                    &self.prefill_scratch.v_buf,
                    kv_out as u32,
                    hidden_size as u32,
                    seq_len as u32,
                );
            }
        } else {
            self.ctx.encode_projection_q4_batch(
                encoder,
                &layer.q_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.q_buf,
                q_out as u32,
                hidden_size as u32,
                seq_len as u32,
            );
            if layer.has_kv {
                self.ctx.encode_projection_q4_batch(
                    encoder,
                    &layer.k_proj,
                    &self.prefill_scratch.normed_buf,
                    &self.prefill_scratch.k_buf,
                    kv_out as u32,
                    hidden_size as u32,
                    seq_len as u32,
                );
                self.ctx.encode_projection_q4_batch(
                    encoder,
                    &layer.v_proj,
                    &self.prefill_scratch.normed_buf,
                    &self.prefill_scratch.v_buf,
                    kv_out as u32,
                    hidden_size as u32,
                    seq_len as u32,
                );
            }
        }

        self.ctx.encode_rmsnorm_batch(
            encoder,
            &self.prefill_scratch.q_buf,
            &layer.q_norm_weight,
            &self.prefill_scratch.q_normed_buf,
            head_dim as u32,
            eps,
            (seq_len * num_heads) as u32,
        );

        if layer.has_kv {
            self.ctx.encode_rmsnorm_batch(
                encoder,
                &self.prefill_scratch.k_buf,
                &layer.k_norm_weight,
                &self.prefill_scratch.k_normed_buf,
                head_dim as u32,
                eps,
                (seq_len * num_kv_heads) as u32,
            );
        }

        self.ctx.encode_transpose_shd(
            encoder,
            &self.prefill_scratch.q_normed_buf,
            &self.prefill_scratch.q_buf,
            seq_len as u32,
            num_heads as u32,
            head_dim as u32,
        );

        if layer.has_kv {
            self.ctx.encode_transpose_shd(
                encoder,
                &self.prefill_scratch.k_normed_buf,
                &self.prefill_scratch.k_buf,
                seq_len as u32,
                num_kv_heads as u32,
                head_dim as u32,
            );

            self.ctx.encode_rmsnorm_noweight_batch(
                encoder,
                &self.prefill_scratch.v_buf,
                &self.prefill_scratch.k_normed_buf,
                head_dim as u32,
                eps,
                (seq_len * num_kv_heads) as u32,
            );

            self.ctx.encode_transpose_shd(
                encoder,
                &self.prefill_scratch.k_normed_buf,
                &self.prefill_scratch.v_buf,
                seq_len as u32,
                num_kv_heads as u32,
                head_dim as u32,
            );
        }

        self.ctx.encode_rotary_batch(
            encoder,
            &self.prefill_scratch.q_buf,
            &self.prefill_scratch.k_buf,
            &self.per_layer_prefill_cos_bufs[layer_idx],
            &self.per_layer_prefill_sin_bufs[layer_idx],
            num_heads as u32,
            if layer.has_kv { num_kv_heads as u32 } else { 0 },
            head_dim as u32,
            seq_len as u32,
        );

        Ok(())
    }

    fn can_use_parallel_prefill_chunk(
        &self,
        start_pos: usize,
        seq_len: usize,
        kv_pool: &KvCachePool,
    ) -> bool {
        if seq_len <= 1 || seq_len > self.prefill_scratch.max_seq_len {
            return false;
        }
        if start_pos + seq_len > kv_pool.capacity() as usize {
            return false;
        }

        true
    }

    fn encode_parallel_prefill_layer(
        &mut self,
        layer_idx: usize,
        seq_len: usize,
        kv_pool: &mut KvCachePool,
        slot: KvSlot,
        start_pos: usize,
    ) -> Result<(), String> {
        let cmd = self.ctx.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        self.encode_parallel_prefill_attention_inputs(encoder, layer_idx, seq_len)?;

        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| format!("invalid layer index {}", layer_idx))?;
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let num_kv_groups = (num_heads / num_kv_heads) as u32;
        let ple_dim = self.config.hidden_size_per_layer_input;
        let eps = self.config.rms_norm_eps as f32;
        let head_dim = layer.head_dim;
        let q_out = layer.q_out_dim;
        let kv_out = layer.kv_out_dim;
        let total_hidden = (seq_len * hidden_size) as u32;
        let total_intermediate = (seq_len * intermediate_size) as u32;
        let scale = 1.0f32;
        let attention_window = if layer.is_full_attention {
            0
        } else {
            self.config.sliding_window as u32
        };

        if layer.has_kv {
            let k_cache = kv_pool
                .layer_k_cache(slot, layer_idx)
                .map_err(|err| err.to_string())?;
            let v_cache = kv_pool
                .layer_v_cache(slot, layer_idx)
                .map_err(|err| err.to_string())?;
            self.ctx.encode_kv_batch_append_f16(
                encoder,
                &self.prefill_scratch.k_buf,
                k_cache,
                num_kv_heads as u32,
                head_dim as u32,
                kv_pool.capacity(),
                start_pos as u32,
                seq_len as u32,
            );
            self.ctx.encode_kv_batch_append_f16(
                encoder,
                &self.prefill_scratch.v_buf,
                v_cache,
                num_kv_heads as u32,
                head_dim as u32,
                kv_pool.capacity(),
                start_pos as u32,
                seq_len as u32,
            );
        }

        let k_cache = kv_pool
            .layer_k_cache(slot, layer.kv_source_layer)
            .map_err(|err| err.to_string())?;
        let v_cache = kv_pool
            .layer_v_cache(slot, layer.kv_source_layer)
            .map_err(|err| err.to_string())?;
        self.ctx.encode_attention_causal_f16(
            encoder,
            &self.prefill_scratch.q_buf,
            k_cache,
            v_cache,
            &self.prefill_scratch.attn_out_buf,
            num_heads as u32,
            num_kv_heads as u32,
            num_kv_groups,
            head_dim as u32,
            (start_pos + seq_len) as u32,
            kv_pool.capacity(),
            scale,
            seq_len as u32,
            start_pos as u32,
            attention_window,
        );

        self.ctx.encode_copy(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.residual_buf,
            total_hidden,
        );
        self.ctx.encode_transpose_hsd(
            encoder,
            &self.prefill_scratch.attn_out_buf,
            &self.prefill_scratch.q_normed_buf,
            seq_len as u32,
            num_heads as u32,
            head_dim as u32,
        );
        self.ctx.encode_projection_f16_batch(
            encoder,
            &layer.o_proj,
            &self.prefill_scratch.q_normed_buf,
            &self.prefill_scratch.o_out_buf,
            hidden_size as u32,
            q_out as u32,
            seq_len as u32,
        );
        self.ctx.encode_rmsnorm_batch(
            encoder,
            &self.prefill_scratch.o_out_buf,
            &layer.post_attention_layernorm_weight,
            &self.prefill_scratch.normed_buf,
            hidden_size as u32,
            eps,
            seq_len as u32,
        );
        self.ctx.encode_vec_add_batch(
            encoder,
            &self.prefill_scratch.residual_buf,
            &self.prefill_scratch.normed_buf,
            &self.prefill_scratch.hidden_buf,
            total_hidden,
        );

        self.ctx.encode_copy(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.residual_buf,
            total_hidden,
        );
        self.ctx.encode_rmsnorm_batch(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &layer.pre_feedforward_layernorm_weight,
            &self.prefill_scratch.normed_buf,
            hidden_size as u32,
            eps,
            seq_len as u32,
        );
        if layer.use_f16 {
            self.ctx.encode_projection_f16_batch(
                encoder,
                &layer.gate_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.gate_buf,
                intermediate_size as u32,
                hidden_size as u32,
                seq_len as u32,
            );
            self.ctx.encode_projection_f16_batch(
                encoder,
                &layer.up_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.up_buf,
                intermediate_size as u32,
                hidden_size as u32,
                seq_len as u32,
            );
        } else {
            self.ctx.encode_projection_q4_batch(
                encoder,
                &layer.gate_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.gate_buf,
                intermediate_size as u32,
                hidden_size as u32,
                seq_len as u32,
            );
            self.ctx.encode_projection_q4_batch(
                encoder,
                &layer.up_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.up_buf,
                intermediate_size as u32,
                hidden_size as u32,
                seq_len as u32,
            );
        }
        self.ctx.encode_gelu_mul(
            encoder,
            &self.prefill_scratch.gate_buf,
            &self.prefill_scratch.up_buf,
            &self.prefill_scratch.gelu_buf,
            total_intermediate,
        );
        if layer.use_f16 {
            self.ctx.encode_projection_f16_batch(
                encoder,
                &layer.down_proj,
                &self.prefill_scratch.gelu_buf,
                &self.prefill_scratch.down_buf,
                hidden_size as u32,
                intermediate_size as u32,
                seq_len as u32,
            );
        } else {
            self.ctx.encode_projection_q4_batch(
                encoder,
                &layer.down_proj,
                &self.prefill_scratch.gelu_buf,
                &self.prefill_scratch.down_buf,
                hidden_size as u32,
                intermediate_size as u32,
                seq_len as u32,
            );
        }
        self.ctx.encode_rmsnorm_batch(
            encoder,
            &self.prefill_scratch.down_buf,
            &layer.post_feedforward_layernorm_weight,
            &self.prefill_scratch.normed_buf,
            hidden_size as u32,
            eps,
            seq_len as u32,
        );
        self.ctx.encode_vec_add_batch(
            encoder,
            &self.prefill_scratch.residual_buf,
            &self.prefill_scratch.normed_buf,
            &self.prefill_scratch.hidden_buf,
            total_hidden,
        );

        self.ctx.encode_copy(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.residual_buf,
            total_hidden,
        );
        self.ctx.encode_projection_f16_batch(
            encoder,
            &layer.per_layer_input_gate_weight,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.gate_buf,
            ple_dim as u32,
            hidden_size as u32,
            seq_len as u32,
        );
        self.ctx.encode_ple_gelu_mul_batch(
            encoder,
            &self.prefill_scratch.gate_buf,
            &self.prefill_scratch.ple_context_proj_buf,
            &self.prefill_scratch.up_buf,
            layer_idx as u32,
            self.config.num_hidden_layers as u32,
            ple_dim as u32,
            seq_len as u32,
        );
        self.ctx.encode_projection_f16_batch(
            encoder,
            &layer.per_layer_projection_weight,
            &self.prefill_scratch.up_buf,
            &self.prefill_scratch.o_out_buf,
            hidden_size as u32,
            ple_dim as u32,
            seq_len as u32,
        );
        self.ctx.encode_rmsnorm_batch(
            encoder,
            &self.prefill_scratch.o_out_buf,
            &layer.post_per_layer_input_norm_weight,
            &self.prefill_scratch.normed_buf,
            hidden_size as u32,
            eps,
            seq_len as u32,
        );
        self.ctx.encode_vec_add_batch(
            encoder,
            &self.prefill_scratch.residual_buf,
            &self.prefill_scratch.normed_buf,
            &self.prefill_scratch.hidden_buf,
            total_hidden,
        );
        self.ctx.encode_vec_scale(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.residual_buf,
            total_hidden,
            layer.layer_scalar,
        );
        self.ctx.encode_copy(
            encoder,
            &self.prefill_scratch.residual_buf,
            &self.prefill_scratch.hidden_buf,
            total_hidden,
        );

        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    fn encode_parallel_prefill_layer_batched(
        &mut self,
        layer_idx: usize,
        total_seq_len: usize,
        segments: &[PrefillBatchSegment],
        kv_pool: &mut KvCachePool,
    ) -> Result<(), String> {
        let cmd = self.ctx.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        self.encode_parallel_prefill_attention_inputs(encoder, layer_idx, total_seq_len)?;

        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| format!("invalid layer index {}", layer_idx))?;
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let num_kv_groups = (num_heads / num_kv_heads) as u32;
        let ple_dim = self.config.hidden_size_per_layer_input;
        let eps = self.config.rms_norm_eps as f32;
        let head_dim = layer.head_dim;
        let q_out = layer.q_out_dim;
        let total_hidden = (total_seq_len * hidden_size) as u32;
        let total_intermediate = (total_seq_len * intermediate_size) as u32;
        let scale = 1.0f32;
        let attention_window = if layer.is_full_attention {
            0
        } else {
            self.config.sliding_window as u32
        };

        if layer.has_kv {
            for segment in segments {
                let k_cache = kv_pool
                    .layer_k_cache(segment.slot, layer_idx)
                    .map_err(|err| err.to_string())?;
                let v_cache = kv_pool
                    .layer_v_cache(segment.slot, layer_idx)
                    .map_err(|err| err.to_string())?;
                self.ctx.encode_kv_batch_append_strided_f16(
                    encoder,
                    &self.prefill_scratch.k_buf,
                    k_cache,
                    num_kv_heads as u32,
                    head_dim as u32,
                    kv_pool.capacity(),
                    segment.start_pos as u32,
                    segment.token_count as u32,
                    total_seq_len as u32,
                    segment.row_start as u32,
                );
                self.ctx.encode_kv_batch_append_strided_f16(
                    encoder,
                    &self.prefill_scratch.v_buf,
                    v_cache,
                    num_kv_heads as u32,
                    head_dim as u32,
                    kv_pool.capacity(),
                    segment.start_pos as u32,
                    segment.token_count as u32,
                    total_seq_len as u32,
                    segment.row_start as u32,
                );
            }
        }

        for segment in segments {
            let k_cache = kv_pool
                .layer_k_cache(segment.slot, layer.kv_source_layer)
                .map_err(|err| err.to_string())?;
            let v_cache = kv_pool
                .layer_v_cache(segment.slot, layer.kv_source_layer)
                .map_err(|err| err.to_string())?;
            self.ctx.encode_attention_causal_strided_f16(
                encoder,
                &self.prefill_scratch.q_buf,
                k_cache,
                v_cache,
                &self.prefill_scratch.attn_out_buf,
                num_heads as u32,
                num_kv_heads as u32,
                num_kv_groups,
                head_dim as u32,
                (segment.start_pos + segment.token_count) as u32,
                kv_pool.capacity(),
                scale,
                segment.token_count as u32,
                segment.start_pos as u32,
                attention_window,
                total_seq_len as u32,
                segment.row_start as u32,
            );
        }

        self.ctx.encode_copy(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.residual_buf,
            total_hidden,
        );
        self.ctx.encode_transpose_hsd(
            encoder,
            &self.prefill_scratch.attn_out_buf,
            &self.prefill_scratch.q_normed_buf,
            total_seq_len as u32,
            num_heads as u32,
            head_dim as u32,
        );
        self.ctx.encode_projection_f16_batch(
            encoder,
            &layer.o_proj,
            &self.prefill_scratch.q_normed_buf,
            &self.prefill_scratch.o_out_buf,
            hidden_size as u32,
            q_out as u32,
            total_seq_len as u32,
        );
        self.ctx.encode_rmsnorm_batch(
            encoder,
            &self.prefill_scratch.o_out_buf,
            &layer.post_attention_layernorm_weight,
            &self.prefill_scratch.normed_buf,
            hidden_size as u32,
            eps,
            total_seq_len as u32,
        );
        self.ctx.encode_vec_add_batch(
            encoder,
            &self.prefill_scratch.residual_buf,
            &self.prefill_scratch.normed_buf,
            &self.prefill_scratch.hidden_buf,
            total_hidden,
        );

        self.ctx.encode_copy(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.residual_buf,
            total_hidden,
        );
        self.ctx.encode_rmsnorm_batch(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &layer.pre_feedforward_layernorm_weight,
            &self.prefill_scratch.normed_buf,
            hidden_size as u32,
            eps,
            total_seq_len as u32,
        );
        if layer.use_f16 {
            self.ctx.encode_projection_f16_batch(
                encoder,
                &layer.gate_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.gate_buf,
                intermediate_size as u32,
                hidden_size as u32,
                total_seq_len as u32,
            );
            self.ctx.encode_projection_f16_batch(
                encoder,
                &layer.up_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.up_buf,
                intermediate_size as u32,
                hidden_size as u32,
                total_seq_len as u32,
            );
        } else {
            self.ctx.encode_projection_q4_batch(
                encoder,
                &layer.gate_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.gate_buf,
                intermediate_size as u32,
                hidden_size as u32,
                total_seq_len as u32,
            );
            self.ctx.encode_projection_q4_batch(
                encoder,
                &layer.up_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.up_buf,
                intermediate_size as u32,
                hidden_size as u32,
                total_seq_len as u32,
            );
        }
        self.ctx.encode_gelu_mul(
            encoder,
            &self.prefill_scratch.gate_buf,
            &self.prefill_scratch.up_buf,
            &self.prefill_scratch.gelu_buf,
            total_intermediate,
        );
        if layer.use_f16 {
            self.ctx.encode_projection_f16_batch(
                encoder,
                &layer.down_proj,
                &self.prefill_scratch.gelu_buf,
                &self.prefill_scratch.down_buf,
                hidden_size as u32,
                intermediate_size as u32,
                total_seq_len as u32,
            );
        } else {
            self.ctx.encode_projection_q4_batch(
                encoder,
                &layer.down_proj,
                &self.prefill_scratch.gelu_buf,
                &self.prefill_scratch.down_buf,
                hidden_size as u32,
                intermediate_size as u32,
                total_seq_len as u32,
            );
        }
        self.ctx.encode_rmsnorm_batch(
            encoder,
            &self.prefill_scratch.down_buf,
            &layer.post_feedforward_layernorm_weight,
            &self.prefill_scratch.normed_buf,
            hidden_size as u32,
            eps,
            total_seq_len as u32,
        );
        self.ctx.encode_vec_add_batch(
            encoder,
            &self.prefill_scratch.residual_buf,
            &self.prefill_scratch.normed_buf,
            &self.prefill_scratch.hidden_buf,
            total_hidden,
        );

        self.ctx.encode_copy(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.residual_buf,
            total_hidden,
        );
        self.ctx.encode_projection_f16_batch(
            encoder,
            &layer.per_layer_input_gate_weight,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.gate_buf,
            ple_dim as u32,
            hidden_size as u32,
            total_seq_len as u32,
        );
        self.ctx.encode_ple_gelu_mul_batch(
            encoder,
            &self.prefill_scratch.gate_buf,
            &self.prefill_scratch.ple_context_proj_buf,
            &self.prefill_scratch.up_buf,
            layer_idx as u32,
            self.config.num_hidden_layers as u32,
            ple_dim as u32,
            total_seq_len as u32,
        );
        self.ctx.encode_projection_f16_batch(
            encoder,
            &layer.per_layer_projection_weight,
            &self.prefill_scratch.up_buf,
            &self.prefill_scratch.o_out_buf,
            hidden_size as u32,
            ple_dim as u32,
            total_seq_len as u32,
        );
        self.ctx.encode_rmsnorm_batch(
            encoder,
            &self.prefill_scratch.o_out_buf,
            &layer.post_per_layer_input_norm_weight,
            &self.prefill_scratch.normed_buf,
            hidden_size as u32,
            eps,
            total_seq_len as u32,
        );
        self.ctx.encode_vec_add_batch(
            encoder,
            &self.prefill_scratch.residual_buf,
            &self.prefill_scratch.normed_buf,
            &self.prefill_scratch.hidden_buf,
            total_hidden,
        );
        self.ctx.encode_vec_scale(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.residual_buf,
            total_hidden,
            layer.layer_scalar,
        );
        self.ctx.encode_copy(
            encoder,
            &self.prefill_scratch.residual_buf,
            &self.prefill_scratch.hidden_buf,
            total_hidden,
        );

        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    fn forward_prefill_chunk_parallel_with_kv_slot(
        &mut self,
        token_ids: &[usize],
        kv_pool: &mut KvCachePool,
        slot: KvSlot,
        start_pos: usize,
    ) -> Result<Vec<f32>, String> {
        let seq_len = token_ids.len();
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let eps = self.config.rms_norm_eps as f32;

        self.encode_parallel_prefill_ple_context(seq_len)?;
        for layer_idx in 0..self.layers.len() {
            self.encode_parallel_prefill_layer(layer_idx, seq_len, kv_pool, slot, start_pos)?;
        }

        let cmd = self.ctx.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        self.ctx.encode_rmsnorm_batch(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &self.final_norm_weight,
            &self.prefill_scratch.normed_buf,
            hidden_size as u32,
            eps,
            seq_len as u32,
        );
        let last_offsets = self.prefill_row_offsets(seq_len - 1);
        self.ctx.encode_matvec_q4_at(
            encoder,
            &self.lm_head_buf,
            &self.prefill_scratch.normed_buf,
            last_offsets.hidden,
            &self.prefill_scratch.logits_buf,
            0,
            vocab_size as u32,
            hidden_size as u32,
        );
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let mut logits = MetalContext::read_buffer(&self.prefill_scratch.logits_buf, vocab_size);
        let cap = self.config.final_logit_softcapping;
        for logit in &mut logits {
            let x = (*logit / cap).clamp(-10.0, 10.0);
            *logit = cap * x.tanh();
        }

        kv_pool
            .with_slot_mut(slot, |slot_state| {
                slot_state.seq_len += seq_len as u32;
                slot_state.total_tokens += seq_len;
            })
            .map_err(|err| err.to_string())?;

        Ok(logits)
    }

    fn forward_decode_batch_encoded_with_kv_slots(
        &mut self,
        inputs: &[(KvSlot, usize)],
        slot_views: &[KvSlotView],
        kv_pool: &mut KvCachePool,
    ) -> Result<Vec<Vec<f32>>, String> {
        let batch_size = inputs.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let vocab_size = self.config.vocab_size;
        let num_layers = self.config.num_hidden_layers;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let num_kv_groups = (num_heads / num_kv_heads) as u32;
        let ple_dim = self.config.hidden_size_per_layer_input;
        let ple_total_dim = num_layers * ple_dim;
        let eps = self.config.rms_norm_eps as f32;
        let context_proj_scale = 1.0f32 / (hidden_size as f32).sqrt();
        let ple_input_scale = 1.0f32 / 2.0f32.sqrt();

        let cmd = self.ctx.queue.new_command_buffer();

        {
            let encoder = cmd.new_compute_command_encoder();
            for batch_idx in 0..batch_size {
                let offsets = self.decode_batch_row_offsets(batch_idx);
                self.ctx.encode_matvec_q4_at(
                    encoder,
                    &self.per_layer_model_projection_weight,
                    &self.decode_batch_scratch.hidden_buf,
                    offsets.hidden,
                    &self.decode_batch_scratch.ple_context_proj_buf,
                    offsets.ple_row,
                    ple_total_dim as u32,
                    hidden_size as u32,
                );
                self.ctx.encode_vec_scale_at(
                    encoder,
                    &self.decode_batch_scratch.ple_context_proj_buf,
                    offsets.ple_row,
                    &self.decode_batch_scratch.ple_combined_buf,
                    offsets.ple_row,
                    ple_total_dim as u32,
                    context_proj_scale,
                );
                self.ctx.encode_rmsnorm_per_head_at(
                    encoder,
                    &self.decode_batch_scratch.ple_combined_buf,
                    offsets.ple_row,
                    &self.per_layer_projection_norm_weight,
                    &self.decode_batch_scratch.ple_context_proj_buf,
                    offsets.ple_row,
                    num_layers as u32,
                    ple_dim as u32,
                    eps,
                );
                self.ctx.encode_vec_add_at(
                    encoder,
                    &self.decode_batch_scratch.ple_context_proj_buf,
                    offsets.ple_row,
                    &self.decode_batch_scratch.ple_token_id_buf,
                    offsets.ple_row,
                    &self.decode_batch_scratch.ple_combined_buf,
                    offsets.ple_row,
                    ple_total_dim as u32,
                );
                self.ctx.encode_vec_scale_at(
                    encoder,
                    &self.decode_batch_scratch.ple_combined_buf,
                    offsets.ple_row,
                    &self.decode_batch_scratch.ple_context_proj_buf,
                    offsets.ple_row,
                    ple_total_dim as u32,
                    ple_input_scale,
                );
            }
            encoder.end_encoding();
        }

        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            let head_dim = layer.head_dim;
            let q_out = layer.q_out_dim;
            let kv_out = layer.kv_out_dim;
            let scale = 1.0f32;

            let encoder = cmd.new_compute_command_encoder();

            for (batch_idx, slot_view) in slot_views.iter().enumerate() {
                let offsets = self.decode_batch_row_offsets(batch_idx);
                let append_pos = slot_view.seq_len;
                let attn_kv_seq = append_pos + 1;
                let effective_kv_seq = if layer.is_full_attention {
                    attn_kv_seq
                } else {
                    attn_kv_seq.min(self.config.sliding_window as u32)
                };
                let kv_start = if !layer.is_full_attention
                    && attn_kv_seq > self.config.sliding_window as u32
                {
                    attn_kv_seq - self.config.sliding_window as u32
                } else {
                    0
                };
                let rotary_offset = Self::f32_byte_offset(batch_idx * head_dim);
                let ple_layer_offset = self.decode_batch_layer_ple_offset(batch_idx, layer_idx);

                self.ctx.encode_copy_at(
                    encoder,
                    &self.decode_batch_scratch.hidden_buf,
                    offsets.hidden,
                    &self.decode_batch_scratch.residual_buf,
                    offsets.hidden,
                    hidden_size as u32,
                );
                self.ctx.encode_rmsnorm_at(
                    encoder,
                    &self.decode_batch_scratch.hidden_buf,
                    offsets.hidden,
                    &layer.input_layernorm_weight,
                    &self.decode_batch_scratch.normed_buf,
                    offsets.hidden,
                    hidden_size as u32,
                    eps,
                );

                if layer.use_f16 {
                    self.ctx.encode_matvec_f16_at(
                        encoder,
                        &layer.q_proj,
                        &self.decode_batch_scratch.normed_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.q_buf,
                        offsets.q,
                        q_out as u32,
                        hidden_size as u32,
                    );
                } else {
                    self.ctx.encode_matvec_q4_at(
                        encoder,
                        &layer.q_proj,
                        &self.decode_batch_scratch.normed_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.q_buf,
                        offsets.q,
                        q_out as u32,
                        hidden_size as u32,
                    );
                }
                self.ctx.encode_rmsnorm_per_head_at(
                    encoder,
                    &self.decode_batch_scratch.q_buf,
                    offsets.q,
                    &layer.q_norm_weight,
                    &self.decode_batch_scratch.q_normed_buf,
                    offsets.q,
                    num_heads as u32,
                    head_dim as u32,
                    eps,
                );
                self.ctx.encode_rotary_at(
                    encoder,
                    &self.decode_batch_scratch.q_normed_buf,
                    offsets.q,
                    &self.decode_batch_scratch.k_normed_buf,
                    offsets.kv,
                    &self.per_layer_decode_batch_cos_bufs[layer_idx],
                    rotary_offset,
                    &self.per_layer_decode_batch_sin_bufs[layer_idx],
                    rotary_offset,
                    num_heads as u32,
                    0,
                    head_dim as u32,
                );

                if layer.has_kv {
                    if layer.use_f16 {
                        self.ctx.encode_matvec_f16_at(
                            encoder,
                            &layer.k_proj,
                            &self.decode_batch_scratch.normed_buf,
                            offsets.hidden,
                            &self.decode_batch_scratch.k_buf,
                            offsets.kv,
                            kv_out as u32,
                            hidden_size as u32,
                        );
                        self.ctx.encode_matvec_f16_at(
                            encoder,
                            &layer.v_proj,
                            &self.decode_batch_scratch.normed_buf,
                            offsets.hidden,
                            &self.decode_batch_scratch.v_buf,
                            offsets.kv,
                            kv_out as u32,
                            hidden_size as u32,
                        );
                    } else {
                        self.ctx.encode_matvec_q4_at(
                            encoder,
                            &layer.k_proj,
                            &self.decode_batch_scratch.normed_buf,
                            offsets.hidden,
                            &self.decode_batch_scratch.k_buf,
                            offsets.kv,
                            kv_out as u32,
                            hidden_size as u32,
                        );
                        self.ctx.encode_matvec_q4_at(
                            encoder,
                            &layer.v_proj,
                            &self.decode_batch_scratch.normed_buf,
                            offsets.hidden,
                            &self.decode_batch_scratch.v_buf,
                            offsets.kv,
                            kv_out as u32,
                            hidden_size as u32,
                        );
                    }

                    self.ctx.encode_rmsnorm_per_head_at(
                        encoder,
                        &self.decode_batch_scratch.k_buf,
                        offsets.kv,
                        &layer.k_norm_weight,
                        &self.decode_batch_scratch.k_normed_buf,
                        offsets.kv,
                        num_kv_heads as u32,
                        head_dim as u32,
                        eps,
                    );
                    self.ctx.encode_rotary_at(
                        encoder,
                        &self.decode_batch_scratch.q_buf,
                        offsets.q,
                        &self.decode_batch_scratch.k_normed_buf,
                        offsets.kv,
                        &self.per_layer_decode_batch_cos_bufs[layer_idx],
                        rotary_offset,
                        &self.per_layer_decode_batch_sin_bufs[layer_idx],
                        rotary_offset,
                        0,
                        num_kv_heads as u32,
                        head_dim as u32,
                    );
                    self.ctx.encode_rmsnorm_per_head_noweight_at(
                        encoder,
                        &self.decode_batch_scratch.v_buf,
                        offsets.kv,
                        &self.decode_batch_scratch.gate_buf,
                        offsets.intermediate,
                        num_kv_heads as u32,
                        head_dim as u32,
                        eps,
                    );

                    let k_cache = kv_pool
                        .layer_k_cache(slot_view.slot, layer_idx)
                        .map_err(|err| err.to_string())?;
                    let v_cache = kv_pool
                        .layer_v_cache(slot_view.slot, layer_idx)
                        .map_err(|err| err.to_string())?;
                    self.ctx.encode_kv_append_f16_at(
                        encoder,
                        &self.decode_batch_scratch.k_normed_buf,
                        offsets.kv,
                        k_cache,
                        num_kv_heads as u32,
                        head_dim as u32,
                        kv_pool.capacity(),
                        append_pos,
                    );
                    self.ctx.encode_kv_append_f16_at(
                        encoder,
                        &self.decode_batch_scratch.gate_buf,
                        offsets.intermediate,
                        v_cache,
                        num_kv_heads as u32,
                        head_dim as u32,
                        kv_pool.capacity(),
                        append_pos,
                    );
                }

                let k_cache = kv_pool
                    .layer_k_cache(slot_view.slot, layer.kv_source_layer)
                    .map_err(|err| err.to_string())?;
                let v_cache = kv_pool
                    .layer_v_cache(slot_view.slot, layer.kv_source_layer)
                    .map_err(|err| err.to_string())?;
                self.ctx.encode_attention_with_offset_f16_at(
                    encoder,
                    &self.decode_batch_scratch.q_normed_buf,
                    offsets.q,
                    k_cache,
                    v_cache,
                    &self.decode_batch_scratch.attn_out_buf,
                    offsets.q,
                    num_heads as u32,
                    num_kv_heads as u32,
                    num_kv_groups,
                    head_dim as u32,
                    effective_kv_seq,
                    kv_pool.capacity(),
                    scale,
                    kv_start,
                );

                self.ctx.encode_matvec_f16_at(
                    encoder,
                    &layer.o_proj,
                    &self.decode_batch_scratch.attn_out_buf,
                    offsets.q,
                    &self.decode_batch_scratch.o_out_buf,
                    offsets.hidden,
                    hidden_size as u32,
                    q_out as u32,
                );
                self.ctx.encode_rmsnorm_at(
                    encoder,
                    &self.decode_batch_scratch.o_out_buf,
                    offsets.hidden,
                    &layer.post_attention_layernorm_weight,
                    &self.decode_batch_scratch.normed_buf,
                    offsets.hidden,
                    hidden_size as u32,
                    eps,
                );
                self.ctx.encode_vec_add_at(
                    encoder,
                    &self.decode_batch_scratch.residual_buf,
                    offsets.hidden,
                    &self.decode_batch_scratch.normed_buf,
                    offsets.hidden,
                    &self.decode_batch_scratch.hidden_buf,
                    offsets.hidden,
                    hidden_size as u32,
                );

                self.ctx.encode_copy_at(
                    encoder,
                    &self.decode_batch_scratch.hidden_buf,
                    offsets.hidden,
                    &self.decode_batch_scratch.residual_buf,
                    offsets.hidden,
                    hidden_size as u32,
                );
                self.ctx.encode_rmsnorm_at(
                    encoder,
                    &self.decode_batch_scratch.hidden_buf,
                    offsets.hidden,
                    &layer.pre_feedforward_layernorm_weight,
                    &self.decode_batch_scratch.normed_buf,
                    offsets.hidden,
                    hidden_size as u32,
                    eps,
                );
                if layer.use_f16 {
                    self.ctx.encode_matvec_f16_at(
                        encoder,
                        &layer.gate_proj,
                        &self.decode_batch_scratch.normed_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.gate_buf,
                        offsets.intermediate,
                        intermediate_size as u32,
                        hidden_size as u32,
                    );
                    self.ctx.encode_matvec_f16_at(
                        encoder,
                        &layer.up_proj,
                        &self.decode_batch_scratch.normed_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.up_buf,
                        offsets.intermediate,
                        intermediate_size as u32,
                        hidden_size as u32,
                    );
                } else {
                    self.ctx.encode_matvec_q4_at(
                        encoder,
                        &layer.gate_proj,
                        &self.decode_batch_scratch.normed_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.gate_buf,
                        offsets.intermediate,
                        intermediate_size as u32,
                        hidden_size as u32,
                    );
                    self.ctx.encode_matvec_q4_at(
                        encoder,
                        &layer.up_proj,
                        &self.decode_batch_scratch.normed_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.up_buf,
                        offsets.intermediate,
                        intermediate_size as u32,
                        hidden_size as u32,
                    );
                }
                self.ctx.encode_gelu_mul_at(
                    encoder,
                    &self.decode_batch_scratch.gate_buf,
                    offsets.intermediate,
                    &self.decode_batch_scratch.up_buf,
                    offsets.intermediate,
                    &self.decode_batch_scratch.gelu_buf,
                    offsets.intermediate,
                    intermediate_size as u32,
                );
                if layer.use_f16 {
                    self.ctx.encode_matvec_f16_at(
                        encoder,
                        &layer.down_proj,
                        &self.decode_batch_scratch.gelu_buf,
                        offsets.intermediate,
                        &self.decode_batch_scratch.down_buf,
                        offsets.hidden,
                        hidden_size as u32,
                        intermediate_size as u32,
                    );
                } else {
                    self.ctx.encode_matvec_q4_at(
                        encoder,
                        &layer.down_proj,
                        &self.decode_batch_scratch.gelu_buf,
                        offsets.intermediate,
                        &self.decode_batch_scratch.down_buf,
                        offsets.hidden,
                        hidden_size as u32,
                        intermediate_size as u32,
                    );
                }
                self.ctx.encode_rmsnorm_at(
                    encoder,
                    &self.decode_batch_scratch.down_buf,
                    offsets.hidden,
                    &layer.post_feedforward_layernorm_weight,
                    &self.decode_batch_scratch.normed_buf,
                    offsets.hidden,
                    hidden_size as u32,
                    eps,
                );
                self.ctx.encode_vec_add_at(
                    encoder,
                    &self.decode_batch_scratch.residual_buf,
                    offsets.hidden,
                    &self.decode_batch_scratch.normed_buf,
                    offsets.hidden,
                    &self.decode_batch_scratch.hidden_buf,
                    offsets.hidden,
                    hidden_size as u32,
                );

                self.ctx.encode_copy_at(
                    encoder,
                    &self.decode_batch_scratch.hidden_buf,
                    offsets.hidden,
                    &self.decode_batch_scratch.residual_buf,
                    offsets.hidden,
                    hidden_size as u32,
                );
                self.ctx.encode_matvec_f16_at(
                    encoder,
                    &layer.per_layer_input_gate_weight,
                    &self.decode_batch_scratch.hidden_buf,
                    offsets.hidden,
                    &self.decode_batch_scratch.gate_buf,
                    offsets.intermediate,
                    ple_dim as u32,
                    hidden_size as u32,
                );
                self.ctx.encode_gelu_mul_at(
                    encoder,
                    &self.decode_batch_scratch.gate_buf,
                    offsets.intermediate,
                    &self.decode_batch_scratch.ple_context_proj_buf,
                    ple_layer_offset,
                    &self.decode_batch_scratch.up_buf,
                    offsets.intermediate,
                    ple_dim as u32,
                );
                self.ctx.encode_matvec_f16_at(
                    encoder,
                    &layer.per_layer_projection_weight,
                    &self.decode_batch_scratch.up_buf,
                    offsets.intermediate,
                    &self.decode_batch_scratch.o_out_buf,
                    offsets.hidden,
                    hidden_size as u32,
                    ple_dim as u32,
                );
                self.ctx.encode_rmsnorm_at(
                    encoder,
                    &self.decode_batch_scratch.o_out_buf,
                    offsets.hidden,
                    &layer.post_per_layer_input_norm_weight,
                    &self.decode_batch_scratch.normed_buf,
                    offsets.hidden,
                    hidden_size as u32,
                    eps,
                );
                self.ctx.encode_vec_add_at(
                    encoder,
                    &self.decode_batch_scratch.residual_buf,
                    offsets.hidden,
                    &self.decode_batch_scratch.normed_buf,
                    offsets.hidden,
                    &self.decode_batch_scratch.hidden_buf,
                    offsets.hidden,
                    hidden_size as u32,
                );
                self.ctx.encode_vec_scale_at(
                    encoder,
                    &self.decode_batch_scratch.hidden_buf,
                    offsets.hidden,
                    &self.decode_batch_scratch.residual_buf,
                    offsets.hidden,
                    hidden_size as u32,
                    layer.layer_scalar,
                );
                self.ctx.encode_copy_at(
                    encoder,
                    &self.decode_batch_scratch.residual_buf,
                    offsets.hidden,
                    &self.decode_batch_scratch.hidden_buf,
                    offsets.hidden,
                    hidden_size as u32,
                );
            }

            encoder.end_encoding();
        }

        {
            let encoder = cmd.new_compute_command_encoder();
            self.ctx.encode_rmsnorm_batch(
                encoder,
                &self.decode_batch_scratch.hidden_buf,
                &self.final_norm_weight,
                &self.decode_batch_scratch.normed_buf,
                hidden_size as u32,
                eps,
                batch_size as u32,
            );
            for batch_idx in 0..batch_size {
                let offsets = self.decode_batch_row_offsets(batch_idx);
                let logits_offset = Self::f32_byte_offset(batch_idx * vocab_size);
                self.ctx.encode_matvec_q4_at(
                    encoder,
                    &self.lm_head_buf,
                    &self.decode_batch_scratch.normed_buf,
                    offsets.hidden,
                    &self.decode_batch_scratch.logits_buf,
                    logits_offset,
                    vocab_size as u32,
                    hidden_size as u32,
                );
            }
            encoder.end_encoding();
        }

        cmd.commit();
        cmd.wait_until_completed();

        let mut logits_batch = MetalContext::read_buffer(
            &self.decode_batch_scratch.logits_buf,
            batch_size * vocab_size,
        );
        let cap = self.config.final_logit_softcapping;
        let mut outputs = Vec::with_capacity(batch_size);
        for batch_idx in 0..batch_size {
            let start = batch_idx * vocab_size;
            let end = start + vocab_size;
            for logit in &mut logits_batch[start..end] {
                let x = (*logit / cap).clamp(-10.0, 10.0);
                *logit = cap * x.tanh();
            }
            outputs.push(logits_batch[start..end].to_vec());
        }

        for slot_view in slot_views {
            kv_pool
                .with_slot_mut(slot_view.slot, |slot_state| {
                    slot_state.seq_len += 1;
                    slot_state.total_tokens += 1;
                })
                .map_err(|err| err.to_string())?;
        }

        Ok(outputs)
    }

    /// Batched prefill: process all prompt tokens sequentially.
    pub fn forward_prefill(&mut self, token_ids: &[usize]) -> Vec<f32> {
        let mut logits = Vec::new();
        for &tid in token_ids {
            logits = self.forward_single_token(tid);
        }
        logits
    }

    pub fn forward_single_token_with_kv_slot(
        &mut self,
        token_id: usize,
        kv_pool: &mut KvCachePool,
        slot: KvSlot,
    ) -> Result<Vec<f32>, KvPoolError> {
        let pool_capacity = kv_pool.capacity();
        kv_pool.with_slot_mut(slot, |slot_state| {
            std::mem::swap(&mut self.k_cache, &mut slot_state.k_cache);
            std::mem::swap(&mut self.v_cache, &mut slot_state.v_cache);

            let legacy_kv_seq_len = self.kv_seq_len;
            let legacy_total_tokens = self.total_tokens;
            let legacy_kv_capacity = self.kv_capacity;
            self.kv_seq_len = slot_state.seq_len;
            self.total_tokens = slot_state.total_tokens;
            self.kv_capacity = pool_capacity;

            let logits = self.forward_single_token(token_id);

            slot_state.seq_len = self.kv_seq_len;
            slot_state.total_tokens = self.total_tokens;
            self.kv_seq_len = legacy_kv_seq_len;
            self.total_tokens = legacy_total_tokens;
            self.kv_capacity = legacy_kv_capacity;

            std::mem::swap(&mut self.v_cache, &mut slot_state.v_cache);
            std::mem::swap(&mut self.k_cache, &mut slot_state.k_cache);

            logits
        })
    }

    pub fn forward_decode_batch_with_kv_slots(
        &mut self,
        inputs: &[(KvSlot, usize)],
        kv_pool: &mut KvCachePool,
    ) -> Vec<Result<Vec<f32>, String>> {
        if inputs.len() > self.max_decode_batch_size() {
            return inputs
                .iter()
                .map(|_| {
                    Err(format!(
                        "decode batch has {} requests, max supported batch is {}",
                        inputs.len(),
                        self.max_decode_batch_size()
                    ))
                })
                .collect();
        }

        let token_ids: Vec<usize> = inputs.iter().map(|&(_, token_id)| token_id).collect();
        if let Err(message) = self.prepare_decode_batch_inputs(&token_ids) {
            return inputs.iter().map(|_| Err(message.clone())).collect();
        }

        let slots: Vec<KvSlot> = inputs.iter().map(|&(slot, _)| slot).collect();
        let slot_views = match kv_pool.slot_views(&slots) {
            Ok(slot_views) => slot_views,
            Err(err) => return inputs.iter().map(|_| Err(err.to_string())).collect(),
        };
        if let Err(message) = self.prepare_decode_batch_rotary(&slot_views) {
            return inputs.iter().map(|_| Err(message.clone())).collect();
        }
        if let Err(message) = self.prepare_decode_batch_attention_metadata(&slot_views) {
            return inputs.iter().map(|_| Err(message.clone())).collect();
        }

        if inputs.len() > 1 {
            return match self.forward_decode_batch_encoded_with_kv_slots(
                inputs,
                &slot_views,
                kv_pool,
            ) {
                Ok(outputs) => outputs.into_iter().map(Ok).collect(),
                Err(message) => inputs.iter().map(|_| Err(message.clone())).collect(),
            };
        }

        inputs
            .iter()
            .map(|&(slot, token_id)| {
                self.forward_single_token_with_kv_slot(token_id, kv_pool, slot)
                    .map_err(|err| err.to_string())
            })
            .collect()
    }

    pub fn forward_prefill_batch_with_kv_slots(
        &mut self,
        inputs: &[(KvSlot, &[usize])],
        kv_pool: &mut KvCachePool,
    ) -> Vec<Result<Vec<f32>, String>> {
        if inputs.is_empty() {
            return Vec::new();
        }
        if inputs.len() == 1 {
            return inputs
                .iter()
                .map(|&(slot, token_ids)| {
                    self.forward_prefill_chunk_with_kv_slot(token_ids, kv_pool, slot)
                })
                .collect();
        }

        let total_seq_len: usize = inputs.iter().map(|(_, token_ids)| token_ids.len()).sum();
        if total_seq_len == 0 {
            return inputs
                .iter()
                .map(|_| Err("prefill token_ids must not be empty".to_string()))
                .collect();
        }
        if total_seq_len > self.max_parallel_prefill_seq() {
            let message = format!(
                "prefill batch has {} total tokens, max supported batch is {}",
                total_seq_len,
                self.max_parallel_prefill_seq()
            );
            return inputs.iter().map(|_| Err(message.clone())).collect();
        }

        let mut flat_tokens = Vec::with_capacity(total_seq_len);
        let mut segments = Vec::with_capacity(inputs.len());
        let mut rotary_segments = Vec::with_capacity(inputs.len());
        let mut row_start = 0usize;

        for &(slot, token_ids) in inputs {
            if token_ids.is_empty() {
                return inputs
                    .iter()
                    .map(|_| Err("prefill token_ids must not be empty".to_string()))
                    .collect();
            }

            let start_pos = match kv_pool.total_tokens(slot) {
                Ok(start_pos) => start_pos,
                Err(err) => return inputs.iter().map(|_| Err(err.to_string())).collect(),
            };
            if start_pos + token_ids.len() > kv_pool.capacity() as usize {
                let message = format!(
                    "KV Cache overflow. Max length {}, current {}, new {}",
                    kv_pool.capacity(),
                    start_pos,
                    token_ids.len()
                );
                return inputs.iter().map(|_| Err(message.clone())).collect();
            }

            flat_tokens.extend_from_slice(token_ids);
            segments.push(PrefillBatchSegment {
                slot,
                row_start,
                token_count: token_ids.len(),
                start_pos,
            });
            rotary_segments.push((start_pos, token_ids.len()));
            row_start += token_ids.len();
        }

        if let Err(message) = self.prepare_parallel_prefill_inputs(&flat_tokens) {
            return inputs.iter().map(|_| Err(message.clone())).collect();
        }
        if let Err(message) = self.prepare_parallel_prefill_rotary_segments(&rotary_segments) {
            return inputs.iter().map(|_| Err(message.clone())).collect();
        }
        if let Err(message) = self.encode_parallel_prefill_ple_context(total_seq_len) {
            return inputs.iter().map(|_| Err(message.clone())).collect();
        }

        for layer_idx in 0..self.layers.len() {
            if let Err(message) = self.encode_parallel_prefill_layer_batched(
                layer_idx,
                total_seq_len,
                &segments,
                kv_pool,
            ) {
                return inputs.iter().map(|_| Err(message.clone())).collect();
            }
        }

        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let eps = self.config.rms_norm_eps as f32;
        let cmd = self.ctx.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        self.ctx.encode_rmsnorm_batch(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &self.final_norm_weight,
            &self.prefill_scratch.normed_buf,
            hidden_size as u32,
            eps,
            total_seq_len as u32,
        );
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let mut outputs = Vec::with_capacity(segments.len());
        for segment in &segments {
            let last_offsets =
                self.prefill_row_offsets(segment.row_start + segment.token_count - 1);
            let cmd = self.ctx.queue.new_command_buffer();
            let encoder = cmd.new_compute_command_encoder();
            self.ctx.encode_matvec_q4_at(
                encoder,
                &self.lm_head_buf,
                &self.prefill_scratch.normed_buf,
                last_offsets.hidden,
                &self.prefill_scratch.logits_buf,
                0,
                vocab_size as u32,
                hidden_size as u32,
            );
            encoder.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            let mut logits =
                MetalContext::read_buffer(&self.prefill_scratch.logits_buf, vocab_size);
            let cap = self.config.final_logit_softcapping;
            for logit in &mut logits {
                let x = (*logit / cap).clamp(-10.0, 10.0);
                *logit = cap * x.tanh();
            }
            outputs.push(Ok(logits));
        }

        for segment in &segments {
            if let Err(err) = kv_pool.with_slot_mut(segment.slot, |slot_state| {
                slot_state.seq_len += segment.token_count as u32;
                slot_state.total_tokens += segment.token_count;
            }) {
                return inputs.iter().map(|_| Err(err.to_string())).collect();
            }
        }

        outputs
    }

    pub fn forward_prefill_with_kv_slot(
        &mut self,
        token_ids: &[usize],
        kv_pool: &mut KvCachePool,
        slot: KvSlot,
    ) -> Result<Vec<f32>, KvPoolError> {
        let mut logits = Vec::new();
        for &tid in token_ids {
            logits = self.forward_single_token_with_kv_slot(tid, kv_pool, slot)?;
        }
        Ok(logits)
    }

    pub fn forward_prefill_chunked_with_kv_slot(
        &mut self,
        token_ids: &[usize],
        kv_pool: &mut KvCachePool,
        slot: KvSlot,
    ) -> Result<Vec<f32>, String> {
        if token_ids.is_empty() {
            return Err("prefill token_ids must not be empty".to_string());
        }

        let mut logits = Vec::new();
        let chunk_size = self.max_parallel_prefill_seq().max(1);

        for chunk in token_ids.chunks(chunk_size) {
            logits = self.forward_prefill_chunk_with_kv_slot(chunk, kv_pool, slot)?;
        }

        Ok(logits)
    }

    pub fn forward_prefill_chunk_with_kv_slot(
        &mut self,
        token_ids: &[usize],
        kv_pool: &mut KvCachePool,
        slot: KvSlot,
    ) -> Result<Vec<f32>, String> {
        self.prepare_parallel_prefill_inputs(token_ids)?;
        let start_pos = kv_pool.total_tokens(slot).map_err(|err| err.to_string())?;
        self.prepare_parallel_prefill_rotary(start_pos, token_ids.len())?;

        if self.can_use_parallel_prefill_chunk(start_pos, token_ids.len(), kv_pool) {
            return self
                .forward_prefill_chunk_parallel_with_kv_slot(token_ids, kv_pool, slot, start_pos);
        }

        let mut logits = Vec::new();
        for &tid in token_ids {
            logits = self
                .forward_single_token_with_kv_slot(tid, kv_pool, slot)
                .map_err(|err| err.to_string())?;
        }
        Ok(logits)
    }
}

/// Decode a safetensors tensor view to Vec<f32>, handling f32/f16/bf16.
fn decode_tensor_to_f32(tensor_view: &safetensors::tensor::TensorView) -> Vec<f32> {
    let dtype = tensor_view.dtype();
    let raw_data = tensor_view.data();

    match dtype {
        safetensors::Dtype::F32 => raw_data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        safetensors::Dtype::F16 => raw_data
            .chunks_exact(2)
            .map(|b| {
                let bits = u16::from_le_bytes([b[0], b[1]]);
                half_to_f32(bits)
            })
            .collect(),
        safetensors::Dtype::BF16 => raw_data
            .chunks_exact(2)
            .map(|b| {
                let bits = u16::from_le_bytes([b[0], b[1]]);
                bf16_to_f32(bits)
            })
            .collect(),
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

/// Convert raw bytes to Vec<u16> (for storing bf16/f16 data compactly).
fn raw_to_u16(data: &[u8]) -> Vec<u16> {
    data.chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}
