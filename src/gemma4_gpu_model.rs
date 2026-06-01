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

    // Embedding table (Q4 quantized for memory efficiency)
    pub embed_tokens_f16: Vec<u16>,  // [vocab_size * hidden_size] as bf16 (~1.3GB)
    // Per-layer embedding table (Q4 quantized, ~1.6GB)
    pub embed_tokens_per_layer_f16: Vec<u16>,  // [vocab_size * num_layers * ple_dim] as bf16

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

    // PLE scratch buffers
    pub ple_embed_buf: Buffer,    // [ple_dim] = 256 (unused now, kept for compat)
    pub ple_gated_buf: Buffer,    // [ple_dim]
    pub ple_normed_buf: Buffer,   // [ple_dim]
    pub ple_projected_buf: Buffer, // [hidden_size]
    pub ple_context_proj_buf: Buffer, // [num_layers * ple_dim] for context projection
    pub ple_token_id_buf: Buffer,     // [num_layers * ple_dim] token identity embedding
    pub ple_combined_buf: Buffer,     // [num_layers * ple_dim] combined output

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
    pub per_layer_ple_bufs: Vec<Buffer>,

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
    pub has_kv: bool,           // false for shared KV layers (layers 24-41)
    pub kv_source_layer: usize, // which layer's KV cache to use
    pub head_dim: usize,
    pub q_out_dim: usize,
    pub kv_out_dim: usize,
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
            let safetensors = SafeTensors::deserialize(&mmap)
                .expect("Failed to deserialize safetensors");

            for (name, tensor_view) in safetensors.tensors() {
                let clean_name = name.strip_prefix(prefix).unwrap_or(&name);

                if clean_name == "embed_tokens.weight" && embed_tokens_f16.is_empty() {
                    embed_tokens_f16 = raw_to_u16(tensor_view.data());
                    println!("    embed_tokens: {:?} (kept as f16, {:.1} MB)",
                             tensor_view.shape(), embed_tokens_f16.len() * 2 / 1024 / 1024);
                } else if clean_name == "embed_tokens_per_layer.weight" && embed_tokens_per_layer_f16.is_empty() {
                    embed_tokens_per_layer_f16 = raw_to_u16(tensor_view.data());
                    println!("    embed_tokens_per_layer: {:?} (kept as f16, {:.1} MB)",
                             tensor_view.shape(), embed_tokens_per_layer_f16.len() * 2 / 1024 / 1024);
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
        assert!(!embed_tokens_per_layer_f16.is_empty(), "embed_tokens_per_layer not found");

        // Create GPU buffer for lm_head (tied embeddings — quantize to Q4_0)
        // Convert bf16 → f32 first, then quantize to Q4_0
        let lm_head_f32: Vec<f32> = embed_tokens_f16.iter()
            .map(|&bits| bf16_to_f32(bits))
            .collect();
        let lm_head_buf = ctx.buffer_from_f32_as_q4(&lm_head_f32, vocab_size, hidden_size);
        println!("    lm_head (tied, Q4_0 on GPU): {:.1} MB",
                 lm_head_buf.length() as f64 / 1024.0 / 1024.0);

        let final_norm_weight = ctx.buffer_from_slice(&final_norm_data);
        let per_layer_projection_norm_weight = ctx.buffer_from_slice(&per_layer_proj_norm_data);
        let per_layer_model_projection_weight = if !per_layer_model_proj_data.is_empty() {
            ctx.buffer_from_f32_as_q4(&per_layer_model_proj_data, num_layers * ple_dim, hidden_size)
        } else {
            // Fallback: create empty buffer (shouldn't happen for E4B)
            println!("  WARNING: per_layer_model_projection not found, PLE context projection disabled");
            ctx.buffer_empty(1)
        };

        // Load all layers
        let num_layers_to_load = num_layers;
        println!("  Loading layers (Q4_0 quantized, {} layers)...", num_layers_to_load);
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
                let safetensors = SafeTensors::deserialize(&mmap)
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
                has_kv: layer_idx < (num_layers - config.num_kv_shared_layers),
                kv_source_layer: 0, // will be computed below
                head_dim,
                q_out_dim: q_out,
                kv_out_dim: kv_out,
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

        println!("  KV sharing: layers 0-{} have own KV, layers {}-{} share",
                 first_kv_shared - 1, first_kv_shared, num_layers - 1);

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
            k_cache.push(ctx.device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared));
            v_cache.push(ctx.device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared));
        }

        // Rotary buffers (allocate for max head_dim)
        let cos_buf = ctx.buffer_empty(max_head_dim);
        let sin_buf = ctx.buffer_empty(max_head_dim);

        // Per-layer persistent buffers for cos/sin/ple (allocated once, overwritten each token)
        let mut per_layer_cos_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_sin_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_ple_bufs = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let hd = config.layer_head_dim(i);
            per_layer_cos_bufs.push(ctx.buffer_empty(hd));
            per_layer_sin_bufs.push(ctx.buffer_empty(hd));
            per_layer_ple_bufs.push(ctx.buffer_empty(ple_dim));
        }

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
            std::slice::from_raw_parts(self.lm_head_buf.contents() as *const u8, lm_head_len as usize)
        };
        file.write_all(lm_head_bytes).unwrap();

        // Save per_layer_model_projection (Q4 on GPU)
        let proj_len = self.per_layer_model_projection_weight.length() as u64;
        file.write_all(&proj_len.to_le_bytes()).unwrap();
        let proj_bytes = unsafe {
            std::slice::from_raw_parts(self.per_layer_model_projection_weight.contents() as *const u8, proj_len as usize)
        };
        file.write_all(proj_bytes).unwrap();

        // Save norms
        let norm_len = self.final_norm_weight.length() as u64;
        file.write_all(&norm_len.to_le_bytes()).unwrap();
        let norm_bytes = unsafe {
            std::slice::from_raw_parts(self.final_norm_weight.contents() as *const u8, norm_len as usize)
        };
        file.write_all(norm_bytes).unwrap();

        let pnorm_len = self.per_layer_projection_norm_weight.length() as u64;
        file.write_all(&pnorm_len.to_le_bytes()).unwrap();
        let pnorm_bytes = unsafe {
            std::slice::from_raw_parts(self.per_layer_projection_norm_weight.contents() as *const u8, pnorm_len as usize)
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
                let bytes = unsafe { std::slice::from_raw_parts(buf.contents() as *const u8, len as usize) };
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
            file.write_all(&(layer.is_full_attention as u8).to_le_bytes()).unwrap();
            file.write_all(&(layer.has_kv as u8).to_le_bytes()).unwrap();
            file.write_all(&(layer.kv_source_layer as u32).to_le_bytes()).unwrap();
            file.write_all(&(layer.head_dim as u32).to_le_bytes()).unwrap();
            file.write_all(&(layer.q_out_dim as u32).to_le_bytes()).unwrap();
            file.write_all(&(layer.kv_out_dim as u32).to_le_bytes()).unwrap();
        }
        println!("  Cache saved: {:.1} MB", file.metadata().map(|m| m.len()).unwrap_or(0) as f64 / 1024.0 / 1024.0);
    }

    /// Load model from pre-quantized cache file (fast path).
    fn load_from_cache(model_dir: &str, cache_path: &Path) -> Self {
        use std::io::Read;

        let config_path = Path::new(model_dir).join("config.json");
        let config_str = fs::read_to_string(&config_path).expect("Failed to read config.json");
        let outer: serde_json::Value = serde_json::from_str(&config_str).expect("Failed to parse config.json");
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
            device.new_buffer_with_data(data.as_ptr() as *const _, len, MTLResourceOptions::StorageModeShared)
        };

        // Load embeddings
        let embed_len = read_len(&mut file) as usize;
        let mut embed_bytes = vec![0u8; embed_len];
        file.read_exact(&mut embed_bytes).unwrap();
        let embed_tokens_f16: Vec<u16> = embed_bytes.chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        println!("    embed_tokens: {:.1} MB", embed_len as f64 / 1024.0 / 1024.0);

        // Load embed_tokens_per_layer
        let ple_embed_len = read_len(&mut file) as usize;
        let mut ple_bytes = vec![0u8; ple_embed_len];
        file.read_exact(&mut ple_bytes).unwrap();
        let embed_tokens_per_layer_f16: Vec<u16> = ple_bytes.chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        println!("    embed_tokens_per_layer: {:.1} MB", ple_embed_len as f64 / 1024.0 / 1024.0);

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
                q_proj, k_proj, v_proj, o_proj,
                gate_proj, up_proj, down_proj,
                input_layernorm_weight, post_attention_layernorm_weight,
                pre_feedforward_layernorm_weight, post_feedforward_layernorm_weight,
                post_per_layer_input_norm_weight,
                per_layer_input_gate_weight, per_layer_projection_weight,
                layer_scalar,
                q_norm_weight, k_norm_weight,
                is_full_attention, has_kv, kv_source_layer,
                head_dim, q_out_dim, kv_out_dim,
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
            k_cache.push(ctx.device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared));
            v_cache.push(ctx.device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared));
        }

        let cos_buf = ctx.buffer_empty(max_head_dim);
        let sin_buf = ctx.buffer_empty(max_head_dim);

        let mut per_layer_cos_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_sin_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_ple_bufs = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let hd = config.layer_head_dim(i);
            per_layer_cos_bufs.push(ctx.buffer_empty(hd));
            per_layer_sin_bufs.push(ctx.buffer_empty(hd));
            per_layer_ple_bufs.push(ctx.buffer_empty(ple_dim));
        }

        println!("  Loaded from Q4 cache successfully");

        Gemma4GpuModel {
            ctx, config,
            embed_tokens_f16, embed_tokens_per_layer_f16,
            lm_head_buf, layers,
            final_norm_weight, per_layer_projection_norm_weight,
            per_layer_model_projection_weight,
            hidden_buf, normed_buf, residual_buf,
            q_buf, k_buf, v_buf, attn_out_buf, o_out_buf,
            gate_buf, up_buf, gelu_buf, down_buf, logits_buf,
            ple_embed_buf, ple_gated_buf, ple_normed_buf, ple_projected_buf,
            ple_context_proj_buf, ple_token_id_buf, ple_combined_buf,
            q_normed_buf, k_normed_buf,
            k_cache, v_cache,
            kv_seq_len: 0, kv_capacity,
            cos_buf, sin_buf,
            per_layer_cos_bufs, per_layer_sin_bufs, per_layer_ple_bufs,
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
                ple_token_identity[i] = bf16_to_f32(self.embed_tokens_per_layer_f16[ple_token_offset + i]) * ple_scale;
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
                let inv_freq = 1.0 / (rope_theta.powf(i as f64 * 2.0 / head_dim as f64) as f32) / rope_factor as f32;
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
            MetalContext::write_buffer(&self.per_layer_cos_bufs[layer_idx], &rotary_cos_per_layer[layer_idx]);
            MetalContext::write_buffer(&self.per_layer_sin_bufs[layer_idx], &rotary_sin_per_layer[layer_idx]);
        }

        let cmd = self.ctx.queue.new_command_buffer();

        // ─── PLE pre-pass: encode into the same command buffer ───
        {
            let ple_enc = cmd.new_compute_command_encoder();

            // Step 2a: context_proj = per_layer_model_projection @ embed (GPU matvec)
            self.ctx.encode_matvec_q4(ple_enc, &self.per_layer_model_projection_weight,
                &self.hidden_buf, &self.ple_context_proj_buf,
                ple_total_dim as u32, hidden_size as u32);

            // Step 2b: context_proj *= 1/sqrt(hidden_size)
            self.ctx.encode_vec_scale(ple_enc, &self.ple_context_proj_buf,
                &self.ple_combined_buf, ple_total_dim as u32, context_proj_scale);

            // Step 2c: RMSNorm per layer (42 chunks of 256)
            self.ctx.encode_rmsnorm_per_head(ple_enc, &self.ple_combined_buf,
                &self.per_layer_projection_norm_weight, &self.ple_context_proj_buf,
                num_layers as u32, ple_dim as u32, eps);

            // Step 3: combined = (context_proj + token_identity) * 1/sqrt(2)
            self.ctx.encode_vec_add(ple_enc, &self.ple_context_proj_buf,
                &self.ple_token_id_buf, &self.ple_combined_buf, ple_total_dim as u32);
            self.ctx.encode_vec_scale(ple_enc, &self.ple_combined_buf,
                &self.ple_context_proj_buf, ple_total_dim as u32, ple_input_scale);

            // Copy per-layer slices from ple_context_proj_buf to per_layer_ple_bufs
            // (GPU-to-GPU copy, avoids CPU readback)
            for i in 0..actual_num_layers {
                let offset = (i * ple_dim * 4) as u64; // byte offset
                ple_enc.set_compute_pipeline_state(&self.ctx.buf_copy_pipeline);
                ple_enc.set_buffer(0, Some(&self.ple_context_proj_buf), offset);
                ple_enc.set_buffer(1, Some(&self.per_layer_ple_bufs[i]), 0);
                let n = ple_dim as u32;
                ple_enc.set_bytes(2, 4, &n as *const u32 as *const _);
                ple_enc.dispatch_threads(MTLSize::new(ple_dim as u64, 1, 1), MTLSize::new(256, 1, 1));
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
            self.ctx.encode_copy(encoder, &self.hidden_buf, &self.residual_buf, hidden_size as u32);

            // Pre-attention norm
            self.ctx.encode_rmsnorm(encoder, &self.hidden_buf,
                &layer.input_layernorm_weight, &self.normed_buf, hidden_size as u32, eps);

            // Q projection (always computed)
            self.ctx.encode_matvec_q4(encoder, &layer.q_proj, &self.normed_buf,
                &self.q_buf, q_out as u32, hidden_size as u32);

            // QK Norm on Q
            self.ctx.encode_rmsnorm_per_head(encoder, &self.q_buf, &layer.q_norm_weight,
                &self.q_normed_buf, num_heads as u32, head_dim as u32, eps);

            // Apply rotary to Q (full head_dim — non-rotary dims have cos=1, sin=0 for pass-through)
            self.ctx.encode_rotary(encoder, &self.q_normed_buf, &self.k_normed_buf,
                &self.per_layer_cos_bufs[layer_idx], &self.per_layer_sin_bufs[layer_idx], num_heads as u32, 0, head_dim as u32);

            // K, V only for non-shared layers
            if layer.has_kv {
                self.ctx.encode_matvec_q4(encoder, &layer.k_proj, &self.normed_buf,
                    &self.k_buf, kv_out as u32, hidden_size as u32);
                self.ctx.encode_matvec_q4(encoder, &layer.v_proj, &self.normed_buf,
                    &self.v_buf, kv_out as u32, hidden_size as u32);

                // K norm + rotary (full head_dim — non-rotary dims pass through)
                self.ctx.encode_rmsnorm_per_head(encoder, &self.k_buf, &layer.k_norm_weight,
                    &self.k_normed_buf, num_kv_heads as u32, head_dim as u32, eps);
                self.ctx.encode_rotary(encoder, &self.q_buf, &self.k_normed_buf,
                    &self.per_layer_cos_bufs[layer_idx], &self.per_layer_sin_bufs[layer_idx], 0, num_kv_heads as u32, head_dim as u32);

                // V norm (no weight)
                self.ctx.encode_rmsnorm_per_head_noweight(encoder, &self.v_buf, &self.gate_buf,
                    num_kv_heads as u32, head_dim as u32, eps);

                // Append to this layer's cache (f16)
                self.ctx.encode_kv_append_f16(encoder, &self.k_normed_buf, &self.k_cache[layer_idx],
                    num_kv_heads as u32, head_dim as u32, self.kv_capacity, kv_seq);
                self.ctx.encode_kv_append_f16(encoder, &self.gate_buf, &self.v_cache[layer_idx],
                    num_kv_heads as u32, head_dim as u32, self.kv_capacity, kv_seq);
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

            self.ctx.encode_attention_with_offset_f16(encoder,
                &self.q_normed_buf, &self.k_cache[layer.kv_source_layer], &self.v_cache[layer.kv_source_layer],
                &self.attn_out_buf,
                num_heads as u32, num_kv_heads as u32, num_kv_groups,
                head_dim as u32, effective_kv_seq, self.kv_capacity, scale, kv_start);

            // O projection
            self.ctx.encode_matvec_q4(encoder, &layer.o_proj, &self.attn_out_buf,
                &self.o_out_buf, hidden_size as u32, q_out as u32);

            // Post-attention norm (cannot be in-place, use normed_buf as temp)
            self.ctx.encode_rmsnorm(encoder, &self.o_out_buf,
                &layer.post_attention_layernorm_weight, &self.normed_buf,
                hidden_size as u32, eps);

            // Residual add: hidden = residual + normed_attn_output
            self.ctx.encode_vec_add(encoder, &self.residual_buf, &self.normed_buf,
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

            // Post-feedforward norm (cannot be in-place, use normed_buf as temp)
            self.ctx.encode_rmsnorm(encoder, &self.down_buf,
                &layer.post_feedforward_layernorm_weight, &self.normed_buf,
                hidden_size as u32, eps);

            // Residual add: hidden = residual + normed_mlp_output
            self.ctx.encode_vec_add(encoder, &self.residual_buf, &self.normed_buf,
                &self.hidden_buf, hidden_size as u32);

            // ─── Per-Layer Embedding (PLE) — after MLP, before layer_scalar ───
            // Save residual for PLE
            self.ctx.encode_copy(encoder, &self.hidden_buf, &self.residual_buf, hidden_size as u32);
            // Gate: ple_gated = per_layer_input_gate(hidden) → [ple_dim]
            self.ctx.encode_matvec_q4(encoder, &layer.per_layer_input_gate_weight,
                &self.hidden_buf, &self.ple_gated_buf, ple_dim as u32, hidden_size as u32);
            // GeLU activation (PLE uses GeLU, not the model's hidden_activation)
            // Reuse gelu_mul with up=ple_embed (element-wise multiply after gelu)
            // gelu_mul does: out = gelu(gate) * up
            self.ctx.encode_gelu_mul(encoder, &self.ple_gated_buf, &self.per_layer_ple_bufs[layer_idx],
                &self.ple_normed_buf, ple_dim as u32);
            // Project back: ple_projected = per_layer_projection(ple_normed) → [hidden]
            self.ctx.encode_matvec_q4(encoder, &layer.per_layer_projection_weight,
                &self.ple_normed_buf, &self.ple_projected_buf, hidden_size as u32, ple_dim as u32);
            // Post-PLE norm
            self.ctx.encode_rmsnorm(encoder, &self.ple_projected_buf,
                &layer.post_per_layer_input_norm_weight, &self.o_out_buf,
                hidden_size as u32, eps);
            // Residual: hidden = hidden + ple_output
            self.ctx.encode_vec_add(encoder, &self.residual_buf, &self.o_out_buf,
                &self.hidden_buf, hidden_size as u32);

            // Layer scalar: hidden *= layer_scalar (prevents signal growth across layers)
            // Use residual_buf as temp to avoid in-place read/write race
            self.ctx.encode_vec_scale(encoder, &self.hidden_buf, &self.residual_buf,
                hidden_size as u32, layer.layer_scalar);
            self.ctx.encode_copy(encoder, &self.residual_buf, &self.hidden_buf, hidden_size as u32);

            encoder.end_encoding();
        }

        // Commit all layer work
        cmd.commit();
        cmd.wait_until_completed();

        // Final norm
        let cmd = self.ctx.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        self.ctx.encode_rmsnorm(encoder, &self.hidden_buf,
            &self.final_norm_weight, &self.normed_buf, hidden_size as u32, eps);
        encoder.end_encoding();

        // Submit and wait
        cmd.commit();
        cmd.wait_until_completed();

        // Compute logits using GPU (tied embeddings = embed_tokens as lm_head)
        // Use matvec_f16: embed_tokens_f16 is (vocab_size, hidden_size) in bf16
        let cmd2 = self.ctx.queue.new_command_buffer();
        let enc2 = cmd2.new_compute_command_encoder();
        self.ctx.encode_matvec_q4(enc2, &self.lm_head_buf, &self.normed_buf,
            &self.logits_buf, vocab_size as u32, hidden_size as u32);
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

/// Convert raw bytes to Vec<u16> (for storing bf16/f16 data compactly).
fn raw_to_u16(data: &[u8]) -> Vec<u16> {
    data.chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}
