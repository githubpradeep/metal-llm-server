use ndarray::{Array2, Array3};
use tokenizers::Tokenizer;

use crate::cache::StreamingKVCache;
use crate::config::LlamaConfig;
use crate::layers::{LlamaDecoderLayer, RMSNorm, RotaryEmbedding};
use crate::quantize::QuantizedLinear;
use crate::weights::ModelWeights;

// ─── LlamaModel ──────────────────────────────────────────────────────────────

pub struct LlamaModel {
    pub embed_tokens: Array2<f32>, // (vocab_size, hidden_size) — stays f32 for lookup
    pub layers: Vec<LlamaDecoderLayer>,
    pub final_layernorm: RMSNorm,
    pub rotary_emb: RotaryEmbedding,
    pub config: LlamaConfig,
}

impl LlamaModel {
    pub fn new(config: &LlamaConfig, weights: &ModelWeights) -> Self {
        let embed_tokens = weights.get_2d("model.embed_tokens.weight");

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            println!("  Loading layer {}/{}", layer_idx + 1, config.num_hidden_layers);
            layers.push(LlamaDecoderLayer::new(config, layer_idx, weights));
        }

        let final_ln_w = weights.get_1d("model.norm.weight");
        let final_layernorm = RMSNorm::new(final_ln_w, config.rms_norm_eps);

        let head_dim = config.head_dim();
        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        );

        LlamaModel {
            embed_tokens,
            layers,
            final_layernorm,
            rotary_emb,
            config: config.clone(),
        }
    }

    pub fn forward(
        &self,
        input_ids: &[Vec<i64>],
        kv_cache: &mut StreamingKVCache,
    ) -> Array3<f32> {
        let batch_size = input_ids.len();
        let seq_length = input_ids[0].len();
        let hidden_size = self.config.hidden_size;

        let cache_len = kv_cache.num_items();
        let position_ids: Vec<Vec<i64>> = (0..batch_size)
            .map(|_| {
                (cache_len..(cache_len + seq_length))
                    .map(|p| p as i64)
                    .collect()
            })
            .collect();

        // Embed tokens
        let embed_raw = self.embed_tokens.as_slice().unwrap();
        let mut hidden_data = vec![0.0f32; batch_size * seq_length * hidden_size];
        for b in 0..batch_size {
            for s_idx in 0..seq_length {
                let token_id = input_ids[b][s_idx] as usize;
                let src_offset = token_id * hidden_size;
                let dst_offset = (b * seq_length + s_idx) * hidden_size;
                hidden_data[dst_offset..dst_offset + hidden_size]
                    .copy_from_slice(&embed_raw[src_offset..src_offset + hidden_size]);
            }
        }

        let (cos, sin) = self.rotary_emb.forward(&position_ids);

        if seq_length == 1 && batch_size == 1 {
            // Fast single-token decode
            let mut current = hidden_data;
            let mut next = vec![0.0f32; hidden_size];

            for layer in &self.layers {
                layer.forward_vec(&current, &cos, &sin, kv_cache, &mut next);
                std::mem::swap(&mut current, &mut next);
            }

            let mut final_out = vec![0.0f32; hidden_size];
            self.final_layernorm.forward_vec(&current, &mut final_out);

            Array3::from_shape_vec((1, 1, hidden_size), final_out).unwrap()
        } else {
            // Multi-token prefill
            let mut hidden_states = Array3::from_shape_vec(
                (batch_size, seq_length, hidden_size),
                hidden_data,
            ).unwrap();

            for layer in &self.layers {
                hidden_states = layer.forward(&hidden_states, &cos, &sin, kv_cache);
            }

            self.final_layernorm.forward(&hidden_states)
        }
    }
}

// ─── LlamaForCausalLM ────────────────────────────────────────────────────────

pub struct LlamaForCausalLM {
    pub model: LlamaModel,
    pub lm_head: QuantizedLinear,
    pub config: LlamaConfig,
}

impl LlamaForCausalLM {
    pub fn new(config: &LlamaConfig, weights: &ModelWeights) -> Self {
        let model = LlamaModel::new(config, weights);

        let lm_head_w = weights.get_2d("lm_head.weight");
        let vocab_size = lm_head_w.shape()[0];
        let hidden_size = lm_head_w.shape()[1];
        let lm_head = QuantizedLinear::from_f32(lm_head_w.as_slice().unwrap(), vocab_size, hidden_size);

        LlamaForCausalLM {
            model,
            lm_head,
            config: config.clone(),
        }
    }

    pub fn forward(
        &self,
        input_ids: &[Vec<i64>],
        kv_cache: &mut StreamingKVCache,
    ) -> Array3<f32> {
        let hidden_states = self.model.forward(input_ids, kv_cache);
        let shape = hidden_states.shape();
        let batch = shape[0];
        let seq = shape[1];
        let m = batch * seq;
        let vocab_size = self.lm_head.out_features;

        let x = hidden_states.as_slice().unwrap();
        let mut logits = vec![0.0f32; m * vocab_size];
        self.lm_head.forward_batch(x, m, &mut logits);

        Array3::from_shape_vec((batch, seq, vocab_size), logits).unwrap()
    }
}

// ─── Model Loading ───────────────────────────────────────────────────────────

pub fn load_model(model_dir: &str) -> (Tokenizer, LlamaForCausalLM, LlamaConfig) {
    let (weights, config) = ModelWeights::load(model_dir);

    println!("Building model (int8 quantized)...");
    let model = LlamaForCausalLM::new(&config, &weights);

    let tokenizer_path = std::path::Path::new(model_dir).join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .expect("Failed to load tokenizer.json");

    (tokenizer, model, config)
}
