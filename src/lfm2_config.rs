//! LFM2 / LFM2.5 GGUF config (LiquidAI hybrid short-conv + attention).

use crate::gguf::Gguf;

#[derive(Clone, Debug)]
pub struct Lfm2Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// Per-layer KV heads; 0 means short-conv (recurrent) layer.
    pub num_key_value_heads: Vec<usize>,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// Short-conv kernel length (`lfm2.shortconv.l_cache`). State width is `l_cache - 1`.
    pub shortconv_l_cache: usize,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
}

impl Lfm2Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn is_shortconv(&self, layer_idx: usize) -> bool {
        self.num_key_value_heads[layer_idx] == 0
    }

    pub fn layer_num_kv_heads(&self, layer_idx: usize) -> usize {
        self.num_key_value_heads[layer_idx]
    }

    pub fn conv_state_width(&self) -> usize {
        self.shortconv_l_cache.saturating_sub(1)
    }

    pub fn conv_state_elems(&self) -> usize {
        self.hidden_size * self.conv_state_width()
    }
}

pub fn lfm2_config_from_gguf(g: &Gguf) -> Lfm2Config {
    let hidden_size = g.get_u32("lfm2.embedding_length").expect("lfm2.embedding_length") as usize;
    let intermediate_size =
        g.get_u32("lfm2.feed_forward_length").expect("lfm2.feed_forward_length") as usize;
    let num_hidden_layers = g.get_u32("lfm2.block_count").expect("lfm2.block_count") as usize;
    let num_attention_heads =
        g.get_u32("lfm2.attention.head_count").expect("lfm2.attention.head_count") as usize;

    let num_key_value_heads = if let Some(arr) = g.get_arr_u32("lfm2.attention.head_count_kv") {
        assert_eq!(arr.len(), num_hidden_layers, "head_count_kv length mismatch");
        arr.iter().map(|&x| x as usize).collect()
    } else if let Some(arr) = g.get_arr_i32("lfm2.attention.head_count_kv") {
        assert_eq!(arr.len(), num_hidden_layers, "head_count_kv length mismatch");
        arr.iter().map(|&x| x as usize).collect()
    } else {
        let n = g
            .get_u32("lfm2.attention.head_count_kv")
            .expect("lfm2.attention.head_count_kv") as usize;
        vec![n; num_hidden_layers]
    };

    let vocab_size = g.get_u32("lfm2.vocab_size").map(|v| v as usize).unwrap_or_else(|| {
        g.get_arr_str("tokenizer.ggml.tokens")
            .map(|t| t.len())
            .expect("lfm2.vocab_size or tokenizer.ggml.tokens")
    });

    let max_position_embeddings =
        g.get_u32("lfm2.context_length").unwrap_or(128_000) as usize;
    let rms_norm_eps = g
        .get_f32("lfm2.attention.layer_norm_rms_epsilon")
        .unwrap_or(1e-5) as f64;
    let rope_theta = g.get_f32("lfm2.rope.freq_base").unwrap_or(10_000_000.0) as f64;
    let shortconv_l_cache = g.get_u32("lfm2.shortconv.l_cache").unwrap_or(3) as usize;
    assert!(
        shortconv_l_cache > 1,
        "lfm2.shortconv.l_cache must be > 1, got {}",
        shortconv_l_cache
    );

    let bos_token_id = g.get_u32("tokenizer.ggml.bos_token_id").unwrap_or(1) as usize;
    let eos_token_id = g.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(2) as usize;

    assert_eq!(
        hidden_size % num_attention_heads,
        0,
        "hidden_size not divisible by num_attention_heads"
    );

    Lfm2Config {
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        vocab_size,
        max_position_embeddings,
        rms_norm_eps,
        rope_theta,
        shortconv_l_cache,
        bos_token_id,
        eos_token_id,
    }
}
