use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4Config {
    pub text_config: Gemma4TextConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[serde(default = "default_global_head_dim")]
    pub global_head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub hidden_activation: String,
    pub rms_norm_eps: f64,
    pub sliding_window: usize,
    pub layer_types: Vec<String>,
    #[serde(default = "default_hidden_size_per_layer")]
    pub hidden_size_per_layer_input: usize,
    #[serde(default)]
    pub num_kv_shared_layers: usize,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_final_logit_softcapping")]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub full_attention: Option<RopeConfig>,
    pub sliding_attention: Option<RopeConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeConfig {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_type: String,
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f64,
    #[serde(default = "default_rope_factor")]
    pub factor: f64,
}

fn default_rope_factor() -> f64 { 1.0 }

fn default_global_head_dim() -> usize { 512 }
fn default_hidden_size_per_layer() -> usize { 256 }
fn default_max_pos() -> usize { 131072 }
fn default_final_logit_softcapping() -> Option<f32> { Some(30.0) }
fn default_rope_theta() -> f64 { 10000.0 }
fn default_partial_rotary() -> f64 { 1.0 }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheType {
    F16,
    Q8_0,
    Q4_0,
}

impl KvCacheType {
    pub fn from_env() -> Self {
        match std::env::var("LLAMA_KV_CACHE_TYPE").as_deref() {
            Ok("q8_0") | Ok("Q8_0") => KvCacheType::Q8_0,
            Ok("q4_0") | Ok("Q4_0") => KvCacheType::Q4_0,
            _ => KvCacheType::F16,
        }
    }

    pub fn bytes_per_row(&self, head_dim: usize) -> usize {
        assert!(head_dim % 32 == 0, "head_dim must be a multiple of 32 for quantized KV cache");
        match self {
            KvCacheType::F16 => head_dim * 2,
            KvCacheType::Q8_0 => (head_dim / 32) * 34,
            KvCacheType::Q4_0 => (head_dim / 32) * 18,
        }
    }
}

impl std::fmt::Display for KvCacheType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvCacheType::F16 => write!(f, "f16"),
            KvCacheType::Q8_0 => write!(f, "q8_0"),
            KvCacheType::Q4_0 => write!(f, "q4_0"),
        }
    }
}

impl Gemma4TextConfig {
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn is_full_attention(&self, layer_idx: usize) -> bool {
        self.layer_types.get(layer_idx).map_or(false, |t| t == "full_attention")
    }

    pub fn is_sliding_attention(&self, layer_idx: usize) -> bool {
        self.layer_types.get(layer_idx).map_or(false, |t| t == "sliding_window_attention")
    }

    pub fn layer_head_dim(&self, layer_idx: usize) -> usize {
        if self.is_full_attention(layer_idx) {
            self.global_head_dim
        } else {
            self.head_dim
        }
    }

    /// Returns the index of the last non-shared layer of the given attention type.
    /// Shared KV layers are at the end of the model and reuse KV from earlier layers.
    /// This matches the HuggingFace/llama.cpp Gemma4 architecture where the MTP
    /// assistant cross-attends to the target's last unique sliding/full KV layers.
    pub fn last_non_shared_layer_of_type(&self, want_full: bool) -> Option<usize> {
        let first_shared = self.num_hidden_layers.saturating_sub(self.num_kv_shared_layers);
        (0..first_shared)
            .rev()
            .find(|&i| self.is_full_attention(i) == want_full)
    }

    pub fn sliding_rope_theta(&self) -> f64 {
        self.rope_parameters.as_ref()
            .and_then(|r| r.sliding_attention.as_ref())
            .map_or(10000.0, |c| c.rope_theta)
    }

    pub fn full_rope_theta(&self) -> f64 {
        self.rope_parameters.as_ref()
            .and_then(|r| r.full_attention.as_ref())
            .map_or(1000000.0, |c| c.rope_theta)
    }

    pub fn full_partial_rotary_factor(&self) -> f64 {
        self.rope_parameters.as_ref()
            .and_then(|r| r.full_attention.as_ref())
            .map_or(0.25, |c| c.partial_rotary_factor)
    }

    pub fn full_rope_factor(&self) -> f64 {
        self.rope_parameters.as_ref()
            .and_then(|r| r.full_attention.as_ref())
            .map_or(1.0, |c| c.factor)
    }

    pub fn sliding_rope_factor(&self) -> f64 {
        self.rope_parameters.as_ref()
            .and_then(|r| r.sliding_attention.as_ref())
            .map_or(1.0, |c| c.factor)
    }
}
