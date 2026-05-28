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
    pub final_logit_softcapping: f32,
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
}

fn default_global_head_dim() -> usize { 512 }
fn default_hidden_size_per_layer() -> usize { 256 }
fn default_max_pos() -> usize { 131072 }
fn default_final_logit_softcapping() -> f32 { 30.0 }
fn default_rope_theta() -> f64 { 10000.0 }
fn default_partial_rotary() -> f64 { 1.0 }

impl Gemma4TextConfig {
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn is_full_attention(&self, layer_idx: usize) -> bool {
        self.layer_types.get(layer_idx).map_or(false, |t| t == "full_attention")
    }

    pub fn layer_head_dim(&self, layer_idx: usize) -> usize {
        if self.is_full_attention(layer_idx) {
            self.global_head_dim
        } else {
            self.head_dim
        }
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
}
