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
    /// Per-layer FFN width when the model uses variable MLP sizes (E2B double-wide).
    #[serde(default)]
    pub intermediate_sizes: Vec<usize>,
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
    /// Per-layer KV head counts. Empty = uniform (use num_key_value_heads).
    #[serde(default)]
    pub num_key_value_heads_per_layer: Vec<usize>,
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
fn default_final_logit_softcapping() -> f32 { 30.0 }
fn default_rope_theta() -> f64 { 10000.0 }
fn default_partial_rotary() -> f64 { 1.0 }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheType {
    F16,
    Q8_0,
    Q4_0,
    /// TurboQuant: Haar-random rotation + quantization of each K/V vector.
    /// `bits == 4` on both sides uses the Q4_0 block layout (affine path);
    /// otherwise V3 Lloyd–Max (fp16 norm + bit-packed indices). Keys/values
    /// may use different bit-widths (e.g. K3/V2).
    TurboQuant { k_bits: u8, v_bits: u8 },
}

impl KvCacheType {
    pub fn from_env() -> Self {
        match std::env::var("LLAMA_KV_CACHE_TYPE").as_deref() {
            Ok("q8_0") | Ok("Q8_0") => KvCacheType::Q8_0,
            Ok("q4_0") | Ok("Q4_0") => KvCacheType::Q4_0,
            Ok("turboquant") | Ok("TurboQuant") | Ok("TURBOQUANT") | Ok("tq") => {
                let parse = |name: &str, default: u8| -> u8 {
                    std::env::var(name)
                        .ok()
                        .and_then(|s| s.trim().parse::<u8>().ok())
                        .filter(|b| (2..=4).contains(b))
                        .unwrap_or(default)
                };
                let bits = parse("TURBOQUANT_BITS", 4);
                let k_bits = parse("TURBOQUANT_K_BITS", bits);
                let v_bits = parse("TURBOQUANT_V_BITS", bits);
                KvCacheType::TurboQuant { k_bits, v_bits }
            }
            _ => KvCacheType::F16,
        }
    }

    /// Fast affine Q4_0 storage when both K and V are 4-bit.
    pub fn tq_affine(&self) -> bool {
        matches!(self, KvCacheType::TurboQuant { k_bits: 4, v_bits: 4 })
    }

    fn tq_side_row_bytes(head_dim: usize, bits: u8, affine: bool) -> usize {
        if affine {
            (head_dim / 32) * 18
        } else {
            2 + (head_dim * (bits as usize)) / 8
        }
    }

    pub fn k_row_bytes(&self, head_dim: usize) -> usize {
        match self {
            KvCacheType::TurboQuant { k_bits, .. } => {
                Self::tq_side_row_bytes(head_dim, *k_bits, self.tq_affine())
            }
            _ => self.bytes_per_row(head_dim),
        }
    }

    pub fn v_row_bytes(&self, head_dim: usize) -> usize {
        match self {
            KvCacheType::TurboQuant { v_bits, .. } => {
                Self::tq_side_row_bytes(head_dim, *v_bits, self.tq_affine())
            }
            _ => self.bytes_per_row(head_dim),
        }
    }

    pub fn bytes_per_row(&self, head_dim: usize) -> usize {
        assert!(head_dim % 32 == 0, "head_dim must be a multiple of 32 for quantized KV cache");
        match self {
            KvCacheType::F16 => head_dim * 2,
            KvCacheType::Q8_0 => (head_dim / 32) * 34,
            KvCacheType::Q4_0 => (head_dim / 32) * 18,
            KvCacheType::TurboQuant { k_bits, .. } => {
                Self::tq_side_row_bytes(head_dim, *k_bits, self.tq_affine())
            }
        }
    }
}

impl std::fmt::Display for KvCacheType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvCacheType::F16 => write!(f, "f16"),
            KvCacheType::Q8_0 => write!(f, "q8_0"),
            KvCacheType::Q4_0 => write!(f, "q4_0"),
            KvCacheType::TurboQuant { k_bits, v_bits } => {
                if k_bits == v_bits {
                    write!(f, "turboquant-{}bit", k_bits)
                } else {
                    write!(f, "turboquant-K{}V{}", k_bits, v_bits)
                }
            }
        }
    }
}

impl Gemma4TextConfig {
    pub fn layer_intermediate_size(&self, layer_idx: usize) -> usize {
        self.intermediate_sizes
            .get(layer_idx)
            .copied()
            .unwrap_or(self.intermediate_size)
    }

    pub fn max_intermediate_size(&self) -> usize {
        self.intermediate_sizes
            .iter()
            .copied()
            .max()
            .unwrap_or(self.intermediate_size)
    }

    pub fn layer_num_kv_heads(&self, layer_idx: usize) -> usize {
        self.num_key_value_heads_per_layer
            .get(layer_idx)
            .copied()
            .unwrap_or(self.num_key_value_heads)
    }

    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn layer_num_kv_groups(&self, layer_idx: usize) -> usize {
        let kv = self.layer_num_kv_heads(layer_idx);
        if kv == 0 { 1 } else { self.num_attention_heads / kv }
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
