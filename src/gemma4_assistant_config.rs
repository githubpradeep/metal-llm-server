use serde::Deserialize;

use crate::gemma4_config::Gemma4TextConfig;

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4AssistantConfig {
    pub backbone_hidden_size: usize,
    pub use_ordered_embeddings: bool,
    pub num_centroids: usize,
    pub centroid_intermediate_top_k: usize,
    pub text_config: Gemma4TextConfig,
}

