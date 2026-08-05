//! DeepSeek-V4-Flash configuration parsed from `deepseek4.*` GGUF metadata.
//!
//! Shape defaults match Flash (43 layers / 4096 embd). We refuse Pro and
//! non-antirez layouts in [`super::gguf_validate`].

use crate::gguf::Gguf;

/// Flash-only constants used when metadata is incomplete (should not happen
/// for valid antirez GGUFs — validation requires the keys).
pub const FLASH_N_LAYER: usize = 43;
pub const FLASH_N_EMBD: usize = 4096;
pub const FLASH_N_VOCAB: usize = 129_280;
pub const FLASH_N_HEAD: usize = 64;
pub const FLASH_N_HEAD_KV: usize = 1;
pub const FLASH_HEAD_DIM: usize = 512;
pub const FLASH_N_ROT: usize = 64;
pub const FLASH_N_LORA_Q: usize = 1024;
pub const FLASH_N_LORA_O: usize = 1024;
pub const FLASH_N_OUT_GROUP: usize = 8;
pub const FLASH_N_EXPERT: usize = 256;
pub const FLASH_N_EXPERT_USED: usize = 6;
pub const FLASH_N_EXPERT_SHARED: usize = 1;
pub const FLASH_N_FF_EXP: usize = 2048;
pub const FLASH_N_HASH_LAYER: usize = 3;
pub const FLASH_N_SWA: usize = 128;
pub const FLASH_N_INDEXER_HEAD: usize = 64;
pub const FLASH_N_INDEXER_HEAD_DIM: usize = 128;
pub const FLASH_N_INDEXER_TOP_K: usize = 512;
pub const FLASH_N_HC: usize = 4;
pub const FLASH_N_HC_SINKHORN_ITER: usize = 20;

pub const DEFAULT_RMS_EPS: f32 = 1e-6;
pub const DEFAULT_HC_EPS: f32 = 1e-6;
pub const DEFAULT_SWIGLU_CLAMP_EXP: f32 = 10.0;
pub const DEFAULT_EXPERT_WEIGHT_SCALE: f32 = 1.5;
pub const DEFAULT_ROPE_FREQ_BASE: f32 = 10_000.0;
pub const DEFAULT_COMPRESS_ROPE_FREQ_BASE: f32 = 160_000.0;
pub const DEFAULT_ROPE_SCALE_FACTOR: f32 = 16.0;
pub const DEFAULT_ROPE_YARN_BETA_FAST: f32 = 32.0;
pub const DEFAULT_ROPE_YARN_BETA_SLOW: f32 = 1.0;
pub const DEFAULT_ROPE_ORIG_CTX: u64 = 65_536;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dsv4Variant {
    Flash,
}

#[derive(Debug, Clone)]
pub struct Dsv4Config {
    pub variant: Dsv4Variant,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_vocab: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub value_dim: usize,
    pub n_rot: usize,
    pub n_lora_q: usize,
    pub n_lora_o: usize,
    pub n_out_group: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_expert_shared: usize,
    pub n_ff_exp: usize,
    pub n_hash_layer: usize,
    pub n_swa: usize,
    pub n_indexer_head: usize,
    pub n_indexer_head_dim: usize,
    pub n_indexer_top_k: usize,
    pub n_hc: usize,
    pub n_hc_sinkhorn_iter: usize,
    pub context_length: u64,
    pub rms_eps: f32,
    pub hc_eps: f32,
    pub expert_weight_scale: f32,
    pub swiglu_clamp_exp: f32,
    pub rope_freq_base: f32,
    pub compress_rope_freq_base: f32,
    pub rope_scale_factor: f32,
    pub rope_yarn_beta_fast: f32,
    pub rope_yarn_beta_slow: f32,
    pub rope_orig_ctx: u64,
    /// Per-layer time-axis compression ratio (0 = raw SWA only, 4 = CSA, 128 = HCA).
    pub compress_ratios: Vec<u32>,
    /// Per-layer SwiGLU clamp (Flash: constant).
    pub swiglu_clamp_exps: Vec<f32>,
}

impl Dsv4Config {
    pub fn expected_compress_ratio(layer: usize) -> u32 {
        if layer < 2 {
            0
        } else if layer % 2 == 0 {
            4
        } else {
            128
        }
    }

    pub fn hc_state_elems(&self) -> usize {
        self.n_hc * self.n_embd
    }

    pub fn is_hash_layer(&self, layer: usize) -> bool {
        layer < self.n_hash_layer
    }

    pub fn has_indexer(&self, layer: usize) -> bool {
        self.compress_ratios.get(layer).copied().unwrap_or(0) == 4
    }

    pub fn compress_ratio(&self, layer: usize) -> u32 {
        self.compress_ratios.get(layer).copied().unwrap_or(0)
    }
}

fn req_u32(g: &Gguf, key: &str) -> u32 {
    g.get_u32(key)
        .unwrap_or_else(|| panic!("deepseek4 GGUF missing required metadata key: {key}"))
}

fn req_u64(g: &Gguf, key: &str) -> u64 {
    g.get_u64(key)
        .unwrap_or_else(|| panic!("deepseek4 GGUF missing required metadata key: {key}"))
}

/// Parse Flash config from an opened GGUF. Does not validate tensor presence.
pub fn dsv4_config_from_gguf(g: &Gguf) -> Dsv4Config {
    let arch = g
        .get_str("general.architecture")
        .unwrap_or("")
        .to_string();
    assert_eq!(
        arch, "deepseek4",
        "expected general.architecture=deepseek4, got {arch:?}"
    );

    let n_layer = req_u32(g, "deepseek4.block_count") as usize;
    let n_embd = req_u32(g, "deepseek4.embedding_length") as usize;
    let n_vocab = req_u32(g, "deepseek4.vocab_size") as usize;
    let n_head = req_u32(g, "deepseek4.attention.head_count") as usize;
    let n_head_kv = req_u32(g, "deepseek4.attention.head_count_kv") as usize;
    let head_dim = req_u32(g, "deepseek4.attention.key_length") as usize;
    let value_dim = req_u32(g, "deepseek4.attention.value_length") as usize;
    let n_rot = req_u32(g, "deepseek4.rope.dimension_count") as usize;
    let n_lora_q = req_u32(g, "deepseek4.attention.q_lora_rank") as usize;
    let n_lora_o = req_u32(g, "deepseek4.attention.output_lora_rank") as usize;
    let n_out_group = req_u32(g, "deepseek4.attention.output_group_count") as usize;
    let n_expert = req_u32(g, "deepseek4.expert_count") as usize;
    let n_expert_used = req_u32(g, "deepseek4.expert_used_count") as usize;
    let n_ff_exp = req_u32(g, "deepseek4.expert_feed_forward_length") as usize;
    let n_expert_shared = req_u32(g, "deepseek4.expert_shared_count") as usize;
    let n_hash_layer = req_u32(g, "deepseek4.hash_layer_count") as usize;
    let n_swa = req_u32(g, "deepseek4.attention.sliding_window") as usize;
    let n_indexer_head = req_u32(g, "deepseek4.attention.indexer.head_count") as usize;
    let n_indexer_head_dim = req_u32(g, "deepseek4.attention.indexer.key_length") as usize;
    let n_indexer_top_k = req_u32(g, "deepseek4.attention.indexer.top_k") as usize;
    let n_hc = req_u32(g, "deepseek4.hyper_connection.count") as usize;
    let n_hc_sinkhorn_iter =
        req_u32(g, "deepseek4.hyper_connection.sinkhorn_iterations") as usize;
    let context_length = req_u64(g, "deepseek4.context_length");

    let compress_ratios = g
        .get_arr_u32("deepseek4.attention.compress_ratios")
        .map(|a| a.to_vec())
        .or_else(|| {
            g.get_arr_i32("deepseek4.attention.compress_ratios")
                .map(|a| a.iter().map(|&x| x as u32).collect())
        })
        .unwrap_or_else(|| {
            panic!("missing deepseek4.attention.compress_ratios");
        });

    let swiglu_clamp_exps = g
        .get_arr_f32("deepseek4.swiglu_clamp_exp")
        .map(|a| a.to_vec())
        .unwrap_or_else(|| vec![DEFAULT_SWIGLU_CLAMP_EXP; n_layer]);

    let rms_eps = g
        .get_f32("deepseek4.attention.layer_norm_rms_epsilon")
        .or_else(|| g.get_f32("deepseek4.rms_norm_eps"))
        .unwrap_or(DEFAULT_RMS_EPS);
    let hc_eps = g
        .get_f32("deepseek4.hyper_connection.eps")
        .unwrap_or(DEFAULT_HC_EPS);
    let expert_weight_scale = g
        .get_f32("deepseek4.expert_weights_scale")
        .unwrap_or(DEFAULT_EXPERT_WEIGHT_SCALE);
    let swiglu_clamp_exp = swiglu_clamp_exps.first().copied().unwrap_or(DEFAULT_SWIGLU_CLAMP_EXP);
    let rope_freq_base = g
        .get_f32("deepseek4.rope.freq_base")
        .unwrap_or(DEFAULT_ROPE_FREQ_BASE);
    let compress_rope_freq_base = g
        .get_f32("deepseek4.rope.compress_freq_base")
        .unwrap_or(DEFAULT_COMPRESS_ROPE_FREQ_BASE);
    let rope_scale_factor = g
        .get_f32("deepseek4.rope.scaling.factor")
        .unwrap_or(DEFAULT_ROPE_SCALE_FACTOR);
    let rope_yarn_beta_fast = g
        .get_f32("deepseek4.rope.scaling.yarn_beta_fast")
        .unwrap_or(DEFAULT_ROPE_YARN_BETA_FAST);
    let rope_yarn_beta_slow = g
        .get_f32("deepseek4.rope.scaling.yarn_beta_slow")
        .unwrap_or(DEFAULT_ROPE_YARN_BETA_SLOW);
    let rope_orig_ctx = g
        .get_u64("deepseek4.rope.scaling.original_context_length")
        .unwrap_or(DEFAULT_ROPE_ORIG_CTX);

    // Flash-only for this port.
    assert_eq!(n_layer, FLASH_N_LAYER, "only Flash (43 layers) is supported");
    assert_eq!(n_embd, FLASH_N_EMBD);
    assert_eq!(n_vocab, FLASH_N_VOCAB);
    assert_eq!(n_head, FLASH_N_HEAD);
    assert_eq!(n_head_kv, FLASH_N_HEAD_KV);
    assert_eq!(head_dim, FLASH_HEAD_DIM);
    assert_eq!(value_dim, FLASH_HEAD_DIM);
    assert_eq!(n_rot, FLASH_N_ROT);
    assert_eq!(n_lora_q, FLASH_N_LORA_Q);
    assert_eq!(n_lora_o, FLASH_N_LORA_O);
    assert_eq!(n_out_group, FLASH_N_OUT_GROUP);
    assert_eq!(n_expert, FLASH_N_EXPERT);
    assert_eq!(n_expert_used, FLASH_N_EXPERT_USED);
    assert_eq!(n_expert_shared, FLASH_N_EXPERT_SHARED);
    assert_eq!(n_ff_exp, FLASH_N_FF_EXP);
    assert_eq!(n_hash_layer, FLASH_N_HASH_LAYER);
    assert_eq!(n_swa, FLASH_N_SWA);
    assert_eq!(n_indexer_head, FLASH_N_INDEXER_HEAD);
    assert_eq!(n_indexer_head_dim, FLASH_N_INDEXER_HEAD_DIM);
    assert_eq!(n_indexer_top_k, FLASH_N_INDEXER_TOP_K);
    assert_eq!(n_hc, FLASH_N_HC);
    assert_eq!(n_hc_sinkhorn_iter, FLASH_N_HC_SINKHORN_ITER);

    assert!(
        compress_ratios.len() >= n_layer,
        "compress_ratios shorter than n_layer"
    );
    for il in 0..n_layer {
        let got = compress_ratios[il];
        let expected = Dsv4Config::expected_compress_ratio(il);
        assert_eq!(
            got, expected,
            "unexpected compress_ratio at layer {il}: got {got}, expected {expected}"
        );
    }

    Dsv4Config {
        variant: Dsv4Variant::Flash,
        n_layer,
        n_embd,
        n_vocab,
        n_head,
        n_head_kv,
        head_dim,
        value_dim,
        n_rot,
        n_lora_q,
        n_lora_o,
        n_out_group,
        n_expert,
        n_expert_used,
        n_expert_shared,
        n_ff_exp,
        n_hash_layer,
        n_swa,
        n_indexer_head,
        n_indexer_head_dim,
        n_indexer_top_k,
        n_hc,
        n_hc_sinkhorn_iter,
        context_length,
        rms_eps,
        hc_eps,
        expert_weight_scale,
        swiglu_clamp_exp,
        rope_freq_base,
        compress_rope_freq_base,
        rope_scale_factor,
        rope_yarn_beta_fast,
        rope_yarn_beta_slow,
        rope_orig_ctx,
        compress_ratios: compress_ratios[..n_layer].to_vec(),
        swiglu_clamp_exps: {
            let mut v = swiglu_clamp_exps;
            v.resize(n_layer, DEFAULT_SWIGLU_CLAMP_EXP);
            v
        },
    }
}
