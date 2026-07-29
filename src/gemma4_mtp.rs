use crate::gemma4_gpu_model::Gemma4GpuModel;
use crate::gpu::MetalContext;
use crate::mtp_host::{MtpBackendKind, MtpHostHead};
use crate::speculative::{DraftScratch, MtpDraftHead};

enum DraftEngine {
    Metal {
        head: MtpDraftHead,
        scratch: DraftScratch,
    },
    Host {
        head: MtpHostHead,
    },
}

/// High-level MTP assistant that wraps the draft head and provides
/// `draft_first`/`draft_tail`/`draft_chain` methods for the generation loop.
pub struct Gemma4MtpAssistant {
    engine: DraftEngine,
    /// Cached `embd_nextn` (h_next) from the last draft step.
    embd_nextn: Vec<f32>,
    /// Cached base-model activation (initial_activation) for first draft step.
    initial_activation: Vec<f32>,
    hidden_backbone: usize,
    pub gpu_passes: u64,
}

impl Gemma4MtpAssistant {
    pub fn new(ctx: &MetalContext, model_path: &str, target: &Gemma4GpuModel) -> Self {
        let backend = MtpBackendKind::from_env();
        eprintln!("  [MTP] Backend: {}", backend.label());

        match backend {
            MtpBackendKind::Metal => {
                let mut head = MtpDraftHead::load_from_gguf(ctx, model_path);
                if target.config.final_logit_softcapping > 0.0 {
                    head.final_logit_softcapping = target.config.final_logit_softcapping;
                }
                remap_metal_layers(&mut head, target);
                let hidden_backbone = head.hidden_backbone;
                let scratch = head.alloc_scratch(ctx);
                print_head_info(
                    head.n_layers,
                    head.hidden_head,
                    head.hidden_backbone,
                    head.vocab,
                    &head.layers.iter().map(|l| LayerInfo {
                        is_full: l.is_full,
                        head_dim: l.head_dim,
                        n_head: l.n_head,
                        ffn_inter: l.ffn_inter,
                        mapped_base_layer: l.mapped_base_layer,
                        base_kv_heads: l.base_kv_heads,
                    }).collect::<Vec<_>>(),
                    target,
                    head.rope_freqs_data.len(),
                    head.full_rope_theta,
                    head.swa_rope_theta,
                    head.full_n_rot,
                    head.swa_n_rot,
                    &head.rope_freqs_data,
                );
                Self {
                    engine: DraftEngine::Metal { head, scratch },
                    embd_nextn: Vec::new(),
                    initial_activation: Vec::new(),
                    hidden_backbone,
                    gpu_passes: 0,
                }
            }
            MtpBackendKind::Host => {
                let mut head = MtpHostHead::load_from_gguf(ctx, model_path);
                if target.config.final_logit_softcapping > 0.0 {
                    head.final_logit_softcapping = target.config.final_logit_softcapping;
                }
                remap_host_layers(&mut head, target);
                let hidden_backbone = head.hidden_backbone;
                print_head_info(
                    head.n_layers,
                    head.hidden_head,
                    head.hidden_backbone,
                    head.vocab,
                    &head.layers.iter().map(|l| LayerInfo {
                        is_full: l.is_full,
                        head_dim: l.head_dim,
                        n_head: l.n_head,
                        ffn_inter: l.ffn_inter,
                        mapped_base_layer: l.mapped_base_layer,
                        base_kv_heads: l.base_kv_heads,
                    }).collect::<Vec<_>>(),
                    target,
                    head.rope_freqs_data.len(),
                    head.full_rope_theta,
                    head.swa_rope_theta,
                    head.full_n_rot,
                    head.swa_n_rot,
                    &head.rope_freqs_data,
                );
                Self {
                    engine: DraftEngine::Host { head },
                    embd_nextn: Vec::new(),
                    initial_activation: Vec::new(),
                    hidden_backbone,
                    gpu_passes: 0,
                }
            }
        }
    }

    /// Draft only the first token (cheap probe before main-model verify).
    pub fn draft_first(
        &mut self,
        initial_token: usize,
        initial_activation: &[f32],
        target: &Gemma4GpuModel,
    ) -> Result<usize, String> {
        let tokens = self.draft_chain(initial_token, initial_activation, 1, target, 0.0)?;
        Ok(tokens[0])
    }

    /// Continue drafting after `draft_first` matched. Reuses cached embd_nextn.
    pub fn draft_tail(
        &mut self,
        from_token: usize,
        steps: usize,
        target: &Gemma4GpuModel,
    ) -> Result<Vec<usize>, String> {
        if steps == 0 {
            return Ok(Vec::new());
        }
        if target.kv_seq_len == 0 {
            return Err("target KV cache is empty".to_string());
        }

        self.sync_kv_mirror(target);

        let mut draft_token = from_token;
        let mut drafts = Vec::with_capacity(steps);
        let initial = self.initial_activation.clone();

        for _ in 0..steps {
            let mut token_embedding = target.token_embedding_raw(draft_token)?;
            let scale = (self.hidden_backbone as f32).sqrt();
            for v in token_embedding.iter_mut() {
                *v *= scale;
            }
            let embd_nextn = std::mem::take(&mut self.embd_nextn);
            let embd = if embd_nextn.is_empty() {
                initial.as_slice()
            } else {
                embd_nextn.as_slice()
            };
            let (next_token, h_next) = self.forward_step(target, &token_embedding, embd)?;
            self.gpu_passes += 1;
            draft_token = next_token as usize;
            self.embd_nextn = h_next;
            drafts.push(draft_token);
        }

        Ok(drafts)
    }

    /// Draft multiple tokens with one RoPE table build and minimal GPU/CPU sync.
    /// When `p_min` is in (0, 1), stop early if the greedy draft's softmax
    /// probability falls below the threshold (reads logits on CPU for that step).
    pub fn draft_chain(
        &mut self,
        initial_token: usize,
        initial_activation: &[f32],
        steps: usize,
        target: &Gemma4GpuModel,
        p_min: f32,
    ) -> Result<Vec<usize>, String> {
        if steps == 0 {
            return Ok(Vec::new());
        }
        if initial_activation.len() != self.hidden_backbone {
            return Err(format!(
                "target activation has {} values, expected {}",
                initial_activation.len(),
                self.hidden_backbone
            ));
        }
        if target.kv_seq_len == 0 {
            return Err("target KV cache is empty".to_string());
        }

        self.initial_activation = initial_activation.to_vec();
        self.sync_kv_mirror(target);

        let mut draft_token = initial_token;
        let mut drafts = Vec::with_capacity(steps);

        for step in 0..steps {
            let mut token_embedding = target.token_embedding_raw(draft_token)?;
            let scale = (self.hidden_backbone as f32).sqrt();
            for v in token_embedding.iter_mut() {
                *v *= scale;
            }
            // Move `embd_nextn` out so the borrow does not collide with the
            // `&mut self` forward call; it is replaced by `h_next` below.
            let prev = if step == 0 {
                Vec::new()
            } else {
                std::mem::take(&mut self.embd_nextn)
            };
            let embd: &[f32] = if step == 0 { initial_activation } else { &prev };
            let (next_token, h_next) = self.forward_step(target, &token_embedding, embd)?;
            self.gpu_passes += 1;
            draft_token = next_token as usize;
            self.embd_nextn = h_next;

            if p_min > 0.0 {
                let logits = self.sync_last_logits();
                let prob = draft_token_confidence(&logits, draft_token, draft_top_k());
                drafts.push(draft_token);
                if prob < p_min {
                    break;
                }
            } else {
                drafts.push(draft_token);
            }
        }

        Ok(drafts)
    }

    fn forward_step(
        &mut self,
        target: &Gemma4GpuModel,
        token_embedding: &[f32],
        embd_nextn: &[f32],
    ) -> Result<(u32, Vec<f32>), String> {
        match &mut self.engine {
            DraftEngine::Metal { head, scratch } => Ok(head.forward_draft_step(
                &target.ctx,
                scratch,
                token_embedding,
                embd_nextn,
                target.total_tokens as u32,
                &target.k_cache,
                &target.v_cache,
                target.kv_seq_len,
                target.kv_capacity,
                target.kv_cache_type,
            )),
            DraftEngine::Host { head } => Ok(head.forward_draft_step(
                &target.ctx,
                token_embedding,
                embd_nextn,
                target.total_tokens as u32,
                &target.k_cache,
                &target.v_cache,
                target.kv_seq_len,
                target.kv_capacity,
                target.kv_cache_type,
            )),
        }
    }

    /// Host backend only: snapshot the target KV rows for this draft chain.
    fn sync_kv_mirror(&mut self, target: &Gemma4GpuModel) {
        if let DraftEngine::Host { head } = &mut self.engine {
            head.sync_kv_mirror(
                &target.k_cache,
                &target.v_cache,
                target.kv_seq_len,
                target.kv_capacity,
                target.kv_cache_type,
            );
        }
    }

    fn sync_last_logits(&mut self) -> Vec<f32> {
        match &mut self.engine {
            DraftEngine::Metal { head, scratch } => {
                MetalContext::read_buffer(&scratch.logits, head.vocab)
            }
            DraftEngine::Host { head } => {
                head.sync_last_logits();
                head.last_logits.clone()
            }
        }
    }
}

struct LayerInfo {
    is_full: bool,
    head_dim: usize,
    n_head: usize,
    ffn_inter: usize,
    mapped_base_layer: usize,
    base_kv_heads: usize,
}

fn remap_metal_layers(head: &mut MtpDraftHead, target: &Gemma4GpuModel) {
    let swa_source = target.mtp_kv_source_layer(false);
    let full_source = target.mtp_kv_source_layer(true);
    let mut remapped = false;
    for l in head.layers.iter_mut() {
        let correct = if l.is_full { full_source } else { swa_source };
        if let Some(src) = correct {
            if l.mapped_base_layer != src {
                eprintln!(
                    "  [MTP] Fixing mapped_base_layer: layer {} ({}_attention): {} -> {}",
                    l.mapped_base_layer,
                    if l.is_full { "full" } else { "SWA" },
                    l.mapped_base_layer,
                    src,
                );
                l.mapped_base_layer = src;
                remapped = true;
            }
        }
    }
    if remapped {
        eprintln!("  [MTP] Fixed KV cache layer mapping (was reading from shared layers with no KV data)");
    }
    for l in head.layers.iter_mut() {
        l.base_kv_heads = target.config.layer_num_kv_heads(l.mapped_base_layer).max(1);
    }
    if let Some(first) = head.layers.first() {
        eprintln!(
            "  [MTP] Base KV heads for draft attention: {} (base num_heads={})",
            first.base_kv_heads, target.config.num_attention_heads,
        );
    }
}

fn remap_host_layers(head: &mut MtpHostHead, target: &Gemma4GpuModel) {
    let swa_source = target.mtp_kv_source_layer(false);
    let full_source = target.mtp_kv_source_layer(true);
    let mut remapped = false;
    for l in head.layers.iter_mut() {
        let correct = if l.is_full { full_source } else { swa_source };
        if let Some(src) = correct {
            if l.mapped_base_layer != src {
                eprintln!(
                    "  [MTP] Fixing mapped_base_layer: layer {} ({}_attention): {} -> {}",
                    l.mapped_base_layer,
                    if l.is_full { "full" } else { "SWA" },
                    l.mapped_base_layer,
                    src,
                );
                l.mapped_base_layer = src;
                remapped = true;
            }
        }
    }
    if remapped {
        eprintln!("  [MTP] Fixed KV cache layer mapping (was reading from shared layers with no KV data)");
    }
    for l in head.layers.iter_mut() {
        l.base_kv_heads = target.config.layer_num_kv_heads(l.mapped_base_layer).max(1);
    }
    if let Some(first) = head.layers.first() {
        eprintln!(
            "  [MTP] Base KV heads for draft attention: {} (base num_heads={})",
            first.base_kv_heads, target.config.num_attention_heads,
        );
    }
}

fn print_head_info(
    n_layers: usize,
    hidden_head: usize,
    hidden_backbone: usize,
    vocab: usize,
    layers: &[LayerInfo],
    target: &Gemma4GpuModel,
    rope_len: usize,
    full_rope_theta: f64,
    swa_rope_theta: f64,
    full_n_rot: usize,
    swa_n_rot: usize,
    rope_freqs_data: &[f32],
) {
    println!(
        "  Gemma4 MTP assistant: {} layers, hidden_head={}, backbone_hidden={}, vocab={}",
        n_layers, hidden_head, hidden_backbone, vocab
    );
    for (idx, l) in layers.iter().enumerate() {
        let base_hd = if l.is_full {
            target.config.global_head_dim
        } else {
            target.config.head_dim
        };
        eprintln!(
            "  [MTP] Draft layer {}: is_full={}, head_dim={}, n_head={}, ffn_inter={}, mapped_base_layer={}, base_layer_head_dim={}",
            idx, l.is_full, l.head_dim, l.n_head, l.ffn_inter, l.mapped_base_layer, base_hd,
        );
    }
    eprintln!(
        "  [MTP] RoPE: rope_freqs_data.len={}, full_rope_theta={}, swa_rope_theta={}, full_n_rot={}, swa_n_rot={}",
        rope_len, full_rope_theta, swa_rope_theta, full_n_rot, swa_n_rot,
    );
    if rope_freqs_data.len() > 4 {
        eprintln!(
            "  [MTP] rope_freqs_data[0..4]={:.4?} ... rope_freqs_data[{}..{}]={:.4?}",
            &rope_freqs_data[..4],
            rope_freqs_data.len() - 4,
            rope_freqs_data.len(),
            &rope_freqs_data[rope_freqs_data.len() - 4..],
        );
    }
    let swa_source = target.mtp_kv_source_layer(false);
    let full_source = target.mtp_kv_source_layer(true);
    if let (Some(sliding), Some(full)) = (swa_source, full_source) {
        println!(
            "  KV sources: sliding_attention=layer {}, full_attention=layer {}",
            sliding, full
        );
    }
}

/// Candidate-set size for the draft confidence softmax. llama.cpp's draft-mtp
/// sampler is top_k=10: the greedy token's probability is normalized over the
/// top 10 candidates only, not the full vocab. 0 = full-vocab softmax.
fn draft_top_k() -> usize {
    static TOP_K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *TOP_K.get_or_init(|| {
        std::env::var("LLAMA_MTP_DRAFT_TOP_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(10)
    })
}

/// Softmax probability of `token` normalized over the `top_k` largest logits
/// (llama.cpp top-k sampler semantics). `top_k == 0` normalizes over the full
/// vocab. `token` is the argmax, so it is always inside the candidate set.
fn draft_token_confidence(logits: &[f32], token: usize, top_k: usize) -> f32 {
    if top_k == 0 || top_k >= logits.len() {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for &l in logits {
            sum += (l - max_logit).exp();
        }
        if sum <= 0.0 {
            return 0.0;
        }
        return (logits[token] - max_logit).exp() / sum;
    }

    // Single pass keeping the top_k largest logits (k is small, ~10).
    let mut top: Vec<f32> = Vec::with_capacity(top_k + 1);
    for &l in logits {
        if top.len() < top_k {
            top.push(l);
            if top.len() == top_k {
                top.sort_by(|a, b| b.partial_cmp(a).unwrap());
            }
        } else if l > top[top_k - 1] {
            let pos = top.partition_point(|&t| t >= l);
            top.insert(pos, l);
            top.pop();
        }
    }
    let max_logit = top[0];
    let sum: f32 = top.iter().map(|&l| (l - max_logit).exp()).sum();
    if sum <= 0.0 {
        return 0.0;
    }
    (logits[token] - max_logit).exp() / sum
}
