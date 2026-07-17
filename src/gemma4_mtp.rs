use crate::gemma4_gpu_model::Gemma4GpuModel;
use crate::gpu::MetalContext;
use crate::speculative::{DraftScratch, MtpDraftHead};

/// High-level MTP assistant that wraps the draft head and provides
/// `draft_first`/`draft_tail`/`draft_chain` methods for the generation loop.
pub struct Gemma4MtpAssistant {
    head: MtpDraftHead,
    scratch: DraftScratch,
    /// Cached `embd_nextn` (h_next) from the last draft step.
    embd_nextn: Vec<f32>,
    /// Cached base-model activation (initial_activation) for first draft step.
    initial_activation: Vec<f32>,
    pub gpu_passes: u64,
}

impl Gemma4MtpAssistant {
    pub fn new(ctx: &MetalContext, model_path: &str, target: &Gemma4GpuModel) -> Self {
        let mut head = MtpDraftHead::load_from_gguf(ctx, model_path);
        // Prefer the target's softcap so draft p_min / greedy path matches verify.
        if target.config.final_logit_softcapping > 0.0 {
            head.final_logit_softcapping = target.config.final_logit_softcapping;
        }
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
        let scratch = head.alloc_scratch(ctx);
        println!(
            "  Gemma4 MTP assistant: {} layers, hidden_head={}, backbone_hidden={}, vocab={}",
            head.n_layers, head.hidden_head, head.hidden_backbone, head.vocab
        );
        for (idx, l) in head.layers.iter().enumerate() {
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
            head.rope_freqs_data.len(),
            head.full_rope_theta,
            head.swa_rope_theta,
            head.full_n_rot,
            head.swa_n_rot,
        );
        if head.rope_freqs_data.len() > 4 {
            eprintln!(
                "  [MTP] rope_freqs_data[0..4]={:.4?} ... rope_freqs_data[{}..{}]={:.4?}",
                &head.rope_freqs_data[..4],
                head.rope_freqs_data.len() - 4,
                head.rope_freqs_data.len(),
                &head.rope_freqs_data[head.rope_freqs_data.len()-4..],
            );
        }
        if let (Some(sliding), Some(full)) = (swa_source, full_source) {
            println!(
                "  KV sources: sliding_attention=layer {}, full_attention=layer {}",
                sliding, full
            );
        }
        Self {
            head,
            scratch,
            embd_nextn: Vec::new(),
            initial_activation: Vec::new(),
            gpu_passes: 0,
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

        let mut draft_token = from_token;
        let mut drafts = Vec::with_capacity(steps);

        for _ in 0..steps {
            let mut token_embedding = target.token_embedding_raw(draft_token)?;
            let scale = (self.head.hidden_backbone as f32).sqrt();
            for v in token_embedding.iter_mut() {
                *v *= scale;
            }
            let embd_nextn = std::mem::take(&mut self.embd_nextn);
            let (next_token, h_next) = self.head.forward_draft_step(
                &target.ctx,
                &self.scratch,
                &token_embedding,
                if embd_nextn.is_empty() { &self.initial_activation } else { &embd_nextn },
                target.total_tokens as u32,
                &target.k_cache,
                &target.v_cache,
                target.kv_seq_len,
                target.kv_capacity,
                target.kv_cache_type,
            );
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
        if initial_activation.len() != self.head.hidden_backbone {
            return Err(format!(
                "target activation has {} values, expected {}",
                initial_activation.len(),
                self.head.hidden_backbone
            ));
        }
        if target.kv_seq_len == 0 {
            return Err("target KV cache is empty".to_string());
        }

        self.initial_activation = initial_activation.to_vec();

        let mut draft_token = initial_token;
        let mut drafts = Vec::with_capacity(steps);

        for step in 0..steps {
            let mut token_embedding = target.token_embedding_raw(draft_token)?;
            let scale = (self.head.hidden_backbone as f32).sqrt();
            for v in token_embedding.iter_mut() {
                *v *= scale;
            }
            let (next_token, h_next) = self.head.forward_draft_step(
                &target.ctx,
                &self.scratch,
                &token_embedding,
                if step == 0 { initial_activation } else { &self.embd_nextn },
                target.total_tokens as u32,
                &target.k_cache,
                &target.v_cache,
                target.kv_seq_len,
                target.kv_capacity,
                target.kv_cache_type,
            );
            self.gpu_passes += 1;
            draft_token = next_token as usize;
            self.embd_nextn = h_next;

            if p_min > 0.0 {
                let logits = MetalContext::read_buffer(&self.scratch.logits, self.head.vocab);
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
