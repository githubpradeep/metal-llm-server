//! Speculative decoding (MTP draft head) for Gemma-4.
//!
//! The drafter is a separate F16 `gemma4-assistant` model (llama.cpp
//! `LLM_ARCH_GEMMA4_ASSISTANT`) that runs on top of the base Q4_K_M model.
//! Its attention reads the base model's KV cache (layers 33 = SWA, 34 = full)
//! and it predicts draft tokens that are then verified by a single batched
//! base forward pass. See `reference/llama.cpp/src/models/gemma4-assistant.cpp`
//! and `llama-context.cpp` (NextN draft/verify/accept driver) for the contract.

use crate::gguf::Gguf;
use crate::gpu::{BufferView, MetalContext, weight_fmt};
use crate::gemma4_config::KvCacheType;
use metal::*;

/// One transformer block of the MTP draft head.
pub struct MtpDraftLayer {
    pub attn_norm: BufferView,
    pub out_scale: BufferView,
    pub ffn_down: BufferView,
    pub ffn_up: BufferView,
    pub ffn_gate: BufferView,
    pub post_attention_norm: BufferView,
    pub post_ffw_norm: BufferView,
    pub ffn_norm: BufferView,
    pub attn_output: BufferView,
    pub attn_q_norm: BufferView,
    pub attn_q: BufferView,
    pub rope_freqs: Option<BufferView>,
    /// Scalar output scale (layer_output_scale.weight, F32 [1]).
    pub out_scale_val: f32,
    /// Non-SWA (full attention) layers carry their own rope_freqs (per the
    /// gemma4-assistant loader, which only creates rope_freqs for full layers).
    pub is_full: bool,
    /// Number of rotated dims for partial rotary (0 for SWA layers, which skip
    /// rope; 128 for full layers with Gemma4 p-RoPE factor 0.25 on hd 512).
    pub n_rot: usize,
    /// Base KV layer this head layer attends to: 33 (SWA) or 34 (full).
    pub mapped_base_layer: usize,
    /// Number of KV heads in the *base* model's cache at `mapped_base_layer`.
    /// The draft's query heads attend GQA-style into this cache, so the
    /// attention kernel must be told the base cache's KV-head count (and the
    /// resulting group fan-out) — hardcoding 1 collapses every draft query head
    /// onto KV head 0 and reads the wrong per-head region for any base model
    /// with GQA (kv_heads > 1). Defaults to 1; set by the assistant loader from
    /// the target config once the base layer mapping is known.
    pub base_kv_heads: usize,
    pub head_dim: usize,
    pub n_head: usize,
    pub ffn_inter: usize,
}

/// Loaded MTP draft head weights (all F16 linears; F32 norms/scales/rope).
pub struct MtpDraftHead {
    pub hidden_backbone: usize, // 1536: base model hidden dim
    pub hidden_head: usize,     // 256: draft head hidden dim
    pub n_layers: usize,
    pub vocab: usize,
    /// Full-attention rope base frequency (`gemma4-assistant.rope.freq_base`).
    pub full_rope_theta: f64,
    /// Full-attention rotary dimension count (`rope.dimension_count`).
    pub full_n_rot: usize,
    /// SWA rope base frequency (`gemma4-assistant.rope.freq_base_swa`).
    pub swa_rope_theta: f64,
    /// SWA rotary dimension count (`rope.dimension_count_swa`).
    pub swa_n_rot: usize,
    /// Attention sliding window (drafter's own, in tokens).
    pub sliding_window: u32,
    /// Final logit softcapping (Gemma). Draft argmax/p_min use the same cap as
    /// the target verifier; assistant GGUF typically omits this key.
    pub final_logit_softcapping: f32,
    /// RMSNorm epsilon from assistant GGUF (fallback 1e-6).
    pub rms_eps: f32,
    /// Output LM head / token embedding of the head [hidden_head, vocab] (F16).
    pub tok_embd: BufferView,
    /// [hidden_head] F32.
    pub output_norm: BufferView,
    /// pre_proj [3072, hidden_head] = matmul(concat(token_emb, embd_nextn)) (F16).
    pub pre_proj: BufferView,
    /// post_proj [hidden_head, hidden_backbone] = next-token seed (F16).
    pub post_proj: BufferView,
    /// [hidden_head] F32, present for the single full-attention layer.
    pub rope_freqs: Option<BufferView>,
    /// CPU copy of the full-attention layer's rope inv-freqs [hidden_head/2].
    pub rope_freqs_data: Vec<f32>,
    /// Max FFN intermediate width across head layers (scratch sizing).
    pub max_ffn_inter: usize,
    pub layers: Vec<MtpDraftLayer>,
}

impl MtpDraftHead {
    /// Load the F16 draft-head GGUF and map its layers onto base KV layers.
    pub fn load_from_gguf(ctx: &MetalContext, path: &str) -> Self {
        let g = Gguf::open(path);

        let f16 = |name: &str| -> BufferView {
            BufferView::from_buffer(ctx.buffer_from_slice_no_copy(g.tensor_raw(name)))
                .with_format(weight_fmt::F16)
        };
        let f32b = |name: &str| -> BufferView {
            BufferView::from_buffer(ctx.buffer_from_slice(&g.dequant_to_f32(name)))
        };
        let tn = |i: usize, s: &str| format!("blk.{}.{}", i, s);

        let n_layers =
            g.get_u32("gemma4-assistant.nextn_predict_layers").unwrap_or(4) as usize;
        let hidden_backbone =
            g.get_u32("gemma4-assistant.embedding_length_out").unwrap_or(1536) as usize;
        let hidden_head =
            g.get_u32("gemma4-assistant.embedding_length").unwrap_or(256) as usize;
        let vocab = g.get_u32("gemma4-assistant.vocab_size").unwrap_or(262144) as usize;
        let full_rope_theta = g
            .get_f32("gemma4-assistant.rope.freq_base")
            .map(|v| v as f64)
            .unwrap_or(1_000_000.0);
        let full_n_rot = g
            .get_u32("gemma4-assistant.rope.dimension_count")
            .unwrap_or(512) as usize;
        let swa_rope_theta = g
            .get_f32("gemma4-assistant.rope.freq_base_swa")
            .map(|v| v as f64)
            .unwrap_or(10_000.0);
        let swa_n_rot = g
            .get_u32("gemma4-assistant.rope.dimension_count_swa")
            .unwrap_or(256) as usize;
        let sliding_window = g
            .get_u32("gemma4-assistant.attention.sliding_window")
            .unwrap_or(512);
        let final_logit_softcapping = g
            .get_f32("gemma4-assistant.final_logit_softcapping")
            .unwrap_or(30.0);
        let rms_eps = g
            .get_f32("gemma4-assistant.attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-6);

        // true => sliding attention, false => full (llama.cpp is_swa_impl).
        let swa_pattern: Vec<bool> = g
            .get_arr_bool("gemma4-assistant.attention.sliding_window_pattern")
            .map(|p| p.to_vec())
            .unwrap_or_else(|| {
                eprintln!(
                    "  [MTP] WARNING: no sliding_window_pattern in assistant GGUF; \
                     falling back to period-4 T,T,T,F"
                );
                (0..n_layers).map(|i| (i % 4) != 3).collect()
            });
        if swa_pattern.len() != n_layers {
            panic!(
                "assistant sliding_window_pattern length {} != n_layers {}",
                swa_pattern.len(),
                n_layers
            );
        }

        let tok_embd = f16("token_embd.weight");
        let output_norm = f32b("output_norm.weight");
        let pre_proj = f16("nextn.pre_projection.weight");
        let post_proj = f16("nextn.post_projection.weight");

        // rope_freqs for full-attention layers. May be at root, or per-block.
        let (rope_freqs, rope_freqs_data) = {
            let mut rope_name: Option<String> = None;
            for candidate in std::iter::once("rope_freqs.weight".to_string())
                .chain((0..n_layers).map(|i| tn(i, "rope_freqs.weight")))
            {
                if g.tensor(&candidate).is_some() {
                    rope_name = Some(candidate);
                    break;
                }
            }
            if let Some(name) = rope_name {
                let b = f32b(&name);
                let d = MetalContext::read_buffer(&b.buffer, b.length as usize / 4);
                (Some(b), d)
            } else {
                (None, Vec::new())
            }
        };

        let mut max_ffn_inter = 0usize;
        let mut layers = Vec::with_capacity(n_layers);
        let mut has_full = false;
        for i in 0..n_layers {
            let q_dims = g.tensor(&tn(i, "attn_q.weight")).expect("attn_q").dims.clone();
            let q_cols = q_dims[1] as usize; // n_head * head_dim
            let head_dim = g
                .tensor(&tn(i, "attn_q_norm.weight"))
                .expect("attn_q_norm")
                .dims[0] as usize;
            let n_head = q_cols / head_dim;
            let ffn_inter = g
                .tensor(&tn(i, "ffn_down.weight"))
                .expect("ffn_down")
                .dims[0] as usize;
            max_ffn_inter = max_ffn_inter.max(ffn_inter);

            // GGUF pattern: true = SWA, false = full (matches llama.cpp is_swa).
            let is_full = !swa_pattern[i];
            if is_full {
                has_full = true;
            }
            let mapped_base_layer = if is_full { 34 } else { 33 };

            let out_scale_buf = f32b(&tn(i, "layer_output_scale.weight"));
            let out_scale_val =
                MetalContext::read_buffer(&out_scale_buf.buffer, 1)[0];

            let layer = MtpDraftLayer {
                attn_norm: f32b(&tn(i, "attn_norm.weight")),
                out_scale: out_scale_buf,
                ffn_down: f16(&tn(i, "ffn_down.weight")),
                ffn_up: f16(&tn(i, "ffn_up.weight")),
                ffn_gate: f16(&tn(i, "ffn_gate.weight")),
                post_attention_norm: f32b(&tn(i, "post_attention_norm.weight")),
                post_ffw_norm: f32b(&tn(i, "post_ffw_norm.weight")),
                ffn_norm: f32b(&tn(i, "ffn_norm.weight")),
                attn_output: f16(&tn(i, "attn_output.weight")),
                attn_q_norm: f32b(&tn(i, "attn_q_norm.weight")),
                attn_q: f16(&tn(i, "attn_q.weight")),
                rope_freqs: if is_full { rope_freqs.clone() } else { None },
                out_scale_val,
                is_full,
                n_rot: if is_full {
                    full_n_rot.min(head_dim)
                } else {
                    swa_n_rot.min(head_dim)
                },
                mapped_base_layer,
                base_kv_heads: 1,
                head_dim,
                n_head,
                ffn_inter,
            };
            layers.push(layer);
        }
        if rope_freqs_data.is_empty() && has_full {
            eprintln!("  [MTP] WARNING: no rope_freqs.weight found in assistant GGUF! p-RoPE will be disabled for full-attention layers.");
        }

        eprintln!(
            "  [MTP] SWA pattern (true=sliding): {:?}, rms_eps={}, logit_softcap={}",
            swa_pattern, rms_eps, final_logit_softcapping
        );

        MtpDraftHead {
            hidden_backbone,
            hidden_head,
            n_layers,
            vocab,
            full_rope_theta,
            full_n_rot,
            swa_rope_theta,
            swa_n_rot,
            sliding_window,
            final_logit_softcapping,
            rms_eps,
            tok_embd,
            output_norm,
            pre_proj,
            post_proj,
            rope_freqs,
            rope_freqs_data,
            max_ffn_inter,
            layers,
        }
    }
}

/// Reusable f32 scratch buffers for one draft-head forward pass.
pub struct DraftScratch {
    pub xh: Buffer,
    pub h: Buffer,
    pub a: Buffer,
    pub q: Buffer,
    pub qn: Buffer,
    pub araw: Buffer,
    pub aproj: Buffer,
    pub apost: Buffer,
    pub aout: Buffer,
    pub ffin: Buffer,
    pub gate: Buffer,
    pub up: Buffer,
    pub inter: Buffer,
    pub fdown: Buffer,
    pub fpost: Buffer,
    pub tmp: Buffer,
    /// Per-layer RoPE tables. A single shared cos/sin pair is wrong: all layers
    /// encode into one command buffer, so CPU overwrites would leave every
    /// rotary kernel reading the last layer's angles (full-attn θ) at GPU time.
    pub cos: Vec<Buffer>,
    pub sin: Vec<Buffer>,
    pub dummy_k: Buffer,
    pub logits: Buffer,
    pub hnext: Buffer,
    pub fnorm: Buffer,
    /// Greedy draft token id (single u32).
    pub argmax_token: Buffer,
}

impl MtpDraftHead {
    pub fn alloc_scratch(&self, ctx: &MetalContext) -> DraftScratch {
        let max_q = self
            .layers
            .iter()
            .map(|l| l.n_head * l.head_dim)
            .max()
            .unwrap_or(1024);
        let max_hd = self
            .layers
            .iter()
            .map(|l| l.head_dim)
            .max()
            .unwrap_or(256);
        let ffn = self.max_ffn_inter.max(1);
        // GGML MWG attention scratch scales with the attention head dimension,
        // not the assistant hidden width. Gemma4's final draft layer uses
        // head_dim=512 while hidden_head=256; sizing from hidden_head let that
        // layer write past the scratch allocation and corrupted Q4 KV drafts.
        let max_heads = self
            .layers
            .iter()
            .map(|l| l.n_head)
            .max()
            .unwrap_or(4);
        let tmp_elems =
            crate::ggml_flash_attn::flash_attn_tmp_bytes(max_heads as u32, max_hd as u32)
                as usize
                / 4;
        let cos: Vec<Buffer> = self
            .layers
            .iter()
            .map(|l| ctx.buffer_empty(l.head_dim / 2))
            .collect();
        let sin: Vec<Buffer> = self
            .layers
            .iter()
            .map(|l| ctx.buffer_empty(l.head_dim / 2))
            .collect();
        DraftScratch {
            xh: ctx.buffer_empty(self.hidden_backbone * 2),
            h: ctx.buffer_empty(self.hidden_head),
            a: ctx.buffer_empty(self.hidden_head),
            q: ctx.buffer_empty(max_q),
            qn: ctx.buffer_empty(max_q),
            araw: ctx.buffer_empty(max_q),
            aproj: ctx.buffer_empty(self.hidden_head),
            apost: ctx.buffer_empty(self.hidden_head),
            aout: ctx.buffer_empty(self.hidden_head),
            ffin: ctx.buffer_empty(self.hidden_head),
            gate: ctx.buffer_empty(ffn),
            up: ctx.buffer_empty(ffn),
            inter: ctx.buffer_empty(ffn),
            fdown: ctx.buffer_empty(self.hidden_head),
            fpost: ctx.buffer_empty(self.hidden_head),
            tmp: ctx.buffer_empty(tmp_elems),
            cos,
            sin,
            dummy_k: ctx.buffer_empty(max_q),
            logits: ctx.buffer_empty(self.vocab),
            hnext: ctx.buffer_empty(self.hidden_backbone),
            fnorm: ctx.buffer_empty(self.hidden_head),
            argmax_token: ctx.buffer_empty_u32(1),
        }
    }

    /// One draft-head forward step. Computes the greedy next draft token and the
    /// `h_next` seed for the following draft step.
    ///
    /// `token_embedding` is `get_rows(base_tok_embd, token) * sqrt(hidden_backbone)`
    /// (caller supplies it). `embd_nextn` is the 1536-dim head input (base output
    /// norm on step 0, head `h_next` on later draft steps). `pos` is the decode
    /// position used for rope. Attention reads the base KV cache (Q4_0).
    pub fn forward_draft_step(
        &self,
        ctx: &MetalContext,
        scratch: &DraftScratch,
        token_embedding: &[f32],
        embd_nextn: &[f32],
        pos: u32,
        base_k_cache: &[Buffer],
        base_v_cache: &[Buffer],
        base_kv_seq: u32,
        base_kv_capacity: u32,
        kv_cache_type: KvCacheType,
    ) -> (u32, Vec<f32>) {
        let sliding_window = self.sliding_window;
        let eps = self.rms_eps;
        let hh = self.hidden_head;
        let softcap = self.final_logit_softcapping;

        // xh = concat(token_embedding [hidden_backbone], embd_nextn [hidden_backbone])
        let mut xh = Vec::with_capacity(self.hidden_backbone * 2);
        xh.extend_from_slice(token_embedding);
        xh.extend_from_slice(embd_nextn);
        MetalContext::write_buffer(&scratch.xh, &xh);

        // Materialize per-layer RoPE tables before encoding so each rotary
        // dispatch binds a stable buffer (SWA θ=1e4 vs full θ=1e6+freq factors).
        for (li, layer) in self.layers.iter().enumerate() {
            let half = layer.head_dim / 2;
            let n_rot = layer.n_rot.max(1);
            let n_rot_half = layer.n_rot / 2;
            let theta = if layer.is_full {
                self.full_rope_theta
            } else {
                self.swa_rope_theta
            };
            let mut cos = vec![0.0f32; half];
            let mut sin = vec![0.0f32; half];
            for i in 0..half {
                if i < n_rot_half {
                    let base_inv = 1.0 / (theta.powf(i as f64 * 2.0 / n_rot as f64) as f32);
                    let inv_freq = if layer.is_full {
                        base_inv / self.rope_freqs_data.get(i).copied().unwrap_or(1.0)
                    } else {
                        base_inv
                    };
                    let ang = pos as f32 * inv_freq;
                    cos[i] = ang.cos();
                    sin[i] = ang.sin();
                } else {
                    cos[i] = 1.0;
                    sin[i] = 0.0;
                }
            }
            MetalContext::write_buffer(&scratch.cos[li], &cos);
            MetalContext::write_buffer(&scratch.sin[li], &sin);
        }

        let command_buffer = ctx.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // cur = pre_proj · xh
        ctx.encode_matvec_f16_view(
            encoder,
            &self.pre_proj,
            &scratch.xh,
            &scratch.h,
            hh as u32,
            (self.hidden_backbone * 2) as u32,
        );

        for (li, layer) in self.layers.iter().enumerate() {
            let hd = layer.head_dim as u32;
            let nh = layer.n_head as u32;
            let qelems = (nh * hd) as u32;

            // a = rmsnorm(cur, attn_norm)
            ctx.encode_rmsnorm_view(encoder, &scratch.h, &layer.attn_norm, &scratch.a, hh as u32, eps);

            // Q = wq · a  ; reshape to (head_dim, n_head)
            ctx.encode_matvec_f16_view(encoder, &layer.attn_q, &scratch.a, &scratch.q, qelems, hh as u32);

            // Q = rmsnorm(Q, attn_q_norm)  (per-head)
            ctx.encode_rmsnorm_per_head_view(encoder, &scratch.q, &layer.attn_q_norm, &scratch.qn, nh, hd, eps);

            // Q = rope(Q). Applied to all layers (llama.cpp always ropes; SWA
            // just omits freq_factors).
            ctx.encode_rotary_at(
                encoder,
                &scratch.qn,
                0,
                &scratch.dummy_k,
                0,
                &scratch.cos[li],
                0,
                &scratch.sin[li],
                0,
                nh,
                0,
                hd,
            );

            // attention: read base KV cache at the mapped layer (Q4_0).
            let (eff, kv_start) = if !layer.is_full {
                let eff = base_kv_seq.min(sliding_window);
                let ks = if base_kv_seq > sliding_window {
                    base_kv_seq - sliding_window
                } else {
                    0
                };
                (eff, ks)
            } else {
                (base_kv_seq, 0)
            };
            let row_bytes = ((layer.head_dim / 32) as u32) * 18;
            if std::env::var("DRAFT_NO_ATTN").is_ok() {
                // Ablation: zero attention output to test residual + MLP path alone.
                let zeros = vec![0.0f32; (nh * hd) as usize];
                MetalContext::write_buffer(&scratch.araw, &zeros);
            } else {
                if std::env::var("DRAFT_READ_KV").is_ok() {
                    let kbuf = &base_k_cache[layer.mapped_base_layer];
                    let klen = kbuf.length() as usize / 4;
                    let kdata = MetalContext::read_buffer(kbuf, klen.min(64));
                    let vbuf = &base_v_cache[layer.mapped_base_layer];
                    let vlen = vbuf.length() as usize / 4;
                    let vdata = MetalContext::read_buffer(vbuf, vlen.min(64));
                    let k_ok = kdata.iter().any(|&v| v != 0.0);
                    let v_ok = vdata.iter().any(|&v| v != 0.0);
                    eprintln!("[DRAFT_READ_KV] base cache idx={}: k[0..5]={:.4?} k_has_nonzero={} v[0..5]={:.4?} v_has_nonzero={}",
                        layer.mapped_base_layer, &kdata[..5.min(klen)], k_ok, &vdata[..5.min(vlen)], v_ok);
                }

                // Draft Q attends GQA-style into the target's KV cache. The
                // base model stores `base_kv_heads` KV heads per token at this
                // layer (GQA), so the kernel must use that count and the matching
                // group fan-out (draft query heads / base KV heads). Hardcoding
                // n_kv=1 only works when the base model has a single KV head
                // (older E-series); for GQA bases (e.g. 12B: 16 Q / 8 KV) it
                // collapses every query head onto KV head 0 and reads garbage.
                let n_kv = (layer.base_kv_heads as u32).max(1).min(nh);
                let n_groups = (nh / n_kv).max(1);
                match kv_cache_type {
                    KvCacheType::F16 => {
                        ctx.encode_attention_with_offset_f16(
                            encoder,
                            &scratch.qn,
                            &base_k_cache[layer.mapped_base_layer],
                            &base_v_cache[layer.mapped_base_layer],
                            &scratch.araw,
                            nh,
                            n_kv,
                            n_groups,
                            hd,
                            eff,
                            base_kv_capacity,
                            1.0,
                            kv_start,
                        );
                    }
                    KvCacheType::Q8_0 => {
                        let groups_per_row = hd / 32;
                        let row_bytes_q8 = groups_per_row * 34;
                        ctx.encode_attention_with_offset_q8_0(
                            encoder,
                            &scratch.qn,
                            &base_k_cache[layer.mapped_base_layer],
                            &base_v_cache[layer.mapped_base_layer],
                            &scratch.araw,
                            nh,
                            n_kv,
                            n_groups,
                            hd,
                            eff,
                            base_kv_capacity,
                            1.0,
                            kv_start,
                            groups_per_row,
                            row_bytes_q8,
                        );
                    }
                    KvCacheType::Q4_0 => {
                        let groups_per_row = hd / 32;
                        ctx.encode_attention_with_offset_q4_0(
                            encoder,
                            &scratch.qn,
                            &base_k_cache[layer.mapped_base_layer],
                            &base_v_cache[layer.mapped_base_layer],
                            &scratch.araw,
                            nh,
                            n_kv,
                            n_groups,
                            hd,
                            eff,
                            base_kv_capacity,
                            1.0,
                            kv_start,
                            groups_per_row,
                            row_bytes,
                        );
                    }
                }
            }

            if std::env::var("DRAFT_READ_ATTN").is_ok() && layer.mapped_base_layer == 13 {
                let araw = MetalContext::read_buffer(&scratch.araw, (nh * hd) as usize);
                let amax = araw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let amin = araw.iter().cloned().fold(f32::INFINITY, f32::min);
                let anonzero = araw.iter().filter(|&&v| v != 0.0).count();
                eprintln!("[DRAFT_READ_ATTN] base_cache={} araw: max={:.4} min={:.4} nonzeros={}/{} first5={:.4?}",
                    layer.mapped_base_layer, amax, amin, anonzero, araw.len(), &araw[..5]);
            }

            // cur = wo · attn_raw
            ctx.encode_matvec_f16_view(encoder, &layer.attn_output, &scratch.araw, &scratch.aproj, hh as u32, qelems);

            // cur = rmsnorm(cur, post_attention_norm)
            ctx.encode_rmsnorm_view(encoder, &scratch.aproj, &layer.post_attention_norm, &scratch.apost, hh as u32, eps);

            // attn_out = cur + inpL (residual)
            ctx.encode_vec_add(encoder, &scratch.apost, &scratch.h, &scratch.aout, hh as u32);

            // ffn_in = rmsnorm(attn_out, ffn_norm)
            ctx.encode_rmsnorm_view(encoder, &scratch.aout, &layer.ffn_norm, &scratch.ffin, hh as u32, eps);

            // gelu(gate) * up
            let fi = layer.ffn_inter as u32;
            ctx.encode_matvec_f16_view(encoder, &layer.ffn_gate, &scratch.ffin, &scratch.gate, fi, hh as u32);
            ctx.encode_matvec_f16_view(encoder, &layer.ffn_up, &scratch.ffin, &scratch.up, fi, hh as u32);
            ctx.encode_gelu_mul(encoder, &scratch.gate, &scratch.up, &scratch.inter, fi);

            // down = w_down · inter
            ctx.encode_matvec_f16_view(encoder, &layer.ffn_down, &scratch.inter, &scratch.fdown, hh as u32, fi);

            // cur = rmsnorm(down, post_ffw_norm)
            ctx.encode_rmsnorm_view(encoder, &scratch.fdown, &layer.post_ffw_norm, &scratch.fpost, hh as u32, eps);

            // cur = (cur + attn_out) * out_scale
            ctx.encode_vec_add(encoder, &scratch.fpost, &scratch.aout, &scratch.h, hh as u32);
            ctx.encode_vec_scale(encoder, &scratch.h, &scratch.h, hh as u32, layer.out_scale_val);
        }

        // result = rmsnorm(cur, output_norm)
        ctx.encode_rmsnorm_view(encoder, &scratch.h, &self.output_norm, &scratch.fnorm, hh as u32, eps);

        // logits = output · result; softcap + greedy (matches target verify).
        // Softcap is monotonic so greedy identity is unchanged vs raw argmax,
        // but p_min confidence (top-k softmax over logits) needs the softcapped
        // distribution to match the verifier.
        ctx.encode_matvec_f16_view(encoder, &self.tok_embd, &scratch.fnorm, &scratch.logits, self.vocab as u32, hh as u32);

        // h_next = post_proj · result   (seed for next draft step)
        ctx.encode_matvec_f16_view(encoder, &self.post_proj, &scratch.fnorm, &scratch.hnext, self.hidden_backbone as u32, hh as u32);

        if softcap > 0.0 {
            ctx.encode_softcap_argmax_rows_f32(
                encoder,
                &scratch.logits,
                &scratch.argmax_token,
                1,
                self.vocab as u32,
                softcap,
            );
        } else {
            ctx.encode_argmax_f32(
                encoder,
                &scratch.logits,
                &scratch.argmax_token,
                self.vocab as u32,
            );
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        if std::env::var("DRAFT_DEBUG").is_ok() {
            let h_end = MetalContext::read_buffer(&scratch.h, hh);
            let hnorm: f32 = h_end.iter().map(|v| v*v).sum::<f32>().sqrt();
            let h_mean = h_end.iter().sum::<f32>() / hh as f32;
            let fnorm = MetalContext::read_buffer(&scratch.fnorm, hh);
            let fnorm_norm: f32 = fnorm.iter().map(|v| v*v).sum::<f32>().sqrt();
            eprintln!("[DRAFT_H_END] ||h||={:.4} mean={:.4} first5={:.4?} ||fnorm||={:.4} fnorm[0..3]={:.4?}",
                hnorm, h_mean, &h_end[..5], fnorm_norm, &fnorm[..3]);
        }

        let best = MetalContext::read_u32_buffer(&scratch.argmax_token, 1)[0] as usize;
        let hnext = MetalContext::read_buffer(&scratch.hnext, self.hidden_backbone);

        if std::env::var("DRAFT_DEBUG").is_ok() {
            let logits = MetalContext::read_buffer(&scratch.logits, self.vocab);
            let mut idx: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
            idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            eprintln!(
                "[DRAFT_LOGITS] argmax={} top3={:?}",
                best,
                &idx[..3]
                    .iter()
                    .map(|(i, v)| format!("{}:{:.2}", i, v))
                    .collect::<Vec<_>>()
            );
        }

        (best as u32, hnext)
    }
}
