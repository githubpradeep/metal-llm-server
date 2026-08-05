//! DeepSeek-V4-Flash forward pass (dense matvecs + SSD experts).

use super::attn::{apply_rope_all_heads, attn_mixed_mqa, attn_swa_mqa, select_compressed_rows};
use super::forward_metal::DenseGpuOp;
use super::hc::{
    hc_expand_post, hc_from_plain_embedding, hc_split_sinkhorn, hc_weighted_sum, rms_norm,
    rms_norm_no_weight,
};
use super::model::Dsv4GpuModel;
use super::moe::{
    hash_experts_for_token, router_probs_sqrt_softplus, select_topk_experts, swiglu,
};
use super::quant::{matvec_iq2_xxs, matvec_q2_k};
use super::ssd::ExpertKey;
use crate::gguf::ggml_type;

impl Dsv4GpuModel {
    fn hc_pre(
        &self,
        il: usize,
        is_attn: bool,
        scale: &[f32],
        base: &[f32],
        residual_hc: &[f32],
        out: &mut [f32],
        post: &mut [f32],
        comb: &mut [f32],
    ) {
        if self.use_metal && self.gpu_layers.is_some() && !super::forward_metal::metal_moe_only() {
            let n_hc = self.cfg.n_hc;
            let n_embd = self.cfg.n_embd;
            let hc_dim = n_hc * n_embd;
            let mix_len = 2 * n_hc + n_hc * n_hc;
            let lg = &self.gpu_layers.as_ref().unwrap()[il];
            let sc = self.scratch.as_ref().unwrap();
            let fn_w = if is_attn { &lg.hc_attn_fn } else { &lg.hc_ffn_fn };
            let scale_b = if is_attn {
                &lg.hc_attn_scale
            } else {
                &lg.hc_ffn_scale
            };
            let base_b = if is_attn {
                &lg.hc_attn_base
            } else {
                &lg.hc_ffn_base
            };
            super::metal_ctx::Dsv4Metal::write_f32(&sc.hc_flat, residual_hc);
            let cmd = self.metal.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            // rms flat in-place via out=mix temp: write to qa then... use mix as flat out
            self.metal.encode_rms_norm(
                &enc,
                &sc.hc_flat,
                None,
                &sc.x, // flat normalized (hc_dim may exceed x — x is 16384, hc_dim=16384 OK)
                hc_dim as i32,
                self.cfg.rms_eps,
            );
            self.metal.encode_matvec_kind(
                &enc,
                fn_w.kind,
                &fn_w.buf,
                &sc.x,
                &sc.mix,
                mix_len as i32,
                hc_dim as i32,
                0,
            );
            self.metal.encode_hc_split_sinkhorn(
                &enc,
                &sc.mix,
                scale_b,
                base_b,
                &sc.split,
                n_hc as i32,
                self.cfg.n_hc_sinkhorn_iter as i32,
                self.cfg.hc_eps,
            );
            // pre weights are first n_hc of split
            self.metal.encode_hc_weighted_sum(
                &enc,
                &sc.hc_flat,
                &sc.split,
                &sc.attn_in,
                n_embd as i32,
                n_hc as i32,
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            let got = super::metal_ctx::Dsv4Metal::read_f32(&sc.attn_in, n_embd);
            out.copy_from_slice(&got);
            let split = super::metal_ctx::Dsv4Metal::read_f32(&sc.split, mix_len);
            post.copy_from_slice(&split[n_hc..2 * n_hc]);
            comb.copy_from_slice(&split[2 * n_hc..]);
            return;
        }
        let n_hc = self.cfg.n_hc;
        let n_embd = self.cfg.n_embd;
        let hc_dim = n_hc * n_embd;
        let mix_len = 2 * n_hc + n_hc * n_hc;
        let mut flat = vec![0.0f32; hc_dim];
        rms_norm_no_weight(&mut flat, residual_hc, self.cfg.rms_eps);
        let mut mix = vec![0.0f32; mix_len];
        let fn_w = if is_attn {
            &self.layers[il].hc_attn_fn
        } else {
            &self.layers[il].hc_ffn_fn
        };
        fn_w.matvec_ggml(&flat, &mut mix);
        let mut split = vec![0.0f32; mix_len];
        let scale3 = [
            scale.first().copied().unwrap_or(1.0),
            scale.get(1).copied().unwrap_or(1.0),
            scale.get(2).copied().unwrap_or(1.0),
        ];
        hc_split_sinkhorn(
            &mut split,
            &mix,
            &scale3,
            base,
            n_hc,
            self.cfg.n_hc_sinkhorn_iter,
            self.cfg.hc_eps,
        );
        hc_weighted_sum(out, residual_hc, &split[..n_hc], n_embd, n_hc);
        post.copy_from_slice(&split[n_hc..2 * n_hc]);
        comb.copy_from_slice(&split[2 * n_hc..]);
    }

    fn layer_attention(&mut self, il: usize, attn_in: &[f32], attn_out: &mut [f32]) {
        let cfg = self.cfg.clone();
        let n_embd = cfg.n_embd;
        let head_dim = cfg.head_dim;
        let n_head = cfg.n_head;
        let n_rot = cfg.n_rot;
        let ratio = cfg.compress_ratio(il);
        let metal = self.use_metal && self.gpu_layers.is_some() && !super::forward_metal::metal_moe_only();

        let mut x = vec![0.0f32; n_embd];
        rms_norm(&mut x, attn_in, &self.layers[il].attn_norm, cfg.rms_eps);

        let qa_dim = self.layers[il]
            .attn_q_a
            .dims()
            .get(1)
            .copied()
            .unwrap_or(cfg.n_lora_q);
        let q_out = self.layers[il]
            .attn_q_b
            .dims()
            .get(1)
            .copied()
            .unwrap_or(n_head * head_dim);
        let mut q = vec![0.0f32; q_out];
        let mut kv_raw = vec![0.0f32; head_dim];

        if metal {
            let lg = &self.gpu_layers.as_ref().unwrap()[il];
            let sc = self.scratch.as_ref().unwrap();
            super::metal_ctx::Dsv4Metal::write_f32(&sc.x, &x);
            let cmd = self.metal.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            // qa
            self.metal.encode_matvec_kind(
                &enc,
                lg.attn_q_a.kind,
                &lg.attn_q_a.buf,
                &sc.x,
                &sc.qa,
                qa_dim as i32,
                n_embd as i32,
                0,
            );
            // rms qa
            let w_qa = lg.attn_q_a_norm.as_ref();
            self.metal.encode_rms_norm(
                &enc,
                &sc.qa,
                w_qa,
                &sc.qa_n,
                qa_dim as i32,
                cfg.rms_eps,
            );
            // qb
            self.metal.encode_matvec_kind(
                &enc,
                lg.attn_q_b.kind,
                &lg.attn_q_b.buf,
                &sc.qa_n,
                &sc.q_heads,
                q_out as i32,
                qa_dim as i32,
                0,
            );
            // kv
            self.metal.encode_matvec_kind(
                &enc,
                lg.attn_kv.kind,
                &lg.attn_kv.buf,
                &sc.x,
                &sc.kv_lat,
                head_dim as i32,
                n_embd as i32,
                0,
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            q = super::metal_ctx::Dsv4Metal::read_f32(&sc.q_heads, q_out);
            kv_raw = super::metal_ctx::Dsv4Metal::read_f32(&sc.kv_lat, head_dim);
        } else {
            let mut qa = vec![0.0f32; qa_dim];
            self.layers[il].attn_q_a.matvec_ggml(&x, &mut qa);
            let mut qa_n = vec![0.0f32; qa.len()];
            if self.layers[il].attn_q_a_norm.len() == qa.len() {
                rms_norm(
                    &mut qa_n,
                    &qa,
                    &self.layers[il].attn_q_a_norm,
                    cfg.rms_eps,
                );
            } else {
                rms_norm_no_weight(&mut qa_n, &qa, cfg.rms_eps);
            }
            self.layers[il].attn_q_b.matvec_ggml(&qa_n, &mut q);
            self.layers[il].attn_kv.matvec_ggml(&x, &mut kv_raw);
        }

        let n_head_eff = q.len() / head_dim;
        for h in 0..n_head_eff {
            let off = h * head_dim;
            let tmp = q[off..off + head_dim].to_vec();
            let mut out_h = vec![0.0f32; head_dim];
            rms_norm_no_weight(&mut out_h, &tmp, cfg.rms_eps);
            q[off..off + head_dim].copy_from_slice(&out_h);
        }

        let mut kv = vec![0.0f32; head_dim];
        if self.layers[il].attn_kv_a_norm.len() == head_dim {
            rms_norm(
                &mut kv,
                &kv_raw,
                &self.layers[il].attn_kv_a_norm,
                cfg.rms_eps,
            );
        } else {
            kv.copy_from_slice(&kv_raw);
        }

        let compress = ratio != 0;
        apply_rope_all_heads(
            &mut q,
            n_head_eff,
            head_dim,
            n_rot,
            self.pos,
            compress,
            false,
        );
        if compress {
            super::attn::rope_tail_compress_inplace(&mut kv, head_dim, n_rot, self.pos, false);
        } else {
            super::attn::rope_tail_inplace(
                &mut kv,
                head_dim,
                n_rot,
                self.pos,
                cfg.rope_freq_base,
                false,
            );
        }
        super::compressor::fp8_e4m3fn_nope_inplace(&mut kv, n_rot);

        self.kv[il].push_raw(&kv, &kv, self.pos);

        let rope_freq = if compress {
            cfg.compress_rope_freq_base
        } else {
            cfg.rope_freq_base
        };
        if let Some(comp_w) = self.layers[il].attn_compressor.as_ref() {
            if let Some(st) = self.attn_comp_state[il].as_mut() {
                if let Some(row) = super::compressor::compressor_step(
                    st,
                    &x,
                    &comp_w.kv,
                    &comp_w.gate,
                    &comp_w.ape,
                    &comp_w.norm,
                    self.pos,
                    n_rot,
                    il,
                    rope_freq,
                    cfg.rms_eps,
                    true,
                ) {
                    self.kv[il].push_compressed(&row, &row);
                }
            }
        }
        if let Some(comp_w) = self.layers[il].indexer_compressor.as_ref() {
            if let Some(st) = self.idx_comp_state[il].as_mut() {
                let _ = super::compressor::compressor_step(
                    st,
                    &x,
                    &comp_w.kv,
                    &comp_w.gate,
                    &comp_w.ape,
                    &comp_w.norm,
                    self.pos,
                    n_rot,
                    il,
                    rope_freq,
                    cfg.rms_eps,
                    true,
                );
            }
        }

        let mut head_out = vec![0.0f32; n_head_eff * head_dim];
        let sinks = Some(self.layers[il].attn_sinks.as_slice());
        if metal && ratio == 0 {
            let n_kv = self.kv[il].raw_len;
            let scale = 1.0 / (head_dim as f32).sqrt();
            let q_buf = self.metal.buffer_from_f32(&q);
            let k_buf = self
                .metal
                .buffer_from_f32(&self.kv[il].raw_k[..n_kv * head_dim]);
            let sinks_buf = self.metal.buffer_from_f32(&self.layers[il].attn_sinks);
            let out_buf = self.metal.buffer_zeros(n_head_eff * head_dim * 4);
            let cmd = self.metal.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            self.metal.encode_attn_swa(
                &enc,
                &q_buf,
                &k_buf,
                &k_buf,
                &sinks_buf,
                &out_buf,
                n_head_eff as i32,
                head_dim as i32,
                n_kv as i32,
                1,
                scale,
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            head_out = super::metal_ctx::Dsv4Metal::read_f32(&out_buf, n_head_eff * head_dim);
        } else if ratio == 0 {
            let n_kv = self.kv[il].raw_len;
            attn_swa_mqa(
                &mut head_out,
                &q,
                &self.kv[il].raw_k[..n_kv * head_dim],
                &self.kv[il].raw_k[..n_kv * head_dim],
                sinks,
                n_head_eff,
                head_dim,
                n_kv,
            );
        } else {
            let comp_idx = if ratio == 4 {
                select_compressed_rows(
                    &q,
                    &self.kv[il],
                    cfg.n_indexer_top_k,
                    cfg.n_indexer_head_dim,
                )
            } else {
                (0..self.kv[il].comp_len).collect()
            };
            attn_mixed_mqa(
                &mut head_out,
                &q,
                &self.kv[il],
                &comp_idx,
                sinks,
                n_head_eff,
                head_dim,
            );
        }

        apply_rope_all_heads(
            &mut head_out,
            n_head_eff,
            head_dim,
            n_rot,
            self.pos,
            compress,
            true,
        );

        let n_groups = cfg.n_out_group;
        let group_heads = n_head_eff / n_groups;
        let group_dim = head_dim * group_heads;
        let rank = cfg.n_lora_o;
        let mut low = vec![0.0f32; n_groups * rank];
        for g in 0..n_groups {
            let xg = &head_out[g * group_dim..(g + 1) * group_dim];
            if metal {
                self.matvec_attn_oa_rows(
                    il,
                    xg,
                    g * rank,
                    rank,
                    &mut low[g * rank..(g + 1) * rank],
                );
            } else {
                self.layers[il].attn_output_a.matvec_rows(
                    xg,
                    g * rank,
                    rank,
                    group_dim,
                    &mut low[g * rank..(g + 1) * rank],
                );
            }
        }
        if metal {
            self.matvec_dense_gpu(il, DenseGpuOp::AttnOb, &low, attn_out);
        } else {
            self.layers[il].attn_output_b.matvec_ggml(&low, attn_out);
        }
    }

    fn layer_ffn(&mut self, il: usize, token: usize, ffn_in: &[f32], ffn_out: &mut [f32]) {
        if self.use_metal && self.gpu_layers.is_some() {
            self.layer_ffn_metal(il, token, ffn_in, ffn_out);
            return;
        }
        let cfg = self.cfg.clone();
        let n_embd = cfg.n_embd;
        let n_ff = cfg.n_ff_exp;

        let mut x = vec![0.0f32; n_embd];
        rms_norm(&mut x, ffn_in, &self.layers[il].ffn_norm, cfg.rms_eps);

        let mut logits = vec![0.0f32; cfg.n_expert];
        self.layers[il].ffn_gate_inp.matvec_ggml(&x, &mut logits);
        let mut probs = vec![0.0f32; cfg.n_expert];
        router_probs_sqrt_softplus(&logits, &mut probs);

        let (ids, weights) = if let Some(ref tid2eid) = self.layers[il].tid2eid {
            hash_experts_for_token(
                tid2eid,
                cfg.n_vocab,
                token,
                cfg.n_expert_used,
                &probs,
                cfg.expert_weight_scale,
            )
        } else {
            select_topk_experts(
                &probs,
                self.layers[il].exp_probs_b.as_deref(),
                cfg.n_expert_used,
                cfg.expert_weight_scale,
            )
        };

        let mut gate_s = vec![0.0f32; n_ff];
        let mut up_s = vec![0.0f32; n_ff];
        self.layers[il].ffn_gate_shexp.matvec_ggml(&x, &mut gate_s);
        self.layers[il].ffn_up_shexp.matvec_ggml(&x, &mut up_s);
        let mut mid_s = vec![0.0f32; n_ff];
        swiglu(&mut mid_s, &gate_s, &up_s, cfg.swiglu_clamp_exp);
        let mut shared = vec![0.0f32; n_embd];
        self.layers[il]
            .ffn_down_shexp
            .matvec_ggml(&mid_s, &mut shared);

        let keys: Vec<ExpertKey> = ids
            .iter()
            .map(|&e| ExpertKey {
                layer: il as u16,
                expert: e as u16,
            })
            .collect();
        let slots = self.ssd.pin(&keys).expect("SSD pin");
        let mut routed = vec![0.0f32; n_embd];
        for (si, &w) in slots.iter().zip(weights.iter()) {
            let gate_w = self.ssd.gate(*si);
            let up_w = self.ssd.up(*si);
            let down_w = self.ssd.down(*si);
            let mut gate = vec![0.0f32; n_ff];
            let mut up = vec![0.0f32; n_ff];
            if self.expert_gate_type == ggml_type::IQ2_XXS {
                matvec_iq2_xxs(gate_w, &x, n_ff, n_embd, &mut gate);
                matvec_iq2_xxs(up_w, &x, n_ff, n_embd, &mut up);
            } else {
                panic!(
                    "unsupported expert gate type {}",
                    crate::gguf::ggml_type_name(self.expert_gate_type)
                );
            }
            let mut mid = vec![0.0f32; n_ff];
            swiglu(&mut mid, &gate, &up, cfg.swiglu_clamp_exp);
            let mut down = vec![0.0f32; n_embd];
            if self.expert_down_type == ggml_type::Q2_K {
                matvec_q2_k(down_w, &mid, n_embd, n_ff, &mut down);
            } else {
                panic!(
                    "unsupported expert down type {}",
                    crate::gguf::ggml_type_name(self.expert_down_type)
                );
            }
            for i in 0..n_embd {
                routed[i] += w * down[i];
            }
        }
        for i in 0..n_embd {
            ffn_out[i] = shared[i] + routed[i];
        }
    }

    pub fn forward_token_logits(&mut self, token: usize) -> Vec<f32> {
        let cfg = self.cfg.clone();
        let n_embd = cfg.n_embd;
        let n_hc = cfg.n_hc;

        let emb = self.embed(token);
        hc_from_plain_embedding(&mut self.hc, &emb, n_embd, n_hc);

        for il in 0..cfg.n_layer {
            let residual = self.hc.clone();
            let mut attn_in = vec![0.0f32; n_embd];
            let mut post = vec![0.0f32; n_hc];
            let mut comb = vec![0.0f32; n_hc * n_hc];
            self.hc_pre(
                il,
                true,
                &self.layers[il].hc_attn_scale.clone(),
                &self.layers[il].hc_attn_base.clone(),
                &residual,
                &mut attn_in,
                &mut post,
                &mut comb,
            );
            let mut attn_out = vec![0.0f32; n_embd];
            self.layer_attention(il, &attn_in, &mut attn_out);
            let mut hc_mid = vec![0.0f32; n_embd * n_hc];
            hc_expand_post(
                &mut hc_mid,
                &attn_out,
                &residual,
                &post,
                &comb,
                n_embd,
                n_hc,
            );

            let residual2 = hc_mid.clone();
            let mut ffn_in = vec![0.0f32; n_embd];
            let mut post2 = vec![0.0f32; n_hc];
            let mut comb2 = vec![0.0f32; n_hc * n_hc];
            self.hc_pre(
                il,
                false,
                &self.layers[il].hc_ffn_scale.clone(),
                &self.layers[il].hc_ffn_base.clone(),
                &residual2,
                &mut ffn_in,
                &mut post2,
                &mut comb2,
            );
            let mut ffn_out = vec![0.0f32; n_embd];
            self.layer_ffn(il, token, &ffn_in, &mut ffn_out);
            hc_expand_post(
                &mut self.hc,
                &ffn_out,
                &residual2,
                &post2,
                &comb2,
                n_embd,
                n_hc,
            );
        }

        let hc_dim = n_hc * n_embd;
        let mut flat = vec![0.0f32; hc_dim];
        rms_norm_no_weight(&mut flat, &self.hc, cfg.rms_eps);
        let mut pre = vec![0.0f32; n_hc];
        if self.use_metal && self.gpu_output_hc_fn.is_some() {
            self.matvec_dense_gpu(0, DenseGpuOp::OutputHcFn, &flat, &mut pre);
        } else {
            self.output_hc_fn.matvec_ggml(&flat, &mut pre);
        }
        let scale0 = self.output_hc_scale.first().copied().unwrap_or(1.0);
        let mut w = vec![0.0f32; n_hc];
        for i in 0..n_hc {
            let z = pre[i] * scale0 + self.output_hc_base.get(i).copied().unwrap_or(0.0);
            w[i] = 1.0 / (1.0 + (-z).exp()) + cfg.hc_eps;
        }
        let mut hidden = vec![0.0f32; n_embd];
        hc_weighted_sum(&mut hidden, &self.hc, &w, n_embd, n_hc);
        let mut normed = vec![0.0f32; n_embd];
        rms_norm(&mut normed, &hidden, &self.output_norm, cfg.rms_eps);
        let mut logits = vec![0.0f32; cfg.n_vocab];
        if self.use_metal && self.gpu_output.is_some() {
            self.matvec_output_metal(&normed, &mut logits);
        } else {
            self.output_weight.matvec_ggml(&normed, &mut logits);
        }
        self.pos += 1;
        logits
    }

    pub fn forward_prefill(&mut self, tokens: &[usize]) -> Vec<f32> {
        let chunk = std::env::var("DSV4_PREFILL_CHUNK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(tokens.len().max(1));
        let mut logits = Vec::new();
        for chunk_toks in tokens.chunks(chunk.max(1)) {
            for &t in chunk_toks {
                logits = self.forward_token_logits(t);
            }
        }
        logits
    }

    pub fn sample_greedy(logits: &[f32]) -> usize {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn generate(&mut self, prompt_tokens: &[usize], n_new: usize) -> Vec<usize> {
        self.generate_with_callback(prompt_tokens, n_new, |_| {})
    }

    pub fn generate_with_callback<F: FnMut(usize)>(
        &mut self,
        prompt_tokens: &[usize],
        n_new: usize,
        mut on_token: F,
    ) -> Vec<usize> {
        self.reset();
        let mut logits = self.forward_prefill(prompt_tokens);
        let mut out = Vec::new();
        for _ in 0..n_new {
            let next = Self::sample_greedy(&logits);
            out.push(next);
            on_token(next);
            logits = self.forward_token_logits(next);
        }
        out
    }
}
