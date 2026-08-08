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
use std::time::Instant;

impl Dsv4GpuModel {
    /// Encode HC-attn pre + QKV (+ optional SWA/CSA/HCA) into an existing encoder.
    fn encode_layer_cb1(
        metal: &super::metal_ctx::Dsv4Metal,
        lg: &super::forward_metal::LayerGpu,
        sc: &super::metal_ctx::MetalScratch,
        gkv: Option<&super::gpu_kv::GpuLayerKv>,
        enc: &metal::ComputeCommandEncoderRef,
        il: usize,
        hc_dim: usize,
        mix_len: usize,
        n_embd: usize,
        n_hc: usize,
        sinkhorn_iters: i32,
        hc_eps: f32,
        qa_dim: usize,
        q_out: usize,
        head_dim: usize,
        n_head_eff: usize,
        n_rot: usize,
        pos: usize,
        rms_eps: f32,
        rope_freq_base: f32,
        compress_rope_freq_base: f32,
        n_swa: usize,
        n_indexer_top_k: usize,
        n_indexer_head_dim: usize,
        swa_raw_len: usize,
        comp_len: usize,
        ratio: u32,
        gpu_swa: bool,
        gpu_full: bool,
    ) {
        metal.encode_rms_norm(enc, &sc.hc_flat, None, &sc.x, hc_dim as i32, rms_eps);
        metal.encode_matvec_kind(
            enc,
            lg.hc_attn_fn.kind,
            &lg.hc_attn_fn.buf,
            &sc.x,
            &sc.mix,
            mix_len as i32,
            hc_dim as i32,
            0,
        );
        metal.encode_hc_split_sinkhorn(
            enc,
            &sc.mix,
            &lg.hc_attn_scale,
            &lg.hc_attn_base,
            &sc.split,
            n_hc as i32,
            sinkhorn_iters,
            hc_eps,
        );
        metal.encode_hc_weighted_sum(
            enc,
            &sc.hc_flat,
            &sc.split,
            &sc.attn_in,
            n_embd as i32,
            n_hc as i32,
        );
        metal.encode_rms_norm(
            enc,
            &sc.attn_in,
            Some(&lg.attn_norm),
            &sc.x,
            n_embd as i32,
            rms_eps,
        );
        metal.encode_matvec_kind(
            enc,
            lg.attn_q_a.kind,
            &lg.attn_q_a.buf,
            &sc.x,
            &sc.qa,
            qa_dim as i32,
            n_embd as i32,
            0,
        );
        metal.encode_rms_norm(
            enc,
            &sc.qa,
            lg.attn_q_a_norm.as_ref(),
            &sc.qa_n,
            qa_dim as i32,
            rms_eps,
        );
        metal.encode_matvec_kind(
            enc,
            lg.attn_q_b.kind,
            &lg.attn_q_b.buf,
            &sc.qa_n,
            &sc.q_heads,
            q_out as i32,
            qa_dim as i32,
            0,
        );
        metal.encode_matvec_kind(
            enc,
            lg.attn_kv.kind,
            &lg.attn_kv.buf,
            &sc.x,
            &sc.kv_lat,
            head_dim as i32,
            n_embd as i32,
            0,
        );
        if gpu_swa {
            let kv_cache = if il == 0 { &sc.swa_kv0 } else { &sc.swa_kv1 };
            metal.encode_swa_rope_attn(
                enc,
                &sc.q_heads,
                &sc.kv_lat,
                kv_cache,
                lg.attn_kv_a_norm.as_ref(),
                &lg.attn_sinks,
                &sc.head_out,
                n_head_eff as i32,
                head_dim as i32,
                n_rot as i32,
                pos as i32,
                rope_freq_base,
                rms_eps,
                swa_raw_len as i32,
                n_swa as i32,
            );
        } else if gpu_full {
            let gkv = gkv.expect("gpu_kv for gpu_full");
            let attn_comp = match (
                lg.attn_comp_kv.as_ref(),
                lg.attn_comp_gate.as_ref(),
                lg.attn_comp_ape.as_ref(),
                lg.attn_comp_norm.as_ref(),
                gkv.state_kv.as_ref(),
                gkv.state_score.as_ref(),
            ) {
                (Some(wkv), Some(wgate), Some(ape), Some(norm), Some(sk), Some(ss))
                    if gkv.width > 0 =>
                {
                    Some((
                        wkv,
                        wgate,
                        ape,
                        norm,
                        sk,
                        ss,
                        gkv.width as i32,
                        head_dim as i32,
                    ))
                }
                _ => None,
            };
            let idx_comp = match (
                lg.idx_comp_kv.as_ref(),
                lg.idx_comp_gate.as_ref(),
                lg.idx_comp_ape.as_ref(),
                lg.idx_comp_norm.as_ref(),
                gkv.idx_state_kv.as_ref(),
                gkv.idx_state_score.as_ref(),
            ) {
                (Some(wkv), Some(wgate), Some(ape), Some(norm), Some(sk), Some(ss))
                    if gkv.idx_width > 0 =>
                {
                    Some((
                        wkv,
                        wgate,
                        ape,
                        norm,
                        sk,
                        ss,
                        gkv.idx_width as i32,
                        n_indexer_head_dim as i32,
                    ))
                }
                _ => None,
            };
            metal.encode_fused_csa_hca(
                enc,
                &sc.q_heads,
                &sc.kv_lat,
                &gkv.raw,
                &gkv.comp,
                lg.attn_kv_a_norm.as_ref(),
                &lg.attn_sinks,
                &sc.head_out,
                &sc.x,
                &sc.gate,
                &sc.up,
                &sc.mid,
                &sc.down,
                &sc.comp_emit,
                &sc.comp_idx,
                &sc.comp_scores,
                attn_comp,
                idx_comp,
                n_head_eff as i32,
                head_dim as i32,
                n_rot as i32,
                pos as i32,
                rms_eps,
                n_embd as i32,
                swa_raw_len as i32,
                n_swa as i32,
                comp_len as i32,
                ratio as i32,
                n_indexer_top_k as i32,
                true,
                compress_rope_freq_base,
            );
        }
    }

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
            self.metal.wait_cmd(&cmd, self.scratch.as_ref());
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
            self.metal.wait_cmd(&cmd, self.scratch.as_ref());
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
            let sc = self.scratch.as_ref().unwrap();
            let need = n_kv * head_dim;
            if need > sc.kv_cache_cap {
                // Fall back to one-shot buffers if SWA window exceeds scratch.
                let q_buf = self.metal.buffer_from_f32(&q);
                let k_buf = self
                    .metal
                    .buffer_from_f32(&self.kv[il].raw_k[..need]);
                let out_buf = self.metal.buffer_zeros(n_head_eff * head_dim * 4);
                let cmd = self.metal.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                self.metal.encode_attn_swa(
                    &enc,
                    &q_buf,
                    &k_buf,
                    &k_buf,
                    &self.gpu_layers.as_ref().unwrap()[il].attn_sinks,
                    &out_buf,
                    n_head_eff as i32,
                    head_dim as i32,
                    n_kv as i32,
                    1,
                    scale,
                );
                enc.end_encoding();
                cmd.commit();
                self.metal.wait_cmd(&cmd, Some(sc));
                head_out = super::metal_ctx::Dsv4Metal::read_f32(&out_buf, n_head_eff * head_dim);
            } else {
                super::metal_ctx::Dsv4Metal::write_f32(&sc.q_heads, &q);
                super::metal_ctx::Dsv4Metal::write_f32(
                    &sc.kv_cache,
                    &self.kv[il].raw_k[..need],
                );
                let cmd = self.metal.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                self.metal.encode_attn_swa(
                    &enc,
                    &sc.q_heads,
                    &sc.kv_cache,
                    &sc.kv_cache,
                    &self.gpu_layers.as_ref().unwrap()[il].attn_sinks,
                    &sc.head_out,
                    n_head_eff as i32,
                    head_dim as i32,
                    n_kv as i32,
                    1,
                    scale,
                );
                enc.end_encoding();
                cmd.commit();
                self.metal.wait_cmd(&cmd, Some(sc));
                head_out =
                    super::metal_ctx::Dsv4Metal::read_f32(&sc.head_out, n_head_eff * head_dim);
            }
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
        if metal {
            // One CB: all grouped LoRA-O rows + output_b (was 9 separate waits).
            let lg = &self.gpu_layers.as_ref().unwrap()[il];
            let sc = self.scratch.as_ref().unwrap();
            super::metal_ctx::Dsv4Metal::write_f32(&sc.head_out, &head_out);
            let row_bytes = match lg.attn_output_a.kind {
                super::metal_ctx::GpuWKind::F16 => group_dim * 2,
                super::metal_ctx::GpuWKind::F32 => group_dim * 4,
                super::metal_ctx::GpuWKind::Q8 => (group_dim / 32) * 34,
                _ => panic!("LoRA-O A must be dense"),
            };
            let cmd = self.metal.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            for g in 0..n_groups {
                self.metal.encode_matvec_kind_off(
                    &enc,
                    lg.attn_output_a.kind,
                    &lg.attn_output_a.buf,
                    &sc.head_out,
                    &sc.low,
                    rank as i32,
                    group_dim as i32,
                    (g * rank * row_bytes) as u64,
                    (g * group_dim * 4) as u64,
                    (g * rank * 4) as u64,
                );
            }
            self.metal.encode_matvec_kind(
                &enc,
                lg.attn_output_b.kind,
                &lg.attn_output_b.buf,
                &sc.low,
                &sc.attn_in,
                n_embd as i32,
                (n_groups * rank) as i32,
                0,
            );
            enc.end_encoding();
            cmd.commit();
            self.metal.wait_cmd(&cmd, Some(sc));
            let got = super::metal_ctx::Dsv4Metal::read_f32(&sc.attn_in, n_embd);
            attn_out.copy_from_slice(&got);
        } else {
            for g in 0..n_groups {
                let xg = &head_out[g * group_dim..(g + 1) * group_dim];
                self.layers[il].attn_output_a.matvec_rows(
                    xg,
                    g * rank,
                    rank,
                    group_dim,
                    &mut low[g * rank..(g + 1) * rank],
                );
            }
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

    /// Record routed expert ids for speculative pin (keep t−1 and t−2).
    fn note_expert_ids(&mut self, il: usize, ids: Vec<u32>) {
        self.prev_expert_ids[il] = std::mem::replace(&mut self.last_expert_ids[il], ids);
    }

    /// Fused Metal layer: ~2–3 GPU waits (vs ~6 with decomposed hc/attn/ffn).
    /// Hash-MoE pins experts before CB1 so SSD I/O overlaps later GPU work.
    fn layer_forward_metal(&mut self, il: usize, token: usize) {
        let profile = std::env::var("DSV4_PROFILE").ok().as_deref() == Some("1");
        // Ablation knobs (PROFILE ceiling micro-bench; wrong text OK).
        let skip_sinkhorn = std::env::var("DSV4_SKIP_SINKHORN").ok().as_deref() == Some("1");
        let skip_hc_expand = std::env::var("DSV4_SKIP_HC_EXPAND").ok().as_deref() == Some("1");
        let skip_moe_routed = std::env::var("DSV4_SKIP_MOE_ROUTED").ok().as_deref() == Some("1");
        let skip_shared = std::env::var("DSV4_SKIP_SHARED").ok().as_deref() == Some("1");
        let t_layer = Instant::now();
        let mut t_attn = 0u64;
        let mut t_gpu = 0u64;
        let mut t_exp = 0u64;
        let cfg = self.cfg.clone();
        let n_embd = cfg.n_embd;
        let n_hc = cfg.n_hc;
        let head_dim = cfg.head_dim;
        let n_rot = cfg.n_rot;
        let ratio = cfg.compress_ratio(il);
        let hc_dim = n_hc * n_embd;
        let mix_len = 2 * n_hc + n_hc * n_hc;
        let n_ff = cfg.n_ff_exp;
        let sinkhorn_iters = std::env::var("DSV4_SINKHORN_ITERS")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(if skip_sinkhorn {
                1
            } else {
                cfg.n_hc_sinkhorn_iter as i32
            });
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
            .unwrap_or(cfg.n_head * head_dim);

        // Hash keys known up-front; pin overlaps CB1 GPU (after commit below).
        // Top-k: speculatively pin union of last two tokens' experts (routing often
        // stable; union raises hit rate when one expert churns per step).
        let hash_keys: Option<Vec<ExpertKey>> =
            self.layers[il].tid2eid.as_ref().map(|tid2eid| {
                let k = cfg.n_expert_used;
                let row = &tid2eid[token * k..(token + 1) * k];
                row.iter()
                    .map(|&e| ExpertKey {
                        layer: il as u16,
                        expert: e as u16,
                    })
                    .collect()
            });
        // Speculative top-k pin (default on). DSV4_SPEC_PIN=0 disables.
        // DSV4_SPEC_PIN_HITS_ONLY=1 skips when the union is not fully resident.
        // DSV4_SPEC_PIN_UNION=0 pins only the previous token (not t−2).
        let spec_keys: Option<Vec<ExpertKey>> = if hash_keys.is_some()
            || std::env::var("DSV4_SPEC_PIN").ok().as_deref() == Some("0")
        {
            None
        } else {
            let use_union = std::env::var("DSV4_SPEC_PIN_UNION").ok().as_deref() != Some("0");
            let mut keys: Vec<ExpertKey> = Vec::with_capacity(cfg.n_expert_used * 2);
            let mut push_ids = |ids: &[u32]| {
                for &e in ids {
                    let k = ExpertKey {
                        layer: il as u16,
                        expert: e as u16,
                    };
                    if !keys.iter().any(|x| *x == k) {
                        keys.push(k);
                    }
                }
            };
            if self.last_expert_ids[il].len() == cfg.n_expert_used {
                push_ids(&self.last_expert_ids[il]);
            }
            if use_union && self.prev_expert_ids[il].len() == cfg.n_expert_used {
                push_ids(&self.prev_expert_ids[il]);
            }
            if keys.is_empty() {
                None
            } else {
                let hits_only =
                    std::env::var("DSV4_SPEC_PIN_HITS_ONLY").ok().as_deref() == Some("1");
                if hits_only {
                    let eg = self.expert_gpu.as_ref().unwrap();
                    if keys.iter().all(|k| eg.contains(*k)) {
                        Some(keys)
                    } else {
                        None
                    }
                } else {
                    Some(keys)
                }
            }
        };
        let early_pin_keys = hash_keys.as_ref().or(spec_keys.as_ref());

        let n_head_eff = q_out / head_dim;
        let gpu_swa = ratio == 0 && super::forward_metal::gpu_swa_enabled();
        let gpu_full = ratio != 0 && super::forward_metal::gpu_full_enabled();
        // Hash layers: CB1+CB2+MoE in one CB when experts are (or will be) resident
        // and attn stays on GPU (no host head_out round-trip).
        let hash_one_cb = hash_keys.is_some()
            && (gpu_swa || gpu_full)
            && std::env::var("DSV4_HASH_ONE_CB").ok().as_deref() != Some("0");
        let swa_raw_len = self.kv[il].raw_len;
        let comp_len = self.kv[il].comp_len;

        // HC lives in scratch.hc_flat across fused layers (uploaded once per token).
        // When attn stays on GPU (SWA/CSA/HCA), merge CB1 into CB2 (one commit).
        // Hash layers additionally fuse MoE into that same CB when experts are resident.
        let merge_cb1 = (gpu_swa || gpu_full)
            && std::env::var("DSV4_MERGE_CB1").ok().as_deref() != Some("0");
        let mut hash_fused_slots: Vec<usize> = Vec::new();
        {
            // I/O-first: start hash/spec preads; when merge_cb1, pin overlaps nothing
            // here (CB1 kernels run in CB2) — pin still overlaps prior deferred MoE.
            let encode_cb1 = || {
                let lg = &self.gpu_layers.as_ref().unwrap()[il];
                let sc = self.scratch.as_ref().unwrap();
                let cmd = self.metal.queue.new_command_buffer().to_owned();
                let enc = cmd.new_compute_command_encoder();
                Self::encode_layer_cb1(
                    &self.metal,
                    lg,
                    sc,
                    self.gpu_kv.as_ref().map(|v| &v[il]),
                    &enc,
                    il,
                    hc_dim,
                    mix_len,
                    n_embd,
                    n_hc,
                    sinkhorn_iters,
                    cfg.hc_eps,
                    qa_dim,
                    q_out,
                    head_dim,
                    n_head_eff,
                    n_rot,
                    self.pos,
                    cfg.rms_eps,
                    cfg.rope_freq_base,
                    cfg.compress_rope_freq_base,
                    cfg.n_swa,
                    cfg.n_indexer_top_k,
                    cfg.n_indexer_head_dim,
                    swa_raw_len,
                    comp_len,
                    ratio,
                    gpu_swa,
                    gpu_full,
                );
                enc.end_encoding();
                cmd.commit();
                cmd
            };
            let cmd1 = if let Some(keys) = early_pin_keys {
                let metal = self.metal.clone();
                let mut eg = self.expert_gpu.take().expect("expert_gpu");
                let t_e0 = Instant::now();
                let all_hit = keys.iter().all(|k| eg.contains(*k));
                let cmd = if merge_cb1 {
                    // Pin only — CB1 kernels merge into CB2 below.
                    if all_hit {
                        for &k in keys {
                            let _ = eg.touch(k);
                        }
                    } else {
                        if let Some(sc) = self.scratch.as_ref() {
                            sc.wait_pending(&self.metal);
                        }
                        let _ = eg
                            .pin_batch_with_gpu(&metal, &self.ssd, keys, &[], || ())
                            .expect("early expert pread");
                    }
                    if hash_one_cb {
                        if let Some(ref hk) = hash_keys {
                            if hk.iter().all(|k| eg.contains(*k)) {
                                let mut tmp = vec![0usize; hk.len()];
                                let _ = eg.classify_hits(hk, &mut tmp);
                                hash_fused_slots = tmp;
                            }
                        }
                    }
                    None
                } else if all_hit {
                    // No pread — safe to overlap deferred MoE with CB1.
                    for &k in keys {
                        let _ = eg.touch(k);
                    }
                    Some(encode_cb1())
                } else {
                    // Drain deferred MoE before Shared preads (unified-memory fight).
                    if let Some(sc) = self.scratch.as_ref() {
                        sc.wait_pending(&self.metal);
                    }
                    Some(
                        eg.pin_batch_with_gpu(&metal, &self.ssd, keys, &[], encode_cb1)
                            .expect("early expert pread")
                            .0,
                    )
                };
                t_exp += t_e0.elapsed().as_nanos() as u64;
                self.expert_gpu = Some(eg);
                cmd
            } else if merge_cb1 {
                None
            } else {
                Some(encode_cb1())
            };
            // Same Metal queue: CB2 may encode/commit without waiting CB1 when
            // head_out stays on GPU (gpu_swa / gpu_full). Legacy path still waits.
            if let Some(cmd1) = cmd1 {
                if !gpu_swa && !gpu_full {
                    let sc = self.scratch.as_ref().unwrap();
                    let t0 = Instant::now();
                    self.metal.wait_owned(&cmd1, Some(sc));
                    let dt = t0.elapsed().as_nanos() as u64;
                    t_gpu += dt;
                    if profile {
                        sc.prof_cb1_ns.set(sc.prof_cb1_ns.get() + dt);
                    }
                }
            }
        }

        let mut head_out = vec![0.0f32; n_head_eff * head_dim];
        let head_out_on_gpu;
        if gpu_swa {
            // GPU SWA ring is source of truth; only advance host len/pos metadata.
            self.kv[il].note_raw_push(self.pos);
            head_out_on_gpu = true;
        } else if gpu_full {
            self.kv[il].note_raw_push(self.pos);
            if (self.pos + 1) % (ratio as usize) == 0 {
                self.kv[il].note_comp_push();
            }
            head_out_on_gpu = true;
        } else {
        let mut q = {
            let sc = self.scratch.as_ref().unwrap();
            super::metal_ctx::Dsv4Metal::read_f32(&sc.q_heads, q_out)
        };
        let kv_raw = {
            let sc = self.scratch.as_ref().unwrap();
            super::metal_ctx::Dsv4Metal::read_f32(&sc.kv_lat, head_dim)
        };
        // sc.x still holds attn-normed residual (compressor input). Keep on GPU.
        // split [pre|post|comb] stays on GPU for CB2 expand (post/comb offsets).
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

        // GPU compressor projections (ds4 graph; was CPU F16 width×4096).
        // Separate CB so hash-expert pread can keep overlapping CB1.
        let lg = &self.gpu_layers.as_ref().unwrap()[il];
        let has_attn_comp = lg.attn_comp_kv.is_some() && self.attn_comp_state[il].is_some();
        let has_idx_comp = lg.idx_comp_kv.is_some() && self.idx_comp_state[il].is_some();
        if has_attn_comp || has_idx_comp {
            let sc = self.scratch.as_ref().unwrap();
            let cmd = self.metal.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            if let (Some(wkv), Some(wgate), Some(st)) = (
                lg.attn_comp_kv.as_ref(),
                lg.attn_comp_gate.as_ref(),
                self.attn_comp_state[il].as_ref(),
            ) {
                let width = st.width as i32;
                self.metal.encode_matvec_kind(
                    &enc,
                    wkv.kind,
                    &wkv.buf,
                    &sc.x,
                    &sc.gate,
                    width,
                    n_embd as i32,
                    0,
                );
                self.metal.encode_matvec_kind(
                    &enc,
                    wgate.kind,
                    &wgate.buf,
                    &sc.x,
                    &sc.up,
                    width,
                    n_embd as i32,
                    0,
                );
            }
            if let (Some(wkv), Some(wgate), Some(st)) = (
                lg.idx_comp_kv.as_ref(),
                lg.idx_comp_gate.as_ref(),
                self.idx_comp_state[il].as_ref(),
            ) {
                let width = st.width as i32;
                self.metal.encode_matvec_kind(
                    &enc,
                    wkv.kind,
                    &wkv.buf,
                    &sc.x,
                    &sc.mid,
                    width,
                    n_embd as i32,
                    0,
                );
                self.metal.encode_matvec_kind(
                    &enc,
                    wgate.kind,
                    &wgate.buf,
                    &sc.x,
                    &sc.down,
                    width,
                    n_embd as i32,
                    0,
                );
            }
            enc.end_encoding();
            cmd.commit();
            let t0 = Instant::now();
            self.metal.wait_cmd(&cmd, Some(sc));
            t_gpu += t0.elapsed().as_nanos() as u64;
        }

        if let Some(comp_w) = self.layers[il].attn_compressor.as_ref() {
            if let Some(st) = self.attn_comp_state[il].as_mut() {
                let width = st.width;
                let sc = self.scratch.as_ref().unwrap();
                let kv_cur = super::metal_ctx::Dsv4Metal::read_f32(&sc.gate, width);
                let sc_cur = super::metal_ctx::Dsv4Metal::read_f32(&sc.up, width);
                if let Some(row) = super::compressor::compressor_step_from_proj(
                    st,
                    &kv_cur,
                    &sc_cur,
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
                let width = st.width;
                let sc = self.scratch.as_ref().unwrap();
                let kv_cur = super::metal_ctx::Dsv4Metal::read_f32(&sc.mid, width);
                let sc_cur = super::metal_ctx::Dsv4Metal::read_f32(&sc.down, width);
                let _ = super::compressor::compressor_step_from_proj(
                    st,
                    &kv_cur,
                    &sc_cur,
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

        let sinks = Some(self.layers[il].attn_sinks.as_slice());
        let t_attn0 = Instant::now();
        if ratio == 0 {
            let n_kv = self.kv[il].raw_len;
            let scale = 1.0 / (head_dim as f32).sqrt();
            let sc = self.scratch.as_ref().unwrap();
            let need = n_kv * head_dim;
            super::metal_ctx::Dsv4Metal::write_f32(&sc.q_heads, &q);
            if need > sc.kv_cache_cap {
                let k_buf = self
                    .metal
                    .buffer_from_f32(&self.kv[il].raw_k[..need]);
                let cmd = self.metal.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                self.metal.encode_attn_swa(
                    &enc,
                    &sc.q_heads,
                    &k_buf,
                    &k_buf,
                    &self.gpu_layers.as_ref().unwrap()[il].attn_sinks,
                    &sc.head_out,
                    n_head_eff as i32,
                    head_dim as i32,
                    n_kv as i32,
                    1,
                    scale,
                );
                enc.end_encoding();
                cmd.commit();
                self.metal.wait_cmd(&cmd, Some(sc));
            } else {
                super::metal_ctx::Dsv4Metal::write_f32(
                    &sc.kv_cache,
                    &self.kv[il].raw_k[..need],
                );
                let cmd = self.metal.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                self.metal.encode_attn_swa(
                    &enc,
                    &sc.q_heads,
                    &sc.kv_cache,
                    &sc.kv_cache,
                    &self.gpu_layers.as_ref().unwrap()[il].attn_sinks,
                    &sc.head_out,
                    n_head_eff as i32,
                    head_dim as i32,
                    n_kv as i32,
                    1,
                    scale,
                );
                enc.end_encoding();
                cmd.commit();
                self.metal.wait_cmd(&cmd, Some(sc));
            }
            head_out =
                super::metal_ctx::Dsv4Metal::read_f32(&sc.head_out, n_head_eff * head_dim);
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
        t_attn += t_attn0.elapsed().as_nanos() as u64;
        head_out_on_gpu = false;
        } // !gpu_swa && !gpu_full

        let n_groups = cfg.n_out_group;
        let group_heads = n_head_eff / n_groups;
        let group_dim = head_dim * group_heads;
        let rank = cfg.n_lora_o;
        let post_off = (n_hc * 4) as u64;
        let comb_off = (2 * n_hc * 4) as u64;

        // MoE encode helpers defined before CB2 wait so after route_ids we can
        // classify → pin_misses immediately (shared+hits ∥ miss preads).
        let down_kind = if self.expert_down_type == ggml_type::Q2_K {
            super::metal_ctx::GpuWKind::Q2K
        } else {
            panic!("unsupported expert down");
        };
        assert_eq!(
            self.expert_gate_type,
            ggml_type::IQ2_XXS,
            "unsupported expert gate"
        );

        let encode_shared = |metal: &super::metal_ctx::Dsv4Metal,
                             lg: &super::forward_metal::LayerGpu,
                             sc: &super::metal_ctx::MetalScratch,
                             enc: &metal::ComputeCommandEncoderRef| {
            if matches!(lg.ffn_gate_shexp.kind, super::metal_ctx::GpuWKind::Q8)
                && matches!(lg.ffn_up_shexp.kind, super::metal_ctx::GpuWKind::Q8)
            {
                metal.encode_q8_pair_swiglu(
                    enc,
                    &lg.ffn_gate_shexp.buf,
                    &lg.ffn_up_shexp.buf,
                    &sc.x,
                    &sc.mid,
                    n_ff as i32,
                    n_embd as i32,
                    cfg.swiglu_clamp_exp,
                );
            } else {
                metal.encode_matvec_kind(
                    enc,
                    lg.ffn_gate_shexp.kind,
                    &lg.ffn_gate_shexp.buf,
                    &sc.x,
                    &sc.gate,
                    n_ff as i32,
                    n_embd as i32,
                    0,
                );
                metal.encode_matvec_kind(
                    enc,
                    lg.ffn_up_shexp.kind,
                    &lg.ffn_up_shexp.buf,
                    &sc.x,
                    &sc.up,
                    n_ff as i32,
                    n_embd as i32,
                    0,
                );
                metal.encode_swiglu(
                    enc,
                    &sc.gate,
                    &sc.up,
                    &sc.mid,
                    n_ff as i32,
                    cfg.swiglu_clamp_exp,
                );
            }
            metal.encode_matvec_kind(
                enc,
                lg.ffn_down_shexp.kind,
                &lg.ffn_down_shexp.buf,
                &sc.mid,
                &sc.shared,
                n_embd as i32,
                n_ff as i32,
                0,
            );
            metal.encode_zero(enc, &sc.routed, n_embd as i32);
        };

        let encode_expert_i = |metal: &super::metal_ctx::Dsv4Metal,
                               eg: &super::expert_gpu::ExpertGpuCache,
                               sc: &super::metal_ctx::MetalScratch,
                               enc: &metal::ComputeCommandEncoderRef,
                               slots: &[usize],
                               i: usize| {
            let gsi = slots[i];
            metal.encode_iq2_pair_swiglu(
                enc,
                eg.gate(gsi),
                eg.up(gsi),
                &sc.x,
                &sc.mid,
                n_ff as i32,
                n_embd as i32,
                cfg.swiglu_clamp_exp,
            );
            metal.encode_matvec_kind(
                enc,
                down_kind,
                eg.down(gsi),
                &sc.mid,
                &sc.exp_down_out[i],
                n_embd as i32,
                n_ff as i32,
                0,
            );
            metal.encode_axpy_w(
                enc,
                &sc.exp_down_out[i],
                &sc.routed,
                &sc.route_w,
                i as i32,
                n_embd as i32,
            );
        };

        let encode_slots6_all = |metal: &super::metal_ctx::Dsv4Metal,
                                 eg: &super::expert_gpu::ExpertGpuCache,
                                 sc: &super::metal_ctx::MetalScratch,
                                 enc: &metal::ComputeCommandEncoderRef,
                                 slots: &[usize]| {
            let gates = [
                eg.gate(slots[0]),
                eg.gate(slots[1]),
                eg.gate(slots[2]),
                eg.gate(slots[3]),
                eg.gate(slots[4]),
                eg.gate(slots[5]),
            ];
            let ups = [
                eg.up(slots[0]),
                eg.up(slots[1]),
                eg.up(slots[2]),
                eg.up(slots[3]),
                eg.up(slots[4]),
                eg.up(slots[5]),
            ];
            let downs = [
                eg.down(slots[0]),
                eg.down(slots[1]),
                eg.down(slots[2]),
                eg.down(slots[3]),
                eg.down(slots[4]),
                eg.down(slots[5]),
            ];
            metal.encode_slots6_iq2_pair_swiglu(
                enc,
                gates,
                ups,
                &sc.x,
                &sc.routed_mid,
                &sc.route_w,
                n_ff as i32,
                n_embd as i32,
                cfg.swiglu_clamp_exp,
            );
            metal.encode_slots6_q2k_sum6(
                enc,
                downs,
                &sc.routed_mid,
                &sc.routed,
                n_embd as i32,
                n_ff as i32,
            );
        };

        let encode_ffn_tail = |metal: &super::metal_ctx::Dsv4Metal,
                               sc: &super::metal_ctx::MetalScratch,
                               enc: &metal::ComputeCommandEncoderRef| {
            metal.encode_add(
                enc,
                &sc.shared,
                &sc.routed,
                &sc.attn_in,
                n_embd as i32,
            );
            metal.encode_hc_expand_post_off(
                enc,
                &sc.attn_in,
                0,
                &sc.y,
                0,
                &sc.split,
                post_off,
                &sc.split,
                comb_off,
                &sc.hc_flat,
                0,
                n_embd as i32,
                n_hc as i32,
            );
        };

        let encode_expert_bufs = |metal: &super::metal_ctx::Dsv4Metal,
                                   sc: &super::metal_ctx::MetalScratch,
                                   enc: &metal::ComputeCommandEncoderRef,
                                   gate: &metal::Buffer,
                                   up: &metal::Buffer,
                                   down: &metal::Buffer,
                                   i: usize| {
            metal.encode_iq2_pair_swiglu(
                enc,
                gate,
                up,
                &sc.x,
                &sc.mid,
                n_ff as i32,
                n_embd as i32,
                cfg.swiglu_clamp_exp,
            );
            metal.encode_matvec_kind(
                enc,
                down_kind,
                down,
                &sc.mid,
                &sc.exp_down_out[i],
                n_embd as i32,
                n_ff as i32,
                0,
            );
            metal.encode_axpy_w(
                enc,
                &sc.exp_down_out[i],
                &sc.routed,
                &sc.route_w,
                i as i32,
                n_embd as i32,
            );
        };

        // Hash: bind concrete slots when resident.
        // Speculative MoE-in-CB (top-k, opt-in DSV4_SPEC_MOE=1): encode MoE with
        // last_ids slots in the same CB as router; wait once; match → done, else
        // restore attn HC snap + corrective MoE. Default off: Flash top-k churns
        // ~3/6 experts/token (exact match ~0%), so fused wait+miss is a regression
        // vs wait(router)+defer(MoE). Hash one-CB fuse stays default-on.
        // Top-k map fuse (opt-in only — default off; =1 is incorrect on misses):
        //   DSV4_FUSE_MAP=1     — defer, no repair (fast, wrong on misses)
        //   DSV4_FUSE_MAP=safe  — wait + miss_flag repair (correct)
        let fuse_map_env = std::env::var("DSV4_FUSE_MAP").ok();
        let fuse_map_optimistic = fuse_map_env.as_deref() == Some("1");
        let fuse_map_safe = fuse_map_env.as_deref() == Some("safe");
        let fuse_map = fuse_map_optimistic || fuse_map_safe;
        let spec_moe = hash_keys.is_none()
            && (gpu_swa || gpu_full)
            && std::env::var("DSV4_SPEC_MOE").ok().as_deref() == Some("1");
        let mut fused_moe = false;
        let mut fuse_slots: Vec<usize> = Vec::new();
        let mut fuse_via_map = false;
        let mut fuse_via_spec = false;
        let mut spec_guess_ids: Option<Vec<u32>> = None;
        if !hash_fused_slots.is_empty() {
            fuse_slots = hash_fused_slots;
            fused_moe = true;
        } else if let Some(ref keys) = hash_keys {
            let eg = self.expert_gpu.as_ref().unwrap();
            if keys.iter().all(|k| eg.contains(*k)) {
                let mut tmp = vec![0usize; keys.len()];
                let _ = self
                    .expert_gpu
                    .as_mut()
                    .unwrap()
                    .classify_hits(keys, &mut tmp);
                fuse_slots = tmp;
                fused_moe = !fuse_slots.is_empty();
            }
        } else if fuse_map {
            let eg = self.expert_gpu.as_ref().unwrap();
            let mut map = vec![-1i32; cfg.n_expert];
            eg.fill_slot_map_for_layer(il as u16, &mut map);
            let n_mapped = map.iter().filter(|&&s| s >= 0).count();
            if n_mapped >= cfg.n_expert_used {
                let sc = self.scratch.as_ref().unwrap();
                super::metal_ctx::Dsv4Metal::write_i32(&sc.slot_map, &map);
                fused_moe = true;
                fuse_via_map = true;
            }
        } else if spec_moe && self.last_expert_ids[il].len() == cfg.n_expert_used {
            let guess = self.last_expert_ids[il].clone();
            let keys: Vec<ExpertKey> = guess
                .iter()
                .map(|&e| ExpertKey {
                    layer: il as u16,
                    expert: e as u16,
                })
                .collect();
            let eg = self.expert_gpu.as_ref().unwrap();
            if keys.iter().all(|k| eg.contains(*k)) {
                let mut tmp = vec![0usize; keys.len()];
                let _ = self
                    .expert_gpu
                    .as_mut()
                    .unwrap()
                    .classify_hits(&keys, &mut tmp);
                fuse_slots = tmp;
                fused_moe = !fuse_slots.is_empty();
                fuse_via_spec = fused_moe;
                if fuse_via_spec {
                    spec_guess_ids = Some(guess);
                }
            }
        }

        let n_routed_known = hash_keys
            .as_ref()
            .map(|k| k.len())
            .unwrap_or(cfg.n_expert_used)
            .min(self.scratch.as_ref().unwrap().max_routed);
        let use_slots6_fuse = n_routed_known == 6
            && std::env::var("DSV4_SLOTS6").ok().as_deref() != Some("0");

        // Set when safe map-fuse needs a miss repair after the CB2 borrow scope.
        let mut map_fuse_ids: Option<Vec<u32>> = None;
        let mut map_fuse_need_repair = false;
        let mut early_fused_return = false;

        // CB2: LoRA-O + attn HC expand + ffn HC + router
        // (+ fused MoE when experts already resident).
        // When merge_cb1, CB1 attn kernels are prepended (one CB for hash = CB1+CB2+MoE).
        {
            let lg = &self.gpu_layers.as_ref().unwrap()[il];
            let sc = self.scratch.as_ref().unwrap();
            if !head_out_on_gpu {
                super::metal_ctx::Dsv4Metal::write_f32(&sc.head_out, &head_out);
            }
            // residual still in hc_flat; attn split still has post|comb
            let cmd = self.metal.queue.new_command_buffer().to_owned();
            let enc = cmd.new_compute_command_encoder();
            if merge_cb1 {
                Self::encode_layer_cb1(
                    &self.metal,
                    lg,
                    sc,
                    self.gpu_kv.as_ref().map(|v| &v[il]),
                    &enc,
                    il,
                    hc_dim,
                    mix_len,
                    n_embd,
                    n_hc,
                    sinkhorn_iters,
                    cfg.hc_eps,
                    qa_dim,
                    q_out,
                    head_dim,
                    n_head_eff,
                    n_rot,
                    self.pos,
                    cfg.rms_eps,
                    cfg.rope_freq_base,
                    cfg.compress_rope_freq_base,
                    cfg.n_swa,
                    cfg.n_indexer_top_k,
                    cfg.n_indexer_head_dim,
                    swa_raw_len,
                    comp_len,
                    ratio,
                    gpu_swa,
                    gpu_full,
                );
            }
            if matches!(lg.attn_output_a.kind, super::metal_ctx::GpuWKind::F16) {
                self.metal.encode_matvec_f16_lora_groups(
                    &enc,
                    &lg.attn_output_a.buf,
                    &sc.head_out,
                    &sc.low,
                    n_groups as i32,
                    rank as i32,
                    group_dim as i32,
                );
            } else if matches!(lg.attn_output_a.kind, super::metal_ctx::GpuWKind::Q8) {
                self.metal.encode_matvec_q8_lora_groups(
                    &enc,
                    &lg.attn_output_a.buf,
                    &sc.head_out,
                    &sc.low,
                    n_groups as i32,
                    rank as i32,
                    group_dim as i32,
                );
            } else {
                let row_bytes = match lg.attn_output_a.kind {
                    super::metal_ctx::GpuWKind::F16 => group_dim * 2,
                    super::metal_ctx::GpuWKind::F32 => group_dim * 4,
                    super::metal_ctx::GpuWKind::Q8 => (group_dim / 32) * 34,
                    _ => panic!("LoRA-O A must be dense"),
                };
                for g in 0..n_groups {
                    self.metal.encode_matvec_kind_off(
                        &enc,
                        lg.attn_output_a.kind,
                        &lg.attn_output_a.buf,
                        &sc.head_out,
                        &sc.low,
                        rank as i32,
                        group_dim as i32,
                        (g * rank * row_bytes) as u64,
                        (g * group_dim * 4) as u64,
                        (g * rank * 4) as u64,
                    );
                }
            }
            self.metal.encode_matvec_kind(
                &enc,
                lg.attn_output_b.kind,
                &lg.attn_output_b.buf,
                &sc.low,
                &sc.attn_in,
                n_embd as i32,
                (n_groups * rank) as i32,
                0,
            );
            if !skip_hc_expand {
                // Expand attn into y (hc_mid). Residual stays in hc_flat.
                self.metal.encode_hc_expand_post_off(
                    &enc,
                    &sc.attn_in,
                    0,
                    &sc.hc_flat,
                    0,
                    &sc.split,
                    post_off,
                    &sc.split,
                    comb_off,
                    &sc.y,
                    0,
                    n_embd as i32,
                    n_hc as i32,
                );
            } else {
                // Ablation: broadcast block into stream0 of y, zero others via expand
                // with identity-ish: just rms on attn_in path via copying into y[0].
                self.metal.encode_zero(&enc, &sc.y, hc_dim as i32);
                self.metal.encode_add(&enc, &sc.attn_in, &sc.y, &sc.y, n_embd as i32);
            }
            // ffn HC pre from hc_mid in y
            self.metal.encode_rms_norm(
                &enc,
                &sc.y,
                None,
                &sc.q_heads,
                hc_dim as i32,
                cfg.rms_eps,
            );
            self.metal.encode_matvec_kind(
                &enc,
                lg.hc_ffn_fn.kind,
                &lg.hc_ffn_fn.buf,
                &sc.q_heads,
                &sc.mix,
                mix_len as i32,
                hc_dim as i32,
                0,
            );
            self.metal.encode_hc_split_sinkhorn(
                &enc,
                &sc.mix,
                &lg.hc_ffn_scale,
                &lg.hc_ffn_base,
                &sc.split,
                n_hc as i32,
                sinkhorn_iters,
                cfg.hc_eps,
            );
            self.metal.encode_hc_weighted_sum(
                &enc,
                &sc.y,
                &sc.split,
                &sc.attn_in,
                n_embd as i32,
                n_hc as i32,
            );
            self.metal.encode_rms_norm(
                &enc,
                &sc.attn_in,
                Some(&lg.ffn_norm),
                &sc.x,
                n_embd as i32,
                cfg.rms_eps,
            );
            // Router + GPU select — wait for route_ids unless MoE fused below.
            self.metal.encode_matvec_kind(
                &enc,
                lg.ffn_gate_inp.kind,
                &lg.ffn_gate_inp.buf,
                &sc.x,
                &sc.logits,
                cfg.n_expert as i32,
                n_embd as i32,
                0,
            );
            // √softplus + top-k / hash select stay on GPU; host only reads ≤k ids for SSD.
            self.metal.encode_router_sqrt_softplus(
                &enc,
                &sc.logits,
                &sc.probs,
                cfg.n_expert as i32,
            );
            if let Some(ref tid2eid) = lg.tid2eid {
                let off = (token * cfg.n_expert_used * 4) as u64;
                self.metal.encode_router_hash_select(
                    &enc,
                    &sc.probs,
                    tid2eid,
                    off,
                    &sc.route_ids,
                    &sc.route_w,
                    cfg.n_expert_used as i32,
                    cfg.expert_weight_scale,
                );
            } else {
                self.metal.encode_router_topk(
                    &enc,
                    &sc.probs,
                    lg.exp_probs_b.as_ref(),
                    &sc.route_ids,
                    &sc.route_w,
                    cfg.n_expert as i32,
                    cfg.n_expert_used as i32,
                    cfg.expert_weight_scale,
                );
            }
            if fused_moe {
                // Same CB: shared + routed + HC expand — no mid-layer wait.
                // Speculative path: snapshot attn-expanded HC (`y`) before MoE
                // so a mismatch can restore and re-apply correct FFN residual.
                if fuse_via_spec {
                    self.metal
                        .encode_copy(&enc, &sc.y, &sc.hc_snap, hc_dim as i32);
                }
                if !skip_shared {
                    encode_shared(&self.metal, lg, sc, &enc);
                } else {
                    self.metal.encode_zero(&enc, &sc.shared, n_embd as i32);
                    self.metal.encode_zero(&enc, &sc.routed, n_embd as i32);
                }
                if !skip_moe_routed {
                    let eg = self.expert_gpu.as_ref().unwrap();
                    if fuse_via_map && use_slots6_fuse {
                        self.metal.encode_map_route_slots(
                            &enc,
                            &sc.route_ids,
                            &sc.slot_map,
                            &sc.route_slots,
                            &sc.moe_miss_flag,
                            &sc.route_ids_hist,
                            cfg.n_expert_used as i32,
                            (il * cfg.n_expert_used) as i32,
                        );
                        self.metal.encode_slots6_iq2_pair_swiglu_packed(
                            &enc,
                            &eg.gate_pack,
                            &eg.up_pack,
                            &sc.route_slots,
                            &sc.x,
                            &sc.routed_mid,
                            &sc.route_w,
                            n_ff as i32,
                            n_embd as i32,
                            cfg.swiglu_clamp_exp,
                            eg.gate_bytes as u32,
                            eg.up_bytes as u32,
                        );
                        self.metal.encode_slots6_q2k_sum6_packed(
                            &enc,
                            &eg.down_pack,
                            &sc.route_slots,
                            &sc.routed_mid,
                            &sc.routed,
                            n_embd as i32,
                            n_ff as i32,
                            eg.down_bytes as u32,
                        );
                    } else if use_slots6_fuse && fuse_slots.len() == 6 {
                        encode_slots6_all(&self.metal, eg, sc, &enc, &fuse_slots);
                    } else {
                        for i in 0..fuse_slots.len().min(n_routed_known) {
                            encode_expert_i(&self.metal, eg, sc, &enc, &fuse_slots, i);
                        }
                    }
                }
                if !skip_hc_expand {
                    encode_ffn_tail(&self.metal, sc, &enc);
                } else {
                    self.metal.encode_add(
                        &enc,
                        &sc.shared,
                        &sc.routed,
                        &sc.attn_in,
                        n_embd as i32,
                    );
                    self.metal.encode_zero(&enc, &sc.hc_flat, hc_dim as i32);
                    self.metal.encode_add(
                        &enc,
                        &sc.attn_in,
                        &sc.hc_flat,
                        &sc.hc_flat,
                        n_embd as i32,
                    );
                }
            }
            enc.end_encoding();
            cmd.commit();
            if fused_moe {
                if fuse_via_spec {
                    // Speculative MoE: wait once, compare route_ids to last_ids.
                    let t0 = Instant::now();
                    self.metal.wait_owned(&cmd, Some(sc));
                    sc.clear_pending();
                    let dt = t0.elapsed().as_nanos() as u64;
                    t_gpu += dt;
                    if profile {
                        sc.prof_cb2_ns.set(sc.prof_cb2_ns.get() + dt);
                    }
                    let ids: Vec<u32> =
                        super::metal_ctx::Dsv4Metal::read_i32(&sc.route_ids, cfg.n_expert_used)
                            .into_iter()
                            .map(|x| x as u32)
                            .collect();
                    let guess = spec_guess_ids.take().unwrap_or_default();
                    if ids == guess {
                        sc.spec_moe_match.set(sc.spec_moe_match.get() + 1);
                        map_fuse_ids = Some(ids);
                        early_fused_return = true;
                    } else {
                        sc.spec_moe_miss.set(sc.spec_moe_miss.get() + 1);
                        let set_hit = ids.iter().filter(|e| guess.contains(e)).count();
                        sc.spec_moe_overlap_sum
                            .set(sc.spec_moe_overlap_sum.get() + set_hit as u64);
                        if set_hit == ids.len() && ids.len() == guess.len() {
                            sc.spec_moe_set_eq.set(sc.spec_moe_set_eq.get() + 1);
                        }
                        if std::env::var("DSV4_SPEC_MOE_DEBUG").ok().as_deref() == Some("1")
                            && il == 5
                        {
                            eprintln!(
                                "SPEC_MOE L{il} overlap={set_hit}/{} guess={guess:?} ids={ids:?}",
                                ids.len()
                            );
                        }
                        map_fuse_ids = Some(ids);
                        map_fuse_need_repair = true;
                        early_fused_return = true;
                    }
                } else if fuse_via_map && fuse_map_safe {
                    // Correct map-fuse: wait, check miss_flag; repair outside this scope.
                    let t0 = Instant::now();
                    self.metal.wait_owned(&cmd, Some(sc));
                    sc.clear_pending();
                    let dt = t0.elapsed().as_nanos() as u64;
                    t_gpu += dt;
                    if profile {
                        sc.prof_cb2_ns.set(sc.prof_cb2_ns.get() + dt);
                    }
                    let miss = super::metal_ctx::Dsv4Metal::read_i32(&sc.moe_miss_flag, 1)
                        .into_iter()
                        .next()
                        .unwrap_or(1);
                    let ids: Vec<u32> =
                        super::metal_ctx::Dsv4Metal::read_i32(&sc.route_ids, cfg.n_expert_used)
                            .into_iter()
                            .map(|x| x as u32)
                            .collect();
                    map_fuse_ids = Some(ids);
                    map_fuse_need_repair = miss != 0;
                    early_fused_return = true;
                } else {
                    // Hash / optimistic map: defer — next layer CB covers this CB.
                    sc.defer_cmd(cmd);
                    if let Some(ref keys) = hash_keys {
                        map_fuse_ids =
                            Some(keys.iter().map(|k| k.expert as u32).collect());
                    }
                    early_fused_return = true;
                }
            } else {
                let t0 = Instant::now();
                self.metal.wait_owned(&cmd, Some(sc));
                // Prior deferred MoE CB completed on the same queue.
                sc.clear_pending();
                let dt = t0.elapsed().as_nanos() as u64;
                t_gpu += dt;
                if profile {
                    sc.prof_cb2_ns.set(sc.prof_cb2_ns.get() + dt);
                }
            }
        }

        if let Some(ids) = map_fuse_ids {
            self.note_expert_ids(il, ids.clone());
            if map_fuse_need_repair {
                let sc = self.scratch.as_ref().unwrap();
                let keys: Vec<ExpertKey> = ids
                    .iter()
                    .map(|&e| ExpertKey {
                        layer: il as u16,
                        expert: e as u16,
                    })
                    .collect();
                let n_routed = keys.len().min(sc.max_routed);
                let mut gpu_slots = vec![0usize; n_routed];
                let (_hits, miss_keys) = {
                    let eg = self.expert_gpu.as_mut().unwrap();
                    eg.classify_hits(&keys[..n_routed], &mut gpu_slots)
                };
                if !miss_keys.is_empty() {
                    let metal = self.metal.clone();
                    let mut eg = self.expert_gpu.take().expect("expert_gpu");
                    let t_e0 = Instant::now();
                    let _ = eg
                        .pin_misses_with_gpu(
                            &metal,
                            &self.ssd,
                            &miss_keys,
                            &mut gpu_slots,
                            &[],
                            || (),
                        )
                        .expect("fuse repair pin");
                    t_exp += t_e0.elapsed().as_nanos() as u64;
                    self.expert_gpu = Some(eg);
                }
                let eg = self.expert_gpu.as_ref().unwrap();
                let lg = &self.gpu_layers.as_ref().unwrap()[il];
                let sc = self.scratch.as_ref().unwrap();
                let cmd_r = self.metal.queue.new_command_buffer().to_owned();
                {
                    let enc = cmd_r.new_compute_command_encoder();
                    // Speculative miss: restore attn-expanded HC before correct MoE.
                    if fuse_via_spec {
                        self.metal
                            .encode_copy(&enc, &sc.hc_snap, &sc.y, hc_dim as i32);
                    }
                    if !skip_shared {
                        encode_shared(&self.metal, lg, sc, &enc);
                    } else {
                        self.metal.encode_zero(&enc, &sc.shared, n_embd as i32);
                        self.metal.encode_zero(&enc, &sc.routed, n_embd as i32);
                    }
                    if !skip_moe_routed {
                        if use_slots6_fuse && n_routed == 6 {
                            encode_slots6_all(&self.metal, eg, sc, &enc, &gpu_slots);
                        } else {
                            for i in 0..n_routed {
                                encode_expert_i(&self.metal, eg, sc, &enc, &gpu_slots, i);
                            }
                        }
                    }
                    if !skip_hc_expand {
                        encode_ffn_tail(&self.metal, sc, &enc);
                    }
                    enc.end_encoding();
                }
                cmd_r.commit();
                sc.defer_cmd(cmd_r);
            }
        }

        if early_fused_return {
            if profile {
                if let Some(sc) = self.scratch.as_ref() {
                    sc.prof_attn_ns.set(sc.prof_attn_ns.get() + t_attn);
                    sc.prof_gpu_ns.set(sc.prof_gpu_ns.get() + t_gpu);
                    sc.prof_copy_ns.set(sc.prof_copy_ns.get() + t_exp);
                    sc.prof_pin_ns
                        .set(sc.prof_pin_ns.get() + t_layer.elapsed().as_nanos() as u64);
                }
            }
            return;
        }

        // After router wait: read ids → classify → pin ASAP (pread ∥ shared+hits).
        let ids: Vec<u32> = {
            let sc = self.scratch.as_ref().unwrap();
            super::metal_ctx::Dsv4Metal::read_i32(&sc.route_ids, cfg.n_expert_used)
                .into_iter()
                .map(|x| x as u32)
                .collect()
        };
        // Speculative pin for the next token (topk layers).
        self.note_expert_ids(il, ids.clone());
        // Weights remain in sc.route_w for encode_axpy_w.

        let slots_or_keys: Vec<ExpertKey> = if let Some(keys) = hash_keys {
            keys
        } else {
            ids.iter()
                .map(|&e| ExpertKey {
                    layer: il as u16,
                    expert: e as u16,
                })
                .collect()
        };

        let n_routed = slots_or_keys
            .len()
            .min(self.scratch.as_ref().unwrap().max_routed);
        let use_slots6 = n_routed == 6
            && std::env::var("DSV4_SLOTS6").ok().as_deref() != Some("0");

        let mut gpu_slots = vec![0usize; n_routed];
        let (hit_idxs, miss_keys) = {
            let eg = self.expert_gpu.as_mut().unwrap();
            eg.classify_hits(&slots_or_keys[..n_routed], &mut gpu_slots)
        };
        // Any mixed set: hit-GPU ∥ miss-pread (ds4's ≥3 gate is for their faster path).
        let split_moe = !hit_idxs.is_empty() && !miss_keys.is_empty();

        if split_moe {
            // I/O-first: miss preads ∥ shared+hit encode/commit; wait once on miss CB.
            let protect: Vec<usize> = hit_idxs.iter().map(|&i| gpu_slots[i]).collect();
            let hit_bufs: Vec<(metal::Buffer, metal::Buffer, metal::Buffer, usize)> = {
                let eg = self.expert_gpu.as_ref().unwrap();
                hit_idxs
                    .iter()
                    .map(|&i| {
                        let si = gpu_slots[i];
                        (
                            eg.gate(si).clone(),
                            eg.up(si).clone(),
                            eg.down(si).clone(),
                            i,
                        )
                    })
                    .collect()
            };
            let metal = self.metal.clone();
            let mut eg = self.expert_gpu.take().expect("expert_gpu");
            let t_e0 = Instant::now();
            let hit_cmd = eg
                .pin_misses_with_gpu(
                    &metal,
                    &self.ssd,
                    &miss_keys,
                    &mut gpu_slots,
                    &protect,
                    || {
                        let lg = &self.gpu_layers.as_ref().unwrap()[il];
                        let sc = self.scratch.as_ref().unwrap();
                        let cmd = self.metal.queue.new_command_buffer().to_owned();
                        let enc = cmd.new_compute_command_encoder();
                        if !skip_shared {
                            encode_shared(&self.metal, lg, sc, &enc);
                        } else {
                            self.metal.encode_zero(&enc, &sc.shared, n_embd as i32);
                            self.metal.encode_zero(&enc, &sc.routed, n_embd as i32);
                        }
                        if !skip_moe_routed {
                            for (gate, up, down, i) in &hit_bufs {
                                encode_expert_bufs(&self.metal, sc, &enc, gate, up, down, *i);
                            }
                        }
                        enc.end_encoding();
                        cmd.commit();
                        cmd
                    },
                )
                .expect("expert pread");
            t_exp += t_e0.elapsed().as_nanos() as u64;
            self.expert_gpu = Some(eg);

            let scratch = self.scratch.as_ref().unwrap();
            let eg = self.expert_gpu.as_ref().unwrap();
            let cmd = self.metal.queue.new_command_buffer().to_owned();
            {
                let enc = cmd.new_compute_command_encoder();
                if !skip_moe_routed {
                    for &(i, _) in &miss_keys {
                        encode_expert_i(&self.metal, eg, scratch, &enc, &gpu_slots, i);
                    }
                }
                if !skip_hc_expand {
                    encode_ffn_tail(&self.metal, scratch, &enc);
                } else {
                    self.metal.encode_add(
                        &enc,
                        &scratch.shared,
                        &scratch.routed,
                        &scratch.attn_in,
                        n_embd as i32,
                    );
                    self.metal
                        .encode_zero(&enc, &scratch.hc_flat, hc_dim as i32);
                    self.metal.encode_add(
                        &enc,
                        &scratch.attn_in,
                        &scratch.hc_flat,
                        &scratch.hc_flat,
                        n_embd as i32,
                    );
                }
                enc.end_encoding();
            }
            cmd.commit();
            let _ = hit_cmd;
            // Fuse MoE wait: next layer CB2 (same queue) covers this CB.
            scratch.defer_cmd(cmd);
        } else if miss_keys.is_empty() {
            // All resident: shared + experts + expand.
            let scratch = self.scratch.as_ref().unwrap();
            let lg = &self.gpu_layers.as_ref().unwrap()[il];
            let eg = self.expert_gpu.as_ref().unwrap();
            let cmd = self.metal.queue.new_command_buffer().to_owned();
            {
                let enc = cmd.new_compute_command_encoder();
                if !skip_shared {
                    encode_shared(&self.metal, lg, scratch, &enc);
                } else {
                    self.metal.encode_zero(&enc, &scratch.shared, n_embd as i32);
                    self.metal.encode_zero(&enc, &scratch.routed, n_embd as i32);
                }
                if !skip_moe_routed {
                    if use_slots6 && n_routed == 6 {
                        encode_slots6_all(&self.metal, eg, scratch, &enc, &gpu_slots);
                    } else {
                        for i in 0..n_routed {
                            encode_expert_i(&self.metal, eg, scratch, &enc, &gpu_slots, i);
                        }
                    }
                }
                if !skip_hc_expand {
                    encode_ffn_tail(&self.metal, scratch, &enc);
                } else {
                    self.metal.encode_add(
                        &enc,
                        &scratch.shared,
                        &scratch.routed,
                        &scratch.attn_in,
                        n_embd as i32,
                    );
                    self.metal
                        .encode_zero(&enc, &scratch.hc_flat, hc_dim as i32);
                    self.metal.encode_add(
                        &enc,
                        &scratch.attn_in,
                        &scratch.hc_flat,
                        &scratch.hc_flat,
                        n_embd as i32,
                    );
                }
                enc.end_encoding();
            }
            cmd.commit();
            let next_keys = if il + 1 < cfg.n_layer {
                self.layers[il + 1].tid2eid.as_ref().map(|tid2eid| {
                    let k = cfg.n_expert_used;
                    let row = &tid2eid[token * k..(token + 1) * k];
                    row.iter()
                        .map(|&e| ExpertKey {
                            layer: (il + 1) as u16,
                            expert: e as u16,
                        })
                        .collect::<Vec<_>>()
                })
            } else {
                None
            };
            let scratch = self.scratch.as_ref().unwrap();
            let t0 = Instant::now();
            if let Some(ref keys) = next_keys {
                let metal = self.metal.clone();
                let protect = gpu_slots.clone();
                let mut eg = self.expert_gpu.take().expect("expert_gpu");
                let t_e0 = Instant::now();
                let _ = eg
                    .pin_batch_with_gpu(&metal, &self.ssd, keys, &protect, || {
                        self.metal.wait_owned(&cmd, Some(scratch));
                        scratch.clear_pending();
                    })
                    .expect("hash readahead");
                t_exp += t_e0.elapsed().as_nanos() as u64;
                self.expert_gpu = Some(eg);
                let dt = t0.elapsed().as_nanos() as u64;
                t_gpu += dt;
                if profile {
                    scratch.prof_cb3_ns.set(scratch.prof_cb3_ns.get() + dt);
                }
            } else {
                scratch.defer_cmd(cmd);
            }
        } else {
            // All missing: I/O-first shared encode ∥ pread, then experts + expand.
            let metal = self.metal.clone();
            let mut eg = self.expert_gpu.take().expect("expert_gpu");
            let t_e0 = Instant::now();
            let shared_cmd = eg
                .pin_misses_with_gpu(
                    &metal,
                    &self.ssd,
                    &miss_keys,
                    &mut gpu_slots,
                    &[],
                    || {
                        let lg = &self.gpu_layers.as_ref().unwrap()[il];
                        let sc = self.scratch.as_ref().unwrap();
                        let cmd = self.metal.queue.new_command_buffer().to_owned();
                        let enc = cmd.new_compute_command_encoder();
                        if !skip_shared {
                            encode_shared(&self.metal, lg, sc, &enc);
                        } else {
                            self.metal.encode_zero(&enc, &sc.shared, n_embd as i32);
                            self.metal.encode_zero(&enc, &sc.routed, n_embd as i32);
                        }
                        enc.end_encoding();
                        cmd.commit();
                        cmd
                    },
                )
                .expect("expert pread");
            t_exp += t_e0.elapsed().as_nanos() as u64;
            self.expert_gpu = Some(eg);

            let scratch = self.scratch.as_ref().unwrap();
            let eg = self.expert_gpu.as_ref().unwrap();
            let cmd = self.metal.queue.new_command_buffer().to_owned();
            {
                let enc = cmd.new_compute_command_encoder();
                if !skip_moe_routed {
                    if use_slots6 && n_routed == 6 {
                        encode_slots6_all(&self.metal, eg, scratch, &enc, &gpu_slots);
                    } else {
                        for i in 0..n_routed {
                            encode_expert_i(&self.metal, eg, scratch, &enc, &gpu_slots, i);
                        }
                    }
                }
                if !skip_hc_expand {
                    encode_ffn_tail(&self.metal, scratch, &enc);
                } else {
                    self.metal.encode_add(
                        &enc,
                        &scratch.shared,
                        &scratch.routed,
                        &scratch.attn_in,
                        n_embd as i32,
                    );
                    self.metal
                        .encode_zero(&enc, &scratch.hc_flat, hc_dim as i32);
                    self.metal.encode_add(
                        &enc,
                        &scratch.attn_in,
                        &scratch.hc_flat,
                        &scratch.hc_flat,
                        n_embd as i32,
                    );
                }
                enc.end_encoding();
            }
            cmd.commit();
            let _ = shared_cmd;
            scratch.defer_cmd(cmd);
        }
        if profile {
            if let Some(sc) = self.scratch.as_ref() {
                sc.prof_attn_ns.set(sc.prof_attn_ns.get() + t_attn);
                sc.prof_gpu_ns.set(sc.prof_gpu_ns.get() + t_gpu);
                sc.prof_copy_ns.set(sc.prof_copy_ns.get() + t_exp);
                sc.prof_pin_ns
                    .set(sc.prof_pin_ns.get() + t_layer.elapsed().as_nanos() as u64);
            }
        }
    }

    pub fn forward_token_logits(&mut self, token: usize) -> Vec<f32> {
        let cfg = self.cfg.clone();
        let n_embd = cfg.n_embd;
        let n_hc = cfg.n_hc;
        let use_fused = self.use_metal
            && self.gpu_layers.is_some()
            && !super::forward_metal::metal_moe_only();

        let emb = self.embed(token);
        hc_from_plain_embedding(&mut self.hc, &emb, n_embd, n_hc);

        if use_fused {
            let sc = self.scratch.as_ref().unwrap();
            super::metal_ctx::Dsv4Metal::write_f32(&sc.hc_flat, &self.hc);

            for il in 0..cfg.n_layer {
                self.layer_forward_metal(il, token);
            }
            let sc = self.scratch.as_ref().unwrap();
            sc.wait_pending(&self.metal);
            self.hc = super::metal_ctx::Dsv4Metal::read_f32(&sc.hc_flat, n_hc * n_embd);
            // Optimistic map-fuse: refresh last_ids from hist for next-token spec pin.
            if std::env::var("DSV4_FUSE_MAP").ok().as_deref() == Some("1") {
                let k = cfg.n_expert_used;
                let hist =
                    super::metal_ctx::Dsv4Metal::read_i32(&sc.route_ids_hist, cfg.n_layer * k);
                let mut pin_keys = Vec::new();
                for il in 0..cfg.n_layer {
                    if self.layers[il].tid2eid.is_some() {
                        continue;
                    }
                    let row = &hist[il * k..(il + 1) * k];
                    if row.iter().all(|&id| id >= 0) {
                        let ids: Vec<u32> = row.iter().map(|&id| id as u32).collect();
                        self.note_expert_ids(il, ids);
                        for &id in row {
                            pin_keys.push(ExpertKey {
                                layer: il as u16,
                                expert: id as u16,
                            });
                        }
                    }
                }
                if !pin_keys.is_empty() {
                    if let Some(eg) = self.expert_gpu.as_mut() {
                        let metal = self.metal.clone();
                        let _ = eg.pin_batch_from_ssd(&metal, &self.ssd, &pin_keys);
                    }
                }
            }
        } else {
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
        self.note_prefill_done();
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
