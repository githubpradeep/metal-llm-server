//! Metal-accelerated decode helpers for DeepSeek-V4-Flash.

use super::metal_ctx::{GpuWKind, GpuWeight, MetalScratch};
use super::model::{Dsv4GpuModel, LayerWeights};
use super::moe::{
    hash_experts_for_token, router_probs_sqrt_softplus, select_topk_experts,
};
use super::ssd::ExpertKey;
use crate::gguf::ggml_type;
use metal::*;

pub struct LayerGpu {
    pub hc_attn_fn: GpuWeight,
    pub hc_ffn_fn: GpuWeight,
    pub attn_q_a: GpuWeight,
    pub attn_q_b: GpuWeight,
    pub attn_kv: GpuWeight,
    pub attn_output_a: GpuWeight,
    pub attn_output_b: GpuWeight,
    pub ffn_gate_inp: GpuWeight,
    pub ffn_gate_shexp: GpuWeight,
    pub ffn_up_shexp: GpuWeight,
    pub ffn_down_shexp: GpuWeight,
    pub attn_norm: Buffer,
    pub ffn_norm: Buffer,
    pub attn_q_a_norm: Option<Buffer>,
    pub attn_kv_a_norm: Option<Buffer>,
    pub attn_sinks: Buffer,
    pub hc_attn_scale: Buffer,
    pub hc_attn_base: Buffer,
    pub hc_ffn_scale: Buffer,
    pub hc_ffn_base: Buffer,
    pub attn_comp_ape: Option<GpuWeight>,
    pub attn_comp_kv: Option<GpuWeight>,
    pub attn_comp_gate: Option<GpuWeight>,
    pub attn_comp_norm: Option<Buffer>,
    pub idx_comp_ape: Option<GpuWeight>,
    pub idx_comp_kv: Option<GpuWeight>,
    pub idx_comp_gate: Option<GpuWeight>,
    pub idx_comp_norm: Option<Buffer>,
    pub exp_probs_b: Option<Buffer>,
    pub tid2eid: Option<Buffer>,
}

pub fn upload_layer_gpu(metal: &super::metal_ctx::Dsv4Metal, lw: &LayerWeights) -> LayerGpu {
    let opt_norm = |v: &[f32]| {
        if v.is_empty() {
            None
        } else {
            Some(metal.buffer_from_f32(v))
        }
    };
    let (attn_comp_ape, attn_comp_kv, attn_comp_gate, attn_comp_norm) =
        if let Some(c) = lw.attn_compressor.as_ref() {
            (
                Some(GpuWeight::from_dense(metal, &c.ape)),
                Some(GpuWeight::from_dense(metal, &c.kv)),
                Some(GpuWeight::from_dense(metal, &c.gate)),
                Some(metal.buffer_from_f32(&c.norm)),
            )
        } else {
            (None, None, None, None)
        };
    let (idx_comp_ape, idx_comp_kv, idx_comp_gate, idx_comp_norm) =
        if let Some(c) = lw.indexer_compressor.as_ref() {
            (
                Some(GpuWeight::from_dense(metal, &c.ape)),
                Some(GpuWeight::from_dense(metal, &c.kv)),
                Some(GpuWeight::from_dense(metal, &c.gate)),
                Some(metal.buffer_from_f32(&c.norm)),
            )
        } else {
            (None, None, None, None)
        };
    LayerGpu {
        hc_attn_fn: GpuWeight::from_dense(metal, &lw.hc_attn_fn),
        hc_ffn_fn: GpuWeight::from_dense(metal, &lw.hc_ffn_fn),
        attn_q_a: GpuWeight::from_dense(metal, &lw.attn_q_a),
        attn_q_b: GpuWeight::from_dense(metal, &lw.attn_q_b),
        attn_kv: GpuWeight::from_dense(metal, &lw.attn_kv),
        attn_output_a: GpuWeight::from_dense(metal, &lw.attn_output_a),
        attn_output_b: GpuWeight::from_dense(metal, &lw.attn_output_b),
        ffn_gate_inp: GpuWeight::from_dense(metal, &lw.ffn_gate_inp),
        ffn_gate_shexp: GpuWeight::from_dense(metal, &lw.ffn_gate_shexp),
        ffn_up_shexp: GpuWeight::from_dense(metal, &lw.ffn_up_shexp),
        ffn_down_shexp: GpuWeight::from_dense(metal, &lw.ffn_down_shexp),
        attn_norm: metal.buffer_from_f32(&lw.attn_norm),
        ffn_norm: metal.buffer_from_f32(&lw.ffn_norm),
        attn_q_a_norm: opt_norm(&lw.attn_q_a_norm),
        attn_kv_a_norm: opt_norm(&lw.attn_kv_a_norm),
        attn_sinks: metal.buffer_from_f32(&lw.attn_sinks),
        hc_attn_scale: metal.buffer_from_f32(&lw.hc_attn_scale),
        hc_attn_base: metal.buffer_from_f32(&lw.hc_attn_base),
        hc_ffn_scale: metal.buffer_from_f32(&lw.hc_ffn_scale),
        hc_ffn_base: metal.buffer_from_f32(&lw.hc_ffn_base),
        attn_comp_ape,
        attn_comp_kv,
        attn_comp_gate,
        attn_comp_norm,
        idx_comp_ape,
        idx_comp_kv,
        idx_comp_gate,
        idx_comp_norm,
        exp_probs_b: lw
            .exp_probs_b
            .as_ref()
            .map(|v| metal.buffer_from_f32(v)),
        tid2eid: lw.tid2eid.as_ref().map(|v| {
            let bytes = unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
            };
            metal.buffer_from_bytes(bytes)
        }),
    }
}

#[derive(Clone, Copy)]
pub enum DenseGpuOp {
    HcAttnFn,
    HcFfnFn,
    AttnQa,
    AttnQb,
    AttnKv,
    AttnOb,
    OutputHcFn,
}

pub fn metal_enabled() -> bool {
    match std::env::var("DSV4_METAL") {
        Ok(v) => v != "0" && v != "false" && v != "off",
        Err(_) => true,
    }
}

/// When set, only MoE + lm_head use Metal (fewer syncs; dense/attn stay CPU).
pub fn metal_moe_only() -> bool {
    matches!(
        std::env::var("DSV4_METAL_MOE_ONLY").ok().as_deref(),
        Some("1") | Some("true") | Some("on")
    )
}

/// Fused GPU RoPE+KV+SWA attn for ratio==0 layers (default on; `DSV4_GPU_SWA=0` disables).
pub fn gpu_swa_enabled() -> bool {
    match std::env::var("DSV4_GPU_SWA") {
        Ok(v) => v != "0" && v != "false" && v != "off",
        Err(_) => true,
    }
}

/// Fused GPU CSA/HCA attn for ratio!=0 layers (default on; `DSV4_GPU_FULL=0` disables).
pub fn gpu_full_enabled() -> bool {
    match std::env::var("DSV4_GPU_FULL") {
        Ok(v) => v != "0" && v != "false" && v != "off",
        Err(_) => true,
    }
}

impl Dsv4GpuModel {
    pub fn metal_enabled() -> bool {
        metal_enabled()
    }

    pub fn ensure_gpu_weights(&mut self) {
        if self.gpu_layers.is_some() {
            return;
        }
        let metal = self.metal.clone();
        let layers: Vec<_> = self
            .layers
            .iter()
            .map(|lw| upload_layer_gpu(&metal, lw))
            .collect();
        let gpu_kv: Vec<_> = (0..self.cfg.n_layer)
            .map(|il| {
                let lcfg = super::kv::LayerKvConfig::from_cfg(&self.cfg, il);
                super::gpu_kv::GpuLayerKv::new(
                    &metal,
                    &lcfg,
                    self.cfg.n_indexer_head_dim,
                    2048,
                )
            })
            .collect();
        let gate_bytes = self.gate_row_bytes * self.cfg.n_ff_exp;
        let up_bytes = self.up_row_bytes * self.cfg.n_ff_exp;
        let down_bytes = self.down_row_bytes * self.cfg.n_embd;
        let scratch = MetalScratch::new(
            &metal,
            self.cfg.n_embd,
            self.cfg.n_ff_exp,
            self.cfg.n_expert,
            self.cfg.n_vocab,
            gate_bytes,
            up_bytes,
            down_bytes,
        );
        self.gpu_output = Some(GpuWeight::from_dense(&metal, &self.output_weight));
        self.gpu_output_hc_fn = Some(GpuWeight::from_dense(&metal, &self.output_hc_fn));
        self.gpu_output_norm = Some(metal.buffer_from_f32(&self.output_norm));
        self.gpu_layers = Some(layers);
        self.gpu_kv = Some(gpu_kv);
        self.scratch = Some(scratch);
        let expert_bytes = gate_bytes + up_bytes + down_bytes;
        let gpu_slots = match std::env::var("DSV4_GPU_EXPERT_SLOTS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            Some(n) => n,
            None => {
                // Dense weights are no-copy Shared views — reclaim the old ~8 GiB
                // duplicate into the expert LRU (60% of recommended working set).
                let ws = metal.device.recommended_max_working_set_size();
                let auto = if ws > 0 && expert_bytes > 0 {
                    let budget = ((ws as f64) * 0.60) / (expert_bytes as f64);
                    (budget as usize).clamp(1000, 3500)
                } else {
                    1000
                };
                let used_gib =
                    (auto as f64) * (expert_bytes as f64) / (1024.0 * 1024.0 * 1024.0);
                println!(
                    "  DSV4_GPU_EXPERT_SLOTS auto={auto} (ws={:.1} GiB, expert={:.2} MiB, {:.2} GiB used, 60% budget)",
                    ws as f64 / (1024.0 * 1024.0 * 1024.0),
                    expert_bytes as f64 / (1024.0 * 1024.0),
                    used_gib
                );
                auto
            }
        };
        self.expert_gpu = Some(super::expert_gpu::ExpertGpuCache::new(
            &metal,
            gpu_slots,
            gate_bytes,
            up_bytes,
            down_bytes,
            self.cfg.n_expert,
        ));
        println!("  dense Metal: no-copy shared views");
        println!(
            "  dsv4 Metal weight residency: {} layers + lm_head",
            self.cfg.n_layer
        );
    }

    pub fn matvec_dense_gpu(&self, il: usize, which: DenseGpuOp, x: &[f32], out: &mut [f32]) {
        let lg = &self.gpu_layers.as_ref().unwrap()[il];
        let g = match which {
            DenseGpuOp::HcAttnFn => &lg.hc_attn_fn,
            DenseGpuOp::HcFfnFn => &lg.hc_ffn_fn,
            DenseGpuOp::AttnQa => &lg.attn_q_a,
            DenseGpuOp::AttnQb => &lg.attn_q_b,
            DenseGpuOp::AttnKv => &lg.attn_kv,
            DenseGpuOp::AttnOb => &lg.attn_output_b,
            DenseGpuOp::OutputHcFn => self.gpu_output_hc_fn.as_ref().unwrap(),
        };
        if let Some(sc) = self.scratch.as_ref() {
            self.metal.matvec_sync_scratch(g, x, out, &sc.x, &sc.y);
            sc.note_wait();
        } else {
            self.metal.matvec_sync(g, x, out);
        }
    }

    pub fn matvec_attn_oa_rows(
        &self,
        il: usize,
        x: &[f32],
        row_start: usize,
        n_out: usize,
        out: &mut [f32],
    ) {
        let lg = &self.gpu_layers.as_ref().unwrap()[il];
        self.metal
            .matvec_rows_sync(&lg.attn_output_a, x, row_start, n_out, out);
        if let Some(sc) = self.scratch.as_ref() {
            sc.note_wait();
        }
    }

    pub fn matvec_output_metal(&self, x: &[f32], out: &mut [f32]) {
        let g = self.gpu_output.as_ref().expect("gpu output");
        if let Some(sc) = self.scratch.as_ref() {
            self.metal.matvec_sync_scratch(g, x, out, &sc.x, &sc.y);
            sc.note_wait();
        } else {
            self.metal.matvec_sync(g, x, out);
        }
    }

    /// FFN with Metal matvecs; SSD pin overlapped with router∥shared GPU work.
    pub fn layer_ffn_metal(&mut self, il: usize, token: usize, ffn_in: &[f32], ffn_out: &mut [f32]) {
        let cfg = self.cfg.clone();
        let n_embd = cfg.n_embd;
        let n_ff = cfg.n_ff_exp;

        let mut x = vec![0.0f32; n_embd];
        super::hc::rms_norm(&mut x, ffn_in, &self.layers[il].ffn_norm, cfg.rms_eps);

        let (
            gate_inp_kind,
            gate_inp_buf,
            gate_sh_kind,
            gate_sh_buf,
            up_sh_kind,
            up_sh_buf,
            down_sh_kind,
            down_sh_buf,
        ) = {
            let lg = &self.gpu_layers.as_ref().expect("gpu layers")[il];
            (
                lg.ffn_gate_inp.kind,
                lg.ffn_gate_inp.buf.clone(),
                lg.ffn_gate_shexp.kind,
                lg.ffn_gate_shexp.buf.clone(),
                lg.ffn_up_shexp.kind,
                lg.ffn_up_shexp.buf.clone(),
                lg.ffn_down_shexp.kind,
                lg.ffn_down_shexp.buf.clone(),
            )
        };

        let scratch = self.scratch.as_ref().expect("scratch");
        super::metal_ctx::Dsv4Metal::write_f32(&scratch.x, &x);

        // One CB: router + shared expert (was 2 waits).
        let cmd = self.metal.queue.new_command_buffer();
        {
            let enc = cmd.new_compute_command_encoder();
            self.metal.encode_matvec_kind(
                &enc,
                gate_inp_kind,
                &gate_inp_buf,
                &scratch.x,
                &scratch.logits,
                cfg.n_expert as i32,
                n_embd as i32,
                0,
            );
            self.metal.encode_matvec_kind(
                &enc,
                gate_sh_kind,
                &gate_sh_buf,
                &scratch.x,
                &scratch.gate,
                n_ff as i32,
                n_embd as i32,
                0,
            );
            self.metal.encode_matvec_kind(
                &enc,
                up_sh_kind,
                &up_sh_buf,
                &scratch.x,
                &scratch.up,
                n_ff as i32,
                n_embd as i32,
                0,
            );
            self.metal.encode_swiglu(
                &enc,
                &scratch.gate,
                &scratch.up,
                &scratch.mid,
                n_ff as i32,
                cfg.swiglu_clamp_exp,
            );
            self.metal.encode_matvec_kind(
                &enc,
                down_sh_kind,
                &down_sh_buf,
                &scratch.mid,
                &scratch.shared,
                n_embd as i32,
                n_ff as i32,
                0,
            );
            self.metal
                .encode_zero(&enc, &scratch.routed, n_embd as i32);
            enc.end_encoding();
        }
        cmd.commit();
        self.metal.wait_cmd(&cmd, Some(scratch));

        let logits = super::metal_ctx::Dsv4Metal::read_f32(&scratch.logits, cfg.n_expert);
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

        let keys: Vec<ExpertKey> = ids
            .iter()
            .map(|&e| ExpertKey {
                layer: il as u16,
                expert: e as u16,
            })
            .collect();

        // Pin experts (I/O). Shared GPU work already finished above; pin is on critical path
        // but avoids holding a CB open across pread.
        let slots = self.ssd.pin(&keys).expect("SSD pin");

        assert_eq!(
            self.expert_gate_type,
            ggml_type::IQ2_XXS,
            "unsupported expert gate"
        );
        let down_kind = if self.expert_down_type == ggml_type::Q2_K {
            GpuWKind::Q2K
        } else {
            panic!("unsupported expert down");
        };

        let n_routed = slots.len().min(self.scratch.as_ref().unwrap().max_routed);
        for (i, &si) in slots.iter().take(n_routed).enumerate() {
            let scratch = self.scratch.as_ref().unwrap();
            super::metal_ctx::Dsv4Metal::write_bytes(&scratch.exp_gate[i], self.ssd.gate(si));
            super::metal_ctx::Dsv4Metal::write_bytes(&scratch.exp_up[i], self.ssd.up(si));
            super::metal_ctx::Dsv4Metal::write_bytes(&scratch.exp_down[i], self.ssd.down(si));
        }
        {
            let scratch = self.scratch.as_ref().unwrap();
            let cmd = self.metal.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            for i in 0..n_routed {
                self.metal.encode_iq2_pair_swiglu(
                    &enc,
                    &scratch.exp_gate[i],
                    &scratch.exp_up[i],
                    &scratch.x,
                    &scratch.mid,
                    n_ff as i32,
                    n_embd as i32,
                    cfg.swiglu_clamp_exp,
                );
                self.metal.encode_matvec_kind(
                    &enc,
                    down_kind,
                    &scratch.exp_down[i],
                    &scratch.mid,
                    &scratch.exp_down_out[i],
                    n_embd as i32,
                    n_ff as i32,
                    0,
                );
                self.metal.encode_axpy(
                    &enc,
                    &scratch.exp_down_out[i],
                    &scratch.routed,
                    weights[i],
                    n_embd as i32,
                );
            }
            enc.end_encoding();
            cmd.commit();
            self.metal.wait_cmd(&cmd, Some(scratch));
        }

        let scratch = self.scratch.as_ref().unwrap();
        let shared = super::metal_ctx::Dsv4Metal::read_f32(&scratch.shared, n_embd);
        let routed = super::metal_ctx::Dsv4Metal::read_f32(&scratch.routed, n_embd);
        for i in 0..n_embd {
            ffn_out[i] = shared[i] + routed[i];
        }
    }
}
