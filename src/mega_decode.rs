//! Single-dispatch decode mega-kernel: builds a GPU op graph once, replays it
//! each token via `decode_mega_gemma4_q4_0`.

use metal::*;
use std::collections::HashMap;

use crate::gemma4_config::{KvCacheType};
use crate::gemma4_gpu_model::{Gemma4GpuModel, WeightFormat};
use crate::gpu::{weight_buf_is_q3, weight_buf_is_q4, BufferView, MetalContext};

pub const MEGA_GRID_TGS: u32 = 512;

pub mod buf {
    pub const HIDDEN: u32 = 0;
    pub const NORMED: u32 = 1;
    pub const Q: u32 = 2;
    pub const K: u32 = 3;
    pub const V: u32 = 4;
    pub const QN: u32 = 5;
    pub const KN: u32 = 6;
    pub const VN: u32 = 7;
    pub const ATTN: u32 = 8;
    pub const O: u32 = 9;
    pub const GATE: u32 = 10;
    pub const UP: u32 = 11;
    pub const GELU: u32 = 12;
    pub const DOWN: u32 = 13;
    pub const PLE_CTX: u32 = 14;
    pub const PLE_TMP: u32 = 15;
    pub const PLE_TOK: u32 = 16;
    pub const LOGITS: u32 = 17;
}

mod op {
    pub const RMS_NORM: u32 = 0;
    pub const RMS_NORM_PER_HEAD: u32 = 1;
    pub const RMS_NORM_PER_HEAD_NOWEIGHT: u32 = 2;
    pub const RMS_NORM_ADD_SAVE: u32 = 3;
    pub const MATVEC_Q4: u32 = 4;
    pub const MATVEC_F16: u32 = 5;
    pub const ROTARY_Q: u32 = 6;
    pub const ROTARY_K: u32 = 7;
    pub const KV_APPEND_Q4: u32 = 8;
    pub const ATTENTION_Q4: u32 = 9;
    pub const VEC_ADD: u32 = 10;
    pub const VEC_SCALE: u32 = 11;
    pub const GELU_MUL: u32 = 12;
    pub const GELU_MUL_AT: u32 = 13;
    pub const VEC_ADD_SCALED: u32 = 14;
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MegaOpDesc {
    pub op_type: u32,
    pub arg0: u32,
    pub arg1: u32,
    pub arg2: u32,
    pub arg3: u32,
    pub num_tgs: u32,
    pub in_buf: u32,
    pub out_buf: u32,
    pub aux_buf: u32,
    pub weight_buf_idx: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MegaParams {
    pub num_ops: u32,
    pub hidden_size: u32,
    pub q_out: u32,
    pub kv_out: u32,
    pub intermediate_size: u32,
    pub vocab_size: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub num_kv_groups: u32,
    pub head_dim: u32,
    pub kv_capacity: u32,
    pub kv_seq: u32,
    pub kv_cache_type: u32,
    pub groups_per_row: u32,
    pub row_bytes: u32,
    pub eps: f32,
    pub attn_scale: f32,
    pub ple_input_scale: f32,
    pub context_proj_scale: f32,
    pub final_logit_cap: f32,
}

struct WeightTables {
    q4: Buffer,
    f16: Buffer,
    f32: Buffer,
    q4_views: Vec<BufferView>,
    f16_views: Vec<BufferView>,
    f32_views: Vec<BufferView>,
    q4_map: HashMap<(u64, u64), u32>,
    f16_map: HashMap<(u64, u64), u32>,
    f32_map: HashMap<(u64, u64), u32>,
}

impl WeightTables {
    fn new(device: &Device) -> Self {
        Self {
            q4: device.new_buffer(4, MTLResourceOptions::StorageModeShared),
            f16: device.new_buffer(4, MTLResourceOptions::StorageModeShared),
            f32: device.new_buffer(4, MTLResourceOptions::StorageModeShared),
            q4_views: Vec::new(),
            f16_views: Vec::new(),
            f32_views: Vec::new(),
            q4_map: HashMap::new(),
            f16_map: HashMap::new(),
            f32_map: HashMap::new(),
        }
    }

    fn key(view: &BufferView) -> (u64, u64) {
        (view.buffer.gpu_address() + view.offset, view.length)
    }

    fn index_q4(&mut self, device: &Device, view: &BufferView) -> u32 {
        let key = Self::key(view);
        if let Some(&idx) = self.q4_map.get(&key) {
            return idx;
        }
        let idx = self.q4_map.len() as u32;
        self.q4_map.insert(key, idx);
        self.q4_views.push(view.clone());
        self.rebuild_q4(device);
        idx
    }

    fn index_f16(&mut self, device: &Device, view: &BufferView) -> u32 {
        let key = Self::key(view);
        if let Some(&idx) = self.f16_map.get(&key) {
            return idx;
        }
        let idx = self.f16_map.len() as u32;
        self.f16_map.insert(key, idx);
        self.f16_views.push(view.clone());
        self.rebuild_f16(device);
        idx
    }

    fn index_f32(&mut self, device: &Device, view: &BufferView) -> u32 {
        let key = Self::key(view);
        if let Some(&idx) = self.f32_map.get(&key) {
            return idx;
        }
        let idx = self.f32_map.len() as u32;
        self.f32_map.insert(key, idx);
        self.f32_views.push(view.clone());
        self.rebuild_f32(device);
        idx
    }

    fn rebuild_q4(&mut self, device: &Device) {
        let mut addrs: Vec<u64> = vec![0; self.q4_map.len()];
        for (&key, &idx) in &self.q4_map {
            addrs[idx as usize] = key.0;
        }
        self.q4 = buffer_u64(device, &addrs);
    }

    fn rebuild_f16(&mut self, device: &Device) {
        let mut addrs: Vec<u64> = vec![0; self.f16_map.len()];
        for (&key, &idx) in &self.f16_map {
            addrs[idx as usize] = key.0;
        }
        self.f16 = buffer_u64(device, &addrs);
    }

    fn rebuild_f32(&mut self, device: &Device) {
        let mut addrs: Vec<u64> = vec![0; self.f32_map.len()];
        for (&key, &idx) in &self.f32_map {
            addrs[idx as usize] = key.0;
        }
        self.f32 = buffer_u64(device, &addrs);
    }
}

fn buffer_u64(device: &Device, data: &[u64]) -> Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (data.len() * std::mem::size_of::<u64>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

pub struct MegaDecodeGraph {
    pub ops: Vec<MegaOpDesc>,
    attn_ops: Vec<(usize, usize)>, // (op_index, layer_index)
    weights: WeightTables,
    k_cache_table: Buffer,
    v_cache_table: Buffer,
    cos_table: Buffer,
    sin_table: Buffer,
    pub ops_buf: Buffer,
    params_template: MegaParams,
}

struct GraphBuilder<'a> {
    ops: Vec<MegaOpDesc>,
    attn_ops: Vec<(usize, usize)>,
    weights: &'a mut WeightTables,
    device: &'a Device,
}

impl<'a> GraphBuilder<'a> {
    fn push(&mut self, mut op: MegaOpDesc) -> usize {
        if op.num_tgs == 0 {
            op.num_tgs = MEGA_GRID_TGS;
        }
        let idx = self.ops.len();
        self.ops.push(op);
        idx
    }

    fn matvec(
        &mut self,
        weight: &BufferView,
        m: u32,
        k: u32,
        in_buf: u32,
        out_buf: u32,
        weight_format: WeightFormat,
    ) {
        let is_q4 = weight_format == WeightFormat::Q4_0 && weight_buf_is_q4(weight, m, k);
        let is_q3 = weight_format == WeightFormat::Q3_0;
        if is_q4 || is_q3 {
            let w = if is_q4 {
                self.weights.index_q4(self.device, weight)
            } else {
                self.weights.index_q4(self.device, weight) // reuse Q4 table for Q3 views
            };
            self.push(MegaOpDesc {
                op_type: op::MATVEC_Q4,
                arg0: m,
                arg1: k,
                num_tgs: MEGA_GRID_TGS,
                in_buf,
                out_buf,
                weight_buf_idx: w,
                ..MegaOpDesc::default()
            });
        } else {
            let w = self.weights.index_f16(self.device, weight);
            self.push(MegaOpDesc {
                op_type: op::MATVEC_F16,
                arg0: m,
                arg1: k,
                num_tgs: MEGA_GRID_TGS,
                in_buf,
                out_buf,
                weight_buf_idx: w,
                ..MegaOpDesc::default()
            });
        }
    }

    fn rmsnorm(&mut self, weight: &BufferView, dim: u32, in_buf: u32, out_buf: u32) {
        let w = self.weights.index_f32(self.device, weight);
        self.push(MegaOpDesc {
            op_type: op::RMS_NORM,
            arg0: dim,
            num_tgs: MEGA_GRID_TGS,
            in_buf,
            out_buf,
            weight_buf_idx: w,
            ..Default::default()
        });
    }

    fn rmsnorm_per_head(
        &mut self,
        weight: &BufferView,
        num_heads: u32,
        head_dim: u32,
        in_buf: u32,
        out_buf: u32,
    ) {
        let w = self.weights.index_f32(self.device, weight);
        self.push(MegaOpDesc {
            op_type: op::RMS_NORM_PER_HEAD,
            arg0: num_heads,
            arg1: head_dim,
            num_tgs: MEGA_GRID_TGS,
            in_buf,
            out_buf,
            weight_buf_idx: w,
            ..Default::default()
        });
    }

    fn rmsnorm_per_head_noweight(
        &mut self,
        num_heads: u32,
        head_dim: u32,
        in_buf: u32,
        out_buf: u32,
    ) {
        self.push(MegaOpDesc {
            op_type: op::RMS_NORM_PER_HEAD_NOWEIGHT,
            arg0: num_heads,
            arg1: head_dim,
            num_tgs: MEGA_GRID_TGS,
            in_buf,
            out_buf,
            ..Default::default()
        });
    }

    fn vec_add(&mut self, n: u32, a: u32, b: u32, out: u32) {
        self.push(MegaOpDesc {
            op_type: op::VEC_ADD,
            arg0: n,
            num_tgs: MEGA_GRID_TGS,
            in_buf: a,
            aux_buf: b,
            out_buf: out,
            ..Default::default()
        });
    }

    fn vec_scale(&mut self, n: u32, scale: f32, in_buf: u32, out_buf: u32) {
        self.push(MegaOpDesc {
            op_type: op::VEC_SCALE,
            arg0: n,
            arg1: scale.to_bits(),
            num_tgs: MEGA_GRID_TGS,
            in_buf,
            out_buf,
            ..Default::default()
        });
    }

    fn gelu_mul(&mut self, n: u32, gate: u32, up: u32, out: u32) {
        self.push(MegaOpDesc {
            op_type: op::GELU_MUL,
            arg0: n,
            num_tgs: MEGA_GRID_TGS,
            in_buf: gate,
            aux_buf: up,
            out_buf: out,
            ..Default::default()
        });
    }

    fn gelu_mul_at(&mut self, ple_dim: u32, layer: u32, gate: u32, ctx: u32, out: u32) {
        self.push(MegaOpDesc {
            op_type: op::GELU_MUL_AT,
            arg0: ple_dim,
            arg1: ple_dim,
            arg2: layer,
            num_tgs: MEGA_GRID_TGS,
            in_buf: gate,
            aux_buf: ctx,
            out_buf: out,
            ..Default::default()
        });
    }

    fn vec_add_scaled(&mut self, n: u32, scale: f32, hidden: u32, ple: u32, out: u32) {
        self.push(MegaOpDesc {
            op_type: op::VEC_ADD_SCALED,
            arg0: n,
            arg1: scale.to_bits(),
            num_tgs: MEGA_GRID_TGS,
            in_buf: hidden,
            aux_buf: ple,
            out_buf: out,
            ..Default::default()
        });
    }

    fn rotary_q(&mut self, layer: u32, q: u32) {
        self.push(MegaOpDesc {
            op_type: op::ROTARY_Q,
            arg0: layer,
            num_tgs: MEGA_GRID_TGS,
            in_buf: q,
            out_buf: q,
            ..Default::default()
        });
    }

    fn rotary_k(&mut self, layer: u32, k: u32) {
        self.push(MegaOpDesc {
            op_type: op::ROTARY_K,
            arg0: layer,
            num_tgs: MEGA_GRID_TGS,
            in_buf: k,
            out_buf: k,
            ..Default::default()
        });
    }

    fn kv_append_k(&mut self, layer: u32, kn: u32) {
        self.push(MegaOpDesc {
            op_type: op::KV_APPEND_Q4,
            arg0: layer,
            arg1: 0,
            num_tgs: MEGA_GRID_TGS,
            in_buf: kn,
            ..Default::default()
        });
    }

    fn kv_append_v(&mut self, layer: u32, vn: u32) {
        self.push(MegaOpDesc {
            op_type: op::KV_APPEND_Q4,
            arg0: layer,
            arg1: 1,
            num_tgs: MEGA_GRID_TGS,
            in_buf: vn,
            ..Default::default()
        });
    }

    fn attention_q4(&mut self, layer_idx: usize, kv_layer: u32) -> usize {
        let idx = self.push(MegaOpDesc {
            op_type: op::ATTENTION_Q4,
            arg0: kv_layer,
            arg1: 0,
            arg2: 0,
            arg3: layer_idx as u32,
            num_tgs: MEGA_GRID_TGS,
            in_buf: buf::QN,
            out_buf: buf::ATTN,
            ..Default::default()
        });
        self.attn_ops.push((idx, layer_idx));
        idx
    }
}

pub fn mega_kernel_enabled() -> bool {
    static WARN: std::sync::Once = std::sync::Once::new();
    if matches!(
        std::env::var("MEGA_KERNEL").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    ) {
        WARN.call_once(|| {
            eprintln!("  MEGA_KERNEL=1: disabled — experimental op graph produced garbage output.");
            eprintln!("  The default path already fuses decode into one command buffer (~24 tok/s).");
            eprintln!("  For throughput tuning try MATVEC_KERNEL=r8 (see gpu.rs).");
        });
    }
    false
}

impl MegaDecodeGraph {
    pub fn build(model: &Gemma4GpuModel) -> Result<Self, String> {
        if model.kv_cache_type != KvCacheType::Q4_0 {
            return Err("MEGA_KERNEL requires LLAMA_KV_CACHE_TYPE=q4_0".into());
        }

        let cfg = &model.config;
        let device = &model.ctx.device;
        let hidden = cfg.hidden_size as u32;
        let inter = cfg.intermediate_size as u32;
        let num_heads = cfg.num_attention_heads as u32;
        let num_kv = cfg.num_key_value_heads as u32;
        let head_dim = cfg.head_dim as u32;
        let q_out = num_heads * head_dim;
        let kv_out = num_kv * head_dim;
        let num_layers = cfg.num_hidden_layers as u32;
        let ple_dim = cfg.hidden_size_per_layer_input as u32;
        let ple_total = num_layers * ple_dim;

        let mut weights = WeightTables::new(device);
        let mut builder = GraphBuilder {
            ops: Vec::new(),
            attn_ops: Vec::new(),
            weights: &mut weights,
            device,
        };

        let ple_input_scale = std::f32::consts::FRAC_1_SQRT_2;
        let context_proj_scale = 1.0 / (hidden as f32).sqrt();

        // PLE pre-pass
        builder.matvec(
            &model.per_layer_model_projection_weight,
            ple_total,
            hidden,
            buf::HIDDEN,
            buf::PLE_CTX,
            WeightFormat::Q4_0,
        );
        builder.vec_scale(ple_total, context_proj_scale, buf::PLE_CTX, buf::PLE_TMP);
        builder.rmsnorm_per_head(
            &model.per_layer_projection_norm_weight,
            num_layers,
            ple_dim,
            buf::PLE_TMP,
            buf::PLE_CTX,
        );
        builder.vec_add(ple_total, buf::PLE_CTX, buf::PLE_TOK, buf::PLE_TMP);
        builder.vec_scale(ple_total, ple_input_scale, buf::PLE_TMP, buf::PLE_CTX);

        for (layer_idx, layer) in model.layers.iter().enumerate() {
            let li = layer_idx as u32;
            let q_out_l = layer.q_out_dim as u32;
            let kv_out_l = layer.kv_out_dim as u32;
            let head_dim_l = layer.head_dim as u32;
            let wf = layer.weight_format;

            builder.rmsnorm(
                &layer.input_layernorm_weight,
                hidden,
                buf::HIDDEN,
                buf::NORMED,
            );
            builder.matvec(&layer.q_proj, q_out_l, hidden, buf::NORMED, buf::Q, wf);
            builder.rmsnorm_per_head(
                &layer.q_norm_weight,
                num_heads,
                head_dim_l,
                buf::Q,
                buf::QN,
            );
            builder.rotary_q(li, buf::QN);

            if layer.has_kv {
                builder.matvec(&layer.k_proj, kv_out_l, hidden, buf::NORMED, buf::K, wf);
                builder.matvec(&layer.v_proj, kv_out_l, hidden, buf::NORMED, buf::V, wf);
                builder.rmsnorm_per_head(
                    &layer.k_norm_weight,
                    num_kv,
                    head_dim_l,
                    buf::K,
                    buf::KN,
                );
                builder.rotary_k(li, buf::KN);
                builder.rmsnorm_per_head_noweight(num_kv, head_dim_l, buf::V, buf::VN);
                builder.kv_append_k(li, buf::KN);
                builder.kv_append_v(li, buf::VN);
            }

            builder.attention_q4(layer_idx, layer.kv_source_layer as u32);
            builder.matvec(&layer.o_proj, hidden, q_out_l, buf::ATTN, buf::O, wf);
            builder.rmsnorm(
                &layer.post_attention_layernorm_weight,
                hidden,
                buf::O,
                buf::NORMED,
            );
            builder.vec_add(hidden, buf::HIDDEN, buf::NORMED, buf::HIDDEN);

            builder.rmsnorm(
                &layer.pre_feedforward_layernorm_weight,
                hidden,
                buf::HIDDEN,
                buf::NORMED,
            );
            builder.matvec(&layer.gate_proj, inter, hidden, buf::NORMED, buf::GATE, wf);
            builder.matvec(&layer.up_proj, inter, hidden, buf::NORMED, buf::UP, wf);
            builder.gelu_mul(inter, buf::GATE, buf::UP, buf::GELU);
            builder.matvec(&layer.down_proj, hidden, inter, buf::GELU, buf::DOWN, wf);
            builder.rmsnorm(
                &layer.post_feedforward_layernorm_weight,
                hidden,
                buf::DOWN,
                buf::NORMED,
            );
            builder.vec_add(hidden, buf::HIDDEN, buf::NORMED, buf::HIDDEN);

            // PLE weights follow buffer layout (auto), not layer.weight_format.
            builder.matvec(
                &layer.per_layer_input_gate_weight,
                ple_dim,
                hidden,
                buf::HIDDEN,
                buf::K,
                WeightFormat::Q4_0,
            );
            builder.gelu_mul_at(ple_dim, li, buf::K, buf::PLE_CTX, buf::DOWN);
            builder.matvec(
                &layer.per_layer_projection_weight,
                hidden,
                ple_dim,
                buf::DOWN,
                buf::O,
                WeightFormat::Q4_0,
            );
            builder.rmsnorm(
                &layer.post_per_layer_input_norm_weight,
                hidden,
                buf::O,
                buf::V,
            );
            builder.vec_add(hidden, buf::HIDDEN, buf::V, buf::HIDDEN);
            builder.vec_scale(hidden, layer.layer_scalar, buf::HIDDEN, buf::HIDDEN);
        }

        builder.rmsnorm(
            &model.final_norm_weight,
            hidden,
            buf::HIDDEN,
            buf::NORMED,
        );
        builder.matvec(
            &model.lm_head_buf,
            cfg.vocab_size as u32,
            hidden,
            buf::NORMED,
            buf::LOGITS,
            WeightFormat::Q4_0,
        );

        let k_addrs: Vec<u64> = model.k_cache.iter().map(|b| b.gpu_address()).collect();
        let v_addrs: Vec<u64> = model.v_cache.iter().map(|b| b.gpu_address()).collect();
        let cos_base = model.decode_rope_cos_packed.gpu_address();
        let sin_base = model.decode_rope_sin_packed.gpu_address();
        let rope_stride = (model.rope_max_head_dim * std::mem::size_of::<f32>()) as u64;
        let cos_addrs: Vec<u64> = (0..model.layers.len())
            .map(|i| cos_base + i as u64 * rope_stride)
            .collect();
        let sin_addrs: Vec<u64> = (0..model.layers.len())
            .map(|i| sin_base + i as u64 * rope_stride)
            .collect();

        let ops_buf = device.new_buffer_with_data(
            builder.ops.as_ptr() as *const _,
            (builder.ops.len() * std::mem::size_of::<MegaOpDesc>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let params_template = MegaParams {
            num_ops: builder.ops.len() as u32,
            hidden_size: hidden,
            q_out,
            kv_out,
            intermediate_size: inter,
            vocab_size: cfg.vocab_size as u32,
            num_heads,
            num_kv_heads: num_kv,
            num_kv_groups: (num_heads / num_kv.max(1)) as u32,
            head_dim,
            kv_capacity: model.kv_capacity,
            kv_seq: 0,
            kv_cache_type: 2,
            groups_per_row: 0,
            row_bytes: 0,
            eps: cfg.rms_norm_eps as f32,
            attn_scale: 1.0,
            ple_input_scale,
            context_proj_scale,
            final_logit_cap: cfg.final_logit_softcapping,
        };

        Ok(Self {
            ops: builder.ops,
            attn_ops: builder.attn_ops,
            weights,
            k_cache_table: buffer_u64(device, &k_addrs),
            v_cache_table: buffer_u64(device, &v_addrs),
            cos_table: buffer_u64(device, &cos_addrs),
            sin_table: buffer_u64(device, &sin_addrs),
            ops_buf,
            params_template,
        })
    }

    fn patch_attention_ops(
        &mut self,
        layers: &[crate::gemma4_gpu_model::Gemma4GpuLayer],
        sliding_window: u32,
        kv_seq: u32,
    ) {
        for &(op_idx, layer_idx) in &self.attn_ops {
            let layer = &layers[layer_idx];
            let attn_kv_seq = kv_seq + 1;
            let effective = if layer.is_full_attention {
                attn_kv_seq
            } else {
                attn_kv_seq.min(sliding_window)
            };
            let kv_start = if !layer.is_full_attention && attn_kv_seq > sliding_window {
                attn_kv_seq - sliding_window
            } else {
                0
            };
            self.ops[op_idx].arg1 = effective;
            self.ops[op_idx].arg2 = kv_start;
        }
        unsafe {
            let ptr = self.ops_buf.contents() as *mut MegaOpDesc;
            std::ptr::copy_nonoverlapping(self.ops.as_ptr(), ptr, self.ops.len());
        }
    }

    pub fn encode(
        &mut self,
        ctx: &MetalContext,
        encoder: &metal::ComputeCommandEncoderRef,
        layers: &[crate::gemma4_gpu_model::Gemma4GpuLayer],
        sliding_window: u32,
        buffers: &MegaScratchBuffers<'_>,
        k_cache: &[Buffer],
        v_cache: &[Buffer],
        cos_packed: &Buffer,
        sin_packed: &Buffer,
        rope_max_head_dim: usize,
        kv_seq: u32,
        sample: Option<(f32, f32, u32)>,
    ) {
        self.patch_attention_ops(layers, sliding_window, kv_seq);

        let params = self.params_template;
        let eps = params.eps;
        let attn_scale = params.attn_scale;

        for op in &self.ops {
            match op.op_type {
                op::RMS_NORM => {
                    let w = &self.weights.f32_views[op.weight_buf_idx as usize];
                    ctx.encode_rmsnorm_view(
                        encoder,
                        buffers.buf(op.in_buf),
                        w,
                        buffers.buf(op.out_buf),
                        op.arg0,
                        eps,
                    );
                }
                op::RMS_NORM_PER_HEAD => {
                    let w = &self.weights.f32_views[op.weight_buf_idx as usize];
                    ctx.encode_rmsnorm_per_head_view(
                        encoder,
                        buffers.buf(op.in_buf),
                        w,
                        buffers.buf(op.out_buf),
                        op.arg0,
                        op.arg1,
                        eps,
                    );
                }
                op::RMS_NORM_PER_HEAD_NOWEIGHT => {
                    ctx.encode_rmsnorm_per_head_noweight(
                        encoder,
                        buffers.buf(op.in_buf),
                        buffers.buf(op.out_buf),
                        op.arg0,
                        op.arg1,
                        eps,
                    );
                }
                op::MATVEC_Q4 => {
                    let w = &self.weights.q4_views[op.weight_buf_idx as usize];
                    if weight_buf_is_q3(w, op.arg0, op.arg1) {
                        ctx.encode_matvec_q3_at_view(
                            encoder,
                            w,
                            buffers.buf(op.in_buf),
                            0,
                            buffers.buf(op.out_buf),
                            0,
                            op.arg0,
                            op.arg1,
                        );
                    } else {
                        ctx.encode_matvec_q4_view(
                            encoder,
                            w,
                            buffers.buf(op.in_buf),
                            buffers.buf(op.out_buf),
                            op.arg0,
                            op.arg1,
                        );
                    }
                }
                op::MATVEC_F16 => {
                    let w = &self.weights.f16_views[op.weight_buf_idx as usize];
                    ctx.encode_matvec_f16_view(
                        encoder,
                        w,
                        buffers.buf(op.in_buf),
                        buffers.buf(op.out_buf),
                        op.arg0,
                        op.arg1,
                    );
                }
                op::ROTARY_Q => {
                    let layer_idx = op.arg0 as usize;
                    let head_dim = layers[layer_idx].head_dim as u32;
                    let rope_off = (layer_idx * rope_max_head_dim * std::mem::size_of::<f32>()) as u64;
                    ctx.encode_rotary_at(
                        encoder,
                        buffers.buf(op.in_buf),
                        0,
                        buffers.buf(buf::KN),
                        0,
                        cos_packed,
                        rope_off,
                        sin_packed,
                        rope_off,
                        params.num_heads,
                        0,
                        head_dim,
                    );
                }
                op::ROTARY_K => {
                    let layer_idx = op.arg0 as usize;
                    let head_dim = layers[layer_idx].head_dim as u32;
                    let rope_off = (layer_idx * rope_max_head_dim * std::mem::size_of::<f32>()) as u64;
                    ctx.encode_rotary_at(
                        encoder,
                        buffers.buf(buf::QN),
                        0,
                        buffers.buf(op.in_buf),
                        0,
                        cos_packed,
                        rope_off,
                        sin_packed,
                        rope_off,
                        0,
                        params.num_kv_heads,
                        head_dim,
                    );
                }
                op::KV_APPEND_Q4 => {
                    let layer_idx = op.arg0 as usize;
                    let head_dim = layers[layer_idx].head_dim as u32;
                    let cache = if op.arg1 == 0 {
                        &k_cache[layer_idx]
                    } else {
                        &v_cache[layer_idx]
                    };
                    ctx.encode_kv_append_q4_0(
                        encoder,
                        buffers.buf(op.in_buf),
                        cache,
                        params.num_kv_heads,
                        head_dim,
                        params.kv_capacity,
                        kv_seq,
                    );
                }
                op::ATTENTION_Q4 => {
                    let layer_idx = op.arg3 as usize;
                    let kv_layer = op.arg0 as usize;
                    let head_dim = layers[layer_idx].head_dim as u32;
                    let groups_per_row = head_dim / 32;
                    let row_bytes = groups_per_row * 18;
                    ctx.encode_attention_with_offset_q4_0(
                        encoder,
                        buffers.buf(op.in_buf),
                        &k_cache[kv_layer],
                        &v_cache[kv_layer],
                        buffers.buf(op.out_buf),
                        params.num_heads,
                        params.num_kv_heads,
                        params.num_kv_groups,
                        head_dim,
                        op.arg1,
                        params.kv_capacity,
                        attn_scale,
                        op.arg2,
                        groups_per_row,
                        row_bytes,
                    );
                }
                op::VEC_ADD => {
                    ctx.encode_vec_add(
                        encoder,
                        buffers.buf(op.in_buf),
                        buffers.buf(op.aux_buf),
                        buffers.buf(op.out_buf),
                        op.arg0,
                    );
                }
                op::VEC_SCALE => {
                    let scale = f32::from_bits(op.arg1);
                    ctx.encode_vec_scale(
                        encoder,
                        buffers.buf(op.in_buf),
                        buffers.buf(op.out_buf),
                        op.arg0,
                        scale,
                    );
                }
                op::GELU_MUL => {
                    ctx.encode_gelu_mul(
                        encoder,
                        buffers.buf(op.in_buf),
                        buffers.buf(op.aux_buf),
                        buffers.buf(op.out_buf),
                        op.arg0,
                    );
                }
                op::GELU_MUL_AT => {
                    let ple_dim = op.arg1;
                    let layer = op.arg2;
                    let off = (layer * ple_dim) as u64;
                    ctx.encode_gelu_mul_at(
                        encoder,
                        buffers.buf(op.in_buf),
                        0,
                        buffers.buf(op.aux_buf),
                        off,
                        buffers.buf(op.out_buf),
                        0,
                        ple_dim,
                    );
                }
                op::VEC_ADD_SCALED => {
                    let scale = f32::from_bits(op.arg1);
                    ctx.encode_vec_add_scaled(
                        encoder,
                        buffers.buf(op.in_buf),
                        buffers.buf(op.aux_buf),
                        buffers.buf(op.out_buf),
                        op.arg0,
                        scale,
                    );
                }
                _ => {}
            }
        }

        if let Some((temperature, min_p, seed)) = sample {
            ctx.encode_sample(
                encoder,
                buffers.logits,
                buffers.sample_out,
                params.vocab_size,
                params.final_logit_cap,
                temperature,
                min_p,
                seed,
            );
        }
    }
}

pub struct MegaScratchBuffers<'a> {
    pub hidden: &'a Buffer,
    pub normed: &'a Buffer,
    pub q: &'a Buffer,
    pub k: &'a Buffer,
    pub v: &'a Buffer,
    pub q_normed: &'a Buffer,
    pub k_normed: &'a Buffer,
    pub gate: &'a Buffer,
    pub attn_out: &'a Buffer,
    pub o_out: &'a Buffer,
    pub up: &'a Buffer,
    pub gelu: &'a Buffer,
    pub down: &'a Buffer,
    pub ple_ctx: &'a Buffer,
    pub ple_tmp: &'a Buffer,
    pub ple_tok: &'a Buffer,
    pub logits: &'a Buffer,
    pub sample_out: &'a Buffer,
}

impl<'a> MegaScratchBuffers<'a> {
    pub fn buf(&self, id: u32) -> &Buffer {
        match id {
            buf::HIDDEN => self.hidden,
            buf::NORMED => self.normed,
            buf::Q => self.q,
            buf::K => self.k,
            buf::V => self.v,
            buf::QN => self.q_normed,
            buf::KN => self.k_normed,
            buf::VN => self.gate,
            buf::ATTN => self.attn_out,
            buf::O => self.o_out,
            buf::GATE => self.gate,
            buf::UP => self.up,
            buf::GELU => self.gelu,
            buf::DOWN => self.down,
            buf::PLE_CTX => self.ple_ctx,
            buf::PLE_TMP => self.ple_tmp,
            buf::PLE_TOK => self.ple_tok,
            buf::LOGITS => self.logits,
            _ => panic!("unknown mega scratch buffer id {id}"),
        }
    }
}
