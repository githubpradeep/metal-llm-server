//! Fused decode layer executor — one code path, maximum fusion, no env-var ladder.
//!
//! Covers Q4_0 and native K-quant (Q4_K_M) weights with Q4_0 KV cache.

use metal::ComputeCommandEncoderRef;

use crate::gemma4_config::KvCacheType;
use crate::gemma4_gpu_model::{Gemma4GpuModel, WeightFormat};
use crate::gpu::{self, BufferView};

/// Global dispatch counter (set `PROFILE_DISPATCHES=1` to print per token).
static DISPATCH_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn fused_decode_enabled() -> bool {
    !matches!(
        std::env::var("FUSED_DECODE").as_deref(),
        Ok("0") | Ok("false") | Ok("FALSE")
    )
}

pub fn profile_dispatches_enabled() -> bool {
    std::env::var("PROFILE_DISPATCHES").is_ok()
}

pub fn take_dispatch_count() -> u32 {
    DISPATCH_COUNTER.swap(0, std::sync::atomic::Ordering::Relaxed)
}

fn bump_dispatches(n: u32) {
    DISPATCH_COUNTER.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
}

/// Print fused-decode eligibility once at model load.
pub fn log_fused_decode_status(model: &Gemma4GpuModel) {
    if !fused_decode_enabled() {
        println!("  Fused decode executor disabled (FUSED_DECODE=0)");
        return;
    }
    if model.kv_cache_type != KvCacheType::Q4_0 {
        println!(
            "  Fused decode executor skipped (KV cache {:?}, need q4_0)",
            model.kv_cache_type
        );
        return;
    }
    if !model.ctx.use_flash_attention {
        println!("  Fused decode executor skipped (flash attention disabled)");
        return;
    }
    if model
        .layers
        .iter()
        .any(|l| l.weight_format == WeightFormat::F16)
    {
        println!("  Fused decode executor skipped (F16 weight layers present)");
        return;
    }
    let kq = model
        .layers
        .iter()
        .filter(|l| l.weight_format.is_kquant())
        .count();
    let q4 = model.layers.len() - kq;
    if kq > 0 {
        println!(
            "  Fused decode executor enabled ({} K-quant + {} Q4_0 layers, Q4_0 KV)",
            kq, q4
        );
    } else {
        println!(
            "  Fused decode executor enabled ({} Q4_0 layers, Q4_0 KV)",
            q4
        );
    }
}

pub struct FusedDecodeScratch<'a> {
    pub hidden: &'a metal::Buffer,
    pub normed: &'a metal::Buffer,
    pub inv_rms: &'a metal::Buffer,
    pub q: &'a metal::Buffer,
    pub k: &'a metal::Buffer,
    pub v: &'a metal::Buffer,
    pub q_normed: &'a metal::Buffer,
    pub k_normed: &'a metal::Buffer,
    pub attn_out: &'a metal::Buffer,
    pub o_out: &'a metal::Buffer,
    pub gate: &'a metal::Buffer,
    pub up: &'a metal::Buffer,
    pub gelu: &'a metal::Buffer,
    pub down: &'a metal::Buffer,
    pub ple_ctx: &'a metal::Buffer,
    pub ple_normed: &'a metal::Buffer,
    pub ple_projected: &'a metal::Buffer,
    pub cos_packed: &'a metal::Buffer,
    pub sin_packed: &'a metal::Buffer,
}

impl Gemma4GpuModel {
    /// Fused fast path: Q4_0 KV cache, Q4_0 and/or K-quant weight layers.
    pub fn fused_decode_eligible(&self) -> bool {
        if !fused_decode_enabled() {
            return false;
        }
        if self.kv_cache_type != KvCacheType::Q4_0 {
            return false;
        }
        if !self.ctx.use_flash_attention {
            return false;
        }
        self.layers.iter().all(|l| {
            l.weight_format == WeightFormat::Q4_0 || l.weight_format.is_kquant()
        })
    }

    /// Encode one decode layer using the maximum-fusion stack.
    pub fn encode_fused_decode_layer(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer_idx: usize,
        kv_seq: u32,
        scratch: &FusedDecodeScratch<'_>,
        skip_attn: bool,
        skip_mlp: bool,
        skip_ple: bool,
    ) {
        let layer = &self.layers[layer_idx];
        let hidden_size = self.config.hidden_size as u32;
        let num_heads = self.config.num_attention_heads as u32;
        let num_kv_heads = self.config.num_key_value_heads as u32;
        let num_kv_groups = (num_heads / num_kv_heads.max(1)) as u32;
        let head_dim = layer.head_dim as u32;
        let q_out = layer.q_out_dim as u32;
        let kv_out = layer.kv_out_dim as u32;
        let intermediate_size = layer.intermediate_size as u32;
        let ple_dim = self.config.hidden_size_per_layer_input as u32;
        let eps = self.config.rms_norm_eps as f32;
        let scale = 1.0f32;
        let rope_off = self.decode_rope_byte_offset(layer_idx);
        let is_full = layer.is_full_attention;
        let attn_kv_seq = kv_seq + 1;
        let effective_kv_seq = if is_full {
            attn_kv_seq
        } else {
            attn_kv_seq.min(self.config.sliding_window as u32)
        };
        let kv_start = if !is_full && attn_kv_seq > self.config.sliding_window as u32 {
            attn_kv_seq - self.config.sliding_window as u32
        } else {
            0u32
        };
        let groups_per_row = head_dim / 32;
        let row_bytes = groups_per_row * 18;
        let mut n = 0u32;

        if !skip_attn {
            n += self.encode_fused_attn_layer(
                encoder,
                layer_idx,
                layer,
                kv_seq,
                scratch,
                num_heads,
                num_kv_heads,
                num_kv_groups,
                head_dim,
                q_out,
                kv_out,
                hidden_size,
                effective_kv_seq,
                kv_start,
                groups_per_row,
                row_bytes,
                rope_off,
                scale,
                eps,
            );
        }

        if !skip_mlp {
            n += self.encode_fused_mlp_layer(
                encoder,
                layer,
                scratch,
                hidden_size,
                intermediate_size,
                eps,
            );
        }

        if !skip_ple {
            n += self.encode_fused_ple_layer(
                encoder,
                layer_idx,
                layer,
                scratch,
                hidden_size,
                ple_dim,
                eps,
            );
        }

        self.ctx.encode_vec_scale(
            encoder,
            scratch.hidden,
            scratch.hidden,
            hidden_size,
            layer.layer_scalar,
        );
        n += 1;

        bump_dispatches(n);
    }

    fn encode_fused_attn_layer(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer_idx: usize,
        layer: &crate::gemma4_gpu_model::Gemma4GpuLayer,
        kv_seq: u32,
        scratch: &FusedDecodeScratch<'_>,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        q_out: u32,
        kv_out: u32,
        hidden_size: u32,
        effective_kv_seq: u32,
        kv_start: u32,
        groups_per_row: u32,
        row_bytes: u32,
        rope_off: u64,
        scale: f32,
        eps: f32,
    ) -> u32 {
        let mut n = 0u32;

        // QKV: inv_rms + fused K-quant or Q4_0 matvec (2 dispatches).
        if layer.weight_format.is_kquant() {
            if layer.has_kv {
                self.ctx.encode_rmsnorm_qkv_kquant_view(
                    encoder,
                    scratch.hidden,
                    &layer.input_layernorm_weight,
                    scratch.inv_rms,
                    &layer.q_proj,
                    &layer.k_proj,
                    &layer.v_proj,
                    scratch.q,
                    scratch.k,
                    scratch.v,
                    q_out,
                    kv_out,
                    hidden_size,
                    eps,
                );
            } else {
                self.ctx.encode_rmsnorm_q_kquant_view(
                    encoder,
                    scratch.hidden,
                    &layer.input_layernorm_weight,
                    scratch.inv_rms,
                    &layer.q_proj,
                    scratch.q,
                    q_out,
                    hidden_size,
                    eps,
                );
            }
            n += 2;
        } else if layer.has_kv {
            self.ctx.encode_rmsnorm_qkv_q4_view(
                encoder,
                scratch.hidden,
                &layer.input_layernorm_weight,
                scratch.inv_rms,
                &layer.q_proj,
                &layer.k_proj,
                &layer.v_proj,
                scratch.q,
                scratch.k,
                scratch.v,
                q_out,
                kv_out,
                hidden_size,
                eps,
            );
            n += 2;
        } else {
            self.ctx.encode_rmsnorm_q_q4_view(
                encoder,
                scratch.hidden,
                &layer.input_layernorm_weight,
                scratch.inv_rms,
                &layer.q_proj,
                scratch.q,
                q_out,
                hidden_size,
                eps,
            );
            n += 2;
        }

        if layer.has_kv
            && matches!(head_dim, 128 | 256 | 512)
            && !gpu::attention_use_ggml_for_layer(true)
        {
            self.ctx.encode_attention_full_fused_q4_0(
                encoder,
                scratch.q,
                &layer.q_norm_weight,
                scratch.cos_packed,
                rope_off,
                scratch.sin_packed,
                rope_off,
                scratch.k,
                &layer.k_norm_weight,
                scratch.v,
                scratch.attn_out,
                &self.k_cache[layer.kv_source_layer],
                &self.v_cache[layer.kv_source_layer],
                num_heads,
                num_kv_heads,
                num_kv_groups,
                head_dim,
                effective_kv_seq,
                self.kv_capacity,
                scale,
                kv_start,
                kv_seq,
                groups_per_row,
                row_bytes,
                eps,
            );
            n += 1;
        } else if !layer.has_kv
            && matches!(head_dim, 128 | 256 | 512)
            && !gpu::attention_use_ggml_for_layer(false)
        {
            self.ctx.encode_attention_qknorm_rope_q4_0(
                encoder,
                scratch.q,
                &layer.q_norm_weight,
                scratch.cos_packed,
                rope_off,
                scratch.sin_packed,
                rope_off,
                &self.k_cache[layer.kv_source_layer],
                &self.v_cache[layer.kv_source_layer],
                scratch.attn_out,
                num_heads,
                num_kv_heads,
                num_kv_groups,
                head_dim,
                effective_kv_seq,
                self.kv_capacity,
                scale,
                kv_start,
                groups_per_row,
                row_bytes,
                eps,
            );
            n += 1;
        } else {
            self.ctx.encode_rmsnorm_per_head_view(
                encoder,
                scratch.q,
                &layer.q_norm_weight,
                scratch.q_normed,
                num_heads,
                head_dim,
                eps,
            );
            self.ctx.encode_rotary_at(
                encoder,
                scratch.q_normed,
                0,
                scratch.k_normed,
                0,
                scratch.cos_packed,
                rope_off,
                scratch.sin_packed,
                rope_off,
                num_heads,
                0,
                head_dim,
            );
            n += 2;

            if layer.has_kv {
                self.ctx.encode_rmsnorm_per_head_view(
                    encoder,
                    scratch.k,
                    &layer.k_norm_weight,
                    scratch.k_normed,
                    num_kv_heads,
                    head_dim,
                    eps,
                );
                self.ctx.encode_rotary_at(
                    encoder,
                    scratch.q,
                    0,
                    scratch.k_normed,
                    0,
                    scratch.cos_packed,
                    rope_off,
                    scratch.sin_packed,
                    rope_off,
                    0,
                    num_kv_heads,
                    head_dim,
                );
                self.ctx.encode_rmsnorm_per_head_noweight(
                    encoder,
                    scratch.v,
                    scratch.gate,
                    num_kv_heads,
                    head_dim,
                    eps,
                );
                n += 3;
                if !gpu::fused_kv_attention_enabled() {
                    self.ctx.encode_kv_append_q4_0(
                        encoder,
                        scratch.k_normed,
                        &self.k_cache[layer_idx],
                        num_kv_heads,
                        head_dim,
                        self.kv_capacity,
                        kv_seq,
                    );
                    self.ctx.encode_kv_append_q4_0(
                        encoder,
                        scratch.gate,
                        &self.v_cache[layer_idx],
                        num_kv_heads,
                        head_dim,
                        self.kv_capacity,
                        kv_seq,
                    );
                    n += 2;
                }
            }

            if gpu::attention_gqa_q4_0_enabled(num_kv_groups) {
                self.ctx.encode_attention_with_offset_q4_0_gqa(
                    encoder,
                    scratch.q_normed,
                    &self.k_cache[layer.kv_source_layer],
                    &self.v_cache[layer.kv_source_layer],
                    scratch.attn_out,
                    num_heads,
                    num_kv_heads,
                    num_kv_groups,
                    head_dim,
                    effective_kv_seq,
                    self.kv_capacity,
                    scale,
                    kv_start,
                    groups_per_row,
                    row_bytes,
                );
            } else {
                self.ctx.encode_attention_with_offset_q4_0(
                    encoder,
                    scratch.q_normed,
                    &self.k_cache[layer.kv_source_layer],
                    &self.v_cache[layer.kv_source_layer],
                    scratch.attn_out,
                    num_heads,
                    num_kv_heads,
                    num_kv_groups,
                    head_dim,
                    effective_kv_seq,
                    self.kv_capacity,
                    scale,
                    kv_start,
                    groups_per_row,
                    row_bytes,
                );
            }
            n += 1;
        }

        self.encode_matvec_quant_layer(
            encoder,
            &layer.o_proj,
            scratch.attn_out,
            scratch.o_out,
            hidden_size,
            q_out,
            layer.weight_format,
        );
        n += 1;
        self.ctx.encode_rmsnorm_acc_view(
            encoder,
            scratch.hidden,
            scratch.o_out,
            &layer.post_attention_layernorm_weight,
            hidden_size,
            eps,
        );
        n += 1;

        n
    }

    /// K-quant MLP: best path matches legacy ladder (rmsnorm+gate∥up+gelu, then Q6_K down).
    fn encode_fused_mlp_layer(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer: &crate::gemma4_gpu_model::Gemma4GpuLayer,
        scratch: &FusedDecodeScratch<'_>,
        hidden_size: u32,
        intermediate_size: u32,
        eps: f32,
    ) -> u32 {
        let mut n = 0u32;
        use crate::gpu::weight_fmt;

        if layer.weight_format.is_kquant() {
            let gate_up_q4k = layer.gate_proj.format == weight_fmt::Q4_K
                && layer.up_proj.format == weight_fmt::Q4_K;

            if gate_up_q4k && gpu::fused_rmsnorm_mlp_kquant_enabled() {
                // inv_rms + Q4_K gate∥up+GeLU from hidden (2 dispatches).
                self.ctx.encode_rmsnorm_qk_gelu_mul_kquant_at_view(
                    encoder,
                    &layer.gate_proj,
                    &layer.up_proj,
                    scratch.hidden,
                    0,
                    &layer.pre_feedforward_layernorm_weight,
                    scratch.inv_rms,
                    scratch.gelu,
                    0,
                    intermediate_size,
                    hidden_size,
                    eps,
                );
                n += 2;
            } else if gate_up_q4k {
                self.ctx.encode_rmsnorm_view(
                    encoder,
                    scratch.hidden,
                    &layer.pre_feedforward_layernorm_weight,
                    scratch.normed,
                    hidden_size,
                    eps,
                );
                n += 1;
                self.ctx.encode_matvec_qk_gelu_mul_at_view(
                    encoder,
                    &layer.gate_proj,
                    &layer.up_proj,
                    scratch.normed,
                    0,
                    scratch.gelu,
                    0,
                    intermediate_size,
                    hidden_size,
                );
                n += 1;
            } else {
                self.ctx.encode_rmsnorm_view(
                    encoder,
                    scratch.hidden,
                    &layer.pre_feedforward_layernorm_weight,
                    scratch.normed,
                    hidden_size,
                    eps,
                );
                n += 1;
                self.encode_matvec_quant_layer(
                    encoder,
                    &layer.gate_proj,
                    scratch.normed,
                    scratch.gate,
                    intermediate_size,
                    hidden_size,
                    layer.weight_format,
                );
                self.encode_matvec_quant_layer(
                    encoder,
                    &layer.up_proj,
                    scratch.normed,
                    scratch.up,
                    intermediate_size,
                    hidden_size,
                    layer.weight_format,
                );
                self.ctx.encode_gelu_mul(
                    encoder,
                    scratch.gate,
                    scratch.up,
                    scratch.gelu,
                    intermediate_size,
                );
                n += 3;
            }

            // Down projection (typically Q6_K on Q4_K_M).
            self.encode_matvec_quant_layer(
                encoder,
                &layer.down_proj,
                scratch.gelu,
                scratch.down,
                hidden_size,
                intermediate_size,
                layer.weight_format,
            );
            n += 1;
        } else if Self::use_packed_mlp_gate_up(layer)
            && gpu::weight_buf_is_q4(
                &layer.gate_proj,
                intermediate_size,
                hidden_size,
            )
        {
            self.ctx.encode_mlp_fused_q4_gelu_down_packed_from_hidden_at_view(
                encoder,
                &layer.gate_up_proj,
                &layer.down_proj,
                &layer.pre_feedforward_layernorm_weight,
                scratch.hidden,
                0,
                scratch.inv_rms,
                0,
                scratch.up,
                0,
                scratch.down,
                0,
                hidden_size,
                intermediate_size,
                eps,
            );
            n += 3;
        } else if gpu::mlp_gate_up_ggml_enabled() {
            self.ctx.encode_rmsnorm_view(
                encoder,
                scratch.hidden,
                &layer.pre_feedforward_layernorm_weight,
                scratch.normed,
                hidden_size,
                eps,
            );
            self.ctx.encode_mlp_fused_q4_gelu_down_ggml_at_view(
                encoder,
                &layer.gate_proj,
                &layer.up_proj,
                &layer.down_proj,
                scratch.normed,
                0,
                scratch.gate,
                scratch.up,
                scratch.gelu,
                0,
                scratch.down,
                0,
                hidden_size,
                intermediate_size,
            );
            n += 2;
        } else {
            self.ctx.encode_rmsnorm_view(
                encoder,
                scratch.hidden,
                &layer.pre_feedforward_layernorm_weight,
                scratch.normed,
                hidden_size,
                eps,
            );
            self.ctx.encode_mlp_fused_q4_gelu_down_packed_at_view(
                encoder,
                &layer.gate_up_proj,
                &layer.down_proj,
                scratch.normed,
                0,
                scratch.up,
                0,
                scratch.down,
                0,
                hidden_size,
                intermediate_size,
            );
            n += 2;
        }

        self.ctx.encode_rmsnorm_acc_view(
            encoder,
            scratch.hidden,
            scratch.down,
            &layer.post_feedforward_layernorm_weight,
            hidden_size,
            eps,
        );
        n += 1;

        n
    }

    fn encode_fused_ple_layer(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer_idx: usize,
        layer: &crate::gemma4_gpu_model::Gemma4GpuLayer,
        scratch: &FusedDecodeScratch<'_>,
        hidden_size: u32,
        ple_dim: u32,
        eps: f32,
    ) -> u32 {
        let mut n = 0u32;
        let ple_off = (layer_idx as u32 * ple_dim * 4) as u64;

        if gpu::fused_mlp_ple_enabled()
            && gpu::weight_buf_is_q4(
                &layer.per_layer_input_gate_weight,
                ple_dim,
                hidden_size,
            )
        {
            self.ctx.encode_ple_matvec_gelu_q4_at_view(
                encoder,
                &layer.per_layer_input_gate_weight,
                scratch.hidden,
                0,
                scratch.ple_ctx,
                ple_off,
                scratch.ple_normed,
                0,
                ple_dim,
                hidden_size,
            );
            n += 1;
        } else {
            // K-quant PLE gate weights route through ggml Q4_K / Q6_K matvec.
            self.encode_matvec_auto_layer(
                encoder,
                &layer.per_layer_input_gate_weight,
                scratch.hidden,
                scratch.gate,
                ple_dim,
                hidden_size,
            );
            self.ctx.encode_gelu_mul_at(
                encoder,
                scratch.gate,
                0,
                scratch.ple_ctx,
                ple_off,
                scratch.ple_normed,
                0,
                ple_dim,
            );
            n += 2;
        }

        self.encode_matvec_auto_layer(
            encoder,
            &layer.per_layer_projection_weight,
            scratch.ple_normed,
            scratch.ple_projected,
            hidden_size,
            ple_dim,
        );
        n += 1;
        self.ctx.encode_rmsnorm_acc_view(
            encoder,
            scratch.hidden,
            scratch.ple_projected,
            &layer.post_per_layer_input_norm_weight,
            hidden_size,
            eps,
        );
        n += 1;

        n
    }

    fn encode_matvec_quant_layer(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &metal::Buffer,
        y_buf: &metal::Buffer,
        m: u32,
        k: u32,
        wf: WeightFormat,
    ) {
        use crate::gpu::weight_fmt;
        match weight.format {
            weight_fmt::Q4_K | weight_fmt::Q6_K => self.ctx.encode_matvec_qk_at_view(
                encoder, weight, x_buf, 0, y_buf, 0, m, k, 1,
            ),
            _ => match wf {
                WeightFormat::Q3_0 => self.ctx.encode_matvec_q3_at_view(
                    encoder, weight, x_buf, 0, y_buf, 0, m, k,
                ),
                WeightFormat::F16 => self.ctx.encode_matvec_f16_view(
                    encoder, weight, x_buf, y_buf, m, k,
                ),
                _ => self.ctx.encode_matvec_q4_at_view(
                    encoder, weight, x_buf, 0, y_buf, 0, m, k,
                ),
            },
        }
    }

    fn encode_matvec_auto_layer(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &metal::Buffer,
        y_buf: &metal::Buffer,
        m: u32,
        k: u32,
    ) {
        self.ctx
            .encode_matvec_auto_view(encoder, weight, x_buf, y_buf, m, k);
    }
}
