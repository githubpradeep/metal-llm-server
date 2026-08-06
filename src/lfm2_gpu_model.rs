//! LFM2 / LFM2.5 Metal GGUF inference (hybrid short-conv + attention).

use metal::*;
use std::sync::Arc;
use std::time::Instant;

use crate::gguf::{self, Gguf};
use crate::gpu::{weight_buf_is_kquant, weight_fmt, BufferView, MetalContext};
use crate::lfm2_config::{lfm2_config_from_gguf, Lfm2Config};

/// GPU RoPE params for one layer (must match `RopeLayerParams` in llama.metal).
#[repr(C)]
struct RopeLayerParams {
    theta: f32,
    factor: f32,
    head_dim: u32,
    rope_angles: u32,
}

fn configured_kv_capacity(max_position_embeddings: usize) -> u32 {
    const DEFAULT_KV_CAPACITY: usize = 8192;
    const ABSOLUTE_MAX_KV_CAPACITY: usize = 200_000;
    let requested = std::env::var("LLAMA_CTX_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_KV_CAPACITY);
    let capped = requested.clamp(256, ABSOLUTE_MAX_KV_CAPACITY);
    if capped > max_position_embeddings {
        eprintln!(
            "  Warning: LLAMA_CTX_SIZE={} > model context_length={} ",
            capped, max_position_embeddings
        );
    }
    capped as u32
}

fn upload_qw(ctx: &MetalContext, g: &Gguf, name: &str, rows: usize, cols: usize) -> BufferView {
    match g.tensor_type(name) {
        gguf::ggml_type::Q4_K => BufferView::from_buffer(ctx.buffer_from_slice_no_copy(g.tensor_raw(name)))
            .with_format(weight_fmt::Q4_K),
        gguf::ggml_type::Q6_K => BufferView::from_buffer(ctx.buffer_from_slice_no_copy(g.tensor_raw(name)))
            .with_format(weight_fmt::Q6_K),
        gguf::ggml_type::Q8_0 => BufferView::from_buffer(ctx.buffer_from_slice_no_copy(g.tensor_raw(name)))
            .with_format(weight_fmt::Q8_0),
        gguf::ggml_type::Q4_0 => BufferView::from_buffer(ctx.buffer_from_slice_no_copy(g.tensor_raw(name)))
            .with_format(weight_fmt::Q4_0),
        gguf::ggml_type::BF16 | gguf::ggml_type::F16 => {
            BufferView::from_buffer(ctx.buffer_from_slice_no_copy(g.tensor_raw(name)))
                .with_format(weight_fmt::F16)
        }
        gguf::ggml_type::F32 => {
            let data = g.dequant_to_f32(name);
            BufferView::from_buffer(ctx.buffer_from_f32_as_f16(&data)).with_format(weight_fmt::F16)
        }
        _ => {
            let data = g.dequant_to_f32(name);
            BufferView::from_buffer(ctx.buffer_from_f32_as_q4(&data, rows, cols))
        }
    }
}

fn upload_f32(ctx: &MetalContext, g: &Gguf, name: &str) -> BufferView {
    BufferView::from_buffer(ctx.buffer_from_slice(&g.dequant_to_f32(name)))
}

struct EmbedTable {
    gguf: Arc<Gguf>,
    ggml_type: u32,
    row_stride: usize,
    cols: usize,
}

impl EmbedTable {
    fn from_gguf(gguf: Arc<Gguf>, vocab_size: usize, cols: usize) -> Self {
        let e = gguf.tensor("token_embd.weight").expect("token_embd.weight");
        let row_stride = e.byte_len() / vocab_size;
        assert_eq!(e.num_elements() / vocab_size, cols);
        Self {
            ggml_type: e.ggml_type,
            row_stride,
            cols,
            gguf,
        }
    }

    fn decode_into(&self, token_id: usize, out: &mut [f32]) {
        let bytes = self
            .gguf
            .tensor_row_bytes("token_embd.weight", token_id, self.row_stride);
        gguf::dequant_row_to_f32(self.ggml_type, bytes, self.cols, out);
    }
}

enum Lfm2LayerKind {
    ShortConv {
        in_proj: BufferView,
        conv: BufferView,
        out_proj: BufferView,
        /// Index into `conv_states`.
        state_idx: usize,
    },
    Attention {
        q_proj: BufferView,
        k_proj: BufferView,
        v_proj: BufferView,
        o_proj: BufferView,
        q_norm: BufferView,
        k_norm: BufferView,
        /// Index into `k_caches` / `v_caches`.
        kv_idx: usize,
        num_kv_heads: usize,
    },
}

fn attn_uses_fused_qkv(kind: &Lfm2LayerKind) -> bool {
    match kind {
        Lfm2LayerKind::Attention {
            q_proj, k_proj, v_proj, ..
        } => {
            weight_buf_is_kquant(q_proj)
                && weight_buf_is_kquant(k_proj)
                && weight_buf_is_kquant(v_proj)
        }
        Lfm2LayerKind::ShortConv { .. } => false,
    }
}

struct Lfm2Layer {
    operator_norm: BufferView,
    ffn_norm: BufferView,
    gate_proj: BufferView,
    up_proj: BufferView,
    down_proj: BufferView,
    kind: Lfm2LayerKind,
}

pub struct Lfm2GpuModel {
    pub ctx: MetalContext,
    pub config: Lfm2Config,
    layers: Vec<Lfm2Layer>,
    final_norm: BufferView,
    lm_head: BufferView,
    embed: EmbedTable,

    residual_buf: Buffer,
    normed_buf: Buffer,
    q_buf: Buffer,
    k_buf: Buffer,
    v_buf: Buffer,
    attn_out_buf: Buffer,
    o_out_buf: Buffer,
    gate_buf: Buffer,
    up_buf: Buffer,
    silu_buf: Buffer,
    down_buf: Buffer,
    logits_buf: Buffer,
    bcx_buf: Buffer,
    shortconv_y_buf: Buffer,
    cos_buf: Buffer,
    sin_buf: Buffer,
    sample_buf: Buffer,
    inv_rms_buf: Buffer,
    rope_layer_params_buf: Buffer,
    embed_scratch: Vec<f32>,

    k_caches: Vec<Buffer>,
    v_caches: Vec<Buffer>,
    conv_states: Vec<Buffer>,
    kv_capacity: u32,
    /// Shared sequence length (attn KV + position).
    pub seq_len: u32,
}

impl Lfm2GpuModel {
    pub fn load_from_gguf(gguf_path: &str) -> Self {
        let load_start = Instant::now();
        let g = Arc::new(Gguf::open(gguf_path));
        let arch = g.get_str("general.architecture").unwrap_or("");
        assert_eq!(arch, "lfm2", "GGUF architecture is '{}', expected 'lfm2'", arch);

        let config = lfm2_config_from_gguf(&g);
        let ctx = MetalContext::new();

        let hidden = config.hidden_size;
        let inter = config.intermediate_size;
        let n_heads = config.num_attention_heads;
        let head_dim = config.head_dim();
        let vocab = config.vocab_size;
        let l_cache = config.shortconv_l_cache;
        let state_elems = config.conv_state_elems();

        let n_attn = config
            .num_key_value_heads
            .iter()
            .filter(|&&h| h > 0)
            .count();
        let n_sc = config.num_hidden_layers - n_attn;

        println!(
            "  LFM2 (GGUF): {} layers ({} attn / {} shortconv), hidden={}, heads={}, head_dim={}, ff={}, vocab={}",
            config.num_hidden_layers, n_attn, n_sc, hidden, n_heads, head_dim, inter, vocab
        );
        println!(
            "  shortconv l_cache={}, RoPE θ={:.0}, rms_eps={:e}",
            l_cache, config.rope_theta, config.rms_norm_eps
        );

        let lm_head = if g.has_tensor("output.weight") {
            upload_qw(&ctx, &g, "output.weight", vocab, hidden)
        } else {
            match g.tensor_type("token_embd.weight") {
                gguf::ggml_type::Q4_K => BufferView::from_buffer(
                    ctx.buffer_from_slice_no_copy(g.tensor_raw("token_embd.weight")),
                )
                .with_format(weight_fmt::Q4_K),
                gguf::ggml_type::Q6_K => BufferView::from_buffer(
                    ctx.buffer_from_slice_no_copy(g.tensor_raw("token_embd.weight")),
                )
                .with_format(weight_fmt::Q6_K),
                gguf::ggml_type::Q4_0 => BufferView::from_buffer(
                    ctx.buffer_from_slice_no_copy(g.tensor_raw("token_embd.weight")),
                )
                .with_format(weight_fmt::Q4_0),
                gguf::ggml_type::F16 | gguf::ggml_type::BF16 => BufferView::from_buffer(
                    ctx.buffer_from_slice_no_copy(g.tensor_raw("token_embd.weight")),
                )
                .with_format(weight_fmt::F16),
                _ => {
                    let data = g.dequant_to_f32("token_embd.weight");
                    BufferView::from_buffer(ctx.buffer_from_f32_as_q4(&data, vocab, hidden))
                }
            }
        };

        let final_norm = upload_f32(&ctx, &g, "token_embd_norm.weight");
        let embed = EmbedTable::from_gguf(Arc::clone(&g), vocab, hidden);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut attn_i = 0usize;
        let mut sc_i = 0usize;
        for il in 0..config.num_hidden_layers {
            let p = |suffix: &str| format!("blk.{}.{}", il, suffix);
            let operator_norm = upload_f32(&ctx, &g, &p("attn_norm.weight"));
            let ffn_norm = upload_f32(&ctx, &g, &p("ffn_norm.weight"));
            let gate_proj = upload_qw(&ctx, &g, &p("ffn_gate.weight"), inter, hidden);
            let up_proj = upload_qw(&ctx, &g, &p("ffn_up.weight"), inter, hidden);
            let down_proj = upload_qw(&ctx, &g, &p("ffn_down.weight"), hidden, inter);

            let kind = if config.is_shortconv(il) {
                let kind = Lfm2LayerKind::ShortConv {
                    in_proj: upload_qw(&ctx, &g, &p("shortconv.in_proj.weight"), 3 * hidden, hidden),
                    conv: upload_f32(&ctx, &g, &p("shortconv.conv.weight")),
                    out_proj: upload_qw(&ctx, &g, &p("shortconv.out_proj.weight"), hidden, hidden),
                    state_idx: sc_i,
                };
                sc_i += 1;
                kind
            } else {
                let n_kv = config.layer_num_kv_heads(il);
                let kv_out = n_kv * head_dim;
                let q_out = n_heads * head_dim;
                let kind = Lfm2LayerKind::Attention {
                    q_proj: upload_qw(&ctx, &g, &p("attn_q.weight"), q_out, hidden),
                    k_proj: upload_qw(&ctx, &g, &p("attn_k.weight"), kv_out, hidden),
                    v_proj: upload_qw(&ctx, &g, &p("attn_v.weight"), kv_out, hidden),
                    o_proj: upload_qw(&ctx, &g, &p("attn_output.weight"), hidden, q_out),
                    q_norm: upload_f32(&ctx, &g, &p("attn_q_norm.weight")),
                    k_norm: upload_f32(&ctx, &g, &p("attn_k_norm.weight")),
                    kv_idx: attn_i,
                    num_kv_heads: n_kv,
                };
                attn_i += 1;
                kind
            };

            layers.push(Lfm2Layer {
                operator_norm,
                ffn_norm,
                gate_proj,
                up_proj,
                down_proj,
                kind,
            });
            if (il + 1) % 5 == 0 || il + 1 == config.num_hidden_layers {
                println!("    loaded layer {}/{}", il + 1, config.num_hidden_layers);
            }
        }

        let kv_capacity = configured_kv_capacity(config.max_position_embeddings);
        let groups_per_row = head_dim / 32;
        let row_bytes = groups_per_row * 18;
        let mut k_caches = Vec::with_capacity(n_attn);
        let mut v_caches = Vec::with_capacity(n_attn);
        for il in 0..config.num_hidden_layers {
            if config.is_shortconv(il) {
                continue;
            }
            let n_kv = config.layer_num_kv_heads(il);
            let bytes = n_kv * kv_capacity as usize * row_bytes;
            k_caches.push(
                ctx.device
                    .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared),
            );
            v_caches.push(
                ctx.device
                    .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared),
            );
        }
        let mut conv_states = Vec::with_capacity(n_sc);
        for _ in 0..n_sc {
            conv_states.push(ctx.buffer_empty(state_elems));
        }

        let max_kv_out = config
            .num_key_value_heads
            .iter()
            .copied()
            .max()
            .unwrap_or(1)
            .max(1)
            * head_dim;

        let residual_buf = ctx.buffer_empty(hidden);
        let normed_buf = ctx.buffer_empty(hidden);
        let q_buf = ctx.buffer_empty(n_heads * head_dim);
        let k_buf = ctx.buffer_empty(max_kv_out);
        let v_buf = ctx.buffer_empty(max_kv_out);
        let attn_out_buf = ctx.buffer_empty(n_heads * head_dim);
        let o_out_buf = ctx.buffer_empty(hidden);
        let gate_buf = ctx.buffer_empty(inter);
        let up_buf = ctx.buffer_empty(inter);
        let silu_buf = ctx.buffer_empty(inter);
        let down_buf = ctx.buffer_empty(hidden);
        let logits_buf = ctx.buffer_empty(vocab);
        let bcx_buf = ctx.buffer_empty(3 * hidden);
        let shortconv_y_buf = ctx.buffer_empty(hidden);
        let cos_buf = ctx.buffer_empty(head_dim);
        let sin_buf = ctx.buffer_empty(head_dim);
        let sample_buf = ctx.buffer_empty_u32(1);
        let inv_rms_buf = ctx.buffer_empty(1);
        let rope_params = [RopeLayerParams {
            theta: config.rope_theta as f32,
            factor: 1.0,
            head_dim: head_dim as u32,
            rope_angles: (head_dim / 2) as u32,
        }];
        let rope_bytes = unsafe {
            std::slice::from_raw_parts(
                rope_params.as_ptr() as *const u8,
                std::mem::size_of_val(&rope_params),
            )
        };
        let rope_layer_params_buf = ctx.buffer_from_bytes(rope_bytes);
        let embed_scratch = vec![0.0; hidden];

        println!(
            "  LFM2 loaded in {:.2}s (KV capacity={}, attn caches={}, conv states={})",
            load_start.elapsed().as_secs_f64(),
            kv_capacity,
            k_caches.len(),
            conv_states.len()
        );

        Self {
            ctx,
            config,
            layers,
            final_norm,
            lm_head,
            embed,
            residual_buf,
            normed_buf,
            q_buf,
            k_buf,
            v_buf,
            attn_out_buf,
            o_out_buf,
            gate_buf,
            up_buf,
            silu_buf,
            down_buf,
            logits_buf,
            bcx_buf,
            shortconv_y_buf,
            cos_buf,
            sin_buf,
            sample_buf,
            inv_rms_buf,
            rope_layer_params_buf,
            embed_scratch,
            k_caches,
            v_caches,
            conv_states,
            kv_capacity,
            seq_len: 0,
        }
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
        let zeros = vec![0.0f32; self.config.conv_state_elems()];
        for st in &self.conv_states {
            MetalContext::write_buffer(st, &zeros);
        }
        // KV caches are overwritten by append; seq_len gate is enough.
    }

    pub fn eos_token_id(&self) -> usize {
        self.config.eos_token_id
    }

    pub fn bos_token_id(&self) -> usize {
        self.config.bos_token_id
    }

    /// Encode one token; if `sample` is true, return greedy argmax token, else update caches only.
    fn forward_one(&mut self, token_id: usize, sample: bool) -> Option<usize> {
        let hidden = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let n_heads = self.config.num_attention_heads;
        let head_dim = self.config.head_dim();
        let eps = self.config.rms_norm_eps as f32;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let l_cache = self.config.shortconv_l_cache as u32;
        let vocab = self.config.vocab_size;
        let groups_per_row = (head_dim / 32) as u32;
        let row_bytes = groups_per_row * 18;
        let n_layers = self.layers.len();

        self.embed.decode_into(token_id, &mut self.embed_scratch);
        MetalContext::write_buffer(&self.residual_buf, &self.embed_scratch);

        let cur_seq = self.seq_len;
        let cmd = self.ctx.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        self.ctx.encode_rope_fill_decode(
            encoder,
            &self.cos_buf,
            &self.sin_buf,
            &self.rope_layer_params_buf,
            1,
            head_dim as u32,
            cur_seq as f32,
        );

        if !attn_uses_fused_qkv(&self.layers[0].kind) {
            self.ctx.encode_rmsnorm_view(
                encoder,
                &self.residual_buf,
                &self.layers[0].operator_norm,
                &self.normed_buf,
                hidden as u32,
                eps,
            );
        }

        for layer_idx in 0..n_layers {
            let layer = &self.layers[layer_idx];

            match &layer.kind {
                Lfm2LayerKind::ShortConv {
                    in_proj,
                    conv,
                    out_proj,
                    state_idx,
                } => {
                    self.ctx.encode_matvec_auto_view(
                        encoder,
                        in_proj,
                        &self.normed_buf,
                        &self.bcx_buf,
                        (3 * hidden) as u32,
                        hidden as u32,
                    );
                    self.ctx.encode_lfm2_shortconv_decode(
                        encoder,
                        &self.bcx_buf,
                        &self.conv_states[*state_idx],
                        conv,
                        &self.shortconv_y_buf,
                        hidden as u32,
                        l_cache,
                    );
                    self.ctx.encode_matvec_auto_view(
                        encoder,
                        out_proj,
                        &self.shortconv_y_buf,
                        &self.o_out_buf,
                        hidden as u32,
                        hidden as u32,
                    );
                }
                Lfm2LayerKind::Attention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm,
                    k_norm,
                    kv_idx,
                    num_kv_heads,
                } => {
                    let n_kv = *num_kv_heads as u32;
                    let n_groups = (n_heads as u32) / n_kv;
                    let q_out = (n_heads * head_dim) as u32;
                    let kv_out = (*num_kv_heads * head_dim) as u32;
                    let fused_qkv = weight_buf_is_kquant(q_proj)
                        && weight_buf_is_kquant(k_proj)
                        && weight_buf_is_kquant(v_proj);

                    if fused_qkv {
                        self.ctx.encode_rmsnorm_qkv_kquant_view(
                            encoder,
                            &self.residual_buf,
                            &layer.operator_norm,
                            &self.inv_rms_buf,
                            q_proj,
                            k_proj,
                            v_proj,
                            &self.q_buf,
                            &self.k_buf,
                            &self.v_buf,
                            q_out,
                            kv_out,
                            hidden as u32,
                            eps,
                        );
                    } else {
                        self.ctx.encode_matvec_auto_view(
                            encoder,
                            q_proj,
                            &self.normed_buf,
                            &self.q_buf,
                            q_out,
                            hidden as u32,
                        );
                        self.ctx.encode_matvec_auto_view(
                            encoder,
                            k_proj,
                            &self.normed_buf,
                            &self.k_buf,
                            kv_out,
                            hidden as u32,
                        );
                        self.ctx.encode_matvec_auto_view(
                            encoder,
                            v_proj,
                            &self.normed_buf,
                            &self.v_buf,
                            kv_out,
                            hidden as u32,
                        );
                    }

                    self.ctx.encode_rmsnorm_per_head_at_view(
                        encoder,
                        &self.q_buf,
                        0,
                        q_norm,
                        &self.q_buf,
                        0,
                        n_heads as u32,
                        head_dim as u32,
                        eps,
                    );
                    self.ctx.encode_rmsnorm_per_head_at_view(
                        encoder,
                        &self.k_buf,
                        0,
                        k_norm,
                        &self.k_buf,
                        0,
                        n_kv,
                        head_dim as u32,
                        eps,
                    );

                    self.ctx.encode_rotary(
                        encoder,
                        &self.q_buf,
                        &self.k_buf,
                        &self.cos_buf,
                        &self.sin_buf,
                        n_heads as u32,
                        n_kv,
                        head_dim as u32,
                    );

                    self.ctx.encode_kv_append_attention_q4_0(
                        encoder,
                        &self.q_buf,
                        &self.k_buf,
                        &self.v_buf,
                        &self.attn_out_buf,
                        &self.k_caches[*kv_idx],
                        &self.v_caches[*kv_idx],
                        n_heads as u32,
                        n_kv,
                        n_groups,
                        head_dim as u32,
                        self.kv_capacity,
                        cur_seq,
                        scale,
                        groups_per_row,
                        row_bytes,
                    );

                    self.ctx.encode_matvec_auto_view(
                        encoder,
                        o_proj,
                        &self.attn_out_buf,
                        &self.o_out_buf,
                        hidden as u32,
                        q_out,
                    );
                }
            }

            let fuse_gate_up = layer.gate_proj.format == weight_fmt::Q4_K
                && layer.up_proj.format == weight_fmt::Q4_K;

            if fuse_gate_up {
                self.ctx.encode_vec_add(
                    encoder,
                    &self.residual_buf,
                    &self.o_out_buf,
                    &self.residual_buf,
                    hidden as u32,
                );
                self.ctx.encode_rmsnorm_qk_silu_mul_kquant_at_view(
                    encoder,
                    &layer.gate_proj,
                    &layer.up_proj,
                    &self.residual_buf,
                    0,
                    &layer.ffn_norm,
                    &self.inv_rms_buf,
                    &self.silu_buf,
                    0,
                    inter as u32,
                    hidden as u32,
                    eps,
                );
            } else {
                self.ctx.encode_rmsnorm_add_save_residual(
                    encoder,
                    &self.residual_buf,
                    &self.o_out_buf,
                    &layer.ffn_norm.buffer,
                    &self.normed_buf,
                    &self.residual_buf,
                    hidden as u32,
                    eps,
                );
                self.ctx.encode_matvec_auto_view(
                    encoder,
                    &layer.gate_proj,
                    &self.normed_buf,
                    &self.gate_buf,
                    inter as u32,
                    hidden as u32,
                );
                self.ctx.encode_matvec_auto_view(
                    encoder,
                    &layer.up_proj,
                    &self.normed_buf,
                    &self.up_buf,
                    inter as u32,
                    hidden as u32,
                );
                self.ctx.encode_silu_mul(
                    encoder,
                    &self.gate_buf,
                    &self.up_buf,
                    &self.silu_buf,
                    inter as u32,
                );
            }

            self.ctx.encode_matvec_auto_view(
                encoder,
                &layer.down_proj,
                &self.silu_buf,
                &self.down_buf,
                hidden as u32,
                inter as u32,
            );

            if layer_idx + 1 < n_layers {
                let next = &self.layers[layer_idx + 1];
                if attn_uses_fused_qkv(&next.kind) {
                    self.ctx.encode_vec_add(
                        encoder,
                        &self.residual_buf,
                        &self.down_buf,
                        &self.residual_buf,
                        hidden as u32,
                    );
                } else {
                    self.ctx.encode_rmsnorm_add_save_residual(
                        encoder,
                        &self.residual_buf,
                        &self.down_buf,
                        &next.operator_norm.buffer,
                        &self.normed_buf,
                        &self.residual_buf,
                        hidden as u32,
                        eps,
                    );
                }
            } else if sample {
                self.ctx.encode_rmsnorm_add(
                    encoder,
                    &self.residual_buf,
                    &self.down_buf,
                    &self.final_norm.buffer,
                    &self.normed_buf,
                    hidden as u32,
                    eps,
                );
                self.ctx.encode_matvec_auto_view(
                    encoder,
                    &self.lm_head,
                    &self.normed_buf,
                    &self.logits_buf,
                    vocab as u32,
                    hidden as u32,
                );
                self.ctx
                    .encode_argmax_f32(encoder, &self.logits_buf, &self.sample_buf, vocab as u32);
            }
        }

        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        self.seq_len += 1;

        if sample {
            Some(MetalContext::read_u32(&self.sample_buf) as usize)
        } else {
            None
        }
    }

    pub fn forward_single_token_sample(
        &mut self,
        token_id: usize,
        _temperature: f32,
        _min_p: f32,
        _seed: u32,
    ) -> usize {
        self.forward_one(token_id, true).expect("sample")
    }

    pub fn forward_prefill_sample_last(
        &mut self,
        token_ids: &[usize],
        temperature: f32,
        min_p: f32,
        seed: u32,
    ) -> usize {
        assert!(!token_ids.is_empty());
        for &tid in &token_ids[..token_ids.len() - 1] {
            let _ = self.forward_one(tid, false);
        }
        self.forward_single_token_sample(*token_ids.last().unwrap(), temperature, min_p, seed)
    }
}
