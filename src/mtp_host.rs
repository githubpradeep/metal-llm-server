//! Host-side MTP draft backend (Accelerate/AMX trunk + host KV mirror).
//!
//! The draft head attends into the *target's* KV cache, which lives in Metal
//! buffers. Dispatching that attention per layer costs a GPU round-trip
//! (write Q, encode, wait, read) — four per draft step, which is what made the
//! first host offload slower than the fused Metal draft.
//!
//! The target's KV does not change while a draft chain runs (the drafter never
//! appends KV; only verify does). So the rows the draft needs are snapshotted
//! into host f32 once per chain and reused for every step and layer. Metal
//! buffers are `StorageModeShared`, so the snapshot is a plain memcpy+dequant
//! off unified memory with no command buffer and no fence.
//!
//! Enable with `MTP_BACKEND=ane` or `MTP_BACKEND=cpu` (aliases). Default remains
//! the full-Metal draft path (`MTP_BACKEND=metal`).

use crate::gemma4_config::KvCacheType;
use crate::gguf::Gguf;
use crate::gpu::{BufferView, MetalContext, f16_to_f32, weight_fmt};
use crate::quantize::QuantizedLinear;
use metal::*;

extern "C" {
    fn cblas_sgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32,
    );
}

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

/// Host f32 copy of the base KV rows the draft attends to, per base layer.
///
/// Rows are append-only and position-major per KV head, so any
/// `[kv_start, kv_start+eff)` window is contiguous and can feed BLAS directly.
struct KvMirror {
    /// `heads[kv_head]` = f32 rows, `valid_len * head_dim` long.
    heads: Vec<Vec<f32>>,
    head_dim: usize,
    valid_len: usize,
}

impl KvMirror {
    fn new(kv_heads: usize, head_dim: usize) -> Self {
        Self {
            heads: vec![Vec::new(); kv_heads.max(1)],
            head_dim,
            valid_len: 0,
        }
    }

    /// Bring the mirror up to `kv_seq` rows, dequantizing only new positions.
    ///
    /// A shorter `kv_seq` means the target rewound rejected drafts; those rows
    /// will be rewritten with different tokens, so they are dropped here and
    /// re-read on the next sync.
    fn sync(&mut self, cache: &Buffer, kv_seq: usize, capacity: usize, ty: KvCacheType) {
        if kv_seq < self.valid_len {
            for h in self.heads.iter_mut() {
                h.truncate(kv_seq * self.head_dim);
            }
            self.valid_len = kv_seq;
            return;
        }
        if kv_seq == self.valid_len {
            return;
        }

        let hd = self.head_dim;
        let bytes = unsafe {
            std::slice::from_raw_parts(cache.contents() as *const u8, cache.length() as usize)
        };
        for (kv_h, rows) in self.heads.iter_mut().enumerate() {
            rows.resize(kv_seq * hd, 0.0);
            for pos in self.valid_len..kv_seq {
                let dst = &mut rows[pos * hd..(pos + 1) * hd];
                decode_kv_row(bytes, kv_h, pos, capacity, hd, ty, dst);
            }
        }
        self.valid_len = kv_seq;
    }

    /// Contiguous `[start, start+len)` window of one KV head.
    fn window(&self, kv_head: usize, start: usize, len: usize) -> &[f32] {
        let hd = self.head_dim;
        &self.heads[kv_head][start * hd..(start + len) * hd]
    }
}

/// Dequantize one KV row (`kv_head`, `pos`) into `dst` (`head_dim` floats).
///
/// Layouts mirror the Metal decode kernels: rows are `[kv_head][capacity][...]`
/// with Q4_0/Q8_0 grouped in 32-value blocks.
fn decode_kv_row(
    bytes: &[u8],
    kv_head: usize,
    pos: usize,
    capacity: usize,
    head_dim: usize,
    ty: KvCacheType,
    dst: &mut [f32],
) {
    match ty {
        KvCacheType::F16 => {
            let base = (kv_head * capacity + pos) * head_dim * 2;
            for d in 0..head_dim {
                let o = base + d * 2;
                dst[d] = f16_to_f32(u16::from_le_bytes([bytes[o], bytes[o + 1]]));
            }
        }
        KvCacheType::Q8_0 => {
            let groups = head_dim / 32;
            let row_bytes = groups * 34;
            let base = kv_head * capacity * row_bytes + pos * row_bytes;
            for g in 0..groups {
                let o = base + g * 34;
                let scale = f16_to_f32(u16::from_le_bytes([bytes[o], bytes[o + 1]]));
                for e in 0..32 {
                    dst[g * 32 + e] = (bytes[o + 2 + e] as i8) as f32 * scale;
                }
            }
        }
        KvCacheType::Q4_0 => {
            let groups = head_dim / 32;
            let row_bytes = groups * 18;
            let base = kv_head * capacity * row_bytes + pos * row_bytes;
            for g in 0..groups {
                let o = base + g * 18;
                let scale = f16_to_f32(u16::from_le_bytes([bytes[o], bytes[o + 1]]));
                let qs = &bytes[o + 2..o + 18];
                for e in 0..16 {
                    dst[g * 32 + e] = ((qs[e] & 0xF) as i32 - 8) as f32 * scale;
                }
                for e in 16..32 {
                    dst[g * 32 + e] = ((qs[e - 16] >> 4) as i32 - 8) as f32 * scale;
                }
            }
        }
    }
}

/// GQA decode attention against mirrored f32 KV. `scores = K·q`, softmax,
/// `out = Vᵀ·p` — both matvecs on Accelerate, matching the Metal kernel's
/// unscaled (`scale=1.0`) draft contract.
fn host_attention(
    k_mirror: &KvMirror,
    v_mirror: &KvMirror,
    q: &[f32],
    out: &mut [f32],
    n_head: usize,
    head_dim: usize,
    kv_heads: usize,
    kv_start: usize,
    eff: usize,
    scores: &mut Vec<f32>,
) {
    let groups = (n_head / kv_heads.max(1)).max(1);
    scores.resize(eff, 0.0);
    for h in 0..n_head {
        let kv_h = (h / groups).min(kv_heads.saturating_sub(1));
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let k = k_mirror.window(kv_h, kv_start, eff);
        let v = v_mirror.window(kv_h, kv_start, eff);
        unsafe {
            cblas_sgemv(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                eff as i32,
                head_dim as i32,
                1.0,
                k.as_ptr(),
                head_dim as i32,
                qh.as_ptr(),
                1,
                0.0,
                scores.as_mut_ptr(),
                1,
            );
        }
        let mut max_s = f32::NEG_INFINITY;
        for &s in scores.iter() {
            if s > max_s {
                max_s = s;
            }
        }
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_s).exp();
            sum += *s;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for s in scores.iter_mut() {
            *s *= inv;
        }
        unsafe {
            cblas_sgemv(
                CBLAS_ROW_MAJOR,
                CBLAS_TRANS,
                eff as i32,
                head_dim as i32,
                1.0,
                v.as_ptr(),
                head_dim as i32,
                scores.as_ptr(),
                1,
                0.0,
                out[h * head_dim..].as_mut_ptr(),
                1,
            );
        }
    }
}

struct HostScratch {
    xh: Vec<f32>,
    h: Vec<f32>,
    a: Vec<f32>,
    q: Vec<f32>,
    araw: Vec<f32>,
    aproj: Vec<f32>,
    apost: Vec<f32>,
    aout: Vec<f32>,
    ffin: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    inter: Vec<f32>,
    fdown: Vec<f32>,
    fpost: Vec<f32>,
    fnorm: Vec<f32>,
}

/// Selected draft compute placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpBackendKind {
    /// Existing full-Metal `forward_draft_step`.
    Metal,
    /// Host Accelerate trunk + Metal attention into base KV.
    Host,
}

impl MtpBackendKind {
    pub fn from_env() -> Self {
        match std::env::var("MTP_BACKEND")
            .unwrap_or_else(|_| "metal".into())
            .to_ascii_lowercase()
            .as_str()
        {
            "ane" | "cpu" | "host" | "amx" => Self::Host,
            _ => Self::Metal,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Host => {
                "host (Accelerate/AMX trunk + mirrored KV attn, Metal lm_head; \
                 not CoreML ANE — measured slower, see tools/ane_mtp_probe.py)"
            }
        }
    }
}

pub struct MtpHostLayer {
    pub attn_norm: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub post_ffw_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub attn_q_norm: Vec<f32>,
    pub attn_q: QuantizedLinear,
    pub attn_output: QuantizedLinear,
    pub ffn_gate: QuantizedLinear,
    pub ffn_up: QuantizedLinear,
    pub ffn_down: QuantizedLinear,
    pub out_scale_val: f32,
    pub is_full: bool,
    pub n_rot: usize,
    pub mapped_base_layer: usize,
    pub base_kv_heads: usize,
    pub head_dim: usize,
    pub n_head: usize,
    pub ffn_inter: usize,
}

/// CPU-resident draft trunk + Metal for base-KV attention and vocab lm_head.
pub struct MtpHostHead {
    pub hidden_backbone: usize,
    pub hidden_head: usize,
    pub n_layers: usize,
    pub vocab: usize,
    pub full_rope_theta: f64,
    pub full_n_rot: usize,
    pub swa_rope_theta: f64,
    pub swa_n_rot: usize,
    pub sliding_window: u32,
    pub final_logit_softcapping: f32,
    pub rms_eps: f32,
    /// Vocab projection stays on Metal (262k×256 dominates host sgemv).
    pub tok_embd: BufferView,
    pub output_norm: Vec<f32>,
    pub pre_proj: QuantizedLinear,
    pub post_proj: BufferView,
    pub rope_freqs_data: Vec<f32>,
    pub layers: Vec<MtpHostLayer>,
    /// Last softcapped logits (for `p_min`).
    pub last_logits: Vec<f32>,
    scratch: HostScratch,
    /// Host KV snapshots keyed by base layer: `(K, V)`.
    kv_mirrors: std::collections::HashMap<usize, (KvMirror, KvMirror)>,
    /// Above this KV length the mirror is skipped and attention stays on Metal,
    /// bounding host memory for very long contexts.
    host_kv_max: usize,
    scores: Vec<f32>,
    q_buf: Buffer,
    araw_buf: Buffer,
    fnorm_buf: Buffer,
    logits_buf: Buffer,
    hnext_buf: Buffer,
    argmax_buf: Buffer,
}

impl MtpHostHead {
    pub fn load_from_gguf(ctx: &MetalContext, path: &str) -> Self {
        let g = Gguf::open(path);
        let tn = |i: usize, s: &str| format!("blk.{}.{}", i, s);

        let linear = |name: &str| -> QuantizedLinear {
            let info = g.tensor(name).unwrap_or_else(|| panic!("missing tensor {name}"));
            let k = info.ne0();
            let m = info.n_rows();
            let w = g.dequant_to_f32(name);
            assert_eq!(w.len(), m * k, "{name}: dequant len {} != {}x{}", w.len(), m, k);
            QuantizedLinear::from_f32(&w, m, k)
        };
        let f16_metal = |name: &str| -> BufferView {
            BufferView::from_buffer(ctx.buffer_from_slice_no_copy(g.tensor_raw(name)))
                .with_format(weight_fmt::F16)
        };
        let f32v = |name: &str| -> Vec<f32> { g.dequant_to_f32(name) };

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

        let swa_pattern: Vec<bool> = g
            .get_arr_bool("gemma4-assistant.attention.sliding_window_pattern")
            .map(|p| p.to_vec())
            .unwrap_or_else(|| (0..n_layers).map(|i| (i % 4) != 3).collect());

        let tok_embd = f16_metal("token_embd.weight");
        let output_norm = f32v("output_norm.weight");
        let pre_proj = linear("nextn.pre_projection.weight");
        let post_proj = f16_metal("nextn.post_projection.weight");

        let rope_freqs_data = {
            let mut rope_name: Option<String> = None;
            for candidate in std::iter::once("rope_freqs.weight".to_string())
                .chain((0..n_layers).map(|i| tn(i, "rope_freqs.weight")))
            {
                if g.tensor(&candidate).is_some() {
                    rope_name = Some(candidate);
                    break;
                }
            }
            rope_name.map(|n| f32v(&n)).unwrap_or_default()
        };

        let mut max_q = 0usize;
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let q_info = g.tensor(&tn(i, "attn_q.weight")).expect("attn_q");
            let head_dim = g
                .tensor(&tn(i, "attn_q_norm.weight"))
                .expect("attn_q_norm")
                .dims[0] as usize;
            let q_cols = q_info.n_rows();
            let n_head = q_cols / head_dim;
            // ggml ffn_down dims [ne0=ffn_inter, ne1=hidden]; gate rows = ffn_inter.
            let ffn_inter = g
                .tensor(&tn(i, "ffn_down.weight"))
                .expect("ffn_down")
                .dims[0] as usize;
            max_q = max_q.max(n_head * head_dim);

            let is_full = !swa_pattern[i];
            let out_scale_val = f32v(&tn(i, "layer_output_scale.weight"))[0];

            layers.push(MtpHostLayer {
                attn_norm: f32v(&tn(i, "attn_norm.weight")),
                post_attention_norm: f32v(&tn(i, "post_attention_norm.weight")),
                post_ffw_norm: f32v(&tn(i, "post_ffw_norm.weight")),
                ffn_norm: f32v(&tn(i, "ffn_norm.weight")),
                attn_q_norm: f32v(&tn(i, "attn_q_norm.weight")),
                attn_q: linear(&tn(i, "attn_q.weight")),
                attn_output: linear(&tn(i, "attn_output.weight")),
                ffn_gate: linear(&tn(i, "ffn_gate.weight")),
                ffn_up: linear(&tn(i, "ffn_up.weight")),
                ffn_down: linear(&tn(i, "ffn_down.weight")),
                out_scale_val,
                is_full,
                n_rot: if is_full {
                    full_n_rot.min(head_dim)
                } else {
                    swa_n_rot.min(head_dim)
                },
                mapped_base_layer: if is_full { 34 } else { 33 },
                base_kv_heads: 1,
                head_dim,
                n_head,
                ffn_inter,
            });
        }

        let max_ffn = layers.iter().map(|l| l.ffn_inter).max().unwrap_or(1);
        let bytes_f32 = |n: usize| -> usize { n.max(1) };
        eprintln!(
            "  [MTP] SWA pattern (true=sliding): {:?}, rms_eps={}, logit_softcap={}",
            swa_pattern, rms_eps, final_logit_softcapping
        );
        eprintln!(
            "  [MTP] Host offload: trunk F16→F32 on Accelerate/AMX; lm_head+post_proj+attn on Metal"
        );

        Self {
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
            rope_freqs_data,
            layers,
            last_logits: vec![0.0; vocab],
            scratch: HostScratch {
                xh: vec![0.0; hidden_backbone * 2],
                h: vec![0.0; hidden_head],
                a: vec![0.0; hidden_head],
                q: vec![0.0; max_q],
                araw: vec![0.0; max_q],
                aproj: vec![0.0; hidden_head],
                apost: vec![0.0; hidden_head],
                aout: vec![0.0; hidden_head],
                ffin: vec![0.0; hidden_head],
                gate: vec![0.0; max_ffn],
                up: vec![0.0; max_ffn],
                inter: vec![0.0; max_ffn],
                fdown: vec![0.0; hidden_head],
                fpost: vec![0.0; hidden_head],
                fnorm: vec![0.0; hidden_head],
            },
            kv_mirrors: std::collections::HashMap::new(),
            host_kv_max: std::env::var("MTP_HOST_KV_MAX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(32768),
            scores: Vec::new(),
            q_buf: ctx.buffer_empty(bytes_f32(max_q)),
            araw_buf: ctx.buffer_empty(bytes_f32(max_q)),
            fnorm_buf: ctx.buffer_empty(bytes_f32(hidden_head)),
            logits_buf: ctx.buffer_empty(bytes_f32(vocab)),
            hnext_buf: ctx.buffer_empty(bytes_f32(hidden_backbone)),
            argmax_buf: ctx.buffer_empty_u32(1),
        }
    }

    pub fn forward_draft_step(
        &mut self,
        ctx: &MetalContext,
        token_embedding: &[f32],
        embd_nextn: &[f32],
        pos: u32,
        base_k_cache: &[Buffer],
        base_v_cache: &[Buffer],
        base_kv_seq: u32,
        base_kv_capacity: u32,
        kv_cache_type: KvCacheType,
    ) -> (u32, Vec<f32>) {
        let eps = self.rms_eps;
        let hh = self.hidden_head;
        let softcap = self.final_logit_softcapping;
        let skip_attn = std::env::var("DRAFT_NO_ATTN").is_ok();

        {
            let xh = &mut self.scratch.xh;
            xh[..self.hidden_backbone].copy_from_slice(token_embedding);
            xh[self.hidden_backbone..].copy_from_slice(embd_nextn);
        }
        self.pre_proj.forward_vec(&self.scratch.xh, &mut self.scratch.h);

        for li in 0..self.layers.len() {
            let hd = self.layers[li].head_dim;
            let nh = self.layers[li].n_head;
            let qelems = nh * hd;
            let fi = self.layers[li].ffn_inter;
            let mapped = self.layers[li].mapped_base_layer;
            let base_kv_heads = self.layers[li].base_kv_heads;
            let is_full = self.layers[li].is_full;
            let n_rot = self.layers[li].n_rot;
            let out_scale = self.layers[li].out_scale_val;

            rmsnorm(
                &self.scratch.h,
                &self.layers[li].attn_norm,
                &mut self.scratch.a,
                eps,
            );
            self.layers[li]
                .attn_q
                .forward_vec(&self.scratch.a, &mut self.scratch.q[..qelems]);
            rmsnorm_per_head_inplace(
                &mut self.scratch.q[..qelems],
                &self.layers[li].attn_q_norm,
                nh,
                hd,
                eps,
            );

            let theta = if is_full {
                self.full_rope_theta
            } else {
                self.swa_rope_theta
            };
            let (cos, sin) = rope_tables(
                pos,
                hd,
                n_rot,
                theta,
                is_full,
                &self.rope_freqs_data,
            );
            apply_rotary_q(&mut self.scratch.q[..qelems], &cos, &sin, nh, hd);

            let (eff, kv_start) = if !is_full {
                let eff = base_kv_seq.min(self.sliding_window);
                let ks = if base_kv_seq > self.sliding_window {
                    base_kv_seq - self.sliding_window
                } else {
                    0
                };
                (eff, ks)
            } else {
                (base_kv_seq, 0)
            };

            if skip_attn {
                self.scratch.araw[..qelems].fill(0.0);
            } else if let Some((k_mirror, v_mirror)) = self.kv_mirrors.get(&mapped) {
                host_attention(
                    k_mirror,
                    v_mirror,
                    &self.scratch.q[..qelems],
                    &mut self.scratch.araw[..qelems],
                    nh,
                    hd,
                    base_kv_heads,
                    kv_start as usize,
                    eff as usize,
                    &mut self.scores,
                );
            } else {
                metal_attention(
                    ctx,
                    &self.q_buf,
                    &self.araw_buf,
                    &self.scratch.q[..qelems],
                    &mut self.scratch.araw[..qelems],
                    &base_k_cache[mapped],
                    &base_v_cache[mapped],
                    nh as u32,
                    base_kv_heads as u32,
                    hd as u32,
                    eff,
                    base_kv_capacity,
                    kv_start,
                    kv_cache_type,
                );
            }

            self.layers[li]
                .attn_output
                .forward_vec(&self.scratch.araw[..qelems], &mut self.scratch.aproj);
            rmsnorm(
                &self.scratch.aproj,
                &self.layers[li].post_attention_norm,
                &mut self.scratch.apost,
                eps,
            );
            for i in 0..hh {
                self.scratch.aout[i] = self.scratch.apost[i] + self.scratch.h[i];
            }

            rmsnorm(
                &self.scratch.aout,
                &self.layers[li].ffn_norm,
                &mut self.scratch.ffin,
                eps,
            );
            self.layers[li]
                .ffn_gate
                .forward_vec(&self.scratch.ffin, &mut self.scratch.gate[..fi]);
            self.layers[li]
                .ffn_up
                .forward_vec(&self.scratch.ffin, &mut self.scratch.up[..fi]);
            for i in 0..fi {
                self.scratch.inter[i] =
                    gelu_pytorch_tanh(self.scratch.gate[i]) * self.scratch.up[i];
            }
            self.layers[li]
                .ffn_down
                .forward_vec(&self.scratch.inter[..fi], &mut self.scratch.fdown);
            rmsnorm(
                &self.scratch.fdown,
                &self.layers[li].post_ffw_norm,
                &mut self.scratch.fpost,
                eps,
            );
            for i in 0..hh {
                self.scratch.h[i] = (self.scratch.fpost[i] + self.scratch.aout[i]) * out_scale;
            }
        }

        rmsnorm(
            &self.scratch.h,
            &self.output_norm,
            &mut self.scratch.fnorm,
            eps,
        );

        // Vocab lm_head + post_proj + softcap/argmax on Metal (one CB).
        MetalContext::write_buffer(&self.fnorm_buf, &self.scratch.fnorm);
        let command_buffer = ctx.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        ctx.encode_matvec_f16_view(
            encoder,
            &self.tok_embd,
            &self.fnorm_buf,
            &self.logits_buf,
            self.vocab as u32,
            hh as u32,
        );
        ctx.encode_matvec_f16_view(
            encoder,
            &self.post_proj,
            &self.fnorm_buf,
            &self.hnext_buf,
            self.hidden_backbone as u32,
            hh as u32,
        );
        if softcap > 0.0 {
            ctx.encode_softcap_argmax_rows_f32(
                encoder,
                &self.logits_buf,
                &self.argmax_buf,
                1,
                self.vocab as u32,
                softcap,
            );
        } else {
            ctx.encode_argmax_f32(encoder, &self.logits_buf, &self.argmax_buf, self.vocab as u32);
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let best = MetalContext::read_u32_buffer(&self.argmax_buf, 1)[0] as usize;
        let hnext = MetalContext::read_buffer(&self.hnext_buf, self.hidden_backbone);
        (best as u32, hnext)
    }

    /// Materialize softcapped logits for `p_min` (GPU→CPU). Call only when needed.
    pub fn sync_last_logits(&mut self) {
        self.last_logits = MetalContext::read_buffer(&self.logits_buf, self.vocab);
    }

    /// Snapshot the base KV rows this draft chain will read.
    ///
    /// Called once per chain: the target's KV is fixed until the next verify, so
    /// every draft step and layer reuses this copy instead of dispatching Metal
    /// attention. Only positions appended since the last snapshot are decoded.
    pub fn sync_kv_mirror(
        &mut self,
        base_k_cache: &[Buffer],
        base_v_cache: &[Buffer],
        base_kv_seq: u32,
        base_kv_capacity: u32,
        kv_cache_type: KvCacheType,
    ) {
        let kv_seq = base_kv_seq as usize;
        if kv_seq > self.host_kv_max {
            self.kv_mirrors.clear();
            return;
        }
        let capacity = base_kv_capacity as usize;
        for layer in self.layers.iter() {
            let l = layer.mapped_base_layer;
            let entry = self.kv_mirrors.entry(l).or_insert_with(|| {
                (
                    KvMirror::new(layer.base_kv_heads, layer.head_dim),
                    KvMirror::new(layer.base_kv_heads, layer.head_dim),
                )
            });
            entry
                .0
                .sync(&base_k_cache[l], kv_seq, capacity, kv_cache_type);
            entry
                .1
                .sync(&base_v_cache[l], kv_seq, capacity, kv_cache_type);
        }
    }
}

fn rmsnorm(x: &[f32], weight: &[f32], out: &mut [f32], eps: f32) {
    let n = x.len();
    debug_assert_eq!(weight.len(), n);
    debug_assert_eq!(out.len(), n);
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let inv = (ss / n as f32 + eps).sqrt().recip();
    for i in 0..n {
        out[i] = x[i] * inv * weight[i];
    }
}

fn rmsnorm_per_head_inplace(x: &mut [f32], weight: &[f32], n_head: usize, head_dim: usize, eps: f32) {
    debug_assert_eq!(weight.len(), head_dim);
    for h in 0..n_head {
        let base = h * head_dim;
        let mut ss = 0.0f32;
        for i in 0..head_dim {
            let v = x[base + i];
            ss += v * v;
        }
        let inv = (ss / head_dim as f32 + eps).sqrt().recip();
        for i in 0..head_dim {
            x[base + i] *= inv * weight[i];
        }
    }
}

fn gelu_pytorch_tanh(x: f32) -> f32 {
    let mut inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    inner = inner.clamp(-10.0, 10.0);
    0.5 * x * (1.0 + inner.tanh())
}

fn rope_tables(
    pos: u32,
    head_dim: usize,
    n_rot: usize,
    theta: f64,
    is_full: bool,
    rope_freqs_data: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let n_rot = n_rot.max(1);
    let n_rot_half = n_rot / 2;
    let mut cos = vec![0.0f32; half];
    let mut sin = vec![0.0f32; half];
    for i in 0..half {
        if i < n_rot_half {
            let base_inv = 1.0 / (theta.powf(i as f64 * 2.0 / n_rot as f64) as f32);
            let inv_freq = if is_full {
                base_inv / rope_freqs_data.get(i).copied().unwrap_or(1.0)
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
    (cos, sin)
}

fn apply_rotary_q(q: &mut [f32], cos: &[f32], sin: &[f32], n_head: usize, head_dim: usize) {
    let half = head_dim / 2;
    for h in 0..n_head {
        let base = h * head_dim;
        for d in 0..half {
            let q1 = q[base + d];
            let q2 = q[base + d + half];
            let c = cos[d];
            let s = sin[d];
            q[base + d] = q1 * c - q2 * s;
            q[base + d + half] = q2 * c + q1 * s;
        }
    }
}

fn metal_attention(
    ctx: &MetalContext,
    q_buf: &Buffer,
    araw_buf: &Buffer,
    q: &[f32],
    araw: &mut [f32],
    k_cache: &Buffer,
    v_cache: &Buffer,
    nh: u32,
    base_kv_heads: u32,
    hd: u32,
    eff: u32,
    capacity: u32,
    kv_start: u32,
    kv_cache_type: KvCacheType,
) {
    MetalContext::write_buffer(q_buf, q);
    let n_kv = base_kv_heads.max(1).min(nh);
    let n_groups = (nh / n_kv).max(1);
    let command_buffer = ctx.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    match kv_cache_type {
        KvCacheType::F16 => {
            ctx.encode_attention_with_offset_f16(
                encoder, q_buf, k_cache, v_cache, araw_buf, nh, n_kv, n_groups, hd, eff,
                capacity, 1.0, kv_start,
            );
        }
        KvCacheType::Q8_0 => {
            let groups_per_row = hd / 32;
            let row_bytes_q8 = groups_per_row * 34;
            ctx.encode_attention_with_offset_q8_0(
                encoder, q_buf, k_cache, v_cache, araw_buf, nh, n_kv, n_groups, hd, eff,
                capacity, 1.0, kv_start, groups_per_row, row_bytes_q8,
            );
        }
        KvCacheType::Q4_0 => {
            let groups_per_row = hd / 32;
            let row_bytes = groups_per_row * 18;
            ctx.encode_attention_with_offset_q4_0(
                encoder, q_buf, k_cache, v_cache, araw_buf, nh, n_kv, n_groups, hd, eff,
                capacity, 1.0, kv_start, groups_per_row, row_bytes,
            );
        }
    }
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let out = MetalContext::read_buffer(araw_buf, araw.len());
    araw.copy_from_slice(&out);
}
