use metal::*;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use memmap2::Mmap;
use safetensors::SafeTensors;
use serde::Deserialize;

use crate::gemma4_config::{Gemma4TextConfig, KvCacheType, RopeConfig, RopeParameters};
use crate::gpu::{BufferView, GpuTimestampProfiler, MetalContext, ProfileAblate, profile_gpu_enabled};
use crate::kv_pool::{KvCachePool, KvPoolError, KvSlot, KvSlotView, TqPoolConfig};

/// Rejected only for paths that still cannot apply TQ consistently (e.g. affine).
const TURBOQUANT_UNSUPPORTED: &str = "TurboQuant KV cache is not supported on this path. \
Use LLAMA_KV_CACHE_TYPE=q4_0, or V3 TurboQuant with TURBOQUANT_HOT_WINDOW>0.";

/// TurboQuant scratch + optional Q4 hot-ring caches (hybrid fast path).
struct TurboQuantState {
    turboquant: Option<crate::turboquant::TurboQuant>,
    tq_q_rot: Buffer,
    tq_k_rot: Buffer,
    tq_v_rot: Buffer,
    tq_out: Buffer,
    tq_scores: Buffer,
    tq_rw_k: Vec<Buffer>,
    tq_rw_v: Vec<Buffer>,
    tq_rw: u32,
    /// Model-frame Q4_0 ring for the most recent `tq_hot_w` tokens. While
    /// `attn_kv_seq <= tq_hot_w`, decode/prefill attend with the fast Q4 flash
    /// path; TQ V3 still receives a dual-write so cold attention stays valid
    /// after the window fills.
    tq_hot_k: Vec<Buffer>,
    tq_hot_v: Vec<Buffer>,
    tq_hot_w: u32,
}

/// Build TurboQuant rotation state (matrices + rotation scratch buffers).
///
/// Returns `None` rotation state for non-TurboQuant cache types, but always
/// allocates the (tiny) scratch buffers so the struct fields are always valid.
fn build_turboquant_state(
    ctx: &MetalContext,
    config: &Gemma4TextConfig,
    kv_cache_type: KvCacheType,
    kv_capacity: u32,
) -> TurboQuantState {
    let max_head_dim = config.head_dim.max(config.global_head_dim);
    let n_heads = config.num_attention_heads;
    let n_kv_heads = config.num_key_value_heads;
    let mk = |elems: usize| -> Buffer {
        ctx.device.new_buffer(
            (elems.max(1) * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let tq_q_rot = mk(n_heads * max_head_dim);
    let tq_out = mk(n_heads * max_head_dim);
    let tq_k_rot = mk(n_kv_heads * max_head_dim);
    let tq_v_rot = mk(n_kv_heads * max_head_dim);
    // Device scores for turboquant_attn_v3: [num_heads, capacity] — avoids
    // overflowing threadgroup memory at long context (8–16k).
    let tq_scores = mk(n_heads * kv_capacity as usize);

    let mut tq_rw_k: Vec<Buffer> = Vec::new();
    let mut tq_rw_v: Vec<Buffer> = Vec::new();
    let mut tq_rw: u32 = 0;
    let mut tq_hot_k: Vec<Buffer> = Vec::new();
    let mut tq_hot_v: Vec<Buffer> = Vec::new();
    let mut tq_hot_w: u32 = 0;

    let turboquant = if let KvCacheType::TurboQuant { k_bits, v_bits } = kv_cache_type {
        let mut dims: Vec<usize> = vec![config.head_dim, config.global_head_dim];
        dims.sort_unstable();
        dims.dedup();
        let affine = kv_cache_type.tq_affine();
        let variant = if affine {
            "rotation + Q4_0"
        } else {
            "rotation + Lloyd-Max V3"
        };

        if !affine {
            tq_rw = std::env::var("TURBOQUANT_RESIDUAL_WINDOW")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(128);
            tq_rw = tq_rw.min(kv_capacity);
            if tq_rw > 0 {
                let window_elems = n_kv_heads * tq_rw as usize * max_head_dim;
                for _ in 0..config.num_hidden_layers {
                    tq_rw_k.push(mk(window_elems));
                    tq_rw_v.push(mk(window_elems));
                }
                let mb = (config.num_hidden_layers * window_elems * 4 * 2) as f64 / 1.0e6;
                println!(
                    "  TurboQuant: residual window {} tokens (fp32 K/V, ~{:.0} MB)",
                    tq_rw, mb
                );
            }

            // Hybrid hot ring: default 2048 so short/medium gens stay on Q4 flash.
            // Set TURBOQUANT_HOT_WINDOW=0 to force pure TQ attention always.
            tq_hot_w = std::env::var("TURBOQUANT_HOT_WINDOW")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(2048);
            tq_hot_w = tq_hot_w.min(kv_capacity);
            if tq_hot_w > 0 {
                for i in 0..config.num_hidden_layers {
                    let hd = config.layer_head_dim(i);
                    let layer_n_kv = config.layer_num_kv_heads(i);
                    let q4_row = (hd / 32) * 18;
                    let bytes =
                        (layer_n_kv * tq_hot_w as usize * q4_row).max(1) as u64;
                    tq_hot_k.push(
                        ctx.device
                            .new_buffer(bytes, MTLResourceOptions::StorageModeShared),
                    );
                    tq_hot_v.push(
                        ctx.device
                            .new_buffer(bytes, MTLResourceOptions::StorageModeShared),
                    );
                }
                let hot_bytes: u64 = tq_hot_k.iter().map(|b| b.length()).sum::<u64>()
                    + tq_hot_v.iter().map(|b| b.length()).sum::<u64>();
                println!(
                    "  TurboQuant: Q4 hot window {} tokens (~{:.1} MB) — fast path while ctx ≤ {}",
                    tq_hot_w,
                    hot_bytes as f64 / 1e6,
                    tq_hot_w
                );
            }
        }

        println!(
            "  TurboQuant: K{}/V{} ({}), Haar rotations + codebooks for head_dims {:?}",
            k_bits, v_bits, variant, dims
        );
        Some(crate::turboquant::TurboQuant::new(
            &ctx.device, &dims, k_bits, v_bits,
        ))
    } else {
        None
    };

    TurboQuantState {
        turboquant,
        tq_q_rot,
        tq_k_rot,
        tq_v_rot,
        tq_out,
        tq_scores,
        tq_rw_k,
        tq_rw_v,
        tq_rw,
        tq_hot_k,
        tq_hot_v,
        tq_hot_w,
    }
}

/// Fused 3→1 operation: projection + norm + residual add
/// 
/// This combines three operations into one:
/// 1. Apply projection to input
/// 2. Apply RMSNorm to projected output
/// 3. Add residual connection
/// 
/// Result: normalized_residual_output
fn encode_proj_norm_residual(
    ctx: &MetalContext,
    encoder: &metal::ComputeCommandEncoderRef,
    input_buf: &metal::Buffer,
    projection_weight: &metal::Buffer,
    norm_weight: &BufferView,
    output_buf: &metal::Buffer,
    hidden_size: u32,
    eps: f32,
) {
    if crate::gpu::fused_rmsnorm_acc_enabled() {
        ctx.encode_rmsnorm_acc_view(
            encoder,
            input_buf,
            projection_weight,
            norm_weight,
            hidden_size,
            eps,
        );
    } else {
        ctx.encode_matvec_auto_view(
            encoder,
            &BufferView::from_buffer(projection_weight.clone()),
            input_buf,
            output_buf,
            hidden_size,
            hidden_size,
        );
        ctx.encode_rmsnorm_view(
            encoder,
            output_buf,
            norm_weight,
            output_buf,
            hidden_size,
            eps,
        );
        ctx.encode_vec_add(
            encoder,
            input_buf,
            output_buf,
            output_buf,
            hidden_size,
        );
    }
}

const DEFAULT_MAX_PREFILL_SEQ: usize = 1024;
const DEFAULT_MAX_DECODE_BATCH: usize = 8;

/// Cumulative per-phase GPU time (ms) and token count for PROFILE_PHASES.
#[derive(Clone, Copy)]
struct PhaseState {
    sums: [f64; 4],
    count: u64,
}

thread_local! {
    static PHASE_STATE: std::cell::Cell<PhaseState> =
        std::cell::Cell::new(PhaseState { sums: [0.0; 4], count: 0 });
}
/// Metal `setBuffer` offset alignment on Apple GPUs.
const WEIGHT_BLOB_ALIGN: usize = 256;
const WEIGHT_CACHE_MAGIC_LEN: usize = 4;

/// GPU RoPE params for one layer (must match `RopeLayerParams` in llama.metal).
#[repr(C)]
struct RopeLayerParams {
    theta: f32,
    factor: f32,
    head_dim: u32,
    rope_angles: u32,
}

fn build_rope_layer_params(config: &Gemma4TextConfig, num_layers: usize) -> Vec<RopeLayerParams> {
    let mut params = Vec::with_capacity(num_layers);
    for layer_idx in 0..num_layers {
        let head_dim = config.layer_head_dim(layer_idx) as u32;
        let is_full = config.is_full_attention(layer_idx);
        let theta = if is_full {
            config.full_rope_theta() as f32
        } else {
            config.sliding_rope_theta() as f32
        };
        let factor = if is_full {
            config.full_rope_factor() as f32
        } else {
            config.sliding_rope_factor() as f32
        };
        let rotary_dim = if is_full {
            (head_dim as f64 * config.full_partial_rotary_factor()) as u32
        } else {
            head_dim
        };
        params.push(RopeLayerParams {
            theta,
            factor,
            head_dim,
            rope_angles: rotary_dim / 2,
        });
    }
    params
}

fn alloc_decode_rope_buffers(
    ctx: &MetalContext,
    config: &Gemma4TextConfig,
    num_layers: usize,
    max_head_dim: usize,
) -> (Buffer, Buffer, Buffer) {
    let params = build_rope_layer_params(config, num_layers);
    let cos = ctx.buffer_empty(num_layers * max_head_dim);
    let sin = ctx.buffer_empty(num_layers * max_head_dim);
    let params_bytes =
        unsafe { std::slice::from_raw_parts(params.as_ptr() as *const u8, params.len() * 16) };
    let params_buf = ctx.buffer_from_bytes(params_bytes);
    (cos, sin, params_buf)
}

fn weight_section_pad(section_offset: usize) -> usize {
    (WEIGHT_BLOB_ALIGN - (section_offset % WEIGHT_BLOB_ALIGN)) % WEIGHT_BLOB_ALIGN
}

fn pad_weights_file_to_section_align(file: &mut fs::File) {
    use std::io::{Seek, Write};
    let pos = file.stream_position().expect("stream position") as usize;
    let pad = weight_section_pad(pos - WEIGHT_CACHE_MAGIC_LEN);
    if pad > 0 {
        file.write_all(&vec![0u8; pad]).expect("write padding");
    }
}

fn configured_max_prefill_seq(kv_capacity: u32) -> usize {
    std::env::var("LLAMA_MAX_PREFILL_SEQ")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_MAX_PREFILL_SEQ)
        .min(kv_capacity as usize)
}

/// KV cache slot capacity (context window). Override with `LLAMA_CTX_SIZE`.
///
/// Default stays 16k for memory; set `LLAMA_CTX_SIZE=200000` for long-context
/// agent workloads. Values above the model's trained `max_position_embeddings`
/// are allowed (up to 200k) but may degrade RoPE quality.
fn configured_kv_capacity(max_position_embeddings: usize) -> u32 {
    const DEFAULT_KV_CAPACITY: usize = 16384;
    const ABSOLUTE_MAX_KV_CAPACITY: usize = 200_000;
    let requested = std::env::var("LLAMA_CTX_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_KV_CAPACITY);
    let capped = requested.clamp(256, ABSOLUTE_MAX_KV_CAPACITY);
    if capped > max_position_embeddings {
        eprintln!(
            "  Warning: LLAMA_CTX_SIZE={} > model max_position_embeddings={} (quality may drop past training length)",
            capped, max_position_embeddings
        );
    }
    capped as u32
}

/// Gemma 4 E4B GPU-resident model with persistent KV cache on Metal.
/// All operations for one token are encoded into a SINGLE command buffer.
/// Result of a single decode forward pass: either the full (softcapped) logit
/// vector (CPU sampling path), the GPU-sampled token id (fused fast path), or
/// KV-only advance (prefill intermediate tokens).
enum DecodeOutput {
    Advanced,
    Logits(Vec<f32>),
    Token(usize),
}

enum DecodeMode {
    /// Update KV cache only; skip final norm, lm_head, and logit readback.
    Advance,
    Logits,
    Sample(f32, f32, u32),
}

pub struct Gemma4GpuModel {
    pub ctx: MetalContext,
    pub config: Gemma4TextConfig,

    // Embedding tables on CPU (mmap'd from cache for instant load; lm_head is separate Q4 on GPU)
    embed_tables: EmbedTables,

    // LM head (tied to embed_tokens, stored as Q4 Metal buffer for GPU matvec)
    pub lm_head_buf: BufferView,

    // Per-layer weights on GPU
    pub layers: Vec<Gemma4GpuLayer>,

    // Shared weights
    pub final_norm_weight: BufferView,
    pub per_layer_projection_norm_weight: BufferView,
    pub per_layer_model_projection_weight: BufferView, // [num_layers * ple_dim, hidden_size] f16/Q4

    // Pre-allocated scratch buffers (reused every token)
    pub hidden_buf: Buffer,
    pub normed_buf: Buffer,
    pub residual_buf: Buffer,
    pub q_buf: Buffer,
    pub k_buf: Buffer,
    pub v_buf: Buffer,
    pub attn_out_buf: Buffer,
    pub o_out_buf: Buffer,
    pub gate_buf: Buffer,
    pub up_buf: Buffer,
    pub gelu_buf: Buffer,
    pub down_buf: Buffer,
    pub logits_buf: Buffer,
    pub sample_out_buf: Buffer, // [1] u32: GPU-sampled token id (decode fast path)
    pub inv_rms_buf: Buffer,    // [1] f32: scratch for fused pre-FF RMSNorm + MLP
    pub prefill_scratch: PrefillScratch,
    pub decode_batch_scratch: DecodeBatchScratch,

    // MTP verify buffers
    pub mtp_verify_logits_buf: Buffer,
    pub mtp_verify_argmax_buf: Buffer,
    pub mtp_verify_hidden_buf: Buffer,
    pub mtp_verify_scratch: MtpVerifyScratch,

    // PLE scratch buffers
    pub ple_embed_buf: Buffer, // [ple_dim] = 256 (unused now, kept for compat)
    pub ple_gated_buf: Buffer, // [ple_dim]
    pub ple_normed_buf: Buffer, // [ple_dim]
    pub ple_projected_buf: Buffer, // [hidden_size]
    pub ple_context_proj_buf: Buffer, // [num_layers * ple_dim] for context projection
    pub ple_token_id_buf: Buffer, // [num_layers * ple_dim] token identity embedding
    pub ple_combined_buf: Buffer, // [num_layers * ple_dim] combined output

    // QK norm scratch
    pub q_normed_buf: Buffer,
    pub k_normed_buf: Buffer,
    pub ggml_fa_tmp_buf: Buffer,

    // GPU-resident KV cache per layer
    pub k_cache: Vec<Buffer>,
    pub v_cache: Vec<Buffer>,
    pub kv_seq_len: u32,
    pub kv_capacity: u32,
    pub kv_cache_type: KvCacheType,

    // TurboQuant: per-head_dim Haar rotation matrices + rotation scratch buffers.
    // `Some` only when `kv_cache_type == KvCacheType::TurboQuant`.
    pub(crate) turboquant: Option<crate::turboquant::TurboQuant>,
    pub(crate) tq_q_rot: Buffer, // [num_heads * max_head_dim] rotated query
    pub(crate) tq_k_rot: Buffer, // [num_kv_heads * max_head_dim] rotated key
    pub(crate) tq_v_rot: Buffer, // [num_kv_heads * max_head_dim] rotated value
    pub(crate) tq_out: Buffer,   // [num_heads * max_head_dim] un-rotated attention output
    pub(crate) tq_scores: Buffer, // [num_heads * capacity] device softmax scores for V3 attn
    // Residual window (V3 2/3-bit only): most recent `tq_rw` tokens' rotated K/V
    pub(crate) tq_rw_k: Vec<Buffer>,
    pub(crate) tq_rw_v: Vec<Buffer>,
    pub(crate) tq_rw: u32,
    // Hybrid Q4_0 hot ring (model frame); see TurboQuantState.
    pub(crate) tq_hot_k: Vec<Buffer>,
    pub(crate) tq_hot_v: Vec<Buffer>,
    pub(crate) tq_hot_w: u32,
    /// Set after the first spill of the Q4 hot ring into TQ cold (ctx > hot).
    pub(crate) tq_hot_spilled: bool,

    // Rotary precomputed buffers (per-layer since sliding/full differ)
    pub cos_buf: Buffer,
    pub sin_buf: Buffer,

    // Packed decode RoPE cos/sin (filled on GPU each token) + static layer params.
    pub decode_rope_cos_packed: Buffer,
    pub decode_rope_sin_packed: Buffer,
    pub rope_layer_params_buf: Buffer,
    pub rope_max_head_dim: usize,

    pub per_layer_prefill_cos_bufs: Vec<Buffer>,
    pub per_layer_prefill_sin_bufs: Vec<Buffer>,
    pub per_layer_decode_batch_cos_bufs: Vec<Buffer>,
    pub per_layer_decode_batch_sin_bufs: Vec<Buffer>,
    pub per_layer_decode_batch_append_pos_bufs: Vec<Buffer>,
    pub per_layer_decode_batch_kv_start_bufs: Vec<Buffer>,
    pub per_layer_decode_batch_kv_len_bufs: Vec<Buffer>,
    pub per_layer_ple_bufs: Vec<Buffer>,

    pub total_tokens: usize,

    // Legacy monolithic cache: offset into embed mmap for fast weights save during migration.
    weights_mmap_offset: Option<usize>,
    embed_decode_scratch: Vec<f32>,
    ple_decode_scratch: Vec<f32>,
}

pub struct PrefillScratch {
    pub max_seq_len: usize,
    pub hidden_buf: Buffer,
    pub normed_buf: Buffer,
    pub residual_buf: Buffer,
    pub q_buf: Buffer,
    pub k_buf: Buffer,
    pub v_buf: Buffer,
    pub attn_out_buf: Buffer,
    pub o_out_buf: Buffer,
    pub gate_buf: Buffer,
    pub up_buf: Buffer,
    pub gelu_buf: Buffer,
    pub down_buf: Buffer,
    pub logits_buf: Buffer,
    pub ple_context_proj_buf: Buffer,
    pub ple_token_id_buf: Buffer,
    pub ple_combined_buf: Buffer,
    pub q_normed_buf: Buffer,
    pub k_normed_buf: Buffer,
    pub qkv_stacked_buf: Buffer,
    pub fa_ext_scratch: Buffer,
    pub fa_ext_layout: crate::ggml_flash_attn_ext::ScratchLayout,
}

pub struct DecodeBatchScratch {
    pub max_batch_size: usize,
    pub hidden_buf: Buffer,
    pub normed_buf: Buffer,
    pub residual_buf: Buffer,
    pub q_buf: Buffer,
    pub k_buf: Buffer,
    pub v_buf: Buffer,
    pub attn_out_buf: Buffer,
    pub o_out_buf: Buffer,
    pub gate_buf: Buffer,
    pub up_buf: Buffer,
    pub gelu_buf: Buffer,
    pub down_buf: Buffer,
    pub inv_rms_buf: Buffer,
    pub logits_buf: Buffer,
    pub ple_context_proj_buf: Buffer,
    pub ple_token_id_buf: Buffer,
    pub ple_combined_buf: Buffer,
    pub q_normed_buf: Buffer,
    pub k_normed_buf: Buffer,
    pub ggml_fa_tmp_buf: Buffer,
}

struct BatchedTokenInputs {
    batch_size: usize,
    hidden: Vec<f32>,
    ple_token_identity: Vec<f32>,
}

struct DecodeBatchRowOffsets {
    hidden: u64,
    q: u64,
    kv: u64,
    intermediate: u64,
    ple_row: u64,
}

struct PrefillRowOffsets {
    hidden: u64,
    q: u64,
    kv: u64,
    intermediate: u64,
    ple_row: u64,
}

#[derive(Clone, Copy)]
struct PrefillBatchSegment {
    slot: KvSlot,
    row_start: usize,
    token_count: usize,
    start_pos: usize,
}

impl DecodeBatchScratch {
    fn new(
        ctx: &MetalContext,
        max_batch_size: usize,
        hidden_size: usize,
        max_q_out: usize,
        max_kv_out: usize,
        intermediate_size: usize,
        vocab_size: usize,
        num_layers: usize,
        ple_dim: usize,
        num_heads: u32,
        max_head_dim: u32,
    ) -> Self {
        let ggml_fa_tmp_elems = (crate::ggml_flash_attn::flash_attn_tmp_bytes(
            num_heads,
            max_head_dim,
        ) / 4) as usize;
        Self {
            max_batch_size,
            hidden_buf: ctx.buffer_empty(max_batch_size * hidden_size),
            normed_buf: ctx.buffer_empty(max_batch_size * hidden_size),
            residual_buf: ctx.buffer_empty(max_batch_size * hidden_size),
            q_buf: ctx.buffer_empty(max_batch_size * max_q_out),
            k_buf: ctx.buffer_empty(max_batch_size * max_kv_out),
            v_buf: ctx.buffer_empty(max_batch_size * max_kv_out),
            attn_out_buf: ctx.buffer_empty(max_batch_size * max_q_out),
            o_out_buf: ctx.buffer_empty(max_batch_size * hidden_size),
            gate_buf: ctx.buffer_empty(max_batch_size * intermediate_size),
            up_buf: ctx.buffer_empty(max_batch_size * intermediate_size),
            gelu_buf: ctx.buffer_empty(max_batch_size * intermediate_size),
            down_buf: ctx.buffer_empty(max_batch_size * hidden_size),
            inv_rms_buf: ctx.buffer_empty(max_batch_size),
            logits_buf: ctx.buffer_empty(max_batch_size * vocab_size),
            ple_context_proj_buf: ctx.buffer_empty(max_batch_size * num_layers * ple_dim),
            ple_token_id_buf: ctx.buffer_empty(max_batch_size * num_layers * ple_dim),
            ple_combined_buf: ctx.buffer_empty(max_batch_size * num_layers * ple_dim),
            q_normed_buf: ctx.buffer_empty(max_batch_size * max_q_out),
            k_normed_buf: ctx.buffer_empty(max_batch_size * max_kv_out),
            ggml_fa_tmp_buf: ctx.buffer_empty(ggml_fa_tmp_elems),
        }
    }
}

/// Scratch buffers for MTP verify batch (parallel decode of draft tokens).
const MAX_MTP_VERIFY_SEQ: usize = 8;

pub(crate) struct MtpVerifyScratch {
    max_seq_len: usize,
    hidden_buf: Buffer,
    ple_token_id_buf: Buffer,
    cos_bufs: Vec<Buffer>,
    sin_bufs: Vec<Buffer>,
}

impl MtpVerifyScratch {
    fn new(
        ctx: &MetalContext,
        max_seq_len: usize,
        hidden_size: usize,
        ple_total_dim: usize,
        head_dims: &[usize],
    ) -> Self {
        Self {
            max_seq_len,
            hidden_buf: ctx.buffer_empty(max_seq_len * hidden_size),
            ple_token_id_buf: ctx.buffer_empty(max_seq_len * ple_total_dim),
            cos_bufs: head_dims
                .iter()
                .map(|&hd| ctx.buffer_empty(max_seq_len * hd))
                .collect(),
            sin_bufs: head_dims
                .iter()
                .map(|&hd| ctx.buffer_empty(max_seq_len * hd))
                .collect(),
        }
    }
}

impl PrefillScratch {
    fn new(
        ctx: &MetalContext,
        max_seq_len: usize,
        hidden_size: usize,
        max_q_out: usize,
        max_kv_out: usize,
        intermediate_size: usize,
        vocab_size: usize,
        num_layers: usize,
        ple_dim: usize,
        kv_capacity: u32,
        num_kv_heads: u32,
        max_head_dim: u32,
    ) -> Self {
        let row_bytes = ((max_head_dim / 32) * 18) as u64;
        let fa_ext_layout = crate::ggml_flash_attn_ext::scratch_layout(
            max_seq_len as u32,
            kv_capacity,
            num_kv_heads,
            row_bytes,
        );
        let fa_ext_elems = ((fa_ext_layout.total + 3) / 4) as usize;
        println!(
            "  fa_ext scratch: {:.1} MB (mask_kv≤{}, max_q={}, dual full/SWA planes)",
            fa_ext_layout.total as f64 / (1024.0 * 1024.0),
            fa_ext_layout.mask_kv_capacity,
            max_seq_len
        );
        Self {
            max_seq_len,
            hidden_buf: ctx.buffer_empty(max_seq_len * hidden_size),
            normed_buf: ctx.buffer_empty(max_seq_len * hidden_size),
            residual_buf: ctx.buffer_empty(max_seq_len * hidden_size),
            q_buf: ctx.buffer_empty(max_seq_len * max_q_out),
            k_buf: ctx.buffer_empty(max_seq_len * max_kv_out),
            v_buf: ctx.buffer_empty(max_seq_len * max_kv_out),
            attn_out_buf: ctx.buffer_empty(max_seq_len * max_q_out),
            o_out_buf: ctx.buffer_empty(max_seq_len * hidden_size),
            gate_buf: ctx.buffer_empty(2 * max_seq_len * intermediate_size),
            up_buf: ctx.buffer_empty(max_seq_len * intermediate_size),
            gelu_buf: ctx.buffer_empty(max_seq_len * intermediate_size),
            down_buf: ctx.buffer_empty(max_seq_len * hidden_size),
            logits_buf: ctx.buffer_empty(vocab_size),
            ple_context_proj_buf: ctx.buffer_empty(max_seq_len * num_layers * ple_dim),
            ple_token_id_buf: ctx.buffer_empty(max_seq_len * num_layers * ple_dim),
            ple_combined_buf: ctx.buffer_empty(max_seq_len * num_layers * ple_dim),
            q_normed_buf: ctx.buffer_empty(max_seq_len * max_q_out),
            k_normed_buf: ctx.buffer_empty(max_seq_len * max_kv_out),
            qkv_stacked_buf: ctx.buffer_empty(max_seq_len * (max_q_out + 2 * max_kv_out)),
            fa_ext_scratch: ctx.buffer_empty(fa_ext_elems),
            fa_ext_layout,
        }
    }
}

/// CPU-resident embedding tables. Cache load mmap's the file (instant); first
/// safetensors load owns bytes; direct GGUF load decodes rows on demand from the
/// native GGUF blocks (no up-front dequant/requant of the multi-GB tables).
struct EmbedTables {
    mmap: Option<Mmap>,
    embed_offset: usize,
    embed_byte_len: usize,
    ple_offset: usize,
    ple_byte_len: usize,
    ple_cols: usize,
    vocab_size: usize,
    owned_embed: Option<Vec<u8>>,
    owned_ple: Option<Vec<u8>>,
    // Native GGUF backing (direct GGUF load): decode rows on demand.
    gguf: Option<std::sync::Arc<crate::gguf::Gguf>>,
    embed_native: Option<(u32, usize, usize)>, // (ggml_type, row_stride, cols=hidden)
    ple_native: Option<(u32, usize, usize)>,    // (ggml_type, row_stride, cols=ple_total)
}

impl EmbedTables {
    fn from_mmap(
        mmap: Mmap,
        embed_offset: usize,
        embed_byte_len: usize,
        ple_offset: usize,
        ple_byte_len: usize,
        ple_cols: usize,
        vocab_size: usize,
    ) -> Self {
        Self {
            mmap: Some(mmap),
            embed_offset,
            embed_byte_len,
            ple_offset,
            ple_byte_len,
            ple_cols,
            vocab_size,
            owned_embed: None,
            owned_ple: None,
            gguf: None,
            embed_native: None,
            ple_native: None,
        }
    }

    fn from_owned(embed: Vec<u8>, ple: Vec<u8>, vocab_size: usize, ple_cols: usize, embed_cols: usize) -> Self {
        let (owned_embed, embed_byte_len) = Self::convert_embed_to_q4(embed, vocab_size, embed_cols);
        let (owned_ple, ple_byte_len) = Self::convert_ple_to_q4(ple, vocab_size, ple_cols);
        Self {
            mmap: None,
            embed_offset: 0,
            embed_byte_len,
            ple_offset: 0,
            ple_byte_len,
            ple_cols,
            vocab_size,
            owned_embed,
            owned_ple,
            gguf: None,
            embed_native: None,
            ple_native: None,
        }
    }

    /// Direct GGUF load: keep the GGUF mmap alive and decode embedding/PLE rows
    /// on demand from the native tensor blocks. Avoids the multi-GB CPU
    /// dequant→bf16→Q4_0 conversion entirely. `embed_cols == hidden_size`,
    /// `ple_total == num_layers * ple_dim`, and the contiguous dim (ne0) of each
    /// tensor is the row length.
    fn from_gguf(
        gguf: std::sync::Arc<crate::gguf::Gguf>,
        vocab_size: usize,
        embed_cols: usize,
        ple_total: usize,
    ) -> Self {
        let (embed_type, embed_row_stride, embed_actual) = {
            let e = gguf
                .tensor("token_embd.weight")
                .expect("token_embd.weight missing");
            (e.ggml_type, e.byte_len() / vocab_size, e.num_elements() / vocab_size)
        };
        assert_eq!(embed_actual, embed_cols, "token_embd row length mismatch");
        let ple_native = if ple_total > 0 {
            let p = gguf
                .tensor("per_layer_token_embd.weight")
                .expect("per_layer_token_embd.weight missing");
            assert_eq!(p.num_elements() / vocab_size, ple_total, "per_layer_token_embd row length mismatch");
            Some((p.ggml_type, p.byte_len() / vocab_size, ple_total))
        } else {
            None
        };
        Self {
            mmap: None,
            embed_offset: 0,
            embed_byte_len: 0,
            ple_offset: 0,
            ple_byte_len: 0,
            ple_cols: ple_total,
            vocab_size,
            owned_embed: None,
            owned_ple: None,
            gguf: Some(gguf),
            embed_native: Some((embed_type, embed_row_stride, embed_cols)),
            ple_native,
        }
    }

    /// Convert bf16 embed data to Q4_0 blocks. Each row of length `embed_cols` becomes
    /// `(embed_cols/32)*18` bytes of Q4_0 blocks.
    fn convert_embed_to_q4(bf16_data: Vec<u8>, vocab_size: usize, embed_cols: usize) -> (Option<Vec<u8>>, usize) {
        assert_eq!(bf16_data.len(), vocab_size * embed_cols * 2,
            "embed data size mismatch: got {} expected {}", bf16_data.len(), vocab_size * embed_cols * 2);
        assert_eq!(embed_cols % 32, 0, "embed cols not divisible by 32");

        let blocks_per_row = embed_cols / 32;
        let q4_row_bytes = blocks_per_row * 18;
        let mut q4_data = vec![0u8; vocab_size * q4_row_bytes];
        let mut f32_row = vec![0.0f32; embed_cols];
        for row in 0..vocab_size {
            let row_start = row * embed_cols * 2;
            for i in 0..embed_cols {
                let byte_off = row_start + i * 2;
                let raw = u16::from_le_bytes([bf16_data[byte_off], bf16_data[byte_off + 1]]);
                f32_row[i] = crate::gpu::bf16_to_f32(raw);
            }
            for g in 0..blocks_per_row {
                let group_start = g * 32;
                let mut max_abs = 0.0f32;
                for &v in f32_row[group_start..group_start + 32].iter() {
                    let a = v.abs();
                    if a > max_abs { max_abs = a; }
                }
                let d = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
                let inv_d = 1.0 / d;
                let d_f16 = crate::gpu::f32_to_f16(d);
                let out_off = row * q4_row_bytes + g * 18;
                q4_data[out_off] = (d_f16 & 0xFF) as u8;
                q4_data[out_off + 1] = (d_f16 >> 8) as u8;
                for i in 0..16 {
                    let v0 = f32_row[group_start + i];
                    let v1 = f32_row[group_start + i + 16];
                    let q0 = ((v0 * inv_d).round() as i32 + 8).clamp(0, 15) as u8;
                    let q1 = ((v1 * inv_d).round() as i32 + 8).clamp(0, 15) as u8;
                    q4_data[out_off + 2 + i] = q0 | (q1 << 4);
                }
            }
        }
        (Some(q4_data), vocab_size * q4_row_bytes)
    }

    /// Convert bf16 PLE data to Q4_0 blocks. Each row of length `ple_cols` becomes
    /// `(ple_cols/32)*18` bytes of Q4_0 blocks.
    fn convert_ple_to_q4(bf16_data: Vec<u8>, vocab_size: usize, ple_cols: usize) -> (Option<Vec<u8>>, usize) {
        assert_eq!(bf16_data.len(), vocab_size * ple_cols * 2,
            "PLE data size mismatch: got {} expected {}", bf16_data.len(), vocab_size * ple_cols * 2);
        assert_eq!(ple_cols % 32, 0, "PLE cols not divisible by 32");

        let blocks_per_row = ple_cols / 32;
        let q4_row_bytes = blocks_per_row * 18;
        let mut q4_data = vec![0u8; vocab_size * q4_row_bytes];

        // Process one row at a time to avoid large f32 temp allocations
        let mut f32_row = vec![0.0f32; ple_cols];
        for row in 0..vocab_size {
            let row_start = row * ple_cols * 2;
            for i in 0..ple_cols {
                let byte_off = row_start + i * 2;
                let raw = u16::from_le_bytes([bf16_data[byte_off], bf16_data[byte_off + 1]]);
                f32_row[i] = crate::gpu::bf16_to_f32(raw);
            }
            // Quantize row to Q4_0
            for g in 0..blocks_per_row {
                let group_start = g * 32;
                let mut max_abs = 0.0f32;
                for &v in f32_row[group_start..group_start + 32].iter() {
                    let a = v.abs();
                    if a > max_abs { max_abs = a; }
                }
                let d = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
                let inv_d = 1.0 / d;
                let d_f16 = crate::gpu::f32_to_f16(d);
                let out_off = row * q4_row_bytes + g * 18;
                q4_data[out_off] = (d_f16 & 0xFF) as u8;
                q4_data[out_off + 1] = (d_f16 >> 8) as u8;
                for i in 0..16 {
                    let v0 = f32_row[group_start + i];
                    let v1 = f32_row[group_start + i + 16];
                    let q0 = ((v0 * inv_d).round() as i32 + 8).clamp(0, 15) as u8;
                    let q1 = ((v1 * inv_d).round() as i32 + 8).clamp(0, 15) as u8;
                    q4_data[out_off + 2 + i] = q0 | (q1 << 4);
                }
            }
        }
        (Some(q4_data), vocab_size * q4_row_bytes)
    }

    fn embed_bytes(&self) -> &[u8] {
        if let Some(mmap) = &self.mmap {
            &mmap[self.embed_offset..self.embed_offset + self.embed_byte_len]
        } else {
            self.owned_embed.as_ref().unwrap()
        }
    }

    fn ple_bytes(&self) -> &[u8] {
        if let Some(mmap) = &self.mmap {
            &mmap[self.ple_offset..self.ple_offset + self.ple_byte_len]
        } else if let Some(owned) = &self.owned_ple {
            owned.as_slice()
        } else {
            &[]
        }
    }

    fn mmap_ref(&self) -> Option<&Mmap> {
        self.mmap.as_ref()
    }

    fn decode_embed_into(&self, token_id: usize, hidden_size: usize, out: &mut [f32]) {
        let scale = (hidden_size as f32).sqrt();
        self.decode_embed_into_impl(token_id, hidden_size, out, scale);
    }

    fn decode_embed_into_no_scale(&self, token_id: usize, hidden_size: usize, out: &mut [f32]) {
        self.decode_embed_into_impl(token_id, hidden_size, out, 1.0);
    }

    fn decode_embed_into_impl(&self, token_id: usize, hidden_size: usize, out: &mut [f32], scale: f32) {
        if let Some((gt, stride, cols)) = self.embed_native {
            let bytes = self
                .gguf
                .as_ref()
                .unwrap()
                .tensor_row_bytes("token_embd.weight", token_id, stride);
            crate::gguf::dequant_row_to_f32(gt, bytes, cols, out);
            for v in out.iter_mut() {
                *v *= scale;
            }
        } else {
            let bf16_row_bytes = hidden_size * 2;
            if self.embed_byte_len as u64 > self.vocab_size as u64 * bf16_row_bytes as u64 / 2 {
                // bf16 format
                let byte_start = token_id * bf16_row_bytes;
                let bytes = &self.embed_bytes()[byte_start..byte_start + bf16_row_bytes];
                for (i, chunk) in bytes.chunks_exact(2).enumerate() {
                    out[i] = crate::gpu::bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])) * scale;
                }
            } else {
                // Q4_0 format
                let blocks = hidden_size / 32;
                let q4_row_bytes = blocks * 18;
                let row_start = token_id * q4_row_bytes;
                let bytes = &self.embed_bytes()[row_start..row_start + q4_row_bytes];
                for g in 0..blocks {
                    let off = g * 18;
                    let raw_d = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
                    let d = crate::gpu::f16_to_f32(raw_d);
                    let out_off = g * 32;
                    for i in 0..16 {
                        let packed = bytes[off + 2 + i];
                        let q0 = packed & 0x0F;
                        let q1 = packed >> 4;
                        out[out_off + i] = d * ((q0 as i32 - 8) as f32) * scale;
                        out[out_off + i + 16] = d * ((q1 as i32 - 8) as f32) * scale;
                    }
                }
            }
        }
    }

    fn decode_ple_into(
        &self,
        token_id: usize,
        ple_total_dim: usize,
        ple_dim: usize,
        out: &mut [f32],
    ) {
        if ple_dim == 0 || ple_total_dim == 0 { return; }
        let scale = (ple_dim as f32).sqrt();
        if let Some((gt, stride, cols)) = self.ple_native {
            let bytes = self
                .gguf
                .as_ref()
                .unwrap()
                .tensor_row_bytes("per_layer_token_embd.weight", token_id, stride);
            crate::gguf::dequant_row_to_f32(gt, bytes, cols, out);
            for v in out.iter_mut() {
                *v *= scale;
            }
        } else if self.ple_byte_len as u64 > self.vocab_size as u64 * ple_total_dim as u64 {
            // Old format: bf16 (detected by large byte length)
            let cols = self.ple_cols;
            let byte_start = token_id * cols * 2;
            let bytes = &self.ple_bytes()[byte_start..byte_start + cols * 2];
            for (i, chunk) in bytes.chunks_exact(2).enumerate() {
                out[i] = crate::gpu::bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])) * scale;
            }
        } else {
            // Q4_0 format
            let blocks = ple_total_dim / 32;
            let q4_row_bytes = blocks * 18;
            let row_start = token_id * q4_row_bytes;
            let bytes = &self.ple_bytes()[row_start..row_start + q4_row_bytes];
            for g in 0..blocks {
                let off = g * 18;
                let raw_d = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
                let d = crate::gpu::f16_to_f32(raw_d);
                let out_off = g * 32;
                for i in 0..16 {
                    let packed = bytes[off + 2 + i];
                    let q0 = packed & 0x0F;
                    let q1 = packed >> 4;
                    out[out_off + i] = d * ((q0 as i32 - 8) as f32) * scale;
                    out[out_off + i + 16] = d * ((q1 as i32 - 8) as f32) * scale;
                }
            }
        }
    }

    fn decode_embed_row_f32(&self, token_id: usize, hidden_size: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; hidden_size];
        self.decode_embed_into(token_id, hidden_size, &mut out);
        out
    }

    fn decode_ple_row_f32(
        &self,
        token_id: usize,
        ple_total_dim: usize,
        ple_dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; ple_total_dim];
        self.decode_ple_into(token_id, ple_total_dim, ple_dim, &mut out);
        out
    }
}

pub struct Gemma4GpuLayer {
    pub q_proj: BufferView,
    pub k_proj: BufferView,
    pub v_proj: BufferView,
    pub o_proj: BufferView,
    /// Vertically stacked [q, k, v rows] for prefill mul_mm (K-quant, KV layers).
    pub qkv_stacked: BufferView,
    pub gate_proj: BufferView,
    pub up_proj: BufferView,
    /// Interleaved [gate_i, up_i] Q4 rows for packed MLP matvec (decode).
    pub gate_up_proj: BufferView,
    /// Vertically stacked [gate rows, up rows] for prefill mul_mm (K-quant).
    pub gate_up_stacked: BufferView,
    pub down_proj: BufferView,

    // 4 norms per layer (Gemma-style)
    pub input_layernorm_weight: BufferView,
    pub post_attention_layernorm_weight: BufferView,
    pub pre_feedforward_layernorm_weight: BufferView,
    pub post_feedforward_layernorm_weight: BufferView,

    // PLE weights
    pub post_per_layer_input_norm_weight: BufferView,
    pub per_layer_input_gate_weight: BufferView, // Q4: [ple_dim, hidden_size]
    pub per_layer_projection_weight: BufferView, // Q4: [hidden_size, ple_dim]
    pub layer_scalar: f32,

    // QK norm weights
    pub q_norm_weight: BufferView,
    pub k_norm_weight: BufferView,

    // Layer properties
    pub is_full_attention: bool,
    pub has_kv: bool,           // false for shared KV layers (layers 24-41)
    pub kv_source_layer: usize, // which layer's KV cache to use
    pub head_dim: usize,
    pub q_out_dim: usize,
    pub kv_out_dim: usize,
    pub intermediate_size: usize,
    pub weight_format: WeightFormat,
}

/// Weight format for a layer's projection matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    F16,
    Q4_0,
    Q3_0,
    /// Mixed K-quant layer (community Q4_K_M). The per-layer tag only marks the
    /// layer as "not pure Q4_0/F16" so the Q4_0-only fused/mega paths are
    /// skipped; the actual kernel is chosen per tensor from `BufferView::format`.
    KQuant,
}

impl WeightFormat {
    pub fn to_u8(self) -> u8 {
        match self {
            WeightFormat::F16 => 0,
            WeightFormat::Q4_0 => 1,
            WeightFormat::Q3_0 => 2,
            WeightFormat::KQuant => 5,
        }
    }

    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => WeightFormat::F16,
            2 => WeightFormat::Q3_0,
            5 => WeightFormat::KQuant,
            _ => WeightFormat::Q4_0,
        }
    }

    pub fn is_quantized(self) -> bool {
        matches!(
            self,
            WeightFormat::Q4_0 | WeightFormat::Q3_0 | WeightFormat::KQuant
        )
    }

    pub fn is_q3(self) -> bool {
        matches!(self, WeightFormat::Q3_0)
    }

    pub fn is_kquant(self) -> bool {
        matches!(self, WeightFormat::KQuant)
    }
}

pub fn env_weight_format() -> WeightFormat {
    match std::env::var("WEIGHT_FORMAT").as_deref() {
        Ok("q4") | Ok("Q4") | Ok("q4_0") | Ok("Q4_0") => WeightFormat::Q4_0,
        Ok("q3") | Ok("Q3") | Ok("q3_0") | Ok("Q3_0") => WeightFormat::Q3_0,
        _ => WeightFormat::Q4_0,
    }
}

/// When `Q3_LAYER_END` is set, returns `Some((start, end))` for the
/// inclusive layer range to quantize with Q3_0 (default start=0).
/// Returns `None` when `Q3_LAYER_END` is unset (no Q3 layers).
fn q3_layer_range() -> Option<(usize, usize)> {
    let start = std::env::var("Q3_LAYER_START")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let end = std::env::var("Q3_LAYER_END")
        .ok()
        .and_then(|v| v.parse().ok())?;
    if end >= start { Some((start, end)) } else { None }
}

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

impl Gemma4GpuModel {
    /// Load model weights. Uses a Q4 cache file for fast subsequent loads.
    /// First run: loads safetensors, quantizes, saves cache (~116s).
    /// Subsequent runs: loads pre-quantized cache directly (~5-10s).
    pub fn new(model_dir: &str) -> Self {
        let cache_path = Path::new(model_dir).join("model.q4cache");
        let embed_cache_path = Path::new(model_dir).join("model.embed.cache");

        if embed_cache_path.exists() && Self::is_stale_split_weights_cache(&cache_path) {
            eprintln!(
                "  Stale Q4 cache (old interleaved layout). Delete model.q4cache and model.embed.cache, then re-run to re-quantize with GGUF layout."
            );
            std::process::exit(1);
        }

        if embed_cache_path.exists() && Self::is_split_weights_cache(&cache_path) {
            println!("  Found split Q4 cache, loading...");
            println!(
                "  (Delete model.q4cache + model.embed.cache to re-quantize with all-Q4 weights.)"
            );
            return Self::load_from_split_cache(model_dir, &embed_cache_path, &cache_path);
        }

        if cache_path.exists() && Self::is_legacy_weights_cache(&cache_path) {
            println!("  Found Q4 cache, loading pre-quantized weights...");
            return Self::load_from_legacy_cache(model_dir, &cache_path);
        }

        if embed_cache_path.exists() && cache_path.exists() {
            eprintln!(
                "  Corrupt cache: model.embed.cache exists but model.q4cache is not a valid split weights file."
            );
            eprintln!(
                "  Delete model.q4cache and model.embed.cache in the model directory, then re-run to re-quantize."
            );
            std::process::exit(1);
        }

        if cache_path.exists() {
            eprintln!("  Unrecognized model.q4cache format. Delete it and re-run to re-quantize.");
            std::process::exit(1);
        }

        println!("  No Q4 cache found, quantizing from safetensors (one-time)...");
        let model = Self::load_from_safetensors(model_dir);

        // Save cache for next time
        println!("  Saving Q4 cache for fast future loads...");
        model.save_cache(&cache_path);

        model
    }

    fn load_from_safetensors(model_dir: &str) -> Self {
        let config_path = Path::new(model_dir).join("config.json");
        let config_str = fs::read_to_string(&config_path).expect("Failed to read config.json");

        // Parse the outer config which wraps text_config
        let outer: serde_json::Value =
            serde_json::from_str(&config_str).expect("Failed to parse config.json");

        let text_config: Gemma4TextConfig = if let Some(tc) = outer.get("text_config") {
            serde_json::from_value(tc.clone()).expect("Failed to parse text_config")
        } else {
            // Flat config (text_config fields at top level)
            serde_json::from_str(&config_str).expect("Failed to parse config as Gemma4TextConfig")
        };

        let config = text_config;
        let ctx = MetalContext::new();

        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let num_layers = config.num_hidden_layers;
        let ple_dim = config.hidden_size_per_layer_input; // 256

        // Determine max head_dim across all layers for buffer allocation
        let max_head_dim = config.global_head_dim; // 512 (full attention layers)
        let max_q_out = num_heads * max_head_dim;
        let max_kv_out = num_kv_heads * max_head_dim;

        println!(
            "  Gemma4 E4B: {} layers, hidden={}, heads={}, kv_heads={}",
            num_layers, hidden_size, num_heads, num_kv_heads
        );
        println!(
            "  Sliding head_dim={}, Full head_dim={}, PLE dim={}",
            config.head_dim, config.global_head_dim, ple_dim
        );

        // Build shard file list
        let index_path = Path::new(model_dir).join("model.safetensors.index.json");
        let shard_files: Vec<String> = if index_path.exists() {
            let index_str = fs::read_to_string(&index_path).unwrap();
            let index: SafetensorsIndex = serde_json::from_str(&index_str).unwrap();
            let mut files: Vec<String> = index.weight_map.values().cloned().collect();
            files.sort();
            files.dedup();
            files
        } else {
            vec!["model.safetensors".to_string()]
        };

        // We'll use a helper that loads a shard and returns a HashMap of tensors
        let prefix = "model.language_model.";

        println!("  Loading embeddings...");

        let mut owned_embed: Option<Vec<u8>> = None;
        let mut owned_ple: Option<Vec<u8>> = None;
        let mut final_norm_data: Vec<f32> = Vec::new();
        let mut per_layer_proj_norm_data: Vec<f32> = Vec::new();
        let mut per_layer_model_proj_data: Vec<f32> = Vec::new();
        let mut lm_head_f32: Vec<f32> = Vec::new();

        // Load global weights from shards — use memory mapping to avoid loading entire file
        for shard_file in &shard_files {
            let shard_path = Path::new(model_dir).join(shard_file);
            let file = fs::File::open(&shard_path)
                .unwrap_or_else(|_| panic!("Failed to open shard: {}", shard_file));
            let mmap = unsafe { memmap2::Mmap::map(&file) }
                .unwrap_or_else(|_| panic!("Failed to mmap shard: {}", shard_file));
            let safetensors =
                SafeTensors::deserialize(&mmap).expect("Failed to deserialize safetensors");

            for (name, tensor_view) in safetensors.tensors() {
                let clean_name = name.strip_prefix(prefix).unwrap_or(&name);

                if clean_name == "embed_tokens.weight" && owned_embed.is_none() {
                    let data = tensor_view.data();
                    println!(
                        "    embed_tokens: {:?} ({:.1} MB)",
                        tensor_view.shape(),
                        data.len() as f64 / 1024.0 / 1024.0
                    );
                    lm_head_f32 = data
                        .chunks_exact(2)
                        .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                        .collect();
                    owned_embed = Some(data.to_vec());
                } else if clean_name == "embed_tokens_per_layer.weight" && owned_ple.is_none() {
                    let data = tensor_view.data();
                    println!(
                        "    embed_tokens_per_layer: {:?} ({:.1} MB)",
                        tensor_view.shape(),
                        data.len() as f64 / 1024.0 / 1024.0
                    );
                    owned_ple = Some(data.to_vec());
                } else if clean_name == "model.norm.weight" || clean_name == "norm.weight" {
                    if final_norm_data.is_empty() {
                        final_norm_data = decode_tensor_to_f32(&tensor_view);
                    }
                } else if clean_name == "model.per_layer_projection_norm.weight"
                    || clean_name == "per_layer_projection_norm.weight"
                {
                    if per_layer_proj_norm_data.is_empty() {
                        per_layer_proj_norm_data = decode_tensor_to_f32(&tensor_view);
                    }
                } else if clean_name == "model.per_layer_model_projection.weight"
                    || clean_name == "per_layer_model_projection.weight"
                {
                    if per_layer_model_proj_data.is_empty() {
                        per_layer_model_proj_data = decode_tensor_to_f32(&tensor_view);
                        println!("    per_layer_model_projection: {:?}", tensor_view.shape());
                    }
                }
            }
            // mmap dropped here
        }

        let ple_total_dim = num_layers * ple_dim;
        let embed_tables = EmbedTables::from_owned(
            owned_embed.expect("embed_tokens not found"),
            owned_ple.expect("embed_tokens_per_layer not found"),
            vocab_size,
            ple_total_dim,
            hidden_size,
        );

        // Create GPU buffer for lm_head (tied embeddings — quantize to Q4_0)
        let lm_head_buf = BufferView::from_buffer(ctx.buffer_from_f32_as_q4(&lm_head_f32, vocab_size, hidden_size));
        println!(
            "    lm_head (tied, Q4_0 on GPU): {:.1} MB",
            lm_head_buf.length as f64 / 1024.0 / 1024.0
        );

        let final_norm_weight = BufferView::from_buffer(ctx.buffer_from_slice(&final_norm_data));
        let per_layer_projection_norm_weight =
            BufferView::from_buffer(ctx.buffer_from_slice(&per_layer_proj_norm_data));
        let per_layer_model_projection_weight = if !per_layer_model_proj_data.is_empty() {
            BufferView::from_buffer(
                ctx.buffer_from_f32_as_f16(&per_layer_model_proj_data),
            )
            .with_format(crate::gpu::weight_fmt::F16)
        } else {
            // Fallback: create empty buffer (shouldn't happen for E4B)
            println!(
                "  WARNING: per_layer_model_projection not found, PLE context projection disabled"
            );
            BufferView::from_buffer(ctx.buffer_empty(1))
        };

        // Load all layers
        let num_layers_to_load = num_layers;
        let q3_range = q3_layer_range();
        let use_q3 = q3_range.is_some();
        let (q3_start, q3_end) = q3_range.unwrap_or((0, 0));
        if use_q3 {
            println!(
                "  Loading layers (Q4_0 + Q3_0 layers {}-{}, {} layers)...",
                q3_start, q3_end, num_layers_to_load
            );
        } else {
            println!(
                "  Loading layers (Q4_0 quantized, {} layers)...",
                num_layers_to_load
            );
        }
        let mut layers = Vec::with_capacity(num_layers_to_load);

        for layer_idx in 0..num_layers_to_load {
            println!("    Layer {}/{}", layer_idx + 1, num_layers);
            let is_full = config.is_full_attention(layer_idx);
            let head_dim = config.layer_head_dim(layer_idx);
            let q_out = num_heads * head_dim;
            let kv_out = num_kv_heads * head_dim;

            // Load this layer's weights from the appropriate shard(s)
            let layer_prefix = format!("layers.{}", layer_idx);
            let mut layer_tensors: HashMap<String, Vec<f32>> = HashMap::new();

            for shard_file in &shard_files {
                let shard_path = Path::new(model_dir).join(shard_file);
                let file = fs::File::open(&shard_path)
                    .unwrap_or_else(|_| panic!("Failed to open shard: {}", shard_file));
                let mmap = unsafe { memmap2::Mmap::map(&file) }
                    .unwrap_or_else(|_| panic!("Failed to mmap shard: {}", shard_file));
                let safetensors =
                    SafeTensors::deserialize(&mmap).expect("Failed to deserialize safetensors");

                for (name, tensor_view) in safetensors.tensors() {
                    let clean_name = name.strip_prefix(prefix).unwrap_or(&name);
                    if clean_name.starts_with(&layer_prefix) {
                        let short_name = clean_name
                            .strip_prefix(&format!("{}.", layer_prefix))
                            .unwrap_or(clean_name);
                        if !layer_tensors.contains_key(short_name) {
                            layer_tensors
                                .insert(short_name.to_string(), decode_tensor_to_f32(&tensor_view));
                        }
                    }
                }
            }

            // Extract layer_scalar
            let layer_scalar = layer_tensors
                .get("layer_scalar")
                .map(|v| v[0])
                .unwrap_or(1.0);

            // Build GPU buffers for this layer
            let q_proj_data = layer_tensors
                .remove("self_attn.q_proj.weight")
                .expect("q_proj missing");
            let k_proj_data = layer_tensors
                .remove("self_attn.k_proj.weight")
                .expect("k_proj missing");
            let v_proj_data = layer_tensors
                .remove("self_attn.v_proj.weight")
                .expect("v_proj missing");
            let o_proj_data = layer_tensors
                .remove("self_attn.o_proj.weight")
                .expect("o_proj missing");
            let gate_proj_data = layer_tensors
                .remove("mlp.gate_proj.weight")
                .expect("gate_proj missing");
            let up_proj_data = layer_tensors
                .remove("mlp.up_proj.weight")
                .expect("up_proj missing");
            let down_proj_data = layer_tensors
                .remove("mlp.down_proj.weight")
                .expect("down_proj missing");

            let input_ln = layer_tensors
                .remove("input_layernorm.weight")
                .expect("input_layernorm missing");
            let post_attn_ln = layer_tensors
                .remove("post_attention_layernorm.weight")
                .expect("post_attention_layernorm missing");
            let pre_ff_ln = layer_tensors
                .remove("pre_feedforward_layernorm.weight")
                .expect("pre_feedforward_layernorm missing");
            let post_ff_ln = layer_tensors
                .remove("post_feedforward_layernorm.weight")
                .expect("post_feedforward_layernorm missing");
            let post_ple_norm = layer_tensors
                .remove("post_per_layer_input_norm.weight")
                .expect("post_per_layer_input_norm missing");

            let ple_gate_data = layer_tensors
                .remove("per_layer_input_gate.weight")
                .expect("per_layer_input_gate missing");
            let ple_proj_data = layer_tensors
                .remove("per_layer_projection.weight")
                .expect("per_layer_projection missing");

            let q_norm_data = layer_tensors
                .remove("self_attn.q_norm.weight")
                .expect("q_norm missing");
            let k_norm_data = layer_tensors
                .remove("self_attn.k_norm.weight")
                .expect("k_norm missing");

            let is_q3 = use_q3 && layer_idx >= q3_start && layer_idx <= q3_end;
            let q_weight = |data: &[f32], rows: usize, cols: usize| -> Buffer {
                if is_q3 {
                    ctx.buffer_from_f32_as_q3(data, rows, cols)
                } else {
                    ctx.buffer_from_f32_as_q4(data, rows, cols)
                }
            };
            let layer_inter = config.layer_intermediate_size(layer_idx);

            let layer = Gemma4GpuLayer {
                q_proj: BufferView::from_buffer(q_weight(&q_proj_data, q_out, hidden_size)),
                k_proj: BufferView::from_buffer(q_weight(&k_proj_data, kv_out, hidden_size)),
                v_proj: BufferView::from_buffer(q_weight(&v_proj_data, kv_out, hidden_size)),
                o_proj: BufferView::from_buffer(q_weight(&o_proj_data, hidden_size, q_out)),

                qkv_stacked: BufferView::from_buffer(ctx.buffer_empty(1)),
                gate_proj: BufferView::from_buffer(q_weight(&gate_proj_data, layer_inter, hidden_size)),
                up_proj: BufferView::from_buffer(q_weight(&up_proj_data, layer_inter, hidden_size)),
                gate_up_proj: BufferView::from_buffer(ctx.buffer_empty(1)),
                gate_up_stacked: BufferView::from_buffer(ctx.buffer_empty(1)),
                down_proj: BufferView::from_buffer(q_weight(&down_proj_data, hidden_size, layer_inter)),

                input_layernorm_weight: BufferView::from_buffer(ctx.buffer_from_slice(&input_ln)),
                post_attention_layernorm_weight: BufferView::from_buffer(ctx.buffer_from_slice(&post_attn_ln)),
                pre_feedforward_layernorm_weight: BufferView::from_buffer(ctx.buffer_from_slice(&pre_ff_ln)),
                post_feedforward_layernorm_weight: BufferView::from_buffer(ctx.buffer_from_slice(&post_ff_ln)),
                post_per_layer_input_norm_weight: BufferView::from_buffer(ctx.buffer_from_slice(&post_ple_norm)),

                per_layer_input_gate_weight: BufferView::from_buffer(
                    ctx.buffer_from_f32_as_f16(&ple_gate_data),
                )
                .with_format(crate::gpu::weight_fmt::F16),
                per_layer_projection_weight: BufferView::from_buffer(
                    ctx.buffer_from_f32_as_f16(&ple_proj_data),
                )
                .with_format(crate::gpu::weight_fmt::F16),
                layer_scalar,

                q_norm_weight: BufferView::from_buffer(ctx.buffer_from_slice(&q_norm_data)),
                k_norm_weight: BufferView::from_buffer(ctx.buffer_from_slice(&k_norm_data)),

                is_full_attention: is_full,
                has_kv: layer_idx < (num_layers - config.num_kv_shared_layers),
                kv_source_layer: 0,
                head_dim,
                q_out_dim: q_out,
                kv_out_dim: kv_out,
                intermediate_size: layer_inter,
                weight_format: if is_q3 { WeightFormat::Q3_0 } else { WeightFormat::Q4_0 },
            };

            layers.push(layer);
            // layer_tensors dropped here, freeing memory
        }

        Self::assemble(
            ctx,
            config,
            embed_tables,
            lm_head_buf,
            final_norm_weight,
            per_layer_projection_norm_weight,
            per_layer_model_projection_weight,
            layers,
        )
    }

    /// Shared model assembly: KV-sharing wiring, gate/up packing, scratch/KV/RoPE
    /// buffer allocation, and final struct construction. Used by both the
    /// safetensors loader and the GGUF loader once per-layer `Gemma4GpuLayer`s
    /// and shared weights have been built.
    fn assemble(
        ctx: MetalContext,
        config: Gemma4TextConfig,
        embed_tables: EmbedTables,
        lm_head_buf: BufferView,
        final_norm_weight: BufferView,
        per_layer_projection_norm_weight: BufferView,
        per_layer_model_projection_weight: BufferView,
        mut layers: Vec<Gemma4GpuLayer>,
    ) -> Self {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads_max = *config.num_key_value_heads_per_layer.iter().max().unwrap_or(&config.num_key_value_heads);
        let vocab_size = config.vocab_size;
        let num_layers = config.num_hidden_layers;
        let ple_dim = config.hidden_size_per_layer_input;
        let max_head_dim = config.global_head_dim;
        let max_q_out = num_heads * max_head_dim;
        let max_kv_out = (0..num_layers).map(|i| config.layer_num_kv_heads(i) * config.layer_head_dim(i)).max().unwrap_or(num_kv_heads_max * max_head_dim);
        // Per-layer max intermediate for scratch buffers

        // Compute kv_source_layer for shared layers
        // For each shared layer, find the last non-shared layer of the same type
        let first_kv_shared = num_layers - config.num_kv_shared_layers;
        for i in first_kv_shared..num_layers {
            let layer_type = &config.layer_types[i];
            // Find the last non-shared layer with the same type
            let mut source = 0;
            for j in (0..first_kv_shared).rev() {
                if &config.layer_types[j] == layer_type {
                    source = j;
                    break;
                }
            }
            layers[i].kv_source_layer = source;
        }
        // Non-shared layers use their own index
        for i in 0..first_kv_shared {
            layers[i].kv_source_layer = i;
        }

        let layers = Self::pack_layers_prefill_stacked(&ctx, layers, hidden_size as u32);

        let max_intermediate_size = layers
            .iter()
            .map(|layer| layer.intermediate_size)
            .max()
            .unwrap_or_else(|| config.max_intermediate_size());

        if first_kv_shared < num_layers {
            println!(
                "  KV sharing: layers 0-{} have own KV, layers {}-{} share",
                first_kv_shared - 1,
                first_kv_shared,
                num_layers - 1
            );
        } else {
            println!("  KV sharing: all {} layers have own KV (no sharing)", num_layers);
        }

        // Pre-allocate scratch buffers
        let hidden_buf = ctx.buffer_empty(hidden_size);
        let normed_buf = ctx.buffer_empty(hidden_size);
        let residual_buf = ctx.buffer_empty(hidden_size);
        let q_buf = ctx.buffer_empty(max_q_out);
        let k_buf = ctx.buffer_empty(max_kv_out);
        let v_buf = ctx.buffer_empty(max_kv_out);
        let attn_out_buf = ctx.buffer_empty(max_q_out);
        let o_out_buf = ctx.buffer_empty(hidden_size);
        let gate_buf = ctx.buffer_empty(max_intermediate_size);
        let up_buf = ctx.buffer_empty(max_intermediate_size);
        let gelu_buf = ctx.buffer_empty(max_intermediate_size);
        let down_buf = ctx.buffer_empty(hidden_size);
        let logits_buf = ctx.buffer_empty(vocab_size);
        let sample_out_buf = ctx.buffer_empty_u32(1);
        let inv_rms_buf = ctx.buffer_empty(1);
        // PLE scratch (allocate at least 1 byte to avoid zero-size Metal buffers)
        let ple_alloc = ple_dim.max(1);
        let ple_embed_buf = ctx.buffer_empty(ple_alloc);
        let ple_gated_buf = ctx.buffer_empty(ple_alloc);
        let ple_normed_buf = ctx.buffer_empty(ple_alloc);
        let ple_projected_buf = ctx.buffer_empty(hidden_size.max(1));
        let ple_ctx_proj_alloc = (num_layers * ple_dim).max(1);
        let ple_context_proj_buf = ctx.buffer_empty(ple_ctx_proj_alloc);
        let ple_token_id_buf = ctx.buffer_empty(ple_ctx_proj_alloc);
        let ple_combined_buf = ctx.buffer_empty(ple_ctx_proj_alloc);

        // QK norm scratch (max head_dim per head)
        let q_normed_buf = ctx.buffer_empty(max_q_out);
        let k_normed_buf = ctx.buffer_empty(max_kv_out);
        let ggml_fa_tmp_elems = (crate::ggml_flash_attn::flash_attn_tmp_bytes(
            num_heads as u32,
            max_head_dim as u32,
        ) / 4) as usize;
        let ggml_fa_tmp_buf = ctx.buffer_empty(ggml_fa_tmp_elems);

        // KV cache: f16 precision to halve memory bandwidth
        let kv_cache_type = KvCacheType::from_env();
        let kv_capacity = configured_kv_capacity(config.max_position_embeddings);
        let mut k_cache = Vec::with_capacity(num_layers);
        let mut v_cache = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let hd = config.layer_head_dim(i);
            let layer_n_kv = config.layer_num_kv_heads(i);
            assert!(hd % 32 == 0, "head_dim must be a multiple of 32 for quantized KV cache");
            // K and V may use different bit-widths (asymmetric TurboQuant).
            let k_byte_len = (layer_n_kv * kv_capacity as usize * kv_cache_type.k_row_bytes(hd)) as u64;
            let v_byte_len = (layer_n_kv * kv_capacity as usize * kv_cache_type.v_row_bytes(hd)) as u64;
            k_cache.push(
                ctx.device
                    .new_buffer(k_byte_len, MTLResourceOptions::StorageModeShared),
            );
            v_cache.push(
                ctx.device
                    .new_buffer(v_byte_len, MTLResourceOptions::StorageModeShared),
            );
        }
        let tq_state = build_turboquant_state(&ctx, &config, kv_cache_type, kv_capacity);
        let TurboQuantState {
            turboquant,
            tq_q_rot,
            tq_k_rot,
            tq_v_rot,
            tq_out,
            tq_scores,
            tq_rw_k,
            tq_rw_v,
            tq_rw,
            tq_hot_k,
            tq_hot_v,
            tq_hot_w,
        } = tq_state;
        let f16_bytes = num_kv_heads_max * kv_capacity as usize * config.head_dim * 2 + num_kv_heads_max * kv_capacity as usize * config.global_head_dim * 2;
        let quant_bytes = num_kv_heads_max * kv_capacity as usize
            * (kv_cache_type.k_row_bytes(config.head_dim) + kv_cache_type.v_row_bytes(config.head_dim))
            / 2
            + num_kv_heads_max * kv_capacity as usize
            * (kv_cache_type.k_row_bytes(config.global_head_dim) + kv_cache_type.v_row_bytes(config.global_head_dim))
            / 2;
        println!("  KV cache type: {}, est. memory per layer: {:.1} MB (vs f16: {:.1} MB, {:.0}% savings)",
            kv_cache_type,
            quant_bytes as f64 / num_layers as f64 / 1024.0 / 1024.0,
            f16_bytes as f64 / num_layers as f64 / 1024.0 / 1024.0,
            (1.0 - quant_bytes as f64 / f16_bytes as f64) * 100.0,
        );
        let max_prefill_seq = configured_max_prefill_seq(kv_capacity);
        let prefill_scratch = PrefillScratch::new(
            &ctx,
            max_prefill_seq,
            hidden_size,
            max_q_out,
            max_kv_out,
            max_intermediate_size,
            vocab_size,
            num_layers,
            ple_dim,
            kv_capacity,
            num_kv_heads_max as u32,
            max_head_dim as u32,
        );
        let decode_batch_scratch = DecodeBatchScratch::new(
            &ctx,
            DEFAULT_MAX_DECODE_BATCH,
            hidden_size,
            max_q_out,
            max_kv_out,
            max_intermediate_size,
            vocab_size,
            num_layers,
            ple_dim,
            num_heads as u32,
            max_head_dim as u32,
        );
        // Rotary buffers (allocate for max head_dim)
        let cos_buf = ctx.buffer_empty(max_head_dim);
        let sin_buf = ctx.buffer_empty(max_head_dim);

        // Per-layer prefill/batch RoPE buffers (decode uses packed GPU fill).
        let (decode_rope_cos_packed, decode_rope_sin_packed, rope_layer_params_buf) =
            alloc_decode_rope_buffers(&ctx, &config, num_layers, max_head_dim);
        let mut per_layer_prefill_cos_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_prefill_sin_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_decode_batch_cos_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_decode_batch_sin_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_decode_batch_append_pos_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_decode_batch_kv_start_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_decode_batch_kv_len_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_ple_bufs = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let hd = config.layer_head_dim(i);
            per_layer_prefill_cos_bufs.push(ctx.buffer_empty(max_prefill_seq * hd));
            per_layer_prefill_sin_bufs.push(ctx.buffer_empty(max_prefill_seq * hd));
            per_layer_decode_batch_cos_bufs.push(ctx.buffer_empty(DEFAULT_MAX_DECODE_BATCH * hd));
            per_layer_decode_batch_sin_bufs.push(ctx.buffer_empty(DEFAULT_MAX_DECODE_BATCH * hd));
            per_layer_decode_batch_append_pos_bufs
                .push(ctx.buffer_empty_u32(DEFAULT_MAX_DECODE_BATCH));
            per_layer_decode_batch_kv_start_bufs
                .push(ctx.buffer_empty_u32(DEFAULT_MAX_DECODE_BATCH));
            per_layer_decode_batch_kv_len_bufs.push(ctx.buffer_empty_u32(DEFAULT_MAX_DECODE_BATCH));
            per_layer_ple_bufs.push(ctx.buffer_empty(ple_dim.max(1)));
        }

        // MTP verify scratch buffers
        let mtp_verify_logits_buf =
            ctx.buffer_empty(MAX_MTP_VERIFY_SEQ * vocab_size);
        let mtp_verify_argmax_buf = ctx.buffer_empty_u32(MAX_MTP_VERIFY_SEQ);
        let mtp_verify_hidden_buf =
            ctx.buffer_empty(MAX_MTP_VERIFY_SEQ * hidden_size);
        let mtp_head_dims: Vec<usize> = (0..num_layers).map(|i| config.layer_head_dim(i)).collect();
        let mtp_verify_scratch = MtpVerifyScratch::new(
            &ctx,
            MAX_MTP_VERIFY_SEQ,
            hidden_size,
            (num_layers * ple_dim).max(1),
            &mtp_head_dims,
        );

        println!("  Decode RoPE: GPU fill (rope_fill_decode kernel)");
        {
            let kq = layers.iter().filter(|l| l.weight_format.is_kquant()).count();
            let f16 = layers
                .iter()
                .filter(|l| l.weight_format == WeightFormat::F16)
                .count();
            let q3 = layers.iter().filter(|l| l.weight_format.is_q3()).count();
            let q4 = num_layers - kq - f16 - q3;
            println!(
                "  Weights: {} layers Q4_0, {} K-quant (Q4_K/Q6_K native), {} f16, {} Q3_0",
                q4, kq, f16, q3
            );
        }

        println!(
            "  Parallel prefill scratch: max_seq={}",
            prefill_scratch.max_seq_len
        );
        println!(
            "  Decode batch scratch: max_batch={}",
            decode_batch_scratch.max_batch_size
        );
        println!("  Gemma4 model loaded successfully (Q4_0 quantized on Metal)");

        let model = Gemma4GpuModel {
            ctx,
            config,
            embed_tables,
            lm_head_buf,
            layers,
            final_norm_weight,
            per_layer_projection_norm_weight,
            per_layer_model_projection_weight,
            hidden_buf,
            normed_buf,
            residual_buf,
            q_buf,
            k_buf,
            v_buf,
            attn_out_buf,
            o_out_buf,
            gate_buf,
            up_buf,
            gelu_buf,
            down_buf,
            logits_buf,
            sample_out_buf,
            inv_rms_buf,
            prefill_scratch,
            decode_batch_scratch,
            mtp_verify_logits_buf,
            mtp_verify_argmax_buf,
            mtp_verify_hidden_buf,
            mtp_verify_scratch,
            ple_embed_buf,
            ple_gated_buf,
            ple_normed_buf,
            ple_projected_buf,
            ple_context_proj_buf,
            ple_token_id_buf,
            ple_combined_buf,
            q_normed_buf,
            k_normed_buf,
            ggml_fa_tmp_buf,
            k_cache,
            v_cache,
            kv_seq_len: 0,
            kv_capacity,
            kv_cache_type,
            turboquant,
            tq_q_rot,
            tq_k_rot,
            tq_v_rot,
            tq_out,
            tq_scores,
            tq_rw_k,
            tq_rw_v,
            tq_rw,
            tq_hot_k,
            tq_hot_v,
            tq_hot_w,
            tq_hot_spilled: false,
            cos_buf,
            sin_buf,
            decode_rope_cos_packed,
            decode_rope_sin_packed,
            rope_layer_params_buf,
            rope_max_head_dim: max_head_dim,
            per_layer_prefill_cos_bufs,
            per_layer_prefill_sin_bufs,
            per_layer_decode_batch_cos_bufs,
            per_layer_decode_batch_sin_bufs,
            per_layer_decode_batch_append_pos_bufs,
            per_layer_decode_batch_kv_start_bufs,
            per_layer_decode_batch_kv_len_bufs,
            per_layer_ple_bufs,
            total_tokens: 0,
            weights_mmap_offset: None,
            embed_decode_scratch: vec![0.0f32; hidden_size],
            ple_decode_scratch: vec![0.0f32; (num_layers * ple_dim).max(1)],
        };

        crate::decode_fused::log_fused_decode_status(&model);
        model
    }

    /// Load a Gemma-4 model directly from a `.gguf` file (architecture `gemma4`).
    /// On first load, dequantizes K-quant embeddings and quantizes weights to Q4_0
    /// on the GPU, saving a Q4 cache for instant loads on subsequent runs.
    pub fn load_from_gguf(gguf_path: &str) -> Self {
        let load_start = Instant::now();

        // Open GGUF for fast metadata parsing (header only) and load weights
        // directly into GPU buffers (no intermediate qcache file). The Gguf is
        // kept alive (Arc) so EmbedTables can decode lookup rows on demand.
        let gguf_path = std::path::Path::new(gguf_path);
        let g = std::sync::Arc::new(crate::gguf::Gguf::open(gguf_path));
        let arch = g.get_str("general.architecture").unwrap_or("");
        assert_eq!(
            arch, "gemma4",
            "GGUF architecture is '{}', expected 'gemma4'",
            arch
        );

        let config = gemma4_config_from_gguf(&g);

        let ctx = MetalContext::new();

        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let num_layers = config.num_hidden_layers;
        let ple_dim = config.hidden_size_per_layer_input;
        let ple_total_dim = num_layers * ple_dim;

        println!(
            "  Gemma4 (GGUF): {} layers, hidden={}, heads={}, kv_heads={}, vocab={}",
            num_layers, hidden_size, num_heads, num_kv_heads, vocab_size
        );
        println!(
            "  Sliding head_dim={}, Full head_dim={}, PLE dim={}, shared_kv_layers={}",
            config.head_dim, config.global_head_dim, ple_dim, config.num_kv_shared_layers
        );
        println!(
            "  RoPE: sliding θ={:.0} (all {} dims), full θ={:.0} p-RoPE factor={:.2} ({} of {} dims)",
            config.sliding_rope_theta(),
            config.head_dim,
            config.full_rope_theta(),
            config.full_partial_rotary_factor(),
            (config.global_head_dim as f64 * config.full_partial_rotary_factor()) as usize,
            config.global_head_dim,
        );

        // --- Embeddings (native GGUF: decode rows on demand, no CPU conversion) ---
        println!("  Loading embeddings directly from GGUF blocks (no conversion)...");
        // lm_head is tied to the input embeddings (no separate output.weight tensor).
        // Upload the native GGUF token_embd blocks directly (Q4_K/Q6_K/F16/...) so the
        // tied logits matrix needs no CPU dequant/requant; logits paths dispatch by
        // format. Falls back to Q4_0 requant only for unusual embedding types.
        let lm_head_buf = {
            use crate::gguf::ggml_type;
            use crate::gpu::weight_fmt;
            match g.tensor_type("token_embd.weight") {
                ggml_type::Q4_0 => BufferView::from_buffer(
                    ctx.buffer_from_slice_no_copy(g.tensor_raw("token_embd.weight")),
                )
                .with_format(weight_fmt::Q4_0),
                ggml_type::Q4_K => BufferView::from_buffer(
                    ctx.buffer_from_slice_no_copy(g.tensor_raw("token_embd.weight")),
                )
                .with_format(weight_fmt::Q4_K),
                ggml_type::Q6_K => BufferView::from_buffer(
                    ctx.buffer_from_slice_no_copy(g.tensor_raw("token_embd.weight")),
                )
                .with_format(weight_fmt::Q6_K),
                ggml_type::F16 | ggml_type::BF16 => BufferView::from_buffer(
                    ctx.buffer_from_slice_no_copy(g.tensor_raw("token_embd.weight")),
                )
                .with_format(weight_fmt::F16),
                _ => BufferView::from_buffer(ctx.buffer_from_f32_as_q4(
                    &g.dequant_to_f32("token_embd.weight"),
                    vocab_size,
                    hidden_size,
                )),
            }
        };
        println!(
            "    lm_head (tied, native {} on GPU): {:.1} MB",
            match lm_head_buf.format {
                crate::gpu::weight_fmt::Q4_K => "Q4_K",
                crate::gpu::weight_fmt::Q6_K => "Q6_K",
                crate::gpu::weight_fmt::F16 => "F16",
                _ => "Q4_0",
            },
            lm_head_buf.length as f64 / 1024.0 / 1024.0
        );
        // Embedding + PLE lookup tables are decoded directly from native GGUF blocks
        // (Q4_K / F16 / ...) on demand, eliminating the multi-GB CPU conversion.
        let embed_tables = EmbedTables::from_gguf(
            std::sync::Arc::clone(&g),
            vocab_size,
            hidden_size,
            ple_total_dim,
        );

        // --- Shared norms / projections ---
        let final_norm_weight =
            BufferView::from_buffer(ctx.buffer_from_slice(&g.dequant_to_f32("output_norm.weight")));
        let per_layer_projection_norm_weight = if ple_total_dim > 0 {
            BufferView::from_buffer(
                ctx.buffer_from_slice(&g.dequant_to_f32("per_layer_proj_norm.weight")),
            )
        } else {
            BufferView::from_buffer(ctx.buffer_empty(1))
        };
        let per_layer_model_projection_weight = if ple_total_dim > 0 && g.has_tensor("per_layer_model_proj.weight") {
            let proj_f32 = g.dequant_to_f32("per_layer_model_proj.weight");
            let pw = {
                use crate::gguf::ggml_type;
                use crate::gpu::weight_fmt;
                match g.tensor_type("per_layer_model_proj.weight") {
                    ggml_type::Q4_K => BufferView::from_buffer(
                        ctx.buffer_from_slice_no_copy(g.tensor_raw("per_layer_model_proj.weight")),
                    )
                    .with_format(weight_fmt::Q4_K),
                    ggml_type::Q6_K => BufferView::from_buffer(
                        ctx.buffer_from_slice_no_copy(g.tensor_raw("per_layer_model_proj.weight")),
                    )
                    .with_format(weight_fmt::Q6_K),
                    ggml_type::BF16 | ggml_type::F16 => BufferView::from_buffer(
                        ctx.buffer_from_slice_no_copy(g.tensor_raw("per_layer_model_proj.weight")),
                    )
                    .with_format(weight_fmt::F16),
                    ggml_type::F32 => BufferView::from_buffer(
                        ctx.buffer_from_f32_as_f16(&proj_f32),
                    )
                    .with_format(weight_fmt::F16),
                    _ => BufferView::from_buffer(ctx.buffer_from_f32_as_q4(
                        &proj_f32,
                        ple_total_dim,
                        hidden_size,
                    )),
                }
            };
            drop(proj_f32);
            println!(
                "  per_layer_model_proj: format={} ({:.1} MB)",
                match pw.format {
                    crate::gpu::weight_fmt::F16 => "F16",
                    crate::gpu::weight_fmt::Q4_K => "Q4_K",
                    crate::gpu::weight_fmt::Q6_K => "Q6_K",
                    _ => "Q4_0",
                },
                pw.length as f64 / (1024.0 * 1024.0)
            );
            pw
        } else {
            BufferView::from_buffer(ctx.buffer_empty(1))
        };

        // --- Layers ---
        let q6k_to_q4 = std::env::var("Q6K_TO_Q4").as_deref() == Ok("1");
        if q6k_to_q4 {
            println!("  Q6K_TO_Q4=1: converting Q6_K tensors to Q4_0 (faster, slightly lower quality)");
        }
        println!(
            "  Loading {} layers from GGUF (native Q4_K/Q6_K kept; other types -> Q4_0)...",
            num_layers
        );
        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let is_full = config.is_full_attention(layer_idx);
            let head_dim = config.layer_head_dim(layer_idx);
            let q_out = num_heads * head_dim;
            let layer_kv_heads = config.layer_num_kv_heads(layer_idx);
            let kv_out = layer_kv_heads * head_dim;
            let p = |suffix: &str| format!("blk.{}.{}", layer_idx, suffix);

            // Quantized projection weight: keep community K-quant blocks native
            // (Q4_K / Q6_K) and tag the view; otherwise dequant + requantize to
            // Q4_0 (lossless for on-disk Q4_0, lossy for Q4_1/Q5_K fallbacks).
            // Set Q6K_TO_Q4=1 to convert Q6_K→Q4_0 at load time (faster but
            // slightly lower quality for the most sensitive weights).
            let qw = |name: String, rows: usize, cols: usize| -> BufferView {
                use crate::gguf::ggml_type;
                use crate::gpu::weight_fmt;
                match g.tensor_type(&name) {
                    ggml_type::Q4_K => BufferView::from_buffer(
                        ctx.buffer_from_slice_no_copy(g.tensor_raw(&name)),
                    )
                    .with_format(weight_fmt::Q4_K),
                    ggml_type::Q6_K if q6k_to_q4 => {
                        let data = g.dequant_to_f32(&name);
                        BufferView::from_buffer(ctx.buffer_from_f32_as_q4(&data, rows, cols))
                    }
                    ggml_type::Q6_K => BufferView::from_buffer(
                        ctx.buffer_from_slice_no_copy(g.tensor_raw(&name)),
                    )
                    .with_format(weight_fmt::Q6_K),
                    // PLE inp_gate/proj are F32 on Q4_K_M; keep dense f16 for mul_mm_f16
                    // (requant→Q4_0 forced the slow projection_q4_batch path — same class
                    // of bug as per_layer_model_proj). Native F16/BF16 upload the raw
                    // blocks zero-copy; F32 is dequantized to f16 (one tensor, small).
                    ggml_type::BF16 | ggml_type::F16 => BufferView::from_buffer(
                        ctx.buffer_from_slice_no_copy(g.tensor_raw(&name)),
                    )
                    .with_format(weight_fmt::F16),
                    ggml_type::F32 => {
                        let data = g.dequant_to_f32(&name);
                        BufferView::from_buffer(ctx.buffer_from_f32_as_f16(&data))
                            .with_format(weight_fmt::F16)
                    }
                    _ => {
                        let data = g.dequant_to_f32(&name);
                        BufferView::from_buffer(ctx.buffer_from_f32_as_q4(&data, rows, cols))
                    }
                }
            };
            let f32buf = |name: String| -> BufferView {
                BufferView::from_buffer(ctx.buffer_from_slice(&g.dequant_to_f32(&name)))
            };

            let layer_scalar = g.dequant_to_f32(&p("layer_output_scale.weight"))[0];
            let layer_inter = config.layer_intermediate_size(layer_idx);

            let q_proj = qw(p("attn_q.weight"), q_out, hidden_size);
            let k_proj = qw(p("attn_k.weight"), kv_out, hidden_size);
            let v_proj = if g.has_tensor(&p("attn_v.weight")) {
                qw(p("attn_v.weight"), kv_out, hidden_size)
            } else {
                // Joint KV attention: no separate V weight; reuse K projection as V.
                qw(p("attn_k.weight"), kv_out, hidden_size)
            };
            let o_proj = qw(p("attn_output.weight"), hidden_size, q_out);
            let gate_proj = qw(p("ffn_gate.weight"), layer_inter, hidden_size);
            let up_proj = qw(p("ffn_up.weight"), layer_inter, hidden_size);
            let down_proj = qw(p("ffn_down.weight"), hidden_size, layer_inter);
            let per_layer_input_gate_weight = if ple_dim > 0 {
                qw(p("inp_gate.weight"), ple_dim, hidden_size)
            } else {
                BufferView::from_buffer(ctx.buffer_empty(1))
            };
            let per_layer_projection_weight = if ple_dim > 0 {
                qw(p("proj.weight"), hidden_size, ple_dim)
            } else {
                BufferView::from_buffer(ctx.buffer_empty(1))
            };

            // A layer is "K-quant" if any projection kept native K-quant blocks;
            // this disables the Q4_0-only fused/mega paths for the layer while
            // each tensor still dispatches by its own `BufferView::format`.
            use crate::gpu::weight_fmt;
            let any_kquant = [
                &q_proj, &k_proj, &v_proj, &o_proj, &gate_proj, &up_proj, &down_proj,
                &per_layer_input_gate_weight, &per_layer_projection_weight,
            ]
            .iter()
            .any(|v| matches!(v.format, weight_fmt::Q4_K | weight_fmt::Q6_K));

            let layer = Gemma4GpuLayer {
                q_proj,
                k_proj,
                v_proj,
                o_proj,

                qkv_stacked: BufferView::from_buffer(ctx.buffer_empty(1)),
                gate_proj,
                up_proj,
                gate_up_proj: BufferView::from_buffer(ctx.buffer_empty(1)),
                gate_up_stacked: BufferView::from_buffer(ctx.buffer_empty(1)),
                down_proj,

                input_layernorm_weight: f32buf(p("attn_norm.weight")),
                post_attention_layernorm_weight: f32buf(p("post_attention_norm.weight")),
                pre_feedforward_layernorm_weight: f32buf(p("ffn_norm.weight")),
                post_feedforward_layernorm_weight: f32buf(p("post_ffw_norm.weight")),
                post_per_layer_input_norm_weight: if ple_dim > 0 { f32buf(p("post_norm.weight")) } else { BufferView::from_buffer(ctx.buffer_empty(1)) },

                per_layer_input_gate_weight,
                per_layer_projection_weight,
                layer_scalar,

                q_norm_weight: f32buf(p("attn_q_norm.weight")),
                k_norm_weight: f32buf(p("attn_k_norm.weight")),

                is_full_attention: is_full,
                has_kv: layer_idx < (num_layers - config.num_kv_shared_layers),
                kv_source_layer: 0,
                head_dim,
                q_out_dim: q_out,
                kv_out_dim: kv_out,
                intermediate_size: layer_inter,
                weight_format: if any_kquant {
                    WeightFormat::KQuant
                } else {
                    WeightFormat::Q4_0
                },
            };
            layers.push(layer);
        }

        let model = Self::assemble(
            ctx,
            config,
            embed_tables,
            lm_head_buf,
            final_norm_weight,
            per_layer_projection_norm_weight,
            per_layer_model_projection_weight,
            layers,
        );

        println!(
            "  Loaded GGUF directly into GPU in {:.2}s (no qcache).",
            load_start.elapsed().as_secs_f64()
        );

        model
    }



    pub(crate) fn decode_rope_byte_offset(&self, layer_idx: usize) -> u64 {
        (layer_idx * self.rope_max_head_dim * std::mem::size_of::<f32>()) as u64
    }

    /// Save split cache: model.embed.cache (CPU mmap) + model.q4cache (GPU weights).
    fn save_cache(&self, path: &Path) {
        let embed_path = path
            .parent()
            .expect("cache path has no parent")
            .join("model.embed.cache");
        self.save_embed_cache(&embed_path);
        self.save_weights_cache(path);
    }



    fn save_embed_cache(&self, path: &Path) {
        use std::io::{Seek, Write};
        let mut file = fs::File::create(path).expect("Failed to create embed cache");
        file.write_all(b"GQ4E").unwrap();
        let embed_bytes = self.embed_tables.embed_bytes();
        file.write_all(&(embed_bytes.len() as u64).to_le_bytes()).unwrap();
        file.write_all(embed_bytes).unwrap();
        let ple_bytes = self.embed_tables.ple_bytes();
        file.write_all(&(ple_bytes.len() as u64).to_le_bytes()).unwrap();
        file.write_all(ple_bytes).unwrap();
        println!(
            "  Embed cache saved: {:.1} MB",
            file.metadata().map(|m| m.len()).unwrap_or(0) as f64 / 1024.0 / 1024.0
        );
    }

    fn save_weights_cache(&self, path: &Path) {
        if let Some(start_offset) = self.weights_mmap_offset {
            let mmap = self
                .embed_tables
                .mmap_ref()
                .expect("weights_mmap_offset set without backing mmap");
            Self::write_weights_cache_from_mmap(mmap, start_offset, path, self.layers.len());
            return;
        }

        use std::io::{Seek, Write};
        let mut file = fs::File::create(path).expect("Failed to create weights cache");
        file.write_all(b"GQ4G").unwrap();
        let save_view = |f: &mut fs::File, view: &BufferView| {
            f.write_all(&view.length.to_le_bytes()).unwrap();
            let pos = f.stream_position().expect("stream position") as usize;
            let pad_before = weight_section_pad(pos - WEIGHT_CACHE_MAGIC_LEN);
            if pad_before > 0 {
                f.write_all(&vec![0u8; pad_before]).unwrap();
            }
            f.write_all(view.as_bytes()).unwrap();
            let pos = f.stream_position().expect("stream position") as usize;
            let pad_after = weight_section_pad(pos - WEIGHT_CACHE_MAGIC_LEN);
            if pad_after > 0 {
                f.write_all(&vec![0u8; pad_after]).unwrap();
            }
        };
        save_view(&mut file, &self.lm_head_buf);
        save_view(&mut file, &self.per_layer_model_projection_weight);
        save_view(&mut file, &self.final_norm_weight);
        save_view(&mut file, &self.per_layer_projection_norm_weight);

        // Save per-layer weights
        let num_layers = self.layers.len() as u32;
        file.write_all(&num_layers.to_le_bytes()).unwrap();
        pad_weights_file_to_section_align(&mut file);
        for layer in &self.layers {
            save_view(&mut file, &layer.q_proj);
            save_view(&mut file, &layer.k_proj);
            save_view(&mut file, &layer.v_proj);
            save_view(&mut file, &layer.o_proj);
            save_view(&mut file, &layer.gate_proj);
            save_view(&mut file, &layer.up_proj);
            save_view(&mut file, &layer.down_proj);
            save_view(&mut file, &layer.input_layernorm_weight);
            save_view(&mut file, &layer.post_attention_layernorm_weight);
            save_view(&mut file, &layer.pre_feedforward_layernorm_weight);
            save_view(&mut file, &layer.post_feedforward_layernorm_weight);
            save_view(&mut file, &layer.post_per_layer_input_norm_weight);
            save_view(&mut file, &layer.per_layer_input_gate_weight);
            save_view(&mut file, &layer.per_layer_projection_weight);
            save_view(&mut file, &layer.q_norm_weight);
            save_view(&mut file, &layer.k_norm_weight);
            file.write_all(&layer.layer_scalar.to_le_bytes()).unwrap();
            file.write_all(&(layer.is_full_attention as u8).to_le_bytes())
                .unwrap();
            file.write_all(&(layer.has_kv as u8).to_le_bytes()).unwrap();
            file.write_all(&[layer.weight_format.to_u8()])
                .unwrap();
            file.write_all(&(layer.kv_source_layer as u32).to_le_bytes())
                .unwrap();
            file.write_all(&(layer.head_dim as u32).to_le_bytes())
                .unwrap();
            file.write_all(&(layer.q_out_dim as u32).to_le_bytes())
                .unwrap();
            file.write_all(&(layer.kv_out_dim as u32).to_le_bytes())
                .unwrap();
            pad_weights_file_to_section_align(&mut file);
        }
        println!(
            "  Weights cache saved: {:.1} MB",
            file.metadata().map(|m| m.len()).unwrap_or(0) as f64 / 1024.0 / 1024.0
        );
    }

    fn write_weights_cache_from_mmap(
        mmap: &Mmap,
        mut offset: usize,
        path: &Path,
        num_layers: usize,
    ) {
        use std::io::{Seek, Write};
        let mut file = fs::File::create(path).expect("Failed to create weights cache");
        file.write_all(b"GQ4G").unwrap();

        let write_blob = |file: &mut fs::File, mmap: &Mmap, offset: &mut usize| {
            let len = Self::read_u64_at(mmap, offset);
            if *offset + len as usize > mmap.len() {
                panic!(
                    "Internal error while writing weights cache: source blob at offset {} \
                     needs {} bytes but mmap is {} bytes",
                    *offset,
                    len,
                    mmap.len()
                );
            }
            file.write_all(&len.to_le_bytes()).unwrap();
            let pos = file.stream_position().expect("stream position") as usize;
            let pad_before = weight_section_pad(pos - WEIGHT_CACHE_MAGIC_LEN);
            if pad_before > 0 {
                file.write_all(&vec![0u8; pad_before]).unwrap();
            }
            file.write_all(&mmap[*offset..*offset + len as usize]).unwrap();
            *offset += len as usize;
            let pos = file.stream_position().expect("stream position") as usize;
            let pad_after = weight_section_pad(pos - WEIGHT_CACHE_MAGIC_LEN);
            if pad_after > 0 {
                file.write_all(&vec![0u8; pad_after]).unwrap();
            }
        };

        write_blob(&mut file, mmap, &mut offset);
        write_blob(&mut file, mmap, &mut offset);
        write_blob(&mut file, mmap, &mut offset);
        write_blob(&mut file, mmap, &mut offset);

        file.write_all(&mmap[offset..offset + 4]).unwrap();
        offset += 4;
        pad_weights_file_to_section_align(&mut file);

        for _ in 0..num_layers {
            for _ in 0..16 {
                write_blob(&mut file, mmap, &mut offset);
            }
            // layer_scalar + 3 bools + 4 u32 metadata fields
            file.write_all(&mmap[offset..offset + 23]).unwrap();
            offset += 23;
            pad_weights_file_to_section_align(&mut file);
        }

        println!(
            "  Weights cache saved: {:.1} MB",
            file.metadata().map(|m| m.len()).unwrap_or(0) as f64 / 1024.0 / 1024.0
        );
    }

    fn read_u64_at(mmap: &[u8], offset: &mut usize) -> u64 {
        if *offset + 8 > mmap.len() {
            panic!(
                "Corrupt weights cache near offset {} (file size {}). \
                 Delete model.q4cache and model.embed.cache, then re-run to re-quantize.",
                *offset,
                mmap.len()
            );
        }
        let value = u64::from_le_bytes(mmap[*offset..*offset + 8].try_into().unwrap());
        *offset += 8;
        value
    }

    fn load_weight_sections(
        data: &[u8],
        offset: &mut usize,
        device: &Device,
        num_layers: usize,
        hidden_size: usize,
        aligned_layout: bool,
        parent: Option<&Buffer>,
        parent_base: usize,
    ) -> (
        BufferView,
        BufferView,
        BufferView,
        BufferView,
        Vec<Gemma4GpuLayer>,
    ) {
        let read_buf = |offset: &mut usize| -> BufferView {
            let len = Self::read_u64_at(data, offset);
            let view = if aligned_layout {
                let parent = parent.expect("aligned layout requires parent buffer");
                let pad_before =
                    weight_section_pad(*offset - parent_base);
                *offset += pad_before;
                if *offset + len as usize > data.len() {
                    panic!(
                        "Corrupt weights cache: blob at offset {} needs {} bytes but file is {} bytes. \
                         Delete model.q4cache and model.embed.cache, then re-run to re-quantize.",
                        *offset,
                        len,
                        data.len()
                    );
                }
                let parent_off = *offset - parent_base;
                assert_eq!(
                    parent_off % WEIGHT_BLOB_ALIGN,
                    0,
                    "unaligned blob at file offset {} (section offset {}). \
                     Delete model.q4cache and model.embed.cache, then re-run to re-quantize.",
                    *offset,
                    parent_off
                );
                let view = BufferView::subrange(parent, parent_off as u64, len);
                *offset += len as usize;
                let pad_after = weight_section_pad(*offset - parent_base);
                *offset += pad_after;
                view
            } else {
                if *offset + len as usize > data.len() {
                    panic!(
                        "Corrupt weights cache: blob at offset {} needs {} bytes but file is {} bytes. \
                         Delete model.q4cache and model.embed.cache, then re-run to re-quantize.",
                        *offset,
                        len,
                        data.len()
                    );
                }
                let buf =
                    MetalContext::buffer_from_slice_parallel(device, &data[*offset..*offset + len as usize]);
                *offset += len as usize;
                BufferView::from_buffer(buf)
            };
            view
        };

        let lm_head_buf = read_buf(offset);
        let per_layer_model_projection_weight = read_buf(offset);
        let final_norm_weight = read_buf(offset);
        let per_layer_projection_norm_weight = read_buf(offset);

        let num_layers_in_cache =
            u32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap()) as usize;
        *offset += 4;
        let pad = weight_section_pad(*offset - parent_base);
        *offset += pad;
        assert_eq!(num_layers_in_cache, num_layers);

        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let q_proj = read_buf(offset);
            let k_proj = read_buf(offset);
            let v_proj = read_buf(offset);
            let o_proj = read_buf(offset);
            let gate_proj = read_buf(offset);
            let layer_inter = Self::infer_mlp_intermediate(&gate_proj, hidden_size);
            let up_proj = read_buf(offset);
            let down_proj = read_buf(offset);
            let input_layernorm_weight = read_buf(offset);
            let post_attention_layernorm_weight = read_buf(offset);
            let pre_feedforward_layernorm_weight = read_buf(offset);
            let post_feedforward_layernorm_weight = read_buf(offset);
            let post_per_layer_input_norm_weight = read_buf(offset);
            let per_layer_input_gate_weight = read_buf(offset);
            let per_layer_projection_weight = read_buf(offset);
            let q_norm_weight = read_buf(offset);
            let k_norm_weight = read_buf(offset);

            let layer_scalar = f32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap());
            *offset += 4;

            let is_full_attention = data[*offset] != 0;
            *offset += 1;
            let has_kv = data[*offset] != 0;
            *offset += 1;
            let wf = WeightFormat::from_u8(data[*offset]);
            *offset += 1;

            let kv_source_layer =
                u32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap()) as usize;
            *offset += 4;
            let head_dim =
                u32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap()) as usize;
            *offset += 4;
            let q_out_dim =
                u32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap()) as usize;
            *offset += 4;
            let kv_out_dim =
                u32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap()) as usize;
            *offset += 4;
            if aligned_layout {
                let pad = weight_section_pad(*offset - parent_base);
                *offset += pad;
            }

            if (layer_idx + 1) % 10 == 0 || layer_idx == num_layers - 1 {
                println!("    Loaded layer {}/{}", layer_idx + 1, num_layers);
            }

            layers.push(Gemma4GpuLayer {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                qkv_stacked: BufferView::from_buffer(
                    device.new_buffer(1, MTLResourceOptions::StorageModeShared),
                ),
                gate_proj,
                up_proj,
                gate_up_proj: BufferView::from_buffer(
                    device.new_buffer(1, MTLResourceOptions::StorageModeShared),
                ),
                gate_up_stacked: BufferView::from_buffer(
                    device.new_buffer(1, MTLResourceOptions::StorageModeShared),
                ),
                down_proj,
                input_layernorm_weight,
                post_attention_layernorm_weight,
                pre_feedforward_layernorm_weight,
                post_feedforward_layernorm_weight,
                post_per_layer_input_norm_weight,
                per_layer_input_gate_weight,
                per_layer_projection_weight,
                layer_scalar,
                q_norm_weight,
                k_norm_weight,
                is_full_attention,
                has_kv,
                kv_source_layer,
                head_dim,
                q_out_dim,
                kv_out_dim,
                intermediate_size: layer_inter,
                weight_format: wf,
            });
        }

        (
            lm_head_buf,
            per_layer_model_projection_weight,
            final_norm_weight,
            per_layer_projection_norm_weight,
            layers,
        )
    }

    /// Legacy monolithic GQ4C cache: mmap embeds on CPU + zero-copy GPU weights.
    fn load_from_legacy_cache(model_dir: &str, cache_path: &Path) -> Self {
        let load_start = Instant::now();

        let config_path = Path::new(model_dir).join("config.json");
        let config_str = fs::read_to_string(&config_path).expect("Failed to read config.json");
        let outer: serde_json::Value =
            serde_json::from_str(&config_str).expect("Failed to parse config.json");
        let config: Gemma4TextConfig = if let Some(tc) = outer.get("text_config") {
            serde_json::from_value(tc.clone()).expect("Failed to parse text_config")
        } else {
            serde_json::from_str(&config_str).expect("Failed to parse config")
        };

        let ctx = MetalContext::new();
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let num_layers = config.num_hidden_layers;
        let ple_dim = config.hidden_size_per_layer_input;
        let max_head_dim = config.global_head_dim;
        let max_q_out = num_heads * max_head_dim;
        let max_kv_out = num_kv_heads * max_head_dim;
        let device = &ctx.device;

        let cache_file = fs::File::open(cache_path).expect("Failed to open cache");
        let mmap = unsafe { Mmap::map(&cache_file).expect("Failed to mmap cache") };
        assert_eq!(&mmap[0..4], b"GQ4C", "Invalid cache magic (expected GQ4C legacy format)");

        let mut offset = 4;
        let embed_len = Self::read_u64_at(&mmap, &mut offset);
        let embed_offset = offset;
        offset += embed_len as usize;
        println!(
            "    embed_tokens: {:.1} MB (mmap)",
            embed_len as f64 / 1024.0 / 1024.0
        );

        let ple_embed_len = Self::read_u64_at(&mmap, &mut offset);
        let ple_offset = offset;
        offset += ple_embed_len as usize;
        println!(
            "    embed_tokens_per_layer: {:.1} MB (mmap)",
            ple_embed_len as f64 / 1024.0 / 1024.0
        );

        let weights_mmap_offset = offset;
        println!("  Loading weights (per-tensor copy)...");
        let weights_load_start = Instant::now();
        let (
            lm_head_buf,
            per_layer_model_projection_weight,
            final_norm_weight,
            per_layer_projection_norm_weight,
            layers,
        ) = Self::load_weight_sections(
            mmap.as_ref(),
            &mut offset,
            device,
            num_layers,
            config.hidden_size,
            false,
            None,
            weights_mmap_offset,
        );
        println!(
            "  Weights loaded in {:.2}s",
            weights_load_start.elapsed().as_secs_f64()
        );

        let ple_cols = num_layers * ple_dim;
        let embed_tables = EmbedTables::from_mmap(
            mmap,
            embed_offset,
            embed_len as usize,
            ple_offset,
            ple_embed_len as usize,
            ple_cols,
            vocab_size,
        );

        let model = Self::finish_cache_load(
            ctx,
            config,
            embed_tables,
            Some(weights_mmap_offset),
            lm_head_buf,
            per_layer_model_projection_weight,
            final_norm_weight,
            per_layer_projection_norm_weight,
            layers,
            load_start,
            "legacy Q4 cache",
        );

        // One-time migration to split cache (instant load on subsequent runs).
        let embed_path = Path::new(model_dir).join("model.embed.cache");
        if !Self::is_split_weights_cache(cache_path) {
            println!("  Migrating to split cache format...");
            if !embed_path.exists() {
                model.save_embed_cache(&embed_path);
            }
            model.save_weights_cache(cache_path);
        }

        model
    }

    fn is_split_weights_cache(path: &Path) -> bool {
        use std::io::Read;
        let file = match fs::File::open(path) {
            Ok(file) => file,
            Err(_) => return false,
        };
        // Weights-only cache should be hundreds of MB, not a truncated header.
        const MIN_BYTES: u64 = 64 * 1024 * 1024;
        if file.metadata().map(|m| m.len()).unwrap_or(0) < MIN_BYTES {
            return false;
        }
        let mut magic = [0u8; 4];
        let mut file = file;
        file.read_exact(&mut magic).is_ok() && magic == *b"GQ4G"
    }

    fn is_stale_split_weights_cache(path: &Path) -> bool {
        use std::io::Read;
        let mut file = match fs::File::open(path) {
            Ok(file) => file,
            Err(_) => return false,
        };
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).is_ok() && (magic == *b"GQ4W" || magic == *b"GQ4A")
    }

    /// Read the 4-byte magic from a weights cache file.
    fn read_weights_magic(path: &Path) -> Option<[u8; 4]> {
        use std::io::Read;
        let mut file = fs::File::open(path).ok()?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).ok()?;
        Some(magic)
    }

    /// True if the file is a GGUF-compatible weights cache
    /// (GQ4G: all-Q4_0 layout, GQ4H: with per-tensor format tags).
    fn is_gguf_weights_cache(path: &Path) -> bool {
        Self::read_weights_magic(path)
            .map_or(false, |m| m == *b"GQ4G" || m == *b"GQ4H")
    }

    fn is_legacy_weights_cache(path: &Path) -> bool {
        use std::io::Read;
        let mut file = match fs::File::open(path) {
            Ok(file) => file,
            Err(_) => return false,
        };
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).is_ok() && magic == *b"GQ4C"
    }

    fn finish_cache_load(
        ctx: MetalContext,
        config: Gemma4TextConfig,
        embed_tables: EmbedTables,
        weights_mmap_offset: Option<usize>,
        lm_head_buf: BufferView,
        per_layer_model_projection_weight: BufferView,
        final_norm_weight: BufferView,
        per_layer_projection_norm_weight: BufferView,
        layers: Vec<Gemma4GpuLayer>,
        load_start: Instant,
        load_label: &str,
    ) -> Self {
        let layers = Self::pack_layers_prefill_stacked(&ctx, layers, config.hidden_size as u32);
        let max_intermediate_size = layers
            .iter()
            .map(|layer| layer.intermediate_size)
            .max()
            .unwrap_or_else(|| config.max_intermediate_size());
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let vocab_size = config.vocab_size;
        let num_layers = config.num_hidden_layers;
        let ple_dim = config.hidden_size_per_layer_input;
        let max_head_dim = config.global_head_dim;
        let max_q_out = num_heads * max_head_dim;
        let max_kv_out = num_kv_heads * max_head_dim;

        let hidden_buf = ctx.buffer_empty(hidden_size);
        let normed_buf = ctx.buffer_empty(hidden_size);
        let residual_buf = ctx.buffer_empty(hidden_size);
        let q_buf = ctx.buffer_empty(max_q_out);
        let k_buf = ctx.buffer_empty(max_kv_out);
        let v_buf = ctx.buffer_empty(max_kv_out);
        let attn_out_buf = ctx.buffer_empty(max_q_out);
        let o_out_buf = ctx.buffer_empty(hidden_size);
        let gate_buf = ctx.buffer_empty(max_intermediate_size);
        let up_buf = ctx.buffer_empty(max_intermediate_size);
        let gelu_buf = ctx.buffer_empty(max_intermediate_size);
        let down_buf = ctx.buffer_empty(hidden_size);
        let logits_buf = ctx.buffer_empty(vocab_size);
        let sample_out_buf = ctx.buffer_empty_u32(1);
        let inv_rms_buf = ctx.buffer_empty(1);
        let ple_embed_buf = ctx.buffer_empty(ple_dim);
        let ple_gated_buf = ctx.buffer_empty(ple_dim);
        let ple_normed_buf = ctx.buffer_empty(ple_dim);
        let ple_projected_buf = ctx.buffer_empty(hidden_size);
        let ple_context_proj_buf = ctx.buffer_empty(num_layers * ple_dim);
        let ple_token_id_buf = ctx.buffer_empty(num_layers * ple_dim);
        let ple_combined_buf = ctx.buffer_empty(num_layers * ple_dim);
        let q_normed_buf = ctx.buffer_empty(max_q_out);
        let k_normed_buf = ctx.buffer_empty(max_kv_out);
        let ggml_fa_tmp_elems = (crate::ggml_flash_attn::flash_attn_tmp_bytes(
            num_heads as u32,
            max_head_dim as u32,
        ) / 4) as usize;
        let ggml_fa_tmp_buf = ctx.buffer_empty(ggml_fa_tmp_elems);

        let kv_cache_type = KvCacheType::from_env();
        let kv_capacity = configured_kv_capacity(config.max_position_embeddings);
        let mut k_cache = Vec::with_capacity(num_layers);
        let mut v_cache = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let hd = config.layer_head_dim(i);
            assert!(hd % 32 == 0, "head_dim must be a multiple of 32 for quantized KV cache");
            // K and V may use different bit-widths (asymmetric TurboQuant).
            let k_byte_len = (num_kv_heads * kv_capacity as usize * kv_cache_type.k_row_bytes(hd)) as u64;
            let v_byte_len = (num_kv_heads * kv_capacity as usize * kv_cache_type.v_row_bytes(hd)) as u64;
            k_cache.push(
                ctx.device
                    .new_buffer(k_byte_len, MTLResourceOptions::StorageModeShared),
            );
            v_cache.push(
                ctx.device
                    .new_buffer(v_byte_len, MTLResourceOptions::StorageModeShared),
            );
        }
        let tq_state = build_turboquant_state(&ctx, &config, kv_cache_type, kv_capacity);
        let TurboQuantState {
            turboquant,
            tq_q_rot,
            tq_k_rot,
            tq_v_rot,
            tq_out,
            tq_scores,
            tq_rw_k,
            tq_rw_v,
            tq_rw,
            tq_hot_k,
            tq_hot_v,
            tq_hot_w,
        } = tq_state;
        let f16_bytes = num_kv_heads * kv_capacity as usize * config.head_dim * 2
            + num_kv_heads * kv_capacity as usize * config.global_head_dim * 2;
        let quant_bytes = num_kv_heads * kv_capacity as usize
            * (kv_cache_type.k_row_bytes(config.head_dim) + kv_cache_type.v_row_bytes(config.head_dim))
            / 2
            + num_kv_heads * kv_capacity as usize
            * (kv_cache_type.k_row_bytes(config.global_head_dim) + kv_cache_type.v_row_bytes(config.global_head_dim))
            / 2;
        println!(
            "  KV cache type: {}, est. memory per layer: {:.1} MB (vs f16: {:.1} MB, {:.0}% savings)",
            kv_cache_type,
            quant_bytes as f64 / num_layers as f64 / 1024.0 / 1024.0,
            f16_bytes as f64 / num_layers as f64 / 1024.0 / 1024.0,
            (1.0 - quant_bytes as f64 / f16_bytes as f64) * 100.0,
        );
        let max_prefill_seq = configured_max_prefill_seq(kv_capacity);
        let prefill_scratch = PrefillScratch::new(
            &ctx,
            max_prefill_seq,
            hidden_size,
            max_q_out,
            max_kv_out,
            max_intermediate_size,
            vocab_size,
            num_layers,
            ple_dim,
            kv_capacity,
            num_kv_heads as u32,
            max_head_dim as u32,
        );
        let decode_batch_scratch = DecodeBatchScratch::new(
            &ctx,
            DEFAULT_MAX_DECODE_BATCH,
            hidden_size,
            max_q_out,
            max_kv_out,
            max_intermediate_size,
            vocab_size,
            num_layers,
            ple_dim,
            num_heads as u32,
            max_head_dim as u32,
        );

        let cos_buf = ctx.buffer_empty(max_head_dim);
        let sin_buf = ctx.buffer_empty(max_head_dim);

        let (decode_rope_cos_packed, decode_rope_sin_packed, rope_layer_params_buf) =
            alloc_decode_rope_buffers(&ctx, &config, num_layers, max_head_dim);
        let mut per_layer_prefill_cos_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_prefill_sin_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_decode_batch_cos_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_decode_batch_sin_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_decode_batch_append_pos_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_decode_batch_kv_start_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_decode_batch_kv_len_bufs = Vec::with_capacity(num_layers);
        let mut per_layer_ple_bufs = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let hd = config.layer_head_dim(i);
            per_layer_prefill_cos_bufs.push(ctx.buffer_empty(max_prefill_seq * hd));
            per_layer_prefill_sin_bufs.push(ctx.buffer_empty(max_prefill_seq * hd));
            per_layer_decode_batch_cos_bufs.push(ctx.buffer_empty(DEFAULT_MAX_DECODE_BATCH * hd));
            per_layer_decode_batch_sin_bufs.push(ctx.buffer_empty(DEFAULT_MAX_DECODE_BATCH * hd));
            per_layer_decode_batch_append_pos_bufs
                .push(ctx.buffer_empty_u32(DEFAULT_MAX_DECODE_BATCH));
            per_layer_decode_batch_kv_start_bufs
                .push(ctx.buffer_empty_u32(DEFAULT_MAX_DECODE_BATCH));
            per_layer_decode_batch_kv_len_bufs.push(ctx.buffer_empty_u32(DEFAULT_MAX_DECODE_BATCH));
            per_layer_ple_bufs.push(ctx.buffer_empty(ple_dim.max(1)));
        }

        if layers.iter().any(|l| l.weight_format == WeightFormat::F16) {
            eprintln!(
                "  Warning: weights cache has f16 layers; delete model.q4cache for all-Q4"
            );
        }

        println!(
            "  Parallel prefill scratch: max_seq={}",
            prefill_scratch.max_seq_len
        );
        println!(
            "  Decode batch scratch: max_batch={}",
            decode_batch_scratch.max_batch_size
        );
        println!(
            "  Loaded from {} in {:.2}s",
            load_label,
            load_start.elapsed().as_secs_f64()
        );

        // MTP verify scratch buffers
        let mtp_verify_logits_buf =
            ctx.buffer_empty(MAX_MTP_VERIFY_SEQ * vocab_size);
        let mtp_verify_argmax_buf = ctx.buffer_empty_u32(MAX_MTP_VERIFY_SEQ);
        let mtp_verify_hidden_buf =
            ctx.buffer_empty(MAX_MTP_VERIFY_SEQ * hidden_size);
        let mtp_head_dims: Vec<usize> = (0..num_layers).map(|i| config.layer_head_dim(i)).collect();
        let mtp_verify_scratch = MtpVerifyScratch::new(
            &ctx,
            MAX_MTP_VERIFY_SEQ,
            hidden_size,
            (num_layers * ple_dim).max(1),
            &mtp_head_dims,
        );

        let model = Gemma4GpuModel {
            ctx,
            config,
            embed_tables,
            lm_head_buf,
            layers,
            final_norm_weight,
            per_layer_projection_norm_weight,
            per_layer_model_projection_weight,
            hidden_buf,
            normed_buf,
            residual_buf,
            q_buf,
            k_buf,
            v_buf,
            attn_out_buf,
            o_out_buf,
            gate_buf,
            up_buf,
            gelu_buf,
            down_buf,
            logits_buf,
            sample_out_buf,
            inv_rms_buf,
            prefill_scratch,
            decode_batch_scratch,
            mtp_verify_logits_buf,
            mtp_verify_argmax_buf,
            mtp_verify_hidden_buf,
            mtp_verify_scratch,
            ple_embed_buf,
            ple_gated_buf,
            ple_normed_buf,
            ple_projected_buf,
            ple_context_proj_buf,
            ple_token_id_buf,
            ple_combined_buf,
            q_normed_buf,
            k_normed_buf,
            ggml_fa_tmp_buf,
            k_cache,
            v_cache,
            kv_seq_len: 0,
            kv_capacity,
            kv_cache_type,
            turboquant,
            tq_q_rot,
            tq_k_rot,
            tq_v_rot,
            tq_out,
            tq_scores,
            tq_rw_k,
            tq_rw_v,
            tq_rw,
            tq_hot_k,
            tq_hot_v,
            tq_hot_w,
            tq_hot_spilled: false,
            cos_buf,
            sin_buf,
            decode_rope_cos_packed,
            decode_rope_sin_packed,
            rope_layer_params_buf,
            rope_max_head_dim: max_head_dim,
            per_layer_prefill_cos_bufs,
            per_layer_prefill_sin_bufs,
            per_layer_decode_batch_cos_bufs,
            per_layer_decode_batch_sin_bufs,
            per_layer_decode_batch_append_pos_bufs,
            per_layer_decode_batch_kv_start_bufs,
            per_layer_decode_batch_kv_len_bufs,
            per_layer_ple_bufs,
            total_tokens: 0,
            weights_mmap_offset,
            embed_decode_scratch: vec![0.0f32; hidden_size],
            ple_decode_scratch: vec![0.0f32; (num_layers * ple_dim).max(1)],
        };
        crate::decode_fused::log_fused_decode_status(&model);
        model
    }

    /// Build interleaved gate∥up Q4 buffers for decode MLP (uzu-style packing).
    /// Build stacked prefill weight buffers (gate∥up, Q∥K∥V) at load time.
    fn pack_layers_prefill_stacked(
        ctx: &MetalContext,
        mut layers: Vec<Gemma4GpuLayer>,
        k: u32,
    ) -> Vec<Gemma4GpuLayer> {
        let pack_interleaved = crate::gpu::packed_mlp_gate_up_enabled()
            || crate::decode_fused::fused_decode_enabled();
        let pack_gate_up = crate::gpu::prefill_gate_up_stacked_enabled();
        let pack_qkv = crate::gpu::prefill_qkv_stacked_enabled();
        if !pack_interleaved && !pack_gate_up && !pack_qkv {
            for layer in layers.iter_mut() {
                layer.gate_up_proj = BufferView::from_buffer(ctx.buffer_empty(1));
                layer.gate_up_stacked = BufferView::from_buffer(ctx.buffer_empty(1));
                layer.qkv_stacked = BufferView::from_buffer(ctx.buffer_empty(1));
            }
            return layers;
        }
        let pack_start = std::time::Instant::now();
        for layer in layers.iter_mut() {
            if pack_interleaved && Self::use_packed_mlp_gate_up(layer) {
                layer.gate_up_proj = ctx.pack_gate_up_interleaved_q4(
                    &layer.gate_proj,
                    &layer.up_proj,
                    layer.intermediate_size as u32,
                    k,
                );
            } else {
                layer.gate_up_proj = BufferView::from_buffer(ctx.buffer_empty(1));
            }
            if pack_gate_up && Self::use_prefill_gate_up_stacked(layer) {
                layer.gate_up_stacked = ctx.pack_gate_up_stacked_kquant(
                    &layer.gate_proj,
                    &layer.up_proj,
                    layer.intermediate_size as u32,
                    k,
                );
            } else {
                layer.gate_up_stacked = BufferView::from_buffer(ctx.buffer_empty(1));
            }
            if pack_qkv && Self::use_prefill_qkv_stacked(layer) {
                layer.qkv_stacked = ctx.pack_qkv_stacked_kquant(
                    &layer.q_proj,
                    &layer.k_proj,
                    &layer.v_proj,
                    layer.q_out_dim as u32,
                    layer.kv_out_dim as u32,
                    k,
                );
            } else {
                layer.qkv_stacked = BufferView::from_buffer(ctx.buffer_empty(1));
            }
        }
        println!(
            "  Packed prefill stacked weights (gate∥up + Q∥K∥V) in {:.2}s",
            pack_start.elapsed().as_secs_f64()
        );
        layers
    }

    pub(crate) fn use_packed_mlp_gate_up(layer: &Gemma4GpuLayer) -> bool {
        (crate::gpu::packed_mlp_gate_up_enabled() || crate::decode_fused::fused_decode_enabled())
            && layer.weight_format == WeightFormat::Q4_0
    }

    pub(crate) fn use_prefill_gate_up_stacked(layer: &Gemma4GpuLayer) -> bool {
        use crate::gpu::weight_fmt;
        crate::gpu::prefill_gate_up_stacked_enabled()
            && layer.gate_proj.format == weight_fmt::Q4_K
            && layer.up_proj.format == weight_fmt::Q4_K
    }

    pub(crate) fn use_prefill_qkv_stacked(layer: &Gemma4GpuLayer) -> bool {
        use crate::gpu::weight_fmt;
        crate::gpu::prefill_qkv_stacked_enabled()
            && layer.has_kv
            && layer.q_proj.format == weight_fmt::Q4_K
            && layer.k_proj.format == weight_fmt::Q4_K
            && layer.v_proj.format == weight_fmt::Q4_K
    }

    fn encode_prefill_attention_qkv(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        layer: &Gemma4GpuLayer,
        seq_len: u32,
        hidden_size: u32,
    ) {
        let q_out = layer.q_out_dim as u32;
        let kv_out = layer.kv_out_dim as u32;
        if layer.weight_format == WeightFormat::F16 {
            self.ctx.encode_projection_f16_batch_view(
                encoder,
                &layer.q_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.q_buf,
                q_out,
                hidden_size,
                seq_len,
            );
            if layer.has_kv {
                self.ctx.encode_projection_f16_batch_view(
                    encoder,
                    &layer.k_proj,
                    &self.prefill_scratch.normed_buf,
                    &self.prefill_scratch.k_buf,
                    kv_out,
                    hidden_size,
                    seq_len,
                );
                self.ctx.encode_projection_f16_batch_view(
                    encoder,
                    &layer.v_proj,
                    &self.prefill_scratch.normed_buf,
                    &self.prefill_scratch.v_buf,
                    kv_out,
                    hidden_size,
                    seq_len,
                );
            }
        } else if Self::use_prefill_qkv_stacked(layer) {
            self.ctx.encode_prefill_qkv_kquant_stacked(
                encoder,
                &layer.qkv_stacked,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.qkv_stacked_buf,
                q_out,
                kv_out,
                hidden_size,
                seq_len,
            );
        } else {
            self.ctx.encode_prefill_projection_q4_batch_view(
                encoder,
                &layer.q_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.q_buf,
                q_out,
                hidden_size,
                seq_len,
            );
            if layer.has_kv {
                self.ctx.encode_prefill_projection_q4_batch_view(
                    encoder,
                    &layer.k_proj,
                    &self.prefill_scratch.normed_buf,
                    &self.prefill_scratch.k_buf,
                    kv_out,
                    hidden_size,
                    seq_len,
                );
                self.ctx.encode_prefill_projection_q4_batch_view(
                    encoder,
                    &layer.v_proj,
                    &self.prefill_scratch.normed_buf,
                    &self.prefill_scratch.v_buf,
                    kv_out,
                    hidden_size,
                    seq_len,
                );
            }
        }
    }

    fn encode_prefill_mlp_gate_up(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        layer: &Gemma4GpuLayer,
        seq_len: u32,
        intermediate_size: u32,
        hidden_size: u32,
        skip_gelu: bool,
    ) {
        let total_intermediate = seq_len * intermediate_size;
        // f16 activations only feed the mul_mm path; the small-seq matvec
        // fallback reads x as f32, so casting there corrupts the MLP.
        let use_f16 = crate::gpu::prefill_mlp_f16_enabled()
            && !crate::gpu::ProfileAblate::from_env().skip_cast()
            && layer.weight_format != WeightFormat::F16
            && crate::gpu::prefill_mul_mm_enabled()
            && crate::ggml_gemv::should_use_mul_mm(hidden_size, seq_len);
        // residual_buf unused during MLP after fused residual path; holds f16 normed.
        // up_buf unused on stacked path; holds f16 gelu for down proj.
        if layer.weight_format == WeightFormat::F16 {
            self.ctx.encode_projection_f16_batch_view(
                encoder,
                &layer.gate_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.gate_buf,
                intermediate_size,
                hidden_size,
                seq_len,
            );
            self.ctx.encode_projection_f16_batch_view(
                encoder,
                &layer.up_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.up_buf,
                intermediate_size,
                hidden_size,
                seq_len,
            );
            if !skip_gelu {
                self.ctx.encode_gelu_mul(
                    encoder,
                    &self.prefill_scratch.gate_buf,
                    &self.prefill_scratch.up_buf,
                    &self.prefill_scratch.gelu_buf,
                    total_intermediate,
                );
            }
        } else if Self::use_prefill_gate_up_stacked(layer) {
            let x_buf = if use_f16 {
                self.ctx.encode_cast_f32_to_f16(
                    encoder,
                    &self.prefill_scratch.normed_buf,
                    &self.prefill_scratch.residual_buf,
                    seq_len * hidden_size,
                );
                &self.prefill_scratch.residual_buf
            } else {
                &self.prefill_scratch.normed_buf
            };
            // MTP verify / small-batch: fused gate∥up+GeLU ext matvec shares
            // activation loads and writes gelu directly (skips 2·M·batch scratch).
            let use_ext_gelu = crate::gpu::prefill_gate_up_ext_gelu_enabled()
                && !use_f16
                && !skip_gelu
                && seq_len >= 2
                && seq_len <= 8
                && layer.gate_proj.format == crate::gpu::weight_fmt::Q4_K
                && layer.up_proj.format == crate::gpu::weight_fmt::Q4_K
                && !crate::ggml_gemv::should_use_mul_mm(hidden_size, seq_len);
            if use_ext_gelu {
                self.ctx.encode_matvec_kq_ext_gelu_mul_at_view(
                    encoder,
                    &layer.gate_proj,
                    &layer.up_proj,
                    x_buf,
                    &self.prefill_scratch.gelu_buf,
                    intermediate_size,
                    hidden_size,
                    seq_len,
                );
            } else {
                self.ctx.encode_prefill_gate_up_kquant_stacked(
                    encoder,
                    &layer.gate_up_stacked,
                    x_buf,
                    &self.prefill_scratch.gate_buf,
                    &self.prefill_scratch.gelu_buf,
                    intermediate_size,
                    hidden_size,
                    seq_len,
                    use_f16,
                    skip_gelu,
                );
            }
        } else {
            let x_buf = if use_f16 {
                self.ctx.encode_cast_f32_to_f16(
                    encoder,
                    &self.prefill_scratch.normed_buf,
                    &self.prefill_scratch.residual_buf,
                    seq_len * hidden_size,
                );
                &self.prefill_scratch.residual_buf
            } else {
                &self.prefill_scratch.normed_buf
            };
            if use_f16 {
                self.ctx.encode_mul_mm_kquant_f16_at_view(
                    encoder,
                    &layer.gate_proj,
                    x_buf,
                    &self.prefill_scratch.gate_buf,
                    intermediate_size,
                    hidden_size,
                    seq_len,
                );
                self.ctx.encode_mul_mm_kquant_f16_at_view(
                    encoder,
                    &layer.up_proj,
                    x_buf,
                    &self.prefill_scratch.up_buf,
                    intermediate_size,
                    hidden_size,
                    seq_len,
                );
            } else {
                self.ctx.encode_prefill_projection_q4_batch_view(
                    encoder,
                    &layer.gate_proj,
                    x_buf,
                    &self.prefill_scratch.gate_buf,
                    intermediate_size,
                    hidden_size,
                    seq_len,
                );
                self.ctx.encode_prefill_projection_q4_batch_view(
                    encoder,
                    &layer.up_proj,
                    x_buf,
                    &self.prefill_scratch.up_buf,
                    intermediate_size,
                    hidden_size,
                    seq_len,
                );
            }
            if !skip_gelu {
                self.ctx.encode_gelu_mul(
                    encoder,
                    &self.prefill_scratch.gate_buf,
                    &self.prefill_scratch.up_buf,
                    &self.prefill_scratch.gelu_buf,
                    total_intermediate,
                );
            }
        }
    }

    /// Infer MLP output width (rows) from a gate projection buffer.
    fn infer_mlp_intermediate(gate: &BufferView, hidden_size: usize) -> usize {
        use crate::gpu::weight_fmt;
        let (epb, bpb) = match gate.format {
            weight_fmt::Q4_K => (256, 144),
            weight_fmt::Q6_K => (256, 210),
            _ => (32, 18),
        };
        let blocks = gate.length as usize / bpb;
        blocks * epb / hidden_size
    }

    /// Dispatch a plain quantized matvec for a layer's weight (Q4 or Q3).
    fn encode_matvec_quant(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &metal::Buffer,
        y_buf: &metal::Buffer,
        m: u32,
        k: u32,
        wf: WeightFormat,
    ) {
        self.encode_matvec_quant_at(encoder, weight, x_buf, 0, y_buf, 0, m, k, wf);
    }

    /// Offset-aware variant of `encode_matvec_quant` for the batched decode path,
    /// where activations for each token live at a per-token offset in a shared
    /// scratch buffer.
    #[allow(clippy::too_many_arguments)]
    fn encode_matvec_quant_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &metal::Buffer,
        x_offset: u64,
        y_buf: &metal::Buffer,
        y_offset: u64,
        m: u32,
        k: u32,
        wf: WeightFormat,
    ) {
        // K-quant is decided per tensor (Q4_K_M mixes Q4_K/Q6_K within a layer),
        // so the per-tensor `BufferView::format` wins over the per-layer `wf`.
        use crate::gpu::weight_fmt;
        match weight.format {
            weight_fmt::Q4_K | weight_fmt::Q6_K => self.ctx.encode_matvec_qk_at_view(
                encoder, weight, x_buf, x_offset, y_buf, y_offset, m, k, 1,
            ),
            _ => match wf {
                WeightFormat::Q3_0 => self.ctx.encode_matvec_q3_at_view(
                    encoder, weight, x_buf, x_offset, y_buf, y_offset, m, k,
                ),
                _ => self.ctx.encode_matvec_q4_at_view(
                    encoder, weight, x_buf, x_offset, y_buf, y_offset, m, k,
                ),
            },
        }
    }

    fn encode_mlp_gate_up_gelu_q4_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        layer: &Gemma4GpuLayer,
        x_buf: &metal::Buffer,
        gelu_buf: &metal::Buffer,
        intermediate_size: u32,
        hidden_size: u32,
    ) {
        self.encode_mlp_gate_up_gelu_q4_at_view(
            encoder,
            layer,
            x_buf,
            0,
            gelu_buf,
            0,
            intermediate_size,
            hidden_size,
        );
    }

    fn encode_mlp_gate_up_gelu_q4_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        layer: &Gemma4GpuLayer,
        x_buf: &metal::Buffer,
        x_offset: u64,
        gelu_buf: &metal::Buffer,
        gelu_offset: u64,
        intermediate_size: u32,
        hidden_size: u32,
    ) {
        if layer.weight_format.is_q3() {
            // Q3_0: use ggml fused gelu_mul kernel
            self.ctx.encode_matvec_ggml_q3_gelu_mul_at_view(
                encoder,
                &layer.gate_proj,
                &layer.up_proj,
                x_buf,
                x_offset,
                gelu_buf,
                gelu_offset,
                intermediate_size,
                hidden_size,
            );
        } else if Self::use_packed_mlp_gate_up(layer) {
            self.ctx.encode_matvec_q4_interleaved_gelu_at_view(
                encoder,
                &layer.gate_up_proj,
                x_buf,
                x_offset,
                gelu_buf,
                gelu_offset,
                intermediate_size,
                hidden_size,
            );
        } else {
            self.ctx.encode_matvec_q4_dual_gelu_at_view(
                encoder,
                &layer.gate_proj,
                &layer.up_proj,
                x_buf,
                x_offset,
                gelu_buf,
                gelu_offset,
                intermediate_size,
                hidden_size,
            );
        }
    }

    /// Split cache: load weights first, then mmap embeddings.
    fn load_from_split_cache(
        model_dir: &str,
        embed_cache_path: &Path,
        weights_cache_path: &Path,
    ) -> Self {
        let load_start = Instant::now();

        let config_path = Path::new(model_dir).join("config.json");
        let config_str = fs::read_to_string(&config_path).expect("Failed to read config.json");
        let outer: serde_json::Value =
            serde_json::from_str(&config_str).expect("Failed to parse config.json");
        let config: Gemma4TextConfig = if let Some(tc) = outer.get("text_config") {
            serde_json::from_value(tc.clone()).expect("Failed to parse text_config")
        } else {
            serde_json::from_str(&config_str).expect("Failed to parse config")
        };

        let ctx = MetalContext::new();
        let device = &ctx.device;
        let num_layers = config.num_hidden_layers;

        let mut weights_file =
            fs::File::open(weights_cache_path).expect("Failed to open weights cache");
        let file_len = weights_file
            .metadata()
            .expect("weights metadata")
            .len();
        let weights_bytes = file_len.saturating_sub(4) as u64;

        let mut magic = [0u8; 4];
        use std::io::Read;
        weights_file
            .read_exact(&mut magic)
            .expect("read weights cache magic");
        let aligned_layout = magic == *b"GQ4G";
        if magic == *b"GQ4W" || magic == *b"GQ4A" {
            panic!(
                "Stale weights cache (old interleaved Q4 layout). \
                 Delete model.q4cache and model.embed.cache, then re-run to re-quantize."
            );
        }
        if !aligned_layout {
            panic!("Invalid weights cache magic (expected GQ4G)");
        }

        let mut offset = 4;
        let (
            lm_head_buf,
            per_layer_model_projection_weight,
            final_norm_weight,
            per_layer_projection_norm_weight,
            layers,
        ) = if aligned_layout {
            println!(
                "  Copying weights to GPU: {:.1} MB (aligned layout)...",
                weights_bytes as f64 / 1024.0 / 1024.0
            );
            let copy_start = Instant::now();
            let weights_buf =
                MetalContext::buffer_read_from_file(device, &mut weights_file, 4, weights_bytes);
            println!(
                "  Weights copied in {:.2}s",
                copy_start.elapsed().as_secs_f64()
            );
            let section = unsafe {
                std::slice::from_raw_parts(weights_buf.contents() as *const u8, weights_bytes as usize)
            };
            let mut section_offset = 0;
            Self::load_weight_sections(
                section,
                &mut section_offset,
                device,
                num_layers,
                config.hidden_size,
                true,
                Some(&weights_buf),
                0,
            )
        } else {
            let weights_mmap =
                unsafe { Mmap::map(&weights_file).expect("Failed to mmap weights cache") };
            println!("  Loading weights (per-tensor copy)...");
            let weights_load_start = Instant::now();
            let sections = Self::load_weight_sections(
                weights_mmap.as_ref(),
                &mut offset,
                device,
                num_layers,
                config.hidden_size,
                false,
                None,
                4,
            );
            println!(
                "  Weights loaded in {:.2}s",
                weights_load_start.elapsed().as_secs_f64()
            );
            sections
        };

        let embed_file = fs::File::open(embed_cache_path).expect("Failed to open embed cache");
        let embed_mmap = unsafe { Mmap::map(&embed_file).expect("Failed to mmap embed cache") };
        assert_eq!(&embed_mmap[0..4], b"GQ4E", "Invalid embed cache magic");
        let mut eoff = 4;
        let embed_len = Self::read_u64_at(&embed_mmap, &mut eoff);
        let embed_offset = eoff;
        eoff += embed_len as usize;
        println!(
            "    embed_tokens: {:.1} MB (mmap)",
            embed_len as f64 / 1024.0 / 1024.0
        );
        let ple_len = Self::read_u64_at(&embed_mmap, &mut eoff);
        let ple_offset = eoff;
        println!(
            "    embed_tokens_per_layer: {:.1} MB (mmap)",
            ple_len as f64 / 1024.0 / 1024.0
        );
        let ple_cols = num_layers * config.hidden_size_per_layer_input;
        let embed_tables = EmbedTables::from_mmap(
            embed_mmap,
            embed_offset,
            embed_len as usize,
            ple_offset,
            ple_len as usize,
            ple_cols,
            config.vocab_size,
        );

        let model = Self::finish_cache_load(
            ctx,
            config,
            embed_tables,
            None,
            lm_head_buf,
            per_layer_model_projection_weight,
            final_norm_weight,
            per_layer_projection_norm_weight,
            layers,
            load_start,
            "split Q4 cache",
        );

        if !aligned_layout {
            println!("  Upgrading weights cache to aligned format for faster loads...");
            model.save_weights_cache(weights_cache_path);
        }

        model
    }

    /// Forward one token through the entire model. One command buffer per layer.
    pub fn forward_single_token(&mut self, token_id: usize) -> Vec<f32> {
        match self.forward_single_token_inner(token_id, DecodeMode::Logits) {
            DecodeOutput::Logits(logits) => logits,
            DecodeOutput::Token(_) | DecodeOutput::Advanced => {
                unreachable!("Logits mode must return logits")
            }
        }
    }

    /// KV-only prefill step: runs layers but skips lm_head and logit readback.
    pub fn forward_advance(&mut self, token_id: usize) {
        match self.forward_single_token_inner(token_id, DecodeMode::Advance) {
            DecodeOutput::Advanced => {}
            _ => unreachable!("Advance mode must return Advanced"),
        }
    }

    /// Fused decode step: runs the full forward pass and GPU-side sampling in a
    /// single command buffer, returning the sampled token id. Only 4 bytes are
    /// read back (the token), avoiding the full-vocab logits readback + CPU
    /// softmax that `forward_single_token` incurs.
    pub fn forward_single_token_sample(
        &mut self,
        token_id: usize,
        temperature: f32,
        min_p: f32,
        seed: u32,
    ) -> usize {
        match self.forward_single_token_inner(
            token_id,
            DecodeMode::Sample(temperature, min_p, seed),
        ) {
            DecodeOutput::Token(token) => token,
            _ => unreachable!("Sample mode must return a token"),
        }
    }

    fn forward_single_token_inner(
        &mut self,
        token_id: usize,
        mode: DecodeMode,
    ) -> DecodeOutput {
        let __profile = std::env::var("PROFILE_DECODE").is_ok();
        // When set, splits the token into per-phase command buffers and times
        // each phase's commit→wait (wall clock). Each phase is its own command
        // buffer dominated by GPU execution, so the numbers isolate where the
        // floor is; they include a small fixed sync per phase (note in output).
        let __pp = std::env::var("PROFILE_PHASES").is_ok();
        let __ablate = ProfileAblate::from_env();
        let layer_count = self.layers.len() as u32;
        let mut __gpu_prof = if !__pp && !__ablate.active() && profile_gpu_enabled() {
            GpuTimestampProfiler::try_new(&self.ctx.device, layer_count)
        } else {
            None
        };
        if __gpu_prof.is_some() {
            eprintln!("  GPU timestamp profiling enabled (PROFILE_GPU=1)");
        }
        __ablate.log_once();
        let __t0 = std::time::Instant::now();
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let num_kv_groups = (num_heads / num_kv_heads) as u32;
        let intermediate_size = self.config.max_intermediate_size();
        let vocab_size = self.config.vocab_size;
        let eps = self.config.rms_norm_eps as f32;
        let ple_dim = self.config.hidden_size_per_layer_input;
        let num_layers = self.config.num_hidden_layers;
        let ple_total_dim = num_layers * ple_dim;

        // CPU embedding lookup (mmap-backed, no GPU gather)
        self.embed_tables
            .decode_embed_into(token_id, hidden_size, &mut self.embed_decode_scratch);
        MetalContext::write_buffer(&self.hidden_buf, &self.embed_decode_scratch);
        if ple_dim > 0 {
            self.embed_tables.decode_ple_into(
                token_id,
                ple_total_dim,
                ple_dim,
                &mut self.ple_decode_scratch,
            );
            MetalContext::write_buffer(&self.ple_token_id_buf, &self.ple_decode_scratch);
        }

        let kv_seq = self.kv_seq_len;
        let pos = self.total_tokens as f32;

        // First token past the Q4 hot window: pack hot → TQ cold once.
        if self.tq_needs_spill(kv_seq) {
            self.spill_tq_hot_to_cold();
        }

        // ─── PLE: Compute per-layer inputs on GPU ───
        let ple_input_scale = std::f32::consts::FRAC_1_SQRT_2; // 1/sqrt(2)
        let context_proj_scale = 1.0 / (hidden_size as f32).sqrt();

        let actual_num_layers = self.layers.len();
        let __t_prep = std::time::Instant::now();
        // `mut` so the phase profiler (or METAL_N_CB>=2) can flush and re-open
        // command buffers mid-token. Default: rope+PLE+layers[0..mid] on CB0,
        // layers[mid..]+head on CB1 — commit CB0 without wait so encode of CB1
        // overlaps GPU execution of CB0.
        let mut cmd = self.ctx.queue.new_command_buffer();
        let mut encoder = cmd.new_compute_command_encoder();

        // GPU RoPE table fill for all layers (replaces CPU sin/cos + write_buffer).
        self.ctx.encode_rope_fill_decode(
            &encoder,
            &self.decode_rope_cos_packed,
            &self.decode_rope_sin_packed,
            &self.rope_layer_params_buf,
            actual_num_layers as u32,
            self.rope_max_head_dim as u32,
            pos,
        );
        if let Some(p) = &mut __gpu_prof {
            p.mark(&encoder);
        }

        // ─── PLE pre-pass ───
        // Produces the per-layer PLE inputs contiguously in ple_context_proj_buf;
        // each layer reads its slice directly (byte offset = layer_idx * ple_dim
        // * 4), so the previous 42 per-layer copy-out dispatches are gone.
        if !__ablate.skip_ple() {
            // Step 2a: context_proj = per_layer_model_projection @ embed
            self.ctx.encode_matvec_auto_view(
                encoder,
                &self.per_layer_model_projection_weight,
                &self.hidden_buf,
                &self.ple_context_proj_buf,
                ple_total_dim as u32,
                hidden_size as u32,
            );
            // Step 2b: context_proj *= 1/sqrt(hidden_size)
            self.ctx.encode_vec_scale(
                encoder,
                &self.ple_context_proj_buf,
                &self.ple_combined_buf,
                ple_total_dim as u32,
                context_proj_scale,
            );
            // Step 2c: RMSNorm per layer
            self.ctx.encode_rmsnorm_per_head_view(
                encoder,
                &self.ple_combined_buf,
                &self.per_layer_projection_norm_weight,
                &self.ple_context_proj_buf,
                num_layers as u32,
                ple_dim as u32,
                eps,
            );
            // Step 3: combined = (context_proj + token_identity) * 1/sqrt(2)
            self.ctx.encode_vec_add(
                encoder,
                &self.ple_context_proj_buf,
                &self.ple_token_id_buf,
                &self.ple_combined_buf,
                ple_total_dim as u32,
            );
            self.ctx.encode_vec_scale(
                encoder,
                &self.ple_combined_buf,
                &self.ple_context_proj_buf,
                ple_total_dim as u32,
                ple_input_scale,
            );
        }
        if let Some(p) = &mut __gpu_prof {
            p.mark(&encoder);
        }

        // Phase boundary: PLE pre-pass (profiling only).
        if __pp {
            encoder.end_encoding();
            let __pt = std::time::Instant::now();
            cmd.commit();
            cmd.wait_until_completed();
            Self::phase_accum(0, __pt.elapsed().as_secs_f64() * 1e3);
            cmd = self.ctx.queue.new_command_buffer();
            encoder = cmd.new_compute_command_encoder();
        }

        let use_fused_decode = self.fused_decode_eligible()
            && !__pp
            && !__ablate.active()
            && __gpu_prof.is_none();

        let metal_n_cb = if use_fused_decode {
            1
        } else if !__pp && !__ablate.active() && __gpu_prof.is_none() {
            crate::gpu::metal_n_cb()
        } else {
            1
        };
        let cb_layer_splits: Vec<usize> = if metal_n_cb >= 2 {
            (1..metal_n_cb as usize)
                .map(|i| actual_num_layers * i / metal_n_cb as usize)
                .collect()
        } else {
            Vec::new()
        };
        let mut cb_split_idx = 0usize;

        if use_fused_decode {
            static FUSED_DECODE_USED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !FUSED_DECODE_USED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "  Decode: fused layer executor active (~{} layers, single CB)",
                    actual_num_layers
                );
            }
            let scratch = crate::decode_fused::FusedDecodeScratch {
                hidden: &self.hidden_buf,
                normed: &self.normed_buf,
                inv_rms: &self.inv_rms_buf,
                q: &self.q_buf,
                k: &self.k_buf,
                v: &self.v_buf,
                q_normed: &self.q_normed_buf,
                k_normed: &self.k_normed_buf,
                attn_out: &self.attn_out_buf,
                o_out: &self.o_out_buf,
                gate: &self.gate_buf,
                up: &self.up_buf,
                gelu: &self.gelu_buf,
                down: &self.down_buf,
                ple_ctx: &self.ple_context_proj_buf,
                ple_normed: &self.ple_normed_buf,
                ple_projected: &self.ple_projected_buf,
                cos_packed: &self.decode_rope_cos_packed,
                sin_packed: &self.decode_rope_sin_packed,
            };
            for layer_idx in 0..actual_num_layers {
                self.encode_fused_decode_layer(
                    &encoder,
                    layer_idx,
                    kv_seq,
                    &scratch,
                    __ablate.skip_attn(),
                    __ablate.skip_mlp(),
                    __ablate.skip_ple(),
                );
            }
            if crate::decode_fused::profile_dispatches_enabled() {
                eprintln!(
                    "  PROFILE_DISPATCHES: {} layer dispatches this token",
                    crate::decode_fused::take_dispatch_count()
                );
            }
        } else {
        for layer_idx in 0..actual_num_layers {
            if cb_split_idx < cb_layer_splits.len() && layer_idx == cb_layer_splits[cb_split_idx] {
                encoder.end_encoding();
                cmd.commit();
                cmd = self.ctx.queue.new_command_buffer();
                encoder = cmd.new_compute_command_encoder();
                cb_split_idx += 1;
            }
            let layer = &self.layers[layer_idx];
            let intermediate_size = layer.intermediate_size;
            let head_dim = layer.head_dim;
            let q_out = layer.q_out_dim;
            let kv_out = layer.kv_out_dim;
            let is_full = layer.is_full_attention;
            // Gemma4 uses attention_scale = 1.0 (QK norm handles scaling)
            let scale = 1.0f32;

            // ─── Attention Block ───
            // Residual is the current hidden_buf, which is not overwritten until
            // the residual add below, so no separate "save residual" copy is
            // needed; the add reads hidden in place.

            if !__ablate.skip_attn() {
            // Pre-attention norm + Q/K/V projections (fused when Q4_0 or K-quant + FUSED_QKV=1).
            if layer.weight_format.is_kquant() && crate::gpu::fused_qkv_enabled() {
                if layer.has_kv {
                    self.ctx.encode_rmsnorm_qkv_kquant_view(
                        encoder,
                        &self.hidden_buf,
                        &layer.input_layernorm_weight,
                        &self.inv_rms_buf,
                        &layer.q_proj,
                        &layer.k_proj,
                        &layer.v_proj,
                        &self.q_buf,
                        &self.k_buf,
                        &self.v_buf,
                        q_out as u32,
                        kv_out as u32,
                        hidden_size as u32,
                        eps,
                    );
                } else {
                    self.ctx.encode_rmsnorm_q_kquant_view(
                        encoder,
                        &self.hidden_buf,
                        &layer.input_layernorm_weight,
                        &self.inv_rms_buf,
                        &layer.q_proj,
                        &self.q_buf,
                        q_out as u32,
                        hidden_size as u32,
                        eps,
                    );
                }
            } else if layer.weight_format == WeightFormat::Q4_0 && crate::gpu::fused_qkv_enabled() {
                if layer.has_kv {
                    self.ctx.encode_rmsnorm_qkv_q4_view(
                        encoder,
                        &self.hidden_buf,
                        &layer.input_layernorm_weight,
                        &self.inv_rms_buf,
                        &layer.q_proj,
                        &layer.k_proj,
                        &layer.v_proj,
                        &self.q_buf,
                        &self.k_buf,
                        &self.v_buf,
                        q_out as u32,
                        kv_out as u32,
                        hidden_size as u32,
                        eps,
                    );
                } else {
                    self.ctx.encode_rmsnorm_q_q4_view(
                        encoder,
                        &self.hidden_buf,
                        &layer.input_layernorm_weight,
                        &self.inv_rms_buf,
                        &layer.q_proj,
                        &self.q_buf,
                        q_out as u32,
                        hidden_size as u32,
                        eps,
                    );
                }
            } else {
            // Pre-attention norm
            self.ctx.encode_rmsnorm_view(
                encoder,
                &self.hidden_buf,
                &layer.input_layernorm_weight,
                &self.normed_buf,
                hidden_size as u32,
                eps,
            );

            // Q projection (always computed)
            if layer.weight_format == WeightFormat::F16 {
                self.ctx.encode_matvec_f16_view(
                    encoder,
                    &layer.q_proj,
                    &self.normed_buf,
                    &self.q_buf,
                    q_out as u32,
                    hidden_size as u32,
                );
            } else {
                self.encode_matvec_quant(
                    encoder,
                    &layer.q_proj,
                    &self.normed_buf,
                    &self.q_buf,
                    q_out as u32,
                    hidden_size as u32,
                    layer.weight_format,
                );
            }

            // K, V only for non-shared layers
            if layer.has_kv {
                if layer.weight_format == WeightFormat::F16 {
                    self.ctx.encode_matvec_f16_view(
                        encoder,
                        &layer.k_proj,
                        &self.normed_buf,
                        &self.k_buf,
                        kv_out as u32,
                        hidden_size as u32,
                    );
                    self.ctx.encode_matvec_f16_view(
                        encoder,
                        &layer.v_proj,
                        &self.normed_buf,
                        &self.v_buf,
                        kv_out as u32,
                        hidden_size as u32,
                    );
                } else {
                    // For Q3_0 and Q4_0: use dual (fused) matvec
                    self.encode_matvec_quant(
                        encoder,
                        &layer.k_proj,
                        &self.normed_buf,
                        &self.k_buf,
                        kv_out as u32,
                        hidden_size as u32,
                        layer.weight_format,
                    );
                    self.encode_matvec_quant(
                        encoder,
                        &layer.v_proj,
                        &self.normed_buf,
                        &self.v_buf,
                        kv_out as u32,
                        hidden_size as u32,
                        layer.weight_format,
                    );
                }
            }
            } // !fused_qkv fallback

            let use_fused_q_attn = layer.weight_format.is_quantized()
                && matches!(self.kv_cache_type, KvCacheType::Q4_0)
                && self.ctx.use_flash_attention
                && crate::gpu::fused_q_attn_enabled()
                && !crate::gpu::attention_use_ggml_for_layer_kv(layer.has_kv, kv_seq + 1)
                && matches!(head_dim, 128 | 256 | 512);
            let use_fused_k_attn =
                use_fused_q_attn && layer.has_kv && crate::gpu::fused_k_attn_enabled();
            let rope_off = self.decode_rope_byte_offset(layer_idx);

            if !use_fused_q_attn {
            // QK Norm on Q
            self.ctx.encode_rmsnorm_per_head_view(
                encoder,
                &self.q_buf,
                &layer.q_norm_weight,
                &self.q_normed_buf,
                num_heads as u32,
                head_dim as u32,
                eps,
            );

            // Apply rotary to Q (full head_dim — non-rotary dims have cos=1, sin=0 for pass-through)
            self.ctx.encode_rotary_at(
                encoder,
                &self.q_normed_buf,
                0,
                &self.k_normed_buf,
                0,
                &self.decode_rope_cos_packed,
                rope_off,
                &self.decode_rope_sin_packed,
                rope_off,
                num_heads as u32,
                0,
                head_dim as u32,
            );
            } // !use_fused_q_attn

            // K norm + rotary (K/V matvecs fused above when FUSED_QKV=1)
            if layer.has_kv && !use_fused_k_attn {
                self.ctx.encode_rmsnorm_per_head_view(
                    encoder,
                    &self.k_buf,
                    &layer.k_norm_weight,
                    &self.k_normed_buf,
                    num_kv_heads as u32,
                    head_dim as u32,
                    eps,
                );
                let rope_off = self.decode_rope_byte_offset(layer_idx);
                self.ctx.encode_rotary_at(
                    encoder,
                    &self.q_buf,
                    0,
                    &self.k_normed_buf,
                    0,
                    &self.decode_rope_cos_packed,
                    rope_off,
                    &self.decode_rope_sin_packed,
                    rope_off,
                    0,
                    num_kv_heads as u32,
                    head_dim as u32,
                );

                // V norm (no weight)
                self.ctx.encode_rmsnorm_per_head_noweight(
                    encoder,
                    &self.v_buf,
                    &self.gate_buf,
                    num_kv_heads as u32,
                    head_dim as u32,
                    eps,
                );

                // Append to this layer's cache
                match self.kv_cache_type {
                    KvCacheType::F16 => {
                        self.ctx.encode_kv_append_f16(
                            encoder,
                            &self.k_normed_buf,
                            &self.k_cache[layer_idx],
                            num_kv_heads as u32,
                            head_dim as u32,
                            self.kv_capacity,
                            kv_seq,
                        );
                        self.ctx.encode_kv_append_f16(
                            encoder,
                            &self.gate_buf,
                            &self.v_cache[layer_idx],
                            num_kv_heads as u32,
                            head_dim as u32,
                            self.kv_capacity,
                            kv_seq,
                        );
                    }
                    KvCacheType::Q8_0 => {
                        self.ctx.encode_kv_append_q8_0(
                            encoder,
                            &self.k_normed_buf,
                            &self.k_cache[layer_idx],
                            num_kv_heads as u32,
                            head_dim as u32,
                            self.kv_capacity,
                            kv_seq,
                        );
                        self.ctx.encode_kv_append_q8_0(
                            encoder,
                            &self.gate_buf,
                            &self.v_cache[layer_idx],
                            num_kv_heads as u32,
                            head_dim as u32,
                            self.kv_capacity,
                            kv_seq,
                        );
                    }
                    KvCacheType::Q4_0 => {
                        if crate::gpu::needs_explicit_kv_append(layer.has_kv, kv_seq + 1) {
                            self.ctx.encode_kv_append_q4_0(
                                encoder,
                                &self.k_normed_buf,
                                &self.k_cache[layer_idx],
                                num_kv_heads as u32,
                                head_dim as u32,
                                self.kv_capacity,
                                kv_seq,
                            );
                            self.ctx.encode_kv_append_q4_0(
                                encoder,
                                &self.gate_buf,
                                &self.v_cache[layer_idx],
                                num_kv_heads as u32,
                                head_dim as u32,
                                self.kv_capacity,
                                kv_seq,
                            );
                        }
                    }
                    KvCacheType::TurboQuant { k_bits, v_bits } => {
                        let tq = self
                            .turboquant
                            .as_ref()
                            .expect("turboquant rotation state");
                        let fwd = tq.fwd(head_dim);
                        if self.kv_cache_type.tq_affine() {
                            self.ctx.encode_turboquant_rotate(
                                encoder,
                                &self.k_normed_buf,
                                &self.tq_k_rot,
                                fwd,
                                num_kv_heads as u32,
                                head_dim as u32,
                            );
                            self.ctx.encode_turboquant_rotate(
                                encoder,
                                &self.gate_buf,
                                &self.tq_v_rot,
                                fwd,
                                num_kv_heads as u32,
                                head_dim as u32,
                            );
                            self.ctx.encode_kv_append_q4_0(
                                encoder,
                                &self.tq_k_rot,
                                &self.k_cache[layer_idx],
                                num_kv_heads as u32,
                                head_dim as u32,
                                self.kv_capacity,
                                kv_seq,
                            );
                            self.ctx.encode_kv_append_q4_0(
                                encoder,
                                &self.tq_v_rot,
                                &self.v_cache[layer_idx],
                                num_kv_heads as u32,
                                head_dim as u32,
                                self.kv_capacity,
                                kv_seq,
                            );
                        } else {
                            let k_row_bytes = self.kv_cache_type.k_row_bytes(head_dim) as u32;
                            let v_row_bytes = self.kv_cache_type.v_row_bytes(head_dim) as u32;
                            let cen_k = tq.centroids_k(head_dim);
                            let cen_v = tq.centroids_v(head_dim);
                            let kwin = if self.tq_rw > 0 {
                                &self.tq_rw_k[layer_idx]
                            } else {
                                &self.tq_k_rot
                            };
                            let vwin = if self.tq_rw > 0 {
                                &self.tq_rw_v[layer_idx]
                            } else {
                                &self.tq_v_rot
                            };
                            let dual_write = matches!(
                                std::env::var("TURBOQUANT_DUAL_WRITE").as_deref(),
                                Ok("1") | Ok("true") | Ok("TRUE")
                            );
                            let write_hot = self.tq_hot_enabled() && kv_seq < self.tq_hot_w;
                            // Prefer Q4 hot-only while ctx fits (skip TQ quant tax).
                            if write_hot {
                                self.ctx.encode_kv_append_q4_0(
                                    encoder,
                                    &self.k_normed_buf,
                                    &self.tq_hot_k[layer_idx],
                                    num_kv_heads as u32,
                                    head_dim as u32,
                                    self.tq_hot_w,
                                    kv_seq,
                                );
                                self.ctx.encode_kv_append_q4_0(
                                    encoder,
                                    &self.gate_buf,
                                    &self.tq_hot_v[layer_idx],
                                    num_kv_heads as u32,
                                    head_dim as u32,
                                    self.tq_hot_w,
                                    kv_seq,
                                );
                            }
                            if dual_write || !write_hot {
                                self.ctx.encode_turboquant_rotate_quant_v3(
                                    encoder,
                                    &self.k_normed_buf,
                                    &self.k_cache[layer_idx],
                                    cen_k,
                                    num_kv_heads as u32,
                                    head_dim as u32,
                                    self.kv_capacity,
                                    kv_seq,
                                    k_bits as u32,
                                    k_row_bytes,
                                    kwin,
                                    self.tq_rw,
                                    fwd,
                                );
                                self.ctx.encode_turboquant_rotate_quant_v3(
                                    encoder,
                                    &self.gate_buf,
                                    &self.v_cache[layer_idx],
                                    cen_v,
                                    num_kv_heads as u32,
                                    head_dim as u32,
                                    self.kv_capacity,
                                    kv_seq,
                                    v_bits as u32,
                                    v_row_bytes,
                                    vwin,
                                    self.tq_rw,
                                    fwd,
                                );
                            }
                        }
                    }
                }
            }

            // Attention
            let attn_kv_seq = kv_seq + 1;

            // For sliding window layers, limit attention to last sliding_window tokens
            let effective_kv_seq = if !is_full {
                attn_kv_seq.min(self.config.sliding_window as u32)
            } else {
                attn_kv_seq
            };

            // For sliding window, adjust the start position in the cache
            let kv_start = if !is_full && attn_kv_seq > self.config.sliding_window as u32 {
                attn_kv_seq - self.config.sliding_window as u32
            } else {
                0u32
            };

            // Buffer feeding the O-projection. TurboQuant affine redirects this
            // to the un-rotated attention output; V3 writes model-frame into attn_out.
            let mut attn_src: &Buffer = &self.attn_out_buf;

            match self.kv_cache_type {
                KvCacheType::F16 => {
                    if crate::gpu::attention_gqa_f16_enabled(num_kv_groups) {
                        self.ctx.encode_attention_with_offset_f16_gqa_at(
                            encoder,
                            &self.q_normed_buf,
                            0,
                            &self.k_cache[layer.kv_source_layer],
                            &self.v_cache[layer.kv_source_layer],
                            &self.attn_out_buf,
                            0,
                            num_heads as u32,
                            num_kv_heads as u32,
                            num_kv_groups,
                            head_dim as u32,
                            effective_kv_seq,
                            self.kv_capacity,
                            scale,
                            kv_start,
                        );
                    } else {
                        self.ctx.encode_attention_with_offset_f16(
                            encoder,
                            &self.q_normed_buf,
                            &self.k_cache[layer.kv_source_layer],
                            &self.v_cache[layer.kv_source_layer],
                            &self.attn_out_buf,
                            num_heads as u32,
                            num_kv_heads as u32,
                            num_kv_groups,
                            head_dim as u32,
                            effective_kv_seq,
                            self.kv_capacity,
                            scale,
                            kv_start,
                        );
                    }
                }
                KvCacheType::Q8_0 => {
                    let groups_per_row = (head_dim / 32) as u32;
                    let row_bytes = groups_per_row * 34;
                    self.ctx.encode_attention_with_offset_q8_0(
                        encoder,
                        &self.q_normed_buf,
                        &self.k_cache[layer.kv_source_layer],
                        &self.v_cache[layer.kv_source_layer],
                        &self.attn_out_buf,
                        num_heads as u32,
                        num_kv_heads as u32,
                        num_kv_groups,
                        head_dim as u32,
                        effective_kv_seq,
                        self.kv_capacity,
                        scale,
                        kv_start,
                        groups_per_row,
                        row_bytes,
                    );
                }
                KvCacheType::Q4_0 => {
                    let groups_per_row = (head_dim / 32) as u32;
                    let row_bytes = groups_per_row * 18;
                    if use_fused_k_attn {
                        self.ctx.encode_attention_full_fused_q4_0(
                            encoder,
                            &self.q_buf,
                            &layer.q_norm_weight,
                            &self.decode_rope_cos_packed,
                            rope_off,
                            &self.decode_rope_sin_packed,
                            rope_off,
                            &self.k_buf,
                            &layer.k_norm_weight,
                            &self.v_buf,
                            &self.attn_out_buf,
                            &self.k_cache[layer.kv_source_layer],
                            &self.v_cache[layer.kv_source_layer],
                            num_heads as u32,
                            num_kv_heads as u32,
                            num_kv_groups,
                            head_dim as u32,
                            effective_kv_seq,
                            self.kv_capacity,
                            scale,
                            kv_start,
                            kv_seq,
                            groups_per_row,
                            row_bytes,
                            eps,
                        );
                    } else if use_fused_q_attn && layer.has_kv && crate::gpu::fused_kv_attention_enabled() {
                        self.ctx.encode_attention_fused_qknorm_rope_q4_0(
                            encoder,
                            &self.q_buf,
                            &layer.q_norm_weight,
                            &self.decode_rope_cos_packed,
                            rope_off,
                            &self.decode_rope_sin_packed,
                            rope_off,
                            &self.k_normed_buf,
                            &self.gate_buf,
                            &self.attn_out_buf,
                            &self.k_cache[layer.kv_source_layer],
                            &self.v_cache[layer.kv_source_layer],
                            num_heads as u32,
                            num_kv_heads as u32,
                            num_kv_groups,
                            head_dim as u32,
                            effective_kv_seq,
                            self.kv_capacity,
                            scale,
                            kv_start,
                            kv_seq,
                            groups_per_row,
                            row_bytes,
                            eps,
                        );
                    } else if use_fused_q_attn {
                        self.ctx.encode_attention_qknorm_rope_q4_0(
                            encoder,
                            &self.q_buf,
                            &layer.q_norm_weight,
                            &self.decode_rope_cos_packed,
                            rope_off,
                            &self.decode_rope_sin_packed,
                            rope_off,
                            &self.k_cache[layer.kv_source_layer],
                            &self.v_cache[layer.kv_source_layer],
                            &self.attn_out_buf,
                            num_heads as u32,
                            num_kv_heads as u32,
                            num_kv_groups,
                            head_dim as u32,
                            effective_kv_seq,
                            self.kv_capacity,
                            scale,
                            kv_start,
                            groups_per_row,
                            row_bytes,
                            eps,
                        );
                    } else if crate::gpu::attention_use_ggml_for_layer_kv(layer.has_kv, effective_kv_seq)
                        && self.ctx.use_flash_attention
                    {
                        self.ctx.encode_attention_ggml_q4_0(
                            encoder,
                            &self.q_normed_buf,
                            &self.k_cache[layer.kv_source_layer],
                            &self.v_cache[layer.kv_source_layer],
                            &self.ggml_fa_tmp_buf,
                            &self.attn_out_buf,
                            num_heads as u32,
                            num_kv_heads as u32,
                            head_dim as u32,
                            effective_kv_seq,
                            self.kv_capacity,
                            scale,
                            kv_start,
                            row_bytes,
                        );
                    } else if layer.has_kv
                        && self.ctx.use_flash_attention
                        && crate::gpu::fused_kv_attention_enabled()
                    {
                        self.ctx.encode_attention_fused_q4_0(
                            encoder,
                            &self.q_normed_buf,
                            &self.k_normed_buf,
                            &self.gate_buf,
                            &self.attn_out_buf,
                            &self.k_cache[layer.kv_source_layer],
                            &self.v_cache[layer.kv_source_layer],
                            num_heads as u32,
                            num_kv_heads as u32,
                            num_kv_groups,
                            head_dim as u32,
                            effective_kv_seq,
                            self.kv_capacity,
                            scale,
                            kv_start,
                            kv_seq,
                            groups_per_row,
                            row_bytes,
                        );
                    } else if self.ctx.use_flash_attention
                        && num_kv_groups > 1
                        && (8 % num_kv_groups) == 0
                        && crate::gpu::attention_gqa_q4_0_enabled(num_kv_groups)
                    {
                        self.ctx.encode_attention_with_offset_q4_0_gqa(
                            encoder,
                            &self.q_normed_buf,
                            &self.k_cache[layer.kv_source_layer],
                            &self.v_cache[layer.kv_source_layer],
                            &self.attn_out_buf,
                            num_heads as u32,
                            num_kv_heads as u32,
                            num_kv_groups,
                            head_dim as u32,
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
                            &self.q_normed_buf,
                            &self.k_cache[layer.kv_source_layer],
                            &self.v_cache[layer.kv_source_layer],
                            &self.attn_out_buf,
                            num_heads as u32,
                            num_kv_heads as u32,
                            num_kv_groups,
                            head_dim as u32,
                            effective_kv_seq,
                            self.kv_capacity,
                            scale,
                            kv_start,
                            groups_per_row,
                            row_bytes,
                        );
                    }
                }
                KvCacheType::TurboQuant { k_bits, v_bits } => {
                    let tq = self
                        .turboquant
                        .as_ref()
                        .expect("turboquant rotation state");
                    if self.kv_cache_type.tq_affine() {
                        self.ctx.encode_turboquant_rotate(
                            encoder,
                            &self.q_normed_buf,
                            &self.tq_q_rot,
                            tq.fwd(head_dim),
                            num_heads as u32,
                            head_dim as u32,
                        );
                        let groups_per_row = (head_dim / 32) as u32;
                        let row_bytes = groups_per_row * 18;
                        self.ctx.encode_attention_with_offset_q4_0(
                            encoder,
                            &self.tq_q_rot,
                            &self.k_cache[layer.kv_source_layer],
                            &self.v_cache[layer.kv_source_layer],
                            &self.attn_out_buf,
                            num_heads as u32,
                            num_kv_heads as u32,
                            num_kv_groups,
                            head_dim as u32,
                            effective_kv_seq,
                            self.kv_capacity,
                            scale,
                            kv_start,
                            groups_per_row,
                            row_bytes,
                        );
                        self.ctx.encode_turboquant_rotate(
                            encoder,
                            &self.attn_out_buf,
                            &self.tq_out,
                            tq.inv(head_dim),
                            num_heads as u32,
                            head_dim as u32,
                        );
                        attn_src = &self.tq_out;
                    } else if self.tq_attn_fits_hot(attn_kv_seq) {
                        // Hybrid fast path: model-frame Q4 over the hot ring
                        // (ggml MWG when ATTENTION_KERNEL=auto and kv≥128).
                        let groups_per_row = (head_dim / 32) as u32;
                        let row_bytes = groups_per_row * 18;
                        let src = layer.kv_source_layer;
                        if crate::gpu::attention_use_ggml_for_layer_kv(
                            layer.has_kv,
                            effective_kv_seq,
                        ) && self.ctx.use_flash_attention
                        {
                            self.ctx.encode_attention_ggml_q4_0(
                                encoder,
                                &self.q_normed_buf,
                                &self.tq_hot_k[src],
                                &self.tq_hot_v[src],
                                &self.ggml_fa_tmp_buf,
                                &self.attn_out_buf,
                                num_heads as u32,
                                num_kv_heads as u32,
                                head_dim as u32,
                                effective_kv_seq,
                                self.tq_hot_w,
                                scale,
                                kv_start,
                                row_bytes,
                            );
                        } else if crate::gpu::attention_gqa_q4_0_enabled(num_kv_groups) {
                            self.ctx.encode_attention_with_offset_q4_0_gqa(
                                encoder,
                                &self.q_normed_buf,
                                &self.tq_hot_k[src],
                                &self.tq_hot_v[src],
                                &self.attn_out_buf,
                                num_heads as u32,
                                num_kv_heads as u32,
                                num_kv_groups,
                                head_dim as u32,
                                effective_kv_seq,
                                self.tq_hot_w,
                                scale,
                                kv_start,
                                groups_per_row,
                                row_bytes,
                            );
                        } else {
                            self.ctx.encode_attention_with_offset_q4_0(
                                encoder,
                                &self.q_normed_buf,
                                &self.tq_hot_k[src],
                                &self.tq_hot_v[src],
                                &self.attn_out_buf,
                                num_heads as u32,
                                num_kv_heads as u32,
                                num_kv_groups,
                                head_dim as u32,
                                effective_kv_seq,
                                self.tq_hot_w,
                                scale,
                                kv_start,
                                groups_per_row,
                                row_bytes,
                            );
                        }
                    } else {
                        let k_row_bytes = self.kv_cache_type.k_row_bytes(head_dim) as u32;
                        let v_row_bytes = self.kv_cache_type.v_row_bytes(head_dim) as u32;
                        // turboquant_attn_v3 treats `kv_seq` as the *absolute end*
                        // position (n_pos = kv_seq - kv_start). Q4 offset kernels
                        // instead take window *length* as kv_seq — do NOT pass
                        // effective_kv_seq here or SWA layers get n_pos=0 once
                        // attn_kv_seq > sliding_window (garbage long-context decode).
                        let window_lo = if self.tq_rw > 0 {
                            kv_start.max(attn_kv_seq.saturating_sub(self.tq_rw))
                        } else {
                            0
                        };
                        let kwin = if self.tq_rw > 0 {
                            &self.tq_rw_k[layer.kv_source_layer]
                        } else {
                            &self.q_normed_buf
                        };
                        let vwin = if self.tq_rw > 0 {
                            &self.tq_rw_v[layer.kv_source_layer]
                        } else {
                            &self.q_normed_buf
                        };
                        self.ctx.encode_turboquant_attn_v3(
                            encoder,
                            &self.q_normed_buf,
                            &self.k_cache[layer.kv_source_layer],
                            &self.v_cache[layer.kv_source_layer],
                            &self.attn_out_buf,
                            tq.centroids_k(head_dim),
                            tq.centroids_v(head_dim),
                            num_heads as u32,
                            num_kv_heads as u32,
                            num_kv_groups,
                            head_dim as u32,
                            attn_kv_seq,
                            self.kv_capacity,
                            scale,
                            kv_start,
                            k_bits as u32,
                            v_bits as u32,
                            k_row_bytes,
                            v_row_bytes,
                            kwin,
                            vwin,
                            self.tq_rw,
                            window_lo,
                            tq.fwd(head_dim),
                            tq.inv(head_dim),
                            &self.tq_scores,
                        );
                    }
                }
            }

            // O projection (Q4 on middle layers, f16 on sensitive layers)
            self.ctx.encode_matvec_auto_view(
                encoder,
                &layer.o_proj,
                attn_src,
                &self.o_out_buf,
                hidden_size as u32,
                q_out as u32,
            );

            // Post-attention norm + residual add (3→1: proj + norm + residual)
            encode_proj_norm_residual(
                &self.ctx,
                encoder,
                &self.hidden_buf,
                &self.o_out_buf,
                &layer.post_attention_layernorm_weight,
                &self.hidden_buf,
                hidden_size as u32,
                eps,
            );
            } // !skip_attn
            if let Some(p) = &mut __gpu_prof {
                p.mark(&encoder);
            }

            // Phase boundary: attention block (profiling only).
            if __pp {
                encoder.end_encoding();
                let __pt = std::time::Instant::now();
                cmd.commit();
                cmd.wait_until_completed();
                Self::phase_accum(1, __pt.elapsed().as_secs_f64() * 1e3);
                cmd = self.ctx.queue.new_command_buffer();
                encoder = cmd.new_compute_command_encoder();
            }

            // ─── MLP Block ───
            // hidden_buf is the residual; it is only read (by the pre-FF norm)
            // until the residual add below, so no save-residual copy is needed.

            if !__ablate.skip_mlp() {
            // MLP: gate_proj, up_proj, GeLU activation, down_proj
            if layer.weight_format == WeightFormat::F16 {
                self.ctx.encode_rmsnorm_view(
                    encoder,
                    &self.hidden_buf,
                    &layer.pre_feedforward_layernorm_weight,
                    &self.normed_buf,
                    hidden_size as u32,
                    eps,
                );
                self.ctx.encode_matvec_f16_view(
                    encoder,
                    &layer.gate_proj,
                    &self.normed_buf,
                    &self.gate_buf,
                    intermediate_size as u32,
                    hidden_size as u32,
                );
                self.ctx.encode_matvec_f16_view(
                    encoder,
                    &layer.up_proj,
                    &self.normed_buf,
                    &self.up_buf,
                    intermediate_size as u32,
                    hidden_size as u32,
                );
                self.ctx.encode_gelu_mul(
                    encoder,
                    &self.gate_buf,
                    &self.up_buf,
                    &self.gelu_buf,
                    intermediate_size as u32,
                );
                self.ctx.encode_matvec_f16_view(
                    encoder,
                    &layer.down_proj,
                    &self.gelu_buf,
                    &self.down_buf,
                    hidden_size as u32,
                    intermediate_size as u32,
                );
            } else if layer.weight_format.is_kquant() {
                // K-quant (Q4_K_M): gate/up (Q4_K) fused with optional pre-FF RMSNorm;
                // down (Q6_K) is a plain matvec.
                if layer.gate_proj.format == crate::gpu::weight_fmt::Q4_K
                    && layer.up_proj.format == crate::gpu::weight_fmt::Q4_K
                    && crate::gpu::fused_rmsnorm_mlp_kquant_enabled()
                {
                    self.ctx.encode_rmsnorm_qk_gelu_mul_kquant_at_view(
                        encoder,
                        &layer.gate_proj,
                        &layer.up_proj,
                        &self.hidden_buf,
                        0,
                        &layer.pre_feedforward_layernorm_weight,
                        &self.inv_rms_buf,
                        &self.gelu_buf,
                        0,
                        intermediate_size as u32,
                        hidden_size as u32,
                        eps,
                    );
                } else if layer.gate_proj.format == crate::gpu::weight_fmt::Q4_K
                    && layer.up_proj.format == crate::gpu::weight_fmt::Q4_K
                {
                    self.ctx.encode_rmsnorm_view(
                        encoder,
                        &self.hidden_buf,
                        &layer.pre_feedforward_layernorm_weight,
                        &self.normed_buf,
                        hidden_size as u32,
                        eps,
                    );
                    self.ctx.encode_matvec_qk_gelu_mul_at_view(
                        encoder,
                        &layer.gate_proj,
                        &layer.up_proj,
                        &self.normed_buf,
                        0,
                        &self.gelu_buf,
                        0,
                        intermediate_size as u32,
                        hidden_size as u32,
                    );
                } else {
                    self.ctx.encode_rmsnorm_view(
                        encoder,
                        &self.hidden_buf,
                        &layer.pre_feedforward_layernorm_weight,
                        &self.normed_buf,
                        hidden_size as u32,
                        eps,
                    );
                    self.encode_matvec_quant(
                        encoder,
                        &layer.gate_proj,
                        &self.normed_buf,
                        &self.gate_buf,
                        intermediate_size as u32,
                        hidden_size as u32,
                        layer.weight_format,
                    );
                    self.encode_matvec_quant(
                        encoder,
                        &layer.up_proj,
                        &self.normed_buf,
                        &self.up_buf,
                        intermediate_size as u32,
                        hidden_size as u32,
                        layer.weight_format,
                    );
                    self.ctx.encode_gelu_mul(
                        encoder,
                        &self.gate_buf,
                        &self.up_buf,
                        &self.gelu_buf,
                        intermediate_size as u32,
                    );
                }
                self.encode_matvec_quant(
                    encoder,
                    &layer.down_proj,
                    &self.gelu_buf,
                    &self.down_buf,
                    hidden_size as u32,
                    intermediate_size as u32,
                    layer.weight_format,
                );
            } else if crate::gpu::fused_mlp_gelu_down_enabled()
                && crate::gpu::fused_rmsnorm_mlp_enabled()
                && Self::use_packed_mlp_gate_up(layer)
                && crate::gpu::weight_buf_is_q4(
                    &layer.gate_proj,
                    intermediate_size as u32,
                    hidden_size as u32,
                )
            {
                self.ctx.encode_mlp_fused_q4_gelu_down_packed_from_hidden_at_view(
                    encoder,
                    &layer.gate_up_proj,
                    &layer.down_proj,
                    &layer.pre_feedforward_layernorm_weight,
                    &self.hidden_buf,
                    0,
                    &self.inv_rms_buf,
                    0,
                    &self.up_buf,
                    0,
                    &self.down_buf,
                    0,
                    hidden_size as u32,
                    intermediate_size as u32,
                    eps,
                );
            } else if crate::gpu::fused_mlp_gelu_down_enabled()
                && !layer.weight_format.is_q3()
                && crate::gpu::weight_buf_is_q4(
                    &layer.gate_proj,
                    intermediate_size as u32,
                    hidden_size as u32,
                )
            {
                self.ctx.encode_rmsnorm_view(
                    encoder,
                    &self.hidden_buf,
                    &layer.pre_feedforward_layernorm_weight,
                    &self.normed_buf,
                    hidden_size as u32,
                    eps,
                );
                if Self::use_packed_mlp_gate_up(layer) {
                    if crate::gpu::mlp_gate_up_ggml_enabled() {
                        self.ctx.encode_mlp_fused_q4_gelu_down_ggml_at_view(
                            encoder,
                            &layer.gate_proj,
                            &layer.up_proj,
                            &layer.down_proj,
                            &self.normed_buf,
                            0,
                            &self.gate_buf,
                            &self.up_buf,
                            &self.gelu_buf,
                            0,
                            &self.down_buf,
                            0,
                            hidden_size as u32,
                            intermediate_size as u32,
                        );
                    } else {
                        self.ctx.encode_mlp_fused_q4_gelu_down_packed_at_view(
                            encoder,
                            &layer.gate_up_proj,
                            &layer.down_proj,
                            &self.normed_buf,
                            0,
                            &self.up_buf,
                            0,
                            &self.down_buf,
                            0,
                            hidden_size as u32,
                            intermediate_size as u32,
                        );
                    }
                } else {
                    self.ctx.encode_mlp_fused_q4_gelu_down_view(
                        encoder,
                        &layer.gate_proj,
                        &layer.up_proj,
                        &layer.down_proj,
                        &self.normed_buf,
                        &self.gelu_buf,
                        &self.down_buf,
                        hidden_size as u32,
                        intermediate_size as u32,
                    );
                }
            } else if Self::use_packed_mlp_gate_up(layer)
                || layer.weight_format.is_q3()
                || crate::gpu::fused_mlp_ple_enabled()
            {
                self.ctx.encode_rmsnorm_view(
                    encoder,
                    &self.hidden_buf,
                    &layer.pre_feedforward_layernorm_weight,
                    &self.normed_buf,
                    hidden_size as u32,
                    eps,
                );
                self.encode_mlp_gate_up_gelu_q4_view(
                    encoder,
                    layer,
                    &self.normed_buf,
                    &self.gelu_buf,
                    intermediate_size as u32,
                    hidden_size as u32,
                );
                self.encode_matvec_quant(
                    encoder,
                    &layer.down_proj,
                    &self.gelu_buf,
                    &self.down_buf,
                    hidden_size as u32,
                    intermediate_size as u32,
                    layer.weight_format,
                );
            } else {
                self.ctx.encode_rmsnorm_view(
                    encoder,
                    &self.hidden_buf,
                    &layer.pre_feedforward_layernorm_weight,
                    &self.normed_buf,
                    hidden_size as u32,
                    eps,
                );
                self.ctx.encode_matvec_q4_dual_view(
                    encoder,
                    &layer.gate_proj,
                    &layer.up_proj,
                    &self.normed_buf,
                    &self.gate_buf,
                    &self.up_buf,
                    intermediate_size as u32,
                    hidden_size as u32,
                );
                self.ctx.encode_gelu_mul(
                    encoder,
                    &self.gate_buf,
                    &self.up_buf,
                    &self.gelu_buf,
                    intermediate_size as u32,
                );
                self.encode_matvec_quant(
                    encoder,
                    &layer.down_proj,
                    &self.gelu_buf,
                    &self.down_buf,
                    hidden_size as u32,
                    intermediate_size as u32,
                    layer.weight_format,
                );
            }

            // Post-feedforward norm + residual add
            if crate::gpu::fused_rmsnorm_acc_enabled() {
                self.ctx.encode_rmsnorm_acc_view(
                    encoder,
                    &self.hidden_buf,
                    &self.down_buf,
                    &layer.post_feedforward_layernorm_weight,
                    hidden_size as u32,
                    eps,
                );
            } else {
                self.ctx.encode_rmsnorm_view(
                    encoder,
                    &self.down_buf,
                    &layer.post_feedforward_layernorm_weight,
                    &self.normed_buf,
                    hidden_size as u32,
                    eps,
                );
                self.ctx.encode_vec_add(
                    encoder,
                    &self.hidden_buf,
                    &self.normed_buf,
                    &self.hidden_buf,
                    hidden_size as u32,
                );
            }
            } // !skip_mlp
            if let Some(p) = &mut __gpu_prof {
                p.mark(&encoder);
            }

            // ─── Per-Layer Embedding (PLE) — after MLP, before layer_scalar ───
            if !__ablate.skip_ple() {
            // hidden_buf is the residual; it is only read (by the input gate
            // matvec) until the residual add below, so no save copy is needed.
            // Gate: ple_gated = per_layer_input_gate(hidden) → [ple_dim]
            if crate::gpu::fused_mlp_ple_enabled()
                && crate::gpu::weight_buf_is_q4(
                    &layer.per_layer_input_gate_weight,
                    ple_dim as u32,
                    hidden_size as u32,
                )
            {
                self.ctx.encode_ple_matvec_gelu_q4_view(
                    encoder,
                    &layer.per_layer_input_gate_weight,
                    &self.hidden_buf,
                    &self.ple_context_proj_buf,
                    (layer_idx * ple_dim * 4) as u64,
                    &self.ple_normed_buf,
                    ple_dim as u32,
                    hidden_size as u32,
                );
            } else {
                self.ctx.encode_matvec_auto_view(
                    encoder,
                    &layer.per_layer_input_gate_weight,
                    &self.hidden_buf,
                    &self.ple_gated_buf,
                    ple_dim as u32,
                    hidden_size as u32,
                );
                self.ctx.encode_gelu_mul_at(
                    encoder,
                    &self.ple_gated_buf,
                    0,
                    &self.ple_context_proj_buf,
                    (layer_idx * ple_dim * 4) as u64,
                    &self.ple_normed_buf,
                    0,
                    ple_dim as u32,
                );
            }
            // Project back: ple_projected = per_layer_projection(ple_normed) → [hidden]
            self.ctx.encode_matvec_auto_view(
                encoder,
                &layer.per_layer_projection_weight,
                &self.ple_normed_buf,
                &self.ple_projected_buf,
                hidden_size as u32,
                ple_dim as u32,
            );
            // Post-PLE norm + residual add (3→1: proj + norm + residual)
            encode_proj_norm_residual(
                &self.ctx,
                encoder,
                &self.hidden_buf,
                &self.ple_projected_buf,
                &layer.post_per_layer_input_norm_weight,
                &self.hidden_buf,
                hidden_size as u32,
                eps,
            );

            } // !skip_ple (per-layer)

            // Layer scalar (in place): hidden *= layer_scalar. vec_scale is a
            // pure elementwise map (dst[i] = scale * src[i]), so src == dst is
            // race-free and the temp copy is unnecessary.
            self.ctx.encode_vec_scale(
                encoder,
                &self.hidden_buf,
                &self.hidden_buf,
                hidden_size as u32,
                layer.layer_scalar,
            );
            if let Some(p) = &mut __gpu_prof {
                p.mark(&encoder);
            }

            // Phase boundary: MLP + PLE block (profiling only).
            if __pp {
                encoder.end_encoding();
                let __pt = std::time::Instant::now();
                cmd.commit();
                cmd.wait_until_completed();
                Self::phase_accum(2, __pt.elapsed().as_secs_f64() * 1e3);
                cmd = self.ctx.queue.new_command_buffer();
                encoder = cmd.new_compute_command_encoder();
            }
        }
        } // !use_fused_decode

        if matches!(mode, DecodeMode::Advance) {
            encoder.end_encoding();
            let __t_encode = std::time::Instant::now();
            cmd.commit();
            cmd.wait_until_completed();
            let __t_gpu = std::time::Instant::now();
            if __profile {
                Self::profile_record(__t0, __t_prep, __t_encode, __t_gpu);
            }
            self.total_tokens += 1;
            self.kv_seq_len += 1;
            return DecodeOutput::Advanced;
        }

        // ─── Final norm + LM head (+ optional sampling) — same encoder ───
        let cap = self.config.final_logit_softcapping;
        if !__ablate.skip_head() {
        self.ctx.encode_rmsnorm_view(
            encoder,
            &self.hidden_buf,
            &self.final_norm_weight,
            &self.normed_buf,
            hidden_size as u32,
            eps,
        );
        // Logits via tied embeddings (native lm_head): logits = lm_head @ normed.
        // Format-aware dispatch handles Q4_0 / Q4_K / Q6_K / F16 lm_head.
        self.ctx.encode_matvec_auto_view(
            encoder,
            &self.lm_head_buf,
            &self.normed_buf,
            &self.logits_buf,
            vocab_size as u32,
            hidden_size as u32,
        );

        } // !skip_head
        if let Some(p) = &mut __gpu_prof {
            p.mark(&encoder);
        }

        let output = match mode {
            DecodeMode::Sample(temperature, min_p, seed) => {
                if __ablate.skip_head() {
                    MetalContext::write_u32_buffer(&self.sample_out_buf, &[0]);
                } else {
                // GPU-side softcap + min-p sampling; read back only the token id.
                self.ctx.encode_sample(
                    encoder,
                    &self.logits_buf,
                    &self.sample_out_buf,
                    vocab_size as u32,
                    cap,
                    temperature,
                    min_p,
                    seed,
                );
                }
                encoder.end_encoding();
                if let Some(p) = &__gpu_prof {
                    p.resolve(&cmd);
                }
                let __t_encode = std::time::Instant::now();
                cmd.commit();
                cmd.wait_until_completed();
                let __t_gpu = std::time::Instant::now();
                if let Some(p) = &__gpu_prof {
                    p.ingest(actual_num_layers as u32);
                }
                if __pp {
                    Self::phase_accum(3, (__t_gpu - __t_encode).as_secs_f64() * 1e3);
                    Self::phase_end_token();
                }
                let tok = MetalContext::read_u32(&self.sample_out_buf) as usize;
                if __profile {
                    Self::profile_record(__t0, __t_prep, __t_encode, __t_gpu);
                }
                DecodeOutput::Token(tok)
            }
            DecodeMode::Logits => {
                encoder.end_encoding();
                if let Some(p) = &__gpu_prof {
                    p.resolve(&cmd);
                }
                let __t_encode = std::time::Instant::now();
                cmd.commit();
                cmd.wait_until_completed();
                let __t_gpu = std::time::Instant::now();
                if let Some(p) = &__gpu_prof {
                    p.ingest(actual_num_layers as u32);
                }
                if __pp {
                    Self::phase_accum(3, (__t_gpu - __t_encode).as_secs_f64() * 1e3);
                    Self::phase_end_token();
                }

                let mut logits = MetalContext::read_buffer(&self.logits_buf, vocab_size);
                // Logit softcapping: logits = cap * tanh(logits / cap)
                // Clamp input to tanh to prevent NaN from overflow.
                for l in logits.iter_mut() {
                    let x = (*l / cap).clamp(-10.0, 10.0);
                    *l = cap * x.tanh();
                }
                if __profile {
                    Self::profile_record(__t0, __t_prep, __t_encode, __t_gpu);
                }
                DecodeOutput::Logits(logits)
            }
            DecodeMode::Advance => unreachable!(),
        };

        // Update state
        self.total_tokens += 1;
        self.kv_seq_len += 1;

        output
    }

    // Per-phase GPU-time accumulators (PROFILE_PHASES). Index: 0=prepass,
    // 1=attention, 2=mlp+ple, 3=head. Sums + token count live in a module-level
    // thread-local (PHASE_STATE) so both helpers share the same window.
    fn phase_accum(idx: usize, ms: f64) {
        PHASE_STATE.with(|s| {
            let mut v = s.get();
            v.sums[idx] += ms;
            s.set(v);
        });
    }

    fn phase_end_token() {
        PHASE_STATE.with(|s| {
            let mut v = s.get();
            v.count += 1;
            s.set(v);
            if v.count % 16 == 0 {
                let c = v.count as f64;
                let s0 = v.sums[0] / c;
                let s1 = v.sums[1] / c;
                let s2 = v.sums[2] / c;
                let s3 = v.sums[3] / c;
                eprintln!(
                    "[phase profile] n={} avg/token commit->wait ms (incl per-phase sync): prepass={:.2} attention={:.2} mlp_ple={:.2} head={:.2} sum={:.2}",
                    v.count, s0, s1, s2, s3, s0 + s1 + s2 + s3
                );
            }
        });
    }

    fn profile_record(
        t0: std::time::Instant,
        t_prep: std::time::Instant,
        t_encode: std::time::Instant,
        t_gpu: std::time::Instant,
    ) {
        use std::cell::Cell;
        thread_local! {
            static ACC: Cell<(f64, f64, f64, f64, u64)> = Cell::new((0.0, 0.0, 0.0, 0.0, 0));
        }
        let prep = (t_prep - t0).as_secs_f64() * 1e3;
        let encode = (t_encode - t_prep).as_secs_f64() * 1e3;
        let gpu = (t_gpu - t_encode).as_secs_f64() * 1e3;
        let read = t_gpu.elapsed().as_secs_f64() * 1e3;
        ACC.with(|a| {
            let (mut p, mut e, mut g, mut r, mut n) = a.get();
            p += prep;
            e += encode;
            g += gpu;
            r += read;
            n += 1;
            if n % 16 == 0 {
                let nf = n as f64;
                eprintln!(
                    "[decode profile] n={} avg/token: cpu_prep={:.2}ms encode={:.2}ms gpu_wait={:.2}ms readback={:.2}ms total={:.2}ms",
                    n, p / nf, e / nf, g / nf, r / nf, (p + e + g + r) / nf
                );
            }
            a.set((p, e, g, r, n));
        });
    }

    pub fn num_items(&self) -> usize {
        self.total_tokens
    }

    /// Live KV meter line for TurboQuant demos:
    /// `KV: X.XX MB / N tok (turboquant-KaVb, vs F16 Y.YY MB, Z×)`.
    /// Returns `None` for non-TurboQuant cache types.
    /// True when V3 TurboQuant keeps a model-frame Q4 hot ring for fast attention.
    #[inline]
    pub fn tq_hot_enabled(&self) -> bool {
        self.tq_hot_w > 0
            && matches!(self.kv_cache_type, KvCacheType::TurboQuant { .. })
            && !self.kv_cache_type.tq_affine()
    }

    /// Attend with Q4 flash on the hot ring while the full context still fits.
    #[inline]
    pub fn tq_attn_fits_hot(&self, attn_kv_seq: u32) -> bool {
        self.tq_hot_enabled() && attn_kv_seq <= self.tq_hot_w
    }

    /// Parallel prefill may use the Q4 hot ring when the whole chunk stays inside it.
    #[inline]
    pub fn tq_prefill_fits_hot(&self, start_pos: usize, seq_len: usize) -> bool {
        self.tq_hot_enabled() && (start_pos + seq_len) as u32 <= self.tq_hot_w
    }

    /// True when the next decode step will leave the hot window and TQ cold is empty.
    #[inline]
    fn tq_needs_spill(&self, kv_seq: u32) -> bool {
        self.tq_hot_enabled() && !self.tq_hot_spilled && kv_seq >= self.tq_hot_w
    }

    /// Spill model-owned hot ring into model cold K/V (CLI / after slot swap).
    fn spill_tq_hot_to_cold(&mut self) {
        if self.tq_hot_spilled || !self.tq_hot_enabled() {
            return;
        }
        let n_tokens = self.tq_hot_w.min(self.kv_seq_len);
        self.spill_tq_hot_buffers(
            &self.tq_hot_k.clone(),
            &self.tq_hot_v.clone(),
            &self.k_cache.clone(),
            &self.v_cache.clone(),
            if self.tq_rw > 0 {
                Some((&self.tq_rw_k, &self.tq_rw_v))
            } else {
                None
            },
            n_tokens,
            self.kv_capacity,
        );
        self.tq_hot_spilled = true;
    }

    /// Pack Q4 hot `[0, n_tokens)` into TQ V3 cold for every KV-owning layer.
    fn spill_tq_hot_buffers(
        &self,
        hot_k: &[Buffer],
        hot_v: &[Buffer],
        cold_k: &[Buffer],
        cold_v: &[Buffer],
        rw: Option<(&[Buffer], &[Buffer])>,
        n_tokens: u32,
        cold_capacity: u32,
    ) {
        if !self.tq_hot_enabled() || n_tokens == 0 {
            return;
        }
        let KvCacheType::TurboQuant { k_bits, v_bits } = self.kv_cache_type else {
            return;
        };
        let Some(tq) = self.turboquant.as_ref() else {
            return;
        };

        let cmd = self.ctx.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if !layer.has_kv {
                continue;
            }
            if layer_idx >= hot_k.len() || layer_idx >= cold_k.len() {
                continue;
            }
            let head_dim = layer.head_dim as u32;
            let num_kv_heads = self.config.layer_num_kv_heads(layer_idx) as u32;
            let q4_row = (head_dim / 32) * 18;
            let k_row = self.kv_cache_type.k_row_bytes(layer.head_dim) as u32;
            let v_row = self.kv_cache_type.v_row_bytes(layer.head_dim) as u32;
            let fwd = tq.fwd(layer.head_dim);
            let (kwin, vwin) = match rw {
                Some((rk, rv)) if layer_idx < rk.len() => (&rk[layer_idx], &rv[layer_idx]),
                _ => (&self.tq_k_rot, &self.tq_v_rot),
            };
            self.ctx.encode_turboquant_spill_q4_to_v3(
                &encoder,
                &hot_k[layer_idx],
                &cold_k[layer_idx],
                tq.centroids_k(layer.head_dim),
                fwd,
                kwin,
                num_kv_heads,
                head_dim,
                self.tq_hot_w,
                cold_capacity,
                n_tokens,
                k_bits as u32,
                q4_row,
                k_row,
                self.tq_rw,
            );
            self.ctx.encode_turboquant_spill_q4_to_v3(
                &encoder,
                &hot_v[layer_idx],
                &cold_v[layer_idx],
                tq.centroids_v(layer.head_dim),
                fwd,
                vwin,
                num_kv_heads,
                head_dim,
                self.tq_hot_w,
                cold_capacity,
                n_tokens,
                v_bits as u32,
                q4_row,
                v_row,
                self.tq_rw,
            );
        }
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        eprintln!(
            "  TurboQuant: spilled {} hot Q4 tokens → TQ cold (ctx now uses attn_v3)",
            n_tokens
        );
    }

    /// Spill a pool slot's hot ring into that slot's cold TQ buffers.
    fn spill_tq_slot(&self, kv_pool: &mut KvCachePool, slot: KvSlot) -> Result<(), String> {
        let spilled = kv_pool.tq_hot_spilled(slot).map_err(|e| e.to_string())?;
        if spilled || !self.tq_hot_enabled() {
            return Ok(());
        }
        let seq_len = kv_pool.seq_len(slot).map_err(|e| e.to_string())?;
        let n_tokens = self.tq_hot_w.min(seq_len);
        if n_tokens == 0 {
            kv_pool
                .set_tq_hot_spilled(slot, true)
                .map_err(|e| e.to_string())?;
            return Ok(());
        }
        // Clone Metal handles (RC) so we can spill without holding slot borrows.
        let (hot_k, hot_v, cold_k, cold_v, rw) = kv_pool
            .with_slot_mut(slot, |s| {
                let rw = if !s.tq_rw_k.is_empty() {
                    Some((s.tq_rw_k.clone(), s.tq_rw_v.clone()))
                } else {
                    None
                };
                (
                    s.tq_hot_k.clone(),
                    s.tq_hot_v.clone(),
                    s.k_cache.clone(),
                    s.v_cache.clone(),
                    rw,
                )
            })
            .map_err(|e| e.to_string())?;
        let rw_ref = rw
            .as_ref()
            .map(|(k, v)| (k.as_slice(), v.as_slice()));
        self.spill_tq_hot_buffers(
            &hot_k,
            &hot_v,
            &cold_k,
            &cold_v,
            rw_ref,
            n_tokens,
            kv_pool.capacity(),
        );
        kv_pool
            .set_tq_hot_spilled(slot, true)
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn kv_meter_line(&self) -> Option<String> {
        let KvCacheType::TurboQuant { .. } = self.kv_cache_type else {
            return None;
        };
        let n = self.kv_seq_len.max(1) as usize;
        let mut tq_bytes = 0usize;
        let mut f16_bytes = 0usize;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if !layer.has_kv {
                continue;
            }
            let hd = layer.head_dim;
            let n_kv = self.config.layer_num_kv_heads(layer_idx);
            tq_bytes += n_kv * n * (self.kv_cache_type.k_row_bytes(hd) + self.kv_cache_type.v_row_bytes(hd));
            f16_bytes += n_kv * n * hd * 2 * 2; // K + V, f16
        }
        let tq_mb = tq_bytes as f64 / (1024.0 * 1024.0);
        let f16_mb = f16_bytes as f64 / (1024.0 * 1024.0);
        let ratio = if tq_mb > 0.0 { f16_mb / tq_mb } else { 0.0 };
        Some(format!(
            "KV: {:.2} MB / {} tok ({}, vs F16 {:.2} MB, {:.1}×)",
            tq_mb, self.kv_seq_len, self.kv_cache_type, f16_mb, ratio
        ))
    }

    pub fn reset_legacy_state(&mut self) {
        self.kv_seq_len = 0;
        self.total_tokens = 0;
        self.tq_hot_spilled = false;
    }

    /// Raw (unscaled) input embedding row, matching a plain `nn.Embedding` lookup.
    /// The MTP drafter concatenates this with the target hidden state and feeds it to
    /// `pre_projection`, bypassing the text model's sqrt(hidden) scaling.
    pub fn token_embedding_raw(&self, token_id: usize) -> Result<Vec<f32>, String> {
        let hidden_size = self.config.hidden_size;
        let mut emb = vec![0.0f32; hidden_size];
        self.embed_tables.decode_embed_into_no_scale(token_id, hidden_size, &mut emb);
        Ok(emb)
    }

    /// Map an MTP assistant layer index to the target model's KV cache layer
    /// that has the same attention type (sliding vs full).
    pub fn mtp_kv_source_layer(&self, is_full_attention: bool) -> Option<usize> {
        let non_shared_layers = self
            .config
            .num_hidden_layers
            .saturating_sub(self.config.num_kv_shared_layers);
        (0..non_shared_layers)
            .rev()
            .find(|&layer_idx| self.config.is_full_attention(layer_idx) == is_full_attention)
    }

    /// Append one token to KV without lm_head readback (decode + final norm only).
    /// Uses the existing `forward_advance` which is KV-only.
    pub fn forward_append_token(&mut self, token_id: usize) {
        // forward_advance already does KV-only with no lm_head
        self.forward_advance(token_id);
    }

    /// Run lm_head on the current `normed_buf` (post-final-norm). Used after
    /// `forward_append_token` when logits were already consumed from verify.
    pub fn forward_logits_from_normed_buf(&self) -> Vec<f32> {
        let vocab_size = self.config.vocab_size;
        let hidden_size = self.config.hidden_size;
        let cmd = self.ctx.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        self.ctx.encode_matvec_auto_view(
            encoder,
            &self.lm_head_buf,
            &self.normed_buf,
            &self.logits_buf,
            vocab_size as u32,
            hidden_size as u32,
        );
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let mut logits = MetalContext::read_buffer(&self.logits_buf, vocab_size);
        let cap = self.config.final_logit_softcapping;
        for l in logits.iter_mut() {
            let x = (*l / cap).clamp(-10.0, 10.0);
            *l = cap * x.tanh();
        }
        logits
    }

    /// Post-final-norm hidden (LM-head input / h_nextn).
    pub fn last_hidden_activation(&self) -> Vec<f32> {
        MetalContext::read_buffer(&self.normed_buf, self.config.hidden_size)
    }

    /// Post-final-norm hidden from an MTP verify row.
    pub fn prefill_hidden_activation_at(&self, row: usize) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let end = (row + 1) * hidden_size;
        let all = MetalContext::read_buffer(&self.mtp_verify_hidden_buf, end);
        all[row * hidden_size..end].to_vec()
    }

    /// Roll back trailing KV entries after a partial MTP accept.
    pub fn truncate_kv(&mut self, tail_count: u32) {
        self.kv_seq_len = self.kv_seq_len.saturating_sub(tail_count);
        self.total_tokens = self
            .total_tokens
            .saturating_sub(tail_count as usize);
    }

    /// Alias this model's `k_cache`/`v_cache` handles to a pool slot's buffers.
    /// Metal buffers are refcounted, so writes via the pool update what the
    /// MTP assistant reads through `target.k_cache`.
    pub fn alias_kv_from_pool(&mut self, kv_pool: &KvCachePool, slot: KvSlot) -> Result<(), String> {
        let (k, v) = kv_pool.slot_buffers(slot).map_err(|e| e.to_string())?;
        self.k_cache = k.to_vec();
        self.v_cache = v.to_vec();
        self.kv_seq_len = kv_pool.seq_len(slot).map_err(|e| e.to_string())?;
        self.total_tokens = kv_pool.total_tokens(slot).map_err(|e| e.to_string())?;
        self.kv_capacity = kv_pool.capacity();
        Ok(())
    }

    fn sync_kv_meta_from_pool(&mut self, kv_pool: &KvCachePool, slot: KvSlot) -> Result<(), String> {
        self.kv_seq_len = kv_pool.seq_len(slot).map_err(|e| e.to_string())?;
        self.total_tokens = kv_pool.total_tokens(slot).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Truncate trailing KV on a live pool slot (MTP partial accept).
    pub fn truncate_kv_pool(
        &mut self,
        kv_pool: &mut KvCachePool,
        slot: KvSlot,
        tail_count: u32,
    ) -> Result<(), String> {
        kv_pool
            .with_slot_mut(slot, |slot_state| {
                slot_state.seq_len = slot_state.seq_len.saturating_sub(tail_count);
                slot_state.total_tokens =
                    slot_state.total_tokens.saturating_sub(tail_count as usize);
            })
            .map_err(|e| e.to_string())?;
        self.sync_kv_meta_from_pool(kv_pool, slot)
    }

    /// After a parallel prefill chunk (or the last chunk of a multi-chunk
    /// prefill), copy the final row of `prefill_scratch.normed_buf` into the
    /// decode-facing `normed_buf` for MTP `h_nextn`.
    ///
    /// Scratch is only `max_parallel_prefill_seq` rows — never the full prompt.
    /// Reading `token_ids.len()` rows here SIGSEGVs once the prompt exceeds one
    /// chunk (MTP serve path; non-MTP never touches this).
    fn capture_prefill_last_normed_row(&self, last_chunk_len: usize) {
        debug_assert!(last_chunk_len > 0);
        let hidden_size = self.config.hidden_size;
        let rows = last_chunk_len.min(self.prefill_scratch.max_seq_len).max(1);
        let normed =
            MetalContext::read_buffer(&self.prefill_scratch.normed_buf, rows * hidden_size);
        let last = (rows - 1) * hidden_size;
        MetalContext::write_buffer(&self.normed_buf, &normed[last..last + hidden_size]);
    }

    /// Parallel prefill into a pool slot; syncs model meta + last-row normed_buf.
    pub fn forward_prefill_pool(
        &mut self,
        token_ids: &[usize],
        kv_pool: &mut KvCachePool,
        slot: KvSlot,
    ) -> Result<Vec<f32>, String> {
        let logits = self.forward_prefill_chunked_with_kv_slot(token_ids, kv_pool, slot)?;
        self.sync_kv_meta_from_pool(kv_pool, slot)?;
        // Only the last chunk remains in prefill_scratch (sized max_seq≤1024).
        let chunk_size = self.max_parallel_prefill_seq().max(1);
        let rem = token_ids.len() % chunk_size;
        let last_chunk_len = if rem == 0 {
            chunk_size.min(token_ids.len())
        } else {
            rem
        };
        self.capture_prefill_last_normed_row(last_chunk_len);
        Ok(logits)
    }

    /// Append tokens via parallel prefill kernels on a pool slot.
    pub fn forward_continue_pool(
        &mut self,
        token_ids: &[usize],
        kv_pool: &mut KvCachePool,
        slot: KvSlot,
    ) -> Result<Vec<f32>, String> {
        if token_ids.is_empty() {
            return Err("continue token_ids must not be empty".to_string());
        }
        // Force the parallel writer even for seq_len==1 (decode fallback would
        // mix KV layouts and break later batched verify).
        self.prepare_parallel_prefill_inputs(token_ids)?;
        let start_pos = kv_pool.total_tokens(slot).map_err(|e| e.to_string())?;
        if !crate::gpu::prefill_gpu_rope_enabled() {
            self.prepare_parallel_prefill_rotary(start_pos, token_ids.len())?;
        }
        let logits = self.forward_prefill_chunk_parallel_with_kv_slot(
            token_ids,
            kv_pool,
            slot,
            start_pos,
            true,
            false,
        )?;
        self.sync_kv_meta_from_pool(kv_pool, slot)?;
        self.capture_prefill_last_normed_row(token_ids.len());
        Ok(logits)
    }

    /// Batched MTP verify on a pool slot. Parallel prefill-at-offset disagrees
    /// with decode-written KV; use decode-faithful sequential verify on the
    /// aliased pool buffers instead (see `forward_verify_batch`).
    pub fn forward_verify_pool(
        &mut self,
        token_ids: &[usize],
        kv_pool: &mut KvCachePool,
        slot: KvSlot,
    ) -> Result<Vec<usize>, String> {
        let _ = (kv_pool, slot);
        self.forward_verify_batch(token_ids)
    }

    /// Batched MTP verify: run draft tokens in a single command buffer.
    /// All tokens share the same KV slot, each at position base_pos + i.
    /// Returns argmax token for each row + populates mtp_verify_hidden_buf.
    pub fn forward_verify_batch(
        &mut self,
        token_ids: &[usize],
    ) -> Result<Vec<usize>, String> {
        if token_ids.is_empty() {
            return Err("verify batch token_ids must not be empty".to_string());
        }
        if token_ids.len() > MAX_MTP_VERIFY_SEQ {
            return Err(format!(
                "verify batch size {} exceeds MAX_MTP_VERIFY_SEQ={}",
                token_ids.len(),
                MAX_MTP_VERIFY_SEQ
            ));
        }

        let batch_size = token_ids.len();

        // Optional crosscheck: compare parallel vs fused-sequential on rewound KV.
        if std::env::var("MTP_VERIFY_CROSSCHECK").is_ok() && batch_size > 1 {
            return self.forward_verify_crosscheck(token_ids);
        }

        // Default: parallel prefill verify (batched ext matvec + causal attn).
        // Opt-in fused sequential (`MTP_VERIFY_SEQUENTIAL=1`) or decode-batch
        // (`MTP_VERIFY_DECODE_BATCH=1`). Per-row decode attention:
        // `MTP_VERIFY_DECODE_FA=1`.
        let want_sequential = std::env::var("MTP_VERIFY_SEQUENTIAL")
            .map(|v| v == "1")
            .unwrap_or(false);
        if want_sequential {
            if !self.fused_decode_eligible() {
                return self.forward_verify_chunk(token_ids);
            }
            return self.forward_verify_sequential(token_ids);
        }

        let want_parallel = batch_size > 1
            && batch_size <= self.prefill_scratch.max_seq_len
            && self.total_tokens + batch_size <= self.kv_capacity as usize
            && !std::env::var("MTP_VERIFY_DECODE_BATCH")
                .map(|v| v == "1")
                .unwrap_or(false);
        if want_parallel {
            return self.forward_verify_parallel(token_ids);
        }

        let want_decode_batch = std::env::var("MTP_VERIFY_DECODE_BATCH")
            .map(|v| v == "1")
            .unwrap_or(false);
        if want_decode_batch
            && batch_size > 1
            && batch_size <= self.max_decode_batch_size()
        {
            return self.forward_verify_decode_batch(token_ids);
        }

        if !self.fused_decode_eligible() {
            return self.forward_verify_chunk(token_ids);
        }
        self.forward_verify_sequential(token_ids)
    }

    fn mtp_bisect_stop_layer() -> Option<usize> {
        std::env::var("MTP_STOP_LAYER").ok().and_then(|v| v.parse().ok())
    }

    /// Compare parallel prefill verify against fused sequential on identical KV.
    fn forward_verify_crosscheck(&mut self, token_ids: &[usize]) -> Result<Vec<usize>, String> {
        let batch_size = token_ids.len();
        let ref_seq = self.kv_seq_len;
        let ref_total = self.total_tokens;

        if std::env::var("MTP_LAYER_BISECT").is_ok() {
            self.mtp_layer_bisect(token_ids[0], ref_seq, ref_total)?;
            std::process::exit(0);
        }

        let reference = self.forward_verify_sequential(token_ids)?;
        self.kv_seq_len = ref_seq;
        self.total_tokens = ref_total;

        // Single-row probe: does the parallel path agree on row 0 alone?
        let single = self.forward_verify_parallel(&token_ids[..1])?;
        self.kv_seq_len = ref_seq;
        self.total_tokens = ref_total;
        eprintln!(
            "MTP VERIFY CROSSCHECK single-row: parallel={} sequential={} {}",
            single[0],
            reference[0],
            if single[0] == reference[0] { "match" } else { "MISMATCH" }
        );

        let parallel = self.forward_verify_parallel(token_ids)?;
        self.kv_seq_len = ref_seq;
        self.total_tokens = ref_total;

        for (row, (&got, &want)) in parallel.iter().zip(reference.iter()).enumerate() {
            if got != want {
                eprintln!(
                    "MTP VERIFY CROSSCHECK row {}: parallel={} sequential={} MISMATCH",
                    row, got, want
                );
            }
        }
        if parallel == reference {
            eprintln!("MTP VERIFY CROSSCHECK: all {} rows match", batch_size);
        }

        // Apply reference result to KV (sequential path).
        let _ = self.forward_verify_sequential(token_ids)?;
        Ok(reference)
    }

    /// Bisect the first layer where the parallel prefill path diverges from the
    /// fused sequential decode path on one token over identical KV state.
    fn mtp_layer_bisect(
        &mut self,
        token: usize,
        ref_seq: u32,
        ref_total: usize,
    ) -> Result<(), String> {
        let hidden_size = self.config.hidden_size;
        let num_layers = self.layers.len();
        eprintln!("MTP LAYER BISECT: token={} base_seq={} total={}", token, ref_seq, ref_total);
        let mut compare = |model: &mut Self, label: &str| -> Result<(), String> {
            model.kv_seq_len = ref_seq;
            model.total_tokens = ref_total;
            let _ = model.forward_verify_sequential(&[token])?;
            let seq_hidden = MetalContext::read_buffer(&model.hidden_buf, hidden_size);

            model.kv_seq_len = ref_seq;
            model.total_tokens = ref_total;
            let _ = model.forward_verify_parallel(&[token])?;
            let par_hidden = MetalContext::read_buffer(
                &model.prefill_scratch.hidden_buf,
                hidden_size,
            );

            let mut dot = 0.0f64;
            let mut n1 = 0.0f64;
            let mut n2 = 0.0f64;
            let mut max_abs = 0.0f32;
            for i in 0..hidden_size {
                let a = seq_hidden[i];
                let b = par_hidden[i];
                dot += (a as f64) * (b as f64);
                n1 += (a as f64) * (a as f64);
                n2 += (b as f64) * (b as f64);
                max_abs = max_abs.max((a - b).abs());
            }
            let cos = if n1 > 0.0 && n2 > 0.0 {
                dot / (n1.sqrt() * n2.sqrt())
            } else {
                0.0
            };
            eprintln!(
                "  {}: cos={:.6} max_abs_diff={:.5} |seq|={:.3} |par|={:.3}",
                label,
                cos,
                max_abs,
                n1.sqrt(),
                n2.sqrt()
            );
            Ok(())
        };

        for stop in 0..=num_layers.min(3) {
            std::env::set_var("MTP_STOP_LAYER", stop.to_string());
            compare(self, &format!("stop_layer={:2} full", stop))?;
        }

        // Component ablation within layer 0 only.
        std::env::set_var("MTP_STOP_LAYER", "1");
        for ablate in ["attn", "mlp", "ple"] {
            std::env::set_var("PROFILE_ABLATE", ablate);
            compare(self, &format!("layer0 skip_{}", ablate))?;
        }
        std::env::remove_var("PROFILE_ABLATE");
        std::env::remove_var("MTP_STOP_LAYER");
        Ok(())
    }

    fn forward_verify_parallel(&mut self, token_ids: &[usize]) -> Result<Vec<usize>, String> {
        let batch_size = token_ids.len();
        let start_pos = self.total_tokens;
        let base_seq = self.kv_seq_len;
        let base_total = self.total_tokens;

        let (mut kv_pool, slot) = self.kv_pool_from_model_caches_at(base_seq, base_total);

        self.prepare_parallel_prefill_inputs(token_ids)?;
        if !crate::gpu::prefill_gpu_rope_enabled() {
            self.prepare_parallel_prefill_rotary(start_pos, batch_size)?;
        }

        self.forward_prefill_chunk_parallel_with_kv_slot(
            token_ids,
            &mut kv_pool,
            slot,
            start_pos,
            false,
            true,
        )?;

        self.kv_seq_len = kv_pool.seq_len(slot).map_err(|e| e.to_string())?;
        self.total_tokens = kv_pool.total_tokens(slot).map_err(|e| e.to_string())?;

        let hidden_size = self.config.hidden_size;
        let hidden =
            MetalContext::read_buffer(&self.mtp_verify_hidden_buf, batch_size * hidden_size);
        let last = (batch_size - 1) * hidden_size;
        MetalContext::write_buffer(&self.normed_buf, &hidden[last..]);

        Ok(self.read_mtp_verify_argmax_tokens(batch_size))
    }

    /// Parallel prefill into this model's own KV (same kernels as verify).
    /// MTP must stay on this path so verify does not mix decode-written KV.
    pub fn forward_prefill_parallel_self(&mut self, token_ids: &[usize]) -> Result<Vec<f32>, String> {
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }

        // TurboQuant hot-only: alias model caches (no fresh Metal alloc). Prefill
        // writes the Q4 hot ring on the model; pool K/V are unused unless dual-write.
        let use_alias = self.tq_prefill_fits_hot(self.kv_seq_len as usize, token_ids.len());
        let (mut kv_pool, slot) = if use_alias {
            self.kv_pool_from_model_caches()
        } else {
            // Fresh pool + swap: matches the historical Q4_0 / MTP prefill path.
            let mut kv_pool = self.create_kv_pool(1, self.kv_capacity);
            let slot = kv_pool
                .allocate()
                .ok_or_else(|| "failed to allocate MTP KV slot".to_string())?;
            kv_pool
                .with_slot_mut(slot, |slot_state| {
                    std::mem::swap(&mut self.k_cache, &mut slot_state.k_cache);
                    std::mem::swap(&mut self.v_cache, &mut slot_state.v_cache);
                    slot_state.seq_len = self.kv_seq_len;
                    slot_state.total_tokens = self.total_tokens;
                })
                .map_err(|e| e.to_string())?;
            (kv_pool, slot)
        };

        let logits = self.forward_prefill_chunked_with_kv_slot(token_ids, &mut kv_pool, slot)?;

        if use_alias {
            self.kv_seq_len = kv_pool.seq_len(slot).map_err(|e| e.to_string())?;
            self.total_tokens = kv_pool.total_tokens(slot).map_err(|e| e.to_string())?;
        } else {
            kv_pool
                .with_slot_mut(slot, |slot_state| {
                    std::mem::swap(&mut self.k_cache, &mut slot_state.k_cache);
                    std::mem::swap(&mut self.v_cache, &mut slot_state.v_cache);
                    self.kv_seq_len = slot_state.seq_len;
                    self.total_tokens = slot_state.total_tokens;
                })
                .map_err(|e| e.to_string())?;
        }

        // Last-row normed → decode-facing buffer for MTP h_nextn.
        let chunk_size = self.max_parallel_prefill_seq().max(1);
        let rem = token_ids.len() % chunk_size;
        let last_chunk_len = if rem == 0 {
            chunk_size.min(token_ids.len())
        } else {
            rem
        };
        self.capture_prefill_last_normed_row(last_chunk_len);

        Ok(logits)
    }

    /// Extend KV with one or more tokens via the parallel prefill kernels.
    pub fn forward_continue_parallel(&mut self, token_ids: &[usize]) -> Result<Vec<f32>, String> {
        self.forward_continue_parallel_chunk(token_ids, true)
    }

    fn forward_continue_parallel_chunk(
        &mut self,
        token_ids: &[usize],
        compute_logits: bool,
    ) -> Result<Vec<f32>, String> {
        if token_ids.is_empty() {
            return Err("continue token_ids must not be empty".to_string());
        }
        if token_ids.len() > self.prefill_scratch.max_seq_len {
            return Err(format!(
                "continue chunk has {} tokens, max is {}",
                token_ids.len(),
                self.prefill_scratch.max_seq_len
            ));
        }
        if self.total_tokens + token_ids.len() > self.kv_capacity as usize {
            return Err("KV capacity exceeded during parallel continue".to_string());
        }

        let mut kv_pool = self.create_kv_pool(1, self.kv_capacity);
        let slot = kv_pool
            .allocate()
            .ok_or_else(|| "failed to allocate MTP KV slot".to_string())?;
        kv_pool
            .with_slot_mut(slot, |slot_state| {
                std::mem::swap(&mut self.k_cache, &mut slot_state.k_cache);
                std::mem::swap(&mut self.v_cache, &mut slot_state.v_cache);
                slot_state.seq_len = self.kv_seq_len;
                slot_state.total_tokens = self.total_tokens;
            })
            .map_err(|e| e.to_string())?;

        let logits =
            self.forward_prefill_chunk_with_kv_slot(token_ids, &mut kv_pool, slot, compute_logits)?;

        kv_pool
            .with_slot_mut(slot, |slot_state| {
                std::mem::swap(&mut self.k_cache, &mut slot_state.k_cache);
                std::mem::swap(&mut self.v_cache, &mut slot_state.v_cache);
                self.kv_seq_len = slot_state.seq_len;
                self.total_tokens = slot_state.total_tokens;
            })
            .map_err(|e| e.to_string())?;

        if compute_logits {
            self.capture_prefill_last_normed_row(token_ids.len());
        }

        Ok(logits)
    }

    /// Decode-faithful same-sequence MTP verify.
    ///
    /// Reuses the multi-request decode batch executor with consecutive
    /// synthetic views of one aliased KV slot. The executor processes one
    /// layer across all rows, appending row `i` at `base_seq + i` and limiting
    /// that row's attention to the same position.
    fn forward_verify_decode_batch(
        &mut self,
        token_ids: &[usize],
    ) -> Result<Vec<usize>, String> {
        let batch_size = token_ids.len();
        if batch_size == 0 || batch_size > self.max_decode_batch_size() {
            return Err(format!(
                "MTP decode verify batch size {} exceeds supported range 1..={}",
                batch_size,
                self.max_decode_batch_size()
            ));
        }
        if self.total_tokens + batch_size > self.kv_capacity as usize {
            return Err("KV capacity exceeded during MTP decode verify".to_string());
        }

        let base_seq = self.kv_seq_len;
        let base_total = self.total_tokens;
        let (mut kv_pool, slot) = self.kv_pool_from_model_caches_at(base_seq, base_total);

        self.prepare_decode_batch_inputs(token_ids)?;
        let slot_views: Vec<KvSlotView> = (0..batch_size)
            .map(|i| KvSlotView {
                slot,
                slot_index: 0,
                seq_len: base_seq + i as u32,
                total_tokens: base_total + i,
            })
            .collect();
        self.prepare_decode_batch_rotary(&slot_views)?;
        self.prepare_decode_batch_attention_metadata(&slot_views)?;

        let inputs: Vec<(KvSlot, usize)> =
            token_ids.iter().copied().map(|token| (slot, token)).collect();
        let outputs = self.forward_decode_batch_encoded_with_kv_slots(
            &inputs,
            &slot_views,
            &mut kv_pool,
            true,
        )?;

        // Preserve decode-facing h_prev from the accepted verify row.
        let hidden_size = self.config.hidden_size;
        let hidden =
            MetalContext::read_buffer(&self.mtp_verify_hidden_buf, batch_size * hidden_size);
        let last = (batch_size - 1) * hidden_size;
        MetalContext::write_buffer(&self.normed_buf, &hidden[last..]);

        self.kv_seq_len = base_seq + batch_size as u32;
        self.total_tokens = base_total + batch_size;

        debug_assert!(outputs.is_empty());
        Ok(self.read_mtp_verify_argmax_tokens(batch_size))
    }

    fn forward_verify_sequential(
        &mut self,
        token_ids: &[usize],
    ) -> Result<Vec<usize>, String> {
        let batch_size = token_ids.len();
        let hidden_size = self.config.hidden_size;
        let ple_dim = self.config.hidden_size_per_layer_input;
        let ple_total_dim = self.config.num_hidden_layers * ple_dim;
        let vocab_size = self.config.vocab_size;
        let base_seq = self.kv_seq_len;
        let eps = self.config.rms_norm_eps as f32;
        let num_layers = self.config.num_hidden_layers as u32;
        let context_proj_scale = 1.0f32 / (hidden_size as f32).sqrt();
        let ple_input_scale = std::f32::consts::FRAC_1_SQRT_2;

        // Prepare embeddings and PLE for all tokens (CPU)
        let mut batch_hidden = vec![0.0f32; batch_size * hidden_size];
        let mut batch_ple = vec![0.0f32; batch_size * ple_total_dim];
        for i in 0..batch_size {
            let emb = &mut batch_hidden[i * hidden_size..(i + 1) * hidden_size];
            self.embed_tables.decode_embed_into(token_ids[i], hidden_size, emb);
            if ple_dim > 0 {
                let ple = &mut batch_ple[i * ple_total_dim..(i + 1) * ple_total_dim];
                self.embed_tables.decode_ple_into(token_ids[i], ple_total_dim, ple_dim, ple);
            }
        }

        MetalContext::write_buffer(&self.decode_batch_scratch.hidden_buf, &batch_hidden);
        MetalContext::write_buffer(&self.decode_batch_scratch.ple_token_id_buf, &batch_ple);

        let cmd = self.ctx.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        // Per-token processing within a single encoder
        for i in 0..batch_size {
            let kv_seq = base_seq + i as u32;
            let pos = (self.total_tokens + i) as f32;

            // Copy token embedding row i → hidden_buf
            self.ctx.encode_copy_at(
                encoder,
                &self.decode_batch_scratch.hidden_buf,
                (i * hidden_size * 4) as u64,
                &self.hidden_buf,
                0,
                hidden_size as u32,
            );

            // PLE context projection
            self.ctx.encode_matvec_auto_view(
                encoder,
                &self.per_layer_model_projection_weight,
                &self.hidden_buf,
                &self.ple_context_proj_buf,
                num_layers * ple_dim as u32,
                hidden_size as u32,
            );

            // PLE scale
            self.ctx.encode_vec_scale(
                encoder,
                &self.ple_context_proj_buf,
                &self.ple_combined_buf,
                num_layers * ple_dim as u32,
                context_proj_scale,
            );

            // PLE norm per layer
            let ple_off = (i * ple_total_dim * 4) as u64;
            self.ctx.encode_rmsnorm_per_head_view(
                encoder,
                &self.ple_combined_buf,
                &self.per_layer_projection_norm_weight,
                &self.ple_context_proj_buf,
                num_layers as u32,
                ple_dim as u32,
                eps,
            );

            // PLE add token identity
            self.ctx.encode_copy_at(
                encoder,
                &self.decode_batch_scratch.ple_token_id_buf,
                ple_off,
                &self.ple_token_id_buf,
                0,
                ple_total_dim as u32,
            );
            self.ctx.encode_vec_add(
                encoder,
                &self.ple_context_proj_buf,
                &self.ple_token_id_buf,
                &self.ple_combined_buf,
                num_layers * ple_dim as u32,
            );

            // PLE final scale
            self.ctx.encode_vec_scale(
                encoder,
                &self.ple_combined_buf,
                &self.ple_context_proj_buf,
                num_layers * ple_dim as u32,
                ple_input_scale,
            );

            // RoPE fill for this token's position
            self.ctx.encode_rope_fill_decode(
                encoder,
                &self.decode_rope_cos_packed,
                &self.decode_rope_sin_packed,
                &self.rope_layer_params_buf,
                self.layers.len() as u32,
                self.rope_max_head_dim as u32,
                pos,
            );

            // Fused decode layers
            let scratch = crate::decode_fused::FusedDecodeScratch {
                hidden: &self.hidden_buf,
                normed: &self.normed_buf,
                inv_rms: &self.inv_rms_buf,
                q: &self.q_buf,
                k: &self.k_buf,
                v: &self.v_buf,
                q_normed: &self.q_normed_buf,
                k_normed: &self.k_normed_buf,
                attn_out: &self.attn_out_buf,
                o_out: &self.o_out_buf,
                gate: &self.gate_buf,
                up: &self.up_buf,
                gelu: &self.gelu_buf,
                down: &self.down_buf,
                ple_ctx: &self.ple_context_proj_buf,
                ple_normed: &self.ple_normed_buf,
                ple_projected: &self.ple_projected_buf,
                cos_packed: &self.decode_rope_cos_packed,
                sin_packed: &self.decode_rope_sin_packed,
            };
            let stop_layer = Self::mtp_bisect_stop_layer().unwrap_or(self.layers.len());
            let ablate = crate::gpu::ProfileAblate::from_env();
            for layer_idx in 0..self.layers.len().min(stop_layer) {
                self.encode_fused_decode_layer(
                    encoder,
                    layer_idx,
                    kv_seq,
                    &scratch,
                    ablate.skip_attn(),
                    ablate.skip_mlp(),
                    ablate.skip_ple(),
                );
            }

            // Final norm (hidden_buf → normed_buf)
            self.ctx.encode_rmsnorm_view(
                encoder,
                &self.hidden_buf,
                &self.final_norm_weight,
                &self.normed_buf,
                hidden_size as u32,
                eps,
            );

            // LM head: logits → temp logits_buf, then copy to batch output
            self.ctx.encode_matvec_auto_view(
                encoder,
                &self.lm_head_buf,
                &self.normed_buf,
                &self.logits_buf,
                vocab_size as u32,
                hidden_size as u32,
            );
            self.ctx.encode_copy_at(
                encoder,
                &self.logits_buf,
                0,
                &self.mtp_verify_logits_buf,
                (i * vocab_size * 4) as u64,
                vocab_size as u32,
            );

            // Copy normed → mtp_verify_hidden_buf row i (for h_nextn)
            self.ctx.encode_copy_at(
                encoder,
                &self.normed_buf,
                0,
                &self.mtp_verify_hidden_buf,
                (i * hidden_size * 4) as u64,
                hidden_size as u32,
            );
        }

        encoder.end_encoding();

        // GPU softcap + argmax over all verify rows (avoids full-vocab readback).
        let cap = self.config.final_logit_softcapping;
        let argmax_encoder = cmd.new_compute_command_encoder();
        self.ctx.encode_softcap_argmax_rows_f32(
            argmax_encoder,
            &self.mtp_verify_logits_buf,
            &self.mtp_verify_argmax_buf,
            batch_size as u32,
            vocab_size as u32,
            cap,
        );
        argmax_encoder.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        let raw_tokens = MetalContext::read_u32_buffer(&self.mtp_verify_argmax_buf, batch_size);
        let tokens: Vec<usize> = raw_tokens.into_iter().map(|t| t as usize).collect();

        // Update global state: KV length and total_tokens advance by batch
        self.kv_seq_len = base_seq + batch_size as u32;
        self.total_tokens += batch_size;

        Ok(tokens)
    }

    fn read_mtp_verify_argmax_tokens(&self, batch_size: usize) -> Vec<usize> {
        MetalContext::read_u32_buffer(&self.mtp_verify_argmax_buf, batch_size)
            .into_iter()
            .map(|t| t as usize)
            .collect()
    }

    /// Batched MTP verify: run draft tokens and return argmax token after each row.
    /// Uses sequential single-token passes (simplest, matches fallback path).
    /// After each verify step, copies the post-norm hidden state to
    /// mtp_verify_hidden_buf so prefill_hidden_activation_at() returns valid data.
    pub fn forward_verify_chunk(
        &mut self,
        token_ids: &[usize],
    ) -> Result<Vec<usize>, String> {
        if token_ids.is_empty() {
            return Err("verify chunk token_ids must not be empty".to_string());
        }
        let hidden_size = self.config.hidden_size;
        let mut tokens = Vec::with_capacity(token_ids.len());
        for (i, &token_id) in token_ids.iter().enumerate() {
            let logits = self.forward_single_token(token_id);
            tokens.push(Self::argmax_cpu(&logits));
            // Copy normed_buf → mtp_verify_hidden_buf row i for later h_nextn readback
            let blit = self.ctx.queue.new_command_buffer();
            let blit_enc = blit.new_blit_command_encoder();
            blit_enc.copy_from_buffer(
                &self.normed_buf,
                0,
                &self.mtp_verify_hidden_buf,
                (i * hidden_size * 4) as u64,
                (hidden_size * 4) as u64,
            );
            blit_enc.end_encoding();
            blit.commit();
            blit.wait_until_completed();
        }
        Ok(tokens)
    }

    fn argmax_cpu(values: &[f32]) -> usize {
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (idx, &value) in values.iter().enumerate() {
            if value > best_val {
                best_val = value;
                best_idx = idx;
            }
        }
        best_idx
    }

    pub fn tq_pool_config(&self) -> TqPoolConfig {
        TqPoolConfig {
            hot_w: if self.tq_hot_enabled() {
                self.tq_hot_w
            } else {
                0
            },
            rw: if self.tq_hot_enabled() && self.tq_rw > 0 {
                self.tq_rw
            } else {
                0
            },
        }
    }

    /// One-slot pool aliasing this model's cold (+ TQ hot/rw) buffers.
    fn kv_pool_from_model_caches(&self) -> (KvCachePool, KvSlot) {
        let tq = self.tq_pool_config();
        let tq_hot = if self.tq_hot_enabled() && !self.tq_hot_k.is_empty() {
            Some((self.tq_hot_k.as_slice(), self.tq_hot_v.as_slice()))
        } else {
            None
        };
        let tq_rw = if self.tq_rw > 0 && !self.tq_rw_k.is_empty() {
            Some((self.tq_rw_k.as_slice(), self.tq_rw_v.as_slice()))
        } else {
            None
        };
        KvCachePool::from_existing(
            &self.k_cache,
            &self.v_cache,
            self.kv_seq_len,
            self.total_tokens,
            self.kv_capacity,
            self.kv_cache_type,
            tq,
            tq_hot,
            tq_rw,
            self.tq_hot_spilled,
        )
    }

    fn kv_pool_from_model_caches_at(
        &self,
        seq_len: u32,
        total_tokens: usize,
    ) -> (KvCachePool, KvSlot) {
        let tq = self.tq_pool_config();
        let tq_hot = if self.tq_hot_enabled() && !self.tq_hot_k.is_empty() {
            Some((self.tq_hot_k.as_slice(), self.tq_hot_v.as_slice()))
        } else {
            None
        };
        let tq_rw = if self.tq_rw > 0 && !self.tq_rw_k.is_empty() {
            Some((self.tq_rw_k.as_slice(), self.tq_rw_v.as_slice()))
        } else {
            None
        };
        KvCachePool::from_existing(
            &self.k_cache,
            &self.v_cache,
            seq_len,
            total_tokens,
            self.kv_capacity,
            self.kv_cache_type,
            tq,
            tq_hot,
            tq_rw,
            self.tq_hot_spilled,
        )
    }

    pub fn create_kv_pool(&self, num_slots: usize, max_seq_len: u32) -> KvCachePool {
        let max_seq_len = max_seq_len.min(self.kv_capacity as u32);
        KvCachePool::new(
            &self.ctx,
            &self.config,
            num_slots,
            max_seq_len,
            self.kv_cache_type,
            self.tq_pool_config(),
        )
    }

    pub fn max_parallel_prefill_seq(&self) -> usize {
        self.prefill_scratch.max_seq_len
    }

    pub fn max_decode_batch_size(&self) -> usize {
        self.decode_batch_scratch.max_batch_size
    }

    pub fn prepare_parallel_prefill_inputs(&mut self, token_ids: &[usize]) -> Result<(), String> {
        if token_ids.is_empty() {
            return Err("prefill token_ids must not be empty".to_string());
        }
        if token_ids.len() > self.prefill_scratch.max_seq_len {
            return Err(format!(
                "prefill chunk has {} tokens, max supported chunk is {}",
                token_ids.len(),
                self.prefill_scratch.max_seq_len
            ));
        }

        let inputs = self.prepare_batched_token_inputs(token_ids)?;
        MetalContext::write_buffer(&self.prefill_scratch.hidden_buf, &inputs.hidden);
        MetalContext::write_buffer(
            &self.prefill_scratch.ple_token_id_buf,
            &inputs.ple_token_identity,
        );
        Ok(())
    }

    fn prepare_parallel_prefill_rotary(
        &mut self,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<(), String> {
        self.prepare_parallel_prefill_rotary_segments(&[(start_pos, seq_len)])
    }

    fn prepare_parallel_prefill_rotary_segments(
        &mut self,
        segments: &[(usize, usize)],
    ) -> Result<(), String> {
        let total_seq_len: usize = segments.iter().map(|(_, seq_len)| *seq_len).sum();
        if total_seq_len == 0 {
            return Err("prefill seq_len must not be empty".to_string());
        }
        if total_seq_len > self.prefill_scratch.max_seq_len {
            return Err(format!(
                "prefill chunk has {} tokens, max supported chunk is {}",
                total_seq_len, self.prefill_scratch.max_seq_len
            ));
        }

        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            let intermediate_size = layer.intermediate_size;
            let head_dim = layer.head_dim;
            let half_dim = head_dim / 2;
            let is_full = layer.is_full_attention;
            let rope_theta = if is_full {
                self.config.full_rope_theta()
            } else {
                self.config.sliding_rope_theta()
            };
            let rope_factor = if is_full {
                self.config.full_rope_factor()
            } else {
                self.config.sliding_rope_factor()
            };
            let rotary_dim = if is_full {
                (head_dim as f64 * self.config.full_partial_rotary_factor()) as usize
            } else {
                head_dim
            };
            let rope_angles = rotary_dim / 2;

            let mut cos_batch = vec![0.0f32; total_seq_len * head_dim];
            let mut sin_batch = vec![0.0f32; total_seq_len * head_dim];

            let mut row_idx = 0;
            for &(start_pos, seq_len) in segments {
                for token_idx in 0..seq_len {
                    let pos = (start_pos + token_idx) as f32;
                    let token_offset = row_idx * head_dim;

                    for i in 0..rope_angles {
                        let inv_freq = 1.0
                            / (rope_theta.powf(i as f64 * 2.0 / head_dim as f64) as f32)
                            / rope_factor as f32;
                        let angle = pos * inv_freq;
                        cos_batch[token_offset + i] = angle.cos();
                        cos_batch[token_offset + i + half_dim] = angle.cos();
                        sin_batch[token_offset + i] = angle.sin();
                        sin_batch[token_offset + i + half_dim] = angle.sin();
                    }

                    for i in rope_angles..half_dim {
                        cos_batch[token_offset + i] = 1.0;
                        cos_batch[token_offset + i + half_dim] = 1.0;
                    }

                    row_idx += 1;
                }
            }

            MetalContext::write_buffer(&self.per_layer_prefill_cos_bufs[layer_idx], &cos_batch);
            MetalContext::write_buffer(&self.per_layer_prefill_sin_bufs[layer_idx], &sin_batch);
        }

        Ok(())
    }

    pub fn prepare_decode_batch_inputs(&mut self, token_ids: &[usize]) -> Result<(), String> {
        if token_ids.is_empty() {
            return Err("decode batch token_ids must not be empty".to_string());
        }
        if token_ids.len() > self.decode_batch_scratch.max_batch_size {
            return Err(format!(
                "decode batch has {} requests, max supported batch is {}",
                token_ids.len(),
                self.decode_batch_scratch.max_batch_size
            ));
        }

        let inputs = self.prepare_batched_token_inputs(token_ids)?;
        MetalContext::write_buffer(&self.decode_batch_scratch.hidden_buf, &inputs.hidden);
        MetalContext::write_buffer(
            &self.decode_batch_scratch.ple_token_id_buf,
            &inputs.ple_token_identity,
        );
        Ok(())
    }

    fn prepare_batched_token_inputs(
        &self,
        token_ids: &[usize],
    ) -> Result<BatchedTokenInputs, String> {
        let hidden_size = self.config.hidden_size;
        let num_layers = self.config.num_hidden_layers;
        let ple_dim = self.config.hidden_size_per_layer_input;
        let ple_total_dim = num_layers * ple_dim;

        let mut hidden = vec![0.0f32; token_ids.len() * hidden_size];
        let mut ple_token_identity = vec![0.0f32; token_ids.len() * ple_total_dim];

        for (pos, &token_id) in token_ids.iter().enumerate() {
            let hidden_offset = pos * hidden_size;
            self.embed_tables.decode_embed_into(
                token_id,
                hidden_size,
                &mut hidden[hidden_offset..hidden_offset + hidden_size],
            );

            if ple_dim > 0 {
                let ple_out_offset = pos * ple_total_dim;
                self.embed_tables.decode_ple_into(
                    token_id,
                    ple_total_dim,
                    ple_dim,
                    &mut ple_token_identity[ple_out_offset..ple_out_offset + ple_total_dim],
                );
            }
        }

        Ok(BatchedTokenInputs {
            batch_size: token_ids.len(),
            hidden,
            ple_token_identity,
        })
    }

    fn prepare_decode_batch_rotary(&mut self, slot_views: &[KvSlotView]) -> Result<(), String> {
        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            let intermediate_size = layer.intermediate_size;
            let head_dim = layer.head_dim;
            let half_dim = head_dim / 2;
            let is_full = layer.is_full_attention;
            let rope_theta = if is_full {
                self.config.full_rope_theta()
            } else {
                self.config.sliding_rope_theta()
            };
            let rope_factor = if is_full {
                self.config.full_rope_factor()
            } else {
                self.config.sliding_rope_factor()
            };
            let rotary_dim = if is_full {
                (head_dim as f64 * self.config.full_partial_rotary_factor()) as usize
            } else {
                head_dim
            };
            let rope_angles = rotary_dim / 2;

            let mut cos_batch = vec![0.0f32; slot_views.len() * head_dim];
            let mut sin_batch = vec![0.0f32; slot_views.len() * head_dim];

            for (batch_idx, slot_view) in slot_views.iter().enumerate() {
                let pos = slot_view.total_tokens as f32;
                let batch_offset = batch_idx * head_dim;

                for i in 0..rope_angles {
                    let inv_freq = 1.0
                        / (rope_theta.powf(i as f64 * 2.0 / head_dim as f64) as f32)
                        / rope_factor as f32;
                    let angle = pos * inv_freq;
                    cos_batch[batch_offset + i] = angle.cos();
                    cos_batch[batch_offset + i + half_dim] = angle.cos();
                    sin_batch[batch_offset + i] = angle.sin();
                    sin_batch[batch_offset + i + half_dim] = angle.sin();
                }

                for i in rope_angles..half_dim {
                    cos_batch[batch_offset + i] = 1.0;
                    cos_batch[batch_offset + i + half_dim] = 1.0;
                }
            }

            MetalContext::write_buffer(
                &self.per_layer_decode_batch_cos_bufs[layer_idx],
                &cos_batch,
            );
            MetalContext::write_buffer(
                &self.per_layer_decode_batch_sin_bufs[layer_idx],
                &sin_batch,
            );
        }

        Ok(())
    }

    fn prepare_decode_batch_attention_metadata(
        &mut self,
        slot_views: &[KvSlotView],
    ) -> Result<(), String> {
        let mut append_positions = vec![0u32; slot_views.len()];
        for (batch_idx, slot_view) in slot_views.iter().enumerate() {
            append_positions[batch_idx] = slot_view.seq_len;
        }

        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            let mut kv_starts = vec![0u32; slot_views.len()];
            let mut kv_lens = vec![0u32; slot_views.len()];

            for (batch_idx, &append_pos) in append_positions.iter().enumerate() {
                let attn_kv_seq = append_pos + 1;
                if layer.is_full_attention {
                    kv_starts[batch_idx] = 0;
                    kv_lens[batch_idx] = attn_kv_seq;
                } else {
                    let window = self.config.sliding_window as u32;
                    kv_lens[batch_idx] = attn_kv_seq.min(window);
                    kv_starts[batch_idx] = attn_kv_seq.saturating_sub(window);
                }
            }

            MetalContext::write_u32_buffer(
                &self.per_layer_decode_batch_append_pos_bufs[layer_idx],
                &append_positions,
            );
            MetalContext::write_u32_buffer(
                &self.per_layer_decode_batch_kv_start_bufs[layer_idx],
                &kv_starts,
            );
            MetalContext::write_u32_buffer(
                &self.per_layer_decode_batch_kv_len_bufs[layer_idx],
                &kv_lens,
            );
        }

        Ok(())
    }

    fn f32_byte_offset(elements: usize) -> u64 {
        (elements * std::mem::size_of::<f32>()) as u64
    }

    fn decode_batch_row_offsets(&self, batch_idx: usize) -> DecodeBatchRowOffsets {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.max_intermediate_size();
        let ple_dim = self.config.hidden_size_per_layer_input;
        let ple_total_dim = self.config.num_hidden_layers * ple_dim;
        let max_q_out = self
            .layers
            .iter()
            .map(|layer| layer.q_out_dim)
            .max()
            .unwrap_or(0);
        let max_kv_out = self
            .layers
            .iter()
            .map(|layer| layer.kv_out_dim)
            .max()
            .unwrap_or(0);

        DecodeBatchRowOffsets {
            hidden: Self::f32_byte_offset(batch_idx * hidden_size),
            q: Self::f32_byte_offset(batch_idx * max_q_out),
            kv: Self::f32_byte_offset(batch_idx * max_kv_out),
            intermediate: Self::f32_byte_offset(batch_idx * intermediate_size),
            ple_row: Self::f32_byte_offset(batch_idx * ple_total_dim),
        }
    }

    fn decode_batch_layer_ple_offset(&self, batch_idx: usize, layer_idx: usize) -> u64 {
        let ple_dim = self.config.hidden_size_per_layer_input;
        let ple_total_dim = self.config.num_hidden_layers * ple_dim;
        Self::f32_byte_offset(batch_idx * ple_total_dim + layer_idx * ple_dim)
    }

    fn prefill_row_offsets(&self, token_idx: usize) -> PrefillRowOffsets {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.max_intermediate_size();
        let ple_dim = self.config.hidden_size_per_layer_input;
        let ple_total_dim = self.config.num_hidden_layers * ple_dim;
        let max_q_out = self
            .layers
            .iter()
            .map(|layer| layer.q_out_dim)
            .max()
            .unwrap_or(0);
        let max_kv_out = self
            .layers
            .iter()
            .map(|layer| layer.kv_out_dim)
            .max()
            .unwrap_or(0);

        PrefillRowOffsets {
            hidden: Self::f32_byte_offset(token_idx * hidden_size),
            q: Self::f32_byte_offset(token_idx * max_q_out),
            kv: Self::f32_byte_offset(token_idx * max_kv_out),
            intermediate: Self::f32_byte_offset(token_idx * intermediate_size),
            ple_row: Self::f32_byte_offset(token_idx * ple_total_dim),
        }
    }

    fn encode_parallel_prefill_ple_context_on_encoder(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        seq_len: usize,
    ) -> Result<(), String> {
        if seq_len == 0 {
            return Err("prefill seq_len must not be empty".to_string());
        }
        if seq_len > self.prefill_scratch.max_seq_len {
            return Err(format!(
                "prefill chunk has {} tokens, max supported chunk is {}",
                seq_len, self.prefill_scratch.max_seq_len
            ));
        }

        let hidden_size = self.config.hidden_size;
        let num_layers = self.config.num_hidden_layers;
        let ple_dim = self.config.hidden_size_per_layer_input;
        let ple_total_dim = num_layers * ple_dim;
        let eps = self.config.rms_norm_eps as f32;
        let context_proj_scale = 1.0f32 / (hidden_size as f32).sqrt();
        let ple_input_scale = 1.0f32 / 2.0f32.sqrt();
        let total_ple = (seq_len * ple_total_dim) as u32;

        self.ctx.encode_prefill_projection_auto_batch_view(
            encoder,
            &self.per_layer_model_projection_weight,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.ple_context_proj_buf,
            ple_total_dim as u32,
            hidden_size as u32,
            seq_len as u32,
        );
        self.ctx.encode_vec_scale(
            encoder,
            &self.prefill_scratch.ple_context_proj_buf,
            &self.prefill_scratch.ple_combined_buf,
            total_ple,
            context_proj_scale,
        );
        self.ctx.encode_rmsnorm_batch_view(
            encoder,
            &self.prefill_scratch.ple_combined_buf,
            &self.per_layer_projection_norm_weight,
            &self.prefill_scratch.ple_context_proj_buf,
            ple_dim as u32,
            eps,
            (seq_len * num_layers) as u32,
        );
        self.ctx.encode_vec_add_batch(
            encoder,
            &self.prefill_scratch.ple_context_proj_buf,
            &self.prefill_scratch.ple_token_id_buf,
            &self.prefill_scratch.ple_combined_buf,
            total_ple,
        );
        self.ctx.encode_vec_scale(
            encoder,
            &self.prefill_scratch.ple_combined_buf,
            &self.prefill_scratch.ple_context_proj_buf,
            total_ple,
            ple_input_scale,
        );
        Ok(())
    }

    fn encode_parallel_prefill_ple_context(&mut self, seq_len: usize) -> Result<(), String> {
        let cmd = self.ctx.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        self.encode_parallel_prefill_ple_context_on_encoder(encoder, seq_len)?;
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    fn encode_parallel_prefill_attention_inputs(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        layer_idx: usize,
        seq_len: usize,
    ) -> Result<(), String> {
        if seq_len == 0 {
            return Err("prefill seq_len must not be empty".to_string());
        }
        if seq_len > self.prefill_scratch.max_seq_len {
            return Err(format!(
                "prefill chunk has {} tokens, max supported chunk is {}",
                seq_len, self.prefill_scratch.max_seq_len
            ));
        }

        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| format!("invalid layer index {}", layer_idx))?;
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = layer.kv_out_dim / layer.head_dim;
        let head_dim = layer.head_dim;
        let eps = self.config.rms_norm_eps as f32;

        self.ctx.encode_rmsnorm_batch_view(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &layer.input_layernorm_weight,
            &self.prefill_scratch.normed_buf,
            hidden_size as u32,
            eps,
            seq_len as u32,
        );

        self.encode_prefill_attention_qkv(
            encoder,
            layer,
            seq_len as u32,
            hidden_size as u32,
        );

        let stacked = Self::use_prefill_qkv_stacked(layer);
        if crate::gpu::prefill_qkv_hsd_enabled() {
            self.ctx.encode_prefill_qkv_postproj_hsd(
                encoder,
                if stacked {
                    Some(&self.prefill_scratch.qkv_stacked_buf)
                } else {
                    None
                },
                &self.prefill_scratch.q_buf,
                &self.prefill_scratch.k_buf,
                &self.prefill_scratch.v_buf,
                &self.prefill_scratch.q_normed_buf,
                &self.prefill_scratch.k_normed_buf,
                &layer.q_norm_weight,
                &layer.k_norm_weight,
                layer.q_out_dim as u32,
                layer.kv_out_dim as u32,
                num_heads as u32,
                num_kv_heads as u32,
                head_dim as u32,
                seq_len as u32,
                eps,
                stacked,
                layer.has_kv,
            );
        } else {
            if stacked {
                self.ctx.encode_qkv_split_stacked_batch(
                    encoder,
                    &self.prefill_scratch.qkv_stacked_buf,
                    &self.prefill_scratch.q_buf,
                    &self.prefill_scratch.k_buf,
                    &self.prefill_scratch.v_buf,
                    layer.q_out_dim as u32,
                    layer.kv_out_dim as u32,
                    seq_len as u32,
                );
            }
            self.ctx.encode_rmsnorm_batch_view(
                encoder,
                &self.prefill_scratch.q_buf,
                &layer.q_norm_weight,
                &self.prefill_scratch.q_normed_buf,
                head_dim as u32,
                eps,
                (seq_len * num_heads) as u32,
            );

            if layer.has_kv {
                self.ctx.encode_rmsnorm_batch_view(
                    encoder,
                    &self.prefill_scratch.k_buf,
                    &layer.k_norm_weight,
                    &self.prefill_scratch.k_normed_buf,
                    head_dim as u32,
                    eps,
                    (seq_len * num_kv_heads) as u32,
                );
            }

            self.ctx.encode_transpose_shd(
                encoder,
                &self.prefill_scratch.q_normed_buf,
                &self.prefill_scratch.q_buf,
                seq_len as u32,
                num_heads as u32,
                head_dim as u32,
            );

            if layer.has_kv {
                self.ctx.encode_transpose_shd(
                    encoder,
                    &self.prefill_scratch.k_normed_buf,
                    &self.prefill_scratch.k_buf,
                    seq_len as u32,
                    num_kv_heads as u32,
                    head_dim as u32,
                );

                self.ctx.encode_rmsnorm_noweight_batch(
                    encoder,
                    &self.prefill_scratch.v_buf,
                    &self.prefill_scratch.k_normed_buf,
                    head_dim as u32,
                    eps,
                    (seq_len * num_kv_heads) as u32,
                );

                self.ctx.encode_transpose_shd(
                    encoder,
                    &self.prefill_scratch.k_normed_buf,
                    &self.prefill_scratch.v_buf,
                    seq_len as u32,
                    num_kv_heads as u32,
                    head_dim as u32,
                );
            }
        }

        self.ctx.encode_rotary_batch(
            encoder,
            &self.prefill_scratch.q_buf,
            &self.prefill_scratch.k_buf,
            &self.per_layer_prefill_cos_bufs[layer_idx],
            &self.per_layer_prefill_sin_bufs[layer_idx],
            num_heads as u32,
            if layer.has_kv { num_kv_heads as u32 } else { 0 },
            head_dim as u32,
            seq_len as u32,
        );

        Ok(())
    }

    fn encode_prefill_gpu_rotary_tables(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<(), String> {
        for layer_idx in 0..self.layers.len() {
            let layer = self
                .layers
                .get(layer_idx)
                .ok_or_else(|| format!("invalid layer index {}", layer_idx))?;
            self.ctx.encode_rope_fill_prefill_batch(
                encoder,
                &self.per_layer_prefill_cos_bufs[layer_idx],
                &self.per_layer_prefill_sin_bufs[layer_idx],
                &self.rope_layer_params_buf,
                layer_idx as u32,
                start_pos as u32,
                seq_len as u32,
                layer.head_dim as u32,
            );
        }
        Ok(())
    }

    fn can_use_parallel_prefill_chunk(
        &self,
        start_pos: usize,
        seq_len: usize,
        kv_pool: &KvCachePool,
    ) -> bool {
        if seq_len <= 1 || seq_len > self.prefill_scratch.max_seq_len {
            return false;
        }
        if start_pos + seq_len > kv_pool.capacity() as usize {
            return false;
        }
        // TurboQuant parallel prefill is only implemented for the Q4 hot window.
        if matches!(self.kv_cache_type, KvCacheType::TurboQuant { .. }) {
            return self.tq_prefill_fits_hot(start_pos, seq_len);
        }

        true
    }

    /// Dual-write one prefill chunk: Q4 into `hot_k`/`hot_v` + optional TQ V3 into cold.
    /// Requires `tq_prefill_fits_hot(start_pos, seq_len)`.
    fn encode_tq_hot_prefill_kv_append(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        layer_idx: usize,
        k_src: &Buffer,
        v_src: &Buffer,
        hot_k: &Buffer,
        hot_v: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        rw_k: Option<&Buffer>,
        rw_v: Option<&Buffer>,
        num_kv_heads: u32,
        head_dim: u32,
        start_pos: u32,
        seq_len: u32,
    ) -> Result<(), String> {
        if !self.tq_prefill_fits_hot(start_pos as usize, seq_len as usize) {
            return Err(TURBOQUANT_UNSUPPORTED.to_string());
        }
        let KvCacheType::TurboQuant { k_bits, v_bits } = self.kv_cache_type else {
            return Err(TURBOQUANT_UNSUPPORTED.to_string());
        };

        self.ctx.encode_kv_batch_append_q4_0(
            encoder,
            k_src,
            hot_k,
            num_kv_heads,
            head_dim,
            self.tq_hot_w,
            start_pos,
            seq_len,
        );
        self.ctx.encode_kv_batch_append_q4_0(
            encoder,
            v_src,
            hot_v,
            num_kv_heads,
            head_dim,
            self.tq_hot_w,
            start_pos,
            seq_len,
        );

        let dual_write = matches!(
            std::env::var("TURBOQUANT_DUAL_WRITE").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE")
        );
        if dual_write {
            let tq = self
                .turboquant
                .as_ref()
                .ok_or_else(|| TURBOQUANT_UNSUPPORTED.to_string())?;
            let fwd = tq.fwd(head_dim as usize);
            let k_row_bytes = self.kv_cache_type.k_row_bytes(head_dim as usize) as u32;
            let v_row_bytes = self.kv_cache_type.v_row_bytes(head_dim as usize) as u32;
            let kwin = rw_k.unwrap_or(&self.tq_k_rot);
            let vwin = rw_v.unwrap_or(&self.tq_v_rot);
            for s in 0..seq_len {
                let cur = start_pos + s;
                self.ctx.encode_turboquant_rotate_quant_v3_strided(
                    encoder,
                    k_src,
                    k_cache,
                    tq.centroids_k(head_dim as usize),
                    num_kv_heads,
                    head_dim,
                    self.kv_capacity,
                    cur,
                    k_bits as u32,
                    k_row_bytes,
                    kwin,
                    self.tq_rw,
                    fwd,
                    seq_len,
                    s,
                );
                self.ctx.encode_turboquant_rotate_quant_v3_strided(
                    encoder,
                    v_src,
                    v_cache,
                    tq.centroids_v(head_dim as usize),
                    num_kv_heads,
                    head_dim,
                    self.kv_capacity,
                    cur,
                    v_bits as u32,
                    v_row_bytes,
                    vwin,
                    self.tq_rw,
                    fwd,
                    seq_len,
                    s,
                );
            }
        } else {
            let _ = (k_cache, v_cache, k_bits, v_bits, layer_idx);
        }
        Ok(())
    }

    /// Resolve slot (or model) TQ hot / residual buffers for a layer.
    fn tq_slot_hot_bufs<'a>(
        &'a self,
        kv_pool: &'a KvCachePool,
        slot: KvSlot,
        layer_idx: usize,
    ) -> Result<(&'a Buffer, &'a Buffer), String> {
        if kv_pool.has_tq_hot(slot).unwrap_or(false) {
            let k = kv_pool
                .layer_tq_hot_k(slot, layer_idx)
                .map_err(|e| e.to_string())?;
            let v = kv_pool
                .layer_tq_hot_v(slot, layer_idx)
                .map_err(|e| e.to_string())?;
            Ok((k, v))
        } else if self.tq_hot_enabled() && layer_idx < self.tq_hot_k.len() {
            Ok((&self.tq_hot_k[layer_idx], &self.tq_hot_v[layer_idx]))
        } else {
            Err(TURBOQUANT_UNSUPPORTED.to_string())
        }
    }

    fn tq_slot_rw_bufs<'a>(
        &'a self,
        kv_pool: &'a KvCachePool,
        slot: KvSlot,
        layer_idx: usize,
    ) -> (Option<&'a Buffer>, Option<&'a Buffer>) {
        if self.tq_rw > 0 && kv_pool.tq_rw > 0 {
            if let (Ok(k), Ok(v)) = (
                kv_pool.layer_tq_rw_k(slot, layer_idx),
                kv_pool.layer_tq_rw_v(slot, layer_idx),
            ) {
                return (Some(k), Some(v));
            }
        }
        if self.tq_rw > 0 && layer_idx < self.tq_rw_k.len() {
            (Some(&self.tq_rw_k[layer_idx]), Some(&self.tq_rw_v[layer_idx]))
        } else {
            (None, None)
        }
    }

    fn encode_parallel_prefill_layer(
        &mut self,
        layer_idx: usize,
        seq_len: usize,
        kv_pool: &mut KvCachePool,
        slot: KvSlot,
        start_pos: usize,
    ) -> Result<(), String> {
        let cmd = self.ctx.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        self.encode_parallel_prefill_attention_inputs(encoder, layer_idx, seq_len)?;

        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| format!("invalid layer index {}", layer_idx))?;
        let hidden_size = self.config.hidden_size;
        let intermediate_size = layer.intermediate_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = layer.kv_out_dim / layer.head_dim;
        let num_kv_groups = (num_heads / num_kv_heads) as u32;
        let ple_dim = self.config.hidden_size_per_layer_input;
        let eps = self.config.rms_norm_eps as f32;
        let head_dim = layer.head_dim;
        let q_out = layer.q_out_dim;
        let kv_out = layer.kv_out_dim;
        let total_hidden = (seq_len * hidden_size) as u32;
        let total_intermediate = (seq_len * intermediate_size) as u32;
        let scale = 1.0f32;
        let attention_window = if layer.is_full_attention {
            0
        } else {
            self.config.sliding_window as u32
        };
        let mut ext_mask_cache = crate::ggml_flash_attn_ext::PrefillExtMaskCache::default();

        if layer.has_kv {
            let k_cache = kv_pool
                .layer_k_cache(slot, layer_idx)
                .map_err(|err| err.to_string())?;
            let v_cache = kv_pool
                .layer_v_cache(slot, layer_idx)
                .map_err(|err| err.to_string())?;
            match self.kv_cache_type {
                KvCacheType::F16 => {
                    self.ctx.encode_kv_batch_append_f16(
                        encoder,
                        &self.prefill_scratch.k_buf,
                        k_cache,
                        num_kv_heads as u32,
                        head_dim as u32,
                        kv_pool.capacity(),
                        start_pos as u32,
                        seq_len as u32,
                    );
                    self.ctx.encode_kv_batch_append_f16(
                        encoder,
                        &self.prefill_scratch.v_buf,
                        v_cache,
                        num_kv_heads as u32,
                        head_dim as u32,
                        kv_pool.capacity(),
                        start_pos as u32,
                        seq_len as u32,
                    );
                }
                KvCacheType::Q8_0 => {
                    self.ctx.encode_kv_batch_append_q8_0(
                        encoder,
                        &self.prefill_scratch.k_buf,
                        k_cache,
                        num_kv_heads as u32,
                        head_dim as u32,
                        kv_pool.capacity(),
                        start_pos as u32,
                        seq_len as u32,
                    );
                    self.ctx.encode_kv_batch_append_q8_0(
                        encoder,
                        &self.prefill_scratch.v_buf,
                        v_cache,
                        num_kv_heads as u32,
                        head_dim as u32,
                        kv_pool.capacity(),
                        start_pos as u32,
                        seq_len as u32,
                    );
                }
                KvCacheType::Q4_0 => {
                    self.ctx.encode_kv_batch_append_q4_0(
                        encoder,
                        &self.prefill_scratch.k_buf,
                        k_cache,
                        num_kv_heads as u32,
                        head_dim as u32,
                        kv_pool.capacity(),
                        start_pos as u32,
                        seq_len as u32,
                    );
                    self.ctx.encode_kv_batch_append_q4_0(
                        encoder,
                        &self.prefill_scratch.v_buf,
                        v_cache,
                        num_kv_heads as u32,
                        head_dim as u32,
                        kv_pool.capacity(),
                        start_pos as u32,
                        seq_len as u32,
                    );
                }
            
                KvCacheType::TurboQuant { .. } => {
                    let (hot_k, hot_v) = self.tq_slot_hot_bufs(kv_pool, slot, layer_idx)?;
                    let (rw_k, rw_v) = self.tq_slot_rw_bufs(kv_pool, slot, layer_idx);
                    self.encode_tq_hot_prefill_kv_append(
                        encoder,
                        layer_idx,
                        &self.prefill_scratch.k_buf,
                        &self.prefill_scratch.v_buf,
                        hot_k,
                        hot_v,
                        k_cache,
                        v_cache,
                        rw_k,
                        rw_v,
                        num_kv_heads as u32,
                        head_dim as u32,
                        start_pos as u32,
                        seq_len as u32,
                    )?;
                }
            }
        }

        let k_cache = kv_pool
            .layer_k_cache(slot, layer.kv_source_layer)
            .map_err(|err| err.to_string())?;
        let v_cache = kv_pool
            .layer_v_cache(slot, layer.kv_source_layer)
            .map_err(|err| err.to_string())?;
        match self.kv_cache_type {
            KvCacheType::F16 => {
                self.ctx.encode_attention_causal_f16(
                    encoder,
                    &self.prefill_scratch.q_buf,
                    k_cache,
                    v_cache,
                    &self.prefill_scratch.attn_out_buf,
                    num_heads as u32,
                    num_kv_heads as u32,
                    num_kv_groups,
                    head_dim as u32,
                    (start_pos + seq_len) as u32,
                    kv_pool.capacity(),
                    scale,
                    seq_len as u32,
                    start_pos as u32,
                    attention_window,
                );
            }
            KvCacheType::Q8_0 => {
                let groups_per_row = (head_dim / 32) as u32;
                let row_bytes = groups_per_row * 34;
                self.ctx.encode_attention_causal_q8_0(
                    encoder,
                    &self.prefill_scratch.q_buf,
                    k_cache,
                    v_cache,
                    &self.prefill_scratch.attn_out_buf,
                    num_heads as u32,
                    num_kv_heads as u32,
                    num_kv_groups,
                    head_dim as u32,
                    (start_pos + seq_len) as u32,
                    kv_pool.capacity(),
                    scale,
                    seq_len as u32,
                    start_pos as u32,
                    attention_window,
                    groups_per_row,
                    row_bytes,
                );
            }
            KvCacheType::Q4_0 => {
                let groups_per_row = (head_dim / 32) as u32;
                let row_bytes = groups_per_row * 18;
                let use_ext_attn = crate::gpu::prefill_flash_attn_ext_enabled()
                    && crate::gpu::prefill_use_flash_attn_ext_tiled(
                        seq_len as u32,
                        head_dim as u32,
                    );
                let attn_out = if use_ext_attn {
                    &self.prefill_scratch.q_normed_buf
                } else {
                    &self.prefill_scratch.attn_out_buf
                };
                let q_f16_scratch = if use_ext_attn {
                    Some(&self.prefill_scratch.attn_out_buf)
                } else {
                    None
                };
                let (fa_scratch, fa_layout) = if use_ext_attn {
                    (
                        Some(&self.prefill_scratch.fa_ext_scratch),
                        Some(&self.prefill_scratch.fa_ext_layout),
                    )
                } else {
                    (None, None)
                };
                self.ctx.encode_prefill_attention_causal_q4_0(
                    encoder,
                    &self.prefill_scratch.q_buf,
                    k_cache,
                    v_cache,
                    attn_out,
                    q_f16_scratch,
                    num_heads as u32,
                    num_kv_heads as u32,
                    num_kv_groups,
                    head_dim as u32,
                    (start_pos + seq_len) as u32,
                    kv_pool.capacity(),
                    scale,
                    seq_len as u32,
                    start_pos as u32,
                    attention_window,
                    groups_per_row,
                    row_bytes,
                    fa_scratch,
                    fa_layout,
                    Some(&mut ext_mask_cache),
                );
            }
        
                KvCacheType::TurboQuant { .. } => {
                    if !self.tq_prefill_fits_hot(start_pos, seq_len) {
                        return Err(TURBOQUANT_UNSUPPORTED.to_string());
                    }
                    let src = layer.kv_source_layer;
                    let (hot_k, hot_v) = self.tq_slot_hot_bufs(kv_pool, slot, src)?;
                    let groups_per_row = (head_dim / 32) as u32;
                    let row_bytes = groups_per_row * 18;
                    let use_ext_attn = crate::gpu::prefill_flash_attn_ext_enabled()
                        && crate::gpu::prefill_use_flash_attn_ext_tiled(
                            seq_len as u32,
                            head_dim as u32,
                        );
                    let attn_out = if use_ext_attn {
                        &self.prefill_scratch.q_normed_buf
                    } else {
                        &self.prefill_scratch.attn_out_buf
                    };
                    let q_f16_scratch = if use_ext_attn {
                        Some(&self.prefill_scratch.attn_out_buf)
                    } else {
                        None
                    };
                    let (fa_scratch, fa_layout) = if use_ext_attn {
                        (
                            Some(&self.prefill_scratch.fa_ext_scratch),
                            Some(&self.prefill_scratch.fa_ext_layout),
                        )
                    } else {
                        (None, None)
                    };
                    self.ctx.encode_prefill_attention_causal_q4_0(
                        encoder,
                        &self.prefill_scratch.q_buf,
                        hot_k,
                        hot_v,
                        attn_out,
                        q_f16_scratch,
                        num_heads as u32,
                        num_kv_heads as u32,
                        num_kv_groups,
                        head_dim as u32,
                        (start_pos + seq_len) as u32,
                        self.tq_hot_w,
                        scale,
                        seq_len as u32,
                        start_pos as u32,
                        attention_window,
                        groups_per_row,
                        row_bytes,
                        fa_scratch,
                        fa_layout,
                        Some(&mut ext_mask_cache),
                    );
                }
            }

        self.ctx.encode_copy(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.residual_buf,
            total_hidden,
        );
        let use_ext_attn = crate::gpu::prefill_flash_attn_ext_enabled()
            && crate::gpu::prefill_use_flash_attn_ext_tiled(seq_len as u32, head_dim as u32)
            && (matches!(self.kv_cache_type, KvCacheType::Q4_0)
                || self.tq_prefill_fits_hot(start_pos, seq_len));
        if !use_ext_attn {
            self.ctx.encode_transpose_hsd(
                encoder,
                &self.prefill_scratch.attn_out_buf,
                &self.prefill_scratch.q_normed_buf,
                seq_len as u32,
                num_heads as u32,
                head_dim as u32,
            );
        }
        self.ctx.encode_prefill_projection_auto_batch_view(
            encoder,
            &layer.o_proj,
            &self.prefill_scratch.q_normed_buf,
            &self.prefill_scratch.o_out_buf,
            hidden_size as u32,
            q_out as u32,
            seq_len as u32,
        );
        self.ctx.encode_rmsnorm_batch_view(
            encoder,
            &self.prefill_scratch.o_out_buf,
            &layer.post_attention_layernorm_weight,
            &self.prefill_scratch.normed_buf,
            hidden_size as u32,
            eps,
            seq_len as u32,
        );
        self.ctx.encode_vec_add_batch(
            encoder,
            &self.prefill_scratch.residual_buf,
            &self.prefill_scratch.normed_buf,
            &self.prefill_scratch.hidden_buf,
            total_hidden,
        );

        self.ctx.encode_copy(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.residual_buf,
            total_hidden,
        );
        self.ctx.encode_rmsnorm_batch_view(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &layer.pre_feedforward_layernorm_weight,
            &self.prefill_scratch.normed_buf,
            hidden_size as u32,
            eps,
            seq_len as u32,
        );
        self.encode_prefill_mlp_gate_up(
            encoder,
            layer,
            seq_len as u32,
            intermediate_size as u32,
            hidden_size as u32,
            false,
        );
        if layer.weight_format == WeightFormat::F16 {
            self.ctx.encode_projection_f16_batch_view(
                encoder,
                &layer.down_proj,
                &self.prefill_scratch.gelu_buf,
                &self.prefill_scratch.down_buf,
                hidden_size as u32,
                intermediate_size as u32,
                seq_len as u32,
            );
        } else {
            self.ctx.encode_prefill_projection_q4_batch_view(
                encoder,
                &layer.down_proj,
                &self.prefill_scratch.gelu_buf,
                &self.prefill_scratch.down_buf,
                hidden_size as u32,
                intermediate_size as u32,
                seq_len as u32,
            );
        }
        self.ctx.encode_rmsnorm_batch_view(
            encoder,
            &self.prefill_scratch.down_buf,
            &layer.post_feedforward_layernorm_weight,
            &self.prefill_scratch.normed_buf,
            hidden_size as u32,
            eps,
            seq_len as u32,
        );
        self.ctx.encode_vec_add_batch(
            encoder,
            &self.prefill_scratch.residual_buf,
            &self.prefill_scratch.normed_buf,
            &self.prefill_scratch.hidden_buf,
            total_hidden,
        );

        self.ctx.encode_copy(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.residual_buf,
            total_hidden,
        );
        self.ctx.encode_prefill_projection_auto_batch_view(
            encoder,
            &layer.per_layer_input_gate_weight,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.gate_buf,
            ple_dim as u32,
            hidden_size as u32,
            seq_len as u32,
        );
        self.ctx.encode_ple_gelu_mul_batch(
            encoder,
            &self.prefill_scratch.gate_buf,
            &self.prefill_scratch.ple_context_proj_buf,
            &self.prefill_scratch.up_buf,
            layer_idx as u32,
            self.config.num_hidden_layers as u32,
            ple_dim as u32,
            seq_len as u32,
        );
        self.ctx.encode_prefill_projection_auto_batch_view(
            encoder,
            &layer.per_layer_projection_weight,
            &self.prefill_scratch.up_buf,
            &self.prefill_scratch.o_out_buf,
            hidden_size as u32,
            ple_dim as u32,
            seq_len as u32,
        );
        self.ctx.encode_rmsnorm_batch_view(
            encoder,
            &self.prefill_scratch.o_out_buf,
            &layer.post_per_layer_input_norm_weight,
            &self.prefill_scratch.normed_buf,
            hidden_size as u32,
            eps,
            seq_len as u32,
        );
        self.ctx.encode_vec_add_batch(
            encoder,
            &self.prefill_scratch.residual_buf,
            &self.prefill_scratch.normed_buf,
            &self.prefill_scratch.hidden_buf,
            total_hidden,
        );
        self.ctx.encode_vec_scale(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.residual_buf,
            total_hidden,
            layer.layer_scalar,
        );
        self.ctx.encode_copy(
            encoder,
            &self.prefill_scratch.residual_buf,
            &self.prefill_scratch.hidden_buf,
            total_hidden,
        );

        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    fn encode_parallel_prefill_layer_batched(
        &mut self,
        layer_idx: usize,
        total_seq_len: usize,
        segments: &[PrefillBatchSegment],
        kv_pool: &mut KvCachePool,
    ) -> Result<(), String> {
        let cmd = self.ctx.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        self.encode_parallel_prefill_attention_inputs(encoder, layer_idx, total_seq_len)?;

        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| format!("invalid layer index {}", layer_idx))?;
        let hidden_size = self.config.hidden_size;
        let intermediate_size = layer.intermediate_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = layer.kv_out_dim / layer.head_dim;
        let num_kv_groups = (num_heads / num_kv_heads) as u32;
        let ple_dim = self.config.hidden_size_per_layer_input;
        let eps = self.config.rms_norm_eps as f32;
        let head_dim = layer.head_dim;
        let q_out = layer.q_out_dim;
        let total_hidden = (total_seq_len * hidden_size) as u32;
        let total_intermediate = (total_seq_len * intermediate_size) as u32;
        let scale = 1.0f32;
        let attention_window = if layer.is_full_attention {
            0
        } else {
            self.config.sliding_window as u32
        };

        if layer.has_kv {
            for segment in segments {
                let k_cache = kv_pool
                    .layer_k_cache(segment.slot, layer_idx)
                    .map_err(|err| err.to_string())?;
                let v_cache = kv_pool
                    .layer_v_cache(segment.slot, layer_idx)
                    .map_err(|err| err.to_string())?;
                match self.kv_cache_type {
                    KvCacheType::F16 => {
                        self.ctx.encode_kv_batch_append_strided_f16(
                            encoder,
                            &self.prefill_scratch.k_buf,
                            k_cache,
                            num_kv_heads as u32,
                            head_dim as u32,
                            kv_pool.capacity(),
                            segment.start_pos as u32,
                            segment.token_count as u32,
                            total_seq_len as u32,
                            segment.row_start as u32,
                        );
                        self.ctx.encode_kv_batch_append_strided_f16(
                            encoder,
                            &self.prefill_scratch.v_buf,
                            v_cache,
                            num_kv_heads as u32,
                            head_dim as u32,
                            kv_pool.capacity(),
                            segment.start_pos as u32,
                            segment.token_count as u32,
                            total_seq_len as u32,
                            segment.row_start as u32,
                        );
                    }
                    KvCacheType::Q8_0 => {
                        self.ctx.encode_kv_batch_append_strided_q8_0(
                            encoder,
                            &self.prefill_scratch.k_buf,
                            k_cache,
                            num_kv_heads as u32,
                            head_dim as u32,
                            kv_pool.capacity(),
                            segment.start_pos as u32,
                            segment.token_count as u32,
                            total_seq_len as u32,
                            segment.row_start as u32,
                        );
                        self.ctx.encode_kv_batch_append_strided_q8_0(
                            encoder,
                            &self.prefill_scratch.v_buf,
                            v_cache,
                            num_kv_heads as u32,
                            head_dim as u32,
                            kv_pool.capacity(),
                            segment.start_pos as u32,
                            segment.token_count as u32,
                            total_seq_len as u32,
                            segment.row_start as u32,
                        );
                    }
                    KvCacheType::Q4_0 => {
                        self.ctx.encode_kv_batch_append_strided_q4_0(
                            encoder,
                            &self.prefill_scratch.k_buf,
                            k_cache,
                            num_kv_heads as u32,
                            head_dim as u32,
                            kv_pool.capacity(),
                            segment.start_pos as u32,
                            segment.token_count as u32,
                            total_seq_len as u32,
                            segment.row_start as u32,
                        );
                        self.ctx.encode_kv_batch_append_strided_q4_0(
                            encoder,
                            &self.prefill_scratch.v_buf,
                            v_cache,
                            num_kv_heads as u32,
                            head_dim as u32,
                            kv_pool.capacity(),
                            segment.start_pos as u32,
                            segment.token_count as u32,
                            total_seq_len as u32,
                            segment.row_start as u32,
                        );
                    }
                
                KvCacheType::TurboQuant { .. } => {
                    let spilled = kv_pool
                        .tq_hot_spilled(segment.slot)
                        .map_err(|e| e.to_string())?;
                    if spilled
                        || !self.tq_prefill_fits_hot(segment.start_pos, segment.token_count)
                    {
                        return Err(
                            "TurboQuant batched prefill requires all segments inside the Q4 hot window"
                                .to_string(),
                        );
                    }
                    let (hot_k, hot_v) =
                        self.tq_slot_hot_bufs(kv_pool, segment.slot, layer_idx)?;
                    self.ctx.encode_kv_batch_append_strided_q4_0(
                        encoder,
                        &self.prefill_scratch.k_buf,
                        hot_k,
                        num_kv_heads as u32,
                        head_dim as u32,
                        self.tq_hot_w,
                        segment.start_pos as u32,
                        segment.token_count as u32,
                        total_seq_len as u32,
                        segment.row_start as u32,
                    );
                    self.ctx.encode_kv_batch_append_strided_q4_0(
                        encoder,
                        &self.prefill_scratch.v_buf,
                        hot_v,
                        num_kv_heads as u32,
                        head_dim as u32,
                        self.tq_hot_w,
                        segment.start_pos as u32,
                        segment.token_count as u32,
                        total_seq_len as u32,
                        segment.row_start as u32,
                    );
                    let _ = (k_cache, v_cache);
                }
            }
            }
        }

        for segment in segments {
            let k_cache = kv_pool
                .layer_k_cache(segment.slot, layer.kv_source_layer)
                .map_err(|err| err.to_string())?;
            let v_cache = kv_pool
                .layer_v_cache(segment.slot, layer.kv_source_layer)
                .map_err(|err| err.to_string())?;
            match self.kv_cache_type {
                KvCacheType::F16 => {
                    self.ctx.encode_attention_causal_strided_f16(
                        encoder,
                        &self.prefill_scratch.q_buf,
                        k_cache,
                        v_cache,
                        &self.prefill_scratch.attn_out_buf,
                        num_heads as u32,
                        num_kv_heads as u32,
                        num_kv_groups,
                        head_dim as u32,
                        (segment.start_pos + segment.token_count) as u32,
                        kv_pool.capacity(),
                        scale,
                        segment.token_count as u32,
                        segment.start_pos as u32,
                        attention_window,
                        total_seq_len as u32,
                        segment.row_start as u32,
                    );
                }
                KvCacheType::Q8_0 => {
                    let groups_per_row = (head_dim / 32) as u32;
                    let row_bytes = groups_per_row * 34;
                    self.ctx.encode_attention_causal_strided_q8_0(
                        encoder,
                        &self.prefill_scratch.q_buf,
                        k_cache,
                        v_cache,
                        &self.prefill_scratch.attn_out_buf,
                        num_heads as u32,
                        num_kv_heads as u32,
                        num_kv_groups,
                        head_dim as u32,
                        (segment.start_pos + segment.token_count) as u32,
                        kv_pool.capacity(),
                        scale,
                        segment.token_count as u32,
                        segment.start_pos as u32,
                        attention_window,
                        total_seq_len as u32,
                        segment.row_start as u32,
                        groups_per_row,
                        row_bytes,
                    );
                }
                KvCacheType::Q4_0 => {
                    let groups_per_row = (head_dim / 32) as u32;
                    let row_bytes = groups_per_row * 18;
                    self.ctx.encode_attention_causal_strided_q4_0(
                        encoder,
                        &self.prefill_scratch.q_buf,
                        k_cache,
                        v_cache,
                        &self.prefill_scratch.attn_out_buf,
                        num_heads as u32,
                        num_kv_heads as u32,
                        num_kv_groups,
                        head_dim as u32,
                        (segment.start_pos + segment.token_count) as u32,
                        kv_pool.capacity(),
                        scale,
                        segment.token_count as u32,
                        segment.start_pos as u32,
                        attention_window,
                        total_seq_len as u32,
                        segment.row_start as u32,
                        groups_per_row,
                        row_bytes,
                    );
                }
                KvCacheType::TurboQuant { .. } => {
                    let spilled = kv_pool
                        .tq_hot_spilled(segment.slot)
                        .map_err(|e| e.to_string())?;
                    if spilled
                        || !self.tq_prefill_fits_hot(segment.start_pos, segment.token_count)
                    {
                        return Err(
                            "TurboQuant batched prefill requires all segments inside the Q4 hot window"
                                .to_string(),
                        );
                    }
                    let src = layer.kv_source_layer;
                    let (hot_k, hot_v) =
                        self.tq_slot_hot_bufs(kv_pool, segment.slot, src)?;
                    let groups_per_row = (head_dim / 32) as u32;
                    let row_bytes = groups_per_row * 18;
                    self.ctx.encode_attention_causal_strided_q4_0(
                        encoder,
                        &self.prefill_scratch.q_buf,
                        hot_k,
                        hot_v,
                        &self.prefill_scratch.attn_out_buf,
                        num_heads as u32,
                        num_kv_heads as u32,
                        num_kv_groups,
                        head_dim as u32,
                        (segment.start_pos + segment.token_count) as u32,
                        self.tq_hot_w,
                        scale,
                        segment.token_count as u32,
                        segment.start_pos as u32,
                        attention_window,
                        total_seq_len as u32,
                        segment.row_start as u32,
                        groups_per_row,
                        row_bytes,
                    );
                    let _ = (k_cache, v_cache);
                }
            }
        }

        // Post-attn / post-MLP / PLE: fuse RMSNorm+add into hidden (no residual copies).
        self.ctx.encode_transpose_hsd(
            encoder,
            &self.prefill_scratch.attn_out_buf,
            &self.prefill_scratch.q_normed_buf,
            total_seq_len as u32,
            num_heads as u32,
            head_dim as u32,
        );
        self.ctx.encode_prefill_projection_auto_batch_view(
            encoder,
            &layer.o_proj,
            &self.prefill_scratch.q_normed_buf,
            &self.prefill_scratch.o_out_buf,
            hidden_size as u32,
            q_out as u32,
            total_seq_len as u32,
        );
        self.ctx.encode_rmsnorm_acc_batch_view(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.o_out_buf,
            &layer.post_attention_layernorm_weight,
            hidden_size as u32,
            eps,
            total_seq_len as u32,
        );

        self.ctx.encode_rmsnorm_batch_view(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &layer.pre_feedforward_layernorm_weight,
            &self.prefill_scratch.normed_buf,
            hidden_size as u32,
            eps,
            total_seq_len as u32,
        );
        self.encode_prefill_mlp_gate_up(
            encoder,
            layer,
            total_seq_len as u32,
            intermediate_size as u32,
            hidden_size as u32,
            false,
        );
        if layer.weight_format == WeightFormat::F16 {
            self.ctx.encode_projection_f16_batch_view(
                encoder,
                &layer.down_proj,
                &self.prefill_scratch.gelu_buf,
                &self.prefill_scratch.down_buf,
                hidden_size as u32,
                intermediate_size as u32,
                total_seq_len as u32,
            );
        } else {
            self.ctx.encode_prefill_projection_q4_batch_view(
                encoder,
                &layer.down_proj,
                &self.prefill_scratch.gelu_buf,
                &self.prefill_scratch.down_buf,
                hidden_size as u32,
                intermediate_size as u32,
                total_seq_len as u32,
            );
        }
        self.ctx.encode_rmsnorm_acc_batch_view(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.down_buf,
            &layer.post_feedforward_layernorm_weight,
            hidden_size as u32,
            eps,
            total_seq_len as u32,
        );

        self.ctx.encode_prefill_projection_auto_batch_view(
            encoder,
            &layer.per_layer_input_gate_weight,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.gate_buf,
            ple_dim as u32,
            hidden_size as u32,
            total_seq_len as u32,
        );
        self.ctx.encode_ple_gelu_mul_batch(
            encoder,
            &self.prefill_scratch.gate_buf,
            &self.prefill_scratch.ple_context_proj_buf,
            &self.prefill_scratch.up_buf,
            layer_idx as u32,
            self.config.num_hidden_layers as u32,
            ple_dim as u32,
            total_seq_len as u32,
        );
        self.ctx.encode_prefill_projection_auto_batch_view(
            encoder,
            &layer.per_layer_projection_weight,
            &self.prefill_scratch.up_buf,
            &self.prefill_scratch.o_out_buf,
            hidden_size as u32,
            ple_dim as u32,
            total_seq_len as u32,
        );
        self.ctx.encode_rmsnorm_acc_batch_view(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.o_out_buf,
            &layer.post_per_layer_input_norm_weight,
            hidden_size as u32,
            eps,
            total_seq_len as u32,
        );
        self.ctx.encode_vec_scale(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.hidden_buf,
            total_hidden,
            layer.layer_scalar,
        );

        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    fn forward_prefill_chunk_parallel_with_kv_slot(
        &mut self,
        token_ids: &[usize],
        kv_pool: &mut KvCachePool,
        slot: KvSlot,
        start_pos: usize,
        compute_logits: bool,
        capture_mtp_rows: bool,
    ) -> Result<Vec<f32>, String> {
        let seq_len = token_ids.len();
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let eps = self.config.rms_norm_eps as f32;

        let timing = crate::gpu::prefill_timing_enabled();
        let t_chunk = std::time::Instant::now();

        // ═══ Encode PLE + all layers (+ optional tail) in one command buffer ═══
        let cmd = self.ctx.queue.new_command_buffer();
        if crate::gpu::prefill_gpu_rope_enabled() {
            let rope_encoder = cmd.new_compute_command_encoder();
            self.encode_prefill_gpu_rotary_tables(rope_encoder, start_pos, seq_len)?;
            rope_encoder.end_encoding();
        }
        let t_after_rope = std::time::Instant::now();
        let ablate_early = crate::gpu::ProfileAblate::from_env();
        if !ablate_early.skip_ple() {
            let ple_encoder = cmd.new_compute_command_encoder();
            self.encode_parallel_prefill_ple_context_on_encoder(ple_encoder, seq_len)?;
            ple_encoder.end_encoding();
        }

        let mut ext_mask_cache = crate::ggml_flash_attn_ext::PrefillExtMaskCache::default();
        let ablate = crate::gpu::ProfileAblate::from_env();
        ablate.log_once();

        // One compute encoder for all layers: fewer encoder create/end costs.
        // Dependent dispatches already share an encoder within each layer.
        let stop_layer = Self::mtp_bisect_stop_layer().unwrap_or(self.layers.len());
        let encoder = cmd.new_compute_command_encoder();
        for layer_idx in 0..self.layers.len().min(stop_layer) {
            let layer = self
                .layers
                .get(layer_idx)
                .ok_or_else(|| format!("invalid layer index {}", layer_idx))?;
            let intermediate_size = layer.intermediate_size;
            let num_heads = self.config.num_attention_heads;
            let num_kv_heads = layer.kv_out_dim / layer.head_dim;
            let num_kv_groups = (num_heads / num_kv_heads) as u32;
            let ple_dim = self.config.hidden_size_per_layer_input;
            let head_dim = layer.head_dim;
            let q_out = layer.q_out_dim;
            let total_hidden = (seq_len * hidden_size) as u32;
            let _total_intermediate = (seq_len * intermediate_size) as u32;
            let scale = 1.0f32;
            let attention_window = if layer.is_full_attention {
                0
            } else {
                self.config.sliding_window as u32
            };
            // MTP verify batches only: keep batched matmuls but use decode offset
            // attention (same kernels as forward_verify_sequential / decode batch).
            // Opt-in per-row decode attention via `MTP_VERIFY_DECODE_FA=1` when
            // batched causal prefill attention disagrees with decode.
            let use_decode_fa = capture_mtp_rows
                && std::env::var("MTP_VERIFY_DECODE_FA")
                    .map(|v| v == "1")
                    .unwrap_or(false)
                && matches!(self.kv_cache_type, KvCacheType::Q4_0)
                && start_pos > 0
                && seq_len <= MAX_MTP_VERIFY_SEQ
                && num_kv_heads == 1;

            if !ablate.skip_attn() {
            if !ablate.skip_attn_qkv() {
                self.encode_parallel_prefill_attention_inputs(encoder, layer_idx, seq_len)?;
            } // !skip_attn_qkv

            if !ablate.skip_attn_flash() {
            if layer.has_kv {
                let k_cache = kv_pool
                    .layer_k_cache(slot, layer_idx)
                    .map_err(|err| err.to_string())?;
                let v_cache = kv_pool
                    .layer_v_cache(slot, layer_idx)
                    .map_err(|err| err.to_string())?;
                match self.kv_cache_type {
                    KvCacheType::F16 => {
                        self.ctx.encode_kv_batch_append_f16(
                            encoder,
                            &self.prefill_scratch.k_buf,
                            k_cache,
                            num_kv_heads as u32,
                            head_dim as u32,
                            kv_pool.capacity(),
                            start_pos as u32,
                            seq_len as u32,
                        );
                        self.ctx.encode_kv_batch_append_f16(
                            encoder,
                            &self.prefill_scratch.v_buf,
                            v_cache,
                            num_kv_heads as u32,
                            head_dim as u32,
                            kv_pool.capacity(),
                            start_pos as u32,
                            seq_len as u32,
                        );
                    }
                    KvCacheType::Q8_0 => {
                        self.ctx.encode_kv_batch_append_q8_0(
                            encoder,
                            &self.prefill_scratch.k_buf,
                            k_cache,
                            num_kv_heads as u32,
                            head_dim as u32,
                            kv_pool.capacity(),
                            start_pos as u32,
                            seq_len as u32,
                        );
                        self.ctx.encode_kv_batch_append_q8_0(
                            encoder,
                            &self.prefill_scratch.v_buf,
                            v_cache,
                            num_kv_heads as u32,
                            head_dim as u32,
                            kv_pool.capacity(),
                            start_pos as u32,
                            seq_len as u32,
                        );
                    }
                    KvCacheType::Q4_0 => {
                        self.ctx.encode_kv_batch_append_q4_0(
                            encoder,
                            &self.prefill_scratch.k_buf,
                            k_cache,
                            num_kv_heads as u32,
                            head_dim as u32,
                            kv_pool.capacity(),
                            start_pos as u32,
                            seq_len as u32,
                        );
                        self.ctx.encode_kv_batch_append_q4_0(
                            encoder,
                            &self.prefill_scratch.v_buf,
                            v_cache,
                            num_kv_heads as u32,
                            head_dim as u32,
                            kv_pool.capacity(),
                            start_pos as u32,
                            seq_len as u32,
                        );
                    }
                
                KvCacheType::TurboQuant { .. } => {
                    let (hot_k, hot_v) = self.tq_slot_hot_bufs(kv_pool, slot, layer_idx)?;
                    let (rw_k, rw_v) = self.tq_slot_rw_bufs(kv_pool, slot, layer_idx);
                    self.encode_tq_hot_prefill_kv_append(
                        encoder,
                        layer_idx,
                        &self.prefill_scratch.k_buf,
                        &self.prefill_scratch.v_buf,
                        hot_k,
                        hot_v,
                        k_cache,
                        v_cache,
                        rw_k,
                        rw_v,
                        num_kv_heads as u32,
                        head_dim as u32,
                        start_pos as u32,
                        seq_len as u32,
                    )?;
                }
            }
            }

            let k_cache = kv_pool
                .layer_k_cache(slot, layer.kv_source_layer)
                .map_err(|err| err.to_string())?;
            let v_cache = kv_pool
                .layer_v_cache(slot, layer.kv_source_layer)
                .map_err(|err| err.to_string())?;
            match self.kv_cache_type {
                KvCacheType::F16 => {
                    self.ctx.encode_attention_causal_f16(
                        encoder,
                        &self.prefill_scratch.q_buf,
                        k_cache,
                        v_cache,
                        &self.prefill_scratch.attn_out_buf,
                        num_heads as u32,
                        num_kv_heads as u32,
                        num_kv_groups,
                        head_dim as u32,
                        (start_pos + seq_len) as u32,
                        kv_pool.capacity(),
                        scale,
                        seq_len as u32,
                        start_pos as u32,
                        attention_window,
                    );
                }
                KvCacheType::Q8_0 => {
                    let groups_per_row = (head_dim / 32) as u32;
                    let row_bytes = groups_per_row * 34;
                    self.ctx.encode_attention_causal_q8_0(
                        encoder,
                        &self.prefill_scratch.q_buf,
                        k_cache,
                        v_cache,
                        &self.prefill_scratch.attn_out_buf,
                        num_heads as u32,
                        num_kv_heads as u32,
                        num_kv_groups,
                        head_dim as u32,
                        (start_pos + seq_len) as u32,
                        kv_pool.capacity(),
                        scale,
                        seq_len as u32,
                        start_pos as u32,
                        attention_window,
                        groups_per_row,
                        row_bytes,
                    );
                }
                KvCacheType::Q4_0 => {
                    let groups_per_row = (head_dim / 32) as u32;
                    let row_bytes = groups_per_row * 18;
                    if use_decode_fa {
                        // Prefill QKV postproj leaves Q in HSD [head][seq][dim] with
                        // QK-norm + RoPE applied; decode attention kernels expect a
                        // token-major [heads*dim] row. Transpose Q back to SHD, then
                        // run the decode offset kernel per row. KV was batch-appended
                        // above (K/V are HSD too, but num_kv_heads==1 makes the
                        // layouts identical); each row reads the cache only up to its
                        // own position, so appending all rows up front stays causal.
                        let sliding_window = self.config.sliding_window as u32;
                        self.ctx.encode_transpose_hsd(
                            encoder,
                            &self.prefill_scratch.q_buf,
                            &self.prefill_scratch.q_normed_buf,
                            seq_len as u32,
                            num_heads as u32,
                            head_dim as u32,
                        );
                        for qi in 0..seq_len {
                            let append_pos = start_pos as u32 + qi as u32;
                            let attn_kv_seq = append_pos + 1;
                            let effective_kv_seq = if layer.is_full_attention {
                                attn_kv_seq
                            } else {
                                attn_kv_seq.min(sliding_window)
                            };
                            let kv_start = if !layer.is_full_attention
                                && attn_kv_seq > sliding_window
                            {
                                attn_kv_seq - sliding_window
                            } else {
                                0
                            };
                            let q_offset = (qi * q_out * 4) as u64;
                            self.ctx.encode_attention_with_offset_q4_0_at(
                                encoder,
                                &self.prefill_scratch.q_normed_buf,
                                q_offset,
                                k_cache,
                                v_cache,
                                &self.prefill_scratch.attn_out_buf,
                                q_offset,
                                num_heads as u32,
                                num_kv_heads as u32,
                                num_kv_groups,
                                head_dim as u32,
                                effective_kv_seq,
                                kv_pool.capacity(),
                                scale,
                                kv_start,
                                groups_per_row,
                                row_bytes,
                            );
                        }
                    } else {
                    let use_ext_attn = crate::gpu::prefill_flash_attn_ext_enabled()
                        && crate::gpu::prefill_use_flash_attn_ext_tiled(
                            seq_len as u32,
                            head_dim as u32,
                        );
                    let attn_out = if use_ext_attn {
                        &self.prefill_scratch.q_normed_buf
                    } else {
                        &self.prefill_scratch.attn_out_buf
                    };
                    let q_f16_scratch = if use_ext_attn {
                        Some(&self.prefill_scratch.attn_out_buf)
                    } else {
                        None
                    };
                    let (fa_scratch, fa_layout) = if use_ext_attn {
                        (
                            Some(&self.prefill_scratch.fa_ext_scratch),
                            Some(&self.prefill_scratch.fa_ext_layout),
                        )
                    } else {
                        (None, None)
                    };
                    self.ctx.encode_prefill_attention_causal_q4_0(
                        encoder,
                        &self.prefill_scratch.q_buf,
                        k_cache,
                        v_cache,
                        attn_out,
                        q_f16_scratch,
                        num_heads as u32,
                        num_kv_heads as u32,
                        num_kv_groups,
                        head_dim as u32,
                        (start_pos + seq_len) as u32,
                        kv_pool.capacity(),
                        scale,
                        seq_len as u32,
                        start_pos as u32,
                        attention_window,
                        groups_per_row,
                        row_bytes,
                        fa_scratch,
                        fa_layout,
                        Some(&mut ext_mask_cache),
                    );
                    }
                }
            
                KvCacheType::TurboQuant { .. } => {
                    if !self.tq_prefill_fits_hot(start_pos, seq_len) {
                        return Err(TURBOQUANT_UNSUPPORTED.to_string());
                    }
                    let src = layer.kv_source_layer;
                    let (hot_k, hot_v) = self.tq_slot_hot_bufs(kv_pool, slot, src)?;
                    let groups_per_row = (head_dim / 32) as u32;
                    let row_bytes = groups_per_row * 18;
                    let use_ext_attn = crate::gpu::prefill_flash_attn_ext_enabled()
                        && crate::gpu::prefill_use_flash_attn_ext_tiled(
                            seq_len as u32,
                            head_dim as u32,
                        );
                    let attn_out = if use_ext_attn {
                        &self.prefill_scratch.q_normed_buf
                    } else {
                        &self.prefill_scratch.attn_out_buf
                    };
                    let q_f16_scratch = if use_ext_attn {
                        Some(&self.prefill_scratch.attn_out_buf)
                    } else {
                        None
                    };
                    let (fa_scratch, fa_layout) = if use_ext_attn {
                        (
                            Some(&self.prefill_scratch.fa_ext_scratch),
                            Some(&self.prefill_scratch.fa_ext_layout),
                        )
                    } else {
                        (None, None)
                    };
                    self.ctx.encode_prefill_attention_causal_q4_0(
                        encoder,
                        &self.prefill_scratch.q_buf,
                        hot_k,
                        hot_v,
                        attn_out,
                        q_f16_scratch,
                        num_heads as u32,
                        num_kv_heads as u32,
                        num_kv_groups,
                        head_dim as u32,
                        (start_pos + seq_len) as u32,
                        self.tq_hot_w,
                        scale,
                        seq_len as u32,
                        start_pos as u32,
                        attention_window,
                        groups_per_row,
                        row_bytes,
                        fa_scratch,
                        fa_layout,
                        Some(&mut ext_mask_cache),
                    );
                }
            }
            } // !skip_attn_flash

            // Post-attn: hidden += RMSNorm(o_proj(attn)). Keep hidden as residual
            // for pre-FFN (no copy into residual_buf).
            if !ablate.skip_attn_o() {
            let use_ext_attn = crate::gpu::prefill_flash_attn_ext_enabled()
                && crate::gpu::prefill_use_flash_attn_ext_tiled(seq_len as u32, head_dim as u32)
                && (matches!(self.kv_cache_type, KvCacheType::Q4_0)
                    || self.tq_prefill_fits_hot(start_pos, seq_len));
            if !use_ext_attn && !use_decode_fa {
                self.ctx.encode_transpose_hsd(
                    encoder,
                    &self.prefill_scratch.attn_out_buf,
                    &self.prefill_scratch.q_normed_buf,
                    seq_len as u32,
                    num_heads as u32,
                    head_dim as u32,
                );
            }
            self.ctx.encode_prefill_projection_auto_batch_view(
                encoder,
                &layer.o_proj,
                if use_decode_fa {
                    &self.prefill_scratch.attn_out_buf
                } else {
                    &self.prefill_scratch.q_normed_buf
                },
                &self.prefill_scratch.o_out_buf,
                hidden_size as u32,
                q_out as u32,
                seq_len as u32,
            );
            self.ctx.encode_rmsnorm_acc_batch_view(
                encoder,
                &self.prefill_scratch.hidden_buf,
                &self.prefill_scratch.o_out_buf,
                &layer.post_attention_layernorm_weight,
                hidden_size as u32,
                eps,
                seq_len as u32,
            );
            } // !skip_attn_o
            } // !skip_attn

            if !ablate.skip_mlp() {
            if !ablate.skip_mlp_gate() {
            self.ctx.encode_rmsnorm_batch_view(
                encoder,
                &self.prefill_scratch.hidden_buf,
                &layer.pre_feedforward_layernorm_weight,
                &self.prefill_scratch.normed_buf,
                hidden_size as u32,
                eps,
                seq_len as u32,
            );
            self.encode_prefill_mlp_gate_up(
                encoder,
                layer,
                seq_len as u32,
                intermediate_size as u32,
                hidden_size as u32,
                ablate.skip_mlp_gelu(),
            );
            } // !skip_mlp_gate
            if !ablate.skip_mlp_down() {
        if layer.weight_format == WeightFormat::F16 {
            self.ctx.encode_projection_f16_batch_view(
                encoder,
                &layer.down_proj,
                    &self.prefill_scratch.gelu_buf,
                    &self.prefill_scratch.down_buf,
                    hidden_size as u32,
                    intermediate_size as u32,
                    seq_len as u32,
                );
            } else if crate::gpu::prefill_mlp_f16_enabled()
                && !ablate.skip_cast()
                && crate::gpu::prefill_mul_mm_enabled()
                && crate::ggml_gemv::should_use_mul_mm(
                    intermediate_size as u32,
                    seq_len as u32,
                )
            {
                // Stacked f16 path already wrote halfs into gelu_buf; skip cast.
                let gelu_f16 = if Self::use_prefill_gate_up_stacked(layer) {
                    &self.prefill_scratch.gelu_buf
                } else {
                    self.ctx.encode_cast_f32_to_f16(
                        encoder,
                        &self.prefill_scratch.gelu_buf,
                        &self.prefill_scratch.up_buf,
                        (seq_len * intermediate_size) as u32,
                    );
                    &self.prefill_scratch.up_buf
                };
                self.ctx.encode_mul_mm_kquant_f16_at_view(
                    encoder,
                    &layer.down_proj,
                    gelu_f16,
                    &self.prefill_scratch.down_buf,
                    hidden_size as u32,
                    intermediate_size as u32,
                    seq_len as u32,
                );
            } else {
                self.ctx.encode_prefill_projection_q4_batch_view(
                    encoder,
                    &layer.down_proj,
                    &self.prefill_scratch.gelu_buf,
                    &self.prefill_scratch.down_buf,
                    hidden_size as u32,
                    intermediate_size as u32,
                    seq_len as u32,
                );
            }
            self.ctx.encode_rmsnorm_acc_batch_view(
                encoder,
                &self.prefill_scratch.hidden_buf,
                &self.prefill_scratch.down_buf,
                &layer.post_feedforward_layernorm_weight,
                hidden_size as u32,
                eps,
                seq_len as u32,
            );
            } // !skip_mlp_down
            } // !skip_mlp

            if !ablate.skip_ple() {
            self.ctx.encode_prefill_projection_auto_batch_view(
                encoder,
                &layer.per_layer_input_gate_weight,
                &self.prefill_scratch.hidden_buf,
                &self.prefill_scratch.gate_buf,
                ple_dim as u32,
                hidden_size as u32,
                seq_len as u32,
            );
            self.ctx.encode_ple_gelu_mul_batch(
                encoder,
                &self.prefill_scratch.gate_buf,
                &self.prefill_scratch.ple_context_proj_buf,
                &self.prefill_scratch.up_buf,
                layer_idx as u32,
                self.config.num_hidden_layers as u32,
                ple_dim as u32,
                seq_len as u32,
            );
            self.ctx.encode_prefill_projection_auto_batch_view(
                encoder,
                &layer.per_layer_projection_weight,
                &self.prefill_scratch.up_buf,
                &self.prefill_scratch.o_out_buf,
                hidden_size as u32,
                ple_dim as u32,
                seq_len as u32,
            );
            self.ctx.encode_rmsnorm_acc_batch_view(
                encoder,
                &self.prefill_scratch.hidden_buf,
                &self.prefill_scratch.o_out_buf,
                &layer.post_per_layer_input_norm_weight,
                hidden_size as u32,
                eps,
                seq_len as u32,
            );
            self.ctx.encode_vec_scale(
                encoder,
                &self.prefill_scratch.hidden_buf,
                &self.prefill_scratch.hidden_buf,
                total_hidden,
                layer.layer_scalar,
            );
            } // !skip_ple
        }
        encoder.end_encoding();

        if compute_logits || capture_mtp_rows {
            // Final norm + lm_head. Normal prefill only needs the last row;
            // MTP verify needs every row's logits + post-norm hidden.
            let encoder = cmd.new_compute_command_encoder();
            self.ctx.encode_rmsnorm_batch_view(
                encoder,
                &self.prefill_scratch.hidden_buf,
                &self.final_norm_weight,
                &self.prefill_scratch.normed_buf,
                hidden_size as u32,
                eps,
                seq_len as u32,
            );
            if capture_mtp_rows {
                // One batched lm_head projection for all rows: the ~440 MB
                // vocab matrix is read once instead of seq_len times.
                self.ctx.encode_prefill_projection_auto_batch_view(
                    encoder,
                    &self.lm_head_buf,
                    &self.prefill_scratch.normed_buf,
                    &self.mtp_verify_logits_buf,
                    vocab_size as u32,
                    hidden_size as u32,
                    seq_len as u32,
                );
                for row in 0..seq_len {
                    let offsets = self.prefill_row_offsets(row);
                    self.ctx.encode_copy_at(
                        encoder,
                        &self.prefill_scratch.normed_buf,
                        offsets.hidden,
                        &self.mtp_verify_hidden_buf,
                        (row * hidden_size * 4) as u64,
                        hidden_size as u32,
                    );
                }
                self.ctx.encode_softcap_argmax_rows_f32(
                    encoder,
                    &self.mtp_verify_logits_buf,
                    &self.mtp_verify_argmax_buf,
                    seq_len as u32,
                    vocab_size as u32,
                    self.config.final_logit_softcapping,
                );
            } else {
                let last_offsets = self.prefill_row_offsets(seq_len - 1);
                self.ctx.encode_matvec_auto_at_view(
                    encoder,
                    &self.lm_head_buf,
                    &self.prefill_scratch.normed_buf,
                    last_offsets.hidden,
                    &self.prefill_scratch.logits_buf,
                    0,
                    vocab_size as u32,
                    hidden_size as u32,
                );
            }
            encoder.end_encoding();
        }

        // ═══ SINGLE commit + wait for PLE + all layers (+ optional tail) ═══
        cmd.commit();
        cmd.wait_until_completed();
        let t_after_gpu = std::time::Instant::now();

        if timing {
            let rope_ms = (t_after_rope - t_chunk).as_secs_f64() * 1e3;
            let gpu_ms = (t_after_gpu - t_after_rope).as_secs_f64() * 1e3;
            let total_ms = (t_after_gpu - t_chunk).as_secs_f64() * 1e3;
            eprintln!(
                "[prefill timing] chunk seq={} start={} rope_gpu={:.1}ms layers_gpu={:.1}ms total={:.1}ms ({:.1} tok/s)",
                seq_len,
                start_pos,
                rope_ms,
                gpu_ms - rope_ms,
                total_ms,
                if total_ms > 0.0 {
                    seq_len as f64 / (total_ms / 1e3)
                } else {
                    0.0
                }
            );
        }

        let logits = if compute_logits {
            let mut logits =
                MetalContext::read_buffer(&self.prefill_scratch.logits_buf, vocab_size);
            let cap = self.config.final_logit_softcapping;
            for logit in &mut logits {
                let x = (*logit / cap).clamp(-10.0, 10.0);
                *logit = cap * x.tanh();
            }
            logits
        } else {
            Vec::new()
        };

        kv_pool
            .with_slot_mut(slot, |slot_state| {
                slot_state.seq_len += seq_len as u32;
                slot_state.total_tokens += seq_len;
            })
            .map_err(|err| err.to_string())?;

        Ok(logits)
    }

    fn forward_decode_batch_encoded_with_kv_slots(
        &mut self,
        inputs: &[(KvSlot, usize)],
        slot_views: &[KvSlotView],
        kv_pool: &mut KvCachePool,
        capture_mtp: bool,
    ) -> Result<Vec<Vec<f32>>, String> {
        let batch_size = inputs.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.max_intermediate_size();
        let vocab_size = self.config.vocab_size;
        let num_layers = self.config.num_hidden_layers;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let num_kv_groups = (num_heads / num_kv_heads) as u32;
        let ple_dim = self.config.hidden_size_per_layer_input;
        let ple_total_dim = num_layers * ple_dim;
        let eps = self.config.rms_norm_eps as f32;
        let context_proj_scale = 1.0f32 / (hidden_size as f32).sqrt();
        let ple_input_scale = 1.0f32 / 2.0f32.sqrt();

        let cmd = self.ctx.queue.new_command_buffer();

        {
            let encoder = cmd.new_compute_command_encoder();
            self.ctx.encode_prefill_projection_auto_batch_view(
                encoder,
                &self.per_layer_model_projection_weight,
                &self.decode_batch_scratch.hidden_buf,
                &self.decode_batch_scratch.ple_context_proj_buf,
                ple_total_dim as u32,
                hidden_size as u32,
                batch_size as u32,
            );
            self.ctx.encode_vec_scale(
                encoder,
                &self.decode_batch_scratch.ple_context_proj_buf,
                &self.decode_batch_scratch.ple_combined_buf,
                (batch_size * ple_total_dim) as u32,
                context_proj_scale,
            );
            self.ctx.encode_rmsnorm_batch_view(
                encoder,
                &self.decode_batch_scratch.ple_combined_buf,
                &self.per_layer_projection_norm_weight,
                &self.decode_batch_scratch.ple_context_proj_buf,
                ple_dim as u32,
                eps,
                (batch_size * num_layers) as u32,
            );
            self.ctx.encode_vec_add_batch(
                encoder,
                &self.decode_batch_scratch.ple_context_proj_buf,
                &self.decode_batch_scratch.ple_token_id_buf,
                &self.decode_batch_scratch.ple_combined_buf,
                (batch_size * ple_total_dim) as u32,
            );
            self.ctx.encode_vec_scale(
                encoder,
                &self.decode_batch_scratch.ple_combined_buf,
                &self.decode_batch_scratch.ple_context_proj_buf,
                (batch_size * ple_total_dim) as u32,
                ple_input_scale,
            );
            encoder.end_encoding();
        }

        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            let intermediate_size = layer.intermediate_size;
            let head_dim = layer.head_dim;
            let num_kv_heads = layer.kv_out_dim / layer.head_dim;
            let num_kv_groups = (num_heads / num_kv_heads) as u32;
            let q_out = layer.q_out_dim;
            let kv_out = layer.kv_out_dim;
            let scale = 1.0f32;

            let encoder = cmd.new_compute_command_encoder();

            for (batch_idx, slot_view) in slot_views.iter().enumerate() {
                let offsets = self.decode_batch_row_offsets(batch_idx);
                let append_pos = slot_view.seq_len;
                let attn_kv_seq = append_pos + 1;
                let effective_kv_seq = if layer.is_full_attention {
                    attn_kv_seq
                } else {
                    attn_kv_seq.min(self.config.sliding_window as u32)
                };
                let kv_start = if !layer.is_full_attention
                    && attn_kv_seq > self.config.sliding_window as u32
                {
                    attn_kv_seq - self.config.sliding_window as u32
                } else {
                    0
                };
                let rotary_offset = Self::f32_byte_offset(batch_idx * head_dim);
                let ple_layer_offset = self.decode_batch_layer_ple_offset(batch_idx, layer_idx);

                self.ctx.encode_copy_at(
                    encoder,
                    &self.decode_batch_scratch.hidden_buf,
                    offsets.hidden,
                    &self.decode_batch_scratch.residual_buf,
                    offsets.hidden,
                    hidden_size as u32,
                );
                self.ctx.encode_rmsnorm_at_view(
                    encoder,
                    &self.decode_batch_scratch.hidden_buf,
                    offsets.hidden,
                    &layer.input_layernorm_weight,
                    &self.decode_batch_scratch.normed_buf,
                    offsets.hidden,
                    hidden_size as u32,
                    eps,
                );

                if layer.weight_format == WeightFormat::F16 {
                    self.ctx.encode_matvec_f16_at_view(
                        encoder,
                        &layer.q_proj,
                        &self.decode_batch_scratch.normed_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.q_buf,
                        offsets.q,
                        q_out as u32,
                        hidden_size as u32,
                    );
                } else {
                    self.ctx.encode_matvec_auto_at_view(
                        encoder,
                        &layer.q_proj,
                        &self.decode_batch_scratch.normed_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.q_buf,
                        offsets.q,
                        q_out as u32,
                        hidden_size as u32,
                    );
                }
                self.ctx.encode_rmsnorm_per_head_at_view(
                    encoder,
                    &self.decode_batch_scratch.q_buf,
                    offsets.q,
                    &layer.q_norm_weight,
                    &self.decode_batch_scratch.q_normed_buf,
                    offsets.q,
                    num_heads as u32,
                    head_dim as u32,
                    eps,
                );
                self.ctx.encode_rotary_at(
                    encoder,
                    &self.decode_batch_scratch.q_normed_buf,
                    offsets.q,
                    &self.decode_batch_scratch.k_normed_buf,
                    offsets.kv,
                    &self.per_layer_decode_batch_cos_bufs[layer_idx],
                    rotary_offset,
                    &self.per_layer_decode_batch_sin_bufs[layer_idx],
                    rotary_offset,
                    num_heads as u32,
                    0,
                    head_dim as u32,
                );

                if layer.has_kv {
                    if layer.weight_format == WeightFormat::F16 {
                        self.ctx.encode_matvec_f16_at_view(
                            encoder,
                            &layer.k_proj,
                            &self.decode_batch_scratch.normed_buf,
                            offsets.hidden,
                            &self.decode_batch_scratch.k_buf,
                            offsets.kv,
                            kv_out as u32,
                            hidden_size as u32,
                        );
                        self.ctx.encode_matvec_f16_at_view(
                            encoder,
                            &layer.v_proj,
                            &self.decode_batch_scratch.normed_buf,
                            offsets.hidden,
                            &self.decode_batch_scratch.v_buf,
                            offsets.kv,
                            kv_out as u32,
                            hidden_size as u32,
                        );
                    } else {
                        self.ctx.encode_matvec_auto_at_view(
                            encoder,
                            &layer.k_proj,
                            &self.decode_batch_scratch.normed_buf,
                            offsets.hidden,
                            &self.decode_batch_scratch.k_buf,
                            offsets.kv,
                            kv_out as u32,
                            hidden_size as u32,
                        );
                        self.ctx.encode_matvec_auto_at_view(
                            encoder,
                            &layer.v_proj,
                            &self.decode_batch_scratch.normed_buf,
                            offsets.hidden,
                            &self.decode_batch_scratch.v_buf,
                            offsets.kv,
                            kv_out as u32,
                            hidden_size as u32,
                        );
                    }

                    self.ctx.encode_rmsnorm_per_head_at_view(
                        encoder,
                        &self.decode_batch_scratch.k_buf,
                        offsets.kv,
                        &layer.k_norm_weight,
                        &self.decode_batch_scratch.k_normed_buf,
                        offsets.kv,
                        num_kv_heads as u32,
                        head_dim as u32,
                        eps,
                    );
                    self.ctx.encode_rotary_at(
                        encoder,
                        &self.decode_batch_scratch.q_buf,
                        offsets.q,
                        &self.decode_batch_scratch.k_normed_buf,
                        offsets.kv,
                        &self.per_layer_decode_batch_cos_bufs[layer_idx],
                        rotary_offset,
                        &self.per_layer_decode_batch_sin_bufs[layer_idx],
                        rotary_offset,
                        0,
                        num_kv_heads as u32,
                        head_dim as u32,
                    );
                    self.ctx.encode_rmsnorm_per_head_noweight_at(
                        encoder,
                        &self.decode_batch_scratch.v_buf,
                        offsets.kv,
                        &self.decode_batch_scratch.gate_buf,
                        offsets.intermediate,
                        num_kv_heads as u32,
                        head_dim as u32,
                        eps,
                    );

                    let k_cache = kv_pool
                        .layer_k_cache(slot_view.slot, layer_idx)
                        .map_err(|err| err.to_string())?;
                    let v_cache = kv_pool
                        .layer_v_cache(slot_view.slot, layer_idx)
                        .map_err(|err| err.to_string())?;
                    match self.kv_cache_type {
                        KvCacheType::F16 => {
                            self.ctx.encode_kv_append_f16_at(
                                encoder,
                                &self.decode_batch_scratch.k_normed_buf,
                                offsets.kv,
                                k_cache,
                                num_kv_heads as u32,
                                head_dim as u32,
                                kv_pool.capacity(),
                                append_pos,
                            );
                            self.ctx.encode_kv_append_f16_at(
                                encoder,
                                &self.decode_batch_scratch.gate_buf,
                                offsets.intermediate,
                                v_cache,
                                num_kv_heads as u32,
                                head_dim as u32,
                                kv_pool.capacity(),
                                append_pos,
                            );
                        }
                        KvCacheType::Q8_0 => {
                            self.ctx.encode_kv_append_q8_0_at(
                                encoder,
                                &self.decode_batch_scratch.k_normed_buf,
                                offsets.kv,
                                k_cache,
                                num_kv_heads as u32,
                                head_dim as u32,
                                kv_pool.capacity(),
                                append_pos,
                            );
                            self.ctx.encode_kv_append_q8_0_at(
                                encoder,
                                &self.decode_batch_scratch.gate_buf,
                                offsets.intermediate,
                                v_cache,
                                num_kv_heads as u32,
                                head_dim as u32,
                                kv_pool.capacity(),
                                append_pos,
                            );
                        }
                        KvCacheType::Q4_0 => {
                            if crate::gpu::needs_explicit_kv_append(layer.has_kv, effective_kv_seq) {
                                self.ctx.encode_kv_append_q4_0_at(
                                    encoder,
                                    &self.decode_batch_scratch.k_normed_buf,
                                    offsets.kv,
                                    k_cache,
                                    num_kv_heads as u32,
                                    head_dim as u32,
                                    kv_pool.capacity(),
                                    append_pos,
                                );
                                self.ctx.encode_kv_append_q4_0_at(
                                    encoder,
                                    &self.decode_batch_scratch.gate_buf,
                                    offsets.intermediate,
                                    v_cache,
                                    num_kv_heads as u32,
                                    head_dim as u32,
                                    kv_pool.capacity(),
                                    append_pos,
                                );
                            }
                        }
                    
                KvCacheType::TurboQuant { .. } => {
                            let spilled = kv_pool
                                .tq_hot_spilled(slot_view.slot)
                                .map_err(|e| e.to_string())?;
                            let use_hot = self.tq_hot_enabled()
                                && !spilled
                                && (append_pos + 1) <= self.tq_hot_w;
                            if !use_hot {
                                return Err(
                                    "TurboQuant decode-batch cold path: use single-slot decode"
                                        .to_string(),
                                );
                            }
                            // Always explicit append into hot — skip fused-append
                            // attention so we don't double-write the ring.
                            let (hot_k, hot_v) =
                                self.tq_slot_hot_bufs(kv_pool, slot_view.slot, layer_idx)?;
                            self.ctx.encode_kv_append_q4_0_at(
                                encoder,
                                &self.decode_batch_scratch.k_normed_buf,
                                offsets.kv,
                                hot_k,
                                num_kv_heads as u32,
                                head_dim as u32,
                                self.tq_hot_w,
                                append_pos,
                            );
                            self.ctx.encode_kv_append_q4_0_at(
                                encoder,
                                &self.decode_batch_scratch.gate_buf,
                                offsets.intermediate,
                                hot_v,
                                num_kv_heads as u32,
                                head_dim as u32,
                                self.tq_hot_w,
                                append_pos,
                            );
                            let _ = (k_cache, v_cache);
                        }
            }
                }

                let k_cache = kv_pool
                    .layer_k_cache(slot_view.slot, layer.kv_source_layer)
                    .map_err(|err| err.to_string())?;
                let v_cache = kv_pool
                    .layer_v_cache(slot_view.slot, layer.kv_source_layer)
                    .map_err(|err| err.to_string())?;
                match self.kv_cache_type {
                    KvCacheType::F16 => {
                        self.ctx.encode_attention_with_offset_f16_at(
                            encoder,
                            &self.decode_batch_scratch.q_normed_buf,
                            offsets.q,
                            k_cache,
                            v_cache,
                            &self.decode_batch_scratch.attn_out_buf,
                            offsets.q,
                            num_heads as u32,
                            num_kv_heads as u32,
                            num_kv_groups,
                            head_dim as u32,
                            effective_kv_seq,
                            kv_pool.capacity(),
                            scale,
                            kv_start,
                        );
                    }
                    KvCacheType::Q8_0 => {
                        let groups_per_row = (head_dim / 32) as u32;
                        let row_bytes = groups_per_row * 34;
                        self.ctx.encode_attention_with_offset_q8_0_at(
                            encoder,
                            &self.decode_batch_scratch.q_normed_buf,
                            offsets.q,
                            k_cache,
                            v_cache,
                            &self.decode_batch_scratch.attn_out_buf,
                            offsets.q,
                            num_heads as u32,
                            num_kv_heads as u32,
                            num_kv_groups,
                            head_dim as u32,
                            effective_kv_seq,
                            kv_pool.capacity(),
                            scale,
                            kv_start,
                            groups_per_row,
                            row_bytes,
                        );
                    }
                    KvCacheType::Q4_0 => {
                        let groups_per_row = (head_dim / 32) as u32;
                        let row_bytes = groups_per_row * 18;
                        if crate::gpu::attention_use_ggml_for_layer_kv(layer.has_kv, effective_kv_seq)
                        && self.ctx.use_flash_attention
                    {
                            self.ctx.encode_attention_ggml_q4_0_at(
                                encoder,
                                &self.decode_batch_scratch.q_normed_buf,
                                offsets.q,
                                k_cache,
                                0,
                                v_cache,
                                0,
                                &self.decode_batch_scratch.ggml_fa_tmp_buf,
                                0,
                                &self.decode_batch_scratch.attn_out_buf,
                                offsets.q,
                                num_heads as u32,
                                num_kv_heads as u32,
                                head_dim as u32,
                                effective_kv_seq,
                                kv_pool.capacity(),
                                scale,
                                kv_start,
                                row_bytes,
                            );
                        } else if layer.has_kv
                            && self.ctx.use_flash_attention
                            && crate::gpu::fused_kv_attention_enabled()
                        {
                            self.ctx.encode_attention_fused_q4_0_at(
                                encoder,
                                &self.decode_batch_scratch.q_normed_buf,
                                offsets.q,
                                &self.decode_batch_scratch.k_normed_buf,
                                offsets.kv,
                                &self.decode_batch_scratch.gate_buf,
                                offsets.intermediate,
                                &self.decode_batch_scratch.attn_out_buf,
                                offsets.q,
                                k_cache,
                                v_cache,
                                num_heads as u32,
                                num_kv_heads as u32,
                                num_kv_groups,
                                head_dim as u32,
                                effective_kv_seq,
                                kv_pool.capacity(),
                                scale,
                                kv_start,
                                append_pos,
                                groups_per_row,
                                row_bytes,
                            );
                        } else if self.ctx.use_flash_attention
                            && num_kv_groups > 1
                            && (8 % num_kv_groups) == 0
                            && crate::gpu::attention_gqa_q4_0_enabled(num_kv_groups)
                        {
                            self.ctx.encode_attention_with_offset_q4_0_gqa_at(
                                encoder,
                                &self.decode_batch_scratch.q_normed_buf,
                                offsets.q,
                                k_cache,
                                v_cache,
                                &self.decode_batch_scratch.attn_out_buf,
                                offsets.q,
                                num_heads as u32,
                                num_kv_heads as u32,
                                num_kv_groups,
                                head_dim as u32,
                                effective_kv_seq,
                                kv_pool.capacity(),
                                scale,
                                kv_start,
                                groups_per_row,
                                row_bytes,
                            );
                        } else {
                            self.ctx.encode_attention_with_offset_q4_0_at(
                                encoder,
                                &self.decode_batch_scratch.q_normed_buf,
                                offsets.q,
                                k_cache,
                                v_cache,
                                &self.decode_batch_scratch.attn_out_buf,
                                offsets.q,
                                num_heads as u32,
                                num_kv_heads as u32,
                                num_kv_groups,
                                head_dim as u32,
                                effective_kv_seq,
                                kv_pool.capacity(),
                                scale,
                                kv_start,
                                groups_per_row,
                                row_bytes,
                            );
                        }
                    }
                
                KvCacheType::TurboQuant { .. } => {
                        let spilled = kv_pool
                            .tq_hot_spilled(slot_view.slot)
                            .map_err(|e| e.to_string())?;
                        let use_hot = self.tq_hot_enabled()
                            && !spilled
                            && effective_kv_seq <= self.tq_hot_w;
                        if !use_hot {
                            return Err(
                                "TurboQuant decode-batch cold path: use single-slot decode"
                                    .to_string(),
                            );
                        }
                        let src = layer.kv_source_layer;
                        let (hot_k, hot_v) =
                            self.tq_slot_hot_bufs(kv_pool, slot_view.slot, src)?;
                        let groups_per_row = (head_dim / 32) as u32;
                        let row_bytes = groups_per_row * 18;
                        // Prefer ggml / offset — never fused-append (KV already written).
                        if crate::gpu::attention_use_ggml_for_layer_kv(
                            layer.has_kv,
                            effective_kv_seq,
                        ) && self.ctx.use_flash_attention
                        {
                            self.ctx.encode_attention_ggml_q4_0_at(
                                encoder,
                                &self.decode_batch_scratch.q_normed_buf,
                                offsets.q,
                                hot_k,
                                0,
                                hot_v,
                                0,
                                &self.decode_batch_scratch.ggml_fa_tmp_buf,
                                0,
                                &self.decode_batch_scratch.attn_out_buf,
                                offsets.q,
                                num_heads as u32,
                                num_kv_heads as u32,
                                head_dim as u32,
                                effective_kv_seq,
                                self.tq_hot_w,
                                scale,
                                kv_start,
                                row_bytes,
                            );
                        } else if self.ctx.use_flash_attention
                            && num_kv_groups > 1
                            && (8 % num_kv_groups) == 0
                            && crate::gpu::attention_gqa_q4_0_enabled(num_kv_groups)
                        {
                            self.ctx.encode_attention_with_offset_q4_0_gqa_at(
                                encoder,
                                &self.decode_batch_scratch.q_normed_buf,
                                offsets.q,
                                hot_k,
                                hot_v,
                                &self.decode_batch_scratch.attn_out_buf,
                                offsets.q,
                                num_heads as u32,
                                num_kv_heads as u32,
                                num_kv_groups,
                                head_dim as u32,
                                effective_kv_seq,
                                self.tq_hot_w,
                                scale,
                                kv_start,
                                groups_per_row,
                                row_bytes,
                            );
                        } else {
                            self.ctx.encode_attention_with_offset_q4_0_at(
                                encoder,
                                &self.decode_batch_scratch.q_normed_buf,
                                offsets.q,
                                hot_k,
                                hot_v,
                                &self.decode_batch_scratch.attn_out_buf,
                                offsets.q,
                                num_heads as u32,
                                num_kv_heads as u32,
                                num_kv_groups,
                                head_dim as u32,
                                effective_kv_seq,
                                self.tq_hot_w,
                                scale,
                                kv_start,
                                groups_per_row,
                                row_bytes,
                            );
                        }
                        let _ = (k_cache, v_cache);
                    }
            }

                self.ctx.encode_matvec_auto_at_view(
                    encoder,
                    &layer.o_proj,
                    &self.decode_batch_scratch.attn_out_buf,
                    offsets.q,
                    &self.decode_batch_scratch.o_out_buf,
                    offsets.hidden,
                    hidden_size as u32,
                    q_out as u32,
                );
                if crate::gpu::fused_rmsnorm_acc_enabled() {
                    self.ctx.encode_rmsnorm_acc_out_at_view(
                        encoder,
                        &self.decode_batch_scratch.residual_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.o_out_buf,
                        offsets.hidden,
                        &layer.post_attention_layernorm_weight,
                        &self.decode_batch_scratch.hidden_buf,
                        offsets.hidden,
                        hidden_size as u32,
                        eps,
                    );
                } else {
                    self.ctx.encode_rmsnorm_at_view(
                        encoder,
                        &self.decode_batch_scratch.o_out_buf,
                        offsets.hidden,
                        &layer.post_attention_layernorm_weight,
                        &self.decode_batch_scratch.normed_buf,
                        offsets.hidden,
                        hidden_size as u32,
                        eps,
                    );
                    self.ctx.encode_vec_add_at(
                        encoder,
                        &self.decode_batch_scratch.residual_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.normed_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.hidden_buf,
                        offsets.hidden,
                        hidden_size as u32,
                    );
                }

                self.ctx.encode_copy_at(
                    encoder,
                    &self.decode_batch_scratch.hidden_buf,
                    offsets.hidden,
                    &self.decode_batch_scratch.residual_buf,
                    offsets.hidden,
                    hidden_size as u32,
                );
                if layer.weight_format == WeightFormat::F16 {
                    self.ctx.encode_rmsnorm_at_view(
                        encoder,
                        &self.decode_batch_scratch.hidden_buf,
                        offsets.hidden,
                        &layer.pre_feedforward_layernorm_weight,
                        &self.decode_batch_scratch.normed_buf,
                        offsets.hidden,
                        hidden_size as u32,
                        eps,
                    );
                    self.ctx.encode_matvec_f16_at_view(
                        encoder,
                        &layer.gate_proj,
                        &self.decode_batch_scratch.normed_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.gate_buf,
                        offsets.intermediate,
                        intermediate_size as u32,
                        hidden_size as u32,
                    );
                    self.ctx.encode_matvec_f16_at_view(
                        encoder,
                        &layer.up_proj,
                        &self.decode_batch_scratch.normed_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.up_buf,
                        offsets.intermediate,
                        intermediate_size as u32,
                        hidden_size as u32,
                    );
                    self.ctx.encode_gelu_mul_at(
                        encoder,
                        &self.decode_batch_scratch.gate_buf,
                        offsets.intermediate,
                        &self.decode_batch_scratch.up_buf,
                        offsets.intermediate,
                        &self.decode_batch_scratch.gelu_buf,
                        offsets.intermediate,
                        intermediate_size as u32,
                    );
                    self.ctx.encode_matvec_f16_at_view(
                        encoder,
                        &layer.down_proj,
                        &self.decode_batch_scratch.gelu_buf,
                        offsets.intermediate,
                        &self.decode_batch_scratch.down_buf,
                        offsets.hidden,
                        hidden_size as u32,
                        intermediate_size as u32,
                    );
                } else if layer.weight_format.is_kquant() {
                    // K-quant (Q4_K_M): gate/up fused; optional pre-FF RMSNorm fold.
                    if layer.gate_proj.format == crate::gpu::weight_fmt::Q4_K
                        && layer.up_proj.format == crate::gpu::weight_fmt::Q4_K
                        && crate::gpu::fused_rmsnorm_mlp_kquant_enabled()
                    {
                        self.ctx.encode_rmsnorm_qk_gelu_mul_kquant_at_view(
                            encoder,
                            &layer.gate_proj,
                            &layer.up_proj,
                            &self.decode_batch_scratch.hidden_buf,
                            offsets.hidden,
                            &layer.pre_feedforward_layernorm_weight,
                            &self.inv_rms_buf,
                            &self.decode_batch_scratch.gelu_buf,
                            offsets.intermediate,
                            intermediate_size as u32,
                            hidden_size as u32,
                            eps,
                        );
                    } else if layer.gate_proj.format == crate::gpu::weight_fmt::Q4_K
                        && layer.up_proj.format == crate::gpu::weight_fmt::Q4_K
                    {
                        self.ctx.encode_rmsnorm_at_view(
                            encoder,
                            &self.decode_batch_scratch.hidden_buf,
                            offsets.hidden,
                            &layer.pre_feedforward_layernorm_weight,
                            &self.decode_batch_scratch.normed_buf,
                            offsets.hidden,
                            hidden_size as u32,
                            eps,
                        );
                        self.ctx.encode_matvec_qk_gelu_mul_at_view(
                            encoder,
                            &layer.gate_proj,
                            &layer.up_proj,
                            &self.decode_batch_scratch.normed_buf,
                            offsets.hidden,
                            &self.decode_batch_scratch.gelu_buf,
                            offsets.intermediate,
                            intermediate_size as u32,
                            hidden_size as u32,
                        );
                    } else {
                        self.ctx.encode_rmsnorm_at_view(
                            encoder,
                            &self.decode_batch_scratch.hidden_buf,
                            offsets.hidden,
                            &layer.pre_feedforward_layernorm_weight,
                            &self.decode_batch_scratch.normed_buf,
                            offsets.hidden,
                            hidden_size as u32,
                            eps,
                        );
                        self.encode_matvec_quant_at(
                            encoder,
                            &layer.gate_proj,
                            &self.decode_batch_scratch.normed_buf,
                            offsets.hidden,
                            &self.decode_batch_scratch.gate_buf,
                            offsets.intermediate,
                            intermediate_size as u32,
                            hidden_size as u32,
                            layer.weight_format,
                        );
                        self.encode_matvec_quant_at(
                            encoder,
                            &layer.up_proj,
                            &self.decode_batch_scratch.normed_buf,
                            offsets.hidden,
                            &self.decode_batch_scratch.up_buf,
                            offsets.intermediate,
                            intermediate_size as u32,
                            hidden_size as u32,
                            layer.weight_format,
                        );
                        self.ctx.encode_gelu_mul_at(
                            encoder,
                            &self.decode_batch_scratch.gate_buf,
                            offsets.intermediate,
                            &self.decode_batch_scratch.up_buf,
                            offsets.intermediate,
                            &self.decode_batch_scratch.gelu_buf,
                            offsets.intermediate,
                            intermediate_size as u32,
                        );
                    }
                    self.encode_matvec_quant_at(
                        encoder,
                        &layer.down_proj,
                        &self.decode_batch_scratch.gelu_buf,
                        offsets.intermediate,
                        &self.decode_batch_scratch.down_buf,
                        offsets.hidden,
                        hidden_size as u32,
                        intermediate_size as u32,
                        layer.weight_format,
                    );
                } else if crate::gpu::fused_mlp_gelu_down_enabled()
                    && crate::gpu::fused_rmsnorm_mlp_enabled()
                    && Self::use_packed_mlp_gate_up(layer)
                    && crate::gpu::weight_buf_is_q4(
                        &layer.gate_proj,
                        intermediate_size as u32,
                        hidden_size as u32,
                    )
                {
                    let inv_rms_off = offsets.hidden / hidden_size as u64;
                    self.ctx.encode_mlp_fused_q4_gelu_down_packed_from_hidden_at_view(
                        encoder,
                        &layer.gate_up_proj,
                        &layer.down_proj,
                        &layer.pre_feedforward_layernorm_weight,
                        &self.decode_batch_scratch.hidden_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.inv_rms_buf,
                        inv_rms_off,
                        &self.decode_batch_scratch.up_buf,
                        offsets.intermediate,
                        &self.decode_batch_scratch.down_buf,
                        offsets.hidden,
                        hidden_size as u32,
                        intermediate_size as u32,
                        eps,
                    );
                } else if crate::gpu::fused_mlp_gelu_down_enabled()
                    && !layer.weight_format.is_q3()
                    && crate::gpu::weight_buf_is_q4(
                        &layer.gate_proj,
                        intermediate_size as u32,
                        hidden_size as u32,
                    )
                {
                    self.ctx.encode_rmsnorm_at_view(
                        encoder,
                        &self.decode_batch_scratch.hidden_buf,
                        offsets.hidden,
                        &layer.pre_feedforward_layernorm_weight,
                        &self.decode_batch_scratch.normed_buf,
                        offsets.hidden,
                        hidden_size as u32,
                        eps,
                    );
                    if Self::use_packed_mlp_gate_up(layer) {
                        self.ctx.encode_mlp_fused_q4_gelu_down_packed_at_view(
                            encoder,
                            &layer.gate_up_proj,
                            &layer.down_proj,
                            &self.decode_batch_scratch.normed_buf,
                            offsets.hidden,
                            &self.decode_batch_scratch.up_buf,
                            offsets.intermediate,
                            &self.decode_batch_scratch.down_buf,
                            offsets.hidden,
                            hidden_size as u32,
                            intermediate_size as u32,
                        );
                    } else {
                        self.ctx.encode_mlp_fused_q4_gelu_down_at_view(
                            encoder,
                            &layer.gate_proj,
                            &layer.up_proj,
                            &layer.down_proj,
                            &self.decode_batch_scratch.normed_buf,
                            offsets.hidden,
                            &self.decode_batch_scratch.gelu_buf,
                            offsets.intermediate,
                            &self.decode_batch_scratch.down_buf,
                            offsets.hidden,
                            hidden_size as u32,
                            intermediate_size as u32,
                        );
                    }
                } else if Self::use_packed_mlp_gate_up(layer)
                    || layer.weight_format.is_q3()
                    || crate::gpu::fused_mlp_ple_enabled()
                {
                    self.ctx.encode_rmsnorm_at_view(
                        encoder,
                        &self.decode_batch_scratch.hidden_buf,
                        offsets.hidden,
                        &layer.pre_feedforward_layernorm_weight,
                        &self.decode_batch_scratch.normed_buf,
                        offsets.hidden,
                        hidden_size as u32,
                        eps,
                    );
                    self.encode_mlp_gate_up_gelu_q4_at_view(
                        encoder,
                        layer,
                        &self.decode_batch_scratch.normed_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.gelu_buf,
                        offsets.intermediate,
                        intermediate_size as u32,
                        hidden_size as u32,
                    );
                    self.encode_matvec_quant(
                        encoder,
                        &layer.down_proj,
                        &self.decode_batch_scratch.gelu_buf,
                        &self.decode_batch_scratch.down_buf,
                        hidden_size as u32,
                        intermediate_size as u32,
                        layer.weight_format,
                    );
                } else {
                    self.ctx.encode_rmsnorm_at_view(
                        encoder,
                        &self.decode_batch_scratch.hidden_buf,
                        offsets.hidden,
                        &layer.pre_feedforward_layernorm_weight,
                        &self.decode_batch_scratch.normed_buf,
                        offsets.hidden,
                        hidden_size as u32,
                        eps,
                    );
                    self.encode_matvec_quant(
                        encoder,
                        &layer.gate_proj,
                        &self.decode_batch_scratch.normed_buf,
                        &self.decode_batch_scratch.gate_buf,
                        intermediate_size as u32,
                        hidden_size as u32,
                        layer.weight_format,
                    );
                    self.encode_matvec_quant(
                        encoder,
                        &layer.up_proj,
                        &self.decode_batch_scratch.normed_buf,
                        &self.decode_batch_scratch.up_buf,
                        intermediate_size as u32,
                        hidden_size as u32,
                        layer.weight_format,
                    );
                    self.ctx.encode_gelu_mul_at(
                        encoder,
                        &self.decode_batch_scratch.gate_buf,
                        offsets.intermediate,
                        &self.decode_batch_scratch.up_buf,
                        offsets.intermediate,
                        &self.decode_batch_scratch.gelu_buf,
                        offsets.intermediate,
                        intermediate_size as u32,
                    );
                    self.encode_matvec_quant(
                        encoder,
                        &layer.down_proj,
                        &self.decode_batch_scratch.gelu_buf,
                        &self.decode_batch_scratch.down_buf,
                        hidden_size as u32,
                        intermediate_size as u32,
                        layer.weight_format,
                    );
                }
                if crate::gpu::fused_rmsnorm_acc_enabled() {
                    self.ctx.encode_rmsnorm_acc_out_at_view(
                        encoder,
                        &self.decode_batch_scratch.residual_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.down_buf,
                        offsets.hidden,
                        &layer.post_feedforward_layernorm_weight,
                        &self.decode_batch_scratch.hidden_buf,
                        offsets.hidden,
                        hidden_size as u32,
                        eps,
                    );
                } else {
                    self.ctx.encode_rmsnorm_at_view(
                        encoder,
                        &self.decode_batch_scratch.down_buf,
                        offsets.hidden,
                        &layer.post_feedforward_layernorm_weight,
                        &self.decode_batch_scratch.normed_buf,
                        offsets.hidden,
                        hidden_size as u32,
                        eps,
                    );
                    self.ctx.encode_vec_add_at(
                        encoder,
                        &self.decode_batch_scratch.residual_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.normed_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.hidden_buf,
                        offsets.hidden,
                        hidden_size as u32,
                    );
                }

                self.ctx.encode_copy_at(
                    encoder,
                    &self.decode_batch_scratch.hidden_buf,
                    offsets.hidden,
                    &self.decode_batch_scratch.residual_buf,
                    offsets.hidden,
                    hidden_size as u32,
                );
                if crate::gpu::fused_mlp_ple_enabled()
                    && crate::gpu::weight_buf_is_q4(
                        &layer.per_layer_input_gate_weight,
                        ple_dim as u32,
                        hidden_size as u32,
                    )
                {
                    self.ctx.encode_ple_matvec_gelu_q4_at_view(
                        encoder,
                        &layer.per_layer_input_gate_weight,
                        &self.decode_batch_scratch.hidden_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.ple_context_proj_buf,
                        ple_layer_offset,
                        &self.decode_batch_scratch.up_buf,
                        offsets.intermediate,
                        ple_dim as u32,
                        hidden_size as u32,
                    );
                } else {
                    self.ctx.encode_matvec_auto_at_view(
                        encoder,
                        &layer.per_layer_input_gate_weight,
                        &self.decode_batch_scratch.hidden_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.gate_buf,
                        offsets.intermediate,
                        ple_dim as u32,
                        hidden_size as u32,
                    );
                    self.ctx.encode_gelu_mul_at(
                        encoder,
                        &self.decode_batch_scratch.gate_buf,
                        offsets.intermediate,
                        &self.decode_batch_scratch.ple_context_proj_buf,
                        ple_layer_offset,
                        &self.decode_batch_scratch.up_buf,
                        offsets.intermediate,
                        ple_dim as u32,
                    );
                }
                self.ctx.encode_matvec_auto_at_view(
                    encoder,
                    &layer.per_layer_projection_weight,
                    &self.decode_batch_scratch.up_buf,
                    offsets.intermediate,
                    &self.decode_batch_scratch.o_out_buf,
                    offsets.hidden,
                    hidden_size as u32,
                    ple_dim as u32,
                );
                if crate::gpu::fused_rmsnorm_acc_enabled() {
                    self.ctx.encode_rmsnorm_acc_out_at_view(
                        encoder,
                        &self.decode_batch_scratch.residual_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.o_out_buf,
                        offsets.hidden,
                        &layer.post_per_layer_input_norm_weight,
                        &self.decode_batch_scratch.hidden_buf,
                        offsets.hidden,
                        hidden_size as u32,
                        eps,
                    );
                } else {
                    self.ctx.encode_rmsnorm_at_view(
                        encoder,
                        &self.decode_batch_scratch.o_out_buf,
                        offsets.hidden,
                        &layer.post_per_layer_input_norm_weight,
                        &self.decode_batch_scratch.normed_buf,
                        offsets.hidden,
                        hidden_size as u32,
                        eps,
                    );
                    self.ctx.encode_vec_add_at(
                        encoder,
                        &self.decode_batch_scratch.residual_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.normed_buf,
                        offsets.hidden,
                        &self.decode_batch_scratch.hidden_buf,
                        offsets.hidden,
                        hidden_size as u32,
                    );
                }
                self.ctx.encode_vec_scale_at(
                    encoder,
                    &self.decode_batch_scratch.hidden_buf,
                    offsets.hidden,
                    &self.decode_batch_scratch.residual_buf,
                    offsets.hidden,
                    hidden_size as u32,
                    layer.layer_scalar,
                );
                self.ctx.encode_copy_at(
                    encoder,
                    &self.decode_batch_scratch.residual_buf,
                    offsets.hidden,
                    &self.decode_batch_scratch.hidden_buf,
                    offsets.hidden,
                    hidden_size as u32,
                );
            }

            encoder.end_encoding();
        }

        {
            let encoder = cmd.new_compute_command_encoder();
            self.ctx.encode_rmsnorm_batch_view(
                encoder,
                &self.decode_batch_scratch.hidden_buf,
                &self.final_norm_weight,
                &self.decode_batch_scratch.normed_buf,
                hidden_size as u32,
                eps,
                batch_size as u32,
            );
            for batch_idx in 0..batch_size {
                let offsets = self.decode_batch_row_offsets(batch_idx);
                let logits_offset = Self::f32_byte_offset(batch_idx * vocab_size);
                self.ctx.encode_matvec_auto_at_view(
                    encoder,
                    &self.lm_head_buf,
                    &self.decode_batch_scratch.normed_buf,
                    offsets.hidden,
                    &self.decode_batch_scratch.logits_buf,
                    logits_offset,
                    vocab_size as u32,
                    hidden_size as u32,
                );
            }
            if capture_mtp {
                self.ctx.encode_copy_at(
                    encoder,
                    &self.decode_batch_scratch.normed_buf,
                    0,
                    &self.mtp_verify_hidden_buf,
                    0,
                    (batch_size * hidden_size) as u32,
                );
                self.ctx.encode_softcap_argmax_rows_f32(
                    encoder,
                    &self.decode_batch_scratch.logits_buf,
                    &self.mtp_verify_argmax_buf,
                    batch_size as u32,
                    vocab_size as u32,
                    self.config.final_logit_softcapping,
                );
            }
            encoder.end_encoding();
        }

        cmd.commit();
        cmd.wait_until_completed();

        let mut outputs = Vec::with_capacity(batch_size);
        if !capture_mtp {
            let mut logits_batch = MetalContext::read_buffer(
                &self.decode_batch_scratch.logits_buf,
                batch_size * vocab_size,
            );
            let cap = self.config.final_logit_softcapping;
            for batch_idx in 0..batch_size {
                let start = batch_idx * vocab_size;
                let end = start + vocab_size;
                for logit in &mut logits_batch[start..end] {
                    let x = (*logit / cap).clamp(-10.0, 10.0);
                    *logit = cap * x.tanh();
                }
                outputs.push(logits_batch[start..end].to_vec());
            }
        }

        for slot_view in slot_views {
            kv_pool
                .with_slot_mut(slot_view.slot, |slot_state| {
                    slot_state.seq_len += 1;
                    slot_state.total_tokens += 1;
                })
                .map_err(|err| err.to_string())?;
        }

        Ok(outputs)
    }

    /// Batched prefill: intermediate tokens skip lm_head; only the last token
    /// computes logits (avoids N× full-vocab readback during prompt processing).
    pub fn forward_prefill(&mut self, token_ids: &[usize]) -> Vec<f32> {
        if token_ids.is_empty() {
            return Vec::new();
        }
        if token_ids.len() == 1 {
            return self.forward_single_token(token_ids[0]);
        }
        // Prefer parallel prefill (Q4_0 always; TurboQuant while prompt fits hot window).
        let can_parallel = !matches!(self.kv_cache_type, KvCacheType::TurboQuant { .. })
            || self.tq_prefill_fits_hot(0, token_ids.len());
        if can_parallel {
            if let Ok(logits) = self.forward_prefill_parallel_self(token_ids) {
                return logits;
            }
        }
        for &tid in &token_ids[..token_ids.len() - 1] {
            self.forward_advance(tid);
        }
        self.forward_single_token(*token_ids.last().unwrap())
    }

    /// Prefill with GPU sampling on the last prompt token (4-byte readback only).
    pub fn forward_prefill_sample_last(
        &mut self,
        token_ids: &[usize],
        temperature: f32,
        min_p: f32,
        seed: u32,
    ) -> usize {
        if token_ids.is_empty() {
            return 0;
        }
        if token_ids.len() == 1 {
            return self.forward_single_token_sample(token_ids[0], temperature, min_p, seed);
        }
        let can_parallel = !matches!(self.kv_cache_type, KvCacheType::TurboQuant { .. })
            || self.tq_prefill_fits_hot(0, token_ids.len());
        if can_parallel {
            if let Ok(logits) = self.forward_prefill_parallel_self(token_ids) {
                // Parallel path materializes last-token logits; sample on CPU.
                return if temperature <= 0.0 {
                    crate::sampling::argmax(&logits)
                } else {
                    crate::sampling::min_p_sampling(&logits, min_p)
                };
            }
        }
        for &tid in &token_ids[..token_ids.len() - 1] {
            self.forward_advance(tid);
        }
        self.forward_single_token_sample(
            *token_ids.last().unwrap(),
            temperature,
            min_p,
            seed,
        )
    }

    pub fn forward_single_token_with_kv_slot(
        &mut self,
        token_id: usize,
        kv_pool: &mut KvCachePool,
        slot: KvSlot,
    ) -> Result<Vec<f32>, KvPoolError> {
        let pool_capacity = kv_pool.capacity();
        kv_pool.with_slot_mut(slot, |slot_state| {
            std::mem::swap(&mut self.k_cache, &mut slot_state.k_cache);
            std::mem::swap(&mut self.v_cache, &mut slot_state.v_cache);
            let swap_hot = !slot_state.tq_hot_k.is_empty() && !self.tq_hot_k.is_empty();
            let swap_rw = !slot_state.tq_rw_k.is_empty() && !self.tq_rw_k.is_empty();
            if swap_hot {
                std::mem::swap(&mut self.tq_hot_k, &mut slot_state.tq_hot_k);
                std::mem::swap(&mut self.tq_hot_v, &mut slot_state.tq_hot_v);
            }
            if swap_rw {
                std::mem::swap(&mut self.tq_rw_k, &mut slot_state.tq_rw_k);
                std::mem::swap(&mut self.tq_rw_v, &mut slot_state.tq_rw_v);
            }
            std::mem::swap(&mut self.tq_hot_spilled, &mut slot_state.tq_hot_spilled);

            let legacy_kv_seq_len = self.kv_seq_len;
            let legacy_total_tokens = self.total_tokens;
            let legacy_kv_capacity = self.kv_capacity;
            self.kv_seq_len = slot_state.seq_len;
            self.total_tokens = slot_state.total_tokens;
            self.kv_capacity = pool_capacity;

            let logits = self.forward_single_token(token_id);

            slot_state.seq_len = self.kv_seq_len;
            slot_state.total_tokens = self.total_tokens;
            self.kv_seq_len = legacy_kv_seq_len;
            self.total_tokens = legacy_total_tokens;
            self.kv_capacity = legacy_kv_capacity;

            std::mem::swap(&mut self.tq_hot_spilled, &mut slot_state.tq_hot_spilled);
            if swap_rw {
                std::mem::swap(&mut self.tq_rw_v, &mut slot_state.tq_rw_v);
                std::mem::swap(&mut self.tq_rw_k, &mut slot_state.tq_rw_k);
            }
            if swap_hot {
                std::mem::swap(&mut self.tq_hot_v, &mut slot_state.tq_hot_v);
                std::mem::swap(&mut self.tq_hot_k, &mut slot_state.tq_hot_k);
            }
            std::mem::swap(&mut self.v_cache, &mut slot_state.v_cache);
            std::mem::swap(&mut self.k_cache, &mut slot_state.k_cache);

            logits
        })
    }

    pub fn forward_decode_batch_with_kv_slots(
        &mut self,
        inputs: &[(KvSlot, usize)],
        kv_pool: &mut KvCachePool,
    ) -> Vec<Result<Vec<f32>, String>> {
        if inputs.len() > self.max_decode_batch_size() {
            return inputs
                .iter()
                .map(|_| {
                    Err(format!(
                        "decode batch has {} requests, max supported batch is {}",
                        inputs.len(),
                        self.max_decode_batch_size()
                    ))
                })
                .collect();
        }

        let token_ids: Vec<usize> = inputs.iter().map(|&(_, token_id)| token_id).collect();
        if let Err(message) = self.prepare_decode_batch_inputs(&token_ids) {
            return inputs.iter().map(|_| Err(message.clone())).collect();
        }

        let slots: Vec<KvSlot> = inputs.iter().map(|&(slot, _)| slot).collect();
        let slot_views = match kv_pool.slot_views(&slots) {
            Ok(slot_views) => slot_views,
            Err(err) => return inputs.iter().map(|_| Err(err.to_string())).collect(),
        };
        if let Err(message) = self.prepare_decode_batch_rotary(&slot_views) {
            return inputs.iter().map(|_| Err(message.clone())).collect();
        }
        if let Err(message) = self.prepare_decode_batch_attention_metadata(&slot_views) {
            return inputs.iter().map(|_| Err(message.clone())).collect();
        }

        if inputs.len() > 1 {
            // TurboQuant: always decode one slot at a time. The encoded multi-row
            // hot path still corrupts concurrent requests (garbage tokens). Slot
            // hot rings remain isolated via swap; only the GPU forward is serial.
            if matches!(self.kv_cache_type, KvCacheType::TurboQuant { .. }) {
                return inputs
                    .iter()
                    .map(|&(slot, token_id)| {
                        let seq = kv_pool.seq_len(slot).unwrap_or(0);
                        let spilled = kv_pool.tq_hot_spilled(slot).unwrap_or(false);
                        if self.tq_hot_enabled() && !spilled && seq >= self.tq_hot_w {
                            if let Err(err) = self.spill_tq_slot(kv_pool, slot) {
                                return Err(err);
                            }
                        }
                        self.forward_single_token_with_kv_slot(token_id, kv_pool, slot)
                            .map_err(|err| err.to_string())
                    })
                    .collect();
            }
            return match self.forward_decode_batch_encoded_with_kv_slots(
                inputs,
                &slot_views,
                kv_pool,
                false,
            ) {
                Ok(outputs) => outputs.into_iter().map(Ok).collect(),
                Err(message) => inputs.iter().map(|_| Err(message.clone())).collect(),
            };
        }

        inputs
            .iter()
            .map(|&(slot, token_id)| {
                self.forward_single_token_with_kv_slot(token_id, kv_pool, slot)
                    .map_err(|err| err.to_string())
            })
            .collect()
    }

    pub fn forward_prefill_batch_with_kv_slots(
        &mut self,
        inputs: &[(KvSlot, &[usize])],
        kv_pool: &mut KvCachePool,
    ) -> Vec<Result<Vec<f32>, String>> {
        if inputs.is_empty() {
            return Vec::new();
        }
        if inputs.len() == 1 {
            let (slot, token_ids) = inputs[0];
            return vec![self.forward_prefill_chunk_with_kv_slot(
                token_ids,
                kv_pool,
                slot,
                true,
            )];
        }

        // TurboQuant: never multi-slot strided prefill in one CB — concurrent
        // batches corrupted output. Isolate per slot (same kernels, separate CBs).
        if matches!(self.kv_cache_type, KvCacheType::TurboQuant { .. }) {
            return inputs
                .iter()
                .map(|&(slot, token_ids)| {
                    self.forward_prefill_chunk_with_kv_slot(token_ids, kv_pool, slot, true)
                })
                .collect();
        }

        let total_seq_len: usize = inputs.iter().map(|(_, token_ids)| token_ids.len()).sum();
        if total_seq_len == 0 {
            return inputs
                .iter()
                .map(|_| Err("prefill token_ids must not be empty".to_string()))
                .collect();
        }
        if total_seq_len > self.max_parallel_prefill_seq() {
            let message = format!(
                "prefill batch has {} total tokens, max supported batch is {}",
                total_seq_len,
                self.max_parallel_prefill_seq()
            );
            return inputs.iter().map(|_| Err(message.clone())).collect();
        }

        let mut flat_tokens = Vec::with_capacity(total_seq_len);
        let mut segments = Vec::with_capacity(inputs.len());
        let mut rotary_segments = Vec::with_capacity(inputs.len());
        let mut row_start = 0usize;

        for &(slot, token_ids) in inputs {
            if token_ids.is_empty() {
                return inputs
                    .iter()
                    .map(|_| Err("prefill token_ids must not be empty".to_string()))
                    .collect();
            }

            let start_pos = match kv_pool.total_tokens(slot) {
                Ok(start_pos) => start_pos,
                Err(err) => return inputs.iter().map(|_| Err(err.to_string())).collect(),
            };
            if start_pos + token_ids.len() > kv_pool.capacity() as usize {
                let message = format!(
                    "KV Cache overflow. Max length {}, current {}, new {}",
                    kv_pool.capacity(),
                    start_pos,
                    token_ids.len()
                );
                return inputs.iter().map(|_| Err(message.clone())).collect();
            }

            flat_tokens.extend_from_slice(token_ids);
            segments.push(PrefillBatchSegment {
                slot,
                row_start,
                token_count: token_ids.len(),
                start_pos,
            });
            rotary_segments.push((start_pos, token_ids.len()));
            row_start += token_ids.len();
        }

        if let Err(message) = self.prepare_parallel_prefill_inputs(&flat_tokens) {
            return inputs.iter().map(|_| Err(message.clone())).collect();
        }
        if let Err(message) = self.prepare_parallel_prefill_rotary_segments(&rotary_segments) {
            return inputs.iter().map(|_| Err(message.clone())).collect();
        }
        if let Err(message) = self.encode_parallel_prefill_ple_context(total_seq_len) {
            return inputs.iter().map(|_| Err(message.clone())).collect();
        }

        for layer_idx in 0..self.layers.len() {
            if let Err(message) = self.encode_parallel_prefill_layer_batched(
                layer_idx,
                total_seq_len,
                &segments,
                kv_pool,
            ) {
                return inputs.iter().map(|_| Err(message.clone())).collect();
            }
        }

        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let eps = self.config.rms_norm_eps as f32;
        let cmd = self.ctx.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        self.ctx.encode_rmsnorm_batch_view(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &self.final_norm_weight,
            &self.prefill_scratch.normed_buf,
            hidden_size as u32,
            eps,
            total_seq_len as u32,
        );
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let mut outputs = Vec::with_capacity(segments.len());
        for segment in &segments {
            let last_offsets =
                self.prefill_row_offsets(segment.row_start + segment.token_count - 1);
            let cmd = self.ctx.queue.new_command_buffer();
            let encoder = cmd.new_compute_command_encoder();
            self.ctx.encode_matvec_auto_at_view(
                encoder,
                &self.lm_head_buf,
                &self.prefill_scratch.normed_buf,
                last_offsets.hidden,
                &self.prefill_scratch.logits_buf,
                0,
                vocab_size as u32,
                hidden_size as u32,
            );
            encoder.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            let mut logits =
                MetalContext::read_buffer(&self.prefill_scratch.logits_buf, vocab_size);
            let cap = self.config.final_logit_softcapping;
            for logit in &mut logits {
                let x = (*logit / cap).clamp(-10.0, 10.0);
                *logit = cap * x.tanh();
            }
            outputs.push(Ok(logits));
        }

        for segment in &segments {
            if let Err(err) = kv_pool.with_slot_mut(segment.slot, |slot_state| {
                slot_state.seq_len += segment.token_count as u32;
                slot_state.total_tokens += segment.token_count;
            }) {
                return inputs.iter().map(|_| Err(err.to_string())).collect();
            }
        }

        outputs
    }

    pub fn forward_prefill_with_kv_slot(
        &mut self,
        token_ids: &[usize],
        kv_pool: &mut KvCachePool,
        slot: KvSlot,
    ) -> Result<Vec<f32>, KvPoolError> {
        let mut logits = Vec::new();
        for &tid in token_ids {
            logits = self.forward_single_token_with_kv_slot(tid, kv_pool, slot)?;
        }
        Ok(logits)
    }

    pub fn forward_prefill_chunked_with_kv_slot(
        &mut self,
        token_ids: &[usize],
        kv_pool: &mut KvCachePool,
        slot: KvSlot,
    ) -> Result<Vec<f32>, String> {
        if token_ids.is_empty() {
            return Err("prefill token_ids must not be empty".to_string());
        }

        let chunk_size = self.max_parallel_prefill_seq().max(1);
        let chunks: Vec<&[usize]> = token_ids.chunks(chunk_size).collect();
        let mut logits = Vec::new();

        for (idx, chunk) in chunks.iter().enumerate() {
            let is_last = idx + 1 == chunks.len();
            logits = self.forward_prefill_chunk_with_kv_slot(
                chunk,
                kv_pool,
                slot,
                is_last,
            )?;
        }

        Ok(logits)
    }

    pub fn forward_prefill_chunk_with_kv_slot(
        &mut self,
        token_ids: &[usize],
        kv_pool: &mut KvCachePool,
        slot: KvSlot,
        compute_logits: bool,
    ) -> Result<Vec<f32>, String> {
        let timing = crate::gpu::prefill_timing_enabled();
        let t_prep = std::time::Instant::now();
        self.prepare_parallel_prefill_inputs(token_ids)?;
        let prepare_ms = t_prep.elapsed().as_secs_f64() * 1e3;
        let start_pos = kv_pool.total_tokens(slot).map_err(|err| err.to_string())?;
        if !crate::gpu::prefill_gpu_rope_enabled() {
            self.prepare_parallel_prefill_rotary(start_pos, token_ids.len())?;
        }
        if timing {
            eprintln!(
                "[prefill timing] cpu_embed_ple_prepare={:.1}ms seq={}",
                prepare_ms,
                token_ids.len()
            );
        }

        if self.can_use_parallel_prefill_chunk(start_pos, token_ids.len(), kv_pool) {
            return self.forward_prefill_chunk_parallel_with_kv_slot(
                token_ids,
                kv_pool,
                slot,
                start_pos,
                compute_logits,
                false,
            );
        }

        let mut logits = Vec::new();
        for (idx, &tid) in token_ids.iter().enumerate() {
            let is_last = idx + 1 == token_ids.len();
            logits = self
                .forward_single_token_with_kv_slot(tid, kv_pool, slot)
                .map_err(|err| err.to_string())?;
            if !is_last || !compute_logits {
                logits.clear();
            }
        }
        Ok(logits)
    }
}

/// Decode a safetensors tensor view to Vec<f32>, handling f32/f16/bf16.
/// Build a `Gemma4TextConfig` from `gemma4.*` GGUF metadata.
fn gemma4_config_from_gguf(g: &crate::gguf::Gguf) -> Gemma4TextConfig {
    let mu = |k: &str| g.get_u32(k).unwrap_or_else(|| panic!("gguf missing u32 key {}", k)) as usize;

    let hidden_size = mu("gemma4.embedding_length");
    let num_hidden_layers = mu("gemma4.block_count");
    let num_attention_heads = mu("gemma4.attention.head_count");
    let head_count_kv_list = g
        .get_usize_list("gemma4.attention.head_count_kv")
        .unwrap_or_else(|| panic!("gguf missing gemma4.attention.head_count_kv"));
    let num_key_value_heads = *head_count_kv_list.iter().max().unwrap_or(&1);
    let num_key_value_heads_per_layer = if head_count_kv_list.len() > 1 {
        head_count_kv_list
    } else {
        Vec::new()
    };
    let head_dim = mu("gemma4.attention.key_length_swa"); // sliding head dim (e.g. 256)
    let global_head_dim = mu("gemma4.attention.key_length"); // full head dim (e.g. 512)
    let intermediate_sizes = g
        .get_usize_list("gemma4.feed_forward_length")
        .unwrap_or_else(|| panic!("gguf missing gemma4.feed_forward_length"));
    let intermediate_size = intermediate_sizes.iter().copied().max().unwrap_or(0);
    assert!(
        !intermediate_sizes.is_empty(),
        "gemma4.feed_forward_length must not be empty"
    );
    if intermediate_sizes.len() == 1 {
        // Uniform FFN (E4B): keep only the scalar in config.intermediate_size.
    } else {
        assert_eq!(
            intermediate_sizes.len(),
            num_hidden_layers,
            "feed_forward_length length {} != block_count {}",
            intermediate_sizes.len(),
            num_hidden_layers
        );
    }
    let uniform_intermediate = if intermediate_sizes.len() == 1 {
        intermediate_sizes[0]
    } else {
        intermediate_size
    };
    let vocab_size = g
        .tensor("token_embd.weight")
        .map(|t| t.n_rows())
        .expect("token_embd.weight missing");
    let rms_norm_eps =
        g.get_f32("gemma4.attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f64;
    let sliding_window = mu("gemma4.attention.sliding_window");
    let hidden_size_per_layer_input = mu("gemma4.embedding_length_per_layer_input");
    let num_kv_shared_layers = mu("gemma4.attention.shared_kv_layers");
    let max_position_embeddings =
        g.get_u32("gemma4.context_length").unwrap_or(131072) as usize;
    let final_logit_softcapping =
        g.get_f32("gemma4.final_logit_softcapping").unwrap_or(30.0);

    // sliding_window_pattern: true => sliding attention, false => full attention.
    let pattern = g
        .get_arr_bool("gemma4.attention.sliding_window_pattern")
        .expect("missing gemma4.attention.sliding_window_pattern");
    assert_eq!(
        pattern.len(),
        num_hidden_layers,
        "sliding_window_pattern length {} != block_count {}",
        pattern.len(),
        num_hidden_layers
    );
    let layer_types: Vec<String> = pattern
        .iter()
        .map(|&sliding| {
            if sliding {
                "sliding_attention".to_string()
            } else {
                "full_attention".to_string()
            }
        })
        .collect();

    let full_theta = g.get_f32("gemma4.rope.freq_base").unwrap_or(1_000_000.0) as f64;
    let sliding_theta = g.get_f32("gemma4.rope.freq_base_swa").unwrap_or(10_000.0) as f64;
    // Sliding layers: dimension_count_swa is the rotated width (usually == head_dim).
    let sliding_rope_dim =
        g.get_u32("gemma4.rope.dimension_count_swa").unwrap_or(head_dim as u32) as f64;
    // Full / global layers use proportional RoPE (p-RoPE, p=0.25): only the first
    // p*head_dim channels are rotated; the rest stay identity. GGUF stores
    // gemma4.rope.dimension_count == global_head_dim (the RoPE *op* width) and
    // rope_freqs.weight with 1.0 on the active angles and ~1e30 on the rest —
    // that is NOT "rotate all dims". Hard-coding p=0.25 matches HF / llama.cpp /
    // the rope_freqs mask (64 of 256 half-dims active on E2B).
    let full_partial_rotary = 0.25f64;

    let rope_parameters = Some(RopeParameters {
        full_attention: Some(RopeConfig {
            rope_theta: full_theta,
            rope_type: "proportional".to_string(),
            partial_rotary_factor: full_partial_rotary,
            factor: 1.0,
        }),
        sliding_attention: Some(RopeConfig {
            rope_theta: sliding_theta,
            rope_type: String::new(),
            partial_rotary_factor: sliding_rope_dim / head_dim as f64,
            factor: 1.0,
        }),
    });

    Gemma4TextConfig {
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        global_head_dim,
        intermediate_size: uniform_intermediate,
        intermediate_sizes: if intermediate_sizes.len() > 1 {
            intermediate_sizes
        } else {
            Vec::new()
        },
        vocab_size,
        hidden_activation: "gelu_pytorch_tanh".to_string(),
        rms_norm_eps,
        sliding_window,
        layer_types,
        hidden_size_per_layer_input,
        num_kv_shared_layers,
        max_position_embeddings,
        final_logit_softcapping,
        tie_word_embeddings: !g.has_tensor("output.weight"),
        rope_parameters,
        num_key_value_heads_per_layer,
    }
}

/// Convert an f32 slice to bf16 little-endian bytes (round to nearest even).
fn f32_to_bf16_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for &v in data {
        let bits = v.to_bits();
        let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
        let bf = ((bits + rounding_bias) >> 16) as u16;
        out.extend_from_slice(&bf.to_le_bytes());
    }
    out
}

fn decode_tensor_to_f32(tensor_view: &safetensors::tensor::TensorView) -> Vec<f32> {
    let dtype = tensor_view.dtype();
    let raw_data = tensor_view.data();

    match dtype {
        safetensors::Dtype::F32 => raw_data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        safetensors::Dtype::F16 => raw_data
            .chunks_exact(2)
            .map(|b| {
                let bits = u16::from_le_bytes([b[0], b[1]]);
                half_to_f32(bits)
            })
            .collect(),
        safetensors::Dtype::BF16 => raw_data
            .chunks_exact(2)
            .map(|b| {
                let bits = u16::from_le_bytes([b[0], b[1]]);
                bf16_to_f32(bits)
            })
            .collect(),
        _ => panic!("Unsupported dtype: {:?}", dtype),
    }
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut e = 1u32;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let f_exp = (127 - 15 + 1 - e) as u32;
        let f_mant = (m & 0x3FF) << 13;
        f32::from_bits((sign << 31) | (f_exp << 23) | f_mant)
    } else if exp == 31 {
        f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
    } else {
        let f_exp = (exp as i32 - 15 + 127) as u32;
        let f_mant = mant << 13;
        f32::from_bits((sign << 31) | (f_exp << 23) | f_mant)
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
