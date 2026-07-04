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
use crate::kv_pool::{KvCachePool, KvPoolError, KvSlot, KvSlotView};
use crate::mega_decode::{mega_kernel_enabled, MegaDecodeGraph, MegaScratchBuffers};

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

const DEFAULT_MAX_PREFILL_SEQ: usize = 128;
const DEFAULT_MAX_DECODE_BATCH: usize = 4;

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
    pub per_layer_model_projection_weight: BufferView, // [num_layers * ple_dim, hidden_size] f16

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

    // GPU-resident KV cache per layer
    pub k_cache: Vec<Buffer>,
    pub v_cache: Vec<Buffer>,
    pub kv_seq_len: u32,
    pub kv_capacity: u32,
    pub kv_cache_type: KvCacheType,

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

    /// Single-dispatch decode graph (MEGA_KERNEL=1).
    mega_graph: Option<MegaDecodeGraph>,
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
    ) -> Self {
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
    ) -> Self {
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
            gate_buf: ctx.buffer_empty(max_seq_len * intermediate_size),
            up_buf: ctx.buffer_empty(max_seq_len * intermediate_size),
            gelu_buf: ctx.buffer_empty(max_seq_len * intermediate_size),
            down_buf: ctx.buffer_empty(max_seq_len * hidden_size),
            logits_buf: ctx.buffer_empty(vocab_size),
            ple_context_proj_buf: ctx.buffer_empty(max_seq_len * num_layers * ple_dim),
            ple_token_id_buf: ctx.buffer_empty(max_seq_len * num_layers * ple_dim),
            ple_combined_buf: ctx.buffer_empty(max_seq_len * num_layers * ple_dim),
            q_normed_buf: ctx.buffer_empty(max_seq_len * max_q_out),
            k_normed_buf: ctx.buffer_empty(max_seq_len * max_kv_out),
        }
    }
}

/// CPU-resident embedding tables. Cache load mmap's the file (instant); first safetensors load owns bytes.
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
        } else {
            self.owned_ple.as_ref().unwrap()
        }
    }

    fn mmap_ref(&self) -> Option<&Mmap> {
        self.mmap.as_ref()
    }

    fn decode_embed_into(&self, token_id: usize, hidden_size: usize, out: &mut [f32]) {
        let scale = (hidden_size as f32).sqrt();
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

    fn decode_ple_into(
        &self,
        token_id: usize,
        ple_total_dim: usize,
        ple_dim: usize,
        out: &mut [f32],
    ) {
        let scale = (ple_dim as f32).sqrt();
        if self.ple_byte_len as u64 > self.vocab_size as u64 * ple_total_dim as u64 {
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
    pub gate_proj: BufferView,
    pub up_proj: BufferView,
    /// Interleaved [gate_i, up_i] Q4 rows for packed MLP matvec (decode).
    pub gate_up_proj: BufferView,
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
                ctx.buffer_from_f32_as_q4(
                    &per_layer_model_proj_data,
                    num_layers * ple_dim,
                    hidden_size,
                ),
            )
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

                gate_proj: BufferView::from_buffer(q_weight(&gate_proj_data, layer_inter, hidden_size)),
                up_proj: BufferView::from_buffer(q_weight(&up_proj_data, layer_inter, hidden_size)),
                gate_up_proj: BufferView::from_buffer(ctx.buffer_empty(1)),
                down_proj: BufferView::from_buffer(q_weight(&down_proj_data, hidden_size, layer_inter)),

                input_layernorm_weight: BufferView::from_buffer(ctx.buffer_from_slice(&input_ln)),
                post_attention_layernorm_weight: BufferView::from_buffer(ctx.buffer_from_slice(&post_attn_ln)),
                pre_feedforward_layernorm_weight: BufferView::from_buffer(ctx.buffer_from_slice(&pre_ff_ln)),
                post_feedforward_layernorm_weight: BufferView::from_buffer(ctx.buffer_from_slice(&post_ff_ln)),
                post_per_layer_input_norm_weight: BufferView::from_buffer(ctx.buffer_from_slice(&post_ple_norm)),

                per_layer_input_gate_weight: BufferView::from_buffer(ctx.buffer_from_f32_as_q4(
                    &ple_gate_data,
                    ple_dim,
                    hidden_size,
                )),
                per_layer_projection_weight: BufferView::from_buffer(ctx.buffer_from_f32_as_q4(
                    &ple_proj_data,
                    hidden_size,
                    ple_dim,
                )),
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
        let num_kv_heads = config.num_key_value_heads;
        let vocab_size = config.vocab_size;
        let num_layers = config.num_hidden_layers;
        let ple_dim = config.hidden_size_per_layer_input;
        let max_head_dim = config.global_head_dim;
        let max_q_out = num_heads * max_head_dim;
        let max_kv_out = num_kv_heads * max_head_dim;

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

        let layers = Self::pack_layers_gate_up(&ctx, layers, hidden_size as u32);

        let max_intermediate_size = layers
            .iter()
            .map(|layer| layer.intermediate_size)
            .max()
            .unwrap_or_else(|| config.max_intermediate_size());

        println!(
            "  KV sharing: layers 0-{} have own KV, layers {}-{} share",
            first_kv_shared - 1,
            first_kv_shared,
            num_layers - 1
        );

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
        // PLE scratch
        let ple_embed_buf = ctx.buffer_empty(ple_dim);
        let ple_gated_buf = ctx.buffer_empty(ple_dim);
        let ple_normed_buf = ctx.buffer_empty(ple_dim);
        let ple_projected_buf = ctx.buffer_empty(hidden_size);
        let ple_context_proj_buf = ctx.buffer_empty(num_layers * ple_dim);
        let ple_token_id_buf = ctx.buffer_empty(num_layers * ple_dim);
        let ple_combined_buf = ctx.buffer_empty(num_layers * ple_dim);

        // QK norm scratch (max head_dim per head)
        let q_normed_buf = ctx.buffer_empty(max_q_out);
        let k_normed_buf = ctx.buffer_empty(max_kv_out);

        // KV cache: f16 precision to halve memory bandwidth
        let kv_cache_type = KvCacheType::from_env();
        let kv_capacity = config.max_position_embeddings.min(4096) as u32;
        let mut k_cache = Vec::with_capacity(num_layers);
        let mut v_cache = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let hd = config.layer_head_dim(i);
            assert!(hd % 32 == 0, "head_dim must be a multiple of 32 for quantized KV cache");
            let bytes_per_row = kv_cache_type.bytes_per_row(hd);
            let byte_len = (num_kv_heads * kv_capacity as usize * bytes_per_row) as u64;
            k_cache.push(
                ctx.device
                    .new_buffer(byte_len, MTLResourceOptions::StorageModeShared),
            );
            v_cache.push(
                ctx.device
                    .new_buffer(byte_len, MTLResourceOptions::StorageModeShared),
            );
        }
        let f16_bytes = num_kv_heads * kv_capacity as usize * config.head_dim * 2 + num_kv_heads * kv_capacity as usize * config.global_head_dim * 2;
        let quant_bytes = num_kv_heads * kv_capacity as usize * kv_cache_type.bytes_per_row(config.head_dim) + num_kv_heads * kv_capacity as usize * kv_cache_type.bytes_per_row(config.global_head_dim);
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
            per_layer_ple_bufs.push(ctx.buffer_empty(ple_dim));
        }

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

        let mut model = Gemma4GpuModel {
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
            ple_embed_buf,
            ple_gated_buf,
            ple_normed_buf,
            ple_projected_buf,
            ple_context_proj_buf,
            ple_token_id_buf,
            ple_combined_buf,
            q_normed_buf,
            k_normed_buf,
            k_cache,
            v_cache,
            kv_seq_len: 0,
            kv_capacity,
            kv_cache_type,
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
            ple_decode_scratch: vec![0.0f32; num_layers * ple_dim],
            mega_graph: None,
        };

        model.attach_mega_graph_if_enabled();
        model
    }

    /// Load a Gemma-4 model directly from a `.gguf` file (architecture `gemma4`).
    /// On first load, dequantizes K-quant embeddings and quantizes weights to Q4_0
    /// on the GPU, saving a Q4 cache for instant loads on subsequent runs.
    pub fn load_from_gguf(gguf_path: &str) -> Self {
        let load_start = Instant::now();

        let gguf_path = std::path::Path::new(gguf_path);
        let gguf_dir = gguf_path.parent().unwrap_or(std::path::Path::new("."));
        let cache_path = gguf_dir.join("model.q4cache");
        let embed_cache_path = gguf_dir.join("model.embed.cache");

        // Open GGUF for fast metadata parsing (header only).
        let g = crate::gguf::Gguf::open(gguf_path);
        let arch = g.get_str("general.architecture").unwrap_or("");
        assert_eq!(
            arch, "gemma4",
            "GGUF architecture is '{}', expected 'gemma4'",
            arch
        );

        let config = gemma4_config_from_gguf(&g);

        // Check for existing Q4 cache (instant load).
        if embed_cache_path.exists() {
            match Self::read_weights_magic(&cache_path) {
                Some(m) if m == *b"GQ4H" => {
                    println!("  Found Q4 cache, loading from cache (instant)...");
                    return Self::load_gguf_from_cache(config, &embed_cache_path, &cache_path, load_start);
                }
                Some(m) if m == *b"GQ4G" => {
                    // GQ4G from an earlier GGUF load lacks per-tensor format tags.
                    // Delete and re-generate as GQ4H.
                    println!("  Stale GGUF Q4 cache (GQ4G without format tags). Re-generating...");
                    let _ = std::fs::remove_file(&embed_cache_path);
                    let _ = std::fs::remove_file(&cache_path);
                }
                _ => {}
            }
        }

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

        // --- Embeddings ---
        println!("  Loading embeddings (dequantizing K-quant tables)...");
        let embed_f32 = g.dequant_to_f32("token_embd.weight");
        assert_eq!(
            embed_f32.len(),
            vocab_size * hidden_size,
            "token_embd size mismatch"
        );
        // lm_head is tied to the input embeddings (no separate output.weight tensor).
        let lm_head_buf =
            BufferView::from_buffer(ctx.buffer_from_f32_as_q4(&embed_f32, vocab_size, hidden_size));
        println!(
            "    lm_head (tied, Q4_0 on GPU): {:.1} MB",
            lm_head_buf.length as f64 / 1024.0 / 1024.0
        );
        // bf16 image for the CPU embedding lookup table (re-quantized to Q4_0 inside).
        let embed_bf16 = f32_to_bf16_bytes(&embed_f32);
        drop(embed_f32);
        // PLE embedding table is multi-GB; stream straight to bf16 bytes.
        let ple_bf16 = g.dequant_to_bf16_bytes("per_layer_token_embd.weight");
        assert_eq!(
            ple_bf16.len(),
            vocab_size * ple_total_dim * 2,
            "per_layer_token_embd size mismatch"
        );
        let embed_tables = EmbedTables::from_owned(
            embed_bf16,
            ple_bf16,
            vocab_size,
            ple_total_dim,
            hidden_size,
        );

        // --- Shared norms / projections ---
        let final_norm_weight =
            BufferView::from_buffer(ctx.buffer_from_slice(&g.dequant_to_f32("output_norm.weight")));
        let per_layer_projection_norm_weight = BufferView::from_buffer(
            ctx.buffer_from_slice(&g.dequant_to_f32("per_layer_proj_norm.weight")),
        );
        let per_layer_model_proj_f32 = g.dequant_to_f32("per_layer_model_proj.weight");
        let per_layer_model_projection_weight = BufferView::from_buffer(ctx.buffer_from_f32_as_q4(
            &per_layer_model_proj_f32,
            ple_total_dim,
            hidden_size,
        ));
        drop(per_layer_model_proj_f32);

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
            let kv_out = num_kv_heads * head_dim;
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
                        ctx.buffer_from_bytes(g.tensor_raw(&name)),
                    )
                    .with_format(weight_fmt::Q4_K),
                    ggml_type::Q6_K if q6k_to_q4 => {
                        let data = g.dequant_to_f32(&name);
                        BufferView::from_buffer(ctx.buffer_from_f32_as_q4(&data, rows, cols))
                    }
                    ggml_type::Q6_K => BufferView::from_buffer(
                        ctx.buffer_from_bytes(g.tensor_raw(&name)),
                    )
                    .with_format(weight_fmt::Q6_K),
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
            let v_proj = qw(p("attn_v.weight"), kv_out, hidden_size);
            let o_proj = qw(p("attn_output.weight"), hidden_size, q_out);
            let gate_proj = qw(p("ffn_gate.weight"), layer_inter, hidden_size);
            let up_proj = qw(p("ffn_up.weight"), layer_inter, hidden_size);
            let down_proj = qw(p("ffn_down.weight"), hidden_size, layer_inter);
            let per_layer_input_gate_weight = qw(p("inp_gate.weight"), ple_dim, hidden_size);
            let per_layer_projection_weight = qw(p("proj.weight"), hidden_size, ple_dim);

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

                gate_proj,
                up_proj,
                gate_up_proj: BufferView::from_buffer(ctx.buffer_empty(1)),
                down_proj,

                input_layernorm_weight: f32buf(p("attn_norm.weight")),
                post_attention_layernorm_weight: f32buf(p("post_attention_norm.weight")),
                pre_feedforward_layernorm_weight: f32buf(p("ffn_norm.weight")),
                post_feedforward_layernorm_weight: f32buf(p("post_ffw_norm.weight")),
                post_per_layer_input_norm_weight: f32buf(p("post_norm.weight")),

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

        // Save cache for future loads
        println!("  Saving Q4 cache for fast future loads...");
        model.save_gguf_cache(&cache_path);

        model
    }

    /// Fast path: load GGUF model from previously-saved Q4 cache files.
    fn load_gguf_from_cache(
        config: Gemma4TextConfig,
        embed_cache_path: &Path,
        weights_cache_path: &Path,
        load_start: Instant,
    ) -> Self {
        let ctx = MetalContext::new();
        let device = &ctx.device;

        // --- Embed tables: mmap from embed cache ---
        let embed_file = fs::File::open(embed_cache_path).expect("Failed to open embed cache");
        let embed_mmap =
            unsafe { Mmap::map(&embed_file).expect("Failed to mmap embed cache") };
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
        let ple_cols = config.num_hidden_layers * config.hidden_size_per_layer_input;
        let embed_tables = EmbedTables::from_mmap(
            embed_mmap,
            embed_offset,
            embed_len as usize,
            ple_offset,
            ple_len as usize,
            ple_cols,
            config.vocab_size,
        );

        // --- Weights: read directly into a single GPU buffer (aligned layout) ---
        let mut weights_file =
            fs::File::open(weights_cache_path).expect("Failed to open weights cache");
        let file_len = weights_file
            .metadata()
            .expect("weights metadata")
            .len();
        let weights_bytes = file_len.saturating_sub(4) as u64;

        use std::io::Read;
        let mut magic = [0u8; 4];
        weights_file
            .read_exact(&mut magic)
            .expect("read weights cache magic");
        let has_tensor_formats = magic == *b"GQ4H";
        assert!(
            magic == *b"GQ4G" || magic == *b"GQ4H",
            "Invalid weights cache magic ({:?}), expected GQ4G or GQ4H",
            magic
        );
        let copy_start = Instant::now();
        let weights_buf =
            MetalContext::buffer_read_from_file(&device, &mut weights_file, 4, weights_bytes);
        println!(
            "  Copying weights to GPU: {:.1} MB ({:.2}s)...",
            weights_bytes as f64 / 1024.0 / 1024.0,
            copy_start.elapsed().as_secs_f64()
        );
        let section = unsafe {
            std::slice::from_raw_parts(weights_buf.contents() as *const u8, weights_bytes as usize)
        };
        let mut section_offset = 0;
        let (mut lm_head_buf, mut per_layer_model_projection_weight,
             mut final_norm_weight, mut per_layer_projection_norm_weight,
             mut layers) =
            Self::load_weight_sections(
                section,
                &mut section_offset,
                &device,
                config.num_hidden_layers,
                config.hidden_size,
                true,
                Some(&weights_buf),
                0,
            );

        // Restore per-tensor format tags from the format table
        // (appended after all layer data by save_gguf_cache).
        if has_tensor_formats {
            let num_tensors = 4 + 16 * config.num_hidden_layers;
            let fmt_bytes = &section[section_offset..section_offset + num_tensors];
            let mut fi = 0;

            let mut global_views = [
                &mut lm_head_buf,
                &mut per_layer_model_projection_weight,
                &mut final_norm_weight,
                &mut per_layer_projection_norm_weight,
            ];
            for v in &mut *global_views.as_mut_slice() {
                v.format = fmt_bytes[fi];
                fi += 1;
            }

            for layer in &mut layers {
                for v in [
                    &mut layer.q_proj, &mut layer.k_proj, &mut layer.v_proj,
                    &mut layer.o_proj, &mut layer.gate_proj, &mut layer.up_proj,
                    &mut layer.down_proj,
                    &mut layer.input_layernorm_weight,
                    &mut layer.post_attention_layernorm_weight,
                    &mut layer.pre_feedforward_layernorm_weight,
                    &mut layer.post_feedforward_layernorm_weight,
                    &mut layer.post_per_layer_input_norm_weight,
                    &mut layer.per_layer_input_gate_weight,
                    &mut layer.per_layer_projection_weight,
                    &mut layer.q_norm_weight, &mut layer.k_norm_weight,
                ] {
                    v.format = fmt_bytes[fi];
                    fi += 1;
                }
            }
            debug_assert_eq!(fi, num_tensors);
        }

        Self::finish_cache_load(
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
            "GGUF Q4 cache",
        )
    }

    fn decode_rope_byte_offset(&self, layer_idx: usize) -> u64 {
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

    /// Save GGUF cache with per-tensor format tags (magic `GQ4H`).
    /// The format byte for each tensor is appended after all layer data
    /// so that the existing `load_weight_sections` can read the tensor
    /// payloads unchanged; the format table is then read separately.
    fn save_gguf_cache(&self, path: &Path) {
        let embed_path = path
            .parent()
            .expect("cache path has no parent")
            .join("model.embed.cache");
        self.save_embed_cache(&embed_path);

        use std::io::{Seek, Write};
        let mut file = fs::File::create(path).expect("Failed to create weights cache");
        file.write_all(b"GQ4H").unwrap();

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

        let collect_formats = |views: &[&BufferView]| -> Vec<u8> {
            views.iter().map(|v| v.format).collect()
        };

        // Global tensors (4)
        let mut formats = Vec::new();
        let global_views = [
            &self.lm_head_buf,
            &self.per_layer_model_projection_weight,
            &self.final_norm_weight,
            &self.per_layer_projection_norm_weight,
        ];
        for &v in &global_views {
            save_view(&mut file, v);
        }
        formats.extend(collect_formats(&global_views));

        // Per-layer weights
        let num_layers = self.layers.len() as u32;
        file.write_all(&num_layers.to_le_bytes()).unwrap();
        pad_weights_file_to_section_align(&mut file);
        for layer in &self.layers {
            let layer_views = [
                &layer.q_proj, &layer.k_proj, &layer.v_proj,
                &layer.o_proj, &layer.gate_proj, &layer.up_proj,
                &layer.down_proj,
                &layer.input_layernorm_weight,
                &layer.post_attention_layernorm_weight,
                &layer.pre_feedforward_layernorm_weight,
                &layer.post_feedforward_layernorm_weight,
                &layer.post_per_layer_input_norm_weight,
                &layer.per_layer_input_gate_weight,
                &layer.per_layer_projection_weight,
                &layer.q_norm_weight, &layer.k_norm_weight,
            ];
            for &v in &layer_views {
                save_view(&mut file, v);
            }
            formats.extend(collect_formats(&layer_views));

            file.write_all(&layer.layer_scalar.to_le_bytes()).unwrap();
            file.write_all(&(layer.is_full_attention as u8).to_le_bytes()).unwrap();
            file.write_all(&(layer.has_kv as u8).to_le_bytes()).unwrap();
            file.write_all(&[layer.weight_format.to_u8()]).unwrap();
            file.write_all(&(layer.kv_source_layer as u32).to_le_bytes()).unwrap();
            file.write_all(&(layer.head_dim as u32).to_le_bytes()).unwrap();
            file.write_all(&(layer.q_out_dim as u32).to_le_bytes()).unwrap();
            file.write_all(&(layer.kv_out_dim as u32).to_le_bytes()).unwrap();
            pad_weights_file_to_section_align(&mut file);
        }

        // Append format table (all format bytes, in tensor order)
        file.write_all(&formats).unwrap();

        println!(
            "  Weights cache saved: {:.1} MB",
            file.metadata().map(|m| m.len()).unwrap_or(0) as f64 / 1024.0 / 1024.0
        );
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
                gate_proj,
                up_proj,
                gate_up_proj: BufferView::from_buffer(
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
        let layers = Self::pack_layers_gate_up(&ctx, layers, config.hidden_size as u32);
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

        let kv_cache_type = KvCacheType::from_env();
        let kv_capacity = config.max_position_embeddings.min(4096) as u32;
        let mut k_cache = Vec::with_capacity(num_layers);
        let mut v_cache = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let hd = config.layer_head_dim(i);
            assert!(hd % 32 == 0, "head_dim must be a multiple of 32 for quantized KV cache");
            let bytes_per_row = kv_cache_type.bytes_per_row(hd);
            let byte_len = (num_kv_heads * kv_capacity as usize * bytes_per_row) as u64;
            k_cache.push(
                ctx.device
                    .new_buffer(byte_len, MTLResourceOptions::StorageModeShared),
            );
            v_cache.push(
                ctx.device
                    .new_buffer(byte_len, MTLResourceOptions::StorageModeShared),
            );
        }
        let f16_bytes = num_kv_heads * kv_capacity as usize * config.head_dim * 2
            + num_kv_heads * kv_capacity as usize * config.global_head_dim * 2;
        let quant_bytes = num_kv_heads * kv_capacity as usize * kv_cache_type.bytes_per_row(config.head_dim)
            + num_kv_heads * kv_capacity as usize * kv_cache_type.bytes_per_row(config.global_head_dim);
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
            per_layer_ple_bufs.push(ctx.buffer_empty(ple_dim));
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

        let mut model = Gemma4GpuModel {
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
            ple_embed_buf,
            ple_gated_buf,
            ple_normed_buf,
            ple_projected_buf,
            ple_context_proj_buf,
            ple_token_id_buf,
            ple_combined_buf,
            q_normed_buf,
            k_normed_buf,
            k_cache,
            v_cache,
            kv_seq_len: 0,
            kv_capacity,
            kv_cache_type,
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
            ple_decode_scratch: vec![0.0f32; num_layers * ple_dim],
            mega_graph: None,
        };
        model.attach_mega_graph_if_enabled();
        model
    }

    fn attach_mega_graph_if_enabled(&mut self) {
        if !mega_kernel_enabled() {
            return;
        }
        // The mega decode graph is hard-wired to Q4_0 block layout; K-quant
        // (Q4_K_M) layers are served by the per-tensor matvec path instead.
        if self.layers.iter().any(|l| l.weight_format.is_kquant()) {
            eprintln!("  MEGA_KERNEL ignored: K-quant weights use the per-tensor matvec path");
            return;
        }
        match MegaDecodeGraph::build(self) {
            Ok(graph) => {
                println!(
                    "  Mega decode enabled ({} ops, single command buffer)",
                    graph.ops.len()
                );
                self.mega_graph = Some(graph);
            }
            Err(e) => {
                eprintln!("  MEGA_KERNEL=1 but mega graph build failed: {e}");
            }
        }
    }

    /// Build interleaved gate∥up Q4 buffers for decode MLP (uzu-style packing).
    fn pack_layers_gate_up(
        ctx: &MetalContext,
        mut layers: Vec<Gemma4GpuLayer>,
        k: u32,
    ) -> Vec<Gemma4GpuLayer> {
        if !crate::gpu::packed_mlp_gate_up_enabled() {
            for layer in layers.iter_mut() {
                layer.gate_up_proj = BufferView::from_buffer(ctx.buffer_empty(1));
            }
            return layers;
        }
        let pack_start = std::time::Instant::now();
        for layer in layers.iter_mut() {
            if Self::use_packed_mlp_gate_up(layer) {
                layer.gate_up_proj = ctx.pack_gate_up_interleaved_q4(
                    &layer.gate_proj,
                    &layer.up_proj,
                    layer.intermediate_size as u32,
                    k,
                );
            } else {
                layer.gate_up_proj = BufferView::from_buffer(ctx.buffer_empty(1));
            }
        }
        println!(
            "  Packed interleaved gate∥up Q4 weights in {:.2}s",
            pack_start.elapsed().as_secs_f64()
        );
        layers
    }

    fn use_packed_mlp_gate_up(layer: &Gemma4GpuLayer) -> bool {
        crate::gpu::packed_mlp_gate_up_enabled() && layer.weight_format == WeightFormat::Q4_0
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
        self.embed_tables.decode_ple_into(
            token_id,
            ple_total_dim,
            ple_dim,
            &mut self.ple_decode_scratch,
        );
        MetalContext::write_buffer(&self.ple_token_id_buf, &self.ple_decode_scratch);

        let kv_seq = self.kv_seq_len;
        let pos = self.total_tokens as f32;

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

        if self.mega_graph.is_some()
            && !__pp
            && !__ablate.active()
            && __gpu_prof.is_none()
            && !matches!(mode, DecodeMode::Advance)
        {
            let cap = self.config.final_logit_softcapping;
            let sliding = self.config.sliding_window as u32;
            let sample = match mode {
                DecodeMode::Sample(t, mp, s) => Some((t, mp, s)),
                DecodeMode::Logits => None,
                DecodeMode::Advance => unreachable!(),
            };
            let scratch = MegaScratchBuffers {
                hidden: &self.hidden_buf,
                normed: &self.normed_buf,
                q: &self.q_buf,
                k: &self.k_buf,
                v: &self.v_buf,
                q_normed: &self.q_normed_buf,
                k_normed: &self.k_normed_buf,
                gate: &self.gate_buf,
                attn_out: &self.attn_out_buf,
                o_out: &self.o_out_buf,
                up: &self.up_buf,
                gelu: &self.gelu_buf,
                down: &self.down_buf,
                ple_ctx: &self.ple_context_proj_buf,
                ple_tmp: &self.ple_combined_buf,
                ple_tok: &self.ple_token_id_buf,
                logits: &self.logits_buf,
                sample_out: &self.sample_out_buf,
            };
            let layers = &self.layers;
            let mega = self.mega_graph.as_mut().unwrap();
            mega.encode(
                &self.ctx,
                &encoder,
                layers,
                sliding,
                &scratch,
                &self.k_cache,
                &self.v_cache,
                &self.decode_rope_cos_packed,
                &self.decode_rope_sin_packed,
                self.rope_max_head_dim,
                kv_seq,
                sample,
            );
            encoder.end_encoding();
            let __t_encode = std::time::Instant::now();
            cmd.commit();
            cmd.wait_until_completed();
            let __t_gpu = std::time::Instant::now();
            let output = match mode {
                DecodeMode::Sample(..) => {
                    let tok = MetalContext::read_u32(&self.sample_out_buf) as usize;
                    if __profile {
                        Self::profile_record(__t0, __t_prep, __t_encode, __t_gpu);
                    }
                    DecodeOutput::Token(tok)
                }
                DecodeMode::Logits => {
                    let mut logits = MetalContext::read_buffer(&self.logits_buf, vocab_size);
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
            self.total_tokens += 1;
            self.kv_seq_len += 1;
            return output;
        }

        // ─── PLE pre-pass ───
        // Produces the per-layer PLE inputs contiguously in ple_context_proj_buf;
        // each layer reads its slice directly (byte offset = layer_idx * ple_dim
        // * 4), so the previous 42 per-layer copy-out dispatches are gone.
        if !__ablate.skip_ple() {
            // Step 2a: context_proj = per_layer_model_projection @ embed
            self.ctx.encode_matvec_q4_view(
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

        let metal_n_cb = if !__pp && !__ablate.active() && __gpu_prof.is_none() {
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
                && !crate::gpu::attention_use_ggml_for_layer(layer.has_kv)
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
                        if !(self.ctx.use_flash_attention
                            && crate::gpu::fused_kv_attention_enabled())
                        {
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
                    } else if crate::gpu::attention_use_ggml_for_layer(layer.has_kv)
                        && self.ctx.use_flash_attention
                    {
                        self.ctx.encode_attention_ggml_q4_0(
                            encoder,
                            &self.q_normed_buf,
                            &self.k_cache[layer.kv_source_layer],
                            &self.v_cache[layer.kv_source_layer],
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
            }

            // O projection (Q4 on middle layers, f16 on sensitive layers)
            self.ctx.encode_matvec_auto_view(
                encoder,
                &layer.o_proj,
                &self.attn_out_buf,
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
        // Logits via tied embeddings (Q4 lm_head): logits = lm_head @ normed
        self.ctx.encode_matvec_q4_view(
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

    pub fn reset_legacy_state(&mut self) {
        self.kv_seq_len = 0;
        self.total_tokens = 0;
    }

    pub fn create_kv_pool(&self, num_slots: usize, max_seq_len: u32) -> KvCachePool {
        let max_seq_len = max_seq_len.min(self.config.max_position_embeddings as u32);
        KvCachePool::new(&self.ctx, &self.config, num_slots, max_seq_len, self.kv_cache_type)
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

            let ple_out_offset = pos * ple_total_dim;
            self.embed_tables.decode_ple_into(
                token_id,
                ple_total_dim,
                ple_dim,
                &mut ple_token_identity[ple_out_offset..ple_out_offset + ple_total_dim],
            );
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

    fn encode_parallel_prefill_ple_context(&mut self, seq_len: usize) -> Result<(), String> {
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

        let cmd = self.ctx.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        let total_ple = (seq_len * ple_total_dim) as u32;

        self.ctx.encode_projection_q4_batch_view(
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
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = layer.head_dim;
        let q_out = layer.q_out_dim;
        let kv_out = layer.kv_out_dim;
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

        if layer.weight_format == WeightFormat::F16 {
            self.ctx.encode_projection_f16_batch_view(
                encoder,
                &layer.q_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.q_buf,
                q_out as u32,
                hidden_size as u32,
                seq_len as u32,
            );
            if layer.has_kv {
                self.ctx.encode_projection_f16_batch_view(
                    encoder,
                    &layer.k_proj,
                    &self.prefill_scratch.normed_buf,
                    &self.prefill_scratch.k_buf,
                    kv_out as u32,
                    hidden_size as u32,
                    seq_len as u32,
                );
                self.ctx.encode_projection_f16_batch_view(
                    encoder,
                    &layer.v_proj,
                    &self.prefill_scratch.normed_buf,
                    &self.prefill_scratch.v_buf,
                    kv_out as u32,
                    hidden_size as u32,
                    seq_len as u32,
                );
            }
        } else {
            // Q4_0 and K-quant (Q4_K_M) share the batched projection entry; the
            // K-quant guard inside picks the right kernel per tensor.
            self.ctx.encode_projection_q4_batch_view(
                encoder,
                &layer.q_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.q_buf,
                q_out as u32,
                hidden_size as u32,
                seq_len as u32,
            );
            if layer.has_kv {
                self.ctx.encode_projection_q4_batch_view(
                    encoder,
                    &layer.k_proj,
                    &self.prefill_scratch.normed_buf,
                    &self.prefill_scratch.k_buf,
                    kv_out as u32,
                    hidden_size as u32,
                    seq_len as u32,
                );
                self.ctx.encode_projection_q4_batch_view(
                    encoder,
                    &layer.v_proj,
                    &self.prefill_scratch.normed_buf,
                    &self.prefill_scratch.v_buf,
                    kv_out as u32,
                    hidden_size as u32,
                    seq_len as u32,
                );
            }
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

        true
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
        let intermediate_size = self.config.max_intermediate_size();
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
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
                self.ctx.encode_attention_causal_q4_0(
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
        }

        self.ctx.encode_copy(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.residual_buf,
            total_hidden,
        );
        self.ctx.encode_transpose_hsd(
            encoder,
            &self.prefill_scratch.attn_out_buf,
            &self.prefill_scratch.q_normed_buf,
            seq_len as u32,
            num_heads as u32,
            head_dim as u32,
        );
        self.ctx.encode_projection_auto_batch_view(
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
        if layer.weight_format == WeightFormat::F16 {
            self.ctx.encode_projection_f16_batch_view(
                encoder,
                &layer.gate_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.gate_buf,
                intermediate_size as u32,
                hidden_size as u32,
                seq_len as u32,
            );
            self.ctx.encode_projection_f16_batch_view(
                encoder,
                &layer.up_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.up_buf,
                intermediate_size as u32,
                hidden_size as u32,
                seq_len as u32,
            );
        } else {
            self.ctx.encode_projection_q4_batch_view(
                encoder,
                &layer.gate_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.gate_buf,
                intermediate_size as u32,
                hidden_size as u32,
                seq_len as u32,
            );
            self.ctx.encode_projection_q4_batch_view(
                encoder,
                &layer.up_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.up_buf,
                intermediate_size as u32,
                hidden_size as u32,
                seq_len as u32,
            );
        }
        self.ctx.encode_gelu_mul(
            encoder,
            &self.prefill_scratch.gate_buf,
            &self.prefill_scratch.up_buf,
            &self.prefill_scratch.gelu_buf,
            total_intermediate,
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
            self.ctx.encode_projection_q4_batch_view(
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
        self.ctx.encode_projection_auto_batch_view(
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
        self.ctx.encode_projection_auto_batch_view(
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
        let intermediate_size = self.config.max_intermediate_size();
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
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
            }
        }

        self.ctx.encode_copy(
            encoder,
            &self.prefill_scratch.hidden_buf,
            &self.prefill_scratch.residual_buf,
            total_hidden,
        );
        self.ctx.encode_transpose_hsd(
            encoder,
            &self.prefill_scratch.attn_out_buf,
            &self.prefill_scratch.q_normed_buf,
            total_seq_len as u32,
            num_heads as u32,
            head_dim as u32,
        );
        self.ctx.encode_projection_auto_batch_view(
            encoder,
            &layer.o_proj,
            &self.prefill_scratch.q_normed_buf,
            &self.prefill_scratch.o_out_buf,
            hidden_size as u32,
            q_out as u32,
            total_seq_len as u32,
        );
        self.ctx.encode_rmsnorm_batch_view(
            encoder,
            &self.prefill_scratch.o_out_buf,
            &layer.post_attention_layernorm_weight,
            &self.prefill_scratch.normed_buf,
            hidden_size as u32,
            eps,
            total_seq_len as u32,
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
            total_seq_len as u32,
        );
        if layer.weight_format == WeightFormat::F16 {
            self.ctx.encode_projection_f16_batch_view(
                encoder,
                &layer.gate_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.gate_buf,
                intermediate_size as u32,
                hidden_size as u32,
                total_seq_len as u32,
            );
            self.ctx.encode_projection_f16_batch_view(
                encoder,
                &layer.up_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.up_buf,
                intermediate_size as u32,
                hidden_size as u32,
                total_seq_len as u32,
            );
        } else {
            self.ctx.encode_projection_q4_batch_view(
                encoder,
                &layer.gate_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.gate_buf,
                intermediate_size as u32,
                hidden_size as u32,
                total_seq_len as u32,
            );
            self.ctx.encode_projection_q4_batch_view(
                encoder,
                &layer.up_proj,
                &self.prefill_scratch.normed_buf,
                &self.prefill_scratch.up_buf,
                intermediate_size as u32,
                hidden_size as u32,
                total_seq_len as u32,
            );
        }
        self.ctx.encode_gelu_mul(
            encoder,
            &self.prefill_scratch.gate_buf,
            &self.prefill_scratch.up_buf,
            &self.prefill_scratch.gelu_buf,
            total_intermediate,
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
            self.ctx.encode_projection_q4_batch_view(
                encoder,
                &layer.down_proj,
                &self.prefill_scratch.gelu_buf,
                &self.prefill_scratch.down_buf,
                hidden_size as u32,
                intermediate_size as u32,
                total_seq_len as u32,
            );
        }
        self.ctx.encode_rmsnorm_batch_view(
            encoder,
            &self.prefill_scratch.down_buf,
            &layer.post_feedforward_layernorm_weight,
            &self.prefill_scratch.normed_buf,
            hidden_size as u32,
            eps,
            total_seq_len as u32,
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
        self.ctx.encode_projection_auto_batch_view(
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
        self.ctx.encode_projection_auto_batch_view(
            encoder,
            &layer.per_layer_projection_weight,
            &self.prefill_scratch.up_buf,
            &self.prefill_scratch.o_out_buf,
            hidden_size as u32,
            ple_dim as u32,
            total_seq_len as u32,
        );
        self.ctx.encode_rmsnorm_batch_view(
            encoder,
            &self.prefill_scratch.o_out_buf,
            &layer.post_per_layer_input_norm_weight,
            &self.prefill_scratch.normed_buf,
            hidden_size as u32,
            eps,
            total_seq_len as u32,
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

    fn forward_prefill_chunk_parallel_with_kv_slot(
        &mut self,
        token_ids: &[usize],
        kv_pool: &mut KvCachePool,
        slot: KvSlot,
        start_pos: usize,
    ) -> Result<Vec<f32>, String> {
        let seq_len = token_ids.len();
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let eps = self.config.rms_norm_eps as f32;

        self.encode_parallel_prefill_ple_context(seq_len)?;

        // ═══ KEY OPTIMIZATION: Encode ALL 42 layers into a SINGLE command buffer ═══
        // Previously each layer did its own commit+wait (42 GPU round-trips).
        // Now we encode the full layer stack + final norm + lm_head in one shot.
        let cmd = self.ctx.queue.new_command_buffer();

        for layer_idx in 0..self.layers.len() {
            let encoder = cmd.new_compute_command_encoder();
            self.encode_parallel_prefill_attention_inputs(encoder, layer_idx, seq_len)?;

            let layer = self
                .layers
                .get(layer_idx)
                .ok_or_else(|| format!("invalid layer index {}", layer_idx))?;
            let intermediate_size = self.config.max_intermediate_size();
            let num_heads = self.config.num_attention_heads;
            let num_kv_heads = self.config.num_key_value_heads;
            let num_kv_groups = (num_heads / num_kv_heads) as u32;
            let ple_dim = self.config.hidden_size_per_layer_input;
            let head_dim = layer.head_dim;
            let q_out = layer.q_out_dim;
            let total_hidden = (seq_len * hidden_size) as u32;
            let total_intermediate = (seq_len * intermediate_size) as u32;
            let scale = 1.0f32;
            let attention_window = if layer.is_full_attention {
                0
            } else {
                self.config.sliding_window as u32
            };

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
                    self.ctx.encode_attention_causal_q4_0(
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
            }

            self.ctx.encode_copy(
                encoder,
                &self.prefill_scratch.hidden_buf,
                &self.prefill_scratch.residual_buf,
                total_hidden,
            );
            self.ctx.encode_transpose_hsd(
                encoder,
                &self.prefill_scratch.attn_out_buf,
                &self.prefill_scratch.q_normed_buf,
                seq_len as u32,
                num_heads as u32,
                head_dim as u32,
            );
            self.ctx.encode_projection_auto_batch_view(
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
        if layer.weight_format == WeightFormat::F16 {
            self.ctx.encode_projection_f16_batch_view(
                encoder,
                &layer.gate_proj,
                    &self.prefill_scratch.normed_buf,
                    &self.prefill_scratch.gate_buf,
                    intermediate_size as u32,
                    hidden_size as u32,
                    seq_len as u32,
                );
                self.ctx.encode_projection_f16_batch_view(
                    encoder,
                    &layer.up_proj,
                    &self.prefill_scratch.normed_buf,
                    &self.prefill_scratch.up_buf,
                    intermediate_size as u32,
                    hidden_size as u32,
                    seq_len as u32,
                );
            } else {
                self.ctx.encode_projection_q4_batch_view(
                    encoder,
                    &layer.gate_proj,
                    &self.prefill_scratch.normed_buf,
                    &self.prefill_scratch.gate_buf,
                    intermediate_size as u32,
                    hidden_size as u32,
                    seq_len as u32,
                );
                self.ctx.encode_projection_q4_batch_view(
                    encoder,
                    &layer.up_proj,
                    &self.prefill_scratch.normed_buf,
                    &self.prefill_scratch.up_buf,
                    intermediate_size as u32,
                    hidden_size as u32,
                    seq_len as u32,
                );
            }
            self.ctx.encode_gelu_mul(
                encoder,
                &self.prefill_scratch.gate_buf,
                &self.prefill_scratch.up_buf,
                &self.prefill_scratch.gelu_buf,
                total_intermediate,
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
                self.ctx.encode_projection_q4_batch_view(
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
            self.ctx.encode_projection_auto_batch_view(
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
            self.ctx.encode_projection_auto_batch_view(
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
        }

        // Final norm + lm_head (still in the same command buffer)
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
        let last_offsets = self.prefill_row_offsets(seq_len - 1);
        self.ctx.encode_matvec_q4_at_view(
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

        // ═══ SINGLE commit + wait for ALL 42 layers + final head ═══
        cmd.commit();
        cmd.wait_until_completed();

        let mut logits = MetalContext::read_buffer(&self.prefill_scratch.logits_buf, vocab_size);
        let cap = self.config.final_logit_softcapping;
        for logit in &mut logits {
            let x = (*logit / cap).clamp(-10.0, 10.0);
            *logit = cap * x.tanh();
        }

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
            for batch_idx in 0..batch_size {
                let offsets = self.decode_batch_row_offsets(batch_idx);
                self.ctx.encode_matvec_q4_at_view(
                    encoder,
                    &self.per_layer_model_projection_weight,
                    &self.decode_batch_scratch.hidden_buf,
                    offsets.hidden,
                    &self.decode_batch_scratch.ple_context_proj_buf,
                    offsets.ple_row,
                    ple_total_dim as u32,
                    hidden_size as u32,
                );
                self.ctx.encode_vec_scale_at(
                    encoder,
                    &self.decode_batch_scratch.ple_context_proj_buf,
                    offsets.ple_row,
                    &self.decode_batch_scratch.ple_combined_buf,
                    offsets.ple_row,
                    ple_total_dim as u32,
                    context_proj_scale,
                );
                self.ctx.encode_rmsnorm_per_head_at_view(
                    encoder,
                    &self.decode_batch_scratch.ple_combined_buf,
                    offsets.ple_row,
                    &self.per_layer_projection_norm_weight,
                    &self.decode_batch_scratch.ple_context_proj_buf,
                    offsets.ple_row,
                    num_layers as u32,
                    ple_dim as u32,
                    eps,
                );
                self.ctx.encode_vec_add_at(
                    encoder,
                    &self.decode_batch_scratch.ple_context_proj_buf,
                    offsets.ple_row,
                    &self.decode_batch_scratch.ple_token_id_buf,
                    offsets.ple_row,
                    &self.decode_batch_scratch.ple_combined_buf,
                    offsets.ple_row,
                    ple_total_dim as u32,
                );
                self.ctx.encode_vec_scale_at(
                    encoder,
                    &self.decode_batch_scratch.ple_combined_buf,
                    offsets.ple_row,
                    &self.decode_batch_scratch.ple_context_proj_buf,
                    offsets.ple_row,
                    ple_total_dim as u32,
                    ple_input_scale,
                );
            }
            encoder.end_encoding();
        }

        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            let intermediate_size = layer.intermediate_size;
            let head_dim = layer.head_dim;
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
                            if !(self.ctx.use_flash_attention
                                && crate::gpu::fused_kv_attention_enabled())
                            {
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
                        if crate::gpu::attention_use_ggml_for_layer(layer.has_kv)
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
                self.ctx.encode_matvec_q4_at_view(
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
            encoder.end_encoding();
        }

        cmd.commit();
        cmd.wait_until_completed();

        let mut logits_batch = MetalContext::read_buffer(
            &self.decode_batch_scratch.logits_buf,
            batch_size * vocab_size,
        );
        let cap = self.config.final_logit_softcapping;
        let mut outputs = Vec::with_capacity(batch_size);
        for batch_idx in 0..batch_size {
            let start = batch_idx * vocab_size;
            let end = start + vocab_size;
            for logit in &mut logits_batch[start..end] {
                let x = (*logit / cap).clamp(-10.0, 10.0);
                *logit = cap * x.tanh();
            }
            outputs.push(logits_batch[start..end].to_vec());
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
            return match self.forward_decode_batch_encoded_with_kv_slots(
                inputs,
                &slot_views,
                kv_pool,
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
            return inputs
                .iter()
                .map(|&(slot, token_ids)| {
                    self.forward_prefill_chunk_with_kv_slot(token_ids, kv_pool, slot)
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
            self.ctx.encode_matvec_q4_at_view(
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

        let mut logits = Vec::new();
        let chunk_size = self.max_parallel_prefill_seq().max(1);

        for chunk in token_ids.chunks(chunk_size) {
            logits = self.forward_prefill_chunk_with_kv_slot(chunk, kv_pool, slot)?;
        }

        Ok(logits)
    }

    pub fn forward_prefill_chunk_with_kv_slot(
        &mut self,
        token_ids: &[usize],
        kv_pool: &mut KvCachePool,
        slot: KvSlot,
    ) -> Result<Vec<f32>, String> {
        self.prepare_parallel_prefill_inputs(token_ids)?;
        let start_pos = kv_pool.total_tokens(slot).map_err(|err| err.to_string())?;
        self.prepare_parallel_prefill_rotary(start_pos, token_ids.len())?;

        if self.can_use_parallel_prefill_chunk(start_pos, token_ids.len(), kv_pool) {
            return self
                .forward_prefill_chunk_parallel_with_kv_slot(token_ids, kv_pool, slot, start_pos);
        }

        let mut logits = Vec::new();
        for &tid in token_ids {
            logits = self
                .forward_single_token_with_kv_slot(tid, kv_pool, slot)
                .map_err(|err| err.to_string())?;
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
    let num_key_value_heads = mu("gemma4.attention.head_count_kv");
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
    // rope.dimension_count tells us how many of the head_dim channels are rotated.
    let full_rope_dim =
        g.get_u32("gemma4.rope.dimension_count").unwrap_or(global_head_dim as u32) as f64;
    let sliding_rope_dim =
        g.get_u32("gemma4.rope.dimension_count_swa").unwrap_or(head_dim as u32) as f64;

    let rope_parameters = Some(RopeParameters {
        full_attention: Some(RopeConfig {
            rope_theta: full_theta,
            rope_type: String::new(),
            partial_rotary_factor: full_rope_dim / global_head_dim as f64,
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
