use metal::*;
use std::path::Path;
use std::sync::OnceLock;

/// Per-tensor weight format tag carried on a `BufferView`. Mirrors
/// `gemma4_gpu_model::WeightFormat::to_u8`. Defaults to Q4_0 so existing
/// (non-GGUF) load paths keep their current behavior with no churn.
pub mod weight_fmt {
    pub const F16: u8 = 0;
    pub const Q4_0: u8 = 1;
    pub const Q3_0: u8 = 2;
    pub const Q4_K: u8 = 3;
    pub const Q6_K: u8 = 4;
}

/// A sub-range view into a Metal buffer (offset applied at kernel bind time).
#[derive(Clone)]
pub struct BufferView {
    pub buffer: Buffer,
    pub offset: u64,
    pub length: u64,
    /// Quantization layout of the bytes in this view (see `weight_fmt`).
    pub format: u8,
}

impl BufferView {
    pub fn from_buffer(buffer: Buffer) -> Self {
        let length = buffer.length();
        Self {
            buffer,
            offset: 0,
            length,
            format: weight_fmt::Q4_0,
        }
    }

    pub fn subrange(buffer: &Buffer, offset: u64, length: u64) -> Self {
        Self {
            buffer: buffer.clone(),
            offset,
            length,
            format: weight_fmt::Q4_0,
        }
    }

    /// Tag this view with a quantization format (builder style).
    pub fn with_format(mut self, format: u8) -> Self {
        self.format = format;
        self
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self.buffer.contents() as *const u8).add(self.offset as usize),
                self.length as usize,
            )
        }
    }
}

/// True when a `[m, k]` weight buffer holds Q4_0 blocks (18 bytes / 32 weights).
/// K-quant buffers (Q4_K is byte-for-byte the same size as Q4_0) must NOT be
/// mistaken for Q4_0 by the Q4_0-only fused kernels, so the format tag wins.
pub fn weight_buf_is_kquant(view: &BufferView) -> bool {
    matches!(view.format, weight_fmt::Q4_K | weight_fmt::Q6_K)
}

pub fn weight_buf_is_q4(view: &BufferView, m: u32, k: u32) -> bool {
    if weight_buf_is_kquant(view) {
        return false;
    }
    let q4_bytes = (m as u64) * (k as u64 / 32) * 18;
    // f16 would be m*k*2 — well above q4_bytes; allow section-alignment padding.
    view.length <= q4_bytes + 256
}

/// True when a `[m, k]` weight buffer holds Q3_0 blocks (14 bytes / 32 weights).
pub fn weight_buf_is_q3(view: &BufferView, m: u32, k: u32) -> bool {
    if weight_buf_is_kquant(view) {
        return false;
    }
    let q3_bytes = (m as u64) * (k as u64 / 32) * 14;
    view.length <= q3_bytes + 256
}

/// Q4 matvec kernel used on the batch-1 decode path. Selectable at runtime via
/// the `MATVEC_KERNEL` env var (`r1` | `r2` | `r4`/`fast` | `splitk`) so the
/// best variant for the model's matvec shapes can be chosen from measurement
/// without a rebuild. Defaults to `Fast` (the previous behavior).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeMatvecKernel {
    /// Per-shape selection (defaults to microbench winners on Apple Silicon).
    Auto,
    R1,
    R2,
    Fast,
    R8,
    SplitK,
    /// llama.cpp / ane-infer style: 4 simdgroups × 32 threads, 2 rows/TG.
    Lc,
    /// ggml-metal `kernel_mul_mv_q4_0_f32` (N_R0=4, N_SG=2, decode default).
    Ggml,
    /// ggml-metal `kernel_mul_mv_ext_q4_0_f32_r1_4` (nxpsg picked by K).
    GgmlExt,
}

impl DecodeMatvecKernel {
    pub fn from_env() -> Self {
        match std::env::var("MATVEC_KERNEL").as_deref() {
            Ok("auto") | Ok("AUTO") => DecodeMatvecKernel::Auto,
            Ok("r1") | Ok("R1") => DecodeMatvecKernel::R1,
            Ok("r2") | Ok("R2") => DecodeMatvecKernel::R2,
            Ok("fast") | Ok("FAST") | Ok("4") => DecodeMatvecKernel::Fast,
            Ok("r8") | Ok("R8") => DecodeMatvecKernel::R8,
            Ok("splitk") | Ok("SplitK") | Ok("SPLITK") => DecodeMatvecKernel::SplitK,
            Ok("lc") | Ok("LC") | Ok("llama") | Ok("LLAMA") => DecodeMatvecKernel::Lc,
            Ok("ggml") | Ok("GGML") => DecodeMatvecKernel::Ggml,
            Ok("ggml-ext") | Ok("GGML-EXT") | Ok("ggml_ext") => DecodeMatvecKernel::GgmlExt,
            _ => DecodeMatvecKernel::Auto,
        }
    }

    /// Pick the kernel for a concrete (M, K) when mode is Auto.
    ///
    /// End-to-end decode on M1 Pro with GGUF Q4 weights: ggml `block_q_n_dot_y`
    /// leads fast/4 (~27 vs ~25 tok/s). Override with MATVEC_KERNEL=fast if needed.
    pub fn pick_for_shape(m: u32, k: u32) -> Self {
        let _ = (m, k);
        DecodeMatvecKernel::Ggml
    }

    pub fn resolve_for_shape(self, m: u32, k: u32) -> Self {
        match self {
            DecodeMatvecKernel::Auto => Self::pick_for_shape(m, k),
            fixed => fixed,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            DecodeMatvecKernel::Auto => "auto",
            DecodeMatvecKernel::R1 => "r1",
            DecodeMatvecKernel::R2 => "r2",
            DecodeMatvecKernel::Fast => "fast",
            DecodeMatvecKernel::R8 => "r8",
            DecodeMatvecKernel::SplitK => "splitk",
            DecodeMatvecKernel::Lc => "lc",
            DecodeMatvecKernel::Ggml => "ggml",
            DecodeMatvecKernel::GgmlExt => "ggml-ext",
        }
    }

    /// Output rows each SIMD group owns for the row-parallel variants. Unused
    /// for SplitK (which uses one threadgroup per row) and Lc (fixed 2 rows/TG).
    pub fn rows_per_sg(self) -> u64 {
        match self {
            DecodeMatvecKernel::Auto => unreachable!("resolve_for_shape before rows_per_sg"),
            DecodeMatvecKernel::R1 => 1,
            DecodeMatvecKernel::R2 => 2,
            DecodeMatvecKernel::Fast => 4,
            DecodeMatvecKernel::R8 => 8,
            DecodeMatvecKernel::SplitK | DecodeMatvecKernel::Lc => 4,
            DecodeMatvecKernel::Ggml | DecodeMatvecKernel::GgmlExt => 4,
        }
    }
}

fn flash_attention_enabled() -> bool {
    !matches!(
        std::env::var("FLASH_ATTN").as_deref(),
        Ok("0") | Ok("false") | Ok("legacy") | Ok("LEGACY")
    )
}

/// Tiled GQA causal prefill (flash_attn_ext-style). Opt in with PREFILL_FLASH_ATTN=1.
pub fn prefill_flash_attn_ext_enabled() -> bool {
    flash_attention_enabled()
        && matches!(
            std::env::var("PREFILL_FLASH_ATTN").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE")
        )
}

/// Tiled flash_attn_ext prefill when q_len ≥ 20 (llama.cpp threshold).
pub fn prefill_use_flash_attn_ext_tiled(q_len: u32, head_dim: u32) -> bool {
    crate::ggml_flash_attn_ext::prefill_use_tiled_ext(q_len, head_dim)
}

pub fn prefill_gpu_rope_enabled() -> bool {
    !matches!(
        std::env::var("PREFILL_GPU_ROPE").as_deref(),
        Ok("0") | Ok("false") | Ok("FALSE")
    )
}

pub fn prefill_timing_enabled() -> bool {
    matches!(
        std::env::var("PREFILL_TIMING").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AttentionKernelMode {
    /// Hybrid by KV length: fused below 128 tokens, ggml MWG at/above 128.
    Auto,
    Ggml,
    Specialized,
    Generic,
}

fn attention_kernel_mode() -> AttentionKernelMode {
    match std::env::var("ATTENTION_KERNEL").as_deref() {
        Ok("ggml") | Ok("GGML") => AttentionKernelMode::Ggml,
        Ok("auto") | Ok("AUTO") => AttentionKernelMode::Auto,
        Ok("specialized") | Ok("SPECIALIZED") => AttentionKernelMode::Specialized,
        Ok("generic") | Ok("GENERIC") => AttentionKernelMode::Generic,
        _ => AttentionKernelMode::Specialized,
    }
}

/// Auto mode hybrid (all layers): fused attention below 128 KV tokens, ggml MWG at/above.
pub fn attention_use_ggml_for_layer_kv(has_kv: bool, kv_seq: u32) -> bool {
    let _ = has_kv;
    match attention_kernel_mode() {
        AttentionKernelMode::Ggml => true,
        AttentionKernelMode::Specialized | AttentionKernelMode::Generic => false,
        AttentionKernelMode::Auto => kv_seq >= 128,
    }
}

pub fn attention_use_ggml_for_layer(has_kv: bool) -> bool {
    attention_use_ggml_for_layer_kv(has_kv, 0)
}

/// KV-owning layers need a separate `encode_kv_append` when attention does not fuse
/// the append (ggml MWG and decomposed flash paths). Hybrid `auto` crosses this at
/// kv_seq ≥ 128 while `fused_kv_attention_enabled()` stays true for fused layers.
pub fn needs_explicit_kv_append(has_kv: bool, effective_kv_seq: u32) -> bool {
    if !has_kv {
        return false;
    }
    if attention_use_ggml_for_layer_kv(has_kv, effective_kv_seq) {
        return true;
    }
    !fused_kv_attention_enabled()
}

/// True when every layer uses ggml FA (ATTENTION_KERNEL=ggml).
pub fn attention_use_ggml() -> bool {
    matches!(attention_kernel_mode(), AttentionKernelMode::Ggml)
}

/// Use the GQA-aware f16 decode attention kernel that processes all query heads
/// sharing a KV head in one threadgroup. Experimental; the simple SIMD-reduction
/// variant currently loses to the tiled flash-decode kernel, so it is opt-in
/// via ATTENTION_GQA_F16=1 until a tiled GQA kernel is implemented.
pub fn attention_gqa_f16_enabled(num_kv_groups: u32) -> bool {
    if num_kv_groups <= 1 {
        return false;
    }
    matches!(
        std::env::var("ATTENTION_GQA_F16").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

/// Use the tiled GQA-aware Q4_0 flash-decode kernel that processes all query
/// heads sharing a KV head in one threadgroup. Experimental; opt-in via
/// ATTENTION_GQA_Q4=1 until it is validated to beat the per-query-head kernel.
pub fn attention_gqa_q4_0_enabled(num_kv_groups: u32) -> bool {
    if num_kv_groups <= 1 {
        return false;
    }
    matches!(
        std::env::var("ATTENTION_GQA_Q4").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

fn attention_q4_hd_specialized() -> bool {
    matches!(
        attention_kernel_mode(),
        AttentionKernelMode::Specialized | AttentionKernelMode::Auto
    )
}

/// Fuse KV Q4_0 append into flash decode attention (default on). Set FUSED_KV_ATTN=0 to disable.
pub fn fused_kv_attention_enabled() -> bool {
    if matches!(attention_kernel_mode(), AttentionKernelMode::Ggml) {
        return false;
    }
    !matches!(
        std::env::var("FUSED_KV_ATTN").as_deref(),
        Ok("0") | Ok("false") | Ok("FALSE")
    )
}

/// Fuse K-norm + RoPE + V-norm + KV append into Q4 flash decode (default on).
/// Requires FUSED_Q_ATTN and FUSED_KV_ATTN. Set FUSED_K_ATTN=0 to use separate K-side dispatches.
pub fn fused_k_attn_enabled() -> bool {
    if !fused_q_attn_enabled() || !fused_kv_attention_enabled() {
        return false;
    }
    !matches!(
        std::env::var("FUSED_K_ATTN").as_deref(),
        Ok("0") | Ok("false") | Ok("FALSE")
    )
}

/// f16 GeLU scratch between gate∥up and down (halves activation bandwidth). Opt-in:
/// ~neutral on M1 Pro e2e; enable with MLP_GELU_F16=1 to experiment.
pub fn mlp_gelu_f16_enabled() -> bool {
    matches!(
        std::env::var("MLP_GELU_F16").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

/// Fuse pre-FF RMSNorm into Q4 gate∥up+GeLU (skip normed scratch). Opt-in:
/// extra inv_rms dispatch per layer regresses ~0.5 tok/s on M1 Pro vs separate rmsnorm.
pub fn fused_rmsnorm_mlp_enabled() -> bool {
    matches!(
        std::env::var("FUSED_RMSNORM_MLP").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

/// Fuse pre-FF RMSNorm into Q4_K gate∥up+GeLU on K-quant decode (default on).
pub fn fused_rmsnorm_mlp_kquant_enabled() -> bool {
    !matches!(
        std::env::var("FUSED_RMSNORM_MLP_KQUANT").as_deref(),
        Ok("0") | Ok("false") | Ok("FALSE")
    )
}

/// Fuse gate+up+GeLU+down MLP in one encoder call (default on).
/// Phase 1: parallel dual_gelu → gelu scratch. Phase 2: down matvec.
/// Set FUSED_MLP_GELU_DOWN=0 to use separate dispatches via FUSED_MLP_PLE.
pub fn fused_mlp_gelu_down_enabled() -> bool {
    !matches!(
        std::env::var("FUSED_MLP_GELU_DOWN").as_deref(),
        Ok("0") | Ok("false") | Ok("FALSE")
    )
}

/// Use two separate ggml-style Q4 matvecs for gate and up (then a separate
/// GeLU multiply) instead of the packed interleaved or dual_gelu kernels.
/// Opt-in; can be faster when the ggml matvec bandwidth outweighs the extra
/// dispatch overhead (MLP_GATE_UP_GGML=1).
pub fn mlp_gate_up_ggml_enabled() -> bool {
    matches!(
        std::env::var("MLP_GATE_UP_GGML").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

/// Fused gate+up+GeLU Q4_0 kernel that writes the gelu scratch directly
/// instead of using separate gate/up scratch buffers + a GeLU multiply
/// dispatch. Enable with MLP_FUSED_GELU_GGML=1 (default on when gate+up ggml
/// path is used).
pub fn mlp_fused_gelu_ggml_enabled() -> bool {
    match std::env::var("MLP_FUSED_GELU_GGML").as_deref() {
        Ok("0") | Ok("false") | Ok("FALSE") => false,
        _ => true,
    }
}

/// Use the lower-register-pressure r2s4 fused gate+up+GeLU kernel. Default on
/// for the fused ggml path; set MLP_FUSED_GELU_GGML_R2S4=0 to use the r4s2
/// layout.
pub fn mlp_fused_gelu_ggml_r2s4_enabled() -> bool {
    match std::env::var("MLP_FUSED_GELU_GGML_R2S4").as_deref() {
        Ok("0") | Ok("false") | Ok("FALSE") => false,
        _ => true,
    }
}

/// Use a single fused gate+up ggml Q4 matvec dispatch instead of two separate
/// dispatches. Shares the x load between gate and up. Experimental: it can
/// hurt bandwidth on some shapes because the two weight streams interleave;
/// enable with MLP_GATE_UP_DUAL=1 to benchmark.
pub fn mlp_gate_up_dual_ggml_enabled() -> bool {
    matches!(
        std::env::var("MLP_GATE_UP_DUAL").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

/// Interleaved gate∥up Q4 weights + single-buffer GeLU matvec (default on).
/// Set PACKED_MLP_GATE_UP=0 to use separate gate/up buffers.
pub fn packed_mlp_gate_up_enabled() -> bool {
    !matches!(
        std::env::var("PACKED_MLP_GATE_UP").as_deref(),
        Ok("0") | Ok("false") | Ok("FALSE")
    )
}

/// Fuse gate+up+GeLU MLP and PLE gate+GeLU decode paths (default on). Set FUSED_MLP_PLE=0 to disable.
pub fn fused_mlp_ple_enabled() -> bool {
    !matches!(
        std::env::var("FUSED_MLP_PLE").as_deref(),
        Ok("0") | Ok("false") | Ok("FALSE")
    )
}

/// Fuse RMSNorm(sublayer)+residual add on post-attn/post-MLP paths (default on).
pub fn fused_rmsnorm_acc_enabled() -> bool {
    !matches!(
        std::env::var("FUSED_RMSNORM_ACC").as_deref(),
        Ok("0") | Ok("false") | Ok("FALSE")
    )
}

/// Number of command buffers per decode token. When >= 2, the layer loop is split
/// so the CPU can encode CB[i+1] while the GPU runs CB[i] (llama.cpp GGML_METAL_N_CB).
/// Default 2; set METAL_N_CB=1 to disable.
pub fn metal_n_cb() -> u32 {
    match std::env::var("METAL_N_CB").as_deref() {
        Ok("1") => 1,
        Ok(s) => s.parse::<u32>().unwrap_or(2).clamp(1, 8),
        _ => 2,
    }
}

/// Fuse pre-attn RMSNorm + Q4 Q/K/V projections into one dispatch (default on).
/// Set FUSED_QKV=0 to use separate rmsnorm + matvec dispatches.
pub fn fused_qkv_enabled() -> bool {
    !matches!(
        std::env::var("FUSED_QKV").as_deref(),
        Ok("0") | Ok("false") | Ok("FALSE")
    )
}

/// Fuse QK-norm + RoPE into Q4 flash decode attention (default on).
/// Set FUSED_Q_ATTN=0 to use separate qknorm + rotary + attention dispatches.
pub fn fused_q_attn_enabled() -> bool {
    !matches!(
        std::env::var("FUSED_Q_ATTN").as_deref(),
        Ok("0") | Ok("false") | Ok("FALSE")
    )
}

/// Prefill Q4_K projections use llama.cpp mul_mm when seq_len > 8 (default on).
/// Set PREFILL_MUL_MM=0 to force matvec for A/B testing.
pub fn prefill_mul_mm_enabled() -> bool {
    !matches!(
        std::env::var("PREFILL_MUL_MM").as_deref(),
        Ok("0") | Ok("false") | Ok("FALSE") | Ok("legacy")
    )
}

/// Prefill MLP gate∥up: one stacked mul_mm + stacked GeLU instead of two mul_mm + gelu_mul.
/// Set PREFILL_GATE_UP_STACKED=0 to use separate gate/up projections.
pub fn prefill_gate_up_stacked_enabled() -> bool {
    !matches!(
        std::env::var("PREFILL_GATE_UP_STACKED").as_deref(),
        Ok("0") | Ok("false") | Ok("FALSE")
    )
}

/// Prefill Q∥K∥V: one stacked mul_mm + split instead of three mul_mm dispatches (KV layers).
/// Set PREFILL_QKV_STACKED=0 to use separate Q/K/V projections.
pub fn prefill_qkv_stacked_enabled() -> bool {
    !matches!(
        std::env::var("PREFILL_QKV_STACKED").as_deref(),
        Ok("0") | Ok("false") | Ok("FALSE")
    )
}

/// Prefill Q/K/V post-projection: HSD layout + fused norm (skip SHD transposes).
/// Set PREFILL_QKV_HSD=0 to use legacy SHD rmsnorm + transpose_shd path.
pub fn prefill_qkv_hsd_enabled() -> bool {
    !matches!(
        std::env::var("PREFILL_QKV_HSD").as_deref(),
        Ok("0") | Ok("false") | Ok("FALSE")
    )
}

/// Output rows per Q4 fast matvec threadgroup (8 simdgroups × 4 rows).
const Q4_MATVEC_ROWS_PER_TG: u32 = 32;

fn attention_threadgroup_size(flash: bool) -> MTLSize {
    if flash {
        MTLSize::new(256, 1, 1)
    } else {
        MTLSize::new(64, 1, 1)
    }
}

/// Metal GPU context holding device, command queue, and compiled pipelines.
pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    pub matvec_pipeline: ComputePipelineState,
    pub matvec_f16_pipeline: ComputePipelineState,
    pub matvec_q4_pipeline: ComputePipelineState,
    pub matvec_q4_fast_pipeline: ComputePipelineState,
    pub matvec_q4_r1_pipeline: ComputePipelineState,
    pub matvec_q4_r2_pipeline: ComputePipelineState,
    pub matvec_q4_r8_pipeline: ComputePipelineState,
    pub matvec_q4_lc_pipeline: ComputePipelineState,
    pub matvec_q4_splitk_pipeline: ComputePipelineState,
    pub matvec_q4_dual_pipeline: ComputePipelineState,
    pub matvec_q4_dual_gelu_pipeline: ComputePipelineState,
    pub matvec_q4_interleaved_gelu_pipeline: ComputePipelineState,
    pub rmsnorm_inv_rms_pipeline: ComputePipelineState,
    pub matvec_q4_interleaved_gelu_hidden_pipeline: ComputePipelineState,
    pub matvec_q4_interleaved_gelu_f16_pipeline: ComputePipelineState,
    pub matvec_ggml_q4_f16x_pipeline: ComputePipelineState,
    pub matvec_q_rmsnorm_inv_q4_pipeline: ComputePipelineState,
    pub matvec_qkv_rmsnorm_inv_q4_pipeline: ComputePipelineState,
    pub matvec_q_rmsnorm_inv_kquant_pipeline: ComputePipelineState,
    pub matvec_qkv_rmsnorm_inv_kquant_pipeline: ComputePipelineState,
    pub ple_matvec_gelu_q4_pipeline: ComputePipelineState,
    pub matvec_ggml_q4_pipeline: ComputePipelineState,
    pub matvec_ggml_q4_dual_pipeline: ComputePipelineState,
    pub matvec_ggml_q4_gelu_mul_pipeline: ComputePipelineState,
    pub matvec_ggml_q4_gelu_mul_r2s4_pipeline: ComputePipelineState,
    // K-quant pipelines (native Q4_K / Q6_K matvec, decode + prefill)
    pub matvec_ggml_q4k_pipeline: ComputePipelineState,
    pub matvec_ggml_q6k_pipeline: ComputePipelineState,
    pub matvec_ggml_q4k_gelu_mul_pipeline: ComputePipelineState,
    pub matvec_ggml_q4k_rmsnorm_gelu_mul_pipeline: ComputePipelineState,
    /// Lazy: prefill Q4_K matrix-matrix (llama.cpp `kernel_mul_mm_q4_K_f32`).
    mul_mm_q4k_pipeline: OnceLock<ComputePipelineState>,
    /// Lazy: prefill Q6_K matrix-matrix (llama.cpp `kernel_mul_mm_q6_K_f32`).
    mul_mm_q6k_pipeline: OnceLock<ComputePipelineState>,
    // Q3_0 pipelines
    pub matvec_ggml_q3_pipeline: ComputePipelineState,
    pub matvec_ggml_q3_dual_pipeline: ComputePipelineState,
    pub matvec_ggml_q3_gelu_mul_pipeline: ComputePipelineState,
    pub matvec_ggml_q3_gelu_mul_r2s4_pipeline: ComputePipelineState,
    pub matvec_ggml_ext_q4_nx4_pipeline: ComputePipelineState,
    pub matvec_ggml_ext_q4_nx8_pipeline: ComputePipelineState,
    pub matvec_ggml_ext_q4_nx16_pipeline: ComputePipelineState,
    /// Decode Q4 matvec variant selected via the MATVEC_KERNEL env var.
    pub decode_matvec_kernel: DecodeMatvecKernel,
    pub projection_f16_batch_pipeline: ComputePipelineState,
    pub projection_q4_batch_pipeline: ComputePipelineState,
    pub projection_f16_batch_tiled_pipeline: ComputePipelineState,
    pub projection_q4_batch_tiled_pipeline: ComputePipelineState,
    pub matmul_pipeline: ComputePipelineState,
    pub rmsnorm_pipeline: ComputePipelineState,
    pub rmsnorm_add_pipeline: ComputePipelineState,
    pub rmsnorm_add_save_residual_pipeline: ComputePipelineState,
    pub rmsnorm_acc_pipeline: ComputePipelineState,
    pub rmsnorm_acc_out_pipeline: ComputePipelineState,
    pub rmsnorm_batch_pipeline: ComputePipelineState,
    pub rmsnorm_noweight_batch_pipeline: ComputePipelineState,
    pub silu_mul_pipeline: ComputePipelineState,
    pub silu_mul_batch_pipeline: ComputePipelineState,
    pub attention_pipeline: ComputePipelineState,
    pub attention_causal_pipeline: ComputePipelineState,
    pub rotary_pipeline: ComputePipelineState,
    pub rope_fill_decode_pipeline: ComputePipelineState,
    pub rope_fill_prefill_batch_pipeline: ComputePipelineState,
    pub rotary_batch_pipeline: ComputePipelineState,
    pub vec_add_pipeline: ComputePipelineState,
    pub vec_add_batch_pipeline: ComputePipelineState,
    pub buf_copy_pipeline: ComputePipelineState,
    pub kv_append_pipeline: ComputePipelineState,
    pub kv_append_f16_pipeline: ComputePipelineState,
    pub kv_batch_append_pipeline: ComputePipelineState,
    pub kv_batch_append_f16_pipeline: ComputePipelineState,
    pub kv_batch_append_strided_f16_pipeline: ComputePipelineState,
    pub transpose_shd_pipeline: ComputePipelineState,
    pub transpose_hsd_pipeline: ComputePipelineState,
    pub gelu_mul_pipeline: ComputePipelineState,
    pub gelu_mul_stacked_batch_pipeline: ComputePipelineState,
    pub qkv_split_stacked_batch_pipeline: ComputePipelineState,
    pub qkv_split_stacked_to_hsd_pipeline: ComputePipelineState,
    pub rmsnorm_hsd_batch_pipeline: ComputePipelineState,
    pub rmsnorm_noweight_hsd_batch_pipeline: ComputePipelineState,
    pub rmsnorm_shd_to_hsd_pipeline: ComputePipelineState,
    pub rmsnorm_noweight_shd_to_hsd_pipeline: ComputePipelineState,
    pub gelu_mul_f16_pipeline: ComputePipelineState,
    pub ple_gelu_mul_batch_pipeline: ComputePipelineState,
    pub vec_mul_pipeline: ComputePipelineState,
    pub vec_add_scaled_pipeline: ComputePipelineState,
    pub rmsnorm_per_head_pipeline: ComputePipelineState,
    pub rmsnorm_per_head_noweight_pipeline: ComputePipelineState,
    pub rotary_partial_pipeline: ComputePipelineState,
    pub attention_offset_pipeline: ComputePipelineState,
    pub attention_offset_f16_pipeline: ComputePipelineState,
    pub attention_offset_f16_gqa_pipeline: ComputePipelineState,
    pub attention_causal_f16_pipeline: ComputePipelineState,
    pub attention_causal_strided_f16_pipeline: ComputePipelineState,
    pub vec_scale_pipeline: ComputePipelineState,
    pub kv_append_q8_0_pipeline: ComputePipelineState,
    pub kv_batch_append_q8_0_pipeline: ComputePipelineState,
    pub kv_batch_append_strided_q8_0_pipeline: ComputePipelineState,
    pub kv_append_q4_0_pipeline: ComputePipelineState,
    pub kv_batch_append_q4_0_pipeline: ComputePipelineState,
    pub kv_batch_append_strided_q4_0_pipeline: ComputePipelineState,
    pub attention_offset_q8_0_pipeline: ComputePipelineState,
    pub attention_causal_q8_0_pipeline: ComputePipelineState,
    pub attention_causal_strided_q8_0_pipeline: ComputePipelineState,
    pub attention_offset_q4_0_pipeline: ComputePipelineState,
    pub attention_offset_q4_0_h256_pipeline: ComputePipelineState,
    pub attention_offset_q4_0_h128_pipeline: ComputePipelineState,
    pub attention_offset_q4_0_h512_pipeline: ComputePipelineState,
    pub attention_offset_q4_0_gqa_pipeline: ComputePipelineState,
    pub attention_fused_q4_0_pipeline: ComputePipelineState,
    pub attention_fused_q4_0_h256_pipeline: ComputePipelineState,
    pub attention_fused_q4_0_h128_pipeline: ComputePipelineState,
    pub attention_fused_q4_0_h512_pipeline: ComputePipelineState,
    pub attention_qknorm_rope_q4_0_h256_pipeline: ComputePipelineState,
    pub attention_qknorm_rope_q4_0_h128_pipeline: ComputePipelineState,
    pub attention_qknorm_rope_q4_0_h512_pipeline: ComputePipelineState,
    pub attention_fused_qknorm_rope_q4_0_h256_pipeline: ComputePipelineState,
    pub attention_fused_qknorm_rope_q4_0_h128_pipeline: ComputePipelineState,
    pub attention_fused_qknorm_rope_q4_0_h512_pipeline: ComputePipelineState,
    pub attention_full_fused_q4_0_h256_pipeline: ComputePipelineState,
    pub attention_full_fused_q4_0_h128_pipeline: ComputePipelineState,
    pub attention_full_fused_q4_0_h512_pipeline: ComputePipelineState,
    pub flash_attn_ggml_q4_h256_pipeline: ComputePipelineState,
    pub flash_attn_ggml_q4_h128_pipeline: ComputePipelineState,
    pub flash_attn_ggml_q4_h512_pipeline: ComputePipelineState,
    pub flash_attn_ggml_reduce_h256_pipeline: ComputePipelineState,
    pub flash_attn_ggml_reduce_h128_pipeline: ComputePipelineState,
    pub flash_attn_ggml_reduce_h512_pipeline: ComputePipelineState,
    pub attention_causal_q4_0_pipeline: ComputePipelineState,
    pub attention_causal_strided_q4_0_pipeline: ComputePipelineState,
    pub attention_causal_q4_0_gqa_h256_pipeline: ComputePipelineState,
    pub attention_causal_q4_0_gqa_h512_pipeline: ComputePipelineState,
    pub flash_attn_ext_prefill_pad_pipeline: ComputePipelineState,
    pub flash_attn_ext_prefill_blk_pipeline: ComputePipelineState,
    pub flash_attn_ext_prefill_q4_h256_pipeline: ComputePipelineState,
    pub flash_attn_ext_prefill_q4_h512_pipeline: ComputePipelineState,
    pub embed_gather_bf16_pipeline: ComputePipelineState,
    pub embed_gather_bf16_batch_pipeline: ComputePipelineState,
    pub sample_token_pipeline: ComputePipelineState,
    /// Tiled online-softmax attention (default). Set FLASH_ATTN=legacy to use
    /// the older per-token / TILE_KV=4 kernels.
    pub use_flash_attention: bool,
    /// Single-dispatch decode mega-kernel (see MEGA_KERNEL env var).
    pub decode_mega_gemma4_pipeline: ComputePipelineState,
}

impl MetalContext {
    pub fn new() -> Self {
        let device = Device::system_default().expect("No Metal GPU found");
        println!("  Metal GPU: {}", device.name());
        let queue = device.new_command_queue();

        let shader_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders/llama.metal");
        let mega_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders/decode_mega.metal");
        let ggml_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders/ggml_mul_mv_q4.metal");
        let ggml_fa_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders/ggml_flash_attn.metal");
        let ggml_fa_ext_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders/ggml_flash_attn_ext.metal");
        let mut shader_src =
            std::fs::read_to_string(&shader_path).expect("Failed to read Metal shader file");
        shader_src.push('\n');
        shader_src.push_str(
            &std::fs::read_to_string(&mega_path).expect("Failed to read decode_mega.metal"),
        );
        shader_src.push('\n');
        shader_src.push_str(
            &std::fs::read_to_string(&ggml_path).expect("Failed to read ggml_mul_mv_q4.metal"),
        );
        shader_src.push('\n');
        shader_src.push_str(
            &std::fs::read_to_string(&ggml_fa_path).expect("Failed to read ggml_flash_attn.metal"),
        );
        shader_src.push('\n');
        shader_src.push_str(
            &std::fs::read_to_string(&ggml_fa_ext_path)
                .expect("Failed to read ggml_flash_attn_ext.metal"),
        );

        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(&shader_src, &options)
            .expect("Failed to compile Metal shaders");

        let get_fn = |name: &str| -> ComputePipelineState {
            let func = library
                .get_function(name, None)
                .unwrap_or_else(|e| panic!("Failed to get function '{}': {:?}", name, e));
            device
                .new_compute_pipeline_state_with_function(&func)
                .unwrap_or_else(|e| panic!("Failed to create pipeline for '{}': {:?}", name, e))
        };

        let matvec_pipeline = get_fn("matvec");
        let matvec_f16_pipeline = get_fn("matvec_f16");
        let matvec_q4_pipeline = get_fn("matvec_q4");
        let matvec_q4_fast_pipeline = get_fn("matvec_q4_fast");
        let matvec_q4_r1_pipeline = get_fn("matvec_q4_r1");
        let matvec_q4_r2_pipeline = get_fn("matvec_q4_r2");
        let matvec_q4_r8_pipeline = get_fn("matvec_q4_r8");
        let matvec_q4_lc_pipeline = get_fn("matvec_q4_lc");
        let matvec_q4_splitk_pipeline = get_fn("matvec_q4_splitk");
        let matvec_q4_dual_pipeline = get_fn("matvec_q4_dual");
        let matvec_q4_dual_gelu_pipeline = get_fn("matvec_q4_dual_gelu");
        let matvec_q4_interleaved_gelu_pipeline = get_fn("matvec_q4_interleaved_gelu");
        let rmsnorm_inv_rms_pipeline = get_fn("rmsnorm_inv_rms");
        let matvec_q4_interleaved_gelu_hidden_pipeline =
            get_fn("matvec_q4_interleaved_gelu_hidden");
        let matvec_q4_interleaved_gelu_f16_pipeline = get_fn("matvec_q4_interleaved_gelu_f16");
        let matvec_ggml_q4_f16x_pipeline = get_fn("matvec_ggml_q4_0_f16x");
        let matvec_q_rmsnorm_inv_q4_pipeline = get_fn("matvec_q_rmsnorm_inv_q4");
        let matvec_qkv_rmsnorm_inv_q4_pipeline = get_fn("matvec_qkv_rmsnorm_inv_q4");
        let matvec_q_rmsnorm_inv_kquant_pipeline = get_fn("matvec_q_rmsnorm_inv_kquant");
        let matvec_qkv_rmsnorm_inv_kquant_pipeline = get_fn("matvec_qkv_rmsnorm_inv_kquant");
        let ple_matvec_gelu_q4_pipeline = get_fn("ple_matvec_gelu_q4");
        let matvec_ggml_q4_pipeline = get_fn("matvec_ggml_q4_0");
        let matvec_ggml_q4_dual_pipeline = get_fn("matvec_ggml_q4_0_dual");
        let matvec_ggml_q4_gelu_mul_pipeline = get_fn("matvec_ggml_q4_0_gelu_mul");
        let matvec_ggml_q4_gelu_mul_r2s4_pipeline = get_fn("matvec_ggml_q4_0_gelu_mul_r2s4");
        let matvec_ggml_q4k_pipeline = get_fn("matvec_ggml_q4_K");
        let matvec_ggml_q6k_pipeline = get_fn("matvec_ggml_q6_K");
        let matvec_ggml_q4k_gelu_mul_pipeline = get_fn("matvec_ggml_q4_K_gelu_mul");
        let matvec_ggml_q4k_rmsnorm_gelu_mul_pipeline = get_fn("matvec_ggml_q4_K_rmsnorm_gelu_mul");
        let matvec_ggml_q3_pipeline = get_fn("matvec_ggml_q3_0");
        let matvec_ggml_q3_dual_pipeline = get_fn("matvec_ggml_q3_0_dual");
        let matvec_ggml_q3_gelu_mul_pipeline = get_fn("matvec_ggml_q3_0_gelu_mul");
        let matvec_ggml_q3_gelu_mul_r2s4_pipeline = get_fn("matvec_ggml_q3_0_gelu_mul_r2s4");
        let matvec_ggml_ext_q4_nx4_pipeline = get_fn("matvec_ggml_ext_q4_nx4_r4");
        let matvec_ggml_ext_q4_nx8_pipeline = get_fn("matvec_ggml_ext_q4_nx8_r4");
        let matvec_ggml_ext_q4_nx16_pipeline = get_fn("matvec_ggml_ext_q4_nx16_r4");
        let decode_matvec_kernel = DecodeMatvecKernel::from_env();
        match decode_matvec_kernel {
            DecodeMatvecKernel::Auto => {
                println!("  Q4 matvec: auto → ggml per-shape (MATVEC_KERNEL=auto; bench with --bench-matvec)");
            }
            DecodeMatvecKernel::Lc => {
                println!("  Q4 matvec: llama.cpp/ane-infer style (MATVEC_KERNEL=lc, 128 threads/TG)");
            }
            DecodeMatvecKernel::Ggml => {
                println!("  Q4 matvec: ggml-metal block_q_n_dot_y (MATVEC_KERNEL=ggml, 8 rows/TG, 64 threads)");
            }
            DecodeMatvecKernel::GgmlExt => {
                println!("  Q4 matvec: ggml-metal mul_mv_ext (MATVEC_KERNEL=ggml-ext, nxpsg by K)");
            }
            k => {
                println!("  Q4 matvec: fixed {} (MATVEC_KERNEL={})", k.label(), k.label());
            }
        }
        let use_flash_attention = flash_attention_enabled();
        let projection_f16_batch_pipeline = get_fn("projection_f16_batch");
        let projection_q4_batch_pipeline = get_fn("projection_q4_batch");
        let projection_f16_batch_tiled_pipeline = get_fn("projection_f16_batch_tiled");
        let projection_q4_batch_tiled_pipeline = get_fn("projection_q4_batch_tiled");
        let matmul_pipeline = get_fn("matmul");
        let rmsnorm_pipeline = get_fn("rmsnorm");
        let rmsnorm_add_pipeline = get_fn("rmsnorm_add");
        let rmsnorm_add_save_residual_pipeline = get_fn("rmsnorm_add_save_residual");
        let rmsnorm_acc_pipeline = get_fn("rmsnorm_acc");
        let rmsnorm_acc_out_pipeline = get_fn("rmsnorm_acc_out");
        let rmsnorm_batch_pipeline = get_fn("rmsnorm_batch");
        let rmsnorm_noweight_batch_pipeline = get_fn("rmsnorm_noweight_batch");
        let silu_mul_pipeline = get_fn("silu_mul");
        let silu_mul_batch_pipeline = get_fn("silu_mul_batch");
        let attention_pipeline = get_fn("attention_single_token");
        let attention_causal_pipeline = get_fn("attention_causal");
        let rotary_pipeline = get_fn("apply_rotary");
        let rope_fill_decode_pipeline = get_fn("rope_fill_decode");
        let rope_fill_prefill_batch_pipeline = get_fn("rope_fill_prefill_batch");
        let rotary_batch_pipeline = get_fn("apply_rotary_batch");
        let vec_add_pipeline = get_fn("vec_add");
        let vec_add_batch_pipeline = get_fn("vec_add_batch");
        let buf_copy_pipeline = get_fn("buf_copy");
        let kv_append_pipeline = get_fn("kv_cache_append");
        let kv_append_f16_pipeline = get_fn("kv_cache_append_f16");
        let kv_batch_append_pipeline = get_fn("kv_cache_batch_append");
        let kv_batch_append_f16_pipeline = get_fn("kv_cache_batch_append_f16");
        let kv_batch_append_strided_f16_pipeline = get_fn("kv_cache_batch_append_strided_f16");
        let transpose_shd_pipeline = get_fn("transpose_shd_to_hsd");
        let transpose_hsd_pipeline = get_fn("transpose_hsd_to_shd");
        let gelu_mul_pipeline = get_fn("gelu_mul");
        let gelu_mul_stacked_batch_pipeline = get_fn("gelu_mul_stacked_batch");
        let qkv_split_stacked_batch_pipeline = get_fn("qkv_split_stacked_batch");
        let qkv_split_stacked_to_hsd_pipeline = get_fn("qkv_split_stacked_to_hsd");
        let rmsnorm_hsd_batch_pipeline = get_fn("rmsnorm_hsd_batch");
        let rmsnorm_noweight_hsd_batch_pipeline = get_fn("rmsnorm_noweight_hsd_batch");
        let rmsnorm_shd_to_hsd_pipeline = get_fn("rmsnorm_shd_to_hsd");
        let rmsnorm_noweight_shd_to_hsd_pipeline = get_fn("rmsnorm_noweight_shd_to_hsd");
        let gelu_mul_f16_pipeline = get_fn("gelu_mul_f16");
        let ple_gelu_mul_batch_pipeline = get_fn("ple_gelu_mul_batch");
        let vec_mul_pipeline = get_fn("vec_mul");
        let vec_add_scaled_pipeline = get_fn("vec_add_scaled");
        let rmsnorm_per_head_pipeline = get_fn("rmsnorm_per_head");
        let rmsnorm_per_head_noweight_pipeline = get_fn("rmsnorm_per_head_noweight");
        let rotary_partial_pipeline = get_fn("apply_rotary_partial");
        let attention_offset_pipeline = get_fn("attention_single_token_offset");
        let attention_offset_f16_pipeline = get_fn(if use_flash_attention {
            "attention_flash_decode_f16"
        } else {
            "attention_single_token_offset_f16"
        });
        let attention_offset_f16_gqa_pipeline = get_fn("attention_single_token_offset_f16_gqa");
        let attention_causal_f16_pipeline = get_fn(if use_flash_attention {
            "attention_flash_causal_f16"
        } else {
            "attention_causal_f16"
        });
        let attention_causal_strided_f16_pipeline = get_fn(if use_flash_attention {
            "attention_flash_causal_strided_f16"
        } else {
            "attention_causal_strided_f16"
        });
        let vec_scale_pipeline = get_fn("vec_scale");

        let kv_append_q8_0_pipeline = get_fn("kv_cache_append_q8_0");
        let kv_batch_append_q8_0_pipeline = get_fn("kv_cache_batch_append_q8_0");
        let kv_batch_append_strided_q8_0_pipeline = get_fn("kv_cache_batch_append_strided_q8_0");
        let kv_append_q4_0_pipeline = get_fn("kv_cache_append_q4_0");
        let kv_batch_append_q4_0_pipeline = get_fn("kv_cache_batch_append_q4_0");
        let kv_batch_append_strided_q4_0_pipeline = get_fn("kv_cache_batch_append_strided_q4_0");
        let attention_offset_q8_0_pipeline = get_fn("attention_single_token_offset_q8_0");
        let attention_causal_q8_0_pipeline = get_fn("attention_causal_q8_0");
        let attention_causal_strided_q8_0_pipeline = get_fn("attention_causal_strided_q8_0");
        let attention_offset_q4_0_pipeline = get_fn(if use_flash_attention {
            "attention_flash_decode_q4_0"
        } else {
            "attention_single_token_offset_q4_0"
        });
        let attention_offset_q4_0_h256_pipeline = get_fn("attention_flash_decode_q4_0_h256");
        let attention_offset_q4_0_h128_pipeline = get_fn("attention_flash_decode_q4_0_h128");
        let attention_offset_q4_0_h512_pipeline = get_fn("attention_flash_decode_q4_0_h512");
        let attention_offset_q4_0_gqa_pipeline = get_fn("attention_flash_decode_q4_0_gqa");
        let attention_fused_q4_0_pipeline = get_fn("attention_flash_decode_fused_q4_0");
        let attention_fused_q4_0_h256_pipeline = get_fn("attention_flash_decode_fused_q4_0_h256");
        let attention_fused_q4_0_h128_pipeline = get_fn("attention_flash_decode_fused_q4_0_h128");
        let attention_fused_q4_0_h512_pipeline = get_fn("attention_flash_decode_fused_q4_0_h512");
        let attention_qknorm_rope_q4_0_h256_pipeline =
            get_fn("attention_flash_decode_qknorm_rope_q4_0_h256");
        let attention_qknorm_rope_q4_0_h128_pipeline =
            get_fn("attention_flash_decode_qknorm_rope_q4_0_h128");
        let attention_qknorm_rope_q4_0_h512_pipeline =
            get_fn("attention_flash_decode_qknorm_rope_q4_0_h512");
        let attention_fused_qknorm_rope_q4_0_h256_pipeline =
            get_fn("attention_flash_decode_fused_qknorm_rope_q4_0_h256");
        let attention_fused_qknorm_rope_q4_0_h128_pipeline =
            get_fn("attention_flash_decode_fused_qknorm_rope_q4_0_h128");
        let attention_fused_qknorm_rope_q4_0_h512_pipeline =
            get_fn("attention_flash_decode_fused_qknorm_rope_q4_0_h512");
        let attention_full_fused_q4_0_h256_pipeline =
            get_fn("attention_flash_decode_full_fused_q4_0_h256");
        let attention_full_fused_q4_0_h128_pipeline =
            get_fn("attention_flash_decode_full_fused_q4_0_h128");
        let attention_full_fused_q4_0_h512_pipeline =
            get_fn("attention_flash_decode_full_fused_q4_0_h512");
        let flash_attn_ggml_q4_h256_pipeline = get_fn("flash_attn_ggml_q4_0_h256");
        let flash_attn_ggml_q4_h128_pipeline = get_fn("flash_attn_ggml_q4_0_h128");
        let flash_attn_ggml_q4_h512_pipeline = get_fn("flash_attn_ggml_q4_0_h512");
        let flash_attn_ggml_reduce_h256_pipeline = get_fn("flash_attn_ggml_q4_0_reduce_h256");
        let flash_attn_ggml_reduce_h128_pipeline = get_fn("flash_attn_ggml_q4_0_reduce_h128");
        let flash_attn_ggml_reduce_h512_pipeline = get_fn("flash_attn_ggml_q4_0_reduce_h512");
        let attention_causal_q4_0_pipeline = get_fn(if use_flash_attention {
            "attention_flash_causal_q4_0"
        } else {
            "attention_causal_q4_0"
        });
        let attention_causal_strided_q4_0_pipeline = get_fn(if use_flash_attention {
            "attention_flash_causal_strided_q4_0"
        } else {
            "attention_causal_strided_q4_0"
        });
        let attention_causal_q4_0_gqa_h256_pipeline =
            get_fn("attention_flash_causal_q4_0_gqa_h256");
        let attention_causal_q4_0_gqa_h512_pipeline =
            get_fn("attention_flash_causal_q4_0_gqa_h512");
        let flash_attn_ext_prefill_pad_pipeline = get_fn("flash_attn_ext_prefill_pad_q4_0");
        let flash_attn_ext_prefill_blk_pipeline = get_fn("flash_attn_ext_prefill_blk");
        let flash_attn_ext_prefill_q4_h256_pipeline = get_fn("flash_attn_ext_prefill_q4_0_h256");
        let flash_attn_ext_prefill_q4_h512_pipeline = get_fn("flash_attn_ext_prefill_q4_0_h512");
        let embed_gather_bf16_pipeline = get_fn("embed_gather_bf16");
        let embed_gather_bf16_batch_pipeline = get_fn("embed_gather_bf16_batch");
        let sample_token_pipeline = get_fn("sample_token");
        let decode_mega_gemma4_pipeline = get_fn("decode_mega_gemma4_q4_0");
        if use_flash_attention {
            println!("  FlashAttention-style tiled kernels enabled (FLASH_ATTN=legacy to disable)");
            if prefill_flash_attn_ext_enabled() {
                println!(
                    "  Q4 prefill attention: llama.cpp flash_attn_ext tiled (PREFILL_FLASH_ATTN=1, q_len≥20)"
                );
            }
            match attention_kernel_mode() {
                AttentionKernelMode::Ggml => {
                    println!("  Q4 decode attention: ggml flash_attn_ext_vec nwg=32 (ATTENTION_KERNEL=ggml)");
                }
                AttentionKernelMode::Auto => {
                    println!(
                        "  Q4 decode attention: auto hybrid — fused <128 tok, ggml MWG ≥128 (ATTENTION_KERNEL=auto)"
                    );
                }
                AttentionKernelMode::Specialized if attention_q4_hd_specialized() => {
                    println!(
                        "  Q4 decode attention: h256/h512 specialized + fused KV (ATTENTION_KERNEL=ggml|auto to try alternatives)"
                    );
                }
                _ => {}
            }
            if fused_kv_attention_enabled() {
                println!("  Fused KV append + Q4 flash attention (FUSED_KV_ATTN=0 to disable)");
            }
            if fused_k_attn_enabled() {
                println!(
                    "  Fused K-norm + RoPE + V-norm + KV append into Q4 flash (FUSED_K_ATTN=0 to disable)"
                );
            }
            if matches!(
                std::env::var("ATTENTION_GQA_Q4").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE")
            ) {
                println!(
                    "  GQA-aware tiled Q4_0 flash-decode attention (ATTENTION_GQA_Q4=1)"
                );
            }
            if packed_mlp_gate_up_enabled() {
                println!("  Interleaved gate∥up Q4 MLP weights (PACKED_MLP_GATE_UP=0 to disable)");
            }
            if mlp_gelu_f16_enabled() {
                println!("  MLP f16 GeLU scratch gate→down (MLP_GELU_F16=1 enabled)");
            }
            if mlp_gate_up_ggml_enabled() {
                println!("  MLP gate+up via separate ggml Q4 matvecs (MLP_GATE_UP_GGML=1)");
            }
            if mlp_fused_gelu_ggml_enabled() {
                println!("  Fused gate+up+GeLU Q4_0 kernel (MLP_FUSED_GELU_GGML=0 to disable)");
            }
            if mlp_fused_gelu_ggml_r2s4_enabled() {
                println!("  Fused gate+up+GeLU r2s4 layout (MLP_FUSED_GELU_GGML_R2S4=0 to disable)");
            }
            if fused_rmsnorm_mlp_kquant_enabled() {
                println!("  Fused pre-FF RMSNorm + Q4_K gate∥up+GeLU on K-quant (FUSED_RMSNORM_MLP_KQUANT=0 to disable)");
            }
            if fused_rmsnorm_mlp_enabled() {
                println!("  Fused pre-FF RMSNorm + gate∥up+GeLU (FUSED_RMSNORM_MLP=0 to disable)");
            }
            if fused_mlp_gelu_down_enabled() {
                println!("  Fused MLP gate+up+GeLU+down pipeline (FUSED_MLP_GELU_DOWN=0 to disable)");
            } else if fused_mlp_ple_enabled() {
                println!("  Fused MLP gate+up+GeLU and PLE gate+GeLU (FUSED_MLP_PLE=0 to disable)");
            }
            if fused_rmsnorm_acc_enabled() {
                println!("  Fused post-attn/post-MLP RMSNorm+residual (FUSED_RMSNORM_ACC=0 to disable)");
            }
            if fused_qkv_enabled() {
                println!("  Fused pre-attn RMSNorm + Q/K/V projections (Q4_0 + K-quant; FUSED_QKV=0 to disable)");
            }
            if fused_q_attn_enabled() {
                println!("  Fused QK-norm + RoPE into Q4 flash attention (FUSED_Q_ATTN=0 to disable)");
            }
        }

        MetalContext {
            device,
            queue,
            matvec_pipeline,
            matvec_f16_pipeline,
            matvec_q4_pipeline,
            matvec_q4_fast_pipeline,
            matvec_q4_r1_pipeline,
            matvec_q4_r2_pipeline,
            matvec_q4_r8_pipeline,
            matvec_q4_lc_pipeline,
            matvec_q4_splitk_pipeline,
            matvec_q4_dual_pipeline,
            matvec_q4_dual_gelu_pipeline,
            matvec_q4_interleaved_gelu_pipeline,
            rmsnorm_inv_rms_pipeline,
            matvec_q4_interleaved_gelu_hidden_pipeline,
            matvec_q4_interleaved_gelu_f16_pipeline,
            matvec_ggml_q4_f16x_pipeline,
            matvec_q_rmsnorm_inv_q4_pipeline,
            matvec_qkv_rmsnorm_inv_q4_pipeline,
            matvec_q_rmsnorm_inv_kquant_pipeline,
            matvec_qkv_rmsnorm_inv_kquant_pipeline,
            ple_matvec_gelu_q4_pipeline,
            matvec_ggml_q4_pipeline,
            matvec_ggml_q4_dual_pipeline,
            matvec_ggml_q4_gelu_mul_pipeline,
            matvec_ggml_q4_gelu_mul_r2s4_pipeline,
            matvec_ggml_q4k_pipeline,
            matvec_ggml_q6k_pipeline,
            matvec_ggml_q4k_gelu_mul_pipeline,
            matvec_ggml_q4k_rmsnorm_gelu_mul_pipeline,
            mul_mm_q4k_pipeline: OnceLock::new(),
            mul_mm_q6k_pipeline: OnceLock::new(),
            matvec_ggml_q3_pipeline,
            matvec_ggml_q3_dual_pipeline,
            matvec_ggml_q3_gelu_mul_pipeline,
            matvec_ggml_q3_gelu_mul_r2s4_pipeline,
            matvec_ggml_ext_q4_nx4_pipeline,
            matvec_ggml_ext_q4_nx8_pipeline,
            matvec_ggml_ext_q4_nx16_pipeline,
            decode_matvec_kernel,
            projection_f16_batch_pipeline,
            projection_q4_batch_pipeline,
            projection_f16_batch_tiled_pipeline,
            projection_q4_batch_tiled_pipeline,
            matmul_pipeline,
            rmsnorm_pipeline,
            rmsnorm_add_pipeline,
            rmsnorm_add_save_residual_pipeline,
            rmsnorm_acc_pipeline,
            rmsnorm_acc_out_pipeline,
            rmsnorm_batch_pipeline,
            rmsnorm_noweight_batch_pipeline,
            silu_mul_pipeline,
            silu_mul_batch_pipeline,
            attention_pipeline,
            attention_causal_pipeline,
            rotary_pipeline,
            rope_fill_decode_pipeline,
            rope_fill_prefill_batch_pipeline,
            rotary_batch_pipeline,
            vec_add_pipeline,
            vec_add_batch_pipeline,
            buf_copy_pipeline,
            kv_append_pipeline,
            kv_append_f16_pipeline,
            kv_batch_append_pipeline,
            kv_batch_append_f16_pipeline,
            kv_batch_append_strided_f16_pipeline,
            transpose_shd_pipeline,
            transpose_hsd_pipeline,
            gelu_mul_pipeline,
            gelu_mul_stacked_batch_pipeline,
            qkv_split_stacked_batch_pipeline,
            qkv_split_stacked_to_hsd_pipeline,
            rmsnorm_hsd_batch_pipeline,
            rmsnorm_noweight_hsd_batch_pipeline,
            rmsnorm_shd_to_hsd_pipeline,
            rmsnorm_noweight_shd_to_hsd_pipeline,
            gelu_mul_f16_pipeline,
            ple_gelu_mul_batch_pipeline,
            vec_mul_pipeline,
            vec_add_scaled_pipeline,
            rmsnorm_per_head_pipeline,
            rmsnorm_per_head_noweight_pipeline,
            rotary_partial_pipeline,
            attention_offset_pipeline,
            attention_offset_f16_pipeline,
            attention_offset_f16_gqa_pipeline,
            attention_causal_f16_pipeline,
            attention_causal_strided_f16_pipeline,
            vec_scale_pipeline,
            kv_append_q8_0_pipeline,
            kv_batch_append_q8_0_pipeline,
            kv_batch_append_strided_q8_0_pipeline,
            kv_append_q4_0_pipeline,
            kv_batch_append_q4_0_pipeline,
            kv_batch_append_strided_q4_0_pipeline,
            attention_offset_q8_0_pipeline,
            attention_causal_q8_0_pipeline,
            attention_causal_strided_q8_0_pipeline,
            attention_offset_q4_0_pipeline,
            attention_offset_q4_0_h256_pipeline,
            attention_offset_q4_0_h128_pipeline,
            attention_offset_q4_0_h512_pipeline,
            attention_offset_q4_0_gqa_pipeline,
            attention_fused_q4_0_pipeline,
            attention_fused_q4_0_h256_pipeline,
            attention_fused_q4_0_h128_pipeline,
            attention_fused_q4_0_h512_pipeline,
            attention_qknorm_rope_q4_0_h256_pipeline,
            attention_qknorm_rope_q4_0_h128_pipeline,
            attention_qknorm_rope_q4_0_h512_pipeline,
            attention_fused_qknorm_rope_q4_0_h256_pipeline,
            attention_fused_qknorm_rope_q4_0_h128_pipeline,
            attention_fused_qknorm_rope_q4_0_h512_pipeline,
            attention_full_fused_q4_0_h256_pipeline,
            attention_full_fused_q4_0_h128_pipeline,
            attention_full_fused_q4_0_h512_pipeline,
            flash_attn_ggml_q4_h256_pipeline,
            flash_attn_ggml_q4_h128_pipeline,
            flash_attn_ggml_q4_h512_pipeline,
            flash_attn_ggml_reduce_h256_pipeline,
            flash_attn_ggml_reduce_h128_pipeline,
            flash_attn_ggml_reduce_h512_pipeline,
            attention_causal_q4_0_pipeline,
            attention_causal_strided_q4_0_pipeline,
            attention_causal_q4_0_gqa_h256_pipeline,
            attention_causal_q4_0_gqa_h512_pipeline,
            flash_attn_ext_prefill_pad_pipeline,
            flash_attn_ext_prefill_blk_pipeline,
            flash_attn_ext_prefill_q4_h256_pipeline,
            flash_attn_ext_prefill_q4_h512_pipeline,
            embed_gather_bf16_pipeline,
            embed_gather_bf16_batch_pipeline,
            sample_token_pipeline,
            use_flash_attention,
            decode_mega_gemma4_pipeline,
        }
    }

    /// Compile prefill mul_mm in a separate Metal library (lazy — decode never pays this cost).
    fn compile_mul_mm_pipeline(device: &Device, entry: &str) -> ComputePipelineState {
        let ggml_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders/ggml_mul_mv_q4.metal");
        let ggml_mm_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders/ggml_mul_mm_q4.metal");
        let mut mul_mm_src =
            std::fs::read_to_string(&ggml_path).expect("Failed to read ggml_mul_mv_q4.metal");
        mul_mm_src.push('\n');
        mul_mm_src.push_str(
            &std::fs::read_to_string(&ggml_mm_path).expect("Failed to read ggml_mul_mm_q4.metal"),
        );
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(&mul_mm_src, &options)
            .expect("Failed to compile ggml_mul_mm Metal shaders");
        let func = library
            .get_function(entry, None)
            .unwrap_or_else(|e| panic!("Failed to get mul_mm function '{}': {:?}", entry, e));
        device
            .new_compute_pipeline_state_with_function(&func)
            .unwrap_or_else(|e| panic!("Failed to create mul_mm pipeline for '{}': {:?}", entry, e))
    }

    fn mul_mm_q4k_pipeline(&self) -> &ComputePipelineState {
        self.mul_mm_q4k_pipeline.get_or_init(|| {
            println!("  Prefill mul_mm: compiling Q4_K simdgroup matmul pipeline");
            Self::compile_mul_mm_pipeline(&self.device, "mul_mm_q4_K_f32")
        })
    }

    fn mul_mm_q6k_pipeline(&self) -> &ComputePipelineState {
        self.mul_mm_q6k_pipeline.get_or_init(|| {
            println!("  Prefill mul_mm: compiling Q6_K simdgroup matmul pipeline");
            Self::compile_mul_mm_pipeline(&self.device, "mul_mm_q6_K_f32")
        })
    }

    /// Copy a mmap'd region into a GPU-accessible shared buffer.
    pub fn buffer_from_mmap_copy(
        device: &Device,
        mmap: &memmap2::Mmap,
        offset: usize,
        len: u64,
    ) -> Buffer {
        Self::buffer_from_slice_parallel(device, &mmap[offset..offset + len as usize])
    }

    /// Read a file region directly into a new shared Metal buffer (one copy, no extra alloc).
    pub fn buffer_read_from_file(
        device: &Device,
        file: &mut std::fs::File,
        offset: u64,
        len: u64,
    ) -> Buffer {
        use std::io::{Read, Seek, SeekFrom};
        let buf = device.new_buffer(len, MTLResourceOptions::StorageModeShared);
        file.seek(SeekFrom::Start(offset)).expect("seek weights file");
        let dst = unsafe {
            std::slice::from_raw_parts_mut(buf.contents() as *mut u8, len as usize)
        };
        file.read_exact(dst).expect("read weights into GPU buffer");
        buf
    }

    /// Copy a byte slice into a new shared Metal buffer (parallel for large payloads).
    pub fn buffer_from_slice_parallel(device: &Device, src: &[u8]) -> Buffer {
        let buf = device.new_buffer(src.len() as u64, MTLResourceOptions::StorageModeShared);
        let dst = unsafe {
            std::slice::from_raw_parts_mut(buf.contents() as *mut u8, src.len())
        };
        Self::parallel_memcpy(dst, src);
        buf
    }

    fn parallel_memcpy(dst: &mut [u8], src: &[u8]) {
        assert_eq!(dst.len(), src.len());
        const CHUNK: usize = 16 * 1024 * 1024;
        if src.len() <= CHUNK {
            dst.copy_from_slice(src);
            return;
        }
        std::thread::scope(|scope| {
            for (d, s) in dst.chunks_mut(CHUNK).zip(src.chunks(CHUNK)) {
                scope.spawn(move || d.copy_from_slice(s));
            }
        });
    }

    /// Zero-copy Metal buffer from mmap. Not safe for GPU kernels when mmap is file-backed.
    pub fn buffer_from_mmap_no_copy(
        device: &Device,
        mmap: &memmap2::Mmap,
        offset: usize,
        len: u64,
    ) -> Buffer {
        device.new_buffer_with_bytes_no_copy(
            unsafe { mmap.as_ptr().add(offset) as *const std::ffi::c_void },
            len,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }

    pub fn buffer_from_bytes(&self, data: &[u8]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    pub fn buffer_from_slice(&self, data: &[f32]) -> Buffer {
        let byte_len = (data.len() * std::mem::size_of::<f32>()) as u64;
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Create a Metal buffer with f16 data converted from f32.
    pub fn buffer_from_f32_as_f16(&self, data: &[f32]) -> Buffer {
        let f16_data: Vec<u16> = data.iter().map(|&v| f32_to_f16(v)).collect();
        let byte_len = (f16_data.len() * 2) as u64;
        self.device.new_buffer_with_data(
            f16_data.as_ptr() as *const _,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Create a Metal buffer with Q4_0 quantized data from f32.
    /// Format: for each group of 32 values: [f16 scale][16 bytes of packed 4-bit pairs]
    /// Total: 18 bytes per 32 weights.
    pub fn buffer_from_f32_as_q4(&self, data: &[f32], rows: usize, cols: usize) -> Buffer {
        let q4_data = quantize_q4_0(data, rows, cols);
        let byte_len = q4_data.len() as u64;
        self.device.new_buffer_with_data(
            q4_data.as_ptr() as *const _,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Create a Metal buffer with Q3_0 quantized data from f32.
    /// Format: for each group of 32 values: [f16 scale][8 bytes low-2-bits][4 bytes high-1-bit]
    /// Total: 14 bytes per 32 weights (~22% smaller than Q4_0's 18 bytes).
    pub fn buffer_from_f32_as_q3(&self, data: &[f32], rows: usize, cols: usize) -> Buffer {
        let q3_data = quantize_q3_0(data, rows, cols);
        let byte_len = q3_data.len() as u64;
        self.device.new_buffer_with_data(
            q3_data.as_ptr() as *const _,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        )
    }

    pub fn buffer_empty(&self, count: usize) -> Buffer {
        let byte_len = (count * std::mem::size_of::<f32>()) as u64;
        self.device
            .new_buffer(byte_len, MTLResourceOptions::StorageModeShared)
    }

    pub fn buffer_empty_u32(&self, count: usize) -> Buffer {
        let byte_len = (count * std::mem::size_of::<u32>()) as u64;
        self.device
            .new_buffer(byte_len, MTLResourceOptions::StorageModeShared)
    }

    pub fn read_buffer(buf: &Buffer, count: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, count).to_vec() }
    }

    pub fn write_buffer(buf: &Buffer, data: &[f32]) {
        let ptr = buf.contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    pub fn write_u32_buffer(buf: &Buffer, data: &[u32]) {
        let ptr = buf.contents() as *mut u32;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    pub fn read_u32(buf: &Buffer) -> u32 {
        let ptr = buf.contents() as *const u32;
        unsafe { *ptr }
    }

    /// Microbenchmark: separates fixed per-dispatch / per-commit overhead from
    /// true Q4 matvec bandwidth. No model needed (uses dummy buffers).
    /// Dispatch one Q4 matvec with an explicit kernel variant (bypasses the
    /// MATVEC_KERNEL selection so the benchmark can compare variants directly).
    fn encode_matvec_q4_variant(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        variant: DecodeMatvecKernel,
        w: &Buffer,
        x: &Buffer,
        y: &Buffer,
        m: u32,
        k: u32,
    ) {
        let variant = variant.resolve_for_shape(m, k);
        if matches!(variant, DecodeMatvecKernel::Ggml | DecodeMatvecKernel::GgmlExt) {
            self.encode_matvec_ggml_at(encoder, w, 0, x, 0, y, 0, m, k, variant);
            return;
        }
        let pipeline = match variant {
            DecodeMatvecKernel::Auto => unreachable!("resolved above"),
            DecodeMatvecKernel::R1 => &self.matvec_q4_r1_pipeline,
            DecodeMatvecKernel::R2 => &self.matvec_q4_r2_pipeline,
            DecodeMatvecKernel::Fast => &self.matvec_q4_fast_pipeline,
            DecodeMatvecKernel::R8 => &self.matvec_q4_r8_pipeline,
            DecodeMatvecKernel::Lc => &self.matvec_q4_lc_pipeline,
            DecodeMatvecKernel::SplitK => &self.matvec_q4_splitk_pipeline,
            DecodeMatvecKernel::Ggml | DecodeMatvecKernel::GgmlExt => unreachable!(),
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(w), 0);
        encoder.set_buffer(1, Some(x), 0);
        encoder.set_buffer(2, Some(y), 0);
        // set_bytes copies M/K into the command buffer — required when many matvecs
        // share one encoder (a shared params buffer would leave every dispatch
        // reading the last-written dimensions at GPU execute time).
        encoder.set_bytes(3, 4, &m as *const u32 as *const _);
        encoder.set_bytes(4, 4, &k as *const u32 as *const _);
        let (num_tgs, tg_size) = Self::matvec_q4_dispatch(variant, m);
        encoder.dispatch_thread_groups(num_tgs, tg_size);
    }

    pub fn bench_matvec(&self) {
        use std::time::Instant;
        let reps = 50;
        let packed = 30;
        // (M, K, label) representative of the E4B per-token decode matvecs.
        let shapes: &[(u32, u32, &str)] = &[
            (2560, 2560, "q/o      hidden x hidden"),
            (2048, 2560, "kv/proj  ~2k    x hidden"),
            (16384, 2560, "gate/up  inter  x hidden"),
            (2560, 16384, "down     hidden x inter"),
            (262144, 2560, "lm_head  vocab  x hidden"),
        ];
        let variants = [
            ("r1", DecodeMatvecKernel::R1),
            ("r2", DecodeMatvecKernel::R2),
            ("fast/4", DecodeMatvecKernel::Fast),
            ("r8", DecodeMatvecKernel::R8),
            ("lc", DecodeMatvecKernel::Lc),
            ("ggml", DecodeMatvecKernel::Ggml),
            ("ggml-ext", DecodeMatvecKernel::GgmlExt),
            ("splitk", DecodeMatvecKernel::SplitK),
        ];

        println!("\n=== matvec_q4 microbenchmark (packed {}/cmdbuf, GB/s) ===", packed);
        println!("Per-op kernel bandwidth; higher is better. Bytes = weights only.");
        println!("M1 Pro peak ~200 GB/s.\n");
        print!("{:<28} {:>8}", "shape", "MB");
        for (name, _) in variants.iter() {
            print!(" {:>8}", name);
        }
        println!(" {:>8} {:>8}", "best", "auto");

        for &(m, k, label) in shapes {
            let num_groups = (k / 32) as u64;
            let wbytes = (m as u64) * num_groups * 18;
            let w = self
                .device
                .new_buffer(wbytes, MTLResourceOptions::StorageModeShared);
            let x = self.buffer_empty(k as usize);
            let y = self.buffer_empty(m as usize);
            let mb = wbytes as f64 / 1e6;

            let mut bws = [0.0f64; 8];
            for (vi, (_, variant)) in variants.iter().enumerate() {
                // warmup
                for _ in 0..5 {
                    let cmd = self.queue.new_command_buffer();
                    let enc = cmd.new_compute_command_encoder();
                    self.encode_matvec_q4_variant(enc, *variant, &w, &x, &y, m, k);
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                }
                let t = Instant::now();
                for _ in 0..reps {
                    let cmd = self.queue.new_command_buffer();
                    let enc = cmd.new_compute_command_encoder();
                    for _ in 0..packed {
                        self.encode_matvec_q4_variant(enc, *variant, &w, &x, &y, m, k);
                    }
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                }
                let per_op_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64 / packed as f64;
                bws[vi] = mb / per_op_ms; // MB/ms == GB/s
            }

            let (best_vi, &best_bw) = bws
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            let auto = DecodeMatvecKernel::pick_for_shape(m, k);

            print!("{:<28} {:>8.2}", label, mb);
            for bw in bws.iter() {
                print!(" {:>8.0}", bw);
            }
            println!(
                " {:>8} {:>8}",
                variants[best_vi].0,
                auto.label()
            );
            let _ = best_bw; // used via best_vi
        }
        println!(
            "\nAuto selection (MATVEC_KERNEL=auto, default): ggml for all shapes on\n\
             end-to-end decode. Set MATVEC_KERNEL=fast|lc|... to override.\n"
        );
    }

    /// Fused logit softcap + temperature + min-p sampling on the GPU.
    /// Writes only the sampled token id (u32) into `out_token_buf`.
    pub fn encode_sample(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        logits_buf: &Buffer,
        out_token_buf: &Buffer,
        vocab_size: u32,
        cap: f32,
        temperature: f32,
        min_p: f32,
        seed: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.sample_token_pipeline);
        encoder.set_buffer(0, Some(logits_buf), 0);
        encoder.set_buffer(1, Some(out_token_buf), 0);
        encoder.set_bytes(2, 4, &vocab_size as *const u32 as *const _);
        encoder.set_bytes(3, 4, &cap as *const f32 as *const _);
        encoder.set_bytes(4, 4, &temperature as *const f32 as *const _);
        encoder.set_bytes(5, 4, &min_p as *const f32 as *const _);
        encoder.set_bytes(6, 4, &seed as *const u32 as *const _);
        // Single threadgroup of 256 threads (matches SAMPLE_TG in the shader).
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_embed_gather_bf16(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        table_buf: &Buffer,
        token_id: u32,
        out_buf: &Buffer,
        row_width: u32,
        scale: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.embed_gather_bf16_pipeline);
        encoder.set_buffer(0, Some(table_buf), 0);
        encoder.set_buffer(1, Some(out_buf), 0);
        encoder.set_bytes(2, 4, &token_id as *const u32 as *const _);
        encoder.set_bytes(3, 4, &row_width as *const u32 as *const _);
        encoder.set_bytes(4, 4, &scale as *const f32 as *const _);
        let num_tgs = MTLSize::new((row_width as u64 + 255) / 256, 1, 1);
        let tg_size = MTLSize::new(256, 1, 1);
        encoder.dispatch_thread_groups(num_tgs, tg_size);
    }

    pub fn encode_embed_gather_bf16_batch(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        table_buf: &Buffer,
        token_ids_buf: &Buffer,
        out_buf: &Buffer,
        batch_size: u32,
        row_width: u32,
        scale: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.embed_gather_bf16_batch_pipeline);
        encoder.set_buffer(0, Some(table_buf), 0);
        encoder.set_buffer(1, Some(token_ids_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 4, &batch_size as *const u32 as *const _);
        encoder.set_bytes(4, 4, &row_width as *const u32 as *const _);
        encoder.set_bytes(5, 4, &scale as *const f32 as *const _);
        let total = batch_size as u64 * row_width as u64;
        let num_tgs = MTLSize::new((total + 255) / 256, 1, 1);
        let tg_size = MTLSize::new(256, 1, 1);
        encoder.dispatch_thread_groups(num_tgs, tg_size);
    }

    // ─── Encoder-based methods (encode into existing encoder) ────────────────

    pub fn encode_matvec(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        w_buf: &Buffer,
        x_buf: &Buffer,
        y_buf: &Buffer,
        m: u32,
        k: u32,
    ) {
        self.encode_matvec_at(encoder, w_buf, 0, x_buf, 0, y_buf, 0, m, k);
    }

    pub fn encode_matvec_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &Buffer,
        y_buf: &Buffer,
        m: u32,
        k: u32,
    ) {
        self.encode_matvec_at(
            encoder,
            &weight.buffer,
            weight.offset,
            x_buf,
            0,
            y_buf,
            0,
            m,
            k,
        );
    }

    pub fn encode_matvec_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        w_buf: &Buffer,
        w_offset: u64,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        m: u32,
        k: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.matvec_pipeline);
        encoder.set_buffer(0, Some(w_buf), w_offset);
        encoder.set_buffer(1, Some(x_buf), x_offset);
        encoder.set_buffer(2, Some(y_buf), y_offset);
        encoder.set_bytes(3, 4, &m as *const u32 as *const _);
        encoder.set_bytes(4, 4, &k as *const u32 as *const _);
        // One threadgroup per row, 32 threads per group (SIMD group)
        let num_tgs = MTLSize::new(m as u64, 1, 1);
        let tg_size = MTLSize::new(32, 1, 1);
        encoder.dispatch_thread_groups(num_tgs, tg_size);
    }

    /// f16 weight matvec: W is half precision, x and y are f32.
    pub fn encode_matvec_f16(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        w_buf: &Buffer,
        x_buf: &Buffer,
        y_buf: &Buffer,
        m: u32,
        k: u32,
    ) {
        self.encode_matvec_f16_at(encoder, w_buf, 0, x_buf, 0, y_buf, 0, m, k);
    }

    pub fn encode_matvec_f16_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &Buffer,
        y_buf: &Buffer,
        m: u32,
        k: u32,
    ) {
        self.encode_matvec_f16_at(
            encoder,
            &weight.buffer,
            weight.offset,
            x_buf,
            0,
            y_buf,
            0,
            m,
            k,
        );
    }

    pub fn encode_matvec_f16_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        m: u32,
        k: u32,
    ) {
        self.encode_matvec_f16_at(
            encoder,
            &weight.buffer,
            weight.offset,
            x_buf,
            x_offset,
            y_buf,
            y_offset,
            m,
            k,
        );
    }

    /// Matvec dispatching to Q4, Q3, or f16 based on the weight buffer layout.
    pub fn encode_matvec_auto_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &Buffer,
        y_buf: &Buffer,
        m: u32,
        k: u32,
    ) {
        if matches!(weight.format, weight_fmt::Q4_K | weight_fmt::Q6_K) {
            self.encode_matvec_qk_at_view(encoder, weight, x_buf, 0, y_buf, 0, m, k, 1);
        } else if weight_buf_is_q3(weight, m, k) {
            self.encode_matvec_q3_at_view(encoder, weight, x_buf, 0, y_buf, 0, m, k);
        } else if weight_buf_is_q4(weight, m, k) {
            self.encode_matvec_q4_view(encoder, weight, x_buf, y_buf, m, k);
        } else {
            self.encode_matvec_f16_view(encoder, weight, x_buf, y_buf, m, k);
        }
    }

    pub fn encode_matvec_auto_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        m: u32,
        k: u32,
    ) {
        if matches!(weight.format, weight_fmt::Q4_K | weight_fmt::Q6_K) {
            self.encode_matvec_qk_at_view(
                encoder, weight, x_buf, x_offset, y_buf, y_offset, m, k, 1,
            );
        } else if weight_buf_is_q3(weight, m, k) {
            self.encode_matvec_q3_at_view(
                encoder, weight, x_buf, x_offset, y_buf, y_offset, m, k,
            );
        } else if weight_buf_is_q4(weight, m, k) {
            self.encode_matvec_q4_at_view(
                encoder, weight, x_buf, x_offset, y_buf, y_offset, m, k,
            );
        } else {
            self.encode_matvec_f16_at_view(
                encoder, weight, x_buf, x_offset, y_buf, y_offset, m, k,
            );
        }
    }

    pub fn encode_projection_auto_batch_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &Buffer,
        y_buf: &Buffer,
        m: u32,
        k: u32,
        seq_len: u32,
    ) {
        if matches!(weight.format, weight_fmt::Q4_K | weight_fmt::Q6_K) {
            self.encode_matvec_qk_at_view(encoder, weight, x_buf, 0, y_buf, 0, m, k, seq_len);
        } else if weight_buf_is_q4(weight, m, k) {
            self.encode_projection_q4_batch_view(encoder, weight, x_buf, y_buf, m, k, seq_len);
        } else {
            self.encode_projection_f16_batch_view(encoder, weight, x_buf, y_buf, m, k, seq_len);
        }
    }

    /// Parallel prefill only — uses mul_mm for K-quant when seq_len > 8 (llama.cpp).
    /// Decode and shared batch paths keep `encode_projection_*_batch_view`.
    pub fn encode_prefill_projection_auto_batch_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &Buffer,
        y_buf: &Buffer,
        m: u32,
        k: u32,
        seq_len: u32,
    ) {
        if matches!(weight.format, weight_fmt::Q4_K | weight_fmt::Q6_K) {
            self.encode_prefill_kquant_projection(encoder, weight, x_buf, y_buf, m, k, seq_len);
        } else if weight_buf_is_q4(weight, m, k) {
            self.encode_prefill_projection_q4_batch_view(
                encoder, weight, x_buf, y_buf, m, k, seq_len,
            );
        } else {
            self.encode_projection_f16_batch_view(encoder, weight, x_buf, y_buf, m, k, seq_len);
        }
    }

    /// Prefill K-quant projection: mul_mm when seq_len > 8 (llama.cpp), else batched matvec.
    /// Opt out with `PREFILL_MUL_MM=0`.
    pub fn encode_prefill_kquant_projection(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &Buffer,
        y_buf: &Buffer,
        m: u32,
        k: u32,
        seq_len: u32,
    ) {
        let use_mm = prefill_mul_mm_enabled()
            && crate::ggml_gemv::should_use_mul_mm(k, seq_len)
            && matches!(weight.format, weight_fmt::Q4_K | weight_fmt::Q6_K);
        if use_mm {
            self.encode_mul_mm_kquant_at_view(encoder, weight, x_buf, y_buf, m, k, seq_len);
        } else {
            self.encode_matvec_qk_at_view(encoder, weight, x_buf, 0, y_buf, 0, m, k, seq_len);
        }
    }

    /// llama.cpp `kernel_mul_mm_{q4,q6}_K_f32` dispatch for prefill projections.
    pub fn encode_mul_mm_kquant_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &Buffer,
        y_buf: &Buffer,
        m: u32,
        k: u32,
        seq_len: u32,
    ) {
        use crate::ggml_gemv::{
            mul_mm_args_k, mul_mm_dispatch, MUL_MM_SMEM, Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES,
        };
        let (pipeline, block_bytes) = match weight.format {
            weight_fmt::Q4_K => (self.mul_mm_q4k_pipeline(), Q4_K_BLOCK_BYTES),
            weight_fmt::Q6_K => (self.mul_mm_q6k_pipeline(), Q6_K_BLOCK_BYTES),
            other => panic!("encode_mul_mm_kquant_at_view: not K-quant ({})", other),
        };
        let args = mul_mm_args_k(m, k, seq_len, block_bytes);
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<crate::ggml_gemv::GgmlMulMmArgs>() as u64,
            &args as *const _ as *const _,
        );
        encoder.set_buffer(1, Some(&weight.buffer), weight.offset);
        encoder.set_buffer(2, Some(x_buf), 0);
        encoder.set_buffer(3, Some(y_buf), 0);
        encoder.set_threadgroup_memory_length(0, MUL_MM_SMEM);
        let (tg_x, tg_y, tg_z, tw, nsg) = mul_mm_dispatch(m, seq_len);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(tg_x, tg_y, tg_z),
            metal::MTLSize::new(tw, nsg, 1),
        );
    }

    pub fn encode_matvec_f16_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        w_buf: &Buffer,
        w_offset: u64,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        m: u32,
        k: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.matvec_f16_pipeline);
        encoder.set_buffer(0, Some(w_buf), w_offset);
        encoder.set_buffer(1, Some(x_buf), x_offset);
        encoder.set_buffer(2, Some(y_buf), y_offset);
        encoder.set_bytes(3, 4, &m as *const u32 as *const _);
        encoder.set_bytes(4, 4, &k as *const u32 as *const _);
        let num_tgs = MTLSize::new(m as u64, 1, 1);
        let tg_size = MTLSize::new(32, 1, 1);
        encoder.dispatch_thread_groups(num_tgs, tg_size);
    }

    /// Q4_0 weight matvec: W is 4-bit quantized, x and y are f32.
    pub fn encode_matvec_q4(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        w_buf: &Buffer,
        x_buf: &Buffer,
        y_buf: &Buffer,
        m: u32,
        k: u32,
    ) {
        self.encode_matvec_q4_at(encoder, w_buf, 0, x_buf, 0, y_buf, 0, m, k);
    }

    pub fn encode_matvec_q4_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &Buffer,
        y_buf: &Buffer,
        m: u32,
        k: u32,
    ) {
        self.encode_matvec_q4_at(
            encoder,
            &weight.buffer,
            weight.offset,
            x_buf,
            0,
            y_buf,
            0,
            m,
            k,
        );
    }

    pub fn encode_matvec_q4_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        m: u32,
        k: u32,
    ) {
        self.encode_matvec_q4_at(
            encoder,
            &weight.buffer,
            weight.offset,
            x_buf,
            x_offset,
            y_buf,
            y_offset,
            m,
            k,
        );
    }

    pub fn encode_matvec_q4_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        w_buf: &Buffer,
        w_offset: u64,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        m: u32,
        k: u32,
    ) {
        let kernel = self.decode_matvec_kernel.resolve_for_shape(m, k);
        if matches!(kernel, DecodeMatvecKernel::Ggml | DecodeMatvecKernel::GgmlExt) {
            self.encode_matvec_ggml_at(
                encoder, w_buf, w_offset, x_buf, x_offset, y_buf, y_offset, m, k, kernel,
            );
            return;
        }

        // Row-parallel variants use 8 simdgroups (256 threads); lc uses 4 (128).
        let pipeline = match kernel {
            DecodeMatvecKernel::Auto => unreachable!("resolved above"),
            DecodeMatvecKernel::R1 => &self.matvec_q4_r1_pipeline,
            DecodeMatvecKernel::R2 => &self.matvec_q4_r2_pipeline,
            DecodeMatvecKernel::Fast => &self.matvec_q4_fast_pipeline,
            DecodeMatvecKernel::R8 => &self.matvec_q4_r8_pipeline,
            DecodeMatvecKernel::Lc => &self.matvec_q4_lc_pipeline,
            DecodeMatvecKernel::SplitK => &self.matvec_q4_splitk_pipeline,
            DecodeMatvecKernel::Ggml | DecodeMatvecKernel::GgmlExt => unreachable!(),
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(w_buf), w_offset);
        encoder.set_buffer(1, Some(x_buf), x_offset);
        encoder.set_buffer(2, Some(y_buf), y_offset);
        encoder.set_bytes(3, 4, &m as *const u32 as *const _);
        encoder.set_bytes(4, 4, &k as *const u32 as *const _);

        let (num_tgs, tg_size) = Self::matvec_q4_dispatch(kernel, m);
        encoder.dispatch_thread_groups(num_tgs, tg_size);
    }

    /// Fused gate+up Q4_0 GEMV: one dispatch, shared x loads, separate outputs.
    pub fn encode_matvec_q4_dual_ggml_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate: &BufferView,
        up: &BufferView,
        x_buf: &Buffer,
        x_offset: u64,
        gate_out: &Buffer,
        gate_offset: u64,
        up_out: &Buffer,
        up_offset: u64,
        m: u32,
        k: u32,
    ) {
        use crate::ggml_gemv::{mul_mv_args, mul_mv_dispatch};
        let args = mul_mv_args(m, k);
        encoder.set_compute_pipeline_state(&self.matvec_ggml_q4_dual_pipeline);
        encoder.set_buffer(0, Some(&gate.buffer), gate.offset);
        encoder.set_buffer(1, Some(&up.buffer), up.offset);
        encoder.set_buffer(2, Some(x_buf), x_offset);
        encoder.set_buffer(3, Some(gate_out), gate_offset);
        encoder.set_buffer(4, Some(up_out), up_offset);
        encoder.set_bytes(
            5,
            std::mem::size_of::<crate::ggml_gemv::GgmlMulMvArgs>() as u64,
            &args as *const _ as *const _,
        );
        let (tg_x, tg_y, tg_z, tw, nsg) = mul_mv_dispatch(m, 1);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(tg_x, tg_y, tg_z),
            metal::MTLSize::new(tw, nsg, 1),
        );
    }

    /// Fused gate+up+GeLU Q4_0 GEMV: gelu = GeLU(W_gate @ x) * (W_up @ x).
    /// Writes a single gelu scratch vector, avoiding separate gate/up outputs
    /// and a follow-up GeLU multiply dispatch.
    pub fn encode_matvec_ggml_q4_gelu_mul_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate: &BufferView,
        up: &BufferView,
        x_buf: &Buffer,
        x_offset: u64,
        gelu_out: &Buffer,
        gelu_offset: u64,
        m: u32,
        k: u32,
    ) {
        use crate::ggml_gemv::mul_mv_args;
        let args = mul_mv_args(m, k);
        let r2s4 = mlp_fused_gelu_ggml_r2s4_enabled();
        encoder.set_compute_pipeline_state(if r2s4 {
            &self.matvec_ggml_q4_gelu_mul_r2s4_pipeline
        } else {
            &self.matvec_ggml_q4_gelu_mul_pipeline
        });
        encoder.set_buffer(0, Some(&gate.buffer), gate.offset);
        encoder.set_buffer(1, Some(&up.buffer), up.offset);
        encoder.set_buffer(2, Some(x_buf), x_offset);
        encoder.set_buffer(3, Some(gelu_out), gelu_offset);
        encoder.set_bytes(
            4,
            std::mem::size_of::<crate::ggml_gemv::GgmlMulMvArgs>() as u64,
            &args as *const _ as *const _,
        );
        if r2s4 {
            // NR0=2, NSG=4 => 8 rows per threadgroup, 4 simdgroups of 32 threads.
            let rows_per_tg = 8u64;
            let tg_x = ((m as u64 + rows_per_tg - 1) / rows_per_tg).max(1);
            encoder.dispatch_thread_groups(
                metal::MTLSize::new(tg_x, 1, 1),
                metal::MTLSize::new(32, 4, 1),
            );
        } else {
            use crate::ggml_gemv::mul_mv_dispatch;
            let (tg_x, tg_y, tg_z, tw, nsg) = mul_mv_dispatch(m, 1);
            encoder.dispatch_thread_groups(
                metal::MTLSize::new(tg_x, tg_y, tg_z),
                metal::MTLSize::new(tw, nsg, 1),
            );
        }
    }

    // ─── K-quant matvec dispatch (Q4_K / Q6_K) ──────────────────────────────

    /// Native K-quant weight matvec for community Q4_K_M GGUFs.
    /// `weight.format` selects Q4_K vs Q6_K; x and y are f32. `batch` is 1 for
    /// decode and `seq_len` for prefill (one kernel covers both via tgpig.y).
    pub fn encode_matvec_qk_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        m: u32,
        k: u32,
        batch: u32,
    ) {
        use crate::ggml_gemv::{
            mul_mv_args_k, mul_mv_k_dispatch, Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES,
        };
        let (pipeline, block_bytes) = match weight.format {
            weight_fmt::Q4_K => (&self.matvec_ggml_q4k_pipeline, Q4_K_BLOCK_BYTES),
            weight_fmt::Q6_K => (&self.matvec_ggml_q6k_pipeline, Q6_K_BLOCK_BYTES),
            other => panic!("encode_matvec_qk_at_view: not a K-quant format ({})", other),
        };
        let args = mul_mv_args_k(m, k, batch, block_bytes);
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&weight.buffer), weight.offset);
        encoder.set_buffer(1, Some(x_buf), x_offset);
        encoder.set_buffer(2, Some(y_buf), y_offset);
        encoder.set_bytes(
            3,
            std::mem::size_of::<crate::ggml_gemv::GgmlMulMvArgs>() as u64,
            &args as *const _ as *const _,
        );
        let (tg_x, tg_y, tg_z, tw, nsg) = mul_mv_k_dispatch(m, batch);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(tg_x, tg_y, tg_z),
            metal::MTLSize::new(tw, nsg, 1),
        );
    }

    /// Fused gate+up Q4_K GEMV with GeLU(gate)*up — one dispatch, shared x loads.
    /// Both `gate` and `up` must be Q4_K. Output is `gelu_out[i] = GeLU(gate·x)·(up·x)`.
    pub fn encode_matvec_qk_gelu_mul_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate: &BufferView,
        up: &BufferView,
        x_buf: &Buffer,
        x_offset: u64,
        gelu_out: &Buffer,
        gelu_offset: u64,
        m: u32,
        k: u32,
    ) {
        debug_assert_eq!(gate.format, weight_fmt::Q4_K);
        debug_assert_eq!(up.format, weight_fmt::Q4_K);
        use crate::ggml_gemv::{mul_mv_args_k, mul_mv_k_dispatch, Q4_K_BLOCK_BYTES};
        let args = mul_mv_args_k(m, k, 1, Q4_K_BLOCK_BYTES);
        encoder.set_compute_pipeline_state(&self.matvec_ggml_q4k_gelu_mul_pipeline);
        encoder.set_buffer(0, Some(&gate.buffer), gate.offset);
        encoder.set_buffer(1, Some(&up.buffer), up.offset);
        encoder.set_buffer(2, Some(x_buf), x_offset);
        encoder.set_buffer(3, Some(gelu_out), gelu_offset);
        encoder.set_bytes(
            4,
            std::mem::size_of::<crate::ggml_gemv::GgmlMulMvArgs>() as u64,
            &args as *const _ as *const _,
        );
        let (tg_x, tg_y, tg_z, tw, nsg) = mul_mv_k_dispatch(m, 1);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(tg_x, tg_y, tg_z),
            metal::MTLSize::new(tw, nsg, 1),
        );
    }

    /// Fused pre-FF RMSNorm + Q4_K gate∥up + GeLU(gate)*up (K-quant MLP decode).
    pub fn encode_rmsnorm_qk_gelu_mul_kquant_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate: &BufferView,
        up: &BufferView,
        hidden: &Buffer,
        hidden_offset: u64,
        norm_weight: &BufferView,
        inv_rms_buf: &Buffer,
        gelu_out: &Buffer,
        gelu_offset: u64,
        m: u32,
        k: u32,
        eps: f32,
    ) {
        debug_assert_eq!(gate.format, weight_fmt::Q4_K);
        debug_assert_eq!(up.format, weight_fmt::Q4_K);
        self.encode_rmsnorm_inv_rms_at_view(
            encoder,
            hidden,
            hidden_offset,
            inv_rms_buf,
            0,
            k,
            eps,
        );
        encoder.set_compute_pipeline_state(&self.matvec_ggml_q4k_rmsnorm_gelu_mul_pipeline);
        encoder.set_buffer(0, Some(&gate.buffer), gate.offset);
        encoder.set_buffer(1, Some(&up.buffer), up.offset);
        encoder.set_buffer(2, Some(hidden), hidden_offset);
        encoder.set_buffer(3, Some(&norm_weight.buffer), norm_weight.offset);
        encoder.set_buffer(4, Some(inv_rms_buf), 0);
        encoder.set_buffer(5, Some(gelu_out), gelu_offset);
        encoder.set_bytes(6, 4, &m as *const u32 as *const _);
        encoder.set_bytes(7, 4, &k as *const u32 as *const _);
        let (tg_x, tg_y, tg_z, tw, nsg) = crate::ggml_gemv::mul_mv_k_dispatch(m, 1);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(tg_x, tg_y, tg_z),
            metal::MTLSize::new(tw, nsg, 1),
        );
    }

    // ─── Q3_0 matvec dispatch ────────────────────────────────────────────────

    /// Q3_0 weight matvec: W is 3-bit quantized, x and y are f32.
    pub fn encode_matvec_q3_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        m: u32,
        k: u32,
    ) {
        use crate::ggml_gemv::{mul_mv_args, mul_mv_dispatch};
        let args = mul_mv_args(m, k);
        encoder.set_compute_pipeline_state(&self.matvec_ggml_q3_pipeline);
        encoder.set_buffer(0, Some(&weight.buffer), weight.offset);
        encoder.set_buffer(1, Some(x_buf), x_offset);
        encoder.set_buffer(2, Some(y_buf), y_offset);
        encoder.set_bytes(
            3,
            std::mem::size_of::<crate::ggml_gemv::GgmlMulMvArgs>() as u64,
            &args as *const _ as *const _,
        );
        let (tg_x, tg_y, tg_z, tw, nsg) = mul_mv_dispatch(m, 1);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(tg_x, tg_y, tg_z),
            metal::MTLSize::new(tw, nsg, 1),
        );
    }

    /// Fused gate+up Q3_0 GEMV with GeLU(gate)*up — one dispatch with shared x loads.
    pub fn encode_matvec_ggml_q3_gelu_mul_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate: &BufferView,
        up: &BufferView,
        x_buf: &Buffer,
        x_offset: u64,
        gelu_out: &Buffer,
        gelu_offset: u64,
        m: u32,
        k: u32,
    ) {
        use crate::ggml_gemv::mul_mv_args;
        let args = mul_mv_args(m, k);
        encoder.set_compute_pipeline_state(&self.matvec_ggml_q3_gelu_mul_pipeline);
        encoder.set_buffer(0, Some(&gate.buffer), gate.offset);
        encoder.set_buffer(1, Some(&up.buffer), up.offset);
        encoder.set_buffer(2, Some(x_buf), x_offset);
        encoder.set_buffer(3, Some(gelu_out), gelu_offset);
        encoder.set_bytes(
            4,
            std::mem::size_of::<crate::ggml_gemv::GgmlMulMvArgs>() as u64,
            &args as *const _ as *const _,
        );
        use crate::ggml_gemv::mul_mv_dispatch;
        let (tg_x, tg_y, tg_z, tw, nsg) = mul_mv_dispatch(m, 1);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(tg_x, tg_y, tg_z),
            metal::MTLSize::new(tw, nsg, 1),
        );
    }

    fn encode_matvec_ggml_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        w_buf: &Buffer,
        w_offset: u64,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        m: u32,
        k: u32,
        kernel: DecodeMatvecKernel,
    ) {
        use crate::ggml_gemv::{mul_mv_args, mul_mv_ext_args, mul_mv_ext_dispatch, mul_mv_dispatch, GgmlMatvecKind};

        encoder.set_buffer(0, Some(w_buf), w_offset);
        encoder.set_buffer(1, Some(x_buf), x_offset);
        encoder.set_buffer(2, Some(y_buf), y_offset);

        match kernel {
            DecodeMatvecKernel::Ggml => {
                let args = mul_mv_args(m, k);
                encoder.set_compute_pipeline_state(&self.matvec_ggml_q4_pipeline);
                encoder.set_bytes(
                    3,
                    std::mem::size_of::<crate::ggml_gemv::GgmlMulMvArgs>() as u64,
                    &args as *const _ as *const _,
                );
                let (tg_x, tg_y, tg_z, tw, nsg) = mul_mv_dispatch(m, 1);
                encoder.dispatch_thread_groups(
                    metal::MTLSize::new(tg_x, tg_y, tg_z),
                    metal::MTLSize::new(tw, nsg, 1),
                );
            }
            DecodeMatvecKernel::GgmlExt => {
                let kind = GgmlMatvecKind::pick_ext_nxpsg(k, 1);
                let pipeline = match kind {
                    GgmlMatvecKind::ExtNx4 => &self.matvec_ggml_ext_q4_nx4_pipeline,
                    GgmlMatvecKind::ExtNx8 => &self.matvec_ggml_ext_q4_nx8_pipeline,
                    GgmlMatvecKind::ExtNx16 => &self.matvec_ggml_ext_q4_nx16_pipeline,
                    GgmlMatvecKind::MulMv => &self.matvec_ggml_q4_pipeline,
                };
                let args = mul_mv_ext_args(m, k, 1, kind);
                encoder.set_compute_pipeline_state(pipeline);
                encoder.set_bytes(
                    3,
                    std::mem::size_of::<crate::ggml_gemv::GgmlMulMvExtArgs>() as u64,
                    &args as *const _ as *const _,
                );
                let (tg_x, tg_y, tg_z, tw, nsg) = mul_mv_ext_dispatch(m, 1, args.nxpsg);
                encoder.dispatch_thread_groups(
                    metal::MTLSize::new(tg_x, tg_y, tg_z),
                    metal::MTLSize::new(tw, nsg as u64, 1),
                );
            }
            _ => unreachable!(),
        }
    }

    fn matvec_q4_dispatch(kernel: DecodeMatvecKernel, m: u32) -> (MTLSize, MTLSize) {
        const SG_PER_TG: u64 = 8;
        match kernel {
            DecodeMatvecKernel::Auto => unreachable!("resolve_for_shape before dispatch"),
            DecodeMatvecKernel::SplitK => (
                MTLSize::new(m as u64, 1, 1),
                MTLSize::new(SG_PER_TG * 32, 1, 1),
            ),
            // ane-infer / llama.cpp: 4 simdgroups, 2 output rows per threadgroup.
            DecodeMatvecKernel::Lc => (
                MTLSize::new(((m as u64) + 1) / 2, 1, 1),
                MTLSize::new(128, 1, 1),
            ),
            k => {
                let rows_per_tg = SG_PER_TG * k.rows_per_sg();
                (
                    MTLSize::new((m as u64 + rows_per_tg - 1) / rows_per_tg, 1, 1),
                    MTLSize::new(SG_PER_TG * 32, 1, 1),
                )
            }
        }
    }

    /// Fused dual Q4 matvec: y0=W0@x, y1=W1@x with one x load per K block (fast/4 geometry).
    pub fn encode_matvec_q4_dual_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        w0: &BufferView,
        w1: &BufferView,
        x_buf: &Buffer,
        y0_buf: &Buffer,
        y1_buf: &Buffer,
        m: u32,
        k: u32,
    ) {
        self.encode_matvec_q4_dual_at_view(
            encoder, w0, w1, x_buf, 0, y0_buf, 0, y1_buf, 0, m, k,
        );
    }

    pub fn encode_matvec_q4_dual_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        w0: &BufferView,
        w1: &BufferView,
        x_buf: &Buffer,
        x_offset: u64,
        y0_buf: &Buffer,
        y0_offset: u64,
        y1_buf: &Buffer,
        y1_offset: u64,
        m: u32,
        k: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.matvec_q4_dual_pipeline);
        encoder.set_buffer(0, Some(&w0.buffer), w0.offset);
        encoder.set_buffer(1, Some(&w1.buffer), w1.offset);
        encoder.set_buffer(2, Some(x_buf), x_offset);
        encoder.set_buffer(3, Some(y0_buf), y0_offset);
        encoder.set_buffer(4, Some(y1_buf), y1_offset);
        encoder.set_bytes(5, 4, &m as *const u32 as *const _);
        encoder.set_bytes(6, 4, &k as *const u32 as *const _);
        const SG_PER_TG: u64 = 8;
        let rows_per_tg = SG_PER_TG * 4;
        let num_tgs = MTLSize::new((m as u64 + rows_per_tg - 1) / rows_per_tg, 1, 1);
        let tg_size = MTLSize::new(SG_PER_TG * 32, 1, 1);
        encoder.dispatch_thread_groups(num_tgs, tg_size);
    }

    /// Fused pre-attn RMSNorm + Q4 Q projection (shared-KV layers).
    /// inv_rms once + Q matvec; skips normed scratch.
    pub fn encode_rmsnorm_q_q4_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        hidden: &Buffer,
        norm_weight: &BufferView,
        inv_rms_buf: &Buffer,
        q_weight: &BufferView,
        q_out: &Buffer,
        m_q: u32,
        k: u32,
        eps: f32,
    ) {
        self.encode_rmsnorm_inv_rms_at_view(encoder, hidden, 0, inv_rms_buf, 0, k, eps);
        encoder.set_compute_pipeline_state(&self.matvec_q_rmsnorm_inv_q4_pipeline);
        encoder.set_buffer(0, Some(hidden), 0);
        encoder.set_buffer(1, Some(&norm_weight.buffer), norm_weight.offset);
        encoder.set_buffer(2, Some(inv_rms_buf), 0);
        encoder.set_buffer(3, Some(&q_weight.buffer), q_weight.offset);
        encoder.set_buffer(4, Some(q_out), 0);
        encoder.set_bytes(5, 4, &m_q as *const u32 as *const _);
        encoder.set_bytes(6, 4, &k as *const u32 as *const _);
        let num_tgs = MTLSize::new((m_q as u64 + Q4_MATVEC_ROWS_PER_TG as u64 - 1) / Q4_MATVEC_ROWS_PER_TG as u64, 1, 1);
        let tg_size = MTLSize::new(256, 1, 1);
        encoder.dispatch_thread_groups(num_tgs, tg_size);
    }

    /// Fused pre-attn RMSNorm + Q4 Q/K/V projections (has_kv layers).
    pub fn encode_rmsnorm_qkv_q4_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        hidden: &Buffer,
        norm_weight: &BufferView,
        inv_rms_buf: &Buffer,
        q_weight: &BufferView,
        k_weight: &BufferView,
        v_weight: &BufferView,
        q_out: &Buffer,
        k_out: &Buffer,
        v_out: &Buffer,
        m_q: u32,
        m_kv: u32,
        k: u32,
        eps: f32,
    ) {
        self.encode_rmsnorm_inv_rms_at_view(encoder, hidden, 0, inv_rms_buf, 0, k, eps);
        encoder.set_compute_pipeline_state(&self.matvec_qkv_rmsnorm_inv_q4_pipeline);
        encoder.set_buffer(0, Some(hidden), 0);
        encoder.set_buffer(1, Some(&norm_weight.buffer), norm_weight.offset);
        encoder.set_buffer(2, Some(inv_rms_buf), 0);
        encoder.set_buffer(3, Some(&q_weight.buffer), q_weight.offset);
        encoder.set_buffer(4, Some(&k_weight.buffer), k_weight.offset);
        encoder.set_buffer(5, Some(&v_weight.buffer), v_weight.offset);
        encoder.set_buffer(6, Some(q_out), 0);
        encoder.set_buffer(7, Some(k_out), 0);
        encoder.set_buffer(8, Some(v_out), 0);
        encoder.set_bytes(9, 4, &m_q as *const u32 as *const _);
        encoder.set_bytes(10, 4, &m_kv as *const u32 as *const _);
        encoder.set_bytes(11, 4, &k as *const u32 as *const _);
        let q_tgs = (m_q + Q4_MATVEC_ROWS_PER_TG - 1) / Q4_MATVEC_ROWS_PER_TG;
        let kv_tgs = (m_kv + Q4_MATVEC_ROWS_PER_TG - 1) / Q4_MATVEC_ROWS_PER_TG;
        let total_tgs = q_tgs + 2 * kv_tgs;
        let num_tgs = MTLSize::new(total_tgs as u64, 1, 1);
        let tg_size = MTLSize::new(256, 1, 1);
        encoder.dispatch_thread_groups(num_tgs, tg_size);
    }

    /// Fused pre-attn RMSNorm + K-quant Q projection (shared-KV layers).
    pub fn encode_rmsnorm_q_kquant_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        hidden: &Buffer,
        norm_weight: &BufferView,
        inv_rms_buf: &Buffer,
        q_weight: &BufferView,
        q_out: &Buffer,
        m_q: u32,
        k: u32,
        eps: f32,
    ) {
        self.encode_rmsnorm_inv_rms_at_view(encoder, hidden, 0, inv_rms_buf, 0, k, eps);
        encoder.set_compute_pipeline_state(&self.matvec_q_rmsnorm_inv_kquant_pipeline);
        encoder.set_buffer(0, Some(hidden), 0);
        encoder.set_buffer(1, Some(&norm_weight.buffer), norm_weight.offset);
        encoder.set_buffer(2, Some(inv_rms_buf), 0);
        encoder.set_buffer(3, Some(&q_weight.buffer), q_weight.offset);
        encoder.set_buffer(4, Some(q_out), 0);
        encoder.set_bytes(5, 4, &m_q as *const u32 as *const _);
        encoder.set_bytes(6, 4, &k as *const u32 as *const _);
        let q_fmt = q_weight.format as u32;
        encoder.set_bytes(7, 4, &q_fmt as *const u32 as *const _);
        let (tg_x, tw, nsg) = crate::ggml_gemv::kquant_fused_qkv_dispatch(m_q, 0, false);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(tg_x, 1, 1),
            metal::MTLSize::new(tw, nsg, 1),
        );
    }

    /// Fused pre-attn RMSNorm + K-quant Q/K/V projections (has_kv layers).
    pub fn encode_rmsnorm_qkv_kquant_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        hidden: &Buffer,
        norm_weight: &BufferView,
        inv_rms_buf: &Buffer,
        q_weight: &BufferView,
        k_weight: &BufferView,
        v_weight: &BufferView,
        q_out: &Buffer,
        k_out: &Buffer,
        v_out: &Buffer,
        m_q: u32,
        m_kv: u32,
        k: u32,
        eps: f32,
    ) {
        self.encode_rmsnorm_inv_rms_at_view(encoder, hidden, 0, inv_rms_buf, 0, k, eps);
        encoder.set_compute_pipeline_state(&self.matvec_qkv_rmsnorm_inv_kquant_pipeline);
        encoder.set_buffer(0, Some(hidden), 0);
        encoder.set_buffer(1, Some(&norm_weight.buffer), norm_weight.offset);
        encoder.set_buffer(2, Some(inv_rms_buf), 0);
        encoder.set_buffer(3, Some(&q_weight.buffer), q_weight.offset);
        encoder.set_buffer(4, Some(&k_weight.buffer), k_weight.offset);
        encoder.set_buffer(5, Some(&v_weight.buffer), v_weight.offset);
        encoder.set_buffer(6, Some(q_out), 0);
        encoder.set_buffer(7, Some(k_out), 0);
        encoder.set_buffer(8, Some(v_out), 0);
        encoder.set_bytes(9, 4, &m_q as *const u32 as *const _);
        encoder.set_bytes(10, 4, &m_kv as *const u32 as *const _);
        encoder.set_bytes(11, 4, &k as *const u32 as *const _);
        let q_fmt = q_weight.format as u32;
        let k_fmt = k_weight.format as u32;
        let v_fmt = v_weight.format as u32;
        encoder.set_bytes(12, 4, &q_fmt as *const u32 as *const _);
        encoder.set_bytes(13, 4, &k_fmt as *const u32 as *const _);
        encoder.set_bytes(14, 4, &v_fmt as *const u32 as *const _);
        let (tg_x, tw, nsg) = crate::ggml_gemv::kquant_fused_qkv_dispatch(m_q, m_kv, true);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(tg_x, 1, 1),
            metal::MTLSize::new(tw, nsg, 1),
        );
    }

    /// Fused gate+up Q4 matvec + GeLU(gate)*up → single output (MLP decode).
    pub fn encode_matvec_q4_dual_gelu_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        w0: &BufferView,
        w1: &BufferView,
        x_buf: &Buffer,
        y_buf: &Buffer,
        m: u32,
        k: u32,
    ) {
        self.encode_matvec_q4_dual_gelu_at_view(
            encoder, w0, w1, x_buf, 0, y_buf, 0, m, k,
        );
    }

    pub fn encode_matvec_q4_dual_gelu_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        w0: &BufferView,
        w1: &BufferView,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        m: u32,
        k: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.matvec_q4_dual_gelu_pipeline);
        encoder.set_buffer(0, Some(&w0.buffer), w0.offset);
        encoder.set_buffer(1, Some(&w1.buffer), w1.offset);
        encoder.set_buffer(2, Some(x_buf), x_offset);
        encoder.set_buffer(3, Some(y_buf), y_offset);
        encoder.set_bytes(4, 4, &m as *const u32 as *const _);
        encoder.set_bytes(5, 4, &k as *const u32 as *const _);
        const SG_PER_TG: u64 = 8;
        let rows_per_tg = SG_PER_TG * 4;
        let num_tgs = MTLSize::new((m as u64 + rows_per_tg - 1) / rows_per_tg, 1, 1);
        let tg_size = MTLSize::new(SG_PER_TG * 32, 1, 1);
        encoder.dispatch_thread_groups(num_tgs, tg_size);
    }

    /// Gate∥up interleaved Q4 matvec + GeLU(gate)*up — gate/up rows adjacent in memory.
    pub fn encode_matvec_q4_interleaved_gelu_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate_up: &BufferView,
        x_buf: &Buffer,
        y_buf: &Buffer,
        m: u32,
        k: u32,
    ) {
        self.encode_matvec_q4_interleaved_gelu_at_view(
            encoder, gate_up, x_buf, 0, y_buf, 0, m, k,
        );
    }

    pub fn encode_matvec_q4_interleaved_gelu_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate_up: &BufferView,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        m: u32,
        k: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.matvec_q4_interleaved_gelu_pipeline);
        encoder.set_buffer(0, Some(&gate_up.buffer), gate_up.offset);
        encoder.set_buffer(1, Some(x_buf), x_offset);
        encoder.set_buffer(2, Some(y_buf), y_offset);
        encoder.set_bytes(3, 4, &m as *const u32 as *const _);
        encoder.set_bytes(4, 4, &k as *const u32 as *const _);
        const SG_PER_TG: u64 = 8;
        let rows_per_tg = SG_PER_TG * 4;
        let num_tgs = MTLSize::new((m as u64 + rows_per_tg - 1) / rows_per_tg, 1, 1);
        let tg_size = MTLSize::new(SG_PER_TG * 32, 1, 1);
        encoder.dispatch_thread_groups(num_tgs, tg_size);
    }

    pub fn encode_rmsnorm_inv_rms_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        x_offset: u64,
        inv_rms_buf: &Buffer,
        inv_rms_offset: u64,
        dim: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.rmsnorm_inv_rms_pipeline);
        encoder.set_buffer(0, Some(x_buf), x_offset);
        encoder.set_buffer(1, Some(inv_rms_buf), inv_rms_offset);
        encoder.set_bytes(2, 4, &dim as *const u32 as *const _);
        encoder.set_bytes(3, 4, &eps as *const f32 as *const _);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_matvec_q4_interleaved_gelu_hidden_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate_up: &BufferView,
        hidden_buf: &Buffer,
        hidden_offset: u64,
        weight: &BufferView,
        inv_rms_buf: &Buffer,
        inv_rms_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        m: u32,
        k: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.matvec_q4_interleaved_gelu_hidden_pipeline);
        encoder.set_buffer(0, Some(&gate_up.buffer), gate_up.offset);
        encoder.set_buffer(1, Some(hidden_buf), hidden_offset);
        encoder.set_buffer(2, Some(&weight.buffer), weight.offset);
        encoder.set_buffer(3, Some(inv_rms_buf), inv_rms_offset);
        encoder.set_buffer(4, Some(y_buf), y_offset);
        encoder.set_bytes(5, 4, &m as *const u32 as *const _);
        encoder.set_bytes(6, 4, &k as *const u32 as *const _);
        const SG_PER_TG: u64 = 8;
        let rows_per_tg = SG_PER_TG * 4;
        let num_tgs = MTLSize::new((m as u64 + rows_per_tg - 1) / rows_per_tg, 1, 1);
        let tg_size = MTLSize::new(SG_PER_TG * 32, 1, 1);
        encoder.dispatch_thread_groups(num_tgs, tg_size);
    }

    pub fn encode_matvec_q4_interleaved_gelu_f16_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate_up: &BufferView,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        m: u32,
        k: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.matvec_q4_interleaved_gelu_f16_pipeline);
        encoder.set_buffer(0, Some(&gate_up.buffer), gate_up.offset);
        encoder.set_buffer(1, Some(x_buf), x_offset);
        encoder.set_buffer(2, Some(y_buf), y_offset);
        encoder.set_bytes(3, 4, &m as *const u32 as *const _);
        encoder.set_bytes(4, 4, &k as *const u32 as *const _);
        const SG_PER_TG: u64 = 8;
        let rows_per_tg = SG_PER_TG * 4;
        let num_tgs = MTLSize::new((m as u64 + rows_per_tg - 1) / rows_per_tg, 1, 1);
        let tg_size = MTLSize::new(SG_PER_TG * 32, 1, 1);
        encoder.dispatch_thread_groups(num_tgs, tg_size);
    }

    pub fn encode_matvec_ggml_q4_f16x_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        w_buf: &Buffer,
        w_offset: u64,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        m: u32,
        k: u32,
    ) {
        use crate::ggml_gemv::{mul_mv_args, mul_mv_dispatch};

        encoder.set_compute_pipeline_state(&self.matvec_ggml_q4_f16x_pipeline);
        encoder.set_buffer(0, Some(w_buf), w_offset);
        encoder.set_buffer(1, Some(x_buf), x_offset);
        encoder.set_buffer(2, Some(y_buf), y_offset);
        let args = mul_mv_args(m, k);
        encoder.set_bytes(
            3,
            std::mem::size_of::<crate::ggml_gemv::GgmlMulMvArgs>() as u64,
            &args as *const _ as *const _,
        );
        let (tg_x, tg_y, tg_z, tw, nsg) = mul_mv_dispatch(m, 1);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(tg_x, tg_y, tg_z),
            metal::MTLSize::new(tw, nsg, 1),
        );
    }

    /// Down matvec reading f16 activations (ggml Q4 weights).
    fn encode_mlp_down_q4_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        down: &BufferView,
        gelu_buf: &Buffer,
        gelu_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        hidden_size: u32,
        intermediate_size: u32,
    ) {
        if mlp_gelu_f16_enabled() {
            self.encode_matvec_ggml_q4_f16x_at_view(
                encoder,
                &down.buffer,
                down.offset,
                gelu_buf,
                gelu_offset,
                y_buf,
                y_offset,
                hidden_size,
                intermediate_size,
            );
        } else {
            self.encode_matvec_q4_at_view(
                encoder,
                down,
                gelu_buf,
                gelu_offset,
                y_buf,
                y_offset,
                hidden_size,
                intermediate_size,
            );
        }
    }

    /// Fused gate∥up+GeLU+down from post-attn hidden (no normed_buf round-trip).
    pub fn encode_mlp_fused_q4_gelu_down_packed_from_hidden_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate_up: &BufferView,
        down: &BufferView,
        ff_norm_weight: &BufferView,
        hidden_buf: &Buffer,
        hidden_offset: u64,
        inv_rms_buf: &Buffer,
        inv_rms_offset: u64,
        gelu_buf: &Buffer,
        gelu_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        hidden_size: u32,
        intermediate_size: u32,
        eps: f32,
    ) {
        self.encode_rmsnorm_inv_rms_at_view(
            encoder,
            hidden_buf,
            hidden_offset,
            inv_rms_buf,
            inv_rms_offset,
            hidden_size,
            eps,
        );
        self.encode_matvec_q4_interleaved_gelu_hidden_at_view(
            encoder,
            gate_up,
            hidden_buf,
            hidden_offset,
            ff_norm_weight,
            inv_rms_buf,
            inv_rms_offset,
            gelu_buf,
            gelu_offset,
            intermediate_size,
            hidden_size,
        );
        self.encode_mlp_down_q4_at_view(
            encoder,
            down,
            gelu_buf,
            gelu_offset,
            y_buf,
            y_offset,
            hidden_size,
            intermediate_size,
        );
    }

    /// Fused gate+up+GeLU+down Q4 MLP: dual_gelu into scratch, then down matvec
    /// (same command-buffer encoder, sequential dispatches — no gelu DRAM round-trip
    /// between separate Rust call sites).
    pub fn encode_mlp_fused_q4_gelu_down_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate: &BufferView,
        up: &BufferView,
        down: &BufferView,
        x_buf: &Buffer,
        gelu_buf: &Buffer,
        y_buf: &Buffer,
        hidden_size: u32,
        intermediate_size: u32,
    ) {
        self.encode_mlp_fused_q4_gelu_down_at_view(
            encoder,
            gate,
            up,
            down,
            x_buf,
            0,
            gelu_buf,
            0,
            y_buf,
            0,
            hidden_size,
            intermediate_size,
        );
    }

    pub fn encode_mlp_fused_q4_gelu_down_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate: &BufferView,
        up: &BufferView,
        down: &BufferView,
        x_buf: &Buffer,
        x_offset: u64,
        gelu_buf: &Buffer,
        gelu_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        hidden_size: u32,
        intermediate_size: u32,
    ) {
        self.encode_matvec_q4_dual_gelu_at_view(
            encoder,
            gate,
            up,
            x_buf,
            x_offset,
            gelu_buf,
            gelu_offset,
            intermediate_size,
            hidden_size,
        );
        self.encode_mlp_down_q4_at_view(
            encoder,
            down,
            gelu_buf,
            gelu_offset,
            y_buf,
            y_offset,
            hidden_size,
            intermediate_size,
        );
    }

    /// Fused interleaved gate∥up+GeLU+down Q4 MLP (packed weights).
    pub fn encode_mlp_fused_q4_gelu_down_packed_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate_up: &BufferView,
        down: &BufferView,
        x_buf: &Buffer,
        x_offset: u64,
        gelu_buf: &Buffer,
        gelu_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        hidden_size: u32,
        intermediate_size: u32,
    ) {
        if mlp_gelu_f16_enabled() {
            self.encode_matvec_q4_interleaved_gelu_f16_at_view(
                encoder,
                gate_up,
                x_buf,
                x_offset,
                gelu_buf,
                gelu_offset,
                intermediate_size,
                hidden_size,
            );
        } else {
            self.encode_matvec_q4_interleaved_gelu_at_view(
                encoder,
                gate_up,
                x_buf,
                x_offset,
                gelu_buf,
                gelu_offset,
                intermediate_size,
                hidden_size,
            );
        }
        self.encode_mlp_down_q4_at_view(
            encoder,
            down,
            gelu_buf,
            gelu_offset,
            y_buf,
            y_offset,
            hidden_size,
            intermediate_size,
        );
    }

    /// Fused gate+up (separate ggml Q4 matvecs) + GeLU + down.
    /// By default uses a single kernel dispatch that computes GeLU(gate) * up
    /// and writes the gelu scratch directly, removing gate_out/up_out scratch
    /// buffers and a separate GeLU multiply dispatch. Set
    /// MLP_FUSED_GELU_GGML=0 to fall back to the dual-matvec + GeLU path.
    pub fn encode_mlp_fused_q4_gelu_down_ggml_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate: &BufferView,
        up: &BufferView,
        down: &BufferView,
        x_buf: &Buffer,
        x_offset: u64,
        gate_out: &Buffer,
        up_out: &Buffer,
        gelu_buf: &Buffer,
        gelu_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        hidden_size: u32,
        intermediate_size: u32,
    ) {
        if mlp_fused_gelu_ggml_enabled() {
            self.encode_matvec_ggml_q4_gelu_mul_at_view(
                encoder,
                gate,
                up,
                x_buf,
                x_offset,
                gelu_buf,
                gelu_offset,
                intermediate_size,
                hidden_size,
            );
        } else {
            if mlp_gate_up_dual_ggml_enabled() {
                self.encode_matvec_q4_dual_ggml_at_view(
                    encoder,
                    gate,
                    up,
                    x_buf,
                    x_offset,
                    gate_out,
                    0,
                    up_out,
                    0,
                    intermediate_size,
                    hidden_size,
                );
            } else {
                self.encode_matvec_q4_at_view(
                    encoder, gate, x_buf, x_offset, gate_out, 0, intermediate_size, hidden_size,
                );
                self.encode_matvec_q4_at_view(
                    encoder, up, x_buf, x_offset, up_out, 0, intermediate_size, hidden_size,
                );
            }
            if mlp_gelu_f16_enabled() {
                self.encode_gelu_mul_f16_at(
                    encoder, gate_out, 0, up_out, 0, gelu_buf, gelu_offset, intermediate_size,
                );
            } else {
                self.encode_gelu_mul_at(
                    encoder, gate_out, 0, up_out, 0, gelu_buf, gelu_offset, intermediate_size,
                );
            }
        }
        self.encode_mlp_down_q4_at_view(
            encoder,
            down,
            gelu_buf,
            gelu_offset,
            y_buf,
            y_offset,
            hidden_size,
            intermediate_size,
        );
    }

    /// Pack separate gate/up Q4 rows into interleaved [gate_i, up_i, …] layout.
    pub fn pack_gate_up_interleaved_q4(
        &self,
        gate: &BufferView,
        up: &BufferView,
        m: u32,
        k: u32,
    ) -> BufferView {
        assert_eq!(
            gate.length, up.length,
            "gate/up Q4 tensors must match for interleaved packing"
        );
        let row_bytes = (k as u64 / 32) * 18;
        let pair_bytes = row_bytes * 2;
        let expected_per_tensor = m as u64 * row_bytes;
        let packed_bytes = m as u64 * pair_bytes;
        assert!(
            gate.length >= expected_per_tensor && up.length >= expected_per_tensor,
            "gate/up buffer too small for interleaved pack (m={m} k={k} need={expected_per_tensor} gate={} up={})",
            gate.length,
            up.length
        );
        let gate_bytes = gate.as_bytes();
        let up_bytes = up.as_bytes();
        let mut packed = vec![0u8; packed_bytes as usize];
        for i in 0..m as u64 {
            let dst = (i * pair_bytes) as usize;
            let src = (i * row_bytes) as usize;
            let rb = row_bytes as usize;
            packed[dst..dst + rb].copy_from_slice(&gate_bytes[src..src + rb]);
            packed[dst + rb..dst + 2 * rb].copy_from_slice(&up_bytes[src..src + rb]);
        }
        BufferView::from_buffer(Self::buffer_from_slice_parallel(&self.device, &packed))
    }

    /// Pack gate/up K-quant rows vertically [gate_0..gate_{m-1}, up_0..up_{m-1}] for one mul_mm.
    pub fn pack_gate_up_stacked_kquant(
        &self,
        gate: &BufferView,
        up: &BufferView,
        m: u32,
        k: u32,
    ) -> BufferView {
        use crate::ggml_gemv::{Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES};
        let block_bytes = match gate.format {
            weight_fmt::Q4_K => Q4_K_BLOCK_BYTES,
            weight_fmt::Q6_K => Q6_K_BLOCK_BYTES,
            other => panic!("pack_gate_up_stacked_kquant: not K-quant ({})", other),
        };
        debug_assert_eq!(gate.format, up.format);
        let row_bytes = (k as u64 / 256) * block_bytes;
        let expected_per_tensor = m as u64 * row_bytes;
        let packed_bytes = expected_per_tensor * 2;
        assert!(
            gate.length >= expected_per_tensor && up.length >= expected_per_tensor,
            "gate/up buffer too small for stacked pack (m={m} k={k} need={expected_per_tensor} gate={} up={})",
            gate.length,
            up.length
        );
        let gate_bytes = gate.as_bytes();
        let up_bytes = up.as_bytes();
        let mut packed = vec![0u8; packed_bytes as usize];
        let rb = row_bytes as usize;
        packed[..rb * m as usize].copy_from_slice(&gate_bytes[..rb * m as usize]);
        let up_off = rb * m as usize;
        packed[up_off..up_off + rb * m as usize]
            .copy_from_slice(&up_bytes[..rb * m as usize]);
        BufferView::from_buffer(Self::buffer_from_slice_parallel(&self.device, &packed))
            .with_format(gate.format)
    }

    /// Pack Q/K/V K-quant rows vertically for one prefill mul_mm.
    pub fn pack_qkv_stacked_kquant(
        &self,
        q: &BufferView,
        k: &BufferView,
        v: &BufferView,
        m_q: u32,
        m_kv: u32,
        k_dim: u32,
    ) -> BufferView {
        use crate::ggml_gemv::{Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES};
        let block_bytes = match q.format {
            weight_fmt::Q4_K => Q4_K_BLOCK_BYTES,
            weight_fmt::Q6_K => Q6_K_BLOCK_BYTES,
            other => panic!("pack_qkv_stacked_kquant: not K-quant ({})", other),
        };
        debug_assert_eq!(q.format, k.format);
        debug_assert_eq!(q.format, v.format);
        let row_bytes = (k_dim as u64 / 256) * block_bytes;
        let q_bytes = m_q as u64 * row_bytes;
        let kv_bytes = m_kv as u64 * row_bytes;
        let packed_bytes = q_bytes + 2 * kv_bytes;
        assert!(
            q.length >= q_bytes && k.length >= kv_bytes && v.length >= kv_bytes,
            "q/k/v buffer too small for stacked pack (m_q={m_q} m_kv={m_kv} k={k_dim})"
        );
        let qb = q.as_bytes();
        let kb = k.as_bytes();
        let vb = v.as_bytes();
        let mut packed = vec![0u8; packed_bytes as usize];
        let rb = row_bytes as usize;
        let q_len = rb * m_q as usize;
        let kv_len = rb * m_kv as usize;
        packed[..q_len].copy_from_slice(&qb[..q_len]);
        packed[q_len..q_len + kv_len].copy_from_slice(&kb[..kv_len]);
        packed[q_len + kv_len..q_len + 2 * kv_len].copy_from_slice(&vb[..kv_len]);
        BufferView::from_buffer(Self::buffer_from_slice_parallel(&self.device, &packed))
            .with_format(q.format)
    }

    /// Prefill stacked gate∥up K-quant: one mul_mm [seq, 2*m] + stacked GeLU → gelu_buf.
    pub fn encode_prefill_gate_up_kquant_stacked(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate_up_stacked: &BufferView,
        x_buf: &Buffer,
        gate_up_act_buf: &Buffer,
        gelu_buf: &Buffer,
        m: u32,
        k: u32,
        seq_len: u32,
    ) {
        let m2 = m * 2;
        self.encode_prefill_kquant_projection(
            encoder,
            gate_up_stacked,
            x_buf,
            gate_up_act_buf,
            m2,
            k,
            seq_len,
        );
        self.encode_gelu_mul_stacked_batch(encoder, gate_up_act_buf, gelu_buf, m, seq_len);
    }

    pub fn encode_gelu_mul_stacked_batch(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate_up_stacked: &Buffer,
        out_buf: &Buffer,
        m: u32,
        seq_len: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.gelu_mul_stacked_batch_pipeline);
        encoder.set_buffer(0, Some(gate_up_stacked), 0);
        encoder.set_buffer(1, Some(out_buf), 0);
        encoder.set_bytes(2, 4, &m as *const u32 as *const _);
        encoder.set_bytes(3, 4, &seq_len as *const u32 as *const _);
        let total = (seq_len * m) as u64;
        encoder.dispatch_threads(MTLSize::new(total, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Prefill stacked Q∥K∥V K-quant: one mul_mm [seq, m_q+2*m_kv] into scratch.
    pub fn encode_prefill_qkv_kquant_stacked(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        qkv_stacked: &BufferView,
        x_buf: &Buffer,
        qkv_act_buf: &Buffer,
        m_q: u32,
        m_kv: u32,
        k: u32,
        seq_len: u32,
    ) {
        let m_total = m_q + m_kv * 2;
        self.encode_prefill_kquant_projection(
            encoder,
            qkv_stacked,
            x_buf,
            qkv_act_buf,
            m_total,
            k,
            seq_len,
        );
    }

    pub fn encode_qkv_split_stacked_to_hsd(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        stacked: &Buffer,
        q_buf: &Buffer,
        k_buf: &Buffer,
        v_buf: &Buffer,
        m_q: u32,
        m_kv: u32,
        head_dim: u32,
        num_heads: u32,
        num_kv_heads: u32,
        seq_len: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.qkv_split_stacked_to_hsd_pipeline);
        encoder.set_buffer(0, Some(stacked), 0);
        encoder.set_buffer(1, Some(q_buf), 0);
        encoder.set_buffer(2, Some(k_buf), 0);
        encoder.set_buffer(3, Some(v_buf), 0);
        encoder.set_bytes(4, 4, &m_q as *const u32 as *const _);
        encoder.set_bytes(5, 4, &m_kv as *const u32 as *const _);
        encoder.set_bytes(6, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(7, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(8, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(9, 4, &seq_len as *const u32 as *const _);
        let total_q = seq_len as u64 * num_heads as u64 * head_dim as u64;
        let total_kv = seq_len as u64 * num_kv_heads as u64 * head_dim as u64;
        let total = total_q + 2 * total_kv;
        encoder.dispatch_threads(MTLSize::new(total, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_rmsnorm_hsd_batch_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        weight: &BufferView,
        out_buf: &Buffer,
        head_dim: u32,
        num_heads: u32,
        seq_len: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.rmsnorm_hsd_batch_pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(&weight.buffer), weight.offset);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &seq_len as *const u32 as *const _);
        encoder.set_bytes(5, 4, &eps as *const f32 as *const _);
        let num_rows = num_heads as u64 * seq_len as u64;
        encoder.dispatch_thread_groups(MTLSize::new(num_rows, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_rmsnorm_noweight_hsd_batch(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        out_buf: &Buffer,
        head_dim: u32,
        num_kv_heads: u32,
        seq_len: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.rmsnorm_noweight_hsd_batch_pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(out_buf), 0);
        encoder.set_bytes(2, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(3, 4, &seq_len as *const u32 as *const _);
        encoder.set_bytes(4, 4, &eps as *const f32 as *const _);
        let num_rows = num_kv_heads as u64 * seq_len as u64;
        encoder.dispatch_thread_groups(MTLSize::new(num_rows, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_rmsnorm_shd_to_hsd_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        weight: &BufferView,
        out_buf: &Buffer,
        head_dim: u32,
        num_heads: u32,
        seq_len: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.rmsnorm_shd_to_hsd_pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(&weight.buffer), weight.offset);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(5, 4, &seq_len as *const u32 as *const _);
        encoder.set_bytes(6, 4, &eps as *const f32 as *const _);
        let num_rows = seq_len as u64 * num_heads as u64;
        encoder.dispatch_thread_groups(MTLSize::new(num_rows, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_rmsnorm_noweight_shd_to_hsd(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        out_buf: &Buffer,
        head_dim: u32,
        num_heads: u32,
        seq_len: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.rmsnorm_noweight_shd_to_hsd_pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(out_buf), 0);
        encoder.set_bytes(2, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(3, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(4, 4, &seq_len as *const u32 as *const _);
        encoder.set_bytes(5, 4, &eps as *const f32 as *const _);
        let num_rows = seq_len as u64 * num_heads as u64;
        encoder.dispatch_thread_groups(MTLSize::new(num_rows, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Post-projection: split (if stacked) + QK-norm + layout → HSD for rotary/flash_attn.
    pub fn encode_prefill_qkv_postproj_hsd(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        qkv_stacked_act: Option<&Buffer>,
        q_buf: &Buffer,
        k_buf: &Buffer,
        v_buf: &Buffer,
        q_norm_weight: &BufferView,
        k_norm_weight: &BufferView,
        m_q: u32,
        m_kv: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
        eps: f32,
        stacked: bool,
        has_kv: bool,
    ) {
        if stacked {
            let stacked_buf = qkv_stacked_act.expect("stacked QKV requires qkv_stacked_act buffer");
            self.encode_qkv_split_stacked_to_hsd(
                encoder,
                stacked_buf,
                q_buf,
                k_buf,
                v_buf,
                m_q,
                m_kv,
                head_dim,
                num_heads,
                num_kv_heads,
                seq_len,
            );
            self.encode_rmsnorm_hsd_batch_view(
                encoder,
                q_buf,
                q_norm_weight,
                q_buf,
                head_dim,
                num_heads,
                seq_len,
                eps,
            );
            self.encode_rmsnorm_hsd_batch_view(
                encoder,
                k_buf,
                k_norm_weight,
                k_buf,
                head_dim,
                num_kv_heads,
                seq_len,
                eps,
            );
            self.encode_rmsnorm_noweight_hsd_batch(
                encoder, v_buf, v_buf, head_dim, num_kv_heads, seq_len, eps,
            );
        } else if has_kv {
            self.encode_rmsnorm_shd_to_hsd_view(
                encoder,
                q_buf,
                q_norm_weight,
                q_buf,
                head_dim,
                num_heads,
                seq_len,
                eps,
            );
            self.encode_rmsnorm_shd_to_hsd_view(
                encoder,
                k_buf,
                k_norm_weight,
                k_buf,
                head_dim,
                num_kv_heads,
                seq_len,
                eps,
            );
            self.encode_rmsnorm_noweight_shd_to_hsd(
                encoder, v_buf, v_buf, head_dim, num_kv_heads, seq_len, eps,
            );
        } else {
            self.encode_rmsnorm_shd_to_hsd_view(
                encoder,
                q_buf,
                q_norm_weight,
                q_buf,
                head_dim,
                num_heads,
                seq_len,
                eps,
            );
        }
    }

    pub fn encode_qkv_split_stacked_batch(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        stacked: &Buffer,
        q_buf: &Buffer,
        k_buf: &Buffer,
        v_buf: &Buffer,
        m_q: u32,
        m_kv: u32,
        seq_len: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.qkv_split_stacked_batch_pipeline);
        encoder.set_buffer(0, Some(stacked), 0);
        encoder.set_buffer(1, Some(q_buf), 0);
        encoder.set_buffer(2, Some(k_buf), 0);
        encoder.set_buffer(3, Some(v_buf), 0);
        encoder.set_bytes(4, 4, &m_q as *const u32 as *const _);
        encoder.set_bytes(5, 4, &m_kv as *const u32 as *const _);
        encoder.set_bytes(6, 4, &seq_len as *const u32 as *const _);
        let m_stacked = (m_q + m_kv * 2) as u64;
        let total = m_stacked * seq_len as u64;
        encoder.dispatch_threads(MTLSize::new(total, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Fused PLE gate Q4 matvec + GeLU(gate)*context slice.
    pub fn encode_ple_matvec_gelu_q4_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate_weight: &BufferView,
        hidden_buf: &Buffer,
        context_buf: &Buffer,
        context_offset: u64,
        out_buf: &Buffer,
        ple_dim: u32,
        hidden_size: u32,
    ) {
        self.encode_ple_matvec_gelu_q4_at_view(
            encoder,
            gate_weight,
            hidden_buf,
            0,
            context_buf,
            context_offset,
            out_buf,
            0,
            ple_dim,
            hidden_size,
        );
    }

    pub fn encode_ple_matvec_gelu_q4_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate_weight: &BufferView,
        hidden_buf: &Buffer,
        hidden_offset: u64,
        context_buf: &Buffer,
        context_offset: u64,
        out_buf: &Buffer,
        out_offset: u64,
        ple_dim: u32,
        hidden_size: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.ple_matvec_gelu_q4_pipeline);
        encoder.set_buffer(0, Some(&gate_weight.buffer), gate_weight.offset);
        encoder.set_buffer(1, Some(hidden_buf), hidden_offset);
        encoder.set_buffer(2, Some(context_buf), context_offset);
        encoder.set_buffer(3, Some(out_buf), out_offset);
        encoder.set_bytes(4, 4, &ple_dim as *const u32 as *const _);
        encoder.set_bytes(5, 4, &hidden_size as *const u32 as *const _);
        const SG_PER_TG: u64 = 8;
        let rows_per_tg = SG_PER_TG * 4;
        let num_tgs = MTLSize::new((ple_dim as u64 + rows_per_tg - 1) / rows_per_tg, 1, 1);
        let tg_size = MTLSize::new(SG_PER_TG * 32, 1, 1);
        encoder.dispatch_thread_groups(num_tgs, tg_size);
    }

    pub fn encode_projection_f16_batch(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        w_buf: &Buffer,
        x_buf: &Buffer,
        y_buf: &Buffer,
        m: u32,
        k: u32,
        seq_len: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.projection_f16_batch_pipeline);
        encoder.set_buffer(0, Some(w_buf), 0);
        encoder.set_buffer(1, Some(x_buf), 0);
        encoder.set_buffer(2, Some(y_buf), 0);
        encoder.set_bytes(3, 4, &m as *const u32 as *const _);
        encoder.set_bytes(4, 4, &k as *const u32 as *const _);
        encoder.set_bytes(5, 4, &seq_len as *const u32 as *const _);
        let n_rows_per_tg = 4u64;
        let num_tgs = MTLSize::new(
            (m as u64 + n_rows_per_tg - 1) / n_rows_per_tg,
            seq_len as u64,
            1,
        );
        let tg_size = MTLSize::new(64, 1, 1);
        encoder.dispatch_thread_groups(num_tgs, tg_size);
    }

    pub fn encode_projection_f16_batch_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &Buffer,
        y_buf: &Buffer,
        m: u32,
        k: u32,
        seq_len: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.projection_f16_batch_pipeline);
        encoder.set_buffer(0, Some(&weight.buffer), weight.offset);
        encoder.set_buffer(1, Some(x_buf), 0);
        encoder.set_buffer(2, Some(y_buf), 0);
        encoder.set_bytes(3, 4, &m as *const u32 as *const _);
        encoder.set_bytes(4, 4, &k as *const u32 as *const _);
        encoder.set_bytes(5, 4, &seq_len as *const u32 as *const _);
        let n_rows_per_tg = 4u64;
        let num_tgs = MTLSize::new(
            (m as u64 + n_rows_per_tg - 1) / n_rows_per_tg,
            seq_len as u64,
            1,
        );
        let tg_size = MTLSize::new(64, 1, 1);
        encoder.dispatch_thread_groups(num_tgs, tg_size);
    }

    pub fn encode_projection_q4_batch(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        w_buf: &Buffer,
        x_buf: &Buffer,
        y_buf: &Buffer,
        m: u32,
        k: u32,
        seq_len: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.projection_q4_batch_pipeline);
        encoder.set_buffer(0, Some(w_buf), 0);
        encoder.set_buffer(1, Some(x_buf), 0);
        encoder.set_buffer(2, Some(y_buf), 0);
        encoder.set_bytes(3, 4, &m as *const u32 as *const _);
        encoder.set_bytes(4, 4, &k as *const u32 as *const _);
        encoder.set_bytes(5, 4, &seq_len as *const u32 as *const _);
        let n_rows_per_tg = 4u64;
        let num_tgs = MTLSize::new(
            (m as u64 + n_rows_per_tg - 1) / n_rows_per_tg,
            seq_len as u64,
            1,
        );
        let tg_size = MTLSize::new(64, 1, 1);
        encoder.dispatch_thread_groups(num_tgs, tg_size);
    }

    pub fn encode_projection_q4_batch_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &Buffer,
        y_buf: &Buffer,
        m: u32,
        k: u32,
        seq_len: u32,
    ) {
        // Native K-quant weights (Q4_K_M) route to the K-quant matvec, which
        // covers the batch via its tgpig.y dimension.
        if matches!(weight.format, weight_fmt::Q4_K | weight_fmt::Q6_K) {
            self.encode_matvec_qk_at_view(encoder, weight, x_buf, 0, y_buf, 0, m, k, seq_len);
            return;
        }
        encoder.set_compute_pipeline_state(&self.projection_q4_batch_pipeline);
        encoder.set_buffer(0, Some(&weight.buffer), weight.offset);
        encoder.set_buffer(1, Some(x_buf), 0);
        encoder.set_buffer(2, Some(y_buf), 0);
        encoder.set_bytes(3, 4, &m as *const u32 as *const _);
        encoder.set_bytes(4, 4, &k as *const u32 as *const _);
        encoder.set_bytes(5, 4, &seq_len as *const u32 as *const _);
        let n_rows_per_tg = 4u64;
        let num_tgs = MTLSize::new(
            (m as u64 + n_rows_per_tg - 1) / n_rows_per_tg,
            seq_len as u64,
            1,
        );
        let tg_size = MTLSize::new(64, 1, 1);
        encoder.dispatch_thread_groups(num_tgs, tg_size);
    }

    /// Parallel prefill only — K-quant uses mul_mm when seq_len > 8.
    pub fn encode_prefill_projection_q4_batch_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &Buffer,
        y_buf: &Buffer,
        m: u32,
        k: u32,
        seq_len: u32,
    ) {
        if matches!(weight.format, weight_fmt::Q4_K | weight_fmt::Q6_K) {
            self.encode_prefill_kquant_projection(encoder, weight, x_buf, y_buf, m, k, seq_len);
            return;
        }
        self.encode_projection_q4_batch_view(encoder, weight, x_buf, y_buf, m, k, seq_len);
    }

    pub fn encode_rmsnorm(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        weight_buf: &Buffer,
        out_buf: &Buffer,
        dim: u32,
        eps: f32,
    ) {
        self.encode_rmsnorm_at(encoder, x_buf, 0, weight_buf, 0, out_buf, 0, dim, eps);
    }

    pub fn encode_rmsnorm_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        x_offset: u64,
        weight_buf: &Buffer,
        weight_offset: u64,
        out_buf: &Buffer,
        out_offset: u64,
        dim: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.rmsnorm_pipeline);
        encoder.set_buffer(0, Some(x_buf), x_offset);
        encoder.set_buffer(1, Some(weight_buf), weight_offset);
        encoder.set_buffer(2, Some(out_buf), out_offset);
        encoder.set_bytes(3, 4, &dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &eps as *const f32 as *const _);
        let tg_size = MTLSize::new(256.min(dim as u64), 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), tg_size);
    }

    pub fn encode_rmsnorm_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        weight: &BufferView,
        out_buf: &Buffer,
        dim: u32,
        eps: f32,
    ) {
        self.encode_rmsnorm_at(
            encoder,
            x_buf,
            0,
            &weight.buffer,
            weight.offset,
            out_buf,
            0,
            dim,
            eps,
        );
    }

    pub fn encode_rmsnorm_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        x_offset: u64,
        weight: &BufferView,
        out_buf: &Buffer,
        out_offset: u64,
        dim: u32,
        eps: f32,
    ) {
        self.encode_rmsnorm_at(
            encoder,
            x_buf,
            x_offset,
            &weight.buffer,
            weight.offset,
            out_buf,
            out_offset,
            dim,
            eps,
        );
    }

    /// Fused RMSNorm + residual add.
    /// Computes: out = RMSNorm(a + b) * weight
    pub fn encode_rmsnorm_add(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        a_buf: &Buffer,
        b_buf: &Buffer,
        weight_buf: &Buffer,
        out_buf: &Buffer,
        dim: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.rmsnorm_add_pipeline);
        encoder.set_buffer(0, Some(a_buf), 0);
        encoder.set_buffer(1, Some(b_buf), 0);
        encoder.set_buffer(2, Some(weight_buf), 0);
        encoder.set_buffer(3, Some(out_buf), 0);
        encoder.set_bytes(4, 4, &dim as *const u32 as *const _);
        encoder.set_bytes(5, 4, &eps as *const f32 as *const _);
        let tg_size = MTLSize::new(256.min(dim as u64), 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), tg_size);
    }

    /// Fused RMSNorm + residual add with residual save.
    /// Computes: out = RMSNorm(a + b) * weight, residual_out = a + b.
    /// a_buf and residual_out_buf may be the same buffer (in-place residual update).
    pub fn encode_rmsnorm_add_save_residual(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        a_buf: &Buffer,
        b_buf: &Buffer,
        weight_buf: &Buffer,
        out_buf: &Buffer,
        residual_out_buf: &Buffer,
        dim: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.rmsnorm_add_save_residual_pipeline);
        encoder.set_buffer(0, Some(a_buf), 0);
        encoder.set_buffer(1, Some(b_buf), 0);
        encoder.set_buffer(2, Some(weight_buf), 0);
        encoder.set_buffer(3, Some(out_buf), 0);
        encoder.set_buffer(4, Some(residual_out_buf), 0);
        encoder.set_bytes(5, 4, &dim as *const u32 as *const _);
        encoder.set_bytes(6, 4, &eps as *const f32 as *const _);
        let tg_size = MTLSize::new(256.min(dim as u64), 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), tg_size);
    }

    /// Fused RMSNorm(x)*weight accumulated into acc in place (Gemma4 post-norm residual).
    pub fn encode_rmsnorm_acc_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        acc_buf: &Buffer,
        x_buf: &Buffer,
        weight: &BufferView,
        dim: u32,
        eps: f32,
    ) {
        self.encode_rmsnorm_acc_at_view(
            encoder, acc_buf, 0, x_buf, 0, weight, dim, eps,
        );
    }

    pub fn encode_rmsnorm_acc_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        acc_buf: &Buffer,
        acc_offset: u64,
        x_buf: &Buffer,
        x_offset: u64,
        weight: &BufferView,
        dim: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.rmsnorm_acc_pipeline);
        encoder.set_buffer(0, Some(acc_buf), acc_offset);
        encoder.set_buffer(1, Some(x_buf), x_offset);
        encoder.set_buffer(2, Some(&weight.buffer), weight.offset);
        encoder.set_bytes(3, 4, &dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &eps as *const f32 as *const _);
        let tg_size = MTLSize::new(256.min(dim as u64), 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), tg_size);
    }

    /// Fused RMSNorm(x)*weight + acc -> out (batch decode residual path).
    pub fn encode_rmsnorm_acc_out_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        acc_buf: &Buffer,
        acc_offset: u64,
        x_buf: &Buffer,
        x_offset: u64,
        weight: &BufferView,
        out_buf: &Buffer,
        out_offset: u64,
        dim: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.rmsnorm_acc_out_pipeline);
        encoder.set_buffer(0, Some(acc_buf), acc_offset);
        encoder.set_buffer(1, Some(x_buf), x_offset);
        encoder.set_buffer(2, Some(&weight.buffer), weight.offset);
        encoder.set_buffer(3, Some(out_buf), out_offset);
        encoder.set_bytes(4, 4, &dim as *const u32 as *const _);
        encoder.set_bytes(5, 4, &eps as *const f32 as *const _);
        let tg_size = MTLSize::new(256.min(dim as u64), 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), tg_size);
    }

    /// GPU-side RoPE table fill for all layers at one decode position.
    pub fn encode_rope_fill_decode(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        cos_packed: &Buffer,
        sin_packed: &Buffer,
        layer_params: &Buffer,
        num_layers: u32,
        max_head_dim: u32,
        position: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.rope_fill_decode_pipeline);
        encoder.set_buffer(0, Some(cos_packed), 0);
        encoder.set_buffer(1, Some(sin_packed), 0);
        encoder.set_buffer(2, Some(layer_params), 0);
        encoder.set_bytes(3, 4, &max_head_dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &position as *const f32 as *const _);
        let threads = MTLSize::new((max_head_dim / 2) as u64, num_layers as u64, 1);
        encoder.dispatch_threads(threads, MTLSize::new(256, 1, 1));
    }

    /// GPU cos/sin tables for one prefill layer × seq_len rows.
    pub fn encode_rope_fill_prefill_batch(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        cos_buf: &Buffer,
        sin_buf: &Buffer,
        layer_params: &Buffer,
        layer_idx: u32,
        start_pos: u32,
        seq_len: u32,
        head_dim: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.rope_fill_prefill_batch_pipeline);
        encoder.set_buffer(0, Some(cos_buf), 0);
        encoder.set_buffer(1, Some(sin_buf), 0);
        encoder.set_buffer(2, Some(layer_params), 0);
        encoder.set_bytes(3, 4, &layer_idx as *const u32 as *const _);
        encoder.set_bytes(4, 4, &start_pos as *const u32 as *const _);
        encoder.set_bytes(5, 4, &seq_len as *const u32 as *const _);
        let threads = MTLSize::new((head_dim / 2) as u64, seq_len as u64, 1);
        encoder.dispatch_threads(threads, MTLSize::new(256, 1, 1));
    }

    pub fn encode_rotary(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        k_buf: &Buffer,
        cos_buf: &Buffer,
        sin_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) {
        self.encode_rotary_at(
            encoder,
            q_buf,
            0,
            k_buf,
            0,
            cos_buf,
            0,
            sin_buf,
            0,
            num_heads,
            num_kv_heads,
            head_dim,
        );
    }

    pub fn encode_rotary_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        q_offset: u64,
        k_buf: &Buffer,
        k_offset: u64,
        cos_buf: &Buffer,
        cos_offset: u64,
        sin_buf: &Buffer,
        sin_offset: u64,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) {
        let half_dim = head_dim / 2;
        let total_threads = num_heads * half_dim + num_kv_heads * half_dim;
        encoder.set_compute_pipeline_state(&self.rotary_pipeline);
        encoder.set_buffer(0, Some(q_buf), q_offset);
        encoder.set_buffer(1, Some(k_buf), k_offset);
        encoder.set_buffer(2, Some(cos_buf), cos_offset);
        encoder.set_buffer(3, Some(sin_buf), sin_offset);
        encoder.set_bytes(4, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(5, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(6, 4, &head_dim as *const u32 as *const _);
        let threads = MTLSize::new(total_threads as u64, 1, 1);
        encoder.dispatch_threads(threads, MTLSize::new(256, 1, 1));
    }

    pub fn encode_kv_append(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        new_data: &Buffer,
        cache: &Buffer,
        num_kv_heads: u32,
        head_dim: u32,
        capacity: u32,
        cur_seq: u32,
    ) {
        let total = num_kv_heads * head_dim;
        encoder.set_compute_pipeline_state(&self.kv_append_pipeline);
        encoder.set_buffer(0, Some(new_data), 0);
        encoder.set_buffer(1, Some(cache), 0);
        encoder.set_bytes(2, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(3, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(5, 4, &cur_seq as *const u32 as *const _);
        let threads = MTLSize::new(total as u64, 1, 1);
        encoder.dispatch_threads(threads, MTLSize::new(256, 1, 1));
    }

    pub fn encode_kv_append_f16(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        new_data: &Buffer,
        cache: &Buffer,
        num_kv_heads: u32,
        head_dim: u32,
        capacity: u32,
        cur_seq: u32,
    ) {
        self.encode_kv_append_f16_at(
            encoder,
            new_data,
            0,
            cache,
            num_kv_heads,
            head_dim,
            capacity,
            cur_seq,
        );
    }

    pub fn encode_kv_append_f16_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        new_data: &Buffer,
        new_data_offset: u64,
        cache: &Buffer,
        num_kv_heads: u32,
        head_dim: u32,
        capacity: u32,
        cur_seq: u32,
    ) {
        let total = num_kv_heads * head_dim;
        encoder.set_compute_pipeline_state(&self.kv_append_f16_pipeline);
        encoder.set_buffer(0, Some(new_data), new_data_offset);
        encoder.set_buffer(1, Some(cache), 0);
        encoder.set_bytes(2, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(3, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(5, 4, &cur_seq as *const u32 as *const _);
        let threads = MTLSize::new(total as u64, 1, 1);
        encoder.dispatch_threads(threads, MTLSize::new(256, 1, 1));
    }

    pub fn encode_attention(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        out_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        k_cap: u32,
        scale: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.attention_pipeline);
        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(k_cache_buf), 0);
        encoder.set_buffer(2, Some(v_cache_buf), 0);
        encoder.set_buffer(3, Some(out_buf), 0);
        encoder.set_bytes(4, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(5, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(6, 4, &num_kv_groups as *const u32 as *const _);
        encoder.set_bytes(7, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(8, 4, &kv_seq as *const u32 as *const _);
        encoder.set_bytes(9, 4, &k_cap as *const u32 as *const _);
        encoder.set_bytes(10, 4, &scale as *const f32 as *const _);
        let tg_size = MTLSize::new(64, 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(num_heads as u64, 1, 1), tg_size);
    }

    pub fn encode_silu_mul(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate_buf: &Buffer,
        up_buf: &Buffer,
        out_buf: &Buffer,
        n: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.silu_mul_pipeline);
        encoder.set_buffer(0, Some(gate_buf), 0);
        encoder.set_buffer(1, Some(up_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 4, &n as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_gelu_mul(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate_buf: &Buffer,
        up_buf: &Buffer,
        out_buf: &Buffer,
        n: u32,
    ) {
        self.encode_gelu_mul_at(encoder, gate_buf, 0, up_buf, 0, out_buf, 0, n);
    }

    pub fn encode_gelu_mul_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate_buf: &Buffer,
        gate_offset: u64,
        up_buf: &Buffer,
        up_offset: u64,
        out_buf: &Buffer,
        out_offset: u64,
        n: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.gelu_mul_pipeline);
        encoder.set_buffer(0, Some(gate_buf), gate_offset);
        encoder.set_buffer(1, Some(up_buf), up_offset);
        encoder.set_buffer(2, Some(out_buf), out_offset);
        encoder.set_bytes(3, 4, &n as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_gelu_mul_f16_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate_buf: &Buffer,
        gate_offset: u64,
        up_buf: &Buffer,
        up_offset: u64,
        out_buf: &Buffer,
        out_offset: u64,
        n: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.gelu_mul_f16_pipeline);
        encoder.set_buffer(0, Some(gate_buf), gate_offset);
        encoder.set_buffer(1, Some(up_buf), up_offset);
        encoder.set_buffer(2, Some(out_buf), out_offset);
        encoder.set_bytes(3, 4, &n as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_vec_mul(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        a_buf: &Buffer,
        b_buf: &Buffer,
        out_buf: &Buffer,
        n: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.vec_mul_pipeline);
        encoder.set_buffer(0, Some(a_buf), 0);
        encoder.set_buffer(1, Some(b_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 4, &n as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_vec_add_scaled(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        a_buf: &Buffer,
        b_buf: &Buffer,
        out_buf: &Buffer,
        n: u32,
        scale: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.vec_add_scaled_pipeline);
        encoder.set_buffer(0, Some(a_buf), 0);
        encoder.set_buffer(1, Some(b_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 4, &n as *const u32 as *const _);
        encoder.set_bytes(4, 4, &scale as *const f32 as *const _);
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_vec_scale(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        src_buf: &Buffer,
        dst_buf: &Buffer,
        n: u32,
        scale: f32,
    ) {
        self.encode_vec_scale_at(encoder, src_buf, 0, dst_buf, 0, n, scale);
    }

    pub fn encode_vec_scale_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        src_buf: &Buffer,
        src_offset: u64,
        dst_buf: &Buffer,
        dst_offset: u64,
        n: u32,
        scale: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.vec_scale_pipeline);
        encoder.set_buffer(0, Some(src_buf), src_offset);
        encoder.set_buffer(1, Some(dst_buf), dst_offset);
        encoder.set_bytes(2, 4, &n as *const u32 as *const _);
        encoder.set_bytes(3, 4, &scale as *const f32 as *const _);
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_rmsnorm_per_head(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        weight_buf: &Buffer,
        out_buf: &Buffer,
        num_heads: u32,
        head_dim: u32,
        eps: f32,
    ) {
        self.encode_rmsnorm_per_head_at(
            encoder, x_buf, 0, weight_buf, 0, out_buf, 0, num_heads, head_dim, eps,
        );
    }

    pub fn encode_rmsnorm_per_head_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        weight: &BufferView,
        out_buf: &Buffer,
        num_heads: u32,
        head_dim: u32,
        eps: f32,
    ) {
        self.encode_rmsnorm_per_head_at(
            encoder,
            x_buf,
            0,
            &weight.buffer,
            weight.offset,
            out_buf,
            0,
            num_heads,
            head_dim,
            eps,
        );
    }

    pub fn encode_rmsnorm_per_head_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        x_offset: u64,
        weight_buf: &Buffer,
        weight_offset: u64,
        out_buf: &Buffer,
        out_offset: u64,
        num_heads: u32,
        head_dim: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.rmsnorm_per_head_pipeline);
        encoder.set_buffer(0, Some(x_buf), x_offset);
        encoder.set_buffer(1, Some(weight_buf), weight_offset);
        encoder.set_buffer(2, Some(out_buf), out_offset);
        encoder.set_bytes(3, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(4, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(5, 4, &eps as *const f32 as *const _);
        let tg_size = MTLSize::new(256.min(head_dim as u64), 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(num_heads as u64, 1, 1), tg_size);
    }

    pub fn encode_rmsnorm_per_head_at_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        x_offset: u64,
        weight: &BufferView,
        out_buf: &Buffer,
        out_offset: u64,
        num_heads: u32,
        head_dim: u32,
        eps: f32,
    ) {
        self.encode_rmsnorm_per_head_at(
            encoder,
            x_buf,
            x_offset,
            &weight.buffer,
            weight.offset,
            out_buf,
            out_offset,
            num_heads,
            head_dim,
            eps,
        );
    }

    pub fn encode_rmsnorm_per_head_noweight(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        out_buf: &Buffer,
        num_heads: u32,
        head_dim: u32,
        eps: f32,
    ) {
        self.encode_rmsnorm_per_head_noweight_at(
            encoder, x_buf, 0, out_buf, 0, num_heads, head_dim, eps,
        );
    }

    pub fn encode_rmsnorm_per_head_noweight_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        x_offset: u64,
        out_buf: &Buffer,
        out_offset: u64,
        num_heads: u32,
        head_dim: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.rmsnorm_per_head_noweight_pipeline);
        encoder.set_buffer(0, Some(x_buf), x_offset);
        encoder.set_buffer(1, Some(out_buf), out_offset);
        encoder.set_bytes(2, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(3, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &eps as *const f32 as *const _);
        let tg_size = MTLSize::new(256.min(head_dim as u64), 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(num_heads as u64, 1, 1), tg_size);
    }

    pub fn encode_rotary_partial(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        k_buf: &Buffer,
        cos_buf: &Buffer,
        sin_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
    ) {
        let half_rot = rotary_dim / 2;
        let total_threads = num_heads * half_rot + num_kv_heads * half_rot;
        encoder.set_compute_pipeline_state(&self.rotary_partial_pipeline);
        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(k_buf), 0);
        encoder.set_buffer(2, Some(cos_buf), 0);
        encoder.set_buffer(3, Some(sin_buf), 0);
        encoder.set_bytes(4, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(5, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(6, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(7, 4, &rotary_dim as *const u32 as *const _);
        let threads = MTLSize::new(total_threads as u64, 1, 1);
        encoder.dispatch_threads(threads, MTLSize::new(256, 1, 1));
    }

    pub fn encode_attention_with_offset(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        out_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        k_cap: u32,
        scale: f32,
        kv_start: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.attention_offset_pipeline);
        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(k_cache_buf), 0);
        encoder.set_buffer(2, Some(v_cache_buf), 0);
        encoder.set_buffer(3, Some(out_buf), 0);
        encoder.set_bytes(4, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(5, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(6, 4, &num_kv_groups as *const u32 as *const _);
        encoder.set_bytes(7, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(8, 4, &kv_seq as *const u32 as *const _);
        encoder.set_bytes(9, 4, &k_cap as *const u32 as *const _);
        encoder.set_bytes(10, 4, &scale as *const f32 as *const _);
        encoder.set_bytes(11, 4, &kv_start as *const u32 as *const _);
        let tg_size = MTLSize::new(64, 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(num_heads as u64, 1, 1), tg_size);
    }

    pub fn encode_attention_with_offset_f16(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        out_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        k_cap: u32,
        scale: f32,
        kv_start: u32,
    ) {
        self.encode_attention_with_offset_f16_at(
            encoder,
            q_buf,
            0,
            k_cache_buf,
            v_cache_buf,
            out_buf,
            0,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            kv_seq,
            k_cap,
            scale,
            kv_start,
        );
    }

    pub fn encode_attention_with_offset_f16_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        q_offset: u64,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        out_buf: &Buffer,
        out_offset: u64,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        k_cap: u32,
        scale: f32,
        kv_start: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.attention_offset_f16_pipeline);
        encoder.set_buffer(0, Some(q_buf), q_offset);
        encoder.set_buffer(1, Some(k_cache_buf), 0);
        encoder.set_buffer(2, Some(v_cache_buf), 0);
        encoder.set_buffer(3, Some(out_buf), out_offset);
        encoder.set_bytes(4, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(5, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(6, 4, &num_kv_groups as *const u32 as *const _);
        encoder.set_bytes(7, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(8, 4, &kv_seq as *const u32 as *const _);
        encoder.set_bytes(9, 4, &k_cap as *const u32 as *const _);
        encoder.set_bytes(10, 4, &scale as *const f32 as *const _);
        encoder.set_bytes(11, 4, &kv_start as *const u32 as *const _);
        let tg_size = attention_threadgroup_size(self.use_flash_attention);
        encoder.dispatch_thread_groups(MTLSize::new(num_heads as u64, 1, 1), tg_size);
    }

    /// GQA-aware f16 decode attention: one threadgroup per KV head, one
    /// simdgroup per query head in the group. Reduces KV cache reads by
    /// num_kv_groups compared to the per-query-head kernel.
    pub fn encode_attention_with_offset_f16_gqa_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        q_offset: u64,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        out_buf: &Buffer,
        out_offset: u64,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        k_cap: u32,
        scale: f32,
        kv_start: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.attention_offset_f16_gqa_pipeline);
        encoder.set_buffer(0, Some(q_buf), q_offset);
        encoder.set_buffer(1, Some(k_cache_buf), 0);
        encoder.set_buffer(2, Some(v_cache_buf), 0);
        encoder.set_buffer(3, Some(out_buf), out_offset);
        encoder.set_bytes(4, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(5, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(6, 4, &num_kv_groups as *const u32 as *const _);
        encoder.set_bytes(7, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(8, 4, &kv_seq as *const u32 as *const _);
        encoder.set_bytes(9, 4, &k_cap as *const u32 as *const _);
        encoder.set_bytes(10, 4, &scale as *const f32 as *const _);
        encoder.set_bytes(11, 4, &kv_start as *const u32 as *const _);
        let tg_size = MTLSize::new((num_kv_groups * 32) as u64, 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(num_kv_heads as u64, 1, 1), tg_size);
    }

    pub fn encode_vec_add(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        a_buf: &Buffer,
        b_buf: &Buffer,
        c_buf: &Buffer,
        n: u32,
    ) {
        self.encode_vec_add_at(encoder, a_buf, 0, b_buf, 0, c_buf, 0, n);
    }

    pub fn encode_vec_add_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        a_buf: &Buffer,
        a_offset: u64,
        b_buf: &Buffer,
        b_offset: u64,
        c_buf: &Buffer,
        c_offset: u64,
        n: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.vec_add_pipeline);
        encoder.set_buffer(0, Some(a_buf), a_offset);
        encoder.set_buffer(1, Some(b_buf), b_offset);
        encoder.set_buffer(2, Some(c_buf), c_offset);
        encoder.set_bytes(3, 4, &n as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_copy(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        src: &Buffer,
        dst: &Buffer,
        n: u32,
    ) {
        self.encode_copy_at(encoder, src, 0, dst, 0, n);
    }

    pub fn encode_copy_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        src: &Buffer,
        src_offset: u64,
        dst: &Buffer,
        dst_offset: u64,
        n: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.buf_copy_pipeline);
        encoder.set_buffer(0, Some(src), src_offset);
        encoder.set_buffer(1, Some(dst), dst_offset);
        encoder.set_bytes(2, 4, &n as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    // ─── Batched encoder methods for prefill ───────────────────────────────────

    pub fn encode_matmul(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        a_buf: &Buffer,
        b_buf: &Buffer,
        c_buf: &Buffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.matmul_pipeline);
        encoder.set_buffer(0, Some(a_buf), 0);
        encoder.set_buffer(1, Some(b_buf), 0);
        encoder.set_buffer(2, Some(c_buf), 0);
        encoder.set_bytes(3, 4, &m as *const u32 as *const _);
        encoder.set_bytes(4, 4, &n as *const u32 as *const _);
        encoder.set_bytes(5, 4, &k as *const u32 as *const _);
        let threads = MTLSize::new(n as u64, m as u64, 1);
        encoder.dispatch_threads(threads, MTLSize::new(16, 16, 1));
    }

    pub fn encode_rmsnorm_batch(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        weight_buf: &Buffer,
        out_buf: &Buffer,
        dim: u32,
        eps: f32,
        seq_len: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.rmsnorm_batch_pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(weight_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 4, &dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &eps as *const f32 as *const _);
        let tg_size = MTLSize::new(256.min(dim as u64), 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(seq_len as u64, 1, 1), tg_size);
    }

    pub fn encode_rmsnorm_batch_view(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        weight: &BufferView,
        out_buf: &Buffer,
        dim: u32,
        eps: f32,
        seq_len: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.rmsnorm_batch_pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(&weight.buffer), weight.offset);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 4, &dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &eps as *const f32 as *const _);
        let tg_size = MTLSize::new(256.min(dim as u64), 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(seq_len as u64, 1, 1), tg_size);
    }

    pub fn encode_rmsnorm_noweight_batch(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        out_buf: &Buffer,
        dim: u32,
        eps: f32,
        num_rows: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.rmsnorm_noweight_batch_pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(out_buf), 0);
        encoder.set_bytes(2, 4, &dim as *const u32 as *const _);
        encoder.set_bytes(3, 4, &eps as *const f32 as *const _);
        let tg_size = MTLSize::new(256.min(dim as u64), 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(num_rows as u64, 1, 1), tg_size);
    }

    pub fn encode_silu_mul_batch(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate_buf: &Buffer,
        up_buf: &Buffer,
        out_buf: &Buffer,
        n: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.silu_mul_batch_pipeline);
        encoder.set_buffer(0, Some(gate_buf), 0);
        encoder.set_buffer(1, Some(up_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 4, &n as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_vec_add_batch(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        a_buf: &Buffer,
        b_buf: &Buffer,
        c_buf: &Buffer,
        n: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.vec_add_batch_pipeline);
        encoder.set_buffer(0, Some(a_buf), 0);
        encoder.set_buffer(1, Some(b_buf), 0);
        encoder.set_buffer(2, Some(c_buf), 0);
        encoder.set_bytes(3, 4, &n as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_rotary_batch(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        k_buf: &Buffer,
        cos_buf: &Buffer,
        sin_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
    ) {
        let half_dim = head_dim / 2;
        let total = num_heads * seq_len * half_dim + num_kv_heads * seq_len * half_dim;
        encoder.set_compute_pipeline_state(&self.rotary_batch_pipeline);
        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(k_buf), 0);
        encoder.set_buffer(2, Some(cos_buf), 0);
        encoder.set_buffer(3, Some(sin_buf), 0);
        encoder.set_bytes(4, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(5, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(6, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(7, 4, &seq_len as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_ple_gelu_mul_batch(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gate_buf: &Buffer,
        context_buf: &Buffer,
        out_buf: &Buffer,
        layer_idx: u32,
        num_layers: u32,
        ple_dim: u32,
        seq_len: u32,
    ) {
        let total = seq_len * ple_dim;
        encoder.set_compute_pipeline_state(&self.ple_gelu_mul_batch_pipeline);
        encoder.set_buffer(0, Some(gate_buf), 0);
        encoder.set_buffer(1, Some(context_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 4, &layer_idx as *const u32 as *const _);
        encoder.set_bytes(4, 4, &num_layers as *const u32 as *const _);
        encoder.set_bytes(5, 4, &ple_dim as *const u32 as *const _);
        encoder.set_bytes(6, 4, &seq_len as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_kv_batch_append(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        new_data: &Buffer,
        cache: &Buffer,
        num_kv_heads: u32,
        head_dim: u32,
        capacity: u32,
        cur_seq: u32,
        seq_len: u32,
    ) {
        let total = num_kv_heads * seq_len * head_dim;
        encoder.set_compute_pipeline_state(&self.kv_batch_append_pipeline);
        encoder.set_buffer(0, Some(new_data), 0);
        encoder.set_buffer(1, Some(cache), 0);
        encoder.set_bytes(2, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(3, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(5, 4, &cur_seq as *const u32 as *const _);
        encoder.set_bytes(6, 4, &seq_len as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_kv_batch_append_f16(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        new_data: &Buffer,
        cache: &Buffer,
        num_kv_heads: u32,
        head_dim: u32,
        capacity: u32,
        cur_seq: u32,
        seq_len: u32,
    ) {
        let total = num_kv_heads * seq_len * head_dim;
        encoder.set_compute_pipeline_state(&self.kv_batch_append_f16_pipeline);
        encoder.set_buffer(0, Some(new_data), 0);
        encoder.set_buffer(1, Some(cache), 0);
        encoder.set_bytes(2, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(3, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(5, 4, &cur_seq as *const u32 as *const _);
        encoder.set_bytes(6, 4, &seq_len as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_kv_batch_append_strided_f16(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        new_data: &Buffer,
        cache: &Buffer,
        num_kv_heads: u32,
        head_dim: u32,
        capacity: u32,
        cur_seq: u32,
        seq_len: u32,
        source_seq_stride: u32,
        source_start: u32,
    ) {
        let total = num_kv_heads * seq_len * head_dim;
        encoder.set_compute_pipeline_state(&self.kv_batch_append_strided_f16_pipeline);
        encoder.set_buffer(0, Some(new_data), 0);
        encoder.set_buffer(1, Some(cache), 0);
        encoder.set_bytes(2, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(3, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(5, 4, &cur_seq as *const u32 as *const _);
        encoder.set_bytes(6, 4, &seq_len as *const u32 as *const _);
        encoder.set_bytes(7, 4, &source_seq_stride as *const u32 as *const _);
        encoder.set_bytes(8, 4, &source_start as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    fn attention_fused_q4_0_pipeline_for(&self, head_dim: u32) -> &ComputePipelineState {
        if self.use_flash_attention && attention_q4_hd_specialized() {
            match head_dim {
                256 => return &self.attention_fused_q4_0_h256_pipeline,
                128 => return &self.attention_fused_q4_0_h128_pipeline,
                512 => return &self.attention_fused_q4_0_h512_pipeline,
                _ => {}
            }
        }
        &self.attention_fused_q4_0_pipeline
    }

    fn attention_qknorm_rope_q4_0_pipeline_for(&self, head_dim: u32) -> &ComputePipelineState {
        match head_dim {
            256 => &self.attention_qknorm_rope_q4_0_h256_pipeline,
            128 => &self.attention_qknorm_rope_q4_0_h128_pipeline,
            512 => &self.attention_qknorm_rope_q4_0_h512_pipeline,
            _ => panic!("qknorm+rope flash attention unsupported head_dim {head_dim}"),
        }
    }

    fn attention_fused_qknorm_rope_q4_0_pipeline_for(&self, head_dim: u32) -> &ComputePipelineState {
        match head_dim {
            256 => &self.attention_fused_qknorm_rope_q4_0_h256_pipeline,
            128 => &self.attention_fused_qknorm_rope_q4_0_h128_pipeline,
            512 => &self.attention_fused_qknorm_rope_q4_0_h512_pipeline,
            _ => panic!("fused qknorm+rope flash attention unsupported head_dim {head_dim}"),
        }
    }

    fn attention_full_fused_q4_0_pipeline_for(&self, head_dim: u32) -> &ComputePipelineState {
        match head_dim {
            256 => &self.attention_full_fused_q4_0_h256_pipeline,
            128 => &self.attention_full_fused_q4_0_h128_pipeline,
            512 => &self.attention_full_fused_q4_0_h512_pipeline,
            _ => panic!("full fused flash attention unsupported head_dim {head_dim}"),
        }
    }

    fn flash_attn_ggml_pipeline_for(&self, head_dim: u32) -> &ComputePipelineState {
        match head_dim {
            256 => &self.flash_attn_ggml_q4_h256_pipeline,
            128 => &self.flash_attn_ggml_q4_h128_pipeline,
            512 => &self.flash_attn_ggml_q4_h512_pipeline,
            _ => panic!("ggml flash attention unsupported head_dim {head_dim} (need 256, 128, or 512)"),
        }
    }

    fn flash_attn_ggml_reduce_pipeline_for(&self, head_dim: u32) -> &ComputePipelineState {
        match head_dim {
            256 => &self.flash_attn_ggml_reduce_h256_pipeline,
            128 => &self.flash_attn_ggml_reduce_h128_pipeline,
            512 => &self.flash_attn_ggml_reduce_h512_pipeline,
            _ => panic!("ggml flash attention reduce unsupported head_dim {head_dim}"),
        }
    }

    pub fn encode_attention_ggml_q4_0(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        tmp_buf: &Buffer,
        out_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        kv_seq: u32,
        capacity: u32,
        scale: f32,
        kv_start: u32,
        row_bytes: u32,
    ) {
        self.encode_attention_ggml_q4_0_at(
            encoder,
            q_buf,
            0,
            k_cache_buf,
            0,
            v_cache_buf,
            0,
            tmp_buf,
            0,
            out_buf,
            0,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_seq,
            capacity,
            scale,
            kv_start,
            row_bytes,
        );
    }

    pub fn encode_attention_ggml_q4_0_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        q_offset: u64,
        k_cache_buf: &Buffer,
        k_offset: u64,
        v_cache_buf: &Buffer,
        v_offset: u64,
        tmp_buf: &Buffer,
        tmp_offset: u64,
        out_buf: &Buffer,
        out_offset: u64,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        kv_seq: u32,
        capacity: u32,
        scale: f32,
        kv_start: u32,
        row_bytes: u32,
    ) {
        use crate::ggml_flash_attn::{
            flash_attn_args, flash_attn_dispatch, flash_attn_reduce_dispatch, flash_attn_smem_bytes,
            GgmlFlashAttnReduceArgs,
        };

        let args = flash_attn_args(
            num_heads,
            num_kv_heads,
            head_dim,
            capacity,
            kv_seq,
            row_bytes as u64,
            scale,
        );
        let kv_off = (kv_start as u64) * row_bytes as u64;
        encoder.set_compute_pipeline_state(self.flash_attn_ggml_pipeline_for(head_dim));
        encoder.set_bytes(
            0,
            std::mem::size_of::<crate::ggml_flash_attn::GgmlFlashAttnArgs>() as u64,
            &args as *const _ as *const _,
        );
        encoder.set_buffer(1, Some(q_buf), q_offset);
        encoder.set_buffer(2, Some(k_cache_buf), k_offset + kv_off);
        encoder.set_buffer(3, Some(v_cache_buf), v_offset + kv_off);
        encoder.set_buffer(4, Some(tmp_buf), tmp_offset);
        encoder.set_threadgroup_memory_length(0, flash_attn_smem_bytes(head_dim));
        let (tg_x, tg_y, tg_z, tw, nsg) = flash_attn_dispatch(num_heads);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(tg_x, tg_y, tg_z),
            metal::MTLSize::new(tw, nsg, 1),
        );

        let reduce_args = GgmlFlashAttnReduceArgs {
            nrows: num_heads as i32,
        };
        encoder.set_compute_pipeline_state(self.flash_attn_ggml_reduce_pipeline_for(head_dim));
        encoder.set_bytes(
            0,
            std::mem::size_of::<GgmlFlashAttnReduceArgs>() as u64,
            &reduce_args as *const _ as *const _,
        );
        encoder.set_buffer(1, Some(tmp_buf), tmp_offset);
        encoder.set_buffer(2, Some(out_buf), out_offset);
        let (rtg_x, rtg_y, rtg_z, rtw, rnsg, _) = flash_attn_reduce_dispatch(num_heads);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(rtg_x, rtg_y, rtg_z),
            metal::MTLSize::new(rtw, rnsg, 1),
        );
    }

    pub fn encode_attention_fused_q4_0(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        k_f32_buf: &Buffer,
        v_f32_buf: &Buffer,
        out_buf: &Buffer,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        capacity: u32,
        scale: f32,
        kv_start: u32,
        cur_seq: u32,
        groups_per_row: u32,
        row_bytes: u32,
    ) {
        self.encode_attention_fused_q4_0_at(
            encoder,
            q_buf,
            0,
            k_f32_buf,
            0,
            v_f32_buf,
            0,
            out_buf,
            0,
            k_cache_buf,
            v_cache_buf,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            kv_seq,
            capacity,
            scale,
            kv_start,
            cur_seq,
            groups_per_row,
            row_bytes,
        );
    }

    pub fn encode_attention_fused_q4_0_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        q_offset: u64,
        k_f32_buf: &Buffer,
        k_f32_offset: u64,
        v_f32_buf: &Buffer,
        v_f32_offset: u64,
        out_buf: &Buffer,
        out_offset: u64,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        capacity: u32,
        scale: f32,
        kv_start: u32,
        cur_seq: u32,
        groups_per_row: u32,
        row_bytes: u32,
    ) {
        encoder.set_compute_pipeline_state(self.attention_fused_q4_0_pipeline_for(head_dim));
        encoder.set_buffer(0, Some(q_buf), q_offset);
        encoder.set_buffer(1, Some(k_f32_buf), k_f32_offset);
        encoder.set_buffer(2, Some(v_f32_buf), v_f32_offset);
        encoder.set_buffer(3, Some(out_buf), out_offset);
        encoder.set_buffer(4, Some(k_cache_buf), 0);
        encoder.set_buffer(5, Some(v_cache_buf), 0);
        encoder.set_bytes(6, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(7, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(8, 4, &num_kv_groups as *const u32 as *const _);
        encoder.set_bytes(9, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(10, 4, &kv_seq as *const u32 as *const _);
        encoder.set_bytes(11, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(12, 4, &scale as *const f32 as *const _);
        encoder.set_bytes(13, 4, &kv_start as *const u32 as *const _);
        encoder.set_bytes(14, 4, &groups_per_row as *const u32 as *const _);
        encoder.set_bytes(15, 4, &row_bytes as *const u32 as *const _);
        encoder.set_bytes(16, 4, &cur_seq as *const u32 as *const _);
        let tg_size = attention_threadgroup_size(self.use_flash_attention);
        encoder.dispatch_thread_groups(MTLSize::new(num_heads as u64, 1, 1), tg_size);
    }

    pub fn encode_attention_qknorm_rope_q4_0(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_raw_buf: &Buffer,
        q_norm_weight: &BufferView,
        cos_buf: &Buffer,
        cos_offset: u64,
        sin_buf: &Buffer,
        sin_offset: u64,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        out_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        capacity: u32,
        scale: f32,
        kv_start: u32,
        groups_per_row: u32,
        row_bytes: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(self.attention_qknorm_rope_q4_0_pipeline_for(head_dim));
        encoder.set_buffer(0, Some(q_raw_buf), 0);
        encoder.set_buffer(1, Some(&q_norm_weight.buffer), q_norm_weight.offset);
        encoder.set_buffer(2, Some(cos_buf), cos_offset);
        encoder.set_buffer(3, Some(sin_buf), sin_offset);
        encoder.set_buffer(4, Some(k_cache_buf), 0);
        encoder.set_buffer(5, Some(v_cache_buf), 0);
        encoder.set_buffer(6, Some(out_buf), 0);
        encoder.set_bytes(7, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(8, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(9, 4, &num_kv_groups as *const u32 as *const _);
        encoder.set_bytes(10, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(11, 4, &kv_seq as *const u32 as *const _);
        encoder.set_bytes(12, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(13, 4, &scale as *const f32 as *const _);
        encoder.set_bytes(14, 4, &kv_start as *const u32 as *const _);
        encoder.set_bytes(15, 4, &groups_per_row as *const u32 as *const _);
        encoder.set_bytes(16, 4, &row_bytes as *const u32 as *const _);
        encoder.set_bytes(17, 4, &eps as *const f32 as *const _);
        let tg_size = attention_threadgroup_size(self.use_flash_attention);
        encoder.dispatch_thread_groups(MTLSize::new(num_heads as u64, 1, 1), tg_size);
    }

    pub fn encode_attention_fused_qknorm_rope_q4_0(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_raw_buf: &Buffer,
        q_norm_weight: &BufferView,
        cos_buf: &Buffer,
        cos_offset: u64,
        sin_buf: &Buffer,
        sin_offset: u64,
        k_f32_buf: &Buffer,
        v_f32_buf: &Buffer,
        out_buf: &Buffer,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        capacity: u32,
        scale: f32,
        kv_start: u32,
        cur_seq: u32,
        groups_per_row: u32,
        row_bytes: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(
            self.attention_fused_qknorm_rope_q4_0_pipeline_for(head_dim),
        );
        encoder.set_buffer(0, Some(q_raw_buf), 0);
        encoder.set_buffer(1, Some(&q_norm_weight.buffer), q_norm_weight.offset);
        encoder.set_buffer(2, Some(cos_buf), cos_offset);
        encoder.set_buffer(3, Some(sin_buf), sin_offset);
        encoder.set_buffer(4, Some(k_f32_buf), 0);
        encoder.set_buffer(5, Some(v_f32_buf), 0);
        encoder.set_buffer(6, Some(out_buf), 0);
        encoder.set_buffer(7, Some(k_cache_buf), 0);
        encoder.set_buffer(8, Some(v_cache_buf), 0);
        encoder.set_bytes(9, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(10, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(11, 4, &num_kv_groups as *const u32 as *const _);
        encoder.set_bytes(12, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(13, 4, &kv_seq as *const u32 as *const _);
        encoder.set_bytes(14, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(15, 4, &scale as *const f32 as *const _);
        encoder.set_bytes(16, 4, &kv_start as *const u32 as *const _);
        encoder.set_bytes(17, 4, &groups_per_row as *const u32 as *const _);
        encoder.set_bytes(18, 4, &row_bytes as *const u32 as *const _);
        encoder.set_bytes(19, 4, &cur_seq as *const u32 as *const _);
        encoder.set_bytes(20, 4, &eps as *const f32 as *const _);
        let tg_size = attention_threadgroup_size(self.use_flash_attention);
        encoder.dispatch_thread_groups(MTLSize::new(num_heads as u64, 1, 1), tg_size);
    }

    pub fn encode_attention_full_fused_q4_0(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_raw_buf: &Buffer,
        q_norm_weight: &BufferView,
        cos_buf: &Buffer,
        cos_offset: u64,
        sin_buf: &Buffer,
        sin_offset: u64,
        k_raw_buf: &Buffer,
        k_norm_weight: &BufferView,
        v_raw_buf: &Buffer,
        out_buf: &Buffer,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        capacity: u32,
        scale: f32,
        kv_start: u32,
        cur_seq: u32,
        groups_per_row: u32,
        row_bytes: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(self.attention_full_fused_q4_0_pipeline_for(head_dim));
        encoder.set_buffer(0, Some(q_raw_buf), 0);
        encoder.set_buffer(1, Some(&q_norm_weight.buffer), q_norm_weight.offset);
        encoder.set_buffer(2, Some(cos_buf), cos_offset);
        encoder.set_buffer(3, Some(sin_buf), sin_offset);
        encoder.set_buffer(4, Some(k_raw_buf), 0);
        encoder.set_buffer(5, Some(&k_norm_weight.buffer), k_norm_weight.offset);
        encoder.set_buffer(6, Some(v_raw_buf), 0);
        encoder.set_buffer(7, Some(out_buf), 0);
        encoder.set_buffer(8, Some(k_cache_buf), 0);
        encoder.set_buffer(9, Some(v_cache_buf), 0);
        encoder.set_bytes(10, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(11, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(12, 4, &num_kv_groups as *const u32 as *const _);
        encoder.set_bytes(13, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(14, 4, &kv_seq as *const u32 as *const _);
        encoder.set_bytes(15, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(16, 4, &scale as *const f32 as *const _);
        encoder.set_bytes(17, 4, &kv_start as *const u32 as *const _);
        encoder.set_bytes(18, 4, &groups_per_row as *const u32 as *const _);
        encoder.set_bytes(19, 4, &row_bytes as *const u32 as *const _);
        encoder.set_bytes(20, 4, &cur_seq as *const u32 as *const _);
        encoder.set_bytes(21, 4, &eps as *const f32 as *const _);
        let tg_size = attention_threadgroup_size(self.use_flash_attention);
        encoder.dispatch_thread_groups(MTLSize::new(num_heads as u64, 1, 1), tg_size);
    }

    pub fn encode_kv_append_attention_q4_0(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        k_buf: &Buffer,
        v_buf: &Buffer,
        output_buf: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        capacity: u32,
        cur_seq: u32,
        scale: f32,
        groups_per_row: u32,
        row_bytes: u32,
    ) {
        let kv_seq = cur_seq + 1;
        self.encode_attention_fused_q4_0(
            encoder,
            q_buf,
            k_buf,
            v_buf,
            output_buf,
            k_cache,
            v_cache,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            kv_seq,
            capacity,
            scale,
            0,
            cur_seq,
            groups_per_row,
            row_bytes,
        );
    }

    pub fn encode_attention_causal(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        out_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        k_cap: u32,
        scale: f32,
        q_len: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.attention_causal_pipeline);
        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(k_cache_buf), 0);
        encoder.set_buffer(2, Some(v_cache_buf), 0);
        encoder.set_buffer(3, Some(out_buf), 0);
        encoder.set_bytes(4, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(5, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(6, 4, &num_kv_groups as *const u32 as *const _);
        encoder.set_bytes(7, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(8, 4, &kv_seq as *const u32 as *const _);
        encoder.set_bytes(9, 4, &k_cap as *const u32 as *const _);
        encoder.set_bytes(10, 4, &scale as *const f32 as *const _);
        encoder.set_bytes(11, 4, &q_len as *const u32 as *const _);
        let tg_size = MTLSize::new(64, 1, 1);
        let num_tgs = num_heads * q_len;
        encoder.dispatch_thread_groups(MTLSize::new(num_tgs as u64, 1, 1), tg_size);
    }

    pub fn encode_attention_causal_f16(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        out_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        k_cap: u32,
        scale: f32,
        q_len: u32,
        q_start: u32,
        attention_window: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.attention_causal_f16_pipeline);
        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(k_cache_buf), 0);
        encoder.set_buffer(2, Some(v_cache_buf), 0);
        encoder.set_buffer(3, Some(out_buf), 0);
        encoder.set_bytes(4, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(5, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(6, 4, &num_kv_groups as *const u32 as *const _);
        encoder.set_bytes(7, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(8, 4, &kv_seq as *const u32 as *const _);
        encoder.set_bytes(9, 4, &k_cap as *const u32 as *const _);
        encoder.set_bytes(10, 4, &scale as *const f32 as *const _);
        encoder.set_bytes(11, 4, &q_len as *const u32 as *const _);
        encoder.set_bytes(12, 4, &q_start as *const u32 as *const _);
        encoder.set_bytes(13, 4, &attention_window as *const u32 as *const _);
        let tg_size = attention_threadgroup_size(self.use_flash_attention);
        let num_tgs = num_heads * q_len;
        encoder.dispatch_thread_groups(MTLSize::new(num_tgs as u64, 1, 1), tg_size);
    }

    pub fn encode_attention_causal_strided_f16(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        out_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        k_cap: u32,
        scale: f32,
        q_len: u32,
        q_pos_start: u32,
        attention_window: u32,
        q_stride: u32,
        q_start_row: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.attention_causal_strided_f16_pipeline);
        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(k_cache_buf), 0);
        encoder.set_buffer(2, Some(v_cache_buf), 0);
        encoder.set_buffer(3, Some(out_buf), 0);
        encoder.set_bytes(4, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(5, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(6, 4, &num_kv_groups as *const u32 as *const _);
        encoder.set_bytes(7, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(8, 4, &kv_seq as *const u32 as *const _);
        encoder.set_bytes(9, 4, &k_cap as *const u32 as *const _);
        encoder.set_bytes(10, 4, &scale as *const f32 as *const _);
        encoder.set_bytes(11, 4, &q_len as *const u32 as *const _);
        encoder.set_bytes(12, 4, &q_pos_start as *const u32 as *const _);
        encoder.set_bytes(13, 4, &attention_window as *const u32 as *const _);
        encoder.set_bytes(14, 4, &q_stride as *const u32 as *const _);
        encoder.set_bytes(15, 4, &q_start_row as *const u32 as *const _);
        let tg_size = attention_threadgroup_size(self.use_flash_attention);
        let num_tgs = num_heads * q_len;
        encoder.dispatch_thread_groups(MTLSize::new(num_tgs as u64, 1, 1), tg_size);
    }

    pub fn encode_transpose_shd(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input: &Buffer,
        output: &Buffer,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
    ) {
        let total = seq_len * num_heads * head_dim;
        encoder.set_compute_pipeline_state(&self.transpose_shd_pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(2, 4, &seq_len as *const u32 as *const _);
        encoder.set_bytes(3, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(4, 4, &head_dim as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_transpose_hsd(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input: &Buffer,
        output: &Buffer,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
    ) {
        let total = seq_len * num_heads * head_dim;
        encoder.set_compute_pipeline_state(&self.transpose_hsd_pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(2, 4, &seq_len as *const u32 as *const _);
        encoder.set_bytes(3, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(4, 4, &head_dim as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_kv_append_q8_0(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        new_data: &Buffer,
        cache: &Buffer,
        num_kv_heads: u32,
        head_dim: u32,
        capacity: u32,
        cur_seq: u32,
    ) {
        let groups_per_row = head_dim / 32;
        let total = num_kv_heads * groups_per_row;
        encoder.set_compute_pipeline_state(&self.kv_append_q8_0_pipeline);
        encoder.set_buffer(0, Some(new_data), 0);
        encoder.set_buffer(1, Some(cache), 0);
        encoder.set_bytes(2, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(3, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(5, 4, &cur_seq as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_kv_append_q8_0_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        new_data: &Buffer,
        new_data_offset: u64,
        cache: &Buffer,
        num_kv_heads: u32,
        head_dim: u32,
        capacity: u32,
        cur_seq: u32,
    ) {
        let groups_per_row = head_dim / 32;
        let total = num_kv_heads * groups_per_row;
        encoder.set_compute_pipeline_state(&self.kv_append_q8_0_pipeline);
        encoder.set_buffer(0, Some(new_data), new_data_offset);
        encoder.set_buffer(1, Some(cache), 0);
        encoder.set_bytes(2, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(3, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(5, 4, &cur_seq as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_kv_append_q4_0(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        new_data: &Buffer,
        cache: &Buffer,
        num_kv_heads: u32,
        head_dim: u32,
        capacity: u32,
        cur_seq: u32,
    ) {
        let groups_per_row = head_dim / 32;
        let total = num_kv_heads * groups_per_row;
        encoder.set_compute_pipeline_state(&self.kv_append_q4_0_pipeline);
        encoder.set_buffer(0, Some(new_data), 0);
        encoder.set_buffer(1, Some(cache), 0);
        encoder.set_bytes(2, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(3, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(5, 4, &cur_seq as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_kv_append_q4_0_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        new_data: &Buffer,
        new_data_offset: u64,
        cache: &Buffer,
        num_kv_heads: u32,
        head_dim: u32,
        capacity: u32,
        cur_seq: u32,
    ) {
        let groups_per_row = head_dim / 32;
        let total = num_kv_heads * groups_per_row;
        encoder.set_compute_pipeline_state(&self.kv_append_q4_0_pipeline);
        encoder.set_buffer(0, Some(new_data), new_data_offset);
        encoder.set_buffer(1, Some(cache), 0);
        encoder.set_bytes(2, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(3, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(5, 4, &cur_seq as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_kv_batch_append_q8_0(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        new_data: &Buffer,
        cache: &Buffer,
        num_kv_heads: u32,
        head_dim: u32,
        capacity: u32,
        cur_seq: u32,
        seq_len: u32,
    ) {
        let groups_per_row = head_dim / 32;
        let total = num_kv_heads * seq_len * groups_per_row;
        encoder.set_compute_pipeline_state(&self.kv_batch_append_q8_0_pipeline);
        encoder.set_buffer(0, Some(new_data), 0);
        encoder.set_buffer(1, Some(cache), 0);
        encoder.set_bytes(2, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(3, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(5, 4, &cur_seq as *const u32 as *const _);
        encoder.set_bytes(6, 4, &seq_len as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_kv_batch_append_strided_q8_0(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        new_data: &Buffer,
        cache: &Buffer,
        num_kv_heads: u32,
        head_dim: u32,
        capacity: u32,
        cur_seq: u32,
        seq_len: u32,
        source_seq_stride: u32,
        source_start: u32,
    ) {
        let groups_per_row = head_dim / 32;
        let total = num_kv_heads * seq_len * groups_per_row;
        encoder.set_compute_pipeline_state(&self.kv_batch_append_strided_q8_0_pipeline);
        encoder.set_buffer(0, Some(new_data), 0);
        encoder.set_buffer(1, Some(cache), 0);
        encoder.set_bytes(2, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(3, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(5, 4, &cur_seq as *const u32 as *const _);
        encoder.set_bytes(6, 4, &seq_len as *const u32 as *const _);
        encoder.set_bytes(7, 4, &source_seq_stride as *const u32 as *const _);
        encoder.set_bytes(8, 4, &source_start as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_kv_batch_append_q4_0(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        new_data: &Buffer,
        cache: &Buffer,
        num_kv_heads: u32,
        head_dim: u32,
        capacity: u32,
        cur_seq: u32,
        seq_len: u32,
    ) {
        let groups_per_row = head_dim / 32;
        let total = num_kv_heads * seq_len * groups_per_row;
        encoder.set_compute_pipeline_state(&self.kv_batch_append_q4_0_pipeline);
        encoder.set_buffer(0, Some(new_data), 0);
        encoder.set_buffer(1, Some(cache), 0);
        encoder.set_bytes(2, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(3, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(5, 4, &cur_seq as *const u32 as *const _);
        encoder.set_bytes(6, 4, &seq_len as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_kv_batch_append_strided_q4_0(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        new_data: &Buffer,
        cache: &Buffer,
        num_kv_heads: u32,
        head_dim: u32,
        capacity: u32,
        cur_seq: u32,
        seq_len: u32,
        source_seq_stride: u32,
        source_start: u32,
    ) {
        let groups_per_row = head_dim / 32;
        let total = num_kv_heads * seq_len * groups_per_row;
        encoder.set_compute_pipeline_state(&self.kv_batch_append_strided_q4_0_pipeline);
        encoder.set_buffer(0, Some(new_data), 0);
        encoder.set_buffer(1, Some(cache), 0);
        encoder.set_bytes(2, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(3, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(5, 4, &cur_seq as *const u32 as *const _);
        encoder.set_bytes(6, 4, &seq_len as *const u32 as *const _);
        encoder.set_bytes(7, 4, &source_seq_stride as *const u32 as *const _);
        encoder.set_bytes(8, 4, &source_start as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_attention_with_offset_q8_0(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        out_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        capacity: u32,
        scale: f32,
        kv_start: u32,
        groups_per_row: u32,
        row_bytes: u32,
    ) {
        self.encode_attention_with_offset_q8_0_at(
            encoder, q_buf, 0, k_cache_buf, v_cache_buf, out_buf, 0,
            num_heads, num_kv_heads, num_kv_groups, head_dim, kv_seq, capacity, scale, kv_start, groups_per_row, row_bytes,
        );
    }

    pub fn encode_attention_with_offset_q8_0_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        q_offset: u64,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        out_buf: &Buffer,
        out_offset: u64,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        capacity: u32,
        scale: f32,
        kv_start: u32,
        groups_per_row: u32,
        row_bytes: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.attention_offset_q8_0_pipeline);
        encoder.set_buffer(0, Some(q_buf), q_offset);
        encoder.set_buffer(1, Some(k_cache_buf), 0);
        encoder.set_buffer(2, Some(v_cache_buf), 0);
        encoder.set_buffer(3, Some(out_buf), out_offset);
        encoder.set_bytes(4, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(5, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(6, 4, &num_kv_groups as *const u32 as *const _);
        encoder.set_bytes(7, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(8, 4, &kv_seq as *const u32 as *const _);
        encoder.set_bytes(9, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(10, 4, &scale as *const f32 as *const _);
        encoder.set_bytes(11, 4, &kv_start as *const u32 as *const _);
        encoder.set_bytes(12, 4, &groups_per_row as *const u32 as *const _);
        encoder.set_bytes(13, 4, &row_bytes as *const u32 as *const _);
        let tg_size = MTLSize::new(64, 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(num_heads as u64, 1, 1), tg_size);
    }

    pub fn encode_attention_with_offset_q4_0(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        out_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        capacity: u32,
        scale: f32,
        kv_start: u32,
        groups_per_row: u32,
        row_bytes: u32,
    ) {
        self.encode_attention_with_offset_q4_0_at(
            encoder, q_buf, 0, k_cache_buf, v_cache_buf, out_buf, 0,
            num_heads, num_kv_heads, num_kv_groups, head_dim, kv_seq, capacity, scale, kv_start, groups_per_row, row_bytes,
        );
    }

    fn attention_offset_q4_0_pipeline_for(&self, head_dim: u32) -> &ComputePipelineState {
        if self.use_flash_attention && attention_q4_hd_specialized() {
            match head_dim {
                256 => return &self.attention_offset_q4_0_h256_pipeline,
                128 => return &self.attention_offset_q4_0_h128_pipeline,
                512 => return &self.attention_offset_q4_0_h512_pipeline,
                _ => {}
            }
        }
        &self.attention_offset_q4_0_pipeline
    }

    pub fn encode_attention_with_offset_q4_0_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        q_offset: u64,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        out_buf: &Buffer,
        out_offset: u64,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        capacity: u32,
        scale: f32,
        kv_start: u32,
        groups_per_row: u32,
        row_bytes: u32,
    ) {
        encoder.set_compute_pipeline_state(self.attention_offset_q4_0_pipeline_for(head_dim));
        encoder.set_buffer(0, Some(q_buf), q_offset);
        encoder.set_buffer(1, Some(k_cache_buf), 0);
        encoder.set_buffer(2, Some(v_cache_buf), 0);
        encoder.set_buffer(3, Some(out_buf), out_offset);
        encoder.set_bytes(4, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(5, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(6, 4, &num_kv_groups as *const u32 as *const _);
        encoder.set_bytes(7, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(8, 4, &kv_seq as *const u32 as *const _);
        encoder.set_bytes(9, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(10, 4, &scale as *const f32 as *const _);
        encoder.set_bytes(11, 4, &kv_start as *const u32 as *const _);
        encoder.set_bytes(12, 4, &groups_per_row as *const u32 as *const _);
        encoder.set_bytes(13, 4, &row_bytes as *const u32 as *const _);
        let tg_size = attention_threadgroup_size(self.use_flash_attention);
        encoder.dispatch_thread_groups(MTLSize::new(num_heads as u64, 1, 1), tg_size);
    }

    /// GQA-aware tiled Q4_0 flash decode: one threadgroup per KV head, all
    /// query heads in the GQA group processed together.
    pub fn encode_attention_with_offset_q4_0_gqa(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        out_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        capacity: u32,
        scale: f32,
        kv_start: u32,
        groups_per_row: u32,
        row_bytes: u32,
    ) {
        self.encode_attention_with_offset_q4_0_gqa_at(
            encoder, q_buf, 0, k_cache_buf, v_cache_buf, out_buf, 0,
            num_heads, num_kv_heads, num_kv_groups, head_dim, kv_seq, capacity,
            scale, kv_start, groups_per_row, row_bytes,
        );
    }

    pub fn encode_attention_with_offset_q4_0_gqa_at(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        q_offset: u64,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        out_buf: &Buffer,
        out_offset: u64,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        capacity: u32,
        scale: f32,
        kv_start: u32,
        groups_per_row: u32,
        row_bytes: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.attention_offset_q4_0_gqa_pipeline);
        encoder.set_buffer(0, Some(q_buf), q_offset);
        encoder.set_buffer(1, Some(k_cache_buf), 0);
        encoder.set_buffer(2, Some(v_cache_buf), 0);
        encoder.set_buffer(3, Some(out_buf), out_offset);
        encoder.set_bytes(4, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(5, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(6, 4, &num_kv_groups as *const u32 as *const _);
        encoder.set_bytes(7, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(8, 4, &kv_seq as *const u32 as *const _);
        encoder.set_bytes(9, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(10, 4, &scale as *const f32 as *const _);
        encoder.set_bytes(11, 4, &kv_start as *const u32 as *const _);
        encoder.set_bytes(12, 4, &groups_per_row as *const u32 as *const _);
        encoder.set_bytes(13, 4, &row_bytes as *const u32 as *const _);
        let tg_size = attention_threadgroup_size(self.use_flash_attention);
        encoder.dispatch_thread_groups(MTLSize::new(num_kv_heads as u64, 1, 1), tg_size);
    }

    pub fn encode_attention_causal_q8_0(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        out_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        capacity: u32,
        scale: f32,
        q_len: u32,
        q_start: u32,
        attention_window: u32,
        groups_per_row: u32,
        row_bytes: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.attention_causal_q8_0_pipeline);
        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(k_cache_buf), 0);
        encoder.set_buffer(2, Some(v_cache_buf), 0);
        encoder.set_buffer(3, Some(out_buf), 0);
        encoder.set_bytes(4, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(5, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(6, 4, &num_kv_groups as *const u32 as *const _);
        encoder.set_bytes(7, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(8, 4, &kv_seq as *const u32 as *const _);
        encoder.set_bytes(9, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(10, 4, &scale as *const f32 as *const _);
        encoder.set_bytes(11, 4, &q_len as *const u32 as *const _);
        encoder.set_bytes(12, 4, &q_start as *const u32 as *const _);
        encoder.set_bytes(13, 4, &attention_window as *const u32 as *const _);
        encoder.set_bytes(14, 4, &groups_per_row as *const u32 as *const _);
        encoder.set_bytes(15, 4, &row_bytes as *const u32 as *const _);
        let tg_size = MTLSize::new(64, 1, 1);
        let num_tgs = num_heads * q_len;
        encoder.dispatch_thread_groups(MTLSize::new(num_tgs as u64, 1, 1), tg_size);
    }

    pub fn encode_attention_causal_strided_q8_0(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        out_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        capacity: u32,
        scale: f32,
        q_len: u32,
        q_pos_start: u32,
        attention_window: u32,
        q_stride: u32,
        q_start_row: u32,
        groups_per_row: u32,
        row_bytes: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.attention_causal_strided_q8_0_pipeline);
        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(k_cache_buf), 0);
        encoder.set_buffer(2, Some(v_cache_buf), 0);
        encoder.set_buffer(3, Some(out_buf), 0);
        encoder.set_bytes(4, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(5, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(6, 4, &num_kv_groups as *const u32 as *const _);
        encoder.set_bytes(7, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(8, 4, &kv_seq as *const u32 as *const _);
        encoder.set_bytes(9, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(10, 4, &scale as *const f32 as *const _);
        encoder.set_bytes(11, 4, &q_len as *const u32 as *const _);
        encoder.set_bytes(12, 4, &q_pos_start as *const u32 as *const _);
        encoder.set_bytes(13, 4, &attention_window as *const u32 as *const _);
        encoder.set_bytes(14, 4, &q_stride as *const u32 as *const _);
        encoder.set_bytes(15, 4, &q_start_row as *const u32 as *const _);
        encoder.set_bytes(16, 4, &groups_per_row as *const u32 as *const _);
        encoder.set_bytes(17, 4, &row_bytes as *const u32 as *const _);
        let tg_size = MTLSize::new(64, 1, 1);
        let num_tgs = num_heads * q_len;
        encoder.dispatch_thread_groups(MTLSize::new(num_tgs as u64, 1, 1), tg_size);
    }

    pub fn encode_attention_causal_q4_0(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        out_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        capacity: u32,
        scale: f32,
        q_len: u32,
        q_start: u32,
        attention_window: u32,
        groups_per_row: u32,
        row_bytes: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.attention_causal_q4_0_pipeline);
        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(k_cache_buf), 0);
        encoder.set_buffer(2, Some(v_cache_buf), 0);
        encoder.set_buffer(3, Some(out_buf), 0);
        encoder.set_bytes(4, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(5, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(6, 4, &num_kv_groups as *const u32 as *const _);
        encoder.set_bytes(7, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(8, 4, &kv_seq as *const u32 as *const _);
        encoder.set_bytes(9, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(10, 4, &scale as *const f32 as *const _);
        encoder.set_bytes(11, 4, &q_len as *const u32 as *const _);
        encoder.set_bytes(12, 4, &q_start as *const u32 as *const _);
        encoder.set_bytes(13, 4, &attention_window as *const u32 as *const _);
        encoder.set_bytes(14, 4, &groups_per_row as *const u32 as *const _);
        encoder.set_bytes(15, 4, &row_bytes as *const u32 as *const _);
        let tg_size = attention_threadgroup_size(self.use_flash_attention);
        let num_tgs = num_heads * q_len;
        encoder.dispatch_thread_groups(MTLSize::new(num_tgs as u64, 1, 1), tg_size);
    }

    /// Prefill causal Q4_0 attention: llama.cpp flash_attn_ext when q_len ≥ 20, else per-query flash.
    pub fn encode_prefill_attention_causal_q4_0(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        out_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        capacity: u32,
        scale: f32,
        q_len: u32,
        q_start: u32,
        attention_window: u32,
        groups_per_row: u32,
        row_bytes: u32,
        fa_ext_scratch: Option<&Buffer>,
        fa_ext_layout: Option<&crate::ggml_flash_attn_ext::ScratchLayout>,
        fa_ext_mask_cache: Option<&mut crate::ggml_flash_attn_ext::PrefillExtMaskCache>,
    ) {
        let _ = (num_kv_groups, groups_per_row);
        if prefill_flash_attn_ext_enabled()
            && prefill_use_flash_attn_ext_tiled(q_len, head_dim)
        {
            if let (Some(scratch), Some(layout)) = (fa_ext_scratch, fa_ext_layout) {
                self.encode_prefill_flash_attn_ext_q4_0(
                    encoder,
                    q_buf,
                    k_cache_buf,
                    v_cache_buf,
                    out_buf,
                    scratch,
                    layout,
                    fa_ext_mask_cache,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    kv_seq,
                    capacity,
                    scale,
                    q_len,
                    q_start,
                    attention_window,
                    row_bytes as u64,
                );
                return;
            }
        }

        self.encode_attention_causal_q4_0(
            encoder,
            q_buf,
            k_cache_buf,
            v_cache_buf,
            out_buf,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            kv_seq,
            capacity,
            scale,
            q_len,
            q_start,
            attention_window,
            groups_per_row,
            row_bytes,
        );
    }

    fn encode_prefill_flash_attn_ext_q4_0(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        out_buf: &Buffer,
        scratch: &Buffer,
        layout: &crate::ggml_flash_attn_ext::ScratchLayout,
        mask_cache: Option<&mut crate::ggml_flash_attn_ext::PrefillExtMaskCache>,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        kv_seq: u32,
        capacity: u32,
        scale: f32,
        q_len: u32,
        q_start: u32,
        attention_window: u32,
        row_bytes: u64,
    ) {
        use crate::ggml_flash_attn_ext::{
            blk_args, fill_causal_mask, flash_attn_ext_args, nsg_for_head_dim, pad_args, smem_bytes,
            FlashAttnExtArgs, FlashAttnExtBlkArgs, FlashAttnExtPadArgs, NCPSG, NQPTG,
        };

        let pad_off = layout.pad_off;
        let blk_off = layout.blk_off;
        let mask_off = layout.mask_off;

        let need_mask_blk = match mask_cache.as_ref() {
            Some(cache) => !cache.matches(q_len, kv_seq, q_start, attention_window),
            None => true,
        };
        if need_mask_blk {
            let mask_len = (q_len as usize) * (kv_seq as usize);
            let mut mask_host = vec![0u16; mask_len];
            fill_causal_mask(
                &mut mask_host,
                q_len,
                kv_seq,
                q_start,
                attention_window,
            );
            unsafe {
                let base = scratch.contents() as *mut u8;
                std::ptr::copy_nonoverlapping(
                    mask_host.as_ptr() as *const u8,
                    base.add(layout.mask_off as usize),
                    mask_len * 2,
                );
            }

            let blk_args = blk_args(q_len, kv_seq);
            encoder.set_compute_pipeline_state(&self.flash_attn_ext_prefill_blk_pipeline);
            encoder.set_bytes(
                0,
                std::mem::size_of::<FlashAttnExtBlkArgs>() as u64,
                &blk_args as *const _ as *const _,
            );
            encoder.set_buffer(1, Some(scratch), mask_off);
            encoder.set_buffer(2, Some(scratch), blk_off);
            let nblk1 = ((q_len + NQPTG - 1) / NQPTG) as u64;
            let nblk0 = ((kv_seq + NCPSG - 1) / NCPSG) as u64;
            encoder.dispatch_thread_groups(
                metal::MTLSize::new(nblk0, nblk1, 1),
                metal::MTLSize::new(32, 1, 1),
            );

            if let Some(cache) = mask_cache {
                cache.mark(q_len, kv_seq, q_start, attention_window);
            }
        }

        let has_kvpad = kv_seq % NCPSG != 0;
        if has_kvpad {
            let pad_args = pad_args(num_kv_heads, kv_seq, row_bytes, q_len);
            encoder.set_compute_pipeline_state(&self.flash_attn_ext_prefill_pad_pipeline);
            encoder.set_bytes(
                0,
                std::mem::size_of::<FlashAttnExtPadArgs>() as u64,
                &pad_args as *const _ as *const _,
            );
            encoder.set_buffer(1, Some(k_cache_buf), 0);
            encoder.set_buffer(2, Some(v_cache_buf), 0);
            encoder.set_buffer(3, Some(scratch), mask_off);
            encoder.set_buffer(4, Some(scratch), pad_off);
            let ne12 = num_kv_heads.max(1);
            encoder.dispatch_thread_groups(
                metal::MTLSize::new(NCPSG as u64, ne12 as u64, 1),
                metal::MTLSize::new(32, 1, 1),
            );
        }

        let nsg = nsg_for_head_dim(head_dim);
        let args = flash_attn_ext_args(
            num_heads,
            num_kv_heads,
            head_dim,
            kv_seq,
            capacity,
            row_bytes,
            q_len,
            scale,
        );
        let main_pipeline = match head_dim {
            256 => &self.flash_attn_ext_prefill_q4_h256_pipeline,
            512 => &self.flash_attn_ext_prefill_q4_h512_pipeline,
            _ => unreachable!("prefill flash_attn_ext only supports h256/h512"),
        };
        encoder.set_compute_pipeline_state(main_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<FlashAttnExtArgs>() as u64,
            &args as *const _ as *const _,
        );
        encoder.set_buffer(1, Some(q_buf), 0);
        encoder.set_buffer(2, Some(k_cache_buf), 0);
        encoder.set_buffer(3, Some(v_cache_buf), 0);
        encoder.set_buffer(4, Some(scratch), mask_off);
        encoder.set_buffer(5, Some(scratch), pad_off);
        encoder.set_buffer(6, Some(scratch), blk_off);
        encoder.set_buffer(7, Some(out_buf), 0);
        encoder.set_threadgroup_memory_length(0, smem_bytes(head_dim, nsg));
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(((q_len + NQPTG - 1) / NQPTG) as u64, num_heads as u64, 1),
            metal::MTLSize::new(32, nsg as u64, 1),
        );
    }

    pub fn encode_attention_causal_strided_q4_0(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        q_buf: &Buffer,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        out_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        capacity: u32,
        scale: f32,
        q_len: u32,
        q_pos_start: u32,
        attention_window: u32,
        q_stride: u32,
        q_start_row: u32,
        groups_per_row: u32,
        row_bytes: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.attention_causal_strided_q4_0_pipeline);
        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(k_cache_buf), 0);
        encoder.set_buffer(2, Some(v_cache_buf), 0);
        encoder.set_buffer(3, Some(out_buf), 0);
        encoder.set_bytes(4, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(5, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(6, 4, &num_kv_groups as *const u32 as *const _);
        encoder.set_bytes(7, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(8, 4, &kv_seq as *const u32 as *const _);
        encoder.set_bytes(9, 4, &capacity as *const u32 as *const _);
        encoder.set_bytes(10, 4, &scale as *const f32 as *const _);
        encoder.set_bytes(11, 4, &q_len as *const u32 as *const _);
        encoder.set_bytes(12, 4, &q_pos_start as *const u32 as *const _);
        encoder.set_bytes(13, 4, &attention_window as *const u32 as *const _);
        encoder.set_bytes(14, 4, &q_stride as *const u32 as *const _);
        encoder.set_bytes(15, 4, &q_start_row as *const u32 as *const _);
        encoder.set_bytes(16, 4, &groups_per_row as *const u32 as *const _);
        encoder.set_bytes(17, 4, &row_bytes as *const u32 as *const _);
        let tg_size = attention_threadgroup_size(self.use_flash_attention);
        let num_tgs = num_heads * q_len;
        encoder.dispatch_thread_groups(MTLSize::new(num_tgs as u64, 1, 1), tg_size);
    }

    // ─── Legacy standalone dispatch methods (kept for compatibility) ─────────

    pub fn matvec(&self, w_buf: &Buffer, x_buf: &Buffer, y_buf: &Buffer, m: u32, k: u32) {
        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        self.encode_matvec(encoder, w_buf, x_buf, y_buf, m, k);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    pub fn rmsnorm(
        &self,
        x_buf: &Buffer,
        weight_buf: &Buffer,
        out_buf: &Buffer,
        dim: u32,
        eps: f32,
    ) {
        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        self.encode_rmsnorm(encoder, x_buf, weight_buf, out_buf, dim, eps);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    pub fn silu_mul(&self, gate_buf: &Buffer, up_buf: &Buffer, out_buf: &Buffer, n: u32) {
        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        self.encode_silu_mul(encoder, gate_buf, up_buf, out_buf, n);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    pub fn attention_single_token(
        &self,
        q_buf: &Buffer,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
        out_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        num_kv_groups: u32,
        head_dim: u32,
        kv_seq: u32,
        k_cap: u32,
        scale: f32,
    ) {
        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        self.encode_attention(
            encoder,
            q_buf,
            k_cache_buf,
            v_cache_buf,
            out_buf,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            kv_seq,
            k_cap,
            scale,
        );
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    pub fn apply_rotary(
        &self,
        q_buf: &Buffer,
        k_buf: &Buffer,
        cos_buf: &Buffer,
        sin_buf: &Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) {
        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        self.encode_rotary(
            encoder,
            q_buf,
            k_buf,
            cos_buf,
            sin_buf,
            num_heads,
            num_kv_heads,
            head_dim,
        );
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    pub fn vec_add(&self, a_buf: &Buffer, b_buf: &Buffer, c_buf: &Buffer, n: u32) {
        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        self.encode_vec_add(encoder, a_buf, b_buf, c_buf, n);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
}

/// Convert f16 (IEEE 754 half-precision) to f32.
pub fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

pub fn f16_to_f32(value: u16) -> f32 {
    let sign = ((value >> 15) as f32) * -2.0 + 1.0;
    let exp = (value >> 10) & 0x1F;
    let mant = value & 0x3FF;
    if exp == 0 {
        // Subnormal or zero
        sign * (mant as f32) * (2.0f32).powi(-24)
    } else if exp == 31 {
        if mant == 0 { sign * f32::INFINITY } else { f32::NAN }
    } else {
        sign * (1.0 + (mant as f32) / 1024.0) * (2.0f32).powi(exp as i32 - 15)
    }
}

/// Convert f32 to f16 (IEEE 754 half-precision).
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3FF;

    if exp <= 0 {
        // Subnormal or zero
        if exp < -10 {
            sign as u16
        } else {
            let mant = (mant | 0x400) >> (1 - exp);
            (sign | mant) as u16
        }
    } else if exp >= 31 {
        // Overflow → infinity
        (sign | 0x7C00) as u16
    } else {
        (sign | ((exp as u32) << 10) | mant) as u16
    }
}

/// Quantize f32 weights to Q3_0 format (3-bit symmetric, group=32).
/// Block = 14 bytes: f16 scale + 8 bytes low-2-bits + 4 bytes high-1-bit.
/// 32 weights × 3 bits = 96 bits = 12 bytes payload + 2 byte scale = 14 bytes.
/// Reconstruction: q ∈ [0,7], value = (q - 4) · d,  d = max_abs / 3.
fn quantize_q3_0(data: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(cols % 32, 0, "cols must be divisible by 32 for Q3_0");
    let num_groups_per_row = cols / 32;
    let bytes_per_row = num_groups_per_row * 14;
    let mut output = vec![0u8; rows * bytes_per_row];

    for row in 0..rows {
        for g in 0..num_groups_per_row {
            let group_start = row * cols + g * 32;
            let group = &data[group_start..group_start + 32];

            let mut max_abs = 0.0f32;
            for &v in group { let a = v.abs(); if a > max_abs { max_abs = a; } }

            let scale = if max_abs > 0.0 { max_abs / 3.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;

            let scale_f16 = f32_to_f16(scale);
            let out_offset = row * bytes_per_row + g * 14;
            output[out_offset] = (scale_f16 & 0xFF) as u8;
            output[out_offset + 1] = (scale_f16 >> 8) as u8;

            // Pack 32 3-bit values: low 2 bits in qs_low[0..7], high 1 bit in qs_high[0..3]
            for i in 0..32 {
                let v = group[i];
                let q = ((v * inv_scale).round() as i32 + 4).clamp(0, 7) as u8;
                let low = q & 0x3;
                let high = (q >> 2) & 0x1;
                // qs_low: 4 weights per byte, 2 bits each
                let low_byte = i / 4;
                let low_shift = (i % 4) * 2;
                output[out_offset + 2 + low_byte] |= low << low_shift;
                // qs_high: 8 weights per byte, 1 bit each
                let high_byte = i / 8;
                let high_shift = i % 8;
                output[out_offset + 10 + high_byte] |= high << high_shift;
            }
        }
    }
    output
}

/// Dequantize Q3_0 to f32 (CPU, for testing).
fn dequantize_q3_0(data: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    let num_groups_per_row = cols / 32;
    let bytes_per_row = num_groups_per_row * 14;
    let mut output = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for g in 0..num_groups_per_row {
            let in_offset = row * bytes_per_row + g * 14;
            let raw_scale = u16::from_le_bytes([data[in_offset], data[in_offset + 1]]);
            let d = f16_to_f32(raw_scale);

            for i in 0..32 {
                let low_byte = i / 4;
                let low_shift = (i % 4) * 2;
                let low = (data[in_offset + 2 + low_byte] >> low_shift) & 0x3;
                let high_byte = i / 8;
                let high_shift = i % 8;
                let high = (data[in_offset + 10 + high_byte] >> high_shift) & 0x1;
                let q = low | (high << 2);
                let v = d * ((q as i32 - 4) as f32);
                output[row * cols + g * 32 + i] = v;
            }
        }
    }
    output
}

/// Quantize f32 weights to Q4_0 format.
/// For each row, process groups of 32 values:
///   - Find max absolute value → scale = max_abs / 7
///   - Quantize each value to 4-bit unsigned: q = round(v / scale) + 8, clamped to [0, 15]
///   - Pack GGUF layout: byte i = low nibble elem i, high nibble elem i+16
///   - Store: [f16 scale][16 bytes packed quants]
fn quantize_q4_0(data: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(cols % 32, 0, "cols must be divisible by 32 for Q4_0");
    let num_groups_per_row = cols / 32;
    let bytes_per_row = num_groups_per_row * 18; // 18 bytes per group
    let mut output = vec![0u8; rows * bytes_per_row];

    for row in 0..rows {
        for g in 0..num_groups_per_row {
            let group_start = row * cols + g * 32;
            let group = &data[group_start..group_start + 32];

            // Find max absolute value
            let mut max_abs = 0.0f32;
            for &v in group {
                let a = v.abs();
                if a > max_abs {
                    max_abs = a;
                }
            }

            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;

            // Write scale as f16
            let scale_f16 = f32_to_f16(scale);
            let out_offset = row * bytes_per_row + g * 18;
            output[out_offset] = (scale_f16 & 0xFF) as u8;
            output[out_offset + 1] = (scale_f16 >> 8) as u8;

            // GGUF Q4_0: bytes 0-15 hold low nibbles of elems 0-15, high nibbles of elems 16-31
            for i in 0..16 {
                let v_lo = group[i];
                let v_hi = group[i + 16];

                let q_lo = ((v_lo * inv_scale).round() as i32 + 8).clamp(0, 15) as u8;
                let q_hi = ((v_hi * inv_scale).round() as i32 + 8).clamp(0, 15) as u8;

                output[out_offset + 2 + i] = q_lo | (q_hi << 4);
            }
        }
    }

    output
}

/// Skip decode sections for bottleneck ablation (`PROFILE_ABLATE=attn|mlp|ple|head`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileAblate {
    None,
    Attn,
    Mlp,
    Ple,
    Head,
}

impl ProfileAblate {
    pub fn from_env() -> Self {
        match std::env::var("PROFILE_ABLATE").as_deref() {
            Ok("attn") | Ok("ATTN") | Ok("attention") => ProfileAblate::Attn,
            Ok("mlp") | Ok("MLP") => ProfileAblate::Mlp,
            Ok("ple") | Ok("PLE") => ProfileAblate::Ple,
            Ok("head") | Ok("HEAD") | Ok("lm_head") => ProfileAblate::Head,
            _ => ProfileAblate::None,
        }
    }

    pub fn active(&self) -> bool {
        !matches!(self, ProfileAblate::None)
    }

    pub fn skip_attn(&self) -> bool {
        matches!(self, ProfileAblate::Attn)
    }

    pub fn skip_mlp(&self) -> bool {
        matches!(self, ProfileAblate::Mlp)
    }

    pub fn skip_ple(&self) -> bool {
        matches!(self, ProfileAblate::Ple)
    }

    pub fn skip_head(&self) -> bool {
        matches!(self, ProfileAblate::Head)
    }

    pub fn log_once(&self) {
        if !self.active() {
            return;
        }
        static WARN: std::sync::Once = std::sync::Once::new();
        WARN.call_once(|| {
            eprintln!(
                "  Ablation profiling: skipping {:?} (PROFILE_ABLATE)",
                std::env::var("PROFILE_ABLATE").unwrap_or_default()
            );
        });
    }
}

pub fn profile_gpu_enabled() -> bool {
    std::env::var("PROFILE_GPU").is_ok()
}

/// Per-layer GPU timestamps inside a single command buffer (Metal counter samples).
/// Sample layout: [0]=post-rope, [1]=post-ple-prepass, per-layer [attn,mlp,ple]×N, [head].
pub struct GpuTimestampProfiler {
    sample_buffer: CounterSampleBuffer,
    resolve_buf: Buffer,
    capacity: u32,
    next: u32,
    ns_per_tick: f64,
}

#[derive(Clone, Copy, Default)]
struct GpuPhaseMs {
    prepass: f64,
    attn: f64,
    mlp: f64,
    ple: f64,
    head: f64,
    total: f64,
}

thread_local! {
    static GPU_PHASE_ACC: std::cell::Cell<(GpuPhaseMs, u64)> =
        std::cell::Cell::new((GpuPhaseMs::default(), 0));
}

impl GpuTimestampProfiler {
    pub fn try_new(device: &Device, num_layers: u32) -> Option<Self> {
        let counter_set = device.counter_sets().into_iter().find(|cs| {
            cs.name().to_ascii_lowercase().contains("timestamp")
        })?;
        let capacity = 2 + num_layers * 3 + 1;
        let desc = CounterSampleBufferDescriptor::new();
        desc.set_counter_set(&counter_set);
        desc.set_sample_count(capacity as u64);
        desc.set_storage_mode(MTLStorageMode::Shared);
        let sample_buffer = device
            .new_counter_sample_buffer_with_descriptor(&desc)
            .ok()?;
        let resolve_bytes = (capacity as u64) * std::mem::size_of::<u64>() as u64;
        let resolve_buf = device.new_buffer(
            resolve_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let mut cpu0 = 0u64;
        let mut gpu0 = 0u64;
        let mut cpu1 = 0u64;
        let mut gpu1 = 0u64;
        device.sample_timestamps(&mut cpu0, &mut gpu0);
        std::thread::sleep(std::time::Duration::from_millis(2));
        device.sample_timestamps(&mut cpu1, &mut gpu1);
        let cpu_delta = cpu1.saturating_sub(cpu0) as f64;
        let gpu_delta = gpu1.saturating_sub(gpu0) as f64;
        let ns_per_tick = if gpu_delta > 0.0 {
            (cpu_delta * 1e6) / gpu_delta
        } else {
            1.0
        };
        // M1/M2 often expose a timestamp counter set but reject encoder sampling.
        let queue = device.new_command_queue();
        let probe_cmd = queue.new_command_buffer();
        let probe_enc = probe_cmd.new_compute_command_encoder();
        probe_enc.sample_counters_in_buffer(&sample_buffer, 0, true);
        probe_enc.end_encoding();
        probe_cmd.commit();
        probe_cmd.wait_until_completed();
        if probe_cmd.status() != MTLCommandBufferStatus::Completed {
            static WARN: std::sync::Once = std::sync::Once::new();
            WARN.call_once(|| {
                eprintln!(
                    "  PROFILE_GPU=1 disabled: sampleCountersInBuffer unsupported on this GPU"
                );
            });
            return None;
        }
        Some(Self {
            sample_buffer,
            resolve_buf,
            capacity,
            next: 0,
            ns_per_tick,
        })
    }

    pub fn mark(&mut self, encoder: &metal::ComputeCommandEncoderRef) {
        if self.next >= self.capacity {
            return;
        }
        encoder.sample_counters_in_buffer(
            &self.sample_buffer,
            self.next as NSUInteger,
            true,
        );
        self.next += 1;
    }

    pub fn resolve(&self, cmd: &CommandBufferRef) {
        let blit = cmd.new_blit_command_encoder();
        blit.resolve_counters(
            &self.sample_buffer,
            NSRange {
                location: 0,
                length: self.next as u64,
            },
            &self.resolve_buf,
            0,
        );
        blit.end_encoding();
    }

    pub fn ingest(&self, num_layers: u32) {
        let n = self.next as usize;
        if n < 3 {
            return;
        }
        let ptr = self.resolve_buf.contents() as *const u64;
        let ticks: Vec<f64> = (0..n)
            .map(|i| unsafe { *ptr.add(i) } as f64 * self.ns_per_tick / 1e6)
            .collect();
        let delta = |a: usize, b: usize| {
            if b < ticks.len() && a < ticks.len() && ticks[b] >= ticks[a] {
                ticks[b] - ticks[a]
            } else {
                0.0
            }
        };
        let prepass = delta(0, 1);
        let mut attn = 0.0;
        let mut mlp = 0.0;
        let mut ple = 0.0;
        for i in 0..num_layers {
            let i = i as usize;
            let attn_start = if i == 0 { 1 } else { 2 + (i - 1) * 3 + 2 };
            let base = 2 + i * 3;
            attn += delta(attn_start, base);
            mlp += delta(base, base + 1);
            ple += delta(base + 1, base + 2);
        }
        let head_start = 2 + (num_layers.saturating_sub(1) as usize) * 3 + 2;
        let head = delta(head_start, n - 1);
        let total = ticks.last().copied().unwrap_or(0.0) - ticks.first().copied().unwrap_or(0.0);
        let phase = GpuPhaseMs {
            prepass,
            attn,
            mlp,
            ple,
            head,
            total,
        };
        GPU_PHASE_ACC.with(|c| {
            let (mut sum, mut count) = c.get();
            sum.prepass += phase.prepass;
            sum.attn += phase.attn;
            sum.mlp += phase.mlp;
            sum.ple += phase.ple;
            sum.head += phase.head;
            sum.total += phase.total;
            count += 1;
            if count % 16 == 0 {
                let nf = count as f64;
                eprintln!(
                    "[gpu profile] n={} avg ms/token (single cmdbuf): prepass={:.2} attn={:.2} mlp={:.2} ple={:.2} head={:.2} sum={:.2}",
                    count,
                    sum.prepass / nf,
                    sum.attn / nf,
                    sum.mlp / nf,
                    sum.ple / nf,
                    sum.head / nf,
                    sum.total / nf,
                );
            }
            c.set((sum, count));
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn metal_context_compiles_shaders() {
        // Creating a context compiles every Metal function in llama.metal.
        let _ctx = MetalContext::new();
    }

    #[test]
    fn q3_0_quantize_dequantize_roundtrip() {
        let mut rng = rand::thread_rng();

        let rows = 4;
        let cols = 32;

        // Random data in [-1, 1]
        let data: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let q3 = quantize_q3_0(&data, rows, cols);
        let deq = dequantize_q3_0(&q3, rows, cols);

        // Check error is reasonable (3-bit symmetric → ~12.5% relative error worst-case)
        let mut max_err = 0.0f32;
        for i in 0..data.len() {
            let err = (data[i] - deq[i]).abs();
            if err > max_err { max_err = err; }
        }
        let data_max = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let rel_err = max_err / data_max.max(1e-6);
        assert!(rel_err < 0.2, "Q3_0 rel error too high: {:.4}", rel_err);
        assert!(max_err < 0.5, "Q3_0 abs error too high: {:.4}", max_err);
    }

    #[test]
    fn q3_0_matvec_matches_cpu_reference() {
        use crate::ggml_gemv::{mul_mv_args, mul_mv_dispatch};
        use metal::*;

        let ctx = MetalContext::new();
        let device = &ctx.device;
        let queue = ctx.queue.clone();

        let rows: u32 = 8;
        let cols: u32 = 64;

        // Random weights
        let mut rng = rand::thread_rng();
        let w_data: Vec<f32> = (0..rows as usize * cols as usize)
            .map(|_| rng.gen_range(-0.5..0.5))
            .collect();

        // Random input vector
        let x_data: Vec<f32> = (0..cols as usize).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // CPU reference: dequantize and gemv
        let q3_data = quantize_q3_0(&w_data, rows as usize, cols as usize);
        let deq = dequantize_q3_0(&q3_data, rows as usize, cols as usize);
        let cpu_out: Vec<f32> = (0..rows as usize)
            .map(|r| {
                deq[r * cols as usize..(r + 1) * cols as usize]
                    .iter()
                    .zip(x_data.iter())
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect();

        // GPU buffers
        let w_buf = device.new_buffer_with_data(
            q3_data.as_ptr() as *const _,
            q3_data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let x_buf = device.new_buffer_with_data(
            x_data.as_ptr() as *const _,
            (x_data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let y_buf = device.new_buffer(
            (rows as u64) * 4,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        let args = mul_mv_args(rows, cols);
        encoder.set_compute_pipeline_state(&ctx.matvec_ggml_q3_pipeline);
        encoder.set_buffer(0, Some(&w_buf), 0);
        encoder.set_buffer(1, Some(&x_buf), 0);
        encoder.set_buffer(2, Some(&y_buf), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<crate::ggml_gemv::GgmlMulMvArgs>() as u64,
            &args as *const _ as *const _,
        );
        let (tg_x, tg_y, tg_z, tw, nsg) = mul_mv_dispatch(rows, 1);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(tg_x, tg_y, tg_z),
            metal::MTLSize::new(tw, nsg, 1),
        );
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let gpu_out: Vec<f32> = {
            let ptr = y_buf.contents() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, rows as usize).to_vec() }
        };

        let mut max_diff = 0.0f32;
        for i in 0..rows as usize {
            let diff = (cpu_out[i] - gpu_out[i]).abs();
            if diff > max_diff { max_diff = diff; }
        }
        let out_scale = cpu_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        let rel_diff = max_diff / out_scale;
        assert!(
            rel_diff < 0.02,
            "Q3_0 GPU vs CPU matvec mismatch: max_diff={:.6} rel={:.6}",
            max_diff,
            rel_diff
        );
    }

    #[test]
    fn q3_0_gelu_mul_matvec_matches_cpu_reference() {
        use crate::ggml_gemv::{mul_mv_args, mul_mv_dispatch};
        use metal::*;

        let ctx = MetalContext::new();
        let device = &ctx.device;
        let queue = ctx.queue.clone();

        let rows: u32 = 8;
        let cols: u32 = 64;

        let mut rng = rand::thread_rng();
        let gate_data: Vec<f32> = (0..rows as usize * cols as usize)
            .map(|_| rng.gen_range(-0.5..0.5))
            .collect();
        let up_data: Vec<f32> = (0..rows as usize * cols as usize)
            .map(|_| rng.gen_range(-0.5..0.5))
            .collect();
        let x_data: Vec<f32> = (0..cols as usize).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // CPU reference
        let cpu_gelu = {
            let q3_gate = quantize_q3_0(&gate_data, rows as usize, cols as usize);
            let q3_up = quantize_q3_0(&up_data, rows as usize, cols as usize);
            let dgate = dequantize_q3_0(&q3_gate, rows as usize, cols as usize);
            let dup = dequantize_q3_0(&q3_up, rows as usize, cols as usize);

            let gate_out: Vec<f32> = (0..rows as usize)
                .map(|r| {
                    dgate[r * cols as usize..(r + 1) * cols as usize]
                        .iter().zip(x_data.iter()).map(|(a, b)| a * b).sum()
                })
                .collect();
            let up_out: Vec<f32> = (0..rows as usize)
                .map(|r| {
                    dup[r * cols as usize..(r + 1) * cols as usize]
                        .iter().zip(x_data.iter()).map(|(a, b)| a * b).sum()
                })
                .collect();
            gate_out.iter().zip(up_out.iter()).map(|(&g, &u)| {
                let inner = 0.7978845608 * (g + 0.044715 * g * g * g);
                let inner_clamped = inner.clamp(-10.0, 10.0);
                (0.5 * g * (1.0 + inner_clamped.tanh())) * u
            }).collect::<Vec<f32>>()
        };
        let cpu_out = cpu_gelu;

        // GPU
        let gate_q3 = quantize_q3_0(&gate_data, rows as usize, cols as usize);
        let up_q3 = quantize_q3_0(&up_data, rows as usize, cols as usize);

        let gate_buf = device.new_buffer_with_data(
            gate_q3.as_ptr() as *const _, gate_q3.len() as u64, MTLResourceOptions::StorageModeShared);
        let up_buf = device.new_buffer_with_data(
            up_q3.as_ptr() as *const _, up_q3.len() as u64, MTLResourceOptions::StorageModeShared);
        let x_buf = device.new_buffer_with_data(
            x_data.as_ptr() as *const _, (x_data.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        let y_buf = device.new_buffer(
            (rows as u64) * 4, MTLResourceOptions::StorageModeShared);

        let cmd = queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        let args = mul_mv_args(rows, cols);
        encoder.set_compute_pipeline_state(&ctx.matvec_ggml_q3_gelu_mul_pipeline);
        encoder.set_buffer(0, Some(&gate_buf), 0);
        encoder.set_buffer(1, Some(&up_buf), 0);
        encoder.set_buffer(2, Some(&x_buf), 0);
        encoder.set_buffer(3, Some(&y_buf), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<crate::ggml_gemv::GgmlMulMvArgs>() as u64,
            &args as *const _ as *const _,
        );
        let (tg_x, tg_y, tg_z, tw, nsg) = mul_mv_dispatch(rows, 1);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(tg_x, tg_y, tg_z),
            metal::MTLSize::new(tw, nsg, 1),
        );
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let gpu_out: Vec<f32> = {
            let ptr = y_buf.contents() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, rows as usize).to_vec() }
        };

        let mut max_diff = 0.0f32;
        for i in 0..rows as usize {
            let diff = (cpu_out[i] - gpu_out[i]).abs();
            if diff > max_diff { max_diff = diff; }
        }
        let out_scale = cpu_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        let rel_diff = max_diff / out_scale;
        assert!(
            rel_diff < 0.05,
            "Q3_0 gelu_mul GPU vs CPU mismatch: max_diff={:.6} rel={:.6}",
            max_diff,
            rel_diff
        );
    }
}
