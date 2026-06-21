use metal::*;
use std::path::Path;

/// A sub-range view into a Metal buffer (offset applied at kernel bind time).
#[derive(Clone)]
pub struct BufferView {
    pub buffer: Buffer,
    pub offset: u64,
    pub length: u64,
}

impl BufferView {
    pub fn from_buffer(buffer: Buffer) -> Self {
        let length = buffer.length();
        Self {
            buffer,
            offset: 0,
            length,
        }
    }

    pub fn subrange(buffer: &Buffer, offset: u64, length: u64) -> Self {
        Self {
            buffer: buffer.clone(),
            offset,
            length,
        }
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
pub fn weight_buf_is_q4(view: &BufferView, m: u32, k: u32) -> bool {
    let q4_bytes = (m as u64) * (k as u64 / 32) * 18;
    // f16 would be m*k*2 — well above q4_bytes; allow section-alignment padding.
    view.length <= q4_bytes + 256
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum AttentionKernelMode {
    /// Shared-KV (full-attn) layers → ggml FA; has_kv layers → fused specialized h256.
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

/// llama.cpp flash_attn_ext_vec for this layer (default auto: shared-KV layers only).
pub fn attention_use_ggml_for_layer(has_kv: bool) -> bool {
    match attention_kernel_mode() {
        AttentionKernelMode::Ggml => true,
        AttentionKernelMode::Specialized | AttentionKernelMode::Generic => false,
        AttentionKernelMode::Auto => !has_kv,
    }
}

/// True when every layer uses ggml FA (ATTENTION_KERNEL=ggml).
pub fn attention_use_ggml() -> bool {
    matches!(attention_kernel_mode(), AttentionKernelMode::Ggml)
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

/// Fuse gate+up+GeLU+down MLP in one encoder call (default on).
/// Phase 1: parallel dual_gelu → gelu scratch. Phase 2: down matvec.
/// Set FUSED_MLP_GELU_DOWN=0 to use separate dispatches via FUSED_MLP_PLE.
pub fn fused_mlp_gelu_down_enabled() -> bool {
    !matches!(
        std::env::var("FUSED_MLP_GELU_DOWN").as_deref(),
        Ok("0") | Ok("false") | Ok("FALSE")
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
    pub ple_matvec_gelu_q4_pipeline: ComputePipelineState,
    pub matvec_ggml_q4_pipeline: ComputePipelineState,
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
    pub ple_gelu_mul_batch_pipeline: ComputePipelineState,
    pub vec_mul_pipeline: ComputePipelineState,
    pub vec_add_scaled_pipeline: ComputePipelineState,
    pub rmsnorm_per_head_pipeline: ComputePipelineState,
    pub rmsnorm_per_head_noweight_pipeline: ComputePipelineState,
    pub rotary_partial_pipeline: ComputePipelineState,
    pub attention_offset_pipeline: ComputePipelineState,
    pub attention_offset_f16_pipeline: ComputePipelineState,
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
    pub flash_attn_ggml_q4_h256_pipeline: ComputePipelineState,
    pub flash_attn_ggml_q4_h128_pipeline: ComputePipelineState,
    pub flash_attn_ggml_q4_h512_pipeline: ComputePipelineState,
    pub attention_causal_q4_0_pipeline: ComputePipelineState,
    pub attention_causal_strided_q4_0_pipeline: ComputePipelineState,
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
        let ple_matvec_gelu_q4_pipeline = get_fn("ple_matvec_gelu_q4");
        let matvec_ggml_q4_pipeline = get_fn("matvec_ggml_q4_0");
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
        let flash_attn_ggml_q4_h256_pipeline = get_fn("flash_attn_ggml_q4_0_h256");
        let flash_attn_ggml_q4_h128_pipeline = get_fn("flash_attn_ggml_q4_0_h128");
        let flash_attn_ggml_q4_h512_pipeline = get_fn("flash_attn_ggml_q4_0_h512");
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
        let embed_gather_bf16_pipeline = get_fn("embed_gather_bf16");
        let embed_gather_bf16_batch_pipeline = get_fn("embed_gather_bf16_batch");
        let sample_token_pipeline = get_fn("sample_token");
        let decode_mega_gemma4_pipeline = get_fn("decode_mega_gemma4_q4_0");
        if use_flash_attention {
            println!("  FlashAttention-style tiled kernels enabled (FLASH_ATTN=legacy to disable)");
            match attention_kernel_mode() {
                AttentionKernelMode::Ggml => {
                    println!("  Q4 decode attention: ggml flash_attn_ext_vec (ATTENTION_KERNEL=ggml)");
                }
                AttentionKernelMode::Auto => {
                    println!(
                        "  Q4 decode attention: auto — fused h256 (has_kv) + ggml h512 (ATTENTION_KERNEL=auto)"
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
            if packed_mlp_gate_up_enabled() {
                println!("  Interleaved gate∥up Q4 MLP weights (PACKED_MLP_GATE_UP=0 to disable)");
            }
            if mlp_gelu_f16_enabled() {
                println!("  MLP f16 GeLU scratch gate→down (MLP_GELU_F16=1 enabled)");
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
                println!("  Fused pre-attn RMSNorm + Q4 Q/K/V projections (FUSED_QKV=0 to disable)");
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
            ple_matvec_gelu_q4_pipeline,
            matvec_ggml_q4_pipeline,
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
            ple_gelu_mul_batch_pipeline,
            vec_mul_pipeline,
            vec_add_scaled_pipeline,
            rmsnorm_per_head_pipeline,
            rmsnorm_per_head_noweight_pipeline,
            rotary_partial_pipeline,
            attention_offset_pipeline,
            attention_offset_f16_pipeline,
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
            flash_attn_ggml_q4_h256_pipeline,
            flash_attn_ggml_q4_h128_pipeline,
            flash_attn_ggml_q4_h512_pipeline,
            attention_causal_q4_0_pipeline,
            attention_causal_strided_q4_0_pipeline,
            embed_gather_bf16_pipeline,
            embed_gather_bf16_batch_pipeline,
            sample_token_pipeline,
            use_flash_attention,
            decode_mega_gemma4_pipeline,
        }
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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

    /// Matvec dispatching to Q4 or f16 based on the weight buffer layout.
    pub fn encode_matvec_auto_view(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &Buffer,
        y_buf: &Buffer,
        m: u32,
        k: u32,
    ) {
        if weight_buf_is_q4(weight, m, k) {
            self.encode_matvec_q4_view(encoder, weight, x_buf, y_buf, m, k);
        } else {
            self.encode_matvec_f16_view(encoder, weight, x_buf, y_buf, m, k);
        }
    }

    pub fn encode_matvec_auto_at_view(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        m: u32,
        k: u32,
    ) {
        if weight_buf_is_q4(weight, m, k) {
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
        encoder: &ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &Buffer,
        y_buf: &Buffer,
        m: u32,
        k: u32,
        seq_len: u32,
    ) {
        if weight_buf_is_q4(weight, m, k) {
            self.encode_projection_q4_batch_view(encoder, weight, x_buf, y_buf, m, k, seq_len);
        } else {
            self.encode_projection_f16_batch_view(encoder, weight, x_buf, y_buf, m, k, seq_len);
        }
    }

    pub fn encode_matvec_f16_at(
        &self,
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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

    fn encode_matvec_ggml_at(
        &self,
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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

    /// Fused gate+up Q4 matvec + GeLU(gate)*up → single output (MLP decode).
    pub fn encode_matvec_q4_dual_gelu_view(
        &self,
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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

    /// Fused PLE gate Q4 matvec + GeLU(gate)*context slice.
    pub fn encode_ple_matvec_gelu_q4_view(
        &self,
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
        weight: &BufferView,
        x_buf: &Buffer,
        y_buf: &Buffer,
        m: u32,
        k: u32,
        seq_len: u32,
    ) {
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

    pub fn encode_rmsnorm(
        &self,
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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

    pub fn encode_rotary(
        &self,
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
        gate_buf: &Buffer,
        up_buf: &Buffer,
        out_buf: &Buffer,
        n: u32,
    ) {
        self.encode_gelu_mul_at(encoder, gate_buf, 0, up_buf, 0, out_buf, 0, n);
    }

    pub fn encode_gelu_mul_at(
        &self,
        encoder: &ComputeCommandEncoderRef,
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

    pub fn encode_vec_mul(
        &self,
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
        src_buf: &Buffer,
        dst_buf: &Buffer,
        n: u32,
        scale: f32,
    ) {
        self.encode_vec_scale_at(encoder, src_buf, 0, dst_buf, 0, n, scale);
    }

    pub fn encode_vec_scale_at(
        &self,
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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

    pub fn encode_vec_add(
        &self,
        encoder: &ComputeCommandEncoderRef,
        a_buf: &Buffer,
        b_buf: &Buffer,
        c_buf: &Buffer,
        n: u32,
    ) {
        self.encode_vec_add_at(encoder, a_buf, 0, b_buf, 0, c_buf, 0, n);
    }

    pub fn encode_vec_add_at(
        &self,
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
        src: &Buffer,
        dst: &Buffer,
        n: u32,
    ) {
        self.encode_copy_at(encoder, src, 0, dst, 0, n);
    }

    pub fn encode_copy_at(
        &self,
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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

    fn flash_attn_ggml_pipeline_for(&self, head_dim: u32) -> &ComputePipelineState {
        match head_dim {
            256 => &self.flash_attn_ggml_q4_h256_pipeline,
            128 => &self.flash_attn_ggml_q4_h128_pipeline,
            512 => &self.flash_attn_ggml_q4_h512_pipeline,
            _ => panic!("ggml flash attention unsupported head_dim {head_dim} (need 256, 128, or 512)"),
        }
    }

    pub fn encode_attention_ggml_q4_0(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q_buf: &Buffer,
        k_cache_buf: &Buffer,
        v_cache_buf: &Buffer,
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
        encoder: &ComputeCommandEncoderRef,
        q_buf: &Buffer,
        q_offset: u64,
        k_cache_buf: &Buffer,
        k_offset: u64,
        v_cache_buf: &Buffer,
        v_offset: u64,
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
        use crate::ggml_flash_attn::{flash_attn_args, flash_attn_dispatch, flash_attn_smem_bytes};

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
        encoder.set_buffer(4, Some(out_buf), out_offset);
        encoder.set_threadgroup_memory_length(0, flash_attn_smem_bytes(head_dim));
        let (tg_x, tg_y, tg_z, tw, nsg) = flash_attn_dispatch(num_heads);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(tg_x, tg_y, tg_z),
            metal::MTLSize::new(tw, nsg, 1),
        );
    }

    pub fn encode_attention_fused_q4_0(
        &self,
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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

    pub fn encode_kv_append_attention_q4_0(
        &self,
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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

    pub fn encode_attention_causal_q8_0(
        &self,
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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
        encoder: &ComputeCommandEncoderRef,
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

    pub fn encode_attention_causal_strided_q4_0(
        &self,
        encoder: &ComputeCommandEncoderRef,
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

    pub fn mark(&mut self, encoder: &ComputeCommandEncoderRef) {
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

    #[test]
    fn metal_context_compiles_shaders() {
        // Creating a context compiles every Metal function in llama.metal.
        let _ctx = MetalContext::new();
    }
}
