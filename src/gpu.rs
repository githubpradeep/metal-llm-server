use metal::*;
use std::path::Path;

/// Metal GPU context holding device, command queue, and compiled pipelines.
pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    pub matvec_pipeline: ComputePipelineState,
    pub matvec_f16_pipeline: ComputePipelineState,
    pub matvec_q4_pipeline: ComputePipelineState,
    pub projection_f16_batch_pipeline: ComputePipelineState,
    pub projection_q4_batch_pipeline: ComputePipelineState,
    pub projection_f16_batch_tiled_pipeline: ComputePipelineState,
    pub projection_q4_batch_tiled_pipeline: ComputePipelineState,
    pub matmul_pipeline: ComputePipelineState,
    pub rmsnorm_pipeline: ComputePipelineState,
    pub rmsnorm_add_pipeline: ComputePipelineState,
    pub rmsnorm_add_save_residual_pipeline: ComputePipelineState,
    pub rmsnorm_batch_pipeline: ComputePipelineState,
    pub rmsnorm_noweight_batch_pipeline: ComputePipelineState,
    pub silu_mul_pipeline: ComputePipelineState,
    pub silu_mul_batch_pipeline: ComputePipelineState,
    pub attention_pipeline: ComputePipelineState,
    pub attention_causal_pipeline: ComputePipelineState,
    pub rotary_pipeline: ComputePipelineState,
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
    pub attention_causal_q4_0_pipeline: ComputePipelineState,
    pub attention_causal_strided_q4_0_pipeline: ComputePipelineState,
    pub ordered_embedding_fill_pipeline: ComputePipelineState,
    pub ordered_embedding_gather_pipeline: ComputePipelineState,
    pub ordered_embedding_scatter_pipeline: ComputePipelineState,
}

impl MetalContext {
    pub fn new() -> Self {
        let device = Device::system_default().expect("No Metal GPU found");
        println!("  Metal GPU: {}", device.name());
        let queue = device.new_command_queue();

        let shader_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders/llama.metal");
        let shader_src =
            std::fs::read_to_string(&shader_path).expect("Failed to read Metal shader file");

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
        let projection_f16_batch_pipeline = get_fn("projection_f16_batch");
        let projection_q4_batch_pipeline = get_fn("projection_q4_batch");
        let projection_f16_batch_tiled_pipeline = get_fn("projection_f16_batch_tiled");
        let projection_q4_batch_tiled_pipeline = get_fn("projection_q4_batch_tiled");
        let matmul_pipeline = get_fn("matmul");
        let rmsnorm_pipeline = get_fn("rmsnorm");
        let rmsnorm_add_pipeline = get_fn("rmsnorm_add");
        let rmsnorm_add_save_residual_pipeline = get_fn("rmsnorm_add_save_residual");
        let rmsnorm_batch_pipeline = get_fn("rmsnorm_batch");
        let rmsnorm_noweight_batch_pipeline = get_fn("rmsnorm_noweight_batch");
        let silu_mul_pipeline = get_fn("silu_mul");
        let silu_mul_batch_pipeline = get_fn("silu_mul_batch");
        let attention_pipeline = get_fn("attention_single_token");
        let attention_causal_pipeline = get_fn("attention_causal");
        let rotary_pipeline = get_fn("apply_rotary");
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
        let attention_offset_f16_pipeline = get_fn("attention_single_token_offset_f16");
        let attention_causal_f16_pipeline = get_fn("attention_causal_f16");
        let attention_causal_strided_f16_pipeline = get_fn("attention_causal_strided_f16");
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
        let attention_offset_q4_0_pipeline = get_fn("attention_single_token_offset_q4_0");
        let attention_causal_q4_0_pipeline = get_fn("attention_causal_q4_0");
        let attention_causal_strided_q4_0_pipeline = get_fn("attention_causal_strided_q4_0");
        let ordered_embedding_fill_pipeline = get_fn("ordered_embedding_fill");
        let ordered_embedding_gather_pipeline = get_fn("ordered_embedding_gather_logits");
        let ordered_embedding_scatter_pipeline = get_fn("ordered_embedding_scatter_logits");

        MetalContext {
            device,
            queue,
            matvec_pipeline,
            matvec_f16_pipeline,
            matvec_q4_pipeline,
            projection_f16_batch_pipeline,
            projection_q4_batch_pipeline,
            projection_f16_batch_tiled_pipeline,
            projection_q4_batch_tiled_pipeline,
            matmul_pipeline,
            rmsnorm_pipeline,
            rmsnorm_add_pipeline,
            rmsnorm_add_save_residual_pipeline,
            rmsnorm_batch_pipeline,
            rmsnorm_noweight_batch_pipeline,
            silu_mul_pipeline,
            silu_mul_batch_pipeline,
            attention_pipeline,
            attention_causal_pipeline,
            rotary_pipeline,
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
            attention_causal_q4_0_pipeline,
            attention_causal_strided_q4_0_pipeline,
            ordered_embedding_fill_pipeline,
            ordered_embedding_gather_pipeline,
            ordered_embedding_scatter_pipeline,
        }
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
        self.encode_matvec_at(encoder, w_buf, x_buf, 0, y_buf, 0, m, k);
    }

    pub fn encode_matvec_at(
        &self,
        encoder: &ComputeCommandEncoderRef,
        w_buf: &Buffer,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        m: u32,
        k: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.matvec_pipeline);
        encoder.set_buffer(0, Some(w_buf), 0);
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
        self.encode_matvec_f16_at(encoder, w_buf, x_buf, 0, y_buf, 0, m, k);
    }

    pub fn encode_matvec_f16_at(
        &self,
        encoder: &ComputeCommandEncoderRef,
        w_buf: &Buffer,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        m: u32,
        k: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.matvec_f16_pipeline);
        encoder.set_buffer(0, Some(w_buf), 0);
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
        self.encode_matvec_q4_at(encoder, w_buf, x_buf, 0, y_buf, 0, m, k);
    }

    pub fn encode_matvec_q4_at(
        &self,
        encoder: &ComputeCommandEncoderRef,
        w_buf: &Buffer,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        m: u32,
        k: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.matvec_q4_pipeline);
        encoder.set_buffer(0, Some(w_buf), 0);
        encoder.set_buffer(1, Some(x_buf), x_offset);
        encoder.set_buffer(2, Some(y_buf), y_offset);
        encoder.set_bytes(3, 4, &m as *const u32 as *const _);
        encoder.set_bytes(4, 4, &k as *const u32 as *const _);
        // 2 SIMD groups (64 threads) per TG, each TG handles 4 rows
        let n_rows_per_tg = 4u64;
        let num_tgs = MTLSize::new((m as u64 + n_rows_per_tg - 1) / n_rows_per_tg, 1, 1);
        let tg_size = MTLSize::new(64, 1, 1); // 2 SIMD groups × 32 threads
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

    pub fn encode_rmsnorm(
        &self,
        encoder: &ComputeCommandEncoderRef,
        x_buf: &Buffer,
        weight_buf: &Buffer,
        out_buf: &Buffer,
        dim: u32,
        eps: f32,
    ) {
        self.encode_rmsnorm_at(encoder, x_buf, 0, weight_buf, out_buf, 0, dim, eps);
    }

    pub fn encode_rmsnorm_at(
        &self,
        encoder: &ComputeCommandEncoderRef,
        x_buf: &Buffer,
        x_offset: u64,
        weight_buf: &Buffer,
        out_buf: &Buffer,
        out_offset: u64,
        dim: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.rmsnorm_pipeline);
        encoder.set_buffer(0, Some(x_buf), x_offset);
        encoder.set_buffer(1, Some(weight_buf), 0);
        encoder.set_buffer(2, Some(out_buf), out_offset);
        encoder.set_bytes(3, 4, &dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &eps as *const f32 as *const _);
        let tg_size = MTLSize::new(256.min(dim as u64), 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), tg_size);
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
            encoder, x_buf, 0, weight_buf, out_buf, 0, num_heads, head_dim, eps,
        );
    }

    pub fn encode_rmsnorm_per_head_at(
        &self,
        encoder: &ComputeCommandEncoderRef,
        x_buf: &Buffer,
        x_offset: u64,
        weight_buf: &Buffer,
        out_buf: &Buffer,
        out_offset: u64,
        num_heads: u32,
        head_dim: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.rmsnorm_per_head_pipeline);
        encoder.set_buffer(0, Some(x_buf), x_offset);
        encoder.set_buffer(1, Some(weight_buf), 0);
        encoder.set_buffer(2, Some(out_buf), out_offset);
        encoder.set_bytes(3, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(4, 4, &head_dim as *const u32 as *const _);
        encoder.set_bytes(5, 4, &eps as *const f32 as *const _);
        let tg_size = MTLSize::new(256.min(head_dim as u64), 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(num_heads as u64, 1, 1), tg_size);
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
        let tg_size = MTLSize::new(64, 1, 1);
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
        let tg_size = MTLSize::new(64, 1, 1);
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
        let tg_size = MTLSize::new(64, 1, 1);
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
        encoder.set_compute_pipeline_state(&self.attention_offset_q4_0_pipeline);
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
        let tg_size = MTLSize::new(64, 1, 1);
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
        let tg_size = MTLSize::new(64, 1, 1);
        let num_tgs = num_heads * q_len;
        encoder.dispatch_thread_groups(MTLSize::new(num_tgs as u64, 1, 1), tg_size);
    }

    // ─── Gemma4 ordered-embedding sparse logits ──────────────────────────────

    pub fn encode_ordered_embedding_fill(
        &self,
        encoder: &ComputeCommandEncoderRef,
        out_buf: &Buffer,
        n: u32,
        value: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.ordered_embedding_fill_pipeline);
        encoder.set_buffer(0, Some(out_buf), 0);
        encoder.set_bytes(1, 4, &n as *const u32 as *const _);
        encoder.set_bytes(2, 4, &value as *const f32 as *const _);
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_ordered_embedding_gather_logits(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lm_head_buf: &Buffer,
        hidden_buf: &Buffer,
        indices_buf: &Buffer,
        selected_logits_buf: &Buffer,
        hidden_size: u32,
        num_selected: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.ordered_embedding_gather_pipeline);
        encoder.set_buffer(0, Some(lm_head_buf), 0);
        encoder.set_buffer(1, Some(hidden_buf), 0);
        encoder.set_buffer(2, Some(indices_buf), 0);
        encoder.set_buffer(3, Some(selected_logits_buf), 0);
        encoder.set_bytes(4, 4, &hidden_size as *const u32 as *const _);
        encoder.set_bytes(5, 4, &num_selected as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(num_selected as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub fn encode_ordered_embedding_scatter_logits(
        &self,
        encoder: &ComputeCommandEncoderRef,
        logits_buf: &Buffer,
        indices_buf: &Buffer,
        selected_logits_buf: &Buffer,
        num_selected: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.ordered_embedding_scatter_pipeline);
        encoder.set_buffer(0, Some(logits_buf), 0);
        encoder.set_buffer(1, Some(indices_buf), 0);
        encoder.set_buffer(2, Some(selected_logits_buf), 0);
        encoder.set_bytes(3, 4, &num_selected as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(num_selected as u64, 1, 1), MTLSize::new(256, 1, 1));
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
///   - Pack pairs of 4-bit values into bytes (low nibble first)
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

            // Quantize and pack pairs
            for i in 0..16 {
                let v0 = group[i * 2];
                let v1 = group[i * 2 + 1];

                let q0 = ((v0 * inv_scale).round() as i32 + 8).clamp(0, 15) as u8;
                let q1 = ((v1 * inv_scale).round() as i32 + 8).clamp(0, 15) as u8;

                output[out_offset + 2 + i] = q0 | (q1 << 4);
            }
        }
    }

    output
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
