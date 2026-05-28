use metal::*;
use std::path::Path;

/// Metal GPU context holding device, command queue, and compiled pipelines.
pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    pub matvec_pipeline: ComputePipelineState,
    pub matmul_pipeline: ComputePipelineState,
    pub rmsnorm_pipeline: ComputePipelineState,
    pub silu_mul_pipeline: ComputePipelineState,
    pub attention_pipeline: ComputePipelineState,
    pub rotary_pipeline: ComputePipelineState,
    pub vec_add_pipeline: ComputePipelineState,
}

impl MetalContext {
    pub fn new() -> Self {
        let device = Device::system_default().expect("No Metal GPU found");
        println!("  Metal GPU: {}", device.name());
        let queue = device.new_command_queue();

        // Compile shader library
        let shader_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/shaders/llama.metal");
        let shader_src = std::fs::read_to_string(&shader_path)
            .expect("Failed to read Metal shader file");

        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(&shader_src, &options)
            .expect("Failed to compile Metal shaders");

        let get_fn = |name: &str| -> ComputePipelineState {
            let func = library.get_function(name, None)
                .unwrap_or_else(|e| panic!("Failed to get function '{}': {:?}", name, e));
            device.new_compute_pipeline_state_with_function(&func)
                .unwrap_or_else(|e| panic!("Failed to create pipeline for '{}': {:?}", name, e))
        };

        let matvec_pipeline = get_fn("matvec");
        let matmul_pipeline = get_fn("matmul");
        let rmsnorm_pipeline = get_fn("rmsnorm");
        let silu_mul_pipeline = get_fn("silu_mul");
        let attention_pipeline = get_fn("attention_single_token");
        let rotary_pipeline = get_fn("apply_rotary");
        let vec_add_pipeline = get_fn("vec_add");

        MetalContext {
            device,
            queue,
            matvec_pipeline,
            matmul_pipeline,
            rmsnorm_pipeline,
            silu_mul_pipeline,
            attention_pipeline,
            rotary_pipeline,
            vec_add_pipeline,
        }
    }

    /// Create a Metal buffer from a slice (shared memory — zero copy on Apple Silicon).
    pub fn buffer_from_slice(&self, data: &[f32]) -> Buffer {
        let byte_len = (data.len() * std::mem::size_of::<f32>()) as u64;
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Create an empty buffer of given float count.
    pub fn buffer_empty(&self, count: usize) -> Buffer {
        let byte_len = (count * std::mem::size_of::<f32>()) as u64;
        self.device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared)
    }

    /// Read floats back from a buffer.
    pub fn read_buffer(buf: &Buffer, count: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
        slice.to_vec()
    }

    /// Write floats into an existing buffer.
    pub fn write_buffer(buf: &Buffer, data: &[f32]) {
        let ptr = buf.contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    /// Dispatch matvec: y = W * x, W is (m, k), x is (k,), y is (m,)
    pub fn matvec(
        &self,
        w_buf: &Buffer,
        x_buf: &Buffer,
        y_buf: &Buffer,
        m: u32,
        k: u32,
    ) {
        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.matvec_pipeline);
        encoder.set_buffer(0, Some(w_buf), 0);
        encoder.set_buffer(1, Some(x_buf), 0);
        encoder.set_buffer(2, Some(y_buf), 0);
        encoder.set_bytes(3, 4, &m as *const u32 as *const _);
        encoder.set_bytes(4, 4, &k as *const u32 as *const _);

        let threads = MTLSize::new(m as u64, 1, 1);
        let tg_size = MTLSize::new(
            self.matvec_pipeline.thread_execution_width().min(m as u64),
            1, 1,
        );
        encoder.dispatch_threads(threads, tg_size);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Dispatch matmul: C = A @ B^T, A is (m, k), B is (n, k), C is (m, n)
    pub fn matmul(
        &self,
        a_buf: &Buffer,
        b_buf: &Buffer,
        c_buf: &Buffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.matmul_pipeline);
        encoder.set_buffer(0, Some(a_buf), 0);
        encoder.set_buffer(1, Some(b_buf), 0);
        encoder.set_buffer(2, Some(c_buf), 0);
        encoder.set_bytes(3, 4, &m as *const u32 as *const _);
        encoder.set_bytes(4, 4, &n as *const u32 as *const _);
        encoder.set_bytes(5, 4, &k as *const u32 as *const _);

        let threads = MTLSize::new(n as u64, m as u64, 1);
        let tg_size = MTLSize::new(16, 16, 1);
        encoder.dispatch_threads(threads, tg_size);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Dispatch RMS norm on a single vector.
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
        encoder.set_compute_pipeline_state(&self.rmsnorm_pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(weight_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 4, &dim as *const u32 as *const _);
        encoder.set_bytes(4, 4, &eps as *const f32 as *const _);

        let tg_size = MTLSize::new(256.min(dim as u64), 1, 1);
        let threads = MTLSize::new(tg_size.width, 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), tg_size);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Dispatch fused SiLU * up.
    pub fn silu_mul(
        &self,
        gate_buf: &Buffer,
        up_buf: &Buffer,
        out_buf: &Buffer,
        n: u32,
    ) {
        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.silu_mul_pipeline);
        encoder.set_buffer(0, Some(gate_buf), 0);
        encoder.set_buffer(1, Some(up_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 4, &n as *const u32 as *const _);

        let threads = MTLSize::new(n as u64, 1, 1);
        let tg_size = MTLSize::new(256, 1, 1);
        encoder.dispatch_threads(threads, tg_size);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Dispatch single-token attention across all heads.
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

        // One threadgroup per head, 64 threads per group
        let tg_size = MTLSize::new(64, 1, 1);
        let num_tgs = MTLSize::new(num_heads as u64, 1, 1);
        encoder.dispatch_thread_groups(num_tgs, tg_size);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Dispatch rotary embedding.
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
        let half_dim = head_dim / 2;
        let total_threads = num_heads * half_dim + num_kv_heads * half_dim;

        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.rotary_pipeline);
        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(k_buf), 0);
        encoder.set_buffer(2, Some(cos_buf), 0);
        encoder.set_buffer(3, Some(sin_buf), 0);
        encoder.set_bytes(4, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(5, 4, &num_kv_heads as *const u32 as *const _);
        encoder.set_bytes(6, 4, &head_dim as *const u32 as *const _);

        let threads = MTLSize::new(total_threads as u64, 1, 1);
        let tg_size = MTLSize::new(256, 1, 1);
        encoder.dispatch_threads(threads, tg_size);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Dispatch vector addition: c = a + b.
    pub fn vec_add(
        &self,
        a_buf: &Buffer,
        b_buf: &Buffer,
        c_buf: &Buffer,
        n: u32,
    ) {
        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.vec_add_pipeline);
        encoder.set_buffer(0, Some(a_buf), 0);
        encoder.set_buffer(1, Some(b_buf), 0);
        encoder.set_buffer(2, Some(c_buf), 0);
        encoder.set_bytes(3, 4, &n as *const u32 as *const _);

        let threads = MTLSize::new(n as u64, 1, 1);
        let tg_size = MTLSize::new(256, 1, 1);
        encoder.dispatch_threads(threads, tg_size);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
}
