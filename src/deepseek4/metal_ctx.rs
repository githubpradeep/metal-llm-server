//! Separate Metal library for DeepSeek-V4 kernels (compiled on demand).

use metal::*;
use std::path::Path;
use std::sync::Arc;

pub struct Dsv4Metal {
    pub device: Device,
    pub queue: CommandQueue,
    pub hc_split_sinkhorn: ComputePipelineState,
    pub hc_weighted_sum: ComputePipelineState,
    pub hc_expand_post: ComputePipelineState,
    pub rms_norm: ComputePipelineState,
    pub matvec_iq2_xxs: ComputePipelineState,
    pub matvec_q2_k: ComputePipelineState,
    pub swiglu: ComputePipelineState,
    pub router_sqrt_softplus: ComputePipelineState,
    pub attn_swa_mqa: ComputePipelineState,
    pub attn_mixed_mqa: ComputePipelineState,
    pub rope_tail: ComputePipelineState,
    pub fp8_store: ComputePipelineState,
    pub compress_mean_pool: ComputePipelineState,
    pub indexer_scores: ComputePipelineState,
}

impl Dsv4Metal {
    pub fn new() -> Arc<Self> {
        let device = Device::system_default().expect("No Metal GPU");
        let queue = device.new_command_queue();
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders");
        let mut src = String::new();
        for name in [
            "dsv4_hc.metal",
            "dsv4_moe.metal",
            "dsv4_attn.metal",
            "dsv4_rope.metal",
            "dsv4_kv.metal",
        ] {
            let p = root.join(name);
            src.push_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| {
                panic!("Failed to read {}: {}", p.display(), e);
            }));
            src.push('\n');
        }
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(&src, &options)
            .unwrap_or_else(|e| panic!("Failed to compile dsv4 Metal shaders: {e}"));
        let get = |name: &str| -> ComputePipelineState {
            let func = library
                .get_function(name, None)
                .unwrap_or_else(|e| panic!("Missing Metal fn '{name}': {e:?}"));
            device
                .new_compute_pipeline_state_with_function(&func)
                .unwrap_or_else(|e| panic!("Pipeline '{name}': {e:?}"))
        };
        let hc_split_sinkhorn = get("dsv4_hc_split_sinkhorn");
        let hc_weighted_sum = get("dsv4_hc_weighted_sum");
        let hc_expand_post = get("dsv4_hc_expand_post");
        let rms_norm = get("dsv4_rms_norm");
        let matvec_iq2_xxs = get("dsv4_matvec_iq2_xxs");
        let matvec_q2_k = get("dsv4_matvec_q2_k");
        let swiglu = get("dsv4_swiglu");
        let router_sqrt_softplus = get("dsv4_router_sqrt_softplus");
        let attn_swa_mqa = get("dsv4_attn_swa_mqa");
        let attn_mixed_mqa = get("dsv4_attn_mixed_mqa");
        let rope_tail = get("dsv4_rope_tail");
        let fp8_store = get("dsv4_fp8_store");
        let compress_mean_pool = get("dsv4_compress_mean_pool");
        let indexer_scores = get("dsv4_indexer_scores");
        Arc::new(Self {
            device,
            queue,
            hc_split_sinkhorn,
            hc_weighted_sum,
            hc_expand_post,
            rms_norm,
            matvec_iq2_xxs,
            matvec_q2_k,
            swiglu,
            router_sqrt_softplus,
            attn_swa_mqa,
            attn_mixed_mqa,
            rope_tail,
            fp8_store,
            compress_mean_pool,
            indexer_scores,
        })
    }

    pub fn buffer_from_f32(&self, data: &[f32]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            (data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    pub fn buffer_zeros(&self, nbytes: usize) -> Buffer {
        self.device
            .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
    }

    pub fn encode_matvec_iq2(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weight: &Buffer,
        x: &Buffer,
        out: &Buffer,
        n_out: i32,
        n_in: i32,
    ) {
        encoder.set_compute_pipeline_state(&self.matvec_iq2_xxs);
        encoder.set_buffer(0, Some(weight), 0);
        encoder.set_buffer(1, Some(x), 0);
        encoder.set_buffer(2, Some(out), 0);
        encoder.set_bytes(3, 4, &n_out as *const i32 as *const _);
        encoder.set_bytes(4, 4, &n_in as *const i32 as *const _);
        let tg = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(n_out as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
    }

    pub fn encode_matvec_q2k(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weight: &Buffer,
        x: &Buffer,
        out: &Buffer,
        n_out: i32,
        n_in: i32,
    ) {
        encoder.set_compute_pipeline_state(&self.matvec_q2_k);
        encoder.set_buffer(0, Some(weight), 0);
        encoder.set_buffer(1, Some(x), 0);
        encoder.set_buffer(2, Some(out), 0);
        encoder.set_bytes(3, 4, &n_out as *const i32 as *const _);
        encoder.set_bytes(4, 4, &n_in as *const i32 as *const _);
        let tg = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(n_out as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
    }

    pub fn read_f32(buf: &Buffer, n: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, n).to_vec() }
    }
}
