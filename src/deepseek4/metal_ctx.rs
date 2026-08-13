//! Separate Metal library for DeepSeek-V4 kernels (compiled on demand).

use metal::*;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::model::DenseW;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuWKind {
    F32,
    F16,
    Q8,
    Iq2,
    Q2K,
}

/// GPU-resident weight view (shared storage with host `DenseW` bytes).
pub struct GpuWeight {
    pub buf: Buffer,
    pub kind: GpuWKind,
    pub n_out: usize,
    pub n_in: usize,
}

impl GpuWeight {
    /// Shared MTLBuffer over host `DenseW` bytes (no second copy when 16-byte
    /// aligned). Caller must keep the underlying `DenseW` alive for the buffer
    /// lifetime (LayerWeights / model output weights).
    pub fn from_dense(metal: &Dsv4Metal, w: &DenseW) -> Self {
        let dims = w.dims();
        let n_in = dims[0];
        let n_out = if dims.len() > 1 { dims[1] } else { 1 };
        match w {
            DenseW::F32 { data, .. } => Self {
                buf: metal.buffer_from_slice_no_copy(unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const u8,
                        data.len() * 4,
                    )
                }),
                kind: GpuWKind::F32,
                n_out,
                n_in,
            },
            DenseW::F16 { data, .. } => Self {
                buf: metal.buffer_from_slice_no_copy(unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const u8,
                        data.len() * 2,
                    )
                }),
                kind: GpuWKind::F16,
                n_out,
                n_in,
            },
            DenseW::Q8 { data, .. } => Self {
                buf: metal.buffer_from_slice_no_copy(data),
                kind: GpuWKind::Q8,
                n_out,
                n_in,
            },
        }
    }

    pub fn from_bytes(
        metal: &Dsv4Metal,
        data: &[u8],
        kind: GpuWKind,
        n_out: usize,
        n_in: usize,
    ) -> Self {
        Self {
            buf: metal.buffer_from_slice_no_copy(data),
            kind,
            n_out,
            n_in,
        }
    }
}

pub struct Dsv4Metal {
    pub device: Device,
    pub queue: CommandQueue,
    pub hc_split_sinkhorn: ComputePipelineState,
    pub hc_weighted_sum: ComputePipelineState,
    pub hc_expand_post: ComputePipelineState,
    pub rms_norm: ComputePipelineState,
    pub matvec_iq2_xxs: ComputePipelineState,
    pub matvec_iq2_xxs_pair_swiglu: ComputePipelineState,
    pub slots6_iq2_pair_swiglu: ComputePipelineState,
    pub slots6_q2k_sum6: ComputePipelineState,
    pub slots6_iq2_pair_swiglu_packed: ComputePipelineState,
    pub slots6_q2k_sum6_packed: ComputePipelineState,
    pub map_route_slots: ComputePipelineState,
    pub matvec_q2_k: ComputePipelineState,
    pub matvec_f16: ComputePipelineState,
    pub matvec_f16_lora_groups: ComputePipelineState,
    pub matvec_f32: ComputePipelineState,
    pub matvec_q8_0: ComputePipelineState,
    pub matvec_q8_0_lora_groups: ComputePipelineState,
    pub matvec_q8_0_pair_swiglu: ComputePipelineState,
    pub swiglu: ComputePipelineState,
    pub router_sqrt_softplus: ComputePipelineState,
    pub router_topk: ComputePipelineState,
    pub router_hash_select: ComputePipelineState,
    pub axpy_w: ComputePipelineState,
    pub attn_swa_mqa: ComputePipelineState,
    pub attn_mixed_mqa: ComputePipelineState,
    pub rope_tail: ComputePipelineState,
    pub rope_tail_ext: ComputePipelineState,
    pub rms_norm_rows: ComputePipelineState,
    pub fp8_e4m3fn_nope: ComputePipelineState,
    pub fp8_store: ComputePipelineState,
    pub kv_store_row: ComputePipelineState,
    pub kv_shift_left: ComputePipelineState,
    pub compress_mean_pool: ComputePipelineState,
    pub indexer_scores: ComputePipelineState,
    pub select_comp_score: ComputePipelineState,
    pub select_comp_topk: ComputePipelineState,
    pub compressor_store_row: ComputePipelineState,
    pub compressor_pool: ComputePipelineState,
    pub compressor_rms_rope_fp8: ComputePipelineState,
    pub compressor_csa_shuffle: ComputePipelineState,
    pub iota_i32: ComputePipelineState,
    pub axpy: ComputePipelineState,
    pub add: ComputePipelineState,
    pub zero: ComputePipelineState,
    pub copy: ComputePipelineState,
    /// CPU→GPU: miss-expert kernels wait until preads have filled Shared slots.
    pread_event: SharedEvent,
    pread_epoch: AtomicU64,
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
            "dsv4_dense.metal",
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
        options.set_fast_math_enabled(true);
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
        let matvec_iq2_xxs_pair_swiglu = get("dsv4_matvec_iq2_xxs_pair_swiglu");
        let slots6_iq2_pair_swiglu = get("dsv4_slots6_iq2_pair_swiglu");
        let slots6_q2k_sum6 = get("dsv4_slots6_q2k_sum6");
        let slots6_iq2_pair_swiglu_packed = get("dsv4_slots6_iq2_pair_swiglu_packed");
        let slots6_q2k_sum6_packed = get("dsv4_slots6_q2k_sum6_packed");
        let map_route_slots = get("dsv4_map_route_slots");
        let matvec_q2_k = get("dsv4_matvec_q2_k");
        let matvec_f16 = get("dsv4_matvec_f16");
        let matvec_f16_lora_groups = get("dsv4_matvec_f16_lora_groups");
        let matvec_f32 = get("dsv4_matvec_f32");
        let matvec_q8_0 = get("dsv4_matvec_q8_0");
        let matvec_q8_0_lora_groups = get("dsv4_matvec_q8_0_lora_groups");
        let matvec_q8_0_pair_swiglu = get("dsv4_matvec_q8_0_pair_swiglu");
        let swiglu = get("dsv4_swiglu");
        let router_sqrt_softplus = get("dsv4_router_sqrt_softplus");
        let router_topk = get("dsv4_router_topk");
        let router_hash_select = get("dsv4_router_hash_select");
        let axpy_w = get("dsv4_axpy_w");
        let attn_swa_mqa = get("dsv4_attn_swa_mqa");
        let attn_mixed_mqa = get("dsv4_attn_mixed_mqa");
        let rope_tail = get("dsv4_rope_tail");
        let rope_tail_ext = get("dsv4_rope_tail_ext");
        let rms_norm_rows = get("dsv4_rms_norm_rows");
        let fp8_e4m3fn_nope = get("dsv4_fp8_e4m3fn_nope");
        let fp8_store = get("dsv4_fp8_store");
        let kv_store_row = get("dsv4_kv_store_row");
        let kv_shift_left = get("dsv4_kv_shift_left");
        let compress_mean_pool = get("dsv4_compress_mean_pool");
        let indexer_scores = get("dsv4_indexer_scores");
        let select_comp_score = get("dsv4_select_comp_score");
        let select_comp_topk = get("dsv4_select_comp_topk");
        let compressor_store_row = get("dsv4_compressor_store_row");
        let compressor_pool = get("dsv4_compressor_pool");
        let compressor_rms_rope_fp8 = get("dsv4_compressor_rms_rope_fp8");
        let compressor_csa_shuffle = get("dsv4_compressor_csa_shuffle");
        let iota_i32 = get("dsv4_iota_i32");
        let axpy = get("dsv4_axpy");
        let add = get("dsv4_add");
        let zero = get("dsv4_zero");
        let copy = get("dsv4_copy");
        let pread_event = device.new_shared_event();
        Arc::new(Self {
            device,
            queue,
            hc_split_sinkhorn,
            hc_weighted_sum,
            hc_expand_post,
            rms_norm,
            matvec_iq2_xxs,
            matvec_iq2_xxs_pair_swiglu,
            slots6_iq2_pair_swiglu,
            slots6_q2k_sum6,
            slots6_iq2_pair_swiglu_packed,
            slots6_q2k_sum6_packed,
            map_route_slots,
            matvec_q2_k,
            matvec_f16,
            matvec_f16_lora_groups,
            matvec_f32,
            matvec_q8_0,
            matvec_q8_0_lora_groups,
            matvec_q8_0_pair_swiglu,
            swiglu,
            router_sqrt_softplus,
            router_topk,
            router_hash_select,
            axpy_w,
            attn_swa_mqa,
            attn_mixed_mqa,
            rope_tail,
            rope_tail_ext,
            rms_norm_rows,
            fp8_e4m3fn_nope,
            fp8_store,
            kv_store_row,
            kv_shift_left,
            compress_mean_pool,
            indexer_scores,
            select_comp_score,
            select_comp_topk,
            compressor_store_row,
            compressor_pool,
            compressor_rms_rope_fp8,
            compressor_csa_shuffle,
            iota_i32,
            axpy,
            add,
            zero,
            copy,
            pread_event,
            pread_epoch: AtomicU64::new(0),
        })
    }

    pub fn buffer_from_bytes(&self, data: &[u8]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Zero-copy shared buffer over `data` when 16-byte aligned; else copies.
    pub fn buffer_from_slice_no_copy(&self, data: &[u8]) -> Buffer {
        if !data.is_empty() && (data.as_ptr() as usize) % 16 == 0 {
            self.device.new_buffer_with_bytes_no_copy(
                data.as_ptr() as *const _,
                data.len() as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        } else {
            self.buffer_from_bytes(data)
        }
    }

    pub fn buffer_from_f32(&self, data: &[f32]) -> Buffer {
        self.buffer_from_bytes(unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        })
    }

    pub fn buffer_zeros(&self, nbytes: usize) -> Buffer {
        self.device
            .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
    }

    pub fn write_f32(buf: &Buffer, data: &[f32]) {
        let dst = buf.contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
    }

    pub fn write_i32(buf: &Buffer, data: &[i32]) {
        let dst = buf.contents() as *mut i32;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
    }

    pub fn write_bytes(buf: &Buffer, data: &[u8]) {
        let dst = buf.contents() as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
    }

    pub fn read_f32(buf: &Buffer, n: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, n).to_vec() }
    }

    pub fn encode_matvec_kind(
        &self,
        encoder: &ComputeCommandEncoderRef,
        kind: GpuWKind,
        weight: &Buffer,
        x: &Buffer,
        out: &Buffer,
        n_out: i32,
        n_in: i32,
        weight_byte_off: u64,
    ) {
        self.encode_matvec_kind_off(
            encoder,
            kind,
            weight,
            x,
            out,
            n_out,
            n_in,
            weight_byte_off,
            0,
            0,
        );
    }

    pub fn encode_matvec_kind_off(
        &self,
        encoder: &ComputeCommandEncoderRef,
        kind: GpuWKind,
        weight: &Buffer,
        x: &Buffer,
        out: &Buffer,
        n_out: i32,
        n_in: i32,
        weight_byte_off: u64,
        x_byte_off: u64,
        out_byte_off: u64,
    ) {
        let pipe = match kind {
            GpuWKind::F16 => &self.matvec_f16,
            GpuWKind::F32 => &self.matvec_f32,
            GpuWKind::Q8 => &self.matvec_q8_0,
            GpuWKind::Iq2 => &self.matvec_iq2_xxs,
            GpuWKind::Q2K => &self.matvec_q2_k,
        };
        encoder.set_compute_pipeline_state(pipe);
        encoder.set_buffer(0, Some(weight), weight_byte_off);
        encoder.set_buffer(1, Some(x), x_byte_off);
        encoder.set_buffer(2, Some(out), out_byte_off);
        encoder.set_bytes(3, 4, &n_out as *const i32 as *const _);
        encoder.set_bytes(4, 4, &n_in as *const i32 as *const _);
        match kind {
            // IQ2 / Q2_K: each simdgroup owns distinct rows (NR0=4, NSG=2).
            GpuWKind::Iq2 | GpuWKind::Q2K => {
                const NR0: u64 = 4;
                const NSG: u64 = 2;
                let rows_per_tg = NR0 * NSG;
                let n_tg = ((n_out as u64) + rows_per_tg - 1) / rows_per_tg;
                if matches!(kind, GpuWKind::Iq2) {
                    encoder.set_threadgroup_memory_length(0, 256 * 8 + 128);
                }
                encoder.dispatch_thread_groups(
                    MTLSize::new(n_tg, 1, 1),
                    MTLSize::new(32 * NSG, 1, 1),
                );
            }
            // Dense F16/F32: ds4 plain mv — NR0=2, NSG=min(8, ceil(K/128)).
            GpuWKind::F16 | GpuWKind::F32 => {
                const NR0: u64 = 2;
                let nsg = ((n_in as u64 + 127) / 128).clamp(1, 8);
                let nsg_i = nsg as i32;
                encoder.set_bytes(5, 4, &nsg_i as *const i32 as *const _);
                encoder.set_threadgroup_memory_length(0, 32 * NR0 * 4);
                let n_tg = ((n_out as u64) + NR0 - 1) / NR0;
                encoder.dispatch_thread_groups(
                    MTLSize::new(n_tg, 1, 1),
                    MTLSize::new(32 * nsg, 1, 1),
                );
            }
            // Dense Q8: ds4 default NR0=2, NSG=4.
            GpuWKind::Q8 => {
                const NR0: u64 = 2;
                const NSG: u64 = 4;
                let nsg_i = NSG as i32;
                encoder.set_bytes(5, 4, &nsg_i as *const i32 as *const _);
                encoder.set_threadgroup_memory_length(0, 32 * NR0 * 4);
                let n_tg = ((n_out as u64) + NR0 - 1) / NR0;
                encoder.dispatch_thread_groups(
                    MTLSize::new(n_tg, 1, 1),
                    MTLSize::new(32 * NSG, 1, 1),
                );
            }
        }
    }

    /// Fused IQ2_XXS gate∥up + SiLU → mid (one dispatch, shared x + TG LUTs).
    pub fn encode_iq2_pair_swiglu(
        &self,
        encoder: &ComputeCommandEncoderRef,
        gate: &Buffer,
        up: &Buffer,
        x: &Buffer,
        mid: &Buffer,
        n_out: i32,
        n_in: i32,
        clampv: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.matvec_iq2_xxs_pair_swiglu);
        encoder.set_buffer(0, Some(gate), 0);
        encoder.set_buffer(1, Some(up), 0);
        encoder.set_buffer(2, Some(x), 0);
        encoder.set_buffer(3, Some(mid), 0);
        encoder.set_bytes(4, 4, &n_out as *const i32 as *const _);
        encoder.set_bytes(5, 4, &n_in as *const i32 as *const _);
        encoder.set_bytes(6, 4, &clampv as *const f32 as *const _);
        encoder.set_threadgroup_memory_length(0, 256 * 8 + 128);
        const NR0: u64 = 4;
        const NSG: u64 = 2;
        let rows_per_tg = NR0 * NSG;
        let n_tg = ((n_out as u64) + rows_per_tg - 1) / rows_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(n_tg, 1, 1),
            MTLSize::new(32 * NSG, 1, 1),
        );
    }

    /// Fused Q8_0 gate∥up + SiLU for shared expert (K-partitioned dense).
    pub fn encode_q8_pair_swiglu(
        &self,
        encoder: &ComputeCommandEncoderRef,
        gate: &Buffer,
        up: &Buffer,
        x: &Buffer,
        mid: &Buffer,
        n_out: i32,
        n_in: i32,
        clampv: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.matvec_q8_0_pair_swiglu);
        encoder.set_buffer(0, Some(gate), 0);
        encoder.set_buffer(1, Some(up), 0);
        encoder.set_buffer(2, Some(x), 0);
        encoder.set_buffer(3, Some(mid), 0);
        encoder.set_bytes(4, 4, &n_out as *const i32 as *const _);
        encoder.set_bytes(5, 4, &n_in as *const i32 as *const _);
        encoder.set_bytes(6, 4, &clampv as *const f32 as *const _);
        const NR0: u64 = 2;
        const NSG: u64 = 4;
        let nsg_i = NSG as i32;
        encoder.set_bytes(7, 4, &nsg_i as *const i32 as *const _);
        encoder.set_threadgroup_memory_length(0, 2 * 32 * NR0 * 4);
        let n_tg = ((n_out as u64) + NR0 - 1) / NR0;
        encoder.dispatch_thread_groups(
            MTLSize::new(n_tg, 1, 1),
            MTLSize::new(32 * NSG, 1, 1),
        );
    }

    /// Flash top-6: one IQ2 gate∥up+SiLU×weight dispatch (tgpig.z = expert).
    pub fn encode_slots6_iq2_pair_swiglu(
        &self,
        encoder: &ComputeCommandEncoderRef,
        gates: [&Buffer; 6],
        ups: [&Buffer; 6],
        x: &Buffer,
        mid: &Buffer,
        weights: &Buffer,
        n_out: i32,
        n_in: i32,
        clampv: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.slots6_iq2_pair_swiglu);
        for i in 0..6 {
            encoder.set_buffer(i as u64, Some(gates[i]), 0);
        }
        for i in 0..6 {
            encoder.set_buffer(6 + i as u64, Some(ups[i]), 0);
        }
        encoder.set_buffer(12, Some(x), 0);
        encoder.set_buffer(13, Some(mid), 0);
        encoder.set_buffer(14, Some(weights), 0);
        encoder.set_bytes(15, 4, &n_out as *const i32 as *const _);
        encoder.set_bytes(16, 4, &n_in as *const i32 as *const _);
        encoder.set_bytes(17, 4, &clampv as *const f32 as *const _);
        encoder.set_threadgroup_memory_length(0, 256 * 8 + 128);
        const NR0: u64 = 4;
        const NSG: u64 = 2;
        let rows_per_tg = NR0 * NSG;
        let n_tg_x = ((n_out as u64) + rows_per_tg - 1) / rows_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(n_tg_x, 1, 6),
            MTLSize::new(32 * NSG, 1, 1),
        );
    }

    /// Flash top-6: one Q2_K down that sums all experts into `out`.
    pub fn encode_slots6_q2k_sum6(
        &self,
        encoder: &ComputeCommandEncoderRef,
        downs: [&Buffer; 6],
        mid: &Buffer,
        out: &Buffer,
        n_out: i32,
        n_in: i32,
    ) {
        encoder.set_compute_pipeline_state(&self.slots6_q2k_sum6);
        for i in 0..6 {
            encoder.set_buffer(i as u64, Some(downs[i]), 0);
        }
        encoder.set_buffer(6, Some(mid), 0);
        encoder.set_buffer(7, Some(out), 0);
        encoder.set_bytes(8, 4, &n_out as *const i32 as *const _);
        encoder.set_bytes(9, 4, &n_in as *const i32 as *const _);
        const NR0: u64 = 4;
        const NSG: u64 = 2;
        let rows_per_tg = NR0 * NSG;
        let n_tg = ((n_out as u64) + rows_per_tg - 1) / rows_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(n_tg, 1, 1),
            MTLSize::new(32 * NSG, 1, 1),
        );
    }

    pub fn encode_map_route_slots(
        &self,
        encoder: &ComputeCommandEncoderRef,
        route_ids: &Buffer,
        slot_map: &Buffer,
        out_slots: &Buffer,
        miss_flag: &Buffer,
        hist: &Buffer,
        k: i32,
        hist_off: i32,
    ) {
        encoder.set_compute_pipeline_state(&self.map_route_slots);
        encoder.set_buffer(0, Some(route_ids), 0);
        encoder.set_buffer(1, Some(slot_map), 0);
        encoder.set_buffer(2, Some(out_slots), 0);
        encoder.set_buffer(3, Some(miss_flag), 0);
        encoder.set_buffer(4, Some(hist), 0);
        encoder.set_bytes(5, 4, &k as *const i32 as *const _);
        encoder.set_bytes(6, 4, &hist_off as *const i32 as *const _);
        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
    }

    pub fn encode_slots6_iq2_pair_swiglu_packed(
        &self,
        encoder: &ComputeCommandEncoderRef,
        gate_pack: &Buffer,
        up_pack: &Buffer,
        slots: &Buffer,
        x: &Buffer,
        mid: &Buffer,
        weights: &Buffer,
        n_out: i32,
        n_in: i32,
        clampv: f32,
        gate_stride: u32,
        up_stride: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.slots6_iq2_pair_swiglu_packed);
        encoder.set_buffer(0, Some(gate_pack), 0);
        encoder.set_buffer(1, Some(up_pack), 0);
        encoder.set_buffer(2, Some(slots), 0);
        encoder.set_buffer(3, Some(x), 0);
        encoder.set_buffer(4, Some(mid), 0);
        encoder.set_buffer(5, Some(weights), 0);
        encoder.set_bytes(6, 4, &n_out as *const i32 as *const _);
        encoder.set_bytes(7, 4, &n_in as *const i32 as *const _);
        encoder.set_bytes(8, 4, &clampv as *const f32 as *const _);
        encoder.set_bytes(9, 4, &gate_stride as *const u32 as *const _);
        encoder.set_bytes(10, 4, &up_stride as *const u32 as *const _);
        encoder.set_threadgroup_memory_length(0, 256 * 8 + 128);
        const NR0: u64 = 4;
        const NSG: u64 = 2;
        let rows_per_tg = NR0 * NSG;
        let n_tg_x = ((n_out as u64) + rows_per_tg - 1) / rows_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(n_tg_x, 1, 6),
            MTLSize::new(32 * NSG, 1, 1),
        );
    }

    pub fn encode_slots6_q2k_sum6_packed(
        &self,
        encoder: &ComputeCommandEncoderRef,
        down_pack: &Buffer,
        slots: &Buffer,
        mid: &Buffer,
        out: &Buffer,
        n_out: i32,
        n_in: i32,
        down_stride: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.slots6_q2k_sum6_packed);
        encoder.set_buffer(0, Some(down_pack), 0);
        encoder.set_buffer(1, Some(slots), 0);
        encoder.set_buffer(2, Some(mid), 0);
        encoder.set_buffer(3, Some(out), 0);
        encoder.set_bytes(4, 4, &n_out as *const i32 as *const _);
        encoder.set_bytes(5, 4, &n_in as *const i32 as *const _);
        encoder.set_bytes(6, 4, &down_stride as *const u32 as *const _);
        const NR0: u64 = 4;
        const NSG: u64 = 2;
        let rows_per_tg = NR0 * NSG;
        let n_tg = ((n_out as u64) + rows_per_tg - 1) / rows_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(n_tg, 1, 1),
            MTLSize::new(32 * NSG, 1, 1),
        );
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
        self.encode_matvec_kind(
            encoder,
            GpuWKind::Iq2,
            weight,
            x,
            out,
            n_out,
            n_in,
            0,
        );
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
        self.encode_matvec_kind(
            encoder,
            GpuWKind::Q2K,
            weight,
            x,
            out,
            n_out,
            n_in,
            0,
        );
    }

    pub fn encode_swiglu(
        &self,
        encoder: &ComputeCommandEncoderRef,
        gate: &Buffer,
        up: &Buffer,
        out: &Buffer,
        n: i32,
        clampv: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.swiglu);
        encoder.set_buffer(0, Some(gate), 0);
        encoder.set_buffer(1, Some(up), 0);
        encoder.set_buffer(2, Some(out), 0);
        encoder.set_bytes(3, 4, &n as *const i32 as *const _);
        encoder.set_bytes(4, 4, &clampv as *const f32 as *const _);
        let tg = MTLSize::new(64, 1, 1);
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), tg);
    }

    pub fn encode_router_sqrt_softplus(
        &self,
        encoder: &ComputeCommandEncoderRef,
        logits: &Buffer,
        probs: &Buffer,
        n: i32,
    ) {
        encoder.set_compute_pipeline_state(&self.router_sqrt_softplus);
        encoder.set_buffer(0, Some(logits), 0);
        encoder.set_buffer(1, Some(probs), 0);
        encoder.set_bytes(2, 4, &n as *const i32 as *const _);
        let tg = MTLSize::new(64, 1, 1);
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), tg);
    }

    pub fn encode_router_topk(
        &self,
        encoder: &ComputeCommandEncoderRef,
        probs: &Buffer,
        bias: Option<&Buffer>,
        out_ids: &Buffer,
        out_w: &Buffer,
        n: i32,
        k: i32,
        weight_scale: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.router_topk);
        encoder.set_buffer(0, Some(probs), 0);
        let b = bias.unwrap_or(probs);
        encoder.set_buffer(1, Some(b), 0);
        encoder.set_buffer(2, Some(out_ids), 0);
        encoder.set_buffer(3, Some(out_w), 0);
        encoder.set_bytes(4, 4, &n as *const i32 as *const _);
        encoder.set_bytes(5, 4, &k as *const i32 as *const _);
        encoder.set_bytes(6, 4, &weight_scale as *const f32 as *const _);
        let has_bias: i32 = if bias.is_some() { 1 } else { 0 };
        encoder.set_bytes(7, 4, &has_bias as *const i32 as *const _);
        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
    }

    pub fn encode_router_hash_select(
        &self,
        encoder: &ComputeCommandEncoderRef,
        probs: &Buffer,
        eids: &Buffer,
        eids_off: u64,
        out_ids: &Buffer,
        out_w: &Buffer,
        k: i32,
        weight_scale: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.router_hash_select);
        encoder.set_buffer(0, Some(probs), 0);
        encoder.set_buffer(1, Some(eids), eids_off);
        encoder.set_buffer(2, Some(out_ids), 0);
        encoder.set_buffer(3, Some(out_w), 0);
        encoder.set_bytes(4, 4, &k as *const i32 as *const _);
        encoder.set_bytes(5, 4, &weight_scale as *const f32 as *const _);
        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
    }

    pub fn encode_axpy_w(
        &self,
        encoder: &ComputeCommandEncoderRef,
        x: &Buffer,
        y: &Buffer,
        scales: &Buffer,
        scale_idx: i32,
        n: i32,
    ) {
        encoder.set_compute_pipeline_state(&self.axpy_w);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(y), 0);
        encoder.set_buffer(2, Some(scales), 0);
        encoder.set_bytes(3, 4, &scale_idx as *const i32 as *const _);
        encoder.set_bytes(4, 4, &n as *const i32 as *const _);
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(64, 1, 1));
    }

    pub fn read_i32(buf: &Buffer, n: usize) -> Vec<i32> {
        let ptr = buf.contents() as *const i32;
        unsafe { std::slice::from_raw_parts(ptr, n).to_vec() }
    }

    pub fn encode_rms_norm(
        &self,
        encoder: &ComputeCommandEncoderRef,
        x: &Buffer,
        weight: Option<&Buffer>,
        out: &Buffer,
        n: i32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.rms_norm);
        encoder.set_buffer(0, Some(x), 0);
        let wbuf = weight.unwrap_or(x);
        encoder.set_buffer(1, Some(wbuf), 0);
        encoder.set_buffer(2, Some(out), 0);
        encoder.set_bytes(3, 4, &n as *const i32 as *const _);
        encoder.set_bytes(4, 4, &eps as *const f32 as *const _);
        let has_w: i32 = if weight.is_some() { 1 } else { 0 };
        encoder.set_bytes(5, 4, &has_w as *const i32 as *const _);
        let tg = MTLSize::new(256, 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), tg);
    }

    pub fn encode_axpy(
        &self,
        encoder: &ComputeCommandEncoderRef,
        x: &Buffer,
        y: &Buffer,
        scale: f32,
        n: i32,
    ) {
        encoder.set_compute_pipeline_state(&self.axpy);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(y), 0);
        encoder.set_bytes(2, 4, &scale as *const f32 as *const _);
        encoder.set_bytes(3, 4, &n as *const i32 as *const _);
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(64, 1, 1));
    }

    pub fn encode_add(
        &self,
        encoder: &ComputeCommandEncoderRef,
        a: &Buffer,
        b: &Buffer,
        y: &Buffer,
        n: i32,
    ) {
        encoder.set_compute_pipeline_state(&self.add);
        encoder.set_buffer(0, Some(a), 0);
        encoder.set_buffer(1, Some(b), 0);
        encoder.set_buffer(2, Some(y), 0);
        encoder.set_bytes(3, 4, &n as *const i32 as *const _);
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(64, 1, 1));
    }

    pub fn encode_zero(&self, encoder: &ComputeCommandEncoderRef, y: &Buffer, n: i32) {
        encoder.set_compute_pipeline_state(&self.zero);
        encoder.set_buffer(0, Some(y), 0);
        encoder.set_bytes(1, 4, &n as *const i32 as *const _);
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(64, 1, 1));
    }

    pub fn encode_copy(
        &self,
        encoder: &ComputeCommandEncoderRef,
        x: &Buffer,
        y: &Buffer,
        n: i32,
    ) {
        encoder.set_compute_pipeline_state(&self.copy);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(y), 0);
        encoder.set_bytes(2, 4, &n as *const i32 as *const _);
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(64, 1, 1));
    }

    pub fn encode_hc_split_sinkhorn(
        &self,
        encoder: &ComputeCommandEncoderRef,
        mix: &Buffer,
        scale: &Buffer,
        base: &Buffer,
        out: &Buffer,
        n_hc: i32,
        iters: i32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.hc_split_sinkhorn);
        encoder.set_buffer(0, Some(mix), 0);
        encoder.set_buffer(1, Some(scale), 0);
        encoder.set_buffer(2, Some(base), 0);
        encoder.set_buffer(3, Some(out), 0);
        encoder.set_bytes(4, 4, &n_hc as *const i32 as *const _);
        encoder.set_bytes(5, 4, &iters as *const i32 as *const _);
        encoder.set_bytes(6, 4, &eps as *const f32 as *const _);
        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
    }

    pub fn encode_hc_weighted_sum(
        &self,
        encoder: &ComputeCommandEncoderRef,
        x_hc: &Buffer,
        weights: &Buffer,
        out: &Buffer,
        n_embd: i32,
        n_hc: i32,
    ) {
        encoder.set_compute_pipeline_state(&self.hc_weighted_sum);
        encoder.set_buffer(0, Some(x_hc), 0);
        encoder.set_buffer(1, Some(weights), 0);
        encoder.set_buffer(2, Some(out), 0);
        encoder.set_bytes(3, 4, &n_embd as *const i32 as *const _);
        encoder.set_bytes(4, 4, &n_hc as *const i32 as *const _);
        encoder.dispatch_threads(
            MTLSize::new(n_embd as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
    }

    pub fn encode_hc_expand_post(
        &self,
        encoder: &ComputeCommandEncoderRef,
        block: &Buffer,
        add_hc: &Buffer,
        post: &Buffer,
        comb: &Buffer,
        hc: &Buffer,
        n_embd: i32,
        n_hc: i32,
    ) {
        self.encode_hc_expand_post_off(encoder, block, 0, add_hc, 0, post, 0, comb, 0, hc, 0, n_embd, n_hc);
    }

    /// `post`/`comb` may live inside the Sinkhorn `split` buffer (`[pre|post|comb]`).
    pub fn encode_hc_expand_post_off(
        &self,
        encoder: &ComputeCommandEncoderRef,
        block: &Buffer,
        block_off: u64,
        add_hc: &Buffer,
        add_off: u64,
        post: &Buffer,
        post_off: u64,
        comb: &Buffer,
        comb_off: u64,
        hc: &Buffer,
        hc_off: u64,
        n_embd: i32,
        n_hc: i32,
    ) {
        encoder.set_compute_pipeline_state(&self.hc_expand_post);
        encoder.set_buffer(0, Some(block), block_off);
        encoder.set_buffer(1, Some(add_hc), add_off);
        encoder.set_buffer(2, Some(post), post_off);
        encoder.set_buffer(3, Some(comb), comb_off);
        encoder.set_buffer(4, Some(hc), hc_off);
        encoder.set_bytes(5, 4, &n_embd as *const i32 as *const _);
        encoder.set_bytes(6, 4, &n_hc as *const i32 as *const _);
        encoder.dispatch_threads(
            MTLSize::new(n_embd as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
    }

    /// Grouped LoRA-O A (F16): one dispatch, tgpig.y = group.
    pub fn encode_matvec_f16_lora_groups(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weight: &Buffer,
        x: &Buffer,
        out: &Buffer,
        n_groups: i32,
        rank: i32,
        group_dim: i32,
    ) {
        encoder.set_compute_pipeline_state(&self.matvec_f16_lora_groups);
        encoder.set_buffer(0, Some(weight), 0);
        encoder.set_buffer(1, Some(x), 0);
        encoder.set_buffer(2, Some(out), 0);
        encoder.set_bytes(3, 4, &n_groups as *const i32 as *const _);
        encoder.set_bytes(4, 4, &rank as *const i32 as *const _);
        encoder.set_bytes(5, 4, &group_dim as *const i32 as *const _);
        const NR0: u64 = 2;
        let nsg = ((group_dim as u64 + 127) / 128).clamp(1, 8);
        let nsg_i = nsg as i32;
        encoder.set_bytes(6, 4, &nsg_i as *const i32 as *const _);
        encoder.set_threadgroup_memory_length(0, 32 * NR0 * 4);
        let n_tg_x = ((rank as u64) + NR0 - 1) / NR0;
        encoder.dispatch_thread_groups(
            MTLSize::new(n_tg_x, n_groups as u64, 1),
            MTLSize::new(32 * nsg, 1, 1),
        );
    }

    /// Grouped LoRA-O A (Q8_0): one dispatch, tgpig.y = group (Flash default).
    pub fn encode_matvec_q8_lora_groups(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weight: &Buffer,
        x: &Buffer,
        out: &Buffer,
        n_groups: i32,
        rank: i32,
        group_dim: i32,
    ) {
        encoder.set_compute_pipeline_state(&self.matvec_q8_0_lora_groups);
        encoder.set_buffer(0, Some(weight), 0);
        encoder.set_buffer(1, Some(x), 0);
        encoder.set_buffer(2, Some(out), 0);
        encoder.set_bytes(3, 4, &n_groups as *const i32 as *const _);
        encoder.set_bytes(4, 4, &rank as *const i32 as *const _);
        encoder.set_bytes(5, 4, &group_dim as *const i32 as *const _);
        const NR0: u64 = 2;
        const NSG: u64 = 4;
        let nsg_i = NSG as i32;
        encoder.set_bytes(6, 4, &nsg_i as *const i32 as *const _);
        encoder.set_threadgroup_memory_length(0, 32 * NR0 * 4);
        let n_tg_x = ((rank as u64) + NR0 - 1) / NR0;
        encoder.dispatch_thread_groups(
            MTLSize::new(n_tg_x, n_groups as u64, 1),
            MTLSize::new(32 * NSG, 1, 1),
        );
    }

    pub fn encode_attn_swa(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        sinks: &Buffer,
        out: &Buffer,
        n_head: i32,
        head_dim: i32,
        n_kv: i32,
        has_sinks: i32,
        scale: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.attn_swa_mqa);
        encoder.set_buffer(0, Some(q), 0);
        encoder.set_buffer(1, Some(k), 0);
        encoder.set_buffer(2, Some(v), 0);
        encoder.set_buffer(3, Some(sinks), 0);
        encoder.set_buffer(4, Some(out), 0);
        encoder.set_bytes(5, 4, &n_head as *const i32 as *const _);
        encoder.set_bytes(6, 4, &head_dim as *const i32 as *const _);
        encoder.set_bytes(7, 4, &n_kv as *const i32 as *const _);
        encoder.set_bytes(8, 4, &has_sinks as *const i32 as *const _);
        encoder.set_bytes(9, 4, &scale as *const f32 as *const _);
        // 1 simdgroup (32 threads) per head — two-pass softmax in dsv4_attn_swa_mqa.
        // scores[128] + inv slot.
        encoder.set_threadgroup_memory_length(0, 129 * 4);
        encoder.dispatch_thread_groups(
            MTLSize::new(n_head as u64, 1, 1),
            MTLSize::new(32, 1, 1),
        );
    }

    pub fn encode_rms_norm_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        x: &Buffer,
        weight: Option<&Buffer>,
        out: &Buffer,
        n_rows: i32,
        row_dim: i32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.rms_norm_rows);
        encoder.set_buffer(0, Some(x), 0);
        let wbuf = weight.unwrap_or(x);
        encoder.set_buffer(1, Some(wbuf), 0);
        encoder.set_buffer(2, Some(out), 0);
        encoder.set_bytes(3, 4, &n_rows as *const i32 as *const _);
        encoder.set_bytes(4, 4, &row_dim as *const i32 as *const _);
        encoder.set_bytes(5, 4, &eps as *const f32 as *const _);
        let has_w: i32 = if weight.is_some() { 1 } else { 0 };
        encoder.set_bytes(6, 4, &has_w as *const i32 as *const _);
        encoder.dispatch_threads(
            MTLSize::new(n_rows as u64, 1, 1),
            MTLSize::new(1, 1, 1),
        );
    }

    pub fn encode_rope_tail(
        &self,
        encoder: &ComputeCommandEncoderRef,
        x: &Buffer,
        n_heads: i32,
        head_dim: i32,
        n_rot: i32,
        pos: i32,
        freq_base: f32,
        inverse: bool,
    ) {
        encoder.set_compute_pipeline_state(&self.rope_tail);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_bytes(1, 4, &n_heads as *const i32 as *const _);
        encoder.set_bytes(2, 4, &head_dim as *const i32 as *const _);
        encoder.set_bytes(3, 4, &n_rot as *const i32 as *const _);
        encoder.set_bytes(4, 4, &pos as *const i32 as *const _);
        encoder.set_bytes(5, 4, &freq_base as *const f32 as *const _);
        let inv: i32 = if inverse { 1 } else { 0 };
        encoder.set_bytes(6, 4, &inv as *const i32 as *const _);
        encoder.dispatch_threads(
            MTLSize::new(n_heads as u64, 1, 1),
            MTLSize::new(1, 1, 1),
        );
    }

    pub fn encode_fp8_e4m3fn_nope(
        &self,
        encoder: &ComputeCommandEncoderRef,
        kv: &Buffer,
        head_dim: i32,
        n_rot: i32,
    ) {
        encoder.set_compute_pipeline_state(&self.fp8_e4m3fn_nope);
        encoder.set_buffer(0, Some(kv), 0);
        encoder.set_bytes(1, 4, &head_dim as *const i32 as *const _);
        encoder.set_bytes(2, 4, &n_rot as *const i32 as *const _);
        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
    }

    pub fn encode_kv_store_row(
        &self,
        encoder: &ComputeCommandEncoderRef,
        src: &Buffer,
        cache: &Buffer,
        head_dim: i32,
        row: i32,
    ) {
        encoder.set_compute_pipeline_state(&self.kv_store_row);
        encoder.set_buffer(0, Some(src), 0);
        encoder.set_buffer(1, Some(cache), 0);
        encoder.set_bytes(2, 4, &head_dim as *const i32 as *const _);
        encoder.set_bytes(3, 4, &row as *const i32 as *const _);
        encoder.dispatch_threads(
            MTLSize::new(head_dim as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
    }

    pub fn encode_kv_shift_left(
        &self,
        encoder: &ComputeCommandEncoderRef,
        cache: &Buffer,
        head_dim: i32,
        swa: i32,
    ) {
        encoder.set_compute_pipeline_state(&self.kv_shift_left);
        encoder.set_buffer(0, Some(cache), 0);
        encoder.set_bytes(1, 4, &head_dim as *const i32 as *const _);
        encoder.set_bytes(2, 4, &swa as *const i32 as *const _);
        encoder.dispatch_threads(
            MTLSize::new(head_dim as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
    }

    /// SWA decode after Q/KV latents: RMS → RoPE → FP8 NoPE → KV store → attn → inv-RoPE.
    pub fn encode_swa_rope_attn(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q_heads: &Buffer,
        kv_lat: &Buffer,
        kv_cache: &Buffer,
        kv_norm: Option<&Buffer>,
        sinks: &Buffer,
        head_out: &Buffer,
        n_head: i32,
        head_dim: i32,
        n_rot: i32,
        pos: i32,
        freq_base: f32,
        eps: f32,
        raw_len: i32,
        swa: i32,
    ) {
        self.encode_rms_norm_rows(encoder, q_heads, None, q_heads, n_head, head_dim, eps);
        self.encode_rms_norm(encoder, kv_lat, kv_norm, kv_lat, head_dim, eps);
        self.encode_rope_tail(
            encoder, q_heads, n_head, head_dim, n_rot, pos, freq_base, false,
        );
        self.encode_rope_tail(
            encoder, kv_lat, 1, head_dim, n_rot, pos, freq_base, false,
        );
        self.encode_fp8_e4m3fn_nope(encoder, kv_lat, head_dim, n_rot);
        let (store_row, n_kv) = if raw_len < swa {
            (raw_len, raw_len + 1)
        } else {
            self.encode_kv_shift_left(encoder, kv_cache, head_dim, swa);
            (swa - 1, swa)
        };
        self.encode_kv_store_row(encoder, kv_lat, kv_cache, head_dim, store_row);
        let scale = 1.0 / (head_dim as f32).sqrt();
        self.encode_attn_swa(
            encoder,
            q_heads,
            kv_cache,
            kv_cache,
            sinks,
            head_out,
            n_head,
            head_dim,
            n_kv,
            1,
            scale,
        );
        self.encode_rope_tail(
            encoder, head_out, n_head, head_dim, n_rot, pos, freq_base, true,
        );
    }

    /// Full YaRN / compress RoPE (all args).
    pub fn encode_rope_tail_ext(
        &self,
        encoder: &ComputeCommandEncoderRef,
        x: &Buffer,
        n_heads: i32,
        head_dim: i32,
        n_rot: i32,
        pos: i32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        n_ctx_orig: u32,
        inverse: bool,
    ) {
        encoder.set_compute_pipeline_state(&self.rope_tail_ext);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_bytes(1, 4, &n_heads as *const i32 as *const _);
        encoder.set_bytes(2, 4, &head_dim as *const i32 as *const _);
        encoder.set_bytes(3, 4, &n_rot as *const i32 as *const _);
        encoder.set_bytes(4, 4, &pos as *const i32 as *const _);
        encoder.set_bytes(5, 4, &freq_base as *const f32 as *const _);
        encoder.set_bytes(6, 4, &freq_scale as *const f32 as *const _);
        encoder.set_bytes(7, 4, &ext_factor as *const f32 as *const _);
        encoder.set_bytes(8, 4, &attn_factor as *const f32 as *const _);
        encoder.set_bytes(9, 4, &beta_fast as *const f32 as *const _);
        encoder.set_bytes(10, 4, &beta_slow as *const f32 as *const _);
        encoder.set_bytes(11, 4, &n_ctx_orig as *const u32 as *const _);
        let inv: i32 = if inverse { 1 } else { 0 };
        encoder.set_bytes(12, 4, &inv as *const i32 as *const _);
        encoder.dispatch_threads(
            MTLSize::new(n_heads as u64, 1, 1),
            MTLSize::new(1, 1, 1),
        );
    }

    /// Flash compress RoPE defaults: base 160000, scale 1/16, YaRN on, attn_factor cancel.
    pub fn encode_rope_tail_compress(
        &self,
        encoder: &ComputeCommandEncoderRef,
        x: &Buffer,
        n_heads: i32,
        head_dim: i32,
        n_rot: i32,
        pos: i32,
        inverse: bool,
    ) {
        let freq_scale = 1.0f32 / 16.0;
        let attn_factor = 1.0 / (1.0 + 0.1 * (1.0 / freq_scale).ln());
        self.encode_rope_tail_ext(
            encoder,
            x,
            n_heads,
            head_dim,
            n_rot,
            pos,
            160_000.0,
            freq_scale,
            1.0,
            attn_factor,
            32.0,
            1.0,
            65_536,
            inverse,
        );
    }

    pub fn encode_attn_mixed_mqa(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q: &Buffer,
        k_raw: &Buffer,
        v_raw: &Buffer,
        k_comp: &Buffer,
        v_comp: &Buffer,
        comp_idx: &Buffer,
        sinks: &Buffer,
        out: &Buffer,
        n_head: i32,
        head_dim: i32,
        n_raw: i32,
        n_comp_sel: i32,
        has_sinks: i32,
        scale: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.attn_mixed_mqa);
        encoder.set_buffer(0, Some(q), 0);
        encoder.set_buffer(1, Some(k_raw), 0);
        encoder.set_buffer(2, Some(v_raw), 0);
        encoder.set_buffer(3, Some(k_comp), 0);
        encoder.set_buffer(4, Some(v_comp), 0);
        encoder.set_buffer(5, Some(comp_idx), 0);
        encoder.set_buffer(6, Some(sinks), 0);
        encoder.set_buffer(7, Some(out), 0);
        encoder.set_bytes(8, 4, &n_head as *const i32 as *const _);
        encoder.set_bytes(9, 4, &head_dim as *const i32 as *const _);
        encoder.set_bytes(10, 4, &n_raw as *const i32 as *const _);
        encoder.set_bytes(11, 4, &n_comp_sel as *const i32 as *const _);
        encoder.set_bytes(12, 4, &has_sinks as *const i32 as *const _);
        encoder.set_bytes(13, 4, &scale as *const f32 as *const _);
        // 1 simdgroup (32 threads) per head — two-pass softmax in dsv4_attn_mixed_mqa.
        // scores[640] + inv slot.
        encoder.set_threadgroup_memory_length(0, 641 * 4);
        encoder.dispatch_thread_groups(
            MTLSize::new(n_head as u64, 1, 1),
            MTLSize::new(32, 1, 1),
        );
    }

    pub fn encode_select_comp_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q: &Buffer,
        k_comp: &Buffer,
        out_idx: &Buffer,
        scores: &Buffer,
        n_head: i32,
        head_dim: i32,
        n_comp: i32,
        top_k: i32,
    ) {
        if n_comp <= 0 {
            return;
        }
        // Parallel score: one thread per compressed row.
        encoder.set_compute_pipeline_state(&self.select_comp_score);
        encoder.set_buffer(0, Some(q), 0);
        encoder.set_buffer(1, Some(k_comp), 0);
        encoder.set_buffer(2, Some(scores), 0);
        encoder.set_bytes(3, 4, &n_head as *const i32 as *const _);
        encoder.set_bytes(4, 4, &head_dim as *const i32 as *const _);
        encoder.set_bytes(5, 4, &n_comp as *const i32 as *const _);
        let tg = 64u64.min(n_comp as u64).max(1);
        encoder.dispatch_threads(
            MTLSize::new(n_comp as u64, 1, 1),
            MTLSize::new(tg, 1, 1),
        );
        // Serial top-k from scores.
        encoder.set_compute_pipeline_state(&self.select_comp_topk);
        encoder.set_buffer(0, Some(scores), 0);
        encoder.set_buffer(1, Some(out_idx), 0);
        encoder.set_bytes(2, 4, &n_comp as *const i32 as *const _);
        encoder.set_bytes(3, 4, &top_k as *const i32 as *const _);
        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
    }

    pub fn encode_iota_i32(
        &self,
        encoder: &ComputeCommandEncoderRef,
        out: &Buffer,
        n: i32,
    ) {
        if n <= 0 {
            return;
        }
        encoder.set_compute_pipeline_state(&self.iota_i32);
        encoder.set_buffer(0, Some(out), 0);
        encoder.set_bytes(1, 4, &n as *const i32 as *const _);
        encoder.dispatch_threads(
            MTLSize::new(n as u64, 1, 1),
            MTLSize::new(64.min(n as u64).max(1), 1, 1),
        );
    }

    /// GPU compressor step (staged parallel kernels).
    /// `ape` is F16 half* from `GpuWeight` buf; `out_row` unused when write_out=0.
    pub fn encode_compressor_update(
        &self,
        encoder: &ComputeCommandEncoderRef,
        kv_cur: &Buffer,
        sc_cur: &Buffer,
        ape: &Buffer,
        norm: &Buffer,
        state_kv: &Buffer,
        state_score: &Buffer,
        out_row: &Buffer,
        ratio: i32,
        width: i32,
        head_dim: i32,
        pos: i32,
        n_rot: i32,
        rms_eps: f32,
        use_compress_rope: bool,
        rope_freq: f32,
        write_out: bool,
    ) {
        let tg = |n: i32| -> u64 { 64u64.min(n.max(1) as u64).max(1) };
        // Stage 1: store
        encoder.set_compute_pipeline_state(&self.compressor_store_row);
        encoder.set_buffer(0, Some(kv_cur), 0);
        encoder.set_buffer(1, Some(sc_cur), 0);
        encoder.set_buffer(2, Some(ape), 0);
        encoder.set_buffer(3, Some(state_kv), 0);
        encoder.set_buffer(4, Some(state_score), 0);
        encoder.set_bytes(5, 4, &ratio as *const i32 as *const _);
        encoder.set_bytes(6, 4, &width as *const i32 as *const _);
        encoder.set_bytes(7, 4, &pos as *const i32 as *const _);
        encoder.dispatch_threads(
            MTLSize::new(width.max(1) as u64, 1, 1),
            MTLSize::new(tg(width), 1, 1),
        );

        let should_compress = (pos + 1) % ratio == 0;
        if !should_compress {
            return;
        }

        if write_out {
            // Stage 2: pool
            encoder.set_compute_pipeline_state(&self.compressor_pool);
            encoder.set_buffer(0, Some(state_kv), 0);
            encoder.set_buffer(1, Some(state_score), 0);
            encoder.set_buffer(2, Some(out_row), 0);
            encoder.set_bytes(3, 4, &ratio as *const i32 as *const _);
            encoder.set_bytes(4, 4, &width as *const i32 as *const _);
            encoder.set_bytes(5, 4, &head_dim as *const i32 as *const _);
            encoder.dispatch_threads(
                MTLSize::new(head_dim.max(1) as u64, 1, 1),
                MTLSize::new(tg(head_dim), 1, 1),
            );
            // Stage 3: rms + rope + fp8
            encoder.set_compute_pipeline_state(&self.compressor_rms_rope_fp8);
            encoder.set_buffer(0, Some(out_row), 0);
            encoder.set_buffer(1, Some(norm), 0);
            encoder.set_bytes(2, 4, &head_dim as *const i32 as *const _);
            encoder.set_bytes(3, 4, &n_rot as *const i32 as *const _);
            encoder.set_bytes(4, 4, &pos as *const i32 as *const _);
            encoder.set_bytes(5, 4, &ratio as *const i32 as *const _);
            encoder.set_bytes(6, 4, &rms_eps as *const f32 as *const _);
            let use_cr: i32 = if use_compress_rope { 1 } else { 0 };
            encoder.set_bytes(7, 4, &use_cr as *const i32 as *const _);
            encoder.set_bytes(8, 4, &rope_freq as *const f32 as *const _);
            encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        }

        // Stage 4: CSA shuffle on emit
        if ratio == 4 {
            for pass in [0i32, 1i32] {
                encoder.set_compute_pipeline_state(&self.compressor_csa_shuffle);
                encoder.set_buffer(0, Some(state_kv), 0);
                encoder.set_buffer(1, Some(state_score), 0);
                encoder.set_bytes(2, 4, &ratio as *const i32 as *const _);
                encoder.set_bytes(3, 4, &width as *const i32 as *const _);
                encoder.set_bytes(4, 4, &pass as *const i32 as *const _);
                encoder.dispatch_threads(
                    MTLSize::new(width.max(1) as u64, 1, 1),
                    MTLSize::new(tg(width), 1, 1),
                );
            }
        }
    }

    /// Fused CSA/HCA after Q/KV latents are already in `q_heads` / `kv_lat`.
    ///
    /// Encodes: RMS(Q/KV) → compress-RoPE → FP8(KV) → store raw ring →
    /// optional attn/idx compressor matvecs+update (+ store emit row) →
    /// select/iota → mixed attn → inv compress-RoPE on `head_out`.
    ///
    /// Host after wait: `note_raw_push`; if `(pos+1)%ratio==0` also `note_comp_push`.
    /// Compressor GPU state lives in the state_* buffers (not CPU `CompressorState`).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_csa_hca(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q_heads: &Buffer,
        kv_lat: &Buffer,
        kv_raw: &Buffer,
        kv_comp: &Buffer,
        kv_norm: Option<&Buffer>,
        sinks: &Buffer,
        head_out: &Buffer,
        x_normed: &Buffer,
        attn_kv_out: &Buffer,
        attn_gate_out: &Buffer,
        idx_kv_out: &Buffer,
        idx_gate_out: &Buffer,
        comp_emit: &Buffer,
        comp_idx: &Buffer,
        comp_scores: &Buffer,
        attn_comp: Option<(
            &GpuWeight,
            &GpuWeight,
            &GpuWeight,
            &Buffer,
            &Buffer,
            &Buffer,
            i32,
            i32,
        )>,
        idx_comp: Option<(
            &GpuWeight,
            &GpuWeight,
            &GpuWeight,
            &Buffer,
            &Buffer,
            &Buffer,
            i32,
            i32,
        )>,
        n_head: i32,
        head_dim: i32,
        n_rot: i32,
        pos: i32,
        eps: f32,
        n_embd: i32,
        raw_len: i32,
        swa: i32,
        comp_len: i32,
        ratio: i32,
        top_k: i32,
        use_compress_rope: bool,
        rope_freq: f32,
    ) {
        self.encode_rms_norm_rows(encoder, q_heads, None, q_heads, n_head, head_dim, eps);
        self.encode_rms_norm(encoder, kv_lat, kv_norm, kv_lat, head_dim, eps);
        self.encode_rope_tail_compress(
            encoder, q_heads, n_head, head_dim, n_rot, pos, false,
        );
        self.encode_rope_tail_compress(
            encoder, kv_lat, 1, head_dim, n_rot, pos, false,
        );
        self.encode_fp8_e4m3fn_nope(encoder, kv_lat, head_dim, n_rot);
        let (store_row, n_raw) = if raw_len < swa {
            (raw_len, raw_len + 1)
        } else {
            self.encode_kv_shift_left(encoder, kv_raw, head_dim, swa);
            (swa - 1, swa)
        };
        self.encode_kv_store_row(encoder, kv_lat, kv_raw, head_dim, store_row);

        let emit = (pos + 1) % ratio == 0;
        let mut did_emit = false;
        if let Some((wkv, wgate, ape, norm, state_kv, state_score, width, hd)) = attn_comp {
            self.encode_matvec_kind(
                encoder,
                wkv.kind,
                &wkv.buf,
                x_normed,
                attn_kv_out,
                width,
                n_embd,
                0,
            );
            self.encode_matvec_kind(
                encoder,
                wgate.kind,
                &wgate.buf,
                x_normed,
                attn_gate_out,
                width,
                n_embd,
                0,
            );
            self.encode_compressor_update(
                encoder,
                attn_kv_out,
                attn_gate_out,
                &ape.buf,
                norm,
                state_kv,
                state_score,
                comp_emit,
                ratio,
                width,
                hd,
                pos,
                n_rot,
                eps,
                use_compress_rope,
                rope_freq,
                emit,
            );
            if emit {
                self.encode_kv_store_row(encoder, comp_emit, kv_comp, head_dim, comp_len);
                did_emit = true;
            }
        }
        if let Some((wkv, wgate, ape, norm, state_kv, state_score, width, hd)) = idx_comp {
            self.encode_matvec_kind(
                encoder,
                wkv.kind,
                &wkv.buf,
                x_normed,
                idx_kv_out,
                width,
                n_embd,
                0,
            );
            self.encode_matvec_kind(
                encoder,
                wgate.kind,
                &wgate.buf,
                x_normed,
                idx_gate_out,
                width,
                n_embd,
                0,
            );
            self.encode_compressor_update(
                encoder,
                idx_kv_out,
                idx_gate_out,
                &ape.buf,
                norm,
                state_kv,
                state_score,
                comp_emit,
                ratio,
                width,
                hd,
                pos,
                n_rot,
                eps,
                use_compress_rope,
                rope_freq,
                false,
            );
        }

        let n_comp_attn = if did_emit { comp_len + 1 } else { comp_len };
        let n_comp_sel = if n_comp_attn <= 0 {
            0
        } else if ratio == 4 {
            let k = top_k.min(n_comp_attn);
            self.encode_select_comp_rows(
                encoder,
                q_heads,
                kv_comp,
                comp_idx,
                comp_scores,
                n_head,
                head_dim,
                n_comp_attn,
                k,
            );
            k
        } else {
            self.encode_iota_i32(encoder, comp_idx, n_comp_attn);
            n_comp_attn
        };

        let scale = 1.0 / (head_dim as f32).sqrt();
        self.encode_attn_mixed_mqa(
            encoder,
            q_heads,
            kv_raw,
            kv_raw,
            kv_comp,
            kv_comp,
            comp_idx,
            sinks,
            head_out,
            n_head,
            head_dim,
            n_raw,
            n_comp_sel,
            1,
            scale,
        );
        self.encode_rope_tail_compress(
            encoder, head_out, n_head, head_dim, n_rot, pos, true,
        );
    }

    /// Synchronous dense matvec using caller-provided scratch x/y buffers when possible.
    pub fn matvec_sync(&self, w: &GpuWeight, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), w.n_in);
        assert_eq!(out.len(), w.n_out);
        let x_buf = self.buffer_from_f32(x);
        let y_buf = self.buffer_zeros(w.n_out * 4);
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        self.encode_matvec_kind(
            &enc,
            w.kind,
            &w.buf,
            &x_buf,
            &y_buf,
            w.n_out as i32,
            w.n_in as i32,
            0,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let got = Self::read_f32(&y_buf, w.n_out);
        out.copy_from_slice(&got);
    }

    pub fn matvec_sync_scratch(
        &self,
        w: &GpuWeight,
        x: &[f32],
        out: &mut [f32],
        x_buf: &Buffer,
        y_buf: &Buffer,
    ) {
        assert_eq!(x.len(), w.n_in);
        assert_eq!(out.len(), w.n_out);
        Self::write_f32(x_buf, x);
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        self.encode_matvec_kind(
            &enc,
            w.kind,
            &w.buf,
            x_buf,
            y_buf,
            w.n_out as i32,
            w.n_in as i32,
            0,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let got = Self::read_f32(y_buf, w.n_out);
        out.copy_from_slice(&got);
    }

    /// Matvec over a contiguous row slice of a weight matrix (grouped LoRA-O).
    pub fn matvec_rows_sync(
        &self,
        w: &GpuWeight,
        x: &[f32],
        row_start: usize,
        n_out: usize,
        out: &mut [f32],
    ) {
        assert_eq!(x.len(), w.n_in);
        assert_eq!(out.len(), n_out);
        let row_bytes = match w.kind {
            GpuWKind::F16 => w.n_in * 2,
            GpuWKind::F32 => w.n_in * 4,
            GpuWKind::Q8 => (w.n_in / 32) * 34,
            _ => panic!("matvec_rows_sync only for dense F16/F32/Q8"),
        };
        let x_buf = self.buffer_from_f32(x);
        let y_buf = self.buffer_zeros(n_out * 4);
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        self.encode_matvec_kind(
            &enc,
            w.kind,
            &w.buf,
            &x_buf,
            &y_buf,
            n_out as i32,
            w.n_in as i32,
            (row_start * row_bytes) as u64,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let got = Self::read_f32(&y_buf, n_out);
        out.copy_from_slice(&got);
    }

    pub fn wait_cmd(&self, cmd: &metal::CommandBufferRef, scratch: Option<&MetalScratch>) {
        cmd.wait_until_completed();
        if let Some(sc) = scratch {
            sc.note_wait();
        }
    }

    /// Wait on an owned CB (same-queue prior commits are covered).
    pub fn wait_owned(&self, cmd: &metal::CommandBuffer, scratch: Option<&MetalScratch>) {
        self.wait_cmd(cmd, scratch);
    }

    /// GPU waits until `signal_pread` (CPU) after miss preads complete.
    pub fn next_pread_epoch(&self) -> u64 {
        self.pread_epoch.fetch_add(1, Ordering::Relaxed) + 1
    }

    pub fn encode_wait_pread(&self, cmd: &metal::CommandBuffer, epoch: u64) {
        let event: &EventRef = self.pread_event.as_ref();
        cmd.encode_wait_for_event(event, epoch);
    }

    pub fn signal_pread(&self, epoch: u64) {
        self.pread_event.set_signaled_value(epoch);
    }
}

/// Reusable activation / expert scratch for Metal FFN encode.
pub struct MetalScratch {
    pub x: Buffer,
    pub y: Buffer,
    pub gate: Buffer,
    pub up: Buffer,
    pub mid: Buffer,
    /// Striped mids for slots6: [max_routed][n_ff].
    pub routed_mid: Buffer,
    pub route_w: Buffer,
    pub route_ids: Buffer,
    /// Expert-id → LRU slot (-1 miss), length n_expert.
    pub slot_map: Buffer,
    /// Top-k slot indices after GPU map.
    pub route_slots: Buffer,
    pub moe_miss_flag: Buffer,
    pub route_ids_hist: Buffer,
    pub down: Buffer,
    pub shared: Buffer,
    pub routed: Buffer,
    pub logits: Buffer,
    pub probs: Buffer,
    pub expert_gate: Buffer,
    pub expert_up: Buffer,
    pub expert_down: Buffer,
    /// Per-routed-expert staging (top-k ≤ 8).
    pub exp_gate: Vec<Buffer>,
    pub exp_up: Vec<Buffer>,
    pub exp_down: Vec<Buffer>,
    pub exp_down_out: Vec<Buffer>,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_expert: usize,
    pub y_cap: usize,
    pub gate_bytes: usize,
    pub up_bytes: usize,
    pub down_bytes: usize,
    pub max_routed: usize,
    pub qa: Buffer,
    pub qa_n: Buffer,
    pub q_heads: Buffer,
    pub kv_lat: Buffer,
    pub head_out: Buffer,
    pub mix: Buffer,
    pub split: Buffer,
    pub hc_flat: Buffer,
    /// Attn-expanded HC snapshot for speculative MoE rollback (`y` after attn expand).
    pub hc_snap: Buffer,
    pub attn_in: Buffer,
    pub low: Buffer,
    pub kv_cache: Buffer,
    pub kv_cache_cap: usize,
    /// Per-SWA-layer GPU rings (Flash layers 0 and 1).
    pub swa_kv0: Buffer,
    pub swa_kv1: Buffer,
    /// CSA top-k / HCA iota indices (≤512).
    pub comp_idx: Buffer,
    /// CSA score scratch for parallel select (≤2048).
    pub comp_scores: Buffer,
    /// Compressor emit row scratch (≤ head_dim=512).
    pub comp_emit: Buffer,
    pub sync_waits: std::cell::Cell<u64>,
    pub prof_pin_ns: std::cell::Cell<u64>,
    pub prof_gpu_ns: std::cell::Cell<u64>,
    pub prof_attn_ns: std::cell::Cell<u64>,
    pub prof_copy_ns: std::cell::Cell<u64>,
    pub prof_cb1_ns: std::cell::Cell<u64>,
    pub prof_cb2_ns: std::cell::Cell<u64>,
    pub prof_cb3_ns: std::cell::Cell<u64>,
    /// `DSV4_STAGE=1`: extra waits splitting CB2 (attn is `prof_cb1` when unmerged).
    pub prof_stage_hc_ns: std::cell::Cell<u64>,
    pub prof_stage_router_ns: std::cell::Cell<u64>,
    /// Speculative MoE-in-CB: ids == last_ids.
    pub spec_moe_match: std::cell::Cell<u64>,
    /// Speculative MoE-in-CB: ids != last_ids (corrective MoE).
    pub spec_moe_miss: std::cell::Cell<u64>,
    /// Same expert set as last_ids but different top-k order.
    pub spec_moe_set_eq: std::cell::Cell<u64>,
    /// Sum of |ids ∩ last_ids| over misses (for avg overlap).
    pub spec_moe_overlap_sum: std::cell::Cell<u64>,
    /// Last MoE CB not yet host-waited (same-queue later waits cover it).
    pub pending_cmd: std::cell::RefCell<Option<metal::CommandBuffer>>,
}

impl MetalScratch {
    pub fn new(
        metal: &Dsv4Metal,
        n_embd: usize,
        n_ff: usize,
        n_expert: usize,
        n_vocab: usize,
        gate_bytes: usize,
        up_bytes: usize,
        down_bytes: usize,
    ) -> Self {
        let y_cap = n_vocab.max(n_ff).max(n_expert).max(n_embd * 4).max(32768);
        let max_routed = 8;
        let mut exp_gate = Vec::with_capacity(max_routed);
        let mut exp_up = Vec::with_capacity(max_routed);
        let mut exp_down = Vec::with_capacity(max_routed);
        let mut exp_down_out = Vec::with_capacity(max_routed);
        for _ in 0..max_routed {
            exp_gate.push(metal.buffer_zeros(gate_bytes));
            exp_up.push(metal.buffer_zeros(up_bytes));
            exp_down.push(metal.buffer_zeros(down_bytes));
            exp_down_out.push(metal.buffer_zeros(n_embd * 4));
        }
        Self {
            x: metal.buffer_zeros(n_embd.max(n_ff).max(16384) * 4),
            y: metal.buffer_zeros(y_cap * 4),
            gate: metal.buffer_zeros(n_ff * 4),
            up: metal.buffer_zeros(n_ff * 4),
            mid: metal.buffer_zeros(n_ff * 4),
            routed_mid: metal.buffer_zeros(max_routed * n_ff * 4),
            route_w: metal.buffer_zeros(max_routed * 4),
            route_ids: metal.buffer_zeros(max_routed * 4),
            slot_map: metal.buffer_zeros(n_expert * 4),
            route_slots: metal.buffer_zeros(max_routed * 4),
            moe_miss_flag: metal.buffer_zeros(4),
            // Per-layer route ids for end-of-token last_ids refresh after fused CBs.
            route_ids_hist: metal.buffer_zeros(64 * max_routed * 4), // up to 64 layers
            down: metal.buffer_zeros(n_embd * 4),
            shared: metal.buffer_zeros(n_embd * 4),
            routed: metal.buffer_zeros(n_embd * 4),
            logits: metal.buffer_zeros(n_expert * 4),
            probs: metal.buffer_zeros(n_expert * 4),
            expert_gate: metal.buffer_zeros(gate_bytes),
            expert_up: metal.buffer_zeros(up_bytes),
            expert_down: metal.buffer_zeros(down_bytes),
            exp_gate,
            exp_up,
            exp_down,
            exp_down_out,
            n_embd,
            n_ff,
            n_expert,
            y_cap,
            gate_bytes,
            up_bytes,
            down_bytes,
            max_routed,
            qa: metal.buffer_zeros(4096 * 4),
            qa_n: metal.buffer_zeros(4096 * 4),
            q_heads: metal.buffer_zeros(64 * 512 * 4),
            kv_lat: metal.buffer_zeros(512 * 4),
            head_out: metal.buffer_zeros(64 * 512 * 4),
            mix: metal.buffer_zeros(64 * 4),
            split: metal.buffer_zeros(64 * 4),
            hc_flat: metal.buffer_zeros(4 * 4096 * 4),
            hc_snap: metal.buffer_zeros(4 * 4096 * 4),
            attn_in: metal.buffer_zeros(4096 * 4),
            low: metal.buffer_zeros(8 * 1024 * 4),
            // SWA window × head_dim; grow on demand via ensure_kv_cache.
            kv_cache: metal.buffer_zeros(128 * 512 * 4),
            kv_cache_cap: 128 * 512,
            swa_kv0: metal.buffer_zeros(128 * 512 * 4),
            swa_kv1: metal.buffer_zeros(128 * 512 * 4),
            comp_idx: metal.buffer_zeros(512 * 4),
            comp_scores: metal.buffer_zeros(2048 * 4),
            comp_emit: metal.buffer_zeros(512 * 4),
            sync_waits: std::cell::Cell::new(0),
            prof_pin_ns: std::cell::Cell::new(0),
            prof_gpu_ns: std::cell::Cell::new(0),
            prof_attn_ns: std::cell::Cell::new(0),
            prof_copy_ns: std::cell::Cell::new(0),
            prof_cb1_ns: std::cell::Cell::new(0),
            prof_cb2_ns: std::cell::Cell::new(0),
            prof_cb3_ns: std::cell::Cell::new(0),
            prof_stage_hc_ns: std::cell::Cell::new(0),
            prof_stage_router_ns: std::cell::Cell::new(0),
            spec_moe_match: std::cell::Cell::new(0),
            spec_moe_miss: std::cell::Cell::new(0),
            spec_moe_set_eq: std::cell::Cell::new(0),
            spec_moe_overlap_sum: std::cell::Cell::new(0),
            pending_cmd: std::cell::RefCell::new(None),
        }
    }

    pub fn note_wait(&self) {
        self.sync_waits.set(self.sync_waits.get() + 1);
    }

    pub fn defer_cmd(&self, cmd: metal::CommandBuffer) {
        // Prior deferred CB is ordered before `cmd` on the same queue.
        *self.pending_cmd.borrow_mut() = Some(cmd);
    }

    pub fn clear_pending(&self) {
        *self.pending_cmd.borrow_mut() = None;
    }

    pub fn wait_pending(&self, metal: &Dsv4Metal) {
        if let Some(cmd) = self.pending_cmd.borrow_mut().take() {
            metal.wait_owned(&cmd, Some(self));
        }
    }

    pub fn reset_profile(&self) {
        self.prof_pin_ns.set(0);
        self.prof_gpu_ns.set(0);
        self.prof_attn_ns.set(0);
        self.prof_copy_ns.set(0);
        self.prof_cb1_ns.set(0);
        self.prof_cb2_ns.set(0);
        self.prof_cb3_ns.set(0);
        self.prof_stage_hc_ns.set(0);
        self.prof_stage_router_ns.set(0);
        self.spec_moe_match.set(0);
        self.spec_moe_miss.set(0);
        self.spec_moe_set_eq.set(0);
        self.spec_moe_overlap_sum.set(0);
    }

    pub fn add_prof(cell: &std::cell::Cell<u64>, start: std::time::Instant) {
        cell.set(cell.get() + start.elapsed().as_nanos() as u64);
    }
}

#[cfg(test)]
mod mv_bench {
    use super::*;
    use std::time::Instant;

    #[test]
    fn bench_f16_q8_matvec() {
        let metal = Dsv4Metal::new();
        let n_out = 4096usize;
        let n_in = 4096usize;
        let mut w_f16 = vec![0u16; n_out * n_in];
        for (i, w) in w_f16.iter_mut().enumerate() {
            *w = crate::gpu::f32_to_f16(((i % 17) as f32) * 0.01);
        }
        let w = GpuWeight {
            buf: metal.buffer_from_bytes(unsafe {
                std::slice::from_raw_parts(w_f16.as_ptr() as *const u8, w_f16.len() * 2)
            }),
            kind: GpuWKind::F16,
            n_out,
            n_in,
        };
        let x = vec![0.01f32; n_in];
        let mut y = vec![0.0f32; n_out];
        // warmup
        metal.matvec_sync(&w, &x, &mut y);
        let t0 = Instant::now();
        let n = 20;
        for _ in 0..n {
            metal.matvec_sync(&w, &x, &mut y);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        eprintln!("F16 {n_out}x{n_in} matvec: {ms:.3} ms/call (sum={:.4})", y.iter().sum::<f32>());
        assert!(ms < 50.0, "F16 matvec unexpectedly slow: {ms} ms");
    }
}
