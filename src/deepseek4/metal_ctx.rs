//! Separate Metal library for DeepSeek-V4 kernels (compiled on demand).

use metal::*;
use std::path::Path;
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
    pub fn from_dense(metal: &Dsv4Metal, w: &DenseW) -> Self {
        let dims = w.dims();
        let n_in = dims[0];
        let n_out = if dims.len() > 1 { dims[1] } else { 1 };
        match w {
            DenseW::F32 { data, .. } => Self {
                buf: metal.buffer_from_bytes(unsafe {
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
                buf: metal.buffer_from_bytes(unsafe {
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
                buf: metal.buffer_from_bytes(data),
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
            buf: metal.buffer_from_bytes(data),
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
    pub matvec_q2_k: ComputePipelineState,
    pub matvec_f16: ComputePipelineState,
    pub matvec_f32: ComputePipelineState,
    pub matvec_q8_0: ComputePipelineState,
    pub swiglu: ComputePipelineState,
    pub router_sqrt_softplus: ComputePipelineState,
    pub attn_swa_mqa: ComputePipelineState,
    pub attn_mixed_mqa: ComputePipelineState,
    pub rope_tail: ComputePipelineState,
    pub fp8_store: ComputePipelineState,
    pub compress_mean_pool: ComputePipelineState,
    pub indexer_scores: ComputePipelineState,
    pub axpy: ComputePipelineState,
    pub add: ComputePipelineState,
    pub zero: ComputePipelineState,
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
        let matvec_f16 = get("dsv4_matvec_f16");
        let matvec_f32 = get("dsv4_matvec_f32");
        let matvec_q8_0 = get("dsv4_matvec_q8_0");
        let swiglu = get("dsv4_swiglu");
        let router_sqrt_softplus = get("dsv4_router_sqrt_softplus");
        let attn_swa_mqa = get("dsv4_attn_swa_mqa");
        let attn_mixed_mqa = get("dsv4_attn_mixed_mqa");
        let rope_tail = get("dsv4_rope_tail");
        let fp8_store = get("dsv4_fp8_store");
        let compress_mean_pool = get("dsv4_compress_mean_pool");
        let indexer_scores = get("dsv4_indexer_scores");
        let axpy = get("dsv4_axpy");
        let add = get("dsv4_add");
        let zero = get("dsv4_zero");
        Arc::new(Self {
            device,
            queue,
            hc_split_sinkhorn,
            hc_weighted_sum,
            hc_expand_post,
            rms_norm,
            matvec_iq2_xxs,
            matvec_q2_k,
            matvec_f16,
            matvec_f32,
            matvec_q8_0,
            swiglu,
            router_sqrt_softplus,
            attn_swa_mqa,
            attn_mixed_mqa,
            rope_tail,
            fp8_store,
            compress_mean_pool,
            indexer_scores,
            axpy,
            add,
            zero,
        })
    }

    pub fn buffer_from_bytes(&self, data: &[u8]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
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
        let pipe = match kind {
            GpuWKind::F16 => &self.matvec_f16,
            GpuWKind::F32 => &self.matvec_f32,
            GpuWKind::Q8 => &self.matvec_q8_0,
            GpuWKind::Iq2 => &self.matvec_iq2_xxs,
            GpuWKind::Q2K => &self.matvec_q2_k,
        };
        encoder.set_compute_pipeline_state(pipe);
        encoder.set_buffer(0, Some(weight), weight_byte_off);
        encoder.set_buffer(1, Some(x), 0);
        encoder.set_buffer(2, Some(out), 0);
        encoder.set_bytes(3, 4, &n_out as *const i32 as *const _);
        encoder.set_bytes(4, 4, &n_in as *const i32 as *const _);
        match kind {
            GpuWKind::F16 | GpuWKind::F32 | GpuWKind::Q8 | GpuWKind::Iq2 | GpuWKind::Q2K => {
                let tg = MTLSize::new(32, 1, 1);
                let grid = MTLSize::new(n_out as u64, 1, 1);
                encoder.dispatch_thread_groups(grid, tg);
            }
        }
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
        encoder.set_compute_pipeline_state(&self.hc_expand_post);
        encoder.set_buffer(0, Some(block), 0);
        encoder.set_buffer(1, Some(add_hc), 0);
        encoder.set_buffer(2, Some(post), 0);
        encoder.set_buffer(3, Some(comb), 0);
        encoder.set_buffer(4, Some(hc), 0);
        encoder.set_bytes(5, 4, &n_embd as *const i32 as *const _);
        encoder.set_bytes(6, 4, &n_hc as *const i32 as *const _);
        encoder.dispatch_threads(
            MTLSize::new(n_embd as u64, 1, 1),
            MTLSize::new(64, 1, 1),
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
        encoder.dispatch_threads(
            MTLSize::new(n_head as u64, 1, 1),
            MTLSize::new(1, 1, 1),
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
}

/// Reusable activation / expert scratch for Metal FFN encode.
pub struct MetalScratch {
    pub x: Buffer,
    pub y: Buffer,
    pub gate: Buffer,
    pub up: Buffer,
    pub mid: Buffer,
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
    pub attn_in: Buffer,
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
            attn_in: metal.buffer_zeros(4096 * 4),
        }
    }
}
