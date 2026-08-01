//! Gemma 4 MoE helpers (26B-A4B): top-k routing, expert views, CPU cross-check.

use crate::gguf::{self, ggml_type, dequant_row_to_f32};
use crate::gpu::{f16_to_f32, BufferView};

/// Softmax over all experts, then top-k with renormalized weights.
pub fn softmax_topk_renorm(logits: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    assert!(!logits.is_empty());
    let k = k.min(logits.len()).max(1);
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&x| (x - max_l).exp()).collect();
    let sum: f32 = probs.iter().sum();
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for p in &mut probs {
        *p *= inv;
    }
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    let wsum: f32 = idx.iter().map(|&i| probs[i]).sum();
    let inv_w = if wsum > 0.0 { 1.0 / wsum } else { 0.0 };
    let weights: Vec<f32> = idx.iter().map(|&i| probs[i] * inv_w).collect();
    (idx, weights)
}

/// Byte stride of one expert's fused gate∥up Q4_K matrix `[n_embd, 2*n_ff]`.
pub fn gate_up_expert_bytes(n_embd: usize, n_ff_exp: usize) -> u64 {
    let n_ff2 = n_ff_exp * 2;
    let blocks_per_row = n_embd / 256;
    (n_ff2 * blocks_per_row * 144) as u64
}

/// Byte offset of gate (or up) rows within one expert's gate_up blob.
/// Gate is the first `n_ff` rows; up is the second (llama.cpp / conversion).
pub fn gate_up_half_offset(n_embd: usize, n_ff_exp: usize, is_up: bool) -> u64 {
    let blocks_per_row = n_embd / 256;
    let row_bytes = (blocks_per_row * 144) as u64;
    if is_up {
        n_ff_exp as u64 * row_bytes
    } else {
        0
    }
}

/// Byte stride of one expert's down matrix `[n_ff_exp, n_embd]` for the given
/// ggml block size (24 for Q5_1, 34 for Q8_0).
pub fn down_expert_bytes(n_ff_exp: usize, n_embd: usize, block_bytes: usize) -> u64 {
    let epb = 32usize;
    let n = n_ff_exp * n_embd;
    ((n / epb) * block_bytes) as u64
}

fn down_block_bytes(format: u8) -> usize {
    match format {
        crate::gpu::weight_fmt::Q5_1 => 24,
        crate::gpu::weight_fmt::Q8_0 => 34,
        other => panic!("unsupported MoE down format {other}"),
    }
}

/// View into one expert's gate or up half of `gate_up_exps` (Q4_K).
pub fn expert_gate_up_view(
    gate_up_exps: &BufferView,
    expert: usize,
    n_embd: usize,
    n_ff_exp: usize,
    is_up: bool,
) -> BufferView {
    let expert_bytes = gate_up_expert_bytes(n_embd, n_ff_exp);
    let half = gate_up_half_offset(n_embd, n_ff_exp, is_up);
    let half_bytes = gate_up_expert_bytes(n_embd, n_ff_exp) / 2;
    BufferView {
        buffer: gate_up_exps.buffer.clone(),
        offset: gate_up_exps.offset + expert as u64 * expert_bytes + half,
        length: half_bytes,
        format: crate::gpu::weight_fmt::Q4_K,
    }
}

/// View into one expert's down matrix `[n_embd rows × n_ff_exp cols]`.
pub fn expert_down_view(
    down_exps: &BufferView,
    expert: usize,
    n_ff_exp: usize,
    n_embd: usize,
) -> BufferView {
    let bpb = down_block_bytes(down_exps.format);
    let stride = down_expert_bytes(n_ff_exp, n_embd, bpb);
    BufferView {
        buffer: down_exps.buffer.clone(),
        offset: down_exps.offset + expert as u64 * stride,
        length: stride,
        format: down_exps.format,
    }
}

fn expert_gate_up_bytes<'a>(
    gate_up_exps: &'a BufferView,
    expert: usize,
    n_embd: usize,
    n_ff_exp: usize,
    is_up: bool,
) -> &'a [u8] {
    let expert_bytes = gate_up_expert_bytes(n_embd, n_ff_exp) as usize;
    let half = gate_up_half_offset(n_embd, n_ff_exp, is_up) as usize;
    let half_bytes = expert_bytes / 2;
    let all = gate_up_exps.as_bytes();
    let base = expert * expert_bytes + half;
    &all[base..base + half_bytes]
}

fn expert_down_bytes<'a>(
    down_exps: &'a BufferView,
    expert: usize,
    n_ff_exp: usize,
    n_embd: usize,
) -> &'a [u8] {
    let bpb = down_block_bytes(down_exps.format);
    let stride = down_expert_bytes(n_ff_exp, n_embd, bpb) as usize;
    let all = down_exps.as_bytes();
    let base = expert * stride;
    &all[base..base + stride]
}

/// CPU Q4_K gemv: `y[m] = W[m, k] @ x[k]`.
pub fn q4_k_gemv(weight_bytes: &[u8], x: &[f32], y: &mut [f32], m: usize, k: usize) {
    assert_eq!(x.len(), k);
    assert_eq!(y.len(), m);
    assert_eq!(k % 256, 0);
    let blocks_per_row = k / 256;
    let row_bytes = blocks_per_row * 144;
    assert_eq!(weight_bytes.len(), m * row_bytes);
    let mut row_f = vec![0.0f32; k];
    for row in 0..m {
        let bytes = &weight_bytes[row * row_bytes..(row + 1) * row_bytes];
        dequant_row_to_f32(ggml_type::Q4_K, bytes, k, &mut row_f);
        let mut acc = 0.0f32;
        for i in 0..k {
            acc += row_f[i] * x[i];
        }
        y[row] = acc;
    }
}

fn gelu_mul_cpu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_560_8;
    const GELU_COEF_A: f32 = 0.044_715;
    for i in 0..gate.len() {
        let x = gate[i];
        let gelu = 0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)).tanh());
        out[i] = gelu * up[i];
    }
}

/// CPU Q5_1 gemv: `y[n_embd] = scale * W[n_embd, n_ff] @ x[n_ff]` (overwrite).
pub fn q5_1_gemv_expert(
    down_bytes: &[u8],
    x: &[f32],
    y: &mut [f32],
    n_ff_exp: usize,
    n_embd: usize,
    scale: f32,
) {
    assert_eq!(x.len(), n_ff_exp);
    assert_eq!(y.len(), n_embd);
    let expected = down_expert_bytes(n_ff_exp, n_embd, 24) as usize;
    assert_eq!(down_bytes.len(), expected);
    let blocks_per_row = n_ff_exp / 32;
    let row_bytes = blocks_per_row * 24;
    for m in 0..n_embd {
        let row = &down_bytes[m * row_bytes..(m + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let base = b * 24;
            let d = f16_to_f32(u16::from_le_bytes([row[base], row[base + 1]]));
            let minv = f16_to_f32(u16::from_le_bytes([row[base + 2], row[base + 3]]));
            let qh = u32::from_le_bytes([
                row[base + 4],
                row[base + 5],
                row[base + 6],
                row[base + 7],
            ]);
            let qs = &row[base + 8..base + 24];
            let xoff = b * 32;
            for i in 0..16 {
                let xh_0 = (((qh >> (i + 0)) << 4) & 0x10) as f32;
                let xh_1 = ((qh >> (i + 12)) & 0x10) as f32;
                let x0 = (qs[i] & 0x0F) as f32 + xh_0;
                let x1 = (qs[i] >> 4) as f32 + xh_1;
                acc += (x0 * d + minv) * x[xoff + i];
                acc += (x1 * d + minv) * x[xoff + i + 16];
            }
        }
        y[m] = scale * acc;
    }
}

/// CPU Q8_0 gemv: `y[n_embd] = scale * W[n_embd, n_ff] @ x[n_ff]`.
pub fn q8_0_gemv_expert(
    down_bytes: &[u8],
    x: &[f32],
    y: &mut [f32],
    n_ff_exp: usize,
    n_embd: usize,
    scale: f32,
) {
    assert_eq!(x.len(), n_ff_exp);
    assert_eq!(y.len(), n_embd);
    let expected = down_expert_bytes(n_ff_exp, n_embd, 34) as usize;
    assert_eq!(down_bytes.len(), expected);
    let blocks_per_row = n_ff_exp / 32;
    let row_bytes = blocks_per_row * 34;
    for m in 0..n_embd {
        let row = &down_bytes[m * row_bytes..(m + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let base = b * 34;
            let d = f16_to_f32(u16::from_le_bytes([row[base], row[base + 1]]));
            let qs = &row[base + 2..base + 34];
            let xoff = b * 32;
            for i in 0..32 {
                acc += (qs[i] as i8 as f32) * d * x[xoff + i];
            }
        }
        y[m] = scale * acc;
    }
}

/// Full CPU expert FFN (for Metal cross-check only).
pub fn expert_ffn_cpu(
    gate_up_exps: &BufferView,
    down_exps: &BufferView,
    x: &[f32],
    expert: usize,
    n_embd: usize,
    n_ff_exp: usize,
    scale: f32,
    swap_gate_up: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let gate_b = expert_gate_up_bytes(gate_up_exps, expert, n_embd, n_ff_exp, swap_gate_up);
    let up_b = expert_gate_up_bytes(gate_up_exps, expert, n_embd, n_ff_exp, !swap_gate_up);
    let down_b = expert_down_bytes(down_exps, expert, n_ff_exp, n_embd);
    let mut gate = vec![0.0f32; n_ff_exp];
    let mut up = vec![0.0f32; n_ff_exp];
    let mut gelu = vec![0.0f32; n_ff_exp];
    let mut down = vec![0.0f32; n_embd];
    q4_k_gemv(gate_b, x, &mut gate, n_ff_exp, n_embd);
    q4_k_gemv(up_b, x, &mut up, n_ff_exp, n_embd);
    gelu_mul_cpu(&gate, &up, &mut gelu);
    match down_exps.format {
        crate::gpu::weight_fmt::Q5_1 => {
            q5_1_gemv_expert(down_b, &gelu, &mut down, n_ff_exp, n_embd, scale)
        }
        crate::gpu::weight_fmt::Q8_0 => {
            q8_0_gemv_expert(down_b, &gelu, &mut down, n_ff_exp, n_embd, scale)
        }
        other => panic!("unsupported MoE down format {other}"),
    }
    (gate, up, gelu, down)
}

pub fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Optional on-disk source for expert miss fills (`pread` + macOS `F_RDADVISE`).
pub struct ExpertIoSource {
    /// One fd per parallel fill worker (avoids contention on a single file table).
    pub files: Vec<std::sync::Arc<std::fs::File>>,
    pub gate_up_file_off: u64,
    pub down_file_off: u64,
}

/// Turbo-fieldfare-style fixed slot cache for routed experts (LFU eviction).
///
/// Keeps `slot_count` hot copies of (gate∥up, down) expert blobs in Metal shared
/// buffers so decode does not thrash the full ~14GB mmap.
pub struct ExpertSlotCache {
    slot_count: usize,
    n_embd: usize,
    n_ff: usize,
    down_format: u8,
    gate_up_stride: usize,
    down_stride: usize,
    gate_up_slots: Vec<metal::Buffer>,
    down_slots: Vec<metal::Buffer>,
    io: Option<ExpertIoSource>,
    state: std::sync::Mutex<ExpertCacheState>,
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
}

/// Process-wide cache counters (per-layer atomics bounce in logs).
static CACHE_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CACHE_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

struct ExpertCacheState {
    slot_expert: Vec<i32>,
    slot_last_use: Vec<u64>,
    expert_use_count: Vec<u32>,
    use_clock: u64,
}

/// Approx reclaimable RAM (free + purgeable pages). Conservative fallback 4 GiB.
fn approx_available_ram_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(1) as u64;
        let read_sysctl_u64 = |name: &std::ffi::CStr| -> Option<u64> {
            let mut val: u64 = 0;
            let mut len = std::mem::size_of::<u64>();
            let rc = unsafe {
                libc::sysctlbyname(
                    name.as_ptr(),
                    &mut val as *mut _ as *mut libc::c_void,
                    &mut len,
                    std::ptr::null_mut(),
                    0,
                )
            };
            if rc == 0 {
                Some(val)
            } else {
                None
            }
        };
        // vm.page_free_count is uint32 on some Darwin versions — try both widths.
        let free_pages = read_sysctl_u64(c"vm.page_free_count").or_else(|| {
            let mut val: u32 = 0;
            let mut len = std::mem::size_of::<u32>();
            let rc = unsafe {
                libc::sysctlbyname(
                    c"vm.page_free_count".as_ptr(),
                    &mut val as *mut _ as *mut libc::c_void,
                    &mut len,
                    std::ptr::null_mut(),
                    0,
                )
            };
            if rc == 0 {
                Some(val as u64)
            } else {
                None
            }
        });
        let purge_pages = read_sysctl_u64(c"vm.page_purgeable_count").or_else(|| {
            let mut val: u32 = 0;
            let mut len = std::mem::size_of::<u32>();
            let rc = unsafe {
                libc::sysctlbyname(
                    c"vm.page_purgeable_count".as_ptr(),
                    &mut val as *mut _ as *mut libc::c_void,
                    &mut len,
                    std::ptr::null_mut(),
                    0,
                )
            };
            if rc == 0 {
                Some(val as u64)
            } else {
                None
            }
        });
        if let Some(free) = free_pages {
            return (free + purge_pages.unwrap_or(0)) * page_size;
        }
    }
    4u64 << 30
}

/// Default slots/layer from free RAM (ds4-style budget). Override with `MOE_EXPERT_SLOTS`.
///
/// A4B: ~110 MB/slot across 30 layers. On a busy 16 GB M1 Pro, 32 slots (~3.3 GB)
/// thrash; 16 matches TurboFieldfare and sustains higher tok/s.
pub fn expert_cache_slot_count(n_expert_used: usize) -> usize {
    if std::env::var("MOE_EXPERT_CACHE").as_deref() == Ok("0") {
        return 0;
    }
    if let Some(raw) = std::env::var("MOE_EXPERT_SLOTS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        return raw.max(n_expert_used).max(1);
    }
    const BYTES_PER_SLOT: u64 = 110 * 1024 * 1024;
    const HEADROOM: u64 = 2500 * 1024 * 1024;
    let avail = approx_available_ram_bytes();
    let budget = avail.saturating_sub(HEADROOM);
    let by_budget = (budget / BYTES_PER_SLOT).clamp(8, 32) as usize;
    // Snap to TF/ds4-friendly sizes so we do not oscillate mid-band.
    let snapped = if by_budget >= 28 {
        32
    } else if by_budget >= 20 {
        24
    } else {
        16
    };
    snapped.max(n_expert_used).max(1)
}

/// Result of LFU slot assignment before miss I/O.
pub struct ExpertCachePlan {
    pub experts: Vec<usize>,
    pub assigned_slots: Vec<usize>,
    pub miss_indices: Vec<usize>,
}

/// Cheap wall-clock MoE phase timers (`MOE_PROFILE=1`).
pub struct MoeProfile {
    pub router_read_ms: f64,
    pub plan_ms: f64,
    pub fill_ms: f64,
    pub shared_wait_ms: f64,
    pub expert_gpu_ms: f64,
    pub calls: u64,
    pub miss_experts: u64,
    pub hit_experts: u64,
    pub fill_bytes: u64,
}

impl MoeProfile {
    pub fn enabled() -> bool {
        std::env::var("MOE_PROFILE").as_deref() == Ok("1")
    }

    fn global() -> &'static std::sync::Mutex<MoeProfile> {
        static P: std::sync::OnceLock<std::sync::Mutex<MoeProfile>> = std::sync::OnceLock::new();
        P.get_or_init(|| {
            std::sync::Mutex::new(MoeProfile {
                router_read_ms: 0.0,
                plan_ms: 0.0,
                fill_ms: 0.0,
                shared_wait_ms: 0.0,
                expert_gpu_ms: 0.0,
                calls: 0,
                miss_experts: 0,
                hit_experts: 0,
                fill_bytes: 0,
            })
        })
    }

    pub fn add(
        router_read_ms: f64,
        plan_ms: f64,
        fill_ms: f64,
        shared_wait_ms: f64,
        expert_gpu_ms: f64,
        hits: u64,
        misses: u64,
        fill_bytes: u64,
    ) {
        if !Self::enabled() {
            return;
        }
        let mut p = Self::global().lock().unwrap();
        p.router_read_ms += router_read_ms;
        p.plan_ms += plan_ms;
        p.fill_ms += fill_ms;
        p.shared_wait_ms += shared_wait_ms;
        p.expert_gpu_ms += expert_gpu_ms;
        p.calls += 1;
        p.hit_experts += hits;
        p.miss_experts += misses;
        p.fill_bytes += fill_bytes;
        if p.calls == 1 || p.calls % 64 == 0 {
            let n = p.calls as f64;
            eprintln!(
                "  [moe-prof] n={} avg_ms router={:.2} plan={:.2} fill={:.2} shared_wait={:.2} expert={:.2} hit/miss={}/{} fill_MB/call={:.2}",
                p.calls,
                p.router_read_ms / n,
                p.plan_ms / n,
                p.fill_ms / n,
                p.shared_wait_ms / n,
                p.expert_gpu_ms / n,
                p.hit_experts,
                p.miss_experts,
                (p.fill_bytes as f64 / n) / (1024.0 * 1024.0),
            );
        }
    }
}

impl ExpertSlotCache {
    pub fn new(
        device: &metal::Device,
        n_expert: usize,
        n_embd: usize,
        n_ff: usize,
        down_format: u8,
        slot_count: usize,
        io: Option<ExpertIoSource>,
    ) -> Self {
        use metal::MTLResourceOptions;
        let gate_up_stride = gate_up_expert_bytes(n_embd, n_ff) as usize;
        let down_stride = down_expert_bytes(n_ff, n_embd, down_block_bytes(down_format)) as usize;
        let mut gate_up_slots = Vec::with_capacity(slot_count);
        let mut down_slots = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            gate_up_slots.push(device.new_buffer(
                gate_up_stride as u64,
                MTLResourceOptions::StorageModeShared,
            ));
            down_slots.push(device.new_buffer(
                down_stride as u64,
                MTLResourceOptions::StorageModeShared,
            ));
        }
        Self {
            slot_count,
            n_embd,
            n_ff,
            down_format,
            gate_up_stride,
            down_stride,
            gate_up_slots,
            down_slots,
            io,
            state: std::sync::Mutex::new(ExpertCacheState {
                slot_expert: vec![-1; slot_count],
                slot_last_use: vec![0; slot_count],
                expert_use_count: vec![0; n_expert.max(1)],
                use_clock: 0,
            }),
            hits: std::sync::atomic::AtomicU64::new(0),
            misses: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn bytes_per_layer(&self) -> u64 {
        (self.slot_count * (self.gate_up_stride + self.down_stride)) as u64
    }

    pub fn bytes_per_expert(&self) -> u64 {
        (self.gate_up_stride + self.down_stride) as u64
    }

    pub fn slot_count(&self) -> usize {
        self.slot_count
    }

    pub fn global_stats() -> (u64, u64) {
        (
            CACHE_HITS.load(std::sync::atomic::Ordering::Relaxed),
            CACHE_MISSES.load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Reserve slots (LFU). Miss slots are marked empty until `fill_misses`.
    pub fn plan(&self, experts: &[usize]) -> ExpertCachePlan {
        assert!(
            experts.len() <= self.slot_count,
            "need ≥{} slots for top-{}, have {}",
            experts.len(),
            experts.len(),
            self.slot_count
        );
        let mut st = self.state.lock().unwrap();
        st.use_clock = st.use_clock.wrapping_add(1);
        let clock = st.use_clock;
        // ds4-style decay: keep LFU from pinning early-prompt experts forever.
        if clock % 16 == 0 {
            for c in st.expert_use_count.iter_mut() {
                *c >>= 1;
            }
        }
        let mut assigned = vec![usize::MAX; experts.len()];
        let mut reserved = vec![false; self.slot_count];

        for (i, &e) in experts.iter().enumerate() {
            for s in 0..self.slot_count {
                if !reserved[s] && st.slot_expert[s] == e as i32 {
                    assigned[i] = s;
                    reserved[s] = true;
                    break;
                }
            }
        }

        let miss_indices: Vec<usize> = assigned
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| if s == usize::MAX { Some(i) } else { None })
            .collect();

        let mut evictable: Vec<usize> = (0..self.slot_count).filter(|&s| !reserved[s]).collect();
        evictable.sort_by(|&a, &b| {
            let ae = st.slot_expert[a];
            let be = st.slot_expert[b];
            if ae < 0 || be < 0 {
                return ae.cmp(&be);
            }
            let ac = st.expert_use_count.get(ae as usize).copied().unwrap_or(0);
            let bc = st.expert_use_count.get(be as usize).copied().unwrap_or(0);
            ac.cmp(&bc)
                .then_with(|| st.slot_last_use[a].cmp(&st.slot_last_use[b]))
        });
        assert!(
            miss_indices.len() <= evictable.len(),
            "expert cache cannot place {} misses into {} free slots",
            miss_indices.len(),
            evictable.len()
        );

        for &e in experts {
            if e < st.expert_use_count.len() {
                st.expert_use_count[e] = st.expert_use_count[e].wrapping_add(1);
            }
        }
        for &s in &assigned {
            if s != usize::MAX {
                st.slot_last_use[s] = clock;
            }
        }
        for (off, &idx) in miss_indices.iter().enumerate() {
            let slot = evictable[off];
            assigned[idx] = slot;
            reserved[slot] = true;
            st.slot_expert[slot] = -1;
            st.slot_last_use[slot] = clock;
        }

        let hits = experts.len() - miss_indices.len();
        self.hits
            .fetch_add(hits as u64, std::sync::atomic::Ordering::Relaxed);
        self.misses.fetch_add(
            miss_indices.len() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        CACHE_HITS.fetch_add(hits as u64, std::sync::atomic::Ordering::Relaxed);
        CACHE_MISSES.fetch_add(
            miss_indices.len() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        if std::env::var("MOE_CACHE_STATS").as_deref() == Ok("1") {
            static CALLS: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let n = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 5 || n % 64 == 0 {
                let (h, m) = Self::global_stats();
                let total = h + m;
                let rate = if total > 0 {
                    100.0 * h as f64 / total as f64
                } else {
                    0.0
                };
                eprintln!(
                    "  [moe-cache] call={} hits={} misses={} hit_rate={:.1}% (this: {} hit / {} miss)",
                    n,
                    h,
                    m,
                    rate,
                    hits,
                    miss_indices.len()
                );
            }
        }

        ExpertCachePlan {
            experts: experts.to_vec(),
            assigned_slots: assigned,
            miss_indices,
        }
    }

    /// Parallel mmap→slot copies for plan misses (safe: distinct slots).
    pub fn fill_misses(
        &self,
        plan: &ExpertCachePlan,
        gate_up_src: &BufferView,
        down_src: &BufferView,
    ) {
        self.fill_misses_scoped(plan, gate_up_src, down_src, || {});
    }

    /// Spawn miss fills, run `between` (e.g. encode cache hits), then join fills
    /// and mark slots occupied. Issues a batch `F_RDADVISE` for all misses first.
    pub fn fill_misses_scoped<R>(
        &self,
        plan: &ExpertCachePlan,
        gate_up_src: &BufferView,
        down_src: &BufferView,
        between: impl FnOnce() -> R,
    ) -> R {
        if plan.miss_indices.is_empty() {
            return between();
        }
        #[cfg(target_os = "macos")]
        if let Some(io) = &self.io {
            use std::os::fd::AsRawFd;
            let fd = io.files[0].as_raw_fd();
            for &idx in &plan.miss_indices {
                let expert = plan.experts[idx];
                let g_off = io.gate_up_file_off + (expert * self.gate_up_stride) as u64;
                let d_off = io.down_file_off + (expert * self.down_stride) as u64;
                let mut ra_g = libc::radvisory {
                    ra_offset: g_off as i64,
                    ra_count: self.gate_up_stride as i32,
                };
                let mut ra_d = libc::radvisory {
                    ra_offset: d_off as i64,
                    ra_count: self.down_stride as i32,
                };
                let _ = unsafe { libc::fcntl(fd, libc::F_RDADVISE, &mut ra_g) };
                let _ = unsafe { libc::fcntl(fd, libc::F_RDADVISE, &mut ra_d) };
            }
        }
        let result = std::thread::scope(|scope| {
            for &idx in &plan.miss_indices {
                let expert = plan.experts[idx];
                let slot = plan.assigned_slots[idx];
                scope.spawn(move || {
                    self.fill_slot(slot, expert, gate_up_src, down_src);
                });
            }
            between()
        });
        let mut st = self.state.lock().unwrap();
        for &idx in &plan.miss_indices {
            st.slot_expert[plan.assigned_slots[idx]] = plan.experts[idx] as i32;
        }
        result
    }

    /// Indices in `plan.experts` that were cache hits at plan time.
    pub fn hit_indices(plan: &ExpertCachePlan) -> Vec<usize> {
        let miss: std::collections::HashSet<usize> =
            plan.miss_indices.iter().copied().collect();
        (0..plan.experts.len())
            .filter(|i| !miss.contains(i))
            .collect()
    }

    /// Gate/up/down views for planned slots. `swap_gate_up=false` → gate first.
    pub fn views(
        &self,
        plan: &ExpertCachePlan,
        swap_gate_up: bool,
    ) -> Vec<(BufferView, BufferView, BufferView)> {
        let gate_is_up = swap_gate_up;
        let half_bytes = self.gate_up_stride / 2;
        let gate_off = gate_up_half_offset(self.n_embd, self.n_ff, gate_is_up);
        let up_off = gate_up_half_offset(self.n_embd, self.n_ff, !gate_is_up);
        plan.assigned_slots
            .iter()
            .map(|&slot| {
                let gate = BufferView {
                    buffer: self.gate_up_slots[slot].clone(),
                    offset: gate_off,
                    length: half_bytes as u64,
                    format: crate::gpu::weight_fmt::Q4_K,
                };
                let up = BufferView {
                    buffer: self.gate_up_slots[slot].clone(),
                    offset: up_off,
                    length: half_bytes as u64,
                    format: crate::gpu::weight_fmt::Q4_K,
                };
                let down = BufferView {
                    buffer: self.down_slots[slot].clone(),
                    offset: 0,
                    length: self.down_stride as u64,
                    format: self.down_format,
                };
                (gate, up, down)
            })
            .collect()
    }

    /// Plan + fill + views (sequential path / tests).
    pub fn ensure(
        &self,
        experts: &[usize],
        gate_up_src: &BufferView,
        down_src: &BufferView,
        swap_gate_up: bool,
    ) -> Vec<(BufferView, BufferView, BufferView)> {
        let plan = self.plan(experts);
        self.fill_misses(&plan, gate_up_src, down_src);
        self.views(&plan, swap_gate_up)
    }

    fn fill_slot(
        &self,
        slot: usize,
        expert: usize,
        gate_up_src: &BufferView,
        down_src: &BufferView,
    ) {
        let g_base = expert * self.gate_up_stride;
        let d_base = expert * self.down_stride;
        unsafe {
            let g_dst = std::slice::from_raw_parts_mut(
                self.gate_up_slots[slot].contents() as *mut u8,
                self.gate_up_stride,
            );
            let d_dst = std::slice::from_raw_parts_mut(
                self.down_slots[slot].contents() as *mut u8,
                self.down_stride,
            );
            if let Some(io) = &self.io {
                use std::os::unix::fs::FileExt;
                let file = &io.files[slot % io.files.len()];
                let g_off = io.gate_up_file_off + g_base as u64;
                let d_off = io.down_file_off + d_base as u64;
                file.read_exact_at(g_dst, g_off)
                    .unwrap_or_else(|e| panic!("expert gate_up pread: {e}"));
                file.read_exact_at(d_dst, d_off)
                    .unwrap_or_else(|e| panic!("expert down pread: {e}"));
            } else {
                let g_src = &gate_up_src.as_bytes()[g_base..g_base + self.gate_up_stride];
                let d_src = &down_src.as_bytes()[d_base..d_base + self.down_stride];
                g_dst.copy_from_slice(g_src);
                d_dst.copy_from_slice(d_src);
            }
        }
    }
}

#[allow(dead_code)]
pub fn assert_q5_1_supported() {
    let _ = ggml_type::Q5_1;
    let _ = gguf::ggml_type_name(ggml_type::Q5_1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_topk_renorm_sums_to_one() {
        let logits = vec![1.0f32, 2.0, 0.5, 3.0, -1.0];
        let (idx, w) = softmax_topk_renorm(&logits, 3);
        assert_eq!(idx.len(), 3);
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        assert_eq!(idx[0], 3);
    }
}
