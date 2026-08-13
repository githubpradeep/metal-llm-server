//! GPU-resident expert cache: Shared MTLBuffers filled by explicit `pread`.
//!
//! ds4 SSD policy: dense weights stay mmap; routed experts live in reusable
//! Shared buffers (unified-memory DRAM). `pread` into buffer contents — not
//! mmap-no-copy (GPU page faults) and not mmap-memcpy on cold pages (fault storm).

use std::collections::{HashMap, HashSet};
use std::os::unix::io::AsRawFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

use metal::Buffer;

use super::metal_ctx::Dsv4Metal;
use super::ssd::{ExpertKey, ExpertSsdCache};

/// One `pread` into a GPU Shared buffer (ds4 stream-expert task).
#[derive(Clone, Copy)]
struct PreadTask {
    dst: usize,
    len: usize,
    offset: u64,
    fd: i32,
}

struct PreadPoolState {
    tasks: Vec<PreadTask>,
    next: usize,
    remaining: usize,
    generation: u64,
    stop: bool,
}

struct PreadPool {
    n_threads: usize,
    mu: Mutex<PreadPoolState>,
    start: Condvar,
    done: Condvar,
}

static PREAD_POOL: OnceLock<PreadPool> = OnceLock::new();
static PREAD_POOL_STARTED: AtomicBool = AtomicBool::new(false);

fn pread_pool_enabled() -> bool {
    std::env::var("DSV4_PREAD_POOL").ok().as_deref() != Some("0")
}

fn pread_pool_threads() -> usize {
    let n = std::env::var("DSV4_PREAD_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(9usize);
    n.clamp(1, 18)
}

/// ds4 streaming cache: evict lowest `route_hotness`, then oldest `last_used`.
/// `DSV4_EXPERT_LFU=0` restores the previous sticky-recency victim policy.
fn expert_lfu_enabled() -> bool {
    std::env::var("DSV4_EXPERT_LFU").ok().as_deref() != Some("0")
}

const HOTNESS_DECAY_TOKENS: u64 = 16;

fn pread_exact(fd: i32, dst: &mut [u8], offset: u64) {
    let mut pos = 0usize;
    while pos < dst.len() {
        let n = unsafe {
            libc::pread(
                fd,
                dst[pos..].as_mut_ptr() as *mut libc::c_void,
                dst.len() - pos,
                (offset as i64).saturating_add(pos as i64),
            )
        };
        if n <= 0 {
            panic!(
                "expert pread failed fd={fd} off={offset}+{pos} len={} n={n}",
                dst.len()
            );
        }
        pos += n as usize;
    }
}

fn run_pread_task(t: PreadTask) {
    let dst = unsafe { std::slice::from_raw_parts_mut(t.dst as *mut u8, t.len) };
    pread_exact(t.fd, dst, t.offset);
}

fn pread_pool() -> &'static PreadPool {
    let n = pread_pool_threads();
    let pool = PREAD_POOL.get_or_init(|| PreadPool {
        n_threads: n,
        mu: Mutex::new(PreadPoolState {
            tasks: Vec::new(),
            next: 0,
            remaining: 0,
            generation: 0,
            stop: false,
        }),
        start: Condvar::new(),
        done: Condvar::new(),
    });
    if !PREAD_POOL_STARTED.swap(true, Ordering::Relaxed) {
        for i in 0..pool.n_threads {
            std::thread::Builder::new()
                .name(format!("dsv4-pread-{i}"))
                .spawn(move || pread_pool_worker(i))
                .expect("pread pool worker");
        }
        println!(
            "  expert pread pool: {} threads (ds4-style, DSV4_PREAD_POOL=0 disables)",
            pool.n_threads
        );
    }
    pool
}

fn pread_pool_worker(_worker_index: usize) {
    let pool = PREAD_POOL.get().expect("pread pool");
    let mut seen = 0u64;
    loop {
        let mut g = pool.mu.lock().unwrap();
        while !g.stop && g.generation == seen {
            g = pool.start.wait(g).unwrap();
        }
        if g.stop {
            break;
        }
        seen = g.generation;
        loop {
            let idx = g.next;
            if idx >= g.tasks.len() {
                break;
            }
            g.next += 1;
            let task = g.tasks[idx];
            drop(g);
            run_pread_task(task);
            g = pool.mu.lock().unwrap();
        }
        if g.remaining > 0 {
            g.remaining -= 1;
            if g.remaining == 0 {
                pool.done.notify_one();
            }
        }
    }
}

fn pread_pool_begin(tasks: &[PreadTask]) -> bool {
    if tasks.is_empty() || !pread_pool_enabled() {
        return false;
    }
    if tasks.len() == 1 {
        run_pread_task(tasks[0]);
        return false;
    }
    let pool = pread_pool();
    let mut g = pool.mu.lock().unwrap();
    g.tasks.clear();
    g.tasks.extend_from_slice(tasks);
    g.next = 0;
    g.remaining = pool.n_threads;
    g.generation += 1;
    pool.start.notify_all();
    true
}

fn pread_pool_wait() {
    let pool = pread_pool();
    let mut g = pool.mu.lock().unwrap();
    while g.remaining != 0 {
        g = pool.done.wait(g).unwrap();
    }
}

struct GpuSlot {
    key: Option<ExpertKey>,
    gate: Option<Buffer>,
    up: Option<Buffer>,
    down: Option<Buffer>,
    last_used: u64,
}

pub struct ExpertGpuCache {
    slots: Vec<GpuSlot>,
    index: HashMap<ExpertKey, usize>,
    clock: u64,
    pub gate_bytes: usize,
    pub up_bytes: usize,
    pub down_bytes: usize,
    /// Contiguous Shared packs (slot `si` at `si * stride`) for GPU-map MoE fuse.
    pub gate_pack: Buffer,
    pub up_pack: Buffer,
    pub down_pack: Buffer,
    /// Per-layer expert-id → slot map (-1 = not resident). Length `n_expert`.
    /// Host mirror; upload to scratch before fused CB2.
    pub layer_slot_map: Vec<i32>,
    pub n_expert: usize,
    n_layer: usize,
    /// ds4 `route_hotness`: counts selected experts even on miss so LFU does
    /// not punish a repeatedly routed expert that was evicted before a 2nd hit.
    route_hotness: Vec<u32>,
    route_notes: u64,
    hotness_decay_notes: u64,
    pub uploads: u64,
    pub skips: u64,
    pub buffer_allocs: u64,
    pub buffer_reuses: u64,
    pub pread_bytes: u64,
}

impl ExpertGpuCache {
    pub fn new(
        metal: &Dsv4Metal,
        n_slots: usize,
        gate_bytes: usize,
        up_bytes: usize,
        down_bytes: usize,
        n_expert: usize,
        n_layer: usize,
    ) -> Self {
        let n_slots = n_slots.max(8);
        let n_layer = n_layer.max(1);
        let n_expert = n_expert.max(1);
        let mib = (gate_bytes + up_bytes + down_bytes) as f64 / (1024.0 * 1024.0);
        let lfu = expert_lfu_enabled();
        println!(
            "  GPU expert LRU: {n_slots} slots (Shared pread, ~{mib:.2} MiB/expert, {})",
            if lfu {
                "route-hotness LFU"
            } else {
                "sticky recency"
            }
        );
        let gate_pack = metal.buffer_zeros(n_slots * gate_bytes);
        let up_pack = metal.buffer_zeros(n_slots * up_bytes);
        let down_pack = metal.buffer_zeros(n_slots * down_bytes);
        Self {
            slots: (0..n_slots)
                .map(|_| GpuSlot {
                    key: None,
                    gate: None,
                    up: None,
                    down: None,
                    last_used: 0,
                })
                .collect(),
            index: HashMap::new(),
            clock: 0,
            gate_bytes,
            up_bytes,
            down_bytes,
            gate_pack,
            up_pack,
            down_pack,
            layer_slot_map: vec![-1i32; n_expert],
            n_expert,
            n_layer,
            route_hotness: vec![0u32; n_layer.saturating_mul(n_expert)],
            route_notes: 0,
            hotness_decay_notes: 0,
            uploads: 0,
            skips: 0,
            buffer_allocs: 0,
            buffer_reuses: 0,
            pread_bytes: 0,
        }
    }

    pub fn n_slots(&self) -> usize {
        self.slots.len()
    }

    pub fn filled_slots(&self) -> usize {
        self.slots.iter().filter(|s| s.key.is_some()).count()
    }

    /// Hit rate over pin classify/touch traffic (`skips / (skips + uploads)`).
    pub fn hit_rate(&self) -> f64 {
        let total = self.uploads.saturating_add(self.skips);
        if total == 0 {
            0.0
        } else {
            self.skips as f64 / total as f64
        }
    }

    /// After prefill: note occupancy, reset traffic counters for gen-only
    /// stats, and age out sticky so gen can replace prefill-only experts
    /// without fighting a hot window — resident keys remain valid hits.
    pub fn compact_or_note_prefill_done(&mut self) {
        let filled = self.filled_slots();
        let n = self.n_slots();
        println!(
            "  GPU expert cache after prefill: {filled}/{n} slots filled ({:.0}%, hit={:.1}%)",
            100.0 * filled as f64 / n.max(1) as f64,
            100.0 * self.hit_rate()
        );
        // Make all resident slots immediately evictable (not sticky).
        for s in &mut self.slots {
            s.last_used = 0;
        }
        self.clock = 0;
        self.uploads = 0;
        self.skips = 0;
        self.pread_bytes = 0;
    }

    fn hotness_of(&self, key: ExpertKey) -> u32 {
        let layer = key.layer as usize;
        let expert = key.expert as usize;
        if layer >= self.n_layer || expert >= self.n_expert {
            return 0;
        }
        self.route_hotness[layer * self.n_expert + expert]
    }

    /// Count a selected id even on miss (ds4 `note_selected_hotness`).
    fn note_route_keys(&mut self, keys: &[ExpertKey]) {
        if keys.is_empty() || !expert_lfu_enabled() {
            return;
        }
        self.route_notes = self.route_notes.saturating_add(1);
        let decay_every = HOTNESS_DECAY_TOKENS.saturating_mul(self.n_layer as u64);
        if decay_every > 0
            && self.route_notes.saturating_sub(self.hotness_decay_notes) >= decay_every
        {
            for h in &mut self.route_hotness {
                *h >>= 1;
            }
            self.hotness_decay_notes = self.route_notes;
        }
        for &key in keys {
            let layer = key.layer as usize;
            let expert = key.expert as usize;
            if layer >= self.n_layer || expert >= self.n_expert {
                continue;
            }
            let h = &mut self.route_hotness[layer * self.n_expert + expert];
            *h = h.saturating_add(1);
        }
    }

    /// Empty first; then ds4 LFU (lowest route-hotness, then oldest last_used).
    /// `DSV4_EXPERT_LFU=0`: previous two-phase sticky recency.
    fn lru_slot_excluding(&self, skip: &HashSet<usize>) -> usize {
        if !expert_lfu_enabled() {
            let sticky = (self.slots.len() / 2).min(800) as u64;
            let mut best_cold: Option<(usize, u64)> = None;
            let mut best_hot: Option<(usize, u64)> = None;
            for (i, s) in self.slots.iter().enumerate() {
                if skip.contains(&i) {
                    continue;
                }
                if s.key.is_none() {
                    return i;
                }
                let age = self.clock.saturating_sub(s.last_used);
                if age >= sticky {
                    if best_cold.map(|(_, t)| s.last_used < t).unwrap_or(true) {
                        best_cold = Some((i, s.last_used));
                    }
                } else if best_hot.map(|(_, t)| s.last_used < t).unwrap_or(true) {
                    best_hot = Some((i, s.last_used));
                }
            }
            return best_cold
                .or(best_hot)
                .map(|(i, _)| i)
                .expect("expert GPU cache exhausted");
        }
        let mut best: Option<(usize, u32, u64)> = None;
        for (i, s) in self.slots.iter().enumerate() {
            if skip.contains(&i) {
                continue;
            }
            if s.key.is_none() {
                return i;
            }
            let hot = self.hotness_of(s.key.unwrap());
            let used = s.last_used;
            let take = match best {
                None => true,
                Some((_, bh, bu)) => hot < bh || (hot == bh && used < bu),
            };
            if take {
                best = Some((i, hot, used));
            }
        }
        best.map(|(i, _, _)| i)
            .expect("expert GPU cache exhausted")
    }

    pub fn contains(&self, key: ExpertKey) -> bool {
        self.index.contains_key(&key)
    }

    /// Slot index without LRU touch (fuse-path lookup).
    pub fn slot_of(&self, key: ExpertKey) -> usize {
        *self.index.get(&key).expect("gpu expert missing")
    }

    pub fn touch(&mut self, key: ExpertKey) -> usize {
        let si = *self.index.get(&key).expect("gpu expert missing");
        self.clock += 1;
        self.slots[si].last_used = self.clock;
        self.skips += 1;
        si
    }

    /// Split keys into cache hits (touched) vs misses. `out_slots[i]` is set for hits.
    pub fn classify_hits(
        &mut self,
        keys: &[ExpertKey],
        out_slots: &mut [usize],
    ) -> (Vec<usize>, Vec<(usize, ExpertKey)>) {
        assert_eq!(keys.len(), out_slots.len());
        self.note_route_keys(keys);
        let mut hits = Vec::new();
        let mut misses = Vec::new();
        for (i, &key) in keys.iter().enumerate() {
            if self.contains(key) {
                out_slots[i] = self.touch(key);
                hits.push(i);
            } else {
                misses.push((i, key));
            }
        }
        (hits, misses)
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    fn fadvise_willneed(file: &std::fs::File, offset: u64, len: usize) {
        let fd = file.as_raw_fd();
        // Soft-fail: advisory only; ignore ENOSYS / EINVAL on exotic FS.
        let _ = unsafe {
            libc::posix_fadvise(
                fd,
                offset as libc::off_t,
                len as libc::off_t,
                libc::POSIX_FADV_WILLNEED,
            )
        };
    }

    /// Darwin has no `posix_fadvise`; `F_RDADVISE` is the file readahead hint.
    #[cfg(target_os = "macos")]
    fn fadvise_willneed(file: &std::fs::File, offset: u64, len: usize) {
        let fd = file.as_raw_fd();
        #[repr(C)]
        struct Radvisory {
            ra_offset: i64,
            ra_count: i32,
        }
        // Soft-fail if unavailable / unsupported.
        const F_RDADVISE: i32 = 44; // fcntl.h
        let adv = Radvisory {
            ra_offset: offset as i64,
            ra_count: len as i32,
        };
        let _ = unsafe { libc::fcntl(fd, F_RDADVISE, &adv as *const Radvisory) };
    }

    #[cfg(not(unix))]
    fn fadvise_willneed(_file: &std::fs::File, _offset: u64, _len: usize) {}

    /// Soft-fail `mlock` on Shared buffer contents (hot expert residency).
    fn try_mlock(ptr: *mut std::ffi::c_void, len: usize) {
        if len == 0 || ptr.is_null() {
            return;
        }
        let _ = unsafe { libc::mlock(ptr, len) };
    }

    /// Pread only the miss list into LRU slots; writes `out_slots[out_i]`.
    /// Reserve miss slots and `pread` them while `gpu` runs on the calling thread
    /// (ds4/gemma I/O-first: hide SSD latency behind encode+commit).
    /// Preads start as soon as each slot is reserved (overlap alloc of later
    /// misses + `gpu` encode/commit). `protect` slots are never evicted.
    ///
    /// `DSV4_SKIP_EXPERT_IO=1`: reserve/install keys + ensure_buffers only — no
    /// preads (all-hit GPU ceiling probe; output text will be wrong).
    pub fn pin_misses_with_gpu<R>(
        &mut self,
        metal: &Dsv4Metal,
        ssd: &ExpertSsdCache,
        misses: &[(usize, ExpertKey)],
        out_slots: &mut [usize],
        protect: &[usize],
        gpu: impl FnOnce() -> R,
    ) -> std::io::Result<R> {
        if misses.is_empty() {
            return Ok(gpu());
        }
        let skip_io = std::env::var("DSV4_SKIP_EXPERT_IO").ok().as_deref() == Some("1");
        let gate_bytes = self.gate_bytes;
        let up_bytes = self.up_bytes;
        let down_bytes = self.down_bytes;
        let file = ssd.file();
        let fd = file.as_raw_fd();
        let mut reserved_sis: HashSet<usize> = protect.iter().copied().collect();
        let mut reserved: Vec<(usize, ExpertKey, usize)> = Vec::with_capacity(misses.len());
        let mut tasks: Vec<PreadTask> = Vec::new();

        for &(out_i, key) in misses {
            if self.contains(key) {
                out_slots[out_i] = self.touch(key);
                continue;
            }
            let layout = ssd.layout(key).clone();
            assert_eq!(layout.gate_bytes, gate_bytes);
            assert_eq!(layout.up_bytes, up_bytes);
            assert_eq!(layout.down_bytes, down_bytes);
            let si = self.lru_slot_excluding(&reserved_sis);
            reserved_sis.insert(si);
            if let Some(old) = self.slots[si].key.take() {
                self.index.remove(&old);
            }
            self.ensure_buffers(metal, si);
            if !skip_io {
                let g_ptr = self.slots[si].gate.as_ref().unwrap().contents() as usize;
                let u_ptr = self.slots[si].up.as_ref().unwrap().contents() as usize;
                let d_ptr = self.slots[si].down.as_ref().unwrap().contents() as usize;
                Self::fadvise_willneed(file, layout.gate_offset, gate_bytes);
                Self::fadvise_willneed(file, layout.up_offset, up_bytes);
                Self::fadvise_willneed(file, layout.down_offset, down_bytes);
                tasks.push(PreadTask {
                    dst: g_ptr,
                    len: gate_bytes,
                    offset: layout.gate_offset,
                    fd,
                });
                tasks.push(PreadTask {
                    dst: u_ptr,
                    len: up_bytes,
                    offset: layout.up_offset,
                    fd,
                });
                tasks.push(PreadTask {
                    dst: d_ptr,
                    len: down_bytes,
                    offset: layout.down_offset,
                    fd,
                });
            }
            reserved.push((out_i, key, si));
        }

        // I/O-first: persistent pool (ds4) overlaps gpu() on this thread.
        let result = if skip_io || tasks.is_empty() {
            gpu()
        } else if pread_pool_begin(&tasks) {
            let r = gpu();
            pread_pool_wait();
            r
        } else if !pread_pool_enabled() {
            std::thread::scope(|scope| {
                for t in &tasks {
                    let t = *t;
                    scope.spawn(move || run_pread_task(t));
                }
                gpu()
            })
        } else {
            // Single-task inline already done in begin; just run gpu.
            gpu()
        };

        for (out_i, key, si) in reserved {
            self.slots[si].key = Some(key);
            self.clock += 1;
            self.slots[si].last_used = self.clock;
            self.index.insert(key, si);
            self.uploads += 1;
            if !skip_io {
                self.pread_bytes += (gate_bytes + up_bytes + down_bytes) as u64;
            }
            out_slots[out_i] = si;
        }
        Ok(result)
    }

    pub fn pin_misses_from_ssd(
        &mut self,
        metal: &Dsv4Metal,
        ssd: &ExpertSsdCache,
        misses: &[(usize, ExpertKey)],
        out_slots: &mut [usize],
    ) -> std::io::Result<()> {
        self.pin_misses_with_gpu(metal, ssd, misses, out_slots, &[], || ())
    }

    /// Pin a key batch while `gpu` runs (hash-layer CB1 overlap).
    /// `protect` slot indices are never chosen for eviction.
    pub fn pin_batch_with_gpu<R>(
        &mut self,
        metal: &Dsv4Metal,
        ssd: &ExpertSsdCache,
        keys: &[ExpertKey],
        protect: &[usize],
        gpu: impl FnOnce() -> R,
    ) -> std::io::Result<(R, Vec<usize>)> {
        self.note_route_keys(keys);
        let mut out = vec![0usize; keys.len()];
        let mut misses = Vec::new();
        for (i, &key) in keys.iter().enumerate() {
            if self.contains(key) {
                out[i] = self.touch(key);
            } else {
                misses.push((i, key));
            }
        }
        let r = self.pin_misses_with_gpu(metal, ssd, &misses, &mut out, protect, gpu)?;
        Ok((r, out))
    }

    fn ensure_buffers(&mut self, metal: &Dsv4Metal, si: usize) {
        if self.slots[si].gate.is_some() {
            self.buffer_reuses += 1;
            return;
        }
        // Alias pack slices via no-copy Shared buffers (16-byte aligned strides).
        let g_ptr = unsafe {
            (self.gate_pack.contents() as *mut u8).add(si * self.gate_bytes)
        };
        let u_ptr = unsafe {
            (self.up_pack.contents() as *mut u8).add(si * self.up_bytes)
        };
        let d_ptr = unsafe {
            (self.down_pack.contents() as *mut u8).add(si * self.down_bytes)
        };
        self.slots[si].gate = Some(metal.buffer_from_slice_no_copy(unsafe {
            std::slice::from_raw_parts(g_ptr, self.gate_bytes)
        }));
        self.slots[si].up = Some(metal.buffer_from_slice_no_copy(unsafe {
            std::slice::from_raw_parts(u_ptr, self.up_bytes)
        }));
        self.slots[si].down = Some(metal.buffer_from_slice_no_copy(unsafe {
            std::slice::from_raw_parts(d_ptr, self.down_bytes)
        }));
        let _ = metal;
        self.buffer_allocs += 1;
        // Wire once per slot. Reuse fills skip mlock (same Shared pages).
        // `DSV4_EXPERT_MLOCK=0` disables — measured 0.48 tok/s from GPU paging.
        if std::env::var("DSV4_EXPERT_MLOCK").ok().as_deref() != Some("0") {
            Self::try_mlock(g_ptr as *mut std::ffi::c_void, self.gate_bytes);
            Self::try_mlock(u_ptr as *mut std::ffi::c_void, self.up_bytes);
            Self::try_mlock(d_ptr as *mut std::ffi::c_void, self.down_bytes);
        }
    }

    /// Fill `out[expert] = slot` for residents of `layer` (-1 if absent).
    pub fn fill_slot_map_for_layer(&self, layer: u16, out: &mut [i32]) {
        assert!(out.len() >= self.n_expert);
        out[..self.n_expert].fill(-1);
        for (&key, &si) in &self.index {
            if key.layer == layer && (key.expert as usize) < self.n_expert {
                out[key.expert as usize] = si as i32;
            }
        }
    }

    pub fn pin_from_ssd(
        &mut self,
        metal: &Dsv4Metal,
        ssd: &ExpertSsdCache,
        key: ExpertKey,
    ) -> std::io::Result<usize> {
        let out = self.pin_batch_from_ssd(metal, ssd, &[key])?;
        Ok(out[0])
    }

    pub fn pin_batch_from_ssd(
        &mut self,
        metal: &Dsv4Metal,
        ssd: &ExpertSsdCache,
        keys: &[ExpertKey],
    ) -> std::io::Result<Vec<usize>> {
        Ok(self.pin_batch_with_gpu(metal, ssd, keys, &[], || ())?.1)
    }

    pub fn gate(&self, si: usize) -> &Buffer {
        self.slots[si].gate.as_ref().expect("empty gpu gate")
    }
    pub fn up(&self, si: usize) -> &Buffer {
        self.slots[si].up.as_ref().expect("empty gpu up")
    }
    pub fn down(&self, si: usize) -> &Buffer {
        self.slots[si].down.as_ref().expect("empty gpu down")
    }
}
