//! GPU-resident expert cache: Shared MTLBuffers filled from the GGUF.
//!
//! ds4 SSD policy: dense weights stay mmap; routed experts live in reusable
//! Shared buffers (unified-memory DRAM). Default fill is memcpy from the
//! process mmap (warm page cache after prefill). `DSV4_EXPERT_MMAP=0` falls
//! back to `pread` into buffer contents.

use std::collections::{HashMap, HashSet};
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;

use metal::Buffer;

use super::metal_ctx::Dsv4Metal;
use super::ssd::{ExpertBlobLayout, ExpertKey, ExpertSsdCache};

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
    ) -> Self {
        let n_slots = n_slots.max(8);
        let mib = (gate_bytes + up_bytes + down_bytes) as f64 / (1024.0 * 1024.0);
        println!(
            "  GPU expert LRU: {n_slots} slots (Shared pread, ~{mib:.2} MiB/expert)"
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

    /// Two-phase sticky LRU: empty first; then oldest outside the sticky window
    /// (`min(n_slots/2, 800)` ticks ≈ ~3 tokens at 258 touches/tok). Only fall
    /// back to sticky slots if every candidate was touched recently.
    fn lru_slot_excluding(&self, skip: &HashSet<usize>) -> usize {
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
        best_cold
            .or(best_hot)
            .map(|(i, _)| i)
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
    ///
    /// Opt-in `DSV4_EXPERT_STAGE=1`: pread into heap staging, then memcpy into MTL
    /// Shared after `gpu` returns (UM-fight experiment; default off was a wash).
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
        // Default off: heap staging added a memcpy without hiding UM contention.
        let stage = std::env::var("DSV4_EXPERT_STAGE").ok().as_deref() == Some("1");
        // Default on: memcpy from process mmap (warm after prefill).
        let use_mmap = std::env::var("DSV4_EXPERT_MMAP").ok().as_deref() != Some("0");
        let gate_bytes = self.gate_bytes;
        let up_bytes = self.up_bytes;
        let down_bytes = self.down_bytes;
        let file = ssd.file();
        let mut reserved_sis: HashSet<usize> = protect.iter().copied().collect();
        let mut reserved: Vec<(usize, ExpertKey, usize)> = Vec::with_capacity(misses.len());
        let mut staged: Vec<(usize, Box<[u8]>, Box<[u8]>, Box<[u8]>)> =
            Vec::with_capacity(misses.len());
        let mut stage_jobs: Vec<(usize, ExpertKey, ExpertBlobLayout)> =
            Vec::with_capacity(misses.len());
        let mut direct_jobs: Vec<(usize, ExpertKey, ExpertBlobLayout)> =
            Vec::with_capacity(misses.len());

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
                if stage {
                    let g = vec![0u8; gate_bytes].into_boxed_slice();
                    let u = vec![0u8; up_bytes].into_boxed_slice();
                    let d = vec![0u8; down_bytes].into_boxed_slice();
                    staged.push((si, g, u, d));
                    stage_jobs.push((staged.len() - 1, key, layout));
                } else {
                    direct_jobs.push((si, key, layout));
                }
            }
            reserved.push((out_i, key, si));
        }

        let stage_ptrs: Vec<(usize, usize, usize, ExpertKey, ExpertBlobLayout)> = stage_jobs
            .iter()
            .map(|&(idx, key, ref layout)| {
                (
                    staged[idx].1.as_mut_ptr() as usize,
                    staged[idx].2.as_mut_ptr() as usize,
                    staged[idx].3.as_mut_ptr() as usize,
                    key,
                    layout.clone(),
                )
            })
            .collect();

        // I/O-first: spawn fills; gpu() overlaps on this thread.
        let result = std::thread::scope(|scope| {
            if !skip_io && stage {
                for &(g_ptr, u_ptr, d_ptr, key, ref layout) in &stage_ptrs {
                    let layout_g = layout.clone();
                    scope.spawn(move || {
                        if use_mmap {
                            let (gs, us, ds) = ssd.mmap_bytes(key);
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    gs.as_ptr(),
                                    g_ptr as *mut u8,
                                    gate_bytes,
                                );
                                std::ptr::copy_nonoverlapping(
                                    us.as_ptr(),
                                    u_ptr as *mut u8,
                                    up_bytes,
                                );
                                std::ptr::copy_nonoverlapping(
                                    ds.as_ptr(),
                                    d_ptr as *mut u8,
                                    down_bytes,
                                );
                            }
                        } else {
                            Self::fadvise_willneed(file, layout_g.gate_offset, gate_bytes);
                            let g = unsafe {
                                std::slice::from_raw_parts_mut(g_ptr as *mut u8, gate_bytes)
                            };
                            file.read_exact_at(g, layout_g.gate_offset)
                                .expect("gate pread");
                            Self::fadvise_willneed(file, layout.up_offset, up_bytes);
                            let u = unsafe {
                                std::slice::from_raw_parts_mut(u_ptr as *mut u8, up_bytes)
                            };
                            file.read_exact_at(u, layout.up_offset).expect("up pread");
                            Self::fadvise_willneed(file, layout.down_offset, down_bytes);
                            let d = unsafe {
                                std::slice::from_raw_parts_mut(d_ptr as *mut u8, down_bytes)
                            };
                            file.read_exact_at(d, layout.down_offset)
                                .expect("down pread");
                        }
                    });
                }
            } else if !skip_io {
                for (si, key, layout) in &direct_jobs {
                    let g_ptr = self.slots[*si].gate.as_ref().unwrap().contents() as usize;
                    let u_ptr = self.slots[*si].up.as_ref().unwrap().contents() as usize;
                    let d_ptr = self.slots[*si].down.as_ref().unwrap().contents() as usize;
                    let key = *key;
                    let layout = layout.clone();
                    if use_mmap {
                        scope.spawn(move || {
                            let (gs, us, ds) = ssd.mmap_bytes(key);
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    gs.as_ptr(),
                                    g_ptr as *mut u8,
                                    gate_bytes,
                                );
                                std::ptr::copy_nonoverlapping(
                                    us.as_ptr(),
                                    u_ptr as *mut u8,
                                    up_bytes,
                                );
                                std::ptr::copy_nonoverlapping(
                                    ds.as_ptr(),
                                    d_ptr as *mut u8,
                                    down_bytes,
                                );
                            }
                        });
                    } else {
                        let layout_g = layout.clone();
                        scope.spawn(move || {
                            Self::fadvise_willneed(file, layout_g.gate_offset, gate_bytes);
                            let g = unsafe {
                                std::slice::from_raw_parts_mut(g_ptr as *mut u8, gate_bytes)
                            };
                            file.read_exact_at(g, layout_g.gate_offset)
                                .expect("gate pread");
                        });
                        let layout_u = layout.clone();
                        scope.spawn(move || {
                            Self::fadvise_willneed(file, layout_u.up_offset, up_bytes);
                            let u = unsafe {
                                std::slice::from_raw_parts_mut(u_ptr as *mut u8, up_bytes)
                            };
                            file.read_exact_at(u, layout_u.up_offset).expect("up pread");
                        });
                        scope.spawn(move || {
                            Self::fadvise_willneed(file, layout.down_offset, down_bytes);
                            let d = unsafe {
                                std::slice::from_raw_parts_mut(d_ptr as *mut u8, down_bytes)
                            };
                            file.read_exact_at(d, layout.down_offset)
                                .expect("down pread");
                        });
                    }
                }
            }
            gpu()
        });

        if !skip_io && stage {
            for (si, g, u, d) in staged {
                let g_dst = unsafe {
                    std::slice::from_raw_parts_mut(
                        self.slots[si].gate.as_ref().unwrap().contents() as *mut u8,
                        gate_bytes,
                    )
                };
                let u_dst = unsafe {
                    std::slice::from_raw_parts_mut(
                        self.slots[si].up.as_ref().unwrap().contents() as *mut u8,
                        up_bytes,
                    )
                };
                let d_dst = unsafe {
                    std::slice::from_raw_parts_mut(
                        self.slots[si].down.as_ref().unwrap().contents() as *mut u8,
                        down_bytes,
                    )
                };
                g_dst.copy_from_slice(&g);
                u_dst.copy_from_slice(&u);
                d_dst.copy_from_slice(&d);
            }
        }

        for (out_i, key, si) in reserved {
            if !skip_io
                && std::env::var("DSV4_EXPERT_MLOCK").ok().as_deref() != Some("0")
            {
                Self::try_mlock(
                    self.slots[si].gate.as_ref().unwrap().contents(),
                    gate_bytes,
                );
                Self::try_mlock(self.slots[si].up.as_ref().unwrap().contents(), up_bytes);
                Self::try_mlock(
                    self.slots[si].down.as_ref().unwrap().contents(),
                    down_bytes,
                );
            }
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
