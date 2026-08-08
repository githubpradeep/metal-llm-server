//! SSD expert streaming for DeepSeek-V4-Flash routed experts.
//!
//! Experts are read from a mmap'd GGUF. The OS page cache is the host cache;
//! optional LRU slots keep hot experts touched (mlock-ish residency via Vec).

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use memmap2::Mmap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertKey {
    pub layer: u16,
    pub expert: u16,
}

#[derive(Debug, Clone)]
pub struct ExpertBlobLayout {
    pub gate_offset: u64,
    pub gate_bytes: usize,
    pub up_offset: u64,
    pub up_bytes: usize,
    pub down_offset: u64,
    pub down_bytes: usize,
}

#[derive(Debug)]
struct Slot {
    key: Option<ExpertKey>,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
    last_used: u64,
}

/// Host-side expert cache backed by an open GGUF file (+ mmap for CPU path).
pub struct ExpertSsdCache {
    path: PathBuf,
    /// Kept open for `pread` into GPU Shared buffers (ds4 SSD streaming).
    file: File,
    mmap: Mmap,
    layouts: HashMap<ExpertKey, ExpertBlobLayout>,
    slots: Vec<Slot>,
    clock: u64,
    pub hits: u64,
    pub misses: u64,
}

impl ExpertSsdCache {
    pub fn new(path: impl AsRef<Path>, n_slots: usize) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        // Safety: GGUF is immutable while we run.
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self {
            path,
            file,
            mmap,
            layouts: HashMap::new(),
            slots: (0..n_slots)
                .map(|_| Slot {
                    key: None,
                    gate: Vec::new(),
                    up: Vec::new(),
                    down: Vec::new(),
                    last_used: 0,
                })
                .collect(),
            clock: 0,
            hits: 0,
            misses: 0,
        })
    }

    pub fn register(&mut self, key: ExpertKey, layout: ExpertBlobLayout) {
        self.layouts.insert(key, layout);
    }

    pub fn n_slots(&self) -> usize {
        self.slots.len()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn file(&self) -> &File {
        &self.file
    }

    pub fn layout(&self, key: ExpertKey) -> &ExpertBlobLayout {
        self.layouts.get(&key).unwrap_or_else(|| {
            panic!(
                "no layout for layer {} expert {}",
                key.layer, key.expert
            )
        })
    }

    /// Zero-copy views into the mmap'd GGUF (page-cache backed).
    pub fn mmap_bytes(&self, key: ExpertKey) -> (&[u8], &[u8], &[u8]) {
        let l = self.layout(key);
        let m = &self.mmap[..];
        let g0 = l.gate_offset as usize;
        let u0 = l.up_offset as usize;
        let d0 = l.down_offset as usize;
        (
            &m[g0..g0 + l.gate_bytes],
            &m[u0..u0 + l.up_bytes],
            &m[d0..d0 + l.down_bytes],
        )
    }

    fn find_slot(&self, key: ExpertKey) -> Option<usize> {
        self.slots.iter().position(|s| s.key == Some(key))
    }

    fn lru_slot(&self) -> usize {
        let mut best = 0;
        let mut best_t = u64::MAX;
        for (i, s) in self.slots.iter().enumerate() {
            if s.key.is_none() {
                return i;
            }
            if s.last_used < best_t {
                best_t = s.last_used;
                best = i;
            }
        }
        best
    }

    /// Ensure experts are resident in the optional LRU; returns slot indices.
    pub fn pin(&mut self, keys: &[ExpertKey]) -> std::io::Result<Vec<usize>> {
        let mut out = Vec::with_capacity(keys.len());
        for &key in keys {
            if let Some(i) = self.find_slot(key) {
                self.clock += 1;
                self.slots[i].last_used = self.clock;
                self.hits += 1;
                out.push(i);
                continue;
            }
            self.misses += 1;
            let layout = self.layout(key).clone();
            let g0 = layout.gate_offset as usize;
            let u0 = layout.up_offset as usize;
            let d0 = layout.down_offset as usize;
            let gate_src = self.mmap[g0..g0 + layout.gate_bytes].to_vec();
            let up_src = self.mmap[u0..u0 + layout.up_bytes].to_vec();
            let down_src = self.mmap[d0..d0 + layout.down_bytes].to_vec();
            let slot_i = self.lru_slot();
            let slot = &mut self.slots[slot_i];
            slot.gate = gate_src;
            slot.up = up_src;
            slot.down = down_src;
            slot.key = Some(key);
            self.clock += 1;
            slot.last_used = self.clock;
            out.push(slot_i);
        }
        Ok(out)
    }

    pub fn gate(&self, slot: usize) -> &[u8] {
        &self.slots[slot].gate
    }
    pub fn up(&self, slot: usize) -> &[u8] {
        &self.slots[slot].up
    }
    pub fn down(&self, slot: usize) -> &[u8] {
        &self.slots[slot].down
    }

    pub fn slots_for_budget(budget_mib: usize, expert_bytes: usize) -> usize {
        if expert_bytes == 0 {
            return 1;
        }
        let budget = budget_mib.saturating_mul(1024 * 1024);
        (budget / expert_bytes).max(8)
    }
}

pub type SharedExpertCache = Arc<Mutex<ExpertSsdCache>>;

use std::thread;

pub fn prefetch_experts_async(
    cache: SharedExpertCache,
    keys: Vec<ExpertKey>,
) -> thread::JoinHandle<std::io::Result<Vec<usize>>> {
    thread::spawn(move || {
        let mut c = cache.lock().unwrap();
        c.pin(&keys)
    })
}

pub fn estimate_expert_bytes(n_embd: usize, n_ff: usize) -> usize {
    use super::quant::{IQ2_XXS_BLOCK_BYTES, Q2_K_BLOCK_BYTES, QK_K};
    let iq2_row = |rows: usize, cols: usize| -> usize {
        rows * (cols / QK_K) * IQ2_XXS_BLOCK_BYTES
    };
    let q2k_row = |rows: usize, cols: usize| -> usize {
        rows * (cols / QK_K) * Q2_K_BLOCK_BYTES
    };
    iq2_row(n_ff, n_embd) + iq2_row(n_ff, n_embd) + q2k_row(n_embd, n_ff)
}
