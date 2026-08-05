//! SSD expert streaming cache for DeepSeek-V4-Flash routed experts.
//!
//! Policy (studied from ds4 docs/behavior, implemented ourselves):
//! - Dense / shared / HC / attn stay resident.
//! - Routed `ffn_{gate,up,down}_exps` pages in per (layer, expert).
//! - Overlap: caller can prefetch while shared expert runs.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

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

/// Host-side expert cache backed by pread from the GGUF file.
pub struct ExpertSsdCache {
    path: PathBuf,
    file: Mutex<File>,
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
        Ok(Self {
            path,
            file: Mutex::new(file),
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

    fn read_exact_at(file: &mut File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(buf)
    }

    /// Ensure experts are resident; returns slot indices in the same order as `keys`.
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
            let layout = self
                .layouts
                .get(&key)
                .cloned()
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("no layout for layer {} expert {}", key.layer, key.expert),
                    )
                })?;
            let slot_i = self.lru_slot();
            let slot = &mut self.slots[slot_i];
            slot.gate.resize(layout.gate_bytes, 0);
            slot.up.resize(layout.up_bytes, 0);
            slot.down.resize(layout.down_bytes, 0);
            {
                let mut file = self.file.lock().unwrap();
                Self::read_exact_at(&mut file, layout.gate_offset, &mut slot.gate)?;
                Self::read_exact_at(&mut file, layout.up_offset, &mut slot.up)?;
                Self::read_exact_at(&mut file, layout.down_offset, &mut slot.down)?;
            }
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

    /// Auto slot count from a memory budget (MiB) and per-expert byte size.
    pub fn slots_for_budget(budget_mib: usize, expert_bytes: usize) -> usize {
        if expert_bytes == 0 {
            return 1;
        }
        let budget = budget_mib.saturating_mul(1024 * 1024);
        (budget / expert_bytes).max(8)
    }
}

/// Shared handle for concurrent prefetch helpers.
pub type SharedExpertCache = Arc<Mutex<ExpertSsdCache>>;

/// Estimate one Flash IQ2 expert footprint (gate IQ2 + up IQ2 + down Q2_K).
pub fn estimate_expert_bytes(n_embd: usize, n_ff: usize) -> usize {
    use super::quant::{IQ2_XXS_BLOCK_BYTES, Q2_K_BLOCK_BYTES, QK_K};
    let iq2_row = |rows: usize, cols: usize| -> usize {
        rows * (cols / QK_K) * IQ2_XXS_BLOCK_BYTES
    };
    let q2k_row = |rows: usize, cols: usize| -> usize {
        rows * (cols / QK_K) * Q2_K_BLOCK_BYTES
    };
    // gate/up: [ff, embd], down: [embd, ff]
    iq2_row(n_ff, n_embd) + iq2_row(n_ff, n_embd) + q2k_row(n_embd, n_ff)
}
