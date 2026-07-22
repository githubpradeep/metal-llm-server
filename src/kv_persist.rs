//! Minimal on-disk KV cache persistence for "warm reopen" — save the GPU KV
//! cache + token history at the end of a turn, reload on next launch, and
//! skip re-prefilling the restored prefix entirely.
//!
//! File layout (all little-endian):
//!   magic:        4 bytes  b"MMKV"
//!   version:      u32
//!   fp_len:       u32
//!   fingerprint:  fp_len bytes (utf8) — model path + size + kv_type + ctx_size
//!   kv_type:      u8       (0=f16, 1=q8_0, 2=q4_0)
//!   ctx_size:     u32      (kv_capacity the buffers were sized for)
//!   kv_seq_len:   u32
//!   total_tokens: u32
//!   history_len:  u32
//!   history:      history_len * u32 token ids
//!   num_kv_layers:u32
//!   per layer: layer_idx:u32, num_kv_heads:u32, head_dim:u32, k_bytes, v_bytes
//!
//! KV cache layout on GPU is `[head][capacity rows][row_bytes]` (see
//! `kv_cache_append_*` kernels), so only the first `kv_seq_len` rows of each
//! head's block are meaningful — we slice those out per head rather than
//! dumping the (possibly huge, e.g. 200k-token) full capacity buffer.
//!
//! Shared-KV layers (`has_kv == false`) read another layer's cache via
//! `kv_source_layer` and own no data of their own, so they're skipped.

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::gemma4_config::KvCacheType;
use crate::gemma4_gpu_model::Gemma4GpuModel;

const MAGIC: &[u8; 4] = b"MMKV";
const VERSION: u32 = 1;

fn kv_type_tag(t: KvCacheType) -> u8 {
    match t {
        KvCacheType::F16 => 0,
        KvCacheType::Q8_0 => 1,
        KvCacheType::Q4_0 => 2,
        // Encode bit-widths in the high nibble so K3/V2 ≠ K4/V4 sessions.
        KvCacheType::TurboQuant { k_bits, v_bits } => {
            0x30 | ((k_bits & 0x3) << 2) | (v_bits & 0x3)
        }
    }
}

/// Build a fingerprint string identifying the model + KV configuration this
/// session was captured against. Any mismatch on load is refused loudly —
/// we never silently replay KV state against the wrong model.
pub fn model_fingerprint(model_path: &str, kv_cache_type: KvCacheType, ctx_size: u32) -> String {
    let size = fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);
    format!(
        "path={}|size={}|kv={}|ctx={}",
        model_path, size, kv_cache_type, ctx_size
    )
}

/// Default session directory: `~/.cache/mega-metal/sessions/` (override with
/// `LLAMA_SESSION_DIR`).
pub fn default_sessions_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("LLAMA_SESSION_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(home) = std::env::var("HOME") {
        return Path::new(&home).join(".cache/mega-metal/sessions");
    }
    PathBuf::from(".sessions")
}

fn session_path(sessions_dir: &Path, session_id: &str) -> PathBuf {
    sessions_dir.join(format!("{}.kv", session_id))
}

fn write_u32(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

/// Read raw bytes directly out of a Metal `StorageModeShared` buffer. Apple
/// Silicon unified memory means this is a plain CPU memcpy — no explicit GPU
/// readback command is needed, but the caller must ensure the GPU has no
/// in-flight writes to this buffer (true here: the decode loop is fully
/// synchronous, so the buffer is settled between tokens).
fn buffer_slice(buf: &metal::Buffer, offset: usize, len: usize) -> Vec<u8> {
    let ptr = buf.contents() as *const u8;
    unsafe { std::slice::from_raw_parts(ptr.add(offset), len).to_vec() }
}

fn buffer_write(buf: &metal::Buffer, offset: usize, data: &[u8]) {
    let ptr = buf.contents() as *mut u8;
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset), data.len());
    }
}

/// Save the model's current KV cache + token history to
/// `<sessions_dir>/<session_id>.kv`. Writes to a `.tmp` file first and
/// renames, so a crash mid-save never corrupts a previously-good session.
pub fn save_session(
    sessions_dir: &Path,
    session_id: &str,
    model_path: &str,
    model: &Gemma4GpuModel,
    token_history: &[u32],
) -> io::Result<(PathBuf, u64)> {
    fs::create_dir_all(sessions_dir)?;
    let final_path = session_path(sessions_dir, session_id);
    let tmp_path = sessions_dir.join(format!("{}.kv.tmp", session_id));

    let fp = model_fingerprint(model_path, model.kv_cache_type, model.kv_capacity);

    let kv_layers: Vec<usize> = model
        .layers
        .iter()
        .enumerate()
        .filter(|(_, l)| l.has_kv)
        .map(|(i, _)| i)
        .collect();

    {
        let file = File::create(&tmp_path)?;
        let mut w = BufWriter::new(file);

        w.write_all(MAGIC)?;
        write_u32(&mut w, VERSION)?;
        write_u32(&mut w, fp.len() as u32)?;
        w.write_all(fp.as_bytes())?;
        w.write_all(&[kv_type_tag(model.kv_cache_type)])?;
        write_u32(&mut w, model.kv_capacity)?;
        write_u32(&mut w, model.kv_seq_len)?;
        write_u32(&mut w, model.total_tokens as u32)?;
        write_u32(&mut w, token_history.len() as u32)?;
        for &tok in token_history {
            write_u32(&mut w, tok)?;
        }
        write_u32(&mut w, kv_layers.len() as u32)?;

        let seq = model.kv_seq_len as usize;
        let capacity = model.kv_capacity as usize;
        for &layer_idx in &kv_layers {
            let num_kv_heads = model.config.layer_num_kv_heads(layer_idx);
            let head_dim = model.config.layer_head_dim(layer_idx);
            let row_bytes = model.kv_cache_type.bytes_per_row(head_dim);

            write_u32(&mut w, layer_idx as u32)?;
            write_u32(&mut w, num_kv_heads as u32)?;
            write_u32(&mut w, head_dim as u32)?;

            for (cache, _name) in [(&model.k_cache, "k"), (&model.v_cache, "v")] {
                let buf = &cache[layer_idx];
                for h in 0..num_kv_heads {
                    let offset = h * capacity * row_bytes;
                    let bytes = buffer_slice(buf, offset, seq * row_bytes);
                    w.write_all(&bytes)?;
                }
            }
        }
        w.flush()?;
    }

    let size = fs::metadata(&tmp_path)?.len();
    fs::rename(&tmp_path, &final_path)?;
    Ok((final_path, size))
}

pub struct LoadedSession {
    pub token_history: Vec<u32>,
    pub kv_seq_len: u32,
    pub total_tokens: usize,
}

/// Load `<sessions_dir>/<session_id>.kv` and write its KV cache directly into
/// `model`'s GPU buffers. Refuses loudly (returns `Err`) on any magic /
/// version / model-fingerprint mismatch — never silently replays KV state
/// captured against a different model or KV configuration.
pub fn load_session(
    sessions_dir: &Path,
    session_id: &str,
    model_path: &str,
    model: &mut Gemma4GpuModel,
) -> io::Result<LoadedSession> {
    let path = session_path(sessions_dir, session_id);
    let file = File::open(&path)?;
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "session {:?}: bad magic header {:?} (expected {:?}) — refusing to load",
                path, magic, MAGIC
            ),
        ));
    }
    let version = read_u32(&mut r)?;
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "session {:?}: version {} unsupported (expected {}) — refusing to load",
                path, version, VERSION
            ),
        ));
    }

    let fp_len = read_u32(&mut r)? as usize;
    let mut fp_bytes = vec![0u8; fp_len];
    r.read_exact(&mut fp_bytes)?;
    let saved_fp = String::from_utf8_lossy(&fp_bytes).to_string();
    let expected_fp = model_fingerprint(model_path, model.kv_cache_type, model.kv_capacity);
    if saved_fp != expected_fp {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "session {:?}: model fingerprint mismatch — refusing to load\n  saved:    {}\n  expected: {}",
                path, saved_fp, expected_fp
            ),
        ));
    }

    let mut kv_type_byte = [0u8; 1];
    r.read_exact(&mut kv_type_byte)?;
    let ctx_size = read_u32(&mut r)?;
    if ctx_size != model.kv_capacity {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "session {:?}: ctx_size {} != current LLAMA_CTX_SIZE {} — refusing to load",
                path, ctx_size, model.kv_capacity
            ),
        ));
    }

    let kv_seq_len = read_u32(&mut r)?;
    let total_tokens = read_u32(&mut r)? as usize;
    let history_len = read_u32(&mut r)? as usize;
    let mut token_history = Vec::with_capacity(history_len);
    for _ in 0..history_len {
        token_history.push(read_u32(&mut r)?);
    }

    let num_kv_layers = read_u32(&mut r)? as usize;
    let seq = kv_seq_len as usize;
    let capacity = model.kv_capacity as usize;

    for _ in 0..num_kv_layers {
        let layer_idx = read_u32(&mut r)? as usize;
        let num_kv_heads = read_u32(&mut r)? as usize;
        let head_dim = read_u32(&mut r)? as usize;

        let expected_heads = model.config.layer_num_kv_heads(layer_idx);
        let expected_hd = model.config.layer_head_dim(layer_idx);
        if num_kv_heads != expected_heads || head_dim != expected_hd {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "session {:?}: layer {} shape mismatch (saved kv_heads={} head_dim={}, model kv_heads={} head_dim={}) — refusing to load",
                    path, layer_idx, num_kv_heads, head_dim, expected_heads, expected_hd
                ),
            ));
        }
        let row_bytes = model.kv_cache_type.bytes_per_row(head_dim);

        for cache in [&model.k_cache, &model.v_cache] {
            let buf = &cache[layer_idx];
            for h in 0..num_kv_heads {
                let mut bytes = vec![0u8; seq * row_bytes];
                r.read_exact(&mut bytes)?;
                let offset = h * capacity * row_bytes;
                buffer_write(buf, offset, &bytes);
            }
        }
    }

    let _ = kv_type_byte; // already validated via fingerprint string
    model.kv_seq_len = kv_seq_len;
    model.total_tokens = total_tokens;

    Ok(LoadedSession {
        token_history,
        kv_seq_len,
        total_tokens,
    })
}

/// List `<sessions_dir>/*.kv` sessions with file sizes, newest first.
pub fn list_sessions(sessions_dir: &Path) -> Vec<(String, u64)> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(sessions_dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().map(|e| e == "kv").unwrap_or(false) {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                out.push((stem.to_string(), size));
            }
        }
    }
    out.sort_by(|a, b| b.1.cmp(&a.1));
    out
}
