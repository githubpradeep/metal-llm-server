//! Colibrì-style next-layer expert prefetch: `F_RDADVISE` only, never MTL Shared.
//!
//! `DSV4_PILOT=1` opts in. A dedicated thread issues readahead hints so the
//! forward thread is not blocked (inline fadvise measured badly in colibrì).
//! Full queue ⇒ drop hints (a missed prefetch is not an error).

use std::os::unix::io::AsRawFd;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

use super::expert_gpu::ExpertGpuCache;
use super::ssd::{ExpertKey, ExpertSsdCache};

#[derive(Clone, Copy)]
struct AdviseTask {
    fd: i32,
    offset: u64,
    len: i32,
}

struct PilotState {
    tasks: Vec<AdviseTask>,
    stop: bool,
}

static PILOT: OnceLock<(Mutex<PilotState>, Condvar)> = OnceLock::new();
static STARTED: AtomicBool = AtomicBool::new(false);
static HINTS: AtomicU64 = AtomicU64::new(0);
static SKIP_RESIDENT: AtomicU64 = AtomicU64::new(0);
static DROPPED: AtomicU64 = AtomicU64::new(0);

pub fn enabled() -> bool {
    std::env::var("DSV4_PILOT").ok().as_deref() == Some("1")
}

pub fn stats() -> (u64, u64, u64) {
    (
        HINTS.load(Ordering::Relaxed),
        SKIP_RESIDENT.load(Ordering::Relaxed),
        DROPPED.load(Ordering::Relaxed),
    )
}

fn fadvise(fd: i32, offset: u64, len: i32) {
    if fd < 0 || len <= 0 {
        return;
    }
    #[cfg(target_os = "macos")]
    {
        #[repr(C)]
        struct Radvisory {
            ra_offset: i64,
            ra_count: i32,
        }
        const F_RDADVISE: i32 = 44;
        let adv = Radvisory {
            ra_offset: offset as i64,
            ra_count: len,
        };
        let _ = unsafe { libc::fcntl(fd, F_RDADVISE, &adv as *const Radvisory) };
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = unsafe {
            libc::posix_fadvise(
                fd,
                offset as libc::off_t,
                len as libc::off_t,
                libc::POSIX_FADV_WILLNEED,
            )
        };
    }
}

fn worker() {
    let (mu, cv) = PILOT.get().expect("pilot");
    loop {
        let mut g = mu.lock().unwrap();
        while !g.stop && g.tasks.is_empty() {
            g = cv.wait(g).unwrap();
        }
        if g.stop && g.tasks.is_empty() {
            break;
        }
        let batch = std::mem::take(&mut g.tasks);
        drop(g);
        for t in batch {
            fadvise(t.fd, t.offset, t.len);
        }
    }
}

fn ensure_worker() {
    let _ = PILOT.get_or_init(|| {
        (
            Mutex::new(PilotState {
                tasks: Vec::new(),
                stop: false,
            }),
            Condvar::new(),
        )
    });
    if !STARTED.swap(true, Ordering::Relaxed) {
        std::thread::Builder::new()
            .name("dsv4-pilot".into())
            .spawn(worker)
            .expect("pilot worker");
        println!("  expert pilot: F_RDADVISE thread (DSV4_PILOT=1)");
    }
}

/// Enqueue WILLNEED for missing keys. Resident GPU slots are skipped.
pub fn submit(ssd: &ExpertSsdCache, keys: &[ExpertKey], eg: Option<&ExpertGpuCache>) {
    if !enabled() || keys.is_empty() {
        return;
    }
    ensure_worker();
    let fd = ssd.file().as_raw_fd();
    let mut tasks = Vec::with_capacity(keys.len() * 3);
    for &key in keys {
        if eg.map(|e| e.contains(key)).unwrap_or(false) {
            SKIP_RESIDENT.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        let layout = ssd.layout(key);
        let mut push = |offset: u64, bytes: usize| {
            if bytes == 0 || bytes > i32::MAX as usize {
                return;
            }
            tasks.push(AdviseTask {
                fd,
                offset,
                len: bytes as i32,
            });
        };
        push(layout.gate_offset, layout.gate_bytes);
        push(layout.up_offset, layout.up_bytes);
        push(layout.down_offset, layout.down_bytes);
        HINTS.fetch_add(1, Ordering::Relaxed);
    }
    if tasks.is_empty() {
        return;
    }
    let (mu, cv) = PILOT.get().expect("pilot");
    let mut g = mu.lock().unwrap();
    if g.tasks.len() + tasks.len() > 384 {
        DROPPED.fetch_add(tasks.len() as u64 / 3, Ordering::Relaxed);
        return;
    }
    g.tasks.extend(tasks);
    cv.notify_one();
}
