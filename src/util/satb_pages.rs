//! Page-COW SATB: page-granularity snapshot-at-the-beginning barrier for
//! concurrent marking, replacing the compiled per-store SATB barrier.
//!
//! At InitialMark the heap is write-protect-armed (bpf_fault WP).  The
//! first write to a page traps in-kernel: the handler copies the PRE-write
//! page into a snapshot arena and sets a byte flag, then clears protection
//! (one fault per page per cycle; ~9us including two page copies).
//! Markers read the LIVE heap; SATB completeness comes from draining the
//! snapshot pages: every reference value present at mark start is either
//! still in the live heap or in some snapshot page, so conservatively
//! enqueueing every plausible reference found in snapshots (VO-bit
//! filtered; over-approximation = floating garbage) preserves everything
//! reachable at mark start.  See docs/satb-pages-design.md.

use crate::util::Address;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::OnceLock;

pub(crate) const LOG_BYTES_IN_PAGE: usize = 12;
pub(crate) const BYTES_IN_PAGE: usize = 1 << LOG_BYTES_IN_PAGE;
const CHUNK: usize = 4 << 20;

static ACTIVE: AtomicBool = AtomicBool::new(false);
static TRACKER: OnceLock<SatbPages> = OnceLock::new();

/// Whether the page-COW SATB barrier is enabled (MMTK_SATB_PAGES).
pub fn satb_pages_active() -> bool {
    ACTIVE.load(Ordering::Relaxed)
}

pub(crate) fn satb_pages() -> Option<&'static SatbPages> {
    TRACKER.get()
}

/// Initialize from the plan constructor when MMTK_SATB_PAGES is set.
pub(crate) fn init_satb_pages(start: Address, end: Address) {
    if std::env::var_os("MMTK_SATB_PAGES").is_none() {
        return;
    }
    let span = end - start;
    TRACKER
        .set(SatbPages::new(start, span))
        .ok()
        .expect("satb pages initialized twice");
    ACTIVE.store(true, Ordering::Relaxed);
}

pub(crate) struct SatbPages {
    base: Address,
    span: usize,
    shim: shim::Shim,
    /// byte-per-page snapshot flags (in the arena, set by the handler)
    flags: *mut u8,
    /// snapshot pages (arena offset 0)
    snaps: *mut u8,
    /// arena chunks already pre-touched (kernel stores to unpopulated
    /// arena pages are silently dropped, so touch before arming)
    touched: Vec<std::sync::atomic::AtomicU64>,
    /// drain resume cursor (page index), for incremental fair draining
    cursor: AtomicUsize,
    /// chunks registered with the WP link (once per chunk, ever)
    registered: std::sync::Mutex<std::collections::HashSet<usize>>,
    /// ranges armed THIS cycle (disarm exactly these: the chunk map may
    /// have grown during marking, and never-registered ranges fail WP-off)
    armed: std::sync::Mutex<Vec<(Address, usize)>>,
    /// 32KB blocks acquired since mark start ("young").  Currently unused
    /// by filtering (recyclable blocks mix mark-start-live objects with
    /// fresh allocation, so block-granular young filtering drops real SATB
    /// edges); kept for a future line-granular variant.
    young_blocks: Vec<std::sync::atomic::AtomicU64>,
    /// Conservative candidates stashed by the CONCURRENT drainer, traced
    /// only at FinalMark: tracing them concurrently races in-flight
    /// allocation (VO bit visible before klass init -> oop_iterate crash);
    /// at the FinalMark safepoint everything is quiesced and fenced.
    stash: std::sync::Mutex<Vec<Address>>,
    /// Drainer-thread handshake: the FinalMark sweep shares the cursor and
    /// stash with the drainer, so the drainer MUST be quiesced first
    /// (unsynchronized overlap loses drained candidates = missed SATB
    /// edges, measured as downstream heap corruption).
    drain_stop: AtomicBool,
    drainer_running: AtomicBool,
}

unsafe impl Sync for SatbPages {}
unsafe impl Send for SatbPages {}

impl SatbPages {
    fn new(base: Address, span: usize) -> Self {
        let shim = shim::Shim::load();
        assert_eq!(shim.init(base, span), 0, "gcsatb_init failed");
        let flags = shim.flags();
        let snaps = shim.snapshots();
        assert!(!flags.is_null() && !snaps.is_null());
        SatbPages {
            base,
            span,
            shim,
            flags,
            snaps,
            touched: (0..(span / CHUNK).div_ceil(64).max(1))
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect(),
            cursor: AtomicUsize::new(0),
            registered: std::sync::Mutex::new(std::collections::HashSet::new()),
            armed: std::sync::Mutex::new(Vec::new()),
            young_blocks: (0..(span >> 15).div_ceil(64).max(1))
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect(),
            stash: std::sync::Mutex::new(Vec::new()),
            drain_stop: AtomicBool::new(false),
            drainer_running: AtomicBool::new(false),
        }
    }

    /// Arm a (chunk-aligned) range: register, pre-touch its arena slice
    /// (first time), and write-protect it.
    pub fn arm(&self, start: Address, bytes: usize) {
        debug_assert!(start >= self.base && start + bytes <= self.base + self.span);
        let mut c = (start - self.base) / CHUNK;
        let end_c = (start + bytes - self.base).div_ceil(CHUNK);
        while c < end_c {
            let (w, b) = (c >> 6, c & 63);
            if self.touched[w].load(Ordering::Relaxed) & (1 << b) == 0 {
                // touch the snapshot slice + flag bytes for this chunk
                unsafe {
                    let mut off = c * CHUNK;
                    let lim = off + CHUNK;
                    while off < lim {
                        std::ptr::write_volatile(self.snaps.add(off), 0);
                        off += BYTES_IN_PAGE;
                    }
                    // flag bytes for this chunk span CHUNK/4096 = 1024 bytes
                    std::ptr::write_volatile(
                        self.flags.add(c * (CHUNK >> LOG_BYTES_IN_PAGE)),
                        0,
                    );
                }
                self.touched[w].fetch_or(1 << b, Ordering::Relaxed);
            }
            c += 1;
        }
        {
            let mut reg = self.registered.lock().unwrap();
            if reg.insert(start.as_usize()) {
                assert_eq!(self.shim.register(start, bytes), 0, "gcsatb register");
            }
        }
        assert_eq!(self.shim.wp(start, bytes, true), 0, "gcsatb wp on");
        self.armed.lock().unwrap().push((start, bytes));
    }

    /// Drop protection from every range armed this cycle (FinalMark
    /// teardown).  Pages that faulted are already unprotected; this
    /// clears the remainder.  Chunks allocated DURING marking were never
    /// armed (their objects allocate black), so they are not touched.
    pub fn disarm_all(&self) {
        for (start, bytes) in self.armed.lock().unwrap().drain(..) {
            assert_eq!(self.shim.wp(start, bytes, false), 0, "gcsatb wp off");
        }
    }

    /// Reset the drain cursor at the start of a mark cycle.
    pub fn reset_cursor(&self) {
        self.cursor.store(0, Ordering::Relaxed);
    }

    /// Mark a 32KB block acquired during marking as young (allocator hook).
    pub fn mark_young_block(&self, start: Address) {
        let b = (start.as_usize().wrapping_sub(self.base.as_usize())) >> 15;
        if b < self.span >> 15 {
            self.young_blocks[b >> 6].fetch_or(1 << (b & 63), Ordering::Release);
        }
    }

    /// Is this address inside a block acquired since mark start?
    pub fn is_young(&self, addr: Address) -> bool {
        let b = (addr.as_usize().wrapping_sub(self.base.as_usize())) >> 15;
        b < self.span >> 15
            && self.young_blocks[b >> 6].load(Ordering::Acquire) & (1 << (b & 63)) != 0
    }

    /// Clear the young set (InitialMark: the frontier is mark start).
    pub fn clear_young(&self) {
        for w in &self.young_blocks {
            w.store(0, Ordering::Relaxed);
        }
    }

    /// Stash conservative candidates for FinalMark tracing.
    pub fn stash_candidates(&self, addrs: &mut Vec<Address>) {
        self.stash.lock().unwrap().append(addrs);
    }

    /// Take all stashed candidates (FinalMark).
    pub fn take_stash(&self) -> Vec<Address> {
        std::mem::take(&mut *self.stash.lock().unwrap())
    }

    /// Drainer lifecycle handshake.
    pub fn drainer_started(&self) {
        self.drainer_running.store(true, Ordering::Release);
    }
    pub fn drainer_exited(&self) {
        self.drainer_running.store(false, Ordering::Release);
    }
    pub fn should_stop(&self) -> bool {
        self.drain_stop.load(Ordering::Acquire)
    }
    /// FinalMark: stop the drainer and wait until it has exited, so the
    /// sweep owns the cursor/flags/stash exclusively.
    pub fn stop_drainer_and_wait(&self) {
        self.drain_stop.store(true, Ordering::Release);
        while self.drainer_running.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
    }
    /// InitialMark: allow the next drainer to run.
    pub fn allow_drainer(&self) {
        self.drain_stop.store(false, Ordering::Release);
    }

    /// Drain up to `max_pages` flagged snapshot pages: conservatively
    /// visit every plausible mark-start reference found (8-aligned value
    /// decoding into the span with the VO bit set), clearing flags.
    /// Returns (pages_drained, wrapped) — wrapped=true when the cursor
    /// completed a full sweep of the span.
    pub fn drain<F: FnMut(Address)>(&self, max_pages: usize, mut visit: F) -> (usize, bool) {
        let total = self.span >> LOG_BYTES_IN_PAGE;
        let mut drained = 0;
        let mut idx = self.cursor.load(Ordering::Relaxed);
        let start_idx = idx;
        loop {
            if idx >= total {
                self.cursor.store(total, Ordering::Relaxed);
                return (drained, true);
            }
            let flag = unsafe { std::ptr::read_volatile(self.flags.add(idx)) };
            if flag != 0 {
                unsafe { std::ptr::write_volatile(self.flags.add(idx), 0) };
                let page = unsafe { self.snaps.add(idx << LOG_BYTES_IN_PAGE) as *const u32 };
                for s in 0..(BYTES_IN_PAGE / 4) {
                    let v = unsafe { std::ptr::read_volatile(page.add(s)) } as usize;
                    // unscaled zero-based compressed oops (heap <= 4GB in
                    // our configs): the dword IS the address; 8-aligned
                    // object starts only.
                    if v == 0 || v & 7 != 0 {
                        continue;
                    }
                    let addr = unsafe { Address::from_usize(v) };
                    if addr >= self.base && addr < self.base + self.span {
                        visit(addr);
                    }
                }
                drained += 1;
                if drained >= max_pages {
                    self.cursor.store(idx + 1, Ordering::Relaxed);
                    return (drained, false);
                }
            }
            idx += 1;
            if idx - start_idx > total {
                return (drained, true);
            }
        }
    }
}

mod shim {
    use super::*;
    use std::ffi::CString;

    type InitFn = unsafe extern "C" fn(u64, u64) -> i32;
    type RangeFn = unsafe extern "C" fn(u64, u64) -> i32;
    type WpFn = unsafe extern "C" fn(u64, u64, i32) -> i32;
    type PtrFn = unsafe extern "C" fn() -> *mut u8;

    pub(super) struct Shim {
        init: InitFn,
        register: RangeFn,
        wp: WpFn,
        flags: PtrFn,
        snapshots: PtrFn,
    }

    impl Shim {
        pub fn load() -> Self {
            let path = std::env::var("MMTK_BPF_SHIM")
                .unwrap_or_else(|_| "/mydata/gc-bpf-fault/shim/libgcbpf.so".to_string());
            let cpath = CString::new(path.clone()).unwrap();
            let handle = unsafe { libc::dlopen(cpath.as_ptr(), libc::RTLD_NOW) };
            assert!(!handle.is_null(), "satb_pages: dlopen {} failed", path);
            let sym = |name: &str| {
                let cname = CString::new(name).unwrap();
                let p = unsafe { libc::dlsym(handle, cname.as_ptr()) };
                assert!(!p.is_null(), "satb shim: missing symbol {}", name);
                p
            };
            unsafe {
                Self {
                    init: std::mem::transmute(sym("gcsatb_init")),
                    register: std::mem::transmute(sym("gcsatb_register")),
                    wp: std::mem::transmute(sym("gcsatb_wp")),
                    flags: std::mem::transmute(sym("gcsatb_flags")),
                    snapshots: std::mem::transmute(sym("gcsatb_snapshots")),
                }
            }
        }
        pub fn init(&self, base: Address, span: usize) -> i32 {
            unsafe { (self.init)(base.as_usize() as u64, span as u64) }
        }
        pub fn register(&self, start: Address, bytes: usize) -> i32 {
            unsafe { (self.register)(start.as_usize() as u64, bytes as u64) }
        }
        pub fn wp(&self, start: Address, bytes: usize, on: bool) -> i32 {
            unsafe { (self.wp)(start.as_usize() as u64, bytes as u64, on as i32) }
        }
        pub fn flags(&self) -> *mut u8 {
            unsafe { (self.flags)() }
        }
        pub fn snapshots(&self) -> *mut u8 {
            unsafe { (self.snapshots)() }
        }
    }
}
