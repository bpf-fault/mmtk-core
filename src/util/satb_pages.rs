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

static VERIFY: AtomicBool = AtomicBool::new(false);
static UFFD_FD: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

/// MMTK_SATB_VERIFY: differential oracle — the compiled SATB barrier
/// stays enabled (correct execution), while the page machinery arms,
/// snapshots and extracts in parallel, CLASSIFYING its output at
/// FinalMark instead of tracing: {valid+marked, valid+unmarked(=refs
/// only pages would rescue), garbage}.  Pinpoints extractor defects
/// without crash-roulette.
pub fn satb_verify() -> bool {
    VERIFY.load(Ordering::Relaxed)
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
    if std::env::var_os("MMTK_SATB_VERIFY").is_some() {
        VERIFY.store(true, Ordering::Relaxed);
    }
    if std::env::var_os("MMTK_SATB_COMMS").is_some() {
        TRACKER.get().unwrap().spawn_comm_dumper();
    }
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
    /// VO-bitmap snapshot taken at InitialMark, BEFORE any of this cycle's
    /// marking.  Conservative candidates are validated against THIS map:
    /// the live VO map is poisoned by our own conservative marks at the
    /// next sweep (CopyFromMarkBits copies mark bits, including marks we
    /// set on candidates, into VO bits) -- measured as ASCII text data
    /// acquiring vo=true and then crashing the tracer.  The snapshot is
    /// pure by induction: marks only ever land on snapshot-validated
    /// genuine objects, so the sweep-copied VO stays a true allocation
    /// map.  Post-mark-start allocations are absent and skipped (they are
    /// allocate-black and need no SATB rescue).
    alloc_map: *mut u8,
}

unsafe impl Sync for SatbPages {}
unsafe impl Send for SatbPages {}

impl SatbPages {
    /// Spawn a thread that dumps the fault-comm table every second
    /// (crash-tolerant: visible before FinalMark's disarm).
    pub fn spawn_comm_dumper(&self) {
        std::thread::spawn(|| unsafe {
            let c = std::ffi::CString::new("gcsatb_dump_comms").unwrap();
            let p = libc::dlsym(libc::RTLD_DEFAULT as *mut libc::c_void, c.as_ptr());
            if p.is_null() {
                return;
            }
            let f: unsafe extern "C" fn() = std::mem::transmute(p);
            loop {
                std::thread::sleep(std::time::Duration::from_millis(1000));
                f();
            }
        });
    }

    fn new(base: Address, span: usize) -> Self {
        let shim = shim::Shim::load();
        assert_eq!(shim.init(base, span), 0, "gcsatb_init failed");
        let flags = shim.flags();
        let snaps = shim.snapshots();
        assert!(!flags.is_null() && !snaps.is_null());
        if std::env::var_os("MMTK_SATB_NOOP").is_some() {
            shim.set_noop();
        }
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
            alloc_map: unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    (span >> 6).max(4096),
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                    -1,
                    0,
                ) as *mut u8
            },
        }
    }

    /// Arm a (chunk-aligned) range: register, pre-touch its arena slice
    /// (first time), and write-protect it.
    pub fn arm(&self, start: Address, bytes: usize) {
        let uffd_mode = std::env::var_os("MMTK_SATB_UFFD").is_some();
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
        if std::env::var_os("MMTK_SATB_UFFD").is_some() {
            self.uffd_arm(start, bytes);
        } else if std::env::var_os("MMTK_SATB_NOWP").is_none() {
            assert_eq!(self.shim.wp(start, bytes, true), 0, "gcsatb wp on");
        }
        self.armed.lock().unwrap().push((start, bytes));
    }

    /// Mainline userfaultfd WP-async arming (MMTK_SATB_UFFD=1): the
    /// SAME wrprotect + in-kernel async resolution flow as bpf-fault WP,
    /// through battle-tested mainline code with zero bpf involvement.
    /// Differential oracle for the kernel path: if ConcurrentImmix
    /// corrupts under THIS arming too, the fault mechanism is exonerated
    /// and the corruption is an upstream plan race exposed by fault
    /// timing; if it is clean, the bug lives in the (small) diff between
    /// the two kernel paths.
    pub fn uffd_arm(&self, start: Address, bytes: usize) {
        let mut fd = UFFD_FD.load(Ordering::Relaxed);
        if fd < 0 {
            unsafe {
                // userfaultfd(O_CLOEXEC) + API handshake with WP_ASYNC
                fd = libc::syscall(libc::SYS_userfaultfd, 0o2000000i32) as i32;
                assert!(fd >= 0, "userfaultfd syscall failed");
                #[repr(C)]
                struct UffdioApi { api: u64, features: u64, ioctls: u64 }
                let mut api = UffdioApi {
                    api: 0xAA,                       // UFFD_API
                    features: 1 << 15,               // UFFD_FEATURE_WP_ASYNC
                    ioctls: 0,
                };
                let r = libc::ioctl(fd, 0xc018aa3f_u64 as _, &mut api); // UFFDIO_API
                assert_eq!(r, 0, "UFFDIO_API failed");
                UFFD_FD.store(fd, Ordering::Relaxed);
            }
        }
        unsafe {
            #[repr(C)]
            struct UffdioRange { start: u64, len: u64 }
            #[repr(C)]
            struct UffdioRegister { range: UffdioRange, mode: u64, ioctls: u64 }
            // Register ONCE per range: re-registering an already-
            // registered range returns EBUSY (this assert firing on the
            // SECOND cycle's arming was the entire "uffd failure" class
            // in the bisect matrix -- a harness bug, not a kernel or
            // plan defect).
            {
                let mut seen = self.registered.lock().unwrap();
                if seen.insert(start.as_usize() | 1) {
                    let mut reg = UffdioRegister {
                        range: UffdioRange {
                            start: start.as_usize() as u64,
                            len: bytes as u64,
                        },
                        mode: 1 << 1,                // UFFDIO_REGISTER_MODE_WP
                        ioctls: 0,
                    };
                    let r = libc::ioctl(fd, 0xc020aa00_u64 as _, &mut reg);
                    assert_eq!(r, 0, "UFFDIO_REGISTER failed");
                }
            }
            #[repr(C)]
            struct UffdioWriteprotect { range: UffdioRange, mode: u64 }
            let mut wp = UffdioWriteprotect {
                range: UffdioRange { start: start.as_usize() as u64, len: bytes as u64 },
                mode: 1,                             // UFFDIO_WRITEPROTECT_MODE_WP
            };
            let r = libc::ioctl(fd, 0xc018aa06_u64 as _, &mut wp); // UFFDIO_WRITEPROTECT
            assert_eq!(r, 0, "UFFDIO_WRITEPROTECT failed");
        }
    }

    /// Resolve uffd write-protection on a range (mode = 0).
    pub fn uffd_disarm(&self, start: Address, bytes: usize) {
        use std::sync::atomic::AtomicI32;
        // same fd as uffd_arm's static
        static UFFD2: AtomicI32 = AtomicI32::new(-1);
        let _ = &UFFD2;
        unsafe {
            #[repr(C)]
            struct UffdioRange { start: u64, len: u64 }
            #[repr(C)]
            struct UffdioWriteprotect { range: UffdioRange, mode: u64 }
            let fd = UFFD_FD.load(Ordering::Relaxed);
            if fd < 0 {
                return;
            }
            let mut wp = UffdioWriteprotect {
                range: UffdioRange { start: start.as_usize() as u64, len: bytes as u64 },
                mode: 0,
            };
            let r = libc::ioctl(fd, 0xc018aa06_u64 as _, &mut wp);
            assert_eq!(r, 0, "UFFDIO_WRITEPROTECT(off) failed");
        }
    }

    /// Arm an arbitrary page-aligned run (LOS object runs, immortal
    /// pages): register+pre-touch its chunk envelope, WP the exact run,
    /// snapshot its alloc-map slice.  Overwritten refs in un-armed spaces
    /// were the measured retention gap (ConcurrentHashMap tables live in
    /// the LOS; their overwritten slots' old targets were lost).
    pub fn arm_pages(&self, start: Address, bytes: usize) {
        let cstart = unsafe { Address::from_usize(start.as_usize() & !(CHUNK - 1)) };
        let cend = (start + bytes).align_up(CHUNK);
        // pre-touch + register the chunk envelope (idempotent)
        let mut c = (cstart - self.base) / CHUNK;
        let end_c = (cend - self.base) / CHUNK;
        while c < end_c {
            let (w, b) = (c >> 6, c & 63);
            if self.touched[w].load(Ordering::Relaxed) & (1 << b) == 0 {
                unsafe {
                    let mut off = c * CHUNK;
                    let lim = off + CHUNK;
                    while off < lim {
                        std::ptr::write_volatile(self.snaps.add(off), 0);
                        off += BYTES_IN_PAGE;
                    }
                    std::ptr::write_volatile(
                        self.flags.add(c * (CHUNK >> LOG_BYTES_IN_PAGE)),
                        0,
                    );
                }
                self.touched[w].fetch_or(1 << b, Ordering::Relaxed);
            }
            let chunk_addr = self.base + c * CHUNK;
            let mut reg = self.registered.lock().unwrap();
            if reg.insert(chunk_addr.as_usize()) {
                assert_eq!(self.shim.register(chunk_addr, CHUNK), 0, "gcsatb register");
            }
            c += 1;
        }
        self.snapshot_alloc_map_range(start, bytes);
        if std::env::var_os("MMTK_SATB_NOWP").is_none() {
            assert_eq!(self.shim.wp(start, bytes, true), 0, "gcsatb wp on (run)");
        }
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

    /// InitialMark: clear the whole alloc-map snapshot before the
    /// per-range copies -- slices for chunks NOT re-snapshotted this
    /// cycle would otherwise hold STALE starts (measured: phantom
    /// spanning-heads from freed chunks feeding garbage to the tracer).
    pub fn clear_alloc_map(&self) {
        unsafe {
            std::ptr::write_bytes(self.alloc_map, 0, self.span >> 6);
        }
    }

    /// Find the last alloc-map start strictly before `addr`, scanning back
    /// at most `max_back` bytes (LOS objects span megabytes; a short
    /// window misses their heads and loses their overwritten slots).
    pub fn prev_start(&self, addr: Address, max_back: usize) -> Option<Address> {
        if addr <= self.base || addr > self.base + self.span {
            return None;
        }
        let w_end = (addr - self.base) >> 3; // exclusive, 8-byte grain
        let w_lo = w_end.saturating_sub(max_back >> 3);
        let mut byte = (w_end + 7) >> 3;
        let byte_lo = w_lo >> 3;
        while byte > byte_lo {
            byte -= 1;
            let v = unsafe { *self.alloc_map.add(byte) };
            if v != 0 {
                // highest set bit whose word index < w_end
                for bit in (0..8).rev() {
                    if v & (1 << bit) != 0 {
                        let w = (byte << 3) | bit;
                        if w < w_end && w >= w_lo {
                            return Some(self.base + (w << 3));
                        }
                    }
                }
            }
        }
        None
    }

    /// InitialMark (before any marking): snapshot the VO bitmap slice for
    /// one chunk into the trusted allocation map.  PER-CHUNK: VO metadata
    /// is only mapped for in-use chunks; a whole-span copy faults on the
    /// gaps (the same lesson as the candidate-filter metadata reads).
    pub fn snapshot_alloc_map_range(&self, start: Address, bytes: usize) {
        #[cfg(feature = "vo_bit")]
        unsafe {
            let spec = &crate::util::metadata::side_metadata::spec_defs::VO_BIT;
            let src = crate::util::metadata::side_metadata::address_to_meta_address(
                spec, start,
            );
            std::ptr::copy_nonoverlapping(
                src.to_ptr::<u8>(),
                self.alloc_map.add((start - self.base) >> 6),
                bytes >> 6,
            );
        }
    }

    /// Sweep all flagged snapshot pages, visiting (page_index) and
    /// clearing flags.  FinalMark only (drainer quiesced).
    pub fn sweep_flags<F: FnMut(usize)>(&self, mut visit: F) {
        let total = self.span >> LOG_BYTES_IN_PAGE;
        for idx in 0..total {
            let flag = unsafe { std::ptr::read_volatile(self.flags.add(idx)) };
            if flag != 0 {
                unsafe { std::ptr::write_volatile(self.flags.add(idx), 0) };
                visit(idx);
            }
        }
    }

    /// Is this page currently flagged (snapshot present)?  For mixed-source
    /// slot reads during the FinalMark sweep, flags must be consulted via
    /// `snapshotted` BEFORE clearing -- so the sweep records indices first.
    pub fn snapshot_page(&self, idx: usize) -> *const u8 {
        assert!(idx < self.span >> LOG_BYTES_IN_PAGE, "snapshot_page OOB");
        unsafe { self.snaps.add(idx << LOG_BYTES_IN_PAGE) }
    }

    /// Page index; usize::MAX for out-of-span addresses (slot iteration
    /// can yield addresses outside the heap span — VM spaces, or wild
    /// values downstream of corruption; never index buffers with them).
    pub fn page_index_of(&self, addr: Address) -> usize {
        if addr < self.base || addr >= self.base + self.span {
            return usize::MAX;
        }
        (addr - self.base) >> LOG_BYTES_IN_PAGE
    }

    pub fn heap_base(&self) -> Address {
        self.base
    }

    pub fn heap_span(&self) -> usize {
        self.span
    }

    /// Scan the alloc-map snapshot for object starts in [start, start+bytes)
    /// (8-byte grain), visiting each start address.
    pub fn alloc_map_starts<F: FnMut(Address)>(&self, start: Address, bytes: usize, mut visit: F) {
        if start < self.base || start + bytes > self.base + self.span {
            return;
        }
        let first = (start - self.base) >> 3;
        let last = (start + bytes - self.base) >> 3;
        for w in first..last {
            let byte = w >> 3;
            let bit = (w & 7) as u8;
            if unsafe { (*self.alloc_map.add(byte) >> bit) & 1 } == 1 {
                visit(self.base + (w << 3));
            }
        }
    }

    /// Was `addr` an allocated object start at mark start (8-byte grain)?
    /// Bounds-checked: extracted values can be wild (space descriptors
    /// cover VA extents far beyond committed memory) — measured as an
    /// out-of-bounds alloc_map read 5.5GB past the buffer.
    pub fn in_alloc_map(&self, addr: Address) -> bool {
        if addr < self.base || addr >= self.base + self.span {
            return false;
        }
        let off = addr - self.base;
        let byte = off >> 6;          /* VO: 1 bit per 8 bytes */
        let bit = ((off >> 3) & 7) as u8;
        unsafe { (*self.alloc_map.add(byte) >> bit) & 1 == 1 }
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
        pub fn set_noop(&self) {
            unsafe {
                let cname = std::ffi::CString::new("gcsatb_set_noop").unwrap();
                let p = libc::dlsym(libc::RTLD_DEFAULT as *mut libc::c_void, cname.as_ptr());
                if !p.is_null() {
                    let f: unsafe extern "C" fn(u32) = std::mem::transmute(p);
                    f(1);
                }
            }
        }
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
