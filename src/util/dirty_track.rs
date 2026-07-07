//! Virtual-memory dirty-page tracking as a replacement for the compiled
//! generational write barrier.
//!
//! When enabled (see the `dirty_tracking` option), generational plans select
//! `NoBarrier` (no compiled barrier code at all) and instead write-protect the
//! mature space's pages at the end of each GC.  The first write a mutator
//! performs to a protected page faults; the fault handler records the page in
//! a dirty bitmap and lifts the protection.  At the start of the next nursery
//! GC the dirty pages are the remembered set.
//!
//! Three backends are provided:
//! - `Bpf`: bpf_fault — an in-kernel eBPF handler sets the dirty bit and
//!   resumes the write (no signal, no handler thread).  Loaded via a small
//!   C shim library (dlopen, `MMTK_BPF_SHIM` env var) so that mmtk-core does
//!   not link libbpf directly.
//! - `Uffd`: userfaultfd write-protect mode with a dedicated handler thread
//!   (WP faults cannot use SIGBUS self-service).
//! - `Segv`: mprotect(PROT_READ) + a chained SIGSEGV handler.

use crate::util::options::DirtyTracking;
use crate::util::Address;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

pub(crate) const LOG_BYTES_IN_PAGE: usize = 12;
pub(crate) const BYTES_IN_PAGE: usize = 1 << LOG_BYTES_IN_PAGE;

static TRACKER: OnceLock<DirtyTracker> = OnceLock::new();

/// Is VM dirty tracking active (i.e. plans should use NoBarrier)?
pub fn is_dirty_tracking_active() -> bool {
    TRACKER.get().is_some()
}

pub(crate) fn dirty_tracker() -> Option<&'static DirtyTracker> {
    TRACKER.get()
}

/// Initialize the global dirty tracker. Called from plan creation when the
/// `dirty_tracking` option is not `Barrier`.
pub(crate) fn init_dirty_tracker(backend: DirtyTracking, start: Address, end: Address) {
    if backend == DirtyTracking::Barrier {
        return;
    }
    let span_bytes = end - start;
    assert!(
        span_bytes <= 256 << 30,
        "dirty tracking: heap span {} too large for a page bitmap; \
         use compressed oops or a smaller heap layout",
        span_bytes
    );
    TRACKER
        .set(DirtyTracker::new(backend, start, end))
        .ok()
        .expect("dirty tracker initialized twice");
}

pub(crate) struct DirtyTracker {
    backend: DirtyTracking,
    span_start: Address,
    span_pages: usize,
    /// Dirty bitmap for the Uffd/Segv backends (Bpf reads the shim's mmaped
    /// BPF map instead).
    user_bitmap: Vec<AtomicU64>,
    /// Page-range starts (chunk granularity) already registered with the
    /// kernel mechanism.
    registered: Mutex<HashSet<Address>>,
    /// Lock-free fast path for `registered` (bit per 4MiB chunk of the
    /// span): set only AFTER successful kernel registration.
    registered_bits: Vec<std::sync::atomic::AtomicU64>,
    /// Chunk starts whose protection was dropped since the last re-arm,
    /// as an atomic bitmap: the promotion acquire hook runs on every GC
    /// worker (locks here measurably regressed lusearch).
    dirty_chunk_bits: Vec<std::sync::atomic::AtomicU64>,
    uffd: uffd::UffdState,
    bpf: bpf::BpfShim,
}

// The shim/uffd fds and pointers are only used in thread-safe ways.
unsafe impl Sync for DirtyTracker {}
unsafe impl Send for DirtyTracker {}

impl DirtyTracker {
    fn new(backend: DirtyTracking, start: Address, end: Address) -> Self {
        let span_pages = (end - start) >> LOG_BYTES_IN_PAGE;
        let words = span_pages.div_ceil(64);
        let mut user_bitmap = Vec::new();
        if matches!(backend, DirtyTracking::Uffd | DirtyTracking::Segv) {
            user_bitmap.resize_with(words, || AtomicU64::new(0));
        }
        let tracker = Self {
            backend,
            span_start: start,
            span_pages,
            user_bitmap,
            registered: Mutex::new(HashSet::new()),
            registered_bits: (0..(span_pages << LOG_BYTES_IN_PAGE >> 22).div_ceil(64).max(1))
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect(),
            dirty_chunk_bits: (0..(span_pages << LOG_BYTES_IN_PAGE >> 22).div_ceil(64).max(1))
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect(),
            uffd: if backend == DirtyTracking::Uffd {
                uffd::UffdState::open()
            } else {
                uffd::UffdState::disabled()
            },
            bpf: if backend == DirtyTracking::Bpf {
                bpf::BpfShim::load(start, end - start)
            } else {
                bpf::BpfShim::disabled()
            },
        };
        if backend == DirtyTracking::Segv {
            segv::install_handler();
        }
        info!(
            "dirty tracking: backend={:?} span={}..{} ({} pages)",
            backend, start, end, span_pages
        );
        tracker
    }

    pub fn backend(&self) -> DirtyTracking {
        self.backend
    }

    fn page_index(&self, addr: Address) -> usize {
        (addr - self.span_start) >> LOG_BYTES_IN_PAGE
    }

    fn in_span(&self, addr: Address) -> bool {
        addr >= self.span_start && self.page_index(addr) < self.span_pages
    }

    /// Mark a page dirty in the user bitmap (Uffd handler thread / Segv
    /// signal handler).
    pub(crate) fn mark_dirty(&self, page: Address) {
        let idx = self.page_index(page);
        self.user_bitmap[idx >> 6].fetch_or(1 << (idx & 63), Ordering::Relaxed);
    }

    /// Record that a chunk's protection was (or will be) dropped this
    /// cycle: dirty-page faults land here at drain time, and the GC copy
    /// allocator's acquire-block hook adds promotion targets.  end_of_gc
    /// re-arms ONLY these chunks (O(dirty) instead of O(mature)), leaving
    /// never-written chunks protected across GCs.
    pub(crate) fn note_unprotected_range(&self, start: Address, bytes: usize) {
        const CHUNK: usize = 4 << 20;
        let mut a = start.align_down(CHUNK);
        let end = (start + bytes).align_up(CHUNK);
        while a < end {
            let c = (a - self.span_start.align_down(CHUNK)) >> 22;
            let w = c >> 6;
            if w < self.dirty_chunk_bits.len() {
                // Skip the RMW when already set (the common case for hot
                // chunks) — a shared-line atomic per promoted block was a
                // measurable regression.
                if self.dirty_chunk_bits[w].load(Ordering::Relaxed) & (1 << (c & 63)) == 0 {
                    self.dirty_chunk_bits[w].fetch_or(1 << (c & 63), Ordering::Relaxed);
                }
            }
            a = a + CHUNK;
        }
    }

    /// Take the set of chunks needing re-protection this cycle.
    pub(crate) fn take_dirty_chunks(&self) -> Vec<Address> {
        const CHUNK: usize = 4 << 20;
        let base = self.span_start.align_down(CHUNK);
        let mut out = Vec::new();
        for (w, word) in self.dirty_chunk_bits.iter().enumerate() {
            let mut v = word.swap(0, Ordering::Relaxed);
            while v != 0 {
                let bit = v.trailing_zeros() as usize;
                v &= v - 1;
                out.push(base + (((w << 6) | bit) << 22));
            }
        }
        out
    }

    /// Unprotect a promotion block (GC copy allocator acquire hook) and
    /// remember its chunk for re-arming.
    pub(crate) fn unprotect_copy_block(&self, start: Address, bytes: usize) {
        self.ensure_registered_range(start, bytes);
        self.unprotect(start, bytes);
        self.note_unprotected_range(start, bytes);
    }

    /// Register a range with the kernel mechanism if not yet registered.
    /// Ranges are tracked by their start address; callers must pass stable
    /// (chunk-aligned) ranges.
    pub(crate) fn ensure_registered(&self, start: Address, bytes: usize) {
        if self.backend == DirtyTracking::Segv {
            return; // mprotect needs no registration
        }
        // Lock-free fast path: bit set only after successful registration.
        const CHUNK: usize = 4 << 20;
        let c = (start.align_down(CHUNK) - self.span_start.align_down(CHUNK)) >> 22;
        let w = c >> 6;
        if w < self.registered_bits.len()
            && self.registered_bits[w].load(Ordering::Acquire) & (1 << (c & 63)) != 0
        {
            return;
        }
        let mut reg = self.registered.lock().unwrap();
        if reg.contains(&start) {
            return;
        }
        match self.backend {
            DirtyTracking::Uffd => self.uffd.register(start, bytes),
            DirtyTracking::Bpf => self.bpf.register(start, bytes),
            _ => unreachable!(),
        }
        reg.insert(start);
        if w < self.registered_bits.len() {
            self.registered_bits[w].fetch_or(1 << (c & 63), Ordering::Release);
        }
    }

    /// Register every 4 MiB-aligned chunk overlapping the range (chunk
    /// starts are stable keys across GCs, unlike LOS object ranges).
    pub(crate) fn ensure_registered_range(&self, start: Address, bytes: usize) {
        const CHUNK: usize = 4 << 20;
        let mut a = start.align_down(CHUNK);
        let end = (start + bytes).align_up(CHUNK);
        while a < end {
            self.ensure_registered(a, CHUNK);
            a = a + CHUNK;
        }
    }

    /// Write-protect a range. The range must have been registered.
    pub(crate) fn protect(&self, start: Address, bytes: usize) {
        match self.backend {
            DirtyTracking::Uffd => self.uffd.writeprotect(start, bytes, true),
            DirtyTracking::Bpf => self.bpf.writeprotect(start, bytes, true),
            DirtyTracking::Segv => segv::set_prot(start, bytes, false),
            _ => unreachable!(),
        }
    }

    /// Remove write protection from a range.
    pub(crate) fn unprotect(&self, start: Address, bytes: usize) {
        match self.backend {
            DirtyTracking::Uffd => self.uffd.writeprotect(start, bytes, false),
            DirtyTracking::Bpf => self.bpf.writeprotect(start, bytes, false),
            DirtyTracking::Segv => segv::set_prot(start, bytes, true),
            _ => unreachable!(),
        }
    }

    /// Drain the dirty-page set, invoking `visit` with each dirty page's
    /// address and clearing the set.  Must only run while mutators are
    /// suspended.
    pub(crate) fn drain_dirty<F: FnMut(Address)>(&self, mut visit: F) -> usize {
        let mut count = 0;
        let words = self.span_pages.div_ceil(64);
        for w in 0..words {
            let val = match self.backend {
                DirtyTracking::Bpf => self.bpf.take_bitmap_word(w),
                _ => self.user_bitmap[w].swap(0, Ordering::Relaxed),
            };
            let mut v = val;
            while v != 0 {
                let bit = v.trailing_zeros() as usize;
                v &= v - 1;
                count += 1;
                let page = self.span_start + ((w << 6 | bit) << LOG_BYTES_IN_PAGE);
                // A dirty page's fault dropped its protection: its chunk
                // needs re-arming at end_of_gc (dirty-chunk re-arm policy).
                self.note_unprotected_range(page, 1 << LOG_BYTES_IN_PAGE);
                visit(page);
            }
        }
        count
    }
}

/* ------------------------------------------------------------------ */
/*  userfaultfd backend                                                */
/* ------------------------------------------------------------------ */

mod uffd {
    use super::*;

    const UFFD_API: u64 = 0xAA;
    const UFFDIO_API: u64 = 0xc018_aa3f;
    const UFFDIO_REGISTER: u64 = 0xc020_aa00;
    const UFFDIO_WRITEPROTECT: u64 = 0xc018_aa06;
    const UFFD_FEATURE_PAGEFAULT_FLAG_WP: u64 = 1 << 0;
    const UFFDIO_REGISTER_MODE_WP: u64 = 1 << 1;
    const UFFDIO_WRITEPROTECT_MODE_WP: u64 = 1 << 0;
    const UFFD_EVENT_PAGEFAULT: u8 = 0x12;

    #[repr(C)]
    struct UffdioApi {
        api: u64,
        features: u64,
        ioctls: u64,
    }
    #[repr(C)]
    struct UffdioRange {
        start: u64,
        len: u64,
    }
    #[repr(C)]
    struct UffdioRegister {
        range: UffdioRange,
        mode: u64,
        ioctls: u64,
    }
    #[repr(C)]
    struct UffdioWriteprotect {
        range: UffdioRange,
        mode: u64,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct UffdMsg {
        event: u8,
        _reserved1: u8,
        _reserved2: u16,
        _reserved3: u32,
        /// pagefault: { flags: u64, address: u64, feat: u32 }
        arg: [u64; 3],
    }

    pub(super) struct UffdState {
        fd: i32,
    }

    impl UffdState {
        pub fn disabled() -> Self {
            Self { fd: -1 }
        }

        pub fn open() -> Self {
            let fd = unsafe {
                libc::syscall(libc::SYS_userfaultfd, libc::O_CLOEXEC | libc::O_NONBLOCK) as i32
            };
            assert!(fd >= 0, "userfaultfd() failed: {}", errno());
            let mut api = UffdioApi {
                api: UFFD_API,
                features: UFFD_FEATURE_PAGEFAULT_FLAG_WP,
                ioctls: 0,
            };
            let r = unsafe { libc::ioctl(fd, UFFDIO_API, &mut api) };
            assert_eq!(r, 0, "UFFDIO_API failed: {}", errno());

            // Handler thread: resolves WP faults and marks pages dirty.
            std::thread::Builder::new()
                .name("mmtk-uffd".into())
                .spawn(move || handler_loop(fd))
                .unwrap();
            Self { fd }
        }

        pub fn register(&self, start: Address, bytes: usize) {
            let mut reg = UffdioRegister {
                range: UffdioRange {
                    start: start.as_usize() as u64,
                    len: bytes as u64,
                },
                mode: UFFDIO_REGISTER_MODE_WP,
                ioctls: 0,
            };
            let r = unsafe { libc::ioctl(self.fd, UFFDIO_REGISTER, &mut reg) };
            assert_eq!(r, 0, "UFFDIO_REGISTER({}, {}) failed: {}", start, bytes, errno());
        }

        pub fn writeprotect(&self, start: Address, bytes: usize, enable: bool) {
            let mut wp = UffdioWriteprotect {
                range: UffdioRange {
                    start: start.as_usize() as u64,
                    len: bytes as u64,
                },
                mode: if enable { UFFDIO_WRITEPROTECT_MODE_WP } else { 0 },
            };
            let r = unsafe { libc::ioctl(self.fd, UFFDIO_WRITEPROTECT, &mut wp) };
            assert_eq!(
                r, 0,
                "UFFDIO_WRITEPROTECT({}, {}, {}) failed: {}",
                start, bytes, enable, errno()
            );
        }
    }

    fn handler_loop(fd: i32) {
        loop {
            let mut pfd = libc::pollfd {
                fd,
                events: libc::POLLIN,
                revents: 0,
            };
            let n = unsafe { libc::poll(&mut pfd, 1, 1000) };
            if n <= 0 {
                continue;
            }
            let mut msg = UffdMsg {
                event: 0,
                _reserved1: 0,
                _reserved2: 0,
                _reserved3: 0,
                arg: [0; 3],
            };
            let r = unsafe {
                libc::read(
                    fd,
                    &mut msg as *mut UffdMsg as *mut libc::c_void,
                    std::mem::size_of::<UffdMsg>(),
                )
            };
            if r <= 0 || msg.event != UFFD_EVENT_PAGEFAULT {
                continue;
            }
            let addr = unsafe {
                Address::from_usize(msg.arg[1] as usize & !(BYTES_IN_PAGE - 1))
            };
            let tracker = dirty_tracker().unwrap();
            if tracker.in_span(addr) {
                tracker.mark_dirty(addr);
            }
            tracker.uffd.writeprotect(addr, BYTES_IN_PAGE, false);
        }
    }

    fn errno() -> i32 {
        unsafe { *libc::__errno_location() }
    }
}

/* ------------------------------------------------------------------ */
/*  SIGSEGV + mprotect backend                                         */
/* ------------------------------------------------------------------ */

mod segv {
    use super::*;
    use std::mem::MaybeUninit;
    use std::sync::atomic::AtomicBool;

    static OLD_ACTION: Mutex<Option<libc::sigaction>> = Mutex::new(None);
    static INSTALLED: AtomicBool = AtomicBool::new(false);
    // The signal handler cannot take locks; keep a raw copy for it.
    // Written once at install time (before any fault can occur), read-only
    // afterwards.
    #[allow(static_mut_refs)]
    static mut OLD_ACTION_RAW: MaybeUninit<libc::sigaction> = MaybeUninit::uninit();

    pub(super) fn install_handler() {
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = handler as usize;
            sa.sa_flags = libc::SA_SIGINFO;
            libc::sigemptyset(&mut sa.sa_mask);
            let mut old: libc::sigaction = std::mem::zeroed();
            let r = libc::sigaction(libc::SIGSEGV, &sa, &mut old);
            assert_eq!(r, 0, "sigaction(SIGSEGV) failed");
            OLD_ACTION_RAW.write(old);
            *OLD_ACTION.lock().unwrap() = Some(old);
            INSTALLED.store(true, Ordering::SeqCst);
        }
    }

    pub(super) fn set_prot(start: Address, bytes: usize, writable: bool) {
        let prot = if writable {
            libc::PROT_READ | libc::PROT_WRITE
        } else {
            libc::PROT_READ
        };
        let r = unsafe { libc::mprotect(start.to_mut_ptr(), bytes, prot) };
        assert_eq!(r, 0, "mprotect({}, {}, {}) failed", start, bytes, writable);
    }

    extern "C" fn handler(
        sig: libc::c_int,
        info: *mut libc::siginfo_t,
        ctx: *mut libc::c_void,
    ) {
        unsafe {
            let addr = Address::from_usize((*info).si_addr() as usize);
            if let Some(tracker) = dirty_tracker() {
                let page = addr.align_down(BYTES_IN_PAGE);
                if tracker.in_span(page) {
                    // Note: only writes can fault here (pages are PROT_READ),
                    // so any fault in span is a write barrier hit.
                    tracker.mark_dirty(page);
                    set_prot(page, BYTES_IN_PAGE, true);
                    return;
                }
            }
            // Not ours: chain to the previously installed handler (HotSpot's).
            let old = OLD_ACTION_RAW.assume_init_ref();
            if old.sa_flags & libc::SA_SIGINFO != 0 {
                let f: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void) =
                    std::mem::transmute(old.sa_sigaction);
                f(sig, info, ctx);
            } else if old.sa_sigaction == libc::SIG_DFL {
                libc::signal(libc::SIGSEGV, libc::SIG_DFL);
                libc::raise(libc::SIGSEGV);
            } else if old.sa_sigaction != libc::SIG_IGN {
                let f: extern "C" fn(libc::c_int) = std::mem::transmute(old.sa_sigaction);
                f(sig);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  bpf_fault backend (via dlopen'ed C shim)                           */
/* ------------------------------------------------------------------ */

mod bpf {
    use super::*;
    use std::ffi::CString;

    type InitFn = unsafe extern "C" fn(u64, u64) -> i32;
    type RangeFn = unsafe extern "C" fn(u64, u64) -> i32;
    type WpFn = unsafe extern "C" fn(u64, u64, i32) -> i32;
    type BitmapFn = unsafe extern "C" fn() -> *mut u64;

    pub(super) struct BpfShim {
        register: Option<RangeFn>,
        wp: Option<WpFn>,
        bitmap: *mut u64,
    }

    impl BpfShim {
        pub fn disabled() -> Self {
            Self {
                register: None,
                wp: None,
                bitmap: std::ptr::null_mut(),
            }
        }

        pub fn load(span_start: Address, span_bytes: usize) -> Self {
            let path = std::env::var("MMTK_BPF_SHIM")
                .unwrap_or_else(|_| "/mydata/gc-bpf-fault/shim/libgcbpf.so".to_string());
            let cpath = CString::new(path.clone()).unwrap();
            let handle = unsafe { libc::dlopen(cpath.as_ptr(), libc::RTLD_NOW) };
            assert!(
                !handle.is_null(),
                "dirty tracking: failed to dlopen bpf shim at {}",
                path
            );
            let sym = |name: &str| -> *mut libc::c_void {
                let cname = CString::new(name).unwrap();
                let p = unsafe { libc::dlsym(handle, cname.as_ptr()) };
                assert!(!p.is_null(), "bpf shim: missing symbol {}", name);
                p
            };
            unsafe {
                let init: InitFn = std::mem::transmute(sym("gcbpf_init"));
                let register: RangeFn = std::mem::transmute(sym("gcbpf_register"));
                let wp: WpFn = std::mem::transmute(sym("gcbpf_wp"));
                let bitmap_fn: BitmapFn = std::mem::transmute(sym("gcbpf_bitmap"));
                let r = init(span_start.as_usize() as u64, span_bytes as u64);
                assert_eq!(r, 0, "gcbpf_init failed: {}", r);
                let bitmap = bitmap_fn();
                assert!(!bitmap.is_null(), "gcbpf_bitmap returned NULL");
                Self {
                    register: Some(register),
                    wp: Some(wp),
                    bitmap,
                }
            }
        }

        pub fn register(&self, start: Address, bytes: usize) {
            let r = unsafe {
                (self.register.unwrap())(start.as_usize() as u64, bytes as u64)
            };
            assert_eq!(r, 0, "gcbpf_register({}, {}) failed: {}", start, bytes, r);
        }

        pub fn writeprotect(&self, start: Address, bytes: usize, enable: bool) {
            let r = unsafe {
                (self.wp.unwrap())(start.as_usize() as u64, bytes as u64, enable as i32)
            };
            assert_eq!(r, 0, "gcbpf_wp({}, {}, {}) failed: {}", start, bytes, enable, r);
        }

        /// Read-and-clear one word of the shim's mmaped dirty bitmap.
        pub fn take_bitmap_word(&self, word: usize) -> u64 {
            unsafe {
                let p = self.bitmap.add(word);
                let v = std::ptr::read_volatile(p);
                if v != 0 {
                    std::ptr::write_volatile(p, 0);
                }
                v
            }
        }
    }
}
