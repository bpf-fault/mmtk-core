//! Fault-driven Compressor compaction (Class B).
//!
//! When the `compact_faults` option selects a backend, the Compressor's
//! per-region compaction is restructured ART-CMC-style:
//!
//! 1. *Flip*: the region's physical pages are moved to a linear from-space
//!    arena via `mremap(MREMAP_DONTUNMAP)` (`arena + (addr - space_base)`),
//!    and the emptied region is registered for missing-fault handling.
//! 2. *Stage*: objects are slid/compacted **inside the arena** (reads and
//!    writes go to alias addresses; all forwarding metadata is queried with
//!    real heap addresses), and each compacted page's state is set to
//!    `Staged`.
//! 3. *Install*: a staged page materializes on first access — the bpf
//!    backend copies arena→page in-kernel inside the fault; the uffd
//!    backend installs via `UFFDIO_COPY`.  Pages beyond the compacted
//!    cursor stay state-0 and zero-fill on demand.
//!
//! B.0 (current): installation happens immediately, still inside the STW
//! pause, validating the mechanism.  B.1 will resume mutators after the
//! root-update pass and let the sweep thread / faulting mutators race.

use crate::util::options::CompactFaults as CompactFaultsBackend;
use crate::util::Address;
use std::sync::OnceLock;

pub(crate) const LOG_BYTES_IN_PAGE: usize = 12;
pub(crate) const BYTES_IN_PAGE: usize = 1 << LOG_BYTES_IN_PAGE;

/// Compressor region granularity (registration tracking unit).
const REGION_BYTES: usize = 1 << 20;

const STATE_ZERO_FILL: u64 = 0;
const STATE_STAGED: u64 = 1;
const STATE_PENDING: u64 = 2;

/// Is a concurrent compaction window currently open?
static WINDOW_OPEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Regions not yet staged+installed in the current window.
static WINDOW_REMAINING: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

static TRACKER: OnceLock<CompactFaults> = OnceLock::new();

pub fn is_compact_faults_active() -> bool {
    TRACKER.get().is_some()
}

pub(crate) fn compact_faults() -> Option<&'static CompactFaults> {
    TRACKER.get()
}

pub(crate) fn init_compact_faults(
    backend: CompactFaultsBackend,
    start: Address,
    end: Address,
) {
    if backend == CompactFaultsBackend::None {
        return;
    }
    let span = end - start;
    assert!(
        span <= 64 << 30,
        "compact_faults: heap span {} too large; use compressed oops",
        span
    );
    TRACKER
        .set(CompactFaults::new(backend, start, span))
        .ok()
        .expect("compact faults initialized twice");
    sigbus::install_handler();
}

pub(crate) struct CompactFaults {
    backend: CompactFaultsBackend,
    space_base: Address,
    span: usize,
    arena_base: Address,
    /// Range starts already registered with bpf_fault (registration
    /// persists across cycles; uffd re-registers every cycle because
    /// finish_region unregisters).
    registered: std::sync::Mutex<std::collections::HashSet<Address>>,
    /// bpf: pointer into the shim's mmaped page_state map.
    /// uffd: our own state array.
    state: *mut u64,
    uffd: i32,
    shim: Option<bpf_shim::Shim>,
}

unsafe impl Sync for CompactFaults {}
unsafe impl Send for CompactFaults {}

impl CompactFaults {
    fn new(backend: CompactFaultsBackend, space_base: Address, span: usize) -> Self {
        match backend {
            CompactFaultsBackend::Bpf => {
                let shim = bpf_shim::Shim::load();
                let arena = shim.init(space_base, span);
                assert!(!arena.is_zero(), "gcb0_init failed");
                let state = shim.state();
                Self {
                    backend,
                    space_base,
                    span,
                    arena_base: arena,
                    registered: std::sync::Mutex::new(std::collections::HashSet::new()),
                    state,
                    uffd: -1,
                    shim: Some(shim),
                }
            }
            CompactFaultsBackend::Uffd => {
                // 2 MiB phase-aligned with the heap so mremap moves whole
                // PMD tables (see the bpf shim for details).
                const PMD: usize = 2 << 20;
                let raw = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        span + PMD,
                        libc::PROT_NONE,
                        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                        -1,
                        0,
                    )
                };
                assert!(raw != libc::MAP_FAILED, "uffd arena mmap failed");
                let mut aligned = ((raw as usize + PMD - 1) & !(PMD - 1))
                    | (space_base.as_usize() & (PMD - 1));
                if aligned < raw as usize {
                    aligned += PMD;
                }
                let arena = aligned as *mut libc::c_void;
                let pages = span >> LOG_BYTES_IN_PAGE;
                let state = unsafe {
                    libc::calloc(pages, std::mem::size_of::<u64>()) as *mut u64
                };
                assert!(!state.is_null());
                let uffd = uffd_open();
                Self {
                    backend,
                    space_base,
                    span,
                    arena_base: Address::from_mut_ptr(arena),
                    registered: std::sync::Mutex::new(std::collections::HashSet::new()),
                    state,
                    uffd,
                    shim: None,
                }
            }
            CompactFaultsBackend::None => unreachable!(),
        }
    }

    /// Offset to add to a heap address to get its arena alias.
    pub fn alias_delta(&self) -> isize {
        self.arena_base.as_usize() as isize - self.space_base.as_usize() as isize
    }

    pub fn alias_of(&self, addr: Address) -> Address {
        self.arena_base + (addr - self.space_base)
    }

    /// Flip a region: move its pages to the arena and register the emptied
    /// range for missing faults.
    pub fn flip(&self, start: Address, bytes: usize) {
        debug_assert!(start >= self.space_base && start + bytes <= self.space_base + self.span);
        match self.backend {
            CompactFaultsBackend::Bpf => {
                let shim = self.shim.as_ref().unwrap();
                // The flip is fast (~1.3ms mremap + ~0.13ms register for a
                // ~540MB live heap, measured): bpf_fault registration does
                // NOT fragment the VMA (it stays a single VMA, so 2MiB PMD
                // moves apply), and a full-range MADV_DONTNEED'd arena slot
                // has empty page tables (pmd_none holds), so move_normal_pmd
                // succeeds.  No kernel change needed — the earlier ~60ms was
                // mremap(MREMAP_FIXED) tearing down the previous cycle's
                // arena pages synchronously, now released concurrently in
                // finish_region.
                let r = shim.flip(start, bytes, false);
                assert_eq!(r, 0, "gcb0_flip({}, {}) failed", start, bytes);
                let mut reg = self.registered.lock().unwrap();
                let mut subs: Vec<(Address, usize)> = vec![];
                let mut a = start;
                while a < start + bytes {
                    if reg.insert(a) {
                        match subs.last_mut() {
                            Some(l) if l.0 + l.1 == a => l.1 += REGION_BYTES,
                            _ => subs.push((a, REGION_BYTES)),
                        }
                    }
                    a = a + REGION_BYTES;
                }
                for &(s2, b2) in &subs {
                    let r = shim.register(s2, b2);
                    assert_eq!(r, 0, "gcb0_register({}, {}) failed", s2, b2);
                }
            }
            CompactFaultsBackend::Uffd => {
                let dst = self.alias_of(start);
                let r = unsafe {
                    libc::mremap(
                        start.to_mut_ptr::<libc::c_void>(),
                        bytes,
                        bytes,
                        libc::MREMAP_MAYMOVE | libc::MREMAP_FIXED | MREMAP_DONTUNMAP,
                        dst.to_mut_ptr::<libc::c_void>(),
                    )
                };
                assert!(r != libc::MAP_FAILED, "uffd flip mremap failed");
                uffd_register_missing(self.uffd, start, bytes);
            }
            CompactFaultsBackend::None => unreachable!(),
        }
    }

    /// Mark pages [start, start+bytes) as staged (arena holds final
    /// contents).
    pub fn stage(&self, start: Address, bytes: usize) {
        let first = (start - self.space_base) >> LOG_BYTES_IN_PAGE;
        let n = bytes >> LOG_BYTES_IN_PAGE;
        for i in first..first + n {
            unsafe {
                std::ptr::write_volatile(self.state.add(i), STATE_STAGED);
            }
        }
    }

    /// Mark pages as pending: live data will land there but the GC has not
    /// staged it yet.  Faults bounce to the SIGBUS handler (wait-mode).
    pub fn set_pending(&self, start: Address, bytes: usize) {
        let first = (start - self.space_base) >> LOG_BYTES_IN_PAGE;
        let n = bytes >> LOG_BYTES_IN_PAGE;
        for i in first..first + n {
            unsafe {
                std::ptr::write_volatile(self.state.add(i), STATE_PENDING);
            }
        }
    }

    fn page_state(&self, addr: Address) -> u64 {
        let idx = (addr - self.space_base) >> LOG_BYTES_IN_PAGE;
        unsafe { std::ptr::read_volatile(self.state.add(idx)) }
    }

    pub fn in_span(&self, addr: Address) -> bool {
        addr >= self.space_base && addr < self.space_base + self.span
    }

    pub fn open_window(&self, regions: usize) {
        WINDOW_REMAINING.store(regions, std::sync::atomic::Ordering::SeqCst);
        WINDOW_OPEN.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Returns true if this was the last region of the window.
    pub fn region_done(&self) -> bool {
        WINDOW_REMAINING.fetch_sub(1, std::sync::atomic::Ordering::SeqCst) == 1
    }

    pub fn close_window(&self) {
        WINDOW_OPEN.store(false, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn window_active(&self) -> bool {
        WINDOW_OPEN.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Handle a SIGBUS on `page` during the concurrent window (wait-mode):
    /// spin until the GC stages the page, then (uffd) install it.  Returns
    /// true if the fault was ours.  Runs in signal context: no locks.
    fn handle_window_fault(&self, page: Address) -> bool {
        if !self.window_active() || !self.in_span(page) {
            return false;
        }
        loop {
            match self.page_state(page) {
                STATE_STAGED => break,
                STATE_PENDING => std::hint::spin_loop(),
                // Zero-fill: bpf handles in-kernel (we never get here);
                // uffd must install a zero page.
                _ => break,
            }
        }
        if self.backend == CompactFaultsBackend::Uffd {
            match self.page_state(page) {
                STATE_STAGED => uffd_copy(self.uffd, page, self.alias_of(page), BYTES_IN_PAGE),
                _ => uffd_zeropage(self.uffd, page, BYTES_IN_PAGE),
            }
        }
        true
    }

    /// Install staged pages now (B.0 STW mode): bpf touches each page (the
    /// in-kernel handler copies from the arena); uffd UFFDIO_COPYs.
    pub fn install(&self, start: Address, bytes: usize) {
        match self.backend {
            CompactFaultsBackend::Bpf => {
                let mut a = start;
                while a < start + bytes {
                    unsafe {
                        std::ptr::read_volatile(a.to_ptr::<u8>());
                    }
                    a = a + BYTES_IN_PAGE;
                }
            }
            CompactFaultsBackend::Uffd => {
                let mut a = start;
                while a < start + bytes {
                    uffd_copy(self.uffd, a, self.alias_of(a), BYTES_IN_PAGE);
                    a = a + BYTES_IN_PAGE;
                }
            }
            CompactFaultsBackend::None => unreachable!(),
        }
    }

    /// Region finished installing (B.0): restore stock fault semantics.
    /// uffd must unregister — with UFFD_FEATURE_SIGBUS, touching an
    /// uninstalled (beyond-cursor) page would SIGBUS instead of zero-fill.
    /// bpf needs nothing: state-0 pages zero-fill in the handler.
    pub fn finish_region(&self, start: Address, bytes: usize) {
        // The region is fully installed.  Unregister it FIRST: while the
        // region stays armed, any later fault on it (kernel reclaim of an
        // installed page, then re-access; or a stray access) re-enters the
        // missing handler, which reads the about-to-be-released arena slot
        // and delivers SIGBUS.  After unregister the heap range is a normal
        // anonymous mapping; installed pages stay present, the next flip
        // re-registers.  (Minimal repro: micro/test_flip_unmap.c — armed
        // region + released arena = SIGBUS; unregister-first = PASS.)
        match self.backend {
            CompactFaultsBackend::Bpf => {
                let r = self.shim.as_ref().unwrap().unregister(start, bytes);
                assert_eq!(r, 0, "gcb0_unregister({}, {}) failed", start, bytes);
                self.registered.lock().unwrap().remove(&start);
            }
            CompactFaultsBackend::Uffd => {}
            CompactFaultsBackend::None => unreachable!(),
        }
        // Now release the arena slot's pages, concurrently with mutators, so
        // the next pause's mremap(MREMAP_FIXED) does not pay the teardown
        // (rmap removal, memcg uncharge, freeing — ~60ms for a ~540MB live
        // heap, vs ~4ms of actual page-table moves).  MADV_DONTNEED, not
        // munmap: munmapping the slot concurrently races a use of the same
        // arena address by HotSpot's resume-time DerivedPointerTable update
        // (a UAF crash); DONTNEED keeps the VMA, freeing only the pages.
        {
            let slot = self.alias_of(start);
            let r = unsafe {
                libc::madvise(
                    slot.to_mut_ptr::<libc::c_void>(),
                    bytes,
                    libc::MADV_DONTNEED,
                )
            };
            debug_assert_eq!(r, 0);
        }
        if self.backend == CompactFaultsBackend::Uffd {
            let mut range = UffdioRange {
                start: start.as_usize() as u64,
                len: bytes as u64,
            };
            let r = unsafe { libc::ioctl(self.uffd, UFFDIO_UNREGISTER, &mut range) };
            assert_eq!(r, 0, "UFFDIO_UNREGISTER({}, {}) failed", start, bytes);
        }
    }

    /// Reset state for a region (next cycle) — pages return to zero-fill.
    pub fn reset_region_state(&self, start: Address, bytes: usize) {
        let first = (start - self.space_base) >> LOG_BYTES_IN_PAGE;
        let n = bytes >> LOG_BYTES_IN_PAGE;
        for i in first..first + n {
            unsafe {
                std::ptr::write_volatile(self.state.add(i), STATE_ZERO_FILL);
            }
        }
    }
}

const MREMAP_DONTUNMAP: libc::c_int = 4;

/* ---------------- uffd helpers (missing mode + COPY) ---------------- */

const UFFD_API: u64 = 0xAA;
const UFFDIO_API: u64 = 0xc018_aa3f;
const UFFDIO_REGISTER: u64 = 0xc020_aa00;
const UFFDIO_UNREGISTER: u64 = 0x8010_aa01;
const UFFDIO_COPY: u64 = 0xc028_aa03;
const UFFDIO_ZEROPAGE: u64 = 0xc020_aa04;
const UFFDIO_REGISTER_MODE_MISSING: u64 = 1 << 0;
const UFFD_FEATURE_SIGBUS: u64 = 1 << 7;

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
struct UffdioCopy {
    dst: u64,
    src: u64,
    len: u64,
    mode: u64,
    copy: i64,
}

fn uffd_open() -> i32 {
    let fd = unsafe {
        libc::syscall(libc::SYS_userfaultfd, libc::O_CLOEXEC) as i32
    };
    assert!(fd >= 0, "userfaultfd() failed");
    // SIGBUS feature: in B.1, mutators self-handle faults on unprocessed
    // pages.  In B.0 (STW) no fault should ever reach a mutator.
    let mut api = UffdioApi {
        api: UFFD_API,
        features: UFFD_FEATURE_SIGBUS,
        ioctls: 0,
    };
    let r = unsafe { libc::ioctl(fd, UFFDIO_API, &mut api) };
    assert_eq!(r, 0, "UFFDIO_API failed");
    fd
}

fn uffd_register_missing(fd: i32, start: Address, bytes: usize) {
    let mut reg = UffdioRegister {
        range: UffdioRange {
            start: start.as_usize() as u64,
            len: bytes as u64,
        },
        mode: UFFDIO_REGISTER_MODE_MISSING,
        ioctls: 0,
    };
    let r = unsafe { libc::ioctl(fd, UFFDIO_REGISTER, &mut reg) };
    assert_eq!(r, 0, "UFFDIO_REGISTER({}, {}) failed", start, bytes);
}

#[repr(C)]
struct UffdioZeropage {
    range: UffdioRange,
    mode: u64,
    zeropage: i64,
}

fn uffd_zeropage(fd: i32, dst: Address, bytes: usize) {
    let mut zp = UffdioZeropage {
        range: UffdioRange {
            start: dst.as_usize() as u64,
            len: bytes as u64,
        },
        mode: 0,
        zeropage: 0,
    };
    let r = unsafe { libc::ioctl(fd, UFFDIO_ZEROPAGE, &mut zp) };
    if r != 0 {
        let errno = unsafe { *libc::__errno_location() };
        // EEXIST: raced install.  ENOENT: the GC installed everything and
        // unregistered the region before our handler ran — the retried
        // access proceeds normally.
        assert!(
            errno == libc::EEXIST || errno == libc::ENOENT,
            "UFFDIO_ZEROPAGE({}) failed: {}",
            dst,
            errno
        );
    }
}

fn uffd_copy(fd: i32, dst: Address, src: Address, bytes: usize) {
    let mut copy = UffdioCopy {
        dst: dst.as_usize() as u64,
        src: src.as_usize() as u64,
        len: bytes as u64,
        mode: 0,
        copy: 0,
    };
    let r = unsafe { libc::ioctl(fd, UFFDIO_COPY, &mut copy) };
    if r != 0 {
        let errno = unsafe { *libc::__errno_location() };
        // EEXIST: page already present (raced install).  ENOENT: the GC
        // installed everything and unregistered the region before our
        // handler ran.
        assert!(
            errno == libc::EEXIST || errno == libc::ENOENT,
            "UFFDIO_COPY({}) failed: {}",
            dst,
            errno
        );
    }
}

/* ---------------- bpf shim (dlopen) ---------------- */

mod bpf_shim {
    use super::*;
    use std::ffi::CString;

    type InitFn = unsafe extern "C" fn(u64, u64) -> u64;
    type FlipFn = unsafe extern "C" fn(u64, u64, i32) -> i32;
    type RangeFn = unsafe extern "C" fn(u64, u64) -> i32;
    type StateFn = unsafe extern "C" fn() -> *mut u64;

    pub(super) struct Shim {
        init: InitFn,
        flip: FlipFn,
        unmap_arena: RangeFn,
        register: RangeFn,
        unregister: RangeFn,
        state: StateFn,
    }

    impl Shim {
        pub fn load() -> Self {
            let path = std::env::var("MMTK_BPF_SHIM")
                .unwrap_or_else(|_| "/mydata/gc-bpf-fault/shim/libgcbpf.so".to_string());
            let cpath = CString::new(path.clone()).unwrap();
            let handle = unsafe { libc::dlopen(cpath.as_ptr(), libc::RTLD_NOW) };
            assert!(!handle.is_null(), "compact_faults: dlopen {} failed", path);
            let sym = |name: &str| {
                let cname = CString::new(name).unwrap();
                let p = unsafe { libc::dlsym(handle, cname.as_ptr()) };
                assert!(!p.is_null(), "shim: missing symbol {}", name);
                p
            };
            unsafe {
                Self {
                    init: std::mem::transmute(sym("gcb0_init")),
                    flip: std::mem::transmute(sym("gcb0_flip")),
                    unmap_arena: std::mem::transmute(sym("gcb0_unmap_arena")),
                    register: std::mem::transmute(sym("gcb0_register")),
                    unregister: std::mem::transmute(sym("gcb0_unregister")),
                    state: std::mem::transmute(sym("gcb0_state")),
                }
            }
        }

        pub fn init(&self, base: Address, span: usize) -> Address {
            unsafe { Address::from_usize((self.init)(base.as_usize() as u64, span as u64) as usize) }
        }

        pub fn flip(&self, start: Address, bytes: usize, do_register: bool) -> i32 {
            unsafe { (self.flip)(start.as_usize() as u64, bytes as u64, do_register as i32) }
        }

        pub fn unmap_arena(&self, start: Address, bytes: usize) -> i32 {
            unsafe { (self.unmap_arena)(start.as_usize() as u64, bytes as u64) }
        }

        pub fn register(&self, start: Address, bytes: usize) -> i32 {
            unsafe { (self.register)(start.as_usize() as u64, bytes as u64) }
        }

        pub fn unregister(&self, start: Address, bytes: usize) -> i32 {
            unsafe { (self.unregister)(start.as_usize() as u64, bytes as u64) }
        }

        pub fn state(&self) -> *mut u64 {
            unsafe { (self.state)() }
        }
    }
}


/* ---------------- chained SIGBUS handler (concurrent window) -------- */

mod sigbus {
    use super::*;
    use std::mem::MaybeUninit;

    // Written once at install time, read-only afterwards.
    #[allow(static_mut_refs)]
    static mut OLD_ACTION_RAW: MaybeUninit<libc::sigaction> = MaybeUninit::uninit();

    pub(super) fn install_handler() {
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = handler as usize;
            sa.sa_flags = libc::SA_SIGINFO;
            libc::sigemptyset(&mut sa.sa_mask);
            let mut old: libc::sigaction = std::mem::zeroed();
            let r = libc::sigaction(libc::SIGBUS, &sa, &mut old);
            assert_eq!(r, 0, "sigaction(SIGBUS) failed");
            #[allow(static_mut_refs)]
            OLD_ACTION_RAW.write(old);
        }
    }

    extern "C" fn handler(
        sig: libc::c_int,
        info: *mut libc::siginfo_t,
        ctx: *mut libc::c_void,
    ) {
        unsafe {
            let addr = Address::from_usize((*info).si_addr() as usize);
            let page = addr.align_down(BYTES_IN_PAGE);
            if let Some(t) = compact_faults() {
                if t.handle_window_fault(page) {
                    return;
                }
            }
            #[allow(static_mut_refs)]
            let old = OLD_ACTION_RAW.assume_init_ref();
            if old.sa_flags & libc::SA_SIGINFO != 0 {
                let f: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void) =
                    std::mem::transmute(old.sa_sigaction);
                f(sig, info, ctx);
            } else if old.sa_sigaction == libc::SIG_DFL {
                libc::signal(libc::SIGBUS, libc::SIG_DFL);
                libc::raise(libc::SIGBUS);
            } else if old.sa_sigaction != libc::SIG_IGN {
                let f: extern "C" fn(libc::c_int) = std::mem::transmute(old.sa_sigaction);
                f(sig);
            }
        }
    }
}
