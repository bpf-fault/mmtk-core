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
//! The `Uffd` backend uses userfaultfd write-protect mode with a dedicated
//! handler thread (WP faults cannot use SIGBUS self-service).

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
    /// Dirty bitmap, written by the uffd handler thread.
    user_bitmap: Vec<AtomicU64>,
    /// Page-range starts (chunk granularity) already registered with the
    /// kernel mechanism.
    registered: Mutex<HashSet<Address>>,
    uffd: uffd::UffdState,
}

// The shim/uffd fds and pointers are only used in thread-safe ways.
unsafe impl Sync for DirtyTracker {}
unsafe impl Send for DirtyTracker {}

impl DirtyTracker {
    fn new(backend: DirtyTracking, start: Address, end: Address) -> Self {
        let span_pages = (end - start) >> LOG_BYTES_IN_PAGE;
        let words = span_pages.div_ceil(64);
        let mut user_bitmap = Vec::new();
        if backend == DirtyTracking::Uffd {
            user_bitmap.resize_with(words, || AtomicU64::new(0));
        }
        let tracker = Self {
            backend,
            span_start: start,
            span_pages,
            user_bitmap,
            registered: Mutex::new(HashSet::new()),
            uffd: if backend == DirtyTracking::Uffd {
                uffd::UffdState::open()
            } else {
                uffd::UffdState::disabled()
            },
        };
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

    /// Register a range with the kernel mechanism if not yet registered.
    /// Ranges are tracked by their start address; callers must pass stable
    /// (chunk-aligned) ranges.
    pub(crate) fn ensure_registered(&self, start: Address, bytes: usize) {
        let mut reg = self.registered.lock().unwrap();
        if reg.contains(&start) {
            return;
        }
        match self.backend {
            DirtyTracking::Uffd => self.uffd.register(start, bytes),
            _ => unreachable!(),
        }
        reg.insert(start);
    }

    /// Write-protect a range. The range must have been registered.
    pub(crate) fn protect(&self, start: Address, bytes: usize) {
        match self.backend {
            DirtyTracking::Uffd => self.uffd.writeprotect(start, bytes, true),
            _ => unreachable!(),
        }
    }

    /// Remove write protection from a range.
    pub(crate) fn unprotect(&self, start: Address, bytes: usize) {
        match self.backend {
            DirtyTracking::Uffd => self.uffd.writeprotect(start, bytes, false),
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
            let val = self.user_bitmap[w].swap(0, Ordering::Relaxed);
            let mut v = val;
            while v != 0 {
                let bit = v.trailing_zeros() as usize;
                v &= v - 1;
                count += 1;
                visit(self.span_start + ((w << 6 | bit) << LOG_BYTES_IN_PAGE));
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
