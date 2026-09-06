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

const STATE_ZERO_FILL: u64 = 0;
const STATE_STAGED: u64 = 1;

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
}

pub(crate) struct CompactFaults {
    backend: CompactFaultsBackend,
    space_base: Address,
    span: usize,
    arena_base: Address,
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
                    state,
                    uffd: -1,
                    shim: Some(shim),
                }
            }
            CompactFaultsBackend::Uffd => {
                let arena = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        span,
                        libc::PROT_NONE,
                        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                        -1,
                        0,
                    )
                };
                assert!(arena != libc::MAP_FAILED, "uffd arena mmap failed");
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
                let r = self.shim.as_ref().unwrap().flip(start, bytes);
                assert_eq!(r, 0, "gcb0_flip({}, {}) failed", start, bytes);
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
        // EEXIST: page already present (e.g. raced install) — fine.
        assert_eq!(errno, libc::EEXIST, "UFFDIO_COPY({}) failed: {}", dst, errno);
    }
}

/* ---------------- bpf shim (dlopen) ---------------- */

mod bpf_shim {
    use super::*;
    use std::ffi::CString;

    type InitFn = unsafe extern "C" fn(u64, u64) -> u64;
    type FlipFn = unsafe extern "C" fn(u64, u64) -> i32;
    type StateFn = unsafe extern "C" fn() -> *mut u64;

    pub(super) struct Shim {
        init: InitFn,
        flip: FlipFn,
        state: StateFn,
    }

    impl Shim {
        pub fn load() -> Self {
            let path = std::env::var("MMTK_BPF_SHIM")
                .unwrap_or_else(|_| "libgcbpf.so".to_string());
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
                    state: std::mem::transmute(sym("gcb0_state")),
                }
            }
        }

        pub fn init(&self, base: Address, span: usize) -> Address {
            unsafe { Address::from_usize((self.init)(base.as_usize() as u64, span as u64) as usize) }
        }

        pub fn flip(&self, start: Address, bytes: usize) -> i32 {
            unsafe { (self.flip)(start.as_usize() as u64, bytes as u64) }
        }

        pub fn state(&self) -> *mut u64 {
            unsafe { (self.state)() }
        }
    }
}
