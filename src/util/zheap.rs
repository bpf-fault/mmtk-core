//! Compressed cold heap (idea 6): pages of the mature space that stay
//! clean for `MMTK_ZHEAP` consecutive GCs are compressed (userspace,
//! zero-suppression codec) into a BPF arena and released with
//! MADV_DONTNEED; the first later access materializes the page via an
//! in-kernel decompressing missing-fault handler (gc_z_ops, ~5us p50).
//! GC tracing included: a trace into a compressed page simply
//! decompresses it.
//!
//! Cold detection uses soft-dirty (pagemap bit 55 + /proc/self/clear_refs
//! "4"): passive, and — unlike the Class A WP link — able to coexist
//! with the missing-fault registration (the kernel allows one fault-ops
//! mode per range: WP or MISSING, not both).
//!
//! Known v1 hazard (documented, not yet closed): a page freed by the
//! allocator while compressed keeps its offset-table entry; if MMTk ever
//! relies on kernel zero-fill for such a page, the stale image would
//! resurrect.  Sweeps invalidate entries for pages that are absent
//! without being compressed and for decompressed pages, and the K-clean
//! threshold keeps churn out of the compressed set.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::util::Address;

pub const BYTES_IN_PAGE: usize = 4096;
const CHUNK: usize = 4 << 20;

struct Zshim {
    register: unsafe extern "C" fn(u64, u64) -> i32,
    compress_page: unsafe extern "C" fn(u64) -> i64,
    invalidate: unsafe extern "C" fn(u64),
    is_compressed: unsafe extern "C" fn(u64) -> i32,
    stats: unsafe extern "C" fn(*mut u64, *mut u64) -> u64,
}

fn sym<T>(name: &str) -> T {
    unsafe {
        let c = std::ffi::CString::new(name).unwrap();
        let p = libc::dlsym(libc::RTLD_DEFAULT as *mut libc::c_void, c.as_ptr());
        assert!(!p.is_null(), "zheap: missing shim symbol {}", name);
        std::mem::transmute_copy(&p)
    }
}

pub struct ZHeap {
    base: Address,
    span: usize,
    threshold: u8,
    /// Sweep every Nth GC: clear_refs write-protects every PTE in the
    /// process (soft-dirty tracking), so each sweep makes the hot set
    /// refault once per page -- measured at ~20% throughput when
    /// sweeping every GC.  N divides that cost.
    every: u64,
    streak: Vec<AtomicU8>,
    compressed: Vec<AtomicU64>,
    registered: Mutex<HashSet<usize>>,
    shim: Zshim,
    pagemap: Mutex<File>,
    log: bool,
    gcs: AtomicU64,
    pages_compressed: AtomicU64,
    pages_resident_saved: AtomicU64,
    /// LOS object extents are immutable while live: cache them so the
    /// end-of-gc enumeration never dereferences object headers (reading
    /// the header faulted back exactly one compressed page per cold LOS
    /// object per sweep -- the measured 16k-page thrash oscillation).
    pub los_extents: Mutex<std::collections::HashMap<usize, usize>>,
}

unsafe impl Sync for ZHeap {}
unsafe impl Send for ZHeap {}

static ZHEAP: OnceLock<ZHeap> = OnceLock::new();
static ACTIVE: AtomicBool = AtomicBool::new(false);

pub fn zheap() -> Option<&'static ZHeap> {
    if ACTIVE.load(Ordering::Relaxed) {
        ZHEAP.get()
    } else {
        None
    }
}

pub fn init_zheap(start: Address, end: Address) {
    let Ok(k) = std::env::var("MMTK_ZHEAP") else {
        return;
    };
    let threshold: u8 = k.parse().unwrap_or(2);
    let every: u64 = std::env::var("MMTK_ZHEAP_EVERY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let span = end - start;
    unsafe {
        let path = std::ffi::CString::new(
            std::env::var("MMTK_GCBPF_SHIM")
                .unwrap_or_else(|_| "/mydata/gc-bpf-fault/shim/libgcbpf.so".into()),
        )
        .unwrap();
        let h = libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL);
        assert!(!h.is_null(), "zheap: dlopen shim failed");
        let init: unsafe extern "C" fn(u64, u64, u64) -> i32 = sym("gcz_init");
        // modest store: the point is net RSS reduction, not coverage
        let store: usize = std::env::var("MMTK_ZHEAP_STORE_MB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256)
            << 20;
        assert_eq!(
            init(start.as_usize() as u64, span as u64, store as u64),
            0,
            "gcz_init failed"
        );
        // selectivity: commit only pages compressing to <= this many bytes
        let max_size: u64 = std::env::var("MMTK_ZHEAP_MAXSZ")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048);
        let set_max: unsafe extern "C" fn(u64) = sym("gcz_set_max_size");
        set_max(max_size);
    }
    let npages = span >> 12;
    let _ = ZHEAP.set(ZHeap {
        base: start,
        span,
        threshold,
        every,
        streak: (0..npages).map(|_| AtomicU8::new(0)).collect(),
        compressed: (0..npages.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(),
        registered: Mutex::new(HashSet::new()),
        shim: Zshim {
            register: sym("gcz_register"),
            compress_page: sym("gcz_compress_page"),
            invalidate: sym("gcz_invalidate"),
            is_compressed: sym("gcz_is_compressed"),
            stats: sym("gcz_stats"),
        },
        pagemap: Mutex::new(File::open("/proc/self/pagemap").expect("pagemap")),
        log: std::env::var_os("MMTK_ZHEAP_LOG").is_some(),
        gcs: AtomicU64::new(0),
        pages_compressed: AtomicU64::new(0),
        pages_resident_saved: AtomicU64::new(0),
        los_extents: Mutex::new(std::collections::HashMap::new()),
    });
    ACTIVE.store(true, Ordering::Relaxed);
    eprintln!("[zheap] active: threshold={} span={}MB", threshold, span >> 20);
}

impl ZHeap {
    fn bit(&self, idx: usize) -> bool {
        self.compressed[idx >> 6].load(Ordering::Relaxed) & (1 << (idx & 63)) != 0
    }
    fn set_bit(&self, idx: usize, v: bool) {
        if v {
            self.compressed[idx >> 6].fetch_or(1 << (idx & 63), Ordering::Relaxed);
        } else {
            self.compressed[idx >> 6].fetch_and(!(1 << (idx & 63)), Ordering::Relaxed);
        }
    }

    /// End-of-GC sweep over mature chunks: age clean pages, compress the
    /// cold ones, then reset soft-dirty for the next window.
    pub fn sweep(&self, chunks: &[(Address, usize)]) {
        let gc = self.gcs.fetch_add(1, Ordering::Relaxed) + 1;
        if gc % self.every != 0 {
            return;
        }
        let mut pm = self.pagemap.lock().unwrap();
        let mut newly = 0u64;
        let mut decompressed = 0u64;
        for &(start, bytes) in chunks {
            let pages = bytes >> 12;
            let mut buf = vec![0u8; pages * 8];
            if pm
                .seek(SeekFrom::Start(((start.as_usize() >> 12) * 8) as u64))
                .is_err()
                || pm.read_exact(&mut buf).is_err()
            {
                continue;
            }
            for p in 0..pages {
                let ent = u64::from_le_bytes(buf[p * 8..p * 8 + 8].try_into().unwrap());
                let present = ent >> 63 & 1 == 1;
                let soft_dirty = ent >> 55 & 1 == 1;
                let addr = start + (p << 12);
                let idx = (addr - self.base) >> 12;
                if self.bit(idx) {
                    if present {
                        // decompressed since last sweep: hot again
                        self.set_bit(idx, false);
                        unsafe { (self.shim.invalidate)(addr.as_usize() as u64) };
                        self.streak[idx].store(0, Ordering::Relaxed);
                        decompressed += 1;
                    }
                    // else: still compressed, leave alone
                    continue;
                }
                if !present {
                    // absent and not ours: make sure no stale image exists
                    unsafe { (self.shim.invalidate)(addr.as_usize() as u64) };
                    self.streak[idx].store(0, Ordering::Relaxed);
                    continue;
                }
                if soft_dirty {
                    self.streak[idx].store(0, Ordering::Relaxed);
                    continue;
                }
                let s = self.streak[idx].load(Ordering::Relaxed).saturating_add(1);
                self.streak[idx].store(s, Ordering::Relaxed);
                if s >= self.threshold {
                    let chunk = addr.as_usize() & !(CHUNK - 1);
                    {
                        let mut reg = self.registered.lock().unwrap();
                        if reg.insert(chunk) {
                            unsafe {
                                assert_eq!(
                                    (self.shim.register)(chunk as u64, CHUNK as u64),
                                    0,
                                    "gcz register"
                                );
                            }
                        }
                    }
                    let r = unsafe { (self.shim.compress_page)(addr.as_usize() as u64) };
                    if r > 0 {
                        self.set_bit(idx, true);
                        self.streak[idx].store(0, Ordering::Relaxed);
                        newly += 1;
                    }
                }
            }
        }
        drop(pm);
        // reset soft-dirty for the next observation window
        if let Ok(mut f) = OpenOptions::new().write(true).open("/proc/self/clear_refs") {
            let _ = f.write_all(b"4");
        }
        self.pages_compressed.fetch_add(newly, Ordering::Relaxed);
        if self.log {
            let (mut orig, mut faults) = (0u64, 0u64);
            let comp = unsafe { (self.shim.stats)(&mut orig, &mut faults) };
            eprintln!(
                "[zheap] gc={} newly={} decomp={} total_comp={}KB orig={}KB faults={}",
                gc,
                newly,
                decompressed,
                comp >> 10,
                orig >> 10,
                faults
            );
        }
        let _ = self.pages_resident_saved;
        let _ = self.span;
        let _ = self.shim.is_compressed;
    }
}
