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

// Per-region staging coordination (steal-mode): exactly one stager per
// region; a faulting mutator either stages the region itself or waits for
// the in-progress stager (bounded by one region, not the whole sweep).
const REGION_UNSTAGED: u8 = 0;
const REGION_STAGING: u8 = 1;
const REGION_DONE: u8 = 2;

/// Plan-registered handler that stages (slide-compacts + installs) one
/// region.  Lets the VM-agnostic SIGBUS handler drive the VM-specific
/// Compressor staging when a mutator faults a not-yet-staged region.
pub trait StealHandler: Send + Sync {
    fn stage(&self, region_index: usize);
    /// Class B v2 deferred forward: `buf` is a private copy of the staged
    /// arena page backing to-space `to_page`; rewrite its reference slots
    /// (located via the reference bitmap) to their forwarded values, in place.
    /// The arena itself stays un-forwarded, so this is idempotent across
    /// concurrent faults.  Only called when `defer_forward()` is set.
    fn forward_buf(&self, to_page: Address, buf: Address);
}

static STEAL: OnceLock<Box<dyn StealHandler>> = OnceLock::new();

/// Register the steal handler (idempotent; first registration wins).
pub fn register_steal_handler(h: Box<dyn StealHandler>) {
    let _ = STEAL.set(h);
}

/// Is a concurrent compaction window currently open?
static WINDOW_OPEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Regions not yet staged+installed in the current window.
static WINDOW_REMAINING: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static WINDOW_FAULTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WINDOW_SPIN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Class B v2 reference bits set this cycle (telemetry/verification).
pub(crate) static REFBITS_POPULATED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
/// Class B v2: defer reference forwarding from staging to install time,
/// driven by the reference bitmap (MMTK_COMPACT_DEFER_FORWARD).
static DEFER_FORWARD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether reference forwarding is deferred to install time (Class B v2).
pub fn defer_forward() -> bool {
    DEFER_FORWARD.load(std::sync::atomic::Ordering::Relaxed)
}

/// R1: full in-kernel compaction.  The eBPF handler builds each to-space page
/// from UN-SLID from-space (no userspace slide-compact); the GC only flips and
/// emits metadata (live-word bitmap, per-page first-source index, reference
/// bitmap at OLD positions, forward table).  MMTK_COMPACT_INKERNEL.
static INKERNEL_COMPACT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Whether full in-kernel compaction (R1) is active.
pub fn inkernel_compact() -> bool {
    INKERNEL_COMPACT.load(std::sync::atomic::Ordering::Relaxed)
}

/// R1 verification mode (MMTK_R1_VERIFY): execute the v2 userspace path
/// (so the run is correct) while also emitting R1 metadata and diffing a
/// userspace emulation of the in-kernel page builder against the staged
/// truth for every region.
static R1_VERIFY: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn r1_verify() -> bool {
    R1_VERIFY.load(std::sync::atomic::Ordering::Relaxed)
}

/// Compressed-oops base/shift, set by the VM binding so the in-kernel (bpf)
/// fixup handler can decode/encode narrow references.
static COOPS_BASE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static COOPS_SHIFT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Provide the VM's compressed-oops base and shift (Class B v2 / bpf backend).
pub fn set_compressed_oops(base: usize, shift: u32) {
    COOPS_BASE.store(base, std::sync::atomic::Ordering::Relaxed);
    COOPS_SHIFT.store(shift, std::sync::atomic::Ordering::Relaxed);
}

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
    if std::env::var_os("MMTK_COMPACT_DEFER_FORWARD").is_some() {
        DEFER_FORWARD.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if std::env::var_os("MMTK_R1_VERIFY").is_some() {
        R1_VERIFY.store(true, std::sync::atomic::Ordering::Relaxed);
        DEFER_FORWARD.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if std::env::var_os("MMTK_COMPACT_INKERNEL").is_some() {
        // In-kernel compaction implies deferred forwarding (the handler does
        // both the compaction copy and the reference forwarding).
        DEFER_FORWARD.store(true, std::sync::atomic::Ordering::Relaxed);
        INKERNEL_COMPACT.store(true, std::sync::atomic::Ordering::Relaxed);
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
    registered: Vec<std::sync::atomic::AtomicBool>,
    /// bpf: pointer into the shim's mmaped page_state map.
    /// uffd: our own state array.
    state: *mut u64,
    /// Per-region staging state (steal-mode coordination), one per
    /// REGION_BYTES of the span (addressed by (addr-base)/REGION_BYTES).
    region_state: Vec<std::sync::atomic::AtomicU8>,
    /// Per-region page-aligned post-compact cursor, set at flip time so the
    /// steal path can stage a region without taking the regions lock.
    region_cursor: Vec<std::sync::atomic::AtomicUsize>,
    uffd: i32,
    shim: Option<bpf_shim::Shim>,
    /// Class B v2 reference bitmap: 1 bit per 4-byte (compressed-oop) slot of
    /// the span, indexed by `(to_space_addr - space_base) >> 2`.  Populated at
    /// staged positions during `stage_region_idx`, so the in-kernel fixup
    /// handler can forward references without object-layout knowledge.  Lives
    /// in its own mapping (NOT MMTk side metadata, whose range does not cover
    /// the staging-arena alias addresses used during compaction).
    refbitmap: *mut u8,
    /// Class B v2 forward table: one u32 (new compressed-oop) per old slot,
    /// indexed by `(old_addr - space_base) >> coops_shift`.  Filled by the GC
    /// during `calculate_offset_vector` (each object's new position is known
    /// there for free); the in-kernel handler does a single direct lookup
    /// instead of an offset-vector transducer scan + side-metadata reads.
    fwdtable: *mut u32,
    /// R1 metadata (in the bpf arena): live-word bitmap (1 bit/8-byte word, old
    /// positions) and the per-to-space-page first-source word index.
    livebits: *mut u8,
    first_src: *mut u32,
}

unsafe impl Sync for CompactFaults {}
unsafe impl Send for CompactFaults {}

impl CompactFaults {
    fn new(backend: CompactFaultsBackend, space_base: Address, span: usize) -> Self {
        // Reference bitmap: 1 bit per 4-byte slot = span/32 bytes, lazily
        // committed.  Shared by both backends.
        let refbitmap = {
            let bytes = (span / 32 + BYTES_IN_PAGE - 1) & !(BYTES_IN_PAGE - 1);
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                    -1,
                    0,
                )
            };
            assert!(p != libc::MAP_FAILED, "refbitmap mmap failed");
            p as *mut u8
        };
        // Forward table: u32 per 8-byte slot of the span (assumes compressed-
        // oops shift >= 3), = span/2 bytes, lazily committed.
        let fwdtable = {
            let bytes = ((span >> 3) * 4 + BYTES_IN_PAGE - 1) & !(BYTES_IN_PAGE - 1);
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                    -1,
                    0,
                )
            };
            assert!(p != libc::MAP_FAILED, "fwdtable mmap failed");
            p as *mut u32
        };
        match backend {
            CompactFaultsBackend::Bpf => {
                let shim = bpf_shim::Shim::load();
                let arena = shim.init(space_base, span);
                assert!(!arena.is_zero(), "gcb0_init failed");
                let state = shim.state();
                // The forward table AND reference bitmap live in the shim's BPF
                // arena (the GC writes them; the in-kernel handler reads them
                // directly).  The anonymous mmaps above are unused for Bpf.
                let fwdtable = shim.fwdtable_base();
                let refbitmap = shim.refbits_base();
                let livebits = shim.livebits_base();
                let first_src = shim.first_src_base();
                assert!(
                    !fwdtable.is_null() && !refbitmap.is_null(),
                    "gcb0 arena bases failed"
                );
                Self {
                    backend,
                    space_base,
                    span,
                    arena_base: arena,
                    registered: (0..span / REGION_BYTES)
                        .map(|_| std::sync::atomic::AtomicBool::new(false))
                        .collect(),
                    state,
                    region_state: (0..span / REGION_BYTES)
                        .map(|_| std::sync::atomic::AtomicU8::new(REGION_DONE))
                        .collect(),
                    region_cursor: (0..span / REGION_BYTES)
                        .map(|_| std::sync::atomic::AtomicUsize::new(0))
                        .collect(),
                    uffd: -1,
                    shim: Some(shim),
                    refbitmap,
                    fwdtable,
                    livebits,
                    first_src,
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
                    registered: (0..span / REGION_BYTES)
                        .map(|_| std::sync::atomic::AtomicBool::new(false))
                        .collect(),
                    state,
                    region_state: (0..span / REGION_BYTES)
                        .map(|_| std::sync::atomic::AtomicU8::new(REGION_DONE))
                        .collect(),
                    region_cursor: (0..span / REGION_BYTES)
                        .map(|_| std::sync::atomic::AtomicUsize::new(0))
                        .collect(),
                    uffd,
                    shim: None,
                    refbitmap,
                    fwdtable,
                    livebits: std::ptr::null_mut(),
                    first_src: std::ptr::null_mut(),
                }
            }
            CompactFaultsBackend::None => unreachable!(),
        }
    }

    pub fn region_index(&self, addr: Address) -> usize {
        (addr - self.space_base) / REGION_BYTES
    }

    pub fn region_count(&self) -> usize {
        self.region_state.len()
    }

    /// Try to become the stager for a region (CAS Unstaged -> Staging).
    /// Returns true if won.
    pub fn claim_region(&self, idx: usize) -> bool {
        use std::sync::atomic::Ordering;
        self.region_state[idx]
            .compare_exchange(REGION_UNSTAGED, REGION_STAGING, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    pub fn mark_region_done(&self, idx: usize) {
        self.region_state[idx].store(REGION_DONE, std::sync::atomic::Ordering::Release);
    }

    pub fn region_is_done(&self, idx: usize) -> bool {
        self.region_state[idx].load(std::sync::atomic::Ordering::Acquire) == REGION_DONE
    }

    /// Window open: mark all regions DONE (non-live = skipped); flip then
    /// marks live ones claimable via `mark_region_live`.
    pub fn reset_region_staging(&self) {
        for r in &self.region_state {
            r.store(REGION_DONE, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Flip time: record a live region (claimable, with its preset cursor).
    pub fn mark_region_live(&self, start: Address, cursor: Address) {
        let idx = self.region_index(start);
        self.region_cursor[idx].store(cursor.as_usize(), std::sync::atomic::Ordering::Relaxed);
        self.region_state[idx].store(REGION_UNSTAGED, std::sync::atomic::Ordering::Relaxed);
    }

    /// (start, preset_cursor) of a region by address-index.
    pub fn region_bounds(&self, idx: usize) -> (Address, Address) {
        let start = self.space_base + idx * REGION_BYTES;
        let cursor =
            unsafe { Address::from_usize(self.region_cursor[idx].load(std::sync::atomic::Ordering::Relaxed)) };
        (start, cursor)
    }

    pub fn region_bytes(&self) -> usize {
        REGION_BYTES
    }

    /// Offset to add to a heap address to get its arena alias.
    pub fn alias_delta(&self) -> isize {
        self.arena_base.as_usize() as isize - self.space_base.as_usize() as isize
    }

    pub fn alias_of(&self, addr: Address) -> Address {
        self.arena_base + (addr - self.space_base)
    }

    /// Set the Class B v2 reference bit for a (to-space) reference-slot
    /// address.  `to_addr` is a real heap address inside the span (the staged
    /// slot's post-compaction location), at 4-byte granularity.
    #[inline]
    pub fn set_ref_bit(&self, to_addr: Address) {
        debug_assert!(to_addr >= self.space_base && to_addr < self.space_base + self.span);
        let slot = (to_addr - self.space_base) >> 2; // 4-byte slots
        let byte = slot >> 3;
        let bit = (slot & 7) as u8;
        unsafe {
            let p = self.refbitmap.add(byte);
            *p |= 1u8 << bit;
        }
        // NOTE: no per-slot global counter here -- a shared atomic
        // fetch_add on every reference slot ping-pongs its cacheline
        // across all staging workers and dominated the window.
    }

    /// Read the reference bit for a to-space slot address (4-byte granularity).
    #[inline]
    pub fn ref_bit(&self, to_addr: Address) -> bool {
        let slot = (to_addr - self.space_base) >> 2;
        let byte = slot >> 3;
        let bit = (slot & 7) as u8;
        unsafe { (*self.refbitmap.add(byte) >> bit) & 1 == 1 }
    }

    /// Clear reference bits covering a to-space byte range [start, start+bytes).
    pub fn clear_ref_bits(&self, start: Address, bytes: usize) {
        let first = (start - self.space_base) >> 2;
        let last = (start + bytes - self.space_base).div_ceil(4);
        let fb = first >> 3;
        let lb = (last + 7) >> 3;
        unsafe {
            std::ptr::write_bytes(self.refbitmap.add(fb), 0, lb - fb);
        }
    }

    /// Heap span base covered by this window.
    pub fn space_base(&self) -> Address {
        self.space_base
    }

    /// R1 debug: emulate the in-kernel page builder for to-space `page`,
    /// reading from-space words from `snapshot` (indexed by region-local
    /// word). `region_w0` is the region's first heap word index. Mirrors
    /// gc_b0_ops.bpf.c exactly.
    pub fn r1_emulate_page(&self, page: usize, snapshot: &[u64],
                            region_w0: usize) -> [u64; 512] {
        const REGION_WORDS: usize = 1 << 17;
        let mut out = [0u64; 512];
        let shift = COOPS_SHIFT.load(std::sync::atomic::Ordering::Relaxed);
        let base = COOPS_BASE.load(std::sync::atomic::Ordering::Relaxed);
        let total_words = self.span >> 3;
        let mut srcw = unsafe { *self.first_src.add(page) } as usize;
        let mut region_end = ((srcw / REGION_WORDS) + 1) * REGION_WORDS;
        if region_end > total_words {
            region_end = total_words;
        }
        let mut outw = 0;
        while outw < 512 && srcw < region_end {
            let live = unsafe {
                (*self.livebits.add(srcw >> 3) >> (srcw & 7)) & 1 == 1
            };
            if !live {
                srcw += 1;
                continue;
            }
            let mut word = if srcw >= region_w0
                && srcw - region_w0 < snapshot.len()
            {
                snapshot[srcw - region_w0]
            } else {
                0xdead_dead_dead_deadu64 // source outside snapshot: flag it
            };
            for h in 0..2 {
                let slot = (srcw << 1) | h;
                let rb = unsafe {
                    (*self.refbitmap.add(slot >> 3) >> (slot & 7)) & 1 == 1
                };
                if !rb {
                    continue;
                }
                let v = if h == 0 { word as u32 } else { (word >> 32) as u32 };
                if v == 0 {
                    continue;
                }
                let old = base.wrapping_add((v as usize) << shift);
                let rel = old.wrapping_sub(self.space_base.as_usize());
                if rel >= self.span {
                    continue;
                }
                let nv = unsafe { *self.fwdtable.add(rel >> 3) };
                if nv == 0 {
                    continue;
                }
                if h == 0 {
                    word = (word & !0xffff_ffffu64) | nv as u64;
                } else {
                    word = (word & 0xffff_ffffu64) | ((nv as u64) << 32);
                }
            }
            out[outw] = word;
            outw += 1;
            srcw += 1;
        }
        out
    }

    /// R1 debug: (first_src[page], live bit at that word) as userspace
    /// reads them through the arena mmap.
    pub fn r1_page_meta(&self, page: usize) -> (u32, bool) {
        unsafe {
            let fs = *self.first_src.add(page);
            let live = (*self.livebits.add((fs as usize) >> 3)
                        >> (fs & 7)) & 1 == 1;
            (fs, live)
        }
    }

    /// R1: clear the live-word bitmap and per-page first-source indexes for
    /// a region before calculate_offset_vector re-emits them.  Neither is
    /// consumed after the window closes, but without clearing, stale bits
    /// and stale page mappings from earlier GC cycles leak into the
    /// in-kernel page builder.
    pub fn clear_r1_meta(&self, start: Address, bytes: usize) {
        let w0 = (start - self.space_base) >> 3;
        let w1 = (start + bytes - self.space_base).div_ceil(8);
        unsafe {
            std::ptr::write_bytes(self.livebits.add(w0 >> 3), 0,
                                  ((w1 + 7) >> 3) - (w0 >> 3));
        }
        let p0 = (start - self.space_base) >> LOG_BYTES_IN_PAGE;
        let p1 = (start + bytes - self.space_base) >> LOG_BYTES_IN_PAGE;
        for p in p0..p1 {
            unsafe { *self.first_src.add(p) = 0; }
        }
    }

    /// Base of the reference bitmap (for the bpf shim / in-kernel handler).
    pub fn refbitmap_base(&self) -> Address {
        Address::from_mut_ptr(self.refbitmap)
    }

    /// Class B v2: record an object's forward (old -> new) in the forward
    /// table, indexed by `(old_addr - space_base) >> coops_shift`, value =
    /// the new compressed-oop.  Called for each live object while the offset
    /// vector is computed.
    #[inline]
    pub fn set_fwd(&self, old_addr: Address, new_addr: Address) {
        let shift = COOPS_SHIFT.load(std::sync::atomic::Ordering::Relaxed);
        let base = COOPS_BASE.load(std::sync::atomic::Ordering::Relaxed);
        let rel = old_addr.as_usize().wrapping_sub(self.space_base.as_usize());
        if rel >= self.span {
            return;
        }
        // Index by word (objects are 8-byte aligned) so the table is dense
        // regardless of the compressed-oops shift (which is 0 for <=4 GiB heaps).
        let idx = rel >> 3;
        let new_narrow = ((new_addr.as_usize() - base) >> shift) as u32;
        unsafe {
            *self.fwdtable.add(idx) = new_narrow;
        }
    }

    /// Base of the forward table (for the bpf shim / in-kernel handler).
    pub fn fwdtable_base(&self) -> Address {
        Address::from_mut_ptr(self.fwdtable)
    }

    /// (compacted words, probe_read failures) from the bpf R1 handler.
    pub fn r1_stats(&self) -> (u64, u64) {
        self.shim.as_ref().map(|s| s.r1_stats()).unwrap_or((0, 0))
    }
    pub fn r1_dbg(&self) {
        if let Some(s) = self.shim.as_ref() { s.dbg_print(); }
    }

    /// R1: mark `old_addr` (a live object word, 8-byte aligned) live in the
    /// live-word bitmap.
    #[inline]
    pub fn set_live_word(&self, old_addr: Address) {
        let w = (old_addr.as_usize().wrapping_sub(self.space_base.as_usize())) >> 3;
        if w >= self.span >> 3 {
            return;
        }
        unsafe {
            *self.livebits.add(w >> 3) |= 1u8 << (w & 7);
        }
    }

    /// R1: record the from-space word index that maps to to-space `page`.
    #[inline]
    pub fn set_first_src(&self, page: usize, old_word: u32) {
        if page < self.span >> LOG_BYTES_IN_PAGE {
            unsafe {
                *self.first_src.add(page) = old_word;
            }
        }
    }

    /// R1: for an object at old `old_start` forwarding to `new_start` and
    /// `nwords` long, set `first_src` for every to-space page boundary its new
    /// range crosses (the object is contiguous, so the source word for a new
    /// word is `old_word0 + (new_word - new_word0)`).
    #[inline]
    pub fn record_first_src(&self, old_start: Address, new_start: Address, nwords: usize) {
        const WPP: usize = BYTES_IN_PAGE / 8; // 512 to-space words / page
        let new_w0 = (new_start.as_usize() - self.space_base.as_usize()) >> 3;
        let old_w0 = (old_start.as_usize() - self.space_base.as_usize()) >> 3;
        let mut pw = new_w0.div_ceil(WPP) * WPP;
        while pw < new_w0 + nwords {
            self.set_first_src(pw / WPP, (old_w0 + (pw - new_w0)) as u32);
            pw += WPP;
        }
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
                let mut subs: Vec<(Address, usize)> = vec![];
                let mut a = start;
                while a < start + bytes {
                    let idx = self.region_index(a);
                    if !self.registered[idx].swap(true, std::sync::atomic::Ordering::Relaxed) {
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
        // Class B v2 (bpf backend): hand the in-kernel handler the params it
        // needs to forward references during page materialization.
        if self.backend == CompactFaultsBackend::Bpf {
            if let Some(shim) = self.shim.as_ref() {
                shim.set_forward(
                    COOPS_BASE.load(std::sync::atomic::Ordering::Relaxed),
                    COOPS_SHIFT.load(std::sync::atomic::Ordering::Relaxed),
                    // Verify mode forwards eagerly in userspace; the kernel
                    // must not re-forward at install time.
                    defer_forward() && !r1_verify(),
                    inkernel_compact(),
                );
            }
        }
        WINDOW_REMAINING.store(regions, std::sync::atomic::Ordering::SeqCst);
        WINDOW_OPEN.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Returns true if this was the last region of the window.
    pub fn region_done(&self) -> bool {
        WINDOW_REMAINING.fetch_sub(1, std::sync::atomic::Ordering::SeqCst) == 1
    }

    pub fn close_window(&self) {
        let faults = WINDOW_FAULTS.swap(0, std::sync::atomic::Ordering::Relaxed);
        let spins = WINDOW_SPIN.swap(0, std::sync::atomic::Ordering::Relaxed);
        if std::env::var("MMTK_WINDOW_STATS").is_ok() {
            eprintln!(
                "window: mutator_faults={} total_spins={} (~{} spins/stalled-fault)",
                faults,
                spins,
                if faults > 0 { spins / faults } else { 0 }
            );
        }
        WINDOW_OPEN.store(false, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn window_active(&self) -> bool {
        WINDOW_OPEN.load(std::sync::atomic::Ordering::SeqCst)
    }

    pub fn stall_report(&self) -> (u64, u64) {
        (
            WINDOW_FAULTS.load(std::sync::atomic::Ordering::Relaxed),
            WINDOW_SPIN.load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Handle a SIGBUS on `page` during the concurrent window (steal-mode):
    /// if the page's region is not yet staged, stage it ourselves (ART-style
    /// self-service) instead of waiting for the address-ordered sweep to
    /// reach it; then (uffd) install the page.  Returns true if the fault
    /// was ours.  Runs in signal context.
    fn handle_window_fault(&self, page: Address) -> bool {
        if !self.window_active() || !self.in_span(page) {
            return false;
        }
        WINDOW_FAULTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if self.page_state(page) == STATE_PENDING {
            let idx = self.region_index(page);
            // Drive the region's staging: claims+stages it ourselves, or
            // no-ops if another thread (GC worker or mutator) already claimed
            // it — in which case we wait below, bounded by ONE region's
            // staging time, not the whole sweep.
            if let Some(steal) = STEAL.get() {
                steal.stage(idx);
            }
            let mut spins: u64 = 0;
            while self.page_state(page) == STATE_PENDING {
                spins += 1;
                unsafe { libc::sched_yield(); }
            }
            if spins > 0 {
                WINDOW_SPIN.fetch_add(spins, std::sync::atomic::Ordering::Relaxed);
            }
        }
        // Now STAGED (installed in-kernel on retry for bpf) or ZERO_FILL.
        // For bpf the stage's install already materialized the page; for
        // uffd we install here (idempotent: EEXIST tolerated).
        if self.backend == CompactFaultsBackend::Uffd {
            match self.page_state(page) {
                STATE_STAGED => self.install_page_uffd(page),
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
                    self.install_page_uffd(a);
                    a = a + BYTES_IN_PAGE;
                }
            }
            CompactFaultsBackend::None => unreachable!(),
        }
    }

    /// Install one staged page via uffd.  When deferring forwarding (Class B
    /// v2), forward a private copy of the arena page first so the install is
    /// idempotent across concurrent faults; otherwise copy the arena directly.
    fn install_page_uffd(&self, page: Address) {
        if defer_forward() {
            let mut buf = [0u8; BYTES_IN_PAGE];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.alias_of(page).to_ptr::<u8>(),
                    buf.as_mut_ptr(),
                    BYTES_IN_PAGE,
                );
            }
            let bufaddr = Address::from_ptr(buf.as_ptr());
            if let Some(s) = STEAL.get() {
                s.forward_buf(page, bufaddr);
            }
            uffd_copy(self.uffd, page, bufaddr, BYTES_IN_PAGE);
        } else {
            uffd_copy(self.uffd, page, self.alias_of(page), BYTES_IN_PAGE);
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
                self.registered[self.region_index(start)]
                    .store(false, std::sync::atomic::Ordering::Relaxed);
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
            // R1 note: an in-flight mutator-fault build may still be reading
            // this alias (its fault started before install() touched the
            // page).  That is safe with the eager release: install() has
            // made every staged page PRESENT before we get here, so a
            // build that reads the released alias necessarily finishes
            // after release and loses the PTE install to the
            // already-present page -- its (garbage) output is discarded by
            // the kernel.  (b0_prefail_live stays the alarm: a released-
            // alias read with live bits would trip it.)
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
    type SetFwdFn = unsafe extern "C" fn(u64, u32, u32, u32);
    type CountFn = unsafe extern "C" fn() -> u64;
    type BaseFn = unsafe extern "C" fn() -> u64;
    type VoidFn = unsafe extern "C" fn();

    pub(super) struct Shim {
        init: InitFn,
        flip: FlipFn,
        unmap_arena: RangeFn,
        register: RangeFn,
        unregister: RangeFn,
        state: StateFn,
        set_forward: SetFwdFn,
        refs_forwarded: CountFn,
        fwdtable_base: BaseFn,
        refbits_base: BaseFn,
        livebits_base: BaseFn,
        first_src_base: BaseFn,
        compact_words: CountFn,
        prefail: CountFn,
        dbg_print: VoidFn,
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
                    set_forward: std::mem::transmute(sym("gcb0_set_forward")),
                    refs_forwarded: std::mem::transmute(sym("gcb0_refs_forwarded")),
                    fwdtable_base: std::mem::transmute(sym("gcb0_fwdtable_base")),
                    refbits_base: std::mem::transmute(sym("gcb0_refbits_base")),
                    livebits_base: std::mem::transmute(sym("gcb0_livebits_base")),
                    first_src_base: std::mem::transmute(sym("gcb0_first_src_base")),
                    compact_words: std::mem::transmute(sym("gcb0_compact_words")),
                    prefail: std::mem::transmute(sym("gcb0_prefail")),
                    dbg_print: std::mem::transmute(sym("gcb0_dbg_print")),
                }
            }
        }

        pub fn set_forward(
            &self,
            coops_base: usize,
            coops_shift: u32,
            defer: bool,
            inkernel: bool,
        ) {
            unsafe {
                (self.set_forward)(coops_base as u64, coops_shift, defer as u32, inkernel as u32)
            }
        }

        pub fn refs_forwarded(&self) -> u64 {
            unsafe { (self.refs_forwarded)() }
        }

        pub fn r1_stats(&self) -> (u64, u64) {
            unsafe { ((self.compact_words)(), (self.prefail)()) }
        }
        pub fn dbg_print(&self) {
            unsafe { (self.dbg_print)() }
        }

        /// Userspace base of the forward-table arena (the GC writes here).
        pub fn fwdtable_base(&self) -> *mut u32 {
            unsafe { (self.fwdtable_base)() as *mut u32 }
        }

        /// Userspace base of the reference bitmap (in the arena).
        pub fn refbits_base(&self) -> *mut u8 {
            unsafe { (self.refbits_base)() as *mut u8 }
        }

        /// Userspace base of the live-word bitmap (R1, in the arena).
        pub fn livebits_base(&self) -> *mut u8 {
            unsafe { (self.livebits_base)() as *mut u8 }
        }

        /// Userspace base of the per-page first-source index (R1, in the arena).
        pub fn first_src_base(&self) -> *mut u32 {
            unsafe { (self.first_src_base)() as *mut u32 }
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
