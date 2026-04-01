//! userfaultfd-based page resolution for the Compressor.
//!
//! This module implements the low-level mechanism needed for an ART-inspired
//! concurrent compaction path:
//! - move a region to a shadow mapping with `mremap(MREMAP_DONTUNMAP)`
//! - register the original range with `userfaultfd`
//! - reconstruct destination pages from the shadow one page at a time
//! - materialize pages back into the original range with `UFFDIO_COPY`
//!
//! The internal page-state protocol is intentionally modeled after Android ART's
//! `mark_compact` collector. We do not yet implement ART's full mutator-resume
//! control flow, but the state machine is structured to make that possible.

use super::CompressorSpace;
use crate::util::Address;
use crate::vm::VMBinding;
use std::fmt;
use std::io;
use std::os::unix::io::RawFd;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

const PAGE_SIZE: usize = 4096;

// userfaultfd ioctl constants
const UFFDIO_API: libc::c_ulong = 0xc018aa3f;
const UFFDIO_REGISTER: libc::c_ulong = 0xc020aa00;
const UFFDIO_WAKE: libc::c_ulong = 0x8010aa02;
const UFFDIO_COPY: libc::c_ulong = 0xc028aa03;
const UFFDIO_UNREGISTER: libc::c_ulong = 0x8010aa01;
const UFFDIO_REGISTER_MODE_MISSING: u64 = 1 << 0;
const UFFDIO_COPY_MODE_DONTWAKE: u64 = 1 << 0;
const MREMAP_DONTUNMAP: libc::c_int = 4;

/// ART-inspired per-page state.
///
/// The numeric values intentionally match ART's `MarkCompact::PageState` order.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PageState {
    /// Not processed yet.
    Unprocessed = 0,
    /// Being processed by GC thread and will not be mapped immediately.
    Processing = 1,
    /// Processed but not mapped yet.
    Processed = 2,
    /// Being processed by GC/fault path and will be mapped immediately.
    ProcessingAndMapping = 3,
    /// Being processed by mutator/fault path.
    MutatorProcessing = 4,
    /// Processed and being mapped.
    ProcessedAndMapping = 5,
    /// Processed and already mapped.
    ProcessedAndMapped = 6,
}

impl PageState {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Unprocessed,
            1 => Self::Processing,
            2 => Self::Processed,
            3 => Self::ProcessingAndMapping,
            4 => Self::MutatorProcessing,
            5 => Self::ProcessedAndMapping,
            6 => Self::ProcessedAndMapped,
            _ => panic!("invalid page state value: {}", value),
        }
    }

    fn is_wait_state(self) -> bool {
        matches!(
            self,
            Self::Processing
                | Self::ProcessingAndMapping
                | Self::MutatorProcessing
                | Self::ProcessedAndMapping
        )
    }
}

impl fmt::Display for PageState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

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
#[derive(Copy, Clone)]
struct UffdMsg {
    event: u8,
    _reserved1: u8,
    _reserved2: u16,
    _reserved3: u32,
    arg: UffdMsgArg,
}

#[repr(C)]
#[derive(Copy, Clone)]
union UffdMsgArg {
    pagefault: UffdMsgPagefault,
    _pad: [u8; 32],
}

#[repr(C)]
#[derive(Copy, Clone)]
struct UffdMsgPagefault {
    flags: u64,
    address: u64,
    _union: u64,
}

#[repr(C)]
struct UffdioCopy {
    dst: u64,
    src: u64,
    len: u64,
    mode: u64,
    copy: i64,
}

/// Describes one region's shadow mapping.
#[derive(Clone)]
pub struct RegionShadow {
    /// Region index in `CompressorSpace`.
    pub region_index: usize,
    /// Original region start address.
    pub original_start: usize,
    /// Shadow address where the original physical pages were moved.
    pub shadow_start: usize,
    /// Size of the moved source prefix kept in shadow for compaction reads.
    pub shadow_size: usize,
    /// Size of the destination prefix registered with userfaultfd.
    pub registered_size: usize,
}

/// Context for a userfaultfd-based compaction epoch.
pub struct UffdContext<VM: VMBinding> {
    /// The userfaultfd file descriptor.
    uffd: RawFd,
    /// Compressor space used to reconstruct destination pages from shadow.
    compressor_space: &'static CompressorSpace<VM>,

    /// Shadow mappings for each region.
    pub shadows: Vec<RegionShadow>,
    /// ART-style page state per region/page.
    page_states: Vec<Vec<AtomicU8>>,
    /// Region-sized contiguous buffers used by the GC path to batch page mapping.
    region_buffers: Vec<Mutex<Option<Vec<u8>>>>,
    /// Whether a region has already been unregistered/unmapped.
    region_cleaned: Vec<AtomicBool>,
    /// Signal to stop the handler thread.
    all_done: Arc<AtomicBool>,
    /// Number of uffd fault messages handled.
    faults_handled: Arc<AtomicU64>,
    /// Number of pages materialized via `UFFDIO_COPY`.
    pages_resolved: Arc<AtomicU64>,
    /// Number of pages compacted by the background GC path.
    gc_pages_processed: Arc<AtomicU64>,
    /// Number of pages compacted by the fault/mutator path.
    mutator_pages_processed: Arc<AtomicU64>,
}

impl<VM: VMBinding> UffdContext<VM> {
    /// Create userfaultfd and set up the API handshake.
    /// Regions are not yet registered — call `mremap_and_register` for each region.
    pub fn new(compressor_space: &'static CompressorSpace<VM>) -> io::Result<Self> {
        let uffd = unsafe { libc::syscall(libc::SYS_userfaultfd, libc::O_NONBLOCK) } as RawFd;
        if uffd < 0 {
            return Err(io::Error::last_os_error());
        }

        let mut api = UffdioApi {
            api: 0xAA,
            features: 0,
            ioctls: 0,
        };
        let ret = unsafe { libc::ioctl(uffd, UFFDIO_API, &mut api as *mut _) };
        if ret != 0 {
            let err = io::Error::last_os_error();
            unsafe { libc::close(uffd) };
            return Err(err);
        }

        Ok(UffdContext {
            uffd,
            compressor_space,
            shadows: Vec::new(),
            page_states: Vec::new(),
            region_buffers: Vec::new(),
            region_cleaned: Vec::new(),
            all_done: Arc::new(AtomicBool::new(false)),
            faults_handled: Arc::new(AtomicU64::new(0)),
            pages_resolved: Arc::new(AtomicU64::new(0)),
            gc_pages_processed: Arc::new(AtomicU64::new(0)),
            mutator_pages_processed: Arc::new(AtomicU64::new(0)),
        })
    }

    /// `mremap` a source prefix to a shadow address and register the destination prefix with userfaultfd.
    pub fn mremap_and_register(
        &mut self,
        region_index: usize,
        original_start: usize,
        shadow_size: usize,
        registered_size: usize,
    ) -> io::Result<()> {
        debug_assert!(registered_size <= shadow_size);
        if shadow_size == 0 {
            self.shadows.push(RegionShadow {
                region_index,
                original_start,
                shadow_start: 0,
                shadow_size: 0,
                registered_size: 0,
            });
            self.page_states.push(Vec::new());
            self.region_buffers.push(Mutex::new(None));
            self.region_cleaned.push(AtomicBool::new(true));
            return Ok(());
        }

        let shadow = unsafe {
            libc::mremap(
                original_start as *mut libc::c_void,
                shadow_size,
                shadow_size,
                libc::MREMAP_MAYMOVE | MREMAP_DONTUNMAP,
            )
        };
        if shadow == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        if registered_size > 0 {
            // ART mremaps the whole source range to shadow but only registers the used
            // destination prefix with UFFD. Fault in one page just beyond the registered
            // prefix first so the split VMAs share anon_vma and can merge back after
            // unregister, then drop the page again.
            if registered_size < shadow_size {
                let tail_page = original_start + registered_size;
                unsafe {
                    std::ptr::write_volatile(tail_page as *mut u8, 0);
                    let ret = libc::madvise(
                        tail_page as *mut libc::c_void,
                        PAGE_SIZE,
                        libc::MADV_DONTNEED,
                    );
                    if ret != 0 {
                        let err = io::Error::last_os_error();
                        libc::munmap(shadow, shadow_size);
                        return Err(err);
                    }
                }
            }

            let mut reg = UffdioRegister {
                range: UffdioRange {
                    start: original_start as u64,
                    len: registered_size as u64,
                },
                mode: UFFDIO_REGISTER_MODE_MISSING,
                ioctls: 0,
            };
            let ret = unsafe { libc::ioctl(self.uffd, UFFDIO_REGISTER, &mut reg as *mut _) };
            if ret != 0 {
                let err = io::Error::last_os_error();
                unsafe { libc::munmap(shadow, shadow_size) };
                return Err(err);
            }
        }

        let num_pages = registered_size / PAGE_SIZE;
        self.shadows.push(RegionShadow {
            region_index,
            original_start,
            shadow_start: shadow as usize,
            shadow_size,
            registered_size,
        });
        self.page_states.push(
            (0..num_pages)
                .map(|_| AtomicU8::new(PageState::Unprocessed as u8))
                .collect(),
        );
        self.region_buffers.push(Mutex::new(None));
        self.region_cleaned.push(AtomicBool::new(false));
        Ok(())
    }

    /// Find which page a faulting address belongs to.
    fn find_page(&self, addr: usize) -> Option<(usize, usize)> {
        for (region_idx, shadow) in self.shadows.iter().enumerate() {
            if addr >= shadow.original_start
                && addr < shadow.original_start + shadow.registered_size
            {
                let page_idx = (addr - shadow.original_start) / PAGE_SIZE;
                return Some((region_idx, page_idx));
            }
        }
        None
    }

    fn load_page_state(&self, region_idx: usize, page_idx: usize) -> PageState {
        PageState::from_u8(self.page_states[region_idx][page_idx].load(Ordering::Acquire))
    }

    fn compare_exchange_page_state(
        &self,
        region_idx: usize,
        page_idx: usize,
        current: PageState,
        new: PageState,
    ) -> Result<PageState, PageState> {
        self.page_states[region_idx][page_idx]
            .compare_exchange(
                current as u8,
                new as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map(|old| PageState::from_u8(old))
            .map_err(PageState::from_u8)
    }

    fn store_page_state(&self, region_idx: usize, page_idx: usize, state: PageState) {
        self.page_states[region_idx][page_idx].store(state as u8, Ordering::Release);
    }

    fn backoff_wait(&self, mut count: u32) {
        if count < 64 {
            std::hint::spin_loop();
        } else {
            thread::yield_now();
            count = 0;
            let _ = count;
        }
    }

    fn process_page_into_region_buffer(&self, region_idx: usize, _page_idx: usize) {
        let shadow = &self.shadows[region_idx];
        let mut region_buf = self.region_buffers[region_idx].lock().unwrap();
        if region_buf.is_none() {
            let mut buf = vec![0u8; shadow.registered_size];
            self.compressor_space.build_region_from_shadow(
                shadow.region_index,
                unsafe { Address::from_usize(shadow.shadow_start) },
                &mut buf,
            );
            if std::env::var_os("MMTK_VALIDATE_UFFD_REGION_OBJECTS").is_some() {
                self.compressor_space
                    .validate_region_buffer_objects(shadow.region_index, &buf)
                    .unwrap_or_else(|e| panic!("{}", e));
            }
            if std::env::var_os("MMTK_VALIDATE_UFFD_REGION_MARK_WORDS").is_some() {
                self.compressor_space
                    .validate_region_buffer_mark_words(
                        shadow.region_index,
                        unsafe { Address::from_usize(shadow.shadow_start) },
                        &buf,
                    )
                    .unwrap_or_else(|e| panic!("{}", e));
            }
            if std::env::var_os("MMTK_VALIDATE_UFFD_REGION_REFS").is_some() {
                self.compressor_space
                    .validate_region_buffer_references(
                        shadow.region_index,
                        unsafe { Address::from_usize(shadow.shadow_start) },
                        &buf,
                    )
                    .unwrap_or_else(|e| panic!("{}", e));
            }
            *region_buf = Some(buf);
        }
    }

    fn map_pages_from_region_buffer(
        &self,
        region_idx: usize,
        start_page_idx: usize,
        num_pages: usize,
    ) -> io::Result<usize> {
        let shadow = &self.shadows[region_idx];
        let start = start_page_idx * PAGE_SIZE;
        let len = num_pages * PAGE_SIZE;
        let dst = shadow.original_start + start;
        let region_buf = self.region_buffers[region_idx].lock().unwrap();
        let buf = region_buf.as_ref().ok_or_else(|| {
            io::Error::other(format!("missing region buffer for region {}", region_idx))
        })?;
        let validate_mapped_pages = std::env::var_os("MMTK_VALIDATE_UFFD_MAPPED_PAGES").is_some();
        let mut copy = UffdioCopy {
            dst: dst as u64,
            src: buf[start..start + len].as_ptr() as u64,
            len: len as u64,
            mode: if validate_mapped_pages {
                UFFDIO_COPY_MODE_DONTWAKE
            } else {
                0
            },
            copy: 0,
        };
        let ret = unsafe { libc::ioctl(self.uffd, UFFDIO_COPY, &mut copy as *mut _) };
        if ret != 0 {
            let err = io::Error::last_os_error();
            if err.raw_os_error() != Some(libc::EEXIST) {
                return Err(err);
            }
        }
        let copied_bytes = if copy.copy > 0 {
            copy.copy as usize
        } else if ret == 0 {
            len
        } else {
            0
        };
        if copied_bytes > 0 && validate_mapped_pages {
            let mapped_slice = unsafe { std::slice::from_raw_parts(dst as *const u8, copied_bytes) };
            let expected_slice = &buf[start..start + copied_bytes];
            if mapped_slice != expected_slice {
                let diff = mapped_slice
                    .iter()
                    .zip(expected_slice.iter())
                    .position(|(actual, expected)| actual != expected)
                    .unwrap_or(0);
                return Err(io::Error::other(format!(
                    "mapped page validation failed for region {} at page {} (+{} bytes): actual=0x{:02x}, expected=0x{:02x}",
                    region_idx,
                    start_page_idx,
                    diff,
                    mapped_slice[diff],
                    expected_slice[diff],
                )));
            }
            let mut wake = UffdioRange {
                start: dst as u64,
                len: copied_bytes as u64,
            };
            let wake_ret = unsafe { libc::ioctl(self.uffd, UFFDIO_WAKE, &mut wake as *mut _) };
            if wake_ret != 0 {
                return Err(io::Error::last_os_error());
            }
        }
        let mapped_pages = copied_bytes / PAGE_SIZE;
        self.pages_resolved
            .fetch_add(mapped_pages as u64, Ordering::Relaxed);
        Ok(mapped_pages)
    }

    fn claim_region_pages_for_gc(&self, region_idx: usize) -> io::Result<Vec<usize>> {
        let num_pages = self.shadows[region_idx].registered_size / PAGE_SIZE;
        let mut claimed = Vec::with_capacity(num_pages);
        for page_idx in (0..num_pages).rev() {
            let mut backoff = 0u32;
            loop {
                let state = self.load_page_state(region_idx, page_idx);
                if state.is_wait_state() {
                    self.backoff_wait(backoff);
                    backoff = backoff.saturating_add(1);
                    continue;
                }
                match state {
                    PageState::Unprocessed => {
                        match self.compare_exchange_page_state(
                            region_idx,
                            page_idx,
                            PageState::Unprocessed,
                            PageState::Processing,
                        ) {
                            Ok(_) => {
                                claimed.push(page_idx);
                                break;
                            }
                            Err(_) => continue,
                        }
                    }
                    PageState::Processed | PageState::ProcessedAndMapped => break,
                    PageState::Processing
                    | PageState::ProcessingAndMapping
                    | PageState::MutatorProcessing
                    | PageState::ProcessedAndMapping => unreachable!(),
                }
            }
        }
        Ok(claimed)
    }

    fn map_processed_pages_for_gc(&self, region_idx: usize) -> io::Result<u64> {
        let num_pages = self.shadows[region_idx].registered_size / PAGE_SIZE;
        let mut idx = 0usize;
        let mut mapped = 0u64;
        while idx < num_pages {
            let mut backoff = 0u32;
            let run_start = loop {
                let state = self.load_page_state(region_idx, idx);
                if state.is_wait_state() {
                    self.backoff_wait(backoff);
                    backoff = backoff.saturating_add(1);
                    continue;
                }
                if state == PageState::Processed {
                    break idx;
                }
                idx += 1;
                if idx >= num_pages {
                    return Ok(mapped);
                }
            };

            let mut run_len = 0usize;
            while idx < num_pages {
                match self.compare_exchange_page_state(
                    region_idx,
                    idx,
                    PageState::Processed,
                    PageState::ProcessedAndMapping,
                ) {
                    Ok(_) => {
                        run_len += 1;
                        idx += 1;
                    }
                    Err(_) => break,
                }
            }

            if run_len == 0 {
                idx += 1;
                continue;
            }

            let mapped_pages = self.map_pages_from_region_buffer(region_idx, run_start, run_len)?;
            if mapped_pages < run_len {
                return Err(io::Error::other(format!(
                    "partial region mapping for region {}: requested {} pages, mapped {}",
                    region_idx, run_len, mapped_pages
                )));
            }
            for page in run_start..run_start + mapped_pages {
                self.store_page_state(region_idx, page, PageState::ProcessedAndMapped);
            }
            mapped += mapped_pages as u64;
        }
        Ok(mapped)
    }

    /// Fault-path processing analogous to ART's `ConcurrentlyProcessMovingPage`.
    fn try_resolve_fault_page(&self, region_idx: usize, page_idx: usize) -> io::Result<u64> {
        let mut backoff = 0u32;
        loop {
            let state = self.load_page_state(region_idx, page_idx);
            if state.is_wait_state() {
                self.backoff_wait(backoff);
                backoff = backoff.saturating_add(1);
                continue;
            }
            match state {
                PageState::Unprocessed => {
                    match self.compare_exchange_page_state(
                        region_idx,
                        page_idx,
                        PageState::Unprocessed,
                        PageState::MutatorProcessing,
                    ) {
                        Ok(_) => {
                            self.process_page_into_region_buffer(region_idx, page_idx);
                            self.mutator_pages_processed.fetch_add(1, Ordering::Relaxed);
                            self.store_page_state(
                                region_idx,
                                page_idx,
                                PageState::ProcessedAndMapping,
                            );
                            let mapped_pages =
                                self.map_pages_from_region_buffer(region_idx, page_idx, 1)?;
                            if mapped_pages != 1 {
                                return Err(io::Error::other(format!(
                                    "expected to map one mutator-processed page for region {} page {}, mapped {}",
                                    region_idx, page_idx, mapped_pages
                                )));
                            }
                            self.store_page_state(
                                region_idx,
                                page_idx,
                                PageState::ProcessedAndMapped,
                            );
                            return Ok(1);
                        }
                        Err(_) => continue,
                    }
                }
                PageState::Processed => {
                    match self.compare_exchange_page_state(
                        region_idx,
                        page_idx,
                        PageState::Processed,
                        PageState::ProcessedAndMapping,
                    ) {
                        Ok(_) => {
                            let mapped_pages =
                                self.map_pages_from_region_buffer(region_idx, page_idx, 1)?;
                            if mapped_pages != 1 {
                                return Err(io::Error::other(format!(
                                    "expected to map one processed page for region {} page {}, mapped {}",
                                    region_idx, page_idx, mapped_pages
                                )));
                            }
                            self.store_page_state(
                                region_idx,
                                page_idx,
                                PageState::ProcessedAndMapped,
                            );
                            return Ok(1);
                        }
                        Err(_) => continue,
                    }
                }
                PageState::ProcessedAndMapped => return Ok(0),
                PageState::Processing
                | PageState::ProcessingAndMapping
                | PageState::MutatorProcessing
                | PageState::ProcessedAndMapping => {
                    self.backoff_wait(backoff);
                    backoff = backoff.saturating_add(1);
                }
            }
        }
    }

    /// Spawn a handler thread that resolves pages on demand.
    pub fn spawn_handler_thread(ctx: Arc<UffdContext<VM>>) -> thread::JoinHandle<()> {
        let uffd = ctx.uffd;
        let all_done = ctx.all_done.clone();
        let faults_handled = ctx.faults_handled.clone();

        thread::spawn(move || {
            let mut msg = std::mem::MaybeUninit::<UffdMsg>::uninit();
            loop {
                let mut pfd = libc::pollfd {
                    fd: uffd,
                    events: libc::POLLIN,
                    revents: 0,
                };
                let ret = unsafe { libc::poll(&mut pfd, 1, 10) };
                if ret <= 0 {
                    if all_done.load(Ordering::Relaxed) {
                        break;
                    }
                    continue;
                }

                let nread = unsafe {
                    libc::read(
                        uffd,
                        msg.as_mut_ptr() as *mut libc::c_void,
                        std::mem::size_of::<UffdMsg>(),
                    )
                };
                if nread <= 0 {
                    if all_done.load(Ordering::Relaxed) {
                        break;
                    }
                    continue;
                }

                let msg_val = unsafe { msg.assume_init_read() };
                if msg_val.event != 0x12 {
                    continue;
                }

                let fault_addr = unsafe { msg_val.arg.pagefault.address } as usize;
                if let Some((region_idx, page_idx)) = ctx.find_page(fault_addr) {
                    if let Err(e) = ctx.try_resolve_fault_page(region_idx, page_idx) {
                        error!(
                            "UffdContext: failed to resolve faulted page r{} p{}: {}",
                            region_idx, page_idx, e
                        );
                    }
                    faults_handled.fetch_add(1, Ordering::Relaxed);
                }
            }
        })
    }

    /// Background-GC processing for one region.
    ///
    /// Like ART's `CompactMovingSpace()`, we walk pages in reverse order so cleanup
    /// can happen incrementally as the concurrent phase advances.
    pub fn process_region_pages_in_reverse(&self, region_idx: usize) -> io::Result<u64> {
        let claimed = self.claim_region_pages_for_gc(region_idx)?;
        if !claimed.is_empty() {
            let shadow = &self.shadows[region_idx];
            {
                let mut region_buf = self.region_buffers[region_idx].lock().unwrap();
                if region_buf.is_none() {
                    *region_buf = Some(vec![0u8; shadow.registered_size]);
                }
                let buf = region_buf.as_mut().unwrap();
                self.compressor_space.build_region_from_shadow(
                    shadow.region_index,
                    unsafe { Address::from_usize(shadow.shadow_start) },
                    buf,
                );
                if std::env::var_os("MMTK_VALIDATE_UFFD_REGION_OBJECTS").is_some() {
                    self.compressor_space
                        .validate_region_buffer_objects(shadow.region_index, buf)
                        .unwrap_or_else(|e| panic!("{}", e));
                }
                if std::env::var_os("MMTK_VALIDATE_UFFD_REGION_MARK_WORDS").is_some() {
                    self.compressor_space
                        .validate_region_buffer_mark_words(
                            shadow.region_index,
                            unsafe { Address::from_usize(shadow.shadow_start) },
                            buf,
                        )
                        .unwrap_or_else(|e| panic!("{}", e));
                }
                if std::env::var_os("MMTK_VALIDATE_UFFD_REGION_REFS").is_some() {
                    self.compressor_space
                        .validate_region_buffer_references(
                            shadow.region_index,
                            unsafe { Address::from_usize(shadow.shadow_start) },
                            buf,
                        )
                        .unwrap_or_else(|e| panic!("{}", e));
                }
            }
            self.gc_pages_processed
                .fetch_add(claimed.len() as u64, Ordering::Relaxed);
            for page_idx in claimed {
                self.store_page_state(region_idx, page_idx, PageState::Processed);
            }
        }
        self.map_processed_pages_for_gc(region_idx)
    }

    /// Unregister one region from userfaultfd and unmap its shadow after every page
    /// in the region has been materialized.
    pub fn cleanup_region(&self, region_idx: usize) -> io::Result<()> {
        if self.region_cleaned[region_idx].load(Ordering::Acquire) {
            return Ok(());
        }

        let shadow = &self.shadows[region_idx];
        if shadow.registered_size > 0 {
            let range = UffdioRange {
                start: shadow.original_start as u64,
                len: shadow.registered_size as u64,
            };
            let ret = unsafe { libc::ioctl(self.uffd, UFFDIO_UNREGISTER, &range as *const _) };
            if ret != 0 {
                return Err(io::Error::last_os_error());
            }
        }

        let ret =
            unsafe { libc::munmap(shadow.shadow_start as *mut libc::c_void, shadow.shadow_size) };
        if ret != 0 {
            return Err(io::Error::last_os_error());
        }
        *self.region_buffers[region_idx].lock().unwrap() = None;
        self.region_cleaned[region_idx].store(true, Ordering::Release);
        Ok(())
    }

    pub fn page_state_counts(&self) -> [u64; 7] {
        let mut counts = [0u64; 7];
        for region in &self.page_states {
            for state in region {
                let idx = state.load(Ordering::Relaxed) as usize;
                counts[idx] += 1;
            }
        }
        counts
    }

    /// Signal the handler thread to stop.
    pub fn signal_done(&self) {
        self.all_done.store(true, Ordering::Relaxed);
    }

    /// Get fault count.
    pub fn faults_handled(&self) -> u64 {
        self.faults_handled.load(Ordering::Relaxed)
    }

    /// Get resolved-page count.
    pub fn pages_resolved(&self) -> u64 {
        self.pages_resolved.load(Ordering::Relaxed)
    }

    pub fn gc_pages_processed(&self) -> u64 {
        self.gc_pages_processed.load(Ordering::Relaxed)
    }

    pub fn mutator_pages_processed(&self) -> u64 {
        self.mutator_pages_processed.load(Ordering::Relaxed)
    }

    /// Cleanup any remaining registered regions and close the UFFD file descriptor.
    pub fn teardown(&self) {
        for region_idx in 0..self.shadows.len() {
            if !self.region_cleaned[region_idx].load(Ordering::Acquire) {
                if let Err(e) = self.cleanup_region(region_idx) {
                    error!(
                        "UffdContext: failed to cleanup remaining region {} during teardown: {}",
                        region_idx, e
                    );
                }
            }
        }
        unsafe {
            libc::close(self.uffd);
        }
    }
}
