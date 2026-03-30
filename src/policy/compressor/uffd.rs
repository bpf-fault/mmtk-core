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
const UFFDIO_COPY: libc::c_ulong = 0xc028aa03;
const UFFDIO_UNREGISTER: libc::c_ulong = 0x8010aa01;
const UFFDIO_REGISTER_MODE_MISSING: u64 = 1 << 0;
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
    /// Full region size (1 MiB aligned).
    pub region_size: usize,
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
    /// Processed-but-not-yet-mapped page buffers.
    page_buffers: Vec<Vec<Mutex<Option<Box<[u8; PAGE_SIZE]>>>>>,
    /// Whether a region has already been unregistered/unmapped.
    region_cleaned: Vec<AtomicBool>,
    /// Signal to stop the handler thread.
    all_done: Arc<AtomicBool>,
    /// Number of uffd fault messages handled.
    faults_handled: Arc<AtomicU64>,
    /// Number of pages materialized via `UFFDIO_COPY`.
    pages_resolved: Arc<AtomicU64>,
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
            page_buffers: Vec::new(),
            region_cleaned: Vec::new(),
            all_done: Arc::new(AtomicBool::new(false)),
            faults_handled: Arc::new(AtomicU64::new(0)),
            pages_resolved: Arc::new(AtomicU64::new(0)),
        })
    }

    /// `mremap` a region to a shadow address and register the original range with userfaultfd.
    pub fn mremap_and_register(
        &mut self,
        region_index: usize,
        original_start: usize,
        region_size: usize,
    ) -> io::Result<()> {
        let shadow = unsafe {
            libc::mremap(
                original_start as *mut libc::c_void,
                region_size,
                region_size,
                libc::MREMAP_MAYMOVE | MREMAP_DONTUNMAP,
            )
        };
        if shadow == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        let mut reg = UffdioRegister {
            range: UffdioRange {
                start: original_start as u64,
                len: region_size as u64,
            },
            mode: UFFDIO_REGISTER_MODE_MISSING,
            ioctls: 0,
        };
        let ret = unsafe { libc::ioctl(self.uffd, UFFDIO_REGISTER, &mut reg as *mut _) };
        if ret != 0 {
            let err = io::Error::last_os_error();
            unsafe { libc::munmap(shadow, region_size) };
            return Err(err);
        }

        let num_pages = region_size / PAGE_SIZE;
        self.shadows.push(RegionShadow {
            region_index,
            original_start,
            shadow_start: shadow as usize,
            region_size,
        });
        self.page_states.push(
            (0..num_pages)
                .map(|_| AtomicU8::new(PageState::Unprocessed as u8))
                .collect(),
        );
        let mut buffers = Vec::with_capacity(num_pages);
        for _ in 0..num_pages {
            buffers.push(Mutex::new(None));
        }
        self.page_buffers.push(buffers);
        self.region_cleaned.push(AtomicBool::new(false));
        Ok(())
    }

    /// Find which page a faulting address belongs to.
    fn find_page(&self, addr: usize) -> Option<(usize, usize)> {
        for (region_idx, shadow) in self.shadows.iter().enumerate() {
            if addr >= shadow.original_start && addr < shadow.original_start + shadow.region_size {
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

    fn build_page_buffer(&self, region_idx: usize, page_idx: usize) -> Box<[u8; PAGE_SIZE]> {
        let shadow = &self.shadows[region_idx];
        let mut buf = Box::new([0u8; PAGE_SIZE]);
        self.compressor_space.build_page_from_shadow(
            shadow.region_index,
            page_idx,
            unsafe { Address::from_usize(shadow.shadow_start) },
            &mut buf[..],
        );
        buf
    }

    fn store_processed_page_buffer(
        &self,
        region_idx: usize,
        page_idx: usize,
        buf: Box<[u8; PAGE_SIZE]>,
    ) {
        let mut slot = self.page_buffers[region_idx][page_idx].lock().unwrap();
        debug_assert!(slot.is_none());
        *slot = Some(buf);
    }

    fn take_processed_page_buffer(
        &self,
        region_idx: usize,
        page_idx: usize,
    ) -> io::Result<Box<[u8; PAGE_SIZE]>> {
        let mut slot = self.page_buffers[region_idx][page_idx].lock().unwrap();
        slot.take().ok_or_else(|| {
            io::Error::other(format!(
                "missing processed page buffer for region {} page {}",
                region_idx, page_idx
            ))
        })
    }

    fn map_page_from_buffer(
        &self,
        region_idx: usize,
        page_idx: usize,
        buf: &[u8; PAGE_SIZE],
    ) -> io::Result<()> {
        let shadow = &self.shadows[region_idx];
        let dst = shadow.original_start + page_idx * PAGE_SIZE;
        let mut copy = UffdioCopy {
            dst: dst as u64,
            src: buf.as_ptr() as u64,
            len: PAGE_SIZE as u64,
            mode: 0,
            copy: 0,
        };
        let ret = unsafe { libc::ioctl(self.uffd, UFFDIO_COPY, &mut copy as *mut _) };
        if ret != 0 {
            let err = io::Error::last_os_error();
            if err.raw_os_error() != Some(libc::EEXIST) {
                return Err(err);
            }
        }
        self.pages_resolved.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// ART-style background processing for a page.
    ///
    /// This mirrors the GC-thread side of `DoPageCompactionWithStateChange()` and
    /// `MapMovingSpacePages()`: compact into a temporary buffer, then map the page.
    fn try_process_or_map_page_for_gc(
        &self,
        region_idx: usize,
        page_idx: usize,
    ) -> io::Result<u64> {
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
                            let buf = self.build_page_buffer(region_idx, page_idx);
                            self.store_processed_page_buffer(region_idx, page_idx, buf);
                            self.store_page_state(region_idx, page_idx, PageState::Processed);
                            continue;
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
                            let buf = self.take_processed_page_buffer(region_idx, page_idx)?;
                            self.map_page_from_buffer(region_idx, page_idx, &buf)?;
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
                            let buf = self.build_page_buffer(region_idx, page_idx);
                            self.store_page_state(
                                region_idx,
                                page_idx,
                                PageState::ProcessedAndMapping,
                            );
                            self.map_page_from_buffer(region_idx, page_idx, &buf)?;
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
                            let buf = self.take_processed_page_buffer(region_idx, page_idx)?;
                            self.map_page_from_buffer(region_idx, page_idx, &buf)?;
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
        let mut mapped = 0u64;
        let num_pages = self.shadows[region_idx].region_size / PAGE_SIZE;
        for page_idx in (0..num_pages).rev() {
            mapped += self.try_process_or_map_page_for_gc(region_idx, page_idx)?;
        }
        Ok(mapped)
    }

    /// Unregister one region from userfaultfd and unmap its shadow after every page
    /// in the region has been materialized.
    pub fn cleanup_region(&self, region_idx: usize) -> io::Result<()> {
        if self.region_cleaned[region_idx].load(Ordering::Acquire) {
            return Ok(());
        }

        let shadow = &self.shadows[region_idx];
        let range = UffdioRange {
            start: shadow.original_start as u64,
            len: shadow.region_size as u64,
        };
        let ret = unsafe { libc::ioctl(self.uffd, UFFDIO_UNREGISTER, &range as *const _) };
        if ret != 0 {
            return Err(io::Error::last_os_error());
        }

        let ret =
            unsafe { libc::munmap(shadow.shadow_start as *mut libc::c_void, shadow.region_size) };
        if ret != 0 {
            return Err(io::Error::last_os_error());
        }
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
