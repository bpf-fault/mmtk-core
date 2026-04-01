use crate::plan::generational::global::RememberedSetMode;
use crate::policy::immix::block::Block;
use crate::policy::immix::ImmixSpace;
use crate::util::constants::LOG_BYTES_IN_PAGE;
use crate::util::linear_scan::Region;
use crate::util::object_enum::BlockMayHaveObjects;
use crate::vm::VMBinding;
use std::collections::HashSet;
use std::env;
use std::mem::{self, MaybeUninit};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

const PAGE_SIZE: usize = 1 << LOG_BYTES_IN_PAGE;

fn parse_remembered_set_mode_from_env() -> RememberedSetMode {
    match env::var("MMTK_UFFD_WP_RS_MODE") {
        Ok(mode) => match mode.as_str() {
            "stats" | "barrier" => RememberedSetMode::Barrier,
            "shadow" => RememberedSetMode::Shadow,
            "replace" => RememberedSetMode::Replace,
            other => {
                warn!(
                    "Unknown MMTK_UFFD_WP_RS_MODE='{}'; falling back to software barrier mode",
                    other
                );
                RememberedSetMode::Barrier
            }
        },
        Err(_) => RememberedSetMode::Barrier,
    }
}

pub struct UffdWpTracker {
    remembered_set_mode: RememberedSetMode,
    #[cfg(target_os = "linux")]
    inner: Option<Arc<LinuxUffdWpTracker>>,
}

impl Default for UffdWpTracker {
    fn default() -> Self {
        Self::new_from_env()
    }
}

impl UffdWpTracker {
    pub fn new_from_env() -> Self {
        let enabled = env::var_os("MMTK_ENABLE_UFFD_WP_TRACKER").is_some();
        if !enabled {
            return Self {
                remembered_set_mode: RememberedSetMode::Barrier,
                #[cfg(target_os = "linux")]
                inner: None,
            };
        }

        let remembered_set_mode = parse_remembered_set_mode_from_env();

        #[cfg(target_os = "linux")]
        {
            match LinuxUffdWpTracker::new(remembered_set_mode) {
                Ok(inner) => Self {
                    remembered_set_mode,
                    inner: Some(Arc::new(inner)),
                },
                Err(err) => {
                    warn!(
                        "Failed to initialize UFFD WP tracker, disabling experiment: {}",
                        err
                    );
                    Self {
                        remembered_set_mode: RememberedSetMode::Barrier,
                        inner: None,
                    }
                }
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            warn!(
                "MMTK_ENABLE_UFFD_WP_TRACKER is set, but this build is not on Linux; disabling experiment"
            );
            Self {
                remembered_set_mode: RememberedSetMode::Barrier,
            }
        }
    }

    pub fn remembered_set_mode(&self) -> RememberedSetMode {
        #[cfg(target_os = "linux")]
        {
            if self.inner.is_none() {
                return RememberedSetMode::Barrier;
            }
        }
        self.remembered_set_mode
    }

    pub fn begin_collection(&self, capture_for_current_gc: bool) {
        #[cfg(target_os = "linux")]
        if let Some(inner) = &self.inner {
            inner.begin_collection(capture_for_current_gc);
        }
    }

    pub fn take_current_gc_dirty_blocks(&self) -> Vec<usize> {
        #[cfg(target_os = "linux")]
        if let Some(inner) = &self.inner {
            return inner.take_current_gc_dirty_blocks();
        }

        vec![]
    }

    pub fn replacement_ready(&self) -> bool {
        #[cfg(target_os = "linux")]
        if let Some(inner) = &self.inner {
            return inner.replacement_ready();
        }

        false
    }

    pub fn record_shadow_barrier_object(&self, object: usize) {
        #[cfg(target_os = "linux")]
        if let Some(inner) = &self.inner {
            inner.record_shadow_barrier_object(object);
        }
    }

    pub fn record_shadow_dirty_object(&self, object: usize) {
        #[cfg(target_os = "linux")]
        if let Some(inner) = &self.inner {
            inner.record_shadow_dirty_object(object);
        }
    }

    pub fn report_shadow_compare(&self) {
        #[cfg(target_os = "linux")]
        if let Some(inner) = &self.inner {
            inner.report_shadow_compare();
        }
    }

    pub fn end_collection_for_immix_space<VM: VMBinding>(&self, immix_space: &ImmixSpace<VM>) {
        #[cfg(target_os = "linux")]
        if let Some(inner) = &self.inner {
            inner.end_collection_for_immix_space(immix_space);
        }
    }
}

#[cfg(target_os = "linux")]
fn shadow_compare_enabled() -> bool {
    env::var_os("MMTK_TRACE_UFFD_WP_RS_COMPARE").is_some()
}

#[cfg(target_os = "linux")]
fn rs_metrics_enabled() -> bool {
    env::var_os("MMTK_TRACE_RS_METRICS").is_some()
}

#[cfg(target_os = "linux")]
struct LinuxUffdWpTracker {
    uffd: libc::c_int,
    remembered_set_mode: RememberedSetMode,
    registered_blocks: Mutex<HashSet<usize>>,
    dirty_blocks: Arc<Mutex<HashSet<usize>>>,
    current_gc_dirty_blocks: Mutex<Vec<usize>>,
    faults: Arc<AtomicU64>,
    protected_epochs: AtomicU64,
    stop: Arc<AtomicBool>,
    trace_faults: bool,
    skip_protect: bool,
    compare_enabled: bool,
    metrics_enabled: bool,
    shadow_barrier_objects: Mutex<HashSet<usize>>,
    shadow_dirty_objects: Mutex<HashSet<usize>>,
}

#[cfg(target_os = "linux")]
impl LinuxUffdWpTracker {
    fn new(remembered_set_mode: RememberedSetMode) -> std::io::Result<Self> {
        let mut uffd_flags = libc::O_CLOEXEC;
        if env::var_os("MMTK_UFFD_WP_TRACKER_BLOCKING").is_none() {
            uffd_flags |= libc::O_NONBLOCK;
        }
        if env::var_os("MMTK_UFFD_WP_TRACKER_DISABLE_USER_MODE_ONLY").is_none() {
            uffd_flags |= UFFD_USER_MODE_ONLY;
        }
        let uffd = unsafe { libc::syscall(libc::SYS_userfaultfd, uffd_flags) } as libc::c_int;
        if uffd < 0 {
            return Err(std::io::Error::last_os_error());
        }

        let mut api = UffdioApi {
            api: UFFD_API,
            features: UFFD_FEATURE_PAGEFAULT_FLAG_WP | UFFD_FEATURE_EXACT_ADDRESS,
            ioctls: 0,
        };
        let ret = unsafe { libc::ioctl(uffd, UFFDIO_API, &mut api as *mut _) };
        if ret != 0 {
            let err = std::io::Error::last_os_error();
            unsafe {
                libc::close(uffd);
            }
            return Err(err);
        }

        let tracker = Self {
            uffd,
            remembered_set_mode,
            registered_blocks: Mutex::new(HashSet::new()),
            dirty_blocks: Arc::new(Mutex::new(HashSet::new())),
            current_gc_dirty_blocks: Mutex::new(vec![]),
            faults: Arc::new(AtomicU64::new(0)),
            protected_epochs: AtomicU64::new(0),
            stop: Arc::new(AtomicBool::new(false)),
            trace_faults: env::var_os("MMTK_TRACE_UFFD_WP_TRACKER").is_some(),
            skip_protect: env::var_os("MMTK_UFFD_WP_TRACKER_SKIP_PROTECT").is_some(),
            compare_enabled: shadow_compare_enabled(),
            metrics_enabled: rs_metrics_enabled(),
            shadow_barrier_objects: Mutex::new(HashSet::new()),
            shadow_dirty_objects: Mutex::new(HashSet::new()),
        };
        tracker.spawn_handler_thread();
        if tracker.trace_faults {
            eprintln!(
                "MMTK UFFD WP tracker: initialized skip_protect={} remembered_set_mode={:?} compare_enabled={} metrics_enabled={} nonblock={} user_mode_only={}",
                tracker.skip_protect,
                tracker.remembered_set_mode,
                tracker.compare_enabled,
                tracker.metrics_enabled,
                (uffd_flags & libc::O_NONBLOCK) != 0,
                (uffd_flags & UFFD_USER_MODE_ONLY) != 0
            );
        }
        info!("Initialized experimental UFFD write-protect tracker");
        Ok(tracker)
    }

    fn spawn_handler_thread(&self) {
        let uffd = self.uffd;
        let dirty_blocks = Arc::clone(&self.dirty_blocks);
        let faults = Arc::clone(&self.faults);
        let stop = Arc::clone(&self.stop);
        let trace_faults = self.trace_faults;

        thread::Builder::new()
            .name("mmtk-uffd-wp".into())
            .spawn(move || {
                let mut pollfd = libc::pollfd {
                    fd: uffd,
                    events: libc::POLLIN,
                    revents: 0,
                };

                loop {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }

                    pollfd.revents = 0;
                    let poll_ret = unsafe { libc::poll(&mut pollfd as *mut _, 1, 100) };
                    if poll_ret < 0 {
                        let err = std::io::Error::last_os_error();
                        if err.raw_os_error() == Some(libc::EINTR) {
                            continue;
                        }
                        warn!("UFFD WP tracker poll failed: {}", err);
                        break;
                    }
                    if poll_ret == 0 {
                        continue;
                    }
                    if trace_faults {
                        eprintln!(
                            "MMTK UFFD WP tracker: poll_ret={} revents=0x{:x}",
                            poll_ret,
                            pollfd.revents
                        );
                    }
                    if (pollfd.revents
                        & (libc::POLLIN | libc::POLLERR | libc::POLLHUP | libc::POLLNVAL))
                        == 0
                    {
                        continue;
                    }

                    let mut msg = MaybeUninit::<UffdMsg>::zeroed();
                    let read_ret = unsafe {
                        libc::read(
                            uffd,
                            msg.as_mut_ptr() as *mut libc::c_void,
                            std::mem::size_of::<UffdMsg>(),
                        )
                    };
                    if read_ret < 0 {
                        let err = std::io::Error::last_os_error();
                        if trace_faults {
                            eprintln!(
                                "MMTK UFFD WP tracker: read failed revents=0x{:x} err={}",
                                pollfd.revents,
                                err
                            );
                        }
                        if matches!(err.raw_os_error(), Some(libc::EAGAIN) | Some(libc::EINTR)) {
                            continue;
                        }
                        warn!("UFFD WP tracker read failed: {}", err);
                        break;
                    }
                    if read_ret == 0 {
                        if trace_faults {
                            eprintln!(
                                "MMTK UFFD WP tracker: read EOF revents=0x{:x}",
                                pollfd.revents
                            );
                        }
                        warn!("UFFD WP tracker reached EOF on uffd");
                        break;
                    }
                    if read_ret as usize != std::mem::size_of::<UffdMsg>() {
                        if trace_faults {
                            eprintln!(
                                "MMTK UFFD WP tracker: short read revents=0x{:x} expected={} got={}",
                                pollfd.revents,
                                std::mem::size_of::<UffdMsg>(),
                                read_ret
                            );
                        }
                        warn!(
                            "UFFD WP tracker short read: expected {}, got {}",
                            std::mem::size_of::<UffdMsg>(),
                            read_ret
                        );
                        continue;
                    }

                    let msg = unsafe { msg.assume_init() };
                    if msg.event != UFFD_EVENT_PAGEFAULT {
                        continue;
                    }

                    let pagefault = unsafe { msg.arg.pagefault };
                    if (pagefault.flags & UFFD_PAGEFAULT_FLAG_WP) == 0 {
                        continue;
                    }

                    let page = page_align_down(pagefault.address as usize);
                    let block = block_align_down(page);
                    dirty_blocks.lock().unwrap().insert(block);
                    faults.fetch_add(1, Ordering::Relaxed);
                    if trace_faults {
                        eprintln!(
                            "MMTK UFFD WP tracker: write fault page={:#x} block={:#x}",
                            page, block
                        );
                    }

                    let mut wp = UffdioWriteProtect {
                        range: UffdioRange {
                            start: block as u64,
                            len: Block::BYTES as u64,
                        },
                        mode: 0,
                    };
                    let ret = unsafe { libc::ioctl(uffd, UFFDIO_WRITEPROTECT, &mut wp as *mut _) };
                    if ret != 0 {
                        warn!(
                            "UFFD WP tracker failed to unprotect block {:#x}: {}",
                            block,
                            std::io::Error::last_os_error()
                        );
                    }
                }
            })
            .expect("failed to spawn UFFD WP tracker thread");
    }

    fn begin_collection(&self, capture_for_current_gc: bool) {
        let epoch = self.protected_epochs.load(Ordering::Relaxed);
        if self.trace_faults {
            eprintln!(
                "MMTK UFFD WP tracker: begin_collection epoch={} start capture_for_current_gc={}",
                epoch + 1,
                capture_for_current_gc
            );
        }
        if epoch > 0 {
            let faults = self.faults.swap(0, Ordering::Relaxed);
            let dirty_blocks = self.dirty_blocks.lock().unwrap().len();
            let tracked_blocks = self.registered_blocks.lock().unwrap().len();
            let tracked_pages_per_block = Block::BYTES / PAGE_SIZE;
            let msg = format!(
                "MMTK UFFD WP tracker epoch {} summary: tracked_blocks={}, tracked_pages={}, dirty_blocks={}, write_faults={}",
                epoch,
                tracked_blocks,
                tracked_blocks * tracked_pages_per_block,
                dirty_blocks,
                faults,
            );
            if self.trace_faults || self.metrics_enabled {
                eprintln!("{}", msg);
            }
            info!("{}", msg);
        }

        let next_current_gc_dirty_blocks = {
            let mut dirty_blocks = self.dirty_blocks.lock().unwrap();
            if capture_for_current_gc {
                dirty_blocks.drain().collect()
            } else {
                dirty_blocks.clear();
                vec![]
            }
        };
        *self.current_gc_dirty_blocks.lock().unwrap() = next_current_gc_dirty_blocks;
        if self.compare_enabled && capture_for_current_gc {
            self.shadow_barrier_objects.lock().unwrap().clear();
            self.shadow_dirty_objects.lock().unwrap().clear();
        }

        let blocks: Vec<usize> = self.registered_blocks.lock().unwrap().iter().copied().collect();
        if !self.skip_protect {
            for block in &blocks {
                if let Err(err) = self.write_protect(*block, Block::BYTES, false) {
                    warn!(
                        "Failed to remove UFFD write protection for block {:#x}: {}",
                        block, err
                    );
                    if self.trace_faults {
                        eprintln!(
                            "MMTK UFFD WP tracker: unprotect failed block={:#x} err={}",
                            block, err
                        );
                    }
                }
            }
        }
        if self.trace_faults {
            eprintln!(
                "MMTK UFFD WP tracker: begin_collection epoch={} done unprotected_blocks={} current_gc_dirty_blocks={} skip_protect={}",
                epoch + 1,
                blocks.len(),
                self.current_gc_dirty_blocks.lock().unwrap().len(),
                self.skip_protect
            );
        }
    }

    fn take_current_gc_dirty_blocks(&self) -> Vec<usize> {
        mem::take(&mut *self.current_gc_dirty_blocks.lock().unwrap())
    }

    fn replacement_ready(&self) -> bool {
        self.protected_epochs.load(Ordering::Relaxed) > 0
    }

    fn record_shadow_barrier_object(&self, object: usize) {
        if self.compare_enabled && self.remembered_set_mode == RememberedSetMode::Shadow {
            self.shadow_barrier_objects.lock().unwrap().insert(object);
        }
    }

    fn record_shadow_dirty_object(&self, object: usize) {
        if self.compare_enabled && self.remembered_set_mode == RememberedSetMode::Shadow {
            self.shadow_dirty_objects.lock().unwrap().insert(object);
        }
    }

    fn report_shadow_compare(&self) {
        if !(self.compare_enabled && self.remembered_set_mode == RememberedSetMode::Shadow) {
            return;
        }

        let barrier = self.shadow_barrier_objects.lock().unwrap();
        let dirty = self.shadow_dirty_objects.lock().unwrap();
        let barrier_only: Vec<usize> = barrier.difference(&dirty).copied().take(8).collect();
        let dirty_only: Vec<usize> = dirty.difference(&barrier).copied().take(8).collect();
        let msg = format!(
            "MMTK UFFD WP tracker shadow compare: barrier_objects={} dirty_objects={} barrier_only={} dirty_only={} barrier_only_sample={:x?} dirty_only_sample={:x?}",
            barrier.len(),
            dirty.len(),
            barrier.len().saturating_sub(barrier.intersection(&dirty).count()),
            dirty.len().saturating_sub(barrier.intersection(&dirty).count()),
            barrier_only,
            dirty_only,
        );
        if self.trace_faults || self.compare_enabled || self.metrics_enabled {
            eprintln!("{}", msg);
        }
        info!("{}", msg);
    }

    fn end_collection_for_immix_space<VM: VMBinding>(&self, immix_space: &ImmixSpace<VM>) {
        let next_epoch = self.protected_epochs.load(Ordering::Relaxed) + 1;
        if self.trace_faults {
            eprintln!(
                "MMTK UFFD WP tracker: end_collection epoch={} start",
                next_epoch
            );
        }
        let current_blocks: HashSet<usize> = immix_space
            .chunk_map
            .all_chunks()
            .flat_map(|chunk| chunk.iter_region::<Block>())
            .filter(|block| block.may_have_objects())
            .map(|block| block.start().as_usize())
            .collect();

        let mut registered = self.registered_blocks.lock().unwrap();

        let stale_blocks: Vec<usize> = registered
            .iter()
            .copied()
            .filter(|block| !current_blocks.contains(block))
            .collect();
        for block in stale_blocks {
            if let Err(err) = self.unregister_range(block, Block::BYTES) {
                warn!(
                    "Failed to unregister stale UFFD range for block {:#x}: {}",
                    block, err
                );
                if self.trace_faults {
                    eprintln!(
                        "MMTK UFFD WP tracker: unregister failed block={:#x} err={}",
                        block, err
                    );
                }
            }
            registered.remove(&block);
        }

        let new_blocks: Vec<usize> = current_blocks
            .iter()
            .copied()
            .filter(|block| !registered.contains(block))
            .collect();
        for block in &new_blocks {
            if let Err(err) = self.register_range(*block, Block::BYTES) {
                warn!(
                    "Failed to register UFFD WP range for block {:#x}: {}",
                    block, err
                );
                if self.trace_faults {
                    eprintln!(
                        "MMTK UFFD WP tracker: register failed block={:#x} err={}",
                        block, err
                    );
                }
                continue;
            }
            if self.trace_faults {
                eprintln!("MMTK UFFD WP tracker: registered block={:#x} bytes={}", block, Block::BYTES);
            }
            registered.insert(*block);
        }

        let protected_blocks: Vec<usize> = registered.iter().copied().collect();
        let protected_count = protected_blocks.len();
        drop(registered);

        if !self.skip_protect {
            for block in protected_blocks {
                if let Err(err) = self.write_protect(block, Block::BYTES, true) {
                    warn!(
                        "Failed to enable UFFD write protection for block {:#x}: {}",
                        block, err
                    );
                    if self.trace_faults {
                        eprintln!(
                            "MMTK UFFD WP tracker: protect failed block={:#x} err={}",
                            block, err
                        );
                    }
                } else if self.trace_faults {
                    eprintln!(
                        "MMTK UFFD WP tracker: protected block={:#x} bytes={}",
                        block,
                        Block::BYTES
                    );
                }
            }
        }

        self.protected_epochs.fetch_add(1, Ordering::Relaxed);
        if self.trace_faults {
            eprintln!(
                "MMTK UFFD WP tracker: end_collection epoch={} done tracked_blocks={} protected_blocks={} skip_protect={}",
                next_epoch,
                current_blocks.len(),
                protected_count,
                self.skip_protect
            );
        }
    }

    fn register_range(&self, start: usize, len: usize) -> std::io::Result<()> {
        let mut reg = UffdioRegister {
            range: UffdioRange {
                start: start as u64,
                len: len as u64,
            },
            mode: UFFDIO_REGISTER_MODE_WP,
            ioctls: 0,
        };
        let ret = unsafe { libc::ioctl(self.uffd, UFFDIO_REGISTER, &mut reg as *mut _) };
        if ret != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    fn unregister_range(&self, start: usize, len: usize) -> std::io::Result<()> {
        let mut range = UffdioRange {
            start: start as u64,
            len: len as u64,
        };
        let ret = unsafe { libc::ioctl(self.uffd, UFFDIO_UNREGISTER, &mut range as *mut _) };
        if ret != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    fn write_protect(&self, start: usize, len: usize, protect: bool) -> std::io::Result<()> {
        let mut wp = UffdioWriteProtect {
            range: UffdioRange {
                start: start as u64,
                len: len as u64,
            },
            mode: if protect {
                UFFDIO_WRITEPROTECT_MODE_WP
            } else {
                0
            },
        };
        let ret = unsafe { libc::ioctl(self.uffd, UFFDIO_WRITEPROTECT, &mut wp as *mut _) };
        if ret != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
impl Drop for LinuxUffdWpTracker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        unsafe {
            libc::close(self.uffd);
        }
    }
}

#[cfg(target_os = "linux")]
fn page_align_down(addr: usize) -> usize {
    addr & !(PAGE_SIZE - 1)
}

#[cfg(target_os = "linux")]
fn block_align_down(addr: usize) -> usize {
    addr & !(Block::BYTES - 1)
}

#[cfg(target_os = "linux")]
const UFFD_API: u64 = 0xAA;
#[cfg(target_os = "linux")]
const UFFDIO_API: libc::c_ulong = 0xc018aa3f;
#[cfg(target_os = "linux")]
const UFFDIO_REGISTER: libc::c_ulong = 0xc020aa00;
#[cfg(target_os = "linux")]
const UFFDIO_UNREGISTER: libc::c_ulong = 0x8010aa01;
#[cfg(target_os = "linux")]
const UFFDIO_WRITEPROTECT: libc::c_ulong = 0xc018aa06;
#[cfg(target_os = "linux")]
const UFFDIO_REGISTER_MODE_WP: u64 = 1 << 1;
#[cfg(target_os = "linux")]
const UFFDIO_WRITEPROTECT_MODE_WP: u64 = 1 << 0;
#[cfg(target_os = "linux")]
const UFFD_EVENT_PAGEFAULT: u8 = 0x12;
#[cfg(target_os = "linux")]
const UFFD_PAGEFAULT_FLAG_WP: u64 = 1 << 1;
#[cfg(target_os = "linux")]
const UFFD_FEATURE_PAGEFAULT_FLAG_WP: u64 = 1 << 0;
#[cfg(target_os = "linux")]
const UFFD_FEATURE_EXACT_ADDRESS: u64 = 1 << 11;
#[cfg(target_os = "linux")]
const UFFD_USER_MODE_ONLY: libc::c_int = 1;

#[cfg(target_os = "linux")]
#[repr(C)]
struct UffdioApi {
    api: u64,
    features: u64,
    ioctls: u64,
}

#[cfg(target_os = "linux")]
#[repr(C)]
struct UffdioRange {
    start: u64,
    len: u64,
}

#[cfg(target_os = "linux")]
#[repr(C)]
struct UffdioRegister {
    range: UffdioRange,
    mode: u64,
    ioctls: u64,
}

#[cfg(target_os = "linux")]
#[repr(C)]
struct UffdioWriteProtect {
    range: UffdioRange,
    mode: u64,
}

#[cfg(target_os = "linux")]
#[repr(C)]
#[derive(Copy, Clone)]
struct UffdMsg {
    event: u8,
    reserved1: u8,
    reserved2: u16,
    reserved3: u32,
    arg: UffdMsgArg,
}

#[cfg(target_os = "linux")]
#[repr(C)]
#[derive(Copy, Clone)]
union UffdMsgArg {
    pagefault: UffdMsgPagefault,
    pad: [u8; 24],
}

#[cfg(target_os = "linux")]
#[repr(C)]
#[derive(Copy, Clone)]
struct UffdMsgPagefault {
    flags: u64,
    address: u64,
    feat: u64,
}
