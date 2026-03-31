use crate::plan::VectorObjectQueue;
use crate::policy::compressor::forwarding;
use crate::policy::gc_work::{TraceKind, TRACE_KIND_TRANSITIVE_PIN};
use crate::policy::largeobjectspace::LargeObjectSpace;
use crate::policy::sft::{GCWorkerMutRef, SFT};
use crate::policy::space::{CommonSpace, Space};
#[cfg(feature = "uffd")]
use crate::plan::AllocationSemantics;
use crate::scheduler::{GCWork, GCWorkScheduler, GCWorker, WorkBucketStage};
#[cfg(feature = "uffd")]
use crate::util::alloc::BumpAllocator;
use crate::util::copy::CopySemantics;
use crate::util::heap::regionpageresource::AllocatedRegion;
use crate::util::heap::{PageResource, RegionPageResource};
use crate::util::linear_scan::Region;
use crate::util::metadata::extract_side_metadata;
#[cfg(feature = "vo_bit")]
use crate::util::metadata::vo_bit;
use crate::util::metadata::MetadataSpec;
use crate::util::object_enum::{self, ObjectEnumerator};
use crate::util::{Address, ObjectReference};
use crate::vm::slot::Slot;
use crate::MMTK;
use crate::{vm::*, ObjectQueue};
use atomic::Ordering;
#[cfg(feature = "uffd")]
use std::collections::HashSet;
#[cfg(feature = "uffd")]
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::Arc;
#[cfg(feature = "uffd")]
use std::sync::{Mutex, OnceLock, RwLock};

pub(crate) const TRACE_KIND_MARK: TraceKind = 0;
pub(crate) const TRACE_KIND_FORWARD_ROOT: TraceKind = 1;

#[cfg(feature = "uffd")]
fn compressor_perf_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("MMTK_TRACE_COMPRESSOR_PERF").is_some())
}

/// [`CompressorSpace`] is a stop-the-world implementation of
/// the Compressor, as described in Kermany and Petrank,
/// [The Compressor: concurrent, incremental, and parallel compaction](https://dl.acm.org/doi/10.1145/1133255.1134023).
///
/// [`CompressorSpace`] makes two main diversions from the paper
/// (aside from [`CompressorSpace`] being stop-the-world):
/// - The heap is structured into regions ([`forwarding::CompressorRegion`])
///   which the collector compacts separately.
/// - The collector compacts each region in-place, instead of using two virtual
///   spaces as in Kermany and Petrank. The virtual spaces side-step a race which
///   would occur if multiple threads attempted to compact one heap in place: one thread
///   could move an object to the location of another object which has yet to be moved by
///   another thread. Kermany and Petrank move objects between from- and to- virtual spaces,
///   preventing the old objects from being overwritten. (They reclaim memory by unmapping
///   pages of the from-virtual space after moving all objects out of said pages.)
///   We instead side-step this race by assigning only a single thread to each region, and
///   running multiple single-threaded Compressors at once.
pub struct CompressorSpace<VM: VMBinding> {
    common: CommonSpace<VM>,
    pr: RegionPageResource<VM, forwarding::CompressorRegion>,
    forwarding: forwarding::ForwardingMetadata<VM>,
    scheduler: Arc<GCWorkScheduler<VM>>,
    #[cfg(feature = "uffd")]
    page_metadata: RwLock<Vec<RegionPageMetadata>>,
    #[cfg(feature = "uffd")]
    black_allocation_buffers: Mutex<Vec<BlackAllocationBuffer>>,
    #[cfg(feature = "uffd")]
    black_allocation_tracking_active: AtomicBool,
    #[cfg(feature = "uffd")]
    compaction_region_limit: AtomicUsize,
}

#[cfg(feature = "uffd")]
#[derive(Clone, Debug)]
struct BlackAllocationBuffer {
    start: Address,
    used_end: Address,
    limit: Address,
}

#[cfg(feature = "uffd")]
#[derive(Clone, Debug)]
pub struct PageCompactMetadata {
    /// The first source object whose compacted destination overlaps this page.
    pub first_obj: Option<ObjectReference>,
    /// If the page starts in the middle of `first_obj`, this is the byte offset
    /// within that object where the page begins. Zero if the page begins at the
    /// start of `first_obj` or if the page has no live data.
    pub first_obj_page_offset: u32,
}

#[cfg(feature = "uffd")]
#[derive(Clone, Debug)]
pub struct RegionPageMetadata {
    pub region_start: Address,
    /// End of the original occupied portion of the region before compaction.
    pub source_end: Address,
    /// End of the compacted destination data in the region.
    pub compacted_end: Address,
    /// Whether any object in this region actually changes address.
    pub has_movement: bool,
    pub pages: Vec<PageCompactMetadata>,
}

impl<VM: VMBinding> SFT for CompressorSpace<VM> {
    fn name(&self) -> &'static str {
        self.get_name()
    }

    fn get_forwarded_object(&self, object: ObjectReference) -> Option<ObjectReference> {
        // Check if forwarding addresses have been calculated before attempting
        // to forward objects
        if self.forwarding.has_calculated_forwarding_addresses() {
            Some(self.forward(object, false))
        } else {
            None
        }
    }

    fn is_live(&self, object: ObjectReference) -> bool {
        Self::is_marked(object)
    }

    #[cfg(feature = "object_pinning")]
    fn pin_object(&self, _object: ObjectReference) -> bool {
        panic!("Cannot pin/unpin objects of CompressorSpace.")
    }

    #[cfg(feature = "object_pinning")]
    fn unpin_object(&self, _object: ObjectReference) -> bool {
        panic!("Cannot pin/unpin objects of CompressorSpace.")
    }

    #[cfg(feature = "object_pinning")]
    fn is_object_pinned(&self, _object: ObjectReference) -> bool {
        false
    }

    fn is_movable(&self) -> bool {
        true
    }

    fn initialize_object_metadata(&self, object: ObjectReference) {
        if self.should_allocate_as_live() {
            forwarding::MARK_SPEC.fetch_or_atomic::<u8>(
                object.to_raw_address(),
                1,
                Ordering::SeqCst,
            );
        }
        #[cfg(feature = "vo_bit")]
        crate::util::metadata::vo_bit::set_vo_bit(object);
    }

    #[cfg(feature = "sanity")]
    fn is_sane(&self) -> bool {
        true
    }

    #[cfg(feature = "vo_bit")]
    fn is_mmtk_object(&self, addr: Address) -> Option<ObjectReference> {
        crate::util::metadata::vo_bit::is_vo_bit_set_for_addr(addr)
    }

    #[cfg(feature = "vo_bit")]
    fn find_object_from_internal_pointer(
        &self,
        ptr: Address,
        max_search_bytes: usize,
    ) -> Option<ObjectReference> {
        crate::util::metadata::vo_bit::find_object_from_internal_pointer::<VM>(
            ptr,
            max_search_bytes,
        )
    }

    fn sft_trace_object(
        &self,
        _queue: &mut VectorObjectQueue,
        _object: ObjectReference,
        _worker: GCWorkerMutRef,
    ) -> ObjectReference {
        // We should not use trace_object for compressor space.
        // Depending on which trace it is, we should manually call either trace_mark or trace_forward.
        panic!("sft_trace_object() cannot be used with CompressorSpace")
    }

    fn debug_print_object_info(&self, object: ObjectReference) {
        println!("marked = {}", CompressorSpace::<VM>::is_marked(object));
        println!("forwarding = {:?}", self.get_forwarded_object(object));
        self.common.debug_print_object_global_info(object);
    }
}

impl<VM: VMBinding> Space<VM> for CompressorSpace<VM> {
    fn as_space(&self) -> &dyn Space<VM> {
        self
    }

    fn as_sft(&self) -> &(dyn SFT + Sync + 'static) {
        self
    }

    fn get_page_resource(&self) -> &dyn PageResource<VM> {
        &self.pr
    }

    fn maybe_get_page_resource_mut(&mut self) -> Option<&mut dyn PageResource<VM>> {
        Some(&mut self.pr)
    }

    fn common(&self) -> &CommonSpace<VM> {
        &self.common
    }

    fn initialize_sft(&self, sft_map: &mut dyn crate::policy::sft_map::SFTMap) {
        self.common().initialize_sft(self.as_sft(), sft_map)
    }

    fn release_multiple_pages(&mut self, _start: Address) {
        panic!("compressorspace only releases pages enmasse")
    }

    fn enumerate_objects(&self, enumerator: &mut dyn ObjectEnumerator) {
        self.pr.enumerate(enumerator);
    }

    fn on_bump_alloc_buffer_retired(&self, start: Address, cursor: Address, limit: Address) {
        #[cfg(feature = "uffd")]
        self.record_retired_bump_alloc_buffer(start, cursor, limit);
        #[cfg(not(feature = "uffd"))]
        let _ = (start, cursor, limit);
    }

    fn clear_side_log_bits(&self) {
        let log_bit = *VM::VMObjectModel::GLOBAL_LOG_BIT_SPEC;
        if log_bit.is_on_side() {
            let side_log_bit = log_bit.extract_side_spec();
            self.pr.enumerate_regions(&mut |region: &AllocatedRegion<
                forwarding::CompressorRegion,
            >| {
                let start = region.region.start();
                let end = region.cursor();
                if start < end {
                    side_log_bit.bzero_metadata(start, end - start);
                }
            });
        } else {
            let mut enumerator = object_enum::ClosureObjectEnumerator::<_, VM>::new(|object| {
                VM::VMObjectModel::GLOBAL_LOG_BIT_SPEC.clear::<VM>(object, Ordering::SeqCst);
            });
            self.pr.enumerate_regions(&mut |region: &AllocatedRegion<
                forwarding::CompressorRegion,
            >| {
                let start = region.region.start();
                let end = region.cursor();
                if start < end {
                    enumerator.visit_address_range(start, end);
                }
            });
        }
    }

    fn set_side_log_bits(&self) {
        let log_bit = *VM::VMObjectModel::GLOBAL_LOG_BIT_SPEC;
        if log_bit.is_on_side() {
            let side_log_bit = log_bit.extract_side_spec();
            self.pr.enumerate_regions(&mut |region: &AllocatedRegion<
                forwarding::CompressorRegion,
            >| {
                let start = region.region.start();
                let end = region.cursor();
                if start < end {
                    side_log_bit.bset_metadata(start, end - start);
                }
            });
        } else {
            let mut enumerator = object_enum::ClosureObjectEnumerator::<_, VM>::new(|object| {
                VM::VMObjectModel::GLOBAL_LOG_BIT_SPEC
                    .mark_as_unlogged::<VM>(object, Ordering::SeqCst);
            });
            self.pr.enumerate_regions(&mut |region: &AllocatedRegion<
                forwarding::CompressorRegion,
            >| {
                let start = region.region.start();
                let end = region.cursor();
                if start < end {
                    enumerator.visit_address_range(start, end);
                }
            });
        }
    }
}

impl<VM: VMBinding> crate::policy::gc_work::PolicyTraceObject<VM> for CompressorSpace<VM> {
    fn trace_object<Q: ObjectQueue, const KIND: crate::policy::gc_work::TraceKind>(
        &self,
        queue: &mut Q,
        object: ObjectReference,
        _copy: Option<CopySemantics>,
        _worker: &mut GCWorker<VM>,
    ) -> ObjectReference {
        debug_assert!(
            KIND != TRACE_KIND_TRANSITIVE_PIN,
            "Compressor does not support transitive pin trace."
        );
        if KIND == TRACE_KIND_MARK {
            self.trace_mark_object(queue, object)
        } else if KIND == TRACE_KIND_FORWARD_ROOT {
            self.trace_forward_root(queue, object)
        } else {
            unreachable!()
        }
    }
    fn may_move_objects<const KIND: crate::policy::gc_work::TraceKind>() -> bool {
        if KIND == TRACE_KIND_MARK {
            false
        } else if KIND == TRACE_KIND_FORWARD_ROOT {
            true
        } else {
            unreachable!()
        }
    }
}

impl<VM: VMBinding> CompressorSpace<VM> {
    pub fn new(args: crate::policy::space::PlanCreateSpaceArgs<VM>) -> Self {
        let vm_map = args.vm_map;
        assert!(
            VM::VMObjectModel::UNIFIED_OBJECT_REFERENCE_ADDRESS,
            "The Compressor requires a unified object reference address model"
        );
        let local_specs = extract_side_metadata(&[
            MetadataSpec::OnSide(forwarding::MARK_SPEC),
            MetadataSpec::OnSide(forwarding::OFFSET_VECTOR_SPEC),
        ]);
        let is_discontiguous = args.vmrequest.is_discontiguous();
        let scheduler = args.scheduler.clone();
        let common = CommonSpace::new(args.into_policy_args(true, false, local_specs));
        CompressorSpace {
            pr: if is_discontiguous {
                RegionPageResource::new_discontiguous(vm_map)
            } else {
                RegionPageResource::new_contiguous(common.start, common.extent, vm_map)
            },
            forwarding: forwarding::ForwardingMetadata::new(),
            common,
            scheduler,
            #[cfg(feature = "uffd")]
            page_metadata: RwLock::new(vec![]),
            #[cfg(feature = "uffd")]
            black_allocation_buffers: Mutex::new(vec![]),
            #[cfg(feature = "uffd")]
            black_allocation_tracking_active: AtomicBool::new(false),
            #[cfg(feature = "uffd")]
            compaction_region_limit: AtomicUsize::new(0),
        }
    }

    pub fn prepare(&self) {
        self.pr
            .enumerate_regions(&mut |r: &AllocatedRegion<forwarding::CompressorRegion>| {
                forwarding::MARK_SPEC
                    .bzero_metadata(r.region.start(), r.region.end() - r.region.start());
            });
    }

    pub fn release(&self) {
        self.forwarding.release();
    }

    pub fn trace_mark_object<Q: ObjectQueue>(
        &self,
        queue: &mut Q,
        object: ObjectReference,
    ) -> ObjectReference {
        #[cfg(feature = "vo_bit")]
        debug_assert!(
            crate::util::metadata::vo_bit::is_vo_bit_set(object),
            "{:x}: VO bit not set",
            object
        );
        if CompressorSpace::<VM>::test_and_mark(object) {
            queue.enqueue(object);
            self.forwarding.mark_last_word_of_object(object);
        }
        object
    }

    pub fn trace_forward_root<Q: ObjectQueue>(
        &self,
        _queue: &mut Q,
        object: ObjectReference,
    ) -> ObjectReference {
        self.forward(object, true)
    }

    pub fn test_and_mark(object: ObjectReference) -> bool {
        forwarding::MARK_SPEC
            .fetch_update_atomic::<u8, _>(
                object.to_raw_address(),
                Ordering::SeqCst,
                Ordering::Relaxed,
                |v| {
                    if v == 0 {
                        Some(1)
                    } else {
                        None
                    }
                },
            )
            .is_ok()
    }

    pub fn is_marked(object: ObjectReference) -> bool {
        let mark_bit =
            forwarding::MARK_SPEC.load_atomic::<u8>(object.to_raw_address(), Ordering::SeqCst);
        mark_bit == 1
    }

    fn generate_tasks(
        &self,
        f: &mut impl FnMut(&AllocatedRegion<forwarding::CompressorRegion>, usize) -> Box<dyn GCWork<VM>>,
    ) -> Vec<Box<dyn GCWork<VM>>> {
        let mut packets = vec![];
        let mut index = 0;
        self.pr.enumerate_regions(&mut |r| {
            packets.push(f(r, index));
            index += 1;
        });
        packets
    }

    pub fn add_offset_vector_tasks(&'static self) {
        let offset_vector_packets: Vec<Box<dyn GCWork<VM>>> = self.generate_tasks(&mut |r, _| {
            Box::new(CalculateOffsetVector::<VM>::new(self, r.region, r.cursor()))
        });
        self.scheduler.work_buckets[WorkBucketStage::CalculateForwarding]
            .bulk_add(offset_vector_packets);
    }

    pub fn calculate_offset_vector_for_region(
        &self,
        region: forwarding::CompressorRegion,
        cursor: Address,
    ) {
        self.forwarding.calculate_offset_vector(region, cursor);
    }

    #[cfg(feature = "uffd")]
    pub fn calculate_offset_vectors_for_region_prefix(&self, count: usize) {
        self.pr.with_regions(&mut |regions| {
            for r in regions.iter().take(count) {
                self.forwarding
                    .calculate_offset_vector(r.region, r.cursor());
            }
        });
    }

    /// Build page-granular compaction metadata for all regions.
    ///
    /// This is the main prerequisite for ART-style page-fault-driven compaction:
    /// for each destination page, we record the first source object whose compacted
    /// bytes overlap that page, and the byte offset within that object where the
    /// page begins.
    #[cfg(feature = "uffd")]
    pub fn build_all_page_metadata(&self) {
        self.build_page_metadata_for_region_prefix(self.num_regions());
    }

    #[cfg(feature = "uffd")]
    pub fn build_page_metadata_for_region_prefix(&self, count: usize) {
        let metadata = self.pr.with_regions(&mut |regions| {
            regions
                .iter()
                .take(count)
                .map(|r| self.build_region_page_metadata(r.region.start(), r.cursor()))
                .collect::<Vec<_>>()
        });
        let total_pages: usize = metadata.iter().map(|m| m.pages.len()).sum();
        if compressor_perf_trace_enabled() {
            info!(
                "Compressor page metadata built: {} regions, {} destination pages",
                metadata.len(),
                total_pages
            );
        }
        *self.page_metadata.write().unwrap() = metadata;
    }

    #[cfg(feature = "uffd")]
    fn with_region_page_metadata<T>(
        &self,
        index: usize,
        f: impl FnOnce(&RegionPageMetadata) -> T,
    ) -> Option<T> {
        let guard = self.page_metadata.read().unwrap();
        guard.get(index).map(f)
    }

    /// Get page metadata for a region previously built by `build_all_page_metadata`.
    #[cfg(feature = "uffd")]
    pub fn region_page_metadata(&self, index: usize) -> Option<RegionPageMetadata> {
        let guard = self.page_metadata.read().unwrap();
        guard.get(index).cloned()
    }

    #[cfg(feature = "uffd")]
    fn rewrite_object_from_shadow_into_temp(
        &self,
        region_start: Address,
        obj: ObjectReference,
        shadow_region_start: Address,
        temp_obj_words: &mut Vec<usize>,
    ) -> (Address, usize) {
        let obj_shadow_addr = shadow_region_start + (obj.to_raw_address() - region_start);
        let shadow_obj = ObjectReference::from_raw_address(obj_shadow_addr)
            .expect("shadow object address should be valid");
        let copied_size = VM::VMObjectModel::get_size_when_copied(shadow_obj);
        let words = copied_size.div_ceil(std::mem::size_of::<usize>());
        if temp_obj_words.len() < words {
            temp_obj_words.resize(words, 0);
        }
        let temp_obj_addr = Address::from_mut_ptr(temp_obj_words.as_mut_ptr() as *mut u8);
        unsafe {
            std::ptr::copy_nonoverlapping(
                obj_shadow_addr.to_ptr::<u8>(),
                temp_obj_addr.to_mut_ptr::<u8>(),
                copied_size,
            );
        }
        let temp_obj = unsafe { ObjectReference::from_raw_address_unchecked(temp_obj_addr) };
        VM::VMScanning::scan_object_for_slot_rewrite(
            crate::util::opaque_pointer::VMWorkerThread(
                crate::util::opaque_pointer::VMThread::UNINITIALIZED,
            ),
            temp_obj,
            &mut |slot: VM::VMSlot| {
                if let Some(o) = slot.load() {
                    let new_ref = if self.in_space(o) && Self::is_marked(o) {
                        self.forward(o, false)
                    } else {
                        o
                    };
                    if new_ref != o {
                        slot.store(new_ref);
                    }
                }
            },
        );
        (temp_obj_addr, copied_size)
    }

    #[cfg(feature = "uffd")]
    pub fn build_region_from_shadow(
        &self,
        region_index: usize,
        shadow_region_start: Address,
        dst_region: &mut [u8],
    ) -> bool {
        let Some((region_start, source_end, compacted_end)) = self
            .with_region_page_metadata(region_index, |meta| {
                (meta.region_start, meta.source_end, meta.compacted_end)
            })
        else {
            return false;
        };
        if compacted_end <= region_start {
            return false;
        }
        dst_region.fill(0);
        let mut copied_any = false;
        let mut temp_obj_words = Vec::new();
        self.forwarding.scan_marked_objects(
            region_start,
            source_end,
            &mut |obj: ObjectReference| {
                let new_obj = self.forward(obj, false);
                let dst_off = new_obj.to_raw_address() - region_start;
                let (temp_obj_addr, copied_size) = self.rewrite_object_from_shadow_into_temp(
                    region_start,
                    obj,
                    shadow_region_start,
                    &mut temp_obj_words,
                );
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        temp_obj_addr.to_ptr::<u8>(),
                        dst_region.as_mut_ptr().add(dst_off),
                        copied_size,
                    );
                }
                copied_any = true;
            },
        );
        copied_any
    }

    /// Reconstruct one compacted destination page from a shadow copy of the region.
    ///
    /// This is a raw-byte page builder: it copies the portions of compacted objects that
    /// overlap the requested destination page from the region shadow into `dst_page`.
    ///
    /// IMPORTANT: this method does **not** update references yet. It is the low-level
    #[cfg(feature = "uffd")]
    /// Reconstruct one destination page from the shadow copy of a region.
    ///
    /// Copies object data from the shadow and rewrites internal references
    /// so that the resulting page is self-consistent for mutator access.
    ///
    /// Returns `true` if any live data was copied into the page, `false` if the page is
    /// logically empty and should be zero-filled.
    pub fn build_page_from_shadow(
        &self,
        region_index: usize,
        page_index: usize,
        shadow_region_start: Address,
        dst_page: &mut [u8],
    ) -> bool {
        const PAGE_SIZE: usize = crate::util::constants::BYTES_IN_PAGE;
        assert!(
            dst_page.len() >= PAGE_SIZE,
            "dst_page must be at least one page"
        );
        dst_page[..PAGE_SIZE].fill(0);

        self.with_region_page_metadata(region_index, |meta| {
            if page_index >= meta.pages.len() {
                return false;
            }
            let Some(first_obj) = meta.pages[page_index].first_obj else {
                return false;
            };

            let region_start = meta.region_start;
            let region_cursor = meta.source_end;
            let page_start = meta.region_start + page_index * PAGE_SIZE;
            let page_end = page_start + PAGE_SIZE;

            let mut started = false;
            let mut done = false;
            let mut copied_any = false;
            let mut temp_obj_words: Vec<usize> = Vec::new();

            self.forwarding.scan_marked_objects(
                first_obj.to_raw_address(),
                region_cursor,
                &mut |obj: ObjectReference| {
                    if done {
                        return;
                    }
                    if !started {
                        debug_assert_eq!(obj, first_obj);
                        started = true;
                    }

                    let (temp_obj_addr, copied_size) = self.rewrite_object_from_shadow_into_temp(
                        region_start,
                        obj,
                        shadow_region_start,
                        &mut temp_obj_words,
                    );
                    let new_obj = self.forward(obj, false);
                    let dst_start = new_obj.to_raw_address();
                    let dst_end = dst_start + copied_size;

                    if dst_start >= page_end {
                        done = true;
                        return;
                    }
                    if dst_end <= page_start {
                        return;
                    }

                    let overlap_start = if dst_start > page_start {
                        dst_start
                    } else {
                        page_start
                    };
                    let overlap_end = if dst_end < page_end {
                        dst_end
                    } else {
                        page_end
                    };
                    if overlap_end <= overlap_start {
                        return;
                    }

                    let src_offset = overlap_start - dst_start;
                    let dst_offset = overlap_start - page_start;
                    let copy_len = overlap_end - overlap_start;

                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            temp_obj_addr.add(src_offset).to_ptr::<u8>(),
                            dst_page.as_mut_ptr().add(dst_offset),
                            copy_len,
                        );
                    }
                    copied_any = true;
                },
            );

            copied_any
        })
        .unwrap_or(false)
    }

    /// Validate the resolved compacted bytes in a region against a direct
    /// object-by-object copy from the shadow region.
    #[cfg(feature = "uffd")]
    pub fn validate_region_compaction_from_shadow(
        &self,
        region_index: usize,
        shadow_region_start: Address,
    ) -> Result<(), String> {
        let Some((region_start, source_end, compacted_end)) = self
            .with_region_page_metadata(region_index, |meta| {
                (meta.region_start, meta.source_end, meta.compacted_end)
            })
        else {
            return Ok(());
        };
        let live_bytes = compacted_end - region_start;
        if live_bytes == 0 {
            return Ok(());
        }

        let mut expected = vec![0u8; live_bytes];
        self.forwarding.scan_marked_objects(
            region_start,
            source_end,
            &mut |obj: ObjectReference| {
                let obj_shadow_addr = shadow_region_start + (obj.to_raw_address() - region_start);
                let shadow_obj = ObjectReference::from_raw_address(obj_shadow_addr)
                    .expect("shadow object address should be valid");
                let copied_size = VM::VMObjectModel::get_size_when_copied(shadow_obj);
                let new_obj = self.forward(obj, false);
                let dst_off = new_obj.to_raw_address() - region_start;
                let words = copied_size.div_ceil(std::mem::size_of::<usize>());
                let mut temp_obj_words = vec![0usize; words];
                let temp_obj_addr = Address::from_mut_ptr(temp_obj_words.as_mut_ptr() as *mut u8);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        obj_shadow_addr.to_ptr::<u8>(),
                        temp_obj_addr.to_mut_ptr::<u8>(),
                        copied_size,
                    );
                }
                let temp_obj =
                    unsafe { ObjectReference::from_raw_address_unchecked(temp_obj_addr) };
                VM::VMScanning::scan_object_for_slot_rewrite(
                    crate::util::opaque_pointer::VMWorkerThread(
                        crate::util::opaque_pointer::VMThread::UNINITIALIZED,
                    ),
                    temp_obj,
                    &mut |slot: VM::VMSlot| {
                        if let Some(o) = slot.load() {
                            let new_ref = if self.in_space(o) && Self::is_marked(o) {
                                self.forward(o, false)
                            } else {
                                o
                            };
                            if new_ref != o {
                                slot.store(new_ref);
                            }
                        }
                    },
                );
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        temp_obj_addr.to_ptr::<u8>(),
                        expected.as_mut_ptr().add(dst_off),
                        copied_size,
                    );
                }
            },
        );

        let actual = unsafe { std::slice::from_raw_parts(region_start.to_ptr::<u8>(), live_bytes) };
        if expected != actual {
            for i in 0..live_bytes {
                if expected[i] != actual[i] {
                    return Err(format!(
                        "region {} mismatch at byte {} (expected 0x{:02x}, actual 0x{:02x})",
                        region_index, i, expected[i], actual[i]
                    ));
                }
            }
            return Err(format!(
                "region {} mismatch with unknown differing byte",
                region_index
            ));
        }

        Ok(())
    }

    /// Collect all valid post-compaction destination object starts.
    #[cfg(feature = "uffd")]
    pub fn collect_destination_objects(&self) -> HashSet<ObjectReference> {
        let page_metadata = self.page_metadata.read().unwrap().clone();
        let mut objects = HashSet::new();
        for meta in page_metadata {
            self.forwarding.scan_marked_objects(
                meta.region_start,
                meta.source_end,
                &mut |obj: ObjectReference| {
                    objects.insert(self.forward(obj, false));
                },
            );
        }
        objects
    }

    /// Validate that all Compressor-space references inside compacted Compressor objects
    /// point to valid destination object starts.
    #[cfg(feature = "uffd")]
    pub fn validate_updated_references(
        &self,
        worker: &mut GCWorker<VM>,
        valid_objects: &HashSet<ObjectReference>,
    ) -> Result<(), String> {
        let page_metadata = self.page_metadata.read().unwrap().clone();
        for meta in page_metadata {
            let mut err = None;
            self.forwarding.scan_marked_objects(
                meta.region_start,
                meta.source_end,
                &mut |obj: ObjectReference| {
                    if err.is_some() {
                        return;
                    }
                    let new_object = self.forward(obj, false);
                    if !VM::VMObjectModel::is_object_sane(new_object) {
                        err = Some(format!(
                            "invalid destination object after fixup: {} (from {})",
                            new_object, obj
                        ));
                        return;
                    }
                    VM::VMScanning::scan_object(worker.tls, new_object, &mut |s: VM::VMSlot| {
                        if err.is_some() {
                            return;
                        }
                        if let Some(referent) = s.load() {
                            if self.in_space(referent) && !valid_objects.contains(&referent) {
                                err = Some(format!(
                                    "stale/invalid Compressor ref {} found in object {} (from {})",
                                    referent, new_object, obj
                                ));
                            }
                        }
                    });
                },
            );
            if let Some(err) = err {
                return Err(err);
            }
        }
        Ok(())
    }

    #[cfg(feature = "uffd")]
    fn build_region_page_metadata(
        &self,
        region_start: Address,
        cursor: Address,
    ) -> RegionPageMetadata {
        const PAGE_SIZE: usize = crate::util::constants::BYTES_IN_PAGE;

        let mut compacted_end = region_start;
        let mut has_movement = false;
        let mut pages: Vec<PageCompactMetadata> = Vec::new();

        self.forwarding
            .scan_marked_objects(region_start, cursor, &mut |obj: ObjectReference| {
                let copied_size = VM::VMObjectModel::get_size_when_copied(obj);
                let new_obj = self.forward(obj, false);
                let dst_start = new_obj.to_raw_address();
                let dst_end = dst_start + copied_size;

                has_movement |= dst_start != obj.to_raw_address();
                if dst_end > compacted_end {
                    compacted_end = dst_end;
                }
                if dst_end <= region_start {
                    return;
                }

                let required_pages = (dst_end - region_start).div_ceil(PAGE_SIZE);
                if pages.len() < required_pages {
                    pages.resize(
                        required_pages,
                        PageCompactMetadata {
                            first_obj: None,
                            first_obj_page_offset: 0,
                        },
                    );
                }

                let first_page = (dst_start - region_start) / PAGE_SIZE;
                let last_page = (dst_end - 1 - region_start) / PAGE_SIZE;
                for page_idx in first_page..=last_page {
                    if pages[page_idx].first_obj.is_none() {
                        let page_start = region_start + page_idx * PAGE_SIZE;
                        let offset = if page_start > dst_start {
                            page_start - dst_start
                        } else {
                            0
                        };
                        pages[page_idx].first_obj = Some(obj);
                        pages[page_idx].first_obj_page_offset = offset as u32;
                    }
                }
            });

        RegionPageMetadata {
            region_start,
            source_end: cursor,
            compacted_end,
            has_movement,
            pages,
        }
    }

    pub fn forward(&self, object: ObjectReference, _vo_bit_valid: bool) -> ObjectReference {
        if !self.in_space(object) {
            return object;
        }
        #[cfg(feature = "uffd")]
        {
            let limit = self.compaction_region_limit.load(Ordering::Acquire);
            if limit > 0 && !self.is_in_compaction_region_prefix(object, limit) {
                return object;
            }
        }
        // We can't expect the VO bit to be valid whilst compacting the heap.
        // If we are fixing a reference to an object which was moved before the referent,
        // the relevant VO bit will have been cleared, and this assertion would fail.
        // Thus we can only ever expect the VO bit to be valid whilst fixing the roots.
        #[cfg(feature = "vo_bit")]
        if _vo_bit_valid {
            debug_assert!(
                crate::util::metadata::vo_bit::is_vo_bit_set(object),
                "{:x}: VO bit not set",
                object
            );
        }
        ObjectReference::from_raw_address(self.forwarding.forward(object.to_raw_address())).unwrap()
    }

    fn update_references(&self, worker: &mut GCWorker<VM>, object: ObjectReference) {
        if VM::VMScanning::support_slot_enqueuing(worker.tls, object) {
            VM::VMScanning::scan_object(worker.tls, object, &mut |s: VM::VMSlot| {
                if let Some(o) = s.load() {
                    s.store(self.forward(o, false));
                }
            });
        } else {
            VM::VMScanning::scan_object_and_trace_edges(worker.tls, object, &mut |o| {
                self.forward(o, false)
            });
        }
    }

    pub fn add_compact_tasks(&'static self) {
        let compact_packets: Vec<Box<dyn GCWork<VM>>> =
            self.generate_tasks(&mut |_, i| Box::new(Compact::<VM>::new(self, i)));
        self.scheduler.work_buckets[WorkBucketStage::Compact].bulk_add(compact_packets);
    }

    pub fn compact_region(&self, worker: &mut GCWorker<VM>, index: usize) {
        self.pr.with_regions(&mut |regions| {
            let r = &regions[index];
            let start = r.region.start();
            let end = r.cursor();
            #[cfg(feature = "vo_bit")]
            {
                #[cfg(debug_assertions)]
                self.forwarding
                    .scan_marked_objects(start, end, &mut |object: ObjectReference| {
                        debug_assert!(
                            crate::util::metadata::vo_bit::is_vo_bit_set(object),
                            "{:x}: VO bit not set",
                            object
                        );
                    });
                crate::util::metadata::vo_bit::bzero_vo_bit(start, end - start);
            }
            let mut to = start;
            self.forwarding
                .scan_marked_objects(start, end, &mut |obj: ObjectReference| {
                    // We set the end bits based on the sizes of objects when they are
                    // marked, and we compute the live data and thus the forwarding
                    // addresses based on those sizes. The forwarding addresses would be
                    // incorrect if the sizes of objects were to change.
                    let copied_size = VM::VMObjectModel::get_size_when_copied(obj);
                    debug_assert!(copied_size == VM::VMObjectModel::get_current_size(obj));
                    let new_object = self.forward(obj, false);
                    debug_assert!(
                        new_object.to_raw_address() >= to,
                        "whilst forwarding {obj}, the new address {0} should be after the end of the last object {to}",
                        new_object.to_raw_address()
                    );
                    // copy object
                    trace!(" copy from {} to {}", obj, new_object);
                    let end_of_new_object =
                        VM::VMObjectModel::copy_to(obj, new_object, Address::ZERO);
                    // update VO bit
                    #[cfg(feature = "vo_bit")]
                    vo_bit::set_vo_bit(new_object);
                    to = new_object.to_object_start::<VM>() + copied_size;
                    debug_assert_eq!(end_of_new_object, to);
                    self.update_references(worker, new_object);
                });
            self.pr.reset_cursor(r, to);
        });
    }

    /// Get the number of allocated regions.
    #[cfg(feature = "uffd")]
    pub fn num_regions(&self) -> usize {
        self.pr.with_regions(&mut |regions| regions.len())
    }

    /// Get the number of data pages reserved by the page resource, excluding
    /// estimated side-metadata overhead.
    #[cfg(feature = "uffd")]
    pub fn data_reserved_pages(&self) -> usize {
        self.pr.reserved_pages()
    }

    #[cfg(feature = "uffd")]
    pub fn update_references_and_reclaim_static_regions(
        &self,
        worker: &mut GCWorker<VM>,
    ) -> (usize, usize) {
        let metadata = self.page_metadata.read().unwrap().clone();
        let mut static_regions = 0usize;
        let mut reclaimed_pages = 0usize;
        for (index, meta) in metadata.iter().enumerate() {
            if meta.has_movement {
                continue;
            }
            static_regions += 1;
            self.forwarding
                .scan_marked_objects(meta.region_start, meta.source_end, &mut |obj| {
                    self.update_references(worker, obj);
                });
            self.pr.with_regions(&mut |regions| {
                if let Some(r) = regions.get(index) {
                    let old = r.cursor();
                    let new = meta
                        .compacted_end
                        .align_up(crate::util::constants::BYTES_IN_PAGE);
                    if old > new {
                        reclaimed_pages += (old - new) / crate::util::constants::BYTES_IN_PAGE;
                    }
                    self.pr.reset_cursor(r, meta.compacted_end);
                }
            });
        }
        (static_regions, reclaimed_pages)
    }

    #[cfg(feature = "uffd")]
    fn is_in_compaction_region_prefix(&self, object: ObjectReference, count: usize) -> bool {
        let addr = object.to_raw_address();
        self.pr.with_regions(&mut |regions| {
            regions
                .iter()
                .take(count)
                .any(|r| addr >= r.region.start() && addr < r.region.end())
        })
    }

    #[cfg(feature = "uffd")]
    pub fn set_compaction_region_limit(&self, count: usize) {
        self.compaction_region_limit.store(count, Ordering::Release);
    }

    #[cfg(feature = "uffd")]
    pub fn clear_compaction_region_limit(&self) {
        self.compaction_region_limit.store(0, Ordering::Release);
    }

    /// Get region start address and cursor for a given index.
    #[cfg(feature = "uffd")]
    pub fn region_info(&self, index: usize) -> (Address, Address) {
        self.pr.with_regions(&mut |regions| {
            let r = &regions[index];
            (r.region.start(), r.cursor())
        })
    }

    /// Begin recording post-InitialMark bump-allocation buffers for the current concurrent-marking epoch.
    #[cfg(feature = "uffd")]
    pub fn snapshot_black_allocation_cursors(&self) {
        self.black_allocation_tracking_active
            .store(true, Ordering::Release);
        self.black_allocation_buffers.lock().unwrap().clear();
    }

    /// Slow-path post-allocation bookkeeping is no longer needed for Compressor black allocations.
    ///
    /// We now catch up on all post-InitialMark allocations by walking the retired/current bump
    /// buffers at FinalMark, which works for both slow-path and JIT fast-path allocations.
    #[cfg(feature = "uffd")]
    pub fn record_black_allocation(&self, object: ObjectReference) {
        let _ = object;
    }

    #[cfg(feature = "uffd")]
    fn record_retired_bump_alloc_buffer(&self, start: Address, cursor: Address, limit: Address) {
        if !self.black_allocation_tracking_active.load(Ordering::Acquire) || cursor <= start {
            return;
        }
        self.black_allocation_buffers
            .lock()
            .unwrap()
            .push(BlackAllocationBuffer {
                start,
                used_end: cursor,
                limit,
            });
    }

    #[cfg(feature = "uffd")]
    fn capture_current_black_allocation_buffers(&self) {
        for mutator in VM::VMActivePlan::mutators() {
            let allocator = unsafe {
                mutator.allocator_impl_for_semantic::<BumpAllocator<VM>>(
                    AllocationSemantics::Default,
                )
            };
            if let Some((start, cursor, limit)) = allocator.current_buffer() {
                self.record_retired_bump_alloc_buffer(start, cursor, limit);
            }
        }
    }

    /// Clear the recorded black-allocation buffer snapshot.
    #[cfg(feature = "uffd")]
    pub fn clear_black_allocation_snapshot(&self) {
        self.black_allocation_tracking_active
            .store(false, Ordering::Release);
        self.black_allocation_buffers.lock().unwrap().clear();
    }

    /// Drain post-InitialMark bump-allocation buffers and reconstruct the objects allocated in them.
    ///
    /// Each retired/current bump buffer represents a contiguous prefix of fully initialized objects
    /// allocated during the concurrent-marking window. Walking these buffers at FinalMark catches up
    /// both slow-path and JIT fast-path allocations, mirroring ART's pause-time black-allocation
    /// update without requiring per-object allocation hooks.
    #[cfg(feature = "uffd")]
    pub fn take_black_allocations(&self) -> Vec<ObjectReference> {
        self.black_allocation_tracking_active
            .store(false, Ordering::Release);

        let mut black_buffers = self.black_allocation_buffers.lock().unwrap();
        let mut buffers = std::mem::take(&mut *black_buffers);
        drop(black_buffers);

        buffers.sort_by_key(|buffer| buffer.start.as_usize());

        let mut objects = vec![];
        let mut early_uninitialized = 0usize;
        let mut early_unsane = 0usize;
        let mut early_size = 0usize;
        for buffer in buffers {
            let mut cursor = buffer.start;
            while cursor < buffer.used_end {
                let object = ObjectReference::from_raw_address(cursor).unwrap_or_else(|| {
                    panic!(
                        "invalid black-allocation object start {} in retired bump buffer [{}, {}, {})",
                        cursor, buffer.start, buffer.used_end, buffer.limit
                    )
                });

                // A mutator may have advanced the bump cursor before fully initializing the next
                // object header. Mirror ART's pause-time catch-up behavior and stop at the first
                // object start the VM reports as not yet initialized.
                if !VM::VMObjectModel::is_object_start_initialized(object) {
                    early_uninitialized += 1;
                    if compressor_perf_trace_enabled() {
                        info!(
                            "Compressor FinalMark: stopped black-buffer walk at uninitialized object start {} in [{}, {}, {})",
                            object, buffer.start, buffer.used_end, buffer.limit
                        );
                    }
                    break;
                }
                if !VM::VMObjectModel::is_object_sane(object) {
                    early_unsane += 1;
                    if compressor_perf_trace_enabled() {
                        info!(
                            "Compressor FinalMark: stopped black-buffer walk at unsane object {} in [{}, {}, {})",
                            object, buffer.start, buffer.used_end, buffer.limit
                        );
                    }
                    break;
                }

                let size = VM::VMObjectModel::get_current_size(object);
                if size == 0 || cursor + size > buffer.used_end {
                    early_size += 1;
                    if compressor_perf_trace_enabled() {
                        info!(
                            "Compressor FinalMark: stopped black-buffer walk at size {} for object {} in [{}, {}, {})",
                            size, object, buffer.start, buffer.used_end, buffer.limit
                        );
                    }
                    break;
                }

                forwarding::MARK_SPEC.fetch_or_atomic::<u8>(
                    object.to_raw_address(),
                    1,
                    Ordering::SeqCst,
                );
                self.forwarding.mark_last_word_of_object(object);
                objects.push(object);
                cursor += size;
            }
        }

        if compressor_perf_trace_enabled() {
            info!(
                "Compressor FinalMark: caught up {} black objects from retired bump buffers (stops: uninitialized={}, unsane={}, size={})",
                objects.len(), early_uninitialized, early_unsane, early_size
            );
        }

        objects
    }

    #[cfg(feature = "uffd")]
    pub fn align_region_cursors_for_concurrent_uffd(&self) -> (usize, usize) {
        self.pr.align_all_region_cursors_up_to_page()
    }

    #[cfg(feature = "uffd")]
    pub fn seal_regions_for_concurrent_uffd(&self) {
        self.pr.seal_existing_regions_for_uffd();
    }

    /// Finalize a region cursor after a concurrent UFFD phase.
    ///
    /// During the UFFD epoch mutators may continue allocating.  To avoid racing with
    /// those allocations, this uses the page-resource lock to reset the cursor only if
    /// it is still exactly at the pre-compaction `source_end`.  If the cursor changed,
    /// mutators allocated in the region after resume and we retain the current cursor.
    #[cfg(feature = "uffd")]
    pub fn finalize_region_cursor_after_concurrent_uffd(&self, index: usize) -> bool {
        let Some((source_end, compacted_end)) =
            self.with_region_page_metadata(index, |meta| (meta.source_end, meta.compacted_end))
        else {
            return false;
        };
        self.pr
            .reset_cursor_if_unchanged(index, source_end, compacted_end)
    }

    pub fn reset_allocator_after_compaction(&self) {
        self.pr.reset_allocator();
    }

    /// Update references from the LOS into Compressor objects after forwarding is known.
    pub fn update_object_references(&self, worker: &mut GCWorker<VM>, object: ObjectReference) {
        self.update_references(worker, object);
    }

    pub fn update_los_references(&self, worker: &mut GCWorker<VM>, los: &LargeObjectSpace<VM>) {
        los.enumerate_to_space_objects(&mut object_enum::ClosureObjectEnumerator::<_, VM>::new(
            &mut |o: ObjectReference| {
                self.update_references(worker, o);
            },
        ));
    }

    pub fn update_space_references(&self, worker: &mut GCWorker<VM>, space: &dyn Space<VM>) {
        let mut enumerator =
            object_enum::ClosureObjectEnumerator::<_, VM>::new(|o: ObjectReference| {
                self.update_references(worker, o);
            });
        space.enumerate_objects(&mut enumerator);
    }

    pub fn after_compact(&self, worker: &mut GCWorker<VM>, los: &LargeObjectSpace<VM>) {
        self.reset_allocator_after_compaction();
        self.update_los_references(worker, los);
    }
}

/// Calculate the offset vector for a region.
pub struct CalculateOffsetVector<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
    region: forwarding::CompressorRegion,
    cursor: Address,
}

impl<VM: VMBinding> GCWork<VM> for CalculateOffsetVector<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        self.compressor_space
            .calculate_offset_vector_for_region(self.region, self.cursor);
    }
}

impl<VM: VMBinding> CalculateOffsetVector<VM> {
    pub fn new(
        compressor_space: &'static CompressorSpace<VM>,
        region: forwarding::CompressorRegion,
        cursor: Address,
    ) -> Self {
        Self {
            compressor_space,
            region,
            cursor,
        }
    }
}

/// Compact live objects in a region.
pub struct Compact<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
    index: usize,
}

impl<VM: VMBinding> GCWork<VM> for Compact<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        self.compressor_space.compact_region(worker, self.index);
    }
}

impl<VM: VMBinding> Compact<VM> {
    pub fn new(compressor_space: &'static CompressorSpace<VM>, index: usize) -> Self {
        Self {
            compressor_space,
            index,
        }
    }
}
