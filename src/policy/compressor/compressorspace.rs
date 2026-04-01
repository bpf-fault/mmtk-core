#[cfg(feature = "uffd")]
use crate::plan::AllocationSemantics;
use crate::plan::VectorObjectQueue;
use crate::policy::compressor::forwarding;
use crate::policy::gc_work::{TraceKind, TRACE_KIND_TRANSITIVE_PIN};
use crate::policy::largeobjectspace::LargeObjectSpace;
use crate::policy::sft::{GCWorkerMutRef, SFT};
use crate::policy::space::{CommonSpace, Space};
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
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};
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

#[cfg(feature = "uffd")]
fn validate_uffd_page_metadata_equiv_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("MMTK_VALIDATE_UFFD_PAGE_METADATA_EQUIV").is_some())
}

#[cfg(feature = "uffd")]
fn format_perf_ms(duration: std::time::Duration) -> String {
    format!("{:.3}", duration.as_secs_f64() * 1000.0)
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
    compaction_summaries: RwLock<Vec<Option<RegionCompactionSummary>>>,
    #[cfg(feature = "uffd")]
    compaction_prepare_snapshots: RwLock<Vec<Option<PreparedRegionCompactionSnapshot>>>,
    #[cfg(feature = "uffd")]
    prepared_region_compaction_data: RwLock<Vec<Option<PreparedRegionCompactionData>>>,
    #[cfg(feature = "uffd")]
    concurrent_mark_activity_epoch: AtomicU64,
    #[cfg(feature = "uffd")]
    compaction_prepare_mark_epoch: AtomicU64,
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageCompactMetadata {
    /// The first source object whose compacted destination overlaps this page.
    pub first_obj: Option<ObjectReference>,
    /// If the page starts in the middle of `first_obj`, this is the byte offset
    /// within that object where the page begins. Zero if the page begins at the
    /// start of `first_obj` or if the page has no live data.
    pub first_obj_page_offset: u32,
}

#[cfg(feature = "uffd")]
#[derive(Clone, Debug, PartialEq, Eq)]
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

#[cfg(feature = "uffd")]
#[derive(Clone, Debug)]
struct RegionCompactionObjectMetadata {
    obj: ObjectReference,
    copied_size: usize,
    dst_start: Address,
    dst_end: Address,
    first_page: usize,
    last_page: usize,
}

#[cfg(feature = "uffd")]
#[derive(Clone, Debug)]
struct RegionCompactionSummary {
    region_start: Address,
    source_end: Address,
    compacted_end: Address,
    compacted_pages: usize,
    has_movement: bool,
    objects: Vec<RegionCompactionObjectMetadata>,
}

#[cfg(feature = "uffd")]
#[derive(Clone, Debug)]
struct PreparedRegionCompactionSnapshot {
    region_start: Address,
    source_end: Address,
}

#[cfg(feature = "uffd")]
#[derive(Clone, Debug)]
struct PreparedRegionCompactionObjectMetadata {
    obj: ObjectReference,
    copied_size: usize,
}

#[cfg(feature = "uffd")]
#[derive(Clone, Debug)]
struct PreparedRegionCompactionData {
    region_start: Address,
    source_end: Address,
    objects: Vec<PreparedRegionCompactionObjectMetadata>,
}

#[cfg(feature = "uffd")]
struct ReferenceUpdateBatchState {
    label: &'static str,
    started: std::time::Instant,
    remaining_packets: AtomicUsize,
    total_objects: usize,
    total_packets: usize,
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
            compaction_summaries: RwLock::new(vec![]),
            #[cfg(feature = "uffd")]
            compaction_prepare_snapshots: RwLock::new(vec![]),
            #[cfg(feature = "uffd")]
            prepared_region_compaction_data: RwLock::new(vec![]),
            #[cfg(feature = "uffd")]
            concurrent_mark_activity_epoch: AtomicU64::new(0),
            #[cfg(feature = "uffd")]
            compaction_prepare_mark_epoch: AtomicU64::new(u64::MAX),
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

    #[cfg(feature = "uffd")]
    pub fn add_offset_vector_tasks_for_final_mark(&'static self) {
        let total_regions = self.compaction_region_count();
        let prepared_prefix = if self.can_use_prepared_region_compaction_data() {
            self.compaction_prepare_snapshots
                .read()
                .unwrap()
                .len()
                .min(total_regions)
        } else {
            0
        };

        self.forwarding.mark_calculated();

        if compressor_perf_trace_enabled() {
            info!(
                "Compressor FinalMark: offset-vector reuse prepared_prefix={}, total_regions={}",
                prepared_prefix, total_regions
            );
        }

        if prepared_prefix >= total_regions {
            return;
        }

        let packets: Vec<Box<dyn GCWork<VM>>> = self.pr.with_regions(&mut |regions| {
            regions
                .iter()
                .skip(prepared_prefix)
                .take(total_regions - prepared_prefix)
                .map(|r| {
                    Box::new(CalculateOffsetVector::<VM>::new(self, r.region, r.cursor()))
                        as Box<dyn GCWork<VM>>
                })
                .collect()
        });
        self.scheduler.work_buckets[WorkBucketStage::CalculateForwarding].bulk_add(packets);
    }

    pub fn calculate_offset_vector_for_region(
        &self,
        region: forwarding::CompressorRegion,
        cursor: Address,
    ) {
        self.forwarding.calculate_offset_vector(region, cursor);
    }

    #[cfg(feature = "uffd")]
    fn calculate_offset_vector_for_prepare_region(
        &self,
        region: forwarding::CompressorRegion,
        cursor: Address,
    ) {
        self.forwarding
            .calculate_offset_vector_for_prepare(region, cursor);
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
        self.build_page_metadata_for_region_prefix(self.compaction_region_count());
    }

    #[cfg(feature = "uffd")]
    pub fn build_page_metadata_for_region_prefix(&self, count: usize) {
        let summarize_start = std::time::Instant::now();
        self.cache_region_compaction_summaries_for_prefix(count);
        let summarize_elapsed = summarize_start.elapsed();

        let materialize_start = std::time::Instant::now();
        self.materialize_page_metadata_from_cached_summaries_for_prefix(count);
        let materialize_elapsed = materialize_start.elapsed();

        if compressor_perf_trace_enabled() {
            info!(
                "Compressor page metadata breakdown: summarize={} ms, materialize={} ms",
                format_perf_ms(summarize_elapsed),
                format_perf_ms(materialize_elapsed)
            );
        }
    }

    #[cfg(feature = "uffd")]
    pub fn has_cached_region_compaction_summaries_for_prefix(&self, count: usize) -> bool {
        let summaries = self.compaction_summaries.read().unwrap();
        summaries.len() >= count && summaries.iter().take(count).all(|s| s.is_some())
    }

    #[cfg(feature = "uffd")]
    pub fn reset_concurrent_mark_activity_epoch(&self) {
        self.concurrent_mark_activity_epoch
            .store(0, Ordering::Release);
        self.compaction_prepare_mark_epoch
            .store(u64::MAX, Ordering::Release);
    }

    #[cfg(feature = "uffd")]
    pub fn note_concurrent_mark_activity(&self) {
        self.concurrent_mark_activity_epoch
            .fetch_add(1, Ordering::AcqRel);
    }

    #[cfg(feature = "uffd")]
    pub fn snapshot_concurrent_compaction_prepare_regions(&self) {
        let snapshots = self.pr.with_regions(&mut |regions| {
            regions
                .iter()
                .map(|region| {
                    Some(PreparedRegionCompactionSnapshot {
                        region_start: region.region.start(),
                        source_end: region.cursor(),
                    })
                })
                .collect::<Vec<_>>()
        });
        let len = snapshots.len();
        *self.compaction_prepare_snapshots.write().unwrap() = snapshots;
        let mut prepared = self.prepared_region_compaction_data.write().unwrap();
        prepared.clear();
        prepared.resize(len, None);
        self.set_compaction_region_limit(len);
        self.compaction_prepare_mark_epoch
            .store(u64::MAX, Ordering::Release);
    }

    #[cfg(feature = "uffd")]
    pub fn start_concurrent_compaction_prepare(
        &'static self,
        on_complete: Arc<dyn Fn() + Send + Sync>,
    ) -> usize {
        let prepare_epoch = self.concurrent_mark_activity_epoch.load(Ordering::Acquire);
        self.compaction_prepare_mark_epoch
            .store(prepare_epoch, Ordering::Release);

        let snapshots = self.compaction_prepare_snapshots.read().unwrap().clone();
        if snapshots.is_empty() {
            return 0;
        }

        let packet_count = snapshots
            .iter()
            .filter(|snapshot| snapshot.is_some())
            .count();
        if packet_count == 0 {
            return 0;
        }

        let batch = Arc::new(CompactionPrepareBatchState::new(on_complete, packet_count));
        let packets = snapshots
            .into_iter()
            .enumerate()
            .filter_map(|(index, snapshot)| {
                let snapshot = snapshot?;
                Some(Box::new(PrepareRegionCompactionData::<VM>::new(
                    self,
                    index,
                    snapshot,
                    batch.clone(),
                )) as Box<dyn GCWork<VM>>)
            })
            .collect::<Vec<_>>();
        self.scheduler.work_buckets[WorkBucketStage::Concurrent].bulk_add(packets);
        packet_count
    }

    #[cfg(feature = "uffd")]
    pub fn add_region_compaction_summary_tasks(&'static self) {
        let count = self.compaction_region_count();
        self.prepare_region_compaction_summary_cache(count);
        let packets: Vec<Box<dyn GCWork<VM>>> = self.pr.with_regions(&mut |regions| {
            regions
                .iter()
                .enumerate()
                .take(count)
                .map(|(index, region)| {
                    Box::new(CacheRegionCompactionSummary::<VM>::new(
                        self,
                        index,
                        region.region.start(),
                        region.cursor(),
                    )) as Box<dyn GCWork<VM>>
                })
                .collect()
        });
        self.scheduler.work_buckets[WorkBucketStage::Compact].bulk_add(packets);
    }

    #[cfg(feature = "uffd")]
    fn cache_region_compaction_summaries_for_prefix(&self, count: usize) {
        self.prepare_region_compaction_summary_cache(count);
        let summaries = self.pr.with_regions(&mut |regions| {
            regions
                .iter()
                .enumerate()
                .take(count)
                .map(|(index, r)| {
                    self.summarize_region_compaction_for_current_region(
                        index,
                        r.region.start(),
                        r.cursor(),
                    )
                })
                .collect::<Vec<_>>()
        });
        let mut cache = self.compaction_summaries.write().unwrap();
        for (index, summary) in summaries.into_iter().enumerate() {
            cache[index] = Some(summary);
        }
    }

    #[cfg(feature = "uffd")]
    fn prepare_region_compaction_summary_cache(&self, count: usize) {
        let mut cache = self.compaction_summaries.write().unwrap();
        cache.clear();
        cache.resize(count, None);
    }

    /// Invalidate all cached compaction summaries and prepared-region data.
    ///
    /// This **must** be called after any operation that changes the mark set
    /// (e.g. inter-pause black-allocation catch-up at the Compaction pause)
    /// so that both offset-vector computation and `build_all_page_metadata()`
    /// recompute from the current mark state instead of reusing stale cached data.
    #[cfg(feature = "uffd")]
    pub fn invalidate_compaction_summary_cache(&self) {
        // Clear the page-metadata / compaction-summary cache.
        self.compaction_summaries.write().unwrap().clear();
        // Also invalidate the prepared-region compaction data (offset vectors)
        // by resetting the mark epoch so can_use_prepared_region_compaction_data()
        // returns false, forcing recomputation in add_offset_vector_tasks_for_final_mark().
        self.compaction_prepare_mark_epoch
            .store(u64::MAX, Ordering::Release);
        self.compaction_prepare_snapshots.write().unwrap().clear();
    }

    #[cfg(feature = "uffd")]
    fn ensure_region_compaction_summary_cache_len(&self, count: usize) {
        let mut cache = self.compaction_summaries.write().unwrap();
        if cache.len() < count {
            cache.resize(count, None);
        }
    }

    #[cfg(feature = "uffd")]
    fn cache_region_compaction_summary_at_index(
        &self,
        index: usize,
        summary: RegionCompactionSummary,
    ) {
        let mut cache = self.compaction_summaries.write().unwrap();
        if index >= cache.len() {
            panic!(
                "region compaction summary index {} out of bounds for cache len {}",
                index,
                cache.len()
            );
        }
        cache[index] = Some(summary);
    }

    #[cfg(feature = "uffd")]
    fn cache_prepared_region_compaction_data_at_index(
        &self,
        index: usize,
        prepared: PreparedRegionCompactionData,
    ) {
        let mut cache = self.prepared_region_compaction_data.write().unwrap();
        if index >= cache.len() {
            cache.resize(index + 1, None);
        }
        cache[index] = Some(prepared);
    }

    #[cfg(feature = "uffd")]
    fn can_use_prepared_region_compaction_data(&self) -> bool {
        let prepared_epoch = self.compaction_prepare_mark_epoch.load(Ordering::Acquire);
        prepared_epoch != u64::MAX
            && prepared_epoch == self.concurrent_mark_activity_epoch.load(Ordering::Acquire)
    }

    #[cfg(feature = "uffd")]
    fn prepared_region_compaction_data_for(
        &self,
        index: usize,
        region_start: Address,
        cursor: Address,
    ) -> Option<PreparedRegionCompactionData> {
        if !self.can_use_prepared_region_compaction_data() {
            return None;
        }
        let cache = self.prepared_region_compaction_data.read().unwrap();
        let prepared = cache.get(index).and_then(|entry| entry.clone())?;
        if prepared.region_start == region_start && prepared.source_end == cursor {
            Some(prepared)
        } else {
            None
        }
    }

    #[cfg(feature = "uffd")]
    fn summarize_region_compaction_for_current_region(
        &self,
        index: usize,
        region_start: Address,
        cursor: Address,
    ) -> RegionCompactionSummary {
        if let Some(prepared) =
            self.prepared_region_compaction_data_for(index, region_start, cursor)
        {
            self.summarize_region_compaction_from_prepared(&prepared)
        } else {
            self.summarize_region_compaction(region_start, cursor)
        }
    }

    #[cfg(feature = "uffd")]
    fn current_region_layout_for_prefix(&self, count: usize) -> Vec<(Address, Address)> {
        self.pr.with_regions(&mut |regions| {
            regions
                .iter()
                .take(count)
                .map(|region| (region.region.start(), region.cursor()))
                .collect::<Vec<_>>()
        })
    }

    #[cfg(feature = "uffd")]
    fn try_build_page_metadata_from_current_cached_summaries_for_prefix(
        &self,
        count: usize,
    ) -> Option<Vec<RegionPageMetadata>> {
        let regions = self.current_region_layout_for_prefix(count);
        let cache = self.compaction_summaries.read().unwrap();
        if cache.len() < count {
            return None;
        }

        let mut metadata = Vec::with_capacity(count);
        for (index, (region_start, cursor)) in regions.into_iter().enumerate() {
            let summary = cache.get(index).and_then(|summary| summary.as_ref())?;
            if summary.region_start != region_start || summary.source_end != cursor {
                return None;
            }
            metadata.push(self.build_region_page_metadata_from_summary(summary));
        }
        Some(metadata)
    }

    #[cfg(feature = "uffd")]
    pub fn validate_region_compaction_layout_for_prefix(
        &self,
        count: usize,
    ) -> Result<(), String> {
        let summaries = self.current_or_recompute_region_compaction_summaries_for_prefix(count);
        for (index, summary) in summaries.iter().enumerate() {
            let region_end = summary.region_start
                + crate::policy::compressor::forwarding::CompressorRegion::BYTES;
            let mut spans = summary
                .objects
                .iter()
                .map(|obj| (obj.obj, obj.dst_start, obj.dst_end))
                .collect::<Vec<_>>();
            spans.sort_by_key(|(_, dst_start, _)| *dst_start);

            for (obj, dst_start, dst_end) in &spans {
                if *dst_start < summary.region_start || *dst_end > region_end || *dst_start >= *dst_end {
                    return Err(format!(
                        "Compaction layout validation failed for region {}: object {} has invalid destination span [{}, {}) within region [{}, {})",
                        index,
                        obj,
                        dst_start,
                        dst_end,
                        summary.region_start,
                        region_end,
                    ));
                }
            }

            for window in spans.windows(2) {
                let (left_obj, left_start, left_end) = window[0];
                let (right_obj, right_start, right_end) = window[1];
                if left_end > right_start {
                    return Err(format!(
                        "Compaction layout validation failed for region {}: overlapping destinations {}:[{}, {}) and {}:[{}, {})",
                        index,
                        left_obj,
                        left_start,
                        left_end,
                        right_obj,
                        right_start,
                        right_end,
                    ));
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "uffd")]
    fn current_or_recompute_region_compaction_summaries_for_prefix(
        &self,
        count: usize,
    ) -> Vec<RegionCompactionSummary> {
        let regions = self.current_region_layout_for_prefix(count);

        self.ensure_region_compaction_summary_cache_len(count);
        let mut summaries = Vec::with_capacity(count);
        let mut reused = 0usize;
        let mut recomputed = 0usize;
        for (index, (region_start, cursor)) in regions.into_iter().enumerate() {
            let cached = {
                let cache = self.compaction_summaries.read().unwrap();
                cache.get(index).and_then(|summary| summary.clone())
            };
            match cached {
                Some(summary)
                    if summary.region_start == region_start && summary.source_end == cursor =>
                {
                    reused += 1;
                    summaries.push(summary);
                }
                _ => {
                    recomputed += 1;
                    let summary = self.summarize_region_compaction_for_current_region(
                        index,
                        region_start,
                        cursor,
                    );
                    self.cache_region_compaction_summary_at_index(index, summary.clone());
                    summaries.push(summary);
                }
            }
        }
        if compressor_perf_trace_enabled() {
            info!(
                "Compressor region compaction summary materialization reuse: reused={}, recomputed={}",
                reused,
                recomputed
            );
        }
        summaries
    }

    #[cfg(feature = "uffd")]
    pub fn materialize_page_metadata_from_cached_summaries_for_prefix(&self, count: usize) {
        let materialize_start = std::time::Instant::now();
        let metadata = if let Some(metadata) =
            self.try_build_page_metadata_from_current_cached_summaries_for_prefix(count)
        {
            if compressor_perf_trace_enabled() {
                info!(
                    "Compressor region compaction summary materialization reuse: reused={}, recomputed=0",
                    count
                );
            }
            metadata
        } else {
            let summaries = self.current_or_recompute_region_compaction_summaries_for_prefix(count);
            self.build_page_metadata_from_summaries(&summaries)
        };
        let materialize_elapsed = materialize_start.elapsed();
        let validate_elapsed = if validate_uffd_page_metadata_equiv_enabled() {
            let validate_start = std::time::Instant::now();
            self.validate_page_metadata_from_summaries(count.min(metadata.len()), &metadata);
            validate_start.elapsed()
        } else {
            std::time::Duration::ZERO
        };
        let total_pages: usize = metadata.iter().map(|m| m.pages.len()).sum();
        if compressor_perf_trace_enabled() {
            info!(
                "Compressor cached page metadata breakdown: materialize={} ms, validate={} ms",
                format_perf_ms(materialize_elapsed),
                format_perf_ms(validate_elapsed)
            );
            info!(
                "Compressor page metadata built: {} regions, {} destination pages",
                metadata.len(),
                total_pages
            );
        }
        *self.page_metadata.write().unwrap() = metadata;
    }

    #[cfg(feature = "uffd")]
    fn build_page_metadata_from_summaries(
        &self,
        summaries: &[RegionCompactionSummary],
    ) -> Vec<RegionPageMetadata> {
        summaries
            .iter()
            .map(|summary| self.build_region_page_metadata_from_summary(summary))
            .collect()
    }

    #[cfg(feature = "uffd")]
    fn validate_page_metadata_from_summaries(&self, count: usize, metadata: &[RegionPageMetadata]) {
        let mismatch = self.pr.with_regions(&mut |regions| {
            regions
                .iter()
                .take(count)
                .enumerate()
                .find_map(|(index, region)| {
                    let legacy = self
                        .build_region_page_metadata_legacy(region.region.start(), region.cursor());
                    let actual = &metadata[index];
                    if actual != &legacy {
                        Some((index, actual.clone(), legacy))
                    } else {
                        None
                    }
                })
        });

        if let Some((index, actual, legacy)) = mismatch {
            panic!(
                "MMTK_VALIDATE_UFFD_PAGE_METADATA_EQUIV: region {} mismatch\nsummary-built={:#?}\nlegacy={:#?}",
                index, actual, legacy
            );
        }

        if compressor_perf_trace_enabled() {
            info!(
                "MMTK_VALIDATE_UFFD_PAGE_METADATA_EQUIV: validated {} regions",
                count
            );
        }
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
    fn validate_rewritten_temp_object(
        &self,
        original_obj: ObjectReference,
        shadow_obj: ObjectReference,
        temp_obj: ObjectReference,
    ) -> Result<(), String> {
        let mut err = None;
        VM::VMScanning::scan_object(
            crate::util::opaque_pointer::VMWorkerThread(
                crate::util::opaque_pointer::VMThread::UNINITIALIZED,
            ),
            temp_obj,
            &mut |slot: VM::VMSlot| {
                if err.is_some() {
                    return;
                }
                let Some(offset) = VM::VMScanning::slot_offset(temp_obj, slot) else {
                    return;
                };
                let slot_desc = VM::VMScanning::describe_slot(temp_obj, slot)
                    .unwrap_or_else(|| format!("slot_offset={offset}"));
                let actual = slot.load();
                let shadow_value = VM::VMScanning::debug_load_slot_at_offset(shadow_obj, offset)
                    .unwrap_or(None);
                let preserve_weak_referent = slot_desc.contains("field=referent")
                    && !slot_desc.contains("reference_type=Final");
                let expected = if preserve_weak_referent {
                    shadow_value
                } else {
                    shadow_value.map(|shadow_ref| {
                        if self.in_space(shadow_ref) && Self::is_marked(shadow_ref) {
                            self.forward(shadow_ref, false)
                        } else {
                            shadow_ref
                        }
                    })
                };
                if actual != expected {
                    err = Some(format!(
                        "Uffd temp rewrite validation failed for object {}: temp={} slot={} actual={:?} shadow={:?} expected={:?}",
                        original_obj,
                        temp_obj,
                        slot_desc,
                        actual,
                        shadow_value,
                        expected,
                    ));
                }
            },
        );
        err.map_or(Ok(()), Err)
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
        VM::VMObjectModel::fixup_copied_object(shadow_obj, temp_obj);
        VM::VMScanning::scan_object_for_slot_rewrite(
            crate::util::opaque_pointer::VMWorkerThread(
                crate::util::opaque_pointer::VMThread::UNINITIALIZED,
            ),
            temp_obj,
            &mut |slot: VM::VMSlot| {
                if let Some(o) = slot.load() {
                    let new_ref = if self.in_space(o) && Self::is_marked(o) {
                        if std::env::var_os("MMTK_TRACE_COMPRESSOR_SLOT_REWRITE_AMBIGUOUS").is_some()
                            || std::env::var_os("MMTK_ABORT_ON_COMPRESSOR_SLOT_REWRITE_AMBIGUOUS").is_some()
                        {
                            let (old_match, dst_match) = self.debug_compaction_address_role(o);
                            let ambiguous_collision = matches!(
                                (&old_match, &dst_match),
                                (Some((old_obj, _)), Some((dst_obj, _))) if old_obj != dst_obj
                            );
                            if ambiguous_collision {
                                static TRACE_BUDGET: std::sync::atomic::AtomicI32 =
                                    std::sync::atomic::AtomicI32::new(64);
                                let slot_desc = VM::VMScanning::describe_slot(temp_obj, slot)
                                    .unwrap_or_else(|| "slot=<unknown>".to_string());
                                let msg = format!(
                                    "Compressor slot rewrite ambiguous forwarding candidate: owner={} shadow_obj={} slot={} value={} role={} old_match={:?} dst_match={:?}",
                                    obj,
                                    shadow_obj,
                                    slot_desc,
                                    o,
                                    self.debug_describe_compaction_object(o),
                                    old_match,
                                    dst_match,
                                );
                                if std::env::var_os("MMTK_TRACE_COMPRESSOR_SLOT_REWRITE_AMBIGUOUS").is_some()
                                    && TRACE_BUDGET.fetch_sub(1, Ordering::Relaxed) > 0
                                {
                                    log::info!("{}", msg);
                                }
                                if std::env::var_os("MMTK_ABORT_ON_COMPRESSOR_SLOT_REWRITE_AMBIGUOUS").is_some() {
                                    panic!("{}", msg);
                                }
                            }
                        }
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
        if std::env::var_os("MMTK_VALIDATE_UFFD_TEMP_REWRITE").is_some() {
            self.validate_rewritten_temp_object(obj, shadow_obj, temp_obj)
                .unwrap_or_else(|e| panic!("{}", e));
        }
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

    #[cfg(feature = "uffd")]
    pub fn validate_region_buffer_objects(
        &self,
        region_index: usize,
        region_buf: &[u8],
    ) -> Result<(), String> {
        let Some((region_start, source_end)) = self
            .with_region_page_metadata(region_index, |meta| (meta.region_start, meta.source_end))
        else {
            return Ok(());
        };

        let buf_addr = Address::from_ptr(region_buf.as_ptr());
        self.forwarding.scan_marked_objects(
            region_start,
            source_end,
            &mut |obj: ObjectReference| {
                let copied_size = VM::VMObjectModel::get_size_when_copied(obj);
                let new_obj = self.forward(obj, false);
                let dst_off = new_obj.to_raw_address() - region_start;
                let temp_obj_addr = buf_addr + dst_off;
                let temp_obj = unsafe { ObjectReference::from_raw_address_unchecked(temp_obj_addr) };
                if !VM::VMObjectModel::is_object_sane(temp_obj) {
                    panic!(
                        "Uffd region object validation failed for region {}: copied object {} -> {} is not sane in region buffer",
                        region_index,
                        obj,
                        new_obj,
                    );
                }
                let actual_size = VM::VMObjectModel::get_current_size(temp_obj);
                if actual_size != copied_size {
                    panic!(
                        "Uffd region object validation failed for region {}: copied object {} -> {} has size {} in region buffer, expected {}",
                        region_index,
                        obj,
                        new_obj,
                        actual_size,
                        copied_size,
                    );
                }
            },
        );

        Ok(())
    }

    #[cfg(feature = "uffd")]
    pub fn debug_count_reference_fields_in_prefix(&self) -> (usize, usize, usize, usize) {
        let limit = self.compaction_region_count();
        if limit == 0 {
            return (0, 0, 0, 0);
        }

        let mut reference_objects = 0usize;
        let mut referent_non_null = 0usize;
        let mut discovered_non_null = 0usize;
        let mut discovered_total = 0usize;
        self.pr.with_regions(&mut |regions| {
            for region in regions.iter().take(limit) {
                self.forwarding.scan_marked_objects(
                    region.region.start(),
                    region.cursor(),
                    &mut |obj: ObjectReference| {
                        let mut object_is_reference = false;
                        VM::VMScanning::scan_object(
                            crate::util::opaque_pointer::VMWorkerThread(
                                crate::util::opaque_pointer::VMThread::UNINITIALIZED,
                            ),
                            obj,
                            &mut |slot: VM::VMSlot| {
                                let Some(desc) = VM::VMScanning::describe_slot(obj, slot) else {
                                    return;
                                };
                                if desc.contains("field=referent") {
                                    object_is_reference = true;
                                    if slot.load().is_some() {
                                        referent_non_null += 1;
                                    }
                                }
                                if desc.contains("field=discovered") {
                                    object_is_reference = true;
                                    discovered_total += 1;
                                    if slot.load().is_some() {
                                        discovered_non_null += 1;
                                    }
                                }
                            },
                        );
                        if object_is_reference {
                            reference_objects += 1;
                        }
                    },
                );
            }
        });
        (
            reference_objects,
            referent_non_null,
            discovered_total,
            discovered_non_null,
        )
    }

    #[cfg(feature = "uffd")]
    pub fn validate_source_prefix_references(&self) -> Result<(), String> {
        let limit = self.compaction_region_count();
        if limit == 0 {
            return Ok(());
        }

        let mut err = None;
        self.pr.with_regions(&mut |regions| {
            for (region_index, region) in regions.iter().take(limit).enumerate() {
                if err.is_some() {
                    break;
                }
                self.forwarding.scan_marked_objects(
                    region.region.start(),
                    region.cursor(),
                    &mut |obj: ObjectReference| {
                        if err.is_some() {
                            return;
                        }
                        VM::VMScanning::scan_object(
                            crate::util::opaque_pointer::VMWorkerThread(
                                crate::util::opaque_pointer::VMThread::UNINITIALIZED,
                            ),
                            obj,
                            &mut |slot: VM::VMSlot| {
                                if err.is_some() {
                                    return;
                                }
                                let Some(referent) = slot.load() else {
                                    return;
                                };
                                if self.in_space(referent)
                                    && self.is_in_compaction_region_prefix(referent, limit)
                                    && !Self::is_marked(referent)
                                {
                                    let slot_desc = VM::VMScanning::describe_slot(obj, slot)
                                        .unwrap_or_else(|| "slot=<unknown>".to_string());
                                    err = Some(format!(
                                        "Compaction source reference validation failed for region {}: object {} contains unmarked prefix ref {} in {}",
                                        region_index,
                                        obj,
                                        referent,
                                        slot_desc,
                                    ));
                                }
                            },
                        );
                    },
                );
            }
        });

        if let Some(err) = err {
            return Err(err);
        }

        Ok(())
    }

    #[cfg(feature = "uffd")]
    pub fn validate_region_buffer_mark_words(
        &self,
        region_index: usize,
        shadow_region_start: Address,
        region_buf: &[u8],
    ) -> Result<(), String> {
        let Some((region_start, source_end)) = self
            .with_region_page_metadata(region_index, |meta| (meta.region_start, meta.source_end))
        else {
            return Ok(());
        };

        let summary = self.summarize_region_compaction_for_current_region(
            region_index,
            region_start,
            source_end,
        );
        let buf_addr = Address::from_ptr(region_buf.as_ptr());
        for obj_meta in &summary.objects {
            let temp_obj_addr = buf_addr + (obj_meta.dst_start - region_start);
            let temp_obj = unsafe { ObjectReference::from_raw_address_unchecked(temp_obj_addr) };
            let Some(mark) = VM::VMObjectModel::debug_mark_word(temp_obj) else {
                continue;
            };
            let shadow_obj_addr = shadow_region_start + (obj_meta.obj.to_raw_address() - region_start);
            let shadow_obj = unsafe { ObjectReference::from_raw_address_unchecked(shadow_obj_addr) };
            let shadow_mark = VM::VMObjectModel::debug_mark_word(shadow_obj);
            let lock_bits = mark & 0x3;
            if lock_bits == 2 {
                let monitor_addr = mark ^ 0x2;
                let shadow_obj_usize = shadow_obj_addr.as_usize();
                let monitor_in_shadow_region = monitor_addr >= shadow_region_start.as_usize()
                    && monitor_addr < shadow_region_start.as_usize() + self.region_page_metadata(region_index).map(|m| (m.source_end - m.region_start)).unwrap_or(0);
                let monitor_matches_shadow_object = monitor_addr == shadow_obj_usize;
                if monitor_addr == 0 || monitor_in_shadow_region {
                    return Err(format!(
                        "Uffd region-buffer mark-word validation failed for region {}: object {} -> {} has suspicious monitor mark word 0x{:x} (shadow_mark={:?}, shadow_obj_addr=0x{:x}, monitor_addr=0x{:x}, monitor_in_shadow_region={}, monitor_matches_shadow_object={})",
                        region_index,
                        obj_meta.obj,
                        unsafe { ObjectReference::from_raw_address_unchecked(obj_meta.dst_start) },
                        mark,
                        shadow_mark,
                        shadow_obj_usize,
                        monitor_addr,
                        monitor_in_shadow_region,
                        monitor_matches_shadow_object,
                    ));
                }
            } else if lock_bits == 3 {
                return Err(format!(
                    "Uffd region-buffer mark-word validation failed for region {}: object {} -> {} retains marked/unused header 0x{:x} (shadow_mark={:?})",
                    region_index,
                    obj_meta.obj,
                    unsafe { ObjectReference::from_raw_address_unchecked(obj_meta.dst_start) },
                    mark,
                    shadow_mark,
                ));
            }
        }

        Ok(())
    }

    #[cfg(feature = "uffd")]
    pub fn validate_region_buffer_references(
        &self,
        region_index: usize,
        shadow_region_start: Address,
        region_buf: &[u8],
    ) -> Result<(), String> {
        let valid_objects = self.collect_destination_objects();
        let Some((region_start, source_end)) = self
            .with_region_page_metadata(region_index, |meta| (meta.region_start, meta.source_end))
        else {
            return Ok(());
        };

        let buf_addr = Address::from_ptr(region_buf.as_ptr());
        let region_summary = self.summarize_region_compaction_for_current_region(
            region_index,
            region_start,
            source_end,
        );
        let mut err = None;
        self.forwarding.scan_marked_objects(
            region_start,
            source_end,
            &mut |obj: ObjectReference| {
                if err.is_some() {
                    return;
                }
                let new_obj = self.forward(obj, false);
                let dst_off = new_obj.to_raw_address() - region_start;
                let temp_obj_addr = buf_addr + dst_off;
                let temp_obj = unsafe { ObjectReference::from_raw_address_unchecked(temp_obj_addr) };
                VM::VMScanning::scan_object(
                    crate::util::opaque_pointer::VMWorkerThread(
                        crate::util::opaque_pointer::VMThread::UNINITIALIZED,
                    ),
                    temp_obj,
                    &mut |slot: VM::VMSlot| {
                        if err.is_some() {
                            return;
                        }
                        let Some(referent) = slot.load() else {
                            return;
                        };
                        if self.in_space(referent) && !valid_objects.contains(&referent) {
                            let referent_marked = Self::is_marked(referent);
                            let referent_mapped = referent.to_raw_address().is_mapped();
                            let referent_initialized = referent_mapped
                                && VM::VMObjectModel::is_object_start_initialized(referent);
                            let referent_in_prefix = self.is_in_compaction_region_prefix(
                                referent,
                                self.compaction_region_count(),
                            );
                            let referent_forwarded = if referent_marked {
                                Some(self.forward(referent, false))
                            } else {
                                None
                            };
                            let slot_desc = VM::VMScanning::describe_slot(temp_obj, slot)
                                .unwrap_or_else(|| "slot=<unknown>".to_string());
                            let shadow_obj_addr = shadow_region_start + (obj.to_raw_address() - region_start);
                            let shadow_obj = unsafe {
                                ObjectReference::from_raw_address_unchecked(shadow_obj_addr)
                            };
                            let slot_offset = VM::VMScanning::slot_offset(temp_obj, slot);
                            let shadow_slot_value = slot_offset.and_then(|offset| {
                                VM::VMScanning::debug_load_slot_at_offset(shadow_obj, offset)
                                    .map(|value| (offset, value))
                            });
                            let expected_from_shadow = shadow_slot_value.map(|(offset, value)| {
                                let expected = value.map(|shadow_ref| {
                                    if self.in_space(shadow_ref) && Self::is_marked(shadow_ref) {
                                        self.forward(shadow_ref, false)
                                    } else {
                                        shadow_ref
                                    }
                                });
                                (offset, value, expected)
                            });
                            let overlap_owner = slot_offset.and_then(|offset| {
                                let slot_addr = new_obj.to_raw_address() + offset;
                                region_summary
                                    .objects
                                    .iter()
                                    .find(|other| {
                                        other.obj != obj
                                            && other.dst_start <= slot_addr
                                            && slot_addr < other.dst_end
                                    })
                                    .map(|other| {
                                        (
                                            other.obj,
                                            other.dst_start,
                                            other.dst_end,
                                            self.forward(other.obj, false),
                                        )
                                    })
                            });
                            err = Some(format!(
                                "Uffd region reference validation failed for region {}: object {} -> {} contains stale Compressor ref {} in {} (mapped={}, initialized={}, marked={}, in_prefix={}, forwarded={:?}, shadow_slot={:?}, expected_from_shadow={:?}, overlap_owner={:?})",
                                region_index,
                                obj,
                                new_obj,
                                referent,
                                slot_desc,
                                referent_mapped,
                                referent_initialized,
                                referent_marked,
                                referent_in_prefix,
                                referent_forwarded,
                                shadow_slot_value,
                                expected_from_shadow,
                                overlap_owner,
                            ));
                        }
                    },
                );
            },
        );

        if let Some(err) = err {
            return Err(err);
        }

        Ok(())
    }

    /// Validate the resolved compacted bytes in a region against a direct
    /// object-by-object copy from the shadow region.
    #[cfg(feature = "uffd")]
    pub fn validate_region_compaction_from_shadow(
        &self,
        region_index: usize,
        shadow_region_start: Address,
    ) -> Result<(), String> {
        const PAGE_SIZE: usize = crate::util::constants::BYTES_IN_PAGE;

        let Some(meta) = self.region_page_metadata(region_index) else {
            return Ok(());
        };
        let region_start = meta.region_start;
        let source_end = meta.source_end;
        let compacted_end = meta.compacted_end;
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
                    let page_idx = i / PAGE_SIZE;
                    let page_offset = i % PAGE_SIZE;
                    let page_meta = meta.pages.get(page_idx).cloned();
                    let page_start = region_start + page_idx * PAGE_SIZE;
                    let page_end = page_start + PAGE_SIZE;
                    let expected_page_end = ((page_idx + 1) * PAGE_SIZE).min(live_bytes);
                    let expected_page = &expected[page_idx * PAGE_SIZE..expected_page_end];
                    let mut built_page = vec![0u8; PAGE_SIZE];
                    let built_live = self.build_page_from_shadow(
                        region_index,
                        page_idx,
                        shadow_region_start,
                        &mut built_page,
                    );
                    let built_matches_expected =
                        built_page[..expected_page.len()] == *expected_page;
                    let mut ground_truth_first_obj = None;
                    self.forwarding.scan_marked_objects(
                        region_start,
                        source_end,
                        &mut |obj: ObjectReference| {
                            if ground_truth_first_obj.is_some() {
                                return;
                            }
                            let new_obj = self.forward(obj, false);
                            let dst_start = new_obj.to_raw_address();
                            let dst_end = dst_start + VM::VMObjectModel::get_size_when_copied(obj);
                            if dst_start < page_end && dst_end > page_start {
                                ground_truth_first_obj = Some(obj);
                            }
                        },
                    );
                    return Err(format!(
                        "region {} mismatch at byte {} (page {}, page_offset {}, expected 0x{:02x}, actual 0x{:02x}, page_meta={:?}, ground_truth_first_obj={:?}, built_live={}, page_builder_matches_expected={})",
                        region_index,
                        i,
                        page_idx,
                        page_offset,
                        expected[i],
                        actual[i],
                        page_meta,
                        ground_truth_first_obj,
                        built_live,
                        built_matches_expected,
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
        let summaries = self.compaction_summaries.read().unwrap().clone();
        let mut objects = HashSet::new();
        let compacted_regions = page_metadata.len();

        let can_use_summaries = summaries.len() >= compacted_regions
            && summaries
                .iter()
                .take(compacted_regions)
                .all(|summary| summary.is_some());
        if can_use_summaries {
            for summary in summaries.iter().take(compacted_regions).flatten() {
                for obj_meta in &summary.objects {
                    objects.insert(unsafe {
                        ObjectReference::from_raw_address_unchecked(obj_meta.dst_start)
                    });
                }
            }
        } else {
            for meta in &page_metadata {
                self.forwarding.scan_marked_objects(
                    meta.region_start,
                    meta.source_end,
                    &mut |obj: ObjectReference| {
                        objects.insert(self.forward(obj, false));
                    },
                );
            }
        }

        self.pr.with_regions(&mut |regions| {
            for region in regions.iter().skip(compacted_regions) {
                self.forwarding.scan_marked_objects(
                    region.region.start(),
                    region.cursor(),
                    &mut |obj: ObjectReference| {
                        objects.insert(obj);
                    },
                );
            }
        });

        objects
    }

    #[cfg(feature = "uffd")]
    pub fn validate_mapped_mark_words(
        &self,
        shadows: &[crate::policy::compressor::uffd::RegionShadow],
    ) -> Result<(), String> {
        let count = self.compaction_region_count();
        let summaries = self.compaction_summaries.read().unwrap();
        if summaries.len() < count || summaries.iter().take(count).any(|s| s.is_none()) {
            return Err("Uffd mark-word validation failed: missing cached compaction summaries".to_string());
        }
        if shadows.len() < count {
            return Err("Uffd mark-word validation failed: missing shadow metadata".to_string());
        }

        let mut neutral = 0usize;
        let mut fast_locked = 0usize;
        let mut monitor = 0usize;
        let mut marked = 0usize;
        let mut inflating = 0usize;

        for (region_idx, summary) in summaries.iter().take(count).flatten().enumerate() {
            let shadow_region_start = unsafe { Address::from_usize(shadows[region_idx].shadow_start) };
            for obj_meta in &summary.objects {
                let new_obj = unsafe { ObjectReference::from_raw_address_unchecked(obj_meta.dst_start) };
                let Some(mark) = VM::VMObjectModel::debug_mark_word(new_obj) else {
                    continue;
                };
                let shadow_obj_addr = shadow_region_start + (obj_meta.obj.to_raw_address() - summary.region_start);
                let shadow_obj = unsafe { ObjectReference::from_raw_address_unchecked(shadow_obj_addr) };
                let shadow_mark = VM::VMObjectModel::debug_mark_word(shadow_obj);
                let lock_bits = mark & 0x3;
                match lock_bits {
                    0 => fast_locked += 1,
                    1 => neutral += 1,
                    2 => {
                        monitor += 1;
                        let monitor_addr = mark ^ 0x2;
                        if monitor_addr == 0 || !unsafe { Address::from_usize(monitor_addr) }.is_mapped() {
                            return Err(format!(
                                "Uffd mark-word validation failed: object {} (from {}) has unmapped monitor mark word 0x{:x} (shadow_mark={:?}, region_idx={})",
                                new_obj,
                                obj_meta.obj,
                                mark,
                                shadow_mark,
                                region_idx,
                            ));
                        }
                    }
                    3 => {
                        marked += 1;
                        return Err(format!(
                            "Uffd mark-word validation failed: object {} (from {}) retains marked/unused header 0x{:x} (shadow_mark={:?}, region_idx={})",
                            new_obj,
                            obj_meta.obj,
                            mark,
                            shadow_mark,
                            region_idx,
                        ));
                    }
                    _ => unreachable!(),
                }
                if mark == 0 {
                    inflating += 1;
                }
            }
        }

        if compressor_perf_trace_enabled() {
            info!(
                "ValidateUffdMarkWords: neutral={}, fast_locked={}, monitor={}, inflating={}, marked={}",
                neutral,
                fast_locked,
                monitor,
                inflating,
                marked,
            );
        }
        Ok(())
    }

    /// Validate that all Compressor-space references inside compacted Compressor objects
    /// point to valid destination object starts.
    #[cfg(feature = "uffd")]
    pub fn validate_mapped_compacted_objects(&self) -> Result<(), String> {
        let valid_objects = self.collect_destination_objects();
        let summaries = self.compaction_summaries.read().unwrap().clone();
        let compacted_regions = self.page_metadata.read().unwrap().len();
        let trace_match_desc = std::env::var("MMTK_TRACE_MATCHING_MAPPED_OBJECT_DESC").ok();
        let trace_match_role = std::env::var("MMTK_TRACE_MATCHING_MAPPED_OBJECT_ROLE").ok();
        let trace_match_referent_desc =
            std::env::var("MMTK_TRACE_MATCHING_MAPPED_REFERENT_DESC").ok();
        let trace_match_referent_role =
            std::env::var("MMTK_TRACE_MATCHING_MAPPED_REFERENT_ROLE").ok();
        let mut trace_budget = 32usize;
        let mut err = None;
        for summary in summaries.into_iter().take(compacted_regions).flatten() {
            if err.is_some() {
                break;
            }
            for obj_meta in &summary.objects {
                if err.is_some() {
                    break;
                }
                let obj = obj_meta.obj;
                let new_obj = unsafe {
                    ObjectReference::from_raw_address_unchecked(obj_meta.dst_start)
                };
                if !VM::VMObjectModel::is_object_sane(new_obj) {
                    err = Some(format!(
                        "Uffd mapped-heap validation failed: destination object {} (from {}) is not sane",
                        new_obj, obj
                    ));
                    break;
                }
                let actual_size = VM::VMObjectModel::get_current_size(new_obj);
                if actual_size != obj_meta.copied_size {
                    err = Some(format!(
                        "Uffd mapped-heap validation failed: destination object {} (from {}) has size {}, expected {}",
                        new_obj, obj, actual_size, obj_meta.copied_size
                    ));
                    break;
                }
                let object_desc = VM::VMObjectModel::debug_object_description(new_obj);
                let object_role = VM::VMObjectModel::debug_object_role(new_obj);
                let should_trace_object = trace_budget > 0
                    && (trace_match_desc.as_ref().is_some_and(|needle| {
                        object_desc
                            .as_ref()
                            .is_some_and(|desc| desc.contains(needle))
                    })
                        || trace_match_role.as_ref().is_some_and(|needle| {
                            object_role
                                .as_ref()
                                .is_some_and(|role| role.contains(needle))
                        }));
                if should_trace_object {
                    info!(
                        "TraceMappedHeapObject: object={} from={} desc={:?} role={:?}",
                        new_obj, obj, object_desc, object_role
                    );
                }
                VM::VMScanning::scan_object(
                    crate::util::opaque_pointer::VMWorkerThread(
                        crate::util::opaque_pointer::VMThread::UNINITIALIZED,
                    ),
                    new_obj,
                    &mut |slot: VM::VMSlot| {
                        if err.is_some() {
                            return;
                        }
                        let Some(referent) = slot.load() else {
                            return;
                        };
                        let slot_desc = VM::VMScanning::describe_slot(new_obj, slot)
                            .unwrap_or_else(|| "slot=<unknown>".to_string());
                        let referent_role = if should_trace_object
                            || trace_match_referent_role.is_some()
                        {
                            VM::VMObjectModel::debug_object_role(referent)
                        } else {
                            None
                        };
                        let referent_desc = if trace_match_referent_desc.is_some() {
                            VM::VMObjectModel::debug_object_description(referent)
                        } else {
                            None
                        };
                        let should_trace_referent = trace_match_referent_desc.as_ref().is_some_and(
                            |needle| referent_desc.as_ref().is_some_and(|desc| desc.contains(needle)),
                        ) || trace_match_referent_role.as_ref().is_some_and(|needle| {
                            referent_role
                                .as_ref()
                                .is_some_and(|role| role.contains(needle))
                        });
                        if (should_trace_object || should_trace_referent) && self.in_space(referent) {
                            info!(
                                "TraceMappedHeapObject: object={} slot={} referent={} desc={:?} role={:?} state=[{}] valid={}",
                                new_obj,
                                slot_desc,
                                referent,
                                referent_desc,
                                referent_role,
                                self.debug_describe_compaction_object(referent),
                                valid_objects.contains(&referent),
                            );
                        }
                        if self.in_space(referent) && !valid_objects.contains(&referent) {
                            let slot_desc = VM::VMScanning::describe_slot(new_obj, slot)
                                .unwrap_or_else(|| "slot=<unknown>".to_string());
                            let source_desc = VM::VMObjectModel::debug_object_description(obj);
                            let source_role = VM::VMObjectModel::debug_object_role(obj);
                            let referent_desc = VM::VMObjectModel::debug_object_description(referent);
                            let referent_role = VM::VMObjectModel::debug_object_role(referent);
                            let referent_mapped = referent.to_raw_address().is_mapped();
                            let referent_initialized = referent_mapped
                                && VM::VMObjectModel::is_object_start_initialized(referent);
                            let referent_marked = Self::is_marked(referent);
                            let referent_in_prefix = self.is_in_compaction_region_prefix(
                                referent,
                                self.compaction_region_count(),
                            );
                            let referent_forwarded = if referent_marked {
                                Some(self.forward(referent, false))
                            } else {
                                None
                            };
                            err = Some(format!(
                                "Uffd mapped-heap validation failed: object {} (from {}) desc={:?} role={:?} source_desc={:?} source_role={:?} contains stale Compressor ref {} desc={:?} role={:?} in {} (mapped={}, initialized={}, marked={}, in_prefix={}, forwarded={:?})",
                                new_obj,
                                obj,
                                object_desc,
                                object_role,
                                source_desc,
                                source_role,
                                referent,
                                referent_desc,
                                referent_role,
                                slot_desc,
                                referent_mapped,
                                referent_initialized,
                                referent_marked,
                                referent_in_prefix,
                                referent_forwarded,
                            ));
                        }
                    },
                );
                if should_trace_object {
                    trace_budget -= 1;
                }
            }
        }
        err.map_or(Ok(()), Err)
    }

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
    fn prepare_region_compaction_data(
        &self,
        region_start: Address,
        cursor: Address,
    ) -> PreparedRegionCompactionData {
        let mut objects = Vec::new();
        self.forwarding
            .scan_marked_objects(region_start, cursor, &mut |obj: ObjectReference| {
                objects.push(PreparedRegionCompactionObjectMetadata {
                    obj,
                    copied_size: VM::VMObjectModel::get_size_when_copied(obj),
                });
            });
        PreparedRegionCompactionData {
            region_start,
            source_end: cursor,
            objects,
        }
    }

    #[cfg(feature = "uffd")]
    fn summarize_region_compaction_from_prepared(
        &self,
        prepared: &PreparedRegionCompactionData,
    ) -> RegionCompactionSummary {
        const PAGE_SIZE: usize = crate::util::constants::BYTES_IN_PAGE;

        let mut objects = Vec::with_capacity(prepared.objects.len());
        let mut compacted_end = prepared.region_start;
        let mut has_movement = false;

        for obj_meta in &prepared.objects {
            let dst_start = self.forward(obj_meta.obj, false).to_raw_address();
            let dst_end = dst_start + obj_meta.copied_size;
            let first_page = (dst_start - prepared.region_start) / PAGE_SIZE;
            let last_page = (dst_end - 1 - prepared.region_start) / PAGE_SIZE;

            has_movement |= dst_start != obj_meta.obj.to_raw_address();
            compacted_end = compacted_end.max(dst_end);
            objects.push(RegionCompactionObjectMetadata {
                obj: obj_meta.obj,
                copied_size: obj_meta.copied_size,
                dst_start,
                dst_end,
                first_page,
                last_page,
            });
        }

        let compacted_pages = if compacted_end > prepared.region_start {
            (compacted_end - prepared.region_start).div_ceil(PAGE_SIZE)
        } else {
            0
        };

        RegionCompactionSummary {
            region_start: prepared.region_start,
            source_end: prepared.source_end,
            compacted_end,
            compacted_pages,
            has_movement,
            objects,
        }
    }

    #[cfg(feature = "uffd")]
    fn summarize_region_compaction(
        &self,
        region_start: Address,
        cursor: Address,
    ) -> RegionCompactionSummary {
        const PAGE_SIZE: usize = crate::util::constants::BYTES_IN_PAGE;

        let mut objects = Vec::new();
        let mut compacted_end = region_start;
        let mut has_movement = false;

        self.forwarding
            .scan_marked_objects(region_start, cursor, &mut |obj: ObjectReference| {
                let copied_size = VM::VMObjectModel::get_size_when_copied(obj);
                let dst_start = self.forward(obj, false).to_raw_address();
                let dst_end = dst_start + copied_size;
                let first_page = (dst_start - region_start) / PAGE_SIZE;
                let last_page = (dst_end - 1 - region_start) / PAGE_SIZE;

                has_movement |= dst_start != obj.to_raw_address();
                compacted_end = compacted_end.max(dst_end);
                objects.push(RegionCompactionObjectMetadata {
                    obj,
                    copied_size,
                    dst_start,
                    dst_end,
                    first_page,
                    last_page,
                });
            });

        let compacted_pages = if compacted_end > region_start {
            (compacted_end - region_start).div_ceil(PAGE_SIZE)
        } else {
            0
        };

        RegionCompactionSummary {
            region_start,
            source_end: cursor,
            compacted_end,
            compacted_pages,
            has_movement,
            objects,
        }
    }

    #[cfg(feature = "uffd")]
    fn build_region_page_metadata_from_summary(
        &self,
        summary: &RegionCompactionSummary,
    ) -> RegionPageMetadata {
        const PAGE_SIZE: usize = crate::util::constants::BYTES_IN_PAGE;

        let region_start = summary.region_start;
        let source_end = summary.source_end;
        let compacted_end = summary.compacted_end;
        let has_movement = summary.has_movement;
        let mut pages = vec![
            PageCompactMetadata {
                first_obj: None,
                first_obj_page_offset: 0,
            };
            summary.compacted_pages
        ];

        for obj_meta in &summary.objects {
            debug_assert_eq!(obj_meta.dst_end, obj_meta.dst_start + obj_meta.copied_size);
            for page_idx in obj_meta.first_page..=obj_meta.last_page {
                if pages[page_idx].first_obj.is_none() {
                    let page_start = region_start + page_idx * PAGE_SIZE;
                    let offset = if page_start > obj_meta.dst_start {
                        page_start - obj_meta.dst_start
                    } else {
                        0
                    };
                    pages[page_idx].first_obj = Some(obj_meta.obj);
                    pages[page_idx].first_obj_page_offset = offset as u32;
                }
            }
        }

        RegionPageMetadata {
            region_start,
            source_end,
            compacted_end,
            has_movement,
            pages,
        }
    }

    #[cfg(feature = "uffd")]
    fn build_region_page_metadata_legacy(
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

    #[cfg(feature = "uffd")]
    pub fn compaction_region_count(&self) -> usize {
        let total = self.num_regions();
        let limit = self.compaction_region_limit.load(Ordering::Acquire);
        if limit == 0 {
            total
        } else {
            limit.min(total)
        }
    }

    #[cfg(feature = "uffd")]
    pub fn debug_is_stale_compaction_source_ref(&self, object: ObjectReference) -> bool {
        if !self.in_space(object) {
            return false;
        }
        let limit = self.compaction_region_limit.load(Ordering::Acquire);
        if limit == 0 || !self.is_in_compaction_region_prefix(object, limit) {
            return false;
        }
        let (old_match, dst_match) = self.debug_compaction_address_role(object);
        if dst_match.is_some() {
            return false;
        }
        if let Some((old_obj, dst_start)) = old_match {
            return old_obj.to_raw_address() != dst_start;
        }
        true
    }

    #[cfg(feature = "uffd")]
    fn debug_compaction_address_role(
        &self,
        object: ObjectReference,
    ) -> (Option<(ObjectReference, Address)>, Option<(ObjectReference, Address)>) {
        let count = self.compaction_region_count();
        let summaries = self.compaction_summaries.read().unwrap();
        let mut old_match = None;
        let mut dst_match = None;
        for summary in summaries.iter().take(count).flatten() {
            for obj_meta in &summary.objects {
                if old_match.is_none() && obj_meta.obj == object {
                    old_match = Some((obj_meta.obj, obj_meta.dst_start));
                }
                if dst_match.is_none() && obj_meta.dst_start == object.to_raw_address() {
                    dst_match = Some((obj_meta.obj, obj_meta.dst_start));
                }
                if old_match.is_some() && dst_match.is_some() {
                    break;
                }
            }
            if old_match.is_some() && dst_match.is_some() {
                break;
            }
        }
        (old_match, dst_match)
    }

    #[cfg(feature = "uffd")]
    pub fn debug_describe_compaction_address_role(&self, object: ObjectReference) -> String {
        let (old_match, dst_match) = self.debug_compaction_address_role(object);
        format!("old_match={:?},dst_match={:?}", old_match, dst_match)
    }

    #[cfg(feature = "uffd")]
    pub fn debug_describe_compaction_object(&self, object: ObjectReference) -> String {
        let in_space = self.in_space(object);
        let mapped = object.to_raw_address().is_mapped();
        let initialized = mapped && VM::VMObjectModel::is_object_start_initialized(object);
        if !in_space {
            return format!(
                "in_space=false,mapped={},initialized={}",
                mapped, initialized
            );
        }

        let limit = self.compaction_region_limit.load(Ordering::Acquire);
        let in_prefix = limit > 0 && self.is_in_compaction_region_prefix(object, limit);
        let marked = Self::is_marked(object);
        let forwarded_raw = if marked {
            Some(self.forwarding.forward(object.to_raw_address()))
        } else {
            None
        };
        let forwarded = forwarded_raw.and_then(ObjectReference::from_raw_address);
        let moved = forwarded.map(|new_object| new_object != object).unwrap_or(false);
        let stale_source = self.debug_is_stale_compaction_source_ref(object);
        let role = self.debug_describe_compaction_address_role(object);
        format!(
            "in_space=true,mapped={},initialized={},in_prefix={},marked={},moved={},forwarded={:?},stale_source={},{}",
            mapped,
            initialized,
            in_prefix,
            marked,
            moved,
            forwarded,
            stale_source,
            role,
        )
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
        if !self
            .black_allocation_tracking_active
            .load(Ordering::Acquire)
            || cursor <= start
        {
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
                mutator
                    .allocator_impl_for_semantic::<BumpAllocator<VM>>(AllocationSemantics::Default)
            };
            if let Some((start, cursor, limit)) = allocator.current_buffer() {
                self.record_retired_bump_alloc_buffer(start, cursor, limit);
            }
        }
    }

    #[cfg(feature = "uffd")]
    pub fn finalize_inter_pause_black_allocations_for_compaction(&self) -> Vec<ObjectReference> {
        self.capture_current_black_allocation_buffers();
        let objects = self.take_black_allocations();
        // Advance region cursors so that offset-vector computation and page-metadata
        // building cover the newly marked inter-pause objects.  Without this, the
        // cursors remain at the pre-FinalMark position, and marked objects beyond
        // that cursor are invisible to `calculate_offset_vector`, `summarize_region_compaction`,
        // and `build_page_from_shadow`, leading to inconsistent compaction layout.
        if !objects.is_empty() {
            self.advance_region_cursors_for_black_allocations(&objects);
        }
        objects
    }

    /// Advance region cursors to cover all inter-pause black-allocated objects.
    ///
    /// Inter-pause allocations may live beyond the cursor recorded in the region
    /// page resource (the bump allocator retired the buffer for tracking but not
    /// for cursor accounting).  We must advance each affected region's cursor so
    /// that offset-vector tasks and summary tasks process these objects.
    #[cfg(feature = "uffd")]
    fn advance_region_cursors_for_black_allocations(&self, objects: &[ObjectReference]) {
        use std::collections::HashMap;
        let region_bytes = forwarding::CompressorRegion::BYTES;
        let limit = self.compaction_region_count();

        // Compute per-region max end address from the black-allocated objects.
        let mut max_ends: HashMap<usize, Address> = HashMap::new();
        for &obj in objects {
            let addr = obj.to_raw_address();
            let size = VM::VMObjectModel::get_current_size(obj);
            let obj_end = addr + size;
            // Determine which region this object belongs to.
            let region_start = addr.align_down(region_bytes);
            if let Some(region_index) = self.pr.with_regions(&mut |regions| {
                regions.iter().position(|r| r.region.start() == region_start)
            }) {
                if region_index < limit {
                    let entry = max_ends.entry(region_index).or_insert(obj_end);
                    if obj_end > *entry {
                        *entry = obj_end;
                    }
                }
            }
        }

        if max_ends.is_empty() {
            return;
        }

        // Advance cursors.
        let mut advanced_regions = 0usize;
        let mut advanced_bytes = 0usize;
        for (&region_index, &new_end) in &max_ends {
            let (did_advance, bytes) = self.pr.advance_cursor_to(region_index, new_end);
            if did_advance {
                advanced_regions += 1;
                advanced_bytes += bytes;
            }
        }

        if compressor_perf_trace_enabled() && advanced_regions > 0 {
            info!(
                "Compressor Compaction: advanced {} region cursors by {} bytes for inter-pause black allocations",
                advanced_regions, advanced_bytes
            );
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

    #[cfg(feature = "uffd")]
    fn add_object_reference_update_tasks(
        &'static self,
        objects: Vec<ObjectReference>,
        label: &'static str,
    ) {
        if objects.is_empty() {
            if compressor_perf_trace_enabled() {
                info!("{} completed in 0.000 ms (objects=0, packets=0)", label);
            }
            return;
        }

        let worker_count = self.scheduler.num_workers().max(1);
        let target_packets = worker_count * 4;
        let chunk_size = objects.len().div_ceil(target_packets).max(1);
        let packet_count = objects.len().div_ceil(chunk_size);
        let batch = Arc::new(ReferenceUpdateBatchState {
            label,
            started: std::time::Instant::now(),
            remaining_packets: AtomicUsize::new(packet_count),
            total_objects: objects.len(),
            total_packets: packet_count,
        });

        if compressor_perf_trace_enabled() {
            info!(
                "{} scheduled with {} objects across {} packets (chunk_size={})",
                label, batch.total_objects, batch.total_packets, chunk_size
            );
        }

        let packets: Vec<Box<dyn GCWork<VM>>> = objects
            .chunks(chunk_size)
            .map(|chunk| {
                Box::new(UpdateObjectReferencesChunk::<VM>::new(
                    self,
                    chunk.to_vec(),
                    batch.clone(),
                )) as Box<dyn GCWork<VM>>
            })
            .collect();
        self.scheduler.work_buckets[WorkBucketStage::Compact].bulk_add(packets);
    }

    #[cfg(feature = "uffd")]
    pub fn schedule_inter_pause_allocation_update_tasks(
        &'static self,
        objects: Vec<ObjectReference>,
    ) {
        if compressor_perf_trace_enabled() && !objects.is_empty() {
            let mut compressor_refs = 0usize;
            let mut marked_compressor_refs = 0usize;
            let mut unmarked_compressor_refs = 0usize;
            for &object in &objects {
                VM::VMScanning::scan_object(
                    crate::util::opaque_pointer::VMWorkerThread(
                        crate::util::opaque_pointer::VMThread::UNINITIALIZED,
                    ),
                    object,
                    &mut |slot: VM::VMSlot| {
                        let Some(referent) = slot.load() else {
                            return;
                        };
                        if self.in_space(referent) {
                            compressor_refs += 1;
                            if Self::is_marked(referent) {
                                marked_compressor_refs += 1;
                            } else {
                                unmarked_compressor_refs += 1;
                            }
                        }
                    },
                );
            }
            info!(
                "Compressor Compaction: inter-pause allocation refs total_objects={}, compressor_refs={}, marked_compressor_refs={}, unmarked_compressor_refs={}",
                objects.len(),
                compressor_refs,
                marked_compressor_refs,
                unmarked_compressor_refs,
            );
        }
        self.add_object_reference_update_tasks(
            objects,
            "Compressor Compaction: inter-pause allocation reference updates",
        );
    }

    #[cfg(feature = "uffd")]
    pub fn add_los_reference_update_tasks(&'static self, los: &'static LargeObjectSpace<VM>) {
        let mut objects = Vec::new();
        los.enumerate_to_space_objects(&mut object_enum::ClosureObjectEnumerator::<_, VM>::new(
            |o: ObjectReference| {
                objects.push(o);
            },
        ));
        self.add_object_reference_update_tasks(
            objects,
            "Compressor FinalMark: LOS reference updates",
        );
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

#[cfg(feature = "uffd")]
struct CompactionPrepareBatchState {
    on_complete: Arc<dyn Fn() + Send + Sync>,
    started: std::time::Instant,
    remaining_packets: AtomicUsize,
    total_packets: usize,
}

#[cfg(feature = "uffd")]
impl CompactionPrepareBatchState {
    fn new(on_complete: Arc<dyn Fn() + Send + Sync>, total_packets: usize) -> Self {
        Self {
            on_complete,
            started: std::time::Instant::now(),
            remaining_packets: AtomicUsize::new(total_packets),
            total_packets,
        }
    }
}

#[cfg(feature = "uffd")]
pub struct PrepareRegionCompactionData<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
    index: usize,
    snapshot: PreparedRegionCompactionSnapshot,
    batch: Arc<CompactionPrepareBatchState>,
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> GCWork<VM> for PrepareRegionCompactionData<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        let prepared = self
            .compressor_space
            .prepare_region_compaction_data(self.snapshot.region_start, self.snapshot.source_end);
        self.compressor_space
            .calculate_offset_vector_for_prepare_region(
                forwarding::CompressorRegion::from_aligned_address(self.snapshot.region_start),
                self.snapshot.source_end,
            );
        self.compressor_space
            .cache_prepared_region_compaction_data_at_index(self.index, prepared);

        if self.batch.remaining_packets.fetch_sub(1, Ordering::AcqRel) == 1 {
            (self.batch.on_complete.as_ref())();
            if compressor_perf_trace_enabled() {
                info!(
                    "Compressor concurrent compaction prepare completed in {} ms (packets={})",
                    format_perf_ms(self.batch.started.elapsed()),
                    self.batch.total_packets,
                );
            }
        }
    }
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> PrepareRegionCompactionData<VM> {
    fn new(
        compressor_space: &'static CompressorSpace<VM>,
        index: usize,
        snapshot: PreparedRegionCompactionSnapshot,
        batch: Arc<CompactionPrepareBatchState>,
    ) -> Self {
        Self {
            compressor_space,
            index,
            snapshot,
            batch,
        }
    }
}

#[cfg(feature = "uffd")]
pub struct CacheRegionCompactionSummary<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
    index: usize,
    region_start: Address,
    cursor: Address,
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> GCWork<VM> for CacheRegionCompactionSummary<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        let summary = self
            .compressor_space
            .summarize_region_compaction_for_current_region(
                self.index,
                self.region_start,
                self.cursor,
            );
        self.compressor_space
            .cache_region_compaction_summary_at_index(self.index, summary);
    }
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> CacheRegionCompactionSummary<VM> {
    pub fn new(
        compressor_space: &'static CompressorSpace<VM>,
        index: usize,
        region_start: Address,
        cursor: Address,
    ) -> Self {
        Self {
            compressor_space,
            index,
            region_start,
            cursor,
        }
    }
}

#[cfg(feature = "uffd")]
pub struct UpdateObjectReferencesChunk<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
    objects: Vec<ObjectReference>,
    batch: Arc<ReferenceUpdateBatchState>,
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> GCWork<VM> for UpdateObjectReferencesChunk<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        for &object in &self.objects {
            self.compressor_space.update_references(worker, object);
        }

        if self.batch.remaining_packets.fetch_sub(1, Ordering::AcqRel) == 1
            && compressor_perf_trace_enabled()
        {
            info!(
                "{} completed in {} ms (objects={}, packets={})",
                self.batch.label,
                format_perf_ms(self.batch.started.elapsed()),
                self.batch.total_objects,
                self.batch.total_packets
            );
        }
    }
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> UpdateObjectReferencesChunk<VM> {
    fn new(
        compressor_space: &'static CompressorSpace<VM>,
        objects: Vec<ObjectReference>,
        batch: Arc<ReferenceUpdateBatchState>,
    ) -> Self {
        Self {
            compressor_space,
            objects,
            batch,
        }
    }
}
