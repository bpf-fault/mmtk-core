use super::gc_work::GenImmixMatureGCWorkContext;
use super::gc_work::GenImmixNurseryGCWorkContext;
use crate::plan::generational::global::CommonGenPlan;
use crate::plan::generational::global::GenerationalPlan;
use crate::plan::global::BasePlan;
use crate::plan::global::CommonPlan;
use crate::plan::global::CreateGeneralPlanArgs;
use crate::plan::global::CreateSpecificPlanArgs;
use crate::plan::AllocationSemantics;
use crate::plan::Plan;
use crate::plan::PlanConstraints;
use crate::policy::gc_work::TraceKind;
use crate::policy::immix::defrag::StatsForDefrag;
use crate::policy::immix::ImmixSpace;
use crate::policy::immix::ImmixSpaceArgs;
use crate::policy::immix::{TRACE_KIND_DEFRAG, TRACE_KIND_FAST};
use crate::policy::space::Space;
use crate::scheduler::GCWorkScheduler;
use crate::scheduler::GCWorker;
use crate::util::alloc::allocators::AllocatorSelector;
use crate::util::copy::*;
use crate::util::heap::gc_trigger::SpaceStats;
use crate::util::heap::VMRequest;
use crate::util::metadata::log_bit::UnlogBitsOperation;
use crate::util::Address;
use crate::util::ObjectReference;
use crate::util::VMWorkerThread;
use crate::vm::*;
use crate::ObjectQueue;

use enum_map::EnumMap;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use mmtk_macros::{HasSpaces, PlanTraceObject};

/// Generational immix. This implements the functionality of a two-generation copying
/// collector where the higher generation is an immix space.
/// See the PLDI'08 paper by Blackburn and McKinley for a description
/// of the algorithm: <http://doi.acm.org/10.1145/1375581.1375586>.
#[derive(HasSpaces, PlanTraceObject)]
pub struct GenImmix<VM: VMBinding> {
    /// Generational plan, which includes a nursery space and operations related with nursery.
    #[parent]
    pub gen: CommonGenPlan<VM>,
    /// An immix space as the mature space.
    #[post_scan]
    #[space]
    #[copy_semantics(CopySemantics::Mature)]
    pub immix_space: ImmixSpace<VM>,
    /// Whether the last GC was a defrag GC for the immix space.
    pub last_gc_was_defrag: AtomicBool,
    /// Whether the last GC was a full heap GC
    pub last_gc_was_full_heap: AtomicBool,
    /// Conservative remembered set for spaces without page dirty tracking
    /// (LOS/immortal/nonmoving): filled in `prepare()` while the LOS
    /// treadmill is still quiescent, consumed by `ScanDirtyStash` in Closure.
    pub(super) dirty_stash: std::sync::Mutex<Vec<ObjectReference>>,
    /// Coalesced page ranges of LOS objects protected at the last
    /// `end_of_gc`, to be unprotected at the next `prepare`.
    los_protected: std::sync::Mutex<Vec<(Address, usize)>>,
}

/// The plan constraints for the generational immix plan.
pub const GENIMMIX_CONSTRAINTS: PlanConstraints = PlanConstraints {
    // The maximum object size that can be allocated without LOS is restricted by the max immix object size.
    // This might be too restrictive, as our default allocator is bump pointer (nursery allocator) which
    // can allocate objects larger than max immix object size. However, for copying, we haven't implemented
    // copying to LOS so we always copy from nursery to the mature immix space. In this case, we should not
    // allocate objects larger than the max immix object size to nursery as well.
    // TODO: We may want to fix this, as this possibly has negative performance impact.
    max_non_los_default_alloc_bytes: crate::util::rust_util::min_of_usize(
        crate::policy::immix::MAX_IMMIX_OBJECT_SIZE,
        crate::plan::generational::GEN_CONSTRAINTS.max_non_los_default_alloc_bytes,
    ),
    ..crate::plan::generational::GEN_CONSTRAINTS
};

impl<VM: VMBinding> Plan for GenImmix<VM> {
    fn constraints(&self) -> &'static PlanConstraints {
        &GENIMMIX_CONSTRAINTS
    }

    fn create_copy_config(&'static self) -> CopyConfig<Self::VM> {
        use enum_map::enum_map;
        CopyConfig {
            copy_mapping: enum_map! {
                CopySemantics::PromoteToMature => CopySelector::ImmixHybrid(0),
                CopySemantics::Mature => CopySelector::ImmixHybrid(0),
                _ => CopySelector::Unused,
            },
            space_mapping: vec![(CopySelector::ImmixHybrid(0), &self.immix_space)],
            constraints: &GENIMMIX_CONSTRAINTS,
        }
    }

    fn last_collection_was_exhaustive(&self) -> bool {
        self.last_gc_was_full_heap.load(Ordering::Relaxed)
            && self
                .immix_space
                .is_last_gc_exhaustive(self.last_gc_was_defrag.load(Ordering::Relaxed))
    }

    fn collection_required(&self, space_full: bool, space: Option<SpaceStats<Self::VM>>) -> bool
    where
        Self: Sized,
    {
        self.gen.collection_required(self, space_full, space)
    }

    // GenImmixMatureProcessEdges<VM, { TraceKind::Defrag }> and GenImmixMatureProcessEdges<VM, { TraceKind::Fast }>
    // are different types. However, it seems clippy does not recognize the constant type parameter and thinks we have identical blocks
    // in different if branches.
    #[allow(clippy::if_same_then_else)]
    #[allow(clippy::branches_sharing_code)]
    fn schedule_collection(&'static self, scheduler: &GCWorkScheduler<Self::VM>) {
        let is_full_heap = self.requires_full_heap_collection();
        probe!(mmtk, gen_full_heap, is_full_heap);

        if !is_full_heap {
            info!("Nursery GC");
            scheduler.schedule_common_work::<GenImmixNurseryGCWorkContext<VM>>(self);
            if crate::util::dirty_track::is_dirty_tracking_active() {
                use crate::plan::generational::gc_work::GenNurseryProcessEdges;
                use crate::policy::gc_work::DEFAULT_TRACE;
                use crate::scheduler::WorkBucketStage;
                type E<VM> = GenNurseryProcessEdges<VM, GenImmix<VM>, DEFAULT_TRACE>;
                #[cfg(feature = "vo_bit")]
                scheduler.work_buckets[WorkBucketStage::Closure].add(
                    crate::plan::generational::gc_work::ScanVMDirtyPages::<E<VM>>::new(),
                );
                #[cfg(not(feature = "vo_bit"))]
                panic!("dirty tracking requires the vo_bit feature");
                // Spaces without page dirty tracking are conservatively
                // re-scanned every nursery GC via the stash filled in
                // `prepare()`.
                scheduler.work_buckets[WorkBucketStage::Closure]
                    .add(super::gc_work::ScanDirtyStash::<VM>::new());
            }
        } else {
            info!("Full heap GC");
            crate::plan::immix::Immix::schedule_immix_full_heap_collection::<
                GenImmix<VM>,
                GenImmixMatureGCWorkContext<VM, TRACE_KIND_FAST>,
                GenImmixMatureGCWorkContext<VM, TRACE_KIND_DEFRAG>,
            >(self, &self.immix_space, scheduler);
        }
    }

    fn get_allocator_mapping(&self) -> &'static EnumMap<AllocationSemantics, AllocatorSelector> {
        &super::mutator::ALLOCATOR_MAPPING
    }

    fn prepare(&mut self, tls: VMWorkerThread) {
        let full_heap = !self.gen.is_current_gc_nursery();
        // Drop page protection on the whole mature space so GC-time writes
        // (promotion into recycled blocks, defrag) never fault.  Mutators are
        // already suspended; the dirty set is drained later in Closure.
        if let Some(tracker) = crate::util::dirty_track::dirty_tracker() {
            self.for_each_mature_chunk(|start, bytes| tracker.unprotect(start, bytes));
            for (start, bytes) in self.los_protected.lock().unwrap().drain(..) {
                tracker.unprotect(start, bytes);
            }
            // Conservative remembered set for the remaining non-tracked
            // spaces (the LOS is page-WP-tracked like the immix space).
            // Must run before `gen.prepare()` (space prepare may race).
            if !full_heap {
                use crate::util::object_enum::ClosureObjectEnumerator;
                let mut stash = self.dirty_stash.lock().unwrap();
                debug_assert!(stash.is_empty());
                let mut enumerator = ClosureObjectEnumerator::<_, VM>::new(|obj| {
                    stash.push(obj);
                });
                self.gen.common.immortal.enumerate_objects(&mut enumerator);
                let mut enumerator = ClosureObjectEnumerator::<_, VM>::new(|obj| {
                    stash.push(obj);
                });
                self.gen.common.nonmoving.enumerate_objects(&mut enumerator);
                probe!(mmtk, stash_size, stash.len());
            }
        }
        self.gen.prepare(tls);
        if full_heap {
            self.immix_space.prepare(
                full_heap,
                Some(StatsForDefrag::new(self)),
                // Bulk clear unlog bits so that we will reconstruct them.
                UnlogBitsOperation::BulkClear,
            );
        } else {
            // We don't do anything special to unlog bits during nursery GC
            // because ProcessModBuf will set the unlog bits back.
        }
    }

    fn release(&mut self, tls: VMWorkerThread) {
        let full_heap = !self.gen.is_current_gc_nursery();
        self.gen.release(tls);
        if full_heap {
            self.immix_space.release(
                full_heap,
                // We reconstructred unlog bits during tracing.  Keep them.
                UnlogBitsOperation::NoOp,
            );
        } else {
            // We don't do anything special to unlog bits during nursery GC
            // because ProcessModBuf has set the unlog bits back.
        }

        self.last_gc_was_full_heap
            .store(full_heap, Ordering::Relaxed);
    }

    fn end_of_gc(&mut self, tls: VMWorkerThread) {
        let next_gc_full_heap = CommonGenPlan::should_next_gc_be_full_heap(self);
        self.gen.end_of_gc(tls, next_gc_full_heap);

        let did_defrag = self.immix_space.end_of_gc();
        self.last_gc_was_defrag.store(did_defrag, Ordering::Relaxed);

        // Re-arm the page-protection write barrier before mutators resume:
        // every mature page is clean now (the nursery is empty, so no
        // old->young refs exist).  Newly mapped chunks are registered first.
        // Dirty bits set by faults during the GC itself are stale; discard.
        if let Some(tracker) = crate::util::dirty_track::dirty_tracker() {
            tracker.drain_dirty(|_| {});
            self.for_each_mature_chunk(|start, bytes| {
                tracker.ensure_registered(start, bytes);
                tracker.protect(start, bytes);
            });
            // Protect the pages of all live LOS objects (coalesced runs).
            // The treadmill is quiescent here (post-release).
            {
                use crate::util::object_enum::ClosureObjectEnumerator;
                const PAGE: usize = crate::util::dirty_track::BYTES_IN_PAGE;
                let mut page_ranges: Vec<(Address, usize)> = Vec::new();
                let mut enumerator = ClosureObjectEnumerator::<_, VM>::new(|obj| {
                    let start = obj.to_object_start::<VM>().align_down(PAGE);
                    let end = (obj.to_object_start::<VM>()
                        + VM::VMObjectModel::get_current_size(obj))
                    .align_up(PAGE);
                    page_ranges.push((start, end - start));
                });
                self.gen.common.los.enumerate_objects(&mut enumerator);
                page_ranges.sort_unstable_by_key(|r| r.0);
                let mut coalesced: Vec<(Address, usize)> = Vec::new();
                for (start, bytes) in page_ranges {
                    match coalesced.last_mut() {
                        Some(last) if start <= last.0 + last.1 => {
                            let end = std::cmp::max(last.0 + last.1, start + bytes);
                            last.1 = end - last.0;
                        }
                        _ => coalesced.push((start, bytes)),
                    }
                }
                probe!(mmtk, los_protect_ranges, coalesced.len());
                for &(start, bytes) in &coalesced {
                    tracker.ensure_registered_range(start, bytes);
                    tracker.protect(start, bytes);
                }
                *self.los_protected.lock().unwrap() = coalesced;
            }
        }
    }

    fn current_gc_may_move_object(&self) -> bool {
        if self.is_current_gc_nursery() {
            true
        } else {
            self.immix_space.in_defrag()
        }
    }

    fn get_collection_reserved_pages(&self) -> usize {
        self.gen.get_collection_reserved_pages() + self.immix_space.defrag_headroom_pages()
    }

    fn get_used_pages(&self) -> usize {
        self.gen.get_used_pages() + self.immix_space.reserved_pages()
    }

    /// Return the number of pages available for allocation. Assuming all future allocations goes to nursery.
    fn get_available_pages(&self) -> usize {
        // super.get_available_pages() / 2 to reserve pages for copying
        (self
            .get_total_pages()
            .saturating_sub(self.get_reserved_pages()))
            >> 1
    }

    fn base(&self) -> &BasePlan<VM> {
        &self.gen.common.base
    }

    fn base_mut(&mut self) -> &mut BasePlan<Self::VM> {
        &mut self.gen.common.base
    }

    fn common(&self) -> &CommonPlan<VM> {
        &self.gen.common
    }

    fn generational(&self) -> Option<&dyn GenerationalPlan<VM = VM>> {
        Some(self)
    }
}

impl<VM: VMBinding> GenerationalPlan for GenImmix<VM> {
    fn is_current_gc_nursery(&self) -> bool {
        self.gen.is_current_gc_nursery()
    }

    fn is_object_in_nursery(&self, object: ObjectReference) -> bool {
        self.gen.nursery.in_space(object)
    }

    fn is_address_in_nursery(&self, addr: Address) -> bool {
        self.gen.nursery.address_in_space(addr)
    }

    fn get_mature_physical_pages_available(&self) -> usize {
        self.immix_space.available_physical_pages()
    }

    fn get_mature_reserved_pages(&self) -> usize {
        self.immix_space.reserved_pages()
    }

    fn force_full_heap_collection(&self) {
        self.gen.force_full_heap_collection()
    }

    fn last_collection_full_heap(&self) -> bool {
        self.gen.last_collection_full_heap()
    }
}

impl<VM: VMBinding> crate::plan::generational::global::GenerationalPlanExt<VM> for GenImmix<VM> {
    fn trace_object_nursery<Q: ObjectQueue, const KIND: TraceKind>(
        &self,
        queue: &mut Q,
        object: ObjectReference,
        worker: &mut GCWorker<VM>,
    ) -> ObjectReference {
        self.gen
            .trace_object_nursery::<Q, KIND>(queue, object, worker)
    }
}

impl<VM: VMBinding> GenImmix<VM> {
    pub fn new(args: CreateGeneralPlanArgs<VM>) -> Self {
        {
            let backend = *args.options.dirty_tracking;
            let vm_layout = crate::util::heap::layout::vm_layout::vm_layout();
            crate::util::dirty_track::init_dirty_tracker(
                backend,
                vm_layout.heap_start,
                vm_layout.heap_end,
            );
        }
        let mut plan_args = CreateSpecificPlanArgs {
            global_args: args,
            constraints: &GENIMMIX_CONSTRAINTS,
            global_side_metadata_specs:
                crate::plan::generational::new_generational_global_metadata_specs::<VM>(),
        };
        let immix_space = ImmixSpace::new(
            plan_args.get_mature_space_args(
                "immix_mature",
                true,
                false,
                VMRequest::discontiguous(),
            ),
            ImmixSpaceArgs {
                // In GenImmix, young objects are not allocated in ImmixSpace directly.
                mixed_age: false,
                never_move_objects: false,
            },
        );

        GenImmix {
            gen: CommonGenPlan::new(plan_args),
            immix_space,
            last_gc_was_defrag: AtomicBool::new(false),
            last_gc_was_full_heap: AtomicBool::new(false),
            dirty_stash: std::sync::Mutex::new(Vec::new()),
            los_protected: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn requires_full_heap_collection(&self) -> bool {
        self.gen.requires_full_heap_collection(self)
    }

    /// Visit every mapped chunk of the mature immix space as
    /// `(start, bytes)`, for page-protection dirty tracking.
    fn for_each_mature_chunk<F: FnMut(Address, usize)>(&self, mut f: F) {
        use crate::util::heap::chunk_map::Chunk;
        use crate::util::linear_scan::Region;
        for chunk in self.immix_space.chunk_map.all_chunks() {
            f(chunk.start(), Chunk::BYTES);
        }
    }
}
