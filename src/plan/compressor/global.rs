#[cfg(not(feature = "uffd"))]
use super::gc_work::AfterCompact;
use super::gc_work::CompressorWorkContext;
use super::gc_work::{ForwardingProcessEdges, GenerateWork, MarkingProcessEdges, UpdateReferences};
use crate::plan::compressor::mutator::ALLOCATOR_MAPPING;
use crate::plan::global::CreateGeneralPlanArgs;
use crate::plan::global::CreateSpecificPlanArgs;
use crate::plan::global::{BasePlan, CommonPlan};
use crate::plan::plan_constraints::MAX_NON_LOS_ALLOC_BYTES_COPYING_PLAN;
use crate::plan::{AllocationSemantics, Plan, PlanConstraints};
use crate::policy::compressor::CompressorSpace;
use crate::policy::space::Space;
use crate::scheduler::gc_work::*;
use crate::scheduler::{GCWorkScheduler, WorkBucketStage};
use crate::util::alloc::allocators::AllocatorSelector;
use crate::util::heap::gc_trigger::SpaceStats;
#[allow(unused_imports)]
use crate::util::heap::VMRequest;
use crate::util::metadata::side_metadata::SideMetadataContext;
use crate::util::opaque_pointer::*;
use crate::vm::VMBinding;
use enum_map::EnumMap;
use mmtk_macros::{HasSpaces, PlanTraceObject};
#[cfg(feature = "uffd")]
use std::sync::atomic::{AtomicBool, Ordering};

/// [`Compressor`] implements a stop-the-world and parallel implementation of
/// the Compressor, as described in Kermany and Petrank,
/// [The Compressor: concurrent, incremental, and parallel compaction](https://dl.acm.org/doi/10.1145/1133255.1134023).
///
/// With `feature = "uffd"` (Linux only), the Compressor uses a single-pause concurrent
/// compaction protocol modeled after Android ART's mark-compact collector:
/// `CompactionPause()` performs setup under STW and `CompactionPhase()` finishes
/// the page processing concurrently.
#[derive(HasSpaces, PlanTraceObject)]
pub struct Compressor<VM: VMBinding> {
    #[parent]
    pub common: CommonPlan<VM>,
    #[space]
    pub compressor_space: CompressorSpace<VM>,
    #[cfg(feature = "uffd")]
    uffd_concurrent_active: AtomicBool,
    #[cfg(feature = "uffd")]
    uffd_epoch_ready: AtomicBool,
}

/// The plan constraints for the Compressor plan.
pub const COMPRESSOR_CONSTRAINTS: PlanConstraints = PlanConstraints {
    max_non_los_default_alloc_bytes: MAX_NON_LOS_ALLOC_BYTES_COPYING_PLAN,
    moves_objects: true,
    needs_forward_after_liveness: true,
    ..PlanConstraints::default()
};

impl<VM: VMBinding> Plan for Compressor<VM> {
    fn constraints(&self) -> &'static PlanConstraints {
        &COMPRESSOR_CONSTRAINTS
    }

    fn collection_required(&self, space_full: bool, _space: Option<SpaceStats<Self::VM>>) -> bool {
        #[cfg(feature = "uffd")]
        if self.uffd_concurrent_active.load(Ordering::Acquire) {
            // Mirror ART's high-level protocol: do not start another GC while
            // the concurrent compaction epoch is still active.
            return false;
        }
        self.base().collection_required(self, space_full)
    }

    fn common(&self) -> &CommonPlan<VM> {
        &self.common
    }

    fn base(&self) -> &BasePlan<VM> {
        &self.common.base
    }

    fn base_mut(&mut self) -> &mut BasePlan<Self::VM> {
        &mut self.common.base
    }

    fn prepare(&mut self, tls: VMWorkerThread) {
        self.common.prepare(tls, true);
        self.compressor_space.prepare();
    }

    fn release(&mut self, tls: VMWorkerThread) {
        #[cfg(feature = "uffd")]
        if self.uffd_epoch_ready.load(Ordering::Acquire) {
            // ART updates roots / non-moving state in the pause before mutator
            // resume, but keeps the moving-space compaction metadata live for
            // the concurrent phase. Release common spaces now and defer only
            // `compressor_space.release()` until the UFFD epoch completes.
            self.common.release(tls, true);
            return;
        }
        self.common.release(tls, true);
        self.compressor_space.release();
    }

    fn end_of_gc(&mut self, tls: VMWorkerThread) {
        self.common.end_of_gc(tls);
        #[cfg(feature = "uffd")]
        if self.uffd_epoch_ready.swap(false, Ordering::AcqRel) {
            // Match ART's CompactionPause()->CompactionPhase() handoff: once
            // mutators resume, new allocations must be treated as already-live.
            self.set_allocate_as_live(true);
            self.uffd_concurrent_active.store(true, Ordering::Release);
        }
    }

    fn get_allocator_mapping(&self) -> &'static EnumMap<AllocationSemantics, AllocatorSelector> {
        &ALLOCATOR_MAPPING
    }

    fn schedule_collection(&'static self, scheduler: &GCWorkScheduler<VM>) {
        #[cfg(feature = "uffd")]
        self.uffd_epoch_ready.store(false, Ordering::Release);

        // Stop & scan mutators
        scheduler.work_buckets[WorkBucketStage::Unconstrained]
            .add(StopMutators::<CompressorWorkContext<VM>>::new());

        // Prepare global/collectors/mutators
        scheduler.work_buckets[WorkBucketStage::Prepare]
            .add(Prepare::<CompressorWorkContext<VM>>::new(self));

        scheduler.work_buckets[WorkBucketStage::CalculateForwarding].add(GenerateWork::new(
            &self.compressor_space,
            CompressorSpace::<VM>::add_offset_vector_tasks,
        ));

        #[cfg(feature = "uffd")]
        {
            // Mirror ART's per-destination-page compaction metadata prepared in
            // the compaction pause (`InitMovingSpaceFirstObjects`).
            scheduler.work_buckets[WorkBucketStage::SecondRoots].add(GenerateWork::new(
                &self.compressor_space,
                CompressorSpace::<VM>::build_all_page_metadata,
            ));
        }

        // Scan roots to update references
        scheduler.work_buckets[WorkBucketStage::SecondRoots].add(UpdateReferences::<VM>::new());

        // Compact phase
        #[cfg(feature = "uffd")]
        {
            scheduler.work_buckets[WorkBucketStage::Compact].add(
                super::gc_work::UffdConcurrentSetup::<VM>::new(
                    self,
                    &self.compressor_space,
                    &self.common.los,
                ),
            );
        }
        #[cfg(not(feature = "uffd"))]
        {
            scheduler.work_buckets[WorkBucketStage::Compact].add(GenerateWork::new(
                &self.compressor_space,
                CompressorSpace::<VM>::add_compact_tasks,
            ));
            scheduler.work_buckets[WorkBucketStage::Compact].set_sentinel(Box::new(
                AfterCompact::<VM>::new(&self.compressor_space, &self.common.los),
            ));
        }

        scheduler.work_buckets[WorkBucketStage::Release]
            .add(Release::<CompressorWorkContext<VM>>::new(self));

        if !*self.base().options.no_reference_types {
            use crate::util::reference_processor::{
                PhantomRefProcessing, SoftRefProcessing, WeakRefProcessing,
            };
            scheduler.work_buckets[WorkBucketStage::SoftRefClosure]
                .add(SoftRefProcessing::<MarkingProcessEdges<VM>>::new());
            scheduler.work_buckets[WorkBucketStage::WeakRefClosure]
                .add(WeakRefProcessing::<VM>::new());
            scheduler.work_buckets[WorkBucketStage::PhantomRefClosure]
                .add(PhantomRefProcessing::<VM>::new());
            use crate::util::reference_processor::RefForwarding;
            scheduler.work_buckets[WorkBucketStage::RefForwarding]
                .add(RefForwarding::<ForwardingProcessEdges<VM>>::new());
            use crate::util::reference_processor::RefEnqueue;
            scheduler.work_buckets[WorkBucketStage::Release].add(RefEnqueue::<VM>::new());
        }

        if !*self.base().options.no_finalizer {
            use crate::util::finalizable_processor::{Finalization, ForwardFinalization};
            scheduler.work_buckets[WorkBucketStage::FinalRefClosure]
                .add(Finalization::<MarkingProcessEdges<VM>>::new());
            scheduler.work_buckets[WorkBucketStage::FinalizableForwarding]
                .add(ForwardFinalization::<ForwardingProcessEdges<VM>>::new());
        }

        scheduler.work_buckets[WorkBucketStage::VMRefClosure]
            .set_sentinel(Box::new(VMProcessWeakRefs::<MarkingProcessEdges<VM>>::new()));
        scheduler.work_buckets[WorkBucketStage::VMRefForwarding]
            .add(VMForwardWeakRefs::<ForwardingProcessEdges<VM>>::new());
        scheduler.work_buckets[WorkBucketStage::Release].add(VMPostForwarding::<VM>::default());

        #[cfg(feature = "analysis")]
        {
            use crate::util::analysis::GcHookWork;
            scheduler.work_buckets[WorkBucketStage::Unconstrained].add(GcHookWork);
        }
        #[cfg(feature = "sanity")]
        scheduler.work_buckets[WorkBucketStage::Final]
            .add(crate::util::sanity::sanity_checker::ScheduleSanityGC::<Self>::new(self));
    }

    fn current_gc_may_move_object(&self) -> bool {
        true
    }

    fn get_used_pages(&self) -> usize {
        self.compressor_space.reserved_pages() + self.common.get_used_pages()
    }
}

impl<VM: VMBinding> Compressor<VM> {
    pub fn new(args: CreateGeneralPlanArgs<VM>) -> Self {
        let mut plan_args = CreateSpecificPlanArgs {
            global_args: args,
            constraints: &COMPRESSOR_CONSTRAINTS,
            global_side_metadata_specs: SideMetadataContext::new_global_specs(&[]),
        };

        let res = Compressor {
            compressor_space: CompressorSpace::new(plan_args.get_normal_space_args(
                "compressor_space",
                true,
                false,
                VMRequest::discontiguous(),
            )),
            common: CommonPlan::new(plan_args),
            #[cfg(feature = "uffd")]
            uffd_concurrent_active: AtomicBool::new(false),
            #[cfg(feature = "uffd")]
            uffd_epoch_ready: AtomicBool::new(false),
        };

        res.verify_side_metadata_sanity();
        res
    }

    #[cfg(feature = "uffd")]
    fn set_allocate_as_live(&self, active: bool) {
        use crate::plan::global::HasSpaces;
        self.for_each_space(&mut |space: &dyn Space<VM>| {
            space.set_allocate_as_live(active);
        });
    }

    #[cfg(feature = "uffd")]
    pub fn mark_uffd_epoch_ready(&self) {
        self.uffd_epoch_ready.store(true, Ordering::Release);
    }

    #[cfg(feature = "uffd")]
    pub fn is_uffd_concurrent_active(&self) -> bool {
        self.uffd_concurrent_active.load(Ordering::Acquire)
    }

    #[cfg(feature = "uffd")]
    pub fn finish_uffd_epoch(&self) {
        self.compressor_space.release();
        self.set_allocate_as_live(false);
        self.uffd_concurrent_active.store(false, Ordering::Release);
    }
}
