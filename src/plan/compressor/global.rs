use super::gc_work::CompressorWorkContext;
use super::gc_work::{
    AfterCompact, ForwardingProcessEdges, GenerateWork, MarkingProcessEdges, UpdateReferences,
};
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
#[cfg(target_os = "linux")]
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
#[cfg(target_os = "linux")]
use std::sync::{Arc, Mutex};
#[cfg(target_os = "linux")]
use std::thread::JoinHandle;

/// [`Compressor`] implements a stop-the-world and parallel implementation of
/// the Compressor, as described in Kermany and Petrank,
/// [The Compressor: concurrent, incremental, and parallel compaction](https://dl.acm.org/doi/10.1145/1133255.1134023).
#[derive(HasSpaces, PlanTraceObject)]
pub struct Compressor<VM: VMBinding> {
    #[parent]
    pub common: CommonPlan<VM>,
    #[space]
    pub compressor_space: CompressorSpace<VM>,
    #[cfg(target_os = "linux")]
    uffd_concurrent_active: AtomicBool,
    #[cfg(target_os = "linux")]
    uffd_concurrent_pause: AtomicU8,
    #[cfg(target_os = "linux")]
    uffd_final_pause_requested: AtomicBool,
    #[cfg(target_os = "linux")]
    uffd_concurrent_regions_remaining: AtomicUsize,
    #[cfg(target_os = "linux")]
    uffd_context: Mutex<Option<Arc<crate::policy::compressor::uffd::UffdContext<VM>>>>,
    #[cfg(target_os = "linux")]
    uffd_handler_thread: Mutex<Option<JoinHandle<()>>>,
}

/// The plan constraints for the Compressor plan.
pub const COMPRESSOR_CONSTRAINTS: PlanConstraints = PlanConstraints {
    max_non_los_default_alloc_bytes: MAX_NON_LOS_ALLOC_BYTES_COPYING_PLAN,
    moves_objects: true,
    needs_forward_after_liveness: true,
    ..PlanConstraints::default()
};

#[cfg(target_os = "linux")]
const UFFD_CONCURRENT_PAUSE_NONE: u8 = 0;
#[cfg(target_os = "linux")]
const UFFD_CONCURRENT_PAUSE_INITIAL: u8 = 1;
#[cfg(target_os = "linux")]
const UFFD_CONCURRENT_PAUSE_FINAL: u8 = 2;

impl<VM: VMBinding> Plan for Compressor<VM> {
    fn constraints(&self) -> &'static PlanConstraints {
        &COMPRESSOR_CONSTRAINTS
    }

    fn collection_required(&self, space_full: bool, _space: Option<SpaceStats<Self::VM>>) -> bool {
        #[cfg(target_os = "linux")]
        if self.is_uffd_concurrent_enabled()
            && self.uffd_concurrent_active.load(Ordering::Acquire)
            && self.common.base.scheduler.work_buckets[WorkBucketStage::Concurrent].is_drained()
        {
            return true;
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
        #[cfg(target_os = "linux")]
        if self.is_uffd_concurrent_enabled()
            && self.current_uffd_concurrent_pause() == UFFD_CONCURRENT_PAUSE_FINAL
        {
            return;
        }
        self.common.prepare(tls, true);
        self.compressor_space.prepare();
    }

    fn release(&mut self, tls: VMWorkerThread) {
        #[cfg(target_os = "linux")]
        if self.is_uffd_concurrent_enabled()
            && self.current_uffd_concurrent_pause() == UFFD_CONCURRENT_PAUSE_INITIAL
        {
            return;
        }
        self.common.release(tls, true);
        self.compressor_space.release();
    }

    fn end_of_gc(&mut self, tls: VMWorkerThread) {
        self.common.end_of_gc(tls);
        #[cfg(target_os = "linux")]
        if self.is_uffd_concurrent_enabled() {
            match self.current_uffd_concurrent_pause() {
                UFFD_CONCURRENT_PAUSE_INITIAL => {
                    // Mutators will resume after this pause while the UFFD epoch is still active.
                    // Match MMTk's concurrent-allocation discipline and tell spaces to allocate
                    // new objects as already-live during this window. This is particularly
                    // important for LOS: otherwise post-resume large-object allocations would be
                    // inserted into the treadmill's alloc_nursery and violate the old STW-only
                    // invariant checked in `LargeObjectSpace::release()`.
                    self.set_uffd_concurrent_allocation_state(true);
                    self.uffd_concurrent_active.store(true, Ordering::Release);
                    self.uffd_final_pause_requested.store(false, Ordering::Release);
                    self.uffd_concurrent_regions_remaining.store(0, Ordering::Release);
                    self.uffd_concurrent_pause
                        .store(UFFD_CONCURRENT_PAUSE_NONE, Ordering::Release);
                }
                UFFD_CONCURRENT_PAUSE_FINAL => {
                    // The split concurrent epoch is complete; restore normal allocation behavior
                    // before mutators resume.
                    self.set_uffd_concurrent_allocation_state(false);
                    self.uffd_concurrent_active.store(false, Ordering::Release);
                    self.uffd_final_pause_requested.store(false, Ordering::Release);
                    self.uffd_concurrent_regions_remaining.store(0, Ordering::Release);
                    self.uffd_concurrent_pause
                        .store(UFFD_CONCURRENT_PAUSE_NONE, Ordering::Release);
                }
                _ => {}
            }
        }
    }

    fn get_allocator_mapping(&self) -> &'static EnumMap<AllocationSemantics, AllocatorSelector> {
        &ALLOCATOR_MAPPING
    }

    fn schedule_collection(&'static self, scheduler: &GCWorkScheduler<VM>) {
        #[cfg(target_os = "linux")]
        if self.is_uffd_concurrent_enabled() && self.uffd_concurrent_active.load(Ordering::Acquire)
        {
            self.uffd_concurrent_pause
                .store(UFFD_CONCURRENT_PAUSE_FINAL, Ordering::Release);
            // ART updates roots before mutator resume in the compaction pause.
            // Our final cleanup pause should only stop/flush mutators and finish
            // the UFFD epoch; it should not enqueue another marking-style root scan.
            scheduler.work_buckets[WorkBucketStage::Unconstrained]
                .add(StopMutators::<CompressorWorkContext<VM>>::new_no_roots());
            scheduler.work_buckets[WorkBucketStage::Compact].add(
                super::gc_work::UffdConcurrentFinish::<VM>::new(
                    self,
                    &self.compressor_space,
                ),
            );
            scheduler.work_buckets[WorkBucketStage::Release]
                .add(Release::<CompressorWorkContext<VM>>::new(self));

            if !*self.base().options.no_reference_types {
                use crate::util::reference_processor::RefEnqueue;
                scheduler.work_buckets[WorkBucketStage::Release].add(RefEnqueue::<VM>::new());
            }

            scheduler.work_buckets[WorkBucketStage::Release].add(VMPostForwarding::<VM>::default());
            return;
        }

        // TODO use schedule_common once it can work with the Compressor
        // The main issue there is that we need to ForwardingProcessEdges
        // in FinalizableForwarding.

        // Stop & scan mutators (mutator scanning can happen before STW)
        scheduler.work_buckets[WorkBucketStage::Unconstrained]
            .add(StopMutators::<CompressorWorkContext<VM>>::new());

        // Prepare global/collectors/mutators
        scheduler.work_buckets[WorkBucketStage::Prepare]
            .add(Prepare::<CompressorWorkContext<VM>>::new(self));

        scheduler.work_buckets[WorkBucketStage::CalculateForwarding].add(GenerateWork::new(
            &self.compressor_space,
            CompressorSpace::<VM>::add_offset_vector_tasks,
        ));

        #[cfg(target_os = "linux")]
        let use_uffd_concurrent = self.is_uffd_concurrent_enabled();
        #[cfg(not(target_os = "linux"))]
        let use_uffd_concurrent = false;
        #[cfg(target_os = "linux")]
        let use_uffd = std::env::var("MMTK_COMPRESSOR_UFFD").map_or(false, |v| v == "1");
        #[cfg(not(target_os = "linux"))]
        let use_uffd = false;

        if use_uffd_concurrent {
            #[cfg(target_os = "linux")]
            self.uffd_concurrent_pause
                .store(UFFD_CONCURRENT_PAUSE_INITIAL, Ordering::Release);
        }

        // Build page-granular metadata whenever any userfaultfd-based Compressor
        // path is enabled.
        if use_uffd_concurrent || use_uffd {
            scheduler.work_buckets[WorkBucketStage::SecondRoots].add(GenerateWork::new(
                &self.compressor_space,
                CompressorSpace::<VM>::build_all_page_metadata,
            ));
        }

        // scan roots to update their references
        scheduler.work_buckets[WorkBucketStage::SecondRoots].add(UpdateReferences::<VM>::new());

        if use_uffd_concurrent {
            #[cfg(target_os = "linux")]
            scheduler.work_buckets[WorkBucketStage::Compact].add(
                super::gc_work::UffdConcurrentSetup::<VM>::new(
                    self,
                    &self.compressor_space,
                    &self.common.los,
                ),
            );
        } else if use_uffd {
            #[cfg(target_os = "linux")]
            {
                scheduler.work_buckets[WorkBucketStage::Compact].add(
                    super::gc_work::UffdCompact::<VM>::new(
                        &self.compressor_space,
                        &self.common.los,
                    ),
                );
            }
        } else {
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

            if !use_uffd_concurrent {
                use crate::util::reference_processor::RefEnqueue;
                scheduler.work_buckets[WorkBucketStage::Release].add(RefEnqueue::<VM>::new());
            }
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

        if !use_uffd_concurrent {
            scheduler.work_buckets[WorkBucketStage::Release].add(VMPostForwarding::<VM>::default());
        }

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
        #[cfg(target_os = "linux")]
        if self.is_uffd_concurrent_enabled() && self.uffd_concurrent_active.load(Ordering::Acquire)
        {
            return true;
        }
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
            #[cfg(target_os = "linux")]
            uffd_concurrent_active: AtomicBool::new(false),
            #[cfg(target_os = "linux")]
            uffd_concurrent_pause: AtomicU8::new(UFFD_CONCURRENT_PAUSE_NONE),
            #[cfg(target_os = "linux")]
            uffd_final_pause_requested: AtomicBool::new(false),
            #[cfg(target_os = "linux")]
            uffd_concurrent_regions_remaining: AtomicUsize::new(0),
            #[cfg(target_os = "linux")]
            uffd_context: Mutex::new(None),
            #[cfg(target_os = "linux")]
            uffd_handler_thread: Mutex::new(None),
        };

        res.verify_side_metadata_sanity();

        res
    }

    #[cfg(target_os = "linux")]
    pub fn is_uffd_concurrent_enabled(&self) -> bool {
        std::env::var("MMTK_COMPRESSOR_UFFD_CONCURRENT").map_or(false, |v| v == "1")
    }

    #[cfg(target_os = "linux")]
    pub fn current_uffd_concurrent_pause(&self) -> u8 {
        self.uffd_concurrent_pause.load(Ordering::Acquire)
    }

    #[cfg(target_os = "linux")]
    fn set_uffd_concurrent_allocation_state(&self, active: bool) {
        use crate::plan::global::HasSpaces;

        self.for_each_space(&mut |space: &dyn Space<VM>| {
            space.set_allocate_as_live(active);
        });
    }

    #[cfg(target_os = "linux")]
    pub fn set_uffd_context(
        &self,
        ctx: Arc<crate::policy::compressor::uffd::UffdContext<VM>>,
        handler: JoinHandle<()>,
    ) {
        *self.uffd_context.lock().unwrap() = Some(ctx);
        *self.uffd_handler_thread.lock().unwrap() = Some(handler);
    }

    #[cfg(target_os = "linux")]
    pub fn uffd_context(&self) -> Option<Arc<crate::policy::compressor::uffd::UffdContext<VM>>> {
        self.uffd_context.lock().unwrap().clone()
    }

    #[cfg(target_os = "linux")]
    pub fn take_uffd_context(
        &self,
    ) -> Option<Arc<crate::policy::compressor::uffd::UffdContext<VM>>> {
        self.uffd_context.lock().unwrap().take()
    }

    #[cfg(target_os = "linux")]
    pub fn take_uffd_handler_thread(&self) -> Option<JoinHandle<()>> {
        self.uffd_handler_thread.lock().unwrap().take()
    }

    #[cfg(target_os = "linux")]
    pub fn set_uffd_concurrent_regions_remaining(&self, regions: usize) {
        self.uffd_concurrent_regions_remaining
            .store(regions, Ordering::Release);
    }

    #[cfg(target_os = "linux")]
    pub fn on_uffd_concurrent_region_processed(&self) -> bool {
        self.uffd_concurrent_regions_remaining
            .fetch_sub(1, Ordering::AcqRel)
            == 1
    }

    #[cfg(target_os = "linux")]
    pub fn request_uffd_final_pause_if_needed(&self) -> bool {
        self.uffd_final_pause_requested
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
}
