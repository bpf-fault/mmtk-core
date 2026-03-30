use super::gc_work::AfterCompact;
#[cfg(feature = "uffd")]
use super::gc_work::CaptureBlackAllocations;
use super::gc_work::CompressorWorkContext;
#[cfg(feature = "uffd")]
use super::gc_work::ConcurrentCompressorGCWorkContext;
use super::gc_work::{ForwardingProcessEdges, GenerateWork, MarkingProcessEdges, UpdateReferences};
#[cfg(feature = "uffd")]
use super::gc_work::{ValidateMutatorRoots, ValidateVmSpecificRoots};
use crate::plan::barriers::BarrierSelector;
use crate::plan::compressor::mutator::ALLOCATOR_MAPPING;
#[cfg(feature = "uffd")]
use crate::plan::concurrent::concurrent_marking_work::ProcessRootSlots;
#[cfg(feature = "uffd")]
use crate::plan::concurrent::global::ConcurrentPlan;
#[cfg(feature = "uffd")]
use crate::plan::concurrent::Pause;
use crate::plan::global::CreateGeneralPlanArgs;
use crate::plan::global::CreateSpecificPlanArgs;
use crate::plan::global::{BasePlan, CommonPlan};
use crate::plan::plan_constraints::MAX_NON_LOS_ALLOC_BYTES_COPYING_PLAN;
use crate::plan::{AllocationSemantics, Plan, PlanConstraints};
use crate::policy::compressor::CompressorSpace;
#[cfg(feature = "uffd")]
use crate::policy::compressor::TRACE_KIND_MARK;
use crate::policy::space::Space;
use crate::scheduler::gc_work::*;
use crate::scheduler::{GCWorkScheduler, WorkBucketStage};
use crate::util::alloc::allocators::AllocatorSelector;
use crate::util::heap::gc_trigger::SpaceStats;
#[allow(unused_imports)]
use crate::util::heap::VMRequest;
use crate::util::metadata::extract_side_metadata;
#[cfg(feature = "uffd")]
use crate::util::metadata::log_bit::UnlogBitsOperation;
use crate::util::metadata::side_metadata::SideMetadataContext;
use crate::util::opaque_pointer::*;
use crate::util::ObjectReference;
#[cfg(feature = "uffd")]
use crate::vm::ActivePlan;
#[cfg(feature = "uffd")]
use crate::vm::ObjectModel;
use crate::vm::VMBinding;
#[cfg(feature = "uffd")]
use atomic::{Atomic, Ordering as AtomicOrdering};
use enum_map::EnumMap;
use mmtk_macros::{HasSpaces, PlanTraceObject};
#[cfg(feature = "uffd")]
use std::sync::atomic::{AtomicBool, Ordering};

/// [`Compressor`] implements a stop-the-world and parallel implementation of
/// the Compressor, as described in Kermany and Petrank,
/// [The Compressor: concurrent, incremental, and parallel compaction](https://dl.acm.org/doi/10.1145/1133255.1134023).
///
/// With `feature = "uffd"` (Linux only), the Compressor follows Android ART's
/// high-level structure more closely:
///
/// 1. `InitialMark` pause seeds concurrent SATB marking.
/// 2. `FinalMark` pause captures black allocations, computes forwarding, and
///    sets up the UFFD epoch.
/// 3. The background compactor resolves pages concurrently after mutator resume.
#[derive(HasSpaces, PlanTraceObject)]
pub struct Compressor<VM: VMBinding> {
    #[parent]
    pub common: CommonPlan<VM>,
    #[space]
    pub compressor_space: CompressorSpace<VM>,
    #[cfg(feature = "uffd")]
    current_pause: Atomic<Option<Pause>>,
    #[cfg(feature = "uffd")]
    previous_pause: Atomic<Option<Pause>>,
    #[cfg(feature = "uffd")]
    concurrent_marking_active: AtomicBool,
    #[cfg(feature = "uffd")]
    should_do_full_gc: AtomicBool,
    #[cfg(feature = "uffd")]
    uffd_compaction_active: AtomicBool,
    #[cfg(feature = "uffd")]
    uffd_epoch_ready: AtomicBool,
}

/// The plan constraints for the Compressor plan.
pub const COMPRESSOR_CONSTRAINTS: PlanConstraints = PlanConstraints {
    max_non_los_default_alloc_bytes: MAX_NON_LOS_ALLOC_BYTES_COPYING_PLAN,
    moves_objects: true,
    needs_forward_after_liveness: true,
    needs_log_bit: cfg!(feature = "uffd"),
    barrier: if cfg!(feature = "uffd") {
        BarrierSelector::SATBBarrier
    } else {
        BarrierSelector::NoBarrier
    },
    needs_prepare_mutator: cfg!(feature = "uffd"),
    ..PlanConstraints::default()
};

impl<VM: VMBinding> Plan for Compressor<VM> {
    fn constraints(&self) -> &'static PlanConstraints {
        &COMPRESSOR_CONSTRAINTS
    }

    fn collection_required(&self, space_full: bool, _space: Option<SpaceStats<Self::VM>>) -> bool {
        #[cfg(feature = "uffd")]
        {
            if self.uffd_compaction_active.load(Ordering::Acquire) {
                return false;
            }

            if self.concurrent_marking_active.load(Ordering::Acquire)
                && self.common.base.scheduler.work_buckets[WorkBucketStage::Concurrent].is_drained()
            {
                return true;
            }

            if self.base().collection_required(self, space_full) {
                self.should_do_full_gc.store(true, Ordering::Release);
                return true;
            }

            if !self.concurrent_marking_active.load(Ordering::Acquire) {
                let threshold = self.get_total_pages() >> 1;
                let used_pages_after_last_gc =
                    self.common.base.global_state.get_used_pages_after_last_gc();
                let used_pages_now = self.get_used_pages();
                let allocated_since_last_gc =
                    used_pages_now.saturating_sub(used_pages_after_last_gc);
                if allocated_since_last_gc > threshold {
                    debug_assert!(
                        self.common.base.scheduler.work_buckets[WorkBucketStage::Concurrent]
                            .is_empty(),
                        "Concurrent bucket should be empty before InitialMark"
                    );
                    return true;
                }
            }

            return false;
        }

        #[cfg(not(feature = "uffd"))]
        {
            self.base().collection_required(self, space_full)
        }
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

    fn schedule_collection(&'static self, scheduler: &GCWorkScheduler<VM>) {
        #[cfg(feature = "uffd")]
        {
            self.uffd_epoch_ready.store(false, Ordering::Release);
            let pause = if self.concurrent_marking_in_progress() {
                Pause::FinalMark
            } else if self.should_do_full_gc.load(Ordering::Acquire) {
                Pause::Full
            } else {
                Pause::InitialMark
            };
            self.current_pause
                .store(Some(pause), AtomicOrdering::SeqCst);

            match pause {
                Pause::InitialMark => self.schedule_initial_mark(scheduler),
                Pause::FinalMark => self.schedule_final_mark(scheduler),
                Pause::Full => self.schedule_full_gc(scheduler),
            }
        }

        #[cfg(not(feature = "uffd"))]
        {
            self.schedule_stw_full_gc(scheduler);
        }
    }

    fn get_allocator_mapping(&self) -> &'static EnumMap<AllocationSemantics, AllocatorSelector> {
        &ALLOCATOR_MAPPING
    }

    fn post_alloc_initialized(&self, object: ObjectReference) {
        #[cfg(feature = "uffd")]
        self.compressor_space.record_black_allocation(object);
        #[cfg(not(feature = "uffd"))]
        let _ = object;
    }

    fn notify_mutators_paused(&self, _scheduler: &GCWorkScheduler<VM>) {
        #[cfg(feature = "uffd")]
        {
            let pause = self.current_pause().unwrap();
            match pause {
                Pause::InitialMark => {
                    debug_assert!(!self.concurrent_marking_in_progress());
                }
                Pause::FinalMark => {
                    debug_assert!(self.concurrent_marking_in_progress());
                    self.set_concurrent_marking_state(false);
                    info!("FinalMark: flushing mutator SATB buffers");
                    for mutator in <VM as VMBinding>::VMActivePlan::mutators() {
                        mutator.barrier.flush();
                    }
                    info!("FinalMark: mutator SATB buffers flushed");
                }
                Pause::Full => {
                    self.set_concurrent_marking_state(false);
                    self.set_ref_closure_buckets_enabled(true);
                    self.compressor_space.clear_black_allocation_snapshot();
                }
            }
            info!("{:?} start", pause);
        }
    }

    fn prepare(&mut self, tls: VMWorkerThread) {
        #[cfg(feature = "uffd")]
        match self.current_pause() {
            Some(Pause::InitialMark) => {
                self.common.prepare(tls, true);
                self.compressor_space.prepare();
                self.compressor_space.snapshot_black_allocation_cursors();
                self.compressor_space.set_side_log_bits();
                self.common
                    .schedule_unlog_bits_op(UnlogBitsOperation::BulkSet);
                self.set_concurrent_marking_state(true);
            }
            Some(Pause::Full) => {
                self.common.prepare(tls, true);
                self.compressor_space.prepare();
            }
            Some(Pause::FinalMark) | None => {}
        }

        #[cfg(not(feature = "uffd"))]
        {
            self.common.prepare(tls, true);
            self.compressor_space.prepare();
        }
    }

    fn release(&mut self, tls: VMWorkerThread) {
        #[cfg(feature = "uffd")]
        match self.current_pause() {
            Some(Pause::InitialMark) => {}
            Some(Pause::FinalMark) => {
                self.compressor_space.clear_side_log_bits();
                self.common
                    .schedule_unlog_bits_op(UnlogBitsOperation::BulkClear);
                self.common.release(tls, true);
            }
            Some(Pause::Full) | None => {
                self.common.release(tls, true);
                self.compressor_space.release();
            }
        }

        #[cfg(not(feature = "uffd"))]
        {
            self.common.release(tls, true);
            self.compressor_space.release();
        }
    }

    fn end_of_gc(&mut self, tls: VMWorkerThread) {
        self.common.end_of_gc(tls);

        #[cfg(feature = "uffd")]
        {
            let pause = self.current_pause().unwrap();
            match pause {
                Pause::InitialMark => {}
                Pause::FinalMark => {
                    let uffd_epoch_ready = self.uffd_epoch_ready.swap(false, Ordering::AcqRel);
                    self.compressor_space.clear_black_allocation_snapshot();
                    if uffd_epoch_ready {
                        self.uffd_compaction_active.store(true, Ordering::Release);
                    } else {
                        // No Compressor regions needed page-fault-driven compaction.
                        self.compressor_space.release();
                    }
                }
                Pause::Full => {
                    self.should_do_full_gc.store(false, Ordering::Release);
                    self.compressor_space.clear_black_allocation_snapshot();
                }
            }
            self.previous_pause
                .store(Some(pause), AtomicOrdering::SeqCst);
            self.current_pause.store(None, AtomicOrdering::SeqCst);
            info!("{:?} end", pause);
        }
    }

    #[cfg(feature = "uffd")]
    fn concurrent(&self) -> Option<&dyn ConcurrentPlan<VM = VM>> {
        Some(self)
    }

    fn current_gc_may_move_object(&self) -> bool {
        true
    }

    fn get_used_pages(&self) -> usize {
        self.compressor_space.reserved_pages() + self.common.get_used_pages()
    }
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> ConcurrentPlan for Compressor<VM> {
    fn concurrent_work_in_progress(&self) -> bool {
        self.concurrent_marking_active.load(Ordering::Acquire)
    }

    fn current_pause(&self) -> Option<Pause> {
        self.current_pause.load(AtomicOrdering::SeqCst)
    }
}

impl<VM: VMBinding> Compressor<VM> {
    pub fn new(args: CreateGeneralPlanArgs<VM>) -> Self {
        #[cfg(feature = "uffd")]
        let global_specs = extract_side_metadata(&[*VM::VMObjectModel::GLOBAL_LOG_BIT_SPEC]);
        #[cfg(not(feature = "uffd"))]
        let global_specs = extract_side_metadata(&[]);

        let mut plan_args = CreateSpecificPlanArgs {
            global_args: args,
            constraints: &COMPRESSOR_CONSTRAINTS,
            global_side_metadata_specs: SideMetadataContext::new_global_specs(&global_specs),
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
            current_pause: Atomic::new(None),
            #[cfg(feature = "uffd")]
            previous_pause: Atomic::new(None),
            #[cfg(feature = "uffd")]
            concurrent_marking_active: AtomicBool::new(false),
            #[cfg(feature = "uffd")]
            should_do_full_gc: AtomicBool::new(false),
            #[cfg(feature = "uffd")]
            uffd_compaction_active: AtomicBool::new(false),
            #[cfg(feature = "uffd")]
            uffd_epoch_ready: AtomicBool::new(false),
        };

        res.verify_side_metadata_sanity();
        res
    }

    fn schedule_stw_full_gc(&'static self, scheduler: &GCWorkScheduler<VM>) {
        scheduler.work_buckets[WorkBucketStage::Unconstrained]
            .add(StopMutators::<CompressorWorkContext<VM>>::new());
        scheduler.work_buckets[WorkBucketStage::Prepare]
            .add(Prepare::<CompressorWorkContext<VM>>::new(self));
        scheduler.work_buckets[WorkBucketStage::CalculateForwarding].add(GenerateWork::new(
            &self.compressor_space,
            CompressorSpace::<VM>::add_offset_vector_tasks,
        ));
        scheduler.work_buckets[WorkBucketStage::SecondRoots].add(UpdateReferences::<VM>::new());
        scheduler.work_buckets[WorkBucketStage::Compact].add(GenerateWork::new(
            &self.compressor_space,
            CompressorSpace::<VM>::add_compact_tasks,
        ));
        scheduler.work_buckets[WorkBucketStage::Compact].set_sentinel(Box::new(
            AfterCompact::<VM>::new(&self.compressor_space, &self.common.los),
        ));
        scheduler.work_buckets[WorkBucketStage::Release]
            .add(Release::<CompressorWorkContext<VM>>::new(self));
        self.schedule_reference_work(scheduler);
    }

    fn schedule_reference_work(&'static self, scheduler: &GCWorkScheduler<VM>) {
        if !*self.base().options.no_reference_types {
            use crate::util::reference_processor::{
                PhantomRefProcessing, RefEnqueue, RefForwarding, SoftRefProcessing,
                WeakRefProcessing,
            };
            scheduler.work_buckets[WorkBucketStage::SoftRefClosure]
                .add(SoftRefProcessing::<MarkingProcessEdges<VM>>::new());
            scheduler.work_buckets[WorkBucketStage::WeakRefClosure]
                .add(WeakRefProcessing::<VM>::new());
            scheduler.work_buckets[WorkBucketStage::PhantomRefClosure]
                .add(PhantomRefProcessing::<VM>::new());
            scheduler.work_buckets[WorkBucketStage::RefForwarding]
                .add(RefForwarding::<ForwardingProcessEdges<VM>>::new());
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

    #[cfg(feature = "uffd")]
    fn set_allocate_as_live(&self, active: bool) {
        use crate::plan::global::HasSpaces;
        self.for_each_space(&mut |space: &dyn Space<VM>| {
            space.set_allocate_as_live(active);
        });
    }

    #[cfg(feature = "uffd")]
    fn set_concurrent_marking_state(&self, active: bool) {
        self.set_allocate_as_live(active);
        self.concurrent_marking_active
            .store(active, Ordering::SeqCst);
    }

    #[cfg(feature = "uffd")]
    pub fn is_concurrent_marking_active(&self) -> bool {
        self.concurrent_marking_active.load(Ordering::SeqCst)
    }

    #[cfg(feature = "uffd")]
    fn concurrent_marking_in_progress(&self) -> bool {
        self.concurrent_marking_active.load(Ordering::Acquire)
    }

    #[cfg(feature = "uffd")]
    fn set_ref_closure_buckets_enabled(&self, do_closure: bool) {
        let scheduler = &self.common.base.scheduler;
        scheduler.work_buckets[WorkBucketStage::VMRefClosure].set_enabled(do_closure);
        scheduler.work_buckets[WorkBucketStage::WeakRefClosure].set_enabled(do_closure);
        scheduler.work_buckets[WorkBucketStage::FinalRefClosure].set_enabled(do_closure);
        scheduler.work_buckets[WorkBucketStage::SoftRefClosure].set_enabled(do_closure);
        scheduler.work_buckets[WorkBucketStage::PhantomRefClosure].set_enabled(do_closure);
    }

    #[cfg(feature = "uffd")]
    fn previous_pause(&self) -> Option<Pause> {
        self.previous_pause.load(AtomicOrdering::SeqCst)
    }

    #[cfg(feature = "uffd")]
    fn schedule_initial_mark(&'static self, scheduler: &GCWorkScheduler<VM>) {
        self.set_ref_closure_buckets_enabled(false);
        scheduler.work_buckets[WorkBucketStage::Unconstrained].add(StopMutators::<
            ConcurrentCompressorGCWorkContext<ProcessRootSlots<VM, Self, TRACE_KIND_MARK>>,
        >::new());
        scheduler.work_buckets[WorkBucketStage::Prepare].add(Prepare::<
            ConcurrentCompressorGCWorkContext<UnsupportedProcessEdges<VM>>,
        >::new(self));
    }

    #[cfg(feature = "uffd")]
    fn schedule_final_mark(&'static self, scheduler: &GCWorkScheduler<VM>) {
        self.set_ref_closure_buckets_enabled(true);
        scheduler.work_buckets[WorkBucketStage::Unconstrained]
            .add(StopMutators::<CompressorWorkContext<VM>>::new());
        scheduler.work_buckets[WorkBucketStage::Closure].add(CaptureBlackAllocations::<VM>::new(
            self,
            &self.compressor_space,
        ));
        scheduler.work_buckets[WorkBucketStage::CalculateForwarding].add(GenerateWork::new(
            &self.compressor_space,
            CompressorSpace::<VM>::add_offset_vector_tasks,
        ));
        scheduler.work_buckets[WorkBucketStage::SecondRoots].add(GenerateWork::new(
            &self.compressor_space,
            CompressorSpace::<VM>::build_all_page_metadata,
        ));
        scheduler.work_buckets[WorkBucketStage::SecondRoots].add(UpdateReferences::<VM>::new());
        if std::env::var_os("MMTK_VALIDATE_MUTATOR_ROOTS").is_some() {
            scheduler.work_buckets[WorkBucketStage::Compact]
                .add(ValidateMutatorRoots::<VM>::new(&self.compressor_space));
        }
        if std::env::var_os("MMTK_VALIDATE_VM_ROOTS").is_some() {
            scheduler.work_buckets[WorkBucketStage::Compact]
                .add(ValidateVmSpecificRoots::<VM>::new(&self.compressor_space));
        }
        scheduler.work_buckets[WorkBucketStage::Compact].add(
            super::gc_work::UffdConcurrentSetup::<VM>::new(
                self,
                &self.compressor_space,
                &self.common.los,
            ),
        );
        scheduler.work_buckets[WorkBucketStage::Release]
            .add(Release::<CompressorWorkContext<VM>>::new(self));
        self.schedule_reference_work(scheduler);
    }

    #[cfg(feature = "uffd")]
    fn schedule_full_gc(&'static self, scheduler: &GCWorkScheduler<VM>) {
        self.set_ref_closure_buckets_enabled(true);
        self.schedule_stw_full_gc(scheduler);
    }

    #[cfg(feature = "uffd")]
    fn current_pause(&self) -> Option<Pause> {
        self.current_pause.load(AtomicOrdering::SeqCst)
    }

    #[cfg(feature = "uffd")]
    pub fn mark_uffd_epoch_ready(&self) {
        self.uffd_epoch_ready.store(true, Ordering::Release);
    }

    #[cfg(feature = "uffd")]
    pub fn is_uffd_compaction_active(&self) -> bool {
        self.uffd_compaction_active.load(Ordering::Acquire)
    }

    #[cfg(feature = "uffd")]
    pub fn finish_uffd_epoch(&self) {
        self.compressor_space.reset_allocator_after_compaction();
        self.compressor_space.release();
        self.set_allocate_as_live(false);
        self.uffd_compaction_active.store(false, Ordering::Release);
    }
}
