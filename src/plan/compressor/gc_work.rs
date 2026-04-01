use super::global::Compressor;
use crate::plan::Plan;
use crate::policy::compressor::{CompressorSpace, TRACE_KIND_FORWARD_ROOT, TRACE_KIND_MARK};
use crate::policy::largeobjectspace::LargeObjectSpace;
#[cfg(feature = "uffd")]
use crate::policy::space::Space;
use crate::scheduler::gc_work::PlanProcessEdges;
use crate::scheduler::gc_work::*;
#[cfg(feature = "uffd")]
use crate::scheduler::ProcessEdgesWork;
use crate::scheduler::{GCWork, GCWorker, WorkBucketStage};
#[cfg(feature = "uffd")]
use crate::util::{linear_scan::Region, Address, ObjectReference};
use crate::vm::slot::Slot;
use crate::vm::{ActivePlan, Scanning, VMBinding};
use crate::MMTK;
use std::marker::{PhantomData, Send};
use std::time::Instant;
#[cfg(feature = "uffd")]
use std::{collections::HashSet, sync::Arc};

fn compressor_perf_trace_enabled() -> bool {
    perf_trace_enabled()
}

/// Generate more packets by calling a method on [`CompressorSpace`].
pub struct GenerateWork<VM: VMBinding, F: Fn(&'static CompressorSpace<VM>) + Send + 'static> {
    compressor_space: &'static CompressorSpace<VM>,
    f: F,
    timing_label: Option<&'static str>,
}

impl<VM: VMBinding, F: Fn(&'static CompressorSpace<VM>) + Send + 'static> GCWork<VM>
    for GenerateWork<VM, F>
{
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        if let Some(label) = self.timing_label {
            start_perf_timing(label);
        }
        (self.f)(self.compressor_space);
    }
}

impl<VM: VMBinding, F: Fn(&'static CompressorSpace<VM>) + Send + 'static> GenerateWork<VM, F> {
    pub fn new(compressor_space: &'static CompressorSpace<VM>, f: F) -> Self {
        Self {
            compressor_space,
            f,
            timing_label: None,
        }
    }

    pub fn new_timed(
        compressor_space: &'static CompressorSpace<VM>,
        f: F,
        timing_label: &'static str,
    ) -> Self {
        Self {
            compressor_space,
            f,
            timing_label: Some(timing_label),
        }
    }
}

pub struct LogBucketTiming<VM: VMBinding> {
    label: &'static str,
    phantom: PhantomData<VM>,
}

impl<VM: VMBinding> LogBucketTiming<VM> {
    pub fn new(label: &'static str) -> Self {
        Self {
            label,
            phantom: PhantomData,
        }
    }
}

impl<VM: VMBinding> GCWork<VM> for LogBucketTiming<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        if let Some(elapsed) = finish_perf_timing(self.label) {
            info!("{} completed in {} ms", self.label, format_perf_ms(elapsed));
        }
    }
}

pub struct LogBucketTimings<VM: VMBinding> {
    labels: &'static [&'static str],
    phantom: PhantomData<VM>,
}

impl<VM: VMBinding> LogBucketTimings<VM> {
    pub fn new(labels: &'static [&'static str]) -> Self {
        Self {
            labels,
            phantom: PhantomData,
        }
    }
}

impl<VM: VMBinding> GCWork<VM> for LogBucketTimings<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        for &label in self.labels {
            if let Some(elapsed) = finish_perf_timing(label) {
                info!("{} completed in {} ms", label, format_perf_ms(elapsed));
            }
        }
    }
}

#[cfg(feature = "uffd")]
pub struct FinalizeInitialMarkPrepare<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
    timing_labels: &'static [&'static str],
}

#[cfg(feature = "uffd")]
pub struct FinalizeCompactionPrepare<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
    timing_labels: &'static [&'static str],
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> FinalizeInitialMarkPrepare<VM> {
    pub fn new(
        compressor_space: &'static CompressorSpace<VM>,
        timing_labels: &'static [&'static str],
    ) -> Self {
        Self {
            compressor_space,
            timing_labels,
        }
    }
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> FinalizeCompactionPrepare<VM> {
    pub fn new(
        compressor_space: &'static CompressorSpace<VM>,
        timing_labels: &'static [&'static str],
    ) -> Self {
        Self {
            compressor_space,
            timing_labels,
        }
    }
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> GCWork<VM> for FinalizeInitialMarkPrepare<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        let start = Instant::now();
        self.compressor_space
            .snapshot_concurrent_compaction_prepare_regions();
        self.compressor_space.snapshot_black_allocation_cursors();
        self.compressor_space.seal_regions_for_concurrent_uffd();
        if compressor_perf_trace_enabled() {
            info!(
                "Compressor InitialMark: finalized concurrent-prepare snapshot in {} ms (regions={})",
                format_perf_ms(start.elapsed()),
                self.compressor_space.num_regions(),
            );
        }
        for &label in self.timing_labels {
            if let Some(elapsed) = finish_perf_timing(label) {
                info!("{} completed in {} ms", label, format_perf_ms(elapsed));
            }
        }
    }
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> GCWork<VM> for FinalizeCompactionPrepare<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        let start = Instant::now();
        let objects = self
            .compressor_space
            .finalize_inter_pause_black_allocations_for_compaction();
        // The inter-pause black-allocation catch-up above marks new objects in
        // prefix regions, changing the mark set that the compaction layout depends
        // on.  Any compaction summaries cached during the concurrent-prepare epoch
        // are now stale because they were built from the FinalMark-era marks only.
        // Invalidate them so that the Compact bucket's CacheRegionCompactionSummary
        // tasks rebuild summaries from the current (post-catch-up) mark state.
        if !objects.is_empty() {
            self.compressor_space.invalidate_compaction_summary_cache();
        }
        // NOTE: We intentionally do NOT schedule inter-pause allocation reference
        // update tasks here.  Those tasks would forward references in-place in
        // compactor-space heap objects BEFORE mremap, causing the shadow to contain
        // already-forwarded references.  The mark-gated fixup in
        // `build_page_from_shadow()` would then double-forward them if the
        // compacted address coincides with another marked object's original address.
        // Instead, page materialization handles all reference forwarding from the
        // shadow, which always contains the original (pre-forwarding) reference
        // values.  Non-moving space updates (LOS, immortal, etc.) are handled
        // separately in UffdConcurrentSetup and are not affected.
        if compressor_perf_trace_enabled() {
            info!(
                "Compressor Compaction: finalized moving-space black allocations in {} ms (objects={})",
                format_perf_ms(start.elapsed()),
                objects.len(),
            );
        }
        for &label in self.timing_labels {
            if let Some(elapsed) = finish_perf_timing(label) {
                info!("{} completed in {} ms", label, format_perf_ms(elapsed));
            }
        }
    }
}

pub struct TimedPrepare<C: crate::scheduler::GCWorkContext> {
    pub plan: *const C::PlanType,
    timing_label: &'static str,
}

unsafe impl<C: crate::scheduler::GCWorkContext> Send for TimedPrepare<C> {}

impl<C: crate::scheduler::GCWorkContext> TimedPrepare<C> {
    pub fn new(plan: *const C::PlanType, timing_label: &'static str) -> Self {
        Self { plan, timing_label }
    }
}

impl<C: crate::scheduler::GCWorkContext> GCWork<C::VM> for TimedPrepare<C> {
    fn do_work(&mut self, worker: &mut GCWorker<C::VM>, mmtk: &'static MMTK<C::VM>) {
        let start = Instant::now();

        trace!("Prepare Global");
        let plan_mut: &mut C::PlanType = unsafe { &mut *(self.plan as *const _ as *mut _) };
        plan_mut.prepare(worker.tls);

        if plan_mut.constraints().needs_prepare_mutator {
            let prepare_mutator_packets = <C::VM as VMBinding>::VMActivePlan::mutators()
                .map(|mutator| Box::new(PrepareMutator::<C::VM>::new(mutator)) as _)
                .collect::<Vec<_>>();
            debug_assert_eq!(
                prepare_mutator_packets.len(),
                <C::VM as VMBinding>::VMActivePlan::number_of_mutators()
            );
            mmtk.scheduler.work_buckets[WorkBucketStage::Prepare].bulk_add(prepare_mutator_packets);
        }

        for w in &mmtk.scheduler.worker_group.workers_shared {
            let result = w.designated_work.push(Box::new(PrepareCollector));
            debug_assert!(result.is_ok());
        }

        if perf_trace_enabled() {
            info!(
                "{} completed in {} ms",
                self.timing_label,
                format_perf_ms(start.elapsed())
            );
        }
    }
}

#[cfg(feature = "uffd")]
pub struct CaptureBlackAllocations<VM: VMBinding> {
    plan: &'static Compressor<VM>,
    compressor_space: &'static CompressorSpace<VM>,
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> GCWork<VM> for CaptureBlackAllocations<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        let start = std::time::Instant::now();
        if compressor_perf_trace_enabled() {
            info!("Compressor FinalMark: capturing black allocations");
        }
        let objects = self.compressor_space.take_black_allocations();
        if compressor_perf_trace_enabled() && !objects.is_empty() {
            let mut compressor_refs = 0usize;
            let mut mapped_compressor_refs = 0usize;
            let mut marked_compressor_refs = 0usize;
            let mut first_bad_ref = None;
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
                        if self.compressor_space.in_space(referent) {
                            compressor_refs += 1;
                            if referent.to_raw_address().is_mapped() {
                                mapped_compressor_refs += 1;
                            } else if first_bad_ref.is_none() {
                                first_bad_ref = Some((object, referent));
                            }
                            if CompressorSpace::<VM>::is_marked(referent) {
                                marked_compressor_refs += 1;
                            }
                        }
                    },
                );
                if first_bad_ref.is_some() {
                    break;
                }
            }
            info!(
                "Compressor FinalMark: black-allocation refs total_objects={}, compressor_refs={}, mapped_compressor_refs={}, marked_compressor_refs={}",
                objects.len(),
                compressor_refs,
                mapped_compressor_refs,
                marked_compressor_refs,
            );
            if let Some((object, referent)) = first_bad_ref {
                panic!(
                    "Compressor FinalMark: black allocation {} contains unmapped Compressor ref {} before closure",
                    object,
                    referent,
                );
            }
        }
        if compressor_perf_trace_enabled() {
            info!(
                "Compressor FinalMark: captured {} black-allocated objects before forwarding in {} ms",
                objects.len(),
                format_perf_ms(start.elapsed())
            );
        }
        if !objects.is_empty() {
            mmtk.scheduler.work_buckets[WorkBucketStage::Closure].add(PlanScanObjects::<
                MarkingProcessEdges<VM>,
                Compressor<VM>,
            >::new(
                self.plan,
                objects,
                false,
                WorkBucketStage::Closure,
            ));
        }
    }
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> CaptureBlackAllocations<VM> {
    pub fn new(
        plan: &'static Compressor<VM>,
        compressor_space: &'static CompressorSpace<VM>,
    ) -> Self {
        Self {
            plan,
            compressor_space,
        }
    }
}

/// Create another round of root scanning work packets
/// to update object references.
pub struct UpdateReferences<VM: VMBinding> {
    p: PhantomData<VM>,
    timing_label: Option<&'static str>,
}

unsafe impl<VM: VMBinding> Send for UpdateReferences<VM> {}

impl<VM: VMBinding> GCWork<VM> for UpdateReferences<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        let start = std::time::Instant::now();
        if let Some(label) = self.timing_label {
            start_perf_timing(label);
        }
        // The following needs to be done right before the second round of root scanning
        VM::VMScanning::prepare_for_roots_re_scanning();
        mmtk.state.prepare_for_stack_scanning();
        #[cfg(feature = "extreme_assertions")]
        mmtk.slot_logger.reset();

        let mut mutator_count = 0usize;
        for mutator in VM::VMActivePlan::mutators() {
            mutator_count += 1;
            mmtk.scheduler.work_buckets[WorkBucketStage::SecondRoots].add(ScanMutatorRoots::<
                CompressorForwardingWorkContext<VM>,
            >(mutator));
        }

        mmtk.scheduler.work_buckets[WorkBucketStage::SecondRoots]
            .add(ScanVMSpecificRoots::<CompressorForwardingWorkContext<VM>>::new());
        if compressor_perf_trace_enabled() {
            info!(
                "Compressor FinalMark: queued second-root rescanning for {} mutators in {} ms",
                mutator_count,
                format_perf_ms(start.elapsed())
            );
        }
    }
}

impl<VM: VMBinding> UpdateReferences<VM> {
    pub fn new() -> Self {
        Self {
            p: PhantomData,
            timing_label: None,
        }
    }

    pub fn new_timed(timing_label: &'static str) -> Self {
        Self {
            p: PhantomData,
            timing_label: Some(timing_label),
        }
    }
}

#[cfg(feature = "uffd")]
struct ValidateMutatorRootsFactory<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
    valid_objects: Arc<HashSet<ObjectReference>>,
    stage: WorkBucketStage,
    validate_object_contents: bool,
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> Clone for ValidateMutatorRootsFactory<VM> {
    fn clone(&self) -> Self {
        Self {
            compressor_space: self.compressor_space,
            valid_objects: self.valid_objects.clone(),
            stage: self.stage,
            validate_object_contents: self.validate_object_contents,
        }
    }
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> ValidateMutatorRootsFactory<VM> {
    fn new(
        compressor_space: &'static CompressorSpace<VM>,
        valid_objects: Arc<HashSet<ObjectReference>>,
        stage: WorkBucketStage,
        validate_object_contents: bool,
    ) -> Self {
        Self {
            compressor_space,
            valid_objects,
            stage,
            validate_object_contents,
        }
    }

    fn validate_object_contents(&self, object: ObjectReference) {
        if !self.validate_object_contents || !self.compressor_space.in_space(object) {
            return;
        }
        VM::VMScanning::scan_object(
            crate::util::opaque_pointer::VMWorkerThread(
                crate::util::opaque_pointer::VMThread::UNINITIALIZED,
            ),
            object,
            &mut |slot: VM::VMSlot| {
                let Some(referent) = slot.load() else {
                    return;
                };
                if self.compressor_space.in_space(referent)
                    && !self.valid_objects.contains(&referent)
                {
                    let slot_desc = VM::VMScanning::describe_slot(object, slot)
                        .unwrap_or_else(|| "slot=<unknown>".to_string());
                    panic!(
                        "ValidateMutatorRoots: rooted object {} contains stale Compressor ref {} in {}",
                        object,
                        referent,
                        slot_desc,
                    );
                }
            },
        );
    }

    fn validate_object(&self, object: ObjectReference) {
        if self.compressor_space.in_space(object) && !self.valid_objects.contains(&object) {
            panic!(
                "ValidateMutatorRoots: stale Compressor root/reference {} is not a valid destination object",
                object
            );
        }
        self.validate_object_contents(object);
    }
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> crate::vm::RootsWorkFactory<VM::VMSlot> for ValidateMutatorRootsFactory<VM> {
    fn roots_work_bucket_stage(&self) -> WorkBucketStage {
        self.stage
    }

    fn process_edges_roots_work_bucket_stage(&self) -> WorkBucketStage {
        self.stage
    }

    fn create_process_roots_work(&mut self, slots: Vec<VM::VMSlot>) {
        for slot in slots {
            if let Some(object) = crate::vm::slot::Slot::load(&slot) {
                self.validate_object(object);
            }
        }
    }

    fn create_process_pinning_roots_work(&mut self, nodes: Vec<ObjectReference>) {
        for object in nodes {
            self.validate_object(object);
        }
    }

    fn create_process_tpinning_roots_work(&mut self, nodes: Vec<ObjectReference>) {
        for object in nodes {
            self.validate_object(object);
        }
    }
}

#[cfg(feature = "uffd")]
pub struct ValidateMutatorRoots<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
}

#[cfg(feature = "uffd")]
pub struct ValidateVmSpecificRoots<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
}

#[cfg(feature = "uffd")]
pub struct ValidateMappedHeapPreResume<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> GCWork<VM> for ValidateMutatorRoots<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        VM::VMScanning::prepare_for_roots_re_scanning();
        mmtk.state.prepare_for_stack_scanning();

        let valid_objects = Arc::new(self.compressor_space.collect_destination_objects());
        let factory = ValidateMutatorRootsFactory::new(
            self.compressor_space,
            valid_objects.clone(),
            WorkBucketStage::Final,
            std::env::var_os("MMTK_VALIDATE_PRE_RESUME_MUTATOR_ROOT_OBJECTS").is_some(),
        );

        for mutator in VM::VMActivePlan::mutators() {
            VM::VMScanning::scan_roots_in_mutator_thread(worker.tls, mutator, factory.clone());
        }

        info!(
            "ValidateMutatorRoots: validated mutator roots against {} destination objects",
            valid_objects.len()
        );
    }
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> ValidateMutatorRoots<VM> {
    pub fn new(compressor_space: &'static CompressorSpace<VM>) -> Self {
        Self { compressor_space }
    }
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> GCWork<VM> for ValidateVmSpecificRoots<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        let valid_objects = Arc::new(self.compressor_space.collect_destination_objects());
        let factory = ValidateMutatorRootsFactory::new(
            self.compressor_space,
            valid_objects.clone(),
            WorkBucketStage::Final,
            std::env::var_os("MMTK_VALIDATE_PRE_RESUME_VM_ROOT_OBJECTS").is_some(),
        );

        VM::VMScanning::scan_vm_specific_roots(
            crate::util::opaque_pointer::VMWorkerThread(
                crate::util::opaque_pointer::VMThread::UNINITIALIZED,
            ),
            factory,
        );

        info!(
            "ValidateVmSpecificRoots: scheduled VM-specific root validation against {} destination objects",
            valid_objects.len()
        );
    }
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> ValidateVmSpecificRoots<VM> {
    pub fn new(compressor_space: &'static CompressorSpace<VM>) -> Self {
        Self { compressor_space }
    }
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> GCWork<VM> for ValidateMappedHeapPreResume<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        self.compressor_space
            .validate_mapped_compacted_objects()
            .unwrap_or_else(|e| panic!("{}", e));
        info!(
            "ValidatePreResumeMappedHeap: validated mapped compacted objects against cached summaries before mutator resume"
        );
    }
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> ValidateMappedHeapPreResume<VM> {
    pub fn new(compressor_space: &'static CompressorSpace<VM>) -> Self {
        Self { compressor_space }
    }
}

/// Reset the allocator and update references in large object space.
pub struct AfterCompact<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
    los: &'static LargeObjectSpace<VM>,
    timing_label: Option<&'static str>,
}

impl<VM: VMBinding> GCWork<VM> for AfterCompact<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        self.compressor_space.after_compact(worker, self.los);
        if let Some(label) = self.timing_label {
            if let Some(elapsed) = finish_perf_timing(label) {
                info!("{} completed in {} ms", label, format_perf_ms(elapsed));
            }
        }
    }
}

impl<VM: VMBinding> AfterCompact<VM> {
    pub fn new(
        compressor_space: &'static CompressorSpace<VM>,
        los: &'static LargeObjectSpace<VM>,
    ) -> Self {
        Self {
            compressor_space,
            los,
            timing_label: None,
        }
    }

    pub fn new_timed(
        compressor_space: &'static CompressorSpace<VM>,
        los: &'static LargeObjectSpace<VM>,
        timing_label: &'static str,
    ) -> Self {
        Self {
            compressor_space,
            los,
            timing_label: Some(timing_label),
        }
    }
}

// ---------------------------------------------------------------------------
// Concurrent UFFD work packets
// ---------------------------------------------------------------------------

#[cfg(feature = "uffd")]
fn spawn_background_compactor<VM: VMBinding>(
    plan: &'static Compressor<VM>,
    compressor_space: &'static CompressorSpace<VM>,
    ctx: std::sync::Arc<crate::policy::compressor::uffd::UffdContext<VM>>,
    handler: std::thread::JoinHandle<()>,
) {
    std::thread::spawn(move || {
        // ART starts `CompactionPhase()` only after `CompactionPause()` resumes
        // mutators.  We mirror that handoff here.
        while !plan.is_uffd_compaction_active() {
            std::thread::yield_now();
        }

        let validate_region_bytes = std::env::var_os("MMTK_VALIDATE_UFFD_REGION_BYTES").is_some();
        let mut finalized_regions = 0usize;
        for region_idx in (0..compressor_space.compaction_region_count()).rev() {
            // ART's `CompactMovingSpace()` walks pages in reverse order so it can
            // reclaim from-space incrementally.  We do the same at region/page
            // granularity and clean each region as soon as it is fully mapped.
            ctx.process_region_pages_in_reverse(region_idx)
                .unwrap_or_else(|e| {
                    panic!(
                        "Uffd background compaction failed for region {}: {}",
                        region_idx, e
                    )
                });
            if validate_region_bytes {
                let shadow_start =
                    unsafe { Address::from_usize(ctx.shadows[region_idx].shadow_start) };
                compressor_space
                    .validate_region_compaction_from_shadow(region_idx, shadow_start)
                    .unwrap_or_else(|e| {
                        panic!(
                            "Uffd region-byte validation failed for region {}: {}",
                            region_idx, e
                        )
                    });
            }
            if compressor_space.finalize_region_cursor_after_concurrent_uffd(region_idx) {
                finalized_regions += 1;
            }
            ctx.cleanup_region(region_idx).unwrap_or_else(|e| {
                panic!(
                    "Uffd background cleanup failed for region {}: {}",
                    region_idx, e
                )
            });
        }

        if std::env::var_os("MMTK_VALIDATE_UFFD_MARK_WORDS").is_some() {
            compressor_space
                .validate_mapped_mark_words(&ctx.shadows)
                .unwrap_or_else(|e| panic!("{}", e));
        }

        if std::env::var_os("MMTK_VALIDATE_UFFD_MAPPED_HEAP").is_some() {
            compressor_space
                .validate_mapped_compacted_objects()
                .unwrap_or_else(|e| panic!("{}", e));
            info!(
                "ValidateUffdMappedHeap: validated mapped compacted objects across {} regions before finish_uffd_epoch",
                compressor_space.compaction_region_count()
            );
        }

        ctx.signal_done();
        handler.join().unwrap_or_else(|e| {
            error!(
                "Uffd background compactor: handler thread panicked: {:?}",
                e
            );
        });

        let state_counts = ctx.page_state_counts();
        let faults = ctx.faults_handled();
        let pages_resolved = ctx.pages_resolved();
        let gc_pages_processed = ctx.gc_pages_processed();
        let mutator_pages_processed = ctx.mutator_pages_processed();
        ctx.teardown();
        plan.finish_uffd_epoch();

        if compressor_perf_trace_enabled() {
            info!(
                "UffdConcurrentPhase: finalized_regions={}, total_pages_resolved={}, gc_pages_processed={}, mutator_pages_processed={}, faults={}, states=[u={}, p={}, pr={}, pm={}, mp={}, pdm={}, m={}]",
                finalized_regions,
                pages_resolved,
                gc_pages_processed,
                mutator_pages_processed,
                faults,
                state_counts[0],
                state_counts[1],
                state_counts[2],
                state_counts[3],
                state_counts[4],
                state_counts[5],
                state_counts[6]
            );
        }
    });
}

#[cfg(feature = "uffd")]
pub struct UffdConcurrentSetup<VM: VMBinding> {
    plan: &'static Compressor<VM>,
    compressor_space: &'static CompressorSpace<VM>,
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> GCWork<VM> for UffdConcurrentSetup<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        use crate::policy::compressor::uffd::UffdContext;
        use std::sync::Arc;
        let start = std::time::Instant::now();

        let num_regions = self.compressor_space.compaction_region_count();
        if num_regions == 0 {
            return;
        }

        if std::env::var_os("MMTK_VALIDATE_COMPACTION_SOURCE_REFS").is_some() {
            self.compressor_space
                .validate_source_prefix_references()
                .unwrap_or_else(|e| panic!("{}", e));
            info!(
                "ValidateCompactionSourceRefs: validated marked prefix objects across {} regions before UFFD setup",
                num_regions
            );
        }

        if std::env::var_os("MMTK_VALIDATE_COMPACTION_LAYOUT").is_some() {
            self.compressor_space
                .validate_region_compaction_layout_for_prefix(num_regions)
                .unwrap_or_else(|e| panic!("{}", e));
            info!(
                "ValidateCompactionLayout: validated destination layout across {} regions before UFFD setup",
                num_regions
            );
        }

        if std::env::var_os("MMTK_TRACE_COMPRESSOR_REFERENCE_FIELDS").is_some() {
            let (reference_objects, referent_non_null, discovered_total, discovered_non_null) =
                self.compressor_space.debug_count_reference_fields_in_prefix();
            info!(
                "Compressor reference-field snapshot before UFFD setup: reference_objects={}, referent_non_null={}, discovered_total={}, discovered_non_null={}",
                reference_objects,
                referent_non_null,
                discovered_total,
                discovered_non_null,
            );
        }

        let page_metadata_start = std::time::Instant::now();
        if self
            .compressor_space
            .has_cached_region_compaction_summaries_for_prefix(num_regions)
        {
            self.compressor_space
                .materialize_page_metadata_from_cached_summaries_for_prefix(num_regions);
        } else {
            self.compressor_space.build_all_page_metadata();
        }
        let page_metadata_elapsed = page_metadata_start.elapsed();

        let validation_objects = if std::env::var_os("MMTK_VALIDATE_MUTATOR_ROOTS").is_some()
            || std::env::var_os("MMTK_VALIDATE_VM_ROOTS").is_some()
            || std::env::var_os("MMTK_VALIDATE_UFFD_SPACE_REFS").is_some()
            || std::env::var_os("MMTK_VALIDATE_UFFD_REF_TABLES").is_some()
        {
            Some(Arc::new(
                self.compressor_space.collect_destination_objects(),
            ))
        } else {
            None
        };

        if std::env::var_os("MMTK_VALIDATE_MUTATOR_ROOTS").is_some() {
            VM::VMScanning::prepare_for_roots_re_scanning();
            mmtk.state.prepare_for_stack_scanning();
            let valid_objects = validation_objects.as_ref().unwrap().clone();
            let factory = ValidateMutatorRootsFactory::new(
                self.compressor_space,
                valid_objects.clone(),
                WorkBucketStage::Compact,
                false,
            );
            let mut mutator_count = 0usize;
            for mutator in VM::VMActivePlan::mutators() {
                mutator_count += 1;
                VM::VMScanning::scan_roots_in_mutator_thread(worker.tls, mutator, factory.clone());
            }
            info!(
                "ValidateMutatorRoots: validated {} mutators against {} destination objects after page-metadata build",
                mutator_count,
                valid_objects.len()
            );
        }

        if std::env::var_os("MMTK_VALIDATE_VM_ROOTS").is_some() {
            VM::VMScanning::prepare_for_roots_re_scanning();
            let valid_objects = validation_objects.as_ref().unwrap().clone();
            let factory = ValidateMutatorRootsFactory::new(
                self.compressor_space,
                valid_objects.clone(),
                WorkBucketStage::Compact,
                false,
            );
            VM::VMScanning::scan_vm_specific_roots(worker.tls, factory);
            info!(
                "ValidateVmSpecificRoots: validated VM roots against {} destination objects after page-metadata build",
                valid_objects.len()
            );
        }

        if std::env::var_os("MMTK_VALIDATE_UFFD_REF_TABLES").is_some() {
            let valid_objects = validation_objects.as_ref().unwrap();
            let counts = mmtk.reference_processors.validate_refs::<VM>(|object| {
                !self.compressor_space.in_space(object) || valid_objects.contains(&object)
            });
            info!(
                "ValidateUffdRefTables: soft=({},{}) weak=({},{}) phantom=({},{})",
                counts[0].0, counts[0].1, counts[1].0, counts[1].1, counts[2].0, counts[2].1,
            );
        }

        let uffd_new_start = std::time::Instant::now();
        let mut uffd_ctx = UffdContext::new(self.compressor_space)
            .unwrap_or_else(|e| panic!("UffdConcurrentSetup: uffd creation failed: {}", e));
        let uffd_new_elapsed = uffd_new_start.elapsed();

        let register_start = std::time::Instant::now();
        let mut total_registered_pages = 0usize;
        let mut total_source_pages = 0usize;
        let mut moving_regions = 0usize;
        let mut static_regions = 0usize;
        for i in 0..num_regions {
            let (start, _cursor) = self.compressor_space.region_info(i);
            let (shadow_size, register_size) = self
                .compressor_space
                .region_page_metadata(i)
                .map(|meta| {
                    let source_pages = (meta.source_end - meta.region_start)
                        .div_ceil(crate::util::constants::BYTES_IN_PAGE);
                    let compacted_pages = (meta.compacted_end - meta.region_start)
                        .div_ceil(crate::util::constants::BYTES_IN_PAGE);
                    total_source_pages += source_pages;
                    if meta.has_movement {
                        moving_regions += 1;
                    } else {
                        static_regions += 1;
                    }
                    (
                        source_pages * crate::util::constants::BYTES_IN_PAGE,
                        compacted_pages * crate::util::constants::BYTES_IN_PAGE,
                    )
                })
                .unwrap_or((0, 0));
            total_registered_pages += register_size / crate::util::constants::BYTES_IN_PAGE;
            uffd_ctx
                .mremap_and_register(i, start.as_usize(), shadow_size, register_size)
                .unwrap_or_else(|e| {
                    panic!(
                        "UffdConcurrentSetup: mremap/register failed for region {}: {}",
                        i, e
                    )
                });
        }

        let register_elapsed = register_start.elapsed();
        if compressor_perf_trace_enabled() {
            info!(
                "UffdConcurrentSetup: registered {} destination pages across {} regions (source_pages={}, moving_regions={}, static_regions={}, max_region_pages={}, page_metadata={} ms, uffd_new={} ms, register={} ms)",
                total_registered_pages,
                num_regions,
                total_source_pages,
                moving_regions,
                static_regions,
                num_regions
                    * (crate::policy::compressor::forwarding::CompressorRegion::BYTES
                        / crate::util::constants::BYTES_IN_PAGE),
                format_perf_ms(page_metadata_elapsed),
                format_perf_ms(uffd_new_elapsed),
                format_perf_ms(register_elapsed)
            );
        }

        // ART's `CompactionPause()` updates immune / non-moving spaces and roots
        // before mutators resume. Update all MMTk-managed non-moving spaces that can
        // hold references into Compressor, not just LOS.
        let los_update_elapsed = std::time::Duration::ZERO;

        let immortal_update_start = std::time::Instant::now();
        self.compressor_space
            .update_space_references(worker, self.plan.common.get_immortal());
        let immortal_update_elapsed = immortal_update_start.elapsed();

        let nonmoving_update_start = std::time::Instant::now();
        self.compressor_space
            .update_space_references(worker, self.plan.common.get_nonmoving());
        let nonmoving_update_elapsed = nonmoving_update_start.elapsed();

        #[cfg(feature = "vm_space")]
        let vm_space_update_elapsed = {
            let vm_space_update_start = std::time::Instant::now();
            self.compressor_space
                .update_space_references(worker, &self.plan.common.base.vm_space);
            vm_space_update_start.elapsed()
        };
        #[cfg(not(feature = "vm_space"))]
        let vm_space_update_elapsed = std::time::Duration::ZERO;

        if std::env::var_os("MMTK_VALIDATE_UFFD_SPACE_REFS").is_some() {
            let valid_objects = validation_objects.as_ref().unwrap();
            let validate_object = |label: &str, object: ObjectReference| {
                VM::VMScanning::scan_object(worker.tls, object, &mut |slot: VM::VMSlot| {
                    let Some(referent) = slot.load() else {
                        return;
                    };
                    if self.compressor_space.in_space(referent)
                        && !valid_objects.contains(&referent)
                    {
                        panic!(
                            "{} contains stale Compressor ref {} in object {}",
                            label, referent, object,
                        );
                    }
                });
            };

            self.plan.common.get_immortal().enumerate_objects(
                &mut crate::util::object_enum::ClosureObjectEnumerator::<_, VM>::new(
                    |o: ObjectReference| validate_object("immortal", o),
                ),
            );
            self.plan.common.get_nonmoving().enumerate_objects(
                &mut crate::util::object_enum::ClosureObjectEnumerator::<_, VM>::new(
                    |o: ObjectReference| validate_object("nonmoving", o),
                ),
            );
            self.plan.common.get_los().enumerate_to_space_objects(
                &mut crate::util::object_enum::ClosureObjectEnumerator::<_, VM>::new(
                    |o: ObjectReference| validate_object("los", o),
                ),
            );
            #[cfg(feature = "vm_space")]
            self.plan.common.base.vm_space.enumerate_objects(
                &mut crate::util::object_enum::ClosureObjectEnumerator::<_, VM>::new(
                    |o: ObjectReference| validate_object("vm_space", o),
                ),
            );
            info!(
                "ValidateUffdSpaceRefs: validated non-moving space references against {} destination objects",
                valid_objects.len()
            );
        }

        let seal_start = std::time::Instant::now();
        self.compressor_space.seal_regions_for_concurrent_uffd();
        let seal_elapsed = seal_start.elapsed();

        let spawn_start = std::time::Instant::now();
        let uffd_arc = Arc::new(uffd_ctx);
        let handler = UffdContext::spawn_handler_thread(uffd_arc.clone());
        spawn_background_compactor(self.plan, self.compressor_space, uffd_arc, handler);
        self.plan.mark_uffd_epoch_ready();
        let spawn_elapsed = spawn_start.elapsed();
        if compressor_perf_trace_enabled() {
            info!(
                "Compressor FinalMark: UffdConcurrentSetup completed in {} ms (page_metadata={} ms, uffd_new={} ms, register={} ms, update_los={} ms, update_immortal={} ms, update_nonmoving={} ms, update_vm_space={} ms, seal={} ms, spawn={} ms)",
                format_perf_ms(start.elapsed()),
                format_perf_ms(page_metadata_elapsed),
                format_perf_ms(uffd_new_elapsed),
                format_perf_ms(register_elapsed),
                format_perf_ms(los_update_elapsed),
                format_perf_ms(immortal_update_elapsed),
                format_perf_ms(nonmoving_update_elapsed),
                format_perf_ms(vm_space_update_elapsed),
                format_perf_ms(seal_elapsed),
                format_perf_ms(spawn_elapsed)
            );
        }
    }
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> UffdConcurrentSetup<VM> {
    pub fn new(
        plan: &'static Compressor<VM>,
        compressor_space: &'static CompressorSpace<VM>,
    ) -> Self {
        Self {
            plan,
            compressor_space,
        }
    }
}

// ---------------------------------------------------------------------------
// Trace types and work contexts
// ---------------------------------------------------------------------------

/// Marking trace
pub type MarkingProcessEdges<VM> = PlanProcessEdges<VM, Compressor<VM>, TRACE_KIND_MARK>;
/// Forwarding trace
pub type ForwardingProcessEdges<VM> = PlanProcessEdges<VM, Compressor<VM>, TRACE_KIND_FORWARD_ROOT>;

#[cfg(feature = "uffd")]
pub(super) struct ConcurrentCompressorGCWorkContext<E: ProcessEdgesWork>(
    std::marker::PhantomData<E>,
);

#[cfg(feature = "uffd")]
impl<E: ProcessEdgesWork> crate::scheduler::GCWorkContext for ConcurrentCompressorGCWorkContext<E> {
    type VM = E::VM;
    type PlanType = Compressor<E::VM>;
    type DefaultProcessEdges = E;
    type PinningProcessEdges = E;
}

pub struct CompressorWorkContext<VM: VMBinding>(std::marker::PhantomData<VM>);
impl<VM: VMBinding> crate::scheduler::GCWorkContext for CompressorWorkContext<VM> {
    type VM = VM;
    type PlanType = Compressor<VM>;
    type DefaultProcessEdges = MarkingProcessEdges<VM>;
    type PinningProcessEdges = UnsupportedProcessEdges<VM>;
}

pub(super) struct CompressorForwardingWorkContext<VM: VMBinding>(std::marker::PhantomData<VM>);
impl<VM: VMBinding> crate::scheduler::GCWorkContext for CompressorForwardingWorkContext<VM> {
    type VM = VM;
    type PlanType = Compressor<VM>;
    type DefaultProcessEdges = ForwardingProcessEdges<VM>;
    type PinningProcessEdges = UnsupportedProcessEdges<VM>;
}
