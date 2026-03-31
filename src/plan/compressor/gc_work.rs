use super::global::Compressor;
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
use crate::vm::{ActivePlan, Scanning, VMBinding};
use crate::MMTK;
use std::marker::{PhantomData, Send};
use std::sync::OnceLock;
#[cfg(feature = "uffd")]
use std::{collections::HashSet, sync::Arc};

fn compressor_perf_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("MMTK_TRACE_COMPRESSOR_PERF").is_some())
}

/// Generate more packets by calling a method on [`CompressorSpace`].
pub struct GenerateWork<VM: VMBinding, F: Fn(&'static CompressorSpace<VM>) + Send + 'static> {
    compressor_space: &'static CompressorSpace<VM>,
    f: F,
}

impl<VM: VMBinding, F: Fn(&'static CompressorSpace<VM>) + Send + 'static> GCWork<VM>
    for GenerateWork<VM, F>
{
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        (self.f)(self.compressor_space);
    }
}

impl<VM: VMBinding, F: Fn(&'static CompressorSpace<VM>) + Send + 'static> GenerateWork<VM, F> {
    pub fn new(compressor_space: &'static CompressorSpace<VM>, f: F) -> Self {
        Self {
            compressor_space,
            f,
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
        if compressor_perf_trace_enabled() {
            info!(
                "Compressor FinalMark: captured {} black-allocated objects before forwarding in {} ms",
                objects.len(),
                start.elapsed().as_millis()
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
}

unsafe impl<VM: VMBinding> Send for UpdateReferences<VM> {}

impl<VM: VMBinding> GCWork<VM> for UpdateReferences<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        let start = std::time::Instant::now();
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
                start.elapsed().as_millis()
            );
        }
    }
}

impl<VM: VMBinding> UpdateReferences<VM> {
    pub fn new() -> Self {
        Self { p: PhantomData }
    }
}

#[cfg(feature = "uffd")]
struct ValidateMutatorRootsFactory<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
    valid_objects: Arc<HashSet<ObjectReference>>,
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> Clone for ValidateMutatorRootsFactory<VM> {
    fn clone(&self) -> Self {
        Self {
            compressor_space: self.compressor_space,
            valid_objects: self.valid_objects.clone(),
        }
    }
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> ValidateMutatorRootsFactory<VM> {
    fn validate_object(&self, object: ObjectReference) {
        if self.compressor_space.in_space(object) && !self.valid_objects.contains(&object) {
            panic!(
                "ValidateMutatorRoots: stale Compressor root/reference {} is not a valid destination object",
                object
            );
        }
    }
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> crate::vm::RootsWorkFactory<VM::VMSlot> for ValidateMutatorRootsFactory<VM> {
    fn roots_work_bucket_stage(&self) -> WorkBucketStage {
        WorkBucketStage::Compact
    }

    fn process_edges_roots_work_bucket_stage(&self) -> WorkBucketStage {
        WorkBucketStage::Compact
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
impl<VM: VMBinding> GCWork<VM> for ValidateMutatorRoots<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        VM::VMScanning::prepare_for_roots_re_scanning();
        mmtk.state.prepare_for_stack_scanning();

        let valid_objects = Arc::new(self.compressor_space.collect_destination_objects());
        let factory = ValidateMutatorRootsFactory {
            compressor_space: self.compressor_space,
            valid_objects: valid_objects.clone(),
        };

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
        let factory = ValidateMutatorRootsFactory {
            compressor_space: self.compressor_space,
            valid_objects: valid_objects.clone(),
        };

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

/// Reset the allocator and update references in large object space.
pub struct AfterCompact<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
    los: &'static LargeObjectSpace<VM>,
}

impl<VM: VMBinding> GCWork<VM> for AfterCompact<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        self.compressor_space.after_compact(worker, self.los);
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
        for region_idx in (0..compressor_space.num_regions()).rev() {
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
    los: &'static LargeObjectSpace<VM>,
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> GCWork<VM> for UffdConcurrentSetup<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        use crate::policy::compressor::uffd::UffdContext;
        use std::sync::Arc;
        let start = std::time::Instant::now();

        let num_regions = self.compressor_space.num_regions();
        if num_regions == 0 {
            return;
        }

        self.compressor_space.build_all_page_metadata();

        let mut uffd_ctx = UffdContext::new(self.compressor_space)
            .unwrap_or_else(|e| panic!("UffdConcurrentSetup: uffd creation failed: {}", e));

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

        if compressor_perf_trace_enabled() {
            info!(
                "UffdConcurrentSetup: registered {} destination pages across {} regions (source_pages={}, moving_regions={}, static_regions={}, max_region_pages={})",
                total_registered_pages,
                num_regions,
                total_source_pages,
                moving_regions,
                static_regions,
                num_regions
                    * (crate::policy::compressor::forwarding::CompressorRegion::BYTES
                        / crate::util::constants::BYTES_IN_PAGE)
            );
        }

        // ART's `CompactionPause()` updates immune / non-moving spaces and roots
        // before mutators resume. Update all MMTk-managed non-moving spaces that can
        // hold references into Compressor, not just LOS.
        self.compressor_space
            .update_los_references(worker, self.los);
        self.compressor_space
            .update_space_references(worker, self.plan.common.get_immortal());
        self.compressor_space
            .update_space_references(worker, self.plan.common.get_nonmoving());
        #[cfg(feature = "vm_space")]
        self.compressor_space
            .update_space_references(worker, &self.plan.common.base.vm_space);

        self.compressor_space.seal_regions_for_concurrent_uffd();

        let uffd_arc = Arc::new(uffd_ctx);
        let handler = UffdContext::spawn_handler_thread(uffd_arc.clone());
        spawn_background_compactor(self.plan, self.compressor_space, uffd_arc, handler);
        self.plan.mark_uffd_epoch_ready();
        if compressor_perf_trace_enabled() {
            info!(
                "Compressor FinalMark: UffdConcurrentSetup completed in {} ms",
                start.elapsed().as_millis()
            );
        }
    }
}

#[cfg(feature = "uffd")]
impl<VM: VMBinding> UffdConcurrentSetup<VM> {
    pub fn new(
        plan: &'static Compressor<VM>,
        compressor_space: &'static CompressorSpace<VM>,
        los: &'static LargeObjectSpace<VM>,
    ) -> Self {
        Self {
            plan,
            compressor_space,
            los,
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
