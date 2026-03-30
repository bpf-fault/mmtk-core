use super::global::Compressor;
use crate::policy::compressor::{CompressorSpace, TRACE_KIND_FORWARD_ROOT, TRACE_KIND_MARK};
use crate::policy::largeobjectspace::LargeObjectSpace;
use crate::scheduler::gc_work::PlanProcessEdges;
use crate::scheduler::gc_work::*;
use crate::scheduler::{GCWork, GCWorker, WorkBucketStage};
use crate::util::linear_scan::Region;
use crate::vm::{ActivePlan, Scanning, VMBinding};
use crate::MMTK;
use std::marker::{PhantomData, Send};

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

/// Create another round of root scanning work packets
/// to update object references.
pub struct UpdateReferences<VM: VMBinding> {
    p: PhantomData<VM>,
}

unsafe impl<VM: VMBinding> Send for UpdateReferences<VM> {}

impl<VM: VMBinding> GCWork<VM> for UpdateReferences<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        // The following needs to be done right before the second round of root scanning
        VM::VMScanning::prepare_for_roots_re_scanning();
        mmtk.state.prepare_for_stack_scanning();
        #[cfg(feature = "extreme_assertions")]
        mmtk.slot_logger.reset();

        for mutator in VM::VMActivePlan::mutators() {
            mmtk.scheduler.work_buckets[WorkBucketStage::SecondRoots].add(ScanMutatorRoots::<
                CompressorForwardingWorkContext<VM>,
            >(mutator));
        }

        mmtk.scheduler.work_buckets[WorkBucketStage::SecondRoots]
            .add(ScanVMSpecificRoots::<CompressorForwardingWorkContext<VM>>::new());
    }
}

impl<VM: VMBinding> UpdateReferences<VM> {
    pub fn new() -> Self {
        Self { p: PhantomData }
    }
}

/// Reset the allocator and update references in large object space.
#[cfg(not(feature = "uffd"))]
pub struct AfterCompact<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
    los: &'static LargeObjectSpace<VM>,
}

#[cfg(not(feature = "uffd"))]
impl<VM: VMBinding> GCWork<VM> for AfterCompact<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        self.compressor_space.after_compact(worker, self.los);
    }
}

#[cfg(not(feature = "uffd"))]
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
        while !plan.is_uffd_concurrent_active() {
            std::thread::yield_now();
        }

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
        ctx.teardown();
        plan.finish_uffd_epoch();

        info!(
            "UffdConcurrentPhase: finalized_regions={}, total_pages_resolved={}, faults={}, states=[u={}, p={}, pr={}, pm={}, mp={}, pdm={}, m={}]",
            finalized_regions,
            pages_resolved,
            faults,
            state_counts[0],
            state_counts[1],
            state_counts[2],
            state_counts[3],
            state_counts[4],
            state_counts[5],
            state_counts[6]
        );
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

        let num_regions = self.compressor_space.num_regions();
        if num_regions == 0 {
            return;
        }

        let region_size = crate::policy::compressor::forwarding::CompressorRegion::BYTES;
        if self.compressor_space.region_page_metadata(0).is_none() {
            self.compressor_space.build_all_page_metadata();
        }

        let mut uffd_ctx = UffdContext::new(self.compressor_space)
            .unwrap_or_else(|e| panic!("UffdConcurrentSetup: uffd creation failed: {}", e));

        for i in 0..num_regions {
            let (start, _cursor) = self.compressor_space.region_info(i);
            uffd_ctx
                .mremap_and_register(i, start.as_usize(), region_size)
                .unwrap_or_else(|e| {
                    panic!(
                        "UffdConcurrentSetup: mremap/register failed for region {}: {}",
                        i, e
                    )
                });
        }

        // ART updates non-moving / root-visible references before mutators resume.
        self.compressor_space
            .update_los_references(worker, self.los);

        let uffd_arc = Arc::new(uffd_ctx);
        let handler = UffdContext::spawn_handler_thread(uffd_arc.clone());
        spawn_background_compactor(self.plan, self.compressor_space, uffd_arc, handler);
        self.plan.mark_uffd_epoch_ready();
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
