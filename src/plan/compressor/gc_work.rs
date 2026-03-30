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

#[cfg(target_os = "linux")]
pub struct UffdConcurrentSetup<VM: VMBinding> {
    plan: &'static Compressor<VM>,
    compressor_space: &'static CompressorSpace<VM>,
    los: &'static LargeObjectSpace<VM>,
}

#[cfg(target_os = "linux")]
impl<VM: VMBinding> GCWork<VM> for UffdConcurrentSetup<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
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

        // In the concurrent UFFD path, mapped pages must be self-consistent before
        // mutators resume, so we always enable page-local fixup.
        let mut uffd_ctx = UffdContext::new(self.compressor_space, true)
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

        // Update non-compressed-space references before mutators resume.
        self.compressor_space.update_los_references(worker, self.los);

        let uffd_arc = Arc::new(uffd_ctx);
        let handler = UffdContext::spawn_handler_thread(uffd_arc.clone());
        self.plan.set_uffd_context(uffd_arc.clone(), handler);
        self.plan.set_uffd_concurrent_regions_remaining(num_regions);

        for region_idx in 0..num_regions {
            mmtk.scheduler.work_buckets[WorkBucketStage::Concurrent].add_no_notify(
                UffdConcurrentProcessRegion::<VM>::new(self.plan, uffd_arc.clone(), region_idx),
            );
        }
    }
}

#[cfg(target_os = "linux")]
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

#[cfg(target_os = "linux")]
pub struct UffdConcurrentProcessRegion<VM: VMBinding> {
    plan: &'static Compressor<VM>,
    ctx: std::sync::Arc<crate::policy::compressor::uffd::UffdContext<VM>>,
    region_idx: usize,
}

#[cfg(target_os = "linux")]
impl<VM: VMBinding> GCWork<VM> for UffdConcurrentProcessRegion<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        self.ctx
            .process_region_pages(self.region_idx)
            .unwrap_or_else(|e| panic!("UffdConcurrentProcessRegion {} failed: {}", self.region_idx, e));
        if self.plan.on_uffd_concurrent_region_processed()
            && self.plan.request_uffd_final_pause_if_needed()
        {
            info!("UffdConcurrent: background processing done, requesting final pause");
            worker.scheduler().request_schedule_collection();
        }
    }
}

#[cfg(target_os = "linux")]
impl<VM: VMBinding> UffdConcurrentProcessRegion<VM> {
    pub fn new(
        plan: &'static Compressor<VM>,
        ctx: std::sync::Arc<crate::policy::compressor::uffd::UffdContext<VM>>,
        region_idx: usize,
    ) -> Self {
        Self { plan, ctx, region_idx }
    }
}

#[cfg(target_os = "linux")]
pub struct UffdConcurrentFinish<VM: VMBinding> {
    plan: &'static Compressor<VM>,
    compressor_space: &'static CompressorSpace<VM>,
}

#[cfg(target_os = "linux")]
impl<VM: VMBinding> GCWork<VM> for UffdConcurrentFinish<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        let Some(ctx) = self.plan.take_uffd_context() else {
            warn!("UffdConcurrentFinish: no active uffd context");
            return;
        };

        ctx.signal_done();
        if let Some(handler) = self.plan.take_uffd_handler_thread() {
            handler.join().unwrap_or_else(|e| {
                error!("UffdConcurrentFinish: handler thread panicked: {:?}", e);
            });
        }

        let resolved_remaining = ctx
            .resolve_remaining_pages()
            .unwrap_or_else(|e| panic!("UffdConcurrentFinish: failed to resolve remaining pages: {}", e));

        let validate_uffd =
            std::env::var("MMTK_COMPRESSOR_UFFD_VALIDATE").map_or(false, |v| v == "1");
        if validate_uffd {
            // Mutators were resumed during the concurrent UFFD phase, so raw bytes in the
            // compacted prefix may have legitimately changed. Only perform structural/reference
            // validation here.
            let valid_objects = self.compressor_space.collect_destination_objects();
            if let Err(msg) = self
                .compressor_space
                .validate_updated_references(worker, &valid_objects)
            {
                panic!("UffdConcurrentFinish reference validation failed: {}", msg);
            }
        }

        for i in 0..self.compressor_space.num_regions() {
            self.compressor_space
                .finalize_region_cursor_after_concurrent_uffd(i);
        }
        // In the concurrent UFFD path, non-moving / LOS references were already
        // updated before mutator resume in `UffdConcurrentSetup`, mirroring ART's
        // pre-resume root/non-moving-space update discipline. Re-running that
        // forwarding logic here would incorrectly try to forward references to
        // objects allocated during the concurrent window.
        self.compressor_space.reset_allocator_after_compaction();

        let state_counts = ctx.page_state_counts();
        let faults = ctx.faults_handled();
        let pages_resolved = ctx.pages_resolved();
        ctx.teardown();

        info!(
            "UffdConcurrentFinish: resolved_remaining={}, total_pages_resolved={}, faults={}, states=[u={}, p={}, pr={}, pm={}, mp={}, pdm={}, m={}]",
            resolved_remaining,
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
    }
}

#[cfg(target_os = "linux")]
impl<VM: VMBinding> UffdConcurrentFinish<VM> {
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

/// Work packet that performs copy-only compaction through userfaultfd.
///
/// This still runs inside a single stop-the-world epoch, but the mechanism is
/// now page-granular and shadow-driven:
/// 1. Build page metadata from the pre-compaction layout.
/// 2. `mremap(MREMAP_DONTUNMAP)` each region to a shadow mapping.
/// 3. Register the original range with `userfaultfd`.
/// 4. Reconstruct destination pages from shadow and materialize them with
///    `UFFDIO_COPY`.
/// 5. Update references after all compacted pages are mapped back.
/// 6. Tear down uffd/shadow state.
#[cfg(target_os = "linux")]
pub struct UffdCompact<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
    los: &'static LargeObjectSpace<VM>,
}

#[cfg(target_os = "linux")]
impl<VM: VMBinding> GCWork<VM> for UffdCompact<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        use crate::policy::compressor::uffd::UffdContext;
        use std::sync::Arc;

        let num_regions = self.compressor_space.num_regions();
        if num_regions == 0 {
            self.compressor_space.after_compact(worker, self.los);
            return;
        }

        let region_size = crate::policy::compressor::forwarding::CompressorRegion::BYTES;

        let validate_uffd =
            std::env::var("MMTK_COMPRESSOR_UFFD_VALIDATE").map_or(false, |v| v == "1");
        let page_local_fixup =
            std::env::var("MMTK_COMPRESSOR_UFFD_PAGE_LOCAL_FIXUP").map_or(false, |v| v == "1");

        // Step 1: Build page metadata from the pre-compaction layout if the
        // scheduler has not already done so for this GC.
        let metadata_start = std::time::Instant::now();
        if self.compressor_space.region_page_metadata(0).is_none() {
            self.compressor_space.build_all_page_metadata();
        }
        let metadata_elapsed = metadata_start.elapsed();

        // Step 2: mremap original regions to shadow and register originals with uffd.
        let setup_start = std::time::Instant::now();
        let mut uffd_ctx = match UffdContext::new(self.compressor_space, page_local_fixup) {
            Ok(ctx) => ctx,
            Err(e) => {
                warn!("UffdCompact: uffd creation failed ({}), skipping uffd resolution", e);
                self.compressor_space.after_compact(worker, self.los);
                return;
            }
        };

        let mut setup_ok = true;
        for i in 0..num_regions {
            let (start, _cursor) = self.compressor_space.region_info(i);
            if let Err(e) = uffd_ctx.mremap_and_register(i, start.as_usize(), region_size) {
                error!("UffdCompact: mremap/register failed for region {}: {}", i, e);
                setup_ok = false;
                break;
            }
        }

        if !setup_ok {
            warn!("UffdCompact: setup failed, tearing down partial state");
            uffd_ctx.teardown();
            panic!("UffdCompact: setup failed after mremap/register; heap state is not recoverable");
        }
        let setup_elapsed = setup_start.elapsed();

        // Step 3: Resolve compacted pages page-by-page through uffd.
        let resolve_start = std::time::Instant::now();
        let uffd_arc = Arc::new(uffd_ctx);
        let handler = UffdContext::spawn_handler_thread(uffd_arc.clone());
        let total_pages = match uffd_arc.resolve_all_pages() {
            Ok(pages) => pages,
            Err(e) => {
                uffd_arc.signal_done();
                let _ = handler.join();
                uffd_arc.teardown();
                panic!("UffdCompact: failed to resolve pages: {}", e);
            }
        };
        uffd_arc.signal_done();
        handler.join().unwrap_or_else(|e| {
            error!("UffdCompact: handler thread panicked: {:?}", e);
        });
        let resolve_elapsed = resolve_start.elapsed();

        if validate_uffd {
            // Debug validation: compare the resolved bytes with a direct object-by-object
            // compaction from shadow before mutating references in-place.
            for shadow in &uffd_arc.shadows {
                if let Err(msg) = self.compressor_space.validate_region_compaction_from_shadow(
                    shadow.region_index,
                    unsafe { crate::util::Address::from_usize(shadow.shadow_start) },
                    page_local_fixup,
                ) {
                    uffd_arc.teardown();
                    panic!("UffdCompact validation failed: {}", msg);
                }
            }
        }

        // Step 4: Now that all pages are mapped back, finish object/reference state.
        let fixup_start = std::time::Instant::now();
        if page_local_fixup {
            for i in 0..num_regions {
                self.compressor_space.reset_region_cursor_to_compacted_end(i);
            }
        } else {
            for i in 0..num_regions {
                self.compressor_space.update_references_region(worker, i);
                self.compressor_space.reset_region_cursor_to_compacted_end(i);
            }
        }
        if validate_uffd {
            let valid_objects = self.compressor_space.collect_destination_objects();
            if let Err(msg) = self
                .compressor_space
                .validate_updated_references(worker, &valid_objects)
            {
                uffd_arc.teardown();
                panic!("UffdCompact reference validation failed: {}", msg);
            }
        }
        self.compressor_space.after_compact(worker, self.los);
        let fixup_elapsed = fixup_start.elapsed();

        let faults = uffd_arc.faults_handled();
        let pages_resolved = uffd_arc.pages_resolved();
        let state_counts = uffd_arc.page_state_counts();

        // Step 5: Teardown.
        uffd_arc.teardown();

        info!(
            "UffdCompact: {} regions, metadata={:?}, setup={:?}, resolve={:?}, fixup={:?}, page_local_fixup={} ({} pages resolved, {} copied, {} faults, states=[u={}, p={}, pr={}, pm={}, mp={}, pdm={}, m={}])",
            num_regions,
            metadata_elapsed,
            setup_elapsed,
            resolve_elapsed,
            fixup_elapsed,
            page_local_fixup,
            pages_resolved,
            total_pages,
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
}

#[cfg(target_os = "linux")]
impl<VM: VMBinding> UffdCompact<VM> {
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

pub struct CompressorForwardingWorkContext<VM: VMBinding>(std::marker::PhantomData<VM>);
impl<VM: VMBinding> crate::scheduler::GCWorkContext for CompressorForwardingWorkContext<VM> {
    type VM = VM;
    type PlanType = Compressor<VM>;
    type DefaultProcessEdges = ForwardingProcessEdges<VM>;
    type PinningProcessEdges = UnsupportedProcessEdges<VM>;
}
