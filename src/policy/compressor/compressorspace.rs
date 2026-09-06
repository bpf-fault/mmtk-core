use crate::plan::VectorObjectQueue;
use crate::policy::compressor::forwarding;
use crate::policy::gc_work::{TraceKind, TRACE_KIND_TRANSITIVE_PIN};
use crate::policy::largeobjectspace::LargeObjectSpace;
use crate::policy::sft::{GCWorkerMutRef, SFT};
use crate::policy::space::{CommonSpace, Space};
use crate::scheduler::{GCWork, GCWorkScheduler, GCWorker, WorkBucketStage};
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
use std::sync::Arc;

pub(crate) const TRACE_KIND_MARK: TraceKind = 0;
pub(crate) const TRACE_KIND_FORWARD_ROOT: TraceKind = 1;

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
    /// Post-compact live end per region (region start -> address), recorded
    /// by the offset-vector calculation; consumed by the concurrent-window
    /// flip for exact cursor presets.
    live_end: std::sync::Mutex<std::collections::HashMap<Address, Address>>,
    pr: RegionPageResource<VM, forwarding::CompressorRegion>,
    forwarding: forwarding::ForwardingMetadata<VM>,
    scheduler: Arc<GCWorkScheduler<VM>>,
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

    fn initialize_object_metadata(&self, _object: ObjectReference) {
        #[cfg(feature = "vo_bit")]
        crate::util::metadata::vo_bit::set_vo_bit(_object);
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

    fn clear_side_log_bits(&self) {
        unimplemented!()
    }

    fn set_side_log_bits(&self) {
        unimplemented!()
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
            live_end: std::sync::Mutex::new(std::collections::HashMap::new()),
            pr: if is_discontiguous {
                RegionPageResource::new_discontiguous(vm_map)
            } else {
                RegionPageResource::new_contiguous(common.start, common.extent, vm_map)
            },
            forwarding: forwarding::ForwardingMetadata::new(),
            common,
            scheduler,
        }
    }

    pub fn prepare(&self) {
        self.pr
            .enumerate_regions(&mut |r: &AllocatedRegion<forwarding::CompressorRegion>| {
                forwarding::MARK_SPEC
                    .bzero_metadata(r.region.start(), r.region.end() - r.region.start());
            });
    }

    /// Concurrent mode: is fault-driven concurrent install enabled?
    pub fn concurrent_install(&self) -> bool {
        crate::util::compact_faults::is_compact_faults_active()
            && *self.common.options.compact_concurrent
    }

    /// Pause-side flip for the concurrent window: per region, set page
    /// states ([start, predicted_to) pending), move pages to the arena,
    /// and preset the allocation cursor to the page-aligned post-compact
    /// position so window-time allocation never shares a page with staged
    /// installs.  Returns the number of regions to stage.
    pub fn flip_all_and_preset(&self) -> usize {
        use crate::util::compact_faults::BYTES_IN_PAGE;
        let cf = crate::util::compact_faults::compact_faults().unwrap();
        let region_bytes = forwarding::CompressorRegion::BYTES;
        // Mark all regions non-claimable (DONE); each live region is marked
        // claimable (UNSTAGED) below.  This is the steal-mode coordination.
        cf.reset_region_staging();
        self.pr.with_regions(&mut |regions| {
            // Coalesce contiguous regions into single mremap+register calls:
            // per-region flips churn VMAs (split per region) and dominate
            // the pause.
            let mut runs: Vec<(Address, usize)> = Vec::new();
            for r in regions.iter() {
                let start = r.region.start();
                let cursor = r.cursor();
                // Exact post-compact cursor, recorded by the offset-vector
                // calculation (the transducer's final position).
                let predicted_to = *self
                    .live_end
                    .lock()
                    .unwrap()
                    .get(&start)
                    .unwrap_or(&start);
                debug_assert!(
                    predicted_to >= start && predicted_to <= cursor,
                    "predicted_to {} outside region {}..{}",
                    predicted_to,
                    start,
                    cursor
                );
                let aligned_to = predicted_to.align_up(BYTES_IN_PAGE);
                cf.reset_region_state(start, region_bytes);
                if aligned_to > start {
                    cf.set_pending(start, aligned_to - start);
                }
                // Mark this region claimable (steal-mode) with its cursor.
                cf.mark_region_live(start, aligned_to);
                match runs.last_mut() {
                    Some(last) if last.0 + last.1 == start => last.1 += region_bytes,
                    _ => runs.push((start, region_bytes)),
                }
                self.pr.reset_cursor(r, aligned_to);
            }
            for &(start, bytes) in &runs {
                cf.flip(start, bytes);
            }
            regions.len()
        })
    }

    /// Concurrent-window staging of one region: slide-compact in the arena,
    /// mark pages staged, install them, clear any leftover pending state,
    /// and finish (uffd unregister).  Runs in the Concurrent bucket while
    /// mutators execute; the cursor was preset in the pause.
    /// Stage one region (address-indexed) during the concurrent window:
    /// claim it (CAS, so exactly one of {GC sweeper, faulting mutator}
    /// stages each region), slide-compact in the arena, install, then mark
    /// it done and decrement the window counter (closing the window on the
    /// last region).  Lock-free: bounds come from the flip-time precomputed
    /// `region_cursor`, so this is safe to call from a mutator SIGBUS
    /// handler.  No-op if the region is already claimed/done.
    pub fn stage_region_idx(&self, tls: crate::util::VMWorkerThread, aidx: usize) {
        use crate::util::compact_faults::BYTES_IN_PAGE;
        let cf = crate::util::compact_faults::compact_faults().unwrap();
        if !cf.claim_region(aidx) {
            return; // another thread is staging or has staged this region
        }
        let region_bytes = cf.region_bytes();
        let (start, end) = cf.region_bounds(aidx);
        let delta = cf.alias_delta();
        let shift = |o: ObjectReference| -> ObjectReference {
            unsafe {
                ObjectReference::from_raw_address_unchecked(Address::from_usize(
                    (o.to_raw_address().as_usize() as isize + delta) as usize,
                ))
            }
        };
        #[cfg(feature = "vo_bit")]
        crate::util::metadata::vo_bit::bzero_vo_bit(start, region_bytes);
        let mut to = start;
        self.forwarding
            .scan_marked_objects(start, start + region_bytes, &mut |obj: ObjectReference| {
                let alias_obj = shift(obj);
                let copied_size = VM::VMObjectModel::get_size_when_copied(alias_obj);
                let new_object = self.forward(obj, false);
                let alias_new = shift(new_object);
                VM::VMObjectModel::copy_to(alias_obj, alias_new, Address::ZERO);
                #[cfg(feature = "vo_bit")]
                vo_bit::set_vo_bit(new_object);
                to = new_object.to_object_start::<VM>() + copied_size;
                self.update_references(tls, alias_new);
            });
        let staged_end = to.align_up(BYTES_IN_PAGE);
        assert!(
            staged_end <= end || end == start,
            "stage_region: staged_end {} exceeds preset cursor {} (region {})",
            staged_end,
            end,
            start
        );
        if staged_end > start {
            cf.stage(start, staged_end - start);
            cf.install(start, staged_end - start);
        }
        // Clear any pending pages we predicted but did not stage, so no
        // mutator waits forever on them.
        if staged_end < start + region_bytes {
            cf.reset_region_state(staged_end, start + region_bytes - staged_end);
        }
        cf.finish_region(start, region_bytes);
        cf.mark_region_done(aidx);
        if cf.region_done() {
            self.close_window();
        }
    }

    /// Sweep all regions, staging each unclaimed one.  Several of these run
    /// in parallel on GC workers during the window; faulting mutators steal
    /// individual regions.  The claim CAS coordinates them.
    pub fn stage_sweep(&self, tls: crate::util::VMWorkerThread) {
        let cf = crate::util::compact_faults::compact_faults().unwrap();
        for aidx in 0..cf.region_count() {
            self.stage_region_idx(tls, aidx);
        }
    }

    pub fn release(&self) {
        if self.concurrent_install() {
            // The concurrent window still needs the forwarding metadata;
            // released by the window-close packet instead.
            return;
        }
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
        let live_end = self.forwarding.calculate_offset_vector(region, cursor);
        self.live_end
            .lock()
            .unwrap()
            .insert(region.start(), live_end);
    }

    pub fn forward(&self, object: ObjectReference, _vo_bit_valid: bool) -> ObjectReference {
        if !self.in_space(object) {
            return object;
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

    fn update_references(&self, tls: crate::util::VMWorkerThread, object: ObjectReference) {
        if VM::VMScanning::support_slot_enqueuing(tls, object) {
            VM::VMScanning::scan_object(tls, object, &mut |s: VM::VMSlot| {
                if let Some(o) = s.load() {
                    s.store(self.forward(o, false));
                }
            });
        } else {
            VM::VMScanning::scan_object_and_trace_edges(tls, object, &mut |o| {
                self.forward(o, false)
            });
        }
    }

    /// Close the concurrent window: release forwarding metadata.
    pub fn close_window(&self) {
        let cf = crate::util::compact_faults::compact_faults().unwrap();
        self.forwarding.release();
        cf.close_window();
    }

    /// Stage tasks for the concurrent window (Concurrent bucket).
    pub fn add_stage_tasks(&'static self) {
        // Several parallel sweepers; each claims unclaimed regions (the CAS
        // coordinates them and any faulting mutators that steal regions).
        let n = self.scheduler.num_workers();
        let packets: Vec<Box<dyn GCWork<VM>>> =
            (0..n).map(|_| Box::new(StageSweep::<VM>::new(self)) as Box<dyn GCWork<VM>>).collect();
        self.scheduler.work_buckets[WorkBucketStage::Concurrent].bulk_add(packets);
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
            // Fault-driven compaction (Class B): flip the region's pages
            // into the from-space arena; all object reads/writes below then
            // go through alias addresses (metadata stays at real addresses).
            let cf = crate::util::compact_faults::compact_faults();
            let region_bytes = forwarding::CompressorRegion::BYTES;
            if let Some(cf) = cf {
                cf.reset_region_state(start, region_bytes);
                cf.flip(start, region_bytes);
            }
            let delta = cf.map_or(0isize, |c| c.alias_delta());
            let shift = |o: ObjectReference| -> ObjectReference {
                unsafe {
                    ObjectReference::from_raw_address_unchecked(Address::from_usize(
                        (o.to_raw_address().as_usize() as isize + delta) as usize,
                    ))
                }
            };
            let mut to = start;
            self.forwarding
                .scan_marked_objects(start, end, &mut |obj: ObjectReference| {
                    // We set the end bits based on the sizes of objects when they are
                    // marked, and we compute the live data and thus the forwarding
                    // addresses based on those sizes. The forwarding addresses would be
                    // incorrect if the sizes of objects were to change.
                    let alias_obj = shift(obj);
                    let copied_size = VM::VMObjectModel::get_size_when_copied(alias_obj);
                    debug_assert!(copied_size == VM::VMObjectModel::get_current_size(alias_obj));
                    let new_object = self.forward(obj, false);
                    debug_assert!(
                        new_object.to_raw_address() >= to,
                        "whilst forwarding {obj}, the new address {0} should be after the end of the last object {to}",
                        new_object.to_raw_address()
                    );
                    // copy object (within the arena when fault-driven)
                    trace!(" copy from {} to {}", obj, new_object);
                    let alias_new = shift(new_object);
                    let end_of_new_object =
                        VM::VMObjectModel::copy_to(alias_obj, alias_new, Address::ZERO);
                    // update VO bit
                    #[cfg(feature = "vo_bit")]
                    vo_bit::set_vo_bit(new_object);
                    to = new_object.to_object_start::<VM>() + copied_size;
                    debug_assert_eq!(
                        end_of_new_object,
                        unsafe {
                            Address::from_usize((to.as_usize() as isize + delta) as usize)
                        }
                    );
                    self.update_references(worker.tls, alias_new);
                });
            if let Some(cf) = cf {
                use crate::util::compact_faults::BYTES_IN_PAGE;
                let staged_end = to.align_up(BYTES_IN_PAGE);
                if staged_end > start {
                    cf.stage(start, staged_end - start);
                    cf.install(start, staged_end - start);
                }
                cf.finish_region(start, region_bytes);
            }
            self.pr.reset_cursor(r, to);
        });
    }

    pub fn after_compact(&self, worker: &mut GCWorker<VM>, los: &LargeObjectSpace<VM>) {
        self.pr.reset_allocator();
        // Update references from the LOS to Compressor too.
        los.enumerate_to_space_objects(&mut object_enum::ClosureObjectEnumerator::<_, VM>::new(
            &mut |o: ObjectReference| {
                self.update_references(worker.tls, o);
            },
        ));
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

/// A parallel sweeper for the concurrent window: stages every region it can
/// claim (window close + per-region done accounting live in
/// `stage_region_idx`).
pub struct StageSweep<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
}

impl<VM: VMBinding> GCWork<VM> for StageSweep<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        self.compressor_space.stage_sweep(worker.tls);
    }
}

impl<VM: VMBinding> StageSweep<VM> {
    pub fn new(compressor_space: &'static CompressorSpace<VM>) -> Self {
        Self { compressor_space }
    }
}

/// Steal handler: lets a faulting mutator stage the region it faulted on.
struct CompressorSteal<VM: VMBinding>(&'static CompressorSpace<VM>);

impl<VM: VMBinding> crate::util::compact_faults::StealHandler for CompressorSteal<VM> {
    fn stage(&self, region_index: usize) {
        // HotSpot's scan_object ignores the worker tls, so a mutator may
        // stage with an uninitialized one.
        let tls = crate::util::VMWorkerThread(crate::util::VMThread::UNINITIALIZED);
        self.0.stage_region_idx(tls, region_index);
    }
}

impl<VM: VMBinding> CompressorSpace<VM> {
    /// Register this space's steal handler (idempotent).
    pub fn register_steal(&'static self) {
        crate::util::compact_faults::register_steal_handler(Box::new(CompressorSteal(self)));
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
