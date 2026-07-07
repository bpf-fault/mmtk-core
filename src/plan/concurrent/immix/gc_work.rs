use crate::plan::concurrent::immix::global::ConcurrentImmix;
use crate::policy::gc_work::{TraceKind, TRACE_KIND_TRANSITIVE_PIN};
use crate::scheduler::gc_work::PlanProcessEdges;
use crate::scheduler::ProcessEdgesWork;
use crate::vm::VMBinding;

pub(super) struct ConcurrentImmixSTWGCWorkContext<VM: VMBinding, const KIND: TraceKind>(
    std::marker::PhantomData<VM>,
);
impl<VM: VMBinding, const KIND: TraceKind> crate::scheduler::GCWorkContext
    for ConcurrentImmixSTWGCWorkContext<VM, KIND>
{
    type VM = VM;
    type PlanType = ConcurrentImmix<VM>;
    type DefaultProcessEdges = PlanProcessEdges<VM, ConcurrentImmix<VM>, KIND>;
    type PinningProcessEdges = PlanProcessEdges<VM, ConcurrentImmix<VM>, TRACE_KIND_TRANSITIVE_PIN>;
}
pub(super) struct ConcurrentImmixGCWorkContext<E: ProcessEdgesWork>(std::marker::PhantomData<E>);

impl<E: ProcessEdgesWork> crate::scheduler::GCWorkContext for ConcurrentImmixGCWorkContext<E> {
    type VM = E::VM;
    type PlanType = ConcurrentImmix<E::VM>;
    type DefaultProcessEdges = E;
    type PinningProcessEdges = E;
}

/* ---------------- page-COW SATB (MMTK_SATB_PAGES) ---------------- */

use crate::plan::concurrent::concurrent_marking_work::ProcessModBufSATB;
use crate::plan::concurrent::global::ConcurrentPlan;
use crate::plan::global::Plan;
use crate::policy::space::Space;
use crate::policy::immix::TRACE_KIND_FAST;
use crate::scheduler::{GCWork, GCWorker, WorkBucketStage};
use crate::util::satb_pages;
use crate::util::ObjectReference;
use crate::MMTK;

/// Convert a conservative snapshot-page value into an SATB node.
/// Only immix-space targets matter (LOS/immortal/nonmoving are wholesale
/// re-scanned at FinalMark), and the space check MUST precede the VO-bit
/// read: side metadata is only mapped for in-use chunks, so probing the
/// VO bit of a garbage candidate outside the space SEGVs.
fn satb_node<VM: VMBinding>(
    plan: &ConcurrentImmix<VM>,
    addr: crate::util::Address,
) -> Option<ObjectReference> {
    if !plan.immix_space.address_in_space(addr) {
        return None;
    }
    // The chunk must be LIVE: released chunks keep stale VO-bit metadata
    // while their heap pages are unmapped (measured MAPERR tracing a
    // candidate in a freed chunk).
    {
        use crate::util::heap::chunk_map::Chunk;
        use crate::util::linear_scan::Region;
        let chunk = Chunk::from_unaligned_address(addr);
        match plan.immix_space.chunk_map.get(chunk) {
            Some(s) if s.is_allocated() => {}
            _ => return None,
        }
    }
    let obj = ObjectReference::from_raw_address(addr)?;
    #[cfg(feature = "vo_bit")]
    if !crate::util::metadata::vo_bit::is_vo_bit_set(obj) {
        return None;
    }
    Some(obj)
}

/// Spawn the concurrent drainer THREAD.  A self-re-enqueueing packet
/// keeps the Concurrent bucket non-empty forever, so marking is never
/// considered finished (measured livelock: mutators + 2ms packet churn
/// at ~33% CPU, FinalMark never scheduled).  A plain thread drains
/// snapshot pages while marking runs and exits when it completes.  The
/// packets it emits are real marking work, so marking cannot terminate
/// with meaningful snapshots undrained; the FinalMark sweep catches the
/// tail race.
pub(super) struct SatbDrainStart<VM: VMBinding>(std::marker::PhantomData<VM>);

impl<VM: VMBinding> SatbDrainStart<VM> {
    pub fn new() -> Self {
        SatbDrainStart(std::marker::PhantomData)
    }
}

impl<VM: VMBinding> GCWork<VM> for SatbDrainStart<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        std::thread::spawn(move || {
            let plan = mmtk
                .get_plan()
                .downcast_ref::<ConcurrentImmix<VM>>()
                .unwrap();
            let Some(t) = satb_pages::satb_pages() else {
                return;
            };
            t.drainer_started();
            // The thread may start before concurrent work is queued: wait
            // for marking to begin (bounded grace), then exit when it ends.
            let mut started = false;
            let mut grace = 0u32;
            loop {
                if t.should_stop() {
                    t.drainer_exited();
                    return;
                }
                if !plan.concurrent_work_in_progress() {
                    if started {
                        t.drainer_exited();
                        return;
                    }
                    grace += 1;
                    if grace > 2500 {
                        t.drainer_exited();
                        return; // marking never started (~5s): bail
                    }
                    std::thread::sleep(std::time::Duration::from_millis(2));
                    continue;
                }
                started = true;
                // Drain concurrently but only STASH candidates: tracing
                // them here races in-flight allocation (VO bit visible
                // before klass init).  FinalMark filters + traces the
                // stash at a safepoint.
                let mut cands: Vec<crate::util::Address> = Vec::new();
                let (drained, wrapped) = t.drain(512, |addr| cands.push(addr));
                if !cands.is_empty() {
                    t.stash_candidates(&mut cands);
                }
                if wrapped {
                    t.reset_cursor();
                }
                if drained == 0 {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
            }
        });
    }
}

/// FinalMark drain: sweep ALL remaining flagged snapshot pages, then
/// conservatively re-enqueue every object of the un-armed spaces
/// (LOS/immortal/nonmoving) — they are treated as live roots, which is
/// exact for immortal spaces and safely conservative for the rest.
pub(super) struct SatbFinalDrain<VM: VMBinding>(std::marker::PhantomData<VM>);

impl<VM: VMBinding> SatbFinalDrain<VM> {
    pub fn new() -> Self {
        SatbFinalDrain(std::marker::PhantomData)
    }
}

impl<VM: VMBinding> GCWork<VM> for SatbFinalDrain<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        let plan = mmtk
            .get_plan()
            .downcast_ref::<ConcurrentImmix<VM>>()
            .unwrap();
        let Some(t) = satb_pages::satb_pages() else {
            return;
        };
        // Quiesce the drainer first: it shares the cursor/flags/stash.
        t.stop_drainer_and_wait();
        t.reset_cursor();
        let mut nodes: Vec<ObjectReference> = Vec::new();
        // Candidates stashed by the concurrent drainer, now safe to filter
        // (safepoint: allocation quiesced, klass writes fenced).
        for addr in t.take_stash() {
            if let Some(o) = satb_node::<VM>(plan, addr) {
                nodes.push(o);
            }
        }
        loop {
            let (_n, wrapped) = t.drain(4096, |addr| {
                if let Some(o) = satb_node::<VM>(plan, addr) {
                    nodes.push(o);
                }
            });
            if nodes.len() >= 8192 {
                let batch = std::mem::take(&mut nodes);
                mmtk.scheduler.work_buckets[WorkBucketStage::Closure].add(ProcessModBufSATB::<
                    VM,
                    ConcurrentImmix<VM>,
                    TRACE_KIND_FAST,
                >::new(batch));
            }
            if wrapped {
                break;
            }
        }
        t.reset_cursor();
        // Un-armed spaces: wholesale rescan as live roots.
        {
            use crate::util::object_enum::ClosureObjectEnumerator;
            let mut enumerator = ClosureObjectEnumerator::<_, VM>::new(|obj| {
                nodes.push(obj);
            });
            plan.common().get_immortal().enumerate_objects(&mut enumerator);
            let mut enumerator = ClosureObjectEnumerator::<_, VM>::new(|obj| {
                nodes.push(obj);
            });
            plan.common().get_los().enumerate_to_space_objects(&mut enumerator);
            let mut enumerator = ClosureObjectEnumerator::<_, VM>::new(|obj| {
                nodes.push(obj);
            });
            plan.common()
                .get_nonmoving()
                .enumerate_objects(&mut enumerator);
        }
        if !nodes.is_empty() {
            mmtk.scheduler.work_buckets[WorkBucketStage::Closure].add(ProcessModBufSATB::<
                VM,
                ConcurrentImmix<VM>,
                TRACE_KIND_FAST,
            >::new(nodes));
        }
    }
}
