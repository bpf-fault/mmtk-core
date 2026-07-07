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
    // Validate against the InitialMark VO SNAPSHOT, never the live VO
    // map: our own conservative marks flow into live VO bits at sweep
    // (CopyFromMarkBits), which would let previously-marked garbage
    // self-legitimize (measured: ASCII data acquiring vo=true).
    if !satb_pages::satb_pages().is_some_and(|t| t.in_alloc_map(addr)) {
        return None;
    }
    ObjectReference::from_raw_address(addr)
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


/// FinalMark validating closure for conservative snapshot candidates.
/// Every hop is re-validated (space -> live chunk -> VO bit) before
/// scanning: an intact-dead candidate (stale VO bit under lazy sweep) is
/// itself safely scannable at a safepoint (its klass is valid), but its
/// CHILDREN may point into reused memory -- feeding them unvalidated into
/// the exact tracer was the crash.  Validated-but-dead objects become
/// bounded floating garbage (reclaimed next cycle).
pub(super) struct SatbValidatingTrace<VM: VMBinding> {
    candidates: Vec<crate::util::Address>,
    /// Genuine objects (e.g., the FinalMark wholesale LOS/immortal
    /// rescan) seeded without candidate validation -- but their CHILDREN
    /// still go through validation: dead-at-mark seeds carry garbage
    /// children, and marking those poisons the VO map at the next sweep
    /// (CopyFromMarkBits), which is exactly how ASCII data acquired VO
    /// bits across cycles.
    trusted: Vec<ObjectReference>,
    _p: std::marker::PhantomData<VM>,
}

impl<VM: VMBinding> SatbValidatingTrace<VM> {
    pub fn new(candidates: Vec<crate::util::Address>) -> Self {
        SatbValidatingTrace {
            candidates,
            trusted: Vec::new(),
            _p: std::marker::PhantomData,
        }
    }

    pub fn new_trusted(trusted: Vec<ObjectReference>) -> Self {
        SatbValidatingTrace {
            candidates: Vec::new(),
            trusted,
            _p: std::marker::PhantomData,
        }
    }
}

struct CollectQueue(Vec<ObjectReference>);
impl crate::plan::ObjectQueue for CollectQueue {
    fn enqueue(&mut self, o: ObjectReference) {
        self.0.push(o);
    }
}

impl<VM: VMBinding> GCWork<VM> for SatbValidatingTrace<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        use crate::plan::PlanTraceObject;
        use crate::plan::tracing::SlotIterator;
        use crate::vm::slot::Slot;

        let plan = mmtk
            .get_plan()
            .downcast_ref::<ConcurrentImmix<VM>>()
            .unwrap();
        let mut queue = CollectQueue(Vec::new());
        for addr in std::mem::take(&mut self.candidates) {
            if let Some(obj) = satb_node::<VM>(plan, addr) {
                plan.trace_object::<CollectQueue, TRACE_KIND_FAST>(&mut queue, obj, worker);
            }
        }
        for obj in std::mem::take(&mut self.trusted) {
            plan.trace_object::<CollectQueue, TRACE_KIND_FAST>(&mut queue, obj, worker);
        }
        // Transitive closure with per-hop validation.
        let debug = std::env::var_os("MMTK_SATB_DEBUG").is_some();
        while let Some(obj) = queue.0.pop() {
            if debug {
                eprintln!(
                    "[satbdbg] scan {:?} vo={} chunk_ok={}",
                    obj,
                    crate::util::metadata::vo_bit::is_vo_bit_set(obj),
                    plan.immix_space.address_in_space(obj.to_raw_address())
                );
            }
            let mut children: Vec<ObjectReference> = Vec::new();
            SlotIterator::<VM>::iterate_fields(obj, worker.tls.0, |s| {
                if let Some(t) = s.load() {
                    children.push(t);
                }
            });
            for t in children {
                if satb_node::<VM>(plan, t.to_raw_address()).is_some() {
                    plan.trace_object::<CollectQueue, TRACE_KIND_FAST>(&mut queue, t, worker);
                }
            }
            plan.post_scan_object(obj);
        }
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
    fn do_work(&mut self, worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        use crate::plan::PlanTraceObject;
        use crate::plan::tracing::SlotIterator;
        use crate::vm::slot::Slot;

        let plan = mmtk
            .get_plan()
            .downcast_ref::<ConcurrentImmix<VM>>()
            .unwrap();
        let Some(t) = satb_pages::satb_pages() else {
            return;
        };
        // Quiesce the drainer (it shares flags with this sweep).
        t.stop_drainer_and_wait();
        t.reset_cursor();

        // EXACT snapshot drain.  For every flagged page: enumerate the
        // objects overlapping it from the ALLOC-MAP SNAPSHOT (object
        // starts at mark start; membership certifies liveness at the
        // previous GC, so both the object and its children are intact
        // memory).  Iterate each object's fields via its LIVE layout
        // (klass immutable), reading each slot VALUE from the snapshot
        // copy when the slot's page was snapshotted (mark-start value)
        // and from live memory otherwise (unwritten => identical).
        // Extracted values are genuine mark-start references: no
        // conservative candidates exist in this scheme, so exact marking
        // never marks non-objects and the VO->snapshot induction holds.
        let mut flagged: Vec<usize> = Vec::new();
        t.sweep_flags(|idx| flagged.push(idx));
        let in_snap = |addr: crate::util::Address, flagged: &[usize], t: &satb_pages::SatbPages| {
            // flagged is sorted (sweep order); binary search page index
            flagged.binary_search(&t.page_index_of(addr)).is_ok()
        };
        let mut nodes: Vec<ObjectReference> = Vec::new();
        let base = t.heap_base();
        for &idx in &flagged {
            let pstart = base + (idx << satb_pages::LOG_BYTES_IN_PAGE);
            let snap = t.snapshot_page(idx);
            // objects starting on this page, plus the spanning head:
            // the nearest alloc-map start within MAX_OBJ before the page
            // whose extent reaches into it.
            let mut objs: Vec<ObjectReference> = Vec::new();
            // spanning head: LOS objects reach megabytes, so search far
            if let Some(h) = t.prev_start(pstart, 64 << 20) {
                if let Some(o) = ObjectReference::from_raw_address(h) {
                    objs.push(o);
                }
            }
            t.alloc_map_starts(pstart, satb_pages::BYTES_IN_PAGE, |a| {
                if let Some(o) = ObjectReference::from_raw_address(a) {
                    objs.push(o);
                }
            });
            for obj in objs {
                SlotIterator::<VM>::iterate_fields(obj, worker.tls.0, |s| {
                    let Some(sa) = s.slot_address() else { return };
                    let v: u32 = if t.page_index_of(sa) == idx {
                        // mark-start value from THIS page's snapshot
                        unsafe {
                            *(snap.add(sa - pstart) as *const u32)
                        }
                    } else if sa >= base
                        && sa < base + t.heap_span()
                        && in_snap(sa, &flagged, t)
                    {
                        let oidx = t.page_index_of(sa);
                        unsafe {
                            *(t.snapshot_page(oidx)
                                .add(sa - (base + (oidx << satb_pages::LOG_BYTES_IN_PAGE)))
                                as *const u32)
                        }
                    } else {
                        // unwritten page: live == mark-start
                        match s.load() {
                            Some(o) => {
                                nodes.push(o);
                                return;
                            }
                            None => return,
                        }
                    };
                    // decode unscaled compressed oop (our testbed configs)
                    if v == 0 {
                        return;
                    }
                    let addr = unsafe { crate::util::Address::from_usize(v as usize) };
                    if let Some(o) = ObjectReference::from_raw_address(addr) {
                        nodes.push(o);
                    }
                });
            }
        }
        // Filter + trace: non-immix refs are genuine (kept spaces);
        // immix refs must be mark-start objects (snapshot membership) —
        // post-mark allocations are allocate-black already.
        let mut queue = CollectQueue(Vec::new());
        for o in nodes {
            let a = o.to_raw_address();
            let in_immix = plan.immix_space.address_in_space(a);
            if !in_immix || t.in_alloc_map(a) {
                plan.trace_object::<CollectQueue, TRACE_KIND_FAST>(&mut queue, o, worker);
            }
        }
        // Children of genuine intact objects are genuine and intact:
        // hand them to the normal exact tracer.
        if !queue.0.is_empty() {
            mmtk.scheduler.work_buckets[WorkBucketStage::Closure].add(ProcessModBufSATB::<
                VM,
                ConcurrentImmix<VM>,
                TRACE_KIND_FAST,
            >::new(std::mem::take(&mut queue.0)));
        }
        t.reset_cursor();
        // Un-armed spaces: wholesale rescan as live roots (their children
        // flow through the same exact tracer; see above for why that is
        // safe for objects live at the previous GC — LOS to_space and
        // immortal/nonmoving enumerations satisfy that).
        {
            use crate::util::object_enum::ClosureObjectEnumerator;
            let mut roots: Vec<ObjectReference> = Vec::new();
            let mut enumerator = ClosureObjectEnumerator::<_, VM>::new(|obj| {
                roots.push(obj);
            });
            plan.common().get_immortal().enumerate_objects(&mut enumerator);
            let mut enumerator = ClosureObjectEnumerator::<_, VM>::new(|obj| {
                roots.push(obj);
            });
            plan.common().get_los().enumerate_to_space_objects(&mut enumerator);
            let mut enumerator = ClosureObjectEnumerator::<_, VM>::new(|obj| {
                roots.push(obj);
            });
            plan.common()
                .get_nonmoving()
                .enumerate_objects(&mut enumerator);
            if !roots.is_empty() {
                mmtk.scheduler.work_buckets[WorkBucketStage::Closure].add(ProcessModBufSATB::<
                    VM,
                    ConcurrentImmix<VM>,
                    TRACE_KIND_FAST,
                >::new(roots));
            }
        }
    }
}
