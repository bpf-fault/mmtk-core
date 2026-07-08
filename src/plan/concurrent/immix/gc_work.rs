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
        // VERIFY mode: pure passive arming — the compiled barrier does
        // all SATB work; skip the drain entirely.  (Classification served
        // its diagnostic purpose; the drain's liveness certificate is
        // additionally unsound under lazy sweeping DURING the cycle —
        // objects in the InitialMark VO snapshot can be swept+recycled
        // mid-mark, so iterating them crashes.  Fix separately for real
        // mode: needs sweep-fence or mark-state filtering.)
        if satb_pages::satb_verify() {
            t.reset_cursor();
            return;
        }
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
        let dbg = std::env::var_os("MMTK_SATB_COMMS").is_some();
        let mut flagged: Vec<usize> = Vec::new();
        t.sweep_flags(|idx| flagged.push(idx));
        if dbg {
            eprintln!("[drain] flagged={}", flagged.len());
        }
        let in_snap = |addr: crate::util::Address, flagged: &[usize], t: &satb_pages::SatbPages| {
            // flagged is sorted (sweep order); binary search page index
            flagged.binary_search(&t.page_index_of(addr)).is_ok()
        };
        let mut nodes: Vec<ObjectReference> = Vec::new();
        let base = t.heap_base();
        let trace = std::env::var_os("MMTK_SATB_TRACE").is_some();
        for &idx in &flagged {
            let pstart = base + (idx << satb_pages::LOG_BYTES_IN_PAGE);
            if trace {
                eprintln!("[xt] page {:x}", pstart.as_usize());
            }
            let snap = t.snapshot_page(idx);
            // objects starting on this page, plus the spanning head:
            // the nearest alloc-map start within MAX_OBJ before the page
            // whose extent reaches into it.
            // Iterate an object only if it exists at mark start (alloc
            // snapshot) AND still exists now (current VO bit): lazy sweep
            // reclaims previous-cycle-dead objects DURING this cycle, so
            // snapshot membership alone admits recycled memory (measured
            // wild-slot crashes).  Mark-start-LIVE objects cannot be
            // swept mid-cycle, so current-VO loses no SATB obligation.
            let alive = |a: crate::util::Address| -> Option<ObjectReference> {
                let o = ObjectReference::from_raw_address(a)?;
                #[cfg(feature = "vo_bit")]
                if !crate::util::metadata::vo_bit::is_vo_bit_set(o) {
                    return None;
                }
                Some(o)
            };
            let mut objs: Vec<ObjectReference> = Vec::new();
            // spanning head: LOS objects reach megabytes, so search far
            if let Some(h) = t.prev_start(pstart, 64 << 20) {
                if let Some(o) = alive(h) {
                    objs.push(o);
                }
            }
            t.alloc_map_starts(pstart, satb_pages::BYTES_IN_PAGE, |a| {
                if let Some(o) = alive(a) {
                    objs.push(o);
                }
            });
            for obj in objs {
                if trace {
                    eprintln!("[xt]   obj {:?}", obj);
                }
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
        // VERIFY mode: classify instead of tracing — the compiled
        // barrier is handling correctness; report the extractor's output
        // quality (valid+marked / valid+unmarked / garbage samples).
        if satb_pages::satb_verify() {
            let (mut ok_marked, mut ok_unmarked, mut garbage) = (0u64, 0u64, 0u64);
            let mut samples: Vec<String> = Vec::new();
            for o in &nodes {
                let a = o.to_raw_address();
                let in_immix = plan.immix_space.address_in_space(a);
                let valid = if in_immix {
                    t.in_alloc_map(a)
                } else {
                    // non-immix: chunk-of-space check only
                    true
                };
                if !valid {
                    garbage += 1;
                    if samples.len() < 8 {
                        samples.push(format!("garbage {:?}", o));
                    }
                    continue;
                }
                #[cfg(feature = "vo_bit")]
                if in_immix && !crate::util::metadata::vo_bit::is_vo_bit_set(*o) {
                    garbage += 1;
                    if samples.len() < 8 {
                        samples.push(format!("no-vo {:?}", o));
                    }
                    continue;
                }
                // marked = the barrier/tracer already reached it
                if plan.immix_space.is_marked(*o) {
                    ok_marked += 1;
                } else {
                    ok_unmarked += 1;
                    if samples.len() < 8 {
                        samples.push(format!("unmarked {:?}", o));
                    }
                }
            }
            eprintln!(
                "[satbverify] extracted={} marked={} unmarked={} garbage={}",
                nodes.len(),
                ok_marked,
                ok_unmarked,
                garbage
            );
            for s in samples {
                eprintln!("[satbverify]   {}", s);
            }
            t.reset_cursor();
            return;
        }
        if dbg {
            eprintln!("[drain] extracted nodes={}", nodes.len());
        }
        // Filter + trace IMMIX refs only, validated as mark-start object
        // starts (alloc snapshot).  Non-immix refs are DROPPED: the
        // wholesale rescan below already retains every LOS/immortal/
        // nonmoving object as a live root, so snapshot refs into those
        // spaces are redundant -- and extracted values can be garbage
        // when a mark-start object's start was recycled mid-cycle (live
        // klass iterated over old snapshot content), which panics
        // vm_trace_object for non-space addresses (measured).
        // Validation per space: immix by mark-start alloc snapshot; LOS/
        // immortal/nonmoving by space membership (SFT chunk-accurate for
        // discontiguous spaces, safe on arbitrary addresses) + current VO
        // bit (metadata mapped once membership holds).  LOS refs MUST be
        // traced: the wholesale rescan enumerates only to_space (already-
        // traced objects), so an LOS object whose only mark-start ref was
        // an overwritten immix slot is rescued exactly here.  Anything
        // else (VM space, garbage) is dropped -- tracing a non-space
        // address panics vm_trace_object.
        let los = plan.common().get_los();
        let immortal = plan.common().get_immortal();
        let nonmoving = plan.common().get_nonmoving();
        let vo_ok = |o: ObjectReference| -> bool {
            #[cfg(feature = "vo_bit")]
            return crate::util::metadata::vo_bit::is_vo_bit_set(o);
            #[cfg(not(feature = "vo_bit"))]
            return true;
        };
        // Collect VALIDATED refs and hand them UNTRACED to the packet:
        // ProcessModBufSATB's tracer performs mark-test-and-scan, and it
        // only SCANS objects it newly marks -- pre-tracing here would
        // mark them without scanning, so their children would never be
        // traced (measured as under-retention: reachable objects swept,
        // Java-level BootstrapMethodError fallout).
        let mut rescued: Vec<ObjectReference> = Vec::new();
        for o in nodes {
            let a = o.to_raw_address();
            // Alignment first: raw snapshot data can decode to misaligned
            // values that ALIAS a genuine start's alloc-map bit (8-byte
            // bit granularity: 0x...01 shares the bit with 0x...00) --
            // measured as a garbage ObjectReference traced into the LOS
            // treadmill and crashing later enumeration.
            if !a.is_aligned_to(8) {
                continue;
            }
            let keep = if plan.immix_space.address_in_space(a) {
                t.in_alloc_map(a)
            } else if los.address_in_space(a)
                || immortal.address_in_space(a)
                || nonmoving.address_in_space(a)
            {
                vo_ok(o)
            } else {
                false
            };
            if keep {
                rescued.push(o);
            }
        }
        // VALIDATING closure over the rescued set: extraction sources
        // include intact-dead objects (stale VO under lazy sweep), whose
        // slot values can be stale; ProcessModBufSATB scans children
        // unvalidated, so a single stale ref reaches iterate_fields on
        // recycled memory (measured).  Validate EVERY hop with the same
        // predicate as the seeds; children failing it are either garbage
        // or allocate-black (already live).  Marks only ever land on
        // validated genuine starts, so the CopyFromMarkBits->VO feedback
        // stays a true allocation map (unlike the conservative-era
        // poisoning).  Intact-dead survivors become bounded floating
        // garbage.
        {
            use crate::plan::tracing::SlotIterator;
            use crate::vm::slot::Slot;
            let validate = |o: ObjectReference| -> bool {
                let a = o.to_raw_address();
                if !a.is_aligned_to(8) {
                    return false;
                }
                if plan.immix_space.address_in_space(a) {
                    t.in_alloc_map(a) && vo_ok(o)
                } else if los.address_in_space(a)
                    || immortal.address_in_space(a)
                    || nonmoving.address_in_space(a)
                {
                    vo_ok(o)
                } else {
                    false
                }
            };
            let mut queue = CollectQueue(Vec::new());
            for o in rescued {
                if validate(o) {
                    plan.trace_object::<CollectQueue, TRACE_KIND_FAST>(&mut queue, o, worker);
                }
            }
            // Wholesale rescan, split by seed liveness certainty:
            //  - immortal (live by definition) and LOS to_space (traced
            //    this cycle => live): children are genuine current refs;
            //    scan them in PARALLEL via ProcessModBufSATB.
            //  - nonmoving (an immix-like space whose enumeration
            //    includes INTACT-DEAD objects with stale slot values):
            //    seed THIS validating closure so children are checked.
            {
                use crate::util::object_enum::ClosureObjectEnumerator;
                let mut live_roots: Vec<ObjectReference> = Vec::new();
                let mut en = ClosureObjectEnumerator::<_, VM>::new(|obj| {
                    live_roots.push(obj);
                });
                plan.common().get_immortal().enumerate_objects(&mut en);
                let mut en = ClosureObjectEnumerator::<_, VM>::new(|obj| {
                    live_roots.push(obj);
                });
                plan.common().get_los().enumerate_to_space_objects(&mut en);
                let mut stale_suspect: Vec<ObjectReference> = Vec::new();
                let mut en = ClosureObjectEnumerator::<_, VM>::new(|obj| {
                    stale_suspect.push(obj);
                });
                plan.common().get_nonmoving().enumerate_objects(&mut en);
                if dbg {
                    eprintln!(
                        "[drain] wholesale live={} suspect={}",
                        live_roots.len(),
                        stale_suspect.len()
                    );
                }
                if !live_roots.is_empty() {
                    mmtk.scheduler.work_buckets[WorkBucketStage::Closure].add(
                        ProcessModBufSATB::<VM, ConcurrentImmix<VM>, TRACE_KIND_FAST>::new(
                            live_roots,
                        ),
                    );
                }
                for o in stale_suspect {
                    plan.trace_object::<CollectQueue, TRACE_KIND_FAST>(&mut queue, o, worker);
                }
            }
            while let Some(obj) = queue.0.pop() {
                let mut children: Vec<ObjectReference> = Vec::new();
                SlotIterator::<VM>::iterate_fields(obj, worker.tls.0, |s| {
                    if let Some(c) = s.load() {
                        children.push(c);
                    }
                });
                for c in children {
                    if validate(c) {
                        plan.trace_object::<CollectQueue, TRACE_KIND_FAST>(&mut queue, c, worker);
                    }
                }
                plan.post_scan_object(obj);
            }
        }
        if dbg {
            eprintln!("[drain] trace-filter done");
        }
        t.reset_cursor();
    }
}
