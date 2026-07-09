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

/// Parallel chunk arming for the InitialMark pause: snapshot the
/// alloc-map slice and WP-arm a batch of chunks.  Chunk batches are
/// disjoint (slices, registration set and armed list are internally
/// synchronized), so batches run concurrently on all GC workers --
/// single-threaded arming measured at ~24ms/cycle, the dominant
/// post-optimization pause cost.
pub(super) struct ArmChunks<VM: VMBinding> {
    chunks: Vec<crate::util::Address>,
    _p: std::marker::PhantomData<VM>,
}

impl<VM: VMBinding> ArmChunks<VM> {
    pub fn new(chunks: Vec<crate::util::Address>) -> Self {
        ArmChunks {
            chunks,
            _p: std::marker::PhantomData,
        }
    }
}

impl<VM: VMBinding> GCWork<VM> for ArmChunks<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        const CHUNK: usize = 4 << 20;
        let Some(t) = satb_pages::satb_pages() else {
            return;
        };
        for &c in &self.chunks {
            t.snapshot_alloc_map_range(c, CHUNK);
            t.arm(c, CHUNK);
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

/// Parallel rescue tracing: validate + trace + guarded transitive scan.
/// Packets share nothing but the (atomic) mark bits: trace_object only
/// enqueues for the NEWLY-marking packet, so each object is scanned once.
pub(super) struct TraceRescued<VM: VMBinding> {
    nodes: Vec<ObjectReference>,
    _p: std::marker::PhantomData<VM>,
}

impl<VM: VMBinding> TraceRescued<VM> {
    pub fn new(nodes: Vec<ObjectReference>) -> Self {
        TraceRescued {
            nodes,
            _p: std::marker::PhantomData,
        }
    }
}

impl<VM: VMBinding> GCWork<VM> for TraceRescued<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        use crate::plan::tracing::SlotIterator;
        use crate::plan::PlanTraceObject;
        use crate::vm::slot::Slot;

        let plan = mmtk
            .get_plan()
            .downcast_ref::<ConcurrentImmix<VM>>()
            .unwrap();
        let Some(t) = satb_pages::satb_pages() else {
            return;
        };
        let los = plan.common().get_los();
        let immortal = plan.common().get_immortal();
        let nonmoving = plan.common().get_nonmoving();
        let in_any = |a: crate::util::Address| {
            if !a.is_aligned_to(8) {
                return false;
            }
            if plan.immix_space.address_in_space(a) {
                t.in_alloc_map(a) && t.klass_plausible(a)
            } else if los.address_in_space(a)
                || immortal.address_in_space(a)
                || nonmoving.address_in_space(a)
            {
                t.klass_plausible(a)
            } else {
                false
            }
        };
        let mut queue = CollectQueue(Vec::new());
        for o in std::mem::take(&mut self.nodes) {
            if in_any(o.to_raw_address()) {
                plan.trace_object::<CollectQueue, TRACE_KIND_FAST>(&mut queue, o, worker);
            }
        }
        while let Some(obj) = queue.0.pop() {
            let mut children: Vec<ObjectReference> = Vec::new();
            SlotIterator::<VM>::iterate_fields(obj, worker.tls.0, |s| {
                if let Some(sa) = s.slot_address() {
                    if crate::mmtk::SFT_MAP.get_checked(sa).name()
                        == crate::policy::sft::EMPTY_SFT_NAME
                    {
                        return;
                    }
                }
                if let Some(c) = s.load() {
                    children.push(c);
                }
            });
            for c in children {
                if in_any(c.to_raw_address()) {
                    plan.trace_object::<CollectQueue, TRACE_KIND_FAST>(&mut queue, c, worker);
                }
            }
            plan.post_scan_object(obj);
        }
    }
}

impl<VM: VMBinding> GCWork<VM> for SatbFinalDrain<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        use crate::plan::tracing::SlotIterator;
        use crate::vm::slot::Slot;

        let plan = mmtk
            .get_plan()
            .downcast_ref::<ConcurrentImmix<VM>>()
            .unwrap();
        let Some(t) = satb_pages::satb_pages() else {
            return;
        };
        if satb_pages::satb_verify() {
            t.reset_cursor();
            return;
        }
        let dbg = std::env::var_os("MMTK_SATB_COMMS").is_some();
        let mut guard = satb_pages::DRAIN_STATE.lock().unwrap();
        let base = t.heap_base();

        if guard.is_none() {
            // FIRST FIRING this pause: collect flags, init state, spawn
            // the wholesale rescan as a parallel packet, re-arm self.
            t.stop_drainer_and_wait();
            t.reset_cursor();
            let mut flagged: Vec<usize> = Vec::new();
            t.sweep_flags(|idx| flagged.push(idx));
            let npages = t.heap_span() >> satb_pages::LOG_BYTES_IN_PAGE;
            let mut flag_bm = vec![0u64; npages.div_ceil(64)];
            for &idx in &flagged {
                flag_bm[idx >> 6] |= 1 << (idx & 63);
            }
            let ngrains = t.heap_span() >> 3;
            *guard = Some(satb_pages::DrainState {
                flagged,
                flag_bm,
                extracted_bm: vec![0u64; ngrains.div_ceil(64)],
                extracted_count: 0,
                rounds: 0,
                t0: std::time::Instant::now(),
            });
            drop(guard);
            {
                use crate::util::object_enum::ClosureObjectEnumerator;
                let mut roots: Vec<ObjectReference> = Vec::new();
                let mut en = ClosureObjectEnumerator::<_, VM>::new(|obj| roots.push(obj));
                plan.common().get_immortal().enumerate_objects(&mut en);
                let mut en = ClosureObjectEnumerator::<_, VM>::new(|obj| roots.push(obj));
                plan.common().get_los().enumerate_to_space_objects(&mut en);
                if dbg {
                    eprintln!("[drain] wholesale roots={}", roots.len());
                }
                if !roots.is_empty() {
                    mmtk.scheduler.work_buckets[WorkBucketStage::Closure]
                        .add(TraceRescued::<VM>::new(roots));
                }
            }
            mmtk.scheduler.work_buckets[WorkBucketStage::Closure]
                .set_sentinel(Box::new(SatbFinalDrain::<VM>::new()));
            return;
        }

        // SUBSEQUENT FIRING: one extraction round over the quiesced mark
        // set (see the fixpoint-soundness comment history: mark-start
        // reachability is covered inductively; only live+snapshot objects
        // are ever iterated).
        let st = guard.as_mut().unwrap();
        st.rounds += 1;
        let mut round_nodes: Vec<ObjectReference> = Vec::new();
        for &idx in &st.flagged {
            let pstart = base + (idx << satb_pages::LOG_BYTES_IN_PAGE);
            let snap = t.snapshot_page(idx);
            let mut objs: Vec<ObjectReference> = Vec::new();
            {
                let extracted_bm = &mut st.extracted_bm;
                let count = &mut st.extracted_count;
                let mut consider = |a: crate::util::Address| {
                    let Some(o) = ObjectReference::from_raw_address(a) else {
                        return;
                    };
                    let g = (a - base) >> 3;
                    if (extracted_bm[g >> 6] >> (g & 63)) & 1 == 1 {
                        return;
                    }
                    let sft = crate::mmtk::SFT_MAP.get_checked(a);
                    if sft.name() == crate::policy::sft::EMPTY_SFT_NAME {
                        return;
                    }
                    if !sft.is_live(o) {
                        return;
                    }
                    extracted_bm[g >> 6] |= 1 << (g & 63);
                    *count += 1;
                    objs.push(o);
                };
                if let Some(h) = t.prev_start(pstart, 64 << 20) {
                    consider(h);
                }
                t.alloc_map_starts(pstart, satb_pages::BYTES_IN_PAGE, |a| consider(a));
            }
            let flag_bm = &st.flag_bm;
            for obj in objs {
                SlotIterator::<VM>::iterate_fields(obj, worker.tls.0, |s| {
                    let Some(sa) = s.slot_address() else { return };
                    if sa < base || sa >= base + t.heap_span() {
                        return;
                    }
                    let pi = t.page_index_of(sa);
                    let v: u32 = if pi == idx {
                        unsafe { *(snap.add(sa - pstart) as *const u32) }
                    } else if pi != usize::MAX && (flag_bm[pi >> 6] >> (pi & 63)) & 1 == 1 {
                        unsafe {
                            *(t.snapshot_page(pi).add(
                                sa - (base + (pi << satb_pages::LOG_BYTES_IN_PAGE)),
                            ) as *const u32)
                        }
                    } else {
                        if crate::mmtk::SFT_MAP.get_checked(sa).name()
                            == crate::policy::sft::EMPTY_SFT_NAME
                        {
                            return;
                        }
                        match s.load() {
                            Some(o) => {
                                round_nodes.push(o);
                                return;
                            }
                            None => return,
                        }
                    };
                    if v == 0 {
                        return;
                    }
                    let addr = unsafe { crate::util::Address::from_usize(v as usize) };
                    if let Some(o) = ObjectReference::from_raw_address(addr) {
                        round_nodes.push(o);
                    }
                });
            }
        }
        if round_nodes.is_empty() {
            if dbg {
                eprintln!(
                    "[drain] fixpoint rounds={} extracted={} flagged={} drain={}us",
                    st.rounds,
                    st.extracted_count,
                    st.flagged.len(),
                    st.t0.elapsed().as_micros()
                );
            }
            *guard = None;
            t.reset_cursor();
            return;
        }
        let per = round_nodes.len().div_ceil(8).max(1);
        for batch in round_nodes.chunks(per) {
            mmtk.scheduler.work_buckets[WorkBucketStage::Closure]
                .add(TraceRescued::<VM>::new(batch.to_vec()));
        }
        drop(guard);
        mmtk.scheduler.work_buckets[WorkBucketStage::Closure]
            .set_sentinel(Box::new(SatbFinalDrain::<VM>::new()));
    }
}
