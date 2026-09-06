use super::global::GenImmix;
use crate::plan::generational::gc_work::GenNurseryProcessEdges;
use crate::policy::gc_work::TraceKind;
use crate::policy::gc_work::DEFAULT_TRACE;
use crate::scheduler::gc_work::PlanProcessEdges;
use crate::scheduler::gc_work::UnsupportedProcessEdges;
use crate::vm::VMBinding;

pub struct GenImmixNurseryGCWorkContext<VM: VMBinding>(std::marker::PhantomData<VM>);
impl<VM: VMBinding> crate::scheduler::GCWorkContext for GenImmixNurseryGCWorkContext<VM> {
    type VM = VM;
    type PlanType = GenImmix<VM>;
    type DefaultProcessEdges = GenNurseryProcessEdges<VM, Self::PlanType, DEFAULT_TRACE>;
    type PinningProcessEdges = UnsupportedProcessEdges<VM>;
}

pub(super) struct GenImmixMatureGCWorkContext<VM: VMBinding, const KIND: TraceKind>(
    std::marker::PhantomData<VM>,
);
impl<VM: VMBinding, const KIND: TraceKind> crate::scheduler::GCWorkContext
    for GenImmixMatureGCWorkContext<VM, KIND>
{
    type VM = VM;
    type PlanType = GenImmix<VM>;
    type DefaultProcessEdges = PlanProcessEdges<VM, GenImmix<VM>, KIND>;
    type PinningProcessEdges = UnsupportedProcessEdges<VM>;
}

/// Consume the conservative remembered set stashed by `GenImmix::prepare()`:
/// all objects in the LOS/immortal/nonmoving spaces (which have no page dirty
/// tracking), enumerated before the LOS treadmill flipped.  Scans their fields
/// for nursery references.
pub(super) struct ScanDirtyStash<VM: VMBinding> {
    phantom: std::marker::PhantomData<VM>,
}

impl<VM: VMBinding> ScanDirtyStash<VM> {
    pub fn new() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<VM: VMBinding> crate::scheduler::GCWork<VM> for ScanDirtyStash<VM> {
    fn do_work(
        &mut self,
        worker: &mut crate::scheduler::GCWorker<VM>,
        mmtk: &'static crate::MMTK<VM>,
    ) {
        use crate::scheduler::gc_work::ScanObjects;
        use crate::scheduler::{GCWork, WorkBucketStage};
        type E<VM> = GenNurseryProcessEdges<VM, GenImmix<VM>, DEFAULT_TRACE>;

        let plan = mmtk.get_plan().downcast_ref::<GenImmix<VM>>().unwrap();
        let objects = std::mem::take(&mut *plan.dirty_stash.lock().unwrap());
        probe!(mmtk, scan_dirty_stash, objects.len());
        if !objects.is_empty() {
            GCWork::do_work(
                &mut ScanObjects::<E<VM>>::new(objects, false, WorkBucketStage::Closure),
                worker,
                mmtk,
            )
        }
    }
}
