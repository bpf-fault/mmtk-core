use super::global::GenImmix;
use crate::plan::generational::gc_work::GenNurseryProcessEdges;
use crate::plan::generational::global::GenerationalPlan;
use crate::policy::gc_work::TraceKind;
use crate::policy::gc_work::DEFAULT_TRACE;
use crate::scheduler::gc_work::{PlanProcessEdges, ProcessEdgesWork, ScanObjects, UnsupportedProcessEdges};
use crate::scheduler::{GCWork, GCWorker, WorkBucketStage};
use crate::util::linear_scan::Region;
use crate::util::ObjectReference;
use crate::vm::VMBinding;
use crate::MMTK;
use std::collections::HashSet;
use std::marker::PhantomData;

pub struct GenImmixNurseryGCWorkContext<VM: VMBinding>(std::marker::PhantomData<VM>);
impl<VM: VMBinding> crate::scheduler::GCWorkContext for GenImmixNurseryGCWorkContext<VM> {
    type VM = VM;
    type PlanType = GenImmix<VM>;
    type DefaultProcessEdges = GenNurseryProcessEdges<VM, Self::PlanType, DEFAULT_TRACE>;
    type PinningProcessEdges = UnsupportedProcessEdges<VM>;
}

pub struct ProcessDirtyBlocks<E: ProcessEdgesWork> {
    phantom: PhantomData<E>,
}

impl<E: ProcessEdgesWork> ProcessDirtyBlocks<E> {
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<E: ProcessEdgesWork> GCWork<E::VM> for ProcessDirtyBlocks<E> {
    fn do_work(&mut self, worker: &mut GCWorker<E::VM>, mmtk: &'static MMTK<E::VM>) {
        let plan = mmtk.get_plan().downcast_ref::<GenImmix<E::VM>>().unwrap();
        if !plan.is_current_gc_nursery() {
            return;
        }
        if !plan
            .uffd_wp_tracker
            .remembered_set_mode()
            .uses_dirty_block_scanning()
        {
            return;
        }

        let dirty_blocks = plan.uffd_wp_tracker.take_current_gc_dirty_blocks();
        if std::env::var_os("MMTK_TRACE_UFFD_WP_RS_COMPARE").is_some()
            || std::env::var_os("MMTK_TRACE_RS_METRICS").is_some()
        {
            eprintln!(
                "MMTK UFFD WP dirty scan: dirty_blocks={}",
                dirty_blocks.len()
            );
        }
        if dirty_blocks.is_empty() {
            return;
        }

        let mut seen = HashSet::<ObjectReference>::new();
        let mut objects = vec![];
        for block_start in dirty_blocks {
            collect_objects_overlapping_block::<E::VM>(block_start, &mut seen, &mut objects);
        }

        if std::env::var_os("MMTK_TRACE_UFFD_WP_RS_COMPARE").is_some()
            || std::env::var_os("MMTK_TRACE_RS_METRICS").is_some()
        {
            eprintln!(
                "MMTK UFFD WP dirty scan: scanned_objects={}",
                objects.len()
            );
        }

        if plan.uffd_wp_tracker.remembered_set_mode().uses_dirty_block_scanning() {
            for object in &objects {
                plan.uffd_wp_tracker
                    .record_shadow_dirty_object(object.to_raw_address().as_usize());
            }
        }

        if !objects.is_empty() {
            GCWork::do_work(
                &mut ScanObjects::<E>::new(objects, false, WorkBucketStage::Closure),
                worker,
                mmtk,
            );
        }
    }
}

fn collect_objects_overlapping_block<VM: VMBinding>(
    block_start: usize,
    seen: &mut HashSet<ObjectReference>,
    objects: &mut Vec<ObjectReference>,
) {
    #[cfg(not(feature = "vo_bit"))]
    {
        let _ = block_start;
        let _ = seen;
        let _ = objects;
        return;
    }

    #[cfg(feature = "vo_bit")]
    {
        use crate::util::metadata::{side_metadata::spec_defs::VO_BIT, vo_bit};
        use crate::vm::ObjectModel;

        let block_start = unsafe { crate::util::Address::from_usize(block_start) };
        let block_end = block_start + crate::policy::immix::block::Block::BYTES;

        if let Some(first) = crate::memory_manager::find_object_from_internal_pointer(
            block_start,
            crate::policy::immix::block::Block::BYTES,
        ) {
            let first_start = first.to_object_start::<VM>();
            let first_end = first_start + VM::VMObjectModel::get_current_size(first);
            if first_end > block_start && first_start < block_end && seen.insert(first) {
                objects.push(first);
            }
        }

        VO_BIT.scan_non_zero_values::<u8>(block_start, block_end, &mut |address| {
            let object = vo_bit::get_object_ref_for_vo_addr(address);
            if seen.insert(object) {
                objects.push(object);
            }
        });
    }
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
