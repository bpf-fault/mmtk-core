#[cfg(feature = "uffd")]
use crate::plan::barriers::SATBBarrier;
use crate::plan::compressor::Compressor;
#[cfg(feature = "uffd")]
use crate::plan::concurrent::barrier::SATBBarrierSemantics;
#[cfg(feature = "uffd")]
use crate::plan::concurrent::Pause;
use crate::plan::mutator_context::common_prepare_func;
use crate::plan::mutator_context::Mutator;
use crate::plan::mutator_context::MutatorBuilder;
use crate::plan::mutator_context::MutatorConfig;
use crate::plan::mutator_context::{
    common_release_func, create_allocator_mapping, create_space_mapping, ReservedAllocators,
};
use crate::plan::AllocationSemantics;
use crate::util::alloc::allocators::AllocatorSelector;
use crate::util::alloc::BumpAllocator;
use crate::util::{VMMutatorThread, VMWorkerThread};
use crate::vm::VMBinding;
use crate::MMTK;
use enum_map::{enum_map, EnumMap};

const RESERVED_ALLOCATORS: ReservedAllocators = ReservedAllocators {
    n_bump_pointer: 1,
    ..ReservedAllocators::DEFAULT
};

lazy_static! {
    /// When compressor_single_space is enabled, force all allocations to go to the default allocator and space.
    static ref ALLOCATOR_MAPPING_SINGLE_SPACE: EnumMap<AllocationSemantics, AllocatorSelector> = enum_map! {
        _ => AllocatorSelector::BumpPointer(0),
    };
    pub static ref ALLOCATOR_MAPPING: EnumMap<AllocationSemantics, AllocatorSelector> = {
        if cfg!(feature = "compressor_single_space") {
            *ALLOCATOR_MAPPING_SINGLE_SPACE
        } else {
            let mut map = create_allocator_mapping(RESERVED_ALLOCATORS, true);
            map[AllocationSemantics::Default] = AllocatorSelector::BumpPointer(0);
            map
        }
    };
}

#[cfg(feature = "uffd")]
type BarrierSemanticsType<VM> =
    SATBBarrierSemantics<VM, Compressor<VM>, { crate::policy::compressor::TRACE_KIND_MARK }>;
#[cfg(feature = "uffd")]
type BarrierType<VM> = SATBBarrier<BarrierSemanticsType<VM>>;

pub fn create_compressor_mutator<VM: VMBinding>(
    mutator_tls: VMMutatorThread,
    mmtk: &'static MMTK<VM>,
) -> Mutator<VM> {
    let plan = mmtk.get_plan().downcast_ref::<Compressor<VM>>().unwrap();
    let config = MutatorConfig {
        allocator_mapping: &ALLOCATOR_MAPPING,
        space_mapping: Box::new({
            let mut vec = create_space_mapping(
                RESERVED_ALLOCATORS,
                !cfg!(feature = "compressor_single_space"),
                plan,
            );
            vec.push((AllocatorSelector::BumpPointer(0), &plan.compressor_space));
            vec
        }),
        prepare_func: &compressor_mutator_prepare,
        release_func: &compressor_mutator_release,
    };

    let builder = MutatorBuilder::new(mutator_tls, mmtk, config);

    #[cfg(feature = "uffd")]
    let mut mutator = builder
        .barrier(Box::new(SATBBarrier::new(BarrierSemanticsType::<VM>::new(
            mmtk,
            mutator_tls,
        ))))
        .build();

    #[cfg(not(feature = "uffd"))]
    let mutator = builder.build();

    #[cfg(feature = "uffd")]
    mutator
        .barrier
        .downcast_mut::<BarrierType<VM>>()
        .unwrap()
        .set_weak_ref_barrier_enabled(plan.is_concurrent_marking_active());

    mutator
}

pub fn compressor_mutator_prepare<VM: VMBinding>(mutator: &mut Mutator<VM>, tls: VMWorkerThread) {
    common_prepare_func(mutator, tls);

    let bump_allocator = unsafe {
        mutator
            .allocators
            .get_allocator_mut(mutator.config.allocator_mapping[AllocationSemantics::Default])
    }
    .downcast_mut::<BumpAllocator<VM>>()
    .unwrap();
    bump_allocator.reset();

    #[cfg(feature = "uffd")]
    {
        let current_pause = mutator.plan.concurrent().unwrap().current_pause().unwrap();
        if current_pause == Pause::InitialMark {
            mutator
                .barrier
                .downcast_mut::<BarrierType<VM>>()
                .unwrap()
                .set_weak_ref_barrier_enabled(true);
        }
    }
}

pub fn compressor_mutator_release<VM: VMBinding>(mutator: &mut Mutator<VM>, tls: VMWorkerThread) {
    // reset the thread-local allocation bump pointer
    let bump_allocator = unsafe {
        mutator
            .allocators
            .get_allocator_mut(mutator.config.allocator_mapping[AllocationSemantics::Default])
    }
    .downcast_mut::<BumpAllocator<VM>>()
    .unwrap();
    bump_allocator.reset();

    #[cfg(feature = "uffd")]
    {
        let current_pause = mutator.plan.concurrent().unwrap().current_pause().unwrap();
        debug_assert_ne!(current_pause, Pause::InitialMark);
        if current_pause == Pause::Full || current_pause == Pause::FinalMark {
            mutator
                .barrier
                .downcast_mut::<BarrierType<VM>>()
                .unwrap()
                .set_weak_ref_barrier_enabled(false);
        }
    }

    common_release_func(mutator, tls);
}
