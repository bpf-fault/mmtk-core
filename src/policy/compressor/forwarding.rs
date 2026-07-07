use crate::util::constants::BYTES_IN_WORD;
use crate::util::linear_scan::{Region, RegionIterator};
use crate::util::metadata::side_metadata::spec_defs::{
    COMPRESSOR_MARK, COMPRESSOR_OFFSET_VECTOR, COMPRESSOR_REFBITS,
};
use crate::util::metadata::side_metadata::SideMetadataSpec;
use crate::util::{Address, ObjectReference};
use crate::vm::object_model::ObjectModel;
use crate::vm::VMBinding;
use atomic::Ordering;
use std::marker::PhantomData;
use std::sync::atomic::AtomicBool;

/// A [`CompressorRegion`] is the granularity at which [`super::CompressorSpace`]
/// compacts the heap. Objects are allocated inside one region, and are only ever
/// moved *within* that region.
#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub(crate) struct CompressorRegion(Address);
impl Region for CompressorRegion {
    const LOG_BYTES: usize = 20; // 1 MiB
    fn from_aligned_address(address: Address) -> Self {
        assert!(
            address.is_aligned_to(Self::BYTES),
            "{address} is not aligned"
        );
        CompressorRegion(address)
    }
    fn start(&self) -> Address {
        self.0
    }
}

/// A finite-state machine which visits the positions of marked bits in
/// the mark bitmap, and accumulates the size of live data that it has
/// seen between marked bits.
///
/// The Compressor caches the state of the transducer at the start of
/// each block by serialising the state using [`Transducer::encode`], and
/// then deserialises the state whilst computing forwarding pointers
/// using [`Transducer::decode`].
#[derive(Debug)]
struct Transducer {
    /// The address for the next object to be copied to, following preceding
    /// objects which were visited by the transducer.
    to: Address,
    /// The address of the last mark bit which the transducer visited.
    last_bit_visited: Address,
    /// Whether or not the transducer is currently inside an object
    /// (i.e. if it has seen a first bit but no matching last bit yet).
    in_object: bool,
}
impl Transducer {
    pub fn new(to: Address) -> Self {
        Self {
            to,
            last_bit_visited: Address::ZERO,
            in_object: false,
        }
    }
    pub fn live_end(&self) -> Address {
        self.to
    }

    pub fn visit_mark_bit(&mut self, address: Address) {
        if self.in_object {
            // The size of an object is the distance between the end and
            // start of the object, and the last word of the object is one
            // word prior to the end of the object. Thus we must add an
            // extra word, in order to compute the size of the object based
            // on the distance between its first and last words.
            let first_word = self.last_bit_visited;
            let last_word = address;
            let size = last_word - first_word + BYTES_IN_WORD;
            self.to += size;
        }
        self.in_object = !self.in_object;
        self.last_bit_visited = address;
    }

    pub fn encode(&self, current_position: Address) -> usize {
        if self.in_object {
            // We count the space between the last mark bit and
            // the current address as live when we stop in the
            // middle of an object.
            self.to.as_usize() + (current_position - self.last_bit_visited) + 1
        } else {
            self.to.as_usize()
        }
    }

    pub fn decode(offset: usize, current_position: Address) -> Self {
        Transducer {
            to: unsafe { Address::from_usize(offset & !1) },
            last_bit_visited: current_position,
            in_object: (offset & 1) == 1,
        }
    }
}

pub struct ForwardingMetadata<VM: VMBinding> {
    calculated: AtomicBool,
    vm: PhantomData<VM>,
}

// A block in the Compressor is the granularity at which we cache
// the amount of live data preceding an address. We set it to 512 bytes
// following the paper.
#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub(crate) struct Block(Address);
impl Region for Block {
    const LOG_BYTES: usize = 9;
    fn from_aligned_address(address: Address) -> Self {
        assert!(address.is_aligned_to(Self::BYTES));
        Block(address)
    }
    fn start(&self) -> Address {
        self.0
    }
}

pub(crate) const MARK_SPEC: SideMetadataSpec = COMPRESSOR_MARK;
pub(crate) const OFFSET_VECTOR_SPEC: SideMetadataSpec = COMPRESSOR_OFFSET_VECTOR;
pub(crate) const REFBITS_SPEC: SideMetadataSpec = COMPRESSOR_REFBITS;

/// Mark `slot` (a reference field's address) as a reference word in the
/// Class B v2 reference bitmap, so the in-kernel fixup handler can forward it.
///
/// NOTE (integration, 2026-06-13): not yet wired into the staging flow. The
/// B.1 `stage_region_idx` path forwards via `update_references` on *aliased*
/// objects (shifted into the staging arena), whose addresses fall outside
/// this side-metadata's mapped range — so the bitmap must instead live in the
/// staging arena and be populated at staged positions (see class-b-design.md).
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn mark_reference_slot(slot: Address) {
    REFBITS_SPEC.store_atomic::<u8>(slot, 1, Ordering::Relaxed);
}

impl<VM: VMBinding> ForwardingMetadata<VM> {
    pub fn new() -> ForwardingMetadata<VM> {
        ForwardingMetadata {
            calculated: AtomicBool::new(false),
            vm: PhantomData,
        }
    }

    pub fn mark_last_word_of_object(&self, object: ObjectReference) {
        let last_word_of_object = object.to_object_start::<VM>()
            + VM::VMObjectModel::get_current_size(object)
            - BYTES_IN_WORD;
        #[cfg(debug_assertions)]
        {
            // We require to be able to iterate upon first and last bits in the
            // same bitmap. Therefore the first and last bits cannot be the
            // same, else we would only encounter one of the two bits.
            // This requirement implies that objects must be at least two words
            // large.
            debug_assert!(
                MARK_SPEC.are_different_metadata_bits(
                    object.to_object_start::<VM>(),
                    last_word_of_object
                ),
                "The first and last mark bits should be different bits."
            );
        }

        // We only mark the last word as input to computing forwarding
        // information, so relaxed consistency is okay.
        MARK_SPEC.fetch_or_atomic::<u8>(last_word_of_object, 1, Ordering::Relaxed);
    }

    /// Returns the final transducer position: the exact post-compact end
    /// of live data in the region.
    pub fn calculate_offset_vector(&self, region: CompressorRegion, cursor: Address) -> Address {
        let mut state = Transducer::new(region.start());
        let first_block = Block::from_aligned_address(region.start());
        let last_block = Block::from_aligned_address(cursor);
        // Class B v2: while we visit each object's start/end mark bits, also
        // record old->new in the forward table for the in-kernel handler.
        // R1: additionally emit the live-word bitmap (old positions) and the
        // per-page first-source index, so the handler can build pages from
        // un-slid from-space.
        let cf = if crate::util::compact_faults::defer_forward() {
            crate::util::compact_faults::compact_faults()
        } else {
            None
        };
        let r1 = crate::util::compact_faults::inkernel_compact();
        if r1 {
            // Live bits are only ever OR'd in; clear this region's slice
            // before re-recording, or stale bits from the previous cycle
            // make the in-kernel build emit dead words.
            if let Some(cf) = cf {
                cf.clear_live_words(region.start(), region.end() - region.start());
            }
        }
        let mut obj_ostart = Address::ZERO;
        let mut obj_nstart = Address::ZERO;
        for block in RegionIterator::<Block>::new(first_block, last_block) {
            OFFSET_VECTOR_SPEC.store_atomic::<usize>(
                block.start(),
                state.encode(block.start()),
                Ordering::Relaxed,
            );
            MARK_SPEC.scan_non_zero_values::<u8>(
                block.start(),
                block.end(),
                &mut |addr: Address| {
                    // A start bit transitions in_object false->true; at that
                    // point state.to is this object's post-compact start.
                    let starting = !state.in_object;
                    if starting {
                        obj_ostart = addr;
                        obj_nstart = state.to;
                    }
                    state.visit_mark_bit(addr);
                    if starting {
                        if let Some(cf) = cf {
                            cf.set_fwd(addr, state.to);
                        }
                    } else if r1 {
                        // end bit: object occupies old [obj_ostart, addr],
                        // i.e. words [obj_ostart..=addr]; new start obj_nstart.
                        if let Some(cf) = cf {
                            let mut w = obj_ostart;
                            while w <= addr {
                                cf.set_live_word(w);
                                w += BYTES_IN_WORD;
                            }
                            // first_src for to-space page boundaries the object's
                            // new range crosses.
                            let nwords = (addr - obj_ostart) / BYTES_IN_WORD + 1;
                            cf.record_first_src(obj_ostart, obj_nstart, nwords);
                        }
                    }
                },
            );
        }
        self.calculated.store(true, Ordering::Relaxed);
        state.live_end()
    }

    pub fn release(&self) {
        self.calculated.store(false, Ordering::Relaxed);
    }

    pub fn forward(&self, address: Address) -> Address {
        debug_assert!(
            self.calculated.load(Ordering::Relaxed),
            "forward() should only be called when we have calculated an offset vector"
        );
        let block = Block::from_unaligned_address(address);
        let mut state = Transducer::decode(
            OFFSET_VECTOR_SPEC.load_atomic::<usize>(block.start(), Ordering::Relaxed),
            block.start(),
        );
        // The transducer in this implementation computes the final
        // address of an object; whereas Total-Live-Data in the paper computes
        // the distance of the object from the start of the block.
        MARK_SPEC.scan_non_zero_values::<u8>(block.start(), address, &mut |addr: Address| {
            state.visit_mark_bit(addr)
        });
        state.to
    }

    pub fn scan_marked_objects(
        &self,
        start: Address,
        end: Address,
        f: &mut impl FnMut(ObjectReference),
    ) {
        let mut in_object = false;
        MARK_SPEC.scan_non_zero_values::<u8>(start, end, &mut |addr: Address| {
            if !in_object {
                let object = ObjectReference::from_raw_address(addr).unwrap();
                f(object);
            }
            in_object = !in_object;
        });
    }

    pub fn has_calculated_forwarding_addresses(&self) -> bool {
        self.calculated.load(Ordering::Relaxed)
    }
}
