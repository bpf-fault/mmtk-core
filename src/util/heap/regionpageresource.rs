use crate::util::constants::BYTES_IN_PAGE;
use crate::util::heap::layout::VMMap;
use crate::util::heap::pageresource::{CommonPageResource, PRAllocFail, PRAllocResult};
use crate::util::heap::space_descriptor::SpaceDescriptor;
use crate::util::heap::{MonotonePageResource, PageResource};
use crate::util::linear_scan::Region;
use crate::util::object_enum::ObjectEnumerator;
use crate::util::Address;
use crate::util::VMThread;
use crate::vm::VMBinding;
use atomic::Atomic;
use std::sync::atomic::Ordering;
use std::sync::RwLock;

/// A region in a [`RegionPageResource`] and its allocation cursor.
pub struct AllocatedRegion<R: Region> {
    pub region: R,
    cursor: Atomic<Address>,
}

impl<R: Region> AllocatedRegion<R> {
    pub fn cursor(&self) -> Address {
        self.cursor.load(Ordering::Relaxed)
    }

    fn set_cursor(&self, a: Address) {
        self.cursor.store(a, Ordering::Relaxed);
    }
}

struct Sync<R: Region> {
    all_regions: Vec<AllocatedRegion<R>>,
    next_region: usize,
}

/// A [`PageResource`] which allocates pages from a region-structured heap.
/// We assume that allocations are much smaller than regions, as we
/// scan linearly over all regions to allocate, and do not revisit regions
/// before a garbage collection cycle.
pub struct RegionPageResource<VM: VMBinding, R: Region> {
    mpr: MonotonePageResource<VM>,
    sync: RwLock<Sync<R>>,
}

impl<VM: VMBinding, R: Region + 'static> PageResource<VM> for RegionPageResource<VM, R> {
    fn common(&self) -> &CommonPageResource {
        self.mpr.common()
    }

    fn common_mut(&mut self) -> &mut CommonPageResource {
        self.mpr.common_mut()
    }

    fn update_discontiguous_start(&mut self, start: Address) {
        self.mpr.update_discontiguous_start(start)
    }

    fn alloc_pages(
        &self,
        space_descriptor: SpaceDescriptor,
        reserved_pages: usize,
        required_pages: usize,
        tls: VMThread,
    ) -> Result<PRAllocResult, PRAllocFail> {
        assert!(reserved_pages <= Self::REGION_PAGES);
        assert!(required_pages <= reserved_pages);
        self.alloc(space_descriptor, reserved_pages, required_pages, tls)
    }

    fn get_available_physical_pages(&self) -> usize {
        self.mpr.get_available_physical_pages()
    }
}

impl<VM: VMBinding, R: Region + 'static> RegionPageResource<VM, R> {
    const REGION_PAGES: usize = R::BYTES / BYTES_IN_PAGE;

    pub fn new_contiguous(start: Address, bytes: usize, vm_map: &'static dyn VMMap) -> Self {
        Self::new(MonotonePageResource::new_contiguous(start, bytes, vm_map))
    }

    pub fn new_discontiguous(vm_map: &'static dyn VMMap) -> Self {
        Self::new(MonotonePageResource::new_discontiguous(vm_map))
    }

    fn new(mpr: MonotonePageResource<VM>) -> Self {
        Self {
            mpr,
            sync: RwLock::new(Sync {
                all_regions: vec![],
                next_region: 0,
            }),
        }
    }

    fn alloc(
        &self,
        space_descriptor: SpaceDescriptor,
        reserved_pages: usize,
        required_pages: usize,
        tls: VMThread,
    ) -> Result<PRAllocResult, PRAllocFail> {
        let mut b = self.sync.write().unwrap();
        let succeed = |start: Address, new_chunk: bool| {
            Result::Ok(PRAllocResult {
                start,
                pages: required_pages,
                new_chunk,
            })
        };
        let bytes = reserved_pages * BYTES_IN_PAGE;
        // First try to reuse a region.
        while b.next_region < b.all_regions.len() {
            let cursor = b.next_region;
            if let Option::Some(address) =
                self.allocate_from_region(&mut b.all_regions[cursor], bytes)
            {
                self.commit_pages(reserved_pages, required_pages, tls);
                return succeed(address, false);
            }
            b.next_region += 1;
        }
        // Else allocate a new region.
        let PRAllocResult {
            start, new_chunk, ..
        } = self.mpr.alloc_pages(
            space_descriptor,
            Self::REGION_PAGES,
            Self::REGION_PAGES,
            tls,
        )?;
        b.all_regions.push(AllocatedRegion {
            region: R::from_aligned_address(start),
            cursor: Atomic::<Address>::new(start),
        });
        let cursor = b.next_region;
        succeed(
            self.allocate_from_region(&mut b.all_regions[cursor], bytes)
                .unwrap(),
            new_chunk,
        )
    }

    fn allocate_from_region(
        &self,
        alloc: &mut AllocatedRegion<R>,
        bytes: usize,
    ) -> Option<Address> {
        let free = alloc.cursor();
        if free + bytes > alloc.region.end() {
            Option::None
        } else {
            alloc.set_cursor(free + bytes);
            Option::Some(free)
        }
    }

    /// Reset the allocation cursor for one region.
    pub fn reset_cursor(&self, alloc: &AllocatedRegion<R>, address: Address) {
        let old = alloc.cursor();
        let new = address.align_up(BYTES_IN_PAGE);
        let pages = (old - new) / BYTES_IN_PAGE;
        self.common().accounting.release(pages);
        alloc.set_cursor(new);
    }

    /// Reset the allocation cursor for a region only if it is still at `current`.
    ///
    /// This is the concurrent-safe counterpart to [`Self::reset_cursor`].  It takes
    /// the page-resource write lock so it does not race with mutator allocations,
    /// mirroring the way ART only reclaims from-space after it knows no future page
    /// processing depends on the old contents.
    pub fn reset_cursor_if_unchanged(
        &self,
        region_index: usize,
        current: Address,
        address: Address,
    ) -> bool {
        let mut sync = self.sync.write().unwrap();
        let Some(alloc) = sync.all_regions.get(region_index) else {
            return false;
        };
        if alloc.cursor() != current {
            return false;
        }

        let new = address.align_up(BYTES_IN_PAGE);
        let pages = (current - new) / BYTES_IN_PAGE;
        self.common().accounting.release(pages);
        alloc.set_cursor(new);
        sync.next_region = sync.next_region.min(region_index);
        true
    }

    /// Advance a region's cursor to at least `new_end`, if `new_end` exceeds the current cursor.
    ///
    /// Used after inter-pause black-allocation catch-up: the bump allocator may have
    /// advanced past the globally recorded cursor, and we need offset-vector / summary
    /// computation to cover those objects.  Page accounting is NOT adjusted here because
    /// the pages were already consumed by the bump allocator.
    pub fn advance_cursor_to(&self, region_index: usize, new_end: Address) -> (bool, usize) {
        let sync = self.sync.read().unwrap();
        if let Some(alloc) = sync.all_regions.get(region_index) {
            let old = alloc.cursor();
            if new_end > old {
                alloc.set_cursor(new_end);
                (true, new_end - old)
            } else {
                (false, 0)
            }
        } else {
            (false, 0)
        }
    }

    /// Align all region cursors up to page boundary.
    ///
    /// ART aligns the post-marking black-allocation boundary to page size before
    /// the compaction pause. This is the per-region analogue for the Compressor:
    /// any future post-pause allocation from a reused region starts at the next
    /// page boundary instead of in the middle of a page that may participate in
    /// the UFFD epoch.
    pub fn align_all_region_cursors_up_to_page(&self) -> (usize, usize) {
        let sync = self.sync.write().unwrap();
        let mut regions_aligned = 0;
        let mut bytes_advanced = 0;
        for alloc in sync.all_regions.iter() {
            let old = alloc.cursor();
            let new = old.align_up(BYTES_IN_PAGE);
            if new > old {
                debug_assert!(new <= alloc.region.end());
                alloc.set_cursor(new);
                regions_aligned += 1;
                bytes_advanced += new - old;
            }
        }
        (regions_aligned, bytes_advanced)
    }

    /// Prevent future allocations from reusing the currently known regions.
    ///
    /// This is used by the concurrent UFFD path so mutators resume allocation in
    /// fresh regions while previously allocated regions are still being compacted
    /// from shadow mappings.
    pub fn seal_existing_regions_for_uffd(&self) {
        let mut sync = self.sync.write().unwrap();
        sync.next_region = sync.all_regions.len();
    }

    /// Reset the allocator state after a collection, so that the allocator will
    /// revisit regions which the garbage collector has compacted.
    pub fn reset_allocator(&self) {
        self.sync.write().unwrap().next_region = 0;
    }

    pub fn enumerate(&self, enumerator: &mut dyn ObjectEnumerator) {
        let sync = self.sync.read().unwrap();
        for alloc in sync.all_regions.iter() {
            enumerator.visit_address_range(alloc.region.start(), alloc.cursor());
        }
    }

    pub fn with_regions<T>(&self, f: &mut impl FnMut(&Vec<AllocatedRegion<R>>) -> T) -> T {
        let sync = self.sync.read().unwrap();
        f(&sync.all_regions)
    }

    pub fn enumerate_regions(&self, enumerator: &mut impl FnMut(&AllocatedRegion<R>)) {
        let sync = self.sync.read().unwrap();
        for alloc in sync.all_regions.iter() {
            enumerator(alloc);
        }
    }
}
