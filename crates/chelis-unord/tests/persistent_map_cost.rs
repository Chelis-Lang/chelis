//! A `PersistentMap` copy costs no allocation, and a change through a copy
//! allocates only for the path it touches: logarithmically many nodes in the
//! map's size, never a number proportional to it.
//!
//! The oracle counts heap allocations with this binary's allocator. Every
//! measured operation runs on the test's own thread, and the counter is
//! thread-local, so the harness's other threads cannot reach it. The sweep
//! covers every size from 1 to 4096 at each operation kind, so the bound is
//! checked for the whole class of tree shapes those sizes produce rather
//! than at one witness size.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use chelis_unord::{PersistentMap, UnordMap};

struct CountingAllocator;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocations<R>(work: impl FnOnce() -> R) -> (R, usize) {
    let before = ALLOCATIONS.with(Cell::get);
    let result = work();
    (result, ALLOCATIONS.with(Cell::get) - before)
}

/// The most nodes one change may copy in a map of `len` entries: an AVL tree
/// is at most `1.45 log2(len + 2)` high, a change copies the nodes on one
/// root-to-leaf path, and rebalancing on the way back up copies at most two
/// more per level.
fn path_bound(len: usize) -> usize {
    let levels = usize::try_from((len + 2).ilog2()).unwrap() * 3 / 2 + 2;
    3 * levels
}

#[test]
fn a_change_through_a_copy_allocates_only_for_its_path() {
    let mut map = PersistentMap::new();
    for len in 1..=4096u64 {
        map.insert(len * 2, len);
        let size = usize::try_from(len).unwrap();
        let bound = path_bound(size);

        let (copy, cost) = allocations(|| map.clone());
        assert_eq!(cost, 0, "copying a map of {len} entries allocated");

        let mut inserted = copy.clone();
        let ((), cost) = allocations(|| {
            inserted.insert(len + 1, 0);
        });
        assert!(
            cost <= bound,
            "an insert into a shared map of {len} copied {cost} nodes"
        );

        let mut removed = copy.clone();
        let ((), cost) = allocations(|| {
            removed.remove(&(len / 2 * 2 + 2));
        });
        assert!(
            cost <= bound,
            "a removal from a shared map of {len} copied {cost} nodes"
        );

        let mut updated = copy.clone();
        let ((), cost) = allocations(|| {
            if let Some(value) = updated.get_mut(&2) {
                *value += 1;
            }
        });
        assert!(
            cost <= bound,
            "an update in a shared map of {len} copied {cost} nodes"
        );

        let ((), cost) = allocations(|| {
            assert!(copy.get(&(len * 2)).is_some());
            assert!(!copy.contains_key(&(len * 2 + 1)));
        });
        assert_eq!(cost, 0, "a lookup allocated");

        let mut absent = copy.clone();
        let ((), cost) = allocations(|| {
            assert!(absent.remove(&(len * 2 + 1)).is_none());
            assert!(absent.get_mut(&(len * 2 + 1)).is_none());
        });
        assert_eq!(
            cost, 0,
            "touching an absent key in a shared map of {len} copied"
        );
        drop((copy, inserted, removed, updated, absent));
    }
}

#[test]
fn an_unshared_map_changes_in_place() {
    let mut map = PersistentMap::new();
    for key in 0..4096u64 {
        let ((), cost) = allocations(|| {
            map.insert(key, key);
        });
        assert_eq!(
            cost, 1,
            "an insert into an unshared map allocated more than its node"
        );
    }
    let ((), cost) = allocations(|| {
        for key in 0..4096u64 {
            map.remove(&key);
        }
    });
    assert_eq!(cost, 0, "a removal from an unshared map allocated");
}

#[test]
fn the_detector_rejects_a_whole_map_copy() {
    // Negative control: the copy the persistent map replaces allocates in
    // proportion to the entries it copies, so the same bound must reject it.
    let mut map = UnordMap::new();
    for len in 1..=4096u64 {
        map.insert(len, len);
    }
    let (copy, cost) = allocations(|| map.clone());
    assert!(
        cost > path_bound(4096),
        "the detector accepted a whole-map copy: {cost} allocations"
    );
    drop(copy);
}
