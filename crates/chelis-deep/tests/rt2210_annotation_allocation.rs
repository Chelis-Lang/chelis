//! Pins the three allocation figures the `Storage` doc comment states:
//! `MetadataValue` is 184 bytes, a one-entry `BTreeMap` costs about two
//! kilobytes of structure, and a one-entry `Vec` costs the `RawVec`
//! four-slot minimum rather than one slot. Nothing else in the tree holds
//! those numbers to account, and the comment reads as fact.
//!
//! From the #2210 review round (rt-2210-round1). Its first draft needed
//! `--test-threads=1`, because one process-global counter is charged for
//! every other test thread's allocations; this version counts per thread,
//! so it runs under the default harness. The reviewer's second test in the
//! same file printed an allocation ladder and asserted nothing, so it is
//! not landed here.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::BTreeMap;

// Per-thread counting allocator. A process-global counter would be charged
// for every other test thread's allocations, which is why an earlier draft
// of this file needed `--test-threads=1`. Thread-local `Cell`s with `const`
// initialisers have no destructor and allocate nothing on first touch, so
// they are safe to read from inside `alloc` itself; `try_with` keeps a
// late-teardown access from recursing.
thread_local! {
    static BYTES: Cell<usize> = const { Cell::new(0) };
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let _ = COUNTING.try_with(|on| {
            if on.get() {
                let _ = BYTES.try_with(|b| b.set(b.get() + l.size()));
            }
        });
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static A: Counting = Counting;

fn measure<T>(f: impl FnOnce() -> T) -> (usize, T) {
    BYTES.with(|b| b.set(0));
    COUNTING.with(|on| on.set(true));
    let out = f();
    COUNTING.with(|on| on.set(false));
    (BYTES.with(|b| b.get()), out)
}

use chelis_deep::Span;
use chelis_deep::annotations::{MetadataKey, MetadataValue, Present, Spanned};

#[test]
fn one_entry_btreemap_costs_about_two_kilobytes_and_the_vector_does_not() {
    let value_size = std::mem::size_of::<MetadataValue>();
    // The String payload allocates too; measure it separately and subtract.
    let (payload_bytes, _) = measure(|| "note".to_string());

    let (map_bytes, map) = measure(|| {
        let mut m: BTreeMap<MetadataKey, MetadataValue> = BTreeMap::new();
        m.insert(
            MetadataKey::Doc,
            MetadataValue::Doc(Spanned::new("note".to_string(), Span::new(0, 0))),
        );
        m
    });
    let (vec_bytes, vec) = measure(|| {
        let mut v: Vec<MetadataValue> = Vec::new();
        v.insert(
            0,
            MetadataValue::Doc(Spanned::new("note".to_string(), Span::new(0, 0))),
        );
        v
    });
    assert_eq!(map.len(), 1);
    assert_eq!(vec.len(), 1);
    println!("size_of::<MetadataValue>() = {value_size}");
    println!(
        "size_of::<MetadataKey>()   = {}",
        std::mem::size_of::<MetadataKey>()
    );
    println!("String payload             = {payload_bytes} B");
    println!(
        "BTreeMap one entry         = {map_bytes} B (structure {} B)",
        map_bytes - payload_bytes
    );
    println!(
        "Vec      one entry         = {vec_bytes} B (structure {} B)",
        vec_bytes - payload_bytes
    );
    println!(
        "ratio                      = {:.1}x",
        map_bytes as f64 / vec_bytes as f64
    );

    assert_eq!(value_size, 184, "PR body states MetadataValue is 184 bytes");
    let map_structure = map_bytes - payload_bytes;
    assert!(
        (1900..=2400).contains(&map_structure),
        "PR body states about 2.1 KB for a one-entry map; measured {map_structure}"
    );
    // Rust's RawVec minimum non-zero capacity is 4 for a 184-byte element,
    // so a one-entry vector allocates four slots, not one.
    assert_eq!(
        vec_bytes - payload_bytes,
        4 * value_size,
        "a one-entry vector allocates the RawVec minimum of four slots"
    );
}

/// The `Storage` comment also claims the vector "stays under the leaf
/// node's footprint up to eight". Eight is the crossover, so pin both
/// sides of it against the leaf node measured in the same run.
#[test]
fn the_vector_stays_under_the_leaf_node_up_to_eight_entries() {
    let (leaf_bytes, _) = measure(|| {
        let mut m: BTreeMap<MetadataKey, MetadataValue> = BTreeMap::new();
        m.insert(
            MetadataKey::Opaque,
            MetadataValue::Opaque(Present::new(Span::new(0, 0))),
        );
        m
    });

    let width = std::mem::size_of::<MetadataValue>();
    let mut vector: Vec<MetadataValue> = Vec::new();
    let footprint = |vector: &mut Vec<MetadataValue>| {
        let at = vector.len();
        vector.insert(at, MetadataValue::Opaque(Present::new(Span::new(0, 0))));
        vector.capacity() * width
    };
    let mut at_eight = 0;
    for _ in 0..8 {
        at_eight = footprint(&mut vector);
    }
    let at_nine = footprint(&mut vector);

    println!("leaf node {leaf_bytes} B, eight entries {at_eight} B, nine {at_nine} B");
    assert!(
        at_eight < leaf_bytes,
        "eight entries occupy {at_eight} B, which the comment says is under the {leaf_bytes} B leaf node"
    );
    assert!(
        at_nine > leaf_bytes,
        "nine entries occupy {at_nine} B, so eight is the crossover the comment names"
    );
}
