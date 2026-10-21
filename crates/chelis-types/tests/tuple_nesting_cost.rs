//! Checking a tuple chain `n` deep allocates linearly in `n` (chelis#2622).
//!
//! The checker records a type for every node, and a node's type contains the
//! type of every node inside it, so a chain `n` deep records types whose sizes
//! sum to about `n^2 / 2`. With owned children each record was a full copy of
//! its subtree; with shared children a record costs its own node.
//!
//! The oracle counts heap bytes with this binary's allocator, which is
//! deterministic for a fixed program. The counter is thread-local and every
//! measured check runs on one measuring thread, so the harness's other
//! threads cannot reach it. Doubling the depth must grow the bytes by at most
//! 2.5 times: linear growth gives about 2, quadratic about 4.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use chelis_types::types::{Prim, Type};

struct CountingAllocator;

thread_local! {
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = BYTES.try_with(|bytes| bytes.set(bytes.get() + layout.size()));
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let _ = BYTES.try_with(|bytes| bytes.set(bytes.get() + size));
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const DEPTHS: [usize; 4] = [128, 256, 512, 1024];

fn allocated<R>(work: impl FnOnce() -> R) -> (R, usize) {
    let before = BYTES.with(Cell::get);
    let result = work();
    (result, BYTES.with(Cell::get) - before)
}

fn grows_linearly(small: usize, large: usize) -> bool {
    large * 2 <= small * 5
}

fn on_measuring_thread<R: Send + 'static>(work: impl FnOnce() -> R + Send + 'static) -> R {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(work)
        .expect("spawn the measuring thread")
        .join()
        .expect("the measuring thread finishes")
}

/// The issue's witness: a tuple chain of literals.
fn literal_chain(depth: usize) -> String {
    let leaf = "(lit {type: (t-prim {} i32)} 1)";
    let body = format!("(tuple {{}} {leaf} ").repeat(depth) + leaf + &")".repeat(depth);
    format!("(def {{}} main (fn {{}} (params {{}}) {body}))\n")
}

/// The same chain over a parameter whose type is a variable.
fn variable_chain(depth: usize) -> String {
    let body = "(tuple {} (var {} x) ".repeat(depth) + "(var {} x)" + &")".repeat(depth);
    format!("(def {{}} f (fn {{}} (params {{}} (x {{type: (t-var {{}} _)}})) {body}))\n")
}

fn check_bytes(source: &str) -> usize {
    let program = chelis_deep::parse_and_stamp_file(source, &chelis_syntax_host::NativeStack)
        .expect("the probe parses");
    let (checked, bytes) = allocated(|| chelis_types::check_typed_program(&program));
    checked.expect("the probe checks");
    bytes
}

fn assert_linear(label: &str, generate: fn(usize) -> String) {
    let bytes = on_measuring_thread(move || DEPTHS.map(|depth| check_bytes(&generate(depth))));
    eprintln!("{label}: check bytes at depth {DEPTHS:?}: {bytes:?}");
    for pair in bytes.windows(2) {
        assert!(
            grows_linearly(pair[0], pair[1]),
            "checking a {label} allocated superlinearly in its depth: {bytes:?}"
        );
    }
}

#[test]
fn checking_a_literal_tuple_chain_allocates_linearly() {
    assert_linear("literal tuple chain", literal_chain);
}

#[test]
fn checking_a_variable_tuple_chain_allocates_linearly() {
    assert_linear("variable tuple chain", variable_chain);
}

/// What an owned child representation pays: one full copy of each level's
/// type, the record a node kept of its subtree.
fn owned_copy(ty: &Type) -> Type {
    match ty {
        Type::Tuple(items) => Type::tuple(items.iter().map(owned_copy).collect()),
        other => other.clone(),
    }
}

#[test]
fn the_detector_rejects_a_full_copy_per_level() {
    fn copies_per_level(depth: usize) -> usize {
        let mut ty = Type::Prim(Prim::Int32);
        let mut records = Vec::new();
        let (_, bytes) = allocated(|| {
            for _ in 0..depth {
                ty = Type::tuple(vec![Type::Prim(Prim::Int32), ty.clone()]);
                records.push(owned_copy(&ty));
            }
        });
        bytes
    }
    let small = copies_per_level(256);
    let large = copies_per_level(512);
    assert!(
        !grows_linearly(small, large),
        "the detector accepted a full type copy per level: {small} -> {large} bytes"
    );
    // A shared record per level, what the checker now keeps, passes.
    fn shares_per_level(depth: usize) -> usize {
        let mut ty = Type::Prim(Prim::Int32);
        let mut records = Vec::new();
        let (_, bytes) = allocated(|| {
            for _ in 0..depth {
                ty = Type::tuple(vec![Type::Prim(Prim::Int32), ty.clone()]);
                records.push(ty.clone());
            }
        });
        bytes
    }
    let small = shares_per_level(256);
    let large = shares_per_level(512);
    assert!(
        grows_linearly(small, large),
        "a shared record per level allocated superlinearly: {small} -> {large} bytes"
    );
}
