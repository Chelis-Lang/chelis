//! A speculative unification costs what it writes, not the substitution it
//! starts from. The checker tries a unification on a copy of the solver's
//! substitution wherever it must be able to discard the attempt: an annotated
//! result is checked against two copies (chelis#3784), and a call suspended on
//! a rigid binder is replayed on one copy per route combination (chelis#2536).
//! The substitution covers the whole module, so while the copy duplicated it,
//! checking `n` definitions copied `n^2` bindings.
//!
//! The oracles count heap bytes with this binary's allocator, which is
//! deterministic for a fixed program. The counter is thread-local and every
//! measured check runs on one measuring thread, so the harness's other
//! threads cannot reach it. Doubling the definitions must grow the bytes by
//! at most 2.5 times: linear growth gives about 2, quadratic about 4.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fmt::Write as _;

use chelis_types::types::{DimVar, Prim, Type, TypeVar};
use chelis_types::unify::Subst;

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

fn allocated<R>(work: impl FnOnce() -> R) -> (R, usize) {
    let before = BYTES.with(Cell::get);
    let result = work();
    (result, BYTES.with(Cell::get) - before)
}

fn grows_linearly(small: usize, large: usize) -> bool {
    large * 2 <= small * 5
}

/// A substitution binding `count` type and dimension variables.
fn substitution(count: u32) -> Subst {
    let mut subst = Subst::new();
    for index in 0..count {
        subst
            .insert_type(TypeVar(index), Type::Prim(Prim::F32))
            .expect("a fresh variable binds");
        subst.insert_dim(
            DimVar(index),
            chelis_types::types::Dim::Lit(i64::from(index)),
        );
    }
    subst
}

#[test]
fn a_substitution_copy_costs_the_same_at_every_size() {
    // The class oracle: a copy of a substitution holding 2, 4, ... 4096
    // bindings allocates the same bytes as a copy of one holding a single
    // binding.
    let single = substitution(1);
    let (copy, baseline) = allocated(|| single.clone());
    drop(copy);
    for exponent in 1..=12 {
        let subst = substitution(1 << exponent);
        let (copy, bytes) = allocated(|| subst.clone());
        assert_eq!(
            bytes,
            baseline,
            "copying a substitution of {} bindings allocated {bytes} bytes, one of a single binding \
             {baseline}",
            1 << exponent
        );
        drop(copy);
    }
}

fn check_bytes(source: &str) -> usize {
    let declarations = chelis_surf::parser::parse_str(source).expect("the probe parses");
    let program = chelis_surf::desugar::desugar_program(&declarations).expect("the probe desugars");
    let (checked, bytes) = allocated(|| chelis_types::check_typed_program(&program));
    checked.expect("the probe checks");
    bytes
}

/// chelis#3784's probe: `n` definitions with an annotated result.
fn annotated_results(n: usize) -> String {
    let mut source = String::new();
    for index in 0..n {
        writeln!(
            source,
            "def f{index}[n](x: &tensor[n, f32]) -> tensor[n, f32] = add(mul(x, x), x)"
        )
        .unwrap();
    }
    source
}

/// chelis#2536's probe: `n` definitions whose call is suspended on a rigid
/// binder.
fn binder_calls(n: usize) -> String {
    let mut source = String::new();
    for index in 0..n {
        writeln!(
            source,
            "def ok{index}[p: Int](a: p, b: p) -> List[i64] = range(a, b)"
        )
        .unwrap();
    }
    source
}

fn on_measuring_thread<R: Send + 'static>(work: impl FnOnce() -> R + Send + 'static) -> R {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(work)
        .expect("spawn the measuring thread")
        .join()
        .expect("the measuring thread finishes")
}

fn assert_linear(label: &str, bytes: &[usize]) {
    eprintln!("{label}: check bytes at n = 128, 256, 512: {bytes:?}");
    for pair in bytes.windows(2) {
        assert!(
            grows_linearly(pair[0], pair[1]),
            "checking {label} allocated superlinearly in the definitions: {bytes:?}"
        );
    }
}

#[test]
fn checking_annotated_results_allocates_linearly() {
    let bytes = on_measuring_thread(|| [128, 256, 512].map(|n| check_bytes(&annotated_results(n))));
    assert_linear("annotated results", &bytes);
}

#[test]
fn checking_binder_suspended_calls_allocates_linearly() {
    let bytes = on_measuring_thread(|| [128, 256, 512].map(|n| check_bytes(&binder_calls(n))));
    assert_linear("binder-suspended calls", &bytes);
}

#[test]
fn the_detector_rejects_a_substitution_copied_per_definition() {
    // Negative control: what the removed copies did, the whole substitution
    // copied once per definition.
    fn copy_per_definition(n: u32) -> usize {
        let mut subst = Subst::new();
        let mut copies = 0;
        for index in 0..n {
            subst
                .insert_type(TypeVar(index), Type::Prim(Prim::F32))
                .expect("a fresh variable binds");
            let copy = subst.types_snapshot();
            copies += copy.len();
        }
        copies
    }
    let (_, small) = allocated(|| copy_per_definition(256));
    let (_, large) = allocated(|| copy_per_definition(512));
    assert!(
        !grows_linearly(small, large),
        "the detector accepted a per-definition substitution copy: {small} -> {large} bytes"
    );
}
