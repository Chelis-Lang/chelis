//! Lowering pays for the names each binder adds, not for the scope it
//! enters. Two walks once copied their whole lexical scope at every binder:
//! `LowerCtx::lower_fn` copied the value scope and the local callables at
//! every function form, and every top-level definition is one, so lowering
//! a library of `n` definitions copied `n^2 / 2` entries (chelis#3774); and
//! the lowerability walk copied its name sets at every `let`, so classifying
//! a chain of `k` bindings copied `k^2 / 2` names (chelis#3763).
//!
//! The oracle is the heap bytes each walk allocates, counted by this binary's
//! allocator, which is deterministic for a fixed program. The counter is
//! thread-local and every measured walk runs on one measuring thread, so the
//! harness's other threads cannot reach it. Doubling the input must grow the
//! bytes by at most 2.5 times: linear growth gives about 2, quadratic about 4.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fmt::Write as _;

use chelis_types::CheckedProgram;

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

fn checked(source: &str) -> CheckedProgram {
    let declarations = chelis_surf::parser::parse_str(source).expect("the probe parses");
    let deep = chelis_surf::desugar::desugar_program(&declarations).expect("the probe desugars");
    let expanded = chelis_macros::expand_program(
        &deep,
        &chelis_macros::ExpansionOptions {
            max_iterations: 100,
            load_std_prelude: false,
        },
    )
    .expect("the probe expands");
    let checked = chelis_types::check_typed_program(expanded.exprs()).expect("the probe checks");
    let checked = chelis_effects::check_program(&checked).expect("the probe has no effect errors");
    chelis_types::check_linearity(&checked).expect("the probe is linear")
}

/// chelis#3774's probe: `n` one-line definitions and one eager use.
fn library(n: usize) -> CheckedProgram {
    let mut source = String::new();
    for index in 0..n {
        writeln!(
            source,
            "def f{index}[n](x: &tensor[n, f32]) -> tensor[n, f32] = add(mul(x, x), x)"
        )
        .unwrap();
    }
    source.push_str("v = to_tensor([1.0f32, 2.0f32, 3.0f32])\nout = f0(&v)\n");
    checked(&source)
}

/// chelis#3763's probe: one definition whose body chains `k` bindings.
fn chain(k: usize) -> CheckedProgram {
    let mut source =
        String::from("def f(x: tensor[3, f32]) -> tensor[3, f32] = {\n  a0 = neg(x)\n");
    for index in 1..k {
        writeln!(source, "  a{index} = neg(a{})", index - 1).unwrap();
    }
    writeln!(source, "  a{}\n}}", k - 1).unwrap();
    source.push_str("out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n");
    checked(&source)
}

fn lowering_bytes(program: &CheckedProgram) -> usize {
    let (lowered, bytes) =
        allocated(|| chelis_ir::lower::try_lower_program_to_library(program).expect("it lowers"));
    drop(lowered);
    bytes
}

fn classification_bytes(program: &CheckedProgram) -> usize {
    let (verdicts, bytes) =
        allocated(|| chelis_ir::lower::top_level_lowering_map(program.exprs(), program.type_env()));
    assert_eq!(
        verdicts.get("f"),
        Some(&true),
        "the chain lowers to the DAG"
    );
    bytes
}

fn on_measuring_thread<R: Send + 'static>(work: impl FnOnce() -> R + Send + 'static) -> R {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(work)
        .expect("spawn the measuring thread")
        .join()
        .expect("the measuring thread finishes")
}

#[test]
fn lowering_a_library_allocates_linearly_in_its_definitions() {
    let bytes = on_measuring_thread(|| [128, 256, 512].map(|n| lowering_bytes(&library(n))));
    eprintln!("lowering bytes at n = 128, 256, 512: {bytes:?}");
    for pair in bytes.windows(2) {
        assert!(
            grows_linearly(pair[0], pair[1]),
            "lowering a library allocated superlinearly in its definitions: {bytes:?}"
        );
    }
}

#[test]
fn classifying_a_let_chain_allocates_linearly_in_its_bindings() {
    let bytes = on_measuring_thread(|| [250, 500, 1000].map(|k| classification_bytes(&chain(k))));
    eprintln!("classification bytes at k = 250, 500, 1000: {bytes:?}");
    for pair in bytes.windows(2) {
        assert!(
            grows_linearly(pair[0], pair[1]),
            "classifying a let chain allocated superlinearly in its bindings: {bytes:?}"
        );
    }
}

#[test]
fn the_detector_rejects_a_scope_copied_per_binder() {
    // Negative control: what the removed copies did, a name set copied at
    // every binder before it adds its own name.
    fn copy_per_binder(k: usize) -> usize {
        let mut scope = std::collections::BTreeSet::new();
        let mut copies = Vec::new();
        for index in 0..k {
            let mut scoped = scope.clone();
            scoped.insert(format!("a{index}"));
            copies.push(scope);
            scope = scoped;
        }
        copies.len()
    }
    let (_, small) = allocated(|| copy_per_binder(500));
    let (_, large) = allocated(|| copy_per_binder(1000));
    assert!(
        !grows_linearly(small, large),
        "the detector accepted a per-binder scope copy: {small} -> {large} bytes"
    );
}
