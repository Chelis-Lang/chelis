//! Host lowering of a program pays per definition for that definition, not
//! for the program. The kernel decision once rebuilt program-wide call-graph
//! facts for every definition it decided: the set of recursive functions
//! (one graph walk per function), the set of function names, and a scan of
//! every caller; and each kernel lowering scanned every declaration to find
//! the one owning its body. Lowering `n` definitions did `n^2` work
//! (chelis#3785).
//!
//! The oracle is the heap bytes host lowering allocates, counted by this
//! binary's allocator, which is deterministic for a fixed program. The counter
//! is thread-local and every measured lowering runs on one measuring thread,
//! so the harness's other threads cannot reach it. Doubling the definitions
//! must grow the bytes by at most 2.5 times: linear growth gives about 2,
//! quadratic about 4.

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

/// One shared helper and `n` definitions that each call it, and one eager
/// use: every definition has a callee, so a per-definition graph walk has work
/// to do, while no definition inlines more than the one helper.
fn shared_callee(n: usize) -> CheckedProgram {
    let mut source = String::from("def g[n](x: &tensor[n, f32]) -> tensor[n, f32] = add(x, x)\n");
    for index in 0..n {
        writeln!(
            source,
            "def f{index}[n](x: &tensor[n, f32]) -> tensor[n, f32] = add(g(&x), x)"
        )
        .unwrap();
    }
    source.push_str("v = to_tensor([1.0f32, 2.0f32, 3.0f32])\nout = f0(&v)\n");
    checked(&source)
}

/// `n` independent definitions and one eager use.
fn independent(n: usize) -> CheckedProgram {
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

fn host_lowering_bytes(program: &CheckedProgram) -> usize {
    let (lowered, bytes) = allocated(|| {
        chelis_ir::host::try_lower_compiled_program(program).expect("the probe lowers")
    });
    drop(lowered);
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

fn assert_linear(label: &str, bytes: &[usize]) {
    eprintln!("{label}: host lowering bytes at n = 128, 256, 512: {bytes:?}");
    for pair in bytes.windows(2) {
        assert!(
            grows_linearly(pair[0], pair[1]),
            "host lowering of {label} allocated superlinearly in its definitions: {bytes:?}"
        );
    }
}

#[test]
fn host_lowering_allocates_linearly_in_independent_definitions() {
    let bytes =
        on_measuring_thread(|| [128, 256, 512].map(|n| host_lowering_bytes(&independent(n))));
    assert_linear("independent definitions", &bytes);
}

#[test]
fn host_lowering_allocates_linearly_in_callers_of_a_shared_helper() {
    let bytes =
        on_measuring_thread(|| [128, 256, 512].map(|n| host_lowering_bytes(&shared_callee(n))));
    assert_linear("callers of a shared helper", &bytes);
}

#[test]
fn the_detector_rejects_a_name_set_copied_per_definition() {
    // Negative control: what the removed per-definition derivation did, a
    // copy of every function name made once for each definition.
    fn copy_per_definition(n: usize) -> usize {
        let names = (0..n).map(|index| format!("f{index}")).collect::<Vec<_>>();
        (0..n)
            .map(|_| {
                names
                    .iter()
                    .cloned()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
            })
            .sum()
    }
    let (_, small) = allocated(|| copy_per_definition(256));
    let (_, large) = allocated(|| copy_per_definition(512));
    assert!(
        !grows_linearly(small, large),
        "the detector accepted a per-definition name-set copy: {small} -> {large} bytes"
    );
}

/// Declarations examined to find a body's owner while host lowering `program`.
fn declaration_visits(program: &CheckedProgram) -> u64 {
    let before = chelis_ir::lower::declaration_body_visits();
    let lowered = chelis_ir::host::try_lower_compiled_program(program).expect("the probe lowers");
    drop(lowered);
    chelis_ir::lower::declaration_body_visits() - before
}

#[test]
fn finding_a_body_owner_examines_a_bounded_number_of_declarations_per_definition() {
    // Every kernel lowering asks which declaration owns its body. The answer
    // must examine the declarations that could match, not every one.
    let sizes = [64, 128, 256, 512];
    let visits = on_measuring_thread(move || sizes.map(|n| declaration_visits(&independent(n))));
    eprintln!("declarations examined at n = {sizes:?}: {visits:?}");
    for (n, visits) in sizes.into_iter().zip(visits) {
        assert!(
            visits <= 4 * n as u64,
            "host lowering of {n} definitions examined {visits} declarations to find body owners"
        );
    }
}
