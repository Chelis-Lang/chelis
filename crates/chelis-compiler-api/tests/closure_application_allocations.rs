//! Applying a closure must not copy its syntax (chelis#3778).
//!
//! A list combinator clones its callback once per element and applies the
//! clone, and the evaluator flattens a block's binding layers each time the
//! block runs. Either one copying syntax makes every application pay for the
//! closure's whole source, including branches it never takes. This receipt
//! counts heap allocations per fold step for one closure whose untaken branch
//! is small and one whose untaken branch is large, inside a block binding.
//! The per-step count must not grow with the untaken branch.
//!
//! Evidentiary status: REGRESSION TEST, proven failing first. With the
//! closure's checked function owned by each clone, or with the block's
//! bindings copied into a flattened binding list, the large closure's
//! per-step count grew with its untaken branch.
use chelis_compiler_api::compiler::eval_selected;
use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind};
#[path = "../../../tests/support/wire_values.rs"]
mod wire_values;
use std::alloc::{GlobalAlloc, Layout, System};
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

struct CountingAllocator;
static ENABLED: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ENABLED.load(Relaxed) {
            CALLS.fetch_add(1, Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if ENABLED.load(Relaxed) {
            CALLS.fetch_add(1, Relaxed);
        }
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// A fold whose callback binds the result of an `if` whose untaken branch
/// nests `depth` additions.
fn source(depth: usize, steps: usize) -> String {
    let mut untaken = "acc".to_string();
    for _ in 0..depth {
        untaken = format!("add({untaken}, x)");
    }
    let mut source = String::new();
    writeln!(
        source,
        "result = fold(fn (acc: i64, x: i64) -> {{\n  \
         y = if gte(x, 0i64) then add(acc, x) else {untaken}\n  \
         y\n\
         }}, 0i64, range(0i64, {steps}i64))"
    )
    .unwrap();
    source
}

fn allocations(depth: usize, steps: usize) -> usize {
    let request = EvalRequest {
        source_kind: SourceKind::Surf,
        source: source(depth, steps),
        bindings: Default::default(),
    };
    CALLS.store(0, Relaxed);
    ENABLED.store(true, Relaxed);
    let result = eval_selected(request, &["result".to_string()]);
    ENABLED.store(false, Relaxed);
    let calls = CALLS.load(Relaxed);
    let result = result.expect("the fold evaluates");
    assert_eq!(
        serde_json::to_value(&result.roots[0].value).unwrap(),
        serde_json::to_value(wire_values::scalar_integer(
            chelis_types::types::Prim::Int64,
            (steps * (steps - 1) / 2) as i64
        ))
        .unwrap(),
        "the fold sums its range"
    );
    calls
}

/// Allocations of one fold step, from the difference of two step counts.
fn per_step(depth: usize) -> usize {
    allocations(depth, 200).saturating_sub(allocations(depth, 100)) / 100
}

#[test]
fn closure_application_allocations_do_not_scale_with_its_syntax() {
    // Debug evaluator frames for the deep untaken branch need more than the
    // default test thread stack.
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let small = per_step(2);
            let large = per_step(256);
            eprintln!("allocations per fold step: {small} (2 additions), {large} (256 additions)");
            assert!(
                small > 0,
                "a fold step allocates, or this receipt measures nothing"
            );
            assert!(
                large <= small + small / 10,
                "a fold step allocated {large} times with a 256-addition untaken branch \
                 against {small} with a 2-addition one"
            );
        })
        .unwrap()
        .join()
        .unwrap();
}
