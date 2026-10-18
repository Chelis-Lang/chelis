//! Applying a closure must not copy the type syntax its bindings carry
//! (chelis#3778).
//!
//! Every application binds the callback's parameters with their declared
//! types, reads each argument's static type, and snapshots the binding types
//! at each `let`. Those types are syntax the program never changes, so an
//! application can share them; copying them made every application pay for
//! the size of its parameter types. This receipt counts heap allocations per
//! fold step for a callback whose accumulator is a two-component tuple and
//! for one whose accumulator is a 64-component tuple. The callback returns its
//! accumulator, so no step builds a value, and the per-step count must not
//! grow with the type.
//!
//! Evidentiary status: REGRESSION TEST, proven failing first. With the type
//! syntax copied per application, the 64-component callback's per-step count
//! grew with its type.
use chelis_compiler_api::compiler::eval_selected;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
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

/// A fold whose callback takes a `width`-component tuple accumulator, binds
/// it, and returns it unchanged.
fn source(width: usize, steps: usize) -> String {
    let ty = vec!["i64"; width].join(", ");
    let zero = vec!["0i64"; width].join(", ");
    let mut source = String::new();
    writeln!(
        source,
        "result = fold(fn (acc: ({ty}), x: i64) -> {{\n  \
         y = acc\n  \
         y\n\
         }}, ({zero}), range(0i64, {steps}i64))"
    )
    .unwrap();
    source
}

fn allocations(width: usize, steps: usize) -> usize {
    let request = EvalRequest {
        source_kind: SourceKind::Surf,
        source: source(width, steps),
        bindings: Default::default(),
    };
    CALLS.store(0, Relaxed);
    ENABLED.store(true, Relaxed);
    let result = eval_selected(request, &["result".to_string()]);
    ENABLED.store(false, Relaxed);
    let calls = CALLS.load(Relaxed);
    result.expect("the fold evaluates");
    calls
}

/// Allocations of one fold step, from the difference of two step counts.
fn per_step(width: usize) -> usize {
    allocations(width, 200).saturating_sub(allocations(width, 100)) / 100
}

#[test]
fn closure_application_allocations_do_not_scale_with_its_type_syntax() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let narrow = per_step(2);
            let wide = per_step(64);
            eprintln!("allocations per fold step: {narrow} (2-component), {wide} (64-component)");
            assert!(
                narrow > 0,
                "a fold step allocates, or this receipt measures nothing"
            );
            assert!(
                wide <= narrow + narrow / 10,
                "a fold step allocated {wide} times with a 64-component accumulator type \
                 against {narrow} with a 2-component one"
            );
        })
        .unwrap()
        .join()
        .unwrap();
}
