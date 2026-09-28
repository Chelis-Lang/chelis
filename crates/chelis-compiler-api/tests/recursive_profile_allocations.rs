//! Source-profile admission must not copy the program at every recursive call.
use chelis_compiler_api::compiler::eval_selected;
use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind};
#[path = "../../../tests/support/wire_values.rs"]
mod wire_values;
use std::alloc::{GlobalAlloc, Layout, System};
use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator;
static BYTES: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        BYTES.fetch_add(size, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocated_for(iterations: usize) -> usize {
    let mut source = String::new();
    for function in 0..12 {
        writeln!(source, "def unused{function}(x: i64) -> i64 = {{").unwrap();
        for binding in 0..48 {
            writeln!(source, "v{binding} = x + {binding}i64").unwrap();
        }
        source.push_str("v47\n}\n");
    }
    source.push_str("def iterate_count(n: i64, acc: i64) -> i64 = if n <= 0i64 then acc else iterate_count(n - 1i64, acc + 1i64)\n");
    writeln!(source, "answer = iterate_count({iterations}i64, 0i64)").unwrap();
    let before = BYTES.load(Ordering::Relaxed);
    let result = eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source,
            bindings: Default::default(),
        },
        &["answer".to_string()],
    )
    .unwrap();
    let allocated = BYTES.load(Ordering::Relaxed) - before;
    assert_eq!(
        serde_json::to_value(&result.roots[0].value).unwrap(),
        serde_json::to_value(wire_values::scalar_integer(
            chelis_types::types::Prim::Int64,
            iterations as i64
        ))
        .unwrap()
    );
    allocated
}

#[test]
fn recursive_calls_do_not_recopy_unrelated_definition_bodies() {
    // This deliberately recursive fixture measures heap allocation. Give debug
    // evaluator frames enough stack without requiring runner environment flags.
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(|| {
            let setup = allocated_for(0);
            let repeated = allocated_for(64);
            let extra = repeated.saturating_sub(setup);
            eprintln!("setup={setup}, repeated={repeated}, extra={extra} bytes");
            assert!(
                extra < 32 * 1024 * 1024,
                "recursive profile admission allocated {extra} extra bytes"
            );
        })
        .unwrap()
        .join()
        .unwrap();
}
