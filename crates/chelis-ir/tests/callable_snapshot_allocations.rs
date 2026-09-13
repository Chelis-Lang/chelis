//! Scope snapshots must not duplicate all previously lowered function bodies.
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

#[test]
fn lowering_scalar_helpers_keeps_scope_snapshot_allocations_bounded() {
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(|| {
            let mut source = String::new();
            for function in 0..16 {
                writeln!(
                    source,
                    "def helper{function}(x: tensor[f64]) -> tensor[f64] = {{"
                )
                .unwrap();
                for index in 0..128 {
                    writeln!(source, "v{index} = x + scalar_to_tensor({index}.0f64)").unwrap();
                }
                source.push_str("v127\n}\n");
            }
            source.push_str("x = (x : tensor[f64])\nout = helper15(x)\n");
            let decls = chelis_surf::parser::parse_str(&source).unwrap();
            let deep = chelis_surf::desugar::desugar_program(&decls);
            let expanded = chelis_macros::expand_program(
                &deep,
                &chelis_macros::ExpansionOptions {
                    max_iterations: 100,
                    load_std_prelude: false,
                },
            )
            .unwrap();
            let checked = chelis_types::check_typed_program(expanded.exprs()).unwrap();
            let checked = chelis_effects::check_program(&checked).unwrap();
            let checked = chelis_types::check_linearity(&checked).unwrap();
            let before = BYTES.load(Ordering::Relaxed);
            let lowered = chelis_ir::lower::try_lower_program_to_library(&checked).unwrap();
            let allocated = BYTES.load(Ordering::Relaxed) - before;
            eprintln!("16 scalar helpers allocated {allocated} bytes during lowering");
            assert!(!lowered.dag().nodes().is_empty());
            assert!(
                allocated < 256 * 1024 * 1024,
                "scope snapshots allocated {allocated} bytes"
            );
        })
        .unwrap()
        .join()
        .unwrap();
}
