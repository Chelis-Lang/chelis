//! Scope snapshots must not duplicate all previously lowered function bodies.
use chelis_deep::Expr;
use std::alloc::{GlobalAlloc, Layout, System};
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
            // Frozen output of the original frontend plus macro expansion (no
            // prelude). Each helper retains all 32 parameters and 128 bindings.
            // Only the nested let tails were detached for JSON serialization:
            // a complete 128-let tree exceeds serde_json's recursion limit.
            let fragments: Vec<Expr> = serde_json::from_str(include_str!(
                "fixtures/core/tests/callable_snapshot_allocations/scalar_helpers.json"
            ))
            .unwrap();
            let mut fragments = fragments.into_iter();
            let mut deep = Vec::with_capacity(34);
            for _function in 0..16 {
                deep.push(fragments.next().unwrap());
                let mut definition = fragments.next().unwrap();
                let bindings: Vec<_> = fragments.by_ref().take(128).collect();
                let mut body = fragments.next().unwrap();
                for mut binding in bindings.into_iter().rev() {
                    let Expr::List(list, _) = &mut binding else {
                        panic!("expected the original legacy let carrier");
                    };
                    list.elements.push(body);
                    body = binding;
                }
                let Expr::List(definition_list, _) = &mut definition else {
                    panic!("expected the original legacy def carrier");
                };
                let Some(Expr::List(function, _)) = definition_list.elements.last_mut() else {
                    panic!("expected the original legacy fn carrier");
                };
                function.elements.push(body);
                deep.push(definition);
            }
            // Original scalar input ascription and helper15 call with 32 args.
            deep.extend(fragments);
            let checked = chelis_types::check_typed_program(&deep).unwrap();
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
