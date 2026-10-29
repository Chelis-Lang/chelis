//! Effects-only checked reconstruction compares source bodies without copying them.
//! The checked-owner/structure contract is pinned by the reannotation mutation suites.

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

fn reconstruction_bytes(size: usize, valid: bool) -> usize {
    let payload = "x".repeat(size);
    let source = format!("payload = \"{payload}\"\nresult = 1i32\n");
    reconstruction_bytes_for_source(&source, valid)
}

fn reconstruction_bytes_for_source(source: &str, valid: bool) -> usize {
    let decls = chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack).unwrap();
    let deep =
        chelis_surf::desugar::desugar_program(&decls, &chelis_syntax_host::NativeStack).unwrap();
    let checked = chelis_types::check_ir_program(&deep).unwrap();
    let mut candidate = checked.annotated_exprs().to_vec();
    if !valid {
        // A real body mutation must still reject after the large unchanged payload.
        candidate[1] = candidate[0].clone();
    }
    let before = BYTES.load(Ordering::Relaxed);
    let result = checked.try_with_effect_annotations(candidate);
    let bytes = BYTES.load(Ordering::Relaxed) - before;
    assert_eq!(result.is_ok(), valid, "{result:?}");
    if let Ok(reconstructed) = result {
        assert_eq!(reconstructed.annotated_exprs(), checked.annotated_exprs());
        assert_eq!(reconstructed.type_env(), checked.type_env());
        assert_eq!(
            reconstructed.local_ascriptions(),
            checked.local_ascriptions()
        );
        assert_eq!(
            reconstructed.signature_inference(),
            checked.signature_inference()
        );
        assert_eq!(reconstructed.library_proof_id(), checked.library_proof_id());
    }
    bytes
}

#[test]
fn effect_reconstruction_does_not_copy_source_bodies_to_compare_owners() {
    chelis_types::run_on_grown_stack(|| {
        for valid in [true, false] {
            let small = reconstruction_bytes(1024, valid);
            for size in [256 * 1024, 1024 * 1024] {
                let large = reconstruction_bytes(size, valid);
                eprintln!("valid={valid} payload={size}: {small} -> {large} allocated bytes");
                assert!(
                    large <= small + 64 * 1024,
                    "effects reconstruction copied source bodies: {small} -> {large}"
                );
            }
        }
        // This exact emitter witness keeps independent input axes and all
        // per-axis singleton/equality checks. The budget covers reconstruction
        // only, after parsing, checking and the caller's candidate copy.
        let bytes = reconstruction_bytes_for_source(
            include_str!("fixtures/hydronnx_symbolic_broadcast_2.ch"),
            true,
        );
        eprintln!("pinned two-Add symbolic-broadcast reconstruction: {bytes} bytes");
        assert!(
            bytes < 40 * 1024 * 1024,
            "model reconstruction allocated {bytes} bytes"
        );
    });
}
