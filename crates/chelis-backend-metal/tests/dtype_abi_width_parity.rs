//! Adversarial probe: does the Metal backend's per-element width agree
//! with the runtime ABI width it stamps into `chelis_alloc`?
//!
//! `dtype::metal_elem_size` documents itself as mirroring "the runtime
//! allocator's `chelis_alloc` element width", and `runtime_dtype_tag`
//! is what the emitter actually passes to `chelis_alloc`. Those two
//! must agree for every dtype in the active Metal matrix, or the
//! device-to-host copy and the host allocation disagree on stride.

use chelis_backend_metal::dtype::{host_sizeof_expr, metal_elem_size};
use chelis_types::types::Prim;

/// Active Metal dtype matrix per spec/04-type-system.md §1.1.3 (f64 excluded).
const ACTIVE_METAL: [Prim; 8] = [
    Prim::F32,
    Prim::F16,
    Prim::Bf16,
    Prim::Int8,
    Prim::Int16,
    Prim::Int32,
    Prim::Int64,
    Prim::Bool,
];

#[test]
fn metal_elem_size_matches_runtime_abi_width() {
    for prec in ACTIVE_METAL {
        // `byte_width` is u32: it describes the runtime ABI, which does not
        // vary with the host pointer width. Widening to compare against a
        // host-side `usize` is the caller's concern.
        let runtime_width = prec
            .runtime_dtype()
            .expect("active Metal dtype must have a runtime dtype")
            .byte_width() as usize;
        assert_eq!(
            metal_elem_size(prec),
            runtime_width,
            "metal_elem_size({}) = {} but the runtime ABI tag it allocates \
             with sizes an element at {} bytes; the device-to-host copy and \
             the chelis_alloc buffer would disagree on stride",
            prec.name(),
            metal_elem_size(prec),
            runtime_width
        );
    }
}

/// The emitter computes the device-to-host byte count as
/// `n * host_sizeof_expr(prec)` but allocates the destination with
/// `chelis_alloc(..., runtime_dtype_tag(prec))`. If the C expression
/// evaluates to a different width than the runtime ABI width, the copy
/// under-fills (or overruns) the destination tensor.
#[test]
fn host_sizeof_expr_matches_runtime_abi_width() {
    // Map the emitted C sizeof expression to the width it evaluates to
    // on every supported host ABI (LP64 / LLP64 both agree on these).
    fn c_sizeof_width(expr: &str) -> usize {
        match expr {
            "sizeof(float)" => 4,
            "sizeof(uint16_t)" => 2,
            "sizeof(int8_t)" => 1,
            "sizeof(int16_t)" => 2,
            "sizeof(int32_t)" => 4,
            "sizeof(int64_t)" => 8,
            "sizeof(bool)" => 1,
            other => panic!("unmapped sizeof expression: {other}"),
        }
    }

    for prec in ACTIVE_METAL {
        let runtime_width = prec
            .runtime_dtype()
            .expect("active Metal dtype must have a runtime dtype")
            .byte_width() as usize;
        let emitted = host_sizeof_expr(prec);
        assert_eq!(
            c_sizeof_width(emitted),
            runtime_width,
            "host_sizeof_expr({}) emits `{}` ({} bytes) but chelis_alloc \
             sizes that dtype at {} bytes/element",
            prec.name(),
            emitted,
            c_sizeof_width(emitted),
            runtime_width
        );
    }
}
