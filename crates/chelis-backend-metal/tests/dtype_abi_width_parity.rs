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

// KNOWN BLIND SPOT: width parity cannot see the Int32 case.
//
// `chelis_tensor_cmplt` states the storage convention directly -- "I32 and
// BOOL storage stays f32-encoded; the f32-strided read is correct for both
// today" -- and reads both through `data_as_f32_const`. Bool is caught by the
// tests below only because its widths differ (1 vs 4). Int32 is 4 bytes on
// both sides, so every assertion here passes while the runtime holds f32 bit
// patterns and Metal's device side holds native `int32_t`. Reinterpreting one
// as the other is silent corruption.
//
// That is why `docs/gap_synthesis.md` rates `CRuntime-I32Storage-F1` MEDIUM /
// "silent type-confusion latent" against `CRuntime-BoolStorage-F1`'s
// LOW-MEDIUM: bool is the loud instance of a quieter class.
//
// Deliberately not asserted here. The evidence is mixed --
// `chelis_scalar_tensor_from_i64` writes a native `i32` while `cmplt` reads
// f32-encoded -- so pinning Int32 needs the storage question settled first,
// not a guess encoded as a test.

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

/// Width parity alone is not sufficient, and this pins why.
///
/// The runtime does not store `bool` as a 1-byte C `bool`. It stores it as
/// `f32`: `chelis-runtime/src/lib.rs` reads `RuntimeDType::Bool` through
/// `*const f32` in the same match arm as `F32`, and `chelis_fill_bool_bits`
/// writes through `data_as_f32` with `f32::from_bits`. Metal's device side
/// uses MSL `bool` (`msl_type(Bool) == "bool"`, and `host_const_fill_body`
/// emits `p[i] = true`).
///
/// So the two lanes disagree on *encoding*, not only on width. Four bytes of
/// packed MSL bools reinterpreted as an `f32` is garbage, not a misaligned
/// stride. A fix that only makes `metal_elem_size(Bool)` return 4 would turn
/// the two tests above green while leaving the copied data wrong, so this
/// test exists to fail in that case.
#[test]
fn bool_device_representation_matches_the_runtime_f32_storage() {
    let runtime_width = Prim::Bool
        .runtime_dtype()
        .expect("bool must have a runtime dtype")
        .byte_width() as usize;

    // The runtime stores bool as f32, so the ABI width is f32's width.
    assert_eq!(
        runtime_width,
        size_of::<f32>(),
        "the runtime stores bool as f32; if this changed, the reasoning below \
         and the Metal copy path both need revisiting"
    );

    // Therefore the device-side element must also be f32-shaped, not a 1-byte
    // MSL bool, or the copy back needs a real conversion rather than a memcpy.
    assert_eq!(
        metal_elem_size(Prim::Bool),
        runtime_width,
        "metal_elem_size(bool) = {} but the runtime stores bool as a {}-byte \
         f32. Matching the width alone is not enough: the device element must \
         hold the same f32 encoding, or the device-to-host path must convert \
         MSL bool -> f32 instead of copying bytes",
        metal_elem_size(Prim::Bool),
        runtime_width
    );
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
