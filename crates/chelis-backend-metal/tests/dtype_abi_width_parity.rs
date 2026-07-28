//! Adversarial probe: does the Metal backend's per-element width agree
//! with the runtime ABI width it stamps into `chelis_alloc`?
//!
//! `dtype::metal_elem_size` documents itself as mirroring "the runtime
//! allocator's `chelis_alloc` element width", and `runtime_dtype_tag`
//! is what the emitter actually passes to `chelis_alloc`. Those two
//! must agree for every dtype in the active Metal matrix, or the
//! device-to-host copy and the host allocation disagree on stride.

use chelis_backend_metal::dtype::{host_sizeof_expr, metal_elem_size, msl_type};
use chelis_types::types::Prim;
use chelis_vocab::Repr;

// Width parity is the weak form of this check and is kept only because it
// produces a more legible failure. The authoritative test is
// `device_representation_matches_the_runtime_representation` below: width is
// a projection of representation, and the projection is not injective, so two
// lanes can agree on width while holding incompatible bytes.
//
// The concrete case that motivated the rewrite: `Int32` is four bytes on both
// sides, so every width assertion passes regardless of encoding. Bool is
// caught by width only by luck, because 1 != 4.

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

/// Metal's device-side encoding for a `Prim`, derived from the MSL type the
/// emitter actually writes into the kernel source.
///
/// Mapping the emitted type rather than asserting a hardcoded table keeps this
/// honest: if `msl_type` changes, this follows it, and the comparison against
/// the runtime representation is what fails.
fn metal_device_repr(prec: Prim) -> Repr {
    match msl_type(prec) {
        "float" => Repr::Ieee754Binary32,
        "half" => Repr::Ieee754Binary16,
        "bfloat" => Repr::Bfloat16,
        "char" => Repr::TwosComplement8,
        "short" => Repr::TwosComplement16,
        "int" => Repr::TwosComplement32,
        "long" => Repr::TwosComplement64,
        // MSL `bool` is a native one-byte boolean. There is no `Repr` variant
        // for it precisely because nothing in the runtime represents bool as
        // itself; that absence is the finding, not an oversight.
        "bool" => panic!(
            "Metal emits a native MSL `bool`, which has no counterpart in the \
             runtime representation vocabulary. The runtime carries bool as an \
             f32 payload (Repr::BoolInBinary32), so the two lanes do not share \
             an encoding and a memcpy between them is not meaningful. Tracked \
             as CRuntime-BoolStorage-F1; the recorded closure is native 1-byte \
             bool in the runtime, at which point add a Repr::Bool8 and map it \
             here."
        ),
        other => panic!("unmapped MSL type: {other}"),
    }
}

/// The authoritative agreement check: same *encoding*, not merely same width.
///
/// This is what a width comparison cannot do. `Int32` passes both, but it
/// passes this one for the right reason -- both lanes are two's-complement 32 --
/// whereas it passes the width test for no reason at all.
#[test]
fn device_representation_matches_the_runtime_representation() {
    for prec in ACTIVE_METAL {
        if prec == Prim::Bool {
            // Covered by the dedicated bool test below, which explains why the
            // mapping cannot be expressed rather than merely failing.
            continue;
        }
        let runtime_repr = prec
            .runtime_dtype()
            .expect("active Metal dtype must have a runtime dtype")
            .repr();
        assert_eq!(
            metal_device_repr(prec),
            runtime_repr,
            "{}: Metal's device encoding and the runtime encoding differ. Equal \
             byte widths would not make these interchangeable; the copy path \
             needs a conversion, not a memcpy",
            prec.name()
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

    // The two lanes therefore do not share an encoding, and a memcpy between
    // them is not meaningful.
    //
    // Direction matters here. Widening Metal to f32 would match the current
    // ABI but move toward the f32-coupling that `CRuntime-F32Coupling`'s
    // closure set out to remove; `docs/gap_synthesis.md` records the intended
    // fix as native 1-byte bool in the runtime. Metal's 1-byte MSL bool is
    // already the target shape. So this test failing means the RUNTIME is the
    // side owing a migration, not Metal.
    assert_eq!(
        metal_elem_size(Prim::Bool),
        runtime_width,
        "metal_elem_size(bool) = {} but the runtime carries bool as a {}-byte \
         f32 payload. Until the runtime migrates to native 1-byte bool \
         (CRuntime-BoolStorage-F1), the device-to-host path must convert \
         MSL bool <-> f32 rather than copy bytes. Do not close this by \
         widening Metal to f32",
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
