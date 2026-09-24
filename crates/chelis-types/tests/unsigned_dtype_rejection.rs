//! WS-A0 RT-1 fixup A1: unsigned dtype names rejected with the
//! §1.1.1 deferral diagnostic.
//!
//! Mirrors the f8e4m3 §1.1.1 rejection contract from
//! `f8e4m3_rejection.rs` but for the unsigned-integer family, reserved
//! as deferred by `spec/04-type-system.md` §1.1.1 (§1.1.2 names the
//! `uint*` spellings canonical). `Prim::parse_name` returns `None`
//! for these spellings, so without a dedicated rejection path the cast
//! falls through to `Type::Error` with no user-facing diagnostic.

use chelis_types::check_ir_program;

fn parse_core(source: &str) -> Vec<chelis_deep::Expr> {
    chelis_deep::parse_and_stamp_file(source).expect("valid core fixture")
}

fn assert_unsigned_rejection(src: &str, dtype_name: &str) {
    let deep = parse_core(src);
    let report = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked)
        .expect_err("a deferred dtype must reject");
    assert!(
        report.errors.iter().any(|error| matches!(
            error.kind,
            chelis_types::errors::CheckErrorKind::UnsupportedTensorPrecision
        ) && error.message.contains(dtype_name)),
        "{dtype_name}: {:?}",
        report.errors
    );
}

#[test]
fn cast_scalar_to_u8_rejected_with_spec_1_1_1_diagnostic() {
    assert_unsigned_rejection(
        include_str!(
            "fixtures/core/tests/unsigned_dtype_rejection/cast_scalar_to_u8_rejected_with_spec_1_1_1_diagnostic.0.deep"
        ),
        "u8",
    );
}

#[test]
fn cast_scalar_to_u16_rejected_with_spec_1_1_1_diagnostic() {
    assert_unsigned_rejection(
        include_str!(
            "fixtures/core/tests/unsigned_dtype_rejection/cast_scalar_to_u16_rejected_with_spec_1_1_1_diagnostic.0.deep"
        ),
        "u16",
    );
}

#[test]
fn cast_scalar_to_u32_rejected_with_spec_1_1_1_diagnostic() {
    assert_unsigned_rejection(
        include_str!(
            "fixtures/core/tests/unsigned_dtype_rejection/cast_scalar_to_u32_rejected_with_spec_1_1_1_diagnostic.0.deep"
        ),
        "u32",
    );
}

#[test]
fn cast_scalar_to_u64_rejected_with_spec_1_1_1_diagnostic() {
    assert_unsigned_rejection(
        include_str!(
            "fixtures/core/tests/unsigned_dtype_rejection/cast_scalar_to_u64_rejected_with_spec_1_1_1_diagnostic.0.deep"
        ),
        "u64",
    );
}

#[test]
fn cast_scalar_to_uint8_alias_rejected_with_spec_1_1_1_diagnostic() {
    assert_unsigned_rejection(
        include_str!(
            "fixtures/core/tests/unsigned_dtype_rejection/cast_scalar_to_uint8_alias_rejected_with_spec_1_1_1_diagnostic.0.deep"
        ),
        "uint8",
    );
}

#[test]
fn cast_scalar_to_uint16_alias_rejected_with_spec_1_1_1_diagnostic() {
    assert_unsigned_rejection(
        include_str!(
            "fixtures/core/tests/unsigned_dtype_rejection/cast_scalar_to_uint16_alias_rejected_with_spec_1_1_1_diagnostic.0.deep"
        ),
        "uint16",
    );
}

#[test]
fn cast_scalar_to_uint32_alias_rejected_with_spec_1_1_1_diagnostic() {
    assert_unsigned_rejection(
        include_str!(
            "fixtures/core/tests/unsigned_dtype_rejection/cast_scalar_to_uint32_alias_rejected_with_spec_1_1_1_diagnostic.0.deep"
        ),
        "uint32",
    );
}

#[test]
fn cast_scalar_to_uint64_alias_rejected_with_spec_1_1_1_diagnostic() {
    assert_unsigned_rejection(
        include_str!(
            "fixtures/core/tests/unsigned_dtype_rejection/cast_scalar_to_uint64_alias_rejected_with_spec_1_1_1_diagnostic.0.deep"
        ),
        "uint64",
    );
}

#[test]
fn tensor_element_u8_rejected_with_spec_1_1_1_diagnostic() {
    let src = include_str!(
        "fixtures/core/tests/unsigned_dtype_rejection/tensor_element_u8_rejected_with_spec_1_1_1_diagnostic.0.deep"
    );
    let deep = parse_core(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    let rep = res.expect_err("tensor[..., u8] must error per spec §1.1.1");
    let messages: Vec<&str> = rep.errors.iter().map(|e| e.message.as_str()).collect();
    assert!(
        messages
            .iter()
            .any(|m| m.contains("unsigned integer types are deferred")),
        "expected §1.1.1 diagnostic for tensor[..., u8]; got: {messages:?}"
    );
}

/// Negative-parity twin: `i32` (a valid signed dtype) must NOT trip
/// the unsigned rejection path. Pinning this guards against an overly
/// eager regex catching `i32` as if it were `uint32`.
#[test]
fn cast_scalar_to_int32_does_not_match_unsigned_family() {
    let src = include_str!(
        "fixtures/core/tests/unsigned_dtype_rejection/cast_scalar_to_int32_does_not_match_unsigned_family.0.deep"
    );
    let deep = parse_core(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    assert!(
        res.is_ok(),
        "i32 cast must NOT match the unsigned family; got: {:?}",
        res.err().map(|e| e.errors)
    );
}
