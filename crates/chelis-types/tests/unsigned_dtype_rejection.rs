//! WS-A0 RT-1 fixup A1: unsigned dtype names rejected with the
//! §1.1.1 deferral diagnostic.
//!
//! Mirrors the f8e4m3 §1.1.1 rejection contract from
//! `f8e4m3_rejection.rs` but for the unsigned-integer family, reserved
//! as deferred by `spec/04-type-system.md` §1.1.1 (§1.1.2 names the
//! `uint*` spellings canonical). `Prim::parse_name` returns `None`
//! for these spellings, so without a dedicated rejection path the cast
//! falls through to `Type::Error` with no user-facing diagnostic.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::check_ir_program;

fn surf_to_deep(source: &str) -> Vec<chelis_deep::Expr> {
    let decls = parse_str(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

fn assert_unsigned_rejection(src: &str, dtype_name: &str) {
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err(&format!(
        "cast to `{dtype_name}` must be a type error per spec §1.1.1"
    ));
    let messages: Vec<&str> = rep.errors.iter().map(|e| e.message.as_str()).collect();
    assert!(
        messages
            .iter()
            .any(|m| m.contains("unsigned integer types are deferred")),
        "expected diagnostic containing literal phrase \
         'unsigned integer types are deferred' for `{dtype_name}`; \
         got: {messages:?}"
    );
    assert!(
        messages
            .iter()
            .any(|m| m.contains("spec/04-type-system.md §1.1.1")),
        "diagnostic must cite spec/04-type-system.md §1.1.1 for `{dtype_name}`; \
         got: {messages:?}"
    );
}

#[test]
fn cast_scalar_to_u8_rejected_with_spec_1_1_1_diagnostic() {
    assert_unsigned_rejection("def main() -> int32 = cast(1, u8)", "u8");
}

#[test]
fn cast_scalar_to_u16_rejected_with_spec_1_1_1_diagnostic() {
    assert_unsigned_rejection("def main() -> int32 = cast(1, u16)", "u16");
}

#[test]
fn cast_scalar_to_u32_rejected_with_spec_1_1_1_diagnostic() {
    assert_unsigned_rejection("def main() -> int32 = cast(1, u32)", "u32");
}

#[test]
fn cast_scalar_to_u64_rejected_with_spec_1_1_1_diagnostic() {
    assert_unsigned_rejection("def main() -> int32 = cast(1, u64)", "u64");
}

#[test]
fn cast_scalar_to_uint8_alias_rejected_with_spec_1_1_1_diagnostic() {
    assert_unsigned_rejection("def main() -> int32 = cast(1, uint8)", "uint8");
}

#[test]
fn cast_scalar_to_uint16_alias_rejected_with_spec_1_1_1_diagnostic() {
    assert_unsigned_rejection("def main() -> int32 = cast(1, uint16)", "uint16");
}

#[test]
fn cast_scalar_to_uint32_alias_rejected_with_spec_1_1_1_diagnostic() {
    assert_unsigned_rejection("def main() -> int32 = cast(1, uint32)", "uint32");
}

#[test]
fn cast_scalar_to_uint64_alias_rejected_with_spec_1_1_1_diagnostic() {
    assert_unsigned_rejection("def main() -> int32 = cast(1, uint64)", "uint64");
}

#[test]
fn tensor_element_u8_rejected_with_spec_1_1_1_diagnostic() {
    let src = "def main() -> tensor[3, int32] = cast(to_tensor([1, 2, 3]), int32)\n\
               def stash() -> tensor[3, u8] = to_tensor([1, 2, 3])";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("tensor[..., u8] must error per spec §1.1.1");
    let messages: Vec<&str> = rep.errors.iter().map(|e| e.message.as_str()).collect();
    assert!(
        messages
            .iter()
            .any(|m| m.contains("unsigned integer types are deferred")),
        "expected §1.1.1 diagnostic for tensor[..., u8]; got: {messages:?}"
    );
}

/// Negative-parity twin: `int32` (a valid signed dtype) must NOT trip
/// the unsigned rejection path. Pinning this guards against an overly
/// eager regex catching `int32` as if it were `uint32`.
#[test]
fn cast_scalar_to_int32_does_not_match_unsigned_family() {
    let src = "def main() -> int32 = cast(1, int32)";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    assert!(
        res.is_ok(),
        "int32 cast must NOT match the unsigned family; got: {:?}",
        res.err().map(|e| e.errors)
    );
}
