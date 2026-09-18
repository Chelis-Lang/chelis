//! chelis#944: the reserved-but-deferred dtype names of
//! `spec/04-type-system.md` §1.1.1 rejected with a §1.1.1-citing
//! diagnostic.
//!
//! Mirrors the f8e4m3 §1.1.1 rejection contract from
//! `f8e4m3_rejection.rs` for the names PR #896 reserved: `f8e5m2`,
//! `int4`/`uint4`, `complex64`/`complex128`, `decimal128`/`decimal256`
//! (the unsigned `uint*` family has its own sibling suite in
//! `unsigned_dtype_rejection.rs`). `Prim::parse_name` returns `None`
//! for every one of these spellings, so without a dedicated rejection
//! path they fall to the generic unknown-name diagnostics with no
//! §1.1.1 citation - which is what the spec's diagnostic contract
//! forbids.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::check_ir_program;
use chelis_types::types::Prim;

const DEFERRED_NAMES: &[&str] = &[
    "f8e5m2",
    "int4",
    "uint4",
    "complex64",
    "complex128",
    "decimal128",
    "decimal256",
];

fn surf_to_deep(source: &str) -> Vec<chelis_deep::Expr> {
    let decls = parse_str(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

fn assert_deferred_rejection(src: &str, dtype_name: &str) {
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err(&format!(
        "`{dtype_name}` must be a type error per spec §1.1.1"
    ));
    let messages: Vec<&str> = rep.errors.iter().map(|e| e.message.as_str()).collect();
    assert!(
        messages
            .iter()
            .any(|m| m.contains(&format!("{dtype_name} is reserved but deferred"))),
        "expected diagnostic containing literal phrase \
         '{dtype_name} is reserved but deferred'; got: {messages:?}"
    );
    assert!(
        messages
            .iter()
            .any(|m| m.contains("spec/04-type-system.md §1.1.1")),
        "diagnostic must cite spec/04-type-system.md §1.1.1 for `{dtype_name}`; \
         got: {messages:?}"
    );
}

/// §1.1.1: none of the reserved names resolves to a `Prim` variant.
/// Reservation is a spelling claim, not an active dtype.
#[test]
fn parse_name_rejects_every_deferred_name() {
    for name in DEFERRED_NAMES {
        assert_eq!(
            Prim::parse_name(name),
            None,
            "spec §1.1.1: `{name}` is reserved but deferred; parse_name must \
             not resolve it"
        );
    }
}

/// Cast-target surface: `cast(1, <name>)` is rejected with the §1.1.1
/// diagnostic for every reserved name.
#[test]
fn cast_scalar_to_deferred_name_rejected_with_spec_1_1_1_diagnostic() {
    for name in DEFERRED_NAMES {
        assert_deferred_rejection(&format!("def main() -> i32 = cast(1, {name})"), name);
    }
}

/// Tensor-element surface: `tensor[3, <name>]` in a value-position
/// annotation is rejected with the §1.1.1 diagnostic for every reserved
/// name.
#[test]
fn tensor_element_deferred_name_rejected_with_spec_1_1_1_diagnostic() {
    for name in DEFERRED_NAMES {
        assert_deferred_rejection(
            &format!("def stash() -> tensor[3, {name}] = to_tensor([1, 2, 3])"),
            name,
        );
    }
}

/// Sig surface: a reserved name in a sig's tensor precision slot must
/// reach the §1.1.1 rejection path, not be admitted as a type variable
/// (the trap the unsigned family's desugar exclusion already guards against).
#[test]
fn sig_precision_deferred_name_rejected_not_quantified() {
    for name in DEFERRED_NAMES {
        let src = format!(
            "sig f[d]: tensor[d, {name}] -> tensor[d, {name}]\n\
             def f(x) = x"
        );
        assert_deferred_rejection(&src, name);
    }
}

/// Negative-parity twin: an ordinary lowercase name explicitly listed in a
/// sig remains a type variable, so the desugar exclusion is exactly the
/// reserved list and nothing wider.
#[test]
fn sig_precision_explicit_ordinary_tvar_still_quantifies() {
    let src = "sig f[d, p]: tensor[d, p] -> tensor[d, p]\n\
               def f(x) = x";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    assert!(
        res.is_ok(),
        "an ordinary lowercase precision name must still quantify; got: {:?}",
        res.err().map(|e| e.errors)
    );
}

/// Negative-parity twin: active dtypes sharing a prefix with reserved
/// names (`i32` vs `int4`, `f16` vs `f8e5m2`) must not trip the
/// deferred rejection path.
#[test]
fn active_dtypes_do_not_match_deferred_family() {
    for (src, what) in [
        ("def main() -> i32 = cast(1, i32)", "i32"),
        ("def main() -> f16 = cast(1.0, f16)", "f16"),
    ] {
        let deep = surf_to_deep(src);
        let res = check_ir_program(&deep);
        assert!(
            res.is_ok(),
            "`{what}` must NOT match the deferred family; got: {:?}",
            res.err().map(|e| e.errors)
        );
    }
}
