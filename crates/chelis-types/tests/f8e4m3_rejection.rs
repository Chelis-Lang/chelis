//! WS-A0 acceptance test (b): the type checker must reject f8e4m3 as a
//! tensor element type and as a scalar cast target with a diagnostic
//! that cites `spec/04-type-system.md §1.1.1` and contains the literal
//! phrase "f8e4m3 is deferred".
//!
//! These two assertions are pinned because they are the user-facing
//! contract: producers (humans, agents, code generators) must be able
//! to grep the diagnostic for a specific spec pointer and decide
//! whether to switch dtype or escalate.

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

#[test]
fn cast_scalar_to_f8e4m3_rejected_with_spec_diagnostic() {
    // `cast(1.0, f8e4m3)` should error at type-check time.
    let src = "def main() -> f32 = cast(1.0, f8e4m3)";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("cast to f8e4m3 must be a type error");
    let messages: Vec<&str> = rep.errors.iter().map(|e| e.message.as_str()).collect();
    assert!(
        messages.iter().any(|m| m.contains("f8e4m3 is deferred")),
        "expected diagnostic containing literal phrase 'f8e4m3 is deferred'; got: {messages:?}"
    );
    assert!(
        messages
            .iter()
            .any(|m| m.contains("spec/04-type-system.md §1.1.1")),
        "expected diagnostic citing spec/04-type-system.md §1.1.1; got: {messages:?}"
    );
}

#[test]
fn cast_tensor_to_f8e4m3_rejected_with_spec_diagnostic() {
    // `cast(t, f8e4m3)` on a tensor should error too, with the same
    // f8e4m3-specific diagnostic shape.
    let src = "def main(x: tensor[3, f32]) -> tensor[3, f32] = cast(x, f8e4m3)";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("cast tensor to f8e4m3 must be a type error");
    let messages: Vec<&str> = rep.errors.iter().map(|e| e.message.as_str()).collect();
    assert!(
        messages.iter().any(|m| m.contains("f8e4m3 is deferred")),
        "expected diagnostic containing literal phrase 'f8e4m3 is deferred'; got: {messages:?}"
    );
    assert!(
        messages
            .iter()
            .any(|m| m.contains("spec/04-type-system.md §1.1.1")),
        "expected diagnostic citing spec/04-type-system.md §1.1.1; got: {messages:?}"
    );
}

#[test]
fn declared_tensor_with_f8e4m3_rejected_with_spec_diagnostic() {
    // Direct `(t-tensor ... (t-prim {} f8e4m3))` use in a declared
    // signature must surface the same f8e4m3-specific message.
    let src = "def x() -> tensor[3, f8e4m3] = cast(to_tensor([1.0, 2.0, 3.0]), f8e4m3)";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("tensor[..., f8e4m3] must be a type error");
    let messages: Vec<&str> = rep.errors.iter().map(|e| e.message.as_str()).collect();
    assert!(
        messages.iter().any(|m| m.contains("f8e4m3 is deferred")),
        "expected diagnostic containing literal phrase 'f8e4m3 is deferred'; got: {messages:?}"
    );
    assert!(
        messages
            .iter()
            .any(|m| m.contains("spec/04-type-system.md §1.1.1")),
        "expected diagnostic citing spec/04-type-system.md §1.1.1; got: {messages:?}"
    );
}

#[test]
fn other_unsupported_precision_does_not_pretend_to_be_f8e4m3_path() {
    // Negative-shape: a generic unsupported precision should NOT trigger
    // the f8e4m3-specific message. Currently every active dtype is
    // supported by `is_valid_*`, so we check the diagnostic doesn't
    // mention the f8e4m3 deferral phrase for an int-precision cast.
    let src = "def main() -> int32 = cast(1, int32)";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            assert!(
                !err.message.contains("f8e4m3 is deferred"),
                "non-f8e4m3 diagnostics must not borrow the f8e4m3 message: {}",
                err.message
            );
        }
    }
}
