//! Linearity-F3 PR 2 fixtures: top-level statements wrapped in a
//! `module` declaration must reject linearity violations as errors,
//! matching the bare-top-level path.
//!
//! Background: PR 1 (#65) extended the pre-declare loop and main walk
//! in `check_linearity` to recurse through `(module {} name ...)`
//! wrappers, and emitted the surfaced violations as warnings on
//! `LinearityInfo::warnings()` for a deprecation window. PR 1's corpus
//! sweep across `examples/`, `examples/illustrative/`, `packages/`,
//! and `crates/*/tests/` found zero surfaced warnings, so PR 2's
//! "fix surfaced violations" step was a no-op. PR 2 flips the severity:
//! module-wrapped violations now route through `Checker::errors`, the
//! warning plumbing on `LinearityInfo` is removed, and the CLI's
//! `warning: linearity: ...` stderr emission is gone.
//!
//! Each fixture asserts that the SAME violation that fires for a bare
//! top-level program ALSO surfaces as a hard error when the program
//! is wrapped in `module Test`. This is the post-PR-2 contract.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::CheckErrorKind;
use chelis_types::{check_linearity, check_typed_program};

/// Run the linearity check on a Surf source. Returns the errors so
/// fixtures that assert a violation can match on
/// `CheckErrorKind::UseAfterConsume` and message content.
fn linearity_errors(source: &str) -> Vec<chelis_types::errors::CheckError> {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    let checked = check_typed_program(&deep).expect("type check should succeed");
    check_linearity(&checked).expect_err("linearity check must error on use-after-consume")
}

#[test]
fn bare_top_level_realize_then_borrow_errors_today() {
    let errors = linearity_errors(
        r#"
x = to_tensor([1.0, 2.0, 3.0], f32)
y = realize(x)
b = add(x, y)
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message.contains("variable `x`")
                && e.message.contains("realize")
        }),
        "bare top-level realize-then-borrow must still error (this is the control); got {errors:?}"
    );
}

#[test]
fn bare_top_level_consuming_call_then_borrow_errors_today() {
    let errors = linearity_errors(
        r#"
def consume_it(t: tensor[3, f32]) -> tensor[3, f32] = realize(t)

x = to_tensor([1.0, 2.0, 3.0], f32)
y = consume_it(x)
b = mul(x, y)
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message.contains("variable `x`")
                && e.message.contains("consume_it")
        }),
        "bare top-level consuming-call-then-borrow must still error (control); got {errors:?}"
    );
}

#[test]
fn bare_top_level_multi_realize_then_borrow_errors_today() {
    let errors = linearity_errors(
        r#"
x = to_tensor([1.0, 2.0, 3.0], f32)
y = realize(x)
b = realize(x)
c = realize(x)
d = add(x, y)
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message.contains("variable `x`")
                && e.message.contains("realize")
        }),
        "bare top-level multi-realize-then-borrow must still error (control); got {errors:?}"
    );
}

#[test]
fn module_wrapped_realize_then_borrow_errors() {
    // The same statements as `bare_top_level_realize_then_borrow_errors_today`,
    // but wrapped in `module Test`. After Linearity-F3 PR 2 the
    // module-recursive walk must surface the violation as a hard error
    // — the warning-mode plumbing from PR 1 is gone.
    let errors = linearity_errors(
        r#"
module Test

x = to_tensor([1.0, 2.0, 3.0], f32)
y = realize(x)
b = add(x, y)
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message.contains("variable `x`")
                && e.message.contains("realize")
        }),
        "expected a UseAfterConsume error for `x` consumed by `realize`; got {errors:?}"
    );
}

#[test]
fn module_wrapped_consuming_call_then_borrow_errors() {
    // Mirrors `bare_top_level_consuming_call_then_borrow_errors_today`
    // but inside a module. The consume site is the call to
    // `consume_it`, so the error message must call that out.
    let errors = linearity_errors(
        r#"
module Test

def consume_it(t: tensor[3, f32]) -> tensor[3, f32] = realize(t)

x = to_tensor([1.0, 2.0, 3.0], f32)
y = consume_it(x)
b = mul(x, y)
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message.contains("variable `x`")
                && e.message.contains("consume_it")
        }),
        "expected a UseAfterConsume error for `x` consumed by `consume_it`; got {errors:?}"
    );
}

#[test]
fn module_wrapped_multi_realize_then_borrow_errors() {
    // PR #60 (V2-F4) tightened the binding-aliasing path for bare
    // top-level statements (`y = x` chains). After Linearity-F3 PR 2
    // the same chain wrapped in `module Test` must surface the use-
    // after-consume as a hard error, not a warning.
    let errors = linearity_errors(
        r#"
module Test

x = to_tensor([1.0, 2.0, 3.0], f32)
y = realize(x)
b = realize(x)
c = realize(x)
d = add(x, y)
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message.contains("variable `x`")
                && e.message.contains("realize")
        }),
        "expected a UseAfterConsume error for `x` consumed by `realize`; got {errors:?}"
    );
}
