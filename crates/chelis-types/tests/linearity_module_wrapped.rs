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
use chelis_types::CopyRepairUseKind;
use chelis_types::errors::CheckErrorKind;
use chelis_types::{check_linearity, check_typed_program};
use copy_repair::assert_copy_repaired;

#[path = "support/copy_repair.rs"]
mod copy_repair;

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
fn bare_top_level_realize_then_borrow_is_copy_repaired() {
    assert_copy_repaired(
        r#"
x = to_tensor([1.0, 2.0, 3.0], f32)
y = realize(x)
b = add(x, y)
"#,
        "x",
        "realize",
        CopyRepairUseKind::Borrow,
    );
}

#[test]
fn bare_top_level_consuming_call_then_borrow_is_copy_repaired() {
    assert_copy_repaired(
        r#"
def consume_it(t: tensor[3, f32]) -> tensor[3, f32] = realize(t)

x = to_tensor([1.0, 2.0, 3.0], f32)
y = consume_it(x)
b = mul(x, y)
"#,
        "x",
        "consume_it",
        CopyRepairUseKind::Borrow,
    );
}

#[test]
fn bare_top_level_multi_realize_then_borrow_is_copy_repaired() {
    assert_copy_repaired(
        r#"
x = to_tensor([1.0, 2.0, 3.0], f32)
y = realize(x)
b = realize(x)
c = realize(x)
d = add(x, y)
"#,
        "x",
        "realize",
        CopyRepairUseKind::Borrow,
    );
}

#[test]
fn module_wrapped_realize_then_borrow_is_copy_repaired() {
    // The module-recursive walk tracks the consume as the bare walk does.
    assert_copy_repaired(
        r#"
module Test

x = to_tensor([1.0, 2.0, 3.0], f32)
y = realize(x)
b = add(x, y)
"#,
        "x",
        "realize",
        CopyRepairUseKind::Borrow,
    );
}

#[test]
fn module_wrapped_consuming_call_then_borrow_is_copy_repaired() {
    assert_copy_repaired(
        r#"
module Test

def consume_it(t: tensor[3, f32]) -> tensor[3, f32] = realize(t)

x = to_tensor([1.0, 2.0, 3.0], f32)
y = consume_it(x)
b = mul(x, y)
"#,
        "x",
        "consume_it",
        CopyRepairUseKind::Borrow,
    );
}

#[test]
fn module_wrapped_multi_realize_then_borrow_is_copy_repaired() {
    assert_copy_repaired(
        r#"
module Test

x = to_tensor([1.0, 2.0, 3.0], f32)
y = realize(x)
b = realize(x)
c = realize(x)
d = add(x, y)
"#,
        "x",
        "realize",
        CopyRepairUseKind::Borrow,
    );
}

/// A use after a `drop` is not fan-out (spec/04 section 8.3, [04-LIN-11]), so
/// it is the violation that must surface as an error, bare or module-wrapped.
#[test]
fn bare_top_level_drop_then_borrow_errors() {
    let errors = linearity_errors(
        r#"
x = to_tensor([1.0, 2.0, 3.0], f32)
y = drop(x)
b = add(x, x)
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message.contains("variable `x`")
                && e.message.contains("drop")
        }),
        "expected a UseAfterConsume error for `x` after `drop`; got {errors:?}"
    );
}

#[test]
fn module_wrapped_drop_then_borrow_errors() {
    let errors = linearity_errors(
        r#"
module Test

x = to_tensor([1.0, 2.0, 3.0], f32)
y = drop(x)
b = add(x, x)
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message.contains("variable `x`")
                && e.message.contains("drop")
        }),
        "expected a UseAfterConsume error for `x` after `drop`; got {errors:?}"
    );
}
