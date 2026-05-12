//! Linearity-F3 PR 1 fixtures: top-level statements wrapped in a
//! `module` declaration must not silently skip the linearity check.
//!
//! Background: PR #29 added cross-statement linearity tracking, and
//! PR #60 (V2-F4) tightened the top-level binding-aliasing path. Both
//! fixes apply only when the `chelis-types` pre-declare loop in
//! `check_linearity` actually pre-declares the top-level names. The
//! loop filters on `get_tag(list) == Some("def")` and therefore misses
//! `(module {} name (def ...) (def ...))` wrappers, which is how every
//! idiomatic Chelis source desugars
//! (`crates/chelis-surf/src/desugar.rs:614-620`). The bug was flagged
//! as a §5 follow-up in
//! `docs/investigations/var_rhs_aliased_fanout_v2_diagnosis.md`
//! ("Out-of-scope" section, item 2).
//!
//! Each fixture asserts that the SAME violation that fires for a
//! bare top-level program ALSO surfaces (as a warning) when the
//! program is wrapped in `module Test`. The fixtures are gated with
//! `#[ignore]` while the warning-mode plumbing is being built; the
//! fix commit removes the gate.
//!
//! For PR 1, surfaced violations are warnings, not errors, so
//! `check_linearity` still returns `Ok(_)` and the violation is
//! reachable via `program.linearity().warnings()`. PR 2 will fix the
//! corpus and flip the severity.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::CheckErrorKind;
use chelis_types::{check_linearity, check_typed_program};

/// Run the linearity check on a Surf source. Returns the
/// `LinearityInfo` so callers can inspect both the warnings and the
/// errors. Errors that surface from `check_linearity` are surfaced as
/// `Err` so a fixture that expects warnings can also assert that no
/// hard errors fire.
fn linearity_of_surf(
    source: &str,
) -> Result<chelis_types::LinearityInfo, Vec<chelis_types::errors::CheckError>> {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    let checked = check_typed_program(&deep).expect("type check should succeed");
    let checked = check_linearity(&checked)?;
    Ok(checked.linearity().clone())
}

/// Sanity check that the bare (no-`module`) variant of each fixture
/// still produces a hard linearity error today. These mirror the
/// existing behavior locked by
/// `crates/chelis-types/tests/linearity.rs::detects_use_after_consume`.
/// Used to prove the warning fixtures below test the same violation,
/// just routed through the module path.
fn bare_program_errors(source: &str) -> Vec<chelis_types::errors::CheckError> {
    linearity_of_surf(source).expect_err("bare top-level program must still error today")
}

#[test]
fn bare_top_level_realize_then_borrow_errors_today() {
    let errors = bare_program_errors(
        r#"
x = to_tensor([1.0, 2.0, 3.0])
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
    let errors = bare_program_errors(
        r#"
def consume_it(t: tensor[3, f32]) -> tensor[3, f32] = realize(t)

x = to_tensor([1.0, 2.0, 3.0])
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
    let errors = bare_program_errors(
        r#"
x = to_tensor([1.0, 2.0, 3.0])
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
fn module_wrapped_realize_then_borrow_emits_warning() {
    // The same statements as `bare_top_level_realize_then_borrow_errors_today`,
    // but wrapped in `module Test`. After Linearity-F3 PR 1 the
    // linearity checker must run on the inner defs and surface the
    // violation as a warning in `LinearityInfo::warnings()`. The
    // checker must still return `Ok(_)` because warnings do not block
    // the build in PR 1.
    let info = linearity_of_surf(
        r#"
module Test

x = to_tensor([1.0, 2.0, 3.0])
y = realize(x)
b = add(x, y)
"#,
    )
    .expect("module-wrapped linearity check must succeed in PR 1 (warnings, not errors)");
    let warnings = info.warnings();
    assert!(
        warnings.iter().any(|w| {
            matches!(w.kind, CheckErrorKind::UseAfterConsume)
                && w.message.contains("variable `x`")
                && w.message.contains("realize")
        }),
        "expected a UseAfterConsume warning for `x` consumed by `realize`; got {warnings:?}"
    );
}

#[test]
fn module_wrapped_consuming_call_then_borrow_emits_warning() {
    // Mirrors `bare_top_level_consuming_call_then_borrow_errors_today`
    // but inside a module. The consume site is the call to
    // `consume_it`, so the warning message must call that out.
    let info = linearity_of_surf(
        r#"
module Test

def consume_it(t: tensor[3, f32]) -> tensor[3, f32] = realize(t)

x = to_tensor([1.0, 2.0, 3.0])
y = consume_it(x)
b = mul(x, y)
"#,
    )
    .expect("module-wrapped linearity check must succeed in PR 1 (warnings, not errors)");
    let warnings = info.warnings();
    assert!(
        warnings.iter().any(|w| {
            matches!(w.kind, CheckErrorKind::UseAfterConsume)
                && w.message.contains("variable `x`")
                && w.message.contains("consume_it")
        }),
        "expected a UseAfterConsume warning for `x` consumed by `consume_it`; got {warnings:?}"
    );
}

#[test]
fn module_wrapped_multi_realize_then_borrow_emits_warning() {
    // PR #60 (V2-F4) tightened the binding-aliasing path for bare
    // top-level statements (`y = x` chains). The same chain wrapped
    // in `module Test` currently slips through because the pre-declare
    // loop never sees the inner defs. After Linearity-F3 PR 1 the
    // module-recursive walk must surface the same use-after-consume
    // as a warning.
    let info = linearity_of_surf(
        r#"
module Test

x = to_tensor([1.0, 2.0, 3.0])
y = realize(x)
b = realize(x)
c = realize(x)
d = add(x, y)
"#,
    )
    .expect("module-wrapped linearity check must succeed in PR 1 (warnings, not errors)");
    let warnings = info.warnings();
    assert!(
        warnings.iter().any(|w| {
            matches!(w.kind, CheckErrorKind::UseAfterConsume)
                && w.message.contains("variable `x`")
                && w.message.contains("realize")
        }),
        "expected a UseAfterConsume warning for `x` consumed by `realize`; got {warnings:?}"
    );
}
