//! Linearity-F1 + Linearity-F2 fixtures: pin the typed `ConsumeKind`
//! discrimination contract and the tuple-destructure linearity gap.
//!
//! Background. Today the linearity checker discriminates aliasing
//! consumes (`alias = x`) from structural consumes (realize / app-arg /
//! pipe-stage / closure-capture / match-scrutinee) by string-prefixing
//! the `ConsumeSite::description` on `"binding "` at
//! `crates/chelis-types/src/linearity.rs:726`. The brittleness was
//! filed as `Linearity-F1` in `docs/gap_synthesis.md`. The fix replaces
//! the string check with a typed `enum ConsumeKind { Aliasing,
//! Structural }` field on `ConsumeSite`.
//!
//! Tuple destructure (`let (a, b) = pair`) synthesizes
//! `__chelis_tmp_N` bindings at
//! `crates/chelis-surf/src/desugar.rs:1135-1156` without calling
//! `inject_type_metadata`. The resulting `(var __chelis_tmp_N)`
//! references have no `:type` entry in the meta-map, so
//! `Checker::expr_is_owned_linear` at L811-814 returns `false` and
//! every consume on a destructured component is silently skipped
//! (`Linearity-F2`).
//!
//! PR 1 of the W1 cascade (this commit) threads the tuple element
//! type onto the tmp-binding value so the references infer as owned
//! linear. The surfaced violations route through
//! `LinearityInfo::warnings` (mirroring the F3 PR 1 deprecation-window
//! pattern) pending W2-cascade flip to errors.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::CheckErrorKind;
use chelis_types::{LinearityInfo, check_linearity, check_typed_program};

/// Run linearity and return the resulting `LinearityInfo`. Fixtures
/// that expect a warning (not an error) call this to inspect
/// `LinearityInfo::warnings`.
fn linearity_info(source: &str) -> LinearityInfo {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    let checked = check_typed_program(&deep).expect("type check should succeed");
    let result = check_linearity(&checked).expect("linearity check should not error");
    result.linearity().clone()
}

/// Run linearity and return the surfaced errors. Fixtures that expect
/// a hard error call this.
fn linearity_errors(source: &str) -> Vec<chelis_types::errors::CheckError> {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    let checked = check_typed_program(&deep).expect("type check should succeed");
    check_linearity(&checked).expect_err("linearity check must error")
}

/// Run linearity and assert no errors and no warnings. Used by positive
/// controls that must stay clean before and after the fix.
fn assert_linearity_clean(source: &str) {
    let info = linearity_info(source);
    assert!(
        info.warnings().is_empty(),
        "expected zero warnings; got {:?}",
        info.warnings()
    );
}

/// Fixture 1: aliasing-consume control. `y = w` is an aliasing consume;
/// the only later use of either name is `realize(y)`. Must pass.
///
/// Before the fix: passes via the string-prefix check at L726.
/// After the fix: passes because the consume on `w` carries
/// `ConsumeKind::Aliasing` and `read_or_error` returns early on that
/// variant. The pinned outcome does not change; what changes is the
/// discrimination mechanism.
#[test]
fn aliasing_consume_control_passes() {
    assert_linearity_clean(
        r#"
def f(w: tensor[4, f32]): tensor[4, f32] =
  {
    y: tensor[4, f32] = w
    realize(y)
  }
"#,
    );
}

/// Fixture 2: structural-consume control. `a = realize(w)` is a
/// structural consume of `w`; a later borrow-read of `w` (passing
/// `w` to a read-only primitive like `add`) must fail.  The
/// discrimination axis is exercised inside `read_or_error` at
/// `linearity.rs:726`: today the string-prefix check on
/// `"binding "` distinguishes Aliasing from Structural consumes;
/// after Linearity-F1 the discrimination is by typed
/// `ConsumeKind::Structural` instead.
///
/// Before the fix: fails (the consume tag is `realize at offset N`,
/// not `binding ...`, so the string-prefix check does not exempt
/// it).  After the fix: fails for the same reason expressed via
/// `ConsumeKind::Structural`.  This mirrors the existing
/// `detects_use_after_consume` fixture in `linearity.rs:22-39`.
#[test]
fn structural_consume_control_errors() {
    let errors = linearity_errors(
        r#"
def f(w: tensor[4, f32]): tensor[4, f32] =
  {
    y: tensor[4, f32] = realize(w)
    add(w, y)
  }
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message.contains("variable `w`")
                && e.message.contains("realize")
        }),
        "expected UseAfterConsume on `w` consumed by `realize` then borrowed by `add`; \
         got {errors:?}"
    );
}

/// Fixture 4: tuple-destructure linearity (Linearity-F2). After
/// destructuring `(a, b) = pair`, a double `realize(a)` must be
/// flagged. PR 1 routes the violation through
/// `LinearityInfo::warnings` (warning-mode, deprecation window);
/// W2-cascade flips it to an error.
///
/// Today this fixture silently passes because the destructured
/// `(var __chelis_tmp_N)` references have no `:type` metadata, so
/// `expr_is_owned_linear` returns false and the linearity check
/// skips both `realize(a)` calls. Gated `#[ignore]` until W1.3
/// threads the type metadata onto the destructured tmp bindings.
#[test]
#[ignore = "blocked on destructure type-metadata threading (Linearity-F2)"]
fn tuple_destructure_double_realize_warns_after_fix() {
    let info = linearity_info(
        r#"
def f(pair: (tensor[4, f32], tensor[4, f32])): tensor[4, f32] =
  {
    (a, b) = pair
    r1: tensor[4, f32] = realize(a)
    realize(a)
  }
"#,
    );
    assert!(
        info.warnings().iter().any(|w| {
            matches!(w.kind, CheckErrorKind::UseAfterConsume) && w.message.contains("variable `a`")
        }),
        "expected a UseAfterConsume warning on `a` after double realize on a destructured \
         tuple component; got {:?}",
        info.warnings()
    );
}

/// Fixture 5: tuple-destructure positive control. `add(realize(a),
/// realize(b))` consumes each destructured component exactly once.
/// Must pass before and after the fix with zero warnings.
#[test]
fn tuple_destructure_single_consume_each_passes() {
    assert_linearity_clean(
        r#"
def f(pair: (tensor[4, f32], tensor[4, f32])): tensor[4, f32] =
  {
    (a, b) = pair
    add(realize(a), realize(b))
  }
"#,
    );
}
