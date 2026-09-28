//! Linearity-F1 + Linearity-F2 fixtures: pin the typed `ConsumeKind`
//! discrimination contract and the tuple-destructure linearity gap.
//!
//! Background. Before Linearity-F1 (W1 PR #83) the linearity checker
//! discriminated aliasing consumes (`alias = x`) from structural
//! consumes (realize / app-arg / pipe-stage / closure-capture /
//! match-scrutinee) by string-prefixing the `ConsumeSite::description`
//! on `"binding "` at `crates/chelis-types/src/linearity.rs:726`. The
//! brittleness was filed as `Linearity-F1` in `docs/archive/reports/gap_synthesis.md`.
//! The fix replaces the string check with a typed
//! `enum ConsumeKind { Aliasing, Structural }` field on `ConsumeSite`.
//!
//! Tuple destructure (`let (a, b) = pair`) synthesizes
//! `__chelis_tmp_N` bindings at
//! `crates/chelis-surf/src/desugar.rs:1135-1156`. Before W1 the
//! resulting `(var __chelis_tmp_N)` references had no `:type` entry
//! in the meta-map, so `Checker::expr_is_owned_linear` at L811-814
//! returned `false` and every consume on a destructured component
//! was silently skipped (`Linearity-F2`).
//!
//! W1 PR #83 closed the false-negative by resolving destructured
//! tuple-component types at linearity-check time via a local
//! `tuple_get_element_type` helper. PR #83 also reintroduced
//! `LinearityInfo::warnings` to route surfaced violations during a
//! deprecation window. The W2-cascade PR (this commit) closes the
//! deprecation window: the corpus survey
//! (`docs/investigations/linearity_destructure_cleanup_survey.md`)
//! confirmed zero in-tree warnings, so the channel removal is total
//! and violations on destructured components surface as errors.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::CheckErrorKind;
use chelis_types::{check_linearity, check_typed_program};

/// Run linearity and return the surfaced errors. Fixtures that
/// expect a hard error call this.
fn linearity_errors(source: &str) -> Vec<chelis_types::errors::CheckError> {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    let checked = check_typed_program(&deep).expect("type check should succeed");
    check_linearity(&checked).expect_err("linearity check must error")
}

/// Run linearity and assert no errors. Used by positive controls
/// that must stay clean.
fn assert_linearity_clean(source: &str) {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    let checked = check_typed_program(&deep).expect("type check should succeed");
    check_linearity(&checked).expect("linearity check should not error");
}

/// Fixture 1: aliasing-consume control. `y = w` is an aliasing consume;
/// the only later use of either name is `realize(y)`. Must pass.
///
/// After Linearity-F1: passes because the consume on `w` carries
/// `ConsumeKind::Aliasing` and `read_or_error` returns early on that
/// variant. The pinned outcome does not change; what changes is the
/// discrimination mechanism (typed field, not string prefix).
#[test]
fn aliasing_consume_control_passes() {
    assert_linearity_clean(
        r#"
def f(w: tensor[4, f32]) -> tensor[4, f32] =
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
/// discrimination axis is exercised inside `read_or_error`: after
/// Linearity-F1 the discrimination is by typed
/// `ConsumeKind::Structural` (replacing the prior string-prefix
/// check on the description). This mirrors the existing
/// `detects_use_after_consume` fixture in `linearity.rs:22-39`.
#[test]
fn structural_consume_control_errors() {
    let errors = linearity_errors(
        r#"
def f(w: tensor[4, f32]) -> tensor[4, f32] =
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
/// destructuring `(a, b) = pair`, a double `realize(a)` is flagged
/// as a hard error. PR #83 closed the false-negative; the W2
/// cascade flips the surfaced violation from warning to error
/// after the corpus survey confirmed zero in-tree warnings to
/// clean up.
#[test]
fn tuple_destructure_double_realize_errors() {
    let errors = linearity_errors(
        r#"
def f(pair: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] =
  {
    (a, b) = pair
    r1: tensor[4, f32] = realize(a)
    realize(a)
  }
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume)
                && e.message.contains("variable `a`")
                && e.message.contains("destructured")
        }),
        "expected UseAfterConsume on `a` after double realize on a destructured \
         tuple component; got {errors:?}"
    );
}

/// Fixture 5: tuple-destructure positive control. `add(realize(a),
/// realize(b))` consumes each destructured component exactly once.
/// Must pass with no errors.
#[test]
fn tuple_destructure_single_consume_each_passes() {
    assert_linearity_clean(
        r#"
def f(pair: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] =
  {
    (a, b) = pair
    add(realize(a), realize(b))
  }
"#,
    );
}

/// A destructuring marker governs the bindings it introduces, not the
/// expression evaluated to produce the first temporary. In particular, a
/// wildcard discard after an earlier use of an ordinary parameter must retain
/// the normal implicit-copy behavior for that parameter.
#[test]
fn destructuring_scope_starts_after_the_root_binding_value() {
    assert_linearity_clean(
        r#"
def f(x: tensor[4, f32]) -> unit =
  {
    y: tensor[4, f32] = realize(x)
    _ = drop(x)
    ()
  }
"#,
    );
}
