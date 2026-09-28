//! Aliasing bindings share values without destroying them; structural
//! consumes invalidate later borrows. Destructured tuple components
//! participate in the same rule and report violations as errors.

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

/// Binding `y = w` shares the value; one `realize(y)` is valid.
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

/// `realize(w)` consumes `w`; borrowing it in `add` reports
/// `UseAfterConsume`.
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

/// A destructured component cannot be realized twice.
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
