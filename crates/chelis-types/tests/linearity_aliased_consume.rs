//! Structural consumption through an alias invalidates later reads of
//! the source binding. Alias links identify binding generations, so
//! multi-level aliases and destructured components follow the value
//! they reference. Violations are hard errors.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::CheckErrorKind;
use chelis_types::{check_linearity, check_typed_program};

fn linearity_errors(source: &str) -> Vec<chelis_types::errors::CheckError> {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    let checked = check_typed_program(&deep).expect("type check should succeed");
    check_linearity(&checked).expect_err("linearity check must error")
}

/// `realize(y)` consumes the value shared by `y` and `w`, so
/// `add(w, z)` reports `UseAfterConsume` on `w`.
#[test]
fn aliased_consume_bypass_errors_after_fix() {
    let errors = linearity_errors(
        r#"
def f(w: tensor[4, f32]) -> tensor[4, f32] =
  {
    y: tensor[4, f32] = w
    z: tensor[4, f32] = realize(y)
    add(w, z)
  }
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume) && e.message.contains("variable `w`")
        }),
        "expected UseAfterConsume on `w` consumed via aliased binding `y`; got {errors:?}"
    );
}

/// A destructured component remains consumed when `realize` acts
/// through an alias; `add(a, r)` must report `UseAfterConsume`.
#[test]
fn destructure_then_alias_consume_errors() {
    let errors = linearity_errors(
        r#"
def f(pair: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] =
  {
    (a, b) = pair
    y: tensor[4, f32] = a
    r: tensor[4, f32] = realize(y)
    add(a, r)
  }
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume) && e.message.contains("variable `a`")
        }),
        "expected UseAfterConsume on `a` via aliased binding `y` after destructure; \
         got {errors:?}"
    );
}
