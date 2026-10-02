//! First-class `count` checker contract (chelis#1287 / [05-OP-29]).
//!
//! Positive and negative cases are paired deliberately: `count` accepts only
//! bool tensors, one or more compile-time i32 axes, and returns an i64
//! tensor with every selected axis removed.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_typed_program;
use chelis_types::errors::CheckError;

fn errors(source: &str) -> Vec<CheckError> {
    let decls = parse_surf(source).expect("Surf fixture parses");
    match check_typed_program(&desugar_program(&decls).expect("Surf fixture must desugar")) {
        Ok(_) => Vec::new(),
        Err(report) => report.errors,
    }
}

fn assert_clean(source: &str) {
    let found = errors(source);
    assert!(found.is_empty(), "expected clean typecheck, got {found:#?}");
}

fn assert_rejects(source: &str, needle: &str) {
    let found = errors(source);
    assert!(
        found.iter().any(|error| match needle {
            "arity" => {
                error.kind.diagnostic_name() == "ArityMismatch"
                    && error.expected.as_deref() == Some("2 arguments")
                    && error.got.as_deref() == Some("1 argument")
                    && error.message.contains("count")
            }
            "axis dtype" => {
                error.kind.diagnostic_name() == "TypeMismatch"
                    && error.expected.as_deref() == Some("i32")
                    && error.got.as_deref() == Some("i64")
                    && error.message.contains("count")
                    && error.message.contains("axis")
            }
            _ => error.message.contains(needle),
        }),
        "count: missing rejection for {needle:?}, got {found:#?}"
    );
}

#[test]
fn concrete_positional_axes_are_variadic_and_order_independent() {
    assert_clean(
        r#"
def f(x: tensor[2, 3, 4, bool]) -> tensor[3, i64] = count(&x, 0, -1)
def g(x: tensor[2, 3, 4, bool]) -> tensor[3, i64] = count(&x, 2, 0)
"#,
    );
}

#[test]
fn bool_tensor_is_required_not_numeric_tensor_or_bool_scalar() {
    assert_clean("def good(x: tensor[4, bool]) -> tensor[i64] = count(&x, 0)");
    assert_rejects(
        "def bad(x: tensor[4, i64]) -> tensor[i64] = count(&x, 0)",
        "bool tensor",
    );
    assert_rejects(
        "def bad(x: bool) -> tensor[i64] = count(x, 0)",
        "tensor input",
    );
    assert_rejects(
        "def bad[p](x: tensor[4, p]) -> tensor[i64] = count(&x, 0)",
        "exactly a bool tensor",
    );
    assert_rejects(
        r#"
def count_any[p](x: tensor[4, p]) -> tensor[i64] = count(&x, 0)
def bad(x: tensor[4, i64]) -> tensor[i64] = count_any(x)
"#,
        "exactly a bool tensor",
    );
}

#[test]
fn axes_are_required_unique_static_int32_and_in_range() {
    assert_rejects(
        "def bad(x: tensor[2, 3, bool]) -> tensor[2, 3, i64] = count(&x)",
        "arity",
    );
    assert_rejects(
        "def bad(x: tensor[2, 3, bool]) -> tensor[i64] = count(&x, 0, 0)",
        "duplicate",
    );
    assert_rejects(
        "def bad(x: tensor[2, 3, bool]) -> tensor[2, i64] = count(&x, 1i64)",
        "axis dtype",
    );
    assert_rejects(
        "def bad(x: tensor[2, 3, bool], axis: i32) -> tensor[2, i64] = count(&x, axis)",
        "compile-time constant",
    );
    assert_rejects(
        "def bad(x: tensor[2, 3, bool]) -> tensor[2, i64] = count(&x, 2)",
        "out of bounds",
    );
}

#[test]
fn variadic_axes_enforce_int32_in_every_position() {
    assert_clean(
        r#"
def first_cast(x: tensor[2, 3, bool]) -> tensor[i64] = count(&x, cast(0, i32), 1)
def later_cast(x: tensor[2, 3, bool]) -> tensor[i64] = count(&x, 0, cast(1, i32))
"#,
    );

    assert_rejects(
        "def bad(x: tensor[2, 3, bool]) -> tensor[i64] = count(&x, 0i64, 1)",
        "axis dtype",
    );
    assert_rejects(
        "def bad(x: tensor[2, 3, bool]) -> tensor[i64] = count(&x, 0, 1i64)",
        "axis dtype",
    );
    assert_rejects(
        "def bad(x: tensor[2, 3, bool]) -> tensor[i64] = count(&x, cast(0, i64), 1)",
        "axis dtype",
    );
    assert_rejects(
        "def bad(x: tensor[2, 3, bool]) -> tensor[i64] = count(&x, 0, cast(1, i64))",
        "axis dtype",
    );
}

#[test]
fn rank_polymorphic_count_uses_only_named_axes() {
    assert_clean(
        r#"
def good[pre, post](x: &tensor[..pre, seq, ..post, bool]) -> tensor[..pre, ..post, i64] = count(&x, seq)
"#,
    );
    assert_rejects(
        r#"
def bad[pre, post](x: &tensor[..pre, seq, ..post, bool]) -> tensor[..pre, ..post, i64] = count(&x, 0)
"#,
        "rank-spread",
    );
    assert_rejects(
        r#"
def bad[pre, post](x: &tensor[..pre, seq, ..post, bool]) -> tensor[..pre, ..post, i64] = count(&x, seq, seq)
"#,
        "duplicate",
    );
    assert_rejects(
        r#"
def bad[pre, post](x: &tensor[..pre, seq, ..post, bool]) -> tensor[..pre, ..post, i64] = count(&x, missing)
"#,
        "no named `missing` axis",
    );
    assert_rejects(
        r#"
def bad[pre, post](x: &tensor[..pre, seq, seq, ..post, bool]) -> tensor[..pre, ..post, i64] = count(&x, seq)
"#,
        "ambiguous",
    );
    assert_rejects(
        r#"
def bad[pre, post](x: &tensor[..pre, seq, ..post, bool]) -> tensor[..pre, ..post, i64] = count(&x, seq, 0)
"#,
        "positional and named axes cannot be mixed",
    );
    assert_rejects(
        r#"
def bad(x: &tensor[row, col, bool]) -> tensor[row, i64] = count(&x, col)
"#,
        "concrete-rank operand requires",
    );
}
