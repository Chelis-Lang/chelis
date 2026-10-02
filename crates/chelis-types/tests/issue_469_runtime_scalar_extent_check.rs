//! chelis#469: the checker admits every `i64` `expand`/`insert` size.
//!
//! `spec/04-type-system.md` section 4.7.2: "`expand(x, axis, size)` and
//! `insert(x, axis, size)` each accept any `i64` `size`", and "A function
//! parameter, top-level or local binding, user-function result, cast, or
//! checked integer-arithmetic expression is equally admissible; no stage may
//! reject an extent because of its provenance or default it to one."
//!
//! The checker used to classify a size by provenance and reject one with no
//! tensor shape source ("no tensor in scope carries it"). Every positive row
//! below is a spelling that classification rejected. The negative rows are the
//! rejections section 4.7.2 does require, which must survive the gate's
//! removal: a static negative size, a size that is not `i64`, an `expand`
//! whose literal operand extent is not 1, an axis out of range, and a literal
//! result claim the size statically refutes.
//!
//! The executed agreement of these programs on `chelis eval` and compiled C is
//! `crates/chelis-cli/tests/issue_469_runtime_scalar_extent.rs`.

use chelis_deep::Expr;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::{InferResult, check_ir_program};

fn surf_to_deep(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls).expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

fn messages(rep: &InferResult) -> Vec<String> {
    rep.errors.iter().map(|e| e.message.clone()).collect()
}

fn assert_checks_clean(source: &str, label: &str) {
    let rep = check_ir_program(&surf_to_deep(source));
    assert!(
        rep.is_ok(),
        "{label}: a runtime i64 size is admissible under section 4.7.2; got {:?}",
        rep.err().map(|r| messages(&r))
    );
}

fn assert_rejects_with(source: &str, label: &str, needle: &str) {
    let rep = check_ir_program(&surf_to_deep(source))
        .err()
        .unwrap_or_else(|| panic!("{label}: must reject at check"));
    let msgs = messages(&rep);
    assert!(
        msgs.iter().any(|m| m.contains(needle)),
        "{label}: expected a diagnostic containing `{needle}`, got {msgs:?}"
    );
    assert!(
        !msgs
            .iter()
            .any(|m| m.contains("no tensor in scope carries it")),
        "{label}: the provenance diagnostic is gone; got {msgs:?}"
    );
}

/// The operand and size spellings, once per operation. `expand` broadcasts an
/// existing unit axis, so its operand is `[1, 2]`; `insert` adds one, so its
/// operand is `[2]`. Both produce `[size, 2]`.
const OPERATIONS: [(&str, &str); 2] = [
    ("expand", "to_tensor([[1i64, 2i64]])"),
    ("insert", "to_tensor([1i64, 2i64])"),
];

fn spellings(op: &str, operand: &str) -> Vec<(&'static str, String)> {
    vec![
        (
            "bare i64 parameter",
            format!("def f(k: i64) = {op}({operand}, 0, k)\n"),
        ),
        (
            "cast of an i32 parameter",
            format!("def f(j: i32) = {op}({operand}, 0, cast(j, i64))\n"),
        ),
        (
            "let-bound parameter",
            format!("def f(k: i64) = {{\n  m = k\n  {op}({operand}, 0, m)\n}}\n"),
        ),
        (
            "checked arithmetic over a parameter",
            format!("def f(k: i64) = {op}({operand}, 0, add(k, 1i64))\n"),
        ),
        (
            "user-function result",
            format!(
                "def g(k: i64) -> i64 = add(k, 1i64)\n\
                 def f(k: i64) = {op}({operand}, 0, g(k))\n"
            ),
        ),
        (
            "top-level runtime binding",
            format!(
                "def g(k: i64) -> i64 = add(k, 1i64)\n\
                 h = g(2i64)\n\
                 out = {op}({operand}, 0, h)\n"
            ),
        ),
        (
            "tuple projection",
            format!("def f(t: (i64, i64)) = {op}({operand}, 0, t.0)\n"),
        ),
        (
            "inline if",
            format!("def f(c: bool) = {op}({operand}, 0, if c then 3i64 else 4i64)\n"),
        ),
        (
            "pipe position",
            format!("def f(j: i32) = {{\n  m = j |> cast(i64)\n  {operand} |> {op}(0i32, m)\n}}\n"),
        ),
        (
            "declared literal result over a parameter",
            format!("def f(k: i64) -> tensor[3, 2, i64] = {op}({operand}, 0, k)\n"),
        ),
    ]
}

#[test]
fn every_runtime_size_spelling_checks_clean() {
    for (op, operand) in OPERATIONS {
        for (label, source) in spellings(op, operand) {
            assert_checks_clean(&source, &format!("{op}: {label}"));
        }
    }
}

/// The Std.Datetime shape that motivated the work: a table whose only extent
/// source is a scalar horizon, then scanned and reduced.
#[test]
fn a_horizon_sized_table_checks_clean() {
    assert_checks_clean(
        "def table(horizon: i64) = cumsum(expand(to_tensor([1i64]), 0, horizon), 0)\n\
         def total(horizon: i64) -> i64 = tensor_to_scalar(sum(table(horizon), 0))\n",
        "horizon table",
    );
}

#[test]
fn a_static_negative_size_is_still_a_type_error() {
    for (op, operand) in OPERATIONS {
        assert_rejects_with(
            &format!("out = {op}({operand}, 0, sub(1i64, 2i64))\n"),
            &format!("{op}: static negative size"),
            "requires non-negative size, got -1",
        );
    }
}

#[test]
fn a_size_that_is_not_i64_is_still_a_type_error() {
    for (op, operand) in OPERATIONS {
        assert_rejects_with(
            &format!("def f(j: i32) = {op}({operand}, 0, j)\n"),
            &format!("{op}: i32 size"),
            "expects an i64 size",
        );
    }
}

#[test]
fn expand_over_a_literal_non_unit_extent_is_still_a_type_error() {
    let rep = check_ir_program(&surf_to_deep(
        "def f(k: i64) = expand(to_tensor([1i64, 2i64]), 0, k)\n",
    ))
    .expect_err("a literal operand extent of 2 cannot be broadcast");
    assert!(
        !messages(&rep)
            .iter()
            .any(|m| m.contains("no tensor in scope carries it")),
        "the rejection is the unit-extent rule, not provenance: {:?}",
        messages(&rep)
    );
}

#[test]
fn an_axis_out_of_range_is_still_a_type_error() {
    assert_rejects_with(
        "def f(k: i64) = insert(to_tensor([1i64, 2i64]), 2, k)\n",
        "insert axis 2 on rank 1",
        "out of bounds",
    );
    assert_rejects_with(
        "def f(k: i64) = expand(to_tensor([[1i64, 2i64]]), 2, k)\n",
        "expand axis 2 on rank 2",
        "out of bounds",
    );
}

#[test]
fn a_literal_claim_a_literal_size_refutes_is_still_a_type_error() {
    for (op, operand) in OPERATIONS {
        let rep = check_ir_program(&surf_to_deep(&format!(
            "def f() -> tensor[3, 2, i64] = {op}({operand}, 0, 4i64)\n"
        )))
        .expect_err("extent 4 cannot satisfy a declared extent 3");
        assert!(
            !messages(&rep).is_empty(),
            "{op}: the refutation carries a diagnostic"
        );
    }
}
