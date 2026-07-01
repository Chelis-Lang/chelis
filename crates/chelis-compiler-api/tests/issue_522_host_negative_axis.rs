//! Issue #522 host-evaluator negative-axis parity (soundness).
//!
//! The host AST evaluator's `normalize_axis` rejected a negative axis for
//! nearly every axis-taking primitive (reductions, `gather`, `scatter`,
//! `cumsum`, `sort`, `split`, `diagonal`, `trace`) while `softmax`/`concat`
//! pre-normalized inline and accepted it. Because the checker
//! (`chelis_types::normalize_static_axis`) and the IR lowerer
//! (`chelis_ir::lower::normalize_axis`) both accept the from-the-end
//! convention, `sum(x, -1)` passed `chelis check`/`chelis build` but failed
//! `chelis eval` — a check↔eval soundness gap (same class as #364).
//!
//! Spec source of truth: `spec/05-risc-primitives.md` §"Axis arguments" —
//! a negative axis `a` denotes `rank + a` UNIFORMLY for every axis-taking
//! primitive.
//!
//! These are check↔eval *parity* oracles: each op is run through the host
//! runtime with a negative axis and with the equivalent positive axis on a
//! rank-2 operand, and the two evaluated roots must be byte-identical. The
//! positive control proves the negative form is not silently degrading to
//! axis 0. The out-of-range cases pin that an axis still out of range after
//! normalization is rejected loud, not silently wrapped.

use std::collections::BTreeMap;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, EvalResult, ExecutionValue, SourceKind};

fn eval_surf(source: &str) -> EvalResult {
    eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    })
    .unwrap_or_else(|err| panic!("eval failed for source:\n{source}\nerror: {err:?}"))
}

/// Serialize the named root's value to JSON so tensors AND tuples (e.g.
/// `sort`, which returns `(values, indices)`) can be compared structurally
/// without requiring `PartialEq` on `ExecutionValue`.
fn root_json(result: &EvalResult, name: &str) -> serde_json::Value {
    let root = result
        .roots
        .iter()
        .find(|r| r.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("missing root {name} in {:?}", result.roots));
    serde_json::to_value(&root.value).expect("root value serializes")
}

/// Assert the negative-axis program and the positive-axis program produce a
/// byte-identical `out` root through the host runtime evaluator.
fn assert_axis_parity(op: &str, neg_src: &str, pos_src: &str) {
    assert_axis_parity_root(op, neg_src, pos_src, "out");
}

/// Like [`assert_axis_parity`] but for an explicitly named root, so tuple
/// returns (e.g. `sort` -> `out.0`, `out.1`) can be compared component-wise.
fn assert_axis_parity_root(op: &str, neg_src: &str, pos_src: &str, root: &str) {
    let neg = eval_surf(neg_src);
    let pos = eval_surf(pos_src);
    assert_eq!(
        root_json(&neg, root),
        root_json(&pos, root),
        "{op}: negative-axis eval must equal positive-axis eval (check↔eval parity) for root {root}"
    );
}

#[test]
fn issue522_sum_negative_axis_matches_positive() {
    assert_axis_parity(
        "sum",
        "x = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\nout = sum(&x, -1)\n",
        "x = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\nout = sum(&x, 1)\n",
    );
}

#[test]
fn issue522_mean_negative_axis_matches_positive() {
    assert_axis_parity(
        "mean",
        "x = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\nout = mean(&x, -1)\n",
        "x = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\nout = mean(&x, 1)\n",
    );
}

#[test]
fn issue522_max_reduce_negative_axis_matches_positive() {
    assert_axis_parity(
        "max_reduce",
        "x = to_tensor([[1.0, 4.0, 2.0], [3.0, 0.5, 5.0]])\nout = max_reduce(&x, -1)\n",
        "x = to_tensor([[1.0, 4.0, 2.0], [3.0, 0.5, 5.0]])\nout = max_reduce(&x, 1)\n",
    );
}

#[test]
fn issue522_cumsum_negative_axis_matches_positive() {
    assert_axis_parity(
        "cumsum",
        "x = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\nout = cumsum(&x, -1)\n",
        "x = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\nout = cumsum(&x, 1)\n",
    );
}

#[test]
fn issue522_sort_negative_axis_matches_positive() {
    // sort returns a `(values, indices)` tuple, which eval splits into the
    // `out.0` (sorted values) and `out.1` (permutation indices) roots; both
    // must match the positive-axis form.
    let neg = "x = to_tensor([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])\nout = sort(&x, -1)\n";
    let pos = "x = to_tensor([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])\nout = sort(&x, 1)\n";
    assert_axis_parity_root("sort values", neg, pos, "out.0");
    assert_axis_parity_root("sort indices", neg, pos, "out.1");
}

#[test]
fn issue522_gather_negative_axis_matches_positive() {
    assert_axis_parity(
        "gather",
        "x = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\n\
         idx = to_tensor([cast(0, int64), cast(2, int64)])\n\
         out = gather(&x, &idx, -1)\n",
        "x = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\n\
         idx = to_tensor([cast(0, int64), cast(2, int64)])\n\
         out = gather(&x, &idx, 1)\n",
    );
}

#[test]
fn issue522_scatter_negative_axis_matches_positive() {
    assert_axis_parity(
        "scatter",
        "base = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\n\
         idx = to_tensor([cast(0, int64), cast(2, int64)])\n\
         upd = to_tensor([[10.0, 20.0], [30.0, 40.0]])\n\
         out = scatter(base, idx, upd, -1, \"replace\")\n",
        "base = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\n\
         idx = to_tensor([cast(0, int64), cast(2, int64)])\n\
         upd = to_tensor([[10.0, 20.0], [30.0, 40.0]])\n\
         out = scatter(base, idx, upd, 1, \"replace\")\n",
    );
}

// ---------------------------------------------------------------------
// Negative parity: an axis still out of range after the from-the-end
// normalization must reject loud at eval, not silently wrap. `rank` and
// `-rank-1` straddle both bounds of a rank-2 operand.
// ---------------------------------------------------------------------

fn eval_is_err(source: &str) -> bool {
    eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    })
    .is_err()
}

#[test]
fn issue522_sum_axis_equal_to_rank_rejects() {
    // rank-2 operand, axis == rank (2) is out of `0..2`.
    assert!(
        eval_is_err("x = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\nout = sum(&x, 2)\n"),
        "sum(x, 2) on a rank-2 operand must reject"
    );
}

#[test]
fn issue522_sum_axis_too_negative_rejects() {
    // -rank-1 == -3 normalizes to -1, still out of `0..2`.
    assert!(
        eval_is_err("x = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\nout = sum(&x, -3)\n"),
        "sum(x, -3) on a rank-2 operand must reject"
    );
}

/// Sanity: the `ExecutionValue::Tensor` variant is what these roots carry,
/// guarding the JSON-compare helper against a silent representation change.
#[test]
fn issue522_sum_root_is_a_tensor() {
    let result =
        eval_surf("x = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\nout = sum(&x, -1)\n");
    let root = result
        .roots
        .iter()
        .find(|r| r.name.as_deref() == Some("out"))
        .expect("out root present");
    assert!(
        matches!(root.value, ExecutionValue::Tensor { .. }),
        "sum root must be a tensor, got {:?}",
        root.value
    );
}
