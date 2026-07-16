//! Issue #230 host-runtime parity for `argmax_reduce` / `argmin_reduce`.
//!
//! `check_reduction_signature` now produces `tensor[..., int64]` for
//! these ops (issue #230 fix). The host-runtime evaluator must
//! continue to dispatch them and produce the expected integer indices
//! along the reduced axis, in the same per-op shape collapsing the
//! IR uses. Internally the runtime stores integer-valued floats per
//! the Phase 3j-pre Batch 1 caveat documented on `RiscOp::Argmax`;
//! this test pins that the produced values are exactly the indices,
//! independent of the type-system label.

use std::collections::BTreeMap;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind};

fn eval_surf(source: &str) -> chelis_compiler_api::schema::EvalResult {
    eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    })
    .unwrap_or_else(|err| panic!("eval failed: {err:?}"))
}

fn root_tensor<'a>(
    result: &'a chelis_compiler_api::schema::EvalResult,
    name: &str,
) -> &'a chelis_compiler_api::schema::TensorValue {
    let root = result
        .roots
        .iter()
        .find(|r| r.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("missing root {name} in {:?}", result.roots));
    match &root.value {
        ExecutionValue::Tensor { value } => value,
        other => panic!("expected tensor for {name}, got {other:?}"),
    }
}

/// EXPECT: `argmax_reduce(x, 0)` over a rank-2 f32 tensor produces
/// the per-column index of the maximum along axis 0 — a rank-1
/// tensor of integer-valued indices.
///
/// `x = [[1.0, 4.0, 2.0],   axis 0 argmax over each column ->
///       [3.0, 0.5, 5.0]]   row 1 wins col 0, row 0 wins col 1,
///                          row 1 wins col 2  =>  [1, 0, 1].
#[test]
fn issue230_argmax_reduce_axis0_runs_and_produces_index_data() {
    let src = r"
make = pad_sequences([[1.0, 4.0, 2.0], [3.0, 0.5, 5.0]], 0.0)
out = argmax_reduce(&make, 0)
";
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![3], "argmax_reduce axis-0 shape");
    assert_eq!(
        out.data,
        vec![1.0, 0.0, 1.0],
        "argmax_reduce axis-0 indices (integer-valued)"
    );
    for v in &out.data {
        assert_eq!(
            *v,
            v.trunc(),
            "argmax_reduce must produce integer-valued indices, got {v}"
        );
    }
}

/// EXPECT: `argmax_reduce(x, 1)` over the same rank-2 f32 tensor
/// produces the per-row index of the maximum along axis 1 — a
/// rank-1 tensor of integer-valued indices.
///
/// `argmax_reduce(x, 1)` -> for row 0 max is 4.0 at col 1 (=> 1),
/// for row 1 max is 5.0 at col 2 (=> 2). So result = [1.0, 2.0].
#[test]
fn issue230_argmax_reduce_axis1_runs_and_produces_index_data() {
    let src = r"
make = pad_sequences([[1.0, 4.0, 2.0], [3.0, 0.5, 5.0]], 0.0)
out = argmax_reduce(&make, 1)
";
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![2], "argmax_reduce axis-1 shape");
    assert_eq!(
        out.data,
        vec![1.0, 2.0],
        "argmax_reduce axis-1 indices (integer-valued)"
    );
    for v in &out.data {
        assert_eq!(
            *v,
            v.trunc(),
            "argmax_reduce must produce integer-valued indices, got {v}"
        );
    }
}

/// EXPECT: `argmin_reduce(x, 0)` produces the per-column index of the
/// minimum along axis 0.
///
/// `argmin_reduce(x, 0)` -> col 0: row 0 (1.0 < 3.0) => 0; col 1:
/// row 1 (0.5 < 4.0) => 1; col 2: row 0 (2.0 < 5.0) => 0. So result
/// = [0.0, 1.0, 0.0].
#[test]
fn issue230_argmin_reduce_axis0_runs_and_produces_index_data() {
    let src = r"
make = pad_sequences([[1.0, 4.0, 2.0], [3.0, 0.5, 5.0]], 0.0)
out = argmin_reduce(&make, 0)
";
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![3], "argmin_reduce axis-0 shape");
    assert_eq!(
        out.data,
        vec![0.0, 1.0, 0.0],
        "argmin_reduce axis-0 indices (integer-valued)"
    );
    for v in &out.data {
        assert_eq!(
            *v,
            v.trunc(),
            "argmin_reduce must produce integer-valued indices, got {v}"
        );
    }
}

/// EXPECT: `argmin_reduce(x, 1)` produces the per-row index of the
/// minimum along axis 1.
///
/// `argmin_reduce(x, 1)` -> row 0: 1.0 at col 0 => 0; row 1: 0.5
/// at col 1 => 1. So result = [0.0, 1.0].
#[test]
fn issue230_argmin_reduce_axis1_runs_and_produces_index_data() {
    let src = r"
make = pad_sequences([[1.0, 4.0, 2.0], [3.0, 0.5, 5.0]], 0.0)
out = argmin_reduce(&make, 1)
";
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![2], "argmin_reduce axis-1 shape");
    assert_eq!(
        out.data,
        vec![0.0, 1.0],
        "argmin_reduce axis-1 indices (integer-valued)"
    );
    for v in &out.data {
        assert_eq!(
            *v,
            v.trunc(),
            "argmin_reduce must produce integer-valued indices, got {v}"
        );
    }
}
