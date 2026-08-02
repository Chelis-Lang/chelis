//! Issue #233 host-runtime storage parity for `argmax_reduce` /
//! `argmin_reduce`.
//!
//! Issue #230 widened the type-system result of these ops to
//! `tensor[..., int64]` (see
//! `crates/chelis-types/src/infer.rs::check_reduction_signature`), but
//! the host-runtime evaluator continued to tag the produced
//! `RuntimeTensorValue` with the input tensor's precision (per the
//! Phase 3j-pre Batch 1 caveat documented on `RiscOp::Argmax`). Any
//! Surf program that fed an `argmax_reduce` result into a primitive
//! that branches on `RuntimeTensorValue::precision` — `eq`, `to_list`,
//! `tensor_to_scalar` — saw the precision-mismatch error
//! "tensor comparison expects matching tensor precision" or got a
//! float scalar where an int64 scalar was contractually expected. This
//! file pins the post-fix invariant that the storage label matches the
//! type-system label.
//!
//! Owner: chelis#233. Acceptance-oracle pattern mirrors
//! `crates/chelis-types/tests/issue_230_argmax_reduce_output_dtype.rs`
//! (positive on each integer-input dtype, negative on a non-int64
//! comparator, plus a sibling regression guard implicit via the
//! existing `issue_230_argmax_argmin_runtime.rs` tests).

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

fn eval_surf_expect_err(source: &str) -> String {
    let outcome = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    });
    match outcome {
        Err(err) => format!("{err:?}"),
        Ok(other) => panic!("expected eval to fail, got {other:?}"),
    }
}

fn root<'a>(result: &'a chelis_compiler_api::schema::EvalResult, name: &str) -> &'a ExecutionValue {
    let root = result
        .roots
        .iter()
        .find(|r| r.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("missing root {name} in {:?}", result.roots));
    &root.value
}

fn root_tensor<'a>(
    result: &'a chelis_compiler_api::schema::EvalResult,
    name: &str,
) -> &'a chelis_compiler_api::schema::TensorValue {
    match root(result, name) {
        ExecutionValue::Tensor { value } => value,
        other => panic!("expected tensor for {name}, got {other:?}"),
    }
}

// ---------------------------------------------------------------------
// Positive: argmax / argmin output compares cleanly against the
// canonical int64 to_tensor literal.
// ---------------------------------------------------------------------

/// EXPECT: comparing `argmax_reduce(x, 1)` (typed `tensor[..., int64]`)
/// against `to_tensor([cast(1, int64), cast(2, int64)])` (storage
/// precision int64) succeeds and produces an all-true mask.
///
/// Before the fix this errored with
/// "tensor comparison expects matching tensor precision" because the
/// argmax storage was still tagged f32 (= input precision).
#[test]
fn issue233_argmax_reduce_eq_int64_literal_comparator_succeeds() {
    let src = r#"
make = pad_sequences([[1.0, 4.0, 2.0], [3.0, 0.5, 5.0]], 0.0)
preds = argmax_reduce(&make, 1)
refs = to_tensor([cast(1, int64), cast(2, int64)])
out = eq(preds, refs)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![2], "eq output shape");
    assert_eq!(
        out.data.to_f64_lossy_vec(),
        vec![1.0, 1.0],
        "argmax_reduce(x, 1) must equal [1, 2] element-wise"
    );
}

/// EXPECT: same shape, but for argmin_reduce against the row-wise
/// minimum indices.
#[test]
fn issue233_argmin_reduce_eq_int64_literal_comparator_succeeds() {
    let src = r#"
make = pad_sequences([[1.0, 4.0, 2.0], [3.0, 0.5, 5.0]], 0.0)
preds = argmin_reduce(&make, 1)
refs = to_tensor([cast(0, int64), cast(1, int64)])
out = eq(preds, refs)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![2], "eq output shape");
    assert_eq!(
        out.data.to_f64_lossy_vec(),
        vec![1.0, 1.0],
        "argmin_reduce(x, 1) must equal [0, 1] element-wise"
    );
}

/// EXPECT: `argmax_reduce(x, 0)` along axis-0 produces an int64 tensor
/// whose `eq` against `to_tensor([cast(1, int64), cast(0, int64),
/// cast(1, int64)])` succeeds.
#[test]
fn issue233_argmax_reduce_axis0_eq_int64_literal_comparator_succeeds() {
    let src = r#"
make = pad_sequences([[1.0, 4.0, 2.0], [3.0, 0.5, 5.0]], 0.0)
preds = argmax_reduce(&make, 0)
refs = to_tensor([cast(1, int64), cast(0, int64), cast(1, int64)])
out = eq(preds, refs)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![3], "eq output shape");
    assert_eq!(out.data.to_f64_lossy_vec(), vec![1.0, 1.0, 1.0]);
}

/// EXPECT: argmax over a cast-widened f64 input still produces an
/// int64 storage tensor — the storage widening is independent of the
/// input precision. Mirrors the type-system test
/// `issue230_argmax_reduce_int64_input_yields_int64`.
#[test]
fn issue233_argmax_reduce_f64_input_storage_is_int64() {
    // `cast(_, f64)` widens the f32 input to f64; the argmax storage
    // tag must still be int64 regardless of the input's float width.
    let src = r#"
make = pad_sequences([[1.0, 4.0, 2.0], [3.0, 0.5, 5.0]], 0.0)
casted = cast(make, f64)
preds = argmax_reduce(&casted, 1)
refs = to_tensor([cast(1, int64), cast(2, int64)])
out = eq(preds, refs)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![2]);
    assert_eq!(out.data.to_f64_lossy_vec(), vec![1.0, 1.0]);
}

// ---------------------------------------------------------------------
// Positive: `to_list(argmax_reduce(...))` returns int64 scalars.
// ---------------------------------------------------------------------

/// EXPECT: `to_list` of an `argmax_reduce` result yields int64 scalars
/// (ExecutionValue::Int64), not Float64. The Phase 3j-pre caveat said
/// "we store integer-valued floats"; with the storage widening, the
/// per-element schema dtype now matches the type-system label.
#[test]
fn issue233_argmax_reduce_to_list_returns_int64_scalars() {
    let src = r#"
make = pad_sequences([[1.0, 4.0, 2.0], [3.0, 0.5, 5.0]], 0.0)
preds = argmax_reduce(&make, 1)
out = to_list(preds)
"#;
    let result = eval_surf(src);
    let ExecutionValue::List { value: items } = root(&result, "out") else {
        panic!("expected list for `out`");
    };
    assert_eq!(items.len(), 2, "to_list length");
    for (i, item) in items.iter().enumerate() {
        match item {
            ExecutionValue::Int64 { value: _ } => {}
            other => panic!("to_list element {i}: expected ExecutionValue::Int64, got {other:?}"),
        }
    }
    // Pin the concrete values too.
    let values: Vec<i64> = items
        .iter()
        .map(|item| match item {
            ExecutionValue::Int64 { value } => *value,
            other => panic!("unexpected element kind: {other:?}"),
        })
        .collect();
    assert_eq!(values, vec![1, 2]);
}

/// EXPECT: `to_list` of an `argmin_reduce` result yields int64 scalars.
#[test]
fn issue233_argmin_reduce_to_list_returns_int64_scalars() {
    let src = r#"
make = pad_sequences([[1.0, 4.0, 2.0], [3.0, 0.5, 5.0]], 0.0)
preds = argmin_reduce(&make, 1)
out = to_list(preds)
"#;
    let result = eval_surf(src);
    let ExecutionValue::List { value: items } = root(&result, "out") else {
        panic!("expected list for `out`");
    };
    let values: Vec<i64> = items
        .iter()
        .map(|item| match item {
            ExecutionValue::Int64 { value } => *value,
            other => panic!("unexpected element kind: {other:?}"),
        })
        .collect();
    assert_eq!(values, vec![0, 1]);
}

// ---------------------------------------------------------------------
// Positive: school PR #52 metrics-style pattern.
// ---------------------------------------------------------------------

/// EXPECT: the school `metrics.ch::accuracy` deferred-case shape works
/// end-to-end. `argmax_reduce` typed `tensor[..., int64]`, compared
/// elementwise to a labels tensor also typed `tensor[..., int64]`, and
/// the boolean mask reduces to integer hits via cast.
#[test]
fn issue233_school_accuracy_pattern_works() {
    // logits[0] = [0.1, 0.7, 0.2]  -> argmax = 1
    // logits[1] = [0.8, 0.1, 0.1]  -> argmax = 0
    // labels    = [1, 0]           -> matches both -> accuracy = 1.0
    let src = r#"
logits = pad_sequences([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]], 0.0)
labels = to_tensor([cast(1, int64), cast(0, int64)])
preds = argmax_reduce(&logits, 1)
hits_mask = eq(preds, labels)
hits = cast(hits_mask, int64)
all_correct = eq(hits, to_tensor([cast(1, int64), cast(1, int64)]))
"#;
    let result = eval_surf(src);
    let all_correct = root_tensor(&result, "all_correct");
    assert_eq!(all_correct.shape, vec![2]);
    assert_eq!(
        all_correct.data.to_f64_lossy_vec(),
        vec![1.0, 1.0],
        "school accuracy pattern: all predictions hit labels"
    );
}

// ---------------------------------------------------------------------
// Negative: precision-mismatch detection still fires for genuinely
// dtype-mismatched comparators.
// ---------------------------------------------------------------------

/// EXPECT: comparing `argmax_reduce(...)` (int64) against a
/// genuinely non-int64 tensor comparator still rejects with a
/// precision-mismatch error somewhere in the pipeline (either the
/// type-checker, since #230 widened the result to `int64`, or the
/// runtime, in any path that bypassed the checker). The fix widens
/// the argmax storage to int64 but must not weaken precision-mismatch
/// detection for genuinely dtype-distinct comparators.
#[test]
fn issue233_argmax_reduce_eq_non_int64_comparator_rejects() {
    let src = r#"
make = pad_sequences([[1.0, 4.0, 2.0], [3.0, 0.5, 5.0]], 0.0)
preds = argmax_reduce(&make, 1)
refs = pad_sequences([[1.0, 2.0]], 0.0)
flat_refs = reshape(refs, [2])
out = eq(preds, flat_refs)
"#;
    let err = eval_surf_expect_err(src);
    assert!(
        err.to_lowercase().contains("precision") || err.to_lowercase().contains("mismatch"),
        "expected precision-mismatch error, got {err}"
    );
    // Pin that the diagnostic mentions int64 — the post-fix argmax
    // result type — so we know the mismatch was detected against the
    // widened storage, not some incidental rank/shape error.
    assert!(
        err.contains("int64"),
        "expected diagnostic to mention int64 (the argmax_reduce result dtype), got {err}"
    );
}

/// EXPECT: comparing `argmin_reduce(...)` (int64) against a
/// genuinely non-int64 tensor comparator rejects with a
/// precision-mismatch error.
#[test]
fn issue233_argmin_reduce_eq_non_int64_comparator_rejects() {
    let src = r#"
make = pad_sequences([[1.0, 4.0, 2.0], [3.0, 0.5, 5.0]], 0.0)
preds = argmin_reduce(&make, 1)
refs = pad_sequences([[0.0, 1.0]], 0.0)
flat_refs = reshape(refs, [2])
out = eq(preds, flat_refs)
"#;
    let err = eval_surf_expect_err(src);
    assert!(
        err.to_lowercase().contains("precision") || err.to_lowercase().contains("mismatch"),
        "expected precision-mismatch error, got {err}"
    );
    assert!(
        err.contains("int64"),
        "expected diagnostic to mention int64 (the argmin_reduce result dtype), got {err}"
    );
}
