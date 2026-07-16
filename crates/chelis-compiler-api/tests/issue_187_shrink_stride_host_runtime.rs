//! Issue #187 host-runtime acceptance: `shrink` and `stride` must be
//! invokable from Surf source AND produce host-runtime output that matches
//! the IR evaluator. Per the project's evaluator-vs-backend agreement gate
//! (the IR evaluator at `crates/chelis-ir/src/eval.rs` is canonical), the
//! host runtime must delegate to the same window arithmetic; this file
//! pins that delegation behavior in terms of exact output values rather
//! than re-checking the IR-level math.
//!
//! Spec source of truth: `spec/05-risc-primitives.md` §2.4.

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

// ---------------------------------------------------------------------------
// Positive parity: parameterized forms run end-to-end and produce exact
// IR-evaluator-equivalent output.
// ---------------------------------------------------------------------------

/// EXPECT: `shrink(&x, [[0, 1], [1, 3]])` on a rank-2 input
///   x = [[0, 1, 2, 3],
///        [4, 5, 6, 7]]
/// returns the axis-0 row 0 sub-block at columns 1..3, i.e. shape [1, 2]
/// data [1.0, 2.0]. Before the fix, no callable form existed so this
/// either failed type-check (parameterized form) or hit
/// `unsupported builtin in host runtime` (bare form).
#[test]
fn issue187_shrink_parameterized_runs_and_matches_ir_eval() {
    let src = r"
make = pad_sequences([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], 0.0)
windowed = shrink(&make, [[0, 1], [1, 3]])
";
    let result = eval_surf(src);
    let win = root_tensor(&result, "windowed");
    assert_eq!(win.shape, vec![1, 2], "windowed shape");
    assert_eq!(win.data, vec![1.0, 2.0], "windowed data");
}

/// EXPECT: `stride(&x, 1, 2)` on the same 2x4 input returns shape [2, 2]
/// with every other element along axis 1:
///   row 0: [0.0, 2.0]
///   row 1: [4.0, 6.0]
#[test]
fn issue187_stride_parameterized_runs_and_matches_ir_eval() {
    let src = r"
make = pad_sequences([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], 0.0)
strided = stride(&make, 1, 2)
";
    let result = eval_surf(src);
    let s = root_tensor(&result, "strided");
    assert_eq!(s.shape, vec![2, 2], "strided shape");
    assert_eq!(s.data, vec![0.0, 2.0, 4.0, 6.0], "strided data");
}

// ---------------------------------------------------------------------------
// Negative parity at the host-runtime layer: malformed window parameters
// surface as errors rather than silent corruption.
// ---------------------------------------------------------------------------

/// EXPECT: a `shrink` with bounds whose axis-0 end exceeds the input
/// dimension fails — either at type-check or at host-runtime, but never
/// silently returning garbage.
#[test]
fn issue187_shrink_out_of_range_bounds_fails_loud() {
    let src = r"
make = pad_sequences([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], 0.0)
bad = shrink(&make, [[0, 5], [1, 3]])
";
    let outcome = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: src.to_string(),
        bindings: BTreeMap::new(),
    });
    assert!(
        outcome.is_err(),
        "out-of-range shrink bounds must surface as an error, got {outcome:?}"
    );
}

/// EXPECT: a `stride` with a zero step fails (either at type-check or at
/// host-runtime), matching the spec-level invariant locked by
/// `c10_stride_zero_step_is_error` on the `RiscOp` side.
#[test]
fn issue187_stride_zero_step_fails_loud() {
    let src = r"
make = pad_sequences([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], 0.0)
bad = stride(&make, 0, 2)
";
    let outcome = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: src.to_string(),
        bindings: BTreeMap::new(),
    });
    assert!(
        outcome.is_err(),
        "zero stride must surface as an error, got {outcome:?}"
    );
}
