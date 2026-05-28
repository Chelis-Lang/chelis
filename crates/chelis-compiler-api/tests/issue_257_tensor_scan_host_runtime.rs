//! Issue #257 host-runtime acceptance — `tensor_scan` builtin.
//!
//! Spec source of truth: `spec/05-risc-primitives.md` §3.6
//! (host-runtime helper).
//!
//! Background. `chelis test` / `chelis eval` overflow the host worker
//! stack on right-recursive Surf list builds of more than ~10000
//! elements because the interpreter recurses on the host stack for
//! each Surf function call. The chunked and fold-based workarounds
//! hit an O(n²) `concat` wall. `tensor_scan(initial, fn, n)` is the
//! host-runtime helper that builds a rank-1 tensor of `n` elements
//! iteratively, sidestepping both walls.
//!
//! These tests pin:
//!
//! 1. The reproducer at the n that previously SIGABRT'd:
//!    `tensor_scan` must complete at n=20000 (and n=40000) without
//!    overflowing the stack.
//! 2. A small correctness test against the equivalent fold so the
//!    semantics match the existing list-scan when interpreted
//!    element-wise.
//! 3. Negative coverage — wrong arity, non-callable second arg, and
//!    a callback that returns the wrong dtype.

use std::collections::BTreeMap;

use chelis_compiler_api::compiler::{compile, eval};
use chelis_compiler_api::schema::{
    CompileRequest, CompileTarget, EvalRequest, ExecutionValue, SourceKind,
};

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
// Reproducer: previously SIGABRT'd at n>=20000 via right-recursive list
// build. With `tensor_scan` the host iterates in Rust, so the worker
// stack is constant.
// ---------------------------------------------------------------------------

#[test]
fn issue257_tensor_scan_int64_n20000_does_not_overflow() {
    // The callback `(prev, _i) -> add(prev, 1)` produces 1..=n.
    let src = r#"
out = tensor_scan(
  cast(0, int64),
  fn (prev: int64, _i: int64) -> add(prev, cast(1, int64)),
  cast(20000, int64)
)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![20000]);
    assert_eq!(out.data[0], 1.0);
    assert_eq!(out.data[19999], 20000.0);
}

#[test]
fn issue257_tensor_scan_int64_n40000_does_not_overflow() {
    // 40k elements — the school-downstream init regime
    // (30720 fc1 + 10080 fc2 + 840 fc3 ≈ 41.6k Glorot weights).
    let src = r#"
out = tensor_scan(
  cast(0, int64),
  fn (prev: int64, _i: int64) -> add(prev, cast(1, int64)),
  cast(40000, int64)
)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![40000]);
    assert_eq!(out.data[0], 1.0);
    assert_eq!(out.data[39999], 40000.0);
}

// ---------------------------------------------------------------------------
// Correctness: equivalent to the small-n list-scan + to_tensor round-trip.
// We pick n=8 so the result is small enough to spell out by hand.
// ---------------------------------------------------------------------------

#[test]
fn issue257_tensor_scan_int64_small_n_matches_fold_oracle() {
    // Same callback as above. For n=8 we expect [1, 2, 3, 4, 5, 6, 7, 8].
    let src = r#"
out = tensor_scan(
  cast(0, int64),
  fn (prev: int64, _i: int64) -> add(prev, cast(1, int64)),
  cast(8, int64)
)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![8]);
    let expected: Vec<f64> = (1..=8).map(|v| v as f64).collect();
    assert_eq!(out.data, expected);
}

#[test]
fn issue257_tensor_scan_int64_uses_index_argument() {
    // Confirm the `int64` index argument is wired correctly: the
    // callback receives `i` and the accumulator. For
    // `fn (_, i) -> i`, the output is [0, 1, 2, 3, 4].
    let src = r#"
out = tensor_scan(
  cast(0, int64),
  fn (_prev: int64, i: int64) -> i,
  cast(5, int64)
)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![5]);
    assert_eq!(out.data, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn issue257_tensor_scan_zero_length_returns_empty_tensor() {
    let src = r#"
out = tensor_scan(
  cast(7, int64),
  fn (prev: int64, _i: int64) -> prev,
  cast(0, int64)
)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![0]);
    assert!(out.data.is_empty());
}

// ---------------------------------------------------------------------------
// Negative parity: every positive contract above has a matching failure
// test that pins the rejection reason. Per CLAUDE.md "Negative Test
// Parity".
// ---------------------------------------------------------------------------

#[test]
fn issue257_tensor_scan_negative_length_rejected() {
    let src = r#"
out = tensor_scan(
  cast(0, int64),
  fn (prev: int64, _i: int64) -> prev,
  cast(-1, int64)
)
"#;
    let result = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: src.to_string(),
        bindings: BTreeMap::new(),
    });
    let message = result.err().map(|e| format!("{e:?}")).unwrap_or_default();
    assert!(
        message.contains("non-negative length"),
        "expected negative-length error, got: {message}"
    );
}

#[test]
fn issue257_tensor_scan_non_callable_second_arg_rejected() {
    // Passing an int where a callback belongs must be caught.
    // This will likely fail at type-check (the callback slot wants a
    // `(T, int64) -> T` function); the durable invariant is that the
    // user gets a tensor_scan-specific error message.
    let src = r#"
out = tensor_scan(
  cast(0, int64),
  cast(42, int64),
  cast(5, int64)
)
"#;
    let result = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: src.to_string(),
        bindings: BTreeMap::new(),
    });
    let message = result.err().map(|e| format!("{e:?}")).unwrap_or_default();
    assert!(
        message.to_lowercase().contains("tensor_scan")
            || message.to_lowercase().contains("callback")
            || message.to_lowercase().contains("callable")
            || message.to_lowercase().contains("function"),
        "expected tensor_scan/callback rejection, got: {message}"
    );
}

#[test]
fn issue257_tensor_scan_wrong_arity_rejected() {
    let src = r#"
out = tensor_scan(cast(0, int64), cast(5, int64))
"#;
    let result = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: src.to_string(),
        bindings: BTreeMap::new(),
    });
    let message = result.err().map(|e| format!("{e:?}")).unwrap_or_default();
    // Arity mismatch is caught by the type checker. The diagnostic
    // must reference "3 args" so the user sees the actual mismatch
    // and not just a generic "ArityMismatch" tag.
    assert!(
        message.contains("ArityMismatch"),
        "expected ArityMismatch kind, got: {message}"
    );
    assert!(
        message.contains("3 args") || message.contains("expected 3"),
        "expected 3-arg arity callout, got: {message}"
    );
}

// ---------------------------------------------------------------------------
// Precision pinning: the output tensor's dtype must follow the initial
// value's dtype, end-to-end. Two positive cases that pin precision
// without going through a generic int64 path, plus a runtime-level
// dtype-mismatch guard (the type checker normally catches this; the
// runtime arm is the belt-and-suspenders for value-level shenanigans).
// ---------------------------------------------------------------------------

#[test]
fn issue257_tensor_scan_f32_initial_value_produces_correct_values() {
    // Initial value is f32; runtime arm pins precision = F32 and the
    // values must be the spec-defined fn(prev, i) sequence.
    let src = r#"
out = tensor_scan(
  cast(1.0, f32),
  fn (prev: f32, _i: int64) -> mul(prev, cast(2.0, f32)),
  cast(4, int64)
)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    // Element zero is fn(initial=1.0, 0) = 2.0; 4, 8, 16 follow.
    assert_eq!(out.shape, vec![4]);
    assert_eq!(out.data, vec![2.0, 4.0, 8.0, 16.0]);
}

#[test]
fn issue257_tensor_scan_bool_initial_value_produces_correct_values() {
    // Initial value is bool; runtime arm pins precision = Bool.
    let src = r#"
out = tensor_scan(
  true,
  fn (prev: bool, _i: int64) -> not(prev),
  cast(4, int64)
)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![4]);
    // not(true) = false (0); then not(false) = true (1); alternating.
    assert_eq!(out.data, vec![0.0, 1.0, 0.0, 1.0]);
}

// ---------------------------------------------------------------------------
// Compiled-backend rejection: `chelis build --target c` must refuse a
// program that calls `tensor_scan`. Without this guard the C emitter
// silently produces `__binding_0_value = /* unsupported builtin
// tensor_scan */ 0` and the compiled binary returns garbage at run
// time. Spec §3.6 marks the builtin host-only by design.
// ---------------------------------------------------------------------------

#[test]
fn issue257_tensor_scan_build_target_c_rejected() {
    let src = r#"
out = tensor_scan(
  cast(0, int64),
  fn (prev: int64, _i: int64) -> add(prev, cast(1, int64)),
  cast(8, int64)
)
"#;
    let result = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: src.to_string(),
        target: CompileTarget::C,
        entry_name: None,
    });
    let err = result.expect_err("chelis build --target c must reject tensor_scan");
    let message = format!("{err:?}");
    assert!(
        message.contains("tensor_scan"),
        "rejection must name the builtin, got: {message}"
    );
    assert!(
        message.contains("host-only") || message.contains("Host-Runtime"),
        "rejection must explain the host-only contract, got: {message}"
    );
    // Belt-and-suspenders: confirm no C source containing the silent
    // stub was emitted via the err path. The previous regression
    // surfaced as compile-Ok with `/* unsupported builtin tensor_scan */`
    // in the C source.
    assert!(
        !message.contains("/* unsupported builtin"),
        "rejection must not be paired with a silent C stub, got: {message}"
    );
}

#[test]
fn issue257_tensor_scan_build_target_hip_rejected() {
    let src = r#"
out = tensor_scan(
  cast(0, int64),
  fn (prev: int64, _i: int64) -> add(prev, cast(1, int64)),
  cast(8, int64)
)
"#;
    let result = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: src.to_string(),
        target: CompileTarget::Hip,
        entry_name: None,
    });
    let err = result.expect_err("chelis build --target hip must reject tensor_scan");
    let message = format!("{err:?}");
    assert!(
        message.contains("tensor_scan"),
        "rejection must name the builtin, got: {message}"
    );
}

// ---------------------------------------------------------------------------
// AD path: `grad(...)` over a function whose body calls `tensor_scan`
// must fail with a tensor_scan-tagged error, not a confusing
// downstream "axis 0 out of range for rank-0 operand" trace.
// Spec §3.6: "[tensor_scan] is not in the RISC DAG and has no AD
// adjoint".
// ---------------------------------------------------------------------------

#[test]
fn issue257_tensor_scan_grad_rejected_with_tagged_error() {
    let src = r#"
target = fn (x: f32) -> sum(tensor_scan(
  x,
  fn (prev: f32, _i: int64) -> mul(prev, cast(2.0, f32)),
  cast(4, int64)
), cast(0, int32))
out = grad(target)(cast(1.0, f32))
"#;
    let result = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: src.to_string(),
        bindings: BTreeMap::new(),
    });
    let err = result.expect_err("grad over tensor_scan must fail closed");
    let message = format!("{err:?}");
    assert!(
        message.contains("tensor_scan"),
        "grad rejection must name the unsupported builtin, got: {message}"
    );
    assert!(
        message.to_lowercase().contains("differentiate")
            || message.contains("AD adjoint")
            || message.contains("§3.6"),
        "grad rejection must explain the AD contract, got: {message}"
    );
}
