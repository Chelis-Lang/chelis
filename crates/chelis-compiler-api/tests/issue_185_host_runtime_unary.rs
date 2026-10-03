//! Issue #185 host-runtime acceptance (Group A — Unary RISC primitives).
//!
//! `BUILTIN_NAMES` accepts every name in this group but the host runtime
//! evaluator (`crates/chelis-compiler-api/src/runtime/eval.rs::eval_builtin`)
//! was missing dispatch arms. Each test below exercises one builtin
//! end-to-end through the eval entry point and pins the exact output
//! against the IR evaluator's reference math.
//!
//! Spec source of truth: `spec/05-risc-primitives.md` §2.2.
// Tests only: Rust std functions on the clippy disallowed list compute
// reference or input values here; the list holds production code to
// chelis-crmath (chelis#2957).
#![allow(clippy::disallowed_methods)]

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

fn assert_close(actual: f64, expected: f64, tol: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tol,
        "{label}: expected {expected}, got {actual} (tol {tol})"
    );
}

// ---------------------------------------------------------------------------
// Positive tests: each unary RISC primitive must run end-to-end and match
// the IR evaluator's elementwise math. Tensors here use 5-element f32
// inputs so the f32-rounding shape is exercised.
// ---------------------------------------------------------------------------

#[test]
fn issue185_abs_runs_and_matches_ir_eval() {
    let src = r#"
make = to_tensor([cast(-3.0, f32), cast(-1.5, f32), cast(0.0, f32), cast(1.5, f32), cast(3.0, f32)])
out = abs(&make)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![5]);
    let expected = [3.0, 1.5, 0.0, 1.5, 3.0];
    for (i, &want) in expected.iter().enumerate() {
        assert_close(
            out.data.element_f64_lossy(i),
            want,
            1e-6,
            &format!("abs[{i}]"),
        );
    }
}

#[test]
fn issue185_cos_runs_and_matches_ir_eval() {
    let src = r#"
make = to_tensor([cast(0.0, f32), cast(1.0, f32), cast(2.0, f32)])
out = cos(&make)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![3]);
    // IR eval uses f64 (`f64::cos`), but the host runtime routes through
    // f32 to mirror the C backend's `cosf`. So we use the f32 reference
    // for byte-identical agreement with the host-runtime emit path.
    let expected = [
        (0.0_f32).cos() as f64,
        (1.0_f32).cos() as f64,
        (2.0_f32).cos() as f64,
    ];
    for (i, &want) in expected.iter().enumerate() {
        assert_close(
            out.data.element_f64_lossy(i),
            want,
            1e-6,
            &format!("cos[{i}]"),
        );
    }
}

#[test]
fn issue185_tan_runs_and_matches_ir_eval() {
    let src = r#"
make = to_tensor([cast(0.0, f32), cast(0.5, f32), cast(1.0, f32)])
out = tan(&make)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![3]);
    let expected = [
        (0.0_f32).tan() as f64,
        (0.5_f32).tan() as f64,
        (1.0_f32).tan() as f64,
    ];
    for (i, &want) in expected.iter().enumerate() {
        assert_close(
            out.data.element_f64_lossy(i),
            want,
            1e-5,
            &format!("tan[{i}]"),
        );
    }
}

#[test]
fn issue185_floor_runs_and_matches_ir_eval() {
    let src = r#"
make = to_tensor([cast(-1.5, f32), cast(-0.5, f32), cast(0.5, f32), cast(1.5, f32), cast(2.0, f32)])
out = floor(&make)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![5]);
    let expected = [-2.0, -1.0, 0.0, 1.0, 2.0];
    for (i, &want) in expected.iter().enumerate() {
        assert_close(
            out.data.element_f64_lossy(i),
            want,
            1e-6,
            &format!("floor[{i}]"),
        );
    }
}

#[test]
fn issue185_ceil_runs_and_matches_ir_eval() {
    let src = r#"
make = to_tensor([cast(-1.5, f32), cast(-0.5, f32), cast(0.5, f32), cast(1.5, f32), cast(2.0, f32)])
out = ceil(&make)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![5]);
    let expected = [-1.0, 0.0, 1.0, 2.0, 2.0];
    for (i, &want) in expected.iter().enumerate() {
        assert_close(
            out.data.element_f64_lossy(i),
            want,
            1e-6,
            &format!("ceil[{i}]"),
        );
    }
}

#[test]
fn issue185_atan_runs_and_matches_ir_eval() {
    let src = r#"
make = to_tensor([cast(-1.0, f32), cast(0.0, f32), cast(1.0, f32)])
out = atan(&make)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![3]);
    let expected = [
        (-1.0_f32).atan() as f64,
        (0.0_f32).atan() as f64,
        (1.0_f32).atan() as f64,
    ];
    for (i, &want) in expected.iter().enumerate() {
        assert_close(
            out.data.element_f64_lossy(i),
            want,
            1e-6,
            &format!("atan[{i}]"),
        );
    }
}

// ---------------------------------------------------------------------------
// Negative tests: each builtin must still reject ill-typed inputs at
// `chelis check`. Passing a string where a tensor is expected is the
// canonical wrong-dtype error.
// ---------------------------------------------------------------------------

fn check_rejects(src: &str, label: &str) {
    let outcome = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: src.to_string(),
        bindings: BTreeMap::new(),
    });
    assert!(
        outcome.is_err(),
        "{label}: expected check/eval to fail, got {outcome:?}"
    );
}

#[test]
fn issue185_abs_rejects_string_input() {
    check_rejects(r#"out = abs("not a tensor")"#, "abs string input");
}

#[test]
fn issue185_cos_rejects_string_input() {
    check_rejects(r#"out = cos("not a tensor")"#, "cos string input");
}

#[test]
fn issue185_tan_rejects_string_input() {
    check_rejects(r#"out = tan("not a tensor")"#, "tan string input");
}

#[test]
fn issue185_floor_rejects_string_input() {
    check_rejects(r#"out = floor("not a tensor")"#, "floor string input");
}

#[test]
fn issue185_ceil_rejects_string_input() {
    check_rejects(r#"out = ceil("not a tensor")"#, "ceil string input");
}

#[test]
fn issue185_atan_rejects_string_input() {
    check_rejects(r#"out = atan("not a tensor")"#, "atan string input");
}
