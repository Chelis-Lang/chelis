//! Issue #185 host-runtime acceptance (Group F — Tensor-bool ops).
//!
//! `and`, `or`, and `not` are dispatched in `eval_builtin` but ONLY for
//! scalar `bool` inputs. The type system already accepts tensor-bool
//! inputs (the typer's `logical_binop` / `logical_unop` signatures pin
//! the input precision to `bool` but allow any dim list), so the host
//! runtime must also dispatch on tensor-bool inputs without falling
//! into the scalar-only error path.
//!
//! Per the brief's pinned decision: add SEPARATE tensor arms. Do not
//! broadcast the scalar arms — tensor-bool semantics differ enough that
//! broadcasting hides bugs.
//!
//! Bool-tensor inputs are constructed via `to_tensor([true, false, ...])`,
//! which the type checker accepts as `tensor[..., bool]`.
//!
//! Spec source of truth: `spec/05-risc-primitives.md` §3.2.

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

#[test]
fn issue185_tensor_and_runs_and_matches_ir_eval() {
    // a = [true,  true,  false, false]
    // b = [true,  false, true,  false]
    // a and b = [true, false, false, false]
    let src = r#"
a = to_tensor([true, true, false, false])
b = to_tensor([true, false, true, false])
out = and(&a, &b)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![4], "tensor and shape");
    assert_eq!(
        out.data.to_f64_lossy_vec(),
        vec![1.0, 0.0, 0.0, 0.0],
        "tensor and data"
    );
}

#[test]
fn issue185_tensor_or_runs_and_matches_ir_eval() {
    // a or b = [true, true, true, false]
    let src = r#"
a = to_tensor([true, true, false, false])
b = to_tensor([true, false, true, false])
out = or(&a, &b)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![4], "tensor or shape");
    assert_eq!(
        out.data.to_f64_lossy_vec(),
        vec![1.0, 1.0, 1.0, 0.0],
        "tensor or data"
    );
}

#[test]
fn issue185_tensor_not_runs_and_matches_ir_eval() {
    // not(a) = [false, false, true, true]
    let src = r#"
a = to_tensor([true, true, false, false])
out = not(&a)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![4], "tensor not shape");
    assert_eq!(
        out.data.to_f64_lossy_vec(),
        vec![0.0, 0.0, 1.0, 1.0],
        "tensor not data"
    );
}

// ---------------------------------------------------------------------------
// Negative parity: tensor-and / tensor-or / tensor-not must still reject
// ill-typed inputs at `chelis check`. A non-bool tensor is the canonical
// wrong-dtype error per the typer's `LOGICAL_OPS` post-check.
// ---------------------------------------------------------------------------

#[test]
fn issue185_tensor_and_rejects_non_bool_tensor_input() {
    // f32 tensors must be rejected — the typer pins the input precision
    // to bool.
    let src = r#"
a = to_tensor([1.0, 2.0], f32)
b = to_tensor([3.0, 4.0], f32)
out = and(&a, &b)
"#;
    let outcome = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: src.to_string(),
        bindings: BTreeMap::new(),
    });
    assert!(
        outcome.is_err(),
        "tensor `and` on non-bool tensors must fail, got {outcome:?}"
    );
}

#[test]
fn issue185_tensor_or_rejects_non_bool_tensor_input() {
    let src = r#"
a = to_tensor([1.0, 2.0], f32)
b = to_tensor([3.0, 4.0], f32)
out = or(&a, &b)
"#;
    let outcome = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: src.to_string(),
        bindings: BTreeMap::new(),
    });
    assert!(
        outcome.is_err(),
        "tensor `or` on non-bool tensors must fail, got {outcome:?}"
    );
}

#[test]
fn issue185_tensor_not_rejects_non_bool_tensor_input() {
    let src = r#"
a = to_tensor([1.0, 2.0], f32)
out = not(&a)
"#;
    let outcome = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: src.to_string(),
        bindings: BTreeMap::new(),
    });
    assert!(
        outcome.is_err(),
        "tensor `not` on non-bool tensors must fail, got {outcome:?}"
    );
}
