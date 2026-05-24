//! Issue #185 host-runtime acceptance (Group C — Binary Tier 2).
//!
//! `BUILTIN_NAMES` accepts `max_elem` and `min_elem` but the host
//! runtime evaluator did not dispatch them. Each is an element-wise
//! binary op; the host runtime must agree with the IR evaluator's
//! `binary_map(.., f64::max)` / `binary_map(.., f64::min)` behavior.
//!
//! Spec source of truth: `spec/05-risc-primitives.md` §3.4.

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
fn issue185_max_elem_runs_and_matches_ir_eval() {
    // a = [1.0, 4.0, 2.0, 5.0]
    // b = [3.0, 1.0, 6.0, 0.5]
    // max_elem(a, b) -> [3.0, 4.0, 6.0, 5.0]
    let src = r#"
a = to_tensor([1.0, 4.0, 2.0, 5.0])
b = to_tensor([3.0, 1.0, 6.0, 0.5])
out = max_elem(&a, &b)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![4], "max_elem shape");
    assert_eq!(out.data, vec![3.0, 4.0, 6.0, 5.0], "max_elem data");
}

#[test]
fn issue185_min_elem_runs_and_matches_ir_eval() {
    // min_elem(a, b) -> [1.0, 1.0, 2.0, 0.5]
    let src = r#"
a = to_tensor([1.0, 4.0, 2.0, 5.0])
b = to_tensor([3.0, 1.0, 6.0, 0.5])
out = min_elem(&a, &b)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![4], "min_elem shape");
    assert_eq!(out.data, vec![1.0, 1.0, 2.0, 0.5], "min_elem data");
}

#[test]
fn issue185_max_elem_rejects_string_input() {
    let src = r#"out = max_elem("a", "b")"#;
    let outcome = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: src.to_string(),
        bindings: BTreeMap::new(),
    });
    assert!(
        outcome.is_err(),
        "max_elem on strings must fail, got {outcome:?}"
    );
}

#[test]
fn issue185_min_elem_rejects_string_input() {
    let src = r#"out = min_elem("a", "b")"#;
    let outcome = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: src.to_string(),
        bindings: BTreeMap::new(),
    });
    assert!(
        outcome.is_err(),
        "min_elem on strings must fail, got {outcome:?}"
    );
}
