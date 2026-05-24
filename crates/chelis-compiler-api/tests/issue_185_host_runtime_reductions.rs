//! Issue #185 host-runtime acceptance (Group B — Reductions).
//!
//! `BUILTIN_NAMES` accepts `max_reduce` but the host runtime evaluator
//! was missing dispatch for it (sibling `min_reduce`/`prod_reduce`/
//! `argmax_reduce`/`argmin_reduce`/`sum` were already wired). This
//! file pins host-runtime parity with the IR evaluator on a small
//! rank-2 input.
//!
//! Spec source of truth: `spec/05-risc-primitives.md` §2.2.

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
fn issue185_max_reduce_axis0_runs_and_matches_ir_eval() {
    // x = [[1.0, 4.0, 2.0],
    //      [3.0, 0.5, 5.0]]
    // max_reduce(x, 0) reduces axis 0 -> shape [3], data [3.0, 4.0, 5.0]
    let src = r#"
make = pad_sequences([[1.0, 4.0, 2.0], [3.0, 0.5, 5.0]], 0.0)
out = max_reduce(&make, 0)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![3], "max_reduce axis-0 shape");
    assert_eq!(out.data, vec![3.0, 4.0, 5.0], "max_reduce axis-0 data");
}

#[test]
fn issue185_max_reduce_axis1_runs_and_matches_ir_eval() {
    // max_reduce(x, 1) reduces axis 1 -> shape [2], data [4.0, 5.0]
    let src = r#"
make = pad_sequences([[1.0, 4.0, 2.0], [3.0, 0.5, 5.0]], 0.0)
out = max_reduce(&make, 1)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![2], "max_reduce axis-1 shape");
    assert_eq!(out.data, vec![4.0, 5.0], "max_reduce axis-1 data");
}

#[test]
fn issue185_max_reduce_rejects_string_input() {
    let src = r#"out = max_reduce("not a tensor", 0)"#;
    let outcome = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: src.to_string(),
        bindings: BTreeMap::new(),
    });
    assert!(
        outcome.is_err(),
        "max_reduce on string must fail, got {outcome:?}"
    );
}
