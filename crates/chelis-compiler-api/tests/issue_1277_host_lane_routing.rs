//! chelis#1277 Slice B2h: which evaluator a user `def` application reaches.
//!
//! `eval_compiled` sends every `Lane::Tensor` manifest entry to
//! `chelis_ir::eval` and every other entry to the host interpreter in
//! `runtime/eval.rs`. The two controls here pin the lane each source form of
//! chelis#1376 lands in, with the manifest's own `lane` field as the witness,
//! so the routing change can be measured against them rather than inferred.
//!
//! `chelis eval --file` constructs its request with an empty bindings map
//! (`main.rs::try_eval_result_for_target`), and even an embedding that binds
//! `f`'s inputs cannot reach the DAG evaluator for it: the lowerer never gives
//! a `shape`-reading def a named root. Both facts are locked below so the
//! routing change is measured against them rather than inferred.

#[path = "../../../tests/support/wire_values.rs"]
mod wire_values;

use std::collections::BTreeMap;

use chelis_compiler_api::compiler::{eval, eval_selected};
use chelis_compiler_api::schema::{
    EvalRequest, EvalResult, ExecutionValue, SourceKind, TensorValue,
};
use chelis_types::types::Lane;

/// chelis#1376's callee: the declared result claims `m` on axis 1 while the
/// `expand` reads `n` for it.
const CALLEE_1376: &str = "def f(x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, m, f32] = \
                           insert(x, 1, shape(x, 0))\n";

/// The CLI form: a nullary `main` applying `f` to two literals.
const HOST_APPLIED_1376: &str = "def f(x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, m, f32] = \
                                 insert(x, 1, shape(x, 0))\n\
                                 def main() = f(to_tensor([1.0f32, 2.0f32]), \
                                 to_tensor([3.0f32, 4.0f32, 5.0f32]))\n";

fn f32_tensor(shape: &[usize], data: &[f32]) -> TensorValue {
    TensorValue {
        shape: shape
            .iter()
            .map(|value| i64::try_from(*value).unwrap())
            .collect(),
        data: wire_values::storage_f32(data.to_vec()),
    }
}

fn root_shape(result: &EvalResult, name: &str) -> Vec<i64> {
    let root = result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("missing root `{name}` in {:?}", result.roots));
    match &root.value {
        ExecutionValue::Tensor { value } => value.shape.clone(),
        other => panic!("root `{name}` is not a tensor: {other:?}"),
    }
}

fn lane_of(result: &EvalResult, name: &str) -> Lane {
    result
        .manifest
        .entries
        .iter()
        .find(|entry| entry.name == name)
        .unwrap_or_else(|| panic!("no manifest entry `{name}` in {:?}", result.manifest))
        .lane
}

/// Selecting a parameterized shape-reading entry still cannot supply these
/// bindings through the Host root (#1397). Preserve the actual host-call
/// argument error rather than replacing it with an unavailable-root wrapper.
/// This is a diagnostic disposition lock, not an execution receipt.
#[test]
fn bound_parameterized_entry_preserves_its_host_call_error() {
    let mut bindings = BTreeMap::new();
    bindings.insert("x".to_string(), f32_tensor(&[2], &[1.0, 2.0]));
    bindings.insert("y".to_string(), f32_tensor(&[3], &[3.0, 4.0, 5.0]));
    let err = eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: CALLEE_1376.to_string(),
            bindings,
        },
        &["f".to_string()],
    )
    .expect_err("a shape-reading def has no named root, so its selected call is a Host root");
    let messages = err
        .errors
        .iter()
        .map(|diagnostic| diagnostic.message.clone())
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(
        messages,
        "kernel `f` parameter `x` expects a tensor or scalar argument, got ()"
    );
}

/// The CLI form: `main` is a nullary fn root, which realizability assigns to
/// the host lane, so the application of `f` is interpreted and the declared
/// claim on `f`'s result is never compared against the value.
///
/// EVIDENTIARY STATUS on `801f92c02`: disposition lock for the lane and for
/// the shape. The `[2, 2]` value is chelis#1376's `silent_unguarded`
/// baseline; the routing alone does not move it (the kernel has no guard for
/// the claim until B2a's DAG-evaluator guard is underneath), so this
/// assertion holds on this head and flips only at the rebase onto B2a.
#[test]
fn host_applied_def_main_is_a_host_lane_root() {
    let result = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: HOST_APPLIED_1376.to_string(),
        bindings: BTreeMap::new(),
    })
    .unwrap_or_else(|err| panic!("eval failed: {err:?}"));

    assert_eq!(lane_of(&result, "main"), Lane::Host);
    assert_eq!(root_shape(&result, "main"), vec![2, 2]);
}
