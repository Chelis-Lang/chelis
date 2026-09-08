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

use std::collections::BTreeMap;

use chelis_compiler_api::compiler::{eval, eval_selected};
use chelis_compiler_api::schema::{
    EvalRequest, EvalResult, ExecutionValue, SourceKind, TensorElements, TensorValue,
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
        shape: shape.to_vec(),
        data: TensorElements::F32(data.to_vec()),
    }
}

fn root_shape(result: &EvalResult, name: &str) -> Vec<usize> {
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

/// The API binding path does NOT reach the DAG evaluator for a `shape`-reading
/// def either. With `x` and `y` bound and `f` selected, `manifested_program_for_eval`
/// finds no named root for `f` (the lowerer's syntactic map classifies a body
/// that applies `shape` as host, `lower.rs:2749`), classifies the selected
/// callable `Lane::Host` with reason `selected-callable-result`, and the host
/// lane cannot serve bound inputs, so the evaluation fails as an unavailable
/// root.
///
/// EVIDENTIARY STATUS on `801f92c02`: disposition lock, measured. It records
/// that the only evaluator a claim-carrying user def can reach from
/// `chelis eval` is the host interpreter, which is why B2h routes the host
/// application of such a def through the kernel C emits for it.
#[test]
fn bound_parameterized_entry_does_not_reach_the_dag_evaluator() {
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
    assert!(
        messages.contains("unavailable root `f` on Host lane"),
        "{messages}"
    );
    assert!(messages.contains("selected-callable-result"), "{messages}");
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

/// `coral_prerequisites.rs`'s comparison roots (the `#[ignore]`d Coral
/// prerequisite asserts `check` only). Each root is a nullary def with a
/// declared tensor result, so the routing applies it through the kernel the C
/// lane emits. The six values are what the compiled C prints and what the host
/// interpreter printed before the routing; the DAG evaluator must print them
/// too.
///
/// RE-VEHICLED by chelis#1506, which resolved the question this comment used to
/// leave open ("chelis#1506 for whether the form should be admitted at all").
/// It is not: `[05-OP-36]` makes a scalar beside a tensor a type error, so the
/// six programs are spelled with the explicit `expand`, which is the migration
/// the rejection's diagnostic names. The SUBJECT is unchanged, and it was never
/// the scalar operand: it is that a host-lane root of this shape evaluates
/// through the kernel to the values the C lane prints. The six expected values
/// are unchanged, because the explicit spelling denotes the same computation.
///
/// EVIDENTIARY STATUS: regression test, watched failing on `3b0e5b1b2` with
/// `unavailable root `above` on Host lane: ... got [] vs [3]`. The re-vehicling
/// preserves that: the operand is now a rank-1 `expand` result rather than a
/// rank-0 `Const`, and the routing property under test is the same.
#[test]
fn scalar_operand_comparison_roots_evaluate_to_the_values_the_c_kernel_prints() {
    const CORAL_COMPARISONS: &str = "module Demo.Main\n\n\
        def above() -> tensor[3, bool] = \
        gt(to_tensor([1.0, 2.0, 3.0]), expand(to_tensor([1.5]), 0i32, 3i64))\n\
        def below() -> tensor[3, bool] = \
        gt(expand(to_tensor([1.5]), 0i32, 3i64), to_tensor([1.0, 2.0, 3.0]))\n\
        def lt_right() -> tensor[3, bool] = \
        lt(to_tensor([1.0, 2.0, 3.0]), expand(to_tensor([2.5]), 0i32, 3i64))\n\
        def lt_left() -> tensor[3, bool] = \
        lt(expand(to_tensor([2.5]), 0i32, 3i64), to_tensor([1.0, 2.0, 3.0]))\n\
        def eq_right() -> tensor[3, bool] = \
        eq(to_tensor([1.0, 2.0, 3.0]), expand(to_tensor([2.0]), 0i32, 3i64))\n\
        def eq_left() -> tensor[3, bool] = \
        eq(expand(to_tensor([2.0]), 0i32, 3i64), to_tensor([1.0, 2.0, 3.0]))\n";
    let result = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: CORAL_COMPARISONS.to_string(),
        bindings: BTreeMap::new(),
    })
    .unwrap_or_else(|err| panic!("eval failed: {err:?}"));
    let expected: [(&str, [bool; 3]); 6] = [
        ("above", [false, true, true]),
        ("below", [true, false, false]),
        ("lt_right", [true, true, false]),
        ("lt_left", [false, false, true]),
        ("eq_right", [false, true, false]),
        ("eq_left", [false, true, false]),
    ];
    for (name, values) in expected {
        let root = result
            .roots
            .iter()
            .find(|root| root.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("missing root `{name}` in {:?}", result.roots));
        match &root.value {
            ExecutionValue::Tensor { value } => {
                assert_eq!(value.shape, vec![3], "{name}");
                assert_eq!(value.data, TensorElements::Bool(values.to_vec()), "{name}");
            }
            other => panic!("root `{name}` is not a tensor: {other:?}"),
        }
    }
}
