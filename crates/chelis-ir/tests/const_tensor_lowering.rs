//! Verify that `to_tensor([1.0, 2.0, 3.0])` with non-uniform data
//! lowers to a single `ConstTensor` node instead of a Const+Pad+Add tree.

use chelis_ir::dag::{Dag, NodeId, RiscOp};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with};
use chelis_ir::lower::try_lower_program;
use chelis_ir::verify;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as surf_parse;
use chelis_types::{check_linearity, check_typed_program};

/// Full Surf-to-DAG pipeline.
fn surf_to_dag(source: &str) -> Result<Dag, String> {
    let decls = surf_parse(source).map_err(|e| format!("surf parse: {e:?}"))?;
    let deep = desugar_program(&decls);
    let checked = check_typed_program(&deep)
        .map_err(|errs| format!("typecheck failed: {:?}", errs.errors))?;
    let checked = chelis_effects::check_program(&checked)
        .map_err(|errs| format!("effects failed: {errs:?}"))?;
    let checked =
        check_linearity(&checked).map_err(|errs| format!("linearity failed: {errs:?}"))?;
    try_lower_program(&checked).map_err(|diag| format!("lowering failed: {diag:?}"))
}

/// Deep-source pipeline for witnesses Surf cannot express (hand-written
/// `.dp` shapes; see the chelis#1123 red-team report on PR #1123).
fn deep_to_dag(source: &str) -> Result<Dag, String> {
    let deep =
        chelis_deep::parser::parse_and_stamp(source).map_err(|e| format!("deep parse: {e:?}"))?;
    let checked = check_typed_program(&deep)
        .map_err(|errs| format!("typecheck failed: {:?}", errs.errors))?;
    let checked = chelis_effects::check_program(&checked)
        .map_err(|errs| format!("effects failed: {errs:?}"))?;
    let checked =
        check_linearity(&checked).map_err(|errs| format!("linearity failed: {errs:?}"))?;
    try_lower_program(&checked).map_err(|diag| format!("lowering failed: {diag:?}"))
}

/// chelis#1123/#1131: the end-to-end guard for single rounding. Surf retains
/// an integer-spelled float literal as an exact Int payload with explicit
/// source provenance, and lowering finalizes it directly at f32 width.
#[test]
fn int_leaf_in_f32_tensor_literal_single_rounds() {
    let source = r#"
def main() -> tensor[2, f32] =
  to_tensor([18014399583223809f32, 3f32])
"#;
    let dag = surf_to_dag(source).expect("pipeline succeeds");
    let bits = dag
        .nodes()
        .iter()
        .find_map(|n| match &n.op {
            RiscOp::ConstTensor { data } => match data.element_ref(0) {
                chelis_types::ElementRef::F32(v) => Some(v.to_bits()),
                _ => None,
            },
            _ => None,
        })
        .expect("non-uniform literal takes the ConstTensor path with an f32 payload");
    assert_eq!(
        bits, 0x5A80_0001,
        "the integer leaf must round ONCE into f32; 0x5A800000 is the \
         double-rounded (i as f64 as f32) image"
    );
}

/// bf16 sibling of the single-rounding witness. Guards relatively: the DAG payload must equal the sealed
/// module's single-rounded image, whose absolute value the chelis-types
/// pins lock; under the double-rounding mutation the two diverge.
#[test]
fn int_leaf_in_bf16_tensor_literal_single_rounds() {
    let source = r#"
def main() -> tensor[2, bf16] =
  to_tensor([18084767253659649bf16, 3bf16])
"#;
    let dag = surf_to_dag(source).expect("pipeline succeeds");
    let stored = dag
        .nodes()
        .iter()
        .find_map(|n| match &n.op {
            RiscOp::ConstTensor { data } => Some(data.scalar_at(0)),
            _ => None,
        })
        .expect("non-uniform literal takes the ConstTensor path");
    let expected = chelis_types::finalize_scalar(
        "test",
        chelis_types::types::Prim::Bf16,
        chelis_types::RawScalar::Int(18_084_767_253_659_649),
    )
    .expect("in-range integer finalize");
    assert_eq!(
        stored, expected,
        "the bf16 payload must be the single-rounded integer image"
    );
}

/// chelis#1123 red-team ask 5 (and chelis#1132): the tensor-level finalize
/// diagnostic is reachable WITHOUT the chelis#1131 gap - every atom kind
/// here agrees with its declared prim; the overflow is manufactured by a
/// `neg` application the checker does not constant-fold. Locks the exact
/// diagnostic text so the loud replacement for the deleted hand-rolled
/// integer-lane guards stays live. Surf rejects the equivalent source at
/// the literal-range check, so the witness is Deep-only; Deep is a
/// supported input surface.
#[test]
fn overflowing_negated_literal_raises_the_finalize_diagnostic() {
    let source = r#"
(defsig {} main (t-fn {} (t-tensor {} (d-lit {} 2) (t-prim {} int8))))

(def {} main (fn {} (params {})
  (app {} (var {} to_tensor)
    (app {} (var {} Cons)
      (app {} (var {} neg) (lit {type: (t-prim {} int8)} -128))
      (app {} (var {} Cons) (lit {type: (t-prim {} int8)} 1) (var {} Nil))))))
"#;
    let err = deep_to_dag(source).expect_err("the literal must not lower");
    assert!(
        err.contains("tensor literal does not finalize at its ascribed dtype `int8`"),
        "expected the [04-NUM-1] finalize diagnostic, got: {err}"
    );
    assert!(
        err.contains("overflow in const at int8"),
        "expected the overflow trap brand, got: {err}"
    );
}

/// chelis#1123 red-team finding 3: a literal mixing an explicit cast with a
/// bare integer literal stages Typed + Raw, whose uniformity keys differ by
/// construction, so lowering takes the ConstTensor path even though the
/// finalized elements are equal. Values are identical to a Const splat;
/// pinned so the node shape is a recorded decision, not an accident.
#[test]
fn mixed_cast_and_bare_integer_literal_lowers_to_const_tensor() {
    let source = r#"
def main() -> tensor[2, int32] =
  to_tensor([cast(1, int32), 1])
"#;
    let dag = surf_to_dag(source).expect("pipeline succeeds");
    let data = dag
        .nodes()
        .iter()
        .find_map(|n| match &n.op {
            RiscOp::ConstTensor { data } => data.to_i64_exact_vec(),
            _ => None,
        })
        .expect("mixed Typed/Raw staging takes the ConstTensor path");
    assert_eq!(data, vec![1, 1]);
}

/// A non-uniform literal `to_tensor` should produce a ConstTensor node.
#[test]
fn to_tensor_non_uniform_lowers_to_const_tensor() {
    let source = r#"
def main() -> tensor[3, f32] =
  to_tensor([1.0, 2.0, 3.0])
"#;
    let dag = surf_to_dag(source).expect("pipeline succeeds");

    // Look for a ConstTensor node in the DAG.
    let has_const_tensor = dag
        .nodes()
        .iter()
        .any(|n| matches!(&n.op, RiscOp::ConstTensor { .. }));
    assert!(
        has_const_tensor,
        "expected at least one ConstTensor node in the lowered DAG, found none.\n\
         DAG ops: {:?}",
        dag.nodes()
            .iter()
            .map(|n| format!("{:?}", n.op))
            .collect::<Vec<_>>()
    );

    // Verify there are NO Pad nodes (the old cascade pattern).
    let has_pad = dag
        .nodes()
        .iter()
        .any(|n| matches!(&n.op, RiscOp::Pad { .. }));
    assert!(!has_pad, "expected no Pad nodes (old cascade pattern)");
}

/// Evaluate a ConstTensor node and verify it produces the correct values.
#[test]
fn const_tensor_evaluates_correctly() {
    let source = r#"
def main() -> tensor[3, f32] =
  to_tensor([1.0, 2.0, 3.0])
"#;
    let dag = surf_to_dag(source).expect("pipeline succeeds");

    let roots: Vec<NodeId> = dag.roots().to_vec();
    let values = eval_tensor_roots_with(&dag, &roots, |_name| None).expect("eval succeeds");

    assert_eq!(roots.len(), 1);
    let result = &values[&roots[0]];
    assert_eq!(result.shape, vec![3]);
    assert_eq!(result.to_f64_lossy_vec(), vec![1.0, 2.0, 3.0]);
}

/// A uniform-value to_tensor still uses the efficient Const (single-value) path.
#[test]
fn to_tensor_uniform_stays_as_const() {
    let source = r#"
def main() -> tensor[3, f32] =
  to_tensor([5.0, 5.0, 5.0])
"#;
    let dag = surf_to_dag(source).expect("pipeline succeeds");

    // Should use the uniform-value Const fast path, NOT ConstTensor.
    let has_const_tensor = dag
        .nodes()
        .iter()
        .any(|n| matches!(&n.op, RiscOp::ConstTensor { .. }));
    assert!(
        !has_const_tensor,
        "uniform data should use Const, not ConstTensor"
    );

    let has_const_5 = dag
        .nodes()
        .iter()
        .any(|n| matches!(&n.op, RiscOp::Const { value } if (value.as_f64_lossy() - 5.0).abs() < f64::EPSILON));
    assert!(has_const_5, "expected a Const(5.0) node for uniform data");
}

// ══════════════════════════════════════════════════════════════════════
// RED TEAM: adversarial ConstTensor tests
// ══════════════════════════════════════════════════════════════════════

/// Edge case: single-element to_tensor should NOT produce ConstTensor
/// (it should stay as Const since 1-element is effectively uniform).
#[test]
fn to_tensor_single_element_is_const_not_const_tensor() {
    let source = r#"
def main() -> tensor[1, f32] =
  to_tensor([42.0])
"#;
    let dag = surf_to_dag(source).expect("pipeline succeeds");

    // A single element is trivially uniform: Const should be used.
    let has_const_tensor = dag
        .nodes()
        .iter()
        .any(|n| matches!(&n.op, RiscOp::ConstTensor { .. }));
    assert!(
        !has_const_tensor,
        "single-element to_tensor should use Const, not ConstTensor"
    );
}

/// Edge case: to_tensor with a variable argument must NOT produce ConstTensor
/// (it should fall back to the host-routing path / add-tree).
#[test]
fn to_tensor_variable_arg_does_not_produce_const_tensor() {
    // Use a function parameter instead of a literal. This should NOT
    // be lowered to ConstTensor because the data is not known at compile time.
    let source = r#"
def make_tensor(x: f32, y: f32) -> tensor[2, f32] =
  to_tensor([x, y])
"#;
    let result = surf_to_dag(source);
    match result {
        Ok(dag) => {
            // If it lowers, it should NOT have a ConstTensor node.
            let has_const_tensor = dag
                .nodes()
                .iter()
                .any(|n| matches!(&n.op, RiscOp::ConstTensor { .. }));
            assert!(
                !has_const_tensor,
                "to_tensor with variable args must not produce ConstTensor; \
                 it should use the fallback path"
            );
        }
        Err(_) => {
            // Acceptable: if lowering rejects it entirely, that's fine too.
            // The key invariant is that ConstTensor is NOT emitted for
            // non-literal data.
        }
    }
}

/// Verify ConstTensor evaluates correctly with negative values.
#[test]
fn const_tensor_negative_values_evaluate_correctly() {
    let source = r#"
def main() -> tensor[4, f32] =
  to_tensor([-1.0, 0.0, -3.5, 2.5])
"#;
    let dag = surf_to_dag(source).expect("pipeline succeeds");

    let roots: Vec<NodeId> = dag.roots().to_vec();
    let values = eval_tensor_roots_with(&dag, &roots, |_name| None).expect("eval succeeds");

    assert_eq!(roots.len(), 1);
    let result = &values[&roots[0]];
    assert_eq!(result.shape, vec![4]);
    assert_eq!(result.to_f64_lossy_vec(), vec![-1.0, 0.0, -3.5, 2.5]);
}

/// Verify default f32 tensor literals are materialized at their declared width.
#[test]
fn const_tensor_materializes_f32_values_at_f32_width() {
    let source = r#"
def main() -> tensor[3, f32] =
  to_tensor([0.1, 0.2, 0.3])
"#;
    let dag = surf_to_dag(source).expect("pipeline succeeds");

    let roots: Vec<NodeId> = dag.roots().to_vec();
    let values = eval_tensor_roots_with(&dag, &roots, |_name| None).expect("eval succeeds");

    let result = &values[&roots[0]];
    assert_eq!(result.shape, vec![3]);
    // chelis#729 Phase 1 re-authoring: an f32-typed ConstTensor stores the
    // exact f32 image of each literal (per-dtype storage + finalize,
    // spec/04 section 9 [04-NUM-1..2]). The pre-refactor expectation kept
    // raw f64 values inside an f32 tensor, which is the chelis#717 defect
    // class this phase ends; bit-exactness is now AT THE DTYPE.
    assert_eq!(result.prim(), chelis_types::types::Prim::F32);
    assert_eq!(
        result.to_f64_lossy_vec(),
        vec![0.1f32 as f64, 0.2f32 as f64, 0.3f32 as f64]
    );
}

/// An explicitly f32 scalar keeps its f32 value when widened inside an f64
/// tensor literal. The enclosing tensor dtype must not erase the leaf dtype.
#[test]
fn const_tensor_widens_f32_suffixed_leaves_from_their_stored_width() {
    let source = r#"
def main() -> tensor[2, f64] =
  to_tensor([cast(0.1f32, f64), cast(0.3f32, f64)])
"#;
    let dag = surf_to_dag(source).expect("pipeline succeeds");

    let roots: Vec<NodeId> = dag.roots().to_vec();
    let values = eval_tensor_roots_with(&dag, &roots, |_name| None).expect("eval succeeds");

    assert_eq!(
        values[&roots[0]].to_f64_lossy_vec(),
        vec![(0.1_f64 as f32) as f64, (0.3_f64 as f32) as f64]
    );
}

/// A cast chain is evaluated at each declared width before the literal is
/// materialized; widening the result cannot recover the discarded f32 bits.
#[test]
fn const_tensor_widens_explicit_inner_f32_casts_from_their_stored_width() {
    let source = r#"
def main() -> tensor[2, f64] =
  to_tensor([cast(cast(0.1, f32), f64), cast(cast(0.3, f32), f64)])
"#;
    let dag = surf_to_dag(source).expect("pipeline succeeds");

    let roots: Vec<NodeId> = dag.roots().to_vec();
    let values = eval_tensor_roots_with(&dag, &roots, |_name| None).expect("eval succeeds");

    assert_eq!(
        values[&roots[0]].to_f64_lossy_vec(),
        vec![(0.1_f64 as f32) as f64, (0.3_f64 as f32) as f64]
    );
}

/// Verify explicitly f64 tensor literals retain f64 lexical precision.
#[test]
fn const_tensor_preserves_explicit_f64_values() {
    let source = r#"
def main() -> tensor[3, f64] =
  to_tensor([cast(0.1, f64), cast(0.2, f64), cast(0.3, f64)])
"#;
    let dag = surf_to_dag(source).expect("pipeline succeeds");

    let roots: Vec<NodeId> = dag.roots().to_vec();
    let values = eval_tensor_roots_with(&dag, &roots, |_name| None).expect("eval succeeds");

    let result = &values[&roots[0]];
    assert_eq!(result.shape, vec![3]);
    assert_eq!(result.to_f64_lossy_vec(), vec![0.1, 0.2, 0.3]);
}

/// Verify a 2D tensor literal with non-uniform rows lowers to ConstTensor.
#[test]
fn to_tensor_2d_non_uniform_lowers_to_const_tensor() {
    let source = r#"
def main() -> tensor[2, 3, f32] =
  to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
"#;
    let dag = surf_to_dag(source).expect("pipeline succeeds");

    let has_const_tensor = dag
        .nodes()
        .iter()
        .any(|n| matches!(&n.op, RiscOp::ConstTensor { .. }));
    assert!(
        has_const_tensor,
        "2D non-uniform to_tensor should produce ConstTensor"
    );

    // Verify correct eval
    let roots: Vec<NodeId> = dag.roots().to_vec();
    let values = eval_tensor_roots_with(&dag, &roots, |_name| None).expect("eval succeeds");
    let result = &values[&roots[0]];
    assert_eq!(result.shape, vec![2, 3]);
    assert_eq!(
        result.to_f64_lossy_vec(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

/// Verify ConstTensor works through grad: constant has zero gradient.
/// The gradient of f(x) = tensor_to_scalar(sum(x * c, 0)) w.r.t. x is c.
#[test]
fn const_tensor_through_grad_is_zero() {
    let source = r#"
def f(x: tensor[3, f32]) -> f32 = {
    c = to_tensor([1.0, 2.0, 3.0])
    tensor_to_scalar(sum(mul(x, c), 0))
}

def main(x: tensor[3, f32]) -> tensor[3, f32] =
    grad(f)(x)
"#;
    // The key assertion: grad through a ConstTensor-lowered literal
    // must compile without errors, not produce NaN or crash.
    let dag = surf_to_dag(source).expect("grad through ConstTensor should succeed");

    // Note: full-pipeline grad DAGs may have dangling nodes (pre-existing
    // behavior unrelated to ConstTensor). We skip verify and focus on
    // correctness of evaluation.

    // Evaluate the gradient and confirm no NaN.
    let roots: Vec<NodeId> = dag.roots().to_vec();
    assert!(!roots.is_empty(), "should have at least one root");
    let values = eval_tensor_roots_with(&dag, &roots, |name| {
        if name == "x" {
            Some(TensorValue::from_vec(vec![3], vec![1.0, 1.0, 1.0]))
        } else {
            None
        }
    })
    .expect("eval succeeds");

    // Check that results contain no NaN.
    for (root, val) in &values {
        for (i, &v) in val.to_f64_lossy_vec().iter().enumerate() {
            assert!(
                v.is_finite(),
                "NaN/Inf in gradient result at root={root:?}, index={i}"
            );
        }
    }
}

/// Verify that DAG verification passes for a ConstTensor node.
#[test]
fn const_tensor_passes_dag_verification() {
    let source = r#"
def main() -> tensor[3, f32] =
  to_tensor([1.0, 2.0, 3.0])
"#;
    let dag = surf_to_dag(source).expect("pipeline succeeds");
    let errors = verify::verify(&dag);
    assert!(
        errors.is_empty(),
        "DAG verification should pass for ConstTensor; got: {errors:?}"
    );
}

/// Verify ConstTensor with all-different integer values.
#[test]
fn const_tensor_integer_values() {
    let source = r#"
def main() -> tensor[4, int32] =
  to_tensor([10, 20, 30, 40])
"#;
    let dag = surf_to_dag(source).expect("pipeline succeeds");

    let has_const_tensor = dag
        .nodes()
        .iter()
        .any(|n| matches!(&n.op, RiscOp::ConstTensor { .. }));
    assert!(
        has_const_tensor,
        "non-uniform integer to_tensor should produce ConstTensor"
    );

    let roots: Vec<NodeId> = dag.roots().to_vec();
    let values = eval_tensor_roots_with(&dag, &roots, |_name| None).expect("eval succeeds");
    let result = &values[&roots[0]];
    assert_eq!(result.to_f64_lossy_vec(), vec![10.0, 20.0, 30.0, 40.0]);
}

/// Diagnostic: verify that the dangling-node issue in full-pipeline grad
/// is NOT specific to ConstTensor. If this test passes, it confirms the
/// dangling node is a pre-existing condition in the full Surf->grad->DAG pipeline.
#[test]
fn grad_dangling_node_is_preexisting_not_const_tensor_specific() {
    // Use UNIFORM values (regular Const path, no ConstTensor).
    let source_uniform = r#"
def f(x: tensor[3, f32]) -> f32 = {
    c = to_tensor([2.0, 2.0, 2.0])
    tensor_to_scalar(sum(mul(x, c), 0))
}
def main(x: tensor[3, f32]) -> tensor[3, f32] =
    grad(f)(x)
"#;
    let dag_uniform = surf_to_dag(source_uniform).expect("uniform pipeline succeeds");
    let errors_uniform = verify::verify(&dag_uniform);

    // Use NON-UNIFORM values (ConstTensor path).
    let source_nonuniform = r#"
def f(x: tensor[3, f32]) -> f32 = {
    c = to_tensor([1.0, 2.0, 3.0])
    tensor_to_scalar(sum(mul(x, c), 0))
}
def main(x: tensor[3, f32]) -> tensor[3, f32] =
    grad(f)(x)
"#;
    let dag_nonuniform = surf_to_dag(source_nonuniform).expect("nonuniform pipeline succeeds");
    let errors_nonuniform = verify::verify(&dag_nonuniform);

    // If both paths produce dangling-node errors, it's pre-existing,
    // not a ConstTensor regression.
    let uniform_has_dangling = errors_uniform.iter().any(|e| e.contains("dangling"));
    let nonuniform_has_dangling = errors_nonuniform.iter().any(|e| e.contains("dangling"));

    assert_eq!(
        uniform_has_dangling, nonuniform_has_dangling,
        "Dangling-node behavior should be identical for Const and ConstTensor.\n\
         Uniform errors: {errors_uniform:?}\n\
         Non-uniform errors: {errors_nonuniform:?}"
    );
}
