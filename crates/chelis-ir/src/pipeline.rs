//! Shared transformation pipelines over RISC DAGs.
//!
//! Phase 1b requires automatic differentiation to run on the ordinary RISC DAG
//! and only fuse the resulting forward+backward graph afterward. This module
//! packages that ordering so callers do not accidentally differentiate a fused
//! DAG, which `grad_dag` intentionally rejects.

use crate::dag::{Dag, NodeId};
use crate::fuse::fuse_with_remap;
use crate::grad::{AdError, GradResult, grad_dag, grad_dag_checked};

/// Differentiate the unfused DAG, then fuse the combined forward+backward DAG.
///
/// This is the Phase 1b-safe ordering:
///
/// ```text
/// grad -> fuse
/// ```
///
/// The more complete Phase 1 GPU path also sandwiches `grad` between
/// optimization passes, but the ordering guarantee enforced here is that AD
/// sees an unfused DAG and fusion runs only on the result.
///
/// Prefer [`grad_then_fuse_checked`] in new code: it surfaces the
/// structured `AdError::NotSupported` rejection for non-differentiable
/// ops (argmax/argmin/floor/ceil/scatter_replace) instead of silently
/// returning `None`.
pub fn grad_then_fuse(forward: &Dag, output: NodeId, wrt: &[NodeId]) -> Option<GradResult> {
    let grad_result = grad_dag(forward, output, wrt)?;
    Some(fuse_grad_result(grad_result))
}

/// Like [`grad_then_fuse`] but returns the structured `AdError` from
/// [`grad_dag_checked`] when AD is not meaningful for the forward DAG
/// (e.g. argmax/argmin in the gradient path, floor/ceil in the
/// gradient path, replace-scatter in the gradient path).
pub fn grad_then_fuse_checked(
    forward: &Dag,
    output: NodeId,
    wrt: &[NodeId],
) -> Result<GradResult, AdError> {
    let grad_result = grad_dag_checked(forward, output, wrt)?;
    Ok(fuse_grad_result(grad_result))
}

fn fuse_grad_result(grad_result: GradResult) -> GradResult {
    let fused = fuse_with_remap(&grad_result.dag);
    let grad_nodes = grad_result
        .grad_nodes
        .into_iter()
        .map(|(wrt_id, grad_id)| {
            let remapped = *fused
                .old_to_new
                .get(&grad_id)
                .unwrap_or_else(|| panic!("gradient node {grad_id:?} missing after fusion"));
            (wrt_id, remapped)
        })
        .collect();
    let output_node = *fused
        .old_to_new
        .get(&grad_result.output_node)
        .unwrap_or_else(|| {
            panic!(
                "output node {:?} missing after fusion",
                grad_result.output_node
            )
        });

    GradResult {
        dag: fused.dag,
        output_node,
        grad_nodes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{DimInfo, RiscOp, TensorType};
    use crate::eval::{TensorValue, eval_tensor_roots_with_strict};
    use chelis_types::types::Prim;
    use std::collections::HashMap;

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    fn vec_f32(n: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(n)],
            precision: Prim::F32,
        }
    }

    fn eval_roots(dag: &Dag, inputs: &HashMap<String, TensorValue>) -> Vec<TensorValue> {
        let roots = dag.roots().to_vec();
        let values = eval_tensor_roots_with_strict(dag, &roots, |name| inputs.get(name).cloned())
            .expect("evaluation should succeed");
        roots.iter().map(|root| values[root].clone()).collect()
    }

    fn assert_outputs_close(actual: &[TensorValue], expected: &[TensorValue], tol: f64) {
        assert_eq!(actual.len(), expected.len(), "root count mismatch");
        for (actual_tensor, expected_tensor) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual_tensor.shape, expected_tensor.shape, "shape mismatch");
            for (actual_value, expected_value) in actual_tensor
                .to_f64_lossy_vec()
                .iter()
                .zip(expected_tensor.to_f64_lossy_vec().iter())
            {
                assert!(
                    (actual_value - expected_value).abs() <= tol,
                    "value mismatch: expected {expected_value}, got {actual_value}"
                );
            }
        }
    }

    #[test]
    fn grad_then_fuse_matches_unfused_grad() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
        let c = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 2.0),
            vec![],
            vec_f32(4),
            None,
        );
        let add = dag.add_node(RiscOp::Add, vec![x, c], vec_f32(4), None);
        let neg = dag.add_node(RiscOp::Neg, vec![add], vec_f32(4), None);
        let sum = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![neg],
            scalar_f32(),
            None,
        );
        dag.add_root(sum);

        let unfused_grad = grad_dag(&dag, sum, &[x]).expect("grad should succeed");
        let fused_grad = grad_then_fuse(&dag, sum, &[x]).expect("grad_then_fuse should succeed");

        let inputs = HashMap::from([(
            "x".to_string(),
            TensorValue::from_vec(vec![4], vec![1.0, -2.0, 3.5, 0.25]),
        )]);

        let unfused_outputs = eval_roots(&unfused_grad.dag, &inputs);
        let fused_outputs = eval_roots(&fused_grad.dag, &inputs);
        assert_outputs_close(&fused_outputs, &unfused_outputs, 1e-6);

        assert!(
            fused_grad
                .dag
                .nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::FusedElem { .. })),
            "fused grad DAG should contain a FusedElem node"
        );
    }

    #[test]
    fn grad_rejects_fused_input_but_grad_then_fuse_succeeds() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
        let c = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        let add = dag.add_node(RiscOp::Add, vec![x, c], vec_f32(4), None);
        let neg = dag.add_node(RiscOp::Neg, vec![add], vec_f32(4), None);
        let sum = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![neg],
            scalar_f32(),
            None,
        );
        dag.add_root(sum);

        let fused_forward = crate::fuse::fuse(&dag);
        let fused_output = *fused_forward
            .roots()
            .first()
            .expect("fused DAG should keep the original root");

        assert!(
            grad_dag(&fused_forward, fused_output, &[x]).is_none(),
            "grad_dag should reject already-fused inputs"
        );
        assert!(
            grad_then_fuse(&dag, sum, &[x]).is_some(),
            "grad_then_fuse should succeed on the unfused DAG"
        );
    }
}
