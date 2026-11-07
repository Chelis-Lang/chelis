//! Native cancellation entry adapters for the portable tensor evaluator.

use chelis_dag_core::dag::{Dag, NodeId};
use chelis_ir_core::eval::{InheritedResultClaim, TensorValue};
use chelis_unord::UnordMap;

pub fn eval_tensor_with<F>(
    dag: &Dag,
    load_input: F,
) -> Result<UnordMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    crate::resources::with_operation(|operation| {
        chelis_ir_core::eval::eval_tensor_with(dag, load_input, operation)
    })
}

pub fn eval_tensor_with_strict<F>(
    dag: &Dag,
    load_input: F,
) -> Result<UnordMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    crate::resources::with_operation(|operation| {
        chelis_ir_core::eval::eval_tensor_with_strict(dag, load_input, operation)
    })
}

pub fn eval_tensor_roots_with<F>(
    dag: &Dag,
    roots: &[NodeId],
    load_input: F,
) -> Result<UnordMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    crate::resources::with_operation(|operation| {
        chelis_ir_core::eval::eval_tensor_roots_with(dag, roots, load_input, operation)
    })
}

pub fn eval_tensor_roots_with_strict<F>(
    dag: &Dag,
    roots: &[NodeId],
    load_input: F,
) -> Result<UnordMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    crate::resources::with_operation(|operation| {
        chelis_ir_core::eval::eval_tensor_roots_with_strict(dag, roots, load_input, operation)
    })
}

pub fn eval_tensor_roots_exact<F>(
    dag: &Dag,
    roots: &[NodeId],
    load_input: F,
) -> Result<UnordMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    crate::resources::with_operation(|operation| {
        chelis_ir_core::eval::eval_tensor_roots_exact(dag, roots, load_input, operation)
    })
}

pub fn eval_tensor_roots_exact_with_result_claims<F>(
    dag: &Dag,
    roots: &[NodeId],
    result_claims: &[InheritedResultClaim],
    load_input: F,
) -> Result<UnordMap<NodeId, TensorValue>, String>
where
    F: FnMut(&str) -> Option<TensorValue>,
{
    crate::resources::with_operation(|operation| {
        chelis_ir_core::eval::eval_tensor_roots_exact_with_result_claims(
            dag,
            roots,
            result_claims,
            load_input,
            operation,
        )
    })
}

pub fn eval_tensor(
    dag: &Dag,
    inputs: &UnordMap<String, TensorValue>,
) -> Result<UnordMap<NodeId, TensorValue>, String> {
    crate::resources::with_operation(|operation| {
        chelis_ir_core::eval::eval_tensor(dag, inputs, operation)
    })
}

/// Evaluate scalar inputs with the existing native panic contract.
pub fn eval_scalar(dag: &Dag, inputs: &UnordMap<String, f64>) -> UnordMap<NodeId, f64> {
    crate::resources::with_operation(|operation| {
        chelis_ir_core::eval::eval_scalar(dag, inputs, operation)
            .expect("scalar evaluation should not fail")
    })
}
