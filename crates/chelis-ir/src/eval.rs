//! Simple scalar evaluator for the RISC DAG.

use std::collections::HashMap;

use crate::dag::*;

/// Evaluate a DAG on scalar values. Each node produces a single f64.
/// Only handles scalar operations (no tensor dims).
pub fn eval_scalar(dag: &Dag, inputs: &HashMap<String, f64>) -> HashMap<NodeId, f64> {
    let mut values: HashMap<NodeId, f64> = HashMap::new();

    for node in dag.nodes() {
        let val = match &node.op {
            RiscOp::Const { value } => *value,
            RiscOp::Load { name } => *inputs.get(name.as_str()).unwrap_or(&0.0),
            RiscOp::Add => values[&node.inputs[0]] + values[&node.inputs[1]],
            RiscOp::Mul => values[&node.inputs[0]] * values[&node.inputs[1]],
            RiscOp::Neg => -values[&node.inputs[0]],
            RiscOp::Exp => values[&node.inputs[0]].exp(),
            RiscOp::Log => values[&node.inputs[0]].ln(),
            RiscOp::Sin => values[&node.inputs[0]].sin(),
            RiscOp::Sqrt => values[&node.inputs[0]].sqrt(),
            RiscOp::MaxElem => values[&node.inputs[0]].max(values[&node.inputs[1]]),
            RiscOp::CmpLt => {
                if values[&node.inputs[0]] < values[&node.inputs[1]] {
                    1.0
                } else {
                    0.0
                }
            }
            _ => 0.0, // Movement/reduction ops not supported in scalar eval
        };
        values.insert(node.id, val);
    }

    values
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    #[test]
    fn eval_add() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
        let vals = eval_scalar(&dag, &HashMap::new());
        assert!((vals[&c] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn eval_neg() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: -5.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Neg, vec![a], scalar_f32());
        let vals = eval_scalar(&dag, &HashMap::new());
        assert!((vals[&b] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn eval_mul() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 4.0 }, vec![], scalar_f32());
        let c = dag.add_node(RiscOp::Mul, vec![a, b], scalar_f32());
        let vals = eval_scalar(&dag, &HashMap::new());
        assert!((vals[&c] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn eval_relu_negative() {
        // relu(x) = max(x, 0) using MaxElem
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: -1.0 }, vec![], scalar_f32());
        let zero = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_f32());
        let r = dag.add_node(RiscOp::MaxElem, vec![x, zero], scalar_f32());
        let vals = eval_scalar(&dag, &HashMap::new());
        assert!((vals[&r] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn eval_relu_positive() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32());
        let zero = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_f32());
        let r = dag.add_node(RiscOp::MaxElem, vec![x, zero], scalar_f32());
        let vals = eval_scalar(&dag, &HashMap::new());
        assert!((vals[&r] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn eval_sub() {
        // sub(a, b) = a + neg(b)
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 10.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let neg_b = dag.add_node(RiscOp::Neg, vec![b], scalar_f32());
        let c = dag.add_node(RiscOp::Add, vec![a, neg_b], scalar_f32());
        let vals = eval_scalar(&dag, &HashMap::new());
        assert!((vals[&c] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn eval_sigmoid_at_zero() {
        // sigmoid(0) = 1 / (1 + exp(-0)) = 0.5
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_f32());
        let neg_x = dag.add_node(RiscOp::Neg, vec![x], scalar_f32());
        let exp_neg = dag.add_node(RiscOp::Exp, vec![neg_x], scalar_f32());
        let one = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let denom = dag.add_node(RiscOp::Add, vec![one, exp_neg], scalar_f32());
        // 1/denom = exp(log(1) - log(denom)) ... but simpler: use log+neg+exp trick
        // Actually, let's use the reciprocal: 1/x = exp(-log(x))
        let log_d = dag.add_node(RiscOp::Log, vec![denom], scalar_f32());
        let neg_log = dag.add_node(RiscOp::Neg, vec![log_d], scalar_f32());
        let result = dag.add_node(RiscOp::Exp, vec![neg_log], scalar_f32());
        let vals = eval_scalar(&dag, &HashMap::new());
        assert!((vals[&result] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn eval_exp_zero() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_f32());
        let e = dag.add_node(RiscOp::Exp, vec![x], scalar_f32());
        let vals = eval_scalar(&dag, &HashMap::new());
        assert!((vals[&e] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn eval_log_one() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let l = dag.add_node(RiscOp::Log, vec![x], scalar_f32());
        let vals = eval_scalar(&dag, &HashMap::new());
        assert!((vals[&l] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn eval_load() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load {
                name: "x".to_string(),
            },
            vec![],
            scalar_f32(),
        );
        let y = dag.add_node(
            RiscOp::Load {
                name: "y".to_string(),
            },
            vec![],
            scalar_f32(),
        );
        let sum = dag.add_node(RiscOp::Add, vec![x, y], scalar_f32());
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), 3.0);
        inputs.insert("y".to_string(), 7.0);
        let vals = eval_scalar(&dag, &inputs);
        assert!((vals[&sum] - 10.0).abs() < 1e-10);
    }
}
