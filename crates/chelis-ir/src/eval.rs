//! Tensor-aware evaluator for the Phase 0 RISC DAG.

use std::collections::HashMap;

use crate::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};

#[derive(Debug, Clone, PartialEq)]
pub struct TensorValue {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl TensorValue {
    pub fn scalar(value: f64) -> Self {
        Self {
            data: vec![value],
            shape: vec![],
        }
    }

    pub fn from_vec(shape: Vec<usize>, data: Vec<f64>) -> Self {
        assert_eq!(numel(&shape), data.len());
        Self { data, shape }
    }
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product::<usize>().max(1)
}

fn concrete_shape(ty: &TensorType) -> Result<Vec<usize>, String> {
    ty.dims
        .iter()
        .map(|dim| match dim {
            DimInfo::Lit(n) => Ok(*n),
            DimInfo::Named(_, Some(n)) => Ok(*n),
            DimInfo::Named(name, None) => {
                Err(format!("cannot evaluate symbolic dimension `{name}`"))
            }
        })
        .collect()
}

fn default_value(ty: &TensorType) -> TensorValue {
    let shape = concrete_shape(ty).unwrap_or_default();
    TensorValue {
        data: vec![0.0; numel(&shape)],
        shape,
    }
}

fn linear_to_index(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut index = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        index[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
    index
}

fn index_to_linear(index: &[usize], shape: &[usize]) -> usize {
    let mut linear = 0usize;
    for (axis, &value) in index.iter().enumerate() {
        linear *= shape[axis];
        linear += value;
    }
    linear
}

fn unary_map(input: &TensorValue, f: impl Fn(f64) -> f64) -> TensorValue {
    TensorValue {
        data: input.data.iter().copied().map(f).collect(),
        shape: input.shape.clone(),
    }
}

fn binary_map(lhs: &TensorValue, rhs: &TensorValue, f: impl Fn(f64, f64) -> f64) -> TensorValue {
    assert_eq!(lhs.shape, rhs.shape);
    TensorValue {
        data: lhs
            .data
            .iter()
            .copied()
            .zip(rhs.data.iter().copied())
            .map(|(a, b)| f(a, b))
            .collect(),
        shape: lhs.shape.clone(),
    }
}

fn reduce(input: &TensorValue, axis: usize, init: f64, f: impl Fn(f64, f64) -> f64) -> TensorValue {
    assert!(axis < input.shape.len());
    let mut out_shape = input.shape.clone();
    out_shape.remove(axis);
    let out_len = numel(&out_shape);
    let mut out = vec![init; out_len];
    for (flat_idx, &value) in input.data.iter().enumerate() {
        let mut idx = linear_to_index(flat_idx, &input.shape);
        idx.remove(axis);
        let out_idx = index_to_linear(&idx, &out_shape);
        out[out_idx] = f(out[out_idx], value);
    }
    TensorValue {
        data: out,
        shape: out_shape,
    }
}

fn reshape(input: &TensorValue, shape: Vec<usize>) -> TensorValue {
    assert_eq!(input.data.len(), numel(&shape));
    TensorValue {
        data: input.data.clone(),
        shape,
    }
}

fn permute(input: &TensorValue, axes: &[usize]) -> TensorValue {
    assert_eq!(axes.len(), input.shape.len());
    let out_shape: Vec<usize> = axes.iter().map(|&axis| input.shape[axis]).collect();
    let out_len = numel(&out_shape);
    let mut out = vec![0.0; out_len];
    for (flat_idx, slot) in out.iter_mut().enumerate() {
        let out_index = linear_to_index(flat_idx, &out_shape);
        let mut in_index = vec![0; input.shape.len()];
        for (out_axis, &in_axis) in axes.iter().enumerate() {
            in_index[in_axis] = out_index[out_axis];
        }
        *slot = input.data[index_to_linear(&in_index, &input.shape)];
    }
    TensorValue {
        data: out,
        shape: out_shape,
    }
}

fn expand(input: &TensorValue, axis: usize, size: usize) -> TensorValue {
    assert!(axis <= input.shape.len());
    let mut out_shape = input.shape.clone();
    out_shape.insert(axis, size);
    let out_len = numel(&out_shape);
    let mut out = vec![0.0; out_len];
    for (flat_idx, slot) in out.iter_mut().enumerate() {
        let mut out_index = linear_to_index(flat_idx, &out_shape);
        out_index.remove(axis);
        *slot = input.data[index_to_linear(&out_index, &input.shape)];
    }
    TensorValue {
        data: out,
        shape: out_shape,
    }
}

fn pad(input: &TensorValue, padding: &[(usize, usize)], fill: f64) -> TensorValue {
    assert_eq!(padding.len(), input.shape.len());
    let out_shape: Vec<usize> = input
        .shape
        .iter()
        .zip(padding.iter())
        .map(|(dim, (before, after))| dim + before + after)
        .collect();
    let mut out = vec![fill; numel(&out_shape)];
    for (flat_idx, &value) in input.data.iter().enumerate() {
        let in_index = linear_to_index(flat_idx, &input.shape);
        let out_index: Vec<usize> = in_index
            .iter()
            .zip(padding.iter())
            .map(|(idx, (before, _))| idx + before)
            .collect();
        let out_flat = index_to_linear(&out_index, &out_shape);
        out[out_flat] = value;
    }
    TensorValue {
        data: out,
        shape: out_shape,
    }
}

fn shrink(input: &TensorValue, bounds: &[(usize, usize)]) -> TensorValue {
    assert_eq!(bounds.len(), input.shape.len());
    let out_shape: Vec<usize> = input
        .shape
        .iter()
        .zip(bounds.iter())
        .map(|(_, (start, end))| end - start)
        .collect();
    let out_len = numel(&out_shape);
    let mut out = vec![0.0; out_len];
    for (flat_idx, slot) in out.iter_mut().enumerate() {
        let out_index = linear_to_index(flat_idx, &out_shape);
        let in_index: Vec<usize> = out_index
            .iter()
            .zip(bounds.iter())
            .map(|(idx, (start, _))| idx + start)
            .collect();
        *slot = input.data[index_to_linear(&in_index, &input.shape)];
    }
    TensorValue {
        data: out,
        shape: out_shape,
    }
}

fn stride(input: &TensorValue, strides: &[usize]) -> TensorValue {
    assert_eq!(strides.len(), input.shape.len());
    let out_shape: Vec<usize> = input
        .shape
        .iter()
        .zip(strides.iter())
        .map(|(dim, step)| {
            if *step == 0 {
                *dim
            } else {
                (*dim).div_ceil(*step)
            }
        })
        .collect();
    let out_len = numel(&out_shape);
    let mut out = vec![0.0; out_len];
    for (flat_idx, slot) in out.iter_mut().enumerate() {
        let out_index = linear_to_index(flat_idx, &out_shape);
        let in_index: Vec<usize> = out_index
            .iter()
            .zip(strides.iter())
            .map(|(idx, step)| idx * step.max(&1))
            .collect();
        *slot = input.data[index_to_linear(&in_index, &input.shape)];
    }
    TensorValue {
        data: out,
        shape: out_shape,
    }
}

pub fn eval_tensor(
    dag: &Dag,
    inputs: &HashMap<String, TensorValue>,
) -> Result<HashMap<NodeId, TensorValue>, String> {
    let mut values: HashMap<NodeId, TensorValue> = HashMap::new();

    for node in dag.nodes() {
        let value = match &node.op {
            RiscOp::Const { value } => {
                let shape = concrete_shape(&node.output_type).unwrap_or_default();
                TensorValue {
                    data: vec![*value; numel(&shape)],
                    shape,
                }
            }
            RiscOp::Load { name } => inputs
                .get(name)
                .cloned()
                .unwrap_or_else(|| default_value(&node.output_type)),
            RiscOp::Store { .. } => values[&node.inputs[0]].clone(),
            RiscOp::Add => binary_map(
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
                |a, b| a + b,
            ),
            RiscOp::Mul => binary_map(
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
                |a, b| a * b,
            ),
            RiscOp::Neg => unary_map(&values[&node.inputs[0]], |x| -x),
            RiscOp::Exp => unary_map(&values[&node.inputs[0]], f64::exp),
            RiscOp::Log => unary_map(&values[&node.inputs[0]], f64::ln),
            RiscOp::Sin => unary_map(&values[&node.inputs[0]], f64::sin),
            RiscOp::Sqrt => unary_map(&values[&node.inputs[0]], f64::sqrt),
            RiscOp::MaxElem => {
                binary_map(&values[&node.inputs[0]], &values[&node.inputs[1]], f64::max)
            }
            RiscOp::CmpLt => binary_map(
                &values[&node.inputs[0]],
                &values[&node.inputs[1]],
                |a, b| {
                    if a < b { 1.0 } else { 0.0 }
                },
            ),
            RiscOp::Sum { axis } => reduce(&values[&node.inputs[0]], *axis, 0.0, |acc, x| acc + x),
            RiscOp::MaxReduce { axis } => {
                reduce(&values[&node.inputs[0]], *axis, f64::NEG_INFINITY, f64::max)
            }
            RiscOp::Reshape { new_shape } => {
                let shape: Vec<usize> = new_shape
                    .iter()
                    .map(|dim| match dim {
                        DimInfo::Lit(n) => Ok(*n),
                        DimInfo::Named(_, Some(n)) => Ok(*n),
                        DimInfo::Named(name, None) => {
                            Err(format!("cannot reshape to symbolic dimension `{name}`"))
                        }
                    })
                    .collect::<Result<_, _>>()?;
                reshape(&values[&node.inputs[0]], shape)
            }
            RiscOp::Permute { axes } => permute(&values[&node.inputs[0]], axes),
            RiscOp::Expand { axis, size } => expand(&values[&node.inputs[0]], *axis, *size),
            RiscOp::Pad { padding, fill } => pad(&values[&node.inputs[0]], padding, *fill),
            RiscOp::Shrink { bounds } => shrink(&values[&node.inputs[0]], bounds),
            RiscOp::Stride { strides } => stride(&values[&node.inputs[0]], strides),
            RiscOp::Cast { .. } => values[&node.inputs[0]].clone(),
        };
        values.insert(node.id, value);
    }

    Ok(values)
}

/// Evaluate a DAG on scalar inputs. Each node produces a single f64.
pub fn eval_scalar(dag: &Dag, inputs: &HashMap<String, f64>) -> HashMap<NodeId, f64> {
    let tensor_inputs: HashMap<String, TensorValue> = inputs
        .iter()
        .map(|(name, value)| (name.clone(), TensorValue::scalar(*value)))
        .collect();
    eval_tensor(dag, &tensor_inputs)
        .expect("scalar evaluation should not fail")
        .into_iter()
        .map(|(id, value)| (id, *value.data.first().unwrap_or(&0.0)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::RiscOp;
    use crate::lower::lower_program;
    use chelis_deep::parser::parse_str;
    use chelis_types::types::Prim;

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    fn vec3_f32() -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        }
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
    fn eval_vector_add() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec3_f32());
        let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec3_f32());
        let c = dag.add_node(RiscOp::Add, vec![a, b], vec3_f32());
        let mut inputs = HashMap::new();
        inputs.insert(
            "a".into(),
            TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
        );
        inputs.insert(
            "b".into(),
            TensorValue::from_vec(vec![3], vec![4.0, 5.0, 6.0]),
        );
        let vals = eval_tensor(&dag, &inputs).unwrap();
        assert_eq!(
            vals[&c],
            TensorValue::from_vec(vec![3], vec![5.0, 7.0, 9.0])
        );
    }

    fn lower(src: &str) -> Dag {
        let exprs = parse_str(src).expect("parse failed");
        let checked = chelis_types::check_phase0e_program(&exprs)
            .unwrap_or_else(|result| panic!("phase 0e check failed: {:?}", result.errors));
        lower_program(&checked)
    }

    #[test]
    fn lowered_relu_has_correct_numeric_result() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x))
            (def {} y (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} (var {} relu) (var {} x)))
        "#;
        let dag = lower(src);
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".into(),
            TensorValue::from_vec(vec![3], vec![-2.0, 0.5, 4.0]),
        );
        let vals = eval_tensor(&dag, &inputs).unwrap();
        let last = vals.get(&NodeId(dag.len() - 1)).unwrap();
        assert_eq!(*last, TensorValue::from_vec(vec![3], vec![0.0, 0.5, 4.0]));
    }

    #[test]
    fn lowered_matmul_has_correct_numeric_result() {
        let src = r#"
            (def {} a (var {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))} a))
            (def {} b (var {type: (t-tensor {} (d-lit {} 3) (d-lit {} 2) (t-prim {} f32))} b))
            (def {} c
              (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))}
                   (var {} matmul) (var {} a) (var {} b)))
        "#;
        let dag = lower(src);
        let mut inputs = HashMap::new();
        inputs.insert(
            "a".into(),
            TensorValue::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );
        inputs.insert(
            "b".into(),
            TensorValue::from_vec(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        );
        let vals = eval_tensor(&dag, &inputs).unwrap();
        let last = vals.get(&NodeId(dag.len() - 1)).unwrap();
        assert_eq!(
            *last,
            TensorValue::from_vec(vec![2, 2], vec![58.0, 64.0, 139.0, 154.0])
        );
    }

    #[test]
    fn lowered_softmax_has_correct_numeric_result() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x))
            (def {} y
              (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
                   (var {} softmax) (var {} x) (lit {} 0)))
        "#;
        let dag = lower(src);
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".into(),
            TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
        );
        let vals = eval_tensor(&dag, &inputs).unwrap();
        let last = vals.get(&NodeId(dag.len() - 1)).unwrap();
        let expected = [0.09003057, 0.24472847, 0.66524096];
        for (actual, target) in last.data.iter().zip(expected.iter()) {
            assert!((actual - target).abs() < 1e-5);
        }
    }

    #[test]
    fn lowered_layer_norm_has_correct_numeric_result() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))} x))
            (def {} gamma (var {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))} gamma))
            (def {} beta (var {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))} beta))
            (def {} y
              (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))}
                   (var {} layer_norm) (var {} x) (var {} gamma) (var {} beta)))
        "#;
        let dag = lower(src);
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".into(),
            TensorValue::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 5.0]),
        );
        inputs.insert(
            "gamma".into(),
            TensorValue::from_vec(vec![2], vec![1.0, 1.5]),
        );
        inputs.insert(
            "beta".into(),
            TensorValue::from_vec(vec![2], vec![0.5, -0.5]),
        );
        let vals = eval_tensor(&dag, &inputs).unwrap();
        let last = vals.get(&NodeId(dag.len() - 1)).unwrap();
        let expected = [-0.49998, 0.99997, -0.499995, 0.99998875];
        for (actual, target) in last.data.iter().zip(expected.iter()) {
            assert!((actual - target).abs() < 2e-4, "{actual} vs {target}");
        }
    }

    #[test]
    fn lowered_conv2d_1x1_has_correct_numeric_result() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 1) (d-lit {} 1) (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))} x))
            (def {} k (var {type: (t-tensor {} (d-lit {} 1) (d-lit {} 1) (d-lit {} 1) (d-lit {} 1) (t-prim {} f32))} k))
            (def {} y
              (app {type: (t-tensor {} (d-lit {} 1) (d-lit {} 1) (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))}
                   (var {} conv2d) (var {} x) (var {} k) (lit {} 1) (lit {} 0)))
        "#;
        let dag = lower(src);
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".into(),
            TensorValue::from_vec(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]),
        );
        inputs.insert(
            "k".into(),
            TensorValue::from_vec(vec![1, 1, 1, 1], vec![2.0]),
        );
        let vals = eval_tensor(&dag, &inputs).unwrap();
        let last = vals.get(&NodeId(dag.len() - 1)).unwrap();
        assert_eq!(
            *last,
            TensorValue::from_vec(vec![1, 1, 2, 2], vec![2.0, 4.0, 6.0, 8.0])
        );
    }

    #[test]
    fn lowered_conv2d_2x2_has_correct_numeric_result() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 1) (d-lit {} 1) (d-lit {} 3) (d-lit {} 3) (t-prim {} f32))} x))
            (def {} k (var {type: (t-tensor {} (d-lit {} 1) (d-lit {} 1) (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))} k))
            (def {} y
              (app {type: (t-tensor {} (d-lit {} 1) (d-lit {} 1) (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))}
                   (var {} conv2d) (var {} x) (var {} k) (lit {} 1) (lit {} 0)))
        "#;
        let dag = lower(src);
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".into(),
            TensorValue::from_vec(
                vec![1, 1, 3, 3],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            ),
        );
        inputs.insert(
            "k".into(),
            TensorValue::from_vec(vec![1, 1, 2, 2], vec![1.0, 1.0, 1.0, 1.0]),
        );
        let vals = eval_tensor(&dag, &inputs).unwrap();
        let last = vals.get(&NodeId(dag.len() - 1)).unwrap();
        assert_eq!(
            *last,
            TensorValue::from_vec(vec![1, 1, 2, 2], vec![12.0, 16.0, 24.0, 28.0])
        );
    }
}
