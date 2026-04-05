//! Reverse-mode automatic differentiation for RISC DAGs.
//!
//! Given a forward DAG computing `f(inputs) -> output`, produces a backward DAG
//! computing gradients of the output with respect to specified input nodes.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::dag::{Dag, DagNode, DimInfo, NodeId, RiscOp, TensorType};
use crate::tier2;

/// Result of reverse-mode AD.
pub struct GradResult {
    /// Combined forward + backward DAG.
    pub dag: Dag,
    /// Maps each requested forward input `NodeId` to its gradient `NodeId` in the combined DAG.
    pub grad_nodes: HashMap<NodeId, NodeId>,
}

/// Run reverse-mode AD on `forward`, differentiating `output` with respect to each node in `wrt`.
pub fn grad_dag(forward: &Dag, output: NodeId, wrt: &[NodeId]) -> GradResult {
    let mut dag = forward.clone();
    let mut adjoints: HashMap<NodeId, NodeId> = HashMap::new();

    // Seed: adjoint of output = 1.0 with same type.
    let output_ty = forward.get(output).unwrap().output_type.clone();
    let seed = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], output_ty);
    adjoints.insert(output, seed);

    // Walk forward topological order in reverse.
    let topo = forward.topological_order();
    for &node_id in topo.iter().rev() {
        let grad_out = match adjoints.get(&node_id) {
            Some(&g) => g,
            None => continue,
        };

        let node = forward.get(node_id).unwrap().clone();
        let input_grads = compute_adjoints(&node, grad_out, forward, &mut dag);

        for (input_id, grad_node) in input_grads {
            match adjoints.entry(input_id) {
                Entry::Vacant(e) => {
                    e.insert(grad_node);
                }
                Entry::Occupied(mut e) => {
                    let existing = *e.get();
                    let ty = dag.get(existing).unwrap().output_type.clone();
                    let sum = dag.add_node(RiscOp::Add, vec![existing, grad_node], ty);
                    e.insert(sum);
                }
            }
        }
    }

    let grad_nodes = wrt
        .iter()
        .filter_map(|&id| adjoints.get(&id).map(|&g| (id, g)))
        .collect();

    GradResult { dag, grad_nodes }
}

/// Compute adjoint contributions for each input of the given node.
/// Returns `(forward_input_id, gradient_node_in_dag)` pairs.
fn compute_adjoints(
    node: &DagNode,
    g: NodeId,
    forward: &Dag,
    dag: &mut Dag,
) -> Vec<(NodeId, NodeId)> {
    match &node.op {
        // --- Binary elementwise ---
        RiscOp::Add => {
            let a = node.inputs[0];
            let b = node.inputs[1];
            vec![(a, g), (b, g)]
        }
        RiscOp::Mul => {
            let a = node.inputs[0];
            let b = node.inputs[1];
            let ty = forward.get(a).unwrap().output_type.clone();
            // da = g * b, db = g * a  (referencing forward nodes directly)
            let da = dag.add_node(RiscOp::Mul, vec![g, b], ty.clone());
            let db = dag.add_node(RiscOp::Mul, vec![g, a], ty);
            vec![(a, da), (b, db)]
        }
        RiscOp::CmpLt => {
            let a = node.inputs[0];
            let b = node.inputs[1];
            let ty_a = forward.get(a).unwrap().output_type.clone();
            let ty_b = forward.get(b).unwrap().output_type.clone();
            let za = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], ty_a);
            let zb = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], ty_b);
            vec![(a, za), (b, zb)]
        }
        RiscOp::MaxElem => {
            // Subgradient: da = g * (b < a), db = g * (a < b)
            let a = node.inputs[0];
            let b = node.inputs[1];
            let ty = forward.get(a).unwrap().output_type.clone();
            let bool_ty = TensorType {
                dims: ty.dims.clone(),
                precision: chelis_types::types::Prim::F32,
            };
            let b_lt_a = dag.add_node(RiscOp::CmpLt, vec![b, a], bool_ty.clone());
            let a_lt_b = dag.add_node(RiscOp::CmpLt, vec![a, b], bool_ty);
            let da = dag.add_node(RiscOp::Mul, vec![g, b_lt_a], ty.clone());
            let db = dag.add_node(RiscOp::Mul, vec![g, a_lt_b], ty);
            vec![(a, da), (b, db)]
        }

        // --- Unary elementwise ---
        RiscOp::Neg => {
            let x = node.inputs[0];
            let ty = forward.get(x).unwrap().output_type.clone();
            let dg = dag.add_node(RiscOp::Neg, vec![g], ty);
            vec![(x, dg)]
        }
        RiscOp::Exp => {
            // d/dx exp(x) = exp(x). Reuse the forward exp node.
            let x = node.inputs[0];
            let ty = forward.get(x).unwrap().output_type.clone();
            let dx = dag.add_node(RiscOp::Mul, vec![g, node.id], ty);
            vec![(x, dx)]
        }
        RiscOp::Log => {
            // d/dx log(x) = 1/x = div(g, x)
            let x = node.inputs[0];
            let ty = forward.get(x).unwrap().output_type.clone();
            let dx = tier2::lower_div(dag, g, x, &ty);
            vec![(x, dx)]
        }
        RiscOp::Sin => {
            // d/dx sin(x) = cos(x) = sin(x + pi/2)
            let x = node.inputs[0];
            let ty = forward.get(x).unwrap().output_type.clone();
            let half_pi = dag.add_node(
                RiscOp::Const {
                    value: std::f64::consts::FRAC_PI_2,
                },
                vec![],
                ty.clone(),
            );
            let shifted = dag.add_node(RiscOp::Add, vec![x, half_pi], ty.clone());
            let cos_x = dag.add_node(RiscOp::Sin, vec![shifted], ty.clone());
            let dx = dag.add_node(RiscOp::Mul, vec![g, cos_x], ty);
            vec![(x, dx)]
        }
        RiscOp::Sqrt => {
            // d/dx sqrt(x) = 1 / (2 * sqrt(x)). Reuse forward sqrt node.
            let x = node.inputs[0];
            let ty = forward.get(x).unwrap().output_type.clone();
            let two = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], ty.clone());
            let two_sqrt = dag.add_node(RiscOp::Mul, vec![two, node.id], ty.clone());
            let dx = tier2::lower_div(dag, g, two_sqrt, &ty);
            vec![(x, dx)]
        }

        // --- Reduction ---
        RiscOp::Sum { axis } => {
            // d/dx sum(x, axis) = expand(g, axis, original_size)
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            let original_size = dim_size(&input_ty.dims[*axis]);
            let dx = dag.add_node(
                RiscOp::Expand {
                    axis: *axis,
                    size: original_size,
                },
                vec![g],
                input_ty,
            );
            vec![(x, dx)]
        }
        RiscOp::MaxReduce { axis } => {
            // Subgradient: gradient flows to elements equal to the max.
            // mask = eq(x, expand(max_reduce(x, axis), axis, size))
            // dx = mul(expand(g, axis, size), mask)
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            let original_size = dim_size(&input_ty.dims[*axis]);

            // Expand forward max_reduce node back to input shape.
            let expanded_max = dag.add_node(
                RiscOp::Expand {
                    axis: *axis,
                    size: original_size,
                },
                vec![node.id],
                input_ty.clone(),
            );

            // Expand gradient to input shape.
            let expanded_g = dag.add_node(
                RiscOp::Expand {
                    axis: *axis,
                    size: original_size,
                },
                vec![g],
                input_ty.clone(),
            );

            // Build equality mask: not(or(cmplt(x, expanded_max), cmplt(expanded_max, x)))
            let mask = tier2::lower_eq(dag, x, expanded_max, &input_ty);

            let dx = dag.add_node(RiscOp::Mul, vec![expanded_g, mask], input_ty);
            vec![(x, dx)]
        }

        // --- Movement ---
        RiscOp::Reshape { .. } => {
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            let original_shape = input_ty.dims.clone();
            let dx = dag.add_node(
                RiscOp::Reshape {
                    new_shape: original_shape,
                },
                vec![g],
                input_ty,
            );
            vec![(x, dx)]
        }
        RiscOp::Permute { axes } => {
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            let inv = inverse_permutation(axes);
            let dx = dag.add_node(RiscOp::Permute { axes: inv }, vec![g], input_ty);
            vec![(x, dx)]
        }
        RiscOp::Expand { axis, .. } => {
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            let dx = dag.add_node(RiscOp::Sum { axis: *axis }, vec![g], input_ty);
            vec![(x, dx)]
        }
        RiscOp::Pad { padding, .. } => {
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            // Shrink: bounds = [(before, before + dim_size), ...] for each axis
            let bounds: Vec<(usize, usize)> = padding
                .iter()
                .zip(input_ty.dims.iter())
                .map(|((before, _after), dim)| (*before, *before + dim_size(dim)))
                .collect();
            let dx = dag.add_node(RiscOp::Shrink { bounds }, vec![g], input_ty);
            vec![(x, dx)]
        }
        RiscOp::Shrink { bounds } => {
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            // Pad: for each axis, before = start, after = original_size - end
            let padding: Vec<(usize, usize)> = bounds
                .iter()
                .zip(input_ty.dims.iter())
                .map(|((start, end), dim)| (*start, dim_size(dim) - end))
                .collect();
            let dx = dag.add_node(RiscOp::Pad { padding, fill: 0.0 }, vec![g], input_ty);
            vec![(x, dx)]
        }
        RiscOp::Stride { .. } => {
            // TODO: complex scatter operation, zero gradient for Phase 0
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            let dx = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], input_ty);
            vec![(x, dx)]
        }

        // --- Memory ---
        RiscOp::Const { .. } => vec![],
        RiscOp::Load { .. } => vec![],
        RiscOp::Store { .. } => {
            let x = node.inputs[0];
            vec![(x, g)]
        }

        // --- Cast ---
        RiscOp::Cast { .. } => {
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            let dx = dag.add_node(
                RiscOp::Cast {
                    new_precision: input_ty.precision,
                },
                vec![g],
                input_ty,
            );
            vec![(x, dx)]
        }
    }
}

fn inverse_permutation(axes: &[usize]) -> Vec<usize> {
    let mut inv = vec![0; axes.len()];
    for (new_pos, &old_pos) in axes.iter().enumerate() {
        inv[old_pos] = new_pos;
    }
    inv
}

fn dim_size(dim: &DimInfo) -> usize {
    match dim {
        DimInfo::Lit(n) => *n,
        DimInfo::Named(_, Some(n)) => *n,
        DimInfo::Named(name, None) => {
            panic!("cannot determine size for symbolic dimension `{name}`")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::eval_scalar;
    use std::collections::HashMap;

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    /// Build a unary DAG: Load("x") -> op -> output.
    fn build_unary_dag(
        op_fn: impl FnOnce(&mut Dag, NodeId, &TensorType) -> NodeId,
    ) -> (Dag, NodeId, NodeId) {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], scalar_f32());
        let y = op_fn(&mut dag, x, &scalar_f32());
        (dag, x, y)
    }

    /// Build a binary DAG: Load("x"), Load("y") -> op -> output.
    fn build_binary_dag(
        op_fn: impl FnOnce(&mut Dag, NodeId, NodeId, &TensorType) -> NodeId,
    ) -> (Dag, NodeId, NodeId, NodeId) {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], scalar_f32());
        let y = dag.add_node(RiscOp::Load { name: "y".into() }, vec![], scalar_f32());
        let out = op_fn(&mut dag, x, y, &scalar_f32());
        (dag, x, y, out)
    }

    /// Compute both analytical (via AD) and numerical (via finite differences) gradients.
    fn finite_diff(
        dag: &Dag,
        output: NodeId,
        wrt: NodeId,
        input_name: &str,
        other_inputs: &[(&str, f64)],
        x0: f64,
        h: f64,
    ) -> (f64, f64) {
        let grad_result = grad_dag(dag, output, &[wrt]);

        // Analytical gradient.
        let mut inputs: HashMap<String, f64> = HashMap::new();
        inputs.insert(input_name.to_string(), x0);
        for &(name, val) in other_inputs {
            inputs.insert(name.to_string(), val);
        }
        let vals = eval_scalar(&grad_result.dag, &inputs);
        let analytical = vals[&grad_result.grad_nodes[&wrt]];

        // Numerical gradient via central differences.
        inputs.insert(input_name.to_string(), x0 + h);
        let f_plus = eval_scalar(dag, &inputs)[&output];
        inputs.insert(input_name.to_string(), x0 - h);
        let f_minus = eval_scalar(dag, &inputs)[&output];
        let numerical = (f_plus - f_minus) / (2.0 * h);

        (analytical, numerical)
    }

    fn assert_grad_close(analytical: f64, numerical: f64) {
        assert!(
            (analytical - numerical).abs() < 1e-4,
            "gradient mismatch: analytical={analytical}, numerical={numerical}"
        );
    }

    // ---- Primitive adjoint tests ----

    #[test]
    fn grad_add() {
        let (dag, x, _y, out) =
            build_binary_dag(|dag, a, b, ty| dag.add_node(RiscOp::Add, vec![a, b], ty.clone()));
        let (a, n) = finite_diff(&dag, out, x, "x", &[("y", 3.0)], 2.0, 1e-5);
        assert_grad_close(a, n);
        assert!((a - 1.0).abs() < 1e-6); // d(x+y)/dx = 1
    }

    #[test]
    fn grad_mul() {
        let (dag, x, _y, out) =
            build_binary_dag(|dag, a, b, ty| dag.add_node(RiscOp::Mul, vec![a, b], ty.clone()));
        let (a, n) = finite_diff(&dag, out, x, "x", &[("y", 3.0)], 2.0, 1e-5);
        assert_grad_close(a, n);
        assert!((a - 3.0).abs() < 1e-6); // d(x*y)/dx = y = 3
    }

    #[test]
    fn grad_neg() {
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Neg, vec![a], ty.clone()));
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.0, 1e-5);
        assert_grad_close(a, n);
        assert!((a - (-1.0)).abs() < 1e-6); // d(-x)/dx = -1
    }

    #[test]
    fn grad_exp() {
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Exp, vec![a], ty.clone()));
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 1.0, 1e-5);
        assert_grad_close(a, n);
        let expected = 1.0_f64.exp();
        assert!((a - expected).abs() < 1e-4); // d(exp(x))/dx = exp(x)
    }

    #[test]
    fn grad_log() {
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Log, vec![a], ty.clone()));
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.0, 1e-5);
        assert_grad_close(a, n);
        assert!((a - 0.5).abs() < 1e-4); // d(log(x))/dx = 1/x = 0.5
    }

    #[test]
    fn grad_sin() {
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Sin, vec![a], ty.clone()));
        let x0 = 1.0;
        let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
        assert_grad_close(a, n);
        let expected = x0.cos();
        assert!((a - expected).abs() < 1e-4);
    }

    #[test]
    fn grad_sqrt() {
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Sqrt, vec![a], ty.clone()));
        let x0 = 4.0;
        let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
        assert_grad_close(a, n);
        let expected = 1.0 / (2.0 * x0.sqrt()); // = 0.25
        assert!((a - expected).abs() < 1e-4);
    }

    #[test]
    fn grad_x_squared() {
        // f(x) = x * x, df/dx = 2x (tests accumulation: x used twice)
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], scalar_f32());
        let x_sq = dag.add_node(RiscOp::Mul, vec![x, x], scalar_f32());

        let (a, n) = finite_diff(&dag, x_sq, x, "x", &[], 3.0, 1e-5);
        assert_grad_close(a, n);
        assert!((a - 6.0).abs() < 1e-4); // 2 * 3 = 6
    }

    #[test]
    fn grad_relu_positive() {
        // relu(x) = max(x, 0), d/dx = 1 when x > 0
        let (dag, x, out) = build_unary_dag(|dag, a, ty| {
            let zero = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], ty.clone());
            dag.add_node(RiscOp::MaxElem, vec![a, zero], ty.clone())
        });
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.0, 1e-5);
        assert_grad_close(a, n);
        assert!((a - 1.0).abs() < 1e-4);
    }

    #[test]
    fn grad_relu_negative() {
        // relu(x) = max(x, 0), d/dx = 0 when x < 0
        let (dag, x, out) = build_unary_dag(|dag, a, ty| {
            let zero = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], ty.clone());
            dag.add_node(RiscOp::MaxElem, vec![a, zero], ty.clone())
        });
        let (a, n) = finite_diff(&dag, out, x, "x", &[], -2.0, 1e-5);
        assert_grad_close(a, n);
        assert!(a.abs() < 1e-4);
    }

    #[test]
    fn grad_chain_exp_neg() {
        // f(x) = exp(-x), df/dx = -exp(-x)
        let (dag, x, out) = build_unary_dag(|dag, a, ty| {
            let neg = dag.add_node(RiscOp::Neg, vec![a], ty.clone());
            dag.add_node(RiscOp::Exp, vec![neg], ty.clone())
        });
        let x0 = 1.0;
        let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
        assert_grad_close(a, n);
        let expected = -(-x0).exp();
        assert!((a - expected).abs() < 1e-4);
    }

    #[test]
    fn grad_multi_input() {
        // f(x, y, z) = x*y + z
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], scalar_f32());
        let y = dag.add_node(RiscOp::Load { name: "y".into() }, vec![], scalar_f32());
        let z = dag.add_node(RiscOp::Load { name: "z".into() }, vec![], scalar_f32());
        let xy = dag.add_node(RiscOp::Mul, vec![x, y], scalar_f32());
        let out = dag.add_node(RiscOp::Add, vec![xy, z], scalar_f32());

        let grad_result = grad_dag(&dag, out, &[x, y, z]);
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), 2.0);
        inputs.insert("y".to_string(), 3.0);
        inputs.insert("z".to_string(), 5.0);
        let vals = eval_scalar(&grad_result.dag, &inputs);

        let dx = vals[&grad_result.grad_nodes[&x]];
        let dy = vals[&grad_result.grad_nodes[&y]];
        let dz = vals[&grad_result.grad_nodes[&z]];
        assert!((dx - 3.0).abs() < 1e-6); // d/dx(x*y + z) = y = 3
        assert!((dy - 2.0).abs() < 1e-6); // d/dy = x = 2
        assert!((dz - 1.0).abs() < 1e-6); // d/dz = 1
    }

    #[test]
    fn grad_accumulation_3x() {
        // f(x) = x + x + x, df/dx = 3
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], scalar_f32());
        let sum1 = dag.add_node(RiscOp::Add, vec![x, x], scalar_f32());
        let out = dag.add_node(RiscOp::Add, vec![sum1, x], scalar_f32());

        let grad_result = grad_dag(&dag, out, &[x]);
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), 7.0);
        let vals = eval_scalar(&grad_result.dag, &inputs);
        let dx = vals[&grad_result.grad_nodes[&x]];
        assert!((dx - 3.0).abs() < 1e-6);
    }

    #[test]
    fn grad_cmplt_zero() {
        let (dag, x, _y, out) =
            build_binary_dag(|dag, a, b, ty| dag.add_node(RiscOp::CmpLt, vec![a, b], ty.clone()));
        let grad_result = grad_dag(&dag, out, &[x]);
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), 1.0);
        inputs.insert("y".to_string(), 2.0);
        let vals = eval_scalar(&grad_result.dag, &inputs);
        let dx = vals[&grad_result.grad_nodes[&x]];
        assert!(dx.abs() < 1e-10);
    }

    #[test]
    fn grad_const_no_gradient() {
        let mut dag = Dag::new();
        let c = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32());
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], scalar_f32());
        let out = dag.add_node(RiscOp::Mul, vec![x, c], scalar_f32());

        let grad_result = grad_dag(&dag, out, &[x]);
        assert!(!grad_result.grad_nodes.contains_key(&c));
    }

    #[test]
    fn grad_sum_expand() {
        // f(x) = sum(x, axis=0) for a 3-element vector, df/dx_i = 1
        use crate::dag::DimInfo;
        use crate::eval::{TensorValue, eval_tensor};

        let vec3_ty = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: chelis_types::types::Prim::F32,
        };
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec3_ty.clone());
        let out = dag.add_node(RiscOp::Sum { axis: 0 }, vec![x], scalar_f32());

        let grad_result = grad_dag(&dag, out, &[x]);
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
        );
        let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
        let grad = &vals[&grad_result.grad_nodes[&x]];
        assert_eq!(grad.data, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn grad_second_order() {
        // f(x) = x*x, f'(x) = 2x, f''(x) = 2
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], scalar_f32());
        let x_sq = dag.add_node(RiscOp::Mul, vec![x, x], scalar_f32());

        // First derivative
        let first = grad_dag(&dag, x_sq, &[x]);
        let dx_node = first.grad_nodes[&x];

        // Second derivative: differentiate the first derivative DAG
        let second = grad_dag(&first.dag, dx_node, &[x]);
        let ddx_node = second.grad_nodes[&x];

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), 5.0);
        let vals = eval_scalar(&second.dag, &inputs);
        let ddx = vals[&ddx_node];
        assert!((ddx - 2.0).abs() < 1e-4); // d^2(x^2)/dx^2 = 2
    }

    #[test]
    fn grad_store_passthrough() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], scalar_f32());
        let exp_x = dag.add_node(RiscOp::Exp, vec![x], scalar_f32());
        let out = dag.add_node(
            RiscOp::Store { name: "out".into() },
            vec![exp_x],
            scalar_f32(),
        );

        let (a, n) = finite_diff(&dag, out, x, "x", &[], 1.0, 1e-5);
        assert_grad_close(a, n);
    }

    #[test]
    fn grad_cast_passthrough() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], scalar_f32());
        let f64_ty = TensorType {
            dims: vec![],
            precision: chelis_types::types::Prim::F64,
        };
        let casted = dag.add_node(
            RiscOp::Cast {
                new_precision: chelis_types::types::Prim::F64,
            },
            vec![x],
            f64_ty,
        );
        let out = dag.add_node(RiscOp::Exp, vec![casted], scalar_f32());

        let grad_result = grad_dag(&dag, out, &[x]);
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), 1.0);
        let vals = eval_scalar(&grad_result.dag, &inputs);
        let dx = vals[&grad_result.grad_nodes[&x]];
        let expected = 1.0_f64.exp();
        assert!((dx - expected).abs() < 1e-4);
    }

    #[test]
    fn grad_reshape_roundtrip() {
        use crate::eval::{TensorValue, eval_tensor};

        let vec6_ty = TensorType {
            dims: vec![DimInfo::Lit(6)],
            precision: chelis_types::types::Prim::F32,
        };
        let mat23_ty = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            precision: chelis_types::types::Prim::F32,
        };

        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec6_ty.clone());
        let reshaped = dag.add_node(
            RiscOp::Reshape {
                new_shape: mat23_ty.dims.clone(),
            },
            vec![x],
            mat23_ty.clone(),
        );
        // Sum all elements to get a scalar
        let sum0 = dag.add_node(
            RiscOp::Sum { axis: 0 },
            vec![reshaped],
            TensorType {
                dims: vec![DimInfo::Lit(3)],
                precision: chelis_types::types::Prim::F32,
            },
        );
        let out = dag.add_node(RiscOp::Sum { axis: 0 }, vec![sum0], scalar_f32());

        let grad_result = grad_dag(&dag, out, &[x]);
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );
        let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
        let grad = &vals[&grad_result.grad_nodes[&x]];
        // d(sum(reshape(x)))/dx = ones
        assert_eq!(grad.data, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(grad.shape, vec![6]);
    }

    #[test]
    fn grad_permute_transpose() {
        use crate::eval::{TensorValue, eval_tensor};

        let mat23_ty = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            precision: chelis_types::types::Prim::F32,
        };
        let mat32_ty = TensorType {
            dims: vec![DimInfo::Lit(3), DimInfo::Lit(2)],
            precision: chelis_types::types::Prim::F32,
        };

        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], mat23_ty.clone());
        let transposed = dag.add_node(
            RiscOp::Permute { axes: vec![1, 0] },
            vec![x],
            mat32_ty.clone(),
        );
        // Sum all elements for a scalar output
        let sum0 = dag.add_node(
            RiscOp::Sum { axis: 0 },
            vec![transposed],
            TensorType {
                dims: vec![DimInfo::Lit(2)],
                precision: chelis_types::types::Prim::F32,
            },
        );
        let out = dag.add_node(RiscOp::Sum { axis: 0 }, vec![sum0], scalar_f32());

        let grad_result = grad_dag(&dag, out, &[x]);
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );
        let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
        let grad = &vals[&grad_result.grad_nodes[&x]];
        // d(sum(permute(x)))/dx = ones, shape should be 2x3
        assert_eq!(grad.data, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(grad.shape, vec![2, 3]);
    }

    #[test]
    fn grad_expand_sum_roundtrip() {
        use crate::eval::{TensorValue, eval_tensor};

        let vec3_ty = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: chelis_types::types::Prim::F32,
        };
        let mat23_ty = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            precision: chelis_types::types::Prim::F32,
        };

        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec3_ty.clone());
        let expanded = dag.add_node(
            RiscOp::Expand { axis: 0, size: 2 },
            vec![x],
            mat23_ty.clone(),
        );
        // Sum back to scalar
        let sum0 = dag.add_node(RiscOp::Sum { axis: 0 }, vec![expanded], vec3_ty.clone());
        let out = dag.add_node(RiscOp::Sum { axis: 0 }, vec![sum0], scalar_f32());

        let grad_result = grad_dag(&dag, out, &[x]);
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
        );
        let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
        let grad = &vals[&grad_result.grad_nodes[&x]];
        // expand by 2 then sum -> each element counted twice -> gradient = 2
        assert_eq!(grad.data, vec![2.0, 2.0, 2.0]);
        assert_eq!(grad.shape, vec![3]);
    }

    #[test]
    fn grad_mul_by_const() {
        // f(x) = 5 * x, df/dx = 5
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], scalar_f32());
        let five = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32());
        let out = dag.add_node(RiscOp::Mul, vec![five, x], scalar_f32());

        let (a, n) = finite_diff(&dag, out, x, "x", &[], 3.0, 1e-5);
        assert_grad_close(a, n);
        assert!((a - 5.0).abs() < 1e-4);
    }

    #[test]
    fn grad_log_chain() {
        // f(x) = log(x^2) = 2*log(x), df/dx = 2/x
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], scalar_f32());
        let x_sq = dag.add_node(RiscOp::Mul, vec![x, x], scalar_f32());
        let out = dag.add_node(RiscOp::Log, vec![x_sq], scalar_f32());

        let x0 = 3.0;
        let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
        assert_grad_close(a, n);
        let expected = 2.0 / x0;
        assert!((a - expected).abs() < 1e-4);
    }

    #[test]
    fn grad_sin_chain() {
        // f(x) = sin(x^2), df/dx = 2x * cos(x^2)
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], scalar_f32());
        let x_sq = dag.add_node(RiscOp::Mul, vec![x, x], scalar_f32());
        let out = dag.add_node(RiscOp::Sin, vec![x_sq], scalar_f32());

        let x0 = 1.5;
        let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
        assert_grad_close(a, n);
    }

    #[test]
    fn grad_max_elem_symmetric() {
        // f(x, y) = max(x, y), test that d/dy = 1 when y > x
        let (dag, _x, y, out) =
            build_binary_dag(|dag, a, b, ty| dag.add_node(RiscOp::MaxElem, vec![a, b], ty.clone()));
        let (a, n) = finite_diff(&dag, out, y, "y", &[("x", 1.0)], 5.0, 1e-5);
        assert_grad_close(a, n);
        assert!((a - 1.0).abs() < 1e-4); // y > x, so d/dy = 1
    }

    #[test]
    fn grad_no_wrt_returns_empty() {
        let (dag, _x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Exp, vec![a], ty.clone()));
        // Ask for gradient wrt a node that doesn't exist
        let grad_result = grad_dag(&dag, out, &[NodeId(999)]);
        assert!(grad_result.grad_nodes.is_empty());
    }

    #[test]
    fn grad_inverse_permutation() {
        assert_eq!(inverse_permutation(&[2, 0, 1]), vec![1, 2, 0]);
        assert_eq!(inverse_permutation(&[0, 1]), vec![0, 1]);
        assert_eq!(inverse_permutation(&[1, 0]), vec![1, 0]);
    }
}
