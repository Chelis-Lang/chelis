//! DAG structural verification.

use crate::dag::{Dag, DimInfo, RiscOp};
use chelis_types::types::Prim;

/// Verify structural invariants of the DAG. Returns a list of error messages (empty = valid).
pub fn verify(dag: &Dag) -> Vec<String> {
    let mut errors = Vec::new();
    for node in dag.nodes() {
        // Check that inputs reference valid, earlier nodes.
        for &input_id in &node.inputs {
            if input_id.0 >= node.id.0 {
                errors.push(format!(
                    "node {} references non-earlier node {}",
                    node.id.0, input_id.0
                ));
            }
            if dag.get(input_id).is_none() {
                errors.push(format!(
                    "node {} references nonexistent node {}",
                    node.id.0, input_id.0
                ));
            }
        }

        // Check arity.
        let arity = node.inputs.len();
        match &node.op {
            RiscOp::Add | RiscOp::Mul | RiscOp::CmpLt | RiscOp::MaxElem => {
                if arity != 2 {
                    errors.push(format!(
                        "binary op at node {} has {} inputs (expected 2)",
                        node.id.0, arity
                    ));
                }

                // C1: precision consistency check for binary ops.
                if arity == 2
                    && let (Some(lhs), Some(rhs)) =
                        (dag.get(node.inputs[0]), dag.get(node.inputs[1]))
                {
                    if lhs.output_type.precision != rhs.output_type.precision {
                        errors.push(format!(
                            "binary op at node {} has mismatched precisions: {:?} vs {:?}",
                            node.id.0, lhs.output_type.precision, rhs.output_type.precision
                        ));
                    }

                    // C2: dimension matching for binary ops.
                    let l_dims = &lhs.output_type.dims;
                    let r_dims = &rhs.output_type.dims;
                    if l_dims.len() != r_dims.len() {
                        errors.push(format!(
                            "binary op at node {} has mismatched dimension count: {} vs {}",
                            node.id.0,
                            l_dims.len(),
                            r_dims.len()
                        ));
                    } else {
                        for (i, (ld, rd)) in l_dims.iter().zip(r_dims.iter()).enumerate() {
                            if !dims_compatible(ld, rd) {
                                errors.push(format!(
                                    "binary op at node {} has mismatched dimension at axis {}: {:?} vs {:?}",
                                    node.id.0, i, ld, rd
                                ));
                            }
                        }
                    }
                }
            }
            RiscOp::Neg
            | RiscOp::Exp
            | RiscOp::Log
            | RiscOp::Sin
            | RiscOp::Sqrt
            | RiscOp::Sum { .. }
            | RiscOp::MaxReduce { .. }
            | RiscOp::Reshape { .. }
            | RiscOp::Permute { .. }
            | RiscOp::Expand { .. }
            | RiscOp::Pad { .. }
            | RiscOp::Shrink { .. }
            | RiscOp::Stride { .. }
            | RiscOp::Cast { .. } => {
                if arity != 1 {
                    errors.push(format!(
                        "unary op at node {} has {} inputs (expected 1)",
                        node.id.0, arity
                    ));
                }
            }
            RiscOp::Const { .. } | RiscOp::Load { .. } => {
                if arity != 0 {
                    errors.push(format!(
                        "memory op at node {} has {} inputs (expected 0)",
                        node.id.0, arity
                    ));
                }
            }
        }

        // C3: validate reduction axis bounds.
        match &node.op {
            RiscOp::Sum { axis } | RiscOp::MaxReduce { axis } => {
                if arity == 1
                    && let Some(input) = dag.get(node.inputs[0])
                {
                    let ndims = input.output_type.dims.len();
                    if ndims == 0 || *axis >= ndims {
                        errors.push(format!(
                            "reduction op at node {} has axis {} but input has {} dimensions",
                            node.id.0, axis, ndims
                        ));
                    }
                }
            }
            _ => {}
        }

        // C4: transcendental ops require float precision.
        match &node.op {
            RiscOp::Exp | RiscOp::Log | RiscOp::Sin | RiscOp::Sqrt => {
                if arity == 1
                    && let Some(input) = dag.get(node.inputs[0])
                    && !input.output_type.precision.is_float()
                {
                    errors.push(format!(
                        "transcendental op {:?} at node {} requires float input, got {:?}",
                        node.op, node.id.0, input.output_type.precision
                    ));
                }
            }
            _ => {}
        }

        // C5: CmpLt output must be Bool.
        if matches!(&node.op, RiscOp::CmpLt) && node.output_type.precision != Prim::Bool {
            errors.push(format!(
                "cmplt at node {} has output precision {:?}, expected Bool",
                node.id.0, node.output_type.precision
            ));
        }
    }
    errors
}

/// Check if two dimension descriptors are compatible.
fn dims_compatible(a: &DimInfo, b: &DimInfo) -> bool {
    match (a, b) {
        (DimInfo::Lit(x), DimInfo::Lit(y)) => x == y,
        (DimInfo::Named(n1, s1), DimInfo::Named(n2, s2)) => {
            if n1 == n2 {
                // Same name: sizes must match if both known.
                match (s1, s2) {
                    (Some(a), Some(b)) => a == b,
                    _ => true,
                }
            } else {
                // Different names: compatible only if sizes match (if known).
                match (s1, s2) {
                    (Some(a), Some(b)) => a == b,
                    _ => true, // unknown sizes are assumed compatible
                }
            }
        }
        (DimInfo::Named(_, Some(s)), DimInfo::Lit(n))
        | (DimInfo::Lit(n), DimInfo::Named(_, Some(s))) => s == n,
        // Named with unknown size vs Lit: assume compatible (can't check).
        (DimInfo::Named(_, None), DimInfo::Lit(_)) | (DimInfo::Lit(_), DimInfo::Named(_, None)) => {
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{NodeId, RiscOp, TensorType};

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    #[test]
    fn valid_dag_no_errors() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
        assert!(verify(&dag).is_empty());
    }

    #[test]
    fn bad_input_reference() {
        let mut dag = Dag::new();
        // Manually create a node that references a future node (impossible via normal API,
        // but we can test via the replace_node backdoor or by constructing the scenario).
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        // Node 1 references itself (not earlier).
        dag.add_node(RiscOp::Neg, vec![NodeId(1)], scalar_f32());
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs[0].contains("non-earlier node"));
        let _ = a;
    }

    #[test]
    fn wrong_arity_binary() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        // Add with only 1 input.
        dag.add_node(RiscOp::Add, vec![a], scalar_f32());
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs[0].contains("binary op"));
    }

    #[test]
    fn wrong_arity_unary() {
        let mut dag = Dag::new();
        // Neg with 0 inputs.
        dag.add_node(RiscOp::Neg, vec![], scalar_f32());
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs[0].contains("unary op"));
    }

    #[test]
    fn wrong_arity_memory() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        // Const with an input (should have 0).
        dag.add_node(RiscOp::Const { value: 2.0 }, vec![a], scalar_f32());
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs[0].contains("memory op"));
    }

    // --- C1: precision consistency ---

    #[test]
    fn c1_mismatched_precision_binary_op() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b_ty = TensorType {
            dims: vec![],
            precision: Prim::F64,
        };
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], b_ty);
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs.iter().any(|e| e.contains("mismatched precisions")));
    }

    #[test]
    fn c1_matching_precision_ok() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Mul, vec![a, b], scalar_f32());
        assert!(verify(&dag).is_empty());
    }

    // --- C2: dimension matching ---

    #[test]
    fn c2_mismatched_dim_count() {
        let mut dag = Dag::new();
        let ty1 = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let ty2 = TensorType {
            dims: vec![DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], ty1);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], ty2);
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("mismatched dimension count"))
        );
    }

    #[test]
    fn c2_mismatched_dim_size() {
        let mut dag = Dag::new();
        let ty1 = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let ty2 = TensorType {
            dims: vec![DimInfo::Lit(5)],
            precision: Prim::F32,
        };
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], ty1);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], ty2);
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("mismatched dimension at axis"))
        );
    }

    // --- C3: reduction axis bounds ---

    #[test]
    fn c3_sum_axis_out_of_bounds() {
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], ty.clone());
        dag.add_node(RiscOp::Sum { axis: 5 }, vec![x], scalar_f32());
        let errs = verify(&dag);
        assert!(errs.iter().any(|e| e.contains("axis 5")));
    }

    #[test]
    fn c3_max_reduce_on_scalar() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::MaxReduce { axis: 0 }, vec![x], scalar_f32());
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("axis 0 but input has 0 dimensions"))
        );
    }

    #[test]
    fn c3_sum_valid_axis() {
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], ty.clone());
        dag.add_node(RiscOp::Sum { axis: 1 }, vec![x], scalar_f32());
        assert!(verify(&dag).is_empty());
    }

    // --- C4: transcendental float-only ---

    #[test]
    fn c4_exp_on_int_is_error() {
        let mut dag = Dag::new();
        let int_ty = TensorType {
            dims: vec![],
            precision: Prim::Int32,
        };
        let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], int_ty.clone());
        dag.add_node(RiscOp::Exp, vec![x], int_ty);
        let errs = verify(&dag);
        assert!(errs.iter().any(|e| e.contains("transcendental")));
    }

    #[test]
    fn c4_sqrt_on_float_ok() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: 4.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Sqrt, vec![x], scalar_f32());
        assert!(verify(&dag).is_empty());
    }

    // --- C5: CmpLt output must be Bool ---

    #[test]
    fn c5_cmplt_non_bool_output_is_error() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        // Wrong: output is F32 instead of Bool.
        dag.add_node(RiscOp::CmpLt, vec![a, b], scalar_f32());
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("cmplt") && e.contains("Bool"))
        );
    }

    #[test]
    fn c5_cmplt_bool_output_ok() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        let bool_ty = TensorType {
            dims: vec![],
            precision: Prim::Bool,
        };
        dag.add_node(RiscOp::CmpLt, vec![a, b], bool_ty);
        assert!(verify(&dag).is_empty());
    }
}
