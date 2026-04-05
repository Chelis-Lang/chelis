//! Tier 2 decomposition helpers.
//!
//! These functions decompose Tier 2 (derived) operations into Tier 1 RISC DAG nodes.

use chelis_types::types::Prim;

use crate::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};

/// `sub(a, b)` = `add(a, neg(b))`
pub fn lower_sub(dag: &mut Dag, a: NodeId, b: NodeId, ty: &TensorType) -> NodeId {
    let neg_b = dag.add_node(RiscOp::Neg, vec![b], ty.clone());
    dag.add_node(RiscOp::Add, vec![a, neg_b], ty.clone())
}

/// `relu(x)` = `max_elem(x, const(0))`
pub fn lower_relu(dag: &mut Dag, x: NodeId, ty: &TensorType) -> NodeId {
    let zero = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], ty.clone());
    dag.add_node(RiscOp::MaxElem, vec![x, zero], ty.clone())
}

/// `sigmoid(x)` = `1 / (1 + exp(-x))`
///
/// Lowered as: `exp(neg(log(add(const(1), exp(neg(x))))))`
pub fn lower_sigmoid(dag: &mut Dag, x: NodeId, ty: &TensorType) -> NodeId {
    let neg_x = dag.add_node(RiscOp::Neg, vec![x], ty.clone());
    let exp_neg = dag.add_node(RiscOp::Exp, vec![neg_x], ty.clone());
    let one = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], ty.clone());
    let sum = dag.add_node(RiscOp::Add, vec![one, exp_neg], ty.clone());
    // recip(sum) = exp(neg(log(sum)))
    let log_sum = dag.add_node(RiscOp::Log, vec![sum], ty.clone());
    let neg_log = dag.add_node(RiscOp::Neg, vec![log_sum], ty.clone());
    dag.add_node(RiscOp::Exp, vec![neg_log], ty.clone())
}

/// `div(a, b)` = `mul(a, recip(b))` where `recip(b) = exp(neg(log(b)))`
pub fn lower_div(dag: &mut Dag, a: NodeId, b: NodeId, ty: &TensorType) -> NodeId {
    let log_b = dag.add_node(RiscOp::Log, vec![b], ty.clone());
    let neg_log = dag.add_node(RiscOp::Neg, vec![log_b], ty.clone());
    let recip_b = dag.add_node(RiscOp::Exp, vec![neg_log], ty.clone());
    dag.add_node(RiscOp::Mul, vec![a, recip_b], ty.clone())
}

/// H1: `gt(a, b)` = `cmplt(b, a)` (swap args)
pub fn lower_gt(dag: &mut Dag, a: NodeId, b: NodeId, ty: &TensorType) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    dag.add_node(RiscOp::CmpLt, vec![b, a], bool_ty)
}

/// H1: `gte(a, b)` = `neg(cmplt(a, b))` — not (a < b)
/// Per spec §3.2: gte uses neg on bool (0/1 convention).
pub fn lower_gte(dag: &mut Dag, a: NodeId, b: NodeId, ty: &TensorType) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    let lt = dag.add_node(RiscOp::CmpLt, vec![a, b], bool_ty.clone());
    // not(lt): cmplt(lt, const(1)) — if lt==0 then 0<1=true, if lt==1 then 1<1=false
    let one = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], bool_ty.clone());
    dag.add_node(RiscOp::CmpLt, vec![lt, one], bool_ty)
}

/// H1: `lte(a, b)` = `neg(cmplt(b, a))` — not (b < a)
pub fn lower_lte(dag: &mut Dag, a: NodeId, b: NodeId, ty: &TensorType) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    let lt = dag.add_node(RiscOp::CmpLt, vec![b, a], bool_ty.clone());
    let one = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], bool_ty.clone());
    dag.add_node(RiscOp::CmpLt, vec![lt, one], bool_ty)
}

/// H1: `eq(a, b)` = not(or(cmplt(a,b), cmplt(b,a)))
/// `or` on bools = `max_elem`, `not` = `cmplt(x, const(1))`
pub fn lower_eq(dag: &mut Dag, a: NodeId, b: NodeId, ty: &TensorType) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    let lt_ab = dag.add_node(RiscOp::CmpLt, vec![a, b], bool_ty.clone());
    let lt_ba = dag.add_node(RiscOp::CmpLt, vec![b, a], bool_ty.clone());
    let or = dag.add_node(RiscOp::MaxElem, vec![lt_ab, lt_ba], bool_ty.clone());
    let one = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], bool_ty.clone());
    dag.add_node(RiscOp::CmpLt, vec![or, one], bool_ty)
}

/// H1: `neq(a, b)` = `or(cmplt(a,b), cmplt(b,a))`
pub fn lower_neq(dag: &mut Dag, a: NodeId, b: NodeId, ty: &TensorType) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    let lt_ab = dag.add_node(RiscOp::CmpLt, vec![a, b], bool_ty.clone());
    let lt_ba = dag.add_node(RiscOp::CmpLt, vec![b, a], bool_ty.clone());
    dag.add_node(RiscOp::MaxElem, vec![lt_ab, lt_ba], bool_ty)
}

/// H1: `min_elem(a, b)` = `neg(max_elem(neg(a), neg(b)))`
pub fn lower_min_elem(dag: &mut Dag, a: NodeId, b: NodeId, ty: &TensorType) -> NodeId {
    let neg_a = dag.add_node(RiscOp::Neg, vec![a], ty.clone());
    let neg_b = dag.add_node(RiscOp::Neg, vec![b], ty.clone());
    let max = dag.add_node(RiscOp::MaxElem, vec![neg_a, neg_b], ty.clone());
    dag.add_node(RiscOp::Neg, vec![max], ty.clone())
}

/// H2: `and(a, b)` on bools = `mul(a, b)`
pub fn lower_and(dag: &mut Dag, a: NodeId, b: NodeId, ty: &TensorType) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    dag.add_node(RiscOp::Mul, vec![a, b], bool_ty)
}

/// H2: `or(a, b)` on bools = `max_elem(a, b)`
pub fn lower_or(dag: &mut Dag, a: NodeId, b: NodeId, ty: &TensorType) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    dag.add_node(RiscOp::MaxElem, vec![a, b], bool_ty)
}

/// H2: `not(a)` on bools = `cmplt(a, const(1))` — flips 0->1, 1->0
pub fn lower_not(dag: &mut Dag, a: NodeId, ty: &TensorType) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    let one = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], bool_ty.clone());
    dag.add_node(RiscOp::CmpLt, vec![a, one], bool_ty)
}

// ---------------------------------------------------------------------------
// Helper functions for type manipulation
// ---------------------------------------------------------------------------

/// Produce a type with the given axis removed (for reductions).
pub fn reduced_type(ty: &TensorType, axis: usize) -> TensorType {
    let mut dims = ty.dims.clone();
    if axis < dims.len() {
        dims.remove(axis);
    }
    TensorType {
        dims,
        precision: ty.precision,
    }
}

/// Extract a concrete dimension size from a tensor type at the given axis.
pub fn dim_size(ty: &TensorType, axis: usize) -> Option<usize> {
    ty.dims.get(axis).and_then(|d| match d {
        DimInfo::Lit(n) => Some(*n),
        DimInfo::Named(_, Some(n)) => Some(*n),
        _ => None,
    })
}

/// A scalar type with no dimensions and the given precision.
pub fn scalar_type(precision: Prim) -> TensorType {
    TensorType {
        dims: vec![],
        precision,
    }
}

// ---------------------------------------------------------------------------
// Tier 2 higher-level decompositions (spec §3.4, §4.1–4.2)
// ---------------------------------------------------------------------------

/// matmul(A: [i, j], B: [j, k]) -> [i, k]
///
/// Lowering (spec §4.1):
///   1. Expand A from [i, j] to [i, j, k] by adding a trailing dim
///   2. Expand B from [j, k] to [i, j, k] by adding a leading dim
///   3. Mul the expanded tensors -> [i, j, k]
///   4. Sum over axis 1 (the j dimension) -> [i, k]
///
/// Phase 0 limitation: if dimension sizes are unknown, placeholder size 1 is used.
pub fn lower_matmul(
    dag: &mut Dag,
    a: NodeId,
    b: NodeId,
    a_ty: &TensorType,
    b_ty: &TensorType,
) -> NodeId {
    // Extract dimension sizes: A is [i, j], B is [j, k].
    let i_size = dim_size(a_ty, 0).unwrap_or(1);
    let j_size = dim_size(a_ty, 1).or_else(|| dim_size(b_ty, 0)).unwrap_or(1);
    let k_size = dim_size(b_ty, 1).unwrap_or(1);

    // The intermediate expanded type is [i, j, k].
    let expanded_ty = TensorType {
        dims: vec![
            DimInfo::Lit(i_size),
            DimInfo::Lit(j_size),
            DimInfo::Lit(k_size),
        ],
        precision: a_ty.precision,
    };

    // The result type is [i, k].
    let result_ty = TensorType {
        dims: vec![DimInfo::Lit(i_size), DimInfo::Lit(k_size)],
        precision: a_ty.precision,
    };

    // 1. Expand A: add dim for k at axis 2 -> [i, j, k]
    let a_expanded = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: k_size,
        },
        vec![a],
        expanded_ty.clone(),
    );

    // 2. Expand B: add dim for i at axis 0 -> [i, j, k]
    let b_expanded = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: i_size,
        },
        vec![b],
        expanded_ty.clone(),
    );

    // 3. Elementwise multiply -> [i, j, k]
    let product = dag.add_node(RiscOp::Mul, vec![a_expanded, b_expanded], expanded_ty);

    // 4. Sum over axis 1 (j) -> [i, k]
    dag.add_node(RiscOp::Sum { axis: 1 }, vec![product], result_ty)
}

/// softmax(x, axis) = exp(x - max_reduce(x, axis)) / sum(exp(x - max_reduce(x, axis)), axis)
///
/// Lowering (spec §4.2): numerically stable softmax via max subtraction.
pub fn lower_softmax(dag: &mut Dag, x: NodeId, axis: usize, ty: &TensorType) -> NodeId {
    let red_ty = reduced_type(ty, axis);
    let size = dim_size(ty, axis).unwrap_or(1);

    // 1. max_reduce(x, axis)
    let max_val = dag.add_node(RiscOp::MaxReduce { axis }, vec![x], red_ty.clone());

    // 2. expand max back to original shape
    let max_expanded = dag.add_node(RiscOp::Expand { axis, size }, vec![max_val], ty.clone());

    // 3. shifted = x - max (numerical stability)
    let shifted = lower_sub(dag, x, max_expanded, ty);

    // 4. exp(shifted)
    let exp_shifted = dag.add_node(RiscOp::Exp, vec![shifted], ty.clone());

    // 5. sum(exp, axis)
    let sum_exp = dag.add_node(RiscOp::Sum { axis }, vec![exp_shifted], red_ty);

    // 6. expand sum back to original shape
    let sum_expanded = dag.add_node(RiscOp::Expand { axis, size }, vec![sum_exp], ty.clone());

    // 7. exp / sum
    lower_div(dag, exp_shifted, sum_expanded, ty)
}

/// mean(x, axis) = sum(x, axis) / dim_size
///
/// Lowering (spec §3.4): sum then divide by the axis size.
pub fn lower_mean(dag: &mut Dag, x: NodeId, axis: usize, ty: &TensorType) -> NodeId {
    let red_ty = reduced_type(ty, axis);
    let dim_size_val = dim_size(ty, axis).unwrap_or(1) as f64;

    // sum(x, axis)
    let sum_node = dag.add_node(RiscOp::Sum { axis }, vec![x], red_ty.clone());

    // const(dim_size)
    let size_const = dag.add_node(
        RiscOp::Const {
            value: dim_size_val,
        },
        vec![],
        scalar_type(ty.precision),
    );

    // sum / dim_size
    lower_div(dag, sum_node, size_const, &red_ty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::RiscOp;
    use crate::verify;

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    #[test]
    fn sub_produces_add_neg() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let result = lower_sub(&mut dag, a, b, &scalar_f32());
        assert!(verify::verify(&dag).is_empty());

        // Should have: Const(5), Const(3), Neg, Add
        assert_eq!(dag.len(), 4);
        let result_node = dag.get(result).unwrap();
        assert_eq!(result_node.op, RiscOp::Add);
        // The Neg node is one of the inputs to Add.
        let neg_id = result_node.inputs[1];
        assert_eq!(dag.get(neg_id).unwrap().op, RiscOp::Neg);
    }

    #[test]
    fn relu_produces_max_elem_const_zero() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: -1.0 }, vec![], scalar_f32());
        let result = lower_relu(&mut dag, x, &scalar_f32());
        assert!(verify::verify(&dag).is_empty());

        // Const(-1), Const(0), MaxElem
        assert_eq!(dag.len(), 3);
        let result_node = dag.get(result).unwrap();
        assert_eq!(result_node.op, RiscOp::MaxElem);
        let zero_id = result_node.inputs[1];
        assert_eq!(dag.get(zero_id).unwrap().op, RiscOp::Const { value: 0.0 });
    }

    #[test]
    fn sigmoid_produces_correct_chain() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let result = lower_sigmoid(&mut dag, x, &scalar_f32());
        assert!(verify::verify(&dag).is_empty());

        // x, neg(x), exp(neg(x)), const(1), add, log, neg, exp
        assert_eq!(dag.len(), 8);
        let result_node = dag.get(result).unwrap();
        assert_eq!(result_node.op, RiscOp::Exp);
    }

    #[test]
    fn div_produces_mul_recip() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 6.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let result = lower_div(&mut dag, a, b, &scalar_f32());
        assert!(verify::verify(&dag).is_empty());

        // a, b, log(b), neg(log(b)), exp(neg(log(b))), mul(a, recip(b))
        assert_eq!(dag.len(), 6);
        let result_node = dag.get(result).unwrap();
        assert_eq!(result_node.op, RiscOp::Mul);
    }

    // --- H1: Tier 2 comparison decompositions ---

    fn scalar_bool() -> TensorType {
        TensorType {
            dims: vec![],
            precision: Prim::Bool,
        }
    }

    #[test]
    fn gt_swaps_args_to_cmplt() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let result = lower_gt(&mut dag, a, b, &scalar_f32());

        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
        // b, a order (swapped).
        assert_eq!(node.inputs, vec![b, a]);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn gte_produces_not_cmplt() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let result = lower_gte(&mut dag, a, b, &scalar_f32());

        // a, b, CmpLt(a,b), Const(1), CmpLt(lt, 1)
        assert_eq!(dag.len(), 5);
        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn lte_produces_not_cmplt_ba() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32());
        let result = lower_lte(&mut dag, a, b, &scalar_f32());

        assert_eq!(dag.len(), 5);
        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn eq_produces_not_or_cmplt() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let result = lower_eq(&mut dag, a, b, &scalar_f32());

        // a, b, CmpLt(a,b), CmpLt(b,a), MaxElem, Const(1), CmpLt(or, 1)
        assert_eq!(dag.len(), 7);
        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn neq_produces_or_cmplt() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32());
        let result = lower_neq(&mut dag, a, b, &scalar_f32());

        // a, b, CmpLt(a,b), CmpLt(b,a), MaxElem
        assert_eq!(dag.len(), 5);
        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::MaxElem);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn min_elem_produces_neg_max_neg() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let result = lower_min_elem(&mut dag, a, b, &scalar_f32());
        assert!(verify::verify(&dag).is_empty());

        // a, b, neg(a), neg(b), max(neg_a, neg_b), neg(max)
        assert_eq!(dag.len(), 6);
        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::Neg);
    }

    // --- H2: Boolean operators ---

    #[test]
    fn and_produces_mul() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_bool());
        let b = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_bool());
        let result = lower_and(&mut dag, a, b, &scalar_bool());

        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::Mul);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn or_produces_max_elem() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_bool());
        let b = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_bool());
        let result = lower_or(&mut dag, a, b, &scalar_bool());

        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::MaxElem);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn not_produces_cmplt_with_one() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_bool());
        let result = lower_not(&mut dag, a, &scalar_bool());

        // a, Const(1), CmpLt(a, 1)
        assert_eq!(dag.len(), 3);
        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    // --- Higher-level decompositions ---

    fn matrix_2x3() -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            precision: Prim::F32,
        }
    }

    fn matrix_3x4() -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        }
    }

    fn vec_5() -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(5)],
            precision: Prim::F32,
        }
    }

    #[test]
    fn matmul_produces_expand_mul_sum() {
        let mut dag = Dag::new();
        let a_ty = matrix_2x3();
        let b_ty = matrix_3x4();
        let a = dag.add_node(RiscOp::Load { name: "A".into() }, vec![], a_ty.clone());
        let b = dag.add_node(RiscOp::Load { name: "B".into() }, vec![], b_ty.clone());
        let result = lower_matmul(&mut dag, a, b, &a_ty, &b_ty);

        // A, B, Expand(A), Expand(B), Mul, Sum
        assert_eq!(dag.len(), 6);

        // Check Expand nodes exist
        let ops: Vec<_> = dag.nodes().iter().map(|n| &n.op).collect();
        let expand_count = ops
            .iter()
            .filter(|op| matches!(op, RiscOp::Expand { .. }))
            .count();
        assert_eq!(expand_count, 2, "expected 2 Expand nodes");
        assert!(
            ops.iter().any(|op| matches!(op, RiscOp::Mul)),
            "expected a Mul node"
        );
        assert!(
            ops.iter().any(|op| matches!(op, RiscOp::Sum { axis: 1 })),
            "expected a Sum{{axis:1}} node"
        );

        // Result type should be [2, 4]
        let result_node = dag.get(result).unwrap();
        assert_eq!(result_node.output_type.dims.len(), 2);
        assert_eq!(result_node.output_type.dims[0], DimInfo::Lit(2));
        assert_eq!(result_node.output_type.dims[1], DimInfo::Lit(4));
    }

    #[test]
    fn softmax_produces_maxreduce_sub_exp_sum_div() {
        let mut dag = Dag::new();
        let ty = vec_5();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone());
        let result = lower_softmax(&mut dag, x, 0, &ty);

        // Check the chain of ops produced.
        let ops: Vec<_> = dag.nodes().iter().map(|n| &n.op).collect();
        assert!(
            ops.iter()
                .any(|op| matches!(op, RiscOp::MaxReduce { axis: 0 })),
            "expected MaxReduce"
        );
        assert!(
            ops.iter().any(|op| matches!(op, RiscOp::Exp)),
            "expected Exp"
        );
        assert!(
            ops.iter().any(|op| matches!(op, RiscOp::Sum { axis: 0 })),
            "expected Sum"
        );
        // Sub produces Add+Neg, Div produces Log+Neg+Exp+Mul
        assert!(
            ops.iter().any(|op| matches!(op, RiscOp::Neg)),
            "expected Neg (from sub)"
        );

        // The final result should be a Mul (from div decomposition)
        let result_node = dag.get(result).unwrap();
        assert_eq!(result_node.op, RiscOp::Mul);
    }

    #[test]
    fn mean_produces_sum_div() {
        let mut dag = Dag::new();
        let ty = vec_5();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone());
        let result = lower_mean(&mut dag, x, 0, &ty);

        let ops: Vec<_> = dag.nodes().iter().map(|n| &n.op).collect();
        // Should have Sum
        assert!(
            ops.iter().any(|op| matches!(op, RiscOp::Sum { axis: 0 })),
            "expected Sum"
        );
        // Should have Const(5.0) for dim size
        assert!(
            ops.iter()
                .any(|op| matches!(op, RiscOp::Const { value } if *value == 5.0)),
            "expected Const(5.0) for dimension size"
        );
        // Result is Mul (from div decomposition: mul(sum, recip(5)))
        let result_node = dag.get(result).unwrap();
        assert_eq!(result_node.op, RiscOp::Mul);
    }
}
