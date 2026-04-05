//! Tier 2 decomposition helpers.
//!
//! These functions decompose Tier 2 (derived) operations into Tier 1 RISC DAG nodes.

use chelis_types::types::Prim;

use crate::dag::{Dag, NodeId, RiscOp, TensorType};

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
}
