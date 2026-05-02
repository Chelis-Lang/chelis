//! Tiled matmul detection + emission for the Metal backend.
//!
//! Mirrors `chelis_backend_hip::blas` so the Metal backend specializes the
//! same lowered matmul subgraph (`expand + mul + sum(axis=1)`) the C and
//! HIP backends already recognize. The emit path uses
//! `kernels::matmul_tiled_kernel` (16x16 tiled MSL kernel) and dispatches
//! via `chelis_metal_launch2d`.

use chelis_ir::dag::{Dag, DagNode, DimInfo, NodeId, RiscOp};

/// Information about a detected matmul pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct MatmulInfo {
    /// NodeId of the left matrix operand (rank-2).
    pub a: NodeId,
    /// NodeId of the right matrix operand (rank-2).
    pub b: NodeId,
    /// Rows of the output (M dimension).
    pub m: usize,
    /// Columns of the output (N dimension).
    pub n: usize,
    /// Inner/contraction dimension (K dimension).
    pub k: usize,
}

/// Try to detect a matmul pattern rooted at the given Sum node.
///
/// Returns `Some(info)` if the Sum is the head of an `expand + mul +
/// sum(axis=1)` matmul subgraph with statically-known rank-2 f32 operands;
/// otherwise `None`.
pub fn detect_matmul_pattern(dag: &Dag, sum_id: NodeId) -> Option<MatmulInfo> {
    let sum_node = dag.get(sum_id)?;
    let axis = match &sum_node.op {
        RiscOp::Sum { axis } => *axis,
        _ => return None,
    };

    if sum_node.inputs.len() != 1 {
        return None;
    }
    let mul_node = dag.get(sum_node.inputs[0])?;
    if !matches!(mul_node.op, RiscOp::Mul) || mul_node.inputs.len() != 2 {
        return None;
    }

    let expand_a = dag.get(mul_node.inputs[0])?;
    let expand_b = dag.get(mul_node.inputs[1])?;

    match (&expand_a.op, &expand_b.op) {
        (RiscOp::Expand { .. }, RiscOp::Expand { .. }) => {
            if expand_a.inputs.len() != 1 || expand_b.inputs.len() != 1 {
                return None;
            }
            let a_id = expand_a.inputs[0];
            let b_id = expand_b.inputs[0];
            let a_node = dag.get(a_id)?;
            let b_node = dag.get(b_id)?;
            extract_matmul_dims(a_node, b_node, axis, a_id, b_id)
        }
        _ => None,
    }
}

fn extract_matmul_dims(
    a_node: &DagNode,
    b_node: &DagNode,
    sum_axis: usize,
    a_id: NodeId,
    b_id: NodeId,
) -> Option<MatmulInfo> {
    if a_node.output_type.dims.len() != 2 || b_node.output_type.dims.len() != 2 {
        return None;
    }
    if a_node.output_type.precision != chelis_types::types::Prim::F32
        || b_node.output_type.precision != chelis_types::types::Prim::F32
    {
        return None;
    }

    let dim_size = |d: &DimInfo| -> Option<usize> {
        match d {
            DimInfo::Lit(n) => Some(*n),
            DimInfo::Named(_, Some(n)) => Some(*n),
            _ => None,
        }
    };

    let m = dim_size(&a_node.output_type.dims[0])?;
    let k_a = dim_size(&a_node.output_type.dims[1])?;
    let k_b = dim_size(&b_node.output_type.dims[0])?;
    let n = dim_size(&b_node.output_type.dims[1])?;

    if k_a != k_b || sum_axis != 1 {
        return None;
    }

    Some(MatmulInfo {
        a: a_id,
        b: b_id,
        m,
        n,
        k: k_a,
    })
}

/// Find every matmul subgraph in the DAG.
///
/// Returns `(sum_node_id, info)` for each detected pattern, ordered by
/// the Sum node's position in the DAG.
pub fn find_all_matmuls(dag: &Dag) -> Vec<(NodeId, MatmulInfo)> {
    let mut found = Vec::new();
    for node in dag.nodes() {
        if let Some(info) = detect_matmul_pattern(dag, node.id) {
            found.push((node.id, info));
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_ir::dag::{Dag, DimInfo, TensorType};
    use chelis_types::types::Prim;

    fn mat_f32(r: usize, c: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
            precision: Prim::F32,
        }
    }

    fn tensor3_f32(a: usize, b: usize, c: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(a), DimInfo::Lit(b), DimInfo::Lit(c)],
            precision: Prim::F32,
        }
    }

    #[test]
    fn detects_canonical_matmul() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::Load { name: "a".into() },
            vec![],
            mat_f32(2, 3),
            None,
        );
        let b = dag.add_node(
            RiscOp::Load { name: "b".into() },
            vec![],
            mat_f32(3, 4),
            None,
        );
        let ea = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: chelis_ir::dag::DimExpr::Concrete(4),
            },
            vec![a],
            tensor3_f32(2, 3, 4),
            None,
        );
        let eb = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::DimExpr::Concrete(2),
            },
            vec![b],
            tensor3_f32(2, 3, 4),
            None,
        );
        let mul = dag.add_node(RiscOp::Mul, vec![ea, eb], tensor3_f32(2, 3, 4), None);
        let sum = dag.add_node(RiscOp::Sum { axis: 1 }, vec![mul], mat_f32(2, 4), None);

        let info = detect_matmul_pattern(&dag, sum).expect("matmul should be detected");
        assert_eq!(info.a, a);
        assert_eq!(info.b, b);
        assert_eq!(info.m, 2);
        assert_eq!(info.n, 4);
        assert_eq!(info.k, 3);
    }

    #[test]
    fn rejects_wrong_sum_axis() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::Load { name: "a".into() },
            vec![],
            mat_f32(2, 3),
            None,
        );
        let b = dag.add_node(
            RiscOp::Load { name: "b".into() },
            vec![],
            mat_f32(3, 4),
            None,
        );
        let ea = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: chelis_ir::dag::DimExpr::Concrete(4),
            },
            vec![a],
            tensor3_f32(2, 3, 4),
            None,
        );
        let eb = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::DimExpr::Concrete(2),
            },
            vec![b],
            tensor3_f32(2, 3, 4),
            None,
        );
        let mul = dag.add_node(RiscOp::Mul, vec![ea, eb], tensor3_f32(2, 3, 4), None);
        // sum axis=0 is the wrong axis; matmul detector should bail.
        let sum = dag.add_node(RiscOp::Sum { axis: 0 }, vec![mul], mat_f32(3, 4), None);

        assert!(detect_matmul_pattern(&dag, sum).is_none());
    }

    #[test]
    fn rejects_mismatched_inner_dims() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::Load { name: "a".into() },
            vec![],
            mat_f32(2, 3),
            None,
        );
        let b = dag.add_node(
            RiscOp::Load { name: "b".into() },
            vec![],
            mat_f32(5, 4),
            None,
        );
        let ea = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: chelis_ir::dag::DimExpr::Concrete(4),
            },
            vec![a],
            tensor3_f32(2, 3, 4),
            None,
        );
        let eb = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::DimExpr::Concrete(2),
            },
            vec![b],
            tensor3_f32(2, 5, 4),
            None,
        );
        let mul = dag.add_node(RiscOp::Mul, vec![ea, eb], tensor3_f32(2, 3, 4), None);
        let sum = dag.add_node(RiscOp::Sum { axis: 1 }, vec![mul], mat_f32(2, 4), None);

        assert!(detect_matmul_pattern(&dag, sum).is_none());
    }
}
