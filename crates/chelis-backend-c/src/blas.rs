//! BLAS pattern matching for matmul subgraphs.
//!
//! Phase 0: detect matmul pattern (Sum whose input is Mul whose inputs are Expand).
//! Actual BLAS emission deferred to when OpenBLAS is available.

use chelis_ir::dag::{Dag, DagNode, DimInfo, NodeId, RiscOp};

/// Information about a detected matmul pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct MatmulInfo {
    /// NodeId of the left matrix operand.
    pub a: NodeId,
    /// NodeId of the right matrix operand.
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
/// The pattern is: Sum { axis } of Mul(Expand(A), Expand(B)).
/// Returns `Some(MatmulInfo)` if the pattern matches, `None` otherwise.
pub fn detect_matmul_pattern(dag: &Dag, sum_id: NodeId) -> Option<MatmulInfo> {
    let sum_node = dag.get(sum_id)?;

    // Must be a Sum node
    let axis = match &sum_node.op {
        RiscOp::Sum { axis } => *axis,
        _ => return None,
    };

    // Its input must be a Mul node
    if sum_node.inputs.len() != 1 {
        return None;
    }
    let mul_node = dag.get(sum_node.inputs[0])?;
    if !matches!(mul_node.op, RiscOp::Mul) {
        return None;
    }

    // Mul must have exactly two inputs, both Expand
    if mul_node.inputs.len() != 2 {
        return None;
    }
    let expand_a = dag.get(mul_node.inputs[0])?;
    let expand_b = dag.get(mul_node.inputs[1])?;

    let (axis_a, _size_a) = match &expand_a.op {
        RiscOp::Expand { axis, size } => (*axis, size.clone()),
        _ => return None,
    };
    let (axis_b, _size_b) = match &expand_b.op {
        RiscOp::Expand { axis, size } => (*axis, size.clone()),
        _ => return None,
    };

    // Get the actual matrix operands (inputs to the Expand nodes)
    if expand_a.inputs.len() != 1 || expand_b.inputs.len() != 1 {
        return None;
    }
    let a_id = expand_a.inputs[0];
    let b_id = expand_b.inputs[0];

    let a_node = dag.get(a_id)?;
    let b_node = dag.get(b_id)?;

    // Extract dimensions: for a standard matmul A[M,K] @ B[K,N] -> C[M,N]
    // Sum reduces the K axis after expand+mul creates a [M,K,N] intermediate
    extract_matmul_dims(a_node, b_node, axis, axis_a, axis_b, a_id, b_id)
}

fn extract_matmul_dims(
    a_node: &DagNode,
    b_node: &DagNode,
    sum_axis: usize,
    _expand_axis_a: usize,
    _expand_axis_b: usize,
    a_id: NodeId,
    b_id: NodeId,
) -> Option<MatmulInfo> {
    // A should be 2D [M, K], B should be 2D [K, N]
    if a_node.output_type.dims.len() != 2 || b_node.output_type.dims.len() != 2 {
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

    if k_a != k_b {
        return None;
    }

    // The sum axis should be the K dimension in the 3D intermediate [M, K, N]
    // which is axis 1
    if sum_axis != 1 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
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

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    #[test]
    fn detects_matmul_pattern() {
        // A[2,3] @ B[3,4] -> C[2,4]
        // Lower as: expand A to [2,3,4], expand B to [2,3,4], mul, sum axis=1
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(2, 3), None);
        let b = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4), None);
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

        let info = detect_matmul_pattern(&dag, sum).unwrap();
        assert_eq!(info.a, a);
        assert_eq!(info.b, b);
        assert_eq!(info.m, 2);
        assert_eq!(info.n, 4);
        assert_eq!(info.k, 3);
    }

    #[test]
    fn rejects_non_sum_node() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        assert!(detect_matmul_pattern(&dag, a).is_none());
    }

    #[test]
    fn rejects_sum_without_mul_input() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::Const { value: 1.0 },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let sum = dag.add_node(RiscOp::Sum { axis: 0 }, vec![a], scalar_f32(), None);
        assert!(detect_matmul_pattern(&dag, sum).is_none());
    }

    #[test]
    fn rejects_mismatched_k_dims() {
        // A[2,3] and B[5,4] — k_a=3 != k_b=5
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(2, 3), None);
        let b = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(5, 4), None);
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

    #[test]
    fn rejects_nonexistent_node() {
        let dag = Dag::new();
        assert!(detect_matmul_pattern(&dag, NodeId(99)).is_none());
    }
}
