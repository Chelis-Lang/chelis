//! hipBLAS matmul pattern matching for HIP codegen.
//!
//! Mirrors the CPU backend's expand+mul+sum detector so the HIP backend can
//! specialize the same lowered matmul subgraph to `hipblasSgemm`.

use chelis_ir::dag::{DagNode, DimInfo, NodeId, RiscOp};
use chelis_ir::ownership::VerifiedDagView;

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
pub fn detect_matmul_pattern(dag: VerifiedDagView<'_>, sum_id: NodeId) -> Option<MatmulInfo> {
    let sum_node = dag.get(sum_id)?;
    let axis = match &sum_node.op {
        RiscOp::Sum { axis, .. } => *axis,
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
        (
            RiscOp::Expand {
                axis: axis_a,
                size: _,
            },
            RiscOp::Expand {
                axis: axis_b,
                size: _,
            },
        ) => {
            if expand_a.inputs.len() != 1 || expand_b.inputs.len() != 1 {
                return None;
            }
            let a_id = expand_a.inputs[0];
            let b_id = expand_b.inputs[0];
            let a_node = dag.get(a_id)?;
            let b_node = dag.get(b_id)?;
            extract_matmul_dims(a_node, b_node, axis, *axis_a, *axis_b, a_id, b_id)
        }
        _ => None,
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_ir::dag::{Dag, DimInfo, TensorType};
    use chelis_types::types::Prim;

    fn detect(dag: &Dag, sum: NodeId) -> Option<MatmulInfo> {
        let verified = chelis_ir::ownership::verify_ownership(
            chelis_ir::ownership::lower_dag_ownership(dag.clone())
                .expect("BLAS detector unit-test DAG must lower ownership"),
        )
        .expect("BLAS detector unit-test DAG must verify ownership");
        super::detect_matmul_pattern(verified.emission(), sum)
    }

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
    fn detects_matmul_pattern() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(mat_f32(2, 3).precision, 1.0),
            vec![],
            mat_f32(2, 3),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(mat_f32(3, 4).precision, 1.0),
            vec![],
            mat_f32(3, 4),
            None,
        );
        let ea = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: chelis_ir::dag::RtDim::Lit(4),
            },
            vec![a],
            tensor3_f32(2, 3, 4),
            None,
        );
        let eb = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::RtDim::Lit(2),
            },
            vec![b],
            tensor3_f32(2, 3, 4),
            None,
        );
        let mul = dag.add_node(RiscOp::Mul, vec![ea, eb], tensor3_f32(2, 3, 4), None);
        let sum = dag.add_node(
            RiscOp::Sum {
                axis: 1,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![mul],
            mat_f32(2, 4),
            None,
        );

        let info = detect(&dag, sum).expect("matmul should be detected");
        assert_eq!(info.a, a);
        assert_eq!(info.b, b);
        assert_eq!(info.m, 2);
        assert_eq!(info.n, 4);
        assert_eq!(info.k, 3);
    }
}
