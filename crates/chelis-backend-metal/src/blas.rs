//! Tiled matmul detection + emission for the Metal backend.
//!
//! Mirrors `chelis_backend_hip::blas` so the Metal backend specializes the
//! same lowered matmul subgraph (`expand + mul + sum(axis=1)`) the C and
//! HIP backends already recognize. The emit path uses
//! `kernels::matmul_tiled_kernel` (16x16 tiled MSL kernel) and dispatches
//! via `chelis_metal_launch2d`.

use chelis_ir::dag::{DagNode, DimInfo, NodeId, RiscOp};
use chelis_ir::ownership::VerifiedDagView;
use chelis_types::types::Prim;

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
    /// Operand precision (both A and B must agree per spec §5.4 — no
    /// implicit promotion). The emitter picks the matmul dispatch
    /// strategy from this field: `f32` / `f16` route through
    /// `MPSMatrixMultiplication`; `bf16` routes through the tiled MSL
    /// kernel; integer matmul is rejected at the F1 codegen guard.
    pub precision: Prim,
}

/// Try to detect a matmul pattern rooted at the given Sum node.
///
/// Returns `Some(info)` if the Sum is the head of an `expand + mul +
/// sum(axis=1)` matmul subgraph with statically-known rank-2 operands at
/// a Metal-supported precision. WS-M1 widens the operand admission
/// from f32-only to the float family (f32/f16/bf16) so the MPS / tiled
/// MSL dispatch in `emit::Emitter::emit_matmul` can route per
/// precision. Integer matmul is rejected at type-check per
/// spec/04-type-system.md §5.7.2 (Wave-2-Fixups B6) so it should never
/// reach this detector with matching operand precision.
pub fn detect_matmul_pattern(dag: VerifiedDagView<'_>, sum_id: NodeId) -> Option<MatmulInfo> {
    let sum_node = dag.get(sum_id)?;
    // Read the Sum node's accumulator field per the destructure-`..`
    // memory rule (do not silently ignore the accumulator). This is
    // membership-style — the matmul detector only checks that the
    // accumulator dispatch is admissible at codegen, not that any
    // particular value is in play. See spec/04-type-system.md §5.7.1.
    let (axis, _accumulator) = match &sum_node.op {
        RiscOp::Sum { axis, accumulator } => (*axis, *accumulator),
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
    let prec_a = a_node.output_type.precision;
    let prec_b = b_node.output_type.precision;
    if prec_a != prec_b {
        return None;
    }
    // Admit the Metal float family. Integer matmul is rejected at
    // type-check (Wave-2-Fixups B6 / spec §5.7.2); the F1 guard in
    // `emit::Emitter::emit_matmul` enforces it defensively too.
    if !matches!(prec_a, Prim::F32 | Prim::F16 | Prim::Bf16) {
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
        precision: prec_a,
    })
}

/// Find every matmul subgraph in the DAG.
///
/// Returns `(sum_node_id, info)` for each detected pattern, ordered by
/// the Sum node's position in the DAG.
pub fn find_all_matmuls(dag: VerifiedDagView<'_>) -> Vec<(NodeId, MatmulInfo)> {
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

    fn detect(dag: &Dag, sum: NodeId) -> Option<MatmulInfo> {
        let verified = crate::testing::verified_dag(dag)
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
        assert_eq!(info.precision, Prim::F32);
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
        // sum axis=0 is the wrong axis; matmul detector should bail.
        let sum = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![mul],
            mat_f32(3, 4),
            None,
        );

        assert!(detect(&dag, sum).is_none());
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
            tensor3_f32(2, 5, 4),
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

        let error = chelis_ir::ownership::lower_dag_ownership(dag)
            .expect_err("dimension-mismatched matmul must not cross the verified boundary");
        assert!(error.to_string().contains("mismatched dimension"));
        let _ = sum;
    }

    fn mat_prec(r: usize, c: usize, prec: Prim) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
            precision: prec,
        }
    }

    fn tensor3_prec(a: usize, b: usize, c: usize, prec: Prim) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(a), DimInfo::Lit(b), DimInfo::Lit(c)],
            precision: prec,
        }
    }

    fn build_matmul(prec: Prim) -> (Dag, NodeId) {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::Load { name: "a".into() },
            vec![],
            mat_prec(2, 3, prec),
            None,
        );
        let b = dag.add_node(
            RiscOp::Load { name: "b".into() },
            vec![],
            mat_prec(3, 4, prec),
            None,
        );
        let ea = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: chelis_ir::dag::RtDim::Lit(4),
            },
            vec![a],
            tensor3_prec(2, 3, 4, prec),
            None,
        );
        let eb = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::RtDim::Lit(2),
            },
            vec![b],
            tensor3_prec(2, 3, 4, prec),
            None,
        );
        let mul = dag.add_node(RiscOp::Mul, vec![ea, eb], tensor3_prec(2, 3, 4, prec), None);
        // Per spec/04-type-system.md §5.7.1 the accumulator for f16/bf16
        // sum is f32; for f32 it stays f32. The detector only inspects
        // axis (membership-style); the codegen path picks the dispatch.
        let acc = match prec {
            Prim::F16 | Prim::Bf16 => Prim::F32,
            other => other,
        };
        let sum = dag.add_node(
            RiscOp::Sum {
                axis: 1,
                accumulator: acc,
            },
            vec![mul],
            mat_prec(2, 4, acc),
            None,
        );
        (dag, sum)
    }

    #[test]
    fn detects_f16_matmul_with_precision_threaded_through() {
        let (dag, sum) = build_matmul(Prim::F16);
        let info = detect(&dag, sum).expect("f16 matmul should be detected");
        assert_eq!(info.precision, Prim::F16);
    }

    #[test]
    fn detects_bf16_matmul_with_precision_threaded_through() {
        let (dag, sum) = build_matmul(Prim::Bf16);
        let info = detect(&dag, sum).expect("bf16 matmul should be detected");
        assert_eq!(info.precision, Prim::Bf16);
    }

    #[test]
    fn rejects_integer_matmul_at_detector() {
        // Integer matmul never reaches a healthy backend (rejected at
        // type-check per §5.7.2); the detector still bails so the F1
        // codegen guard is never asked to handle this case in practice.
        let (dag, sum) = build_matmul(Prim::Int32);
        assert!(detect(&dag, sum).is_none());
    }

    #[test]
    fn rejects_mixed_precision_matmul() {
        // Per spec §5.4 there is no implicit precision promotion; an
        // f32 × f16 matmul is a type error upstream, but if it ever
        // synthesizes here the detector must not pretend to handle it.
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::Load { name: "a".into() },
            vec![],
            mat_prec(2, 3, Prim::F32),
            None,
        );
        let b = dag.add_node(
            RiscOp::Load { name: "b".into() },
            vec![],
            mat_prec(3, 4, Prim::F16),
            None,
        );
        let ea = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: chelis_ir::dag::RtDim::Lit(4),
            },
            vec![a],
            tensor3_prec(2, 3, 4, Prim::F32),
            None,
        );
        let eb = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::RtDim::Lit(2),
            },
            vec![b],
            tensor3_prec(2, 3, 4, Prim::F16),
            None,
        );
        let mul = dag.add_node(
            RiscOp::Mul,
            vec![ea, eb],
            tensor3_prec(2, 3, 4, Prim::F32),
            None,
        );
        let sum = dag.add_node(
            RiscOp::Sum {
                axis: 1,
                accumulator: Prim::F32,
            },
            vec![mul],
            mat_prec(2, 4, Prim::F32),
            None,
        );
        let error = chelis_ir::ownership::lower_dag_ownership(dag)
            .expect_err("mixed-precision matmul must not cross the verified boundary");
        assert!(error.to_string().contains("mismatched precisions"));
        let _ = sum;
    }
}
