//! GPU memory cleanup for generated HIP code.
//!
//! Phase 1a: allocate-per-node, free-all-at-end.
//! Views (from movement ops) use `chelis_gpu_free_view` (host struct only).
//! Allocations use `chelis_gpu_free` (host struct + device data).

use chelis_ir::dag::{Dag, NodeId, RiscOp};

/// Returns true if the node is borrowed from the caller (Load) and must never be freed.
fn is_borrowed(op: &RiscOp) -> bool {
    matches!(op, RiscOp::Load { .. })
}

/// Returns true if the op creates a view (shares device memory with its input).
fn is_view(op: &RiscOp) -> bool {
    matches!(
        op,
        RiscOp::Reshape { .. }
            | RiscOp::Permute { .. }
            | RiscOp::Expand { .. }
            | RiscOp::Stride { .. }
    )
}

/// Emit cleanup calls for all GPU tensors except outputs.
///
/// Views use `chelis_gpu_free_view()` (frees host struct only, not device data).
/// Allocations use `chelis_gpu_free()` (frees both host struct and device data).
/// Loads are never freed (borrowed from caller).
pub fn emit_cleanup(dag: &Dag, output_ids: &[NodeId]) -> Vec<String> {
    let mut lines = Vec::new();
    for n in dag.nodes() {
        if output_ids.contains(&n.id) {
            continue;
        }
        if is_borrowed(&n.op) {
            continue;
        }
        if is_view(&n.op) {
            lines.push(format!("    chelis_gpu_free_view(d_t{});", n.id.0));
        } else {
            lines.push(format!("    chelis_gpu_free(d_t{});", n.id.0));
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    fn mat_f32(r: usize, c: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
            precision: chelis_types::types::Prim::F32,
        }
    }

    #[test]
    fn cleanup_skips_outputs() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
        let lines = emit_cleanup(&dag, &[c]);
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("d_t0"));
        assert!(lines[1].contains("d_t1"));
        assert!(!lines.iter().any(|l| l.contains("d_t2")));
    }

    #[test]
    fn cleanup_skips_loads() {
        let mut dag = Dag::new();
        dag.add_node(RiscOp::Load { name: "x".into() }, vec![], scalar_f32());
        let lines = emit_cleanup(&dag, &[]);
        assert!(lines.is_empty());
    }

    #[test]
    fn cleanup_views_use_free_view() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(2, 3));
        let _p = dag.add_node(RiscOp::Permute { axes: vec![1, 0] }, vec![x], mat_f32(3, 2));
        let lines = emit_cleanup(&dag, &[]);
        // Const (t0) should use chelis_gpu_free
        assert!(lines[0].contains("chelis_gpu_free(d_t0)"));
        // Permute (t1) should use chelis_gpu_free_view
        assert!(lines[1].contains("chelis_gpu_free_view(d_t1)"));
    }

    #[test]
    fn cleanup_reshape_is_view() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(2, 3));
        dag.add_node(
            RiscOp::Reshape {
                new_shape: vec![DimInfo::Lit(6)],
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(6)],
                precision: chelis_types::types::Prim::F32,
            },
        );
        let lines = emit_cleanup(&dag, &[]);
        assert!(lines[1].contains("chelis_gpu_free_view"));
    }

    #[test]
    fn cleanup_expand_is_view() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        dag.add_node(
            RiscOp::Expand { axis: 0, size: 4 },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: chelis_types::types::Prim::F32,
            },
        );
        let lines = emit_cleanup(&dag, &[]);
        assert!(lines[1].contains("chelis_gpu_free_view"));
    }
}
