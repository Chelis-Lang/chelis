//! Memory planning for generated C code.
//!
//! Phase 0: simple allocate-per-node, free-all-at-end strategy.

use chelis_ir::dag::{Dag, NodeId, RiscOp};

/// Returns true if the node's op aliases its input's data pointer (no owned allocation).
fn is_data_alias(op: &RiscOp) -> bool {
    matches!(
        op,
        RiscOp::Permute { .. }
            | RiscOp::Expand { .. }
            | RiscOp::Stride { .. }
            | RiscOp::Store { .. }
    )
}

/// Emit `chelis_free()` calls for all nodes except the specified output nodes.
/// For nodes that alias their input's data pointer (Permute, Expand, Stride, Store),
/// null the data pointer before freeing to avoid double-free.
pub fn emit_cleanup(dag: &Dag, output_ids: &[NodeId]) -> Vec<String> {
    let mut lines = Vec::new();
    for n in dag.nodes() {
        if output_ids.contains(&n.id) {
            continue;
        }
        if is_data_alias(&n.op) {
            lines.push(format!("    t{}->data = NULL;", n.id.0));
        }
        lines.push(format!("    chelis_free(t{});", n.id.0));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_ir::dag::{Dag, RiscOp, TensorType};

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    #[test]
    fn cleanup_skips_output_nodes() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());

        let lines = emit_cleanup(&dag, &[c]);
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("t0"));
        assert!(lines[1].contains("t1"));
        assert!(!lines.iter().any(|l| l.contains("t2")));
    }

    #[test]
    fn cleanup_empty_dag() {
        let dag = Dag::new();
        let lines = emit_cleanup(&dag, &[]);
        assert!(lines.is_empty());
    }

    #[test]
    fn cleanup_all_outputs_means_no_frees() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let lines = emit_cleanup(&dag, &[a]);
        assert!(lines.is_empty());
    }

    #[test]
    fn cleanup_multiple_outputs() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());

        let lines = emit_cleanup(&dag, &[a, c]);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("t1"));
    }

    #[test]
    fn cleanup_format_matches_indent() {
        let mut dag = Dag::new();
        dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let lines = emit_cleanup(&dag, &[]);
        assert!(lines[0].starts_with("    chelis_free(t0)"));
    }
}
