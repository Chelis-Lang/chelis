//! Basic DAG optimization passes.

use std::collections::HashMap;

use crate::dag::{Dag, NodeId, RiscOp};

/// Constant folding: if a binary op has two Const inputs, evaluate it.
///
/// Span propagation per spec/design/chelis_span_survival.md §2.3
/// Constant fold row: the replacement node inherits the **operation
/// node's** `span_id` (preserved automatically by `replace_node`, which
/// rewrites op/inputs/output_type but leaves span metadata in place).
/// Operand spans (canonical and merged_spans) that differ from the
/// operation's `span_id` append to the folded node's `merged_spans`,
/// lex-sorted and deduped. No `__synthesized_*__` marker is minted —
/// the operation node had a real source span (or `None`) before the
/// fold, and that's what survives.
pub fn constant_fold(dag: &mut Dag) {
    // Collect fold candidates first, then apply (to avoid borrow issues).
    // Each entry is (op_node_id, folded_value, operand_spans_to_merge).
    // operand_spans_to_merge = the union of each operand's `span_id`
    // (when distinct from the operation's) and each operand's existing
    // `merged_spans` — i.e. the operand's full provenance flowing onto
    // the folded result.
    let mut replacements: Vec<(NodeId, f64, Vec<String>)> = Vec::new();

    for node in dag.nodes() {
        if node.inputs.len() == 2 {
            let lhs = dag.get(node.inputs[0]);
            let rhs = dag.get(node.inputs[1]);
            if let (Some(l), Some(r)) = (lhs, rhs)
                && let (RiscOp::Const { value: lv }, RiscOp::Const { value: rv }) = (&l.op, &r.op)
            {
                let result = match &node.op {
                    RiscOp::Add => Some(lv + rv),
                    RiscOp::Mul => Some(lv * rv),
                    RiscOp::CmpLt => Some(if lv < rv { 1.0 } else { 0.0 }),
                    RiscOp::MaxElem => Some(if lv >= rv { *lv } else { *rv }),
                    _ => None,
                };
                if let Some(val) = result {
                    let merge_spans = collect_operand_spans(node, &[l, r]);
                    replacements.push((node.id, val, merge_spans));
                }
            }
        }
        // Unary constant folding.
        if node.inputs.len() == 1 {
            let input = dag.get(node.inputs[0]);
            if let Some(inp) = input
                && let RiscOp::Const { value: v } = &inp.op
            {
                let result = match &node.op {
                    RiscOp::Neg => Some(-v),
                    RiscOp::Exp => Some(v.exp()),
                    RiscOp::Log => Some(v.ln()),
                    RiscOp::Sin => Some(v.sin()),
                    RiscOp::Sqrt => Some(v.sqrt()),
                    RiscOp::Cos => Some(v.cos()),
                    RiscOp::Tan => Some(v.tan()),
                    RiscOp::Atan => Some(v.atan()),
                    RiscOp::Abs => Some(v.abs()),
                    RiscOp::Floor => Some(v.floor()),
                    RiscOp::Ceil => Some(v.ceil()),
                    RiscOp::Round => Some(v.round_ties_even()),
                    _ => None,
                };
                if let Some(val) = result {
                    let merge_spans = collect_operand_spans(node, &[inp]);
                    replacements.push((node.id, val, merge_spans));
                }
            }
        }
    }

    for (id, val, operand_spans) in replacements {
        let ty = dag.get(id).unwrap().output_type.clone();
        dag.replace_node(id, RiscOp::Const { value: val }, vec![], ty);
        // The operation's own `span_id` is preserved by `replace_node`
        // (it rewrites op/inputs/output_type, never span metadata).
        // Append each operand's full provenance to the folded node's
        // `merged_spans`. The shared helper handles None-no-op,
        // canonical-no-op (operand span equal to the op's own
        // `span_id`), dedup, and lex-sort.
        crate::span_merge::append_spans_to_node(dag, id, &operand_spans);
    }
}

/// Collect operand provenance to merge onto a folded result. For each
/// operand: include its `span_id` (if any) and its existing
/// `merged_spans`. The shared helper later dedups against the operation
/// node's own `span_id`, so we don't filter that here — we just collect
/// every operand-side span.
fn collect_operand_spans(
    _op_node: &crate::dag::DagNode,
    operands: &[&crate::dag::DagNode],
) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for operand in operands {
        if let Some(s) = &operand.span_id
            && !out.contains(s)
        {
            out.push(s.clone());
        }
        for s in &operand.merged_spans {
            if !out.contains(s) {
                out.push(s.clone());
            }
        }
    }
    out
}

/// Dead code elimination: build a new DAG with only reachable nodes.
///
/// Marks the last node and all Store nodes as live, propagates liveness
/// backward through inputs, then rebuilds the DAG with only live nodes
/// and remapped NodeIds.
pub fn dead_code_eliminate(dag: &Dag) -> Dag {
    dead_code_eliminate_with_remap(dag).0
}

/// Same as [`dead_code_eliminate`] but also returns the `old_id -> new_id`
/// remapping. Phase F (`lower_program_with_context`) needs the remap to
/// rewrite the library's name → NodeId symbol table after DCE renumbering.
pub fn dead_code_eliminate_with_remap(dag: &Dag) -> (Dag, HashMap<NodeId, NodeId>) {
    let n = dag.len();
    if n == 0 {
        return (Dag::new(), HashMap::new());
    }

    // Mark live nodes: DAG roots + all Store nodes.
    let mut live = vec![false; n];
    if dag.roots().is_empty() {
        live[n - 1] = true;
    } else {
        for &root in dag.roots() {
            live[root.0] = true;
        }
    }
    for node in dag.nodes() {
        if matches!(node.op, RiscOp::Store { .. }) {
            live[node.id.0] = true;
        }
    }

    // Propagate liveness backward.
    for i in (0..n).rev() {
        if live[i] {
            for &input in &dag.nodes()[i].inputs {
                live[input.0] = true;
            }
            if let Some(reusable_input) = dag.nodes()[i].reusable_input {
                live[reusable_input.0] = true;
            }
        }
    }

    // Rebuild with only live nodes, remapping IDs.
    let mut new_dag = Dag::new();
    let mut id_map: HashMap<usize, NodeId> = HashMap::new();

    for (old_id, node) in dag.nodes().iter().enumerate() {
        if live[old_id] {
            let new_inputs: Vec<NodeId> = node
                .inputs
                .iter()
                .map(|&old| *id_map.get(&old.0).unwrap())
                .collect();
            // DCE is a pure copy of surviving nodes — clone span_id and
            // merged_spans verbatim per `spec/design/chelis_span_survival.md`
            // §2.3 (DCE/remap row). This is required for the S2 oracle
            // (`lower_program` runs DCE inside `lower_program_to_library`,
            // and the audit invariant says input Deep spans must appear
            // on at least one IR node post-pipeline).
            let new_id = new_dag.add_node(
                node.op.clone(),
                new_inputs,
                node.output_type.clone(),
                node.span_id.clone(),
            );
            if let Some(reusable_input) = node.reusable_input
                && let Some(&mapped_input) = id_map.get(&reusable_input.0)
            {
                new_dag.set_reusable_input(new_id, mapped_input);
            }
            // Preserve merged_spans across DCE (S3 will populate them but
            // the invariant of pure-copy DCE means they must survive when
            // present).
            if !node.merged_spans.is_empty()
                && let Some(new_node) = new_dag.node_mut(new_id)
            {
                new_node.merged_spans = node.merged_spans.clone();
            }
            id_map.insert(old_id, new_id);
        }
    }

    for &root in dag.roots() {
        if let Some(&new_root) = id_map.get(&root.0) {
            new_dag.add_root(new_root);
        }
    }

    let node_remap: HashMap<NodeId, NodeId> = id_map
        .into_iter()
        .map(|(old, new)| (NodeId(old), new))
        .collect();
    (new_dag, node_remap)
}

/// Common subexpression elimination: build a new DAG, merging nodes
/// that have identical (op, remapped_inputs) keys.
///
/// Span propagation per spec/design/chelis_span_survival.md §2.3 CSE
/// row: the survivor (first node seen with a given key) keeps its own
/// `span_id`. When a duplicate is found, the duplicate's full
/// provenance — its `span_id` and its existing `merged_spans` — folds
/// into the survivor's `merged_spans` (lex-sorted, deduped) via
/// `crate::span_merge::merge_duplicate_into_survivor`. Survivor's
/// canonical span and any duplicate-canonical that equals it are
/// dedup'd by the helper.
pub fn common_subexpr_eliminate(dag: &Dag) -> Dag {
    let mut new_dag = Dag::new();
    let mut id_map: HashMap<usize, NodeId> = HashMap::new();
    let mut seen: HashMap<(String, Vec<NodeId>), NodeId> = HashMap::new();

    for node in dag.nodes() {
        let remapped_inputs: Vec<NodeId> = node
            .inputs
            .iter()
            .map(|&old| *id_map.get(&old.0).unwrap_or(&old))
            .collect();

        let op_key = format!("{:?}", node.op);
        let cse_key = (op_key, remapped_inputs.clone());

        if let Some(&existing) = seen.get(&cse_key) {
            // Duplicate: its full provenance (canonical + merged) folds
            // onto the survivor so the audit chain through the dropped
            // node is preserved.
            crate::span_merge::merge_duplicate_into_survivor(
                &mut new_dag,
                existing,
                node.span_id.as_deref(),
                &node.merged_spans,
            );
            id_map.insert(node.id.0, existing);
        } else {
            // Survivor: clone its own span_id and merged_spans onto the
            // new node so the canonical provenance flows through CSE.
            let new_id = new_dag.add_node(
                node.op.clone(),
                remapped_inputs,
                node.output_type.clone(),
                node.span_id.clone(),
            );
            if !node.merged_spans.is_empty()
                && let Some(new_node) = new_dag.node_mut(new_id)
            {
                new_node.merged_spans = node.merged_spans.clone();
            }
            if let Some(reusable_input) = node.reusable_input
                && let Some(&mapped_input) = id_map.get(&reusable_input.0)
            {
                new_dag.set_reusable_input(new_id, mapped_input);
            }
            id_map.insert(node.id.0, new_id);
            seen.insert(cse_key, new_id);
        }
    }

    for &root in dag.roots() {
        if let Some(&new_root) = id_map.get(&root.0) {
            new_dag.add_root(new_root);
        }
    }

    new_dag
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{Dag, RiscOp, TensorType};

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    #[test]
    fn constant_fold_add() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);

        constant_fold(&mut dag);

        let result = dag.get(NodeId(2)).unwrap();
        assert_eq!(result.op, RiscOp::Const { value: 3.0 });
        assert!(result.inputs.is_empty());
    }

    #[test]
    fn constant_fold_mul() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 4.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::Mul, vec![a, b], scalar_f32(), None);

        constant_fold(&mut dag);

        let result = dag.get(NodeId(2)).unwrap();
        assert_eq!(result.op, RiscOp::Const { value: 12.0 });
    }

    #[test]
    fn constant_fold_neg() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::Neg, vec![a], scalar_f32(), None);

        constant_fold(&mut dag);

        let result = dag.get(NodeId(1)).unwrap();
        assert_eq!(result.op, RiscOp::Const { value: -5.0 });
    }

    #[test]
    fn dce_removes_dead_nodes() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let _dead = dag.add_node(RiscOp::Const { value: 99.0 }, vec![], scalar_f32(), None);
        let live = dag.add_node(RiscOp::Neg, vec![a], scalar_f32(), None);
        dag.add_root(live);

        let new_dag = dead_code_eliminate(&dag);
        // Dead const(99) should be removed; only 2 nodes remain.
        assert_eq!(new_dag.len(), 2);
    }

    #[test]
    fn dce_keeps_store_nodes() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        dag.add_node(
            RiscOp::Store { name: "out".into() },
            vec![a],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32(), None);
        let live = dag.add_node(RiscOp::Neg, vec![b], scalar_f32(), None);
        dag.add_root(live);

        let new_dag = dead_code_eliminate(&dag);
        // Store + its input const + second const + neg = 4 nodes all live.
        assert_eq!(new_dag.len(), 4);
    }

    #[test]
    fn cse_deduplicates_consts() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let sum = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
        dag.add_root(sum);

        let new_dag = common_subexpr_eliminate(&dag);

        // CSE merges the two identical Consts, so only 2 nodes (1 Const + 1 Add).
        assert_eq!(new_dag.len(), 2);
        // The Add node should reference the same Const twice.
        let add_node = new_dag.get(NodeId(1)).unwrap();
        assert_eq!(add_node.inputs[0], add_node.inputs[1]);
    }

    #[test]
    fn dce_keeps_all_roots() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32(), None);
        dag.add_root(a);
        dag.add_root(b);

        let new_dag = dead_code_eliminate(&dag);
        assert_eq!(new_dag.len(), 2);
        assert_eq!(new_dag.roots().len(), 2);
    }
}
