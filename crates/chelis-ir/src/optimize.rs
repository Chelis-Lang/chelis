//! Basic DAG optimization passes.

use std::collections::{HashMap, HashSet};

use crate::dag::{Dag, NodeId, RiscOp};

/// Constant folding: if a binary op has two Const inputs, evaluate it.
pub fn constant_fold(dag: &mut Dag) {
    // Collect fold candidates first, then apply (to avoid borrow issues).
    let mut replacements: Vec<(NodeId, f64)> = Vec::new();

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
                    replacements.push((node.id, val));
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
                    _ => None,
                };
                if let Some(val) = result {
                    replacements.push((node.id, val));
                }
            }
        }
    }

    for (id, val) in replacements {
        let ty = dag.get(id).unwrap().output_type.clone();
        dag.replace_node(id, RiscOp::Const { value: val }, vec![], ty);
    }
}

/// Dead code elimination: mark nodes reachable from the last node, remove the rest.
///
/// Note: Since the DAG is a flat vector with index-based IDs, "removing" nodes
/// means replacing them with no-op Const(0) nodes. A full compaction would
/// require rewriting all NodeId references, which we skip for Phase 0.
pub fn dead_code_eliminate(dag: &mut Dag) {
    if dag.is_empty() {
        return;
    }
    // The "output" is the last node.
    let output = NodeId(dag.len() - 1);
    let mut reachable = HashSet::new();
    mark_reachable(dag, output, &mut reachable);

    for i in 0..dag.len() {
        let id = NodeId(i);
        if !reachable.contains(&id) {
            let ty = dag.get(id).unwrap().output_type.clone();
            dag.replace_node(id, RiscOp::Const { value: 0.0 }, vec![], ty);
        }
    }
}

fn mark_reachable(dag: &Dag, id: NodeId, visited: &mut HashSet<NodeId>) {
    if !visited.insert(id) {
        return;
    }
    if let Some(node) = dag.get(id) {
        for &input in &node.inputs {
            mark_reachable(dag, input, visited);
        }
    }
}

/// Common subexpression elimination: hash nodes by (op_discriminant, inputs),
/// merge duplicates by rewriting references.
///
/// Phase 0: simple version that detects duplicate Const nodes with the same value.
pub fn common_subexpr_eliminate(dag: &mut Dag) {
    // Map from (op-as-string, inputs) to the first NodeId with that signature.
    let mut seen: HashMap<(String, Vec<usize>), NodeId> = HashMap::new();
    // Map from old NodeId -> canonical NodeId.
    let mut remap: HashMap<NodeId, NodeId> = HashMap::new();

    for i in 0..dag.len() {
        let id = NodeId(i);
        let node = dag.get(id).unwrap();

        // Remap inputs through existing remaps.
        let canonical_inputs: Vec<usize> = node
            .inputs
            .iter()
            .map(|inp| remap.get(inp).unwrap_or(inp).0)
            .collect();

        let key = (format!("{:?}", node.op), canonical_inputs.clone());

        if let Some(&canonical) = seen.get(&key) {
            remap.insert(id, canonical);
        } else {
            seen.insert(key, id);
        }
    }

    // Rewrite inputs to use canonical IDs.
    if !remap.is_empty() {
        for i in 0..dag.len() {
            let id = NodeId(i);
            let node = dag.get(id).unwrap();
            let new_inputs: Vec<NodeId> = node
                .inputs
                .iter()
                .map(|inp| *remap.get(inp).unwrap_or(inp))
                .collect();
            if new_inputs != dag.get(id).unwrap().inputs {
                let op = dag.get(id).unwrap().op.clone();
                let ty = dag.get(id).unwrap().output_type.clone();
                dag.replace_node(id, op, new_inputs, ty);
            }
        }
    }
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
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());

        constant_fold(&mut dag);

        let result = dag.get(NodeId(2)).unwrap();
        assert_eq!(result.op, RiscOp::Const { value: 3.0 });
        assert!(result.inputs.is_empty());
    }

    #[test]
    fn constant_fold_mul() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 4.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Mul, vec![a, b], scalar_f32());

        constant_fold(&mut dag);

        let result = dag.get(NodeId(2)).unwrap();
        assert_eq!(result.op, RiscOp::Const { value: 12.0 });
    }

    #[test]
    fn constant_fold_neg() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Neg, vec![a], scalar_f32());

        constant_fold(&mut dag);

        let result = dag.get(NodeId(1)).unwrap();
        assert_eq!(result.op, RiscOp::Const { value: -5.0 });
    }

    #[test]
    fn cse_deduplicates_consts() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());

        common_subexpr_eliminate(&mut dag);

        // The Add node should now reference the same Const twice.
        let add_node = dag.get(NodeId(2)).unwrap();
        assert_eq!(add_node.inputs[0], add_node.inputs[1]);
    }
}
