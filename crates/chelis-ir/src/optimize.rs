//! Basic DAG optimization passes.

use std::collections::HashMap;

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

/// Dead code elimination: build a new DAG with only reachable nodes.
///
/// Marks the last node and all Store nodes as live, propagates liveness
/// backward through inputs, then rebuilds the DAG with only live nodes
/// and remapped NodeIds.
pub fn dead_code_eliminate(dag: &Dag) -> Dag {
    let n = dag.len();
    if n == 0 {
        return Dag::new();
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
            let new_id = new_dag.add_node(node.op.clone(), new_inputs, node.output_type.clone());
            if let Some(reusable_input) = node.reusable_input
                && let Some(&mapped_input) = id_map.get(&reusable_input.0)
            {
                new_dag.set_reusable_input(new_id, mapped_input);
            }
            id_map.insert(old_id, new_id);
        }
    }

    for &root in dag.roots() {
        if let Some(&new_root) = id_map.get(&root.0) {
            new_dag.add_root(new_root);
        }
    }

    new_dag
}

/// Common subexpression elimination: build a new DAG, merging nodes
/// that have identical (op, remapped_inputs) keys.
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
            id_map.insert(node.id.0, existing);
        } else {
            let new_id =
                new_dag.add_node(node.op.clone(), remapped_inputs, node.output_type.clone());
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
    fn dce_removes_dead_nodes() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let _dead = dag.add_node(RiscOp::Const { value: 99.0 }, vec![], scalar_f32());
        let live = dag.add_node(RiscOp::Neg, vec![a], scalar_f32());
        dag.add_root(live);

        let new_dag = dead_code_eliminate(&dag);
        // Dead const(99) should be removed; only 2 nodes remain.
        assert_eq!(new_dag.len(), 2);
    }

    #[test]
    fn dce_keeps_store_nodes() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::Store { name: "out".into() }, vec![a], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        let live = dag.add_node(RiscOp::Neg, vec![b], scalar_f32());
        dag.add_root(live);

        let new_dag = dead_code_eliminate(&dag);
        // Store + its input const + second const + neg = 4 nodes all live.
        assert_eq!(new_dag.len(), 4);
    }

    #[test]
    fn cse_deduplicates_consts() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let sum = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
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
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        dag.add_root(a);
        dag.add_root(b);

        let new_dag = dead_code_eliminate(&dag);
        assert_eq!(new_dag.len(), 2);
        assert_eq!(new_dag.roots().len(), 2);
    }
}
