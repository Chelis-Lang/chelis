//! Graph ownership normalization without source policy.

use crate::collections::{HashMap, HashSet};
use crate::dag::{Dag, NodeId, RiscOp, TensorType};
use alloc::vec;
use alloc::vec::Vec;
use chelis_numeric_core::prim::Prim;
use chelis_unord::UnordMap;

/// The graph and its original-node map.
pub struct OwnershipResult {
    pub dag: Dag,
    pub old_to_new: UnordMap<NodeId, NodeId>,
}

/// Insert copies before consuming fanout, then close unused value lifetimes.
pub fn normalize_ownership(dag: &Dag) -> OwnershipResult {
    let (dag, old_to_new) = insert_copy_nodes_for_consuming_fanout(dag);
    OwnershipResult {
        dag: insert_drop_nodes_for_unconsumed_values(dag),
        old_to_new,
    }
}

pub fn insert_copy_nodes_for_consuming_fanout(dag: &Dag) -> (Dag, UnordMap<NodeId, NodeId>) {
    let mut consuming_uses = HashMap::<NodeId, usize>::new();
    for node in dag.nodes() {
        if !op_consumes_inputs(&node.op) {
            continue;
        }
        for input in &node.inputs {
            *consuming_uses.entry(*input).or_default() += 1;
        }
    }

    let mut seen_consuming_uses = HashMap::<NodeId, usize>::new();
    let mut out = Dag::new();
    out.inherit_declarations(dag);
    let mut id_map = UnordMap::<NodeId, NodeId>::new();

    for node in dag.nodes() {
        let mut inputs = Vec::with_capacity(node.inputs.len());
        for input in &node.inputs {
            let mapped = *id_map
                .get(input)
                .expect("input must have been remapped before consumer");
            let total = consuming_uses.get(input).copied().unwrap_or_default();
            if op_consumes_inputs(&node.op) && total > 1 {
                let seen = seen_consuming_uses.entry(*input).or_default();
                *seen += 1;
                if *seen < total {
                    let input_ty = out
                        .get(mapped)
                        .map(|n| n.output_type.clone())
                        .unwrap_or_else(TensorType::scalar_f32);
                    let copy = out.add_node(
                        node.owner.remap(&id_map),
                        RiscOp::Copy,
                        vec![mapped],
                        input_ty,
                        node.span_id.clone(),
                    );
                    inputs.push(copy);
                    continue;
                }
            }
            inputs.push(mapped);
        }

        let reusable_input = node.reusable_input.and_then(|old| {
            node.inputs
                .iter()
                .position(|input| *input == old)
                .and_then(|slot| inputs.get(slot).copied())
                .or_else(|| id_map.get(&old).copied())
        });

        let new_id = out.add_node(
            node.owner.remap(&id_map),
            node.op.clone(),
            inputs,
            node.output_type.clone(),
            node.span_id.clone(),
        );
        if let Some(new_node) = out.node_mut(new_id) {
            new_node.reusable_input = reusable_input;
            new_node.merged_spans = node.merged_spans.clone();
            // chelis#384/#397: preserve (remapped) shape-only deps so an
            // `expand` shape-derived extent source survives this rebuild. Deps are
            // earlier in topo order, so already remapped in `id_map`.
            new_node.shape_deps = node
                .shape_deps
                .iter()
                .filter_map(|old| id_map.get(old).copied())
                .collect();
            new_node.result_claim_deps = node
                .result_claim_deps
                .iter()
                .map(|old| {
                    *id_map
                        .get(old)
                        .unwrap_or_else(|| panic!("unmapped result claim dependency {old:?}"))
                })
                .collect();
        }
        id_map.insert(node.id, new_id);
    }

    for root in dag.roots() {
        if let Some(new_root) = id_map.get(root) {
            out.add_root(*new_root);
        }
    }

    (out, id_map)
}

fn op_consumes_inputs(op: &RiscOp) -> bool {
    matches!(op, RiscOp::Realize | RiscOp::Drop | RiscOp::Store { .. })
}

pub fn insert_drop_nodes_for_unconsumed_values(mut dag: Dag) -> Dag {
    let mut consumed = HashSet::<NodeId>::new();
    for node in dag.nodes() {
        if !op_consumes_inputs(&node.op) {
            continue;
        }
        consumed.extend(node.inputs.iter().copied());
    }
    let roots = dag.roots().iter().copied().collect::<HashSet<_>>();
    let values_to_drop = dag
        .nodes()
        .iter()
        .filter(|node| !roots.contains(&node.id))
        .filter(|node| !consumed.contains(&node.id))
        .filter(|node| {
            !matches!(
                node.op,
                RiscOp::Load { .. } | RiscOp::Drop | RiscOp::Store { .. }
            )
        })
        // A key, at any rank, is never dropped. Its one use is a draw, a key
        // operation or a root (spec/10 §3.2), and a `Drop` would be a second
        // reader and a key reaching an operation that does not consume keys.
        // Its storage is still released exactly once: an owned value with no
        // terminal of its own gets the ownership plan's scope-end release.
        .filter(|node| node.output_type.precision != Prim::Key)
        .map(|node| {
            (
                node.id,
                node.output_type.clone(),
                node.span_id.clone(),
                node.merged_spans.clone(),
                node.owner,
            )
        })
        .collect::<Vec<_>>();

    // A discarded value's terminal belongs to the owner of the value it
    // drops (chelis#2476: an unrooted `Drop` owned by no root).
    for (id, ty, span_id, merged_spans, owner) in values_to_drop {
        let drop = dag.add_node(owner, RiscOp::Drop, vec![id], ty, span_id);
        if let Some(node) = dag.node_mut(drop) {
            node.merged_spans = merged_spans;
        }
    }

    dag
}

pub fn strip_drop_nodes(dag: &Dag) -> OwnershipResult {
    let mut out = Dag::new();
    out.inherit_declarations(dag);
    let mut id_map = UnordMap::<NodeId, NodeId>::new();

    for node in dag.nodes() {
        if matches!(node.op, RiscOp::Drop) {
            continue;
        }
        let inputs = node
            .inputs
            .iter()
            .filter_map(|input| id_map.get(input).copied())
            .collect::<Vec<_>>();
        let new_id = out.add_node(
            node.owner.remap(&id_map),
            node.op.clone(),
            inputs,
            node.output_type.clone(),
            node.span_id.clone(),
        );
        if let Some(new_node) = out.node_mut(new_id) {
            new_node.reusable_input = node
                .reusable_input
                .and_then(|old| id_map.get(&old).copied());
            new_node.merged_spans = node.merged_spans.clone();
            // chelis#384/#397: preserve (remapped) shape-only deps so an
            // `expand` shape-derived extent source survives this rebuild. Deps are
            // earlier in topo order, so already remapped in `id_map`.
            new_node.shape_deps = node
                .shape_deps
                .iter()
                .filter_map(|old| id_map.get(old).copied())
                .collect();
            new_node.result_claim_deps = node
                .result_claim_deps
                .iter()
                .map(|old| {
                    *id_map
                        .get(old)
                        .unwrap_or_else(|| panic!("unmapped result claim dependency {old:?}"))
                })
                .collect();
        }
        id_map.insert(node.id, new_id);
    }

    for root in dag.roots() {
        if let Some(new_root) = id_map.get(root) {
            out.add_root(*new_root);
        }
    }

    OwnershipResult {
        dag: out,
        old_to_new: id_map,
    }
}
