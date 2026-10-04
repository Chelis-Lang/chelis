//! Post-AD decomposition of retained activation identities.
use crate::dag::{Dag, NodeId, RiscOp};
use chelis_unord::UnordMap;

/// Recreate [05-OP-48]'s stable forward graph only after AD selected its rule.
/// Idempotent, with every declaration and non-value dependency preserved.
pub fn decompose(dag: &Dag) -> Dag {
    if !dag
        .nodes()
        .iter()
        .any(|node| matches!(node.op, RiscOp::Softmax { .. }))
    {
        return dag.clone();
    }
    let mut out = Dag::new();
    out.inherit_declarations(dag);
    let mut map: UnordMap<NodeId, NodeId> = UnordMap::new();
    for node in dag.nodes() {
        let inputs = node.inputs.iter().map(|id| map[id]).collect::<Vec<_>>();
        let owner = node.owner.remap(&map);
        let result = if let RiscOp::Softmax { axis } = node.op {
            crate::tier2::decompose_softmax(
                owner,
                &mut out,
                inputs[0],
                axis,
                &node.output_type,
                node.span_id.as_deref(),
            )
        } else {
            out.add_node(
                owner,
                node.op.clone(),
                inputs,
                node.output_type.clone(),
                node.span_id.clone(),
            )
        };
        if let Some(reusable) = node.reusable_input {
            out.set_reusable_input(result, map[&reusable]);
        }
        out.node_mut(result)
            .unwrap()
            .merged_spans
            .extend(node.merged_spans.clone());
        out.preserve_shape_deps_strict(result, &node.shape_deps, &map)
            .expect("complete softmax decomposition shape dependency map");
        out.preserve_result_claim_deps(result, &node.result_claim_deps, &map);
        map.insert(node.id, result);
    }
    for root in dag.roots() {
        out.add_root(map[root]);
    }
    out
}
