//! DAG-to-DAG fusion pass.
//!
//! Greedy kernel fusion: merge adjacent single-consumer elementwise ops into
//! `FusedElem` nodes that emit as single GPU kernels. Also fuses elementwise
//! chains into trailing reductions (the elementwise ops become part of the
//! reduction's inner loop).
//!
//! **Invariant:** fusion never duplicates computation. A node with multiple
//! consumers is never absorbed into a fused chain.

use chelis_unord::{UnordMap, UnordSet};

use crate::dag::{Dag, DagNode, FusedInput, FusedStep, FusedStepOp, NodeId, RiscOp};

/// Result of running the fusion pass.
pub struct FuseResult {
    /// Fused DAG.
    pub dag: Dag,
    /// Mapping from original node IDs to node IDs in the fused DAG.
    pub old_to_new: UnordMap<NodeId, NodeId>,
}

/// Apply greedy kernel fusion to a DAG.
/// Returns a new DAG with fusible chains replaced by FusedElem nodes.
pub fn fuse(dag: &Dag) -> Dag {
    fuse_with_remap(dag).dag
}

/// Apply greedy kernel fusion to a DAG and return the old→new node remap.
pub fn fuse_with_remap(dag: &Dag) -> FuseResult {
    if dag.is_empty() {
        return FuseResult {
            dag: Dag::new(),
            old_to_new: UnordMap::new(),
        };
    }

    // 1. Build consumer counts.
    let consumer_count = build_consumer_counts(dag);

    // 2. Identify chains: walk forward, greedily extend.
    let chains = find_chains(dag, &consumer_count);

    // 3. Rebuild DAG with fused nodes.
    let (dag, old_to_new) = rebuild_with_fusion(dag, &chains);
    FuseResult { dag, old_to_new }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Count how many nodes consume each node's output.
fn build_consumer_counts(dag: &Dag) -> Vec<usize> {
    let mut counts = vec![0usize; dag.len()];
    for node in dag.nodes() {
        for &input in &node.inputs {
            counts[input.0] += 1;
        }
    }
    // Roots count as consumers (they must be materialized).
    for &root in dag.roots() {
        counts[root.0] += 1;
    }
    counts
}

/// Returns true if the op is an elementwise op that can participate in fusion.
fn is_fusible_elementwise(node: &DagNode) -> bool {
    // chelis#729 Phase 3 / chelis#699: the typed backends now have trapping
    // direct integer Abs/Sub/extrema kernels, while their general fused
    // integer kernels are still deliberately unavailable. Keep those integer
    // identities materialized so ordinary source programs cannot be optimized
    // back onto a float-only fused path. Their float forms remain fusible.
    if matches!(
        node.op,
        RiscOp::Abs | RiscOp::Sub | RiscOp::MaxElem | RiscOp::MinElem
    ) && node.output_type.precision.is_integer()
    {
        return false;
    }
    matches!(
        node.op,
        RiscOp::Add
            | RiscOp::Sub
            | RiscOp::Mul
            | RiscOp::Div
            | RiscOp::FloorDiv
            | RiscOp::TruncDiv
            | RiscOp::MaxElem
            | RiscOp::MinElem
            | RiscOp::Neg
            | RiscOp::Recip
            | RiscOp::Exp
            | RiscOp::Log
            | RiscOp::Sin
            | RiscOp::Sqrt
            | RiscOp::Cos
            | RiscOp::Tan
            | RiscOp::Atan
            | RiscOp::Abs
            | RiscOp::Floor
            | RiscOp::Ceil
            | RiscOp::Round
    )
}

/// Convert a RiscOp to its FusedStepOp equivalent.
fn to_fused_step_op(op: &RiscOp) -> FusedStepOp {
    match op {
        RiscOp::Add => FusedStepOp::Add,
        RiscOp::Sub => FusedStepOp::Sub,
        RiscOp::Mul => FusedStepOp::Mul,
        RiscOp::Div => FusedStepOp::Div,
        RiscOp::FloorDiv => FusedStepOp::FloorDiv,
        RiscOp::TruncDiv => FusedStepOp::TruncDiv,
        RiscOp::MaxElem => FusedStepOp::MaxElem,
        RiscOp::MinElem => FusedStepOp::MinElem,
        RiscOp::Neg => FusedStepOp::Neg,
        RiscOp::Recip => FusedStepOp::Recip,
        RiscOp::Exp => FusedStepOp::Exp,
        RiscOp::Log => FusedStepOp::Log,
        RiscOp::Sin => FusedStepOp::Sin,
        RiscOp::Sqrt => FusedStepOp::Sqrt,
        RiscOp::Cos => FusedStepOp::Cos,
        RiscOp::Tan => FusedStepOp::Tan,
        RiscOp::Atan => FusedStepOp::Atan,
        RiscOp::Abs => FusedStepOp::Abs,
        RiscOp::Floor => FusedStepOp::Floor,
        RiscOp::Ceil => FusedStepOp::Ceil,
        RiscOp::Round => FusedStepOp::Round,
        _ => panic!("not a fusible elementwise op: {op:?}"),
    }
}

/// A chain of fusible nodes identified in the DAG.
/// `nodes` is in execution order: nodes[0] is the first op, nodes[last] is the output.
#[derive(Debug)]
struct Chain {
    /// Node IDs in execution order.
    nodes: Vec<NodeId>,
}

/// Find all maximal fusible chains in the DAG.
///
/// A chain is a sequence of elementwise nodes where each node (except the last
/// in the chain) has exactly one consumer, and that consumer is the next node
/// in the chain.
fn find_chains(dag: &Dag, consumer_count: &[usize]) -> Vec<Chain> {
    let mut chains = Vec::new();
    let mut in_chain: Vec<bool> = vec![false; dag.len()];
    let claimed_producers = crate::axis_sources::claimed_producers(dag);
    let is_claim_barrier = |node: &DagNode| claimed_producers[node.id.0];

    // Walk in topological order.
    for node in dag.nodes() {
        let id = node.id.0;
        if in_chain[id] {
            continue;
        }
        if !is_fusible_elementwise(node) || is_claim_barrier(node) {
            continue;
        }

        // Start a new chain from this node.
        let mut chain = vec![node.id];
        in_chain[id] = true;

        // Extend forward: find the single consumer, check if fusible.
        let mut current = node.id;
        loop {
            // Can we extend? Current node must have exactly 1 consumer.
            if consumer_count[current.0] != 1 {
                break;
            }
            // Find the single consumer.
            let consumer = dag
                .nodes()
                .iter()
                .find(|n| n.inputs.contains(&current) && !in_chain[n.id.0]);
            match consumer {
                Some(c) if is_fusible_elementwise(c) && !is_claim_barrier(c) => {
                    // Check all of this consumer's inputs: only fuse if the
                    // consumer's chain-internal inputs are all single-consumer.
                    // (Other inputs are external and fine.)
                    chain.push(c.id);
                    in_chain[c.id.0] = true;
                    current = c.id;
                }
                _ => break,
            }
        }

        if chain.len() >= 2 {
            chains.push(Chain { nodes: chain });
        }
    }

    chains
}

/// Rebuild the DAG, replacing chain nodes with FusedElem nodes.
fn rebuild_with_fusion(dag: &Dag, chains: &[Chain]) -> (Dag, UnordMap<NodeId, NodeId>) {
    // Map old node ID → chain index (if part of a chain).
    let mut node_to_chain: UnordMap<usize, usize> = UnordMap::new();
    // For each chain, which node is the "representative" (last node, produces output).
    let mut chain_output: UnordMap<usize, NodeId> = UnordMap::new();

    for (ci, chain) in chains.iter().enumerate() {
        for &nid in &chain.nodes {
            node_to_chain.insert(nid.0, ci);
        }
        chain_output.insert(ci, *chain.nodes.last().unwrap());
    }

    let mut new_dag = Dag::new();
    new_dag.inherit_declarations(dag);
    let mut id_map: UnordMap<usize, NodeId> = UnordMap::new();
    let claimed_producers = crate::axis_sources::claimed_producers(dag);

    for node in dag.nodes() {
        let old_id = node.id.0;

        if let Some(&ci) = node_to_chain.get(&old_id) {
            // This node is part of a chain.
            let chain_out = chain_output[&ci];
            if node.id != chain_out {
                // Not the output node of the chain — skip (will be absorbed).
                continue;
            }

            // This is the chain output node — emit a FusedElem.
            let chain = &chains[ci];
            assert!(
                chain.nodes.iter().all(|id| !claimed_producers[id.0]),
                "fusion chain contains a producer-owned extent claim"
            );
            let (fused_op, external_inputs) = build_fused_elem(dag, chain, &id_map);
            let reusable_input = reusable_external_input(dag, chain, &external_inputs);

            let remapped_inputs: Vec<NodeId> = external_inputs
                .iter()
                .map(|&old| {
                    *id_map
                        .get(&old.0)
                        .unwrap_or_else(|| panic!("unmapped input {old:?} in fusion"))
                })
                .collect();

            let output_type = dag.get(chain_out).unwrap().output_type.clone();
            // Span propagation per spec/design/chelis_span_survival.md
            // §2.3 Fusion row: FusedElem's span_id = first contributor's
            // span_id; merged_spans = sort_dedup(rest contributors'
            // spans ∪ each contributor's pre-existing merged_spans).
            let first = dag.get(chain.nodes[0]).expect("chain head exists");
            let new_id = new_dag.add_node(
                dag.get(chain_out).expect("chain output exists").decl,
                fused_op,
                remapped_inputs,
                output_type,
                first.span_id.clone(),
            );
            if let Some(reusable) = reusable_input {
                let mapped = *id_map
                    .get(&reusable.0)
                    .unwrap_or_else(|| panic!("unmapped reusable input {reusable:?} in fusion"));
                new_dag.set_reusable_input(new_id, mapped);
            }
            // Carry the first contributor's pre-existing merged_spans
            // verbatim onto the FusedElem (they already belong to the
            // canonical contributor's audit chain).
            if !first.merged_spans.is_empty()
                && let Some(new_node) = new_dag.node_mut(new_id)
            {
                new_node.merged_spans = first.merged_spans.clone();
            }
            // Append every other contributor's full provenance —
            // canonical span_id + pre-existing merged_spans — onto the
            // FusedElem's merged_spans (the shared helper handles
            // dedup-against-canonical and lex-sort).
            for &nid in &chain.nodes[1..] {
                let contributor = dag.get(nid).expect("contributor exists");
                crate::span_merge::append_span_to_node(
                    &mut new_dag,
                    new_id,
                    contributor.span_id.as_deref(),
                );
                crate::span_merge::append_spans_to_node(
                    &mut new_dag,
                    new_id,
                    &contributor.merged_spans,
                );
            }
            // Map ALL chain nodes to this new ID (consumers reference chain internals).
            for &nid in &chain.nodes {
                id_map.insert(nid.0, new_id);
            }
        } else {
            // Not part of a chain — emit as-is with remapped inputs.
            // This is a pure copy: clone span_id and merged_spans
            // verbatim (no cross-contributor merge for unfused nodes).
            let new_inputs: Vec<NodeId> = node
                .inputs
                .iter()
                .map(|&old| {
                    *id_map
                        .get(&old.0)
                        .unwrap_or_else(|| panic!("unmapped input {old:?}"))
                })
                .collect();
            let new_id = new_dag.add_node(
                node.decl,
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
            if !node.merged_spans.is_empty()
                && let Some(new_node) = new_dag.node_mut(new_id)
            {
                new_node.merged_spans = node.merged_spans.clone();
            }
            // chelis#384/#397: preserve (remapped) shape-derived `expand` deps.
            if !node.shape_deps.is_empty() {
                let mapped: Vec<NodeId> = node
                    .shape_deps
                    .iter()
                    .filter_map(|old| id_map.get(&old.0).copied())
                    .collect();
                if let Some(new_node) = new_dag.node_mut(new_id) {
                    new_node.shape_deps = mapped;
                }
            }
            if !node.result_claim_deps.is_empty() {
                let mapped = node
                    .result_claim_deps
                    .iter()
                    .map(|old| {
                        *id_map
                            .get(&old.0)
                            .unwrap_or_else(|| panic!("unmapped result claim dependency {old:?}"))
                    })
                    .collect();
                if let Some(new_node) = new_dag.node_mut(new_id) {
                    new_node.result_claim_deps = mapped;
                }
            }
            id_map.insert(old_id, new_id);
        }
    }

    // Remap roots.
    for &root in dag.roots() {
        if let Some(&new_root) = id_map.get(&root.0) {
            new_dag.add_root(new_root);
        }
    }

    let old_to_new = id_map
        .into_sorted()
        .into_iter()
        .map(|(old_id, new_id)| (NodeId(old_id), new_id))
        .collect();

    (new_dag, old_to_new)
}

/// Build a `FusedElem` op from a chain of nodes.
///
/// Returns the `RiscOp::FusedElem` and the list of external input NodeIds
/// (in the original DAG's ID space).
fn build_fused_elem(
    dag: &Dag,
    chain: &Chain,
    _id_map: &UnordMap<usize, NodeId>,
) -> (RiscOp, Vec<NodeId>) {
    let chain_set: chelis_unord::UnordSet<usize> = chain.nodes.iter().map(|n| n.0).collect();

    // Collect external inputs: inputs to chain nodes that are NOT other chain nodes.
    let mut external_inputs: Vec<NodeId> = Vec::new();
    let mut ext_index: UnordMap<usize, usize> = UnordMap::new(); // old_id → index in external_inputs

    // Also track: for each chain node, its step index.
    let mut step_index: UnordMap<usize, usize> = UnordMap::new();

    let mut steps: Vec<FusedStep> = Vec::new();

    for (si, &nid) in chain.nodes.iter().enumerate() {
        step_index.insert(nid.0, si);
        let node = dag.get(nid).unwrap();

        let mut input_indices = Vec::new();
        for &inp in &node.inputs {
            if chain_set.contains(&inp.0) {
                // Reference to a previous chain node.
                let prev_step = step_index[&inp.0];
                input_indices.push(FusedInput::PreviousStep(prev_step));
            } else {
                // External input.
                let ei = *ext_index.entry(inp.0).or_insert_with(|| {
                    let idx = external_inputs.len();
                    external_inputs.push(inp);
                    idx
                });
                input_indices.push(FusedInput::External(ei));
            }
        }

        steps.push(FusedStep {
            op: to_fused_step_op(&node.op),
            input_indices,
        });
    }

    (RiscOp::FusedElem { ops: steps }, external_inputs)
}

fn reusable_external_input(dag: &Dag, chain: &Chain, external_inputs: &[NodeId]) -> Option<NodeId> {
    let external: UnordSet<NodeId> = external_inputs.iter().copied().collect();
    let mut reusable = None;

    for &nid in &chain.nodes {
        let Some(candidate) = dag.get(nid).and_then(|node| node.reusable_input) else {
            continue;
        };
        if !external.contains(&candidate) {
            continue;
        }
        match reusable {
            None => reusable = Some(candidate),
            Some(existing) if existing == candidate => {}
            Some(_) => return None,
        }
    }

    reusable
}

/// Identify FusedElem nodes whose sole consumer is a reduction (Sum or MaxReduce).
///
/// These nodes can be inlined into the reduction's inner loop at emit time,
/// eliminating the intermediate buffer. Returns a set of node IDs that the
/// emitter should skip (no allocation, no standalone emission) and the
/// reduction should handle by inlining the fused steps.
pub fn reduction_inlined_fused_elems(dag: &Dag) -> UnordSet<NodeId> {
    let consumer_count = build_consumer_counts(dag);
    let mut inlined = UnordSet::new();

    for node in dag.nodes() {
        let is_reduction = matches!(node.op, RiscOp::Sum { .. } | RiscOp::MaxReduce { .. });
        if !is_reduction {
            continue;
        }
        // Reduction has exactly one input.
        let input_id = node.inputs[0];
        let input_node = dag.get(input_id).unwrap();
        if !matches!(input_node.op, RiscOp::FusedElem { .. }) {
            continue;
        }
        // Only inline if the FusedElem has exactly one consumer (the reduction).
        // consumer_count includes root references, so a root FusedElem that is also
        // consumed by a reduction would have count >= 2 — correctly excluded.
        if consumer_count[input_id.0] == 1 {
            inlined.insert(input_id);
        }
    }

    inlined
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{DimInfo, RiscOp, TensorType};

    fn vec_f32(n: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(n)],
            precision: chelis_types::types::Prim::F32,
        }
    }

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    fn vec_i64(n: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(n)],
            precision: chelis_types::types::Prim::Int64,
        }
    }

    #[test]
    fn consumer_counts_basic() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        let c = dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_f32(), None);
        dag.add_root(c);
        let counts = build_consumer_counts(&dag);
        assert_eq!(counts[a.0], 1); // consumed by c
        assert_eq!(counts[b.0], 1); // consumed by c
        assert_eq!(counts[c.0], 1); // root
    }

    #[test]
    fn multi_consumer_blocks_chain() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(4).precision, 2.0),
            vec![],
            vec_f32(4),
            None,
        );
        let shared = dag.add_node(decl, RiscOp::Add, vec![a, b], vec_f32(4), None);
        let _left = dag.add_node(decl, RiscOp::Neg, vec![shared], vec_f32(4), None);
        let _right = dag.add_node(decl, RiscOp::Exp, vec![shared], vec_f32(4), None);

        let counts = build_consumer_counts(&dag);
        assert_eq!(counts[shared.0], 2); // two consumers

        let chains = find_chains(&dag, &counts);
        // shared has 2 consumers → cannot be start of a chain that extends forward
        for chain in &chains {
            assert!(
                chain.nodes.len() <= 1,
                "multi-consumer node should not form a chain"
            );
        }
    }

    #[test]
    fn simple_chain_found() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(4).precision, 2.0),
            vec![],
            vec_f32(4),
            None,
        );
        let c = dag.add_node(decl, RiscOp::Add, vec![a, b], vec_f32(4), None);
        let d = dag.add_node(decl, RiscOp::Neg, vec![c], vec_f32(4), None);
        dag.add_root(d);

        let counts = build_consumer_counts(&dag);
        let chains = find_chains(&dag, &counts);
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].nodes.len(), 2); // add, neg
    }

    #[test]
    fn integer_abs_stays_materialized_until_typed_fused_kernels_exist() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec_i64(4),
            None,
        );
        let one = dag.add_node(
            decl,
            RiscOp::synth_const(vec_i64(4).precision, 1.0),
            vec![],
            vec_i64(4),
            None,
        );
        let abs = dag.add_node(decl, RiscOp::Abs, vec![x], vec_i64(4), None);
        let add = dag.add_node(decl, RiscOp::Add, vec![abs, one], vec_i64(4), None);
        dag.add_root(add);

        let fused = fuse(&dag);
        assert!(
            fused
                .nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::Abs)),
            "integer abs must reach the typed direct kernel instead of a float-only fused emitter"
        );
        assert!(
            !fused.nodes().iter().any(|node| {
                matches!(
                    &node.op,
                    RiscOp::FusedElem { ops }
                        if ops.iter().any(|step| step.op == FusedStepOp::Abs)
                )
            }),
            "integer abs cannot be fused until the fused kernel carries exact traps"
        );
    }

    #[test]
    fn fusion_preserves_unambiguous_external_reusable_input() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec_f32(4),
            None,
        );
        let bias = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        let add = dag.add_node(decl, RiscOp::Add, vec![x, bias], vec_f32(4), None);
        dag.set_reusable_input(add, x);
        let scale = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(4).precision, 2.0),
            vec![],
            vec_f32(4),
            None,
        );
        let mul = dag.add_node(decl, RiscOp::Mul, vec![add, scale], vec_f32(4), None);
        dag.add_root(mul);

        let fused = fuse_with_remap(&dag);
        let fused_id = fused.old_to_new[&mul];
        let fused_node = fused.dag.get(fused_id).unwrap();

        assert!(
            matches!(fused_node.op, RiscOp::FusedElem { .. }),
            "add/mul chain should fuse"
        );
        assert_eq!(
            fused_node.reusable_input,
            Some(fused.old_to_new[&x]),
            "fusion should preserve the reusable external input"
        );
    }

    #[test]
    fn fusion_drops_ambiguous_reusable_inputs() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec_f32(4),
            None,
        );
        let y = dag.add_node(
            decl,
            RiscOp::Load { name: "y".into() },
            vec![],
            vec_f32(4),
            None,
        );
        let add = dag.add_node(decl, RiscOp::Add, vec![x, y], vec_f32(4), None);
        dag.set_reusable_input(add, x);
        let mul = dag.add_node(decl, RiscOp::Mul, vec![add, y], vec_f32(4), None);
        dag.set_reusable_input(mul, y);
        dag.add_root(mul);

        let fused = fuse_with_remap(&dag);
        let fused_id = fused.old_to_new[&mul];
        let fused_node = fused.dag.get(fused_id).unwrap();

        assert!(
            matches!(fused_node.op, RiscOp::FusedElem { .. }),
            "add/mul chain should fuse"
        );
        assert_eq!(
            fused_node.reusable_input, None,
            "fusion must not choose between conflicting reusable external inputs"
        );
    }
}
