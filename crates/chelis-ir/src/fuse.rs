//! DAG-to-DAG fusion pass.
//!
//! Greedy kernel fusion: merge adjacent single-consumer elementwise ops into
//! `FusedElem` nodes that emit as single GPU kernels. Also fuses elementwise
//! chains into trailing reductions (the elementwise ops become part of the
//! reduction's inner loop).
//!
//! **Invariant:** fusion never duplicates computation. A node with multiple
//! consumers is never absorbed into a fused chain.

use std::collections::{HashMap, HashSet};

use crate::dag::{Dag, FusedInput, FusedStep, FusedStepOp, NodeId, RiscOp};

/// Result of running the fusion pass.
pub struct FuseResult {
    /// Fused DAG.
    pub dag: Dag,
    /// Mapping from original node IDs to node IDs in the fused DAG.
    pub old_to_new: HashMap<NodeId, NodeId>,
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
            old_to_new: HashMap::new(),
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
fn is_fusible_elementwise(op: &RiscOp) -> bool {
    matches!(
        op,
        RiscOp::Add
            | RiscOp::Mul
            | RiscOp::MaxElem
            | RiscOp::CmpLt
            | RiscOp::Neg
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
    )
}

/// Convert a RiscOp to its FusedStepOp equivalent.
fn to_fused_step_op(op: &RiscOp) -> FusedStepOp {
    match op {
        RiscOp::Add => FusedStepOp::Add,
        RiscOp::Mul => FusedStepOp::Mul,
        RiscOp::MaxElem => FusedStepOp::MaxElem,
        RiscOp::CmpLt => FusedStepOp::CmpLt,
        RiscOp::Neg => FusedStepOp::Neg,
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

    // Walk in topological order.
    for node in dag.nodes() {
        let id = node.id.0;
        if in_chain[id] {
            continue;
        }
        if !is_fusible_elementwise(&node.op) {
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
                Some(c) if is_fusible_elementwise(&c.op) => {
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
fn rebuild_with_fusion(dag: &Dag, chains: &[Chain]) -> (Dag, HashMap<NodeId, NodeId>) {
    // Map old node ID → chain index (if part of a chain).
    let mut node_to_chain: HashMap<usize, usize> = HashMap::new();
    // For each chain, which node is the "representative" (last node, produces output).
    let mut chain_output: HashMap<usize, NodeId> = HashMap::new();

    for (ci, chain) in chains.iter().enumerate() {
        for &nid in &chain.nodes {
            node_to_chain.insert(nid.0, ci);
        }
        chain_output.insert(ci, *chain.nodes.last().unwrap());
    }

    let mut new_dag = Dag::new();
    let mut id_map: HashMap<usize, NodeId> = HashMap::new();

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
            let (fused_op, external_inputs) = build_fused_elem(dag, chain, &id_map);

            let remapped_inputs: Vec<NodeId> = external_inputs
                .iter()
                .map(|&old| {
                    *id_map
                        .get(&old.0)
                        .unwrap_or_else(|| panic!("unmapped input {old:?} in fusion"))
                })
                .collect();

            let output_type = dag.get(chain_out).unwrap().output_type.clone();
            let new_id = new_dag.add_node(fused_op, remapped_inputs, output_type);
            // Map ALL chain nodes to this new ID (consumers reference chain internals).
            for &nid in &chain.nodes {
                id_map.insert(nid.0, new_id);
            }
        } else {
            // Not part of a chain — emit as-is with remapped inputs.
            let new_inputs: Vec<NodeId> = node
                .inputs
                .iter()
                .map(|&old| {
                    *id_map
                        .get(&old.0)
                        .unwrap_or_else(|| panic!("unmapped input {old:?}"))
                })
                .collect();
            let new_id = new_dag.add_node(node.op.clone(), new_inputs, node.output_type.clone());
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
    _id_map: &HashMap<usize, NodeId>,
) -> (RiscOp, Vec<NodeId>) {
    let chain_set: std::collections::HashSet<usize> = chain.nodes.iter().map(|n| n.0).collect();

    // Collect external inputs: inputs to chain nodes that are NOT other chain nodes.
    let mut external_inputs: Vec<NodeId> = Vec::new();
    let mut ext_index: HashMap<usize, usize> = HashMap::new(); // old_id → index in external_inputs

    // Also track: for each chain node, its step index.
    let mut step_index: HashMap<usize, usize> = HashMap::new();

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

/// Identify FusedElem nodes whose sole consumer is a reduction (Sum or MaxReduce).
///
/// These nodes can be inlined into the reduction's inner loop at emit time,
/// eliminating the intermediate buffer. Returns a set of node IDs that the
/// emitter should skip (no allocation, no standalone emission) and the
/// reduction should handle by inlining the fused steps.
pub fn reduction_inlined_fused_elems(dag: &Dag) -> HashSet<NodeId> {
    let consumer_count = build_consumer_counts(dag);
    let mut inlined = HashSet::new();

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

    #[test]
    fn consumer_counts_basic() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
        dag.add_root(c);
        let counts = build_consumer_counts(&dag);
        assert_eq!(counts[a.0], 1); // consumed by c
        assert_eq!(counts[b.0], 1); // consumed by c
        assert_eq!(counts[c.0], 1); // root
    }

    #[test]
    fn multi_consumer_blocks_chain() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(4));
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(4));
        let shared = dag.add_node(RiscOp::Add, vec![a, b], vec_f32(4));
        let _left = dag.add_node(RiscOp::Neg, vec![shared], vec_f32(4));
        let _right = dag.add_node(RiscOp::Exp, vec![shared], vec_f32(4));

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
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(4));
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(4));
        let c = dag.add_node(RiscOp::Add, vec![a, b], vec_f32(4));
        let d = dag.add_node(RiscOp::Neg, vec![c], vec_f32(4));
        dag.add_root(d);

        let counts = build_consumer_counts(&dag);
        let chains = find_chains(&dag, &counts);
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].nodes.len(), 2); // add, neg
    }
}
