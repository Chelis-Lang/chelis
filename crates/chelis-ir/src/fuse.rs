//! DAG-to-DAG fusion pass.
//!
//! Greedy kernel fusion: merge adjacent single-consumer elementwise ops into
//! `FusedElem` nodes that emit as single GPU kernels. Also fuses elementwise
//! chains into trailing reductions (the elementwise ops become part of the
//! reduction's inner loop).
//!
//! **Invariant:** fusion never duplicates computation. A node with multiple
//! consumers is never absorbed into a fused chain.

use chelis_numeric_core::prim::Prim;
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
        // An activation is read by the node it owns, so a node that is some
        // node's activation stays materialized rather than being absorbed
        // into a chain as an intermediate.
        if let Some(activation) = node.owner.activation {
            counts[activation.0] += 1;
        }
    }
    // Roots count as consumers (they must be materialized).
    for &root in dag.roots() {
        counts[root.0] += 1;
    }
    counts
}

/// Returns true if the op is an elementwise op that can participate in fusion.
fn is_fusible_elementwise(seeds: &crate::dag::TrapSeeds<'_>, node: &DagNode) -> bool {
    // A checking operation under an activation substitutes operands its
    // checks accept where the activation is false (spec/10 section 3.2);
    // it stays its own kernel, where both lanes substitute them.
    if seeds.is_activation_gated(node) {
        return false;
    }
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
    fused_step_op(&node.op).is_some()
}

/// The FusedStepOp a fusible elementwise RiscOp becomes. This one table
/// decides both eligibility and conversion, so they cannot drift apart.
fn fused_step_op(op: &RiscOp) -> Option<FusedStepOp> {
    match op {
        RiscOp::Add => Some(FusedStepOp::Add),
        RiscOp::Sub => Some(FusedStepOp::Sub),
        RiscOp::Mul => Some(FusedStepOp::Mul),
        RiscOp::Div => Some(FusedStepOp::Div),
        RiscOp::Pow => Some(FusedStepOp::Pow),
        RiscOp::FloorDiv => Some(FusedStepOp::FloorDiv),
        RiscOp::TruncDiv => Some(FusedStepOp::TruncDiv),
        RiscOp::MaxElem => Some(FusedStepOp::MaxElem),
        RiscOp::MinElem => Some(FusedStepOp::MinElem),
        RiscOp::Neg => Some(FusedStepOp::Neg),
        RiscOp::Recip => Some(FusedStepOp::Recip),
        RiscOp::Exp => Some(FusedStepOp::Exp),
        RiscOp::Log => Some(FusedStepOp::Log),
        RiscOp::Sin => Some(FusedStepOp::Sin),
        RiscOp::Sqrt => Some(FusedStepOp::Sqrt),
        RiscOp::Cos => Some(FusedStepOp::Cos),
        RiscOp::Tan => Some(FusedStepOp::Tan),
        RiscOp::Atan => Some(FusedStepOp::Atan),
        RiscOp::Tanh => Some(FusedStepOp::Tanh),
        RiscOp::Erf => Some(FusedStepOp::Erf),
        RiscOp::Erfc => Some(FusedStepOp::Erfc),
        RiscOp::Abs => Some(FusedStepOp::Abs),
        RiscOp::Floor => Some(FusedStepOp::Floor),
        RiscOp::Ceil => Some(FusedStepOp::Ceil),
        RiscOp::Round => Some(FusedStepOp::Round),
        _ => None,
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

    let seeds = dag.trap_seeds();
    // Walk in topological order.
    for node in dag.nodes() {
        let id = node.id.0;
        if in_chain[id] {
            continue;
        }
        if !is_fusible_elementwise(&seeds, node) || is_claim_barrier(node) {
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
                // One fused kernel runs under one owner: a node under another
                // activation checks under that activation, so it starts its
                // own chain (spec/10 section 3.2).
                Some(c)
                    if is_fusible_elementwise(&seeds, c)
                        && !is_claim_barrier(c)
                        && c.owner == node.owner =>
                {
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
            // Every chain member has the same owner (see `find_chains`).
            let new_id = new_dag.add_node(
                dag.get(chain_out)
                    .expect("chain output exists")
                    .owner
                    .remap_with(|old| id_map.get(&old.0).copied()),
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
                node.owner.remap_with(|old| id_map.get(&old.0).copied()),
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
            op: fused_step_op(&node.op).expect("a chain holds only fusible elementwise ops"),
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

/// A float `Sum(Mul(Expand(x), Expand(y)))`, spec/05 section 4.1's matmul
/// graph among others, whose product a lane may compute leaf by leaf inside
/// the sum instead of materializing the expanded operands and their product
/// (chelis#3370).
///
/// Each leaf is still the product the `Mul` defines, entering the `Sum`'s
/// canonical tree ([05-OP-30]) in the same position, so the result bits are
/// those of the materialized graph. Float multiplication has no trap, and
/// both expansions keep their checked plans, so the traps are too.
///
/// Either `Mul` operand may instead be an exact float widening `Cast` of its
/// expansion, the conversion [05-OP-33]'s einsum makes before it multiplies
/// at a wider accumulator (chelis#3783); the lane converts each element as it
/// reads it. The `Sum` may likewise end in a float `Cast`, the narrowing to a
/// reduced-float result, which the lane applies to each finalized total as it
/// stores it. Neither conversion traps, so the bits and traps are still the
/// materialized graph's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InlinedProduct {
    pub sum: NodeId,
    pub product: NodeId,
    /// The `Mul`'s two `Expand` operands, in operand order.
    pub views: [NodeId; 2],
    /// The exact widening `Cast` between each expansion and the `Mul`, if
    /// any.
    pub casts: [Option<NodeId>; 2],
    /// The float `Cast` that is the `Sum`'s one consumer, if the lane stores
    /// the converted total in its place.
    pub narrowing: Option<NodeId>,
    /// The `Reshape` each expansion reads, if the expansion is its one
    /// element reader: the lane reads the reshape's dense operand in place
    /// through the reshape's metadata instead of storing it (chelis#3802).
    pub view_reshapes: [Option<NodeId>; 2],
    /// The `Reshape` that is the total's one consumer, if the lane stores the
    /// total directly in the reshape's slot: a reshape keeps row-major
    /// element order, so each total lands at its own flat position
    /// (chelis#3802).
    pub reshaped: Option<NodeId>,
}

impl InlinedProduct {
    /// The nodes the lane does not store: the product, its expansions and
    /// their conversions, and the sum when its narrowing stores instead.
    pub fn skipped(&self) -> Vec<NodeId> {
        let mut skipped = vec![self.product, self.views[0], self.views[1]];
        skipped.extend(self.casts.iter().flatten());
        if self.output() != self.sum {
            skipped.push(self.sum);
        }
        if let (Some(narrowing), Some(_)) = (self.narrowing, self.reshaped) {
            skipped.push(narrowing);
        }
        skipped
    }

    /// The node whose position and storage hold the lane's kernel.
    pub fn output(&self) -> NodeId {
        self.reshaped.or(self.narrowing).unwrap_or(self.sum)
    }

    /// The node holding the total before any trailing reshape: the sum, or
    /// its narrowing.
    pub fn total(&self) -> NodeId {
        self.narrowing.unwrap_or(self.sum)
    }
}

/// Whether every value of `from` converts exactly to `to`, a wider float.
fn exact_float_widening(from: Prim, to: Prim) -> bool {
    matches!(
        (from, to),
        (Prim::F16 | Prim::Bf16, Prim::F32 | Prim::F64) | (Prim::F32, Prim::F64)
    )
}

/// Every [`InlinedProduct`] in `dag`.
///
/// A match requires that nothing else reads the intermediates: the product,
/// each expansion and each conversion have exactly one consumer, are no other
/// node's shape or claim dependency, and carry no activation, and no `Drop`
/// releases an expansion's operand before the kernel reads it. The product
/// shares its rank with the expansions and its dtype with the expansions or
/// their widening casts. Each lane must check actual operand-shape
/// agreement at the original product position, before its result claims,
/// even when the expansions' declared dimension identities differ. An extent claim an intermediate owns stays
/// the lane's to check at that intermediate's position, from the extents
/// the expansions' checked plans hold.
pub fn reduction_inlined_products(dag: &Dag) -> Vec<InlinedProduct> {
    let consumer_count = build_consumer_counts(dag);
    let mut dependency = vec![false; dag.len()];
    let mut consumer = vec![None; dag.len()];
    // Reads of a node's elements, as opposed to its extents.
    let mut payload_readers = vec![0usize; dag.len()];
    for node in dag.nodes() {
        for id in node.shape_deps.iter().chain(&node.result_claim_deps) {
            dependency[id.0] = true;
        }
        for (slot, id) in node.inputs.iter().enumerate() {
            consumer[id.0] = Some(node.id);
            if crate::contraction_views::payload_slot(node, slot) {
                payload_readers[id.0] += 1;
            }
        }
    }
    for root in dag.roots() {
        payload_readers[root.0] += 1;
    }
    let private = |node: &DagNode| {
        consumer_count[node.id.0] == 1 && !dependency[node.id.0] && node.owner.activation.is_none()
    };
    let mut first_drop = vec![usize::MAX; dag.len()];
    for node in dag.nodes() {
        if matches!(node.op, RiscOp::Drop) {
            for input in &node.inputs {
                first_drop[input.0] = first_drop[input.0].min(node.id.0);
            }
        }
    }
    let mut found = Vec::new();
    for sum in dag.nodes() {
        if !matches!(sum.op, RiscOp::Sum { .. })
            || sum.inputs.len() != 1
            || sum.owner.activation.is_some()
        {
            continue;
        }
        let Some(product) = dag.get(sum.inputs[0]) else {
            continue;
        };
        let precision = product.output_type.precision;
        if !matches!(product.op, RiscOp::Mul)
            || product.inputs.len() != 2
            || !matches!(precision, Prim::F32 | Prim::F64 | Prim::Bf16 | Prim::F16)
            || !private(product)
        {
            continue;
        }
        let rank = product.output_type.dims.len();
        // An operand is an expansion at the product dtype, or an exact
        // widening of an expansion to it.
        let operand = |id: NodeId| -> Option<(NodeId, Option<NodeId>)> {
            let node = dag.get(id)?;
            let (view, cast) = match node.op {
                RiscOp::Cast { new_precision }
                    if new_precision == precision && node.inputs.len() == 1 && private(node) =>
                {
                    (dag.get(node.inputs[0])?, Some(node.id))
                }
                _ => (node, None),
            };
            let converts = match cast {
                Some(_) => exact_float_widening(view.output_type.precision, precision),
                None => view.output_type.precision == precision,
            };
            (matches!(view.op, RiscOp::Expand { .. })
                && private(view)
                && converts
                && view.output_type.dims.len() == rank)
                .then_some((view.id, cast))
        };
        let (Some((lhs, lhs_cast)), Some((rhs, rhs_cast))) =
            (operand(product.inputs[0]), operand(product.inputs[1]))
        else {
            continue;
        };
        // The kernel reads the expansions' operands where it runs.
        let released_before = |kernel: NodeId| {
            [lhs, rhs]
                .iter()
                .filter_map(|view| dag.get(*view))
                .flat_map(|view| &view.inputs)
                .any(|id| first_drop[id.0] < kernel.0)
        };
        if released_before(sum.id) {
            continue;
        }
        // Only `Drop`s, which never trap, may run between the sum and its
        // narrowing, so running the kernel there moves no trap past another.
        let narrowing = consumer[sum.id.0]
            .filter(|_| private(sum))
            .and_then(|id| dag.get(id))
            .filter(|cast| {
                matches!(cast.op, RiscOp::Cast { new_precision } if new_precision.is_float())
                    && cast.owner.activation.is_none()
                    && dag.nodes()[sum.id.0 + 1..cast.id.0]
                        .iter()
                        .all(|between| matches!(between.op, RiscOp::Drop))
                    && !released_before(cast.id)
            })
            .map(|cast| cast.id);
        // A private reshape of the total may hold it directly, under the same
        // conditions as the narrowing. Its target is literal or symbolic, so
        // it reads no other node.
        let total = narrowing.unwrap_or(sum.id);
        let reshaped = dag
            .get(total)
            .filter(|node| private(node))
            .and_then(|_| consumer[total.0])
            .and_then(|id| dag.get(id))
            .filter(|reshape| {
                matches!(reshape.op, RiscOp::Reshape { .. })
                    && reshape.inputs.len() == 1
                    && reshape.owner.activation.is_none()
                    && dag.nodes()[total.0 + 1..reshape.id.0]
                        .iter()
                        .all(|between| matches!(between.op, RiscOp::Drop))
                    && !released_before(reshape.id)
            })
            .map(|reshape| reshape.id);
        // An expansion's reshape of a dense operand reorders no element, so
        // the kernel can read the operand through it, when the expansion is
        // its one element reader; other nodes may read its extents.
        let view_reshape = |view: NodeId| {
            let source = dag.get(dag.get(view)?.inputs[0])?;
            (matches!(source.op, RiscOp::Reshape { .. })
                && payload_readers[source.id.0] == 1
                && !dependency[source.id.0]
                && source.owner.activation.is_none()
                && source.output_type.precision.is_float())
            .then_some(source.id)
        };
        found.push(InlinedProduct {
            sum: sum.id,
            product: product.id,
            views: [lhs, rhs],
            casts: [lhs_cast, rhs_cast],
            narrowing,
            view_reshapes: [view_reshape(lhs), view_reshape(rhs)],
            reshaped,
        });
    }
    found
}

/// A private Cartesian index matrix consumed by a flat gather. The descriptor
/// records existing nodes; lanes retain their checks at those node positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CartesianGatherIndices {
    pub gather: NodeId,
    pub index_sum: NodeId,
    pub views: [NodeId; 2],
    /// One data reader; auxiliary Expand InputAxis readers observe extents.
    pub private_gather: bool,
}

pub fn cartesian_gather_indices(dag: &Dag) -> Vec<CartesianGatherIndices> {
    let counts = build_consumer_counts(dag);
    // Auxiliary InputAxis reads observe only extents. A window's one data
    // reader can coexist with those reads without requiring dense windows.
    let mut data_counts = counts.clone();
    for node in dag.nodes() {
        if let RiscOp::Expand {
            size: crate::dag::RtDim::InputAxis { tensor, .. },
            ..
        } = &node.op
            && *tensor != 0
            && let Some(input) = node.inputs.get(*tensor)
        {
            data_counts[input.0] -= 1;
        }
    }
    let mut dependency = vec![false; dag.len()];
    for node in dag.nodes() {
        for id in node.shape_deps.iter().chain(&node.result_claim_deps) {
            dependency[id.0] = true;
        }
    }
    let private = |node: &DagNode| {
        counts[node.id.0] == 1 && !dependency[node.id.0] && node.owner.activation.is_none()
    };
    let mut found = Vec::new();
    for gather in dag.nodes() {
        if !matches!(
            gather.op,
            RiscOp::Gather {
                axis: 0,
                batch_rank: 0
            }
        ) || gather.inputs.len() != 2
            || gather.owner.activation.is_some()
        {
            continue;
        }
        let (Some(source), Some(sum)) = (dag.get(gather.inputs[0]), dag.get(gather.inputs[1]))
        else {
            continue;
        };
        if source.output_type.dims.len() != 1
            || !matches!(sum.op, RiscOp::Add)
            || sum.inputs.len() != 2
            || !sum.output_type.precision.is_integer()
            || !private(sum)
        {
            continue;
        }
        let (Some(row), Some(col)) = (dag.get(sum.inputs[0]), dag.get(sum.inputs[1])) else {
            continue;
        };
        if !matches!(row.op, RiscOp::Expand { axis: 1, .. })
            || !matches!(col.op, RiscOp::Expand { axis: 0, .. })
            || row.inputs.is_empty()
            || col.inputs.is_empty()
            || !private(row)
            || !private(col)
            || row.output_type != sum.output_type
            || col.output_type != sum.output_type
            || sum.output_type.dims.len() != 2
        {
            continue;
        }
        if ![row, col].iter().all(|view| {
            dag.get(view.inputs[0]).is_some_and(|base| {
                base.output_type.dims.len() == 1
                    && base.output_type.precision == sum.output_type.precision
            })
        }) {
            continue;
        }
        found.push(CartesianGatherIndices {
            gather: gather.id,
            index_sum: sum.id,
            views: [row.id, col.id],
            private_gather: data_counts[gather.id.0] == 1
                && !dependency[gather.id.0]
                && gather.owner.activation.is_none(),
        });
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{DimInfo, RiscOp, TensorType};

    fn vec_f32(n: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(n)],
            precision: chelis_numeric_core::prim::Prim::F32,
        }
    }

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    fn vec_i64(n: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(n)],
            precision: chelis_numeric_core::prim::Prim::Int64,
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

    /// spec/05 section 4.1's `[2,3] x [3,4]` matmul graph at `prim`, with
    /// `extra` applied before the sum's root is added. Returns the DAG, the
    /// two operand loads, both expansions, the product, and the sum.
    fn matmul_graph(prim: Prim, extra: impl FnOnce(&mut Dag, [NodeId; 5])) -> (Dag, [NodeId; 6]) {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let ty = |dims: &[usize]| TensorType {
            dims: dims.iter().copied().map(DimInfo::Lit).collect(),
            precision: prim,
        };
        let a = dag.add_node(
            decl,
            RiscOp::Load { name: "a".into() },
            vec![],
            ty(&[2, 3]),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::Load { name: "b".into() },
            vec![],
            ty(&[3, 4]),
            None,
        );
        let expand = |axis, size| RiscOp::Expand {
            axis,
            size: crate::dag::RtDim::Lit(size),
        };
        let lhs = dag.add_node(decl, expand(2, 4), vec![a], ty(&[2, 3, 4]), None);
        let rhs = dag.add_node(decl, expand(0, 2), vec![b], ty(&[2, 3, 4]), None);
        let product = dag.add_node(decl, RiscOp::Mul, vec![lhs, rhs], ty(&[2, 3, 4]), None);
        extra(&mut dag, [a, b, lhs, rhs, product]);
        let sum = dag.add_node(
            decl,
            RiscOp::Sum {
                axis: 1,
                accumulator: if prim.is_float() && !matches!(prim, Prim::Bf16 | Prim::F16) {
                    prim
                } else if prim.is_float() {
                    Prim::F32
                } else {
                    Prim::Int64
                },
            },
            vec![product],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(4)],
                precision: if matches!(prim, Prim::Bf16 | Prim::F16) {
                    Prim::F32
                } else if prim.is_float() {
                    prim
                } else {
                    Prim::Int64
                },
            },
            None,
        );
        dag.add_root(sum);
        (dag, [a, b, lhs, rhs, product, sum])
    }

    #[test]
    fn matmul_products_are_inlined_at_every_float_dtype() {
        for prim in [Prim::F32, Prim::F64, Prim::Bf16, Prim::F16] {
            let (dag, [_, _, lhs, rhs, product, sum]) = matmul_graph(prim, |_, _| {});
            assert_eq!(
                reduction_inlined_products(&dag),
                vec![InlinedProduct {
                    sum,
                    product,
                    views: [lhs, rhs],
                    casts: [None, None],
                    narrowing: None,
                    view_reshapes: [None, None],
                    reshaped: None,
                }],
                "{prim:?}"
            );
        }
    }

    #[test]
    fn wildcard_products_require_original_position_agreement_without_storage() {
        let (mut dag, [_, _, lhs, rhs, product, sum]) = matmul_graph(Prim::F32, |_, _| {});
        for id in [lhs, rhs, product] {
            let node = dag.node_mut(id).unwrap();
            node.output_type.dims = (0..3)
                .map(|axis| DimInfo::Named(format!("__wild_{}_{}", id.0, axis), None))
                .collect();
        }
        assert_eq!(
            reduction_inlined_products(&dag),
            vec![InlinedProduct {
                sum,
                product,
                views: [lhs, rhs],
                casts: [None, None],
                narrowing: None,
                view_reshapes: [None, None],
                reshaped: None,
            }]
        );
        dag.add_root(rhs);
        assert!(reduction_inlined_products(&dag).is_empty());
    }

    #[test]
    fn observed_or_trapping_products_are_not_inlined() {
        type Extra = fn(&mut Dag, [NodeId; 5]);
        let observed: [(&str, Prim, Extra); 6] = [
            // Integer products trap on overflow before any sum addition.
            ("integer", Prim::Int64, |_, _| {}),
            ("product root", Prim::F32, |dag, ids| dag.add_root(ids[4])),
            ("expansion root", Prim::F32, |dag, ids| dag.add_root(ids[2])),
            ("product read twice", Prim::F32, |dag, ids| {
                let decl = dag.declare("other");
                let ty = dag.get(ids[4]).unwrap().output_type.clone();
                let neg = dag.add_node(decl, RiscOp::Neg, vec![ids[4]], ty, None);
                dag.add_root(neg);
            }),
            ("product is a shape dependency", Prim::F32, |dag, ids| {
                let decl = dag.declare("other");
                let ty = dag.get(ids[0]).unwrap().output_type.clone();
                let neg = dag.add_node(decl, RiscOp::Neg, vec![ids[0]], ty, None);
                dag.node_mut(neg).unwrap().shape_deps.push(ids[4]);
                dag.add_root(neg);
            }),
            ("operand released before the sum", Prim::F32, |dag, ids| {
                let decl = dag.declare("other");
                let ty = dag.get(ids[0]).unwrap().output_type.clone();
                dag.add_node(decl, RiscOp::Drop, vec![ids[0]], ty, None);
            }),
        ];
        for (case, prim, extra) in observed {
            let (dag, _) = matmul_graph(prim, extra);
            assert_eq!(reduction_inlined_products(&dag), vec![], "{case}");
        }
    }

    /// [05-OP-33]'s einsum at a wider accumulator: each `stored` expansion
    /// widens to `accumulator`, which the product and sum hold, and the sum
    /// narrows to `stored` after `between` runs, which also sees an
    /// unrelated load. Returns the DAG, the two loads, both expansions, both
    /// widenings, the product, the sum and the narrowing.
    fn converted_graph(
        stored: Prim,
        accumulator: Prim,
        between: impl FnOnce(&mut Dag, [NodeId; 3]),
    ) -> (Dag, [NodeId; 9]) {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let ty = |dims: &[usize], precision| TensorType {
            dims: dims.iter().copied().map(DimInfo::Lit).collect(),
            precision,
        };
        let load = |dag: &mut Dag, name: &str, dims: &[usize]| {
            dag.add_node(
                decl,
                RiscOp::Load { name: name.into() },
                vec![],
                ty(dims, stored),
                None,
            )
        };
        let unrelated = load(&mut dag, "c", &[1]);
        let a = load(&mut dag, "a", &[2, 3]);
        let b = load(&mut dag, "b", &[3, 4]);
        let expand = |axis, size| RiscOp::Expand {
            axis,
            size: crate::dag::RtDim::Lit(size),
        };
        let widen = RiscOp::Cast {
            new_precision: accumulator,
        };
        let lhs = dag.add_node(decl, expand(2, 4), vec![a], ty(&[2, 3, 4], stored), None);
        let rhs = dag.add_node(decl, expand(0, 2), vec![b], ty(&[2, 3, 4], stored), None);
        let wide_lhs = dag.add_node(
            decl,
            widen.clone(),
            vec![lhs],
            ty(&[2, 3, 4], accumulator),
            None,
        );
        let wide_rhs = dag.add_node(decl, widen, vec![rhs], ty(&[2, 3, 4], accumulator), None);
        let product = dag.add_node(
            decl,
            RiscOp::Mul,
            vec![wide_lhs, wide_rhs],
            ty(&[2, 3, 4], accumulator),
            None,
        );
        let sum = dag.add_node(
            decl,
            RiscOp::Sum {
                axis: 1,
                accumulator,
            },
            vec![product],
            ty(&[2, 4], accumulator),
            None,
        );
        between(&mut dag, [a, b, unrelated]);
        let narrowing = dag.add_node(
            decl,
            RiscOp::Cast {
                new_precision: stored,
            },
            vec![sum],
            ty(&[2, 4], stored),
            None,
        );
        dag.add_root(narrowing);
        (
            dag,
            [a, b, lhs, rhs, wide_lhs, wide_rhs, product, sum, narrowing],
        )
    }

    /// chelis#3783: an exact widening of each expansion and a float cast of
    /// the sum join the inlined product, at every widening the float
    /// dtypes have.
    #[test]
    fn exact_widenings_and_the_sum_narrowing_are_inlined() {
        for (stored, accumulator) in [
            (Prim::F16, Prim::F32),
            (Prim::Bf16, Prim::F32),
            (Prim::F16, Prim::F64),
            (Prim::Bf16, Prim::F64),
            (Prim::F32, Prim::F64),
        ] {
            let (dag, [_, _, lhs, rhs, wide_lhs, wide_rhs, product, sum, narrowing]) =
                converted_graph(stored, accumulator, |_, _| {});
            let inlined = InlinedProduct {
                sum,
                product,
                views: [lhs, rhs],
                casts: [Some(wide_lhs), Some(wide_rhs)],
                narrowing: Some(narrowing),
                view_reshapes: [None, None],
                reshaped: None,
            };
            assert_eq!(
                reduction_inlined_products(&dag),
                vec![inlined],
                "{stored:?}"
            );
            assert_eq!(inlined.output(), narrowing);
            assert!(inlined.skipped().contains(&sum));
        }
        // A `Drop` that releases nothing the kernel reads may run between.
        let (dag, [.., narrowing]) = converted_graph(Prim::F16, Prim::F32, |dag, [.., other]| {
            let decl = dag.declare("other");
            let ty = dag.get(other).unwrap().output_type.clone();
            dag.add_node(decl, RiscOp::Drop, vec![other], ty, None);
        });
        assert_eq!(
            reduction_inlined_products(&dag)[0].narrowing,
            Some(narrowing)
        );
    }

    /// A conversion that is not exact, that something else reads, or that
    /// would move the kernel past a release or another operation stays a
    /// node of its own.
    #[test]
    fn inexact_observed_or_displaced_conversions_are_not_inlined() {
        // Neither reduced float holds the other exactly, and a narrowing
        // is no widening.
        for (stored, accumulator) in [
            (Prim::F16, Prim::Bf16),
            (Prim::Bf16, Prim::F16),
            (Prim::F64, Prim::F32),
        ] {
            let (dag, _) = converted_graph(stored, accumulator, |_, _| {});
            assert_eq!(
                reduction_inlined_products(&dag),
                vec![],
                "{stored:?} to {accumulator:?}"
            );
        }
        // A widening read twice is stored.
        let (mut dag, [.., wide_lhs, _, _, _, _]) =
            converted_graph(Prim::F16, Prim::F32, |_, _| {});
        dag.add_root(wide_lhs);
        assert_eq!(reduction_inlined_products(&dag), vec![]);
        // A sum read twice stores itself, and its cast stays a cast.
        let (mut dag, [.., sum, _]) = converted_graph(Prim::F16, Prim::F32, |_, _| {});
        dag.add_root(sum);
        let inlined = reduction_inlined_products(&dag);
        assert_eq!(inlined.len(), 1);
        assert_eq!(inlined[0].narrowing, None);
        // An operand released, or any other operation run, between the sum
        // and its narrowing keeps the kernel at the sum.
        type Between = fn(&mut Dag, [NodeId; 3]);
        let displaced: [(&str, Between); 2] = [
            ("operand released", |dag, [a, ..]| {
                let decl = dag.declare("other");
                let ty = dag.get(a).unwrap().output_type.clone();
                dag.add_node(decl, RiscOp::Drop, vec![a], ty, None);
            }),
            ("another operation", |dag, [.., other]| {
                let decl = dag.declare("other");
                let ty = dag.get(other).unwrap().output_type.clone();
                let neg = dag.add_node(decl, RiscOp::Neg, vec![other], ty, None);
                dag.add_root(neg);
            }),
        ];
        for (case, between) in displaced {
            let (dag, [.., sum, _]) = converted_graph(Prim::F16, Prim::F32, between);
            let inlined = reduction_inlined_products(&dag);
            assert_eq!(inlined.len(), 1, "{case}");
            assert_eq!(
                (inlined[0].sum, inlined[0].narrowing),
                (sum, None),
                "{case}"
            );
        }
    }

    /// A row-flattened product sum: `x [2, 3, 4]` reshaped to `[6, 4]`,
    /// expanded against `w [4, 5]`, summed over the shared axis, and the
    /// `[6, 5]` total reshaped back to `[2, 3, 5]`, with `between` run before
    /// the trailing reshape. Returns the DAG, the row reshape and the trailing
    /// reshape.
    fn reshaped_graph(between: impl FnOnce(&mut Dag, [NodeId; 3])) -> (Dag, [NodeId; 2]) {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let ty = |dims: &[usize]| TensorType {
            dims: dims.iter().copied().map(DimInfo::Lit).collect(),
            precision: Prim::F32,
        };
        let lit = |dims: &[usize]| dims.iter().map(|&n| crate::dag::RtDim::Lit(n)).collect();
        let load = |dag: &mut Dag, name: &str, dims: &[usize]| {
            dag.add_node(
                decl,
                RiscOp::Load { name: name.into() },
                vec![],
                ty(dims),
                None,
            )
        };
        let x = load(&mut dag, "x", &[2, 3, 4]);
        let w = load(&mut dag, "w", &[4, 5]);
        let rows = dag.add_node(
            decl,
            RiscOp::Reshape {
                new_shape: lit(&[6, 4]),
            },
            vec![x],
            ty(&[6, 4]),
            None,
        );
        let expand = |axis, size| RiscOp::Expand {
            axis,
            size: crate::dag::RtDim::Lit(size),
        };
        let lhs = dag.add_node(decl, expand(2, 5), vec![rows], ty(&[6, 4, 5]), None);
        let rhs = dag.add_node(decl, expand(0, 6), vec![w], ty(&[6, 4, 5]), None);
        let product = dag.add_node(decl, RiscOp::Mul, vec![lhs, rhs], ty(&[6, 4, 5]), None);
        let sum = dag.add_node(
            decl,
            RiscOp::Sum {
                axis: 1,
                accumulator: Prim::F32,
            },
            vec![product],
            ty(&[6, 5]),
            None,
        );
        between(&mut dag, [x, rows, sum]);
        let out = dag.add_node(
            decl,
            RiscOp::Reshape {
                new_shape: lit(&[2, 3, 5]),
            },
            vec![sum],
            ty(&[2, 3, 5]),
            None,
        );
        dag.add_root(out);
        (dag, [rows, out])
    }

    /// chelis#3802: the expansion's reshape is read in place and the total's
    /// reshape holds it, so the kernel runs at the trailing reshape.
    #[test]
    fn row_and_output_reshapes_join_the_inlined_product() {
        let (dag, [rows, out]) = reshaped_graph(|_, _| {});
        let inlined = reduction_inlined_products(&dag);
        assert_eq!(inlined.len(), 1);
        assert_eq!(inlined[0].view_reshapes, [Some(rows), None]);
        assert_eq!(inlined[0].reshaped, Some(out));
        assert_eq!(inlined[0].output(), out);
        assert!(inlined[0].skipped().contains(&inlined[0].sum));
        assert!(!inlined[0].skipped().contains(&rows));
        // Another node may read the row reshape's extents.
        let (dag, [rows, _]) = reshaped_graph(|dag, [x, rows, _]| {
            let decl = dag.declare("other");
            let ty = dag.get(x).unwrap().output_type.clone();
            let read = dag.add_node(
                decl,
                RiscOp::Expand {
                    axis: 0,
                    size: crate::dag::RtDim::InputAxis {
                        tensor: 1,
                        axis: crate::dag::RtAxis::Lit(0),
                    },
                },
                vec![x, rows],
                TensorType {
                    dims: [vec![DimInfo::Lit(6)], ty.dims].concat(),
                    precision: ty.precision,
                },
                None,
            );
            dag.add_root(read);
        });
        assert_eq!(
            reduction_inlined_products(&dag)[0].view_reshapes[0],
            Some(rows)
        );
    }

    /// A reshape something else reads as data, a total read twice, or another
    /// operation between the total and its reshape keeps that reshape a node
    /// of its own.
    #[test]
    fn observed_or_displaced_reshapes_stay_their_own_nodes() {
        let (dag, [_, _]) = reshaped_graph(|dag, [_, rows, _]| dag.add_root(rows));
        assert_eq!(
            reduction_inlined_products(&dag)[0].view_reshapes,
            [None, None]
        );
        let (dag, [_, _]) = reshaped_graph(|dag, [_, _, sum]| dag.add_root(sum));
        assert_eq!(reduction_inlined_products(&dag)[0].reshaped, None);
        let (dag, [_, _]) = reshaped_graph(|dag, [x, _, _]| {
            let decl = dag.declare("other");
            let ty = dag.get(x).unwrap().output_type.clone();
            let neg = dag.add_node(decl, RiscOp::Neg, vec![x], ty, None);
            dag.add_root(neg);
        });
        let inlined = reduction_inlined_products(&dag);
        assert_eq!(
            (inlined[0].reshaped, inlined[0].output()),
            (None, inlined[0].sum)
        );
    }
}
