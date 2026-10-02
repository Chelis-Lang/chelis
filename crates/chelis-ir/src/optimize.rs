//! Basic DAG optimization passes.

use chelis_unord::UnordMap;

use crate::dag::{ComparisonKind, Dag, LogicalKind, NodeId, Owner, RiscOp};

type CseKey = (Owner, String, Vec<NodeId>, Vec<NodeId>, Vec<NodeId>);

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
    let mut replacements: Vec<(NodeId, chelis_types::ScalarValue, Vec<String>)> = Vec::new();
    let claimed_producers = crate::axis_sources::claimed_producers(dag);

    // Direct subtraction and value extrema fold through their exact typed
    // kernels below. The remaining legacy fold set computes on the f64 wide
    // image (the chelis#680 residue) and FINALIZES the result at the node's
    // dtype through the sealed module. Two decline rules keep that older set
    // conservative per the section C2 contract: an integer payload whose f64
    // image is not exact declines (never bake a collapsed value in,
    // chelis#856), and a result that does not finalize at the node's dtype
    // declines (never bake a trap away nor in - the runtime evaluates the
    // unfolded graph and traps with its full diagnostic).
    let wide_image = |value: &chelis_types::ScalarValue| -> Option<f64> {
        if let Some(i) = value.as_i64_exact()
            && (i as f64) as i128 != i as i128
        {
            return None;
        }
        Some(value.as_f64_lossy())
    };

    for node in dag.nodes() {
        if claimed_producers[node.id.0] {
            continue;
        }
        if node.inputs.len() == 2 {
            let lhs = dag.get(node.inputs[0]);
            let rhs = dag.get(node.inputs[1]);
            if let (Some(l), Some(r)) = (lhs, rhs)
                && let (RiscOp::Const { value: lval }, RiscOp::Const { value: rval }) =
                    (&l.op, &r.op)
            {
                let direct = match &node.op {
                    RiscOp::Sub => Some(if node.output_type.precision.is_integer() {
                        chelis_types::int_binop(chelis_types::IntBinOp::Sub, *lval, *rval)
                    } else {
                        chelis_types::float_binop(chelis_types::FloatBinOp::Sub, *lval, *rval)
                    }),
                    RiscOp::MaxElem => Some(if node.output_type.precision.is_integer() {
                        chelis_types::int_binop(chelis_types::IntBinOp::Max, *lval, *rval)
                    } else {
                        chelis_types::float_binop(chelis_types::FloatBinOp::Max, *lval, *rval)
                    }),
                    RiscOp::MinElem => Some(if node.output_type.precision.is_integer() {
                        chelis_types::int_binop(chelis_types::IntBinOp::Min, *lval, *rval)
                    } else {
                        chelis_types::float_binop(chelis_types::FloatBinOp::Min, *lval, *rval)
                    }),
                    RiscOp::Compare(kind) => Some(
                        chelis_types::compare_scalars(
                            match kind {
                                ComparisonKind::CmpLt | ComparisonKind::Lt => {
                                    chelis_types::CompareOp::Lt
                                }
                                ComparisonKind::Eq => chelis_types::CompareOp::Eq,
                                ComparisonKind::Neq => chelis_types::CompareOp::Ne,
                                ComparisonKind::Gt => chelis_types::CompareOp::Gt,
                                ComparisonKind::Gte => chelis_types::CompareOp::Gte,
                                ComparisonKind::Lte => chelis_types::CompareOp::Lte,
                            },
                            *lval,
                            *rval,
                        )
                        .and_then(|result| {
                            Ok(chelis_types::scalar_from_i64(
                                "const",
                                chelis_types::types::Prim::Bool,
                                i64::from(result),
                            )?)
                        }),
                    ),
                    RiscOp::Logical(LogicalKind::And | LogicalKind::Or) => Some(
                        lval.as_bool_exact()
                            .zip(rval.as_bool_exact())
                            .ok_or({
                                chelis_types::NumericKernelError::Trap(
                                    chelis_types::NumericTrap::Domain {
                                        op: "logical",
                                        prim: node.output_type.precision,
                                    },
                                )
                            })
                            .and_then(|(lhs, rhs)| {
                                let result = match node.op {
                                    RiscOp::Logical(LogicalKind::And) => lhs && rhs,
                                    RiscOp::Logical(LogicalKind::Or) => lhs || rhs,
                                    _ => unreachable!(),
                                };
                                Ok(chelis_types::scalar_from_i64(
                                    "const",
                                    chelis_types::types::Prim::Bool,
                                    i64::from(result),
                                )?)
                            }),
                    ),
                    _ => None,
                };
                if let Some(result) = direct {
                    if let Ok(sealed) = result
                        && sealed.prim() == node.output_type.precision
                    {
                        let merge_spans = collect_operand_spans(node, &[l, r]);
                        replacements.push((node.id, sealed, merge_spans));
                    }
                    // A direct typed fold that traps or finds a malformed
                    // dtype declines the fold; it must never fall through to
                    // the legacy f64-wide optimizer path.
                    continue;
                }

                let (Some(lv), Some(rv)) = (wide_image(lval), wide_image(rval)) else {
                    continue;
                };
                let result = match &node.op {
                    RiscOp::Add => Some(lv + rv),
                    RiscOp::Mul => Some(lv * rv),
                    _ => None,
                };
                if let Some(val) = result
                    && let Ok(sealed) =
                        chelis_types::scalar_from_f64("const", node.output_type.precision, val)
                {
                    let merge_spans = collect_operand_spans(node, &[l, r]);
                    replacements.push((node.id, sealed, merge_spans));
                }
            }
        }
        // Unary constant folding.
        if node.inputs.len() == 1 {
            let input = dag.get(node.inputs[0]);
            // The transcendentals fold through the evaluator's declared-width
            // kernel, never the f64 wide image: a folded literal must carry the
            // same correctly rounded bits ([05-OP-46]) as the unfolded node.
            let transcendental = match &node.op {
                RiscOp::Exp => Some(chelis_types::FloatUnOp::Exp),
                RiscOp::Log => Some(chelis_types::FloatUnOp::Log),
                RiscOp::Sin => Some(chelis_types::FloatUnOp::Sin),
                RiscOp::Cos => Some(chelis_types::FloatUnOp::Cos),
                RiscOp::Tan => Some(chelis_types::FloatUnOp::Tan),
                RiscOp::Atan => Some(chelis_types::FloatUnOp::Atan),
                RiscOp::Tanh => Some(chelis_types::FloatUnOp::Tanh),
                _ => None,
            };
            if let Some(inp) = input
                && let RiscOp::Const { value: inner } = &inp.op
                && let Some(op) = transcendental
            {
                if inner.prim() == node.output_type.precision
                    && let Ok(folded) = chelis_types::float_unop(op, *inner)
                {
                    let merge_spans = collect_operand_spans(node, &[inp]);
                    replacements.push((node.id, folded, merge_spans));
                }
            } else if let Some(inp) = input
                && let RiscOp::Const { value: inner } = &inp.op
                && let Some(v) = wide_image(inner)
            {
                let result = match &node.op {
                    RiscOp::Logical(LogicalKind::Not) => {
                        inner.as_bool_exact().map(|value| i64::from(!value) as f64)
                    }
                    RiscOp::Neg => Some(-v),
                    RiscOp::Sqrt => Some(v.sqrt()),
                    RiscOp::Abs => Some(v.abs()),
                    RiscOp::Floor => Some(v.floor()),
                    RiscOp::Ceil => Some(v.ceil()),
                    RiscOp::Round => Some(v.round_ties_even()),
                    _ => None,
                };
                if let Some(val) = result
                    && let Ok(sealed) =
                        chelis_types::scalar_from_f64("const", node.output_type.precision, val)
                {
                    let merge_spans = collect_operand_spans(node, &[inp]);
                    replacements.push((node.id, sealed, merge_spans));
                }
            }
        }
        if node.inputs.len() == 3
            && matches!(node.op, RiscOp::Where)
            && let (Some(condition), Some(then_value), Some(else_value)) = (
                dag.get(node.inputs[0]),
                dag.get(node.inputs[1]),
                dag.get(node.inputs[2]),
            )
            && let (
                RiscOp::Const {
                    value: condition_value,
                },
                RiscOp::Const { value: then_scalar },
                RiscOp::Const { value: else_scalar },
            ) = (&condition.op, &then_value.op, &else_value.op)
            && let Some(selected) = condition_value.as_bool_exact().map(|condition| {
                if condition {
                    *then_scalar
                } else {
                    *else_scalar
                }
            })
        {
            let merge_spans = collect_operand_spans(node, &[condition, then_value, else_value]);
            replacements.push((node.id, selected, merge_spans));
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
pub fn dead_code_eliminate_with_remap(dag: &Dag) -> (Dag, UnordMap<NodeId, NodeId>) {
    dead_code_eliminate_with_retained(dag, &[])
}

/// Extra execution roots belong to an evaluator plan, not to the public
/// value-root interface. Keep their dependencies without publishing outputs.
pub(crate) fn dead_code_eliminate_with_retained(
    dag: &Dag,
    retained: &[NodeId],
) -> (Dag, UnordMap<NodeId, NodeId>) {
    dead_code_eliminate_impl(dag, retained, true)
}

fn dead_code_eliminate_impl(
    dag: &Dag,
    retained: &[NodeId],
    implicit_observations: bool,
) -> (Dag, UnordMap<NodeId, NodeId>) {
    let n = dag.len();
    if n == 0 {
        return (Dag::new(), UnordMap::new());
    }

    // Mark live nodes: DAG roots + all Store nodes.
    let mut live = vec![false; n];
    for id in retained {
        live[id.0] = true;
    }
    if dag.roots().is_empty() && implicit_observations {
        live[n - 1] = true;
    } else {
        for &root in dag.roots() {
            live[root.0] = true;
        }
    }
    // chelis#2476: the roots are the selection, so an abort or a trapping
    // draw in a declaration they do not enter is not this graph's concern,
    // as it is not the evaluator's ([`Dag::outside_selection`]). Without
    // this an entry selected from a lowered program keeps every other
    // declaration's draws, and their parameters become its inputs.
    let selection = dag
        .roots()
        .iter()
        .chain(retained)
        .copied()
        .collect::<Vec<_>>();
    let outside = dag.outside_selection(&selection);
    let seeds = dag.trap_seeds();
    for node in dag.nodes() {
        let observed = matches!(node.op, RiscOp::Store { .. });
        // chelis#2368: an unconditional effect is live regardless of
        // `implicit_observations`. A projected slice may legitimately drop an
        // unrelated `Store`, but never an abort: [05-OP-68] says it may not
        // be removed, and a slice that silently skipped one would report a
        // successful result for a program that aborts. chelis#2440 and
        // chelis#2413: a potentially trapping node, numeric or random, is in
        // that class too ([`crate::dag::TrapSeeds::is_observable_root`]).
        if (implicit_observations && observed)
            || (seeds.is_observable_root(node) && !outside[node.id.0])
        {
            live[node.id.0] = true;
        }
    }
    if implicit_observations {
        for (id, claimed) in crate::axis_sources::claimed_producers(dag)
            .into_iter()
            .enumerate()
        {
            live[id] |= claimed;
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
            // chelis#384/#397: a shape-only dependency (the `x` whose
            // runtime shape supplies an `expand` extent) is consumed for
            // its shape, not its data, so it is not in `inputs`. Keep it
            // live so its `Load` survives and the symbolic dim it declares
            // retains its source. See `DagNode::shape_deps`.
            for &dep in &dag.nodes()[i].shape_deps {
                live[dep.0] = true;
            }
            for &dep in &dag.nodes()[i].result_claim_deps {
                live[dep.0] = true;
            }
            // A node's activation decides whether it checks anything, so it
            // lives as long as the node does ([`crate::dag::Owner`]).
            if let Some(activation) = dag.nodes()[i].owner.activation {
                live[activation.0] = true;
            }
        }
    }

    // chelis#1277 C2.4 rule 2: no guard is ever discharged. An INTERFACE
    // witness - a `Load` axis a class groups - is an observable root, because
    // the guard comparing it can trap and a trap is an observation under
    // `spec/06` section 5.2. Without this a witness read by nothing is
    // dropped, its claim is left with one witness, the class dissolves and
    // the guard silently disappears, which is chelis#1374's shape.
    //
    // Two bounds keep this from resurrecting dead computation. It runs AFTER
    // ordinary liveness, and it force-keeps only members whose source is an
    // external axis. A LOCAL member is the operation introducing the extent:
    // if that operation is dead, no lane emits its guard, and C4.5 derives
    // classes "from the DAG a lane consumes, after the last rewrite" - a node
    // a rewrite has replaced is not in that graph. Forcing local members live
    // made the liveness circular, since a dead `Expand` carrying a claim
    // became a member and the membership then kept it alive; that resurrected
    // the dense product path specialization had just replaced with a
    // `BlasMatmul`.
    let mut extra = Vec::new();
    for class in crate::axis_sources::derive_runtime_dim_classes(dag)
        .into_iter()
        .filter(|_| implicit_observations)
    {
        for member in &class.members {
            if let crate::axis_sources::AxisSource::ExternalAxis { load, .. } = member.source
                && !live[load.0]
            {
                extra.push(load);
            }
        }
    }
    for load in extra {
        live[load.0] = true;
    }

    // Rebuild with only live nodes, remapping IDs.
    let mut new_dag = Dag::new();
    new_dag.inherit_declarations(dag);
    let mut id_map: UnordMap<usize, NodeId> = UnordMap::new();

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
            // Preserve merged_spans across DCE (S3 will populate them but
            // the invariant of pure-copy DCE means they must survive when
            // present).
            if !node.merged_spans.is_empty()
                && let Some(new_node) = new_dag.node_mut(new_id)
            {
                new_node.merged_spans = node.merged_spans.clone();
            }
            // chelis#384/#397: preserve (remapped) shape-only deps. Each was
            // marked live above and has a lower id in the topo-ordered DAG, so
            // it is already in `node_remap` by the time this node is rebuilt.
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

    for &root in dag.roots() {
        if let Some(&new_root) = id_map.get(&root.0) {
            new_dag.add_root(new_root);
        }
    }

    let node_remap: UnordMap<NodeId, NodeId> = id_map
        .into_sorted()
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
    new_dag.inherit_declarations(dag);
    let mut id_map: UnordMap<usize, NodeId> = UnordMap::new();
    let mut seen: UnordMap<CseKey, NodeId> = UnordMap::new();
    let claimed_producers = crate::axis_sources::claimed_producers(dag);

    for node in dag.nodes() {
        let remapped_inputs: Vec<NodeId> = node
            .inputs
            .iter()
            .map(|&old| *id_map.get(&old.0).unwrap_or(&old))
            .collect();
        // chelis#384/#397: two otherwise-identical nodes that depend on
        // DIFFERENT shape sources (an `expand` extent) are NOT
        // interchangeable — merging them would drop one source. Fold the
        // remapped shape-deps into the CSE key so such nodes stay distinct.
        let remapped_shape_deps: Vec<NodeId> = node
            .shape_deps
            .iter()
            .map(|&old| *id_map.get(&old.0).unwrap_or(&old))
            .collect();
        let remapped_result_claims: Vec<NodeId> = node
            .result_claim_deps
            .iter()
            .map(|old| {
                *id_map
                    .get(&old.0)
                    .unwrap_or_else(|| panic!("unmapped result claim dependency {old:?}"))
            })
            .collect();

        let op_key = format!("{:?}", node.op);
        // Nodes of two owners never merge: a parameter is its declaration and
        // its name, a node's declaration decides which selection runs it, and
        // its activation decides whether it checks (the same operation under
        // `c` and under `Not c` stays two nodes).
        let owner = node.owner.remap_with(|old| id_map.get(&old.0).copied());
        let cse_key = (
            owner,
            op_key,
            remapped_inputs.clone(),
            remapped_shape_deps.clone(),
            remapped_result_claims.clone(),
        );

        if !claimed_producers[node.id.0]
            && !matches!(
                node.op,
                RiscOp::ExtentWitness { .. }
                    | RiscOp::CheckedReshapeExtent { .. }
                    | RiscOp::CheckedUnitAxis { .. }
                    // Equal capture values in two List calls still have
                    // distinct invocation identities for a later AD pass.
                    | RiscOp::ListMapCapture { .. }
            )
            // Two key operations with equal inputs produce equal bits, but
            // merging them would hand one key to both consumers, which the
            // key rules reject: a key is consumed once in the graph, not
            // merely once per value.
            && !matches!(
                node.op,
                RiscOp::KeyFromSeed
                    | RiscOp::Split { .. }
                    | RiscOp::FoldIn
                    | RiscOp::SplitN { .. }
                    | RiscOp::KeySelect
            )
            && let Some(&existing) = seen.get(&cse_key)
        {
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
                owner,
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
            // chelis#384/#397: preserve the (remapped) shape-derived `expand`
            // shape-deps so CSE does not drop the liveness edge.
            if !remapped_shape_deps.is_empty()
                && let Some(new_node) = new_dag.node_mut(new_id)
            {
                new_node.shape_deps = remapped_shape_deps;
            }
            if !remapped_result_claims.is_empty()
                && let Some(new_node) = new_dag.node_mut(new_id)
            {
                new_node.result_claim_deps = remapped_result_claims;
            }
            if let Some(reusable_input) = node.reusable_input
                && let Some(&mapped_input) = id_map.get(&reusable_input.0)
            {
                new_dag.set_reusable_input(new_id, mapped_input);
            }
            id_map.insert(node.id.0, new_id);
            if !claimed_producers[node.id.0] {
                seen.insert(cse_key, new_id);
            }
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
    use crate::dag::{
        Dag, DeclId, DimInfo, ExtentWitnessSite, FusedInput, FusedStep, FusedStepOp,
        ReduceWindowKind, RiscOp, RtAxis, RtDim, TensorType,
    };
    use chelis_types::{ElementRef, scalar_from_f64, scalar_from_i64};

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    fn named_claim(dag: &mut Dag, decl: DeclId, input: NodeId) -> NodeId {
        dag.add_node(
            decl,
            RiscOp::ExtentWitness {
                site: ExtentWitnessSite::ResultClaim {
                    claim: "n".into(),
                    axis: RtAxis::Lit(0),
                },
                parameter: "x".into(),
                axis: RtAxis::Lit(0),
                requirements: Vec::new(),
                claims: Vec::new(),
            },
            vec![input],
            TensorType {
                dims: Vec::new(),
                precision: chelis_types::types::Prim::Int64,
            },
            None,
        )
    }

    #[test]
    fn claimed_producers_are_neither_folded_merged_nor_dropped() {
        let tensor = TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: chelis_types::types::Prim::F32,
        };

        let mut folded = Dag::new();
        let folded_decl = folded.declare("test");
        let left = folded.add_node(
            folded_decl,
            RiscOp::synth_const(tensor.precision, 1.0),
            Vec::new(),
            tensor.clone(),
            None,
        );
        let right = folded.add_node(
            folded_decl,
            RiscOp::synth_const(tensor.precision, 2.0),
            Vec::new(),
            tensor.clone(),
            None,
        );
        let claim = named_claim(&mut folded, folded_decl, left);
        let add = folded.add_node(
            folded_decl,
            RiscOp::Add,
            vec![left, right],
            tensor.clone(),
            None,
        );
        folded.add_result_claim_dep(add, claim);
        constant_fold(&mut folded);
        assert!(matches!(folded.get(add).unwrap().op, RiscOp::Add));

        let mut duplicate = Dag::new();
        let duplicate_decl = duplicate.declare("test");
        let input = duplicate.add_node(
            duplicate_decl,
            RiscOp::Load { name: "x".into() },
            Vec::new(),
            tensor.clone(),
            None,
        );
        let claim = named_claim(&mut duplicate, duplicate_decl, input);
        for _ in 0..2 {
            let cast = duplicate.add_node(
                duplicate_decl,
                RiscOp::Cast {
                    new_precision: chelis_types::types::Prim::F32,
                },
                vec![input],
                tensor.clone(),
                None,
            );
            duplicate.add_result_claim_dep(cast, claim);
            duplicate.add_root(cast);
        }
        let duplicate = common_subexpr_eliminate(&duplicate);
        assert_eq!(
            duplicate
                .nodes()
                .iter()
                .filter(|node| matches!(node.op, RiscOp::Cast { .. }))
                .count(),
            2,
            "CSE cannot merge two potentially trapping claimed producers"
        );

        let mut dead = Dag::new();
        let dead_decl = dead.declare("test");
        let input = dead.add_node(
            dead_decl,
            RiscOp::Load { name: "x".into() },
            Vec::new(),
            tensor.clone(),
            None,
        );
        let claim = named_claim(&mut dead, dead_decl, input);
        let cast = dead.add_node(
            dead_decl,
            RiscOp::Cast {
                new_precision: chelis_types::types::Prim::F32,
            },
            vec![input],
            tensor,
            None,
        );
        dead.add_result_claim_dep(cast, claim);
        let root = dead.add_node(
            dead_decl,
            RiscOp::synth_const(chelis_types::types::Prim::F32, 0.0),
            Vec::new(),
            scalar_f32(),
            None,
        );
        dead.add_root(root);
        let dead = dead_code_eliminate(&dead);
        assert!(
            dead.nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::Cast { .. })),
            "full DCE must retain a potentially trapping claimed producer"
        );
    }

    /// chelis#2440: the same obligation without a result claim.
    ///
    /// `spec/06-transformations.md` §5.2 is prescriptive — "Mark every
    /// effectful node, every potentially trapping node, every `Store`, and
    /// every designated output as live", and "purity alone does not make a
    /// possible trap dead". The retention above hung on the node carrying a
    /// `result_claim_dep`; a plain discarded trapping node had nothing to
    /// keep it, so its trap simply did not occur.
    #[test]
    fn full_dce_retains_a_trapping_node_with_no_claim() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let source = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            Vec::new(),
            scalar_f32(),
            None,
        );
        let trapping = dag.add_node(
            decl,
            RiscOp::Cast {
                new_precision: chelis_types::types::Prim::Int32,
            },
            vec![source],
            TensorType {
                dims: Vec::new(),
                precision: chelis_types::types::Prim::Int32,
            },
            None,
        );
        let root = dag.add_node(
            decl,
            RiscOp::synth_const(chelis_types::types::Prim::F32, 0.0),
            Vec::new(),
            scalar_f32(),
            None,
        );
        dag.add_root(root);
        let pruned = dead_code_eliminate(&dag);
        assert!(
            pruned
                .nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::Cast { .. })),
            "a discarded cast into an integer width can trap, so §5.2 makes it \
             an observable root even with no claim and no consumer"
        );
        let _ = trapping;
    }

    /// One retained/eliminated pair per member of
    /// [`RuntimeCheck::OperandValues`], plus two neighbouring classes that
    /// must NOT seed.
    ///
    /// `runtime_check`'s op match is exhaustive, so a *new* operation cannot
    /// be added without classifying it. What that does not protect is an
    /// existing arm being **reclassified**, and that is what this pins. Each
    /// row flips with its own arm, at sub-arm granularity (`Neg` alone,
    /// `Mod` alone, one shift alone, one `ReduceWindowKind` alone) and
    /// per-dtype, and in **both** directions: an arm that stops seeding and
    /// an arm that starts seeding when it should not.
    ///
    /// `Bitwise` is keyed on the operation KIND rather than the dtype (only
    /// a shift checks anything), `ReduceWindow` on the reducer (only a sum
    /// or mean is arithmetic), and `Iota` checks unconditionally; all three
    /// are enumerated for the same reason.
    ///
    /// `EmptyAxis`, `MovementBounds`, `ExtentClaims` and `Random` are *not*
    /// enumerated: their seed turns on a static fact rather than a dtype or
    /// kind, and each is caught elsewhere — verified by mutation, `Random`
    /// and `ExtentClaims` within `chelis-ir`, `EmptyAxis` and
    /// `MovementBounds` in `chelis-backend-c::exec_compile`. No row here
    /// belongs to those classes. The three non-seeding controls below are
    /// float `Div` (`MeanDivisor`) and `Reshape` and `Expand` (`Ungated`).
    ///
    /// [`RuntimeCheck::SparseIndex`] is enumerated in full (chelis#2440):
    /// its seed turns on neither a dtype nor a kind, so no member would be
    /// singled out by a row that varied either, and a reclassification of
    /// any one of them would otherwise be silent. It does turn on a
    /// per-node fact -- the node must carry no activation -- which every row
    /// here satisfies, because `Dag::add_node` gives each node an
    /// unconditional owner. The false-activation half is pinned at the CLI
    /// surface instead, in `issue_2440_discarded_sparse_index_traps`.
    ///
    /// This comment deliberately does not claim which arms nothing else
    /// catches. Two review rounds each refuted such a claim by running a
    /// target the previous measurement had missed, so the coverage is stated
    /// as what this test pins, not as what only it pins.
    #[test]
    fn every_dtype_conditional_trap_class_member_is_seeded_and_its_twin_is_not() {
        use chelis_types::types::Prim;

        fn ty(precision: Prim) -> TensorType {
            TensorType {
                dims: vec![DimInfo::Lit(2)],
                precision,
            }
        }

        // (label, op, arity, precision, must the discarded node survive DCE?)
        let rows: Vec<(&str, RiscOp, usize, Prim, bool)> = vec![
            // OperandValues: integer arithmetic overflows, float does not.
            ("add", RiscOp::Add, 2, Prim::Int32, true),
            ("add float twin", RiscOp::Add, 2, Prim::F32, false),
            ("sub", RiscOp::Sub, 2, Prim::Int64, true),
            ("sub float twin", RiscOp::Sub, 2, Prim::F64, false),
            ("mul", RiscOp::Mul, 2, Prim::Int32, true),
            ("mul float twin", RiscOp::Mul, 2, Prim::F32, false),
            ("neg", RiscOp::Neg, 1, Prim::Int32, true),
            ("neg float twin", RiscOp::Neg, 1, Prim::F32, false),
            ("abs", RiscOp::Abs, 1, Prim::Int32, true),
            ("abs float twin", RiscOp::Abs, 1, Prim::F32, false),
            // OperandValues: division by zero and MIN / -1. The escape above.
            ("floor_div", RiscOp::FloorDiv, 2, Prim::Int32, true),
            (
                "floor_div float twin",
                RiscOp::FloorDiv,
                2,
                Prim::F32,
                false,
            ),
            ("trunc_div", RiscOp::TruncDiv, 2, Prim::Int64, true),
            (
                "trunc_div float twin",
                RiscOp::TruncDiv,
                2,
                Prim::F32,
                false,
            ),
            ("mod", RiscOp::Mod, 2, Prim::Int32, true),
            ("mod float twin", RiscOp::Mod, 2, Prim::F32, false),
            // `Div` splits by dtype into two different classes: integer is
            // OperandValues, float is MeanDivisor, which does not seed.
            ("div integer", RiscOp::Div, 2, Prim::Int32, true),
            ("div float is MeanDivisor", RiscOp::Div, 2, Prim::F32, false),
            // OperandValues: integer reductions overflow.
            (
                "sum",
                RiscOp::Sum {
                    axis: 0,
                    accumulator: Prim::Int32,
                },
                1,
                Prim::Int32,
                true,
            ),
            (
                "sum float twin",
                RiscOp::Sum {
                    axis: 0,
                    accumulator: Prim::F32,
                },
                1,
                Prim::F32,
                false,
            ),
            (
                "prod_reduce",
                RiscOp::ProdReduce { axis: 0 },
                1,
                Prim::Int64,
                true,
            ),
            (
                "prod_reduce float twin",
                RiscOp::ProdReduce { axis: 0 },
                1,
                Prim::F32,
                false,
            ),
            // OperandValues: a shift checks its shift count; the bitwise
            // logical ops check nothing. This member is keyed on the KIND
            // rather than the dtype, so both sides are integer.
            (
                "shift_left",
                RiscOp::Bitwise(chelis_types::bitwise::BitwiseKind::ShiftLeft),
                2,
                Prim::Int32,
                true,
            ),
            (
                "shift_right",
                RiscOp::Bitwise(chelis_types::bitwise::BitwiseKind::ShiftRight),
                2,
                Prim::Int32,
                true,
            ),
            (
                "bitand is not a shift",
                RiscOp::Bitwise(chelis_types::bitwise::BitwiseKind::And),
                2,
                Prim::Int32,
                false,
            ),
            (
                "bitxor is not a shift",
                RiscOp::Bitwise(chelis_types::bitwise::BitwiseKind::Xor),
                2,
                Prim::Int32,
                false,
            ),
            // OperandValues unconditionally: `iota`'s length is checked in
            // mathematical integers before allocation.
            ("iota", RiscOp::Iota, 2, Prim::Int64, true),
            // OperandValues: an integer windowed SUM or MEAN overflows; max
            // and min select without arithmetic, so they check nothing even
            // at an integer dtype.
            (
                "reduce_window sum",
                RiscOp::ReduceWindow {
                    reducer: ReduceWindowKind::Sum,
                    window_shape: vec![1],
                    strides: vec![1],
                },
                1,
                Prim::Int32,
                true,
            ),
            (
                "reduce_window sum float twin",
                RiscOp::ReduceWindow {
                    reducer: ReduceWindowKind::Sum,
                    window_shape: vec![1],
                    strides: vec![1],
                },
                1,
                Prim::F32,
                false,
            ),
            (
                "reduce_window mean",
                RiscOp::ReduceWindow {
                    reducer: ReduceWindowKind::Mean,
                    window_shape: vec![1],
                    strides: vec![1],
                },
                1,
                Prim::Int32,
                true,
            ),
            (
                "reduce_window min is not arithmetic",
                RiscOp::ReduceWindow {
                    reducer: ReduceWindowKind::Min,
                    window_shape: vec![1],
                    strides: vec![1],
                },
                1,
                Prim::Int32,
                false,
            ),
            (
                "reduce_window max is not arithmetic",
                RiscOp::ReduceWindow {
                    reducer: ReduceWindowKind::Max,
                    window_shape: vec![1],
                    strides: vec![1],
                },
                1,
                Prim::Int32,
                false,
            ),
            // OperandValues: a fused chain inherits its steps' checks.
            (
                "fused_elem",
                RiscOp::FusedElem {
                    ops: vec![FusedStep {
                        op: FusedStepOp::Add,
                        input_indices: vec![FusedInput::External(0), FusedInput::External(0)],
                    }],
                },
                1,
                Prim::Int32,
                true,
            ),
            (
                "fused_elem float twin",
                RiscOp::FusedElem {
                    ops: vec![FusedStep {
                        op: FusedStepOp::Add,
                        input_indices: vec![FusedInput::External(0), FusedInput::External(0)],
                    }],
                },
                1,
                Prim::F32,
                false,
            ),
            // OperandValues: a cast into an integer or bool width checks its
            // domain and range; one into a float width cannot.
            (
                "cast to integer",
                RiscOp::Cast {
                    new_precision: Prim::Int32,
                },
                1,
                Prim::Int32,
                true,
            ),
            (
                "cast to bool",
                RiscOp::Cast {
                    new_precision: Prim::Bool,
                },
                1,
                Prim::Bool,
                true,
            ),
            (
                "cast to float twin",
                RiscOp::Cast {
                    new_precision: Prim::F32,
                },
                1,
                Prim::F32,
                false,
            ),
            (
                "cast_trunc to integer",
                RiscOp::CastTrunc {
                    new_precision: Prim::Int64,
                },
                1,
                Prim::Int64,
                true,
            ),
            (
                "cast_trunc to float twin",
                RiscOp::CastTrunc {
                    new_precision: Prim::F32,
                },
                1,
                Prim::F32,
                false,
            ),
            // SparseIndex: [05-OP-52] makes an out-of-bounds index fail
            // loudly, and the index is data, so every member seeds whatever
            // its dtype. Each row flips with its own sub-arm.
            (
                "gather",
                RiscOp::Gather {
                    axis: 0,
                    batch_rank: 0,
                },
                2,
                Prim::F32,
                true,
            ),
            (
                "scatter add",
                RiscOp::ScatterAdd {
                    axis: 0,
                    batch_rank: 0,
                },
                3,
                Prim::F32,
                true,
            ),
            (
                "scatter replace",
                RiscOp::Scatter {
                    axis: 0,
                    batch_rank: 0,
                },
                3,
                Prim::F32,
                true,
            ),
            (
                "scatter_elements",
                RiscOp::ScatterElements { axis: 0 },
                3,
                Prim::F32,
                true,
            ),
            ("one_hot", RiscOp::OneHot { vocab: 2 }, 1, Prim::Int32, true),
            // Ungated: still not a seeded class. A reshape DOES carry a
            // runtime element-count check of its own, distinct from the
            // `CheckedReshapeExtent` extent claim (`ExtentClaims`); it is
            // simply not seeded, which chelis#2440 lists as residual.
            (
                "reshape is Ungated",
                RiscOp::Reshape {
                    new_shape: vec![RtDim::Lit(2)],
                },
                1,
                Prim::Int32,
                false,
            ),
            (
                "expand is Ungated",
                RiscOp::Expand {
                    axis: 0,
                    size: RtDim::Lit(2),
                },
                1,
                Prim::Int32,
                false,
            ),
        ];

        for (label, op, arity, precision, expect_retained) in rows {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let source = dag.add_node(
                decl,
                RiscOp::Load { name: "x".into() },
                Vec::new(),
                ty(precision),
                None,
            );
            let discarded =
                dag.add_node(decl, op.clone(), vec![source; arity], ty(precision), None);
            let root = dag.add_node(
                decl,
                RiscOp::synth_const(Prim::F32, 0.0),
                Vec::new(),
                scalar_f32(),
                None,
            );
            dag.add_root(root);

            let retained = dead_code_eliminate(&dag)
                .nodes()
                .iter()
                .any(|node| node.op == op);
            assert_eq!(
                retained, expect_retained,
                "{label} at {precision:?}: expected retained={expect_retained}, got \
                 {retained} (discarded node {discarded:?})"
            );
        }
    }

    /// The negative control: §5.2's own example removes a FLOAT `Add`, which
    /// cannot trap. Over-retaining is safe but not free, so the predicate
    /// must stay dtype-aware rather than keeping all arithmetic alive.
    #[test]
    fn full_dce_still_removes_a_non_trapping_dead_node() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let source = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            Vec::new(),
            scalar_f32(),
            None,
        );
        let one = dag.add_node(
            decl,
            RiscOp::synth_const(chelis_types::types::Prim::F32, 1.0),
            Vec::new(),
            scalar_f32(),
            None,
        );
        // Float addition produces infinities, never a trap.
        let dead = dag.add_node(decl, RiscOp::Add, vec![source, one], scalar_f32(), None);
        let root = dag.add_node(
            decl,
            RiscOp::synth_const(chelis_types::types::Prim::F32, 0.0),
            Vec::new(),
            scalar_f32(),
            None,
        );
        dag.add_root(root);
        let pruned = dead_code_eliminate(&dag);
        assert!(
            !pruned
                .nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::Add)),
            "a dead float add cannot trap and must still be eliminated"
        );
        let _ = dead;
    }

    #[test]
    fn constant_fold_add() {
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
        dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_f32(), None);

        constant_fold(&mut dag);

        let result = dag.get(NodeId(2)).unwrap();
        assert_eq!(
            result.op,
            RiscOp::synth_const(chelis_types::types::Prim::F32, 3.0)
        );
        assert!(result.inputs.is_empty());
    }

    #[test]
    fn constant_fold_mul() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 4.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Mul, vec![a, b], scalar_f32(), None);

        constant_fold(&mut dag);

        let result = dag.get(NodeId(2)).unwrap();
        assert_eq!(
            result.op,
            RiscOp::synth_const(chelis_types::types::Prim::F32, 12.0)
        );
    }

    #[test]
    fn constant_fold_neg() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 5.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Neg, vec![a], scalar_f32(), None);

        constant_fold(&mut dag);

        let result = dag.get(NodeId(1)).unwrap();
        assert_eq!(
            result.op,
            RiscOp::synth_const(chelis_types::types::Prim::F32, -5.0)
        );
    }

    #[test]
    fn constant_fold_direct_integer_subtraction_is_exact_or_declines_overflow() {
        let scalar_i64 = TensorType {
            dims: vec![],
            precision: chelis_types::types::Prim::Int64,
        };
        let mut exact = Dag::new();
        let exact_decl = exact.declare("test");
        let lhs = exact.add_node(
            exact_decl,
            RiscOp::Const {
                value: scalar_from_i64("test", scalar_i64.precision, 9_007_199_254_740_993)
                    .unwrap(),
            },
            vec![],
            scalar_i64.clone(),
            None,
        );
        let rhs = exact.add_node(
            exact_decl,
            RiscOp::Const {
                value: scalar_from_i64("test", scalar_i64.precision, 1).unwrap(),
            },
            vec![],
            scalar_i64.clone(),
            None,
        );
        exact.add_node(
            exact_decl,
            RiscOp::Sub,
            vec![lhs, rhs],
            scalar_i64.clone(),
            None,
        );
        constant_fold(&mut exact);
        match &exact.get(NodeId(2)).unwrap().op {
            RiscOp::Const { value } => {
                assert_eq!(value.as_i64_exact(), Some(9_007_199_254_740_992));
            }
            other => panic!("exact i64 subtraction must fold directly, got {other:?}"),
        }

        let mut overflow = Dag::new();
        let overflow_decl = overflow.declare("test");
        let lhs = overflow.add_node(
            overflow_decl,
            RiscOp::Const {
                value: scalar_from_i64("test", scalar_i64.precision, i64::MAX).unwrap(),
            },
            vec![],
            scalar_i64.clone(),
            None,
        );
        let rhs = overflow.add_node(
            overflow_decl,
            RiscOp::Const {
                value: scalar_from_i64("test", scalar_i64.precision, -1).unwrap(),
            },
            vec![],
            scalar_i64.clone(),
            None,
        );
        overflow.add_node(overflow_decl, RiscOp::Sub, vec![lhs, rhs], scalar_i64, None);
        constant_fold(&mut overflow);
        assert!(matches!(overflow.get(NodeId(2)).unwrap().op, RiscOp::Sub));
    }

    #[test]
    fn constant_fold_direct_extrema_preserves_selected_f64_bits() {
        let scalar_f64 = TensorType {
            dims: vec![],
            precision: chelis_types::types::Prim::F64,
        };
        let cases = [
            (
                f64::from_bits(0x7ff8_1111_2222_3333),
                1.0,
                0x7ff8_1111_2222_3333,
            ),
            (
                1.0,
                f64::from_bits(0xfff8_4444_5555_6666),
                0xfff8_4444_5555_6666,
            ),
            (0.0, -0.0, 0),
            (-0.0, 0.0, 0x8000_0000_0000_0000),
        ];
        for op in [RiscOp::MaxElem, RiscOp::MinElem] {
            for (lhs_value, rhs_value, expected_bits) in cases {
                let mut dag = Dag::new();
                let decl = dag.declare("test");
                let lhs = dag.add_node(
                    decl,
                    RiscOp::Const {
                        value: scalar_from_f64("test", scalar_f64.precision, lhs_value).unwrap(),
                    },
                    vec![],
                    scalar_f64.clone(),
                    None,
                );
                let rhs = dag.add_node(
                    decl,
                    RiscOp::Const {
                        value: scalar_from_f64("test", scalar_f64.precision, rhs_value).unwrap(),
                    },
                    vec![],
                    scalar_f64.clone(),
                    None,
                );
                dag.add_node(decl, op.clone(), vec![lhs, rhs], scalar_f64.clone(), None);
                constant_fold(&mut dag);
                match &dag.get(NodeId(2)).unwrap().op {
                    RiscOp::Const { value } => match value.element_ref() {
                        ElementRef::F64(observed) => {
                            assert_eq!(observed.to_bits(), expected_bits, "{op:?}")
                        }
                        other => panic!("expected f64 folded value, got {other:?}"),
                    },
                    other => panic!("direct extrema must fold through exact selection: {other:?}"),
                }
            }
        }
    }

    #[test]
    fn constant_fold_comparisons_obey_ieee_nan_and_signed_zero_rules() {
        let scalar_f64 = TensorType {
            dims: vec![],
            precision: chelis_types::types::Prim::F64,
        };
        let scalar_bool = TensorType {
            dims: vec![],
            precision: chelis_types::types::Prim::Bool,
        };
        let nan = f64::from_bits(0x7ff8_1234_5678_9abc);
        let cases = [
            (
                nan,
                1.0,
                [
                    (ComparisonKind::CmpLt, false),
                    (ComparisonKind::Lt, false),
                    (ComparisonKind::Eq, false),
                    (ComparisonKind::Neq, true),
                    (ComparisonKind::Gt, false),
                    (ComparisonKind::Gte, false),
                    (ComparisonKind::Lte, false),
                ],
            ),
            (
                0.0,
                -0.0,
                [
                    (ComparisonKind::CmpLt, false),
                    (ComparisonKind::Lt, false),
                    (ComparisonKind::Eq, true),
                    (ComparisonKind::Neq, false),
                    (ComparisonKind::Gt, false),
                    (ComparisonKind::Gte, true),
                    (ComparisonKind::Lte, true),
                ],
            ),
        ];

        for (lhs_value, rhs_value, comparisons) in cases {
            for (kind, expected) in comparisons {
                let mut dag = Dag::new();
                let decl = dag.declare("test");
                let lhs = dag.add_node(
                    decl,
                    RiscOp::Const {
                        value: scalar_from_f64("test", scalar_f64.precision, lhs_value).unwrap(),
                    },
                    vec![],
                    scalar_f64.clone(),
                    None,
                );
                let rhs = dag.add_node(
                    decl,
                    RiscOp::Const {
                        value: scalar_from_f64("test", scalar_f64.precision, rhs_value).unwrap(),
                    },
                    vec![],
                    scalar_f64.clone(),
                    None,
                );
                let comparison = dag.add_node(
                    decl,
                    RiscOp::Compare(kind),
                    vec![lhs, rhs],
                    scalar_bool.clone(),
                    None,
                );

                constant_fold(&mut dag);

                match &dag.get(comparison).unwrap().op {
                    RiscOp::Const { value } => assert_eq!(
                        value.as_bool_exact(),
                        Some(expected),
                        "{kind:?}({lhs_value:?}, {rhs_value:?})"
                    ),
                    other => panic!("constant comparison must fold, got {other:?}"),
                }
            }
        }
    }

    #[test]
    fn constant_fold_bool_logical_truth_tables() {
        let scalar_bool = TensorType {
            dims: vec![],
            precision: chelis_types::types::Prim::Bool,
        };

        for (kind, truth_table) in [
            (
                LogicalKind::And,
                [
                    (false, false, false),
                    (false, true, false),
                    (true, false, false),
                    (true, true, true),
                ],
            ),
            (
                LogicalKind::Or,
                [
                    (false, false, false),
                    (false, true, true),
                    (true, false, true),
                    (true, true, true),
                ],
            ),
        ] {
            for (lhs_value, rhs_value, expected) in truth_table {
                let mut dag = Dag::new();
                let decl = dag.declare("test");
                let lhs = dag.add_node(
                    decl,
                    RiscOp::Const {
                        value: scalar_from_i64("test", scalar_bool.precision, i64::from(lhs_value))
                            .unwrap(),
                    },
                    vec![],
                    scalar_bool.clone(),
                    None,
                );
                let rhs = dag.add_node(
                    decl,
                    RiscOp::Const {
                        value: scalar_from_i64("test", scalar_bool.precision, i64::from(rhs_value))
                            .unwrap(),
                    },
                    vec![],
                    scalar_bool.clone(),
                    None,
                );
                let logical = dag.add_node(
                    decl,
                    RiscOp::Logical(kind),
                    vec![lhs, rhs],
                    scalar_bool.clone(),
                    None,
                );

                constant_fold(&mut dag);

                match &dag.get(logical).unwrap().op {
                    RiscOp::Const { value } => assert_eq!(
                        value.as_bool_exact(),
                        Some(expected),
                        "{kind:?}({lhs_value}, {rhs_value})"
                    ),
                    other => panic!("constant logical operation must fold, got {other:?}"),
                }
            }
        }

        for (input_value, expected) in [(false, true), (true, false)] {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let input = dag.add_node(
                decl,
                RiscOp::Const {
                    value: scalar_from_i64("test", scalar_bool.precision, i64::from(input_value))
                        .unwrap(),
                },
                vec![],
                scalar_bool.clone(),
                None,
            );
            let logical = dag.add_node(
                decl,
                RiscOp::Logical(LogicalKind::Not),
                vec![input],
                scalar_bool.clone(),
                None,
            );

            constant_fold(&mut dag);

            match &dag.get(logical).unwrap().op {
                RiscOp::Const { value } => {
                    assert_eq!(value.as_bool_exact(), Some(expected), "Not({input_value})")
                }
                other => panic!("constant logical not must fold, got {other:?}"),
            }
        }
    }

    #[test]
    fn constant_fold_where_preserves_selected_f64_bits() {
        let scalar_bool = TensorType {
            dims: vec![],
            precision: chelis_types::types::Prim::Bool,
        };
        let scalar_f64 = TensorType {
            dims: vec![],
            precision: chelis_types::types::Prim::F64,
        };
        let then_bits = 0x7ff8_1234_5678_9abc;
        let else_bits = 0x8000_0000_0000_0000;

        for (condition_value, expected_bits) in [(true, then_bits), (false, else_bits)] {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let condition = dag.add_node(
                decl,
                RiscOp::Const {
                    value: scalar_from_i64(
                        "test",
                        scalar_bool.precision,
                        i64::from(condition_value),
                    )
                    .unwrap(),
                },
                vec![],
                scalar_bool.clone(),
                None,
            );
            let then_value = dag.add_node(
                decl,
                RiscOp::Const {
                    value: scalar_from_f64("test", scalar_f64.precision, f64::from_bits(then_bits))
                        .unwrap(),
                },
                vec![],
                scalar_f64.clone(),
                None,
            );
            let else_value = dag.add_node(
                decl,
                RiscOp::Const {
                    value: scalar_from_f64("test", scalar_f64.precision, f64::from_bits(else_bits))
                        .unwrap(),
                },
                vec![],
                scalar_f64.clone(),
                None,
            );
            let selected = dag.add_node(
                decl,
                RiscOp::Where,
                vec![condition, then_value, else_value],
                scalar_f64.clone(),
                None,
            );

            constant_fold(&mut dag);

            match &dag.get(selected).unwrap().op {
                RiscOp::Const { value } => match value.element_ref() {
                    ElementRef::F64(observed) => {
                        assert_eq!(observed.to_bits(), expected_bits, "{condition_value}")
                    }
                    other => panic!("expected f64 folded value, got {other:?}"),
                },
                other => panic!("constant where must fold, got {other:?}"),
            }
        }
    }

    #[test]
    fn constant_fold_new_ops_decline_nonconstant_and_type_invalid_inputs() {
        let scalar_bool = TensorType {
            dims: vec![],
            precision: chelis_types::types::Prim::Bool,
        };
        let scalar_f32 = scalar_f32();
        let scalar_f64 = TensorType {
            dims: vec![],
            precision: chelis_types::types::Prim::F64,
        };

        let mut mismatched_comparison = Dag::new();
        let mismatched_comparison_decl = mismatched_comparison.declare("test");
        let lhs = mismatched_comparison.add_node(
            mismatched_comparison_decl,
            RiscOp::Const {
                value: scalar_from_f64("test", scalar_f32.precision, 1.0).unwrap(),
            },
            vec![],
            scalar_f32.clone(),
            None,
        );
        let rhs = mismatched_comparison.add_node(
            mismatched_comparison_decl,
            RiscOp::Const {
                value: scalar_from_f64("test", scalar_f64.precision, 1.0).unwrap(),
            },
            vec![],
            scalar_f64,
            None,
        );
        let comparison = mismatched_comparison.add_node(
            mismatched_comparison_decl,
            RiscOp::Compare(ComparisonKind::Eq),
            vec![lhs, rhs],
            scalar_bool.clone(),
            None,
        );
        constant_fold(&mut mismatched_comparison);
        assert!(matches!(
            mismatched_comparison.get(comparison).unwrap().op,
            RiscOp::Compare(ComparisonKind::Eq)
        ));

        let mut invalid_logical = Dag::new();
        let invalid_logical_decl = invalid_logical.declare("test");
        let lhs = invalid_logical.add_node(
            invalid_logical_decl,
            RiscOp::synth_const(scalar_f32.precision, 1.0),
            vec![],
            scalar_f32.clone(),
            None,
        );
        let rhs = invalid_logical.add_node(
            invalid_logical_decl,
            RiscOp::synth_const(scalar_f32.precision, 0.0),
            vec![],
            scalar_f32.clone(),
            None,
        );
        let logical = invalid_logical.add_node(
            invalid_logical_decl,
            RiscOp::Logical(LogicalKind::And),
            vec![lhs, rhs],
            scalar_bool.clone(),
            None,
        );
        constant_fold(&mut invalid_logical);
        assert!(matches!(
            invalid_logical.get(logical).unwrap().op,
            RiscOp::Logical(LogicalKind::And)
        ));

        let mut nonconstant_logical = Dag::new();
        let nonconstant_logical_decl = nonconstant_logical.declare("test");
        let lhs = nonconstant_logical.add_node(
            nonconstant_logical_decl,
            RiscOp::Load {
                name: "flag".into(),
            },
            vec![],
            scalar_bool.clone(),
            None,
        );
        let rhs = nonconstant_logical.add_node(
            nonconstant_logical_decl,
            RiscOp::synth_const(scalar_bool.precision, 1.0),
            vec![],
            scalar_bool.clone(),
            None,
        );
        let logical = nonconstant_logical.add_node(
            nonconstant_logical_decl,
            RiscOp::Logical(LogicalKind::Or),
            vec![lhs, rhs],
            scalar_bool.clone(),
            None,
        );
        constant_fold(&mut nonconstant_logical);
        assert!(matches!(
            nonconstant_logical.get(logical).unwrap().op,
            RiscOp::Logical(LogicalKind::Or)
        ));

        let mut invalid_where = Dag::new();
        let invalid_where_decl = invalid_where.declare("test");
        let condition = invalid_where.add_node(
            invalid_where_decl,
            RiscOp::synth_const(scalar_f32.precision, 1.0),
            vec![],
            scalar_f32.clone(),
            None,
        );
        let then_value = invalid_where.add_node(
            invalid_where_decl,
            RiscOp::synth_const(scalar_f32.precision, 2.0),
            vec![],
            scalar_f32.clone(),
            None,
        );
        let else_value = invalid_where.add_node(
            invalid_where_decl,
            RiscOp::synth_const(scalar_f32.precision, 3.0),
            vec![],
            scalar_f32.clone(),
            None,
        );
        let selected = invalid_where.add_node(
            invalid_where_decl,
            RiscOp::Where,
            vec![condition, then_value, else_value],
            scalar_f32.clone(),
            None,
        );
        constant_fold(&mut invalid_where);
        assert!(matches!(
            invalid_where.get(selected).unwrap().op,
            RiscOp::Where
        ));

        let mut nonconstant_where = Dag::new();
        let nonconstant_where_decl = nonconstant_where.declare("test");
        let condition = nonconstant_where.add_node(
            nonconstant_where_decl,
            RiscOp::synth_const(scalar_bool.precision, 1.0),
            vec![],
            scalar_bool,
            None,
        );
        let then_value = nonconstant_where.add_node(
            nonconstant_where_decl,
            RiscOp::Load {
                name: "then".into(),
            },
            vec![],
            scalar_f32.clone(),
            None,
        );
        let else_value = nonconstant_where.add_node(
            nonconstant_where_decl,
            RiscOp::synth_const(scalar_f32.precision, 3.0),
            vec![],
            scalar_f32.clone(),
            None,
        );
        let selected = nonconstant_where.add_node(
            nonconstant_where_decl,
            RiscOp::Where,
            vec![condition, then_value, else_value],
            scalar_f32,
            None,
        );
        constant_fold(&mut nonconstant_where);
        assert!(matches!(
            nonconstant_where.get(selected).unwrap().op,
            RiscOp::Where
        ));
    }

    #[test]
    fn dce_removes_dead_nodes() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let _dead = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 99.0),
            vec![],
            scalar_f32(),
            None,
        );
        let live = dag.add_node(decl, RiscOp::Neg, vec![a], scalar_f32(), None);
        dag.add_root(live);

        let new_dag = dead_code_eliminate(&dag);
        // Dead const(99) should be removed; only 2 nodes remain.
        assert_eq!(new_dag.len(), 2);
    }

    #[test]
    fn dce_keeps_store_nodes() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Store { name: "out".into() },
            vec![a],
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
        let live = dag.add_node(decl, RiscOp::Neg, vec![b], scalar_f32(), None);
        dag.add_root(live);

        let new_dag = dead_code_eliminate(&dag);
        // Store + its input const + second const + neg = 4 nodes all live.
        assert_eq!(new_dag.len(), 4);
    }

    #[test]
    fn cse_keeps_each_key_operation_and_its_draw() {
        // Two `key_from_seed` nodes over one seed produce equal bits, but
        // merging them would hand one key to two draws, which the key rules
        // reject (spec/10 section 3.2, V2): CSE keeps both key operations and
        // both draws, and the two draws, as pure functions of equal keys,
        // produce equal values.
        use chelis_types::types::Prim;
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let template = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::F32, 0.0),
            vec![],
            scalar_f32(),
            None,
        );
        let seed = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::Int64, 42.0),
            vec![],
            TensorType {
                dims: vec![],
                precision: Prim::Int64,
            },
            None,
        );
        let low = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::F32, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        let high = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::F32, 5.0),
            vec![],
            scalar_f32(),
            None,
        );
        for _ in 0..2 {
            let key = dag.add_node(
                decl,
                RiscOp::KeyFromSeed,
                vec![seed],
                TensorType {
                    dims: vec![],
                    precision: Prim::Key,
                },
                None,
            );
            let draw = dag.add_node(
                decl,
                RiscOp::UniformLike,
                vec![template, low, high, key],
                scalar_f32(),
                None,
            );
            dag.add_root(draw);
        }
        let optimized = common_subexpr_eliminate(&dag);
        assert_eq!(crate::verify::verify(&optimized), Vec::<String>::new());
        assert_eq!(
            optimized
                .nodes()
                .iter()
                .filter(|node| matches!(node.op, RiscOp::KeyFromSeed))
                .count(),
            2,
            "equal key operations must not merge"
        );
        assert_eq!(optimized.roots().len(), 2);
        assert_ne!(optimized.roots()[0], optimized.roots()[1]);
        let values =
            crate::eval::eval_tensor_roots_exact(&optimized, optimized.roots(), |_| None).unwrap();
        assert_eq!(values[&optimized.roots()[0]], values[&optimized.roots()[1]]);
    }

    #[test]
    fn cse_deduplicates_consts() {
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
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let sum = dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_f32(), None);
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
        dag.add_root(a);
        dag.add_root(b);

        let new_dag = dead_code_eliminate(&dag);
        assert_eq!(new_dag.len(), 2);
        assert_eq!(new_dag.roots().len(), 2);
    }
}
