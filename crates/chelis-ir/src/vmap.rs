use chelis_unord::{UnordMap, UnordSet};

use crate::dag::{Dag, DimInfo, NodeId, RiscOp, RtAxis, RtDim, TensorType};

pub fn vectorize_axis0(dag: &Dag, batch_dim: DimInfo) -> Result<Dag, String> {
    vectorize_axis0_with_node_map(dag, batch_dim).map(|(batched, _)| batched)
}

/// Vectorize `dag` while treating the named lexical loads as loop-invariant
/// captures rather than mapped formals.
///
/// A capture keeps its authored load type. Its mapped identity is an explicit
/// rank-inserting [`RiscOp::Expand`] (the IR representation of source
/// `insert`) so downstream elementwise nodes still receive shape-equal
/// operands without acquiring implicit broadcasting semantics.
pub fn vectorize_axis0_with_captures(
    dag: &Dag,
    batch_dim: DimInfo,
    captured_loads: &UnordSet<String>,
) -> Result<Dag, String> {
    vectorize_axis0_with_node_map_and_captures(dag, batch_dim, captured_loads)
        .map(|(batched, _)| batched)
}

/// [`vectorize_axis0`] plus the batched id of every input node, indexed by the
/// input `NodeId`.
///
/// The rebuild is not id-preserving in general: a shared extent scalar gains a
/// batch-expansion node, which shifts every later id. A caller that has to name
/// one specific input node inside the batched DAG therefore needs this map
/// rather than the id it started with (chelis#1821 names the forward activation
/// under `vmap(grad(...))` so it survives dead-code elimination).
pub fn vectorize_axis0_with_node_map(
    dag: &Dag,
    batch_dim: DimInfo,
) -> Result<(Dag, Vec<NodeId>), String> {
    vectorize_axis0_with_node_map_and_captures(dag, batch_dim, &UnordSet::new())
}

/// [`vectorize_axis0_with_captures`] plus the mapped identity of every source
/// node. Captured `Load`s map to their explicit batch lift, not to the raw
/// authored-rank load.
pub fn vectorize_axis0_with_node_map_and_captures(
    dag: &Dag,
    batch_dim: DimInfo,
    captured_loads: &UnordSet<String>,
) -> Result<(Dag, Vec<NodeId>), String> {
    vectorize_axis0_impl(dag, batch_dim, captured_loads, false)
}

/// List callbacks retain scalar capture-edge provenance for reverse mode;
/// authored vmap continues to use the ordinary Expand adjoint.
pub fn vectorize_list_map(
    dag: &Dag,
    batch_dim: DimInfo,
    captured_loads: &UnordSet<String>,
) -> Result<Dag, String> {
    vectorize_axis0_impl(dag, batch_dim, captured_loads, true).map(|(dag, _)| dag)
}

fn vectorize_axis0_impl(
    dag: &Dag,
    batch_dim: DimInfo,
    captured_loads: &UnordSet<String>,
    list_map: bool,
) -> Result<(Dag, Vec<NodeId>), String> {
    let mut out = Dag::new();
    out.inherit_declarations(dag);
    let concrete_batch = match &batch_dim {
        DimInfo::Lit(size) => Some(*size),
        DimInfo::Named(_, Some(size)) => Some(*size),
        DimInfo::Named(_, None) => None,
    };
    // `ToEnd` is the structural full-axis sentinel only while the mapped
    // batch remains symbolic. Once vmap knows the batch cardinality, spell
    // the bound as that exact literal: downstream backends deliberately
    // reject a sentinel paired with a concrete result dimension because it
    // can otherwise conceal a malformed movement rewrite.
    let batch_shrink_end = concrete_batch.map(RtDim::Lit).unwrap_or(RtDim::ToEnd);

    // A node-valued movement extent is one scalar for the whole mapped
    // invocation, not one scalar per element. Prove that its dependency
    // closure is shape-only (or a scalar parameter/constant) before cloning.
    // An element-derived scalar would make the result ragged, which the dense
    // tensor IR cannot represent.
    let shared_bound_nodes = shared_bound_nodes(dag)?;
    // A captured key that nothing reads is not broadcast to any row; the
    // lexical scope seeds a Load for every enclosing binding, used or not.
    let read_nodes = dag
        .nodes()
        .iter()
        .flat_map(|node| node.inputs.iter().chain(node.shape_deps.iter()).copied())
        .chain(dag.roots().iter().copied())
        .collect::<UnordSet<NodeId>>();
    let mut mapped_ids = Vec::with_capacity(dag.nodes().len());
    let mut expanded_shared = UnordMap::<NodeId, NodeId>::new();
    let mut list_invocation = None;

    for node in dag.nodes() {
        let shared = shared_bound_nodes.contains(&node.id);
        let captured_load = matches!(
            &node.op,
            RiscOp::Load { name } if captured_loads.contains(name.as_str())
        );
        // spec/design/randomness_explicit_keys.md §3: each row of a vmapped
        // draw consumes its own row of a mapped `tensor[n, key]`. Broadcasting
        // one captured key to every row would consume it once per row
        // (chelis#2409).
        if captured_load
            && node.output_type.precision == chelis_types::types::Prim::Key
            && read_nodes.contains(&node.id)
        {
            return Err(format!(
                "vmap cannot broadcast captured key {:?} to every row; a vmapped key must be a mapped tensor of keys",
                node.op
            ));
        }
        let output_type = if shared {
            node.output_type.clone()
        } else {
            prepend_batch_type(&node.output_type, &batch_dim)
        };
        let op = match &node.op {
            RiscOp::ListMapCapture { .. } | RiscOp::OrderedAdjointSum { .. } => {
                return Err(
                    "batching an ordered List cotangent requires nested invocation provenance"
                        .into(),
                );
            }
            RiscOp::Iota => {
                return Err(
                    "runtime range inside vmap requires a shared cardinality representation".into(),
                );
            }
            RiscOp::Sum { axis, accumulator } => RiscOp::Sum {
                axis: axis + 1,
                accumulator: *accumulator,
            },
            RiscOp::Count { axes } => RiscOp::Count {
                axes: axes.iter().map(|axis| axis + 1).collect(),
            },
            RiscOp::MaxReduce { axis } => RiscOp::MaxReduce { axis: axis + 1 },
            RiscOp::MinReduce { axis } => RiscOp::MinReduce { axis: axis + 1 },
            RiscOp::ProdReduce { axis } => RiscOp::ProdReduce { axis: axis + 1 },
            RiscOp::Argmax { axis } => RiscOp::Argmax { axis: axis + 1 },
            RiscOp::Argmin { axis } => RiscOp::Argmin { axis: axis + 1 },
            RiscOp::Reshape { new_shape } => RiscOp::Reshape {
                // `RtDim::Node` slots index this node's `inputs`, which vmap
                // clones verbatim, so prepending a target axis shifts nothing.
                new_shape: std::iter::once(RtDim::from_dim_info(&batch_dim))
                    .chain(new_shape.iter().map(shift_input_axis))
                    .collect(),
            },
            RiscOp::Permute { axes } => RiscOp::Permute {
                axes: std::iter::once(0)
                    .chain(axes.iter().map(|axis| axis + 1))
                    .collect(),
            },
            RiscOp::Expand { axis, size } => RiscOp::Expand {
                axis: axis + 1,
                size: shift_input_axis(size),
            },
            RiscOp::Pad { padding, fill } => RiscOp::Pad {
                padding: std::iter::once((RtDim::Lit(0), RtDim::Lit(0)))
                    .chain(
                        padding
                            .iter()
                            .map(|(start, end)| (shift_input_axis(start), shift_input_axis(end))),
                    )
                    .collect(),
                fill: *fill,
            },
            RiscOp::Shrink { bounds } => RiscOp::Shrink {
                bounds: std::iter::once((RtDim::Lit(0), batch_shrink_end.clone()))
                    .chain(
                        bounds
                            .iter()
                            .map(|(start, end)| (shift_input_axis(start), shift_input_axis(end))),
                    )
                    .collect(),
            },
            RiscOp::Stride { strides } => RiscOp::Stride {
                strides: std::iter::once(RtDim::Lit(1))
                    .chain(strides.iter().map(shift_input_axis))
                    .collect(),
            },
            RiscOp::Shape { axis } if shared => RiscOp::Shape { axis: axis + 1 },
            RiscOp::ExtentWitness {
                site,
                parameter,
                axis: RtAxis::Lit(axis),
                requirements,
                claims,
            } => RiscOp::ExtentWitness {
                site: match site {
                    crate::dag::ExtentWitnessSite::ResultClaim {
                        claim,
                        axis: RtAxis::Lit(axis),
                    } => crate::dag::ExtentWitnessSite::ResultClaim {
                        claim: claim.clone(),
                        axis: RtAxis::Lit(axis.checked_add(1).expect("vmap result axis fits i32")),
                    },
                    crate::dag::ExtentWitnessSite::LocalAscriptionClaim {
                        ascription_id,
                        binding,
                        claim,
                        axis: RtAxis::Lit(axis),
                    } => crate::dag::ExtentWitnessSite::LocalAscriptionClaim {
                        ascription_id: *ascription_id,
                        binding: binding.clone(),
                        claim: claim.clone(),
                        axis: RtAxis::Lit(
                            axis.checked_add(1)
                                .expect("vmap local claim axis fits int32"),
                        ),
                    },
                    other => other.clone(),
                },
                parameter: parameter.clone(),
                axis: RtAxis::Lit(axis.checked_add(1).expect("vmap axis fits i32")),
                requirements: requirements.clone(),
                // A named claim relates two witnesses, and both shift by the
                // same prepended batch axis, so the obligation is unchanged.
                claims: claims.clone(),
            },
            RiscOp::CheckedReshapeExtent {
                claims,
                axis: RtAxis::Lit(axis),
            } => RiscOp::CheckedReshapeExtent {
                claims: claims.clone(),
                axis: RtAxis::Lit(axis + 1),
            },
            RiscOp::CheckedUnitAxis {
                axis: RtAxis::Lit(axis),
            } => RiscOp::CheckedUnitAxis {
                axis: RtAxis::Lit(axis.checked_add(1).expect("vmap axis fits i32")),
            },
            RiscOp::Load { name } => RiscOp::Load { name: name.clone() },
            other => other.clone(),
        };

        // The activation maps like a value input: an `if` over a row makes
        // it the row's Bool, so a batched node checks row by row.
        let owner = node
            .owner
            .try_remap_with(|activation| mapped_ids.get(activation.0).copied())
            .map_err(|message| format!("vmap node {}: {message}", node.id.0))?;
        let bound_slots = bound_input_slots(&node.op);
        let batch_witness = node
            .inputs
            .iter()
            .copied()
            .find(|input| !shared_bound_nodes.contains(input))
            .map(|input| mapped_ids[input.0])
            .or_else(|| {
                mapped_ids.iter().enumerate().find_map(|(index, mapped)| {
                    (!shared_bound_nodes.contains(&NodeId(index))).then_some(*mapped)
                })
            });
        let mut inputs = Vec::with_capacity(node.inputs.len());
        for (slot, input) in node.inputs.iter().copied().enumerate() {
            let mapped = mapped_ids[input.0];
            if shared || bound_slots.contains(&slot) || !shared_bound_nodes.contains(&input) {
                inputs.push(mapped);
                continue;
            }

            let expanded = if let Some(expanded) = expanded_shared.get(&input) {
                *expanded
            } else {
                let (size, expand_inputs) = match concrete_batch {
                    Some(batch) => (RtDim::Lit(batch), vec![mapped]),
                    None => {
                        let witness = batch_witness.ok_or_else(|| {
                            "vmap cannot locate a batched tensor witness for a shared extent \
                             scalar used as ordinary data"
                                .to_string()
                        })?;
                        (
                            RtDim::InputAxis {
                                tensor: 1,
                                axis: RtAxis::Lit(0),
                            },
                            vec![mapped, witness],
                        )
                    }
                };
                let expanded = out.add_node(
                    owner,
                    RiscOp::Expand { axis: 0, size },
                    expand_inputs,
                    prepend_batch_type(&dag.get(input).unwrap().output_type, &batch_dim),
                    node.span_id.clone(),
                );
                expanded_shared.insert(input, expanded);
                expanded
            };
            inputs.push(expanded);
        }

        // A lexical capture is one loop-invariant value, not an additional
        // mapped argument. Preserve its authored-rank Load and make the
        // source node's mapped identity an explicit inserted batch axis.
        // This is deliberately the same structural movement used for an
        // authored constant payload; elementwise operators remain exact-
        // shape operations.
        if !shared && captured_load {
            let raw = out.add_node(
                owner,
                node.op.clone(),
                Vec::new(),
                node.output_type.clone(),
                node.span_id.clone(),
            );
            if let Some(raw_node) = out.node_mut(raw)
                && !node.merged_spans.is_empty()
            {
                raw_node.merged_spans = node.merged_spans.clone();
            }
            let (size, expand_inputs) = match concrete_batch {
                Some(batch) => (RtDim::Lit(batch), vec![raw]),
                None => {
                    let witness = batch_witness.ok_or_else(|| {
                        format!(
                            "vmap cannot locate a batched tensor witness for captured load {}",
                            match &node.op {
                                RiscOp::Load { name } => name.as_str(),
                                _ => unreachable!("captured_load only marks Load"),
                            }
                        )
                    })?;
                    (
                        RtDim::InputAxis {
                            tensor: 1,
                            axis: RtAxis::Lit(0),
                        },
                        vec![raw, witness],
                    )
                }
            };
            let (capture_op, expand_inputs) = if list_map
                && node.output_type.precision.is_float()
                && read_nodes.contains(&node.id)
            {
                if !node.output_type.dims.is_empty() {
                    return Err("ordered List capture must be scalar".into());
                }
                let first = list_invocation.is_none();
                let invocation = list_invocation
                    .or(batch_witness)
                    .ok_or("List map has no invocation carrier")?;
                (RiscOp::ListMapCapture { first }, vec![raw, invocation])
            } else {
                (RiscOp::Expand { axis: 0, size }, expand_inputs)
            };
            let new_id = out.add_node(
                owner,
                capture_op,
                expand_inputs,
                output_type,
                node.span_id.clone(),
            );
            if matches!(
                out.get(new_id).unwrap().op,
                RiscOp::ListMapCapture { first: true }
            ) {
                list_invocation = Some(new_id);
            }
            mapped_ids.push(new_id);
            let remapped_shape_deps = remap_shape_deps(node.id, &node.shape_deps, &mapped_ids)?;
            let remapped_result_claims =
                remap_result_claim_deps(node.id, &node.result_claim_deps, &mapped_ids)?;
            if let Some(new_node) = out.node_mut(new_id) {
                new_node.merged_spans = node.merged_spans.clone();
                new_node.shape_deps = remapped_shape_deps;
                new_node.result_claim_deps = remapped_result_claims;
            }
            if let Some(reusable_input) = node.reusable_input {
                out.set_reusable_input(new_id, mapped_ids[reusable_input.0]);
            }
            continue;
        }

        // A non-shared constant has one authored payload, not one
        // payload per mapped lane. Preserve that payload at its original
        // type and make the old node's mapped identity an explicit batch
        // broadcast. Merely prepending the batch dimension while cloning the
        // flat storage creates a malformed constant whose cardinality no
        // longer matches its declared type (chelis#1932).
        if !shared
            && (matches!(node.op, RiscOp::ConstTensor { .. })
                || (matches!(node.op, RiscOp::Const { .. }) && node.output_type.dims.is_empty()))
        {
            let raw_owner = if matches!(node.op, RiscOp::Const { .. }) {
                crate::dag::Owner::new(owner.decl, None)
            } else {
                owner
            };
            let raw = out.add_node(
                raw_owner,
                node.op.clone(),
                Vec::new(),
                node.output_type.clone(),
                node.span_id.clone(),
            );
            if let Some(raw_node) = out.node_mut(raw)
                && !node.merged_spans.is_empty()
            {
                raw_node.merged_spans = node.merged_spans.clone();
            }
            let (size, expand_inputs) = match concrete_batch {
                Some(batch) => (RtDim::Lit(batch), vec![raw]),
                None => {
                    let witness = batch_witness.ok_or_else(|| {
                        "vmap cannot locate a batched tensor witness for a constant tensor"
                            .to_string()
                    })?;
                    (
                        RtDim::InputAxis {
                            tensor: 1,
                            axis: RtAxis::Lit(0),
                        },
                        vec![raw, witness],
                    )
                }
            };
            let new_id = out.add_node(
                owner,
                RiscOp::Expand { axis: 0, size },
                expand_inputs,
                output_type,
                node.span_id.clone(),
            );
            mapped_ids.push(new_id);
            let remapped_shape_deps = remap_shape_deps(node.id, &node.shape_deps, &mapped_ids)?;
            let remapped_result_claims =
                remap_result_claim_deps(node.id, &node.result_claim_deps, &mapped_ids)?;
            if let Some(new_node) = out.node_mut(new_id) {
                new_node.merged_spans = node.merged_spans.clone();
                new_node.shape_deps = remapped_shape_deps;
                new_node.result_claim_deps = remapped_result_claims;
            }
            if let Some(reusable_input) = node.reusable_input {
                out.set_reusable_input(new_id, mapped_ids[reusable_input.0]);
            }
            continue;
        }

        // Vmap is a pure clone of the per-node operator (with axis
        // shifts) onto a new DAG. Per spec/design/chelis_span_survival.md
        // §2.3 vmap row, span_id and merged_spans are cloned unchanged
        // — every input span survives the pass.
        let new_id = out.add_node(owner, op, inputs, output_type, node.span_id.clone());
        mapped_ids.push(new_id);
        let remapped_shape_deps = remap_shape_deps(node.id, &node.shape_deps, &mapped_ids)?;
        let remapped_result_claims =
            remap_result_claim_deps(node.id, &node.result_claim_deps, &mapped_ids)?;
        if let Some(new_node) = out.node_mut(new_id) {
            if !node.merged_spans.is_empty() {
                new_node.merged_spans = node.merged_spans.clone();
            }
            // chelis#384/#397: an `expand` shape dependency is remapped
            // through `mapped_ids`, not copied verbatim. This rebuild is NOT
            // an id-preserving clone in general: a shared extent scalar with
            // an ordinary consumer gains a batch expansion that shifts every
            // later id, which is exactly why `vectorize_axis0_with_node_map`
            // returns the mapping. The remap below is therefore the
            // correctness step, not a no-op that happens to look like one.
            new_node.shape_deps = remapped_shape_deps;
            // A scalar comparison has no axes to witness before batching.
            // Its new anonymous batch axis belongs to its actual operand,
            // just as the corresponding source-level tensor operation does.
            if !shared
                && matches!(new_node.op, RiscOp::Compare(_) | RiscOp::Logical(_))
                && let Some(&source) = new_node.inputs.first()
                && !new_node.shape_deps.contains(&source)
            {
                new_node.shape_deps.push(source);
            }
            new_node.result_claim_deps = remapped_result_claims;
        }
        if let Some(reusable_input) = node.reusable_input {
            out.set_reusable_input(new_id, mapped_ids[reusable_input.0]);
        }
    }

    for root in dag.roots() {
        let mapped = mapped_ids[root.0];
        if !shared_bound_nodes.contains(root) {
            out.add_root(mapped);
            continue;
        }
        let expanded = if let Some(expanded) = expanded_shared.get(root) {
            *expanded
        } else {
            let (size, expand_inputs) = match concrete_batch {
                Some(batch) => (RtDim::Lit(batch), vec![mapped]),
                None => {
                    let witness = mapped_ids
                        .iter()
                        .enumerate()
                        .find_map(|(index, mapped)| {
                            (!shared_bound_nodes.contains(&NodeId(index))).then_some(*mapped)
                        })
                        .ok_or_else(|| {
                            "vmap cannot locate a batched tensor witness for a shared extent \
                             scalar root"
                                .to_string()
                        })?;
                    (
                        RtDim::InputAxis {
                            tensor: 1,
                            axis: RtAxis::Lit(0),
                        },
                        vec![mapped, witness],
                    )
                }
            };
            let root_owner = dag
                .get(*root)
                .unwrap()
                .owner
                .try_remap_with(|activation| mapped_ids.get(activation.0).copied())?;
            let expanded = out.add_node(
                root_owner,
                RiscOp::Expand { axis: 0, size },
                expand_inputs,
                prepend_batch_type(&dag.get(*root).unwrap().output_type, &batch_dim),
                dag.get(*root).unwrap().span_id.clone(),
            );
            expanded_shared.insert(*root, expanded);
            expanded
        };
        out.add_root(expanded);
    }

    Ok((out, mapped_ids))
}

fn remap_shape_deps(
    owner: NodeId,
    source_deps: &[NodeId],
    mapped_ids: &[NodeId],
) -> Result<Vec<NodeId>, String> {
    source_deps
        .iter()
        .map(|dep| {
            mapped_ids.get(dep.0).copied().ok_or_else(|| {
                format!("vmap shape dependency {dep:?} of node {owner:?} has no mapped identity")
            })
        })
        .collect()
}

fn remap_result_claim_deps(
    owner: NodeId,
    source_deps: &[NodeId],
    mapped_ids: &[NodeId],
) -> Result<Vec<NodeId>, String> {
    source_deps
        .iter()
        .map(|dep| {
            mapped_ids.get(dep.0).copied().ok_or_else(|| {
                format!(
                    "vmap result claim dependency {dep:?} of node {owner:?} has no mapped identity"
                )
            })
        })
        .collect()
}

fn shift_input_axis(dim: &RtDim) -> RtDim {
    match dim {
        RtDim::InputAxis {
            tensor,
            axis: RtAxis::Lit(axis),
        } => RtDim::InputAxis {
            tensor: *tensor,
            axis: RtAxis::Lit(axis.checked_add(1).expect("vmap axis fits i32")),
        },
        other => other.clone(),
    }
}

fn bound_input_slots(op: &RiscOp) -> UnordSet<usize> {
    let mut slots = UnordSet::new();
    let mut add = |dim: &RtDim| {
        if let RtDim::Node(slot) = dim {
            slots.insert(*slot);
        }
    };
    match op {
        RiscOp::Expand { size, .. } => add(size),
        // A key split's count is one shared extent for every row.
        RiscOp::SplitN { count } => add(count),
        RiscOp::Reshape { new_shape } => new_shape.iter().for_each(add),
        RiscOp::Pad { padding, .. } | RiscOp::Shrink { bounds: padding } => {
            for (start, end) in padding {
                add(start);
                add(end);
            }
        }
        RiscOp::Stride { strides } => strides.iter().for_each(add),
        RiscOp::CheckedUnitAxis { .. } => {
            slots.insert(1);
        }
        RiscOp::CheckedReshapeExtent { claims, .. } => {
            slots.extend(0..=claims.len());
        }
        _ => {}
    }
    slots
}

fn shared_bound_nodes(dag: &Dag) -> Result<UnordSet<NodeId>, String> {
    let mut shared = UnordSet::new();
    for owner in dag.nodes() {
        if matches!(owner.op, RiscOp::ExtentWitness { .. }) {
            shared.insert(owner.id);
        }
        // A shape read stays rank zero even when used only as ordinary data
        // or retained forward work. Its operand is still batched; the read's
        // axis shifts above, and existing consumer/root expansion broadcasts
        // its scalar result (spec/05 §2.5.1, spec/06 §3.7; chelis#2003).
        if matches!(
            owner.op,
            RiscOp::CheckedReshapeExtent { .. } | RiscOp::Shape { .. }
        ) {
            mark_shared_bound(dag, owner.id, &mut shared)?;
        }
        // Operand-slot order is the IR's canonical order for this dependency walk.
        for slot in bound_input_slots(&owner.op).into_sorted() {
            let Some(source) = owner.inputs.get(slot).copied() else {
                continue;
            };
            mark_shared_bound(dag, source, &mut shared)?;
        }
    }
    // Once an extent producer is shared, a scalar-only computation over
    // shared values is shared too. This preserves the spec/06 §3.7 rule for
    // a shape read used through a cast/arithmetic chain both as a bound and
    // as an ordinary value; the chain executes once and is broadcast only at
    // its first batched consumer or root.
    loop {
        let mut changed = false;
        for node in dag.nodes() {
            if shared.contains(&node.id)
                || !node.output_type.dims.is_empty()
                || node.inputs.is_empty()
                || !node.inputs.iter().all(|input| shared.contains(input))
                || !shared_scalar_op(&node.op)
            {
                continue;
            }
            shared.insert(node.id);
            changed = true;
        }
        if !changed {
            break;
        }
    }
    Ok(shared)
}

fn shared_scalar_op(op: &RiscOp) -> bool {
    matches!(
        op,
        RiscOp::CheckedReshapeExtent { .. }
            | RiscOp::Cast { .. }
            | RiscOp::CastTrunc { .. }
            | RiscOp::Copy
            | RiscOp::Realize
            | RiscOp::Neg
            | RiscOp::Abs
            | RiscOp::Add
            | RiscOp::Sub
            | RiscOp::Mul
            | RiscOp::FloorDiv
            | RiscOp::TruncDiv
            | RiscOp::Mod
            | RiscOp::Bitwise(_)
            | RiscOp::MaxElem
            | RiscOp::MinElem
    )
}

fn mark_shared_bound(dag: &Dag, id: NodeId, shared: &mut UnordSet<NodeId>) -> Result<(), String> {
    if shared.contains(&id) {
        return Ok(());
    }
    let node = dag.get(id).ok_or_else(|| {
        format!(
            "batch_varying_extent: bound references missing node {}",
            id.0
        )
    })?;
    if !node.output_type.dims.is_empty() {
        return Err(format!(
            "batch_varying_extent: node {} produces rank {} rather than one shared scalar",
            id.0,
            node.output_type.dims.len()
        ));
    }

    match &node.op {
        RiscOp::Shape { .. }
        | RiscOp::ExtentWitness { .. }
        | RiscOp::Const { .. }
        | RiscOp::Load { .. } => {}
        RiscOp::Cast { .. }
        | RiscOp::CastTrunc { .. }
        | RiscOp::Copy
        | RiscOp::Realize
        | RiscOp::Neg
        | RiscOp::Abs => {
            for input in &node.inputs {
                mark_shared_bound(dag, *input, shared)?;
            }
        }
        RiscOp::CheckedReshapeExtent { .. }
        | RiscOp::Add
        | RiscOp::Sub
        | RiscOp::Mul
        | RiscOp::FloorDiv
        | RiscOp::TruncDiv
        | RiscOp::Mod
        | RiscOp::Bitwise(_)
        | RiscOp::MaxElem
        | RiscOp::MinElem => {
            for input in &node.inputs {
                mark_shared_bound(dag, *input, shared)?;
            }
        }
        other => {
            return Err(format!(
                "batch_varying_extent: node {} derives a movement extent from element data via {other:?}",
                id.0
            ));
        }
    }
    shared.insert(id);
    Ok(())
}

fn prepend_batch_type(ty: &TensorType, batch_dim: &DimInfo) -> TensorType {
    TensorType {
        dims: prepend_batch_dims(&ty.dims, batch_dim),
        precision: ty.precision,
    }
}

fn prepend_batch_dims(dims: &[DimInfo], batch_dim: &DimInfo) -> Vec<DimInfo> {
    let mut out = Vec::with_capacity(dims.len() + 1);
    out.push(batch_dim.clone());
    out.extend(dims.iter().cloned());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{NodeId, RiscOp, TensorType};
    use crate::eval::{TensorValue, eval_tensor_roots_with_strict};
    use chelis_types::types::Prim;
    use chelis_unord::UnordMap;

    fn vec_f32(n: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(n)],
            precision: Prim::F32,
        }
    }

    fn mat_f32(m: usize, n: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(m), DimInfo::Lit(n)],
            precision: Prim::F32,
        }
    }

    fn batched_mat_f32(batch: usize, m: usize, n: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(batch), DimInfo::Lit(m), DimInfo::Lit(n)],
            precision: Prim::F32,
        }
    }

    fn eval_root(dag: &Dag, root: NodeId, inputs: &UnordMap<String, TensorValue>) -> TensorValue {
        let values = eval_tensor_roots_with_strict(dag, &[root], |name| inputs.get(name).cloned())
            .expect("evaluation should succeed");
        values[&root].clone()
    }

    #[test]
    fn elementwise_vmap_prepends_batch_axis() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec_f32(3),
            None,
        );
        let y = dag.add_node(decl, RiscOp::Neg, vec![x], vec_f32(3), None);
        dag.add_root(y);

        let vmapped = vectorize_axis0(&dag, DimInfo::Lit(2)).expect("vmap should succeed");
        let root = vmapped.roots()[0];
        let actual = eval_root(
            &vmapped,
            root,
            &UnordMap::from([(
                "x".to_string(),
                TensorValue::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, -5.0, 6.0]),
            )]),
        );
        assert_eq!(actual.shape, vec![2, 3]);
        assert_eq!(
            actual.to_f64_lossy_vec(),
            vec![-1.0, -2.0, -3.0, -4.0, 5.0, -6.0]
        );
    }

    #[test]
    fn reduction_vmap_shifts_axis() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            mat_f32(2, 3),
            None,
        );
        let y = dag.add_node(
            decl,
            RiscOp::Sum {
                axis: 1,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![x],
            vec_f32(2),
            None,
        );
        dag.add_root(y);

        let vmapped = vectorize_axis0(&dag, DimInfo::Lit(2)).expect("vmap should succeed");
        let root = vmapped.roots()[0];
        let actual = eval_root(
            &vmapped,
            root,
            &UnordMap::from([(
                "x".to_string(),
                TensorValue::from_vec(
                    vec![2, 2, 3],
                    vec![
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 7.0, 8.0, 9.0,
                    ],
                ),
            )]),
        );
        assert_eq!(actual.shape, vec![2, 2]);
        assert_eq!(actual.to_f64_lossy_vec(), vec![6.0, 15.0, 60.0, 24.0]);
    }

    #[test]
    fn nested_vmap_adds_two_batch_axes() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec_f32(4),
            None,
        );
        dag.add_root(x);

        let inner = vectorize_axis0(&dag, DimInfo::Lit(3)).expect("inner vmap should succeed");
        let outer = vectorize_axis0(&inner, DimInfo::Lit(2)).expect("outer vmap should succeed");
        let root = outer.roots()[0];
        let node = outer.get(root).expect("root node");
        assert_eq!(
            node.output_type,
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            }
        );
    }

    #[test]
    fn batched_matmul_shape_matches_expand_mul_sum_pattern() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::Load { name: "a".into() },
            vec![],
            mat_f32(2, 3),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::Load { name: "b".into() },
            vec![],
            mat_f32(3, 4),
            None,
        );
        let a_exp = dag.add_node(
            decl,
            RiscOp::Expand {
                axis: 2,
                size: RtDim::Lit(4),
            },
            vec![a],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let b_exp = dag.add_node(
            decl,
            RiscOp::Expand {
                axis: 0,
                size: RtDim::Lit(2),
            },
            vec![b],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let prod = dag.add_node(
            decl,
            RiscOp::Mul,
            vec![a_exp, b_exp],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let out = dag.add_node(
            decl,
            RiscOp::Sum {
                axis: 1,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![prod],
            mat_f32(2, 4),
            None,
        );
        dag.add_root(out);

        let vmapped = vectorize_axis0(&dag, DimInfo::Lit(5)).expect("vmap should succeed");
        let root = vmapped.roots()[0];
        let node = vmapped.get(root).expect("root");
        assert_eq!(node.output_type, batched_mat_f32(5, 2, 4));
        assert!(
            vmapped
                .nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::Expand { axis: 3, .. })),
            "expected vmap to preserve the batched matmul decomposition shape"
        );
    }
}
