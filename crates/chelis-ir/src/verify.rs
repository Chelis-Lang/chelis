//! DAG structural verification.

use crate::dag::{
    ComparisonKind, Dag, DimExpr, DimInfo, ExtentWitnessSite, NodeId, RiscOp, RtAxis, RtDim,
};
#[allow(unused_imports)]
use chelis_types::types::Prim;
use chelis_unord::{UnordMap, UnordSet};

/// Verify structural invariants of the DAG. Returns a list of error messages (empty = valid).
// `collapsible_match` (rust 1.95+) flags `match { X => { if cond { ... } } }` patterns.
// The validation arms here have large bodies and no else branch, so converting to match
// guards would require dedenting ~90 lines per arm with no readability win; the `if`
// inside an otherwise-empty arm body is the clearer expression of intent.
#[allow(clippy::collapsible_match)]
pub fn verify(dag: &Dag) -> Vec<String> {
    verify_with_dangling_policy(dag, true)
}

/// Structural verifier used immediately before ownership lowering. Dangling
/// producers are intentionally admitted here because the ownership plan
/// attaches their required `ScopeDrop`; every other DAG invariant remains
/// identical to [`verify`].
pub(crate) fn verify_ownership_input(dag: &Dag) -> Vec<String> {
    verify_with_dangling_policy(dag, false)
}

fn scalar_axis_source(dag: &Dag, value: NodeId) -> Option<(NodeId, usize)> {
    let scalar = dag.get(value)?;
    match &scalar.op {
        RiscOp::Shape { axis } => Some((*scalar.inputs.first()?, *axis)),
        RiscOp::ExtentWitness {
            axis: RtAxis::Lit(axis),
            ..
        } => Some((*scalar.inputs.first()?, usize::try_from(*axis).ok()?)),
        _ => None,
    }
}

fn anonymous_declared_shape_source(
    dag: &Dag,
    owner: &crate::dag::DagNode,
    relevant_shape_sources: &[NodeId],
) -> Option<NodeId> {
    if !matches!(owner.op, RiscOp::Const { .. } | RiscOp::ConstTensor { .. }) {
        return None;
    }
    owner.shape_deps.iter().copied().find(|source| {
        relevant_shape_sources.contains(source)
            && dag
                .get(*source)
                .is_some_and(|source| source.output_type.dims.len() == owner.output_type.dims.len())
    })
}

fn witnessed_extent_origin_equal(
    dag: &Dag,
    left: &crate::axis_sources::ExtentOrigin,
    right: &crate::axis_sources::ExtentOrigin,
    relevant_shape_sources: &[NodeId],
) -> bool {
    if left == right {
        return true;
    }
    let cutoff = relevant_shape_sources
        .iter()
        .map(|node| node.0)
        .max()
        .unwrap_or(dag.len());
    let observed = |witness: NodeId| {
        let node = dag.get(witness)?;
        let RiscOp::ExtentWitness {
            axis: RtAxis::Lit(axis),
            ..
        } = node.op
        else {
            return None;
        };
        crate::axis_sources::resolve_axis_extent(
            dag,
            *node.inputs.first()?,
            usize::try_from(axis).ok()?,
        )
    };
    let mut edges = Vec::new();
    for node in dag.nodes().iter().take(cutoff) {
        let RiscOp::ExtentWitness { claims, .. } = &node.op else {
            continue;
        };
        let Some(here) = observed(node.id) else {
            continue;
        };
        for requirement in node.inputs.iter().skip(1).take(claims.len()) {
            if requirement.0 >= node.id.0 {
                continue;
            }
            if let Some(there) = observed(*requirement) {
                edges.push((here.clone(), there));
            }
        }
    }
    let mut pending = vec![left.clone()];
    let mut seen = Vec::new();
    while let Some(origin) = pending.pop() {
        if &origin == right {
            return true;
        }
        if seen.contains(&origin) {
            continue;
        }
        seen.push(origin.clone());
        for (a, b) in &edges {
            if a == &origin {
                pending.push(b.clone());
            } else if b == &origin {
                pending.push(a.clone());
            }
        }
    }
    false
}

fn semantic_dim_expr(
    dag: &Dag,
    node: NodeId,
    axis: usize,
    fuel: usize,
    relevant_shape_sources: &[NodeId],
) -> Option<DimExpr> {
    if fuel == 0 {
        return None;
    }
    let owner = dag.get(node)?;
    let dim = owner.output_type.dims.get(axis)?;
    if matches!(
        dim,
        DimInfo::Named(name, None) if name.is_empty() || name == "*"
    ) {
        if let Some(source) = anonymous_declared_shape_source(dag, owner, relevant_shape_sources) {
            return semantic_dim_expr(dag, source, axis, fuel - 1, relevant_shape_sources);
        }
        if matches!(owner.op, RiscOp::Where)
            && let Some(dim) = owner.inputs.iter().skip(1).find_map(|source| {
                let source = dag.get(*source)?;
                (source.output_type.dims.len() == owner.output_type.dims.len())
                    .then(|| {
                        semantic_dim_expr(dag, source.id, axis, fuel - 1, relevant_shape_sources)
                    })
                    .flatten()
                    .filter(
                        |dim| !matches!(dim, DimExpr::Sym(name) if name.is_empty() || name == "*"),
                    )
            })
        {
            return Some(dim);
        }
        return None;
    }
    Some(DimExpr::from(dim))
}

fn semantic_axis_origin(
    dag: &Dag,
    node: NodeId,
    axis: usize,
    fuel: usize,
    relevant_shape_sources: &[NodeId],
) -> Option<crate::axis_sources::ExtentOrigin> {
    if fuel == 0 {
        return None;
    }
    let owner = dag.get(node)?;
    if let Ok(Some(agreement)) = crate::axis_sources::same_shape_result_agreement(dag, node) {
        let mut origins = agreement.members().iter().map(|source| {
            semantic_axis_origin(dag, *source, axis, fuel - 1, relevant_shape_sources)
        });
        let first = origins.next()??;
        return origins
            .all(|origin| {
                origin.is_some_and(|origin| {
                    witnessed_extent_origin_equal(dag, &first, &origin, relevant_shape_sources)
                })
            })
            .then_some(first);
    }
    if matches!(
        owner.output_type.dims.get(axis),
        Some(DimInfo::Named(name, None)) if name.is_empty() || name == "*"
    ) {
        if let Some(source) = anonymous_declared_shape_source(dag, owner, relevant_shape_sources) {
            return semantic_axis_origin(dag, source, axis, fuel - 1, relevant_shape_sources);
        }
        if matches!(owner.op, RiscOp::Where)
            && let Some(origin) = owner.inputs.iter().skip(1).find_map(|source| {
                semantic_axis_origin(dag, *source, axis, fuel - 1, relevant_shape_sources)
            })
        {
            return Some(origin);
        }
    }
    let origin = crate::axis_sources::resolve_axis_extent(dag, node, axis)?;
    match origin {
        crate::axis_sources::ExtentOrigin::ScalarInput { value, at, axis } => {
            let Some((source, read_axis)) = scalar_axis_source(dag, value) else {
                return Some(crate::axis_sources::ExtentOrigin::ScalarInput { value, at, axis });
            };
            semantic_axis_origin(dag, source, read_axis, fuel - 1, relevant_shape_sources)
        }
        other => Some(other),
    }
}

fn static_axis_extent(
    dag: &Dag,
    node: NodeId,
    axis: usize,
    fuel: usize,
    relevant_shape_sources: &[NodeId],
) -> Option<usize> {
    if fuel == 0 {
        return None;
    }
    let owner = dag.get(node)?;
    if let Ok(Some(agreement)) = crate::axis_sources::same_shape_result_agreement(dag, node) {
        let mut extents = agreement
            .members()
            .iter()
            .map(|source| static_axis_extent(dag, *source, axis, fuel - 1, relevant_shape_sources));
        let first = extents.next()??;
        return extents.all(|extent| extent == Some(first)).then_some(first);
    }
    if matches!(
        owner.output_type.dims.get(axis),
        Some(DimInfo::Named(name, None)) if name.is_empty() || name == "*"
    ) {
        if let Some(source) = anonymous_declared_shape_source(dag, owner, relevant_shape_sources) {
            return static_axis_extent(dag, source, axis, fuel - 1, relevant_shape_sources);
        }
        if matches!(owner.op, RiscOp::Where)
            && let Some(extent) = owner.inputs.iter().skip(1).find_map(|source| {
                static_axis_extent(dag, *source, axis, fuel - 1, relevant_shape_sources)
            })
        {
            return Some(extent);
        }
    }
    if let Some(extent) = owner
        .output_type
        .dims
        .get(axis)
        .map(DimExpr::from)
        .and_then(|dim| dim.as_concrete())
    {
        return Some(extent);
    }
    match crate::axis_sources::resolve_axis_extent(dag, node, axis)? {
        crate::axis_sources::ExtentOrigin::Literal(extent) => usize::try_from(extent).ok(),
        crate::axis_sources::ExtentOrigin::ExternalAxis { load, axis } => dag
            .get(load)?
            .output_type
            .dims
            .get(axis)
            .map(DimExpr::from)
            .and_then(|dim| dim.as_concrete()),
        crate::axis_sources::ExtentOrigin::ScalarInput { value, .. } => {
            let (source, read_axis) = scalar_axis_source(dag, value)?;
            static_axis_extent(dag, source, read_axis, fuel - 1, relevant_shape_sources)
        }
        crate::axis_sources::ExtentOrigin::OpComputed { op, axis } => {
            crate::axis_sources::static_op_computed_axis_extent(dag, op, axis)
        }
    }
}

fn node_shapes_semantically_equivalent(
    dag: &Dag,
    left: NodeId,
    right: NodeId,
    relevant_shape_sources: &[NodeId],
) -> bool {
    let (Some(left_node), Some(right_node)) = (dag.get(left), dag.get(right)) else {
        return false;
    };
    left_node.output_type.dims.len() == right_node.output_type.dims.len()
        && left_node
            .output_type
            .dims
            .iter()
            .zip(&right_node.output_type.dims)
            .enumerate()
            .all(|(axis, _)| {
                semantic_dim_expr(dag, left, axis, dag.len(), relevant_shape_sources)
                    .zip(semantic_dim_expr(
                        dag,
                        right,
                        axis,
                        dag.len(),
                        relevant_shape_sources,
                    ))
                    .is_some_and(|(left_dim, right_dim)| left_dim == right_dim)
                    || semantic_axis_origin(dag, left, axis, dag.len(), relevant_shape_sources)
                        .zip(semantic_axis_origin(
                            dag,
                            right,
                            axis,
                            dag.len(),
                            relevant_shape_sources,
                        ))
                        .is_some_and(|(left_origin, right_origin)| {
                            witnessed_extent_origin_equal(
                                dag,
                                &left_origin,
                                &right_origin,
                                relevant_shape_sources,
                            )
                        })
                    || static_axis_extent(dag, left, axis, dag.len(), relevant_shape_sources)
                        .zip(static_axis_extent(
                            dag,
                            right,
                            axis,
                            dag.len(),
                            relevant_shape_sources,
                        ))
                        .is_some_and(|(left_extent, right_extent)| left_extent == right_extent)
            })
}

fn node_types_semantically_equivalent(
    dag: &Dag,
    left: NodeId,
    right: NodeId,
    relevant_shape_sources: &[NodeId],
) -> bool {
    dag.get(left)
        .zip(dag.get(right))
        .is_some_and(|(left_node, right_node)| {
            left_node.output_type.precision == right_node.output_type.precision
                && node_shapes_semantically_equivalent(dag, left, right, relevant_shape_sources)
        })
}

fn has_anonymous_dims(node: &crate::dag::DagNode) -> bool {
    node.output_type
        .dims
        .iter()
        .any(|dim| matches!(dim, DimInfo::Named(name, None) if name.is_empty() || name == "*"))
}

fn anonymous_output_has_input_authority(dag: &Dag, node: &crate::dag::DagNode) -> bool {
    !has_anonymous_dims(node)
        || node.shape_deps.iter().any(|source| {
            node.inputs.contains(source)
                && dag.get(*source).is_some_and(|source| {
                    source.output_type.dims.len() == node.output_type.dims.len()
                })
        })
}

/// Complete correspondence required before a mapped gradient may leave
/// lowering. These are typed node identities, not positional guesses:
/// vectorization and splice both may insert nodes and therefore must provide
/// explicit maps for every authored carrier.
pub(crate) struct MappedGradientClosure<'a> {
    pub source: &'a Dag,
    pub mapped: &'a Dag,
    pub node_map: &'a [NodeId],
    pub root_map: &'a UnordMap<NodeId, NodeId>,
    pub spliced: &'a Dag,
    pub splice_map: &'a UnordMap<NodeId, NodeId>,
    pub forward_source: NodeId,
    pub forward_activation: NodeId,
    pub cotangents: &'a [NodeId],
    pub expected_cotangents: usize,
}

/// Fail closed after the mapped-gradient splice if any authored entry
/// witness, root, shape-only dependency, rendered extent origin, forward
/// activation, or cotangent loses its typed identity.
pub(crate) fn verify_mapped_gradient_closure(
    closure: MappedGradientClosure<'_>,
) -> Result<(), String> {
    let MappedGradientClosure {
        source,
        mapped,
        node_map,
        root_map,
        spliced,
        splice_map,
        forward_source,
        forward_activation,
        cotangents,
        expected_cotangents,
    } = closure;

    if node_map.len() != source.len() {
        return Err(format!(
            "vectorization node map has {} entries for {} source nodes",
            node_map.len(),
            source.len()
        ));
    }
    let mut mapped_identities = UnordSet::new();
    for source_node in source.nodes() {
        let mapped_id = node_map[source_node.id.0];
        let Some(mapped_node) = mapped.get(mapped_id) else {
            return Err(format!(
                "vectorization node map sends {:?} to invalid node {mapped_id:?}",
                source_node.id
            ));
        };
        if !mapped_identities.insert(mapped_id) {
            return Err(format!(
                "vectorization node map gives multiple source nodes the identity {mapped_id:?}"
            ));
        }
        let expected_deps = source_node
            .shape_deps
            .iter()
            .map(|dep| {
                node_map.get(dep.0).copied().ok_or_else(|| {
                    format!(
                        "activation shape dependencies of {:?} include unmapped node {dep:?}",
                        source_node.id
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if mapped_node.shape_deps != expected_deps {
            return Err(format!(
                "activation shape dependencies of {:?} were not preserved by vectorization",
                source_node.id
            ));
        }
        let expected_result_claims = source_node
            .result_claim_deps
            .iter()
            .map(|dep| {
                node_map.get(dep.0).copied().ok_or_else(|| {
                    format!(
                        "activation result claims of {:?} include unmapped node {dep:?}",
                        source_node.id
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if mapped_node.result_claim_deps != expected_result_claims {
            return Err(format!(
                "activation result claims of {:?} were not preserved by vectorization",
                source_node.id
            ));
        }
        if let RiscOp::ExtentWitness {
            site,
            parameter,
            axis: RtAxis::Lit(axis),
            requirements,
            claims,
        } = &source_node.op
        {
            let shifted_site = match site {
                ExtentWitnessSite::ResultClaim {
                    claim,
                    axis: RtAxis::Lit(result_axis),
                } => ExtentWitnessSite::ResultClaim {
                    claim: claim.clone(),
                    axis: RtAxis::Lit(
                        result_axis
                            .checked_add(1)
                            .ok_or_else(|| "entry witness result axis overflow".to_string())?,
                    ),
                },
                ExtentWitnessSite::LocalAscriptionClaim {
                    ascription_id,
                    binding,
                    claim,
                    axis: RtAxis::Lit(result_axis),
                } => ExtentWitnessSite::LocalAscriptionClaim {
                    ascription_id: *ascription_id,
                    binding: binding.clone(),
                    claim: claim.clone(),
                    axis: RtAxis::Lit(
                        result_axis
                            .checked_add(1)
                            .ok_or_else(|| "local ascription result axis overflow".to_string())?,
                    ),
                },
                other => other.clone(),
            };
            let expected = RiscOp::ExtentWitness {
                site: shifted_site,
                parameter: parameter.clone(),
                axis: RtAxis::Lit(
                    axis.checked_add(1)
                        .ok_or_else(|| "entry witness axis overflow".to_string())?,
                ),
                requirements: requirements.clone(),
                claims: claims.clone(),
            };
            if mapped_node.op != expected {
                return Err(format!(
                    "entry witness {:?} lost its site, parameter, shifted axes, requirements, or claims",
                    source_node.id
                ));
            }
            let expected_inputs = source_node
                .inputs
                .iter()
                .map(|input| node_map[input.0])
                .collect::<Vec<_>>();
            if mapped_node.inputs != expected_inputs {
                return Err(format!(
                    "entry witness {:?} lost its observing input correspondence",
                    source_node.id
                ));
            }
        }
    }

    if root_map.len() != source.roots().len() || mapped.roots().len() != source.roots().len() {
        return Err("mapped gradient root correspondence is incomplete".into());
    }
    for (&source_root, &mapped_root) in source.roots().iter().zip(mapped.roots()) {
        if root_map.get(&source_root) != Some(&mapped_root) {
            return Err(format!(
                "mapped gradient root {source_root:?} has no exact vectorized identity"
            ));
        }
    }

    for mapped_node in mapped.nodes() {
        let Some(spliced_id) = splice_map.get(&mapped_node.id).copied() else {
            return Err(format!(
                "mapped node {:?} has no splice correspondence",
                mapped_node.id
            ));
        };
        let Some(spliced_node) = spliced.get(spliced_id) else {
            return Err(format!(
                "mapped node {:?} splices to invalid node {spliced_id:?}",
                mapped_node.id
            ));
        };
        if !matches!(mapped_node.op, RiscOp::Load { .. }) {
            let expected_inputs = mapped_node
                .inputs
                .iter()
                .map(|input| {
                    splice_map.get(input).copied().ok_or_else(|| {
                        format!(
                            "value input {input:?} of mapped node {:?} has no splice correspondence",
                            mapped_node.id
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if spliced_node.inputs != expected_inputs {
                return Err(format!(
                    "mapped node {:?} lost its value-input splice correspondence",
                    mapped_node.id
                ));
            }
            let expected_deps = mapped_node
                .shape_deps
                .iter()
                .map(|dep| {
                    splice_map.get(dep).copied().ok_or_else(|| {
                        format!(
                            "shape dependency {dep:?} of mapped node {:?} has no splice correspondence",
                            mapped_node.id
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut expected_deps = expected_deps;
            if cotangents.contains(&spliced_id)
                && spliced_id != forward_activation
                && !expected_deps.contains(&forward_activation)
            {
                expected_deps.push(forward_activation);
            }
            if spliced_node.shape_deps != expected_deps {
                return Err(format!(
                    "spliced shape dependencies of mapped node {:?} are incomplete: expected {:?}, got {:?} at {:?}",
                    mapped_node.id, expected_deps, spliced_node.shape_deps, spliced_id
                ));
            }
            let expected_result_claims = mapped_node
                .result_claim_deps
                .iter()
                .map(|dep| {
                    splice_map.get(dep).copied().ok_or_else(|| {
                        format!(
                            "result claim dependency {dep:?} of mapped node {:?} has no splice correspondence",
                            mapped_node.id
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if spliced_node.result_claim_deps != expected_result_claims {
                return Err(format!(
                    "spliced result claims of mapped node {:?} are incomplete",
                    mapped_node.id
                ));
            }
            if spliced_node.op != mapped_node.op
                || spliced_node.output_type != mapped_node.output_type
            {
                return Err(format!(
                    "mapped node {:?} changed operator or type during splice",
                    mapped_node.id
                ));
            }
        }
    }

    let expected_forward = node_map
        .get(forward_source.0)
        .and_then(|mapped_id| splice_map.get(mapped_id))
        .copied()
        .ok_or_else(|| "forward activation has no complete vectorize/splice mapping".to_string())?;
    if expected_forward != forward_activation || spliced.get(forward_activation).is_none() {
        return Err("forward activation identity disagrees with its correspondence maps".into());
    }
    if cotangents.len() != expected_cotangents {
        return Err(format!(
            "cotangent packing produced {} values for {expected_cotangents} selected parameters",
            cotangents.len()
        ));
    }
    for cotangent in cotangents {
        let Some(node) = spliced.get(*cotangent) else {
            return Err(format!(
                "cotangent {cotangent:?} is not present after splice"
            ));
        };
        if *cotangent != forward_activation && !node.shape_deps.contains(&forward_activation) {
            return Err(format!(
                "cotangent {cotangent:?} does not retain forward activation {forward_activation:?}"
            ));
        }
    }

    // The vectorized callee is an intermediate graph: its mapped batch extent
    // may be supplied only by the caller actual introduced by splice. Validate
    // rendered identifiers at the post-splice artifact boundary, where that
    // correspondence must be complete.
    crate::axis_sources::check_rendered_dim_origins(
        spliced,
        chelis_types::unsupported::Stage::Lowering,
    )
    .map_err(|error| format!("spliced rendered dimension origin is unresolved: {error}"))?;
    Ok(())
}

#[allow(clippy::collapsible_match)]
fn verify_with_dangling_policy(dag: &Dag, reject_dangling: bool) -> Vec<String> {
    let mut errors = Vec::new();
    let mut consumers = vec![0usize; dag.len()];
    let mut load_types = chelis_unord::UnordMap::<String, crate::dag::TensorType>::new();
    for node in dag.nodes() {
        for &input_id in &node.inputs {
            if input_id.0 < consumers.len() {
                consumers[input_id.0] += 1;
            }
        }
        // chelis#384/#397/#616: a shape-only dependency (an `expand` shape
        // source or a runtime-dim declarer kept alive for its extent) is a
        // real consumption — the dependent reads the node's shape, not its
        // value — so its target is not dangling.
        for &dep in &node.shape_deps {
            if dep.0 < consumers.len() {
                consumers[dep.0] += 1;
            }
        }
        for &dep in &node.result_claim_deps {
            if dep.0 < consumers.len() {
                consumers[dep.0] += 1;
            }
        }

        if let RiscOp::Load { name } = &node.op {
            if let Some(prev_ty) = load_types.get(name.as_str()) {
                if prev_ty != &node.output_type {
                    errors.push(format!(
                        "load '{}' has inconsistent tensor types: {:?} vs {:?}",
                        name, prev_ty, node.output_type
                    ));
                }
            } else {
                load_types.insert(name.as_str().to_string(), node.output_type.clone());
            }
        }
    }

    for node in dag.nodes() {
        // Check that inputs reference valid, earlier nodes.
        for &input_id in &node.inputs {
            if input_id.0 >= node.id.0 {
                errors.push(format!(
                    "node {} references non-earlier node {}",
                    node.id.0, input_id.0
                ));
            }
            if dag.get(input_id).is_none() {
                errors.push(format!(
                    "node {} references nonexistent node {}",
                    node.id.0, input_id.0
                ));
            }
        }
        for &dependency in node.shape_deps.iter().chain(&node.result_claim_deps) {
            if dependency.0 >= node.id.0 {
                errors.push(format!(
                    "node {} references non-earlier dependency {}",
                    node.id.0, dependency.0
                ));
            }
            if dag.get(dependency).is_none() {
                errors.push(format!(
                    "node {} references nonexistent dependency {}",
                    node.id.0, dependency.0
                ));
            }
        }

        // Check arity.
        let arity = node.inputs.len();
        match &node.op {
            RiscOp::Compare(kind) => {
                if arity != 2 {
                    errors.push(format!(
                        "comparison at node {} has {} inputs (expected 2)",
                        node.id.0, arity
                    ));
                } else if let (Some(lhs), Some(rhs)) =
                    (dag.get(node.inputs[0]), dag.get(node.inputs[1]))
                {
                    let shape_participants = [node.inputs[0], node.inputs[1], node.id];
                    if lhs.output_type.precision != rhs.output_type.precision {
                        errors.push(format!(
                            "comparison at node {} has mismatched precision {:?} vs {:?}",
                            node.id.0, lhs.output_type.precision, rhs.output_type.precision
                        ));
                    }
                    if !node_shapes_semantically_equivalent(
                        dag,
                        node.inputs[0],
                        node.inputs[1],
                        &shape_participants,
                    ) {
                        errors.push(format!(
                            "comparison at node {} requires exactly matching operand shape",
                            node.id.0
                        ));
                    }
                    let active_numeric = lhs.output_type.precision.is_numeric()
                        && lhs.output_type.precision.is_admissible_active();
                    if !active_numeric && lhs.output_type.precision != Prim::Bool {
                        errors.push(format!(
                            "comparison {} at node {} requires active numeric or bool operands",
                            kind.surf_name(),
                            node.id.0
                        ));
                    } else if matches!(
                        kind,
                        ComparisonKind::CmpLt
                            | ComparisonKind::Lt
                            | ComparisonKind::Gt
                            | ComparisonKind::Gte
                            | ComparisonKind::Lte
                    ) && !active_numeric
                    {
                        errors.push(format!(
                            "ordered comparison {} at node {} requires active numeric operands",
                            kind.surf_name(),
                            node.id.0
                        ));
                    }
                    if node.output_type.precision != Prim::Bool {
                        errors.push(format!(
                            "comparison {} at node {} has output precision {:?}, expected Bool",
                            kind.surf_name(),
                            node.id.0,
                            node.output_type.precision
                        ));
                    }
                    if !anonymous_output_has_input_authority(dag, node)
                        || !node_shapes_semantically_equivalent(
                            dag,
                            node.id,
                            node.inputs[0],
                            &shape_participants,
                        )
                    {
                        errors.push(format!(
                            "comparison at node {} output shape must match its operands",
                            node.id.0
                        ));
                    }
                }
            }
            RiscOp::Logical(kind) => {
                let expected = kind.arity();
                if arity != expected {
                    errors.push(format!(
                        "logical {} at node {} has {} inputs (expected {})",
                        kind.surf_name(),
                        node.id.0,
                        arity,
                        expected
                    ));
                } else {
                    let shape_participants = node
                        .inputs
                        .iter()
                        .copied()
                        .chain(std::iter::once(node.id))
                        .collect::<Vec<_>>();
                    let inputs = node
                        .inputs
                        .iter()
                        .filter_map(|input| dag.get(*input))
                        .collect::<Vec<_>>();
                    for input in &inputs {
                        if input.output_type.precision != Prim::Bool {
                            errors.push(format!(
                                "logical {} at node {} requires bool operands",
                                kind.surf_name(),
                                node.id.0
                            ));
                        }
                        if !anonymous_output_has_input_authority(dag, node)
                            || !node_shapes_semantically_equivalent(
                                dag,
                                input.id,
                                node.id,
                                &shape_participants,
                            )
                        {
                            errors.push(format!(
                                "logical {} at node {} requires exactly matching shape",
                                kind.surf_name(),
                                node.id.0
                            ));
                        }
                    }
                    if node.output_type.precision != Prim::Bool {
                        errors.push(format!(
                            "logical {} at node {} requires Bool output",
                            kind.surf_name(),
                            node.id.0
                        ));
                    }
                }
            }
            RiscOp::Where => {
                if arity != 3 {
                    errors.push(format!(
                        "where at node {} has {} inputs (expected 3)",
                        node.id.0, arity
                    ));
                } else if let (Some(condition), Some(then_value), Some(_)) = (
                    dag.get(node.inputs[0]),
                    dag.get(node.inputs[1]),
                    dag.get(node.inputs[2]),
                ) {
                    let shape_participants =
                        [node.inputs[0], node.inputs[1], node.inputs[2], node.id];
                    if condition.output_type.precision != Prim::Bool {
                        errors.push(format!(
                            "where at node {} condition must be Bool",
                            node.id.0
                        ));
                    }
                    if !node_types_semantically_equivalent(
                        dag,
                        node.inputs[1],
                        node.inputs[2],
                        &shape_participants,
                    ) {
                        errors.push(format!(
                            "where at node {} branches must have exactly matching type",
                            node.id.0
                        ));
                    }
                    if !node_types_semantically_equivalent(
                        dag,
                        node.id,
                        node.inputs[1],
                        &shape_participants,
                    ) {
                        errors.push(format!(
                            "where at node {} output must match its branches",
                            node.id.0
                        ));
                    }
                    if !node_shapes_semantically_equivalent(
                        dag,
                        node.inputs[0],
                        node.inputs[1],
                        &shape_participants,
                    ) {
                        errors.push(format!(
                            "where at node {} condition and branches must have exactly matching shape",
                            node.id.0
                        ));
                    }
                    if !then_value.output_type.precision.is_valid_tensor_precision() {
                        errors.push(format!(
                            "where at node {} branches must use an active tensor element dtype",
                            node.id.0
                        ));
                    }
                }
            }
            RiscOp::Add
            | RiscOp::Sub
            | RiscOp::Mul
            | RiscOp::Div
            | RiscOp::FloorDiv
            | RiscOp::TruncDiv
            | RiscOp::Mod
            | RiscOp::MaxElem
            | RiscOp::MinElem => {
                if arity != 2 {
                    errors.push(format!(
                        "binary op at node {} has {} inputs (expected 2)",
                        node.id.0, arity
                    ));
                }

                if matches!(node.op, RiscOp::Mod) && !node.output_type.precision.is_integer() {
                    errors.push(format!(
                        "mod at node {} requires an integer dtype",
                        node.id.0
                    ));
                }

                // C1: precision consistency check for binary ops.
                if arity == 2
                    && let (Some(lhs), Some(rhs)) =
                        (dag.get(node.inputs[0]), dag.get(node.inputs[1]))
                {
                    if lhs.output_type.precision != rhs.output_type.precision {
                        errors.push(format!(
                            "binary op at node {} has mismatched precisions: {:?} vs {:?}",
                            node.id.0, lhs.output_type.precision, rhs.output_type.precision
                        ));
                    }

                    if matches!(node.op, RiscOp::Mod)
                        && (node.output_type.precision != lhs.output_type.precision
                            || node.output_type.dims.len() != lhs.output_type.dims.len()
                            || node
                                .output_type
                                .dims
                                .iter()
                                .zip(&lhs.output_type.dims)
                                .any(|(out, input)| !dims_compatible(out, input)))
                    {
                        errors.push(format!(
                            "mod at node {} output must match its input shape and dtype",
                            node.id.0
                        ));
                    }

                    // C2: dimension matching for binary ops.
                    let l_dims = &lhs.output_type.dims;
                    let r_dims = &rhs.output_type.dims;
                    if l_dims.len() != r_dims.len() {
                        errors.push(format!(
                            "binary op at node {} has mismatched dimension count: {} vs {}",
                            node.id.0,
                            l_dims.len(),
                            r_dims.len()
                        ));
                    } else {
                        for (i, (ld, rd)) in l_dims.iter().zip(r_dims.iter()).enumerate() {
                            if !dims_compatible(ld, rd) {
                                errors.push(format!(
                                    "binary op at node {} has mismatched dimension at axis {}: {:?} vs {:?}",
                                    node.id.0, i, ld, rd
                                ));
                            }
                        }
                    }
                }
            }
            RiscOp::ReduceWindowGrad { .. } => {
                // Adjoint of ReduceWindow: inputs are `[x, g]` with
                // *different* shapes (forward input vs forward output), so
                // this is not an elementwise binary op. Check arity only;
                // the output shape equals `x`'s shape and is set at
                // construction by `chelis_ir::grad`.
                if arity != 2 {
                    errors.push(format!(
                        "reduce_window_grad at node {} has {} inputs (expected 2)",
                        node.id.0, arity
                    ));
                }
            }
            RiscOp::ExtremaAdjoint { .. } => {
                if arity != 3 {
                    errors.push(format!(
                        "extrema adjoint at node {} has {} inputs (expected 3)",
                        node.id.0, arity
                    ));
                } else {
                    let inputs = node
                        .inputs
                        .iter()
                        .filter_map(|input| dag.get(*input))
                        .collect::<Vec<_>>();
                    if inputs.len() == 3 {
                        for input in &inputs {
                            if input.output_type != node.output_type {
                                errors.push(format!(
                                    "extrema adjoint at node {} has input type {:?}, expected {:?}",
                                    node.id.0, input.output_type, node.output_type
                                ));
                            }
                        }
                        if !node.output_type.precision.is_float() {
                            errors.push(format!(
                                "extrema adjoint at node {} requires a float dtype, found {:?}",
                                node.id.0, node.output_type.precision
                            ));
                        }
                    }
                }
            }
            RiscOp::BlasMatmul {
                batch_dims,
                m,
                n,
                k,
                ..
            } => {
                if arity != 2 {
                    errors.push(format!(
                        "blas matmul at node {} has {} inputs (expected 2)",
                        node.id.0, arity
                    ));
                }
                if arity == 2
                    && let (Some(lhs), Some(rhs)) =
                        (dag.get(node.inputs[0]), dag.get(node.inputs[1]))
                {
                    if lhs.output_type.precision != rhs.output_type.precision {
                        errors.push(format!(
                            "blas matmul at node {} has mismatched precisions: {:?} vs {:?}",
                            node.id.0, lhs.output_type.precision, rhs.output_type.precision
                        ));
                    }
                    if lhs.output_type.dims.len() < 2 || rhs.output_type.dims.len() < 2 {
                        errors.push(format!(
                            "blas matmul at node {} expects rank >= 2 inputs, got rank {} and {}",
                            node.id.0,
                            lhs.output_type.dims.len(),
                            rhs.output_type.dims.len()
                        ));
                    }
                    if node.output_type.dims.len() < 2 {
                        errors.push(format!(
                            "blas matmul at node {} expects rank >= 2 output, got rank {}",
                            node.id.0,
                            node.output_type.dims.len()
                        ));
                    }
                    if m.as_concrete() == Some(0)
                        || n.as_concrete() == Some(0)
                        || k.as_concrete() == Some(0)
                    {
                        errors.push(format!(
                            "blas matmul at node {} has zero dimension m={m} n={n} k={k}",
                            node.id.0
                        ));
                    }
                    if node.output_type.dims.len() >= 2 {
                        let out_batch_len = node.output_type.dims.len() - 2;
                        if batch_dims.len() != out_batch_len {
                            errors.push(format!(
                                "blas matmul at node {} has {} batch dims for rank {} output",
                                node.id.0,
                                batch_dims.len(),
                                node.output_type.dims.len()
                            ));
                        }
                        let expected_out_dims = batch_dims
                            .iter()
                            .cloned()
                            .chain([m.clone(), n.clone()])
                            .collect::<Vec<_>>();
                        let actual_out_dims = node
                            .output_type
                            .dims
                            .iter()
                            .map(crate::dag::DimExpr::from)
                            .collect::<Vec<_>>();
                        if expected_out_dims != actual_out_dims {
                            errors.push(format!(
                                "blas matmul at node {} has output dims {:?}, expected {:?}",
                                node.id.0, actual_out_dims, expected_out_dims
                            ));
                        }
                    }
                    if lhs.output_type.dims.len() >= 2 && rhs.output_type.dims.len() >= 2 {
                        let lhs_dims = lhs
                            .output_type
                            .dims
                            .iter()
                            .map(crate::dag::DimExpr::from)
                            .collect::<Vec<_>>();
                        let rhs_dims = rhs
                            .output_type
                            .dims
                            .iter()
                            .map(crate::dag::DimExpr::from)
                            .collect::<Vec<_>>();
                        let lhs_matrix = &lhs_dims[lhs_dims.len() - 2..];
                        let rhs_matrix = &rhs_dims[rhs_dims.len() - 2..];
                        if lhs_matrix != [m.clone(), k.clone()]
                            || rhs_matrix != [k.clone(), n.clone()]
                        {
                            errors.push(format!(
                                "blas matmul at node {} has incompatible matrix dims",
                                node.id.0
                            ));
                        }
                    }
                }
            }
            RiscOp::Gather { axis } => {
                if arity != 2 {
                    errors.push(format!(
                        "gather at node {} has {} inputs (expected 2)",
                        node.id.0, arity
                    ));
                }
                if arity == 2
                    && let (Some(values), Some(indices)) =
                        (dag.get(node.inputs[0]), dag.get(node.inputs[1]))
                {
                    if !matches!(indices.output_type.precision, Prim::Int32 | Prim::Int64) {
                        errors.push(format!(
                            "gather at node {} requires i32/i64 indices, got {:?}",
                            node.id.0, indices.output_type.precision
                        ));
                    }
                    if node.output_type.precision != values.output_type.precision {
                        errors.push(format!(
                            "gather at node {} output precision {:?} must match values precision {:?}",
                            node.id.0,
                            node.output_type.precision,
                            values.output_type.precision
                        ));
                    }
                    if *axis >= values.output_type.dims.len() {
                        errors.push(format!(
                            "gather at node {} has axis {} out of bounds for rank {}",
                            node.id.0,
                            axis,
                            values.output_type.dims.len()
                        ));
                    } else {
                        let mut expected = Vec::new();
                        expected.extend_from_slice(&values.output_type.dims[..*axis]);
                        expected.extend(indices.output_type.dims.iter().cloned());
                        expected.extend_from_slice(&values.output_type.dims[*axis + 1..]);
                        if node.output_type.dims != expected {
                            errors.push(format!(
                                "gather at node {} has output dims {:?}, expected {:?}",
                                node.id.0, node.output_type.dims, expected
                            ));
                        }
                    }
                }
            }
            RiscOp::ScatterAdd { axis } => {
                verify_scatter_like(node, dag, *axis, "scatter_add", &mut errors);
            }
            RiscOp::Scatter { axis } => {
                // Replace-scatter (last-write-wins) shares the input
                // arity / index-precision / shape contract with
                // scatter_add — only the duplicate-index semantics
                // differ (replace vs accumulate), which is a runtime
                // concern not a structural one.
                verify_scatter_like(node, dag, *axis, "scatter_replace", &mut errors);
            }
            RiscOp::ScatterElements { axis } => {
                // Element-wise scatter (ONNX `ScatterElements`,
                // spec §3.5.1) has a distinct shape contract
                // (indices.dims == updates.dims; output.dims ==
                // data.dims; shared rank), so it verifies separately.
                verify_scatter_elements(node, dag, *axis, &mut errors);
            }
            RiscOp::Count { axes } => {
                if arity != 1 {
                    errors.push(format!(
                        "count at node {} has {} inputs (expected 1)",
                        node.id.0, arity
                    ));
                }
                if axes.is_empty() {
                    errors.push(format!(
                        "count at node {} requires a non-empty axis list",
                        node.id.0
                    ));
                }
                if axes.windows(2).any(|pair| pair[0] <= pair[1]) {
                    errors.push(format!(
                        "count at node {} axes must be unique and strictly descending",
                        node.id.0
                    ));
                }
                if arity == 1
                    && let Some(input) = dag.get(node.inputs[0])
                {
                    let rank = input.output_type.dims.len();
                    if axes.iter().any(|&axis| axis >= rank) {
                        errors.push(format!(
                            "count at node {} has an axis out of range for rank {}",
                            node.id.0, rank
                        ));
                    }
                    if input.output_type.precision != Prim::Bool {
                        errors.push(format!(
                            "count at node {} requires bool input, got {:?}",
                            node.id.0, input.output_type.precision
                        ));
                    }
                    if node.output_type.precision != Prim::Int64 {
                        errors.push(format!(
                            "count at node {} requires i64 output, got {:?}",
                            node.id.0, node.output_type.precision
                        ));
                    }
                    if axes.iter().all(|&axis| axis < rank) {
                        let expected: Vec<_> = input
                            .output_type
                            .dims
                            .iter()
                            .enumerate()
                            .filter_map(|(axis, dim)| {
                                (!axes.contains(&axis)).then_some(dim.clone())
                            })
                            .collect();
                        if node.output_type.dims != expected {
                            errors.push(format!(
                                "count at node {} has output shape {:?}, expected {:?}",
                                node.id.0, node.output_type.dims, expected
                            ));
                        }
                    }
                }
            }
            RiscOp::Neg
            | RiscOp::Relu
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
            | RiscOp::Dropout { .. }
            | RiscOp::Copy
            | RiscOp::Drop
            | RiscOp::Realize
            | RiscOp::Sum { .. }
            | RiscOp::MaxReduce { .. }
            | RiscOp::MinReduce { .. }
            | RiscOp::ProdReduce { .. }
            | RiscOp::ReduceWindow { .. }
            | RiscOp::Argmax { .. }
            | RiscOp::Argmin { .. }
            | RiscOp::Permute { .. }
            | RiscOp::OneHot { .. }
            | RiscOp::Shape { .. }
            | RiscOp::Cast { .. }
            | RiscOp::CastTrunc { .. } => {
                if arity != 1 {
                    errors.push(format!(
                        "unary op at node {} has {} inputs (expected 1)",
                        node.id.0, arity
                    ));
                }
            }
            RiscOp::ExtentWitness {
                site: crate::dag::ExtentWitnessSite::LiteralResultClaim,
                ..
            } => {
                if arity != 0 {
                    errors.push(format!(
                        "literal result claim at node {} must have no inputs",
                        node.id.0
                    ));
                }
            }
            RiscOp::ExtentWitness {
                site: crate::dag::ExtentWitnessSite::LocalAscriptionClaim { .. },
                requirements,
                ..
            } => {
                let expected = usize::from(requirements.is_empty());
                if arity != expected {
                    errors.push(format!(
                        "local ascription claim at node {} must carry either one literal or one observed tensor",
                        node.id.0
                    ));
                }
            }
            RiscOp::ExtentWitness { claims, .. } => {
                if arity != claims.len() + 1 {
                    errors.push(format!(
                        "extent witness at node {} requires its observed tensor and one witness input per named claim",
                        node.id.0
                    ));
                }
            }
            RiscOp::CheckedReshapeExtent { claims, .. } => {
                if claims.is_empty() || arity != claims.len() + 1 {
                    errors.push(format!(
                        "checked reshape extent at node {} requires an actual and one scalar input per nonempty claim",
                        node.id.0
                    ));
                }
            }
            RiscOp::CheckedUnitAxis { .. } | RiscOp::ReluAdjoint => {
                if arity != 2 {
                    errors.push(format!(
                        "binary checked op at node {} has {} inputs (expected 2)",
                        node.id.0, arity
                    ));
                }
            }
            RiscOp::UniformLike { .. } => {
                if !matches!(arity, 1 | 2) {
                    errors.push(format!(
                        "op {:?} at node {} expects 1 or 2 inputs, got {}",
                        node.op, node.id.0, arity
                    ));
                }
                if arity == 2
                    && let Some(activation) = dag.get(node.inputs[1])
                    && (activation.output_type.precision != Prim::Bool
                        || !activation.output_type.dims.is_empty())
                {
                    errors.push(format!(
                        "uniform_like at node {} requires a scalar Bool path activation, got {:?}",
                        node.id.0, activation.output_type
                    ));
                }
            }
            // chelis#616: movement ops (and `Reshape`, whose runtime target
            // extents work the same way) carry a tensor at `inputs[0]` plus zero
            // or more rank-0 integer bound scalars at `inputs[1..]` (node-valued
            // runtime bounds). Their arity + bound-source validity is checked in
            // the dedicated arms below.
            RiscOp::Pad { .. }
            | RiscOp::Shrink { .. }
            | RiscOp::Stride { .. }
            | RiscOp::Reshape { .. }
            | RiscOp::Expand { .. } => {
                if arity < 1 {
                    errors.push(format!(
                        "movement op at node {} has {} inputs (expected at least 1)",
                        node.id.0, arity
                    ));
                }
            }
            RiscOp::Store { .. } => {
                if arity != 1 {
                    errors.push(format!(
                        "store op at node {} has {} inputs (expected 1)",
                        node.id.0, arity
                    ));
                }
            }
            RiscOp::Const { .. } | RiscOp::ConstTensor { .. } | RiscOp::Load { .. } => {
                if arity != 0 {
                    errors.push(format!(
                        "memory op at node {} has {} inputs (expected 0)",
                        node.id.0, arity
                    ));
                }
                if let RiscOp::ConstTensor { data } = &node.op {
                    let mut expected = Some(1usize);
                    for dim in &node.output_type.dims {
                        let extent = match dim {
                            DimInfo::Lit(extent) | DimInfo::Named(_, Some(extent)) => *extent,
                            DimInfo::Named(_, None) => {
                                expected = None;
                                break;
                            }
                        };
                        expected = expected.and_then(|count| count.checked_mul(extent));
                        if expected.is_none() {
                            errors.push(format!(
                                "constant tensor at node {} has a concrete shape whose cardinality overflows usize",
                                node.id.0
                            ));
                            break;
                        }
                    }
                    if let Some(expected) = expected
                        && data.len() != expected
                    {
                        errors.push(format!(
                            "constant tensor at node {} stores {} values but its concrete shape requires {}",
                            node.id.0,
                            data.len(),
                            expected
                        ));
                    }
                }
            }
            RiscOp::FusedElem { ops } => {
                if ops.is_empty() {
                    errors.push(format!("fused elem at node {} has no steps", node.id.0));
                }
            }
        }

        if matches!(node.op, RiscOp::Relu | RiscOp::ReluAdjoint) {
            if !node.output_type.precision.is_float() {
                errors.push(format!(
                    "relu op at node {} requires a float output, got {:?}",
                    node.id.0, node.output_type.precision
                ));
            }
            for input in &node.inputs {
                if let Some(input) = dag.get(*input)
                    && (input.output_type != node.output_type
                        || !input.output_type.precision.is_float())
                {
                    errors.push(format!(
                        "relu op at node {} requires same-shape, same-dtype float inputs",
                        node.id.0
                    ));
                }
            }
        }

        // C3: validate reduction axis bounds.
        match &node.op {
            RiscOp::Sum { axis, .. }
            | RiscOp::MaxReduce { axis }
            | RiscOp::MinReduce { axis }
            | RiscOp::ProdReduce { axis }
            | RiscOp::Argmax { axis }
            | RiscOp::Argmin { axis } => {
                if arity == 1
                    && let Some(input) = dag.get(node.inputs[0])
                {
                    let ndims = input.output_type.dims.len();
                    if ndims == 0 || *axis >= ndims {
                        errors.push(format!(
                            "reduction op at node {} has axis {} but input has {} dimensions",
                            node.id.0, axis, ndims
                        ));
                    }
                }
            }
            _ => {}
        }

        // Shape query (chelis#513/#558): the read axis must be in range
        // of the input rank, and the output must be a rank-0 exact i64
        // scalar (the runtime extent). A different output type would
        // misdeclare the value node to the backend or narrow its carrier.
        if let RiscOp::Shape { axis } = &node.op {
            if arity == 1
                && let Some(input) = dag.get(node.inputs[0])
            {
                let ndims = input.output_type.dims.len();
                if *axis >= ndims {
                    errors.push(format!(
                        "shape read at node {} has axis {} but input has {} dimensions",
                        node.id.0, axis, ndims
                    ));
                }
            }
            if !node.output_type.dims.is_empty() {
                errors.push(format!(
                    "shape read at node {} must produce a rank-0 scalar, got rank {}",
                    node.id.0,
                    node.output_type.dims.len()
                ));
            }
            if node.output_type.precision != Prim::Int64 {
                errors.push(format!(
                    "shape read at node {} must produce an exact i64 scalar, got precision `{}`",
                    node.id.0,
                    node.output_type.precision.name()
                ));
            }
        }

        if let RiscOp::ExtentWitness {
            site,
            axis: crate::dag::RtAxis::Lit(axis),
            requirements,
            claims,
            ..
        } = &node.op
        {
            if let crate::dag::ExtentWitnessSite::LocalAscriptionClaim {
                binding,
                claim,
                axis: crate::dag::RtAxis::Lit(claimed_axis),
                ..
            } = site
            {
                let RiscOp::ExtentWitness { parameter, .. } = &node.op else {
                    unreachable!()
                };
                let owners = dag
                    .nodes()
                    .iter()
                    .filter(|owner| owner.shape_deps.contains(&node.id))
                    .collect::<Vec<_>>();
                if let [owner] = owners.as_slice()
                    && let Err(reason) = crate::axis_sources::local_ascription_guard_activation(
                        dag, owner.id, node.id,
                    )
                {
                    errors.push(reason);
                }
                let literal = node.inputs.is_empty()
                    && node.shape_deps.is_empty()
                    && matches!(&node.op, RiscOp::ExtentWitness { parameter, .. } if parameter.is_empty())
                    && requirements.len() == 1;
                let named = requirements.is_empty()
                    && node.inputs.len() == 1
                    && node.shape_deps.len() == 1
                    && !parameter.is_empty()
                    && dag.get(node.shape_deps[0]).is_some_and(|declared| {
                        matches!(declared.op, RiscOp::ExtentWitness {
                            site: crate::dag::ExtentWitnessSite::Caller,
                            axis: crate::dag::RtAxis::Lit(observed),
                            ..
                        } if observed == *axis)
                            && declared.inputs.first() == node.inputs.first()
                            && declared.id.0 < node.id.0
                    });
                if binding.is_empty()
                    || claim.is_empty()
                    || *claimed_axis < 0
                    // A literal claim observes its owner's output axis
                    // directly. A named claim may instead source that extent
                    // from a different tensor axis (for example, insert
                    // output axis 1 from `shape(x, 2)`); each coordinate is
                    // validated against its own tensor below.
                    || (literal && claimed_axis != axis)
                    || !claims.is_empty()
                    || owners.len() != 1
                    || (!literal && !named)
                    || requirements.iter().any(|value| {
                        value.prim() != Prim::Int64
                            || value.as_i64_exact().is_none_or(|value| value < 0)
                    })
                    || !node.output_type.dims.is_empty()
                    || node.output_type.precision != Prim::Int64
                {
                    errors.push(format!(
                        "local ascription claim at node {} requires exact nonempty provenance, one literal or declaring witness, one initializer owner, a normalized claimed axis, and scalar int64 output",
                        node.id.0
                    ));
                }
                continue;
            }
            if matches!(site, crate::dag::ExtentWitnessSite::LiteralResultClaim) {
                let owners = dag
                    .nodes()
                    .iter()
                    .filter(|owner| {
                        owner.shape_deps.contains(&node.id)
                            || owner.result_claim_deps.contains(&node.id)
                    })
                    .count();
                if owners != 1 {
                    errors.push(format!(
                        "literal result claim at node {} requires exactly one producing owner",
                        node.id.0
                    ));
                }
                let RiscOp::ExtentWitness { parameter, .. } = &node.op else {
                    unreachable!()
                };
                if *axis < 0
                    || !parameter.is_empty()
                    || !claims.is_empty()
                    || !node.inputs.is_empty()
                    || !node.shape_deps.is_empty()
                    || !node.result_claim_deps.is_empty()
                    || requirements.len() != 1
                    || requirements.first().is_none_or(|value| {
                        value.prim() != Prim::Int64
                            || value.as_i64_exact().is_none_or(|value| value < 0)
                    })
                    || !node.output_type.dims.is_empty()
                    || node.output_type.precision != Prim::Int64
                {
                    errors.push(format!("literal result claim at node {} requires one nonnegative i64 literal, normalized axis, scalar output, and no observing or entry dependencies", node.id.0));
                }
            } else {
                if let crate::dag::ExtentWitnessSite::ResultClaim {
                    claim,
                    axis: crate::dag::RtAxis::Lit(result_axis),
                } = site
                    && (claim.is_empty()
                        || *result_axis < 0
                        || !requirements.is_empty()
                        || !claims.is_empty())
                {
                    errors.push(format!("result claim witness at node {} requires a nonempty label, normalized axis, and no entry obligations", node.id.0));
                }
                if matches!(site, crate::dag::ExtentWitnessSite::ResultClaim { .. }) {
                    let declared = node
                        .shape_deps
                        .as_slice()
                        .first()
                        .and_then(|required| dag.get(*required));
                    let same_observation = node.shape_deps.len() == 1 && declared.is_some_and(|declared| {
                    matches!(declared.op, RiscOp::ExtentWitness { site: crate::dag::ExtentWitnessSite::Caller, axis: crate::dag::RtAxis::Lit(observed), .. } if observed == *axis)
                        && declared.inputs.first() == node.inputs.first()
                        && declared.id.0 < node.id.0
                });
                    if !same_observation {
                        errors.push(format!("result claim witness at node {} requires its exact earlier declaring observation", node.id.0));
                    }
                }
                // chelis#1374: a named claim's diagnostic reads the DECLARING
                // parameter and axis off the requirement's own node, so the edge
                // must be a witness and not merely an i64 scalar. The edge is
                // also strictly earlier, which is what keeps the check due "at
                // the later of its two witnesses" (spec/04 §4.7) and the topology
                // acyclic.
                if arity == claims.len() + 1 {
                    for edge in node.inputs.iter().skip(1) {
                        let earlier = edge.0 < node.id.0;
                        let witness = dag.get(*edge).is_some_and(|edge| {
                            matches!(edge.op, RiscOp::ExtentWitness { ref site, .. } if !matches!(site, crate::dag::ExtentWitnessSite::LiteralResultClaim))
                                && edge.output_type.dims.is_empty()
                                && edge.output_type.precision == Prim::Int64
                        });
                        if !earlier || !witness {
                            errors.push(format!(
                            "extent witness at node {} requires each named claim to name an earlier rank-0 i64 extent witness",
                            node.id.0
                        ));
                        }
                    }
                }
                if claims.iter().any(|claim| claim.claim.is_empty()) {
                    errors.push(format!(
                        "extent witness at node {} requires a nonempty binder for each named claim",
                        node.id.0
                    ));
                }
                if arity >= 1
                    && let Some(input) = dag.get(node.inputs[0])
                    && usize::try_from(*axis)
                        .map_or(true, |axis| axis >= input.output_type.dims.len())
                {
                    errors.push(format!(
                        "extent witness at node {} has an invalid input axis",
                        node.id.0
                    ));
                }
                if !node.output_type.dims.is_empty() || node.output_type.precision != Prim::Int64 {
                    errors.push(format!(
                        "extent witness at node {} must produce a rank-0 i64 scalar",
                        node.id.0
                    ));
                }
                if requirements.iter().any(|value| {
                    value.prim() != Prim::Int64
                        || value.as_i64_exact().is_none_or(|value| value < 0)
                }) {
                    errors.push(format!(
                        "extent witness at node {} requires nonnegative i64 literals",
                        node.id.0
                    ));
                }
            }
        }

        for required in node.shape_deps.iter().chain(&node.result_claim_deps) {
            if let Some(crate::dag::DagNode {
                op:
                    RiscOp::ExtentWitness {
                        site:
                            crate::dag::ExtentWitnessSite::LocalAscriptionClaim {
                                axis: crate::dag::RtAxis::Lit(axis),
                                ..
                            },
                        ..
                    },
                ..
            }) = dag.get(*required)
            {
                if required.0 >= node.id.0
                    || usize::try_from(*axis)
                        .map_or(true, |axis| axis >= node.output_type.dims.len())
                {
                    errors.push(format!(
                        "local ascription claim at node {} requires an earlier token and a valid initializer axis",
                        node.id.0
                    ));
                }
                continue;
            }
            if let Some(crate::dag::DagNode {
                op:
                    RiscOp::ExtentWitness {
                        site: crate::dag::ExtentWitnessSite::LiteralResultClaim,
                        axis: crate::dag::RtAxis::Lit(axis),
                        ..
                    },
                ..
            }) = dag.get(*required)
            {
                if required.0 >= node.id.0
                    || usize::try_from(*axis)
                        .map_or(true, |axis| axis >= node.output_type.dims.len())
                {
                    errors.push(format!("literal result claim at node {} requires an earlier token and a valid producing axis", node.id.0));
                }
                continue;
            }
            let Some(crate::dag::DagNode {
                op:
                    RiscOp::ExtentWitness {
                        site:
                            crate::dag::ExtentWitnessSite::ResultClaim {
                                axis: crate::dag::RtAxis::Lit(axis),
                                ..
                            },
                        ..
                    },
                ..
            }) = dag.get(*required)
            else {
                continue;
            };
            let supported = usize::try_from(*axis).ok().is_some_and(|axis| {
                axis < node.output_type.dims.len()
                    && (crate::axis_sources::expand_or_reshape_carrier(&node.op, axis).is_some()
                        || crate::axis_sources::op_computed_axis_extent(&node.op, axis).is_some()
                        || matches!(
                            crate::axis_sources::same_shape_result_agreement(dag, node.id),
                            Ok(Some(_))
                        ))
            });
            if required.0 >= node.id.0 || !supported {
                errors.push(format!("result claim at node {} requires an earlier witness and a supported producing axis", node.id.0));
            }
        }
        if let Err(reason) = crate::axis_sources::same_shape_result_agreement(dag, node.id) {
            errors.push(reason);
        }
        if !node.result_claim_deps.is_empty() {
            if !crate::axis_sources::is_same_shape_result_op(&node.op) {
                errors.push(format!(
                    "producer result claims at node {} require a same-shape operation",
                    node.id.0
                ));
            }
            for dependency in &node.result_claim_deps {
                if !matches!(
                    dag.get(*dependency).map(|claim| &claim.op),
                    Some(RiscOp::ExtentWitness {
                        site: ExtentWitnessSite::ResultClaim { .. }
                            | ExtentWitnessSite::LiteralResultClaim,
                        ..
                    })
                ) {
                    errors.push(format!(
                        "producer result claim at node {} requires an extent-claim witness",
                        node.id.0
                    ));
                }
            }
        }

        if let RiscOp::CheckedReshapeExtent {
            axis: crate::dag::RtAxis::Lit(axis),
            ..
        } = node.op
        {
            if axis < 0 {
                errors.push(format!(
                    "checked reshape extent at node {} requires a nonnegative result axis",
                    node.id.0
                ));
            }
            let scalar =
                |ty: &crate::dag::TensorType| ty.dims.is_empty() && ty.precision == Prim::Int64;
            if !scalar(&node.output_type)
                || node.inputs.iter().any(|input| {
                    dag.get(*input)
                        .is_none_or(|input| !scalar(&input.output_type))
                })
            {
                errors.push(format!(
                    "checked reshape extent at node {} requires scalar i64 inputs and output",
                    node.id.0
                ));
            }
        }
        if let RiscOp::CheckedUnitAxis {
            axis: crate::dag::RtAxis::Lit(axis),
        } = node.op
        {
            let valid = (|| {
                let [input, witness] = node.inputs.as_slice() else {
                    return None;
                };
                let input_node = dag.get(*input)?;
                let witness = dag.get(*witness)?;
                let RiscOp::ExtentWitness {
                    axis: crate::dag::RtAxis::Lit(observed_axis),
                    requirements,
                    ..
                } = &witness.op
                else {
                    return None;
                };
                if witness.inputs.as_slice() != [*input]
                    || *observed_axis != axis
                    || !requirements
                        .iter()
                        .any(|value| value.prim() == Prim::Int64 && value.as_i64_exact() == Some(1))
                {
                    return None;
                }
                let mut refined = input_node.output_type.clone();
                *refined.dims.get_mut(usize::try_from(axis).ok()?)? = DimInfo::Lit(1);
                (refined == node.output_type).then_some(())
            })()
            .is_some();
            if !valid {
                errors.push(format!("checked unit axis at node {} requires its own tensor-axis witness with requirement one and only that axis refined", node.id.0));
            }
        }

        // C3a (WS-A0): per spec/04-type-system.md §5.7.1 the result
        // precision of `reduce_sum` IS the accumulator precision; the
        // IR invariant is `Sum.output_type.precision == accumulator`.
        // For BlasMatmul, the accumulator must be at least as wide as
        // the operand precision and at least as wide as the spec
        // default for that operand precision.
        match &node.op {
            RiscOp::Sum { accumulator, .. } => {
                // IR invariant: Sum.output_type.precision must equal
                // Sum.accumulator. The §5.7.1 result-precision-table
                // column is the user-facing rule, and lowering inserts
                // a downcast `Cast` node for the bf16/f16 row so the
                // user-visible result returns to operand precision; at
                // the IR level the Sum node itself outputs the
                // accumulator precision.
                if node.output_type.precision != *accumulator {
                    errors.push(format!(
                        "reduce_sum at node {} has output precision `{}` but \
                         accumulator `{}`; per spec/04-type-system.md §5.7.1 \
                         the IR-level result precision of `reduce_sum` is the \
                         accumulator precision (lowering inserts an explicit \
                         downcast for bf16/f16 to recover the operand-precision \
                         result per the §5.7.1 table)",
                        node.id.0,
                        node.output_type.precision.name(),
                        accumulator.name()
                    ));
                }
                // RT-2 fixup B4: spec §5.7.1 narrowness rule for Sum.
                // Symmetric with the BlasMatmul check below. The
                // `sum_with_accumulator` constructor enforces the
                // same rule, but any direct construction of
                // `RiscOp::Sum { .. }` (e.g. by lowering or by a
                // hand-built test) bypasses it; the verify-layer
                // check is the defense-in-depth that prevents a
                // narrow accumulator from reaching the backend.
                if arity == 1
                    && let Some(operand_node) = dag.get(node.inputs[0])
                {
                    let operand = operand_node.output_type.precision;
                    match RiscOp::default_reduce_sum_accumulator(operand) {
                        Ok(default) => {
                            if !crate::dag::accumulator_at_least_as_wide(
                                operand,
                                *accumulator,
                                default,
                            ) {
                                errors.push(format!(
                                    "reduce_sum at node {} has accumulator `{}` narrower than \
                                     the spec/04-type-system.md §5.7.1 default `{}` for \
                                     operand precision `{}`",
                                    node.id.0,
                                    accumulator.name(),
                                    default.name(),
                                    operand.name(),
                                ));
                            }
                        }
                        Err(msg) => {
                            errors.push(format!("reduce_sum at node {}: {msg}", node.id.0));
                        }
                    }
                }
            }
            RiscOp::BlasMatmul { accumulator, .. } => {
                if arity == 2
                    && let Some(lhs) = dag.get(node.inputs[0])
                {
                    let operand = lhs.output_type.precision;
                    match RiscOp::default_matmul_accumulator(operand) {
                        Ok(default) => {
                            if !crate::dag::accumulator_at_least_as_wide(
                                operand,
                                *accumulator,
                                default,
                            ) {
                                errors.push(format!(
                                    "matmul at node {} has accumulator `{}` narrower than \
                                     the spec/04-type-system.md §5.7.1 default `{}` for \
                                     operand precision `{}`",
                                    node.id.0,
                                    accumulator.name(),
                                    default.name(),
                                    operand.name(),
                                ));
                            }
                        }
                        Err(msg) => errors.push(format!("matmul at node {}: {msg}", node.id.0)),
                    }
                    // F1 (WS-A0 RT-1 fixup, tactical) — lifted by WS-A1
                    // (C backend f64), WS-A2 (HIP backend f64), and
                    // WS-A3 (HIP backend bf16/f16).
                    //
                    // Original guard: every non-f32 BlasMatmul was
                    // rejected because the C/HIP/Metal backends
                    // destructured BlasMatmul with `..` and called
                    // single-precision GEMM regardless of operand
                    // precision, producing silent precision loss for
                    // non-f32 source. The C backend now dispatches
                    // `cblas_sgemm`/`cblas_dgemm` for f32/f64 (WS-A1).
                    // The HIP backend binds the accumulator field
                    // explicitly and routes f32/f64 through
                    // `hipblasSgemm`/`hipblasDgemm` (WS-A2) and bf16/f16
                    // through `hipblasGemmEx` (WS-A3) per
                    // spec/04-type-system.md §5.7.1.
                    //
                    // This IR guard's remaining job is to keep the
                    // still-unsupported operand precisions (integer
                    // matmul per §5.7.2, lifted in WS-A4) from ever
                    // reaching any backend's destructure-`..` footgun.
                    // The literal "F1:" tag keeps the remaining lift
                    // trivial to grep for.
                    let backend_supported =
                        matches!(operand, Prim::F32 | Prim::F64 | Prim::Bf16 | Prim::F16);
                    if !backend_supported {
                        // RT-2 fixup P3 (comment): the integer-matmul
                        // arm of this guard is permanent, not pending
                        // a future "lift". Spec §5.7.2 declares
                        // integer matmul not admitted, so the F1
                        // guard's residual purpose is to keep
                        // integer-precision BlasMatmul from ever
                        // reaching a backend even if a hand-built or
                        // future-pass IR slips one through. The
                        // type-checker now rejects integer matmul
                        // upfront with a §5.7.2-citing diagnostic
                        // (RT-2 B6), so this branch is defense in
                        // depth.
                        errors.push(format!(
                            "F1: BlasMatmul on operand precision `{}` is not admitted; \
                             node {} (accumulator `{}`). \
                             spec/04-type-system.md §5.7.2 declares integer matmul not \
                             admitted in this cycle; the type checker rejects integer \
                             operands upfront and this verify-level guard is defense in \
                             depth. Float operand precisions f32/f64/bf16/f16 are \
                             admitted; deferred dtype `f8e4m3` is rejected per §1.1.1.",
                            operand.name(),
                            node.id.0,
                            accumulator.name(),
                        ));
                    }
                }
            }
            _ => {}
        }

        // C4: transcendental ops require float precision. `abs` is the
        // exact numeric exception: it has typed float and signed-integer
        // kernels. Integer floor/ceil/round are canonicalized to identity
        // during lowering, so seeing one of those integer nodes is still a
        // structural error rather than permission to enter a float backend.
        match &node.op {
            RiscOp::Exp
            | RiscOp::Log
            | RiscOp::Sin
            | RiscOp::Sqrt
            | RiscOp::Cos
            | RiscOp::Tan
            | RiscOp::Atan
            | RiscOp::Floor
            | RiscOp::Ceil
            | RiscOp::Round => {
                if arity == 1
                    && let Some(input) = dag.get(node.inputs[0])
                    && !input.output_type.precision.is_float()
                {
                    errors.push(format!(
                        "transcendental op {:?} at node {} requires float input, got {:?}",
                        node.op, node.id.0, input.output_type.precision
                    ));
                }
            }
            RiscOp::Abs => {
                if arity == 1
                    && let Some(input) = dag.get(node.inputs[0])
                    && !input.output_type.precision.is_float()
                    && !input.output_type.precision.is_integer()
                {
                    errors.push(format!(
                        "numeric op Abs at node {} requires float or signed-integer input, got {:?}",
                        node.id.0, input.output_type.precision
                    ));
                }
            }
            RiscOp::UniformLike { .. } => {
                if matches!(arity, 1 | 2)
                    && let Some(input) = dag.get(node.inputs[0])
                {
                    if !input.output_type.precision.is_float() {
                        errors.push(format!(
                            "uniform_like at node {} requires a float template, got {:?}",
                            node.id.0, input.output_type.precision
                        ));
                    }
                    if node.output_type.precision != input.output_type.precision {
                        errors.push(format!(
                            "uniform_like at node {} output precision {:?} must match template precision {:?}",
                            node.id.0,
                            node.output_type.precision,
                            input.output_type.precision
                        ));
                    }
                }
            }
            _ => {}
        }

        // C6: Permute validation.
        if let RiscOp::Permute { axes } = &node.op
            && arity == 1
        {
            let input = dag.get(node.inputs[0]).unwrap();
            let rank = input.output_type.dims.len();
            if axes.len() != rank {
                errors.push(format!(
                    "permute at node {}: axes len {} != input rank {}",
                    node.id.0,
                    axes.len(),
                    rank
                ));
            }
            let mut seen = vec![false; rank];
            for &a in axes {
                if a >= rank {
                    errors.push(format!(
                        "permute at node {}: axis {} >= rank {}",
                        node.id.0, a, rank
                    ));
                } else if seen[a] {
                    errors.push(format!(
                        "permute at node {}: duplicate axis {}",
                        node.id.0, a
                    ));
                } else {
                    seen[a] = true;
                }
            }
        }

        // C7: Reshape validation — static product check when every extent is
        // compile-time known; node-valued (runtime) targets are guarded at
        // eval / C runtime instead (chelis#616). Each `Node` target must
        // reference a valid rank-0 integer bound-scalar slot, and the
        // shrink-only `ToEnd` sentinel is never a valid target.
        if let RiscOp::Reshape { new_shape } = &node.op
            && arity >= 1
        {
            let input = dag.get(node.inputs[0]).unwrap();
            let old_product = dim_product(&input.output_type.dims);
            let new_product = new_shape
                .iter()
                .try_fold(1usize, |acc, dim| dim.as_lit().map(|n| acc * n));
            if let (Some(old), Some(new)) = (old_product, new_product)
                && old != new
            {
                errors.push(format!(
                    "reshape at node {}: product mismatch {} vs {}",
                    node.id.0, old, new
                ));
            }
            for (axis, dim) in new_shape.iter().enumerate() {
                check_bound_source(
                    dag,
                    node,
                    dim,
                    &format!("reshape at node {} target axis {}", node.id.0, axis),
                    &mut errors,
                );
                if matches!(dim, RtDim::ToEnd) {
                    errors.push(format!(
                        "reshape at node {}: target axis {} uses the ToEnd sentinel, \
                         which is only valid as a shrink end",
                        node.id.0, axis
                    ));
                }
            }
            check_exact_bound_inputs(node, new_shape, "reshape", &mut errors);
        }

        // C8: Cast validation — dims must not change, output precision must match target.
        // Both ladder rungs share the shape rule; only their element
        // semantics differ.
        if let RiscOp::Cast { new_precision } | RiscOp::CastTrunc { new_precision } = &node.op
            && arity == 1
        {
            let input = dag.get(node.inputs[0]).unwrap();
            if input.output_type.dims != node.output_type.dims {
                errors.push(format!("cast at node {}: dims changed", node.id.0));
            }
            if node.output_type.precision != *new_precision {
                errors.push(format!(
                    "cast at node {}: output precision doesn't match cast target",
                    node.id.0
                ));
            }
        }

        // C9: Reduction output rank check.
        match &node.op {
            RiscOp::Sum { .. }
            | RiscOp::MaxReduce { .. }
            | RiscOp::MinReduce { .. }
            | RiscOp::ProdReduce { .. }
            | RiscOp::Argmax { .. }
            | RiscOp::Argmin { .. } => {
                if arity == 1 {
                    let input = dag.get(node.inputs[0]).unwrap();
                    let expected_rank = input.output_type.dims.len().saturating_sub(1);
                    if node.output_type.dims.len() != expected_rank {
                        errors.push(format!(
                            "reduction at node {}: output rank {} != expected {}",
                            node.id.0,
                            node.output_type.dims.len(),
                            expected_rank
                        ));
                    }
                }
            }
            _ => {}
        }

        // C10: Movement op shape validation.
        match &node.op {
            RiscOp::Expand { axis, size } => {
                let expected_arity = match size {
                    RtDim::Lit(_) => 1,
                    RtDim::Node(1) | RtDim::InputAxis { tensor: 1, .. } => 2,
                    RtDim::Node(_) | RtDim::InputAxis { .. } => {
                        errors.push(format!(
                            "expand at node {}: runtime size must reference absolute input slot 1, got {size:?}",
                            node.id.0
                        ));
                        arity
                    }
                    RtDim::ToEnd | RtDim::Sym(_) => {
                        errors.push(format!(
                            "expand at node {}: size carrier {size:?} is forbidden; expected Lit, Node, or InputAxis",
                            node.id.0
                        ));
                        arity
                    }
                };
                if arity != expected_arity {
                    errors.push(format!(
                        "expand at node {} has {} inputs (expected {} for {size:?})",
                        node.id.0, arity, expected_arity
                    ));
                }
                check_bound_source(
                    dag,
                    node,
                    size,
                    &format!("Expand at node {}", node.id.0),
                    &mut errors,
                );
                if arity >= 1 {
                    let input = dag.get(node.inputs[0]).unwrap();
                    let input_rank = input.output_type.dims.len();
                    let output_rank = node.output_type.dims.len();
                    if *axis > input_rank {
                        errors.push(format!(
                            "expand at node {}: axis {} > input rank {}",
                            node.id.0, axis, input_rank
                        ));
                    }
                    if node.output_type.precision != input.output_type.precision {
                        errors.push(format!(
                            "expand at node {}: output precision {:?} != input precision {:?}",
                            node.id.0, node.output_type.precision, input.output_type.precision
                        ));
                    }
                    if output_rank == input_rank + 1 && *axis <= input_rank {
                        for out_i in 0..output_rank {
                            if out_i == *axis {
                                if let Some(out_size) =
                                    dim_known_size(&node.output_type.dims[out_i])
                                    && size.as_lit().is_some_and(|size| out_size != size)
                                {
                                    errors.push(format!(
                                        "expand at node {}: inserted axis {} has size {}, expected {}",
                                        node.id.0,
                                        axis,
                                        out_size,
                                        size.as_lit().unwrap()
                                    ));
                                }
                            } else {
                                let in_i = if out_i < *axis { out_i } else { out_i - 1 };
                                if !dims_compatible(
                                    &node.output_type.dims[out_i],
                                    &input.output_type.dims[in_i],
                                ) {
                                    errors.push(format!(
                                        "expand at node {}: output axis {} {:?} incompatible with input axis {} {:?}",
                                        node.id.0, out_i, node.output_type.dims[out_i], in_i, input.output_type.dims[in_i]
                                    ));
                                }
                            }
                        }
                    } else if output_rank == input_rank {
                        if *axis >= input_rank {
                            errors.push(format!(
                                "expand at node {}: axis {} >= input rank {} for same-rank expand",
                                node.id.0, axis, input_rank
                            ));
                        } else {
                            if let Some(in_size) = dim_known_size(&input.output_type.dims[*axis])
                                && in_size != 1
                            {
                                errors.push(format!(
                                    "expand at node {}: same-rank expand requires input axis {} to have size 1, got {}",
                                    node.id.0, axis, in_size
                                ));
                            }
                            if let Some(out_size) = dim_known_size(&node.output_type.dims[*axis])
                                && size.as_lit().is_some_and(|size| out_size != size)
                            {
                                errors.push(format!(
                                    "expand at node {}: output axis {} has size {}, expected {}",
                                    node.id.0,
                                    axis,
                                    out_size,
                                    size.as_lit().unwrap()
                                ));
                            }
                            for out_i in 0..output_rank {
                                if out_i == *axis {
                                    continue;
                                }
                                if !dims_compatible(
                                    &node.output_type.dims[out_i],
                                    &input.output_type.dims[out_i],
                                ) {
                                    errors.push(format!(
                                        "expand at node {}: output axis {} {:?} incompatible with input axis {} {:?}",
                                        node.id.0, out_i, node.output_type.dims[out_i], out_i, input.output_type.dims[out_i]
                                    ));
                                }
                            }
                        }
                    } else {
                        errors.push(format!(
                            "expand at node {}: output rank {} must equal input rank {} or {}",
                            node.id.0,
                            output_rank,
                            input_rank,
                            input_rank + 1
                        ));
                    }
                }
            }
            RiscOp::OneHot { vocab } => {
                if arity == 1 {
                    let input = dag.get(node.inputs[0]).unwrap();
                    if *vocab == 0 {
                        errors.push(format!("one_hot at node {}: vocab must be > 0", node.id.0));
                    }
                    if !matches!(input.output_type.precision, Prim::Int32 | Prim::Int64) {
                        errors.push(format!(
                            "one_hot at node {} requires i32/i64 indices, got {:?}",
                            node.id.0, input.output_type.precision
                        ));
                    }
                    if node.output_type.precision != Prim::F32 {
                        errors.push(format!(
                            "one_hot at node {} output precision {:?}, expected F32",
                            node.id.0, node.output_type.precision
                        ));
                    }
                    let mut expected = input.output_type.dims.clone();
                    expected.push(DimInfo::Lit(*vocab));
                    if node.output_type.dims != expected {
                        errors.push(format!(
                            "one_hot at node {} has output dims {:?}, expected {:?}",
                            node.id.0, node.output_type.dims, expected
                        ));
                    }
                }
            }
            RiscOp::Pad { padding, fill } => {
                if !node.inputs.is_empty() {
                    let input = dag.get(node.inputs[0]).unwrap();
                    let input_rank = input.output_type.dims.len();
                    if padding.len() != input_rank {
                        errors.push(format!(
                            "pad at node {}: padding len {} != input rank {}",
                            node.id.0,
                            padding.len(),
                            input_rank
                        ));
                    }
                    if node.output_type.precision != input.output_type.precision {
                        errors.push(format!(
                            "pad at node {}: output precision {:?} != input precision {:?}",
                            node.id.0, node.output_type.precision, input.output_type.precision
                        ));
                    }
                    if fill.prim() != node.output_type.precision {
                        errors.push(format!(
                            "pad at node {}: fill precision {:?} != output precision {:?}",
                            node.id.0,
                            fill.prim(),
                            node.output_type.precision
                        ));
                    }
                    // chelis#616: validate node-valued bound sources; `ToEnd` is
                    // shrink-only.
                    for (axis, (before, after)) in padding.iter().enumerate() {
                        check_bound_source(
                            dag,
                            node,
                            before,
                            &format!("pad at node {} axis {} before", node.id.0, axis),
                            &mut errors,
                        );
                        check_bound_source(
                            dag,
                            node,
                            after,
                            &format!("pad at node {} axis {} after", node.id.0, axis),
                            &mut errors,
                        );
                        if matches!(before, RtDim::ToEnd) || matches!(after, RtDim::ToEnd) {
                            errors.push(format!(
                                "pad at node {}: axis {} uses the ToEnd sentinel, which is \
                                 only valid as a shrink end",
                                node.id.0, axis
                            ));
                        }
                        if matches!(before, RtDim::Sym(_)) || matches!(after, RtDim::Sym(_)) {
                            errors.push(format!(
                                "pad at node {}: axis {} uses a symbolic dim, which is only \
                                 valid as a reshape target",
                                node.id.0, axis
                            ));
                        }
                        if matches!(before, RtDim::InputAxis { .. })
                            || matches!(after, RtDim::InputAxis { .. })
                        {
                            errors.push(format!(
                                "Pad at node {} axis {} cannot own RtDim::InputAxis; folded shape reads are legal only for Expand and Reshape",
                                node.id.0, axis
                            ));
                        }
                    }
                    check_exact_bound_inputs(
                        node,
                        padding.iter().flat_map(|(before, after)| [before, after]),
                        "pad",
                        &mut errors,
                    );
                    if node.output_type.dims.len() != input_rank {
                        errors.push(format!(
                            "pad at node {}: output rank {} != input rank {}",
                            node.id.0,
                            node.output_type.dims.len(),
                            input_rank
                        ));
                    } else {
                        for (axis, ((before, after), in_dim)) in padding
                            .iter()
                            .zip(input.output_type.dims.iter())
                            .enumerate()
                        {
                            // Static size check only for compile-time bounds.
                            if let (Some(before), Some(after)) = (before.as_lit(), after.as_lit())
                                && let Some(in_size) = dim_known_size(in_dim)
                            {
                                let expected = in_size + before + after;
                                if let Some(out_size) = dim_known_size(&node.output_type.dims[axis])
                                    && out_size != expected
                                {
                                    errors.push(format!(
                                        "pad at node {}: output axis {} has size {}, expected {}",
                                        node.id.0, axis, out_size, expected
                                    ));
                                }
                            }
                        }
                    }
                }
            }
            RiscOp::Shrink { bounds } => {
                if !node.inputs.is_empty() {
                    let input = dag.get(node.inputs[0]).unwrap();
                    let input_rank = input.output_type.dims.len();
                    if bounds.len() != input_rank {
                        errors.push(format!(
                            "shrink at node {}: bounds len {} != input rank {}",
                            node.id.0,
                            bounds.len(),
                            input_rank
                        ));
                    }
                    if node.output_type.precision != input.output_type.precision {
                        errors.push(format!(
                            "shrink at node {}: output precision {:?} != input precision {:?}",
                            node.id.0, node.output_type.precision, input.output_type.precision
                        ));
                    }
                    // chelis#616: validate node-valued bound sources; a `ToEnd`
                    // start is malformed (the sentinel is an `end`-only marker).
                    for (axis, (start, end)) in bounds.iter().enumerate() {
                        check_bound_source(
                            dag,
                            node,
                            start,
                            &format!("shrink at node {} axis {} start", node.id.0, axis),
                            &mut errors,
                        );
                        check_bound_source(
                            dag,
                            node,
                            end,
                            &format!("shrink at node {} axis {} end", node.id.0, axis),
                            &mut errors,
                        );
                        if matches!(start, RtDim::ToEnd) {
                            errors.push(format!(
                                "shrink at node {}: axis {} start uses the ToEnd sentinel, \
                                 which is only valid as an end",
                                node.id.0, axis
                            ));
                        }
                        // chelis#1480, `spec/05` section 2.4.1: a `ToEnd` end
                        // is well formed only when the start paired with it is
                        // `Lit(0)`. A `ToEnd` end over any other start is a
                        // malformed bound, rejected rather than resolved to a
                        // slice.
                        if matches!(end, RtDim::ToEnd) && start.as_lit() != Some(0) {
                            errors.push(format!(
                                "shrink at node {}: axis {} pairs the ToEnd sentinel with a \
                                 start that is not Lit(0), which is a malformed bound",
                                node.id.0, axis
                            ));
                        }
                        if matches!(start, RtDim::Sym(_)) || matches!(end, RtDim::Sym(_)) {
                            errors.push(format!(
                                "shrink at node {}: axis {} uses a symbolic dim, which is \
                                 only valid as a reshape target",
                                node.id.0, axis
                            ));
                        }
                        if matches!(start, RtDim::InputAxis { .. })
                            || matches!(end, RtDim::InputAxis { .. })
                        {
                            errors.push(format!(
                                "Shrink at node {} axis {} cannot own RtDim::InputAxis; materialize the shape read as a Node bound",
                                node.id.0, axis
                            ));
                        }
                    }
                    check_exact_bound_inputs(
                        node,
                        bounds.iter().flat_map(|(start, end)| [start, end]),
                        "shrink",
                        &mut errors,
                    );
                    if node.output_type.dims.len() != input_rank {
                        errors.push(format!(
                            "shrink at node {}: output rank {} != input rank {}",
                            node.id.0,
                            node.output_type.dims.len(),
                            input_rank
                        ));
                    } else {
                        for (axis, ((start, end), in_dim)) in
                            bounds.iter().zip(input.output_type.dims.iter()).enumerate()
                        {
                            // Static checks only for compile-time `(Lit, Lit)`
                            // bounds; `Node`/`ToEnd` are validated at runtime /
                            // bind time.
                            let (Some(start), Some(end)) = (start.as_lit(), end.as_lit()) else {
                                continue;
                            };
                            if start > end {
                                errors.push(format!(
                                    "shrink at node {}: axis {} has invalid bounds ({}, {})",
                                    node.id.0, axis, start, end
                                ));
                                continue;
                            }
                            if let Some(in_size) = dim_known_size(in_dim)
                                && end > in_size
                            {
                                errors.push(format!(
                                    "shrink at node {}: axis {} end {} > input size {}",
                                    node.id.0, axis, end, in_size
                                ));
                            }
                            let expected = end - start;
                            if let Some(out_size) = dim_known_size(&node.output_type.dims[axis])
                                && out_size != expected
                            {
                                errors.push(format!(
                                    "shrink at node {}: output axis {} has size {}, expected {}",
                                    node.id.0, axis, out_size, expected
                                ));
                            }
                        }
                    }
                }
            }
            RiscOp::Stride { strides } => {
                if !node.inputs.is_empty() {
                    let input = dag.get(node.inputs[0]).unwrap();
                    let input_rank = input.output_type.dims.len();
                    if strides.len() != input_rank {
                        errors.push(format!(
                            "stride at node {}: strides len {} != input rank {}",
                            node.id.0,
                            strides.len(),
                            input_rank
                        ));
                    }
                    if node.output_type.precision != input.output_type.precision {
                        errors.push(format!(
                            "stride at node {}: output precision {:?} != input precision {:?}",
                            node.id.0, node.output_type.precision, input.output_type.precision
                        ));
                    }
                    // chelis#616: validate node-valued step sources; `ToEnd` is
                    // never a valid stride step.
                    for (axis, step) in strides.iter().enumerate() {
                        check_bound_source(
                            dag,
                            node,
                            step,
                            &format!("stride at node {} axis {} step", node.id.0, axis),
                            &mut errors,
                        );
                        if matches!(step, RtDim::ToEnd) {
                            errors.push(format!(
                                "stride at node {}: axis {} step uses the ToEnd sentinel, \
                                 which is not a valid stride",
                                node.id.0, axis
                            ));
                        }
                        if matches!(step, RtDim::Sym(_)) {
                            errors.push(format!(
                                "stride at node {}: axis {} step uses a symbolic dim, which \
                                 is only valid as a reshape target",
                                node.id.0, axis
                            ));
                        }
                        if matches!(step, RtDim::InputAxis { .. }) {
                            errors.push(format!(
                                "Stride at node {} axis {} cannot own RtDim::InputAxis; materialize the shape read as a Node bound",
                                node.id.0, axis
                            ));
                        }
                    }
                    check_exact_bound_inputs(node, strides, "stride", &mut errors);
                    if node.output_type.dims.len() != input_rank {
                        errors.push(format!(
                            "stride at node {}: output rank {} != input rank {}",
                            node.id.0,
                            node.output_type.dims.len(),
                            input_rank
                        ));
                    } else {
                        for (axis, (step, in_dim)) in strides
                            .iter()
                            .zip(input.output_type.dims.iter())
                            .enumerate()
                        {
                            let Some(step) = step.as_lit() else {
                                continue;
                            };
                            if step == 0 {
                                errors.push(format!(
                                    "stride at node {}: axis {} has invalid step 0",
                                    node.id.0, axis
                                ));
                                continue;
                            }
                            if let Some(in_size) = dim_known_size(in_dim) {
                                let expected = in_size.div_ceil(step);
                                if let Some(out_size) = dim_known_size(&node.output_type.dims[axis])
                                    && out_size != expected
                                {
                                    errors.push(format!(
                                        "stride at node {}: output axis {} has size {}, expected {}",
                                        node.id.0, axis, out_size, expected
                                    ));
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        // C11: Store arity (already checked above, but explicit message).
        if let RiscOp::Store { .. } = &node.op
            && node.inputs.len() != 1
        {
            errors.push(format!(
                "store at node {} has {} inputs (expected 1)",
                node.id.0,
                node.inputs.len()
            ));
        }

        if matches!(node.op, RiscOp::Drop) {
            if dag.is_root(node.id) {
                errors.push(format!(
                    "drop at node {} is terminal and cannot be a DAG root",
                    node.id.0
                ));
            }
            if consumers[node.id.0] != 0 {
                errors.push(format!(
                    "drop at node {} is terminal and cannot be consumed",
                    node.id.0
                ));
            }
        }

        let is_implicit_root = dag.roots().is_empty() && node.id.0 + 1 == dag.len();
        if reject_dangling
            && !dag.is_root(node.id)
            && !is_implicit_root
            && consumers[node.id.0] == 0
            && !matches!(node.op, RiscOp::Store { .. } | RiscOp::Drop)
        {
            errors.push(format!(
                "node {} is dangling: it has no consumers and is not a DAG root",
                node.id.0
            ));
        }
    }

    // chelis#1277 C4.1: every realized output axis has one checked extent
    // source. `verify` is one of the production paths this runs on, not the
    // only one: it runs in tests and at the end of `grad_dag`, while eval
    // and the three codegen entries call `check_axis_sources` themselves.
    if let Err(unsupported) =
        crate::axis_sources::check_axis_sources(dag, chelis_types::unsupported::Stage::Lowering)
    {
        errors.push(unsupported.to_string());
    }

    errors
}

/// Shared structural verification for replace-scatter and scatter-add.
/// Inputs are `[target, indices, updates]`; output shape equals
/// `target`; `updates` shape equals `target.dims[..axis] +
/// indices.dims + target.dims[axis+1..]`. Indices must be i32/i64.
fn verify_scatter_like(
    node: &crate::dag::DagNode,
    dag: &Dag,
    axis: usize,
    label: &str,
    errors: &mut Vec<String>,
) {
    let arity = node.inputs.len();
    if arity != 3 {
        errors.push(format!(
            "{label} at node {} has {} inputs (expected 3)",
            node.id.0, arity
        ));
        return;
    }
    if let (Some(target), Some(indices), Some(updates)) = (
        dag.get(node.inputs[0]),
        dag.get(node.inputs[1]),
        dag.get(node.inputs[2]),
    ) {
        if !matches!(indices.output_type.precision, Prim::Int32 | Prim::Int64) {
            errors.push(format!(
                "{label} at node {} requires i32/i64 indices, got {:?}",
                node.id.0, indices.output_type.precision
            ));
        }
        if updates.output_type.precision != target.output_type.precision {
            errors.push(format!(
                "{label} at node {} update precision {:?} must match target precision {:?}",
                node.id.0, updates.output_type.precision, target.output_type.precision
            ));
        }
        if node.output_type != target.output_type {
            errors.push(format!(
                "{label} at node {} output type must match target",
                node.id.0
            ));
        }
        if axis >= target.output_type.dims.len() {
            errors.push(format!(
                "{label} at node {} has axis {} out of bounds for rank {}",
                node.id.0,
                axis,
                target.output_type.dims.len()
            ));
        } else {
            let mut expected_updates = Vec::new();
            expected_updates.extend_from_slice(&target.output_type.dims[..axis]);
            expected_updates.extend(indices.output_type.dims.iter().cloned());
            expected_updates.extend_from_slice(&target.output_type.dims[axis + 1..]);
            if updates.output_type.dims != expected_updates {
                errors.push(format!(
                    "{label} at node {} has update dims {:?}, expected {:?}",
                    node.id.0, updates.output_type.dims, expected_updates
                ));
            }
        }
    }
}

/// Verify the ONNX `ScatterElements` structural contract
/// (`spec/05-risc-primitives.md` §3.5.1). Distinct from
/// [`verify_scatter_like`]: `data`, `indices`, and `updates` share a
/// rank, `indices.dims == updates.dims`, and `output.dims ==
/// data.dims`.
fn verify_scatter_elements(
    node: &crate::dag::DagNode,
    dag: &Dag,
    axis: usize,
    errors: &mut Vec<String>,
) {
    let label = "scatter_elements";
    let arity = node.inputs.len();
    if arity != 3 {
        errors.push(format!(
            "{label} at node {} has {} inputs (expected 3)",
            node.id.0, arity
        ));
        return;
    }
    if let (Some(data), Some(indices), Some(updates)) = (
        dag.get(node.inputs[0]),
        dag.get(node.inputs[1]),
        dag.get(node.inputs[2]),
    ) {
        if !matches!(indices.output_type.precision, Prim::Int32 | Prim::Int64) {
            errors.push(format!(
                "{label} at node {} requires i32/i64 indices, got {:?}",
                node.id.0, indices.output_type.precision
            ));
        }
        if updates.output_type.precision != data.output_type.precision {
            errors.push(format!(
                "{label} at node {} update precision {:?} must match data precision {:?}",
                node.id.0, updates.output_type.precision, data.output_type.precision
            ));
        }
        if node.output_type != data.output_type {
            errors.push(format!(
                "{label} at node {} output type must match data",
                node.id.0
            ));
        }
        if indices.output_type.dims != updates.output_type.dims {
            errors.push(format!(
                "{label} at node {} requires indices.dims == updates.dims, got {:?} vs {:?}",
                node.id.0, indices.output_type.dims, updates.output_type.dims
            ));
        }
        if indices.output_type.dims.len() != data.output_type.dims.len() {
            errors.push(format!(
                "{label} at node {} requires data, indices, and updates to share a rank: \
                 data rank {} vs indices rank {}",
                node.id.0,
                data.output_type.dims.len(),
                indices.output_type.dims.len()
            ));
        }
        if axis >= data.output_type.dims.len() {
            errors.push(format!(
                "{label} at node {} has axis {} out of bounds for rank {}",
                node.id.0,
                axis,
                data.output_type.dims.len()
            ));
        }
    }
}

/// Compute the product of known dimension sizes. Returns None if any dim is unknown.
fn dim_product(dims: &[DimInfo]) -> Option<usize> {
    let mut product = 1usize;
    for d in dims {
        match d {
            DimInfo::Lit(n) => product *= n,
            DimInfo::Named(_, Some(n)) => product *= n,
            DimInfo::Named(_, None) => return None,
        }
    }
    Some(product)
}

fn dim_known_size(dim: &DimInfo) -> Option<usize> {
    match dim {
        DimInfo::Lit(n) => Some(*n),
        DimInfo::Named(_, Some(n)) => Some(*n),
        DimInfo::Named(_, None) => None,
    }
}

/// chelis#616: validate a single movement `RtDim` against the owning node. A
/// `RtDim::Node(i)` must reference a rank-0 i64 input; `InputAxis` must
/// reference a tensor input and a normalized in-range axis.
fn check_bound_source(
    dag: &Dag,
    node: &crate::dag::DagNode,
    bound: &RtDim,
    label: &str,
    errors: &mut Vec<String>,
) {
    match bound {
        RtDim::Node(i) => {
            let i = *i;
            if i == 0 || i >= node.inputs.len() {
                errors.push(format!(
                    "{label}: node-valued bound references invalid input slot {i} \
                     (inputs len {})",
                    node.inputs.len()
                ));
                return;
            }
            let src = dag.get(node.inputs[i]).unwrap();
            if !src.output_type.dims.is_empty() {
                errors.push(format!(
                    "{label}: node-valued bound source (input slot {i}) must be a rank-0 \
                     scalar, got rank {}",
                    src.output_type.dims.len()
                ));
            }
            if src.output_type.precision != Prim::Int64 {
                errors.push(format!(
                    "{label}: node-valued bound source (input slot {i}) must be i64, got `{}`",
                    src.output_type.precision.name()
                ));
            }
        }
        RtDim::InputAxis {
            tensor,
            axis: crate::dag::RtAxis::Lit(axis),
        } => {
            if *tensor == 0 || *tensor >= node.inputs.len() {
                errors.push(format!(
                    "{label}: InputAxis references invalid tensor input slot {tensor} (inputs len {})",
                    node.inputs.len()
                ));
                return;
            }
            let source = dag.get(node.inputs[*tensor]).unwrap();
            let Ok(axis) = usize::try_from(*axis) else {
                errors.push(format!("{label}: InputAxis axis {axis} is not normalized"));
                return;
            };
            if axis >= source.output_type.dims.len() {
                errors.push(format!(
                    "{label}: InputAxis axis {axis} out of bounds for rank {} tensor in input slot {tensor}",
                    source.output_type.dims.len()
                ));
            }
        }
        RtDim::Lit(_) | RtDim::ToEnd | RtDim::Sym(_) => {}
    }
}

/// Every non-data movement input is an explicit runtime-extent edge. Reject
/// stale or accidental inputs that no typed `RtDim` owns instead of letting
/// serialization or a rebuild silently preserve an ambiguous dependency.
fn check_exact_bound_inputs<'a>(
    node: &crate::dag::DagNode,
    bounds: impl IntoIterator<Item = &'a RtDim>,
    label: &str,
    errors: &mut Vec<String>,
) {
    let owned = bounds
        .into_iter()
        .filter_map(|bound| match bound {
            RtDim::Node(input) | RtDim::InputAxis { tensor: input, .. } => Some(*input),
            RtDim::Lit(_) | RtDim::ToEnd | RtDim::Sym(_) => None,
        })
        .collect::<std::collections::BTreeSet<_>>();
    for input in 1..node.inputs.len() {
        if !owned.contains(&input) {
            errors.push(format!(
                "{label} at node {} has unowned runtime extent input slot {input}",
                node.id.0
            ));
        }
    }
}

/// IR validation pass for the Metal target's admissible dtype matrix.
///
/// Per spec/04-type-system.md §1.1.3 the Metal backend rejects f64
/// because Apple Silicon GPUs lack FP64 ALUs (software emulation is
/// explicitly out of scope). The spec names three rejection surfaces:
/// the CLI gate, the IR validation pass, and the codegen entry. This
/// function is the IR validation pass; the CLI gate is
/// `chelis-cli::reject_unsupported_metal_ops` and the codegen entry is
/// `chelis-backend-metal::emit::Emitter::require_metal_admissible`.
/// All three surfaces emit the same diagnostic text so a regression
/// in one is caught by the same string-match tests used by the
/// others.
///
/// Returns `Ok(())` if every node's tensor precision is admissible on
/// Metal, or `Err(message)` with the spec-pinned diagnostic for the
/// first f64 / unsupported-precision node encountered. The message
/// uses the same wording as the CLI gate and codegen entry so the
/// three surfaces speak with one voice.
pub fn validate_metal_admissible_precisions(dag: &Dag) -> Result<(), String> {
    for node in dag.nodes() {
        match node.output_type.precision {
            Prim::F32
            | Prim::F16
            | Prim::Bf16
            | Prim::Int8
            | Prim::Int16
            | Prim::Int32
            | Prim::Int64
            | Prim::Bool => {}
            Prim::F64 => {
                return Err(format!(
                    "IR validation rejects FP64 for `--target metal` (node {}): \
                     Apple Silicon GPUs lack FP64 ALUs; use `--target c` or \
                     `--target hip` for f64 workloads. \
                     See spec/04-type-system.md §1.1.3.",
                    node.id.0
                ));
            }
            other => {
                return Err(format!(
                    "IR validation rejects precision `{}` for `--target metal` \
                     (node {}). The Metal backend admits the active dtype set \
                     per spec/04-type-system.md §1.1.3 except f64; supported: \
                     f32/f16/bf16/i8/i16/i32/i64/bool.",
                    other.name(),
                    node.id.0
                ));
            }
        }
    }
    Ok(())
}

/// Check if two dimension descriptors are compatible.
fn dims_compatible(a: &DimInfo, b: &DimInfo) -> bool {
    match (a, b) {
        (DimInfo::Lit(x), DimInfo::Lit(y)) => x == y,
        (DimInfo::Named(n1, s1), DimInfo::Named(n2, s2)) => {
            if n1 == n2 {
                // Same name: sizes must match if both known.
                match (s1, s2) {
                    (Some(a), Some(b)) => a == b,
                    _ => true,
                }
            } else {
                // Different names: compatible only if sizes match (if known).
                match (s1, s2) {
                    (Some(a), Some(b)) => a == b,
                    _ => true, // unknown sizes are assumed compatible
                }
            }
        }
        (DimInfo::Named(_, Some(s)), DimInfo::Lit(n))
        | (DimInfo::Lit(n), DimInfo::Named(_, Some(s))) => s == n,
        // Named with unknown size vs Lit: assume compatible (can't check).
        (DimInfo::Named(_, None), DimInfo::Lit(_)) | (DimInfo::Lit(_), DimInfo::Named(_, None)) => {
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{ExtentWitnessSite, NodeId, RiscOp, RtAxis, TensorType};

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    fn tensor_ty(dims: &[usize], precision: Prim) -> TensorType {
        TensorType {
            dims: dims.iter().copied().map(DimInfo::Lit).collect(),
            precision,
        }
    }

    struct MappedFixture {
        source: Dag,
        mapped: Dag,
        node_map: Vec<NodeId>,
        root_map: chelis_unord::UnordMap<NodeId, NodeId>,
        spliced: Dag,
        splice_map: chelis_unord::UnordMap<NodeId, NodeId>,
        forward_source: NodeId,
        forward_activation: NodeId,
        cotangent: NodeId,
    }

    fn mapped_fixture() -> MappedFixture {
        let scalar_i64 = tensor_ty(&[], Prim::Int64);
        let scalar_f32 = tensor_ty(&[], Prim::F32);
        let vec2 = tensor_ty(&[2], Prim::F32);
        let mat22 = tensor_ty(&[2, 2], Prim::F32);

        let mut source = Dag::new();
        let source_load = source.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            vec2.clone(),
            None,
        );
        let source_witness = source.add_node(
            RiscOp::ExtentWitness {
                site: ExtentWitnessSite::Caller,
                parameter: "x".into(),
                axis: RtAxis::Lit(0),
                requirements: vec![],
                claims: vec![],
            },
            vec![source_load],
            scalar_i64.clone(),
            None,
        );
        let source_forward = source.add_node(
            RiscOp::synth_const(Prim::F32, 1.0),
            vec![],
            scalar_f32,
            None,
        );
        source.add_shape_dep(source_forward, source_witness);
        source.add_root(source_forward);

        let mut mapped = Dag::new();
        let mapped_load = mapped.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            mat22.clone(),
            None,
        );
        let mapped_witness = mapped.add_node(
            RiscOp::ExtentWitness {
                site: ExtentWitnessSite::Caller,
                parameter: "x".into(),
                axis: RtAxis::Lit(1),
                requirements: vec![],
                claims: vec![],
            },
            vec![mapped_load],
            scalar_i64,
            None,
        );
        let mapped_forward = mapped.add_node(
            RiscOp::synth_const_tensor(Prim::F32, vec![1.0, 1.0]),
            vec![],
            vec2,
            None,
        );
        mapped.add_shape_dep(mapped_forward, mapped_witness);
        mapped.add_root(mapped_forward);

        let mut spliced = Dag::new();
        let actual = spliced.add_node(
            RiscOp::Load {
                name: "actual".into(),
            },
            vec![],
            mat22.clone(),
            None,
        );
        let spliced_witness = spliced.add_node(
            mapped.get(mapped_witness).unwrap().op.clone(),
            vec![actual],
            mapped.get(mapped_witness).unwrap().output_type.clone(),
            None,
        );
        let spliced_forward = spliced.add_node(
            mapped.get(mapped_forward).unwrap().op.clone(),
            vec![],
            mapped.get(mapped_forward).unwrap().output_type.clone(),
            None,
        );
        spliced.add_shape_dep(spliced_forward, spliced_witness);
        let cotangent = spliced.add_node(
            RiscOp::synth_const_tensor(Prim::F32, vec![0.0; 4]),
            vec![],
            mat22,
            None,
        );
        spliced.add_shape_dep(cotangent, spliced_forward);

        MappedFixture {
            source,
            mapped,
            node_map: vec![mapped_load, mapped_witness, mapped_forward],
            root_map: chelis_unord::UnordMap::from([(source_forward, mapped_forward)]),
            spliced,
            splice_map: chelis_unord::UnordMap::from([
                (mapped_load, actual),
                (mapped_witness, spliced_witness),
                (mapped_forward, spliced_forward),
            ]),
            forward_source: source_forward,
            forward_activation: spliced_forward,
            cotangent,
        }
    }

    fn verify_fixture(fixture: &MappedFixture) -> Result<(), String> {
        verify_mapped_gradient_closure(MappedGradientClosure {
            source: &fixture.source,
            mapped: &fixture.mapped,
            node_map: &fixture.node_map,
            root_map: &fixture.root_map,
            spliced: &fixture.spliced,
            splice_map: &fixture.splice_map,
            forward_source: fixture.forward_source,
            forward_activation: fixture.forward_activation,
            cotangents: &[fixture.cotangent],
            expected_cotangents: 1,
        })
    }

    #[test]
    fn mapped_gradient_closure_accepts_complete_correspondence() {
        verify_fixture(&mapped_fixture()).unwrap();
    }

    #[test]
    fn mapped_gradient_closure_rejects_every_missing_identity_class() {
        let mut fixture = mapped_fixture();
        fixture.node_map.pop();
        assert!(verify_fixture(&fixture).unwrap_err().contains("node map"));

        let mut fixture = mapped_fixture();
        fixture.root_map.clear();
        assert!(verify_fixture(&fixture).unwrap_err().contains("root"));

        let mut fixture = mapped_fixture();
        fixture.mapped.node_mut(fixture.node_map[1]).unwrap().op = RiscOp::ExtentWitness {
            site: ExtentWitnessSite::Caller,
            parameter: "x".into(),
            axis: RtAxis::Lit(0),
            requirements: vec![],
            claims: vec![],
        };
        assert!(
            verify_fixture(&fixture)
                .unwrap_err()
                .contains("entry witness")
        );

        let mut fixture = mapped_fixture();
        fixture
            .mapped
            .node_mut(fixture.node_map[fixture.forward_source.0])
            .unwrap()
            .shape_deps
            .clear();
        assert!(
            verify_fixture(&fixture)
                .unwrap_err()
                .contains("activation shape dependencies")
        );

        let mut fixture = mapped_fixture();
        fixture.splice_map.remove(&fixture.node_map[1]);
        assert!(verify_fixture(&fixture).unwrap_err().contains("splice"));

        let mut fixture = mapped_fixture();
        fixture
            .spliced
            .node_mut(fixture.forward_activation)
            .unwrap()
            .shape_deps
            .clear();
        assert!(
            verify_fixture(&fixture)
                .unwrap_err()
                .contains("spliced shape dependencies")
        );

        let mut fixture = mapped_fixture();
        fixture
            .spliced
            .node_mut(fixture.cotangent)
            .unwrap()
            .shape_deps
            .clear();
        assert!(verify_fixture(&fixture).unwrap_err().contains("cotangent"));

        let mut fixture = mapped_fixture();
        fixture
            .spliced
            .node_mut(fixture.cotangent)
            .unwrap()
            .output_type
            .dims[0] = DimInfo::Named("lost_rendered_extent".into(), None);
        assert!(
            verify_fixture(&fixture)
                .unwrap_err()
                .contains("rendered dimension origin")
        );
    }

    #[test]
    fn shape_read_requires_exact_int64_scalar_output() {
        let mut dag = Dag::new();
        let input = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            tensor_ty(&[3], Prim::F32),
            None,
        );
        dag.add_node(
            RiscOp::Shape { axis: 0 },
            vec![input],
            TensorType {
                dims: vec![],
                precision: Prim::Int32,
            },
            None,
        );

        let errors = verify(&dag);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("shape read") && error.contains("i64")),
            "i32 shape output must fail the exact runtime-extent invariant: {errors:?}"
        );
    }

    #[test]
    fn same_shape_rank_relation_is_verified_without_a_result_claim() {
        let mut dag = Dag::new();
        let vector = dag.add_node(
            RiscOp::Load {
                name: "vector".into(),
            },
            vec![],
            tensor_ty(&[8], Prim::F32),
            None,
        );
        let matrix = dag.add_node(
            RiscOp::Load {
                name: "matrix".into(),
            },
            vec![],
            tensor_ty(&[2, 4], Prim::F32),
            None,
        );
        let result = dag.add_node(
            RiscOp::Add,
            vec![vector, matrix],
            tensor_ty(&[2, 4], Prim::F32),
            None,
        );
        dag.add_root(result);

        let errors = verify(&dag);
        assert!(
            errors.iter().any(|error| {
                error.contains("same-shape result")
                    && error.contains("positive-rank operand")
                    && error.contains("rank 1")
                    && error.contains("expected rank 2")
            }),
            "mixed-positive-rank same-shape operation must fail without relying on a result claim: {errors:?}"
        );
    }

    #[test]
    fn valid_dag_no_errors() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
        assert!(verify(&dag).is_empty());
    }

    #[test]
    fn constant_tensor_payload_cardinality_must_match_its_concrete_type() {
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::synth_const_tensor(Prim::F32, vec![1.0, 2.0, 3.0]),
            vec![],
            tensor_ty(&[2, 3], Prim::F32),
            None,
        );
        let errors = verify(&dag);
        assert!(
            errors.iter().any(|error| {
                error.contains("constant tensor")
                    && error.contains("3 values")
                    && error.contains("requires 6")
            }),
            "a cloned unbatched payload may not claim a batched concrete type: {errors:?}"
        );
    }

    #[test]
    fn div_arity_one_rejected() {
        // Div is binary; a single-input Div node must surface the
        // binary-arity diagnostic alongside Add/Mul/Compare/MaxElem.
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Div, vec![a], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("binary op") && e.contains("expected 2")),
            "Div with 1 input must produce a binary-arity error; got {errs:?}"
        );
    }

    #[test]
    fn div_arity_three_rejected() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        let c = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Div, vec![a, b, c], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("binary op") && e.contains("expected 2")),
            "Div with 3 inputs must produce a binary-arity error; got {errs:?}"
        );
    }

    #[test]
    fn recip_arity_two_rejected() {
        // Recip is unary; a two-input Recip node must surface the
        // unary-arity diagnostic alongside the other unary elementwise
        // ops.
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Recip, vec![a, b], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("unary op") && e.contains("expected 1")),
            "Recip with 2 inputs must produce a unary-arity error; got {errs:?}"
        );
    }

    #[test]
    fn recip_arity_zero_rejected() {
        let mut dag = Dag::new();
        dag.add_node(RiscOp::Recip, vec![], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("unary op") && e.contains("expected 1")),
            "Recip with 0 inputs must produce a unary-arity error; got {errs:?}"
        );
    }

    #[test]
    fn div_and_recip_arity_two_and_one_accepted() {
        // Positive parity: well-formed Div(binary) and Recip(unary)
        // nodes must verify cleanly. Recip feeds Div so the DAG has a
        // single root and no dangling nodes.
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 6.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        let recip = dag.add_node(RiscOp::Recip, vec![b], scalar_f32(), None);
        let div = dag.add_node(RiscOp::Div, vec![a, recip], scalar_f32(), None);
        dag.add_root(div);
        let errs = verify(&dag);
        assert!(
            errs.is_empty(),
            "well-formed Div(binary) and Recip(unary) must verify clean; got {errs:?}"
        );
    }

    /// F4: the IR validation pass for Metal must accept admissible
    /// dtypes and reject f64 with the spec-pinned diagnostic.
    #[test]
    fn metal_validation_accepts_admissible_dtypes() {
        for prec in [
            Prim::F32,
            Prim::F16,
            Prim::Bf16,
            Prim::Int8,
            Prim::Int16,
            Prim::Int32,
            Prim::Int64,
            Prim::Bool,
        ] {
            let mut dag = Dag::new();
            dag.add_node(
                RiscOp::Load { name: "x".into() },
                vec![],
                tensor_ty(&[4], prec),
                None,
            );
            assert!(
                validate_metal_admissible_precisions(&dag).is_ok(),
                "{prec:?} should be admissible on Metal"
            );
        }
    }

    #[test]
    fn metal_validation_rejects_f64_with_spec_diagnostic() {
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            tensor_ty(&[4], Prim::F64),
            None,
        );
        let err =
            validate_metal_admissible_precisions(&dag).expect_err("f64 must be rejected on Metal");
        assert!(
            err.contains("FP64") && err.contains("Apple Silicon"),
            "diagnostic must cite the spec hardware constraint: {err}"
        );
        assert!(
            err.contains("§1.1.3"),
            "diagnostic must cite spec section: {err}"
        );
        assert!(
            err.contains("--target c") || err.contains("--target hip"),
            "diagnostic must point at the alternate targets: {err}"
        );
    }

    #[test]
    fn drop_root_is_rejected() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let drop = dag.add_node(RiscOp::Drop, vec![a], scalar_f32(), None);
        dag.add_root(drop);
        let errs = verify(&dag);
        assert!(
            errs.iter().any(|e| e.contains("cannot be a DAG root")),
            "{errs:?}"
        );
    }

    #[test]
    fn consumed_drop_is_rejected() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let drop = dag.add_node(RiscOp::Drop, vec![a], scalar_f32(), None);
        dag.add_node(RiscOp::Neg, vec![drop], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(
            errs.iter().any(|e| e.contains("cannot be consumed")),
            "{errs:?}"
        );
    }

    #[test]
    fn copy_and_drop_require_one_input() {
        let mut copy_dag = Dag::new();
        copy_dag.add_node(RiscOp::Copy, vec![], scalar_f32(), None);
        let copy_errs = verify(&copy_dag);
        assert!(
            copy_errs.iter().any(|e| e.contains("unary op")),
            "{copy_errs:?}"
        );

        let mut drop_dag = Dag::new();
        drop_dag.add_node(RiscOp::Drop, vec![], scalar_f32(), None);
        let drop_errs = verify(&drop_dag);
        assert!(
            drop_errs.iter().any(|e| e.contains("unary op")),
            "{drop_errs:?}"
        );
    }

    #[test]
    fn bad_input_reference() {
        let mut dag = Dag::new();
        // Manually create a node that references a future node (impossible via normal API,
        // but we can test via the replace_node backdoor or by constructing the scenario).
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        // Node 1 references itself (not earlier).
        dag.add_node(RiscOp::Neg, vec![NodeId(1)], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs.iter().any(|e| e.contains("non-earlier node")));
        let _ = a;
    }

    #[test]
    fn wrong_arity_binary() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        // Add with only 1 input.
        dag.add_node(RiscOp::Add, vec![a], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs[0].contains("binary op"));
    }

    #[test]
    fn wrong_arity_unary() {
        let mut dag = Dag::new();
        // Neg with 0 inputs.
        dag.add_node(RiscOp::Neg, vec![], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs[0].contains("unary op"));
    }

    #[test]
    fn wrong_arity_memory() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        // Const with an input (should have 0).
        dag.add_node(
            RiscOp::synth_const(Prim::F32, 2.0),
            vec![a],
            scalar_f32(),
            None,
        );
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs[0].contains("memory op"));
    }

    // --- C1: precision consistency ---

    #[test]
    fn c1_mismatched_precision_binary_op() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b_ty = TensorType {
            dims: vec![],
            precision: Prim::F64,
        };
        let b = dag.add_node(RiscOp::synth_const(b_ty.precision, 2.0), vec![], b_ty, None);
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs.iter().any(|e| e.contains("mismatched precisions")));
    }

    #[test]
    fn c1_matching_precision_ok() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Mul, vec![a, b], scalar_f32(), None);
        assert!(verify(&dag).is_empty());
    }

    /// WS-A4 negative coverage: i8 + i32 add must be rejected by the
    /// IR verifier per spec/04-type-system.md §5.1 (no implicit
    /// precision promotion). This is the i8-specific instance of the
    /// generic `c1_mismatched_precision_binary_op` test above; pinning
    /// it explicitly so a future refactor that special-cases narrow
    /// integers cannot silently widen i8 to i32 at the binary-op site.
    #[test]
    fn ws_a4_i8_plus_i32_add_is_precision_mismatch() {
        let mut dag = Dag::new();
        let i8_ty = TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::Int8,
        };
        let i32_ty = TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::Int32,
        };
        let a = dag.add_node(
            RiscOp::Load { name: "a".into() },
            vec![],
            i8_ty.clone(),
            None,
        );
        let b = dag.add_node(
            RiscOp::Load { name: "b".into() },
            vec![],
            i32_ty.clone(),
            None,
        );
        // The output type doesn't matter — the verifier rejects on the
        // operand mismatch first.
        dag.add_node(RiscOp::Add, vec![a, b], i32_ty, None);
        let errs = verify(&dag);
        assert!(
            errs.iter().any(|e| e.contains("mismatched precisions")),
            "i8 + i32 add must be rejected with a precision-mismatch \
             diagnostic per spec §5.1; got: {errs:?}"
        );
    }

    // --- C2: dimension matching ---

    #[test]
    fn c2_mismatched_dim_count() {
        let mut dag = Dag::new();
        let ty1 = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let ty2 = TensorType {
            dims: vec![DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let a = dag.add_node(RiscOp::synth_const(ty1.precision, 1.0), vec![], ty1, None);
        let b = dag.add_node(RiscOp::synth_const(ty2.precision, 2.0), vec![], ty2, None);
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("mismatched dimension count"))
        );
    }

    #[test]
    fn c2_mismatched_dim_size() {
        let mut dag = Dag::new();
        let ty1 = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let ty2 = TensorType {
            dims: vec![DimInfo::Lit(5)],
            precision: Prim::F32,
        };
        let a = dag.add_node(RiscOp::synth_const(ty1.precision, 1.0), vec![], ty1, None);
        let b = dag.add_node(RiscOp::synth_const(ty2.precision, 2.0), vec![], ty2, None);
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("mismatched dimension at axis"))
        );
    }

    // --- C3: reduction axis bounds ---

    #[test]
    fn c3_sum_axis_out_of_bounds() {
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let x = dag.add_node(
            RiscOp::synth_const(ty.precision, 1.0),
            vec![],
            ty.clone(),
            None,
        );
        dag.add_node(
            RiscOp::Sum {
                axis: 5,
                accumulator: Prim::F32,
            },
            vec![x],
            scalar_f32(),
            None,
        );
        let errs = verify(&dag);
        assert!(errs.iter().any(|e| e.contains("axis 5")));
    }

    #[test]
    fn c3_max_reduce_on_scalar() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::MaxReduce { axis: 0 }, vec![x], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("axis 0 but input has 0 dimensions"))
        );
    }

    #[test]
    fn c3_sum_valid_axis() {
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let out_ty = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::synth_const(ty.precision, 1.0), vec![], ty, None);
        dag.add_node(
            RiscOp::Sum {
                axis: 1,
                accumulator: Prim::F32,
            },
            vec![x],
            out_ty,
            None,
        );
        assert!(verify(&dag).is_empty());
    }

    // --- C4: transcendental float-only ---

    #[test]
    fn c4_exp_on_int_is_error() {
        let mut dag = Dag::new();
        let int_ty = TensorType {
            dims: vec![],
            precision: Prim::Int32,
        };
        let x = dag.add_node(
            RiscOp::synth_const(int_ty.precision, 1.0),
            vec![],
            int_ty.clone(),
            None,
        );
        dag.add_node(RiscOp::Exp, vec![x], int_ty, None);
        let errs = verify(&dag);
        assert!(errs.iter().any(|e| e.contains("transcendental")));
    }

    #[test]
    fn c4_sqrt_on_float_ok() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 4.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(RiscOp::Sqrt, vec![x], scalar_f32(), None);
        assert!(verify(&dag).is_empty());
    }

    #[test]
    fn c4_abs_on_signed_integer_is_valid_exact_ir() {
        let mut dag = Dag::new();
        let int_ty = TensorType {
            dims: vec![DimInfo::Lit(1)],
            precision: Prim::Int64,
        };
        let x = dag.add_node(
            RiscOp::synth_const(int_ty.precision, -1.0),
            vec![],
            int_ty.clone(),
            None,
        );
        dag.add_node(RiscOp::Abs, vec![x], int_ty, None);
        assert!(verify(&dag).is_empty());
    }

    #[test]
    fn c4_integer_rounding_node_is_rejected_as_noncanonical() {
        let mut dag = Dag::new();
        let int_ty = TensorType {
            dims: vec![DimInfo::Lit(1)],
            precision: Prim::Int64,
        };
        let x = dag.add_node(
            RiscOp::synth_const(int_ty.precision, -1.0),
            vec![],
            int_ty.clone(),
            None,
        );
        dag.add_node(RiscOp::Floor, vec![x], int_ty, None);
        let errs = verify(&dag);
        assert!(errs.iter().any(|error| error.contains("requires float")));
    }

    #[test]
    fn c4_abs_on_bool_is_error() {
        let mut dag = Dag::new();
        let bool_ty = TensorType {
            dims: vec![DimInfo::Lit(1)],
            precision: Prim::Bool,
        };
        let x = dag.add_node(
            RiscOp::synth_const(bool_ty.precision, 1.0),
            vec![],
            bool_ty.clone(),
            None,
        );
        dag.add_node(RiscOp::Abs, vec![x], bool_ty, None);
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|error| error.contains("float or signed-integer"))
        );
    }

    // --- C5: Compare(CmpLt) output must be Bool ---

    #[test]
    fn c5_cmplt_non_bool_output_is_error() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        // Wrong: output is F32 instead of Bool.
        dag.add_node(
            RiscOp::Compare(ComparisonKind::CmpLt),
            vec![a, b],
            scalar_f32(),
            None,
        );
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("cmplt") && e.contains("Bool"))
        );
    }

    #[test]
    fn c5_cmplt_bool_output_ok() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        let bool_ty = TensorType {
            dims: vec![],
            precision: Prim::Bool,
        };
        dag.add_node(
            RiscOp::Compare(ComparisonKind::CmpLt),
            vec![a, b],
            bool_ty,
            None,
        );
        assert!(verify(&dag).is_empty());
    }

    // --- C10: movement shape validation ---

    #[test]
    fn c10_expand_axis_out_of_bounds_is_error() {
        let mut dag = Dag::new();
        let input_ty = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], input_ty, None);
        dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: crate::dag::RtDim::Lit(4),
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let errs = verify(&dag);
        assert!(errs.iter().any(|e| e.contains("expand")));
    }

    #[test]
    fn c10_expand_same_rank_broadcast_ok() {
        let mut dag = Dag::new();
        let input_ty = TensorType {
            dims: vec![DimInfo::Lit(1)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], input_ty, None);
        dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: crate::dag::RtDim::Lit(4),
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        assert!(verify(&dag).is_empty());
    }

    #[test]
    fn c10_expand_same_rank_requires_unit_input_axis() {
        let mut dag = Dag::new();
        let input_ty = TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], input_ty, None);
        dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: crate::dag::RtDim::Lit(4),
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("requires input axis 0 to have size 1"))
        );
    }

    #[test]
    fn c10_pad_wrong_rank_is_error() {
        let mut dag = Dag::new();
        let input_ty = TensorType {
            dims: vec![DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], input_ty, None);
        dag.add_node(
            RiscOp::zero_pad(Prim::F32, vec![(RtDim::Lit(1), RtDim::Lit(1))]),
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(5), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let errs = verify(&dag);
        assert!(errs.iter().any(|e| e.contains("pad")));
    }

    #[test]
    fn pad_fill_dtype_mismatch_is_error() {
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::Int64,
        };
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
        let wrong_fill = chelis_types::scalar_from_i64("pad", Prim::Int32, 0).unwrap();
        dag.add_node(
            RiscOp::pad(vec![(RtDim::Lit(1), RtDim::Lit(1))], wrong_fill),
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::Int64,
            },
            None,
        );
        assert!(
            verify(&dag)
                .iter()
                .any(|error| error.contains("fill precision Int32 != output precision Int64"))
        );
    }

    #[test]
    fn uniform_like_rejects_non_float_template() {
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::Int64,
        };
        let template = dag.add_node(
            RiscOp::Load {
                name: "template".into(),
            },
            vec![],
            ty.clone(),
            None,
        );
        dag.add_node(
            RiscOp::UniformLike {
                low: 0.0,
                high: 1.0,
                seed: 7,
            },
            vec![template],
            ty,
            None,
        );
        assert!(
            verify(&dag)
                .iter()
                .any(|error| error.contains("requires a float template"))
        );
    }

    #[test]
    fn uniform_like_path_activation_requires_scalar_bool() {
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::F32,
        };
        let template = dag.add_node(
            RiscOp::Load {
                name: "template".into(),
            },
            vec![],
            ty.clone(),
            None,
        );
        let wrong_activation = dag.add_node(
            RiscOp::Load {
                name: "activation".into(),
            },
            vec![],
            TensorType::scalar_f32(),
            None,
        );
        dag.add_node(
            RiscOp::UniformLike {
                low: 0.0,
                high: 1.0,
                seed: 7,
            },
            vec![template, wrong_activation],
            ty,
            None,
        );
        assert!(
            verify(&dag)
                .iter()
                .any(|error| error.contains("requires a scalar Bool path activation"))
        );
    }

    #[test]
    fn c10_shrink_invalid_bounds_is_error() {
        let mut dag = Dag::new();
        let input_ty = TensorType {
            dims: vec![DimInfo::Lit(8)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], input_ty, None);
        dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(6), RtDim::Lit(2))],
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let errs = verify(&dag);
        assert!(errs.iter().any(|e| e.contains("shrink")));
    }

    #[test]
    fn c10_stride_zero_step_is_error() {
        let mut dag = Dag::new();
        let input_ty = TensorType {
            dims: vec![DimInfo::Lit(8)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], input_ty, None);
        dag.add_node(
            RiscOp::Stride {
                strides: vec![RtDim::Lit(0)],
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(8)],
                precision: Prim::F32,
            },
            None,
        );
        let errs = verify(&dag);
        assert!(errs.iter().any(|e| e.contains("stride")));
    }

    // --- C11: Store arity ---

    #[test]
    fn store_correct_arity() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(
            RiscOp::Store { name: "out".into() },
            vec![x],
            scalar_f32(),
            None,
        );
        assert!(verify(&dag).is_empty());
    }

    #[test]
    fn store_wrong_arity_zero_inputs() {
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::Store { name: "out".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs.iter().any(|e| e.contains("store")));
    }

    #[test]
    fn store_wrong_arity_two_inputs() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(
            RiscOp::Store { name: "out".into() },
            vec![a, b],
            scalar_f32(),
            None,
        );
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs.iter().any(|e| e.contains("store")));
    }

    #[test]
    fn dangling_nonfinal_node_is_error() {
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        let errs = verify(&dag);
        assert!(errs.iter().any(|e| e.contains("dangling")));
    }

    #[test]
    fn root_nodes_are_not_dangling() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_root(a);
        dag.add_root(b);
        assert!(verify(&dag).is_empty());
    }

    #[test]
    fn load_name_type_mismatch_is_error() {
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(2)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(3)],
                precision: Prim::F32,
            },
            None,
        );
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("load 'x' has inconsistent tensor types"))
        );
    }

    #[test]
    fn gather_requires_integer_indices_and_matching_output_precision() {
        let mut dag = Dag::new();
        let values = dag.add_node(
            RiscOp::Load {
                name: "values".into(),
            },
            vec![],
            tensor_ty(&[4, 2], Prim::F64),
            None,
        );
        let bad_indices = dag.add_node(
            RiscOp::synth_const(tensor_ty(&[3], Prim::F32).precision, 0.0),
            vec![],
            tensor_ty(&[3], Prim::F32),
            None,
        );
        dag.add_node(
            RiscOp::Gather { axis: 0 },
            vec![values, bad_indices],
            tensor_ty(&[3, 2], Prim::F32),
            None,
        );

        let errs = verify(&dag);
        assert!(
            errs.iter().any(|e| e.contains("requires i32/i64 indices")),
            "expected integer-index diagnostic, got {errs:?}"
        );
        assert!(
            errs.iter().any(|e| e.contains("output precision")),
            "expected output-precision diagnostic, got {errs:?}"
        );
    }

    #[test]
    fn scatter_add_requires_integer_indices_and_matching_update_precision() {
        let mut dag = Dag::new();
        let target = dag.add_node(
            RiscOp::Load {
                name: "target".into(),
            },
            vec![],
            tensor_ty(&[4, 2], Prim::F64),
            None,
        );
        let bad_indices = dag.add_node(
            RiscOp::synth_const(tensor_ty(&[3], Prim::F32).precision, 0.0),
            vec![],
            tensor_ty(&[3], Prim::F32),
            None,
        );
        let bad_updates = dag.add_node(
            RiscOp::synth_const(tensor_ty(&[3, 2], Prim::F32).precision, 1.0),
            vec![],
            tensor_ty(&[3, 2], Prim::F32),
            None,
        );
        dag.add_node(
            RiscOp::ScatterAdd { axis: 0 },
            vec![target, bad_indices, bad_updates],
            tensor_ty(&[4, 2], Prim::F64),
            None,
        );

        let errs = verify(&dag);
        assert!(
            errs.iter().any(|e| e.contains("requires i32/i64 indices")),
            "expected integer-index diagnostic, got {errs:?}"
        );
        assert!(
            errs.iter().any(|e| e.contains("update precision")),
            "expected update-precision diagnostic, got {errs:?}"
        );
    }
}
