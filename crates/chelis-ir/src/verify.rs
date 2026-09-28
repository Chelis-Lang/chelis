//! DAG structural verification.

use crate::dag::{
    ComparisonKind, Dag, DimExpr, DimInfo, ExtentWitnessSite, NodeId, RiscOp, RtAxis, RtDim,
};
#[allow(unused_imports)]
use chelis_types::key_admission::{KeyAdmission, KeyPrimitive};
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
        && (0..left_node.output_type.dims.len()).all(|axis| {
            axis_extents_semantically_equivalent(dag, left, right, axis, relevant_shape_sources)
        })
}

/// The origin `left`'s and `right`'s extents on `axis` both resolve to, when
/// it is one and the same origin.
pub(crate) fn shared_axis_origin(
    dag: &Dag,
    left: NodeId,
    right: NodeId,
    axis: usize,
    relevant_shape_sources: &[NodeId],
) -> Option<crate::axis_sources::ExtentOrigin> {
    let left = semantic_axis_origin(dag, left, axis, dag.len(), relevant_shape_sources)?;
    let right = semantic_axis_origin(dag, right, axis, dag.len(), relevant_shape_sources)?;
    (left == right).then_some(left)
}

/// Whether `left`'s and `right`'s extents on `axis` are one extent: the
/// same dimension expression, the same witnessed origin, or the same static
/// extent. Where's operand-agreement rule reads it per axis, and so does the
/// runtime `if`'s join condition (`lower.rs`, `join_condition_extents`).
pub(crate) fn axis_extents_semantically_equivalent(
    dag: &Dag,
    left: NodeId,
    right: NodeId,
    axis: usize,
    relevant_shape_sources: &[NodeId],
) -> bool {
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

/// The operand and result rules of the key operations ([05-OP-69..72]) and
/// the key-operand random nodes (spec/10 §3.2), rule V5 among them. The IR
/// verifier and the wire decoder both call this one implementation, as they
/// call [`verify_key_rules`], so the two sides of the codec cannot differ on
/// a shape.
///
/// - A key operation is element-wise over one exact shape: `KeyFromSeed`,
///   `Split` and `FoldIn` produce their operands' dims, and `SplitN` appends
///   its count axis, from a literal or input slot 1, to its key's dims.
/// - `Dropout` and `DropoutReplay` produce their data input's exact type, and
///   `UniformLike` its template's. `UniformBoundAdjoint`'s cotangent has its
///   template's exact type, and its result has that dtype.
/// - V5: a random primitive's key has any rank `r`, and its dims are its
///   data's leading `r` dims (a bound adjoint's data is its cotangent). Each
///   control, the node's own activation and a bound adjoint's result has the
///   dims of the key's leading `c` axes for some `c <= r`. Row `b` of the
///   data, the elements whose leading `r` indices are key index `b` in
///   row-major order, draws with `key[b]` and reads the element of each such
///   operand that its leading `c` indices name.
/// - Every one of them reads exactly its operands. A draw's, a replay's and
///   a key operation's activation is its own ([`KeyGraph::activation`]),
///   never an input, and a key-consuming key operation's is shaped like a
///   leading part of its key's shape; a join's two activations are its last
///   two inputs.
///
/// Dims compare exactly, name and extent alike. The messages are the wire
/// decoder's, each naming its node.
pub fn verify_random_operands(graph: &impl KeyGraph, errors: &mut Vec<String>) {
    for node in 0..graph.node_count() {
        match graph.role(node) {
            role @ (KeyRole::KeyFromSeed
            | KeyRole::Split { .. }
            | KeyRole::FoldIn
            | KeyRole::SplitN { .. }) => key_operation_operands(graph, node, role, errors),
            role @ (KeyRole::Dropout
            | KeyRole::DropoutReplay
            | KeyRole::UniformLike
            | KeyRole::UniformBoundAdjoint) => random_node_operands(graph, node, role, errors),
            KeyRole::KeySelect => key_select_operands(graph, node, errors),
            _ => {}
        }
    }
}

/// Whether `node`'s own activation, if it has one, is a Bool shaped like a
/// leading part of `batch`, the key shape it consumes or reads (rule V5).
fn own_activation_is_per_row(graph: &impl KeyGraph, node: usize, batch: &[DimInfo]) -> bool {
    graph.activation(node).is_none_or(|active| {
        graph.dtype(active) == Some(Prim::Bool)
            && graph
                .dims(active)
                .is_some_and(|active| batch.starts_with(&active))
    })
}

/// A join's operands: two keys of its own exact type, then two Bool
/// activations, each shaped like a leading part of that type's dims.
fn key_select_operands(graph: &impl KeyGraph, node: usize, errors: &mut Vec<String>) {
    let described = graph.describe_node(node);
    let inputs = (0..)
        .map_while(|slot| graph.input(node, slot))
        .collect::<Vec<_>>();
    if inputs.len() != 4 {
        errors.push(format!(
            "{described} must read two keys and then two activations"
        ));
        return;
    }
    let Some(dims) = graph.dims(node) else {
        errors.push(format!("{described} has no dims"));
        return;
    };
    let key_ok = |input: usize| {
        graph.dtype(input) == Some(Prim::Key) && graph.dims(input).is_some_and(|key| key == dims)
    };
    if graph.dtype(node) != Some(Prim::Key) || !key_ok(inputs[0]) || !key_ok(inputs[1]) {
        errors.push(format!(
            "{described} must read two keys of its own exact shape and produce a key"
        ));
    }
    let active_ok = |input: usize| {
        graph.dtype(input) == Some(Prim::Bool)
            && graph
                .dims(input)
                .is_some_and(|active| dims.starts_with(&active))
    };
    if !active_ok(inputs[2]) || !active_ok(inputs[3]) {
        errors.push(format!(
            "{described}'s two activations must be Bools shaped like a leading part of its key's shape"
        ));
    }
}

fn key_operation_operands(
    graph: &impl KeyGraph,
    node: usize,
    role: KeyRole,
    errors: &mut Vec<String>,
) {
    let at = graph.describe_node(node);
    let (operand, arity) = match role {
        KeyRole::KeyFromSeed => (Prim::Int64, 1),
        KeyRole::FoldIn => (Prim::Key, 2),
        KeyRole::SplitN {
            count: SplitCount::Input(_),
        } => (Prim::Key, 2),
        _ => (Prim::Key, 1),
    };
    let inputs = (0..)
        .map_while(|slot| graph.input(node, slot))
        .collect::<Vec<_>>();
    // Its activation is its own, never an input (spec/10 §3.2).
    if inputs.len() != arity {
        errors.push(format!(
            "{at}: key operation has the wrong number of inputs"
        ));
        return;
    }
    if graph.dtype(inputs[0]) != Some(operand) || graph.dtype(node) != Some(Prim::Key) {
        errors.push(format!(
            "{at}: key operation reads the wrong operand dtype or does not produce keys"
        ));
    }
    let (Some(key), Some(dims)) = (graph.dims(inputs[0]), graph.dims(node)) else {
        errors.push(format!("{at}: key operation has an operand without dims"));
        return;
    };
    if role.consumes() && !own_activation_is_per_row(graph, node, &key) {
        errors.push(format!(
            "{at}: key operation's activation must be a Bool shaped like a leading part of its key's shape"
        ));
    }
    let valid = match role {
        KeyRole::SplitN { count } => {
            dims.len() == key.len() + 1
                && dims.starts_with(&key)
                && match count {
                    // A named axis with a known extent claims that extent;
                    // an unresolved one is checked when the split runs.
                    SplitCount::Lit(value) => dims.last().is_some_and(|axis| match axis {
                        DimInfo::Lit(size) | DimInfo::Named(_, Some(size)) => *size == value,
                        DimInfo::Named(_, None) => true,
                    }),
                    SplitCount::Input(slot) => slot == 1,
                    SplitCount::Other => false,
                }
        }
        KeyRole::FoldIn => {
            graph.dtype(inputs[1]) == Some(Prim::Int64)
                && key == dims
                && graph.dims(inputs[1]).is_some_and(|indices| indices == dims)
        }
        _ => key == dims,
    };
    if !valid {
        errors.push(format!(
            "{at}: key operation must be element-wise over one exact shape, with a split's count axis appended last"
        ));
    }
}

fn random_node_operands(
    graph: &impl KeyGraph,
    node: usize,
    role: KeyRole,
    errors: &mut Vec<String>,
) {
    let at = graph.describe_node(node);
    let fixed = match role {
        KeyRole::UniformLike => 4,
        _ => 3,
    };
    let arity = (0..)
        .take_while(|slot| graph.input(node, *slot).is_some())
        .count();
    if arity < fixed {
        errors.push(format!("{at}: random operation is missing an input"));
        return;
    }
    // Its activation is its own, never an input (spec/10 §3.2).
    if arity > fixed {
        errors.push(format!(
            "{at}: random operation has the wrong number of inputs"
        ));
        return;
    }
    let input = |slot: usize| graph.input(node, slot);
    let dtype = |slot: usize| input(slot).and_then(|input| graph.dtype(input));
    let dims = |slot: usize| input(slot).and_then(|input| graph.dims(input));
    let key_slot = role.key_slot();
    // The key batch's dims, when the primitive draws a key batch. Every
    // operand of a primitive whose key is missing or no key is rank 0, which
    // is its own error below.
    let batch = match key_slot
        .filter(|slot| dtype(*slot) == Some(Prim::Key))
        .and_then(dims)
    {
        Some(batch) => batch,
        None => std::borrow::Cow::Borrowed(&[][..]),
    };
    let per_row = |value: Option<std::borrow::Cow<'_, [DimInfo]>>| {
        value.is_some_and(|value| batch.starts_with(&value))
    };
    if !own_activation_is_per_row(graph, node, &batch) {
        errors.push(format!(
            "{at}: random operation's activation must be a Bool shaped like a leading part of its key's shape"
        ));
    }
    if let Some(slot) = key_slot {
        let data = usize::from(role == KeyRole::UniformBoundAdjoint);
        if dtype(slot) != Some(Prim::Key)
            || !dims(data).is_some_and(|data| data.starts_with(&batch))
        {
            errors.push(format!(
                "{at}: random operation requires a key batch matching its data's leading axes"
            ));
        }
    }
    let output = graph.dtype(node);
    let float = |prim: Option<Prim>| prim.is_some_and(|prim| prim.is_float());
    // Whether input `slot` is a control of dtype `draw` (or f32, for a
    // uniform draw's bound), shaped like a leading part of the key.
    let control = |slot: usize, draw: Option<Prim>, uniform: bool| {
        per_row(dims(slot))
            && dtype(slot).is_some_and(|prim| Some(prim) == draw || (uniform && prim == Prim::F32))
    };
    let control_error = || {
        format!(
            "{at}: random control must be a value of the draw's dtype (f32 bounds admitted), shaped like a leading part of its key's shape"
        )
    };
    let bounds_error = || format!("{at}: uniform_like bounds must share one dtype");
    let same_as = |slot: usize| {
        dtype(slot) == output && dims(slot).is_some_and(|dims| graph.dims(node) == Some(dims))
    };
    match role {
        KeyRole::UniformLike => {
            if !float(output) || !same_as(0) {
                errors.push(format!(
                    "{at}: uniform_like must preserve its float template's exact shape and dtype"
                ));
            }
            if !control(1, output, true) || !control(2, output, true) {
                errors.push(control_error());
            }
            if dtype(1) != dtype(2) {
                errors.push(bounds_error());
            }
        }
        KeyRole::Dropout | KeyRole::DropoutReplay => {
            if !float(output) || !same_as(0) {
                errors.push(format!(
                    "{at}: dropout must preserve its float data input's exact shape and dtype"
                ));
            }
            if !control(1, output, false) {
                errors.push(control_error());
            }
        }
        KeyRole::UniformBoundAdjoint => {
            let cotangent_ok = dtype(1) == dtype(0) && dims(1).is_some_and(|g| dims(0) == Some(g));
            if !float(output) || !per_row(graph.dims(node)) || dtype(0) != output || !cotangent_ok {
                errors.push(format!(
                    "{at}: a uniform bound adjoint is a value of its template's dtype, shaped like a leading part of its key's shape, over a cotangent of its template's exact type"
                ));
            }
        }
        _ => unreachable!("random_node_operands reads only random operations"),
    }
}

/// A node's part in the key rules of [`verify_key_rules`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyRole {
    /// `KeyFromSeed`, which produces a key from an i64 seed.
    KeyFromSeed,
    /// One `Split` branch, which consumes the key at input 0.
    Split { branch: crate::dag::KeyBranch },
    /// `FoldIn`, which consumes the key at input 0.
    FoldIn,
    /// `SplitN`, which consumes the key at input 0 and appends `count`.
    SplitN { count: SplitCount },
    /// `KeySelect`, a branch's join, which consumes the key at input 0 under
    /// the activation at input 2 and the key at input 1 under the one at
    /// input 3.
    KeySelect,
    /// A `Load`, which may enter a key into the graph.
    Load,
    /// A `Store`, a named graph output: a key `Store` is the key at its
    /// input 0, observed by the root that names it.
    Store,
    /// `Dropout`, which consumes the key at input 2.
    Dropout,
    /// `UniformLike`, which consumes the key at input 3.
    UniformLike,
    /// `DropoutReplay`, which reads its forward `Dropout`'s key at input 2.
    DropoutReplay,
    /// `UniformBoundAdjoint`, which reads its forward `UniformLike`'s key at
    /// input 2.
    UniformBoundAdjoint,
    /// A `Drop`, source `drop` ([05-OP-67]), which consumes the key at
    /// input 0 and produces no key a later node reads.
    Drop,
    /// A two-input `And`, read by the activation exclusivity rule.
    And,
    /// A `Not`, read by the activation exclusivity rule.
    Not,
    /// The rank-0 Bool constant `false`, read by the activation exclusivity
    /// rule.
    ConstFalse,
    /// The rank-0 Bool constant `true`, read by the join's arm rule, where
    /// folding has left it of one arm.
    ConstTrue,
    /// An operation that takes no key.
    Other,
}

impl KeyRole {
    /// The one input slot at which a key operation or a random primitive
    /// reads its key; `None` for a join, which reads two
    /// ([`Self::key_slots`]), and for any node that reads none.
    fn key_slot(self) -> Option<usize> {
        match self {
            Self::Split { .. } | Self::FoldIn | Self::SplitN { .. } | Self::Drop => Some(0),
            Self::Dropout | Self::DropoutReplay | Self::UniformBoundAdjoint => Some(2),
            Self::UniformLike => Some(3),
            Self::KeyFromSeed
            | Self::KeySelect
            | Self::Load
            | Self::Store
            | Self::And
            | Self::Not
            | Self::ConstFalse
            | Self::ConstTrue
            | Self::Other => None,
        }
    }

    /// The allow-list entry this node's operation is
    /// (`chelis_types::key_admission`), which the checker reads too, or
    /// `None` for an operation that takes no key. Exhaustive, with no
    /// wildcard arm.
    pub fn admission(self) -> Option<KeyAdmission> {
        match self {
            Self::Split { .. } => Some(KeyAdmission::Primitive(KeyPrimitive::SplitKey)),
            Self::SplitN { .. } => Some(KeyAdmission::Primitive(KeyPrimitive::SplitKeys)),
            Self::FoldIn => Some(KeyAdmission::Primitive(KeyPrimitive::FoldIn)),
            // A replay reads the key its forward draw consumed.
            Self::Dropout | Self::DropoutReplay => {
                Some(KeyAdmission::Primitive(KeyPrimitive::Dropout))
            }
            Self::UniformLike | Self::UniformBoundAdjoint => {
                Some(KeyAdmission::Primitive(KeyPrimitive::UniformLike))
            }
            Self::Drop => Some(KeyAdmission::Drop),
            Self::KeySelect => Some(KeyAdmission::Join),
            Self::Store => Some(KeyAdmission::Root),
            Self::KeyFromSeed
            | Self::Load
            | Self::And
            | Self::Not
            | Self::ConstFalse
            | Self::ConstTrue
            | Self::Other => None,
        }
    }

    /// Every input slot at which this node reads a key (rule V4): none
    /// unless the allow-list admits its operation ([`Self::admission`]).
    pub fn key_slots(self) -> &'static [usize] {
        if self.admission().is_none() {
            return &[];
        }
        match self {
            Self::KeySelect => &[0, 1],
            Self::Store => &[0],
            _ => match self.key_slot() {
                Some(0) => &[0],
                Some(2) => &[2],
                Some(3) => &[3],
                _ => &[],
            },
        }
    }

    /// Whether this node's output must be a key.
    fn produces_key(self) -> bool {
        matches!(
            self,
            Self::KeyFromSeed
                | Self::Split { .. }
                | Self::FoldIn
                | Self::SplitN { .. }
                | Self::KeySelect
        )
    }

    /// Whether this node derives its key from the key at input 0, so that
    /// the derived key inherits the parent's confinement ([`verify_confinement`]).
    fn derives(self) -> bool {
        matches!(
            self,
            Self::Split { .. } | Self::FoldIn | Self::SplitN { .. }
        )
    }

    /// How a diagnostic names this node's operation.
    fn operation(self) -> Option<&'static str> {
        Some(match self {
            Self::KeyFromSeed => "`key_from_seed`",
            Self::Split {
                branch: crate::dag::KeyBranch::Left,
            } => "left `split_key`",
            Self::Split {
                branch: crate::dag::KeyBranch::Right,
            } => "right `split_key`",
            Self::FoldIn => "`fold_in`",
            Self::SplitN { .. } => "`split_keys`",
            Self::KeySelect => "`if` join",
            Self::Dropout => "`dropout`",
            Self::UniformLike => "`uniform_like`",
            Self::DropoutReplay => "`dropout` replay",
            Self::UniformBoundAdjoint => "`uniform_like` bound adjoint",
            Self::Drop => "`drop`",
            Self::Load
            | Self::Store
            | Self::And
            | Self::Not
            | Self::ConstFalse
            | Self::ConstTrue
            | Self::Other => return None,
        })
    }

    fn is_draw(self) -> bool {
        matches!(self, Self::Dropout | Self::UniformLike)
    }

    /// Whether reading the key at [`Self::key_slot`] consumes it. A replay
    /// reads its forward draw's key without consuming it.
    fn consumes(self) -> bool {
        self.is_draw() || self.derives() || matches!(self, Self::KeySelect | Self::Drop)
    }
}

/// What an operation reads through one input slot ([`slot_read`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotRead {
    /// The operand's value.
    Value,
    /// Only the operand's extent. A key tensor's extent is not key material,
    /// so a key there is observed, not used, and stays live ([04-LIN-9]).
    Extent(ExtentSlot),
}

impl SlotRead {
    /// The allow-list entry a key at input `slot` of a node of `role` is, or
    /// `None` where the key rules refuse it (rule V4): an extent slot is an
    /// extent observation whatever the operation, and a value slot admits a
    /// key exactly where the operation's own admission does.
    pub fn admission(self, role: KeyRole, slot: usize) -> Option<KeyAdmission> {
        match self {
            Self::Extent(_) => Some(KeyAdmission::ExtentObservation),
            Self::Value => role
                .admission()
                .filter(|_| role.key_slots().contains(&slot)),
        }
    }
}

/// Every carrier through which an operation reads only an input's extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExtentSlot {
    /// `Shape`'s input 0 (source `shape` and `numel`).
    Shape,
    /// `ExtentWitness`'s input 0, the tensor whose axis a call's extent
    /// contract reads.
    ExtentWitness,
    /// An `Expand` size (source `expand` and `insert`) read from an input's
    /// axis.
    ExpandSize,
    /// A `Reshape` target read from an input's axis.
    ReshapeTarget,
    /// A `Pad` bound read from an input's axis.
    PadBound,
    /// A `Shrink` bound read from an input's axis.
    ShrinkBound,
    /// A `Stride` step read from an input's axis.
    StrideStep,
    /// A `SplitN` count read from an input's axis.
    SplitCount,
}

impl ExtentSlot {
    pub const ALL: [Self; 8] = [
        Self::Shape,
        Self::ExtentWitness,
        Self::ExpandSize,
        Self::ReshapeTarget,
        Self::PadBound,
        Self::ShrinkBound,
        Self::StrideStep,
        Self::SplitCount,
    ];
}

/// The input a runtime bound reads, as [`bound_slot_read`] takes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundInput {
    /// A literal, the end of an axis or a symbol: no input.
    None,
    /// `Node(slot)`: the value of a rank-0 integer input.
    Value(usize),
    /// `InputAxis { tensor: slot, .. }`: the extent of a tensor input.
    Extent(usize),
}

impl BoundInput {
    pub fn of(bound: &RtDim) -> Self {
        match bound {
            RtDim::Node(slot) => Self::Value(*slot),
            RtDim::InputAxis { tensor, .. } => Self::Extent(*tensor),
            RtDim::Lit(_) | RtDim::ToEnd | RtDim::Sym(_) => Self::None,
        }
    }
}

/// The read at input `slot` of an operation whose data is input 0 and whose
/// runtime bounds are `bounds`: the extent of an input a bound names by its
/// axis and no bound reads the value of, otherwise a value. The IR table
/// ([`slot_read`]) and the wire decoder's read their bounds through it.
pub fn bound_slot_read(
    kind: ExtentSlot,
    bounds: impl IntoIterator<Item = BoundInput>,
    slot: usize,
) -> SlotRead {
    let (mut extent, mut value) = (false, slot == 0);
    for bound in bounds {
        match bound {
            BoundInput::Extent(read) => extent |= read == slot,
            BoundInput::Value(read) => value |= read == slot,
            BoundInput::None => {}
        }
    }
    if extent && !value {
        SlotRead::Extent(kind)
    } else {
        SlotRead::Value
    }
}

/// The read at input `slot` of an operation that reads only input 0's
/// extent and every other input's value.
pub fn operand_extent_read(kind: ExtentSlot, slot: usize) -> SlotRead {
    if slot == 0 {
        SlotRead::Extent(kind)
    } else {
        SlotRead::Value
    }
}

/// What `op` reads through input `slot`: the one table of value and extent
/// slots, exhaustive with no wildcard arm, so no operation reaches the key
/// rules without declaring which of its inputs it reads only for their
/// extent. A dependency that reads only an extent never uses a key, whether
/// it is one of these slots or a shape dependency, which the key rules do
/// not see ([`KeyGraph`]). A symbolic bound (`RtDim::Sym`) names a dimension
/// and reads no input.
pub fn slot_read(op: &RiscOp, slot: usize) -> SlotRead {
    let bounds = |kind, bounds: &mut dyn Iterator<Item = &RtDim>| {
        bound_slot_read(kind, bounds.map(BoundInput::of), slot)
    };
    match op {
        RiscOp::Shape { .. } => operand_extent_read(ExtentSlot::Shape, slot),
        RiscOp::ExtentWitness { .. } => operand_extent_read(ExtentSlot::ExtentWitness, slot),
        RiscOp::Expand { size, .. } => bounds(ExtentSlot::ExpandSize, &mut std::iter::once(size)),
        RiscOp::Reshape { new_shape } => bounds(ExtentSlot::ReshapeTarget, &mut new_shape.iter()),
        RiscOp::Pad { padding, .. } => bounds(
            ExtentSlot::PadBound,
            &mut padding.iter().flat_map(|(before, after)| [before, after]),
        ),
        RiscOp::Shrink { bounds: pairs } => bounds(
            ExtentSlot::ShrinkBound,
            &mut pairs.iter().flat_map(|(start, end)| [start, end]),
        ),
        RiscOp::Stride { strides } => bounds(ExtentSlot::StrideStep, &mut strides.iter()),
        RiscOp::SplitN { count } => bounds(ExtentSlot::SplitCount, &mut std::iter::once(count)),
        RiscOp::Iota
        | RiscOp::Add
        | RiscOp::Sub
        | RiscOp::Mul
        | RiscOp::Div
        | RiscOp::FloorDiv
        | RiscOp::TruncDiv
        | RiscOp::Mod
        | RiscOp::Compare(_)
        | RiscOp::Logical(_)
        | RiscOp::Where
        | RiscOp::GuardedFail { .. }
        | RiscOp::MaxElem
        | RiscOp::MinElem
        | RiscOp::ExtremaAdjoint { .. }
        | RiscOp::Relu
        | RiscOp::ReluAdjoint
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
        | RiscOp::Round
        | RiscOp::Recip
        | RiscOp::UniformLike
        | RiscOp::Dropout
        | RiscOp::DropoutReplay
        | RiscOp::UniformBoundAdjoint { .. }
        | RiscOp::KeyFromSeed
        | RiscOp::Split { .. }
        | RiscOp::FoldIn
        | RiscOp::KeySelect
        | RiscOp::Sum { .. }
        | RiscOp::Count { .. }
        | RiscOp::MaxReduce { .. }
        | RiscOp::MinReduce { .. }
        | RiscOp::ProdReduce { .. }
        | RiscOp::ReduceWindow { .. }
        | RiscOp::ReduceWindowGrad { .. }
        | RiscOp::Argmax { .. }
        | RiscOp::Argmin { .. }
        | RiscOp::Permute { .. }
        | RiscOp::OneHot { .. }
        // Every input and the result are i64 scalars.
        | RiscOp::CheckedReshapeExtent { .. }
        // Input 0 is the refined tensor itself, not only its extent.
        | RiscOp::CheckedUnitAxis { .. }
        | RiscOp::Const { .. }
        | RiscOp::ConstTensor { .. }
        | RiscOp::Load { .. }
        | RiscOp::Store { .. }
        | RiscOp::Copy
        | RiscOp::Drop
        | RiscOp::Realize
        | RiscOp::Cast { .. }
        | RiscOp::CastTrunc { .. }
        | RiscOp::FusedElem { .. }
        // Its dims are symbolic expressions, which name no input.
        | RiscOp::BlasMatmul { .. }
        | RiscOp::Gather { .. }
        | RiscOp::ScatterAdd { .. }
        | RiscOp::Scatter { .. }
        | RiscOp::ScatterElements { .. } => SlotRead::Value,
    }
}

/// A `SplitN` count as the operand rules of [`verify_random_operands`] read
/// it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitCount {
    /// A literal count.
    Lit(usize),
    /// A runtime count read from the given input slot.
    Input(usize),
    /// A bound no split takes.
    Other,
}

/// The view of a graph that [`verify_key_rules`] and
/// [`verify_random_operands`] read. The IR [`Dag`] implements it, and so
/// does the compiler API's decoded wire graph, so the key rules and the
/// random operand rules each have one implementation on both sides of the
/// codec. A node is named by its position; a position that names no node
/// has no dtype and no dims.
///
/// The view has a node's inputs, what it reads through each
/// ([`Self::slot_read`]), and its activation, and no other edge: a shape
/// dependency reads only its node's extent, which is not key material
/// ([04-LIN-9]), and a result-claim dependency orders a witness, so no key
/// rule can count either as a use.
pub trait KeyGraph {
    fn node_count(&self) -> usize;
    fn role(&self, node: usize) -> KeyRole;
    /// What `node` reads through input `slot`: [`slot_read`] of its
    /// operation.
    fn slot_read(&self, node: usize, slot: usize) -> SlotRead;
    /// The dtype `node` produces, or `None` when no node or no known dtype.
    fn dtype(&self, node: usize) -> Option<Prim>;
    /// Whether two existing nodes produce the same tensor type.
    fn same_type(&self, left: usize, right: usize) -> bool;
    /// The node at `node`'s input `slot`, if the slot exists. Slots are
    /// dense: the first missing slot ends the node's inputs.
    fn input(&self, node: usize, slot: usize) -> Option<usize>;
    /// `node`'s own activation (spec/10 §3.2), if it has one.
    fn activation(&self, node: usize) -> Option<usize>;
    fn roots(&self) -> impl Iterator<Item = usize> + '_;
    /// The parameter a `Load` at `node` reads, or `None` for any other node.
    fn load_name(&self, node: usize) -> Option<&str>;
    /// The name of the declaration `node` belongs to.
    fn declaration(&self, node: usize) -> &str;
    /// Whether two nodes belong to one declaration.
    fn same_declaration(&self, left: usize, right: usize) -> bool;
    /// The dims `node` produces, or `None` when no node or when its dims
    /// have no IR reading.
    fn dims(&self, node: usize) -> Option<std::borrow::Cow<'_, [DimInfo]>>;

    /// How a key-rule diagnostic names the key `node` produces (spec/10
    /// §3.2): a parameter by its name and declaration, "key `k` of `f`",
    /// and any other key by the operation that produces it, its node and its
    /// declaration, "the `fold_in` key at node 7 of `f`". Both sides of the
    /// codec word it here, so their diagnostics agree.
    fn describe_key(&self, node: usize) -> String {
        let of = match self.declaration(node) {
            "" => String::new(),
            declaration => format!(" of `{declaration}`"),
        };
        match (self.load_name(node), self.role(node).operation()) {
            (Some(name), _) => format!("key `{name}`{of}"),
            (None, Some(operation)) => format!("the {operation} key at node {node}{of}"),
            (None, None) => format!("the key at node {node}{of}"),
        }
    }

    /// How a key-rule diagnostic names the node that reads or produces a
    /// key: its operation, its node and its declaration, "the `dropout` at
    /// node 9 of `f`".
    fn describe_node(&self, node: usize) -> String {
        let of = match self.declaration(node) {
            "" => String::new(),
            declaration => format!(" of `{declaration}`"),
        };
        match self.role(node).operation() {
            Some(operation) => format!("the {operation} at node {node}{of}"),
            None => format!("node {node}{of}"),
        }
    }
}

impl KeyGraph for Dag {
    fn node_count(&self) -> usize {
        self.len()
    }

    fn role(&self, node: usize) -> KeyRole {
        match self.get(NodeId(node)).map(|node| &node.op) {
            Some(RiscOp::KeyFromSeed) => KeyRole::KeyFromSeed,
            Some(RiscOp::Split { branch }) => KeyRole::Split { branch: *branch },
            Some(RiscOp::FoldIn) => KeyRole::FoldIn,
            Some(RiscOp::KeySelect) => KeyRole::KeySelect,
            Some(RiscOp::SplitN { count }) => KeyRole::SplitN {
                count: match count {
                    RtDim::Lit(value) => SplitCount::Lit(*value),
                    RtDim::Node(slot) => SplitCount::Input(*slot),
                    _ => SplitCount::Other,
                },
            },
            Some(RiscOp::Load { .. }) => KeyRole::Load,
            Some(RiscOp::Store { .. }) => KeyRole::Store,
            Some(RiscOp::Dropout) => KeyRole::Dropout,
            Some(RiscOp::UniformLike) => KeyRole::UniformLike,
            Some(RiscOp::DropoutReplay) => KeyRole::DropoutReplay,
            Some(RiscOp::UniformBoundAdjoint { .. }) => KeyRole::UniformBoundAdjoint,
            Some(RiscOp::Drop) => KeyRole::Drop,
            Some(RiscOp::Logical(crate::dag::LogicalKind::And)) => KeyRole::And,
            Some(RiscOp::Logical(crate::dag::LogicalKind::Not)) => KeyRole::Not,
            Some(RiscOp::Const { value }) if is_const_false(value) => KeyRole::ConstFalse,
            Some(RiscOp::Const { value }) if is_const_true(value) => KeyRole::ConstTrue,
            _ => KeyRole::Other,
        }
    }

    fn slot_read(&self, node: usize, slot: usize) -> SlotRead {
        self.get(NodeId(node))
            .map_or(SlotRead::Value, |node| slot_read(&node.op, slot))
    }

    fn dtype(&self, node: usize) -> Option<Prim> {
        self.get(NodeId(node))
            .map(|node| node.output_type.precision)
    }

    fn same_type(&self, left: usize, right: usize) -> bool {
        match (self.get(NodeId(left)), self.get(NodeId(right))) {
            (Some(left), Some(right)) => left.output_type == right.output_type,
            _ => false,
        }
    }

    fn input(&self, node: usize, slot: usize) -> Option<usize> {
        Some(self.get(NodeId(node))?.inputs.get(slot)?.0)
    }

    fn activation(&self, node: usize) -> Option<usize> {
        Some(self.get(NodeId(node))?.owner.activation?.0)
    }

    fn roots(&self) -> impl Iterator<Item = usize> + '_ {
        Dag::roots(self).iter().map(|root| root.0)
    }

    fn load_name(&self, node: usize) -> Option<&str> {
        match &self.get(NodeId(node))?.op {
            RiscOp::Load { name } => Some(name.as_ref()),
            _ => None,
        }
    }

    fn declaration(&self, node: usize) -> &str {
        &self.declaration(self.nodes()[node].owner.decl).name
    }

    fn same_declaration(&self, left: usize, right: usize) -> bool {
        self.nodes()[left].owner.decl == self.nodes()[right].owner.decl
    }

    fn dims(&self, node: usize) -> Option<std::borrow::Cow<'_, [DimInfo]>> {
        self.get(NodeId(node))
            .map(|node| std::borrow::Cow::Borrowed(node.output_type.dims.as_slice()))
    }
}

/// The nodes an activation implies: the activation itself and, through
/// every `And`, both conjuncts, transitively.
fn activation_conjuncts(graph: &impl KeyGraph, activation: usize) -> Vec<usize> {
    let mut seen = Vec::new();
    let mut stack = vec![activation];
    while let Some(node) = stack.pop() {
        if seen.contains(&node) {
            continue;
        }
        seen.push(node);
        if graph.role(node) == KeyRole::And {
            stack.extend((0..2).filter_map(|slot| graph.input(node, slot)));
        }
    }
    seen
}

/// Whether a scalar is the Bool constant `false`, the literal that makes an
/// activation implying it exclusive with every other.
pub fn is_const_false(value: &chelis_types::ScalarValue) -> bool {
    value.prim() == Prim::Bool && value.as_bool_exact() == Some(false)
}

/// Whether a scalar is the Bool constant `true`, what folding leaves of a
/// join's arm whose condition became a constant.
pub fn is_const_true(value: &chelis_types::ScalarValue) -> bool {
    value.prim() == Prim::Bool && value.as_bool_exact() == Some(true)
}

/// Rule V3: two activations are structurally exclusive when one implies a
/// node `X` and the other implies `Not(X)`, or when either implies the
/// constant `false`, whose draw never runs. That covers `lower_if`'s arms,
/// `And(P, X)` against `And(P, Not X)`, any arm nested inside one of them,
/// and what constant folding leaves of such a pair when `X` is a constant:
/// one of `X` and `Not(X)` folds to `false`. Activations that are merely
/// never both true at run time do not count.
fn activations_exclusive(graph: &impl KeyGraph, left: usize, right: usize) -> bool {
    let left = activation_conjuncts(graph, left);
    let right = activation_conjuncts(graph, right);
    let negates = |node: usize, other: usize| {
        graph.role(node) == KeyRole::Not && graph.input(node, 0) == Some(other)
    };
    left.iter()
        .chain(&right)
        .any(|node| graph.role(*node) == KeyRole::ConstFalse)
        || left
            .iter()
            .any(|a| right.iter().any(|b| negates(*a, *b) || negates(*b, *a)))
}

/// The key rules of spec/10 §3.2 (`spec/design/randomness_explicit_keys.md`
/// §4, rules V1 to V4; V5, the batched-draw shapes, is an operand rule).
///
/// - V1: a key is produced by a key operation, a join or a key-typed `Load`,
///   and a key may be a graph root, directly or through the `Store` that
///   names it; a key `Store` is the key it stores.
/// - V2: a key's uses are exactly one draw, one `FoldIn`, one `SplitN`, one
///   `Drop`, one slot of one join, or one root, or at most one `Split` of
///   each branch.
///   Every `Load` of one parameter of one declaration is one key.
/// - V3: two uses of one key, by draws, key operations and join slots alike,
///   other than one `Split` of each branch, may share it only when each
///   consumes it under an activation and the two activations are
///   structurally exclusive: one implies `X` and the other `Not(X)`, or
///   either implies `false`. A draw or key operation consumes its key under
///   its own activation ([`KeyGraph::activation`]). A join consumes input 0
///   under input 2 and input 1 under input 3, and those are its own
///   activation conjoined with the two arms of one branch ([`join_arms`]).
///   Rule S: a key derived under such sharing is consumed only under that
///   activation, or by the join of its branch ([`verify_confinement`]).
/// - V4: a key reaching any other operation or slot is rejected: the slots
///   are those of the key allow-list's graph admissions
///   ([`SlotRead::admission`]). Replays read their forward draw's key
///   without consuming it. An extent slot ([`slot_read`]) reads its key
///   tensor's extent, which is not a use; so does a node that names a key in
///   its shape dependencies, which [`KeyGraph`] does not show.
///
/// Each message names a key by [`KeyGraph::describe_key`] and a node by
/// [`KeyGraph::describe_node`].
pub fn verify_key_rules(graph: &impl KeyGraph, errors: &mut Vec<String>) {
    let is_key = |node: usize| graph.dtype(node) == Some(Prim::Key);
    // The key a node's value is: a key `Store` is the key it names, every
    // key `Load` of one parameter is the first such `Load`, and any other
    // key is its own node. A parameter is its declaration and its name: two
    // declarations' `k` are two keys.
    let identity = |node: usize| {
        let mut node = node;
        while is_key(node) && graph.role(node) == KeyRole::Store {
            match graph.input(node, 0) {
                Some(stored) if stored < node => node = stored,
                _ => break,
            }
        }
        match graph.load_name(node) {
            Some(name) if is_key(node) => (0..node)
                .find(|earlier| {
                    is_key(*earlier)
                        && graph.load_name(*earlier) == Some(name)
                        && graph.same_declaration(*earlier, node)
                })
                .unwrap_or(node),
            _ => node,
        }
    };
    let key = |node: usize| graph.describe_key(identity(node));
    let mut consumers = vec![Vec::<KeyUse>::new(); graph.node_count()];
    for node in 0..graph.node_count() {
        let role = graph.role(node);
        if role.produces_key() && !is_key(node) {
            errors.push(format!(
                "{} is a key operation that does not produce a key",
                graph.describe_node(node)
            ));
        }
        // A `Store` is the key it names and a `Drop` mirrors the type of the
        // key it consumes; neither is a new key.
        if is_key(node)
            && !role.produces_key()
            && !matches!(role, KeyRole::Load | KeyRole::Store | KeyRole::Drop)
        {
            errors.push(format!(
                "{} produces a key, but only a key operation, a join or a Load produces one",
                graph.describe_node(node)
            ));
        }
        let inputs = (0..).map_while(|slot| graph.input(node, slot));
        for (slot, input) in inputs.enumerate().filter(|(_, input)| is_key(*input)) {
            let read = graph.slot_read(node, slot);
            if read.admission(role, slot).is_none() {
                errors.push(format!(
                    "{} reaches input {slot} of {}; only a key operation, a join or a random primitive consumes a key",
                    key(input),
                    graph.describe_node(node)
                ));
                continue;
            }
            if read == SlotRead::Value && role.consumes() {
                consumers[identity(input)].push(KeyUse { node, slot });
            }
        }
        if role == KeyRole::KeySelect {
            let arms = join_decompositions(graph, node);
            if arms.is_empty() {
                errors.push(format!(
                    "{} joins keys under activations that are not the two arms of one branch",
                    graph.describe_node(node)
                ));
            } else if !arms
                .iter()
                .any(|enclosing| join_encloses_own(graph, node, enclosing))
            {
                errors.push(format!(
                    "{} joins keys under two arms that are not its own activation conjoined with a condition and with its negation",
                    graph.describe_node(node)
                ));
            }
        }
    }
    let mut exclusive = vec![false; graph.node_count()];
    for (shared, uses) in consumers.iter().enumerate() {
        if uses.len() > 1 {
            verify_shared_key(graph, &key(shared), uses, &mut exclusive, errors);
        }
    }
    // A root is a use: returning a key hands it to the caller.
    let mut rooted = vec![0usize; graph.node_count()];
    for root in graph.roots().filter(|root| is_key(*root)) {
        let Some(uses) = rooted.get_mut(identity(root)) else {
            continue;
        };
        *uses += 1;
        if *uses == 2 {
            errors.push(format!("{} is a graph root twice", key(root)));
        }
    }
    for (shared, uses) in rooted.iter().enumerate() {
        if let (true, Some(consumer)) = (*uses > 0, consumers[shared].first()) {
            errors.push(format!(
                "{} is a graph root and is also consumed by {}",
                key(shared),
                graph.describe_node(consumer.node)
            ));
        }
    }
    verify_confinement(graph, identity, &consumers, &exclusive, &rooted, errors);
    // Replay reads: each must read a key that a forward draw of the matching
    // kind consumes under the same rate or template type and under the same
    // activation, its own.
    for node in 0..graph.node_count() {
        let forward_role = match graph.role(node) {
            KeyRole::DropoutReplay => KeyRole::Dropout,
            KeyRole::UniformBoundAdjoint => KeyRole::UniformLike,
            _ => continue,
        };
        let Some(read) = graph.input(node, 2) else {
            continue;
        };
        let forwards = consumers
            .get(identity(read))
            .into_iter()
            .flatten()
            .map(|forward| forward.node)
            .filter(|forward| graph.role(*forward).is_draw())
            .collect::<Vec<_>>();
        if forwards.is_empty() {
            errors.push(format!(
                "{} reads {}, which no forward random primitive consumes",
                graph.describe_node(node),
                key(read)
            ));
            continue;
        }
        let matches = |forward: usize| {
            graph.role(forward) == forward_role
                && match forward_role {
                    KeyRole::Dropout => {
                        graph.input(node, 1) == graph.input(forward, 1)
                            && graph.same_type(node, forward)
                    }
                    _ => graph
                        .input(node, 0)
                        .is_some_and(|template| graph.same_type(template, forward)),
                }
                && graph.activation(node) == graph.activation(forward)
        };
        if !forwards.iter().any(|forward| matches(*forward)) {
            errors.push(format!(
                "{} changes the mask contract of {}",
                graph.describe_node(node),
                graph.describe_node(forwards[0])
            ));
        }
    }
}

/// One consuming read of a key: input `slot` of `node`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KeyUse {
    node: usize,
    slot: usize,
}

/// Rules V2 and V3 for a key `key` names that several uses consume, pair by
/// pair: a `Left` and a `Right` split are the two halves of one
/// `split_key`, and any other pair, whatever the consumers' kinds and
/// whichever of a join's slots, must consume it under structurally exclusive
/// activations. Each key operation that shares the key only through
/// exclusive activations is marked in `exclusive`, for
/// [`verify_confinement`].
fn verify_shared_key(
    graph: &impl KeyGraph,
    key: &str,
    uses: &[KeyUse],
    exclusive: &mut [bool],
    errors: &mut Vec<String>,
) {
    for (index, &left) in uses.iter().enumerate() {
        for &right in &uses[index + 1..] {
            let (left_role, right_role) = (graph.role(left.node), graph.role(right.node));
            if let (KeyRole::Split { branch: first }, KeyRole::Split { branch: second }) =
                (left_role, right_role)
                && first != second
            {
                continue;
            }
            let (first, second) = (
                graph.describe_node(left.node),
                graph.describe_node(right.node),
            );
            match (consuming_activation(graph, left), consuming_activation(graph, right)) {
                (Some(left_active), Some(right_active))
                    if activations_exclusive(graph, left_active, right_active) =>
                {
                    exclusive[left.node] = true;
                    exclusive[right.node] = true;
                }
                (Some(_), Some(_)) => errors.push(format!(
                    "{key} is consumed twice, by {first} and {second}, whose activations are not exclusive"
                )),
                _ => errors.push(match left_role {
                    KeyRole::Split { .. } if left_role == right_role => {
                        format!("{key} is split twice for one branch, by {first} and {second}")
                    }
                    _ => format!("{key} is consumed twice, by {first} and {second}"),
                }),
            }
        }
    }
}

/// The activation a use consumes its key under: a join's slot 0 under its
/// input 2 and slot 1 under its input 3, and a draw or key operation under
/// its own activation, if it has one.
fn consuming_activation(graph: &impl KeyGraph, key_use: KeyUse) -> Option<usize> {
    match graph.role(key_use.node) {
        KeyRole::KeySelect => graph.input(key_use.node, key_use.slot + 2),
        _ => graph.activation(key_use.node),
    }
}

/// The nodes a conjunction implies at its top level: through every `And`,
/// both conjuncts, transitively, stopping at `atom` and at every node that
/// is not an `And`.
fn conjunction_atoms(graph: &impl KeyGraph, activation: usize, atom: Option<usize>) -> Vec<usize> {
    let mut atoms = Vec::new();
    let mut seen = Vec::new();
    let mut stack = vec![activation];
    while let Some(node) = stack.pop() {
        if seen.contains(&node) {
            continue;
        }
        seen.push(node);
        if Some(node) != atom && graph.role(node) == KeyRole::And {
            stack.extend((0..2).filter_map(|slot| graph.input(node, slot)));
        } else {
            atoms.push(node);
        }
    }
    atoms
}

/// A join's two activations as the two arms of one branch: the then
/// activation is a conjunction `S And X` and the else activation `S And
/// Not(X)` (either way round, `S` possibly empty), for one node `X` and one
/// set `S` of conjuncts, or `X` and `Not(X)` are the constants `true` and
/// `false` that folding leaves of them. Then where the enclosing activation
/// `S` holds exactly one arm does, which makes the join's result a key under
/// `S`. Returns the atoms of every such `S`, sorted, without repeats; empty
/// when the activations are not two arms of one branch.
fn join_decompositions(graph: &impl KeyGraph, node: usize) -> Vec<Vec<usize>> {
    let mut decompositions = Vec::<Vec<usize>>::new();
    let (Some(then_active), Some(else_active)) = (graph.input(node, 2), graph.input(node, 3))
    else {
        return decompositions;
    };
    let then_conjuncts = activation_conjuncts(graph, then_active);
    let else_conjuncts = activation_conjuncts(graph, else_active);
    let negates = |node: usize, other: usize| {
        graph.role(node) == KeyRole::Not && graph.input(node, 0) == Some(other)
    };
    let constants = |node: usize, other: usize| {
        graph.role(node) == KeyRole::ConstFalse && graph.role(other) == KeyRole::ConstTrue
    };
    let complementary =
        |a: usize, b: usize| negates(a, b) || negates(b, a) || constants(a, b) || constants(b, a);
    for &then_atom in &then_conjuncts {
        for &else_atom in &else_conjuncts {
            if !complementary(then_atom, else_atom) {
                continue;
            }
            let mut enclosing = conjunction_atoms(graph, then_active, Some(then_atom));
            enclosing.retain(|atom| *atom != then_atom);
            enclosing.sort_unstable();
            let mut other = conjunction_atoms(graph, else_active, Some(else_atom));
            other.retain(|atom| *atom != else_atom);
            other.sort_unstable();
            if enclosing == other && !decompositions.contains(&enclosing) {
                decompositions.push(enclosing);
            }
        }
    }
    decompositions
}

/// Whether `enclosing`, the atoms of an enclosing activation that
/// [`join_decompositions`] finds for the join `node`, is the join's own
/// activation ([`KeyGraph::activation`]): the same atoms through every
/// `And`, and none when it has no activation. The constant `true`, which
/// folding may leave of a conjunct, implies nothing and is set aside on both
/// sides. So the two slot activations are the join's own activation
/// conjoined with a condition and with its negation, and exactly one of
/// them holds wherever the join's own does (spec/10 §3.2).
fn join_encloses_own(graph: &impl KeyGraph, node: usize, enclosing: &[usize]) -> bool {
    let informative = |atoms: &mut Vec<usize>| {
        atoms.retain(|atom| graph.role(*atom) != KeyRole::ConstTrue);
        atoms.sort_unstable();
    };
    let mut own = graph
        .activation(node)
        .map_or_else(Vec::new, |active| conjunction_atoms(graph, active, None));
    informative(&mut own);
    let mut enclosing = enclosing.to_vec();
    informative(&mut enclosing);
    own == enclosing
}

/// The enclosing activation of the join `node` ([`join_decompositions`]):
/// the one that is its own activation where one is, and otherwise the first,
/// or `None` when its activations are not two arms of one branch.
fn join_arms(graph: &impl KeyGraph, node: usize) -> Option<Vec<usize>> {
    let decompositions = join_decompositions(graph, node);
    decompositions
        .iter()
        .find(|enclosing| join_encloses_own(graph, node, enclosing))
        .or(decompositions.first())
        .cloned()
}

/// Rule S. A key operation that shares its key with an exclusive consumer
/// may derive what that consumer derives: two exclusive `Split{Left}`s
/// derive one key, and so do a `FoldIn` of `n` and row `n` of a `SplitN`.
/// Such keys stay apart only while each is used under the activation it was
/// derived under. So a key derived by an operation that `exclusive` marks,
/// and every key derived from it in turn, is consumed only under that
/// activation: by consumers whose activations imply it (their `And`
/// conjuncts contain it) or contain the constant `false`, a join's slot
/// among them, and it is never a root.
///
/// A join of a branch ([`join_arms`]) consumes each slot under its arm, so
/// its result carries only the requirements its arms share, which are the
/// enclosing activation's: those its slots' keys carry that both arms imply,
/// and every atom of the enclosing activation itself. On a path where that
/// holds exactly one arm does, and the result is that arm's key, used where
/// every requirement of it holds; the enclosing requirement keeps the result
/// out of every path where no arm holds. `identity` names the key a node's
/// value is, as in [`verify_key_rules`].
fn verify_confinement(
    graph: &impl KeyGraph,
    identity: impl Fn(usize) -> usize,
    consumers: &[Vec<KeyUse>],
    exclusive: &[bool],
    rooted: &[usize],
    errors: &mut Vec<String>,
) {
    // The activations each key must be used under. Nodes precede their
    // consumers, so a parent's entry is complete before its children read it.
    let mut required = vec![Vec::<usize>::new(); graph.node_count()];
    let inherited = |required: &[Vec<usize>], slot: usize, node: usize| {
        graph
            .input(node, slot)
            .map(&identity)
            .and_then(|parent| required.get(parent))
            .map_or_else(Vec::new, Clone::clone)
    };
    for node in 0..graph.node_count() {
        let role = graph.role(node);
        if role.derives() {
            // A key operation without a parent key in this graph is the
            // operand rules' error, and derives nothing these rules follow.
            let mut under = inherited(&required, 0, node);
            if exclusive[node]
                && let Some(active) = graph.activation(node)
                && !under.contains(&active)
            {
                under.push(active);
            }
            required[node] = under;
        } else if role == KeyRole::KeySelect
            && let Some(enclosing) = join_arms(graph, node)
        {
            let (Some(then_active), Some(else_active)) =
                (graph.input(node, 2), graph.input(node, 3))
            else {
                continue;
            };
            let then_conjuncts = activation_conjuncts(graph, then_active);
            let else_conjuncts = activation_conjuncts(graph, else_active);
            let mut under = enclosing;
            for active in inherited(&required, 0, node)
                .into_iter()
                .chain(inherited(&required, 1, node))
            {
                if then_conjuncts.contains(&active)
                    && else_conjuncts.contains(&active)
                    && !under.contains(&active)
                {
                    under.push(active);
                }
            }
            required[node] = under;
        }
    }
    for (key, under) in required.iter().enumerate() {
        let Some(&first) = under.first() else {
            continue;
        };
        let described = graph.describe_key(key);
        if rooted[key] > 0 {
            errors.push(format!(
                "{described} is confined to the activation at node {first}, and is a graph root; a key derived in a branch arm leaves it only through that branch's join"
            ));
        }
        for &key_use in &consumers[key] {
            let consumer = graph.describe_node(key_use.node);
            let Some(active) = consuming_activation(graph, key_use) else {
                errors.push(format!(
                    "{described} is confined to the activation at node {first}, but {consumer} consumes it outside that activation; a key derived in a branch arm leaves it only through that branch's join"
                ));
                continue;
            };
            let conjuncts = activation_conjuncts(graph, active);
            if conjuncts
                .iter()
                .any(|node| graph.role(*node) == KeyRole::ConstFalse)
            {
                continue;
            }
            if let Some(missing) = under.iter().find(|active| !conjuncts.contains(active)) {
                errors.push(format!(
                    "{described} is confined to the activation at node {missing}, but {consumer} consumes it outside that activation; a key derived in a branch arm leaves it only through that branch's join"
                ));
            }
        }
    }
}

#[allow(clippy::collapsible_match)]
fn verify_with_dangling_policy(dag: &Dag, reject_dangling: bool) -> Vec<String> {
    let mut errors = Vec::new();
    let mut consumers = vec![0usize; dag.len()];
    let seeds = dag.trap_seeds();
    // chelis#2413: a `Load` reads its declaration's parameter, so two
    // declarations may each name a parameter `x` with its own type.
    let mut load_types =
        chelis_unord::UnordMap::<(crate::dag::DeclId, String), &crate::dag::TensorType>::new();
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
        // spec/10 section 3.2: a node's activation is an earlier Bool node of
        // the graph, and the node reads it to decide whether it checks.
        if let Some(activation) = node.owner.activation {
            match dag.get(activation) {
                Some(source) if activation < node.id => {
                    consumers[activation.0] += 1;
                    if source.output_type.precision != chelis_types::types::Prim::Bool {
                        errors.push(format!(
                            "{}'s activation {} is not a Bool",
                            dag.describe_node(node.id),
                            dag.describe_node(activation)
                        ));
                    }
                }
                _ => errors.push(format!(
                    "{}'s activation {} is not an earlier node of the graph",
                    dag.describe_node(node.id),
                    activation.0
                )),
            }
        }

        if let RiscOp::Load { name } = &node.op {
            let prev_ty = *load_types
                .entry((node.owner.decl, name.as_str().to_string()))
                .or_insert(&node.output_type);
            if prev_ty != &node.output_type {
                errors.push(format!(
                    "{} has inconsistent tensor types: {:?} vs {:?}",
                    dag.describe_node(node.id),
                    prev_ty,
                    node.output_type
                ));
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
            RiscOp::Iota => {
                if node
                    .owner
                    .activation
                    .and_then(|id| dag.get(id))
                    .is_some_and(|active| !active.output_type.dims.is_empty())
                {
                    errors.push(format!(
                        "iota at node {} requires scalar activation",
                        node.id.0
                    ));
                }
                if arity != 2 {
                    errors.push(format!("iota at node {} requires two inputs", node.id.0));
                }
                for &input in &node.inputs {
                    if let Some(input) = dag.get(input)
                        && (input.output_type.precision != Prim::Int64
                            || !input.output_type.dims.is_empty())
                    {
                        errors.push(format!(
                            "iota at node {} requires rank-zero i64 endpoints",
                            node.id.0
                        ));
                    }
                }
                if node.output_type.precision != Prim::Int64 || node.output_type.dims.len() != 1 {
                    errors.push(format!(
                        "iota at node {} requires rank-one i64 output",
                        node.id.0
                    ));
                }
            }

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
            // chelis#1464 / [05-OP-68]: the guard's own contract, checked
            // here so a malformed abort node cannot reach a backend. An
            // empty message is rejected because the atom forbids a
            // synthesized or defaulted one.
            RiscOp::GuardedFail { message, .. } => {
                if arity != 2 {
                    errors.push(format!(
                        "guarded_fail at node {} has {} inputs (expected 2)",
                        node.id.0, arity
                    ));
                } else if let (Some(condition), Some(_)) =
                    (dag.get(node.inputs[0]), dag.get(node.inputs[1]))
                {
                    if condition.output_type.precision != Prim::Bool {
                        errors.push(format!(
                            "guarded_fail at node {} condition must be Bool",
                            node.id.0
                        ));
                    }
                    if condition.output_type.dims.len() > 1 {
                        errors.push(format!(
                            "guarded_fail at node {} condition must be rank-0, or rank-1 \
                             when mapped over a batch axis",
                            node.id.0
                        ));
                    }
                    // Structural equality, deliberately NOT the semantic
                    // helper the sibling arms use. [05-OP-68] says the result
                    // has EXACTLY the fallback's shape and dtype, and
                    // `guarded_fail_value` builds it from that node's own
                    // type, so exact equality is both the rule and the
                    // construction.
                    //
                    // The semantic helper is vacuous here: `axis_sources`
                    // attributes this op's output axes to slot 1, so the
                    // guard and its fallback always share an axis origin and
                    // the comparison succeeds for any dims at all. It passed
                    // a `[3]` fallback under a `[4]` result until a negative
                    // test caught it.
                    let fallback_type = dag
                        .get(node.inputs[1])
                        .map(|fallback| fallback.output_type.clone());
                    if fallback_type.as_ref() != Some(&node.output_type) {
                        errors.push(format!(
                            "guarded_fail at node {} must have exactly its fallback's type",
                            node.id.0
                        ));
                    }
                }
                if message.is_empty() {
                    errors.push(format!(
                        "guarded_fail at node {} has an empty message",
                        node.id.0
                    ));
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
                    // spec/04 §1.1: `where` names no `key`, so its branches
                    // range over the data element dtypes; keys select through
                    // activations instead.
                    if !then_value.output_type.precision.is_data_element_dtype() {
                        errors.push(format!(
                            "where at node {} branches must use an active data element dtype",
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
            // The random and key operations' operand rules are
            // `verify_random_operands`, which the wire decoder shares; only
            // the IR's own bound carrier is checked here.
            RiscOp::SplitN { count } => check_bound_source(
                dag,
                node,
                count,
                &format!("split_keys at node {}", node.id.0),
                &mut errors,
            ),
            RiscOp::UniformLike
            | RiscOp::Dropout
            | RiscOp::DropoutReplay
            | RiscOp::UniformBoundAdjoint { .. }
            | RiscOp::KeyFromSeed
            | RiscOp::Split { .. }
            | RiscOp::FoldIn
            | RiscOp::KeySelect => {}
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
                        // [05-OP-71]: a split's count axis produces its extent
                        // as an expansion's size does, and the split checks
                        // every extent its type declares before any key
                        // exists, in both lanes.
                        || matches!(node.op, RiscOp::SplitN { .. })
                            && axis + 1 == node.output_type.dims.len()
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
            // Its witness checks the unit requirement where the refinement
            // runs, never on a path the refinement is not on.
            if let Some(witness) = node.inputs.get(1).and_then(|witness| dag.get(*witness))
                && witness.owner.activation != node.owner.activation
            {
                errors.push(format!(
                    "checked unit axis at node {} is checked under activation {:?}, but its witness {} under {:?}",
                    node.id.0,
                    node.owner.activation.map(|activation| activation.0),
                    witness.id.0,
                    witness.owner.activation.map(|activation| activation.0)
                ));
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
            // chelis#2368 / chelis#2440: an observable root may legitimately
            // have no consumer; firing, or trapping, is the point. Requiring
            // one would force it back into the value graph, which is exactly
            // the reachability criterion that let DCE sweep it. This is the
            // verifier agreeing with the DCE seed ([`crate::dag::TrapSeeds::is_observable_root`],
            // `spec/06-transformations.md` §5.2) rather than carving an
            // exception out of it.
            && !matches!(node.op, RiscOp::Store { .. } | RiscOp::Drop)
            && !seeds.is_observable_root(node)
        {
            errors.push(format!(
                "node {} is dangling: it has no consumers and is not a DAG root",
                node.id.0
            ));
        }
    }

    verify_random_operands(dag, &mut errors);
    verify_key_rules(dag, &mut errors);
    verify_declaration_sharing(dag, &mut errors);

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

/// #2413 (spec/10 section 3.2): a node reads a node of another declaration
/// only when none of that declaration's nodes can trap
/// ([`crate::dag::TrapSeeds::is_observable_root`]). A reference to a value declaration whose
/// initializer may trap lowers the initializer again at the reference site,
/// under that site's owner, so its checks run exactly where and when the
/// reference is reached. A shared node of such a declaration would instead
/// run its declaration's checks wherever any reader is evaluated, in no
/// reader's activation.
pub fn verify_declaration_sharing(dag: &Dag, errors: &mut Vec<String>) {
    let mut may_trap = vec![false; dag.declarations().len()];
    let seeds = dag.trap_seeds();
    for node in dag.nodes() {
        if seeds.is_observable_root(node)
            && let Some(flag) = may_trap.get_mut(node.owner.decl.0 as usize)
        {
            *flag = true;
        }
    }
    for node in dag.nodes() {
        for dependency in node.dependencies() {
            let Some(source) = dag.get(dependency) else {
                continue;
            };
            if source.owner.decl != node.owner.decl
                && may_trap
                    .get(source.owner.decl.0 as usize)
                    .copied()
                    .unwrap_or(false)
            {
                errors.push(format!(
                    "{} reads {}, but {} has a node that can trap: another declaration reads \
                     it only inlined at the reference, never shared",
                    dag.describe_node(node.id),
                    dag.describe_node(dependency),
                    dag.describe_declaration(source.owner.decl)
                ));
            }
        }
    }
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
    use crate::dag::{DeclId, ExtentWitnessSite, NodeId, RiscOp, RtAxis, TensorType};

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    fn tensor_ty(dims: &[usize], precision: Prim) -> TensorType {
        TensorType {
            dims: dims.iter().copied().map(DimInfo::Lit).collect(),
            precision,
        }
    }

    /// A graph with one group of nodes per entry, each group built by
    /// `group` in the named declaration: a repeated name is one declaration.
    fn declared_groups(declarations: &[&str], mut group: impl FnMut(&mut Dag, DeclId)) -> Dag {
        let mut dag = Dag::new();
        for name in declarations {
            let existing = dag
                .declarations()
                .iter()
                .position(|entry| entry.name == *name);
            let decl = match existing {
                Some(index) => DeclId(u32::try_from(index).unwrap()),
                None => dag.declare(*name),
            };
            group(&mut dag, decl);
        }
        dag
    }

    /// chelis#2413 B2: a parameter is its declaration and its name. Two
    /// declarations that each read a key parameter `k` hold two keys, and
    /// one declaration's two `Load`s of `k` are one key.
    ///
    /// Evidentiary status: REGRESSION TEST. At ff8957386 the verifier
    /// identified a key `Load` by its name alone, so the first row failed.
    #[test]
    fn a_key_parameter_is_its_declaration_and_its_name() {
        let errors = |declarations: &[&str]| {
            let dag = declared_groups(declarations, |dag, decl| {
                let k = dag.add_node(
                    decl,
                    RiscOp::Load { name: "k".into() },
                    vec![],
                    tensor_ty(&[], Prim::Key),
                    None,
                );
                let x = dag.add_node(
                    decl,
                    RiscOp::Load { name: "x".into() },
                    vec![],
                    tensor_ty(&[2], Prim::F32),
                    None,
                );
                let rate = dag.add_node(
                    decl,
                    RiscOp::synth_const(Prim::F32, 0.5),
                    vec![],
                    scalar_f32(),
                    None,
                );
                let drawn = dag.add_node(
                    decl,
                    RiscOp::Dropout,
                    vec![x, rate, k],
                    tensor_ty(&[2], Prim::F32),
                    None,
                );
                dag.add_root(drawn);
            });
            let mut errors = Vec::new();
            verify_key_rules(&dag, &mut errors);
            errors
        };
        assert_eq!(errors(&["a", "b"]), Vec::<String>::new());
        assert!(
            errors(&["a", "a"])
                .iter()
                .any(|error| error.contains("is consumed twice"))
        );
    }

    /// [04-LIN-9], spec/10 §3.2: a key tensor's extent is not key material.
    /// An `ExtentWitness` or a `Shape` reads a key tensor without a use, so
    /// the key verifies beside one draw or as a root, and a second draw is
    /// still its second use.
    ///
    /// Evidentiary status: REGRESSION TEST. At `096daea8c` every row
    /// reported "key `ks` of `f` reaches input 0 of node 1".
    #[test]
    fn an_extent_read_of_a_key_tensor_is_not_a_use() {
        let errors = |observer: RiscOp, draws: usize, rooted: bool| {
            let mut dag = Dag::new();
            let f = dag.declare("f");
            let ks = dag.add_node(
                f,
                RiscOp::Load { name: "ks".into() },
                vec![],
                tensor_ty(&[3], Prim::Key),
                None,
            );
            let extent = dag.add_node(f, observer, vec![ks], tensor_ty(&[], Prim::Int64), None);
            dag.add_root(extent);
            let x = dag.add_node(
                f,
                RiscOp::Load { name: "x".into() },
                vec![],
                tensor_ty(&[3], Prim::F32),
                None,
            );
            let rate = dag.add_node(
                f,
                RiscOp::synth_const(Prim::F32, 0.5),
                vec![],
                scalar_f32(),
                None,
            );
            for _ in 0..draws {
                let drawn = dag.add_node(
                    f,
                    RiscOp::Dropout,
                    vec![x, rate, ks],
                    tensor_ty(&[3], Prim::F32),
                    None,
                );
                dag.add_root(drawn);
            }
            if rooted {
                dag.add_root(ks);
            }
            let mut errors = Vec::new();
            verify_key_rules(&dag, &mut errors);
            errors
        };
        for observer in [
            RiscOp::ExtentWitness {
                site: ExtentWitnessSite::Caller,
                parameter: "ks".into(),
                axis: RtAxis::Lit(0),
                requirements: Vec::new(),
                claims: Vec::new(),
            },
            RiscOp::Shape { axis: 0 },
        ] {
            assert_eq!(errors(observer.clone(), 1, false), Vec::<String>::new());
            assert_eq!(errors(observer.clone(), 0, true), Vec::<String>::new());
            let twice = errors(observer, 2, false);
            assert!(
                twice
                    .iter()
                    .any(|error| error.contains("is consumed twice")),
                "{twice:?}"
            );
        }
    }

    /// A parameter's type is its declaration's: two declarations may each
    /// read an `x` of a different type, and one declaration's two `Load`s of
    /// `x` must agree. The error names the parameter and its declaration.
    ///
    /// Evidentiary status: REGRESSION TEST. At ff8957386 the first row
    /// reported "load 'x' has inconsistent tensor types".
    #[test]
    fn a_parameter_type_is_its_declarations() {
        let errors = |declarations: &[&str]| {
            let mut shapes = [vec![4], vec![2, 2]].into_iter().cycle();
            let dag = declared_groups(declarations, |dag, decl| {
                let x = dag.add_node(
                    decl,
                    RiscOp::Load { name: "x".into() },
                    vec![],
                    tensor_ty(&shapes.next().unwrap(), Prim::F32),
                    None,
                );
                dag.add_root(x);
            });
            verify(&dag)
                .into_iter()
                .filter(|error| error.contains("inconsistent tensor types"))
                .collect::<Vec<_>>()
        };
        assert_eq!(errors(&["a", "b"]), Vec::<String>::new());
        let same = errors(&["keep", "keep"]);
        assert_eq!(same.len(), 1, "{same:?}");
        assert!(
            same[0].starts_with("parameter `x` of `keep` has inconsistent tensor types"),
            "{same:?}"
        );
    }

    /// #2413 (spec/10 section 3.2): a node reads another declaration's node
    /// only when none of that declaration's nodes can trap. `s` reads `d`'s
    /// value; the verifier accepts it while `d` is a total `mul` and rejects
    /// it once `d` holds an integer `cast`, which can trap, naming both
    /// declarations. A reference to such a value is its initializer inlined
    /// at the reference, so lowering never builds the second graph.
    ///
    /// Evidentiary status: REGRESSION TEST. At ad0abe6a9 no rule read a
    /// node's declaration against its readers', so the trapping row
    /// verified.
    #[test]
    fn a_node_that_can_trap_is_never_shared_with_another_declaration() {
        let sharing = |trapping: bool| {
            let mut dag = Dag::new();
            let d = dag.declare_value("d");
            let s = dag.declare("s");
            let x = dag.add_node(
                d,
                RiscOp::Load { name: "x".into() },
                vec![],
                tensor_ty(&[4], Prim::F32),
                None,
            );
            let value = if trapping {
                dag.add_node(
                    d,
                    RiscOp::Cast {
                        new_precision: Prim::Int32,
                    },
                    vec![x],
                    tensor_ty(&[4], Prim::Int32),
                    None,
                )
            } else {
                dag.add_node(d, RiscOp::Mul, vec![x, x], tensor_ty(&[4], Prim::F32), None)
            };
            let read = dag.add_node(
                s,
                RiscOp::Copy,
                vec![value],
                dag.get(value).unwrap().output_type.clone(),
                None,
            );
            dag.add_root(read);
            verify(&dag)
                .into_iter()
                .filter(|error| error.contains("has a node that can trap"))
                .collect::<Vec<_>>()
        };
        assert_eq!(sharing(false), Vec::<String>::new());
        let shared = sharing(true);
        assert_eq!(shared.len(), 1, "{shared:?}");
        assert!(
            shared[0].starts_with("node 2 of `s` reads node 1 of `d`, but `d` has a node"),
            "{shared:?}"
        );
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

    /// chelis#1464 / [05-OP-68]: the guard's own contract. Every rule the
    /// atom states gets a negative control here, because a malformed abort
    /// node reaching a backend is how the placeholder class came back.
    mod guarded_fail {
        use super::*;

        fn bool_scalar() -> TensorType {
            TensorType {
                dims: Vec::new(),
                precision: Prim::Bool,
            }
        }

        /// A well-formed guard: rank-0 bool condition, fallback carrying the
        /// result type, non-empty message.
        fn well_formed() -> Dag {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let cond = dag.add_node(
                decl,
                RiscOp::synth_const(Prim::Bool, 1.0),
                vec![],
                bool_scalar(),
                None,
            );
            let fallback = dag.add_node(
                decl,
                RiscOp::synth_const(Prim::F32, 2.0),
                vec![],
                scalar_f32(),
                None,
            );
            dag.add_node(
                decl,
                RiscOp::GuardedFail {
                    message: "boom".to_string(),
                    trap_on_true: true,
                },
                vec![cond, fallback],
                scalar_f32(),
                None,
            );
            dag
        }

        #[test]
        fn a_well_formed_guard_verifies() {
            assert!(
                verify(&well_formed()).is_empty(),
                "the positive control must be clean, or the negatives below prove nothing"
            );
        }

        #[test]
        fn an_empty_message_is_rejected() {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let cond = dag.add_node(
                decl,
                RiscOp::synth_const(Prim::Bool, 1.0),
                vec![],
                bool_scalar(),
                None,
            );
            let fallback = dag.add_node(
                decl,
                RiscOp::synth_const(Prim::F32, 2.0),
                vec![],
                scalar_f32(),
                None,
            );
            dag.add_node(
                decl,
                RiscOp::GuardedFail {
                    message: String::new(),
                    trap_on_true: true,
                },
                vec![cond, fallback],
                scalar_f32(),
                None,
            );
            let errors = verify(&dag);
            assert!(
                errors.iter().any(|error| error.contains("empty message")),
                "[05-OP-68] forbids a synthesized or defaulted message; got: {errors:?}"
            );
        }

        #[test]
        fn a_non_bool_condition_is_rejected() {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let cond = dag.add_node(
                decl,
                RiscOp::synth_const(Prim::F32, 1.0),
                vec![],
                scalar_f32(),
                None,
            );
            let fallback = dag.add_node(
                decl,
                RiscOp::synth_const(Prim::F32, 2.0),
                vec![],
                scalar_f32(),
                None,
            );
            dag.add_node(
                decl,
                RiscOp::GuardedFail {
                    message: "boom".to_string(),
                    trap_on_true: true,
                },
                vec![cond, fallback],
                scalar_f32(),
                None,
            );
            let errors = verify(&dag);
            assert!(
                errors
                    .iter()
                    .any(|error| error.contains("condition must be Bool")),
                "the atom admits bool alone for the condition; got: {errors:?}"
            );
        }

        #[test]
        fn a_condition_above_rank_one_is_rejected() {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let cond = dag.add_node(
                decl,
                RiscOp::synth_const(Prim::Bool, 1.0),
                vec![],
                tensor_ty(&[2, 2], Prim::Bool),
                None,
            );
            let fallback = dag.add_node(
                decl,
                RiscOp::synth_const(Prim::F32, 2.0),
                vec![],
                scalar_f32(),
                None,
            );
            dag.add_node(
                decl,
                RiscOp::GuardedFail {
                    message: "boom".to_string(),
                    trap_on_true: true,
                },
                vec![cond, fallback],
                scalar_f32(),
                None,
            );
            let errors = verify(&dag);
            assert!(
                errors.iter().any(|error| error.contains("must be rank-0")),
                "rank-0, or rank-1 when mapped over a batch axis, is the admitted set; \
                 got: {errors:?}"
            );
        }

        #[test]
        fn a_batched_rank_one_condition_is_admitted() {
            // The negative above must not over-reject the `vmap` form.
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let cond = dag.add_node(
                decl,
                RiscOp::synth_const(Prim::Bool, 1.0),
                vec![],
                tensor_ty(&[2], Prim::Bool),
                None,
            );
            let fallback = dag.add_node(
                decl,
                RiscOp::synth_const(Prim::F32, 2.0),
                vec![],
                tensor_ty(&[2], Prim::F32),
                None,
            );
            dag.add_node(
                decl,
                RiscOp::GuardedFail {
                    message: "boom".to_string(),
                    trap_on_true: true,
                },
                vec![cond, fallback],
                tensor_ty(&[2], Prim::F32),
                None,
            );
            assert!(
                !verify(&dag)
                    .iter()
                    .any(|error| error.contains("guarded_fail")),
                "a mapped rank-1 condition is admitted by [05-OP-68]"
            );
        }

        #[test]
        fn a_wrong_arity_is_rejected() {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let cond = dag.add_node(
                decl,
                RiscOp::synth_const(Prim::Bool, 1.0),
                vec![],
                bool_scalar(),
                None,
            );
            dag.add_node(
                decl,
                RiscOp::GuardedFail {
                    message: "boom".to_string(),
                    trap_on_true: true,
                },
                vec![cond],
                scalar_f32(),
                None,
            );
            let errors = verify(&dag);
            assert!(
                errors.iter().any(|error| error.contains("expected 2")),
                "the guard takes exactly (condition, fallback); got: {errors:?}"
            );
        }

        #[test]
        fn a_result_type_differing_from_the_fallback_is_rejected() {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let cond = dag.add_node(
                decl,
                RiscOp::synth_const(Prim::Bool, 1.0),
                vec![],
                bool_scalar(),
                None,
            );
            let fallback = dag.add_node(
                decl,
                RiscOp::synth_const(Prim::F32, 2.0),
                vec![],
                tensor_ty(&[3], Prim::F32),
                None,
            );
            dag.add_node(
                decl,
                RiscOp::GuardedFail {
                    message: "boom".to_string(),
                    trap_on_true: true,
                },
                vec![cond, fallback],
                tensor_ty(&[4], Prim::F32),
                None,
            );
            let errors = verify(&dag);
            assert!(
                errors
                    .iter()
                    .any(|error| error.contains("must have exactly its fallback's type")),
                "the guard is type-transparent: the result IS the fallback; got: {errors:?}"
            );
        }
    }

    fn mapped_fixture() -> MappedFixture {
        let scalar_i64 = tensor_ty(&[], Prim::Int64);
        let scalar_f32 = tensor_ty(&[], Prim::F32);
        let vec2 = tensor_ty(&[2], Prim::F32);
        let mat22 = tensor_ty(&[2, 2], Prim::F32);

        let mut source = Dag::new();
        let source_decl = source.declare("test");
        let source_load = source.add_node(
            source_decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec2.clone(),
            None,
        );
        let source_witness = source.add_node(
            source_decl,
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
            source_decl,
            RiscOp::synth_const(Prim::F32, 1.0),
            vec![],
            scalar_f32,
            None,
        );
        source.add_shape_dep(source_forward, source_witness);
        source.add_root(source_forward);

        let mut mapped = Dag::new();
        let mapped_decl = mapped.declare("test");
        let mapped_load = mapped.add_node(
            mapped_decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            mat22.clone(),
            None,
        );
        let mapped_witness = mapped.add_node(
            mapped_decl,
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
            mapped_decl,
            RiscOp::synth_const_tensor(Prim::F32, vec![1.0, 1.0]),
            vec![],
            vec2,
            None,
        );
        mapped.add_shape_dep(mapped_forward, mapped_witness);
        mapped.add_root(mapped_forward);

        let mut spliced = Dag::new();
        let spliced_decl = spliced.declare("test");
        let actual = spliced.add_node(
            spliced_decl,
            RiscOp::Load {
                name: "actual".into(),
            },
            vec![],
            mat22.clone(),
            None,
        );
        let spliced_witness = spliced.add_node(
            spliced_decl,
            mapped.get(mapped_witness).unwrap().op.clone(),
            vec![actual],
            mapped.get(mapped_witness).unwrap().output_type.clone(),
            None,
        );
        let spliced_forward = spliced.add_node(
            spliced_decl,
            mapped.get(mapped_forward).unwrap().op.clone(),
            vec![],
            mapped.get(mapped_forward).unwrap().output_type.clone(),
            None,
        );
        spliced.add_shape_dep(spliced_forward, spliced_witness);
        let cotangent = spliced.add_node(
            spliced_decl,
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
        let decl = dag.declare("test");
        let input = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            tensor_ty(&[3], Prim::F32),
            None,
        );
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let vector = dag.add_node(
            decl,
            RiscOp::Load {
                name: "vector".into(),
            },
            vec![],
            tensor_ty(&[8], Prim::F32),
            None,
        );
        let matrix = dag.add_node(
            decl,
            RiscOp::Load {
                name: "matrix".into(),
            },
            vec![],
            tensor_ty(&[2, 4], Prim::F32),
            None,
        );
        let result = dag.add_node(
            decl,
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
        assert!(verify(&dag).is_empty());
    }

    #[test]
    fn constant_tensor_payload_cardinality_must_match_its_concrete_type() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Div, vec![a], scalar_f32(), None);
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
        let c = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Div, vec![a, b, c], scalar_f32(), None);
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
        dag.add_node(decl, RiscOp::Recip, vec![a, b], scalar_f32(), None);
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
        let decl = dag.declare("test");
        dag.add_node(decl, RiscOp::Recip, vec![], scalar_f32(), None);
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
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 6.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        let recip = dag.add_node(decl, RiscOp::Recip, vec![b], scalar_f32(), None);
        let div = dag.add_node(decl, RiscOp::Div, vec![a, recip], scalar_f32(), None);
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
            let decl = dag.declare("test");
            dag.add_node(
                decl,
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
        let decl = dag.declare("test");
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let drop = dag.add_node(decl, RiscOp::Drop, vec![a], scalar_f32(), None);
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
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let drop = dag.add_node(decl, RiscOp::Drop, vec![a], scalar_f32(), None);
        dag.add_node(decl, RiscOp::Neg, vec![drop], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(
            errs.iter().any(|e| e.contains("cannot be consumed")),
            "{errs:?}"
        );
    }

    #[test]
    fn copy_and_drop_require_one_input() {
        let mut copy_dag = Dag::new();
        let copy_dag_decl = copy_dag.declare("test");
        copy_dag.add_node(copy_dag_decl, RiscOp::Copy, vec![], scalar_f32(), None);
        let copy_errs = verify(&copy_dag);
        assert!(
            copy_errs.iter().any(|e| e.contains("unary op")),
            "{copy_errs:?}"
        );

        let mut drop_dag = Dag::new();
        let drop_dag_decl = drop_dag.declare("test");
        drop_dag.add_node(drop_dag_decl, RiscOp::Drop, vec![], scalar_f32(), None);
        let drop_errs = verify(&drop_dag);
        assert!(
            drop_errs.iter().any(|e| e.contains("unary op")),
            "{drop_errs:?}"
        );
    }

    #[test]
    fn bad_input_reference() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        // Manually create a node that references a future node (impossible via normal API,
        // but we can test via the replace_node backdoor or by constructing the scenario).
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        // Node 1 references itself (not earlier).
        dag.add_node(decl, RiscOp::Neg, vec![NodeId(1)], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs.iter().any(|e| e.contains("non-earlier node")));
        let _ = a;
    }

    #[test]
    fn wrong_arity_binary() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        // Add with only 1 input.
        dag.add_node(decl, RiscOp::Add, vec![a], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs[0].contains("binary op"));
    }

    #[test]
    fn wrong_arity_unary() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        // Neg with 0 inputs.
        dag.add_node(decl, RiscOp::Neg, vec![], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs[0].contains("unary op"));
    }

    #[test]
    fn wrong_arity_memory() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        // Const with an input (should have 0).
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b_ty = TensorType {
            dims: vec![],
            precision: Prim::F64,
        };
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(b_ty.precision, 2.0),
            vec![],
            b_ty,
            None,
        );
        dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs.iter().any(|e| e.contains("mismatched precisions")));
    }

    #[test]
    fn c1_matching_precision_ok() {
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
        dag.add_node(decl, RiscOp::Mul, vec![a, b], scalar_f32(), None);
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
        let decl = dag.declare("test");
        let i8_ty = TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::Int8,
        };
        let i32_ty = TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::Int32,
        };
        let a = dag.add_node(
            decl,
            RiscOp::Load { name: "a".into() },
            vec![],
            i8_ty.clone(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::Load { name: "b".into() },
            vec![],
            i32_ty.clone(),
            None,
        );
        // The output type doesn't matter — the verifier rejects on the
        // operand mismatch first.
        dag.add_node(decl, RiscOp::Add, vec![a, b], i32_ty, None);
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
        let decl = dag.declare("test");
        let ty1 = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let ty2 = TensorType {
            dims: vec![DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(ty1.precision, 1.0),
            vec![],
            ty1,
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(ty2.precision, 2.0),
            vec![],
            ty2,
            None,
        );
        dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("mismatched dimension count"))
        );
    }

    #[test]
    fn c2_mismatched_dim_size() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let ty1 = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let ty2 = TensorType {
            dims: vec![DimInfo::Lit(5)],
            precision: Prim::F32,
        };
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(ty1.precision, 1.0),
            vec![],
            ty1,
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(ty2.precision, 2.0),
            vec![],
            ty2,
            None,
        );
        dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_f32(), None);
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
        let decl = dag.declare("test");
        let ty = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(ty.precision, 1.0),
            vec![],
            ty.clone(),
            None,
        );
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::MaxReduce { axis: 0 },
            vec![x],
            scalar_f32(),
            None,
        );
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("axis 0 but input has 0 dimensions"))
        );
    }

    #[test]
    fn c3_sum_valid_axis() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let ty = TensorType {
            dims: vec![DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let out_ty = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(ty.precision, 1.0),
            vec![],
            ty,
            None,
        );
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let int_ty = TensorType {
            dims: vec![],
            precision: Prim::Int32,
        };
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(int_ty.precision, 1.0),
            vec![],
            int_ty.clone(),
            None,
        );
        dag.add_node(decl, RiscOp::Exp, vec![x], int_ty, None);
        let errs = verify(&dag);
        assert!(errs.iter().any(|e| e.contains("transcendental")));
    }

    #[test]
    fn c4_sqrt_on_float_ok() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 4.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Sqrt, vec![x], scalar_f32(), None);
        assert!(verify(&dag).is_empty());
    }

    #[test]
    fn c4_abs_on_signed_integer_is_valid_exact_ir() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let int_ty = TensorType {
            dims: vec![DimInfo::Lit(1)],
            precision: Prim::Int64,
        };
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(int_ty.precision, -1.0),
            vec![],
            int_ty.clone(),
            None,
        );
        dag.add_node(decl, RiscOp::Abs, vec![x], int_ty, None);
        assert!(verify(&dag).is_empty());
    }

    #[test]
    fn c4_integer_rounding_node_is_rejected_as_noncanonical() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let int_ty = TensorType {
            dims: vec![DimInfo::Lit(1)],
            precision: Prim::Int64,
        };
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(int_ty.precision, -1.0),
            vec![],
            int_ty.clone(),
            None,
        );
        dag.add_node(decl, RiscOp::Floor, vec![x], int_ty, None);
        let errs = verify(&dag);
        assert!(errs.iter().any(|error| error.contains("requires float")));
    }

    #[test]
    fn c4_abs_on_bool_is_error() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let bool_ty = TensorType {
            dims: vec![DimInfo::Lit(1)],
            precision: Prim::Bool,
        };
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(bool_ty.precision, 1.0),
            vec![],
            bool_ty.clone(),
            None,
        );
        dag.add_node(decl, RiscOp::Abs, vec![x], bool_ty, None);
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
        // Wrong: output is F32 instead of Bool.
        dag.add_node(
            decl,
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
        let bool_ty = TensorType {
            dims: vec![],
            precision: Prim::Bool,
        };
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let input_ty = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            input_ty,
            None,
        );
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let input_ty = TensorType {
            dims: vec![DimInfo::Lit(1)],
            precision: Prim::F32,
        };
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            input_ty,
            None,
        );
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let input_ty = TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::F32,
        };
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            input_ty,
            None,
        );
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let input_ty = TensorType {
            dims: vec![DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            input_ty,
            None,
        );
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let ty = TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::Int64,
        };
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty.clone(),
            None,
        );
        let wrong_fill = chelis_types::scalar_from_i64("pad", Prim::Int32, 0).unwrap();
        dag.add_node(
            decl,
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

    /// A key-operand `UniformLike` over `template` keyed by
    /// `key_from_seed(7)` with optional `activation`.
    fn keyed_uniform(dag: &mut Dag, decl: DeclId, template: NodeId, activation: Option<NodeId>) {
        let ty = dag.get(template).unwrap().output_type.clone();
        let low = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::F32, 0.0),
            vec![],
            TensorType::scalar_f32(),
            None,
        );
        let high = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::F32, 1.0),
            vec![],
            TensorType::scalar_f32(),
            None,
        );
        let seed = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::Int64, 7.0),
            vec![],
            TensorType {
                dims: vec![],
                precision: Prim::Int64,
            },
            None,
        );
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
        dag.add_node(
            crate::dag::Owner::new(decl, activation),
            RiscOp::UniformLike,
            vec![template, low, high, key],
            ty,
            None,
        );
    }

    #[test]
    fn uniform_like_rejects_non_float_template() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let ty = TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::Int64,
        };
        let template = dag.add_node(
            decl,
            RiscOp::Load {
                name: "template".into(),
            },
            vec![],
            ty,
            None,
        );
        keyed_uniform(&mut dag, decl, template, None);
        assert!(verify(&dag).iter().any(|error| {
            error.contains("must preserve its float template's exact shape and dtype")
        }));
    }

    #[test]
    fn uniform_like_path_activation_requires_scalar_bool() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let ty = TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::F32,
        };
        let template = dag.add_node(
            decl,
            RiscOp::Load {
                name: "template".into(),
            },
            vec![],
            ty,
            None,
        );
        let wrong_activation = dag.add_node(
            decl,
            RiscOp::Load {
                name: "activation".into(),
            },
            vec![],
            TensorType::scalar_f32(),
            None,
        );
        keyed_uniform(&mut dag, decl, template, Some(wrong_activation));
        assert!(
            verify(&dag)
                .iter()
                .any(|error| error.contains("random operation's activation must be a Bool"))
        );
    }

    #[test]
    fn c10_shrink_invalid_bounds_is_error() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let input_ty = TensorType {
            dims: vec![DimInfo::Lit(8)],
            precision: Prim::F32,
        };
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            input_ty,
            None,
        );
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let input_ty = TensorType {
            dims: vec![DimInfo::Lit(8)],
            precision: Prim::F32,
        };
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            input_ty,
            None,
        );
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        dag.add_node(
            decl,
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
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        let errs = verify(&dag);
        assert!(errs.iter().any(|e| e.contains("dangling")));
    }

    /// chelis#2440 widened the dangling exemption from a `GuardedFail`
    /// match to every observable root. The pair below is the boundary: an
    /// unconsumed INTEGER add is retained because it can overflow, and the
    /// otherwise identical FLOAT add is still rejected. Without the second
    /// half the exemption could drift into "any arithmetic may dangle".
    #[test]
    fn an_unconsumed_integer_add_is_not_dangling_but_its_float_twin_is() {
        fn dag_with(precision: Prim) -> Dag {
            let ty = TensorType {
                dims: vec![DimInfo::Lit(2)],
                precision,
            };
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let x = dag.add_node(
                decl,
                RiscOp::Load { name: "x".into() },
                vec![],
                ty.clone(),
                None,
            );
            let root = dag.add_node(decl, RiscOp::Add, vec![x, x], ty.clone(), None);
            dag.add_node(decl, RiscOp::Mul, vec![x, x], ty, None);
            dag.add_root(root);
            dag
        }

        let integer = verify(&dag_with(Prim::Int32));
        assert!(
            !integer.iter().any(|e| e.contains("dangling")),
            "an unconsumed integer node can overflow, so it is an observable \
             root, not dangling: {integer:?}"
        );

        let float = verify(&dag_with(Prim::F32));
        assert!(
            float.iter().any(|e| e.contains("dangling")),
            "float arithmetic cannot trap, so an unconsumed float node is \
             still dangling: {float:?}"
        );
    }

    #[test]
    fn root_nodes_are_not_dangling() {
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
        assert!(verify(&dag).is_empty());
    }

    #[test]
    fn load_name_type_mismatch_is_error() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(2)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_node(
            decl,
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
                .any(|e| e.contains("parameter `x` of `test` has inconsistent tensor types"))
        );
    }

    #[test]
    fn gather_requires_integer_indices_and_matching_output_precision() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let values = dag.add_node(
            decl,
            RiscOp::Load {
                name: "values".into(),
            },
            vec![],
            tensor_ty(&[4, 2], Prim::F64),
            None,
        );
        let bad_indices = dag.add_node(
            decl,
            RiscOp::synth_const(tensor_ty(&[3], Prim::F32).precision, 0.0),
            vec![],
            tensor_ty(&[3], Prim::F32),
            None,
        );
        dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let target = dag.add_node(
            decl,
            RiscOp::Load {
                name: "target".into(),
            },
            vec![],
            tensor_ty(&[4, 2], Prim::F64),
            None,
        );
        let bad_indices = dag.add_node(
            decl,
            RiscOp::synth_const(tensor_ty(&[3], Prim::F32).precision, 0.0),
            vec![],
            tensor_ty(&[3], Prim::F32),
            None,
        );
        let bad_updates = dag.add_node(
            decl,
            RiscOp::synth_const(tensor_ty(&[3, 2], Prim::F32).precision, 1.0),
            vec![],
            tensor_ty(&[3, 2], Prim::F32),
            None,
        );
        dag.add_node(
            decl,
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
