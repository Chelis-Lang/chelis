//! Reverse-mode automatic differentiation for RISC DAGs.
//!
//! Given a forward DAG computing `f(inputs) -> output`, produces a backward DAG
//! computing gradients of the output with respect to specified input nodes.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt;

use crate::dag::{Dag, DagNode, DimExpr, DimInfo, NodeId, RiscOp, RtDim, TensorType};
use crate::tier2;
use chelis_types::types::Prim;

/// Result of reverse-mode AD.
pub struct GradResult {
    /// Combined forward + backward DAG.
    pub dag: Dag,
    /// The remapped forward output node inside `dag`.
    pub output_node: NodeId,
    /// Maps each requested forward input `NodeId` to its gradient `NodeId` in the combined DAG.
    pub grad_nodes: HashMap<NodeId, NodeId>,
}

/// Structured reason for an AD rejection.
///
/// Variants are intentionally distinct so downstream consumers can
/// programmatically match on the rejection class rather than parsing
/// free-text. New variants are added as new AD-rejected ops ship; the
/// `Other` catch-all carries the legacy free-text message until each
/// case is given a structured variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdRejectionReason {
    /// The op is non-differentiable because its output is an integer
    /// index (e.g. `Argmax`, `Argmin`).
    IntegerIndexOutput,
    /// The op is piecewise constant; the analytic derivative is zero
    /// almost everywhere and undefined at the breakpoints (e.g.
    /// `Floor`, `Ceil`).
    PiecewiseConstant,
    /// The op is non-deterministic over duplicate target indices, so
    /// no well-defined reverse-mode adjoint exists. This is the
    /// fail-closed contract for replace-scatter (`Scatter`): when two
    /// updates target the same cell, the forward result depends on
    /// the iteration order, so the backward direction cannot
    /// distribute a single output gradient between the colliding
    /// updates without an arbitrary policy. Use `ScatterAdd` (whose
    /// adjoint is `Gather`) when an accumulating semantic is
    /// acceptable.
    NonDeterministicAtDuplicateIndices,
    /// The forward DAG was empty or the output node did not exist.
    EmptyOrMissingOutput,
    /// The output node's type is not a scalar float — reverse-mode AD
    /// requires a scalar loss.
    NonScalarOutput,
    /// Catch-all for legacy free-text rejection reasons that have not
    /// yet been given a structured variant. Carries the original
    /// message verbatim. Adding a new structured variant should
    /// migrate the corresponding message out of `Other` and into the
    /// dedicated enum case.
    Other(String),
}

/// Structured error returned by [`grad_dag_checked`].
///
/// The public AD-error surface is **programmatically matchable**:
/// downstream consumers should pattern-match on the enum variant and
/// fields, not parse the rendered `Display` string. The rendered
/// string is provided for human consumption only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdError {
    /// AD is not supported for the named op for the given structural
    /// reason. `op` uses the canonical `RiscOp` snake-case name (e.g.
    /// `"scatter_replace"` for the replace-scatter primitive, which
    /// matches the user-facing Surf builtin name).
    NotSupported {
        op: &'static str,
        reason: AdRejectionReason,
    },
}

impl AdError {
    /// Construct a `NotSupported` error.
    pub fn not_supported(op: &'static str, reason: AdRejectionReason) -> Self {
        AdError::NotSupported { op, reason }
    }
}

impl fmt::Display for AdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdError::NotSupported { op, reason } => match reason {
                AdRejectionReason::IntegerIndexOutput => write!(
                    f,
                    "grad: {op} is non-differentiable (integer-index output); \
                     remove it from the gradient path or wrap it in a stop-gradient"
                ),
                AdRejectionReason::PiecewiseConstant => write!(
                    f,
                    "grad: {op} is non-differentiable (piecewise constant); \
                     remove it from the gradient path or wrap it in a stop-gradient"
                ),
                AdRejectionReason::NonDeterministicAtDuplicateIndices => write!(
                    f,
                    "grad: {op} is non-differentiable (non-deterministic at duplicate \
                     indices -- last-write-wins forward semantics has no well-defined \
                     adjoint); use scatter_add (whose adjoint is gather) or wrap \
                     {op} in a stop-gradient"
                ),
                AdRejectionReason::EmptyOrMissingOutput => write!(
                    f,
                    "grad: cannot differentiate ({op}: empty DAG or missing output node)"
                ),
                AdRejectionReason::NonScalarOutput => {
                    write!(f, "grad: output of {op} must be a scalar float")
                }
                AdRejectionReason::Other(msg) => write!(f, "{msg}"),
            },
        }
    }
}

impl std::error::Error for AdError {}

/// Run reverse-mode AD on `forward` and return a diagnostic error if the
/// gradient cannot be computed for a structural reason (non-differentiable
/// ops such as `Argmax`/`Argmin` on the live forward graph, or an unsupported
/// output shape).
///
/// Prefer this over [`grad_dag`] in new code — it distinguishes
/// "differentiation is not meaningful here" from "no gradient requested".
pub fn grad_dag_checked(
    forward: &Dag,
    output: NodeId,
    wrt: &[NodeId],
) -> Result<GradResult, AdError> {
    if forward.is_empty() {
        return Err(AdError::NotSupported {
            op: "<empty>",
            reason: AdRejectionReason::EmptyOrMissingOutput,
        });
    }
    let out_node = forward.get(output).ok_or(AdError::NotSupported {
        op: "<missing>",
        reason: AdRejectionReason::EmptyOrMissingOutput,
    })?;
    if !is_scalar_float(&out_node.output_type) {
        return Err(AdError::NotSupported {
            op: risc_op_name(&out_node.op),
            reason: AdRejectionReason::NonScalarOutput,
        });
    }

    // Walk the subgraph of nodes reachable from `output` and look for ops
    // whose adjoint is intentionally undefined.
    //
    // chelis#616: a movement op's bound-source inputs (`inputs[1..]` — the
    // rank-0 integer `Shape`/arithmetic scalars that compute a runtime
    // `shrink`/`stride`/`pad` bound or a runtime `reshape` target extent, e.g.
    // the window count `m`) are INDEX MATH, not data. They carry no cotangent
    // (the adjoint routes gradient only to `inputs[0]`), so they are a
    // stop-gradient boundary and must not pull their producers — which may be
    // intentionally non-differentiable (e.g. the window-count `floor_div`) —
    // into the differentiability check. A bound scalar that is ALSO reached via
    // a genuine data edge stays live through that edge and is still checked.
    let mut live = vec![false; forward.len()];
    live[output.0] = true;
    for i in (0..forward.len()).rev() {
        if live[i] {
            let node = &forward.nodes()[i];
            let differentiable_inputs: &[NodeId] = match &node.op {
                RiscOp::Shrink { .. }
                | RiscOp::Stride { .. }
                | RiscOp::Pad { .. }
                | RiscOp::Reshape { .. } => &node.inputs[..node.inputs.len().min(1)],
                _ => &node.inputs,
            };
            for input in differentiable_inputs {
                live[input.0] = true;
            }
        }
    }
    for node in forward.nodes() {
        if !live[node.id.0] {
            continue;
        }
        match &node.op {
            RiscOp::Argmax { .. } => {
                return Err(AdError::NotSupported {
                    op: "argmax",
                    reason: AdRejectionReason::IntegerIndexOutput,
                });
            }
            RiscOp::Argmin { .. } => {
                return Err(AdError::NotSupported {
                    op: "argmin",
                    reason: AdRejectionReason::IntegerIndexOutput,
                });
            }
            RiscOp::Floor => {
                return Err(AdError::NotSupported {
                    op: "floor",
                    reason: AdRejectionReason::PiecewiseConstant,
                });
            }
            RiscOp::Ceil => {
                return Err(AdError::NotSupported {
                    op: "ceil",
                    reason: AdRejectionReason::PiecewiseConstant,
                });
            }
            RiscOp::Round => {
                return Err(AdError::NotSupported {
                    op: "round",
                    reason: AdRejectionReason::PiecewiseConstant,
                });
            }
            // chelis#178: floor / truncating integer division are
            // piecewise-constant (the quotient jumps at integer
            // boundaries), so the analytic derivative is zero almost
            // everywhere and undefined at the jumps. `grad` rejects them,
            // same treatment as `floor` / `ceil` / `round`.
            RiscOp::FloorDiv => {
                return Err(AdError::NotSupported {
                    op: "floor_div",
                    reason: AdRejectionReason::PiecewiseConstant,
                });
            }
            RiscOp::TruncDiv => {
                return Err(AdError::NotSupported {
                    op: "trunc_div",
                    reason: AdRejectionReason::PiecewiseConstant,
                });
            }
            // [05-OP-6]: `cast_trunc` is piecewise constant, so its
            // adjoint is zero almost everywhere and undefined at every
            // integer boundary. Rejecting is the point of the atom's
            // `no_grad` rule: a silent zero here would mask a modeling
            // bug rather than report it. The checked `cast` keeps its
            // float-to-float adjoint.
            RiscOp::CastTrunc { .. } => {
                return Err(AdError::NotSupported {
                    op: "cast_trunc",
                    reason: AdRejectionReason::PiecewiseConstant,
                });
            }
            RiscOp::Scatter { .. } => {
                // Last-write-wins replace-scatter is fail-closed for
                // AD: the forward result depends on iteration order at
                // duplicate target indices, so no well-defined adjoint
                // exists. See `RiscOp::Scatter` doc and
                // `spec/05-risc-primitives.md` §3.5.
                return Err(AdError::NotSupported {
                    op: "scatter_replace",
                    reason: AdRejectionReason::NonDeterministicAtDuplicateIndices,
                });
            }
            RiscOp::ScatterElements { .. } => {
                // Element-wise replace-scatter is fail-closed for AD for
                // the same reason as `Scatter` (last-write-wins at
                // duplicate indices). See `spec/05-risc-primitives.md`
                // §3.5.1.
                return Err(AdError::NotSupported {
                    op: "scatter_elements",
                    reason: AdRejectionReason::NonDeterministicAtDuplicateIndices,
                });
            }
            // `reduce_window_*` now has a reverse-mode adjoint
            // (`RiscOp::ReduceWindowGrad`, lowered in `grad_dag` below) per
            // spec/05-risc-primitives.md §2.3.1, so it is no longer rejected
            // here.
            _ => {}
        }
    }

    grad_dag_result(forward, output, wrt).map_err(|why| AdError::NotSupported {
        op: "<unknown>",
        reason: AdRejectionReason::Other(format!("failed to construct backward DAG ({why})")),
    })
}

/// Canonical snake-case name for a `RiscOp` for use in `AdError`'s
/// `op` field. Mirrors the user-facing Surf builtin name where one
/// exists.
fn risc_op_name(op: &RiscOp) -> &'static str {
    match op {
        RiscOp::Add => "add",
        RiscOp::Mul => "mul",
        RiscOp::Div => "div",
        RiscOp::FloorDiv => "floor_div",
        RiscOp::TruncDiv => "trunc_div",
        RiscOp::CmpLt => "cmplt",
        RiscOp::MaxElem => "max_elem",
        RiscOp::Neg => "neg",
        RiscOp::Recip => "recip",
        RiscOp::Exp => "exp",
        RiscOp::Log => "log",
        RiscOp::Sin => "sin",
        RiscOp::Sqrt => "sqrt",
        RiscOp::Cos => "cos",
        RiscOp::Tan => "tan",
        RiscOp::Atan => "atan",
        RiscOp::Abs => "abs",
        RiscOp::Floor => "floor",
        RiscOp::Ceil => "ceil",
        RiscOp::Round => "round",
        RiscOp::UniformLike { .. } => "uniform_like",
        RiscOp::Dropout { .. } => "dropout",
        RiscOp::Sum { .. } => "sum",
        RiscOp::MaxReduce { .. } => "max_reduce",
        RiscOp::MinReduce { .. } => "min_reduce",
        RiscOp::ProdReduce { .. } => "prod_reduce",
        RiscOp::ReduceWindow { reducer, .. } => reducer.surf_name(),
        RiscOp::ReduceWindowGrad { .. } => "reduce_window_grad",
        RiscOp::Argmax { .. } => "argmax",
        RiscOp::Argmin { .. } => "argmin",
        RiscOp::Reshape { .. } => "reshape",
        RiscOp::Permute { .. } => "permute",
        RiscOp::Expand { .. } => "expand",
        RiscOp::OneHot { .. } => "one_hot",
        RiscOp::Pad { .. } => "pad",
        RiscOp::Shrink { .. } => "shrink",
        RiscOp::Stride { .. } => "stride",
        RiscOp::Shape { .. } => "shape",
        RiscOp::Const { .. } => "const",
        RiscOp::ConstTensor { .. } => "const_tensor",
        RiscOp::Load { .. } => "load",
        RiscOp::Store { .. } => "store",
        RiscOp::Copy => "copy",
        RiscOp::Drop => "drop",
        RiscOp::Realize => "realize",
        RiscOp::Cast { .. } => "cast",
        RiscOp::CastTrunc { .. } => "cast_trunc",
        RiscOp::FusedElem { .. } => "fused_elem",
        RiscOp::BlasMatmul { .. } => "blas_matmul",
        RiscOp::Gather { .. } => "gather",
        RiscOp::ScatterAdd { .. } => "scatter_add",
        RiscOp::Scatter { .. } => "scatter_replace",
        RiscOp::ScatterElements { .. } => "scatter_elements",
    }
}

/// Canonical synthesized marker for AD backward (adjoint) nodes. Locked
/// by spec/03-deep-syntax.md §1.1.1 — double-underscore wrap, lowercase
/// pass name. Every adjoint node carries this as its `span_id` and
/// records the corresponding forward node's `span_id` (and any of its
/// pre-existing `merged_spans`) inside `merged_spans` so the audit
/// chain (backward → forward → LaTeX source) is reconstructible.
pub const GRAD_SYNTH_MARKER: &str = "__synthesized_grad__";

/// Stamp the AD synthesized-marker rule onto nodes added during a
/// single adjoint construction. `dag_size_before` is the dag length
/// captured immediately before the adjoint helper ran; every node from
/// that index onward is a backward node corresponding to `forward_node`
/// (the forward operation being differentiated). Each backward node:
///   * gets `span_id = "__synthesized_grad__"` (overwriting whatever
///     the constructor wrote — including `__synthesized_tier2__` for
///     adjoints that go through a Tier 2 helper),
///   * gets the forward node's `span_id` and `merged_spans` folded into
///     its `merged_spans` (lex-sorted, deduped) so the forward span is
///     always present alongside the synthesized marker.
fn stamp_grad_marker(dag: &mut Dag, dag_size_before: usize, forward_node: &DagNode) {
    let new_len = dag.len();
    for idx in dag_size_before..new_len {
        let id = NodeId(idx);
        if let Some(node) = dag.node_mut(id) {
            node.span_id = Some(GRAD_SYNTH_MARKER.to_owned());
            // Wipe any prior merged_spans (the constructor may have
            // populated some via tier2 helpers; we overwrite with the
            // forward-side provenance to maintain the canonical order).
            node.merged_spans.clear();
        }
        crate::span_merge::append_span_to_node(dag, id, forward_node.span_id.as_deref());
        crate::span_merge::append_spans_to_node(dag, id, &forward_node.merged_spans);
    }
}

/// Run reverse-mode AD on `forward`, differentiating `output` with respect to each node in `wrt`.
///
/// Returns `None` if the forward DAG is empty or the output node doesn't exist.
pub fn grad_dag(forward: &Dag, output: NodeId, wrt: &[NodeId]) -> Option<GradResult> {
    grad_dag_result(forward, output, wrt).ok()
}

/// Like [`grad_dag`] but returns a structured failure string instead of
/// a bare `None` when backward construction fails. The string names the
/// concrete cause — either an unsupported op whose adjoint is undefined
/// or the post-construction verifier diagnostics — so callers
/// (`grad_dag_checked` and its user-facing lowering error) can report
/// *why* the backward DAG could not be built rather than the legacy
/// opaque "unsupported op or verification failure".
fn grad_dag_result(forward: &Dag, output: NodeId, wrt: &[NodeId]) -> Result<GradResult, String> {
    if forward.is_empty() {
        return Err("grad: forward DAG is empty".to_string());
    }
    let output_ty = forward
        .get(output)
        .ok_or_else(|| "grad: output node is missing from the forward DAG".to_string())?
        .output_type
        .clone();
    if !is_scalar_float(&output_ty) {
        return Err(format!(
            "grad: output node type {output_ty:?} is not a scalar float"
        ));
    }
    // Forward nodes clone span_id + merged_spans unchanged via Dag::clone()
    // — `forward.clone()` deep-copies the DagNodes, and the existing
    // serde derives include the span fields. Per
    // spec/design/chelis_span_survival.md §2.3 AD row: "Forward nodes:
    // clone span_id and merged_spans."
    let mut dag = forward.clone();
    let mut adjoints: HashMap<NodeId, NodeId> = HashMap::new();
    // Seed the gradient at `output` (∂output/∂output = 1). This is a
    // backward node corresponding to the forward `output`, so it
    // carries the grad marker.
    let output_node = forward.get(output).unwrap().clone();
    let dag_before_seed = dag.len();
    let seed = dag.add_node(
        RiscOp::synth_const(output_ty.precision, 1.0),
        vec![],
        output_ty,
        None,
    );
    stamp_grad_marker(&mut dag, dag_before_seed, &output_node);
    adjoints.insert(output, seed);

    // Walk forward topological order in reverse.
    let topo = forward.topological_order();
    for &node_id in topo.iter().rev() {
        let grad_out = match adjoints.get(&node_id) {
            Some(&g) => g,
            None => continue,
        };

        let node = forward.get(node_id).unwrap().clone();
        let dag_size_before = dag.len();
        let input_grads =
            compute_adjoints(&node, grad_out, forward, &mut dag).ok_or_else(|| {
                format!(
                    "grad: no reverse-mode adjoint is defined for `{}` (node {})",
                    risc_op_name(&node.op),
                    node.id.0
                )
            })?;
        // Every node added inside compute_adjoints is a backward
        // (adjoint) node for `node`. Stamp the grad marker + the
        // forward span onto each.
        stamp_grad_marker(&mut dag, dag_size_before, &node);

        for (input_id, grad_node) in input_grads {
            match adjoints.entry(input_id) {
                Entry::Vacant(e) => {
                    e.insert(grad_node);
                }
                Entry::Occupied(mut e) => {
                    let existing = *e.get();
                    let ty = dag.get(existing).unwrap().output_type.clone();
                    // Sum-accumulator for multi-consumer forward nodes
                    // — also a backward node, attributed to the
                    // forward input being accumulated.
                    let dag_before_sum = dag.len();
                    let sum = dag.add_node(RiscOp::Add, vec![existing, grad_node], ty, None);
                    let input_forward = forward.get(input_id).unwrap().clone();
                    stamp_grad_marker(&mut dag, dag_before_sum, &input_forward);
                    e.insert(sum);
                }
            }
        }
    }

    let grad_nodes = wrt
        .iter()
        .filter_map(|&id| adjoints.get(&id).map(|&g| (id, g)))
        .collect::<HashMap<_, _>>();

    dag.add_root(output);
    for &grad in grad_nodes.values() {
        dag.add_root(grad);
    }

    // chelis#616: a backward node may reference an op-declared runtime dim
    // (e.g. the Sum adjoint's Expand over a runtime reshape extent) whose
    // declaring forward node's VALUE is otherwise dead. Record shape-deps so
    // the pruning below keeps each declarer and its bound-scalar chain.
    crate::dag::record_runtime_dim_shape_deps(&mut dag);

    let (dag, output_node, grad_nodes) = prune_to_requested_outputs(&dag, output, &grad_nodes);

    let verify_errors = crate::verify::verify(&dag);
    if !verify_errors.is_empty() {
        return Err(format!(
            "grad: constructed backward DAG failed verification: {}",
            verify_errors.join("; ")
        ));
    }

    Ok(GradResult {
        dag,
        output_node,
        grad_nodes,
    })
}

fn is_scalar_float(ty: &TensorType) -> bool {
    ty.dims.is_empty() && ty.precision.is_float()
}

fn prune_to_requested_outputs(
    dag: &Dag,
    output: NodeId,
    grad_nodes: &HashMap<NodeId, NodeId>,
) -> (Dag, NodeId, HashMap<NodeId, NodeId>) {
    if dag.is_empty() {
        return (Dag::new(), NodeId(0), HashMap::new());
    }

    let mut live = vec![false; dag.len()];
    for &root in dag.roots() {
        live[root.0] = true;
    }
    for node in dag.nodes() {
        if matches!(node.op, RiscOp::Store { .. }) {
            live[node.id.0] = true;
        }
    }
    for i in (0..dag.len()).rev() {
        if live[i] {
            for &input in &dag.nodes()[i].inputs {
                live[input.0] = true;
            }
            // chelis#384/#397/#616: a shape-only dep (a Form-3 `expand`
            // source or a runtime-dim declarer) keeps its source alive; the
            // consumer needs the extent even though it never reads the value.
            for &dep in &dag.nodes()[i].shape_deps {
                live[dep.0] = true;
            }
        }
    }

    let mut new_dag = Dag::new();
    let mut id_map = HashMap::<usize, NodeId>::new();
    for node in dag.nodes() {
        if live[node.id.0] {
            let new_inputs = node
                .inputs
                .iter()
                .map(|input| *id_map.get(&input.0).expect("live input must be remapped"))
                .collect();
            // Pruning is a pure copy: thread span_id and merged_spans
            // through unchanged. Same rule as DCE/remap (§2.3 DCE row);
            // grad's pruner is a separate code path that must not
            // drift. Without this the AD seed/backward nodes would lose
            // their `__synthesized_grad__` markers post-prune.
            let new_id = new_dag.add_node(
                node.op.clone(),
                new_inputs,
                node.output_type.clone(),
                node.span_id.clone(),
            );
            if !node.merged_spans.is_empty()
                && let Some(new_node) = new_dag.node_mut(new_id)
            {
                new_node.merged_spans = node.merged_spans.clone();
            }
            // chelis#384/#397: preserve (remapped) Form-3 `expand` shape-deps.
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
            id_map.insert(node.id.0, new_id);
        }
    }

    for &root in dag.roots() {
        if let Some(&new_root) = id_map.get(&root.0) {
            new_dag.add_root(new_root);
        }
    }

    let new_grad_nodes = grad_nodes
        .iter()
        .filter_map(|(old_wrt, old_grad)| {
            id_map
                .get(&old_grad.0)
                .copied()
                .map(|new_grad| (*old_wrt, new_grad))
        })
        .collect();

    let new_output = *id_map
        .get(&output.0)
        .unwrap_or_else(|| panic!("output node {output:?} missing after grad pruning"));

    (new_dag, new_output, new_grad_nodes)
}

/// Compute adjoint contributions for each input of the given node.
/// Returns `(forward_input_id, gradient_node_in_dag)` pairs.
fn compute_adjoints(
    node: &DagNode,
    g: NodeId,
    forward: &Dag,
    dag: &mut Dag,
) -> Option<Vec<(NodeId, NodeId)>> {
    match &node.op {
        // --- Binary elementwise ---
        RiscOp::Add => {
            let a = node.inputs[0];
            let b = node.inputs[1];
            Some(vec![(a, g), (b, g)])
        }
        RiscOp::Mul => {
            let a = node.inputs[0];
            let b = node.inputs[1];
            let ty = forward.get(a).unwrap().output_type.clone();
            // da = g * b, db = g * a  (referencing forward nodes directly)
            let da = dag.add_node(RiscOp::Mul, vec![g, b], ty.clone(), None);
            let db = dag.add_node(RiscOp::Mul, vec![g, a], ty, None);
            Some(vec![(a, da), (b, db)])
        }
        RiscOp::Div => {
            // y = a / b
            // dL/da = g / b           = Div(g, b)
            // dL/db = -g * a / b^2    = -g * y / b   (using y = a/b ⇒ a/b² = y/b)
            let a = node.inputs[0];
            let b = node.inputs[1];
            let ty = forward.get(a).unwrap().output_type.clone();
            let da = dag.add_node(RiscOp::Div, vec![g, b], ty.clone(), None);
            let g_times_y = dag.add_node(RiscOp::Mul, vec![g, node.id], ty.clone(), None);
            let g_y_over_b = dag.add_node(RiscOp::Div, vec![g_times_y, b], ty.clone(), None);
            let db = dag.add_node(RiscOp::Neg, vec![g_y_over_b], ty, None);
            Some(vec![(a, da), (b, db)])
        }
        RiscOp::CmpLt => {
            let a = node.inputs[0];
            let b = node.inputs[1];
            let ty_a = forward.get(a).unwrap().output_type.clone();
            let ty_b = forward.get(b).unwrap().output_type.clone();
            let za = dag.add_node(RiscOp::synth_const(ty_a.precision, 0.0), vec![], ty_a, None);
            let zb = dag.add_node(RiscOp::synth_const(ty_b.precision, 0.0), vec![], ty_b, None);
            Some(vec![(a, za), (b, zb)])
        }
        RiscOp::MaxElem => {
            // Subgradient per spec: da = g * (x >= y), db = g * (x < y)
            // (x >= y) = NOT(x < y) = 1 - cmplt(a, b)
            let a = node.inputs[0];
            let b = node.inputs[1];
            let ty = forward.get(a).unwrap().output_type.clone();
            let bool_ty = TensorType {
                dims: ty.dims.clone(),
                precision: Prim::Bool,
            };
            let a_lt_b_bool = dag.add_node(RiscOp::CmpLt, vec![a, b], bool_ty, None);
            let a_lt_b = dag.add_node(
                RiscOp::Cast {
                    new_precision: ty.precision,
                },
                vec![a_lt_b_bool],
                ty.clone(),
                None,
            );
            let one = dag.add_node(
                RiscOp::synth_const(ty.precision, 1.0),
                vec![],
                ty.clone(),
                None,
            );
            // a_ge_b = 1 - cmplt(a, b)  (NOT via subtraction since bools are 0/1)
            let neg_a_lt_b = dag.add_node(RiscOp::Neg, vec![a_lt_b], ty.clone(), None);
            let a_ge_b = dag.add_node(RiscOp::Add, vec![one, neg_a_lt_b], ty.clone(), None);
            let da = dag.add_node(RiscOp::Mul, vec![g, a_ge_b], ty.clone(), None);
            let db = dag.add_node(RiscOp::Mul, vec![g, a_lt_b], ty, None);
            Some(vec![(a, da), (b, db)])
        }

        // --- Unary elementwise ---
        RiscOp::Neg => {
            let x = node.inputs[0];
            let ty = forward.get(x).unwrap().output_type.clone();
            let dg = dag.add_node(RiscOp::Neg, vec![g], ty, None);
            Some(vec![(x, dg)])
        }
        RiscOp::Recip => {
            // y = 1/x  ⇒  dL/dx = -g * y * y   (using y = 1/x ⇒ -1/x² = -y²)
            let x = node.inputs[0];
            let ty = forward.get(x).unwrap().output_type.clone();
            let y_sq = dag.add_node(RiscOp::Mul, vec![node.id, node.id], ty.clone(), None);
            let g_y_sq = dag.add_node(RiscOp::Mul, vec![g, y_sq], ty.clone(), None);
            let dx = dag.add_node(RiscOp::Neg, vec![g_y_sq], ty, None);
            Some(vec![(x, dx)])
        }
        RiscOp::Exp => {
            // d/dx exp(x) = exp(x). Reuse the forward exp node.
            let x = node.inputs[0];
            let ty = forward.get(x).unwrap().output_type.clone();
            let dx = dag.add_node(RiscOp::Mul, vec![g, node.id], ty, None);
            Some(vec![(x, dx)])
        }
        RiscOp::Log => {
            // d/dx log(x) = 1/x = div(g, x)
            let x = node.inputs[0];
            let ty = forward.get(x).unwrap().output_type.clone();
            let dx = tier2::lower_div(dag, g, x, &ty, None);
            Some(vec![(x, dx)])
        }
        RiscOp::Sin => {
            // d/dx sin(x) = cos(x) = sin(x + pi/2)
            let x = node.inputs[0];
            let ty = forward.get(x).unwrap().output_type.clone();
            let half_pi = dag.add_node(
                RiscOp::synth_const(ty.precision, std::f64::consts::FRAC_PI_2),
                vec![],
                ty.clone(),
                None,
            );
            let shifted = dag.add_node(RiscOp::Add, vec![x, half_pi], ty.clone(), None);
            let cos_x = dag.add_node(RiscOp::Sin, vec![shifted], ty.clone(), None);
            let dx = dag.add_node(RiscOp::Mul, vec![g, cos_x], ty, None);
            Some(vec![(x, dx)])
        }
        RiscOp::Sqrt => {
            // d/dx sqrt(x) = 1 / (2 * sqrt(x)). Reuse forward sqrt node.
            let x = node.inputs[0];
            let ty = forward.get(x).unwrap().output_type.clone();
            let two = dag.add_node(
                RiscOp::synth_const(ty.precision, 2.0),
                vec![],
                ty.clone(),
                None,
            );
            let two_sqrt = dag.add_node(RiscOp::Mul, vec![two, node.id], ty.clone(), None);
            let dx = tier2::lower_div(dag, g, two_sqrt, &ty, None);
            Some(vec![(x, dx)])
        }
        RiscOp::Cos => {
            // d/dx cos(x) = -sin(x)
            let x = node.inputs[0];
            let ty = forward.get(x).unwrap().output_type.clone();
            let sin_x = dag.add_node(RiscOp::Sin, vec![x], ty.clone(), None);
            let neg_sin_x = dag.add_node(RiscOp::Neg, vec![sin_x], ty.clone(), None);
            let dx = dag.add_node(RiscOp::Mul, vec![g, neg_sin_x], ty, None);
            Some(vec![(x, dx)])
        }
        RiscOp::Tan => {
            // d/dx tan(x) = 1 / cos²(x) = g / (cos(x) * cos(x))
            let x = node.inputs[0];
            let ty = forward.get(x).unwrap().output_type.clone();
            let cos_x = dag.add_node(RiscOp::Cos, vec![x], ty.clone(), None);
            let cos_sq = dag.add_node(RiscOp::Mul, vec![cos_x, cos_x], ty.clone(), None);
            let dx = tier2::lower_div(dag, g, cos_sq, &ty, None);
            Some(vec![(x, dx)])
        }
        RiscOp::Atan => {
            // d/dx atan(x) = 1 / (1 + x²) = g / (1 + x*x)
            let x = node.inputs[0];
            let ty = forward.get(x).unwrap().output_type.clone();
            let one = dag.add_node(
                RiscOp::synth_const(ty.precision, 1.0),
                vec![],
                ty.clone(),
                None,
            );
            let x_sq = dag.add_node(RiscOp::Mul, vec![x, x], ty.clone(), None);
            let denom = dag.add_node(RiscOp::Add, vec![one, x_sq], ty.clone(), None);
            let dx = tier2::lower_div(dag, g, denom, &ty, None);
            Some(vec![(x, dx)])
        }
        RiscOp::Abs => {
            // d/dx abs(x) = sign(x): 1 if x > 0, -1 if x < 0, 0 if x = 0
            // Expressed as: (x > 0) - (x < 0) cast to float, then * g
            // Equivalently: cast(cmplt(zero, x)) - cast(cmplt(x, zero)) multiplied by g
            let x = node.inputs[0];
            let ty = forward.get(x).unwrap().output_type.clone();
            let bool_ty = TensorType {
                dims: ty.dims.clone(),
                precision: chelis_types::types::Prim::Bool,
            };
            let zero = dag.add_node(
                RiscOp::synth_const(ty.precision, 0.0),
                vec![],
                ty.clone(),
                None,
            );
            // positive mask: x > 0  i.e. cmplt(0, x)
            let pos_bool = dag.add_node(RiscOp::CmpLt, vec![zero, x], bool_ty.clone(), None);
            let pos = dag.add_node(
                RiscOp::Cast {
                    new_precision: ty.precision,
                },
                vec![pos_bool],
                ty.clone(),
                None,
            );
            // negative mask: x < 0  i.e. cmplt(x, 0)
            let neg_bool = dag.add_node(RiscOp::CmpLt, vec![x, zero], bool_ty, None);
            let neg_cast = dag.add_node(
                RiscOp::Cast {
                    new_precision: ty.precision,
                },
                vec![neg_bool],
                ty.clone(),
                None,
            );
            // sign = pos - neg_cast  (tier2 sub)
            let sign = tier2::lower_sub(dag, pos, neg_cast, &ty, None);
            let dx = dag.add_node(RiscOp::Mul, vec![sign, g], ty, None);
            Some(vec![(x, dx)])
        }
        RiscOp::Floor => {
            // floor is non-differentiable — grad_dag_checked will have already
            // rejected this; this arm is a safety net returning zero gradient.
            let x = node.inputs[0];
            let ty = forward.get(x).unwrap().output_type.clone();
            let zero = dag.add_node(RiscOp::synth_const(ty.precision, 0.0), vec![], ty, None);
            Some(vec![(x, zero)])
        }
        RiscOp::Ceil => {
            // ceil is non-differentiable — grad_dag_checked will have already
            // rejected this; this arm is a safety net returning zero gradient.
            let x = node.inputs[0];
            let ty = forward.get(x).unwrap().output_type.clone();
            let zero = dag.add_node(RiscOp::synth_const(ty.precision, 0.0), vec![], ty, None);
            Some(vec![(x, zero)])
        }
        RiscOp::Round => {
            // round is non-differentiable — grad_dag_checked will have already
            // rejected this; this arm is a safety net returning zero gradient.
            let x = node.inputs[0];
            let ty = forward.get(x).unwrap().output_type.clone();
            let zero = dag.add_node(RiscOp::synth_const(ty.precision, 0.0), vec![], ty, None);
            Some(vec![(x, zero)])
        }
        RiscOp::FloorDiv | RiscOp::TruncDiv => {
            // chelis#178: floor / truncating integer division are
            // non-differentiable (piecewise constant) — grad_dag_checked
            // will have already rejected these; this arm is a safety net
            // returning zero gradient to both operands.
            let a = node.inputs[0];
            let b = node.inputs[1];
            let ty_a = forward.get(a).unwrap().output_type.clone();
            let ty_b = forward.get(b).unwrap().output_type.clone();
            let za = dag.add_node(RiscOp::synth_const(ty_a.precision, 0.0), vec![], ty_a, None);
            let zb = dag.add_node(RiscOp::synth_const(ty_b.precision, 0.0), vec![], ty_b, None);
            Some(vec![(a, za), (b, zb)])
        }
        RiscOp::UniformLike { .. } => {
            let x = node.inputs[0];
            let ty = forward.get(x).unwrap().output_type.clone();
            let zero = dag.add_node(RiscOp::synth_const(ty.precision, 0.0), vec![], ty, None);
            Some(vec![(x, zero)])
        }
        RiscOp::Dropout { rate, seed } => {
            let x = node.inputs[0];
            let ty = forward.get(x).unwrap().output_type.clone();
            let dx = dag.add_node(
                RiscOp::Dropout {
                    rate: *rate,
                    seed: *seed,
                },
                vec![g],
                ty,
                None,
            );
            Some(vec![(x, dx)])
        }

        // --- Reduction ---
        RiscOp::Sum { axis, .. } => {
            // d/dx sum(x, axis) = expand(g, axis, original_size)
            //
            // WS-A3 fix: when the upstream gradient `g` has a different
            // precision than the operand `x` (which happens whenever
            // the Sum carries a wider accumulator per
            // spec/04-type-system.md §5.7.1 — bf16/f16 operands sum
            // into f32, integer-narrow operands sum into i32), insert
            // an explicit cast back to the operand precision before
            // the Expand. Without the cast, Expand would carry an
            // input of one dtype and output of another, which IR
            // validation rejects. The cast direction is from the wider
            // accumulator back to the operand precision so the
            // returned adjoint has `input_ty.precision`, matching the
            // spec rule "gradient precision = operand precision".
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            let original_size = DimExpr::from(&input_ty.dims[*axis]);
            let g_node = dag
                .get(g)
                .expect("upstream adjoint must exist in the AD DAG");
            let g_for_expand = if g_node.output_type.precision == input_ty.precision {
                g
            } else {
                let g_ty = TensorType {
                    dims: g_node.output_type.dims.clone(),
                    precision: input_ty.precision,
                };
                dag.add_node(
                    RiscOp::Cast {
                        new_precision: input_ty.precision,
                    },
                    vec![g],
                    g_ty,
                    None,
                )
            };
            let dx = dag.add_node(
                RiscOp::Expand {
                    axis: *axis,
                    size: original_size.clone(),
                },
                vec![g_for_expand],
                input_ty,
                None,
            );
            // chelis#616: the expand restores the forward input's shape; a
            // non-concrete size (a runtime or wildcard axis) resolves at
            // eval time from the input's actual value via this shape-dep.
            if original_size.as_concrete().is_none() {
                dag.add_shape_dep(dx, x);
            }
            Some(vec![(x, dx)])
        }
        RiscOp::MaxReduce { axis } => {
            // Subgradient: gradient flows to elements equal to the max.
            // mask = eq(x, expand(max_reduce(x, axis), axis, size))
            // dx = mul(expand(g, axis, size), mask)
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            let original_size = DimExpr::from(&input_ty.dims[*axis]);

            // Expand forward max_reduce node back to input shape.
            let expanded_max = dag.add_node(
                RiscOp::Expand {
                    axis: *axis,
                    size: original_size.clone(),
                },
                vec![node.id],
                input_ty.clone(),
                None,
            );

            // Expand gradient to input shape.
            let expanded_g = dag.add_node(
                RiscOp::Expand {
                    axis: *axis,
                    size: original_size,
                },
                vec![g],
                input_ty.clone(),
                None,
            );

            // Build equality mask: not(or(cmplt(x, expanded_max), cmplt(expanded_max, x)))
            let mask_bool = tier2::lower_eq(dag, x, expanded_max, &input_ty, None);
            let mask = dag.add_node(
                RiscOp::Cast {
                    new_precision: input_ty.precision,
                },
                vec![mask_bool],
                input_ty.clone(),
                None,
            );

            let dx = dag.add_node(RiscOp::Mul, vec![expanded_g, mask], input_ty, None);
            Some(vec![(x, dx)])
        }
        RiscOp::MinReduce { axis } => {
            // Subgradient mirrors MaxReduce: gradient flows to elements equal
            // to the min. This is a first-class rule, NOT composed as
            // neg(max_reduce(neg(x))) — that would work but obscures the
            // numerical semantics and makes autodiff graph inspection
            // harder. Spec §3j-pre: ship the rule explicitly.
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            let original_size = DimExpr::from(&input_ty.dims[*axis]);

            let expanded_min = dag.add_node(
                RiscOp::Expand {
                    axis: *axis,
                    size: original_size.clone(),
                },
                vec![node.id],
                input_ty.clone(),
                None,
            );
            let expanded_g = dag.add_node(
                RiscOp::Expand {
                    axis: *axis,
                    size: original_size,
                },
                vec![g],
                input_ty.clone(),
                None,
            );
            let mask_bool = tier2::lower_eq(dag, x, expanded_min, &input_ty, None);
            let mask = dag.add_node(
                RiscOp::Cast {
                    new_precision: input_ty.precision,
                },
                vec![mask_bool],
                input_ty.clone(),
                None,
            );
            let dx = dag.add_node(RiscOp::Mul, vec![expanded_g, mask], input_ty, None);
            Some(vec![(x, dx)])
        }
        RiscOp::ProdReduce { axis } => {
            // Safe prefix*suffix product adjoint. The naive form
            // `g * prod / x` divides by zero whenever any element in the
            // reduced slice is zero (and the gradient at that element is
            // precisely `prod_of_everyone_else`, a finite number that the
            // naive form cannot represent).
            //
            // Construction: split the input along `axis` into k 1-wide
            // slices (via Shrink). For each slice i, build
            //   prefix_i = prod_{j<i} slice_j  (running product)
            //   suffix_i = prod_{j>i} slice_j
            //   ∂prod/∂slice_i = prefix_i * suffix_i
            // then Pad each contribution back to the full axis width and
            // sum them (via element-wise Add since each contribution is
            // zero outside its slice). Finally multiply by the expanded
            // upstream gradient and return.
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            let axis_size = match &input_ty.dims[*axis] {
                DimInfo::Lit(n) => *n,
                DimInfo::Named(_, Some(n)) => *n,
                DimInfo::Named(name, None) => panic!(
                    "prod_reduce adjoint requires a concrete axis size; got symbolic `{name}`"
                ),
            };
            let rank = input_ty.dims.len();
            let original_size = DimExpr::from(&input_ty.dims[*axis]);

            // Build slice types: same as input_ty but with axis dim = 1.
            let mut slice_dims = input_ty.dims.clone();
            slice_dims[*axis] = DimInfo::Lit(1);
            let slice_ty = TensorType {
                dims: slice_dims,
                precision: input_ty.precision,
            };

            // Slice each element along `axis`.
            //
            // chelis#513 gap 3 (structural slice): a bystander NON-reduced
            // axis takes the whole axis (`(0, full_extent)`), so a symbolic
            // bystander dim uses the `SHRINK_TO_END` full-axis sentinel
            // instead of demanding a concrete size; `bind_symbolic_dims` (eval
            // lane) and `emit_shrink` (C lane) resolve it. Only the REDUCED
            // axis needs a concrete size (the slice count), which stays a
            // hard error above -- no silent guess.
            let mut slices: Vec<NodeId> = Vec::with_capacity(axis_size);
            for i in 0..axis_size {
                let bounds: Vec<(RtDim, RtDim)> = (0..rank)
                    .map(|d| {
                        if d == *axis {
                            (RtDim::Lit(i), RtDim::Lit(i + 1))
                        } else {
                            match &input_ty.dims[d] {
                                DimInfo::Named(_, None) => (RtDim::Lit(0), RtDim::ToEnd),
                                dim => (RtDim::Lit(0), RtDim::Lit(dim_size(dim))),
                            }
                        }
                    })
                    .collect();
                let s = dag.add_node(RiscOp::Shrink { bounds }, vec![x], slice_ty.clone(), None);
                slices.push(s);
            }

            // Prefix products: prefix[i] = prod_{j<i} slices[j], with prefix[0] = 1.
            let one = dag.add_node(
                RiscOp::synth_const(slice_ty.precision, 1.0),
                vec![],
                slice_ty.clone(),
                None,
            );
            let mut prefix: Vec<NodeId> = Vec::with_capacity(axis_size);
            prefix.push(one);
            for i in 1..axis_size {
                let p = dag.add_node(
                    RiscOp::Mul,
                    vec![prefix[i - 1], slices[i - 1]],
                    slice_ty.clone(),
                    None,
                );
                prefix.push(p);
            }

            // Suffix products: suffix[i] = prod_{j>i} slices[j], with suffix[n-1] = 1.
            let mut suffix: Vec<NodeId> = vec![one; axis_size];
            if axis_size >= 2 {
                for i in (0..axis_size - 1).rev() {
                    suffix[i] = dag.add_node(
                        RiscOp::Mul,
                        vec![suffix[i + 1], slices[i + 1]],
                        slice_ty.clone(),
                        None,
                    );
                }
            }

            // Per-slice local gradient = prefix[i] * suffix[i], padded back to
            // the full axis width. Sum them into a single full-shape tensor.
            let zero_const = 0.0f64;
            let mut acc: Option<NodeId> = None;
            for i in 0..axis_size {
                let local = dag.add_node(
                    RiscOp::Mul,
                    vec![prefix[i], suffix[i]],
                    slice_ty.clone(),
                    None,
                );
                let padding: Vec<(RtDim, RtDim)> = (0..rank)
                    .map(|d| {
                        if d == *axis {
                            (RtDim::Lit(i), RtDim::Lit(axis_size - i - 1))
                        } else {
                            (RtDim::Lit(0), RtDim::Lit(0))
                        }
                    })
                    .collect();
                let padded = dag.add_node(
                    RiscOp::Pad {
                        padding,
                        fill: zero_const,
                    },
                    vec![local],
                    input_ty.clone(),
                    None,
                );
                acc = Some(match acc {
                    None => padded,
                    Some(prev) => {
                        dag.add_node(RiscOp::Add, vec![prev, padded], input_ty.clone(), None)
                    }
                });
            }

            let local_grad = acc.expect("prod_reduce adjoint needs axis_size >= 1");

            // Upstream gradient expanded back to input shape, multiplied by local.
            let expanded_g = dag.add_node(
                RiscOp::Expand {
                    axis: *axis,
                    size: original_size,
                },
                vec![g],
                input_ty.clone(),
                None,
            );
            let dx = dag.add_node(RiscOp::Mul, vec![expanded_g, local_grad], input_ty, None);
            Some(vec![(x, dx)])
        }
        RiscOp::Argmax { .. } | RiscOp::Argmin { .. } => {
            // Non-differentiable: integer-index outputs have zero gradient
            // almost everywhere and undefined gradient on ties. Returning
            // None here means grad_dag propagates a "cannot differentiate"
            // signal; `grad_dag_checked` below surfaces this as an explicit
            // error with a helpful message rather than a silent zero.
            None
        }

        // --- Movement ---
        RiscOp::Reshape { .. } => {
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            // chelis#616: a runtime input axis (a movement/reshape output
            // extent, whose name is not Load-declarable) restores through an
            // explicit Shape read of the forward input.
            let mut inputs = vec![g];
            let original_shape = restore_target(dag, forward, x, &input_ty.dims, &mut inputs);
            let dx = dag.add_node(
                RiscOp::Reshape {
                    new_shape: original_shape,
                },
                inputs,
                input_ty,
                None,
            );
            Some(vec![(x, dx)])
        }
        RiscOp::Permute { axes } => {
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            let inv = inverse_permutation(axes);
            let dx = dag.add_node(RiscOp::Permute { axes: inv }, vec![g], input_ty, None);
            Some(vec![(x, dx)])
        }
        RiscOp::Expand { axis, .. } => {
            // Two forward `Expand` shapes exist (see `verify.rs` C10):
            //
            //   * rank-INCREASING: a new axis is inserted at `axis`, so
            //     `output_rank == source_rank + 1`. The broadcast copies
            //     the source across the new axis; the adjoint is a `Sum`
            //     over that axis, which *removes* it and recovers the
            //     source rank exactly.
            //
            //   * SAME-RANK: an existing size-1 axis is broadcast to size
            //     n, so `output_rank == source_rank`. The adjoint must
            //     `Sum` over `axis` (which removes it, giving rank
            //     `source_rank - 1`) and then restore the collapsed size-1
            //     axis so the cotangent matches the source shape
            //     `[..., 1, ...]`.
            //
            // The previous rule emitted `Sum { axis }` with the SOURCE
            // type as the output for both shapes. For the same-rank case
            // that mislabels a rank `source_rank - 1` reduction as the
            // full rank-`source_rank` source type, so the cotangent flows
            // on with the wrong shape and a downstream elementwise op
            // fails verification with a dimension mismatch (issue #288:
            // `expand(scalar_to_tensor(c), 0, n)`, where the constant
            // lowers to a rank-1 size-1 `tensor[1]` source and the expand
            // is a same-rank `1 -> n` broadcast). Branch on the forward
            // shape and reshape the same-rank result back to the source.
            let x = node.inputs[0];
            let source_ty = forward.get(x).unwrap().output_type.clone();
            let source_rank = source_ty.dims.len();
            let output_rank = node.output_type.dims.len();
            let same_rank = output_rank == source_rank;
            // The gradient sum runs over the operand precision; use the
            // spec-default accumulator so the AD path tracks WS-A0 §5.7.1.
            let acc = RiscOp::default_reduce_sum_accumulator(source_ty.precision)
                .unwrap_or(source_ty.precision);

            // `Sum { axis }` over the cotangent removes `axis`. Its
            // resulting dims depend on the forward expand shape:
            //
            //   * RANK-INCREASING: the cotangent's `axis` is the inserted
            //     axis, which is NOT present in the source, so removing it
            //     yields the source dims unchanged.
            //   * SAME-RANK: the cotangent's `axis` IS the source's
            //     broadcast (size-1) axis, so removing it yields the
            //     source dims with that axis collapsed away; a follow-up
            //     reshape restores it to size 1.
            let summed_dims: Vec<DimInfo> = if same_rank {
                let mut dims = source_ty.dims.clone();
                if *axis < dims.len() {
                    dims.remove(*axis);
                }
                dims
            } else {
                source_ty.dims.clone()
            };
            let summed = dag.add_node(
                RiscOp::Sum {
                    axis: *axis,
                    accumulator: acc,
                },
                vec![g],
                TensorType {
                    dims: summed_dims.clone(),
                    precision: acc,
                },
                None,
            );

            // WS-A3: the §5.7.1 accumulator may be wider than the source
            // precision (bf16/f16 sum into f32); the gradient must be in
            // the source precision, so cast back when they differ. This
            // mirrors the `Sum` adjoint above.
            let summed_in_source_prec = if acc == source_ty.precision {
                summed
            } else {
                dag.add_node(
                    RiscOp::Cast {
                        new_precision: source_ty.precision,
                    },
                    vec![summed],
                    TensorType {
                        dims: summed_dims,
                        precision: source_ty.precision,
                    },
                    None,
                )
            };

            if same_rank {
                // Same-rank broadcast of a size-1 axis: restore the
                // collapsed size-1 axis so the cotangent matches the
                // source shape `[..., 1, ...]`.
                let mut inputs = vec![summed_in_source_prec];
                let new_shape = restore_target(dag, forward, x, &source_ty.dims, &mut inputs);
                let dx = dag.add_node(RiscOp::Reshape { new_shape }, inputs, source_ty, None);
                Some(vec![(x, dx)])
            } else {
                // Rank-increasing broadcast: the single `Sum` already
                // recovered the source rank (this also covers a rank-0
                // source, whose cotangent is rank 1 and sums to a scalar).
                Some(vec![(x, summed_in_source_prec)])
            }
        }
        RiscOp::OneHot { .. } => Some(vec![]),
        RiscOp::Pad { padding, .. } => {
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            // Shrink: bounds = [(before, before + dim_size), ...] for each axis.
            //
            // chelis#368: a Pad whose padded axis is concrete but whose
            // OTHER axes are runtime-derived symbolic dims (the differentiable
            // `concat` lowering pads each element along a concrete concat axis
            // and leaves every other axis `(0, 0)`) cannot bake a literal
            // `end` for those symbolic no-pad axes. For an unpadded axis the
            // Shrink is a full-axis identity (`(0, full_extent)`), so emit the
            // `SHRINK_TO_END` sentinel; `bind_symbolic_dims` resolves it to the
            // axis's runtime extent before eval. A symbolic dim on an axis that
            // WAS padded would still need a concrete extent — that stays a
            // hard error via `dim_size` (no silent guess).
            let mut shrink_inputs = vec![g];
            let mut bounds: Vec<(RtDim, RtDim)> = Vec::with_capacity(padding.len());
            for (axis, ((before, after), dim)) in
                padding.iter().zip(input_ty.dims.iter()).enumerate()
            {
                if matches!(dim, DimInfo::Named(_, None))
                    && before.as_lit() == Some(0)
                    && after.as_lit() == Some(0)
                {
                    bounds.push((RtDim::Lit(0), RtDim::ToEnd));
                } else if let (Some(before), Some(n)) = (before.as_lit(), static_dim(dim)) {
                    bounds.push((RtDim::Lit(before), RtDim::Lit(before + n)));
                } else {
                    // chelis#616: runtime bounds. `before` re-slots the
                    // forward bound scalar; `end = before + shape(x, axis)`
                    // is fresh runtime arithmetic over a Shape read of the
                    // forward input.
                    let start = reslot_bound(node, before, &mut shrink_inputs);
                    let before_scalar = bound_scalar(dag, node, before);
                    let extent = shape_scalar(dag, x, axis);
                    let end_scalar = int_scalar_binary(dag, RiscOp::Add, before_scalar, extent);
                    let slot = shrink_inputs.len();
                    shrink_inputs.push(end_scalar);
                    bounds.push((start, RtDim::Node(slot)));
                }
            }
            let dx = dag.add_node(RiscOp::Shrink { bounds }, shrink_inputs, input_ty, None);
            Some(vec![(x, dx)])
        }
        RiscOp::Shrink { bounds } => {
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            // Pad: for each axis, before = start, after = original_size - end.
            //
            // chelis#513 gap 3: a `(0, SHRINK_TO_END)` bound is the full-axis
            // identity sentinel (emitted by the Pad and Stride adjoints on
            // symbolic bystander axes), whose exact adjoint is no padding at
            // all -- `(0, 0)` -- with no axis size needed. Anything else
            // carrying the sentinel is a producing-pass bug: fail loud rather
            // than let the `dim_size(dim) - end` subtraction wrap. A symbolic
            // dim under a CONCRETE sub-range bound still needs the runtime
            // size for `after` and stays a hard error via `dim_size`.
            let mut pad_inputs = vec![g];
            let mut padding: Vec<(RtDim, RtDim)> = Vec::with_capacity(bounds.len());
            for (axis, ((start, end), dim)) in bounds.iter().zip(input_ty.dims.iter()).enumerate() {
                if matches!(end, RtDim::ToEnd) {
                    assert_eq!(
                        start.as_lit(),
                        Some(0),
                        "malformed ToEnd sentinel in shrink adjoint: nonzero start {start:?}"
                    );
                    padding.push((RtDim::Lit(0), RtDim::Lit(0)));
                } else if let (Some(start), Some(end), Some(n)) =
                    (start.as_lit(), end.as_lit(), static_dim(dim))
                {
                    padding.push((RtDim::Lit(start), RtDim::Lit(n - end)));
                } else {
                    // chelis#616: runtime bounds. `before` re-slots the
                    // forward start scalar; `after = shape(x, axis) - end`
                    // is fresh runtime arithmetic over a Shape read of the
                    // forward input.
                    let before = reslot_bound(node, start, &mut pad_inputs);
                    let end_scalar = bound_scalar(dag, node, end);
                    let extent = shape_scalar(dag, x, axis);
                    let precision = dag.get(end_scalar).unwrap().output_type.precision;
                    let neg_end =
                        dag.add_node(RiscOp::Neg, vec![end_scalar], scalar_int(precision), None);
                    let after_scalar = int_scalar_binary(dag, RiscOp::Add, extent, neg_end);
                    let slot = pad_inputs.len();
                    pad_inputs.push(after_scalar);
                    padding.push((before, RtDim::Node(slot)));
                }
            }
            let dx = dag.add_node(
                RiscOp::Pad { padding, fill: 0.0 },
                pad_inputs,
                input_ty,
                None,
            );
            Some(vec![(x, dx)])
        }
        RiscOp::Stride { strides } => {
            // Forward `stride(x, s)`: `out[i] = x[i .* s]` along each
            // axis, with `out` axis size `ceil(in / s)` (verify.rs C10).
            // The exact reverse-mode adjoint scatters each cotangent
            // element `g[i]` back to source position `i .* s` and zeros
            // every skipped slot (spec/05-risc-primitives.md §2.4: stride
            // adjoint = "appropriate expand/scatter").
            //
            // No new primitive is needed: the scatter is the separable
            // "insert `s - 1` zeros after each element, then trim to the
            // original size" upsample, which is exactly `pad` of a
            // freshly-inserted minor axis followed by `shrink`, applied
            // one axis at a time. For a single axis `a` with step `s_a`
            // and source size `n_a` (so the strided size is
            // `m_a = ceil(n_a / s_a)`):
            //
            //   reshape : [.., m_a, ..]      -> [.., m_a, 1, ..]
            //   pad     : [.., m_a, 1, ..]   -> [.., m_a, s_a, ..]   (axis a+1, after = s_a - 1)
            //   reshape : [.., m_a, s_a, ..] -> [.., m_a * s_a, ..]
            //   shrink  : [.., m_a * s_a, ..]-> [.., n_a, ..]        (axis a, [0, n_a))
            //
            // which places `g[.., i, ..]` at source index `i * s_a` and 0
            // at every `i * s_a + 1 ..= i * s_a + (s_a - 1)`. A step of 1
            // is the identity (`m_a == n_a`) and is skipped. All four ops
            // already have evaluator, C-backend, and adjoint coverage, so
            // this is sound under eval-vs-backend agreement and supports
            // higher-order AD.
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            let precision = input_ty.precision;
            // chelis#513 gap 3 / chelis#616: a STRIDED axis (step > 1) needs
            // its source size `n_a` for the trim bound and the strided size
            // `m_a` for the merge extent. When both are compile-time the
            // cascade stays fully static (the pre-#616 structural slice).
            // When either is runtime-symbolic (a windowed axis), the cascade
            // is built with runtime scalars instead: `m_a = shape(g, axis)`
            // read from the cotangent, the merge extent `m_a * step` as a
            // node-valued reshape target, and the trim bound
            // `(0, shape(x, axis))` read from the forward input. A symbolic
            // bystander axis (step <= 1) flows through untouched; its trim
            // bound is the `SHRINK_TO_END` full-axis sentinel.

            // Running cotangent; its dims mutate axis-by-axis from the
            // strided shape back toward the source shape.
            let mut cur = g;
            let mut cur_dims: Vec<DimInfo> = node.output_type.dims.clone();

            for (axis, step) in strides.iter().enumerate() {
                // chelis#616: a runtime (node-valued) stride step would need
                // a runtime-extent axis INSERTION (the pad's `step` axis);
                // that stays fail-closed loud. The oracle's literal steps
                // fold at lowering (`extract_int_for_dim` handles
                // cast-of-literal), so only a genuinely runtime step lands
                // here.
                let step = step
                    .as_lit()
                    .expect("a runtime (node-valued) stride STEP has no structural adjoint yet (chelis#616)");
                if step <= 1 {
                    // Identity stride on this axis: m_a == n_a already.
                    continue;
                }
                let n_a_static = static_dim(&input_ty.dims[axis]);
                let m_a_static = static_dim(&cur_dims[axis]);
                let runtime_axis = n_a_static.is_none() || m_a_static.is_none();

                // reshape: insert a size-1 axis after `axis`. On the runtime
                // path every non-static target extent is an explicit Shape
                // read of the source cotangent (wildcard dim names are not
                // stable symbols, so no Sym resolution is relied upon).
                let mut split_dims = cur_dims.clone();
                split_dims.insert(axis + 1, DimInfo::Lit(1));
                let (split_targets, split_inputs) = if runtime_axis {
                    let mut targets = Vec::with_capacity(split_dims.len());
                    let mut inputs = vec![cur];
                    for (j, dim) in split_dims.iter().enumerate() {
                        if j == axis + 1 {
                            targets.push(RtDim::Lit(1));
                            continue;
                        }
                        let src_axis = if j <= axis { j } else { j - 1 };
                        match static_dim(dim) {
                            Some(n) => targets.push(RtDim::Lit(n)),
                            None => {
                                let read = shape_scalar(dag, cur, src_axis);
                                let slot = inputs.len();
                                inputs.push(read);
                                targets.push(RtDim::Node(slot));
                            }
                        }
                    }
                    (targets, inputs)
                } else {
                    (reshape_target(&split_dims), vec![cur])
                };
                let split = dag.add_node(
                    RiscOp::Reshape {
                        new_shape: split_targets,
                    },
                    split_inputs,
                    TensorType {
                        dims: split_dims.clone(),
                        precision,
                    },
                    None,
                );

                // pad the new minor axis with (0, step - 1).
                let mut padding = vec![(RtDim::Lit(0), RtDim::Lit(0)); split_dims.len()];
                padding[axis + 1] = (RtDim::Lit(0), RtDim::Lit(step - 1));
                let mut padded_dims = split_dims.clone();
                padded_dims[axis + 1] = DimInfo::Lit(step);
                let padded = dag.add_node(
                    RiscOp::Pad { padding, fill: 0.0 },
                    vec![split],
                    TensorType {
                        dims: padded_dims.clone(),
                        precision,
                    },
                    None,
                );

                // reshape: merge axis and axis+1 back into one axis of
                // size m_a * step.
                let mut merged_dims = cur_dims.clone();
                let (merged_targets, merged_inputs) = if runtime_axis {
                    let m_a_read = shape_scalar(dag, cur, axis);
                    let step_const = dag.add_node(
                        RiscOp::synth_const(scalar_int(Prim::Int32).precision, step as f64),
                        vec![],
                        scalar_int(Prim::Int32),
                        None,
                    );
                    let merged_scalar = int_scalar_binary(dag, RiscOp::Mul, m_a_read, step_const);
                    merged_dims[axis] =
                        DimInfo::Named(format!("_rt_dim_{}_{axis}", merged_scalar.0), None);
                    let mut targets = Vec::with_capacity(merged_dims.len());
                    let mut inputs = vec![padded];
                    for (j, dim) in merged_dims.iter().enumerate() {
                        if j == axis {
                            let slot = inputs.len();
                            inputs.push(merged_scalar);
                            targets.push(RtDim::Node(slot));
                            continue;
                        }
                        match static_dim(dim) {
                            Some(n) => targets.push(RtDim::Lit(n)),
                            None => {
                                // Bystander runtime axis: same extent as the
                                // pre-split cotangent's axis `j`.
                                let read = shape_scalar(dag, cur, j);
                                let slot = inputs.len();
                                inputs.push(read);
                                targets.push(RtDim::Node(slot));
                            }
                        }
                    }
                    (targets, inputs)
                } else {
                    let m_a = m_a_static.expect("static path has a static strided size");
                    merged_dims[axis] = DimInfo::Lit(m_a * step);
                    (reshape_target(&merged_dims), vec![padded])
                };
                let merged = dag.add_node(
                    RiscOp::Reshape {
                        new_shape: merged_targets,
                    },
                    merged_inputs,
                    TensorType {
                        dims: merged_dims.clone(),
                        precision,
                    },
                    None,
                );

                // shrink axis back to [0, n_a). m_a * step >= n_a always
                // (ceil), so this is a valid trim of the trailing
                // overshoot from the final group. Symbolic bystander axes
                // take the full-axis identity sentinel (chelis#513 gap 3);
                // on the runtime path the trim bound reads the forward
                // input's extent (`shape(x, axis)`) — the sentinel would be
                // the MERGED extent, which overshoots.
                let mut shrink_inputs = vec![merged];
                let mut bounds: Vec<(RtDim, RtDim)> = merged_dims
                    .iter()
                    .map(|d| match d {
                        DimInfo::Named(_, None) => (RtDim::Lit(0), RtDim::ToEnd),
                        dim => (RtDim::Lit(0), RtDim::Lit(dim_size(dim))),
                    })
                    .collect();
                let mut trimmed_dims = merged_dims.clone();
                if let Some(n_a) = n_a_static {
                    bounds[axis] = (RtDim::Lit(0), RtDim::Lit(n_a));
                    trimmed_dims[axis] = DimInfo::Lit(n_a);
                } else {
                    let n_a_read = shape_scalar(dag, x, axis);
                    let slot = shrink_inputs.len();
                    shrink_inputs.push(n_a_read);
                    bounds[axis] = (RtDim::Lit(0), RtDim::Node(slot));
                    trimmed_dims[axis] = input_ty.dims[axis].clone();
                }
                let trimmed = dag.add_node(
                    RiscOp::Shrink { bounds },
                    shrink_inputs,
                    TensorType {
                        dims: trimmed_dims.clone(),
                        precision,
                    },
                    None,
                );

                cur = trimmed;
                cur_dims = trimmed_dims;
            }

            // The accumulated cotangent now has the source shape exactly;
            // label the terminal node with the original input type.
            if cur == g {
                // All steps were identity (every step <= 1): stride was a
                // no-op, so the adjoint is the cotangent unchanged.
                Some(vec![(x, g)])
            } else {
                debug_assert_eq!(
                    cur_dims.len(),
                    input_ty.dims.len(),
                    "stride adjoint must reconstruct the source rank",
                );
                debug_assert!(
                    cur_dims
                        .iter()
                        .zip(input_ty.dims.iter())
                        .all(|(got, want)| match (static_dim(got), static_dim(want)) {
                            (Some(g), Some(w)) => g == w,
                            // Runtime axes are reconstructed by runtime
                            // bounds; only static extents are checkable here.
                            _ => true,
                        }),
                    "stride adjoint must reconstruct the source shape: got {cur_dims:?}, want {:?}",
                    input_ty.dims,
                );
                Some(vec![(x, cur)])
            }
        }

        // --- Shape query ---
        RiscOp::Shape { .. } => {
            // `shape(x, axis)` reads only the input's shape metadata, not
            // its element values, so its output is constant w.r.t. those
            // values: the cotangent to the input tensor is exactly zero
            // (differentiable in the trivial constant sense per
            // chelis#558). Emit a zero of the INPUT's type (the cotangent
            // `g` has the scalar output's type, which differs from the
            // input's, so it is not reused here).
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            let zero = dag.add_node(
                RiscOp::synth_const(input_ty.precision, 0.0),
                vec![],
                input_ty,
                None,
            );
            Some(vec![(x, zero)])
        }

        // --- Memory ---
        RiscOp::Const { .. } => Some(vec![]),
        RiscOp::ConstTensor { .. } => Some(vec![]),
        RiscOp::Load { .. } => Some(vec![]),
        RiscOp::Store { .. } => {
            let x = node.inputs[0];
            Some(vec![(x, g)])
        }
        RiscOp::Realize => {
            let x = node.inputs[0];
            Some(vec![(x, g)])
        }

        // --- Cast ---
        RiscOp::Cast { .. } => {
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            if input_ty.precision.is_float() && node.output_type.precision.is_float() {
                let dx = dag.add_node(
                    RiscOp::Cast {
                        new_precision: input_ty.precision,
                    },
                    vec![g],
                    input_ty,
                    None,
                );
                Some(vec![(x, dx)])
            } else {
                let zero = dag.add_node(
                    RiscOp::synth_const(input_ty.precision, 0.0),
                    vec![],
                    input_ty,
                    None,
                );
                Some(vec![(x, zero)])
            }
        }
        // [05-OP-6] carries the `no_grad` rule: there is NO adjoint, and
        // emitting a zero here would be exactly the silent masking the
        // atom forbids. `grad_dag_checked`'s live-node scan already
        // rejected it with the structured `AdError`; this arm keeps the
        // unchecked entry point from inventing one.
        RiscOp::CastTrunc { .. } => None,
        RiscOp::FusedElem { .. } => {
            // Fused nodes should be un-fused before AD; gradient through fusion
            // is not yet supported.
            None
        }
        RiscOp::Copy => Some(vec![(node.inputs[0], g)]),
        RiscOp::Drop => None,
        RiscOp::Gather { axis } => {
            let values = node.inputs[0];
            let indices = node.inputs[1];
            let values_ty = forward.get(values).unwrap().output_type.clone();
            let zero = dag.add_node(
                RiscOp::synth_const(values_ty.precision, 0.0),
                vec![],
                values_ty.clone(),
                None,
            );
            let dvalues = dag.add_node(
                RiscOp::ScatterAdd { axis: *axis },
                vec![zero, indices, g],
                values_ty,
                None,
            );
            Some(vec![(values, dvalues)])
        }
        RiscOp::ScatterAdd { .. } => None,
        RiscOp::ReduceWindow {
            reducer,
            window_shape,
            strides,
        } => {
            // Reverse-mode adjoint of `reduce_window_*`
            // (spec/05-risc-primitives.md §2.3.1): lower to a single
            // `ReduceWindowGrad` node carrying the same window contract.
            // It scatters/overlap-adds (Sum/Mean) or routes-to-extreme
            // (Max/Min) the upstream cotangent `g` back to the input shape.
            //
            // No `Cast` is needed (unlike `Sum`): `reduce_window` does not
            // widen its accumulator — the forward output precision equals
            // the input precision — so `g` and `x` share a precision and
            // the adjoint carries the operand precision directly.
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            let din = dag.add_node(
                RiscOp::ReduceWindowGrad {
                    reducer: *reducer,
                    window_shape: window_shape.clone(),
                    strides: strides.clone(),
                },
                vec![x, g],
                input_ty,
                None,
            );
            Some(vec![(x, din)])
        }
        RiscOp::ReduceWindowGrad { .. } => {
            // Second-order AD through the windowed adjoint itself is not
            // defined; fail closed rather than synthesize a wrong adjoint.
            None
        }
        RiscOp::Scatter { .. } => {
            // Replace-scatter (last-write-wins) is non-differentiable.
            // `grad_dag_checked` rejects this case before reaching here
            // with a structured `AdError::NotSupported { op:
            // "scatter_replace", reason:
            // AdRejectionReason::NonDeterministicAtDuplicateIndices }`.
            // Returning `None` here keeps the legacy `grad_dag` path
            // fail-closed (rather than synthesizing a silent-zero or
            // arbitrary adjoint) for any caller that still uses the
            // un-checked entry point.
            None
        }
        RiscOp::ScatterElements { .. } => {
            // Element-wise replace-scatter is non-differentiable for the
            // same reason as `Scatter`; `grad_dag_checked` rejects it
            // before reaching here. `None` keeps the legacy path
            // fail-closed.
            None
        }
        RiscOp::BlasMatmul {
            batch_dims,
            m,
            n,
            k,
            ..
        } => {
            // Forward: Y = A @ B with A: [..., m, k], B: [..., k, n],
            // Y: [..., m, n]. Standard reverse-mode adjoint
            // (well-known matrix-multiply gradient):
            //   dA = g @ B^T   (shape [..., m, k])
            //   dB = A^T @ g   (shape [..., k, n])
            //
            // We express both via `BlasMatmul` nodes whose accumulator
            // follows the spec §5.7.1 default for the operand
            // precision. The transposes use `Permute` over the last
            // two axes so the rule supports batched matmuls
            // (`batch_dims.len() >= 0`) without special-casing rank-2
            // vs rank-N.
            let a_id = node.inputs[0];
            let b_id = node.inputs[1];
            let a_ty = forward.get(a_id).unwrap().output_type.clone();
            let b_ty = forward.get(b_id).unwrap().output_type.clone();
            debug_assert!(
                a_ty.precision == b_ty.precision,
                "blas matmul operand precisions must match (verifier-checked)",
            );
            let operand_prim = a_ty.precision;
            let adjoint_accumulator =
                RiscOp::default_matmul_accumulator(operand_prim).unwrap_or(operand_prim);

            // Build the last-two-axes transpose permutation. For
            // rank-2 inputs this is [1, 0]; for rank-N (N >= 2) it is
            // [0, 1, ..., N-3, N-1, N-2].
            let transpose_last_two = |rank: usize| -> Vec<usize> {
                assert!(rank >= 2, "blas matmul operand must be rank >= 2");
                let mut axes: Vec<usize> = (0..rank).collect();
                axes.swap(rank - 2, rank - 1);
                axes
            };

            // --- dA = g @ B^T ---
            //
            // B has type [..., k, n] -> B^T has type [..., n, k].
            let bt_axes = transpose_last_two(b_ty.dims.len());
            let mut bt_dims = b_ty.dims.clone();
            let bt_rank = bt_dims.len();
            bt_dims.swap(bt_rank - 2, bt_rank - 1);
            let bt_ty = TensorType {
                dims: bt_dims,
                precision: operand_prim,
            };
            let b_transposed =
                dag.add_node(RiscOp::Permute { axes: bt_axes }, vec![b_id], bt_ty, None);
            // dA shape = A's shape.
            let da_ty = a_ty.clone();
            let da_op = RiscOp::matmul_with_accumulator(
                batch_dims.clone(),
                m.clone(),
                k.clone(),
                n.clone(),
                operand_prim,
                adjoint_accumulator,
            )
            .expect(
                "adjoint matmul accumulator must be valid for the operand precision \
                 (spec/04-type-system.md §5.7.1 default for non-integer operand)",
            );
            let da = dag.add_node(da_op, vec![g, b_transposed], da_ty, None);

            // --- dB = A^T @ g ---
            //
            // A has type [..., m, k] -> A^T has type [..., k, m].
            let at_axes = transpose_last_two(a_ty.dims.len());
            let mut at_dims = a_ty.dims.clone();
            let at_rank = at_dims.len();
            at_dims.swap(at_rank - 2, at_rank - 1);
            let at_ty = TensorType {
                dims: at_dims,
                precision: operand_prim,
            };
            let a_transposed =
                dag.add_node(RiscOp::Permute { axes: at_axes }, vec![a_id], at_ty, None);
            let db_ty = b_ty.clone();
            let db_op = RiscOp::matmul_with_accumulator(
                batch_dims.clone(),
                k.clone(),
                n.clone(),
                m.clone(),
                operand_prim,
                adjoint_accumulator,
            )
            .expect(
                "adjoint matmul accumulator must be valid for the operand precision \
                 (spec/04-type-system.md §5.7.1 default for non-integer operand)",
            );
            let db = dag.add_node(db_op, vec![a_transposed, g], db_ty, None);

            Some(vec![(a_id, da), (b_id, db)])
        }
    }
}

fn inverse_permutation(axes: &[usize]) -> Vec<usize> {
    let mut inv = vec![0; axes.len()];
    for (new_pos, &old_pos) in axes.iter().enumerate() {
        inv[old_pos] = new_pos;
    }
    inv
}

fn dim_size(dim: &DimInfo) -> usize {
    match dim {
        DimInfo::Lit(n) => *n,
        DimInfo::Named(_, Some(n)) => *n,
        DimInfo::Named(name, None) => {
            panic!("cannot determine size for symbolic dimension `{name}`")
        }
    }
}

/// chelis#616: build a `Reshape` target from a static dim list. Known extents
/// become `Lit`, unbound named dims stay symbolic (`Sym`) so a Load-declared
/// bystander axis keeps resolving through `bind_symbolic_dims`.
fn reshape_target(dims: &[DimInfo]) -> Vec<RtDim> {
    dims.iter().map(RtDim::from_dim_info).collect()
}

/// The static extent of a dim, if it has one (a non-panicking [`dim_size`]).
fn static_dim(dim: &DimInfo) -> Option<usize> {
    match dim {
        DimInfo::Lit(n) => Some(*n),
        DimInfo::Named(_, Some(n)) => Some(*n),
        DimInfo::Named(_, None) => None,
    }
}

/// chelis#616: rank-0 integer scalar type for grad-built bound arithmetic.
fn scalar_int(precision: Prim) -> TensorType {
    TensorType {
        dims: Vec::new(),
        precision,
    }
}

/// chelis#616: a fresh `Shape(x, axis)` read — the runtime extent of `x`
/// along `axis` as a rank-0 int32 scalar (mirrors the Surf `shape()`
/// lowering's precision pin).
fn shape_scalar(dag: &mut Dag, x: NodeId, axis: usize) -> NodeId {
    dag.add_node(
        RiscOp::Shape { axis },
        vec![x],
        scalar_int(Prim::Int32),
        None,
    )
}

/// chelis#616: cast a rank-0 integer scalar to `precision` if needed.
fn cast_scalar(dag: &mut Dag, scalar: NodeId, precision: Prim) -> NodeId {
    let current = dag.get(scalar).unwrap().output_type.precision;
    if current == precision {
        return scalar;
    }
    dag.add_node(
        RiscOp::Cast {
            new_precision: precision,
        },
        vec![scalar],
        scalar_int(precision),
        None,
    )
}

/// chelis#616: a rank-0 integer binary op over two scalars, unified to the
/// wider of the two integer precisions (explicit casts; no implicit
/// promotion).
fn int_scalar_binary(dag: &mut Dag, op: RiscOp, a: NodeId, b: NodeId) -> NodeId {
    let pa = dag.get(a).unwrap().output_type.precision;
    let pb = dag.get(b).unwrap().output_type.precision;
    let precision = if pa == Prim::Int64 || pb == Prim::Int64 {
        Prim::Int64
    } else {
        Prim::Int32
    };
    let a = cast_scalar(dag, a, precision);
    let b = cast_scalar(dag, b, precision);
    dag.add_node(op, vec![a, b], scalar_int(precision), None)
}

/// chelis#616: materialize a forward movement bound as a rank-0 integer
/// scalar node. A `Node` bound reuses the forward op's bound-source scalar
/// directly (the backward DAG extends the forward one); a `Lit` becomes an
/// int32 Const. `ToEnd`/`Sym` never reach the runtime adjoint paths (the
/// sentinel is handled first and verify rejects `Sym` in movement bounds).
fn bound_scalar(dag: &mut Dag, forward_node: &DagNode, bound: &RtDim) -> NodeId {
    match bound {
        RtDim::Node(i) => forward_node.inputs[*i],
        RtDim::Lit(n) => dag.add_node(
            RiscOp::synth_const(scalar_int(Prim::Int32).precision, *n as f64),
            vec![],
            scalar_int(Prim::Int32),
            None,
        ),
        other => panic!("movement adjoint bound {other:?} has no runtime scalar form"),
    }
}

/// chelis#616: re-slot a forward movement bound onto a new (backward) op:
/// `Lit` passes through, a `Node` bound's forward scalar is appended to the
/// new op's `inputs` and referenced by its new absolute slot.
fn reslot_bound(forward_node: &DagNode, bound: &RtDim, inputs: &mut Vec<NodeId>) -> RtDim {
    match bound {
        RtDim::Node(i) => {
            let slot = inputs.len();
            inputs.push(forward_node.inputs[*i]);
            RtDim::Node(slot)
        }
        other => other.clone(),
    }
}

/// chelis#616: whether any forward `Load`'s type carries `name` — i.e. the
/// symbol is declarable from an input shape in every lane (including HIP,
/// which rejects `Shape` reads).
fn load_declares(forward: &Dag, name: &str) -> bool {
    forward.nodes().iter().any(|n| {
        matches!(&n.op, RiscOp::Load { .. })
            && n.output_type
                .dims
                .iter()
                .any(|d| matches!(d, DimInfo::Named(s, _) if s == name))
    })
}

/// chelis#616: build a `Reshape` target that restores `dims`, where `source`
/// is a forward tensor whose axes correspond 1:1 to `dims`. Static extents
/// become `Lit`; a Load-declared symbol stays `Sym` (resolvable in every
/// lane with no extra nodes, the pre-#616 behavior); any other symbolic dim
/// (a runtime movement/reshape extent, or a checker wildcard whose name is
/// not a stable symbol) becomes an explicit runtime `Shape` read on
/// `source`, appended to `inputs` and referenced as `Node`.
fn restore_target(
    dag: &mut Dag,
    forward: &Dag,
    source: NodeId,
    dims: &[DimInfo],
    inputs: &mut Vec<NodeId>,
) -> Vec<RtDim> {
    dims.iter()
        .enumerate()
        .map(|(axis, dim)| match dim {
            DimInfo::Lit(n) | DimInfo::Named(_, Some(n)) => RtDim::Lit(*n),
            DimInfo::Named(name, None) if load_declares(forward, name) => RtDim::Sym(name.clone()),
            DimInfo::Named(_, None) => {
                let read = shape_scalar(dag, source, axis);
                let slot = inputs.len();
                inputs.push(read);
                RtDim::Node(slot)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{TensorValue, eval_scalar};
    use std::collections::HashMap;

    // chelis#729 Phase 1: these fixtures verify the AUTODIFF machinery
    // against f64-precision finite differences, so the DAG type is f64.
    // (Pre-refactor they were typed f32 but evaluated at raw f64, the
    // chelis#717 shape; per-dtype finalize now makes an f32-typed DAG
    // genuinely round at f32, which the matrix suites cover.)
    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    fn scalar_f64() -> TensorType {
        TensorType {
            dims: vec![],
            precision: Prim::F64,
        }
    }

    /// Build a unary DAG: Load("x") -> op -> output.
    fn build_unary_dag(
        op_fn: impl FnOnce(&mut Dag, NodeId, &TensorType) -> NodeId,
    ) -> (Dag, NodeId, NodeId) {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let y = op_fn(&mut dag, x, &scalar_f64());
        (dag, x, y)
    }

    /// Build a binary DAG: Load("x"), Load("y") -> op -> output.
    fn build_binary_dag(
        op_fn: impl FnOnce(&mut Dag, NodeId, NodeId, &TensorType) -> NodeId,
    ) -> (Dag, NodeId, NodeId, NodeId) {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let y = dag.add_node(
            RiscOp::Load { name: "y".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let out = op_fn(&mut dag, x, y, &scalar_f64());
        (dag, x, y, out)
    }

    /// Compute both analytical (via AD) and numerical (via finite differences) gradients.
    fn finite_diff(
        dag: &Dag,
        output: NodeId,
        wrt: NodeId,
        input_name: &str,
        other_inputs: &[(&str, f64)],
        x0: f64,
        h: f64,
    ) -> (f64, f64) {
        let grad_result = grad_dag(dag, output, &[wrt]).unwrap();

        // Analytical gradient.
        let mut inputs: HashMap<String, f64> = HashMap::new();
        inputs.insert(input_name.to_string(), x0);
        for &(name, val) in other_inputs {
            inputs.insert(name.to_string(), val);
        }
        let vals = eval_scalar(&grad_result.dag, &inputs);
        let analytical = vals[&grad_result.grad_nodes[&wrt]];

        // Numerical gradient via central differences.
        inputs.insert(input_name.to_string(), x0 + h);
        let f_plus = eval_scalar(dag, &inputs)[&output];
        inputs.insert(input_name.to_string(), x0 - h);
        let f_minus = eval_scalar(dag, &inputs)[&output];
        let numerical = (f_plus - f_minus) / (2.0 * h);

        (analytical, numerical)
    }

    fn assert_grad_close(analytical: f64, numerical: f64) {
        assert!(
            (analytical - numerical).abs() < 1e-4,
            "gradient mismatch: analytical={analytical}, numerical={numerical}"
        );
    }

    // ---- Primitive adjoint tests ----

    #[test]
    fn grad_add() {
        let (dag, x, _y, out) = build_binary_dag(|dag, a, b, ty| {
            dag.add_node(RiscOp::Add, vec![a, b], ty.clone(), None)
        });
        let (a, n) = finite_diff(&dag, out, x, "x", &[("y", 3.0)], 2.0, 1e-5);
        assert_grad_close(a, n);
        assert!((a - 1.0).abs() < 1e-6); // d(x+y)/dx = 1
    }

    #[test]
    fn grad_mul() {
        let (dag, x, _y, out) = build_binary_dag(|dag, a, b, ty| {
            dag.add_node(RiscOp::Mul, vec![a, b], ty.clone(), None)
        });
        let (a, n) = finite_diff(&dag, out, x, "x", &[("y", 3.0)], 2.0, 1e-5);
        assert_grad_close(a, n);
        assert!((a - 3.0).abs() < 1e-6); // d(x*y)/dx = y = 3
    }

    #[test]
    fn grad_div_lhs() {
        let (dag, x, _y, out) = build_binary_dag(|dag, a, b, ty| {
            dag.add_node(RiscOp::Div, vec![a, b], ty.clone(), None)
        });
        let (a, n) = finite_diff(&dag, out, x, "x", &[("y", 4.0)], 2.0, 1e-5);
        assert_grad_close(a, n);
        assert!(
            (a - 0.25).abs() < 1e-6,
            "d(x/y)/dx at y=4 should be 0.25, got {a}"
        );
    }

    #[test]
    fn grad_div_rhs() {
        let (dag, _x, y, out) = build_binary_dag(|dag, a, b, ty| {
            dag.add_node(RiscOp::Div, vec![a, b], ty.clone(), None)
        });
        let (a, n) = finite_diff(&dag, out, y, "y", &[("x", 2.0)], 4.0, 1e-5);
        assert_grad_close(a, n);
        assert!(
            (a - (-0.125)).abs() < 1e-6,
            "d(x/y)/dy at x=2,y=4 should be -0.125, got {a}"
        );
    }

    #[test]
    fn grad_recip() {
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Recip, vec![a], ty.clone(), None));
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.0, 1e-5);
        assert_grad_close(a, n);
        assert!(
            (a - (-0.25)).abs() < 1e-6,
            "d(1/x)/dx at x=2 should be -0.25, got {a}"
        );
    }

    #[test]
    fn grad_neg() {
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Neg, vec![a], ty.clone(), None));
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.0, 1e-5);
        assert_grad_close(a, n);
        assert!((a - (-1.0)).abs() < 1e-6); // d(-x)/dx = -1
    }

    #[test]
    fn grad_exp() {
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Exp, vec![a], ty.clone(), None));
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 1.0, 1e-5);
        assert_grad_close(a, n);
        let expected = 1.0_f64.exp();
        assert!((a - expected).abs() < 1e-4); // d(exp(x))/dx = exp(x)
    }

    #[test]
    fn grad_log() {
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Log, vec![a], ty.clone(), None));
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.0, 1e-5);
        assert_grad_close(a, n);
        assert!((a - 0.5).abs() < 1e-4); // d(log(x))/dx = 1/x = 0.5
    }

    #[test]
    fn grad_sin() {
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Sin, vec![a], ty.clone(), None));
        let x0 = 1.0;
        let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
        assert_grad_close(a, n);
        let expected = x0.cos();
        assert!((a - expected).abs() < 1e-4);
    }

    #[test]
    fn grad_sqrt() {
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Sqrt, vec![a], ty.clone(), None));
        let x0 = 4.0;
        let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
        assert_grad_close(a, n);
        let expected = 1.0 / (2.0 * x0.sqrt()); // = 0.25
        assert!((a - expected).abs() < 1e-4);
    }

    #[test]
    fn grad_x_squared() {
        // f(x) = x * x, df/dx = 2x (tests accumulation: x used twice)
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let x_sq = dag.add_node(RiscOp::Mul, vec![x, x], scalar_f64(), None);

        let (a, n) = finite_diff(&dag, x_sq, x, "x", &[], 3.0, 1e-5);
        assert_grad_close(a, n);
        assert!((a - 6.0).abs() < 1e-4); // 2 * 3 = 6
    }

    #[test]
    fn grad_relu_positive() {
        // relu(x) = max(x, 0), d/dx = 1 when x > 0
        let (dag, x, out) = build_unary_dag(|dag, a, ty| {
            let zero = dag.add_node(
                RiscOp::synth_const(ty.precision, 0.0),
                vec![],
                ty.clone(),
                None,
            );
            dag.add_node(RiscOp::MaxElem, vec![a, zero], ty.clone(), None)
        });
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.0, 1e-5);
        assert_grad_close(a, n);
        assert!((a - 1.0).abs() < 1e-4);
    }

    #[test]
    fn grad_relu_negative() {
        // relu(x) = max(x, 0), d/dx = 0 when x < 0
        let (dag, x, out) = build_unary_dag(|dag, a, ty| {
            let zero = dag.add_node(
                RiscOp::synth_const(ty.precision, 0.0),
                vec![],
                ty.clone(),
                None,
            );
            dag.add_node(RiscOp::MaxElem, vec![a, zero], ty.clone(), None)
        });
        let (a, n) = finite_diff(&dag, out, x, "x", &[], -2.0, 1e-5);
        assert_grad_close(a, n);
        assert!(a.abs() < 1e-4);
    }

    #[test]
    fn grad_chain_exp_neg() {
        // f(x) = exp(-x), df/dx = -exp(-x)
        let (dag, x, out) = build_unary_dag(|dag, a, ty| {
            let neg = dag.add_node(RiscOp::Neg, vec![a], ty.clone(), None);
            dag.add_node(RiscOp::Exp, vec![neg], ty.clone(), None)
        });
        let x0 = 1.0;
        let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
        assert_grad_close(a, n);
        let expected = -(-x0).exp();
        assert!((a - expected).abs() < 1e-4);
    }

    #[test]
    fn grad_multi_input() {
        // f(x, y, z) = x*y + z
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let y = dag.add_node(
            RiscOp::Load { name: "y".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let z = dag.add_node(
            RiscOp::Load { name: "z".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let xy = dag.add_node(RiscOp::Mul, vec![x, y], scalar_f64(), None);
        let out = dag.add_node(RiscOp::Add, vec![xy, z], scalar_f64(), None);

        let grad_result = grad_dag(&dag, out, &[x, y, z]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), 2.0);
        inputs.insert("y".to_string(), 3.0);
        inputs.insert("z".to_string(), 5.0);
        let vals = eval_scalar(&grad_result.dag, &inputs);

        let dx = vals[&grad_result.grad_nodes[&x]];
        let dy = vals[&grad_result.grad_nodes[&y]];
        let dz = vals[&grad_result.grad_nodes[&z]];
        assert!((dx - 3.0).abs() < 1e-6); // d/dx(x*y + z) = y = 3
        assert!((dy - 2.0).abs() < 1e-6); // d/dy = x = 2
        assert!((dz - 1.0).abs() < 1e-6); // d/dz = 1
    }

    #[test]
    fn grad_accumulation_3x() {
        // f(x) = x + x + x, df/dx = 3
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let sum1 = dag.add_node(RiscOp::Add, vec![x, x], scalar_f64(), None);
        let out = dag.add_node(RiscOp::Add, vec![sum1, x], scalar_f64(), None);

        let grad_result = grad_dag(&dag, out, &[x]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), 7.0);
        let vals = eval_scalar(&grad_result.dag, &inputs);
        let dx = vals[&grad_result.grad_nodes[&x]];
        assert!((dx - 3.0).abs() < 1e-6);
    }

    #[test]
    fn grad_cmplt_zero() {
        let bool_ty = TensorType {
            dims: vec![],
            precision: Prim::Bool,
        };
        let (dag, x, _y, out) = build_binary_dag(|dag, a, b, _ty| {
            dag.add_node(RiscOp::CmpLt, vec![a, b], bool_ty.clone(), None)
        });
        assert!(
            grad_dag(&dag, out, &[x]).is_none(),
            "grad requires a scalar floating output and should reject bool outputs"
        );
    }

    #[test]
    fn grad_const_no_gradient() {
        let mut dag = Dag::new();
        let c = dag.add_node(
            RiscOp::synth_const(scalar_f64().precision, 5.0),
            vec![],
            scalar_f64(),
            None,
        );
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let out = dag.add_node(RiscOp::Mul, vec![x, c], scalar_f64(), None);

        let grad_result = grad_dag(&dag, out, &[x]).unwrap();
        assert!(!grad_result.grad_nodes.contains_key(&c));
    }

    #[test]
    fn grad_sum_expand() {
        // f(x) = sum(x, axis=0) for a 3-element vector, df/dx_i = 1
        use crate::dag::DimInfo;
        use crate::eval::{TensorValue, eval_tensor};

        let vec3_ty = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: chelis_types::types::Prim::F32,
        };
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            vec3_ty.clone(),
            None,
        );
        let out = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![x],
            scalar_f32(),
            None,
        );

        let grad_result = grad_dag(&dag, out, &[x]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
        );
        let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
        let grad = &vals[&grad_result.grad_nodes[&x]];
        assert_eq!(grad.to_f64_lossy_vec(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn grad_second_order() {
        // f(x) = x*x, f'(x) = 2x, f''(x) = 2
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let x_sq = dag.add_node(RiscOp::Mul, vec![x, x], scalar_f64(), None);

        // First derivative
        let first = grad_dag(&dag, x_sq, &[x]).unwrap();
        let dx_node = first.grad_nodes[&x];

        // Second derivative: differentiate the first derivative DAG
        let second = grad_dag(&first.dag, dx_node, &[x]).unwrap();
        let ddx_node = second.grad_nodes[&x];

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), 5.0);
        let vals = eval_scalar(&second.dag, &inputs);
        let ddx = vals[&ddx_node];
        assert!((ddx - 2.0).abs() < 1e-4); // d^2(x^2)/dx^2 = 2
    }

    #[test]
    fn grad_store_passthrough() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let exp_x = dag.add_node(RiscOp::Exp, vec![x], scalar_f64(), None);
        let out = dag.add_node(
            RiscOp::Store { name: "out".into() },
            vec![exp_x],
            scalar_f64(),
            None,
        );

        let (a, n) = finite_diff(&dag, out, x, "x", &[], 1.0, 1e-5);
        assert_grad_close(a, n);
    }

    #[test]
    fn grad_cast_passthrough() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let f64_ty = TensorType {
            dims: vec![],
            precision: chelis_types::types::Prim::F64,
        };
        let casted = dag.add_node(
            RiscOp::Cast {
                new_precision: chelis_types::types::Prim::F64,
            },
            vec![x],
            f64_ty,
            None,
        );
        let out = dag.add_node(RiscOp::Exp, vec![casted], scalar_f64(), None);

        let grad_result = grad_dag(&dag, out, &[x]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), 1.0);
        let vals = eval_scalar(&grad_result.dag, &inputs);
        let dx = vals[&grad_result.grad_nodes[&x]];
        let expected = 1.0_f64.exp();
        assert!((dx - expected).abs() < 1e-4);
    }

    #[test]
    fn grad_cast_to_int_is_zero() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let int_ty = TensorType {
            dims: vec![],
            precision: Prim::Int32,
        };
        let casted = dag.add_node(
            RiscOp::Cast {
                new_precision: Prim::Int32,
            },
            vec![x],
            int_ty.clone(),
            None,
        );
        let recast = dag.add_node(
            RiscOp::Cast {
                new_precision: Prim::F32,
            },
            vec![casted],
            scalar_f32(),
            None,
        );
        let out = dag.add_node(RiscOp::Add, vec![recast, recast], scalar_f32(), None);

        let grad_result = grad_dag(&dag, out, &[x]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), 1.0);
        let vals = eval_scalar(&grad_result.dag, &inputs);
        let dx = vals[&grad_result.grad_nodes[&x]];
        assert!(
            dx.abs() < 1e-6,
            "integer cast gradient should be zero, got {dx}"
        );
    }

    #[test]
    fn grad_reshape_roundtrip() {
        use crate::eval::{TensorValue, eval_tensor};

        let vec6_ty = TensorType {
            dims: vec![DimInfo::Lit(6)],
            precision: chelis_types::types::Prim::F32,
        };
        let mat23_ty = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            precision: chelis_types::types::Prim::F32,
        };

        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            vec6_ty.clone(),
            None,
        );
        let reshaped = dag.add_node(
            RiscOp::Reshape {
                new_shape: reshape_target(&mat23_ty.dims),
            },
            vec![x],
            mat23_ty.clone(),
            None,
        );
        // Sum all elements to get a scalar
        let sum0 = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![reshaped],
            TensorType {
                dims: vec![DimInfo::Lit(3)],
                precision: chelis_types::types::Prim::F32,
            },
            None,
        );
        let out = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![sum0],
            scalar_f32(),
            None,
        );

        let grad_result = grad_dag(&dag, out, &[x]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );
        let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
        let grad = &vals[&grad_result.grad_nodes[&x]];
        // d(sum(reshape(x)))/dx = ones
        assert_eq!(grad.to_f64_lossy_vec(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(grad.shape, vec![6]);
    }

    #[test]
    fn grad_permute_transpose() {
        use crate::eval::{TensorValue, eval_tensor};

        let mat23_ty = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            precision: chelis_types::types::Prim::F32,
        };
        let mat32_ty = TensorType {
            dims: vec![DimInfo::Lit(3), DimInfo::Lit(2)],
            precision: chelis_types::types::Prim::F32,
        };

        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            mat23_ty.clone(),
            None,
        );
        let transposed = dag.add_node(
            RiscOp::Permute { axes: vec![1, 0] },
            vec![x],
            mat32_ty.clone(),
            None,
        );
        // Sum all elements for a scalar output
        let sum0 = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![transposed],
            TensorType {
                dims: vec![DimInfo::Lit(2)],
                precision: chelis_types::types::Prim::F32,
            },
            None,
        );
        let out = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![sum0],
            scalar_f32(),
            None,
        );

        let grad_result = grad_dag(&dag, out, &[x]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );
        let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
        let grad = &vals[&grad_result.grad_nodes[&x]];
        // d(sum(permute(x)))/dx = ones, shape should be 2x3
        assert_eq!(grad.to_f64_lossy_vec(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(grad.shape, vec![2, 3]);
    }

    #[test]
    fn grad_expand_sum_roundtrip() {
        use crate::eval::{TensorValue, eval_tensor};

        let vec3_ty = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: chelis_types::types::Prim::F32,
        };
        let mat23_ty = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            precision: chelis_types::types::Prim::F32,
        };

        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            vec3_ty.clone(),
            None,
        );
        let expanded = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: crate::dag::DimExpr::Concrete(2),
            },
            vec![x],
            mat23_ty.clone(),
            None,
        );
        // Sum back to scalar
        let sum0 = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![expanded],
            vec3_ty.clone(),
            None,
        );
        let out = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![sum0],
            scalar_f32(),
            None,
        );

        let grad_result = grad_dag(&dag, out, &[x]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
        );
        let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
        let grad = &vals[&grad_result.grad_nodes[&x]];
        // expand by 2 then sum -> each element counted twice -> gradient = 2
        assert_eq!(grad.to_f64_lossy_vec(), vec![2.0, 2.0, 2.0]);
        assert_eq!(grad.shape, vec![3]);
    }

    #[test]
    fn grad_mul_by_const() {
        // f(x) = 5 * x, df/dx = 5
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let five = dag.add_node(
            RiscOp::synth_const(scalar_f64().precision, 5.0),
            vec![],
            scalar_f64(),
            None,
        );
        let out = dag.add_node(RiscOp::Mul, vec![five, x], scalar_f64(), None);

        let (a, n) = finite_diff(&dag, out, x, "x", &[], 3.0, 1e-5);
        assert_grad_close(a, n);
        assert!((a - 5.0).abs() < 1e-4);
    }

    #[test]
    fn grad_log_chain() {
        // f(x) = log(x^2) = 2*log(x), df/dx = 2/x
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let x_sq = dag.add_node(RiscOp::Mul, vec![x, x], scalar_f64(), None);
        let out = dag.add_node(RiscOp::Log, vec![x_sq], scalar_f64(), None);

        let x0 = 3.0;
        let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
        assert_grad_close(a, n);
        let expected = 2.0 / x0;
        assert!((a - expected).abs() < 1e-4);
    }

    #[test]
    fn grad_sin_chain() {
        // f(x) = sin(x^2), df/dx = 2x * cos(x^2)
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let x_sq = dag.add_node(RiscOp::Mul, vec![x, x], scalar_f64(), None);
        let out = dag.add_node(RiscOp::Sin, vec![x_sq], scalar_f64(), None);

        let x0 = 1.5;
        let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
        assert_grad_close(a, n);
    }

    #[test]
    fn grad_max_elem_symmetric() {
        // f(x, y) = max(x, y), test that d/dy = 1 when y > x
        let (dag, _x, y, out) = build_binary_dag(|dag, a, b, ty| {
            dag.add_node(RiscOp::MaxElem, vec![a, b], ty.clone(), None)
        });
        let (a, n) = finite_diff(&dag, out, y, "y", &[("x", 1.0)], 5.0, 1e-5);
        assert_grad_close(a, n);
        assert!((a - 1.0).abs() < 1e-4); // y > x, so d/dy = 1
    }

    #[test]
    fn grad_no_wrt_returns_empty() {
        let (dag, _x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Exp, vec![a], ty.clone(), None));
        // Ask for gradient wrt a node that doesn't exist
        let grad_result = grad_dag(&dag, out, &[NodeId(999)]).unwrap();
        assert!(grad_result.grad_nodes.is_empty());
    }

    #[test]
    fn grad_inverse_permutation() {
        assert_eq!(inverse_permutation(&[2, 0, 1]), vec![1, 2, 0]);
        assert_eq!(inverse_permutation(&[0, 1]), vec![0, 1]);
        assert_eq!(inverse_permutation(&[1, 0]), vec![1, 0]);
    }

    #[test]
    fn grad_result_is_verified_and_rooted() {
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Exp, vec![a], ty.clone(), None));
        let grad_result = grad_dag(&dag, out, &[x]).unwrap();
        assert!(grad_result.dag.is_root(out));
        assert!(grad_result.dag.is_root(grad_result.grad_nodes[&x]));
        assert!(
            crate::verify::verify(&grad_result.dag).is_empty(),
            "grad result should be structurally valid"
        );
    }

    // ================================================================
    // ADVERSARIAL TESTS — Categories A through E
    // ================================================================

    // ---- Category A: Failure modes ----

    #[test]
    fn adv_empty_dag() {
        // An empty DAG returns None
        let dag = Dag::new();
        assert!(
            grad_dag(&dag, NodeId(0), &[]).is_none(),
            "empty DAG should return None"
        );
    }

    #[test]
    fn adv_output_node_doesnt_exist() {
        let mut dag = Dag::new();
        let _x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f64(),
            None,
        );
        // NodeId(999) doesn't exist — should return None
        assert!(
            grad_dag(&dag, NodeId(999), &[]).is_none(),
            "non-existent output should return None"
        );
    }

    #[test]
    fn adv_non_scalar_output_rejected() {
        let vec2_ty = TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::F32,
        };
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            vec2_ty.clone(),
            None,
        );
        let y = dag.add_node(RiscOp::Exp, vec![x], vec2_ty, None);
        assert!(
            grad_dag(&dag, y, &[x]).is_none(),
            "non-scalar outputs should be rejected without an explicit seed gradient"
        );
    }

    #[test]
    fn adv_wrt_non_leaf_node() {
        // Ask for gradient wrt a non-leaf node (e.g., an Add node)
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let y = dag.add_node(
            RiscOp::Load { name: "y".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let add_node = dag.add_node(RiscOp::Add, vec![x, y], scalar_f64(), None);
        let out = dag.add_node(RiscOp::Exp, vec![add_node], scalar_f64(), None);

        // wrt the Add node, not a leaf
        let grad_result = grad_dag(&dag, out, &[add_node]).unwrap();
        // This should work -- it's asking for d(out)/d(add_node).
        // The answer should be exp(x+y), i.e., exp evaluated at the add node.
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), 1.0);
        inputs.insert("y".to_string(), 2.0);
        let vals = eval_scalar(&grad_result.dag, &inputs);
        let d_add = vals[&grad_result.grad_nodes[&add_node]];
        let expected = (1.0_f64 + 2.0).exp();
        assert!(
            (d_add - expected).abs() < 1e-4,
            "wrt non-leaf: expected {expected}, got {d_add}"
        );
    }

    // ---- Category B: Numerical correctness at specific values ----

    #[test]
    fn adv_exp_at_2() {
        // d(exp(x))/dx at x=2.0 should be exp(2) ≈ 7.389
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Exp, vec![a], ty.clone(), None));
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.0, 1e-5);
        assert_grad_close(a, n);
        let expected = 2.0_f64.exp();
        assert!(
            (a - expected).abs() < 1e-4,
            "d(exp(x))/dx at x=2: expected {expected}, got {a}"
        );
    }

    #[test]
    fn adv_log_at_0_5() {
        // d(log(x))/dx at x=0.5 should be 2.0
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Log, vec![a], ty.clone(), None));
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 0.5, 1e-5);
        assert_grad_close(a, n);
        assert!(
            (a - 2.0).abs() < 1e-4,
            "d(log(x))/dx at x=0.5: expected 2.0, got {a}"
        );
    }

    #[test]
    fn adv_sin_at_1() {
        // d(sin(x))/dx at x=1.0 should be cos(1) ≈ 0.5403
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Sin, vec![a], ty.clone(), None));
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 1.0, 1e-5);
        assert_grad_close(a, n);
        let expected = 1.0_f64.cos();
        assert!(
            (a - expected).abs() < 1e-4,
            "d(sin(x))/dx at x=1: expected {expected}, got {a}"
        );
    }

    #[test]
    fn adv_sqrt_at_4() {
        // d(sqrt(x))/dx at x=4.0 should be 0.25
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Sqrt, vec![a], ty.clone(), None));
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 4.0, 1e-5);
        assert_grad_close(a, n);
        assert!(
            (a - 0.25).abs() < 1e-4,
            "d(sqrt(x))/dx at x=4: expected 0.25, got {a}"
        );
    }

    #[test]
    fn adv_max_x_gt_y() {
        // d(max(x,y))/dx at x=3, y=1 should be 1.0
        let (dag, x, _y, out) = build_binary_dag(|dag, a, b, ty| {
            dag.add_node(RiscOp::MaxElem, vec![a, b], ty.clone(), None)
        });
        let (a, n) = finite_diff(&dag, out, x, "x", &[("y", 1.0)], 3.0, 1e-5);
        assert_grad_close(a, n);
        assert!(
            (a - 1.0).abs() < 1e-4,
            "d(max(x,y))/dx at x=3,y=1: expected 1.0, got {a}"
        );
    }

    #[test]
    fn adv_max_x_lt_y() {
        // d(max(x,y))/dx at x=1, y=3 should be 0.0
        let (dag, x, _y, out) = build_binary_dag(|dag, a, b, ty| {
            dag.add_node(RiscOp::MaxElem, vec![a, b], ty.clone(), None)
        });
        let (a, n) = finite_diff(&dag, out, x, "x", &[("y", 3.0)], 1.0, 1e-5);
        assert_grad_close(a, n);
        assert!(
            a.abs() < 1e-4,
            "d(max(x,y))/dx at x=1,y=3: expected 0.0, got {a}"
        );
    }

    #[test]
    fn adv_max_equal_inputs() {
        // Phase 0 tie-break convention routes the gradient to the left-hand side.
        let (dag, x, y, out) = build_binary_dag(|dag, a, b, ty| {
            dag.add_node(RiscOp::MaxElem, vec![a, b], ty.clone(), None)
        });
        let grad_result = grad_dag(&dag, out, &[x, y]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), 2.0);
        inputs.insert("y".to_string(), 2.0);
        let vals = eval_scalar(&grad_result.dag, &inputs);
        let dx = vals[&grad_result.grad_nodes[&x]];
        let dy = vals[&grad_result.grad_nodes[&y]];
        assert!(
            (dx - 1.0).abs() < 1e-6,
            "expected left tie gradient 1.0, got {dx}"
        );
        assert!(dy.abs() < 1e-6, "expected right tie gradient 0.0, got {dy}");
    }

    // ---- Category C: Composed operations ----

    #[test]
    fn adv_exp_log_identity() {
        // d(exp(log(x)))/dx should be 1.0 (since exp(log(x)) = x)
        let (dag, x, out) = build_unary_dag(|dag, a, ty| {
            let log_x = dag.add_node(RiscOp::Log, vec![a], ty.clone(), None);
            dag.add_node(RiscOp::Exp, vec![log_x], ty.clone(), None)
        });
        let x0 = 2.7;
        let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
        assert_grad_close(a, n);
        assert!(
            (a - 1.0).abs() < 1e-3,
            "d(exp(log(x)))/dx: expected 1.0, got {a}"
        );
    }

    #[test]
    fn adv_x_times_exp_x() {
        // d(x * exp(x))/dx = (1+x) * exp(x)
        let (dag, x, out) = build_unary_dag(|dag, a, ty| {
            let exp_x = dag.add_node(RiscOp::Exp, vec![a], ty.clone(), None);
            dag.add_node(RiscOp::Mul, vec![a, exp_x], ty.clone(), None)
        });
        let x0 = 1.5;
        let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
        assert_grad_close(a, n);
        let expected = (1.0 + x0) * x0.exp();
        assert!(
            (a - expected).abs() < 1e-3,
            "d(x*exp(x))/dx at x={x0}: expected {expected}, got {a}"
        );
    }

    #[test]
    fn adv_sin_x_squared() {
        // d(sin(x^2))/dx = 2x * cos(x^2)
        let (dag, x, out) = build_unary_dag(|dag, a, ty| {
            let x_sq = dag.add_node(RiscOp::Mul, vec![a, a], ty.clone(), None);
            dag.add_node(RiscOp::Sin, vec![x_sq], ty.clone(), None)
        });
        let x0 = 1.5;
        let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
        assert_grad_close(a, n);
        let expected = 2.0 * x0 * (x0 * x0).cos();
        assert!(
            (a - expected).abs() < 1e-3,
            "d(sin(x^2))/dx at x={x0}: expected {expected}, got {a}"
        );
    }

    #[test]
    fn adv_sigmoid_gradient() {
        // sigmoid(x) = 1 / (1 + exp(-x))
        // Lowered as: div(1, add(1, exp(neg(x))))
        // d(sigmoid)/dx = sigmoid * (1 - sigmoid)
        let (dag, x, out) = build_unary_dag(|dag, a, ty| {
            let neg_x = dag.add_node(RiscOp::Neg, vec![a], ty.clone(), None);
            let exp_neg_x = dag.add_node(RiscOp::Exp, vec![neg_x], ty.clone(), None);
            let one = dag.add_node(
                RiscOp::synth_const(ty.precision, 1.0),
                vec![],
                ty.clone(),
                None,
            );
            let denom = dag.add_node(RiscOp::Add, vec![one, exp_neg_x], ty.clone(), None);
            // div(1, denom) = 1 * recip(denom) = exp(-log(denom))
            crate::tier2::lower_div(dag, one, denom, ty, None)
        });
        let x0 = -0.3;
        let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
        assert_grad_close(a, n);
        let sig = 1.0 / (1.0 + (-x0).exp());
        let expected = sig * (1.0 - sig);
        assert!(
            (a - expected).abs() < 1e-3,
            "sigmoid gradient at x={x0}: expected {expected}, got {a}"
        );
    }

    // ---- Category D: Edge cases ----

    #[test]
    fn adv_all_zero_grad_out() {
        // If grad_out is Const(0) everywhere (degenerate), gradient should still compute.
        // This happens when the output doesn't depend on x but we ask for dx anyway.
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let c = dag.add_node(
            RiscOp::synth_const(scalar_f64().precision, 42.0),
            vec![],
            scalar_f64(),
            None,
        );
        // output is a constant, doesn't depend on x
        let grad_result = grad_dag(&dag, c, &[x]).unwrap();
        // x might not even be in grad_nodes since no path from c to x
        if let Some(&gx) = grad_result.grad_nodes.get(&x) {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), 7.0);
            let vals = eval_scalar(&grad_result.dag, &inputs);
            let dx = vals[&gx];
            assert!(
                dx.abs() < 1e-10,
                "gradient of constant wrt x should be 0, got {dx}"
            );
        }
        // If x is not in grad_nodes, that's also fine (no path)
    }

    #[test]
    fn adv_node_used_5_times() {
        // add(add(add(add(x, x), x), x), x) = 5x, gradient should be 5
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let s1 = dag.add_node(RiscOp::Add, vec![x, x], scalar_f64(), None);
        let s2 = dag.add_node(RiscOp::Add, vec![s1, x], scalar_f64(), None);
        let s3 = dag.add_node(RiscOp::Add, vec![s2, x], scalar_f64(), None);
        let out = dag.add_node(RiscOp::Add, vec![s3, x], scalar_f64(), None);

        let grad_result = grad_dag(&dag, out, &[x]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), 2.7);
        let vals = eval_scalar(&grad_result.dag, &inputs);
        let dx = vals[&grad_result.grad_nodes[&x]];
        assert!(
            (dx - 5.0).abs() < 1e-6,
            "5x gradient: expected 5.0, got {dx}"
        );
    }

    #[test]
    fn adv_deep_neg_chain_even() {
        // neg(neg(neg(neg(x)))) = x, gradient = 1.0
        let (dag, x, out) = build_unary_dag(|dag, a, ty| {
            let n1 = dag.add_node(RiscOp::Neg, vec![a], ty.clone(), None);
            let n2 = dag.add_node(RiscOp::Neg, vec![n1], ty.clone(), None);
            let n3 = dag.add_node(RiscOp::Neg, vec![n2], ty.clone(), None);
            dag.add_node(RiscOp::Neg, vec![n3], ty.clone(), None)
        });
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.7, 1e-5);
        assert_grad_close(a, n);
        assert!(
            (a - 1.0).abs() < 1e-6,
            "4x neg gradient: expected 1.0, got {a}"
        );
    }

    #[test]
    fn adv_deep_neg_chain_odd() {
        // neg(neg(neg(x))) = -x, gradient = -1.0
        let (dag, x, out) = build_unary_dag(|dag, a, ty| {
            let n1 = dag.add_node(RiscOp::Neg, vec![a], ty.clone(), None);
            let n2 = dag.add_node(RiscOp::Neg, vec![n1], ty.clone(), None);
            dag.add_node(RiscOp::Neg, vec![n2], ty.clone(), None)
        });
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.7, 1e-5);
        assert_grad_close(a, n);
        assert!(
            (a - (-1.0)).abs() < 1e-6,
            "3x neg gradient: expected -1.0, got {a}"
        );
    }

    // ---- Category E: MaxReduce adjoint (critical for MNIST) ----

    #[test]
    fn adv_max_reduce_gradient() {
        // Build a 2x3 tensor, apply max_reduce on axis 0, then sum to scalar.
        // Check gradient via finite differences on each element.
        use crate::eval::{TensorValue, eval_tensor};

        let mat23_ty = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            precision: chelis_types::types::Prim::F64,
        };
        let vec3_ty = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: chelis_types::types::Prim::F64,
        };

        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            mat23_ty.clone(),
            None,
        );
        let maxr = dag.add_node(
            RiscOp::MaxReduce { axis: 0 },
            vec![x],
            vec3_ty.clone(),
            None,
        );
        let out = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F64,
            },
            vec![maxr],
            scalar_f64(),
            None,
        );

        let grad_result = grad_dag(&dag, out, &[x]).unwrap();

        // Input: [[1, 5, 3], [4, 2, 6]]
        // max_reduce(axis=0) = [4, 5, 6]
        // sum = 15
        // Gradient: 1 where element equals max along axis 0, 0 otherwise
        // Expected gradient: [[0, 1, 0], [1, 0, 1]]
        let input_data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![2, 3], input_data.clone()),
        );
        let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
        let grad = &vals[&grad_result.grad_nodes[&x]];

        let expected_grad = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        for (i, (actual, expected)) in grad
            .to_f64_lossy_vec()
            .iter()
            .zip(expected_grad.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() < 1e-6,
                "max_reduce gradient at index {i}: expected {expected}, got {actual}"
            );
        }

        // Also verify via finite differences on each element
        let h = 1e-5;
        for elem_idx in 0..6 {
            let mut data_plus = input_data.clone();
            data_plus[elem_idx] += h;
            let mut data_minus = input_data.clone();
            data_minus[elem_idx] -= h;

            let mut inputs_plus = HashMap::new();
            inputs_plus.insert(
                "x".to_string(),
                TensorValue::from_vec(vec![2, 3], data_plus),
            );
            let vals_plus = eval_tensor(&dag, &inputs_plus).unwrap();
            let f_plus = vals_plus[&out].to_f64_lossy_vec()[0];

            let mut inputs_minus = HashMap::new();
            inputs_minus.insert(
                "x".to_string(),
                TensorValue::from_vec(vec![2, 3], data_minus),
            );
            let vals_minus = eval_tensor(&dag, &inputs_minus).unwrap();
            let f_minus = vals_minus[&out].to_f64_lossy_vec()[0];

            let numerical = (f_plus - f_minus) / (2.0 * h);
            let analytical = grad.to_f64_lossy_vec()[elem_idx];
            assert!(
                (analytical - numerical).abs() < 1e-3,
                "max_reduce finite diff at elem {elem_idx}: analytical={analytical}, numerical={numerical}"
            );
        }
    }

    #[test]
    fn adv_max_reduce_with_ties() {
        // What happens when multiple elements tie for max?
        // Input: [[3, 3], [3, 1]]
        // max_reduce(axis=0) = [3, 3]
        // The gradient should flow to ALL elements equal to max (eq mask).
        // So row 0 col 0 AND row 1 col 0 should both get gradient.
        use crate::eval::{TensorValue, eval_tensor};

        let mat22_ty = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(2)],
            precision: chelis_types::types::Prim::F32,
        };
        let vec2_ty = TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: chelis_types::types::Prim::F32,
        };

        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            mat22_ty.clone(),
            None,
        );
        let maxr = dag.add_node(
            RiscOp::MaxReduce { axis: 0 },
            vec![x],
            vec2_ty.clone(),
            None,
        );
        let out = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![maxr],
            scalar_f32(),
            None,
        );

        let grad_result = grad_dag(&dag, out, &[x]).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![2, 2], vec![3.0, 3.0, 3.0, 1.0]),
        );
        let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
        let grad = &vals[&grad_result.grad_nodes[&x]];

        // With ties, eq mask gives 1 for ALL elements equal to max.
        // Column 0: both rows are 3 (the max), so both get gradient 1.
        // Column 1: row 0 is 3 (the max), row 1 is 1, so only row 0 gets gradient.
        // Expected: [[1, 1], [1, 0]]
        //
        // BUT: The gradient sum for column 0 is 2, not 1! This means
        // the gradient is INFLATED when there are ties. The correct subgradient
        // should only assign gradient to ONE element per reduction, not all ties.
        let _expected_if_all_ties = [1.0, 1.0, 1.0, 0.0];
        let grad_col0 = grad.to_f64_lossy_vec()[0] + grad.to_f64_lossy_vec()[2]; // should be 1.0 but will be 2.0
        eprintln!(
            "FINDING: max_reduce with ties: gradient = {:?}, column 0 sum = {grad_col0}",
            grad.to_f64_lossy_vec()
        );
        if (grad_col0 - 2.0).abs() < 1e-6 {
            eprintln!(
                "CONFIRMED: max_reduce gradient is DOUBLED when elements tie. \
                 The eq mask gives 1 to ALL tied elements, but the gradient should \
                 only flow to one element (or be divided among tied elements)."
            );
        }
    }

    #[test]
    fn adv_max_reduce_axis1() {
        // Test max_reduce along axis 1 (the last axis)
        use crate::eval::{TensorValue, eval_tensor};

        let mat23_ty = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            precision: chelis_types::types::Prim::F64,
        };
        let vec2_ty = TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: chelis_types::types::Prim::F64,
        };

        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            mat23_ty.clone(),
            None,
        );
        let maxr = dag.add_node(
            RiscOp::MaxReduce { axis: 1 },
            vec![x],
            vec2_ty.clone(),
            None,
        );
        let out = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F64,
            },
            vec![maxr],
            scalar_f64(),
            None,
        );

        let grad_result = grad_dag(&dag, out, &[x]).unwrap();

        // Input: [[1, 5, 3], [4, 2, 6]]
        // max_reduce(axis=1) = [5, 6]
        // sum = 11
        // Gradient: 1 at positions of max along axis 1
        // Row 0: max is 5 at col 1 -> [0, 1, 0]
        // Row 1: max is 6 at col 2 -> [0, 0, 1]
        // Expected: [[0, 1, 0], [0, 0, 1]]
        let input_data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![2, 3], input_data.clone()),
        );
        let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
        let grad = &vals[&grad_result.grad_nodes[&x]];

        let expected_grad = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for (i, (actual, expected)) in grad
            .to_f64_lossy_vec()
            .iter()
            .zip(expected_grad.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() < 1e-6,
                "max_reduce axis=1 gradient at index {i}: expected {expected}, got {actual}"
            );
        }

        // Verify via finite differences
        let h = 1e-5;
        for elem_idx in 0..6 {
            let mut data_plus = input_data.clone();
            data_plus[elem_idx] += h;
            let mut data_minus = input_data.clone();
            data_minus[elem_idx] -= h;

            let mut inputs_plus = HashMap::new();
            inputs_plus.insert(
                "x".to_string(),
                TensorValue::from_vec(vec![2, 3], data_plus),
            );
            let vals_plus = eval_tensor(&dag, &inputs_plus).unwrap();
            let f_plus = vals_plus[&out].to_f64_lossy_vec()[0];

            let mut inputs_minus = HashMap::new();
            inputs_minus.insert(
                "x".to_string(),
                TensorValue::from_vec(vec![2, 3], data_minus),
            );
            let vals_minus = eval_tensor(&dag, &inputs_minus).unwrap();
            let f_minus = vals_minus[&out].to_f64_lossy_vec()[0];

            let numerical = (f_plus - f_minus) / (2.0 * h);
            let analytical = grad.to_f64_lossy_vec()[elem_idx];
            assert!(
                (analytical - numerical).abs() < 1e-3,
                "max_reduce axis=1 finite diff at elem {elem_idx}: analytical={analytical}, numerical={numerical}"
            );
        }
    }

    #[test]
    fn adv_log_at_negative_input() {
        // log(x) at x=-1 is NaN. What does the gradient look like?
        // This shouldn't crash.
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Log, vec![a], ty.clone(), None));
        let grad_result = grad_dag(&dag, out, &[x]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), -1.0);
        let vals = eval_scalar(&grad_result.dag, &inputs);
        let dx = vals[&grad_result.grad_nodes[&x]];
        // Should be NaN or -1 (1/x = 1/-1 = -1, but log(-1) is NaN so the
        // chain involves NaN). Just verify no crash.
        eprintln!("log gradient at x=-1: {dx} (expected NaN or similar)");
    }

    #[test]
    fn adv_sqrt_at_zero() {
        // d(sqrt(x))/dx = 1/(2*sqrt(x)), at x=0 this is infinity.
        // Just verify no crash.
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Sqrt, vec![a], ty.clone(), None));
        let grad_result = grad_dag(&dag, out, &[x]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), 0.0);
        let vals = eval_scalar(&grad_result.dag, &inputs);
        let dx = vals[&grad_result.grad_nodes[&x]];
        eprintln!("sqrt gradient at x=0: {dx} (expected inf)");
    }

    #[test]
    fn adv_div_gradient() {
        // div(a, b) lowers to RiscOp::Div(a, b); the closed-form
        // adjoint pair is da = g / b, db = -g * y / b (where y =
        // a / b is the forward output). At a=6, b=3 the expected
        // gradients are d/da = 1/3, d/db = -6/9 = -2/3.
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::Load { name: "a".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let b = dag.add_node(
            RiscOp::Load { name: "b".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let out = crate::tier2::lower_div(&mut dag, a, b, &scalar_f64(), None);

        // Test d/da
        let (ana_a, num_a) = finite_diff(&dag, out, a, "a", &[("b", 3.0)], 6.0, 1e-5);
        assert_grad_close(ana_a, num_a);
        assert!(
            (ana_a - 1.0 / 3.0).abs() < 1e-3,
            "d(a/b)/da at a=6,b=3: expected {}, got {ana_a}",
            1.0 / 3.0
        );

        // Test d/db
        let (ana_b, num_b) = finite_diff(&dag, out, b, "b", &[("a", 6.0)], 3.0, 1e-5);
        assert_grad_close(ana_b, num_b);
        let expected_db = -6.0 / 9.0;
        assert!(
            (ana_b - expected_db).abs() < 1e-3,
            "d(a/b)/db at a=6,b=3: expected {expected_db}, got {ana_b}"
        );
    }

    #[test]
    fn adv_pad_shrink_gradient() {
        // Test pad adjoint: pad then sum, check gradient
        use crate::eval::{TensorValue, eval_tensor};

        let vec3_ty = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: chelis_types::types::Prim::F32,
        };
        let vec5_ty = TensorType {
            dims: vec![DimInfo::Lit(5)],
            precision: chelis_types::types::Prim::F32,
        };

        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            vec3_ty.clone(),
            None,
        );
        let padded = dag.add_node(
            RiscOp::Pad {
                padding: vec![(RtDim::Lit(1), RtDim::Lit(1))],
                fill: 0.0,
            },
            vec![x],
            vec5_ty.clone(),
            None,
        );
        let out = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![padded],
            scalar_f32(),
            None,
        );

        let grad_result = grad_dag(&dag, out, &[x]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
        );
        let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
        let grad = &vals[&grad_result.grad_nodes[&x]];
        // d(sum(pad(x)))/dx = ones (pad adds zeros, but gradient only flows to original elements)
        assert_eq!(grad.to_f64_lossy_vec(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn adv_shrink_gradient() {
        // Test shrink adjoint: shrink then sum
        use crate::eval::{TensorValue, eval_tensor};

        let vec5_ty = TensorType {
            dims: vec![DimInfo::Lit(5)],
            precision: chelis_types::types::Prim::F32,
        };
        let vec3_ty = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: chelis_types::types::Prim::F32,
        };

        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            vec5_ty.clone(),
            None,
        );
        let shrunk = dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(1), RtDim::Lit(4))],
            },
            vec![x],
            vec3_ty.clone(),
            None,
        );
        let out = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![shrunk],
            scalar_f32(),
            None,
        );

        let grad_result = grad_dag(&dag, out, &[x]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        );
        let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
        let grad = &vals[&grad_result.grad_nodes[&x]];
        // d(sum(shrink(x, [1,4])))/dx = [0, 1, 1, 1, 0]
        assert_eq!(
            grad.to_f64_lossy_vec(),
            vec![0.0, 1.0, 1.0, 1.0, 0.0],
            "shrink gradient mismatch: got {:?}",
            grad.to_f64_lossy_vec()
        );
    }

    #[test]
    fn adv_max_elem_spec_compliance() {
        let (dag, x, _y, out) = build_binary_dag(|dag, a, b, ty| {
            dag.add_node(RiscOp::MaxElem, vec![a, b], ty.clone(), None)
        });
        let grad_result = grad_dag(&dag, out, &[x]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), 5.0);
        inputs.insert("y".to_string(), 3.0);
        let vals = eval_scalar(&grad_result.dag, &inputs);
        let dx_gt = vals[&grad_result.grad_nodes[&x]];
        assert!(
            (dx_gt - 1.0).abs() < 1e-6,
            "max_elem x>y: dx should be 1.0, got {dx_gt}"
        );

        inputs.insert("x".to_string(), 3.0);
        inputs.insert("y".to_string(), 3.0);
        let vals2 = eval_scalar(&grad_result.dag, &inputs);
        let dx_eq = vals2[&grad_result.grad_nodes[&x]];
        assert!(
            (dx_eq - 1.0).abs() < 1e-6,
            "max_elem x==y should route gradient to lhs, got {dx_eq}"
        );
    }

    #[test]
    fn adv_log_second_order() {
        // f(x) = log(x), f'(x) = 1/x, f''(x) = -1/x^2
        // At x=2: f''(2) = -0.25
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let out = dag.add_node(RiscOp::Log, vec![x], scalar_f64(), None);

        let first = grad_dag(&dag, out, &[x]).unwrap();
        let dx_node = first.grad_nodes[&x];

        let second = grad_dag(&first.dag, dx_node, &[x]).unwrap();
        let ddx_node = second.grad_nodes[&x];

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), 2.0);
        let vals = eval_scalar(&second.dag, &inputs);
        let ddx = vals[&ddx_node];
        let expected = -1.0 / 4.0; // -1/x^2 = -0.25
        assert!(
            (ddx - expected).abs() < 1e-3,
            "d^2(log(x))/dx^2 at x=2: expected {expected}, got {ddx}"
        );
    }

    #[test]
    fn adv_stride_gradient_scatters_into_strided_slots() {
        // Issue #291: the stride adjoint scatters the cotangent back into
        // the strided source slots (zeros elsewhere) using existing
        // movement primitives (reshape/pad/shrink). `f(x) = sum(stride(x,
        // 2)) = x0 + x2`, so `df/dx = [1, 0, 1, 0]`.
        use crate::eval::{TensorValue, eval_tensor};

        let vec4_ty = TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: chelis_types::types::Prim::F32,
        };
        let vec2_ty = TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: chelis_types::types::Prim::F32,
        };

        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            vec4_ty.clone(),
            None,
        );
        let strided = dag.add_node(
            RiscOp::Stride {
                strides: vec![RtDim::Lit(2)],
            },
            vec![x],
            vec2_ty.clone(),
            None,
        );
        let out = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![strided],
            scalar_f32(),
            None,
        );
        let grad_result =
            grad_dag(&dag, out, &[x]).expect("stride gradients are supported (issue #291)");
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]),
        );
        let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
        let grad = &vals[&grad_result.grad_nodes[&x]];
        assert_eq!(
            grad.to_f64_lossy_vec(),
            vec![1.0, 0.0, 1.0, 0.0],
            "stride gradient must scatter into strided slots: got {:?}",
            grad.to_f64_lossy_vec()
        );
    }

    #[test]
    fn adv_gradient_at_nontrivial_values() {
        // Test all ops at non-trivial values (2.7, -0.3, 1.5)
        // to catch any issues with specific values.

        // exp at 2.7
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Exp, vec![a], ty.clone(), None));
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.7, 1e-5);
        assert_grad_close(a, n);

        // log at 1.5
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Log, vec![a], ty.clone(), None));
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 1.5, 1e-5);
        assert_grad_close(a, n);

        // sin at -0.3
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Sin, vec![a], ty.clone(), None));
        let (a, n) = finite_diff(&dag, out, x, "x", &[], -0.3, 1e-5);
        assert_grad_close(a, n);

        // sqrt at 2.7
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Sqrt, vec![a], ty.clone(), None));
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.7, 1e-5);
        assert_grad_close(a, n);
    }

    // ---- Phase 3j-pre: adjoints for new reductions ----

    #[test]
    fn adv_min_reduce_gradient() {
        // Mirrors adv_max_reduce_gradient. Input 2x3:
        //   [[1, 5, 3], [4, 2, 6]]
        // min_reduce(axis=0) = [1, 2, 3]
        // sum(min) = 6
        // Gradient: 1 where element equals min along axis 0, else 0.
        // Expected: [[1, 0, 1], [0, 1, 0]]
        use crate::eval::{TensorValue, eval_tensor};

        let mat23 = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let vec3 = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };

        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            mat23.clone(),
            None,
        );
        let minr = dag.add_node(RiscOp::MinReduce { axis: 0 }, vec![x], vec3, None);
        let out = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![minr],
            scalar_f32(),
            None,
        );

        let grad_result = grad_dag(&dag, out, &[x]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![2, 3], vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]),
        );
        let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
        let grad = &vals[&grad_result.grad_nodes[&x]];
        let expected = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        for (i, (got, want)) in grad
            .to_f64_lossy_vec()
            .iter()
            .zip(expected.iter())
            .enumerate()
        {
            assert!(
                (got - want).abs() < 1e-6,
                "min_reduce grad index {i}: expected {want}, got {got}"
            );
        }
    }

    #[test]
    fn adv_prod_reduce_gradient_nonzero() {
        // Input: 1D length-4 vector [2, 3, 4, 5]
        // prod = 120
        // ∂prod/∂x_i = prod(x) / x_i = [60, 40, 30, 24]
        use crate::eval::{TensorValue, eval_tensor};

        let vec4 = TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec4, None);
        let p = dag.add_node(RiscOp::ProdReduce { axis: 0 }, vec![x], scalar_f32(), None);
        dag.add_root(p);

        let grad_result = grad_dag(&dag, p, &[x]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![4], vec![2.0, 3.0, 4.0, 5.0]),
        );
        let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
        let grad = &vals[&grad_result.grad_nodes[&x]];
        let expected = [60.0, 40.0, 30.0, 24.0];
        for (i, (got, want)) in grad
            .to_f64_lossy_vec()
            .iter()
            .zip(expected.iter())
            .enumerate()
        {
            assert!(
                (got - want).abs() < 1e-4,
                "prod_reduce grad index {i}: expected {want}, got {got}"
            );
        }
    }

    /// The critical safety test: one element of the reduced slice is zero.
    /// The naive `g * prod(x) / x_i` rule divides by zero; the prefix*suffix
    /// rule must produce finite, correct gradients.
    ///
    /// Input: [2, 0, 4, 5]
    /// prod = 0
    /// ∂prod/∂x_0 = 0*4*5 = 0
    /// ∂prod/∂x_1 = 2*4*5 = 40   (the interesting one — nonzero even though x_1=0)
    /// ∂prod/∂x_2 = 2*0*5 = 0
    /// ∂prod/∂x_3 = 2*0*4 = 0
    #[test]
    fn adv_prod_reduce_gradient_with_zero_element() {
        use crate::eval::{TensorValue, eval_tensor};

        let vec4 = TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec4, None);
        let p = dag.add_node(RiscOp::ProdReduce { axis: 0 }, vec![x], scalar_f32(), None);
        dag.add_root(p);

        let grad_result = grad_dag(&dag, p, &[x]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![4], vec![2.0, 0.0, 4.0, 5.0]),
        );
        let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
        let grad = &vals[&grad_result.grad_nodes[&x]];
        for v in &grad.to_f64_lossy_vec() {
            assert!(v.is_finite(), "prod_reduce grad must be finite; got {v}");
        }
        let expected = [0.0, 40.0, 0.0, 0.0];
        for (i, (got, want)) in grad
            .to_f64_lossy_vec()
            .iter()
            .zip(expected.iter())
            .enumerate()
        {
            assert!(
                (got - want).abs() < 1e-4,
                "prod_reduce zero-element grad index {i}: expected {want}, got {got}"
            );
        }
    }

    #[test]
    fn adv_argmax_on_grad_path_errors_cleanly() {
        // sum(argmax(x, axis=0)) — non-differentiable. grad_dag_checked must
        // refuse with a message naming the offending op, not silently zero.
        let mat23 = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let vec3 = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], mat23, None);
        let am = dag.add_node(RiscOp::Argmax { axis: 0 }, vec![x], vec3, None);
        let out = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![am],
            scalar_f64(),
            None,
        );

        let err = match grad_dag_checked(&dag, out, &[x]) {
            Err(e) => e,
            Ok(_) => panic!("argmax on the gradient path must error, not succeed"),
        };
        assert!(
            matches!(
                err,
                AdError::NotSupported {
                    op: "argmax",
                    reason: AdRejectionReason::IntegerIndexOutput,
                }
            ),
            "argmax must be rejected with structured AdError::NotSupported \
             {{ op: \"argmax\", reason: IntegerIndexOutput }}; got: {err:?}"
        );
    }

    #[test]
    fn adv_argmin_on_grad_path_errors_cleanly() {
        let mat23 = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let vec3 = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], mat23, None);
        let am = dag.add_node(RiscOp::Argmin { axis: 1 }, vec![x], vec3, None);
        let out = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![am],
            scalar_f64(),
            None,
        );

        let err = match grad_dag_checked(&dag, out, &[x]) {
            Err(e) => e,
            Ok(_) => panic!("argmin on the gradient path must error, not succeed"),
        };
        assert!(
            matches!(
                err,
                AdError::NotSupported {
                    op: "argmin",
                    reason: AdRejectionReason::IntegerIndexOutput,
                }
            ),
            "argmin must be rejected with structured AdError::NotSupported \
             {{ op: \"argmin\", reason: IntegerIndexOutput }}; got: {err:?}"
        );
    }

    // ---- New scalar builtin AD tests ----

    #[test]
    fn grad_cos() {
        // d/dx cos(x) = -sin(x)
        // At x=0.5: -sin(0.5) ≈ -0.4794
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Cos, vec![a], ty.clone(), None));
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 0.5, 1e-5);
        assert_grad_close(a, n);
        let expected = -(0.5_f64).sin();
        assert!(
            (a - expected).abs() < 1e-4,
            "grad of cos at 0.5 should be ≈ {expected}, got {a}"
        );
    }

    #[test]
    fn grad_tan() {
        // d/dx tan(x) = 1 / cos²(x)
        // At x=0.5: 1/cos²(0.5) ≈ 1.298
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Tan, vec![a], ty.clone(), None));
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 0.5, 1e-5);
        assert_grad_close(a, n);
    }

    #[test]
    fn grad_atan() {
        // d/dx atan(x) = 1 / (1 + x²)
        // At x=1.0: 1 / (1+1) = 0.5
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Atan, vec![a], ty.clone(), None));
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 1.0, 1e-5);
        assert_grad_close(a, n);
        assert!(
            (a - 0.5).abs() < 1e-4,
            "grad of atan at 1.0 should be ≈ 0.5, got {a}"
        );
    }

    #[test]
    fn grad_abs_positive() {
        // d/dx abs(x) = 1 for x > 0
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Abs, vec![a], ty.clone(), None));
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.0, 1e-5);
        assert_grad_close(a, n);
        assert!(
            (a - 1.0).abs() < 1e-4,
            "grad of abs at 2.0 should be 1.0, got {a}"
        );
    }

    #[test]
    fn grad_abs_negative() {
        // d/dx abs(x) = -1 for x < 0
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Abs, vec![a], ty.clone(), None));
        let (a, n) = finite_diff(&dag, out, x, "x", &[], -2.0, 1e-5);
        assert_grad_close(a, n);
        assert!(
            (a - (-1.0)).abs() < 1e-4,
            "grad of abs at -2.0 should be -1.0, got {a}"
        );
    }

    #[test]
    fn floor_on_grad_path_errors_cleanly() {
        // floor is non-differentiable: grad_dag_checked must return a clean error.
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Floor, vec![a], ty.clone(), None));
        let err = match grad_dag_checked(&dag, out, &[x]) {
            Err(e) => e,
            Ok(_) => panic!("floor on the gradient path must error, not succeed"),
        };
        assert!(
            matches!(
                err,
                AdError::NotSupported {
                    op: "floor",
                    reason: AdRejectionReason::PiecewiseConstant,
                }
            ),
            "floor must be rejected with structured AdError::NotSupported \
             {{ op: \"floor\", reason: PiecewiseConstant }}; got: {err:?}"
        );
    }

    #[test]
    fn ceil_on_grad_path_errors_cleanly() {
        // ceil is non-differentiable: grad_dag_checked must return a clean error.
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Ceil, vec![a], ty.clone(), None));
        let err = match grad_dag_checked(&dag, out, &[x]) {
            Err(e) => e,
            Ok(_) => panic!("ceil on the gradient path must error, not succeed"),
        };
        assert!(
            matches!(
                err,
                AdError::NotSupported {
                    op: "ceil",
                    reason: AdRejectionReason::PiecewiseConstant,
                }
            ),
            "ceil must be rejected with structured AdError::NotSupported \
             {{ op: \"ceil\", reason: PiecewiseConstant }}; got: {err:?}"
        );
    }

    #[test]
    fn round_on_grad_path_errors_cleanly() {
        // round is non-differentiable: grad_dag_checked must return a clean error.
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Round, vec![a], ty.clone(), None));
        let err = match grad_dag_checked(&dag, out, &[x]) {
            Err(e) => e,
            Ok(_) => panic!("round on the gradient path must error, not succeed"),
        };
        assert!(
            matches!(
                err,
                AdError::NotSupported {
                    op: "round",
                    reason: AdRejectionReason::PiecewiseConstant,
                }
            ),
            "round must be rejected with structured AdError::NotSupported \
             {{ op: \"round\", reason: PiecewiseConstant }}; got: {err:?}"
        );
    }

    // ---- ADVERSARIAL TESTS: missing coverage from red-team spec ----

    /// abs(x) at x=0 must give exactly 0.0 (sign convention: sign(0) = 0).
    /// This is NOT covered by grad_abs_positive (x=2.0) or grad_abs_negative (x=-2.0).
    #[test]
    fn adv_grad_abs_at_zero_is_zero() {
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Abs, vec![a], ty.clone(), None));
        let grad_result = grad_dag(&dag, out, &[x]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), 0.0_f64);
        let vals = eval_scalar(&grad_result.dag, &inputs);
        let a = vals[&grad_result.grad_nodes[&x]];
        assert!(
            a.abs() < 1e-9,
            "abs gradient at x=0 must be 0.0 (sign(0)=0), got {a}"
        );
    }

    /// abs(x) at x=-2.7 must give exactly -1.0.
    /// Spec says: abs(x) at x=-2.7 → -1.0.
    #[test]
    fn adv_grad_abs_at_neg2_7_is_neg1() {
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Abs, vec![a], ty.clone(), None));
        let grad_result = grad_dag(&dag, out, &[x]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), -2.7_f64);
        let vals = eval_scalar(&grad_result.dag, &inputs);
        let a = vals[&grad_result.grad_nodes[&x]];
        assert!(
            (a - (-1.0)).abs() < 1e-6,
            "abs gradient at x=-2.7 must be -1.0, got {a}"
        );
    }

    /// atan(x) at x=1.5: grad = 1/(1+1.5²) = 1/3.25 ≈ 0.3077.
    /// Spec asks for this point; existing test only covers x=1.0.
    #[test]
    fn adv_grad_atan_at_1_5() {
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Atan, vec![a], ty.clone(), None));
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 1.5, 1e-5);
        assert_grad_close(a, n);
        let expected = 1.0 / (1.0 + 1.5_f64 * 1.5);
        assert!(
            (a - expected).abs() < 1e-4,
            "grad of atan at 1.5 should be ≈ {expected} (1/3.25), got {a}"
        );
    }

    /// tan(x) at x=0.3: grad = 1/cos²(0.3) ≈ 1.047.
    /// Spec asks for this exact point with exact value; existing test uses x=0.5 with no exact check.
    #[test]
    fn adv_grad_tan_at_0_3_exact() {
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Tan, vec![a], ty.clone(), None));
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 0.3, 1e-5);
        assert_grad_close(a, n);
        let expected = 1.0 / (0.3_f64.cos() * 0.3_f64.cos());
        assert!(
            (a - expected).abs() < 1e-4,
            "grad of tan at 0.3 should be ≈ {expected} (1/cos²(0.3)), got {a}"
        );
    }

    /// cos(add(x, const(0.5))) composition — grad should be -sin(x+0.5).
    #[test]
    fn adv_grad_cos_of_add_composition() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f64(),
            None,
        );
        let half = dag.add_node(
            RiscOp::synth_const(scalar_f64().precision, 0.5),
            vec![],
            scalar_f64(),
            None,
        );
        let xp = dag.add_node(RiscOp::Add, vec![x, half], scalar_f64(), None);
        let out = dag.add_node(RiscOp::Cos, vec![xp], scalar_f64(), None);

        let x0 = 0.7_f64;
        let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
        assert_grad_close(a, n);
        let expected = -(x0 + 0.5).sin();
        assert!(
            (a - expected).abs() < 1e-4,
            "grad of cos(x+0.5) at x=0.7 should be -sin(1.2) ≈ {expected}, got {a}"
        );
    }

    /// floor must give a CLEAN structured error, not a silent zero gradient.
    /// Pattern-matches on the AdError enum, not the rendered string.
    #[test]
    fn adv_floor_grad_dag_checked_error_is_not_silent_zero() {
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Floor, vec![a], ty.clone(), None));
        // grad_dag (unchecked) returns a zero gradient silently — that is the safety net.
        // grad_dag_checked must be the gate that rejects it.
        let result = grad_dag_checked(&dag, out, &[x]);
        match result {
            Err(err) => {
                assert!(
                    matches!(
                        err,
                        AdError::NotSupported {
                            op: "floor",
                            reason: AdRejectionReason::PiecewiseConstant,
                        }
                    ),
                    "floor must be rejected with structured AdError::NotSupported \
                     {{ op: \"floor\", reason: PiecewiseConstant }}; got: {err:?}"
                );
            }
            Ok(_) => panic!("floor must be rejected by grad_dag_checked. Got Ok"),
        }
    }

    /// ceil must give a CLEAN structured error, not succeed.
    #[test]
    fn adv_ceil_grad_dag_checked_error_is_not_silent_zero() {
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Ceil, vec![a], ty.clone(), None));
        let result = grad_dag_checked(&dag, out, &[x]);
        match result {
            Err(err) => {
                assert!(
                    matches!(
                        err,
                        AdError::NotSupported {
                            op: "ceil",
                            reason: AdRejectionReason::PiecewiseConstant,
                        }
                    ),
                    "ceil must be rejected with structured AdError::NotSupported \
                     {{ op: \"ceil\", reason: PiecewiseConstant }}; got: {err:?}"
                );
            }
            Ok(_) => panic!("ceil must be rejected by grad_dag_checked. Got Ok"),
        }
    }

    /// cos(x) at x=0.5: exact value check. Spec requires -sin(0.5) ≈ -0.4794.
    #[test]
    fn adv_grad_cos_at_0_5_exact_value() {
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Cos, vec![a], ty.clone(), None));
        let grad_result = grad_dag(&dag, out, &[x]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), 0.5_f64);
        let vals = eval_scalar(&grad_result.dag, &inputs);
        let a = vals[&grad_result.grad_nodes[&x]];
        let expected = -(0.5_f64).sin(); // ≈ -0.4794
        assert!(
            (a - expected).abs() < 1e-5,
            "cos gradient at x=0.5 must be -sin(0.5) ≈ {expected:.6}, got {a:.6}"
        );
    }

    // --- chelis#513 gap 3: structural symbolic-bystander-axis adjoints ---

    fn sym_batch_ty(rest: &[usize]) -> TensorType {
        let mut dims = vec![DimInfo::Named("batch".into(), None)];
        dims.extend(rest.iter().map(|n| DimInfo::Lit(*n)));
        TensorType {
            dims,
            precision: chelis_types::types::Prim::F32,
        }
    }

    /// Reduce `node` (rank-2 `[batch, k]`) to a scalar via two Sums, so the
    /// DAG has the scalar-loss output `grad_dag` requires.
    fn reduce_rank2_to_scalar(dag: &mut Dag, node: NodeId, mid_dims: Vec<DimInfo>) -> NodeId {
        let mid = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![node],
            TensorType {
                dims: mid_dims,
                precision: chelis_types::types::Prim::F32,
            },
            None,
        );
        dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![mid],
            scalar_f32(),
            None,
        )
    }

    /// Stride along the concrete axis of a `[batch, 4]` input: backward
    /// construction must succeed and the trim `Shrink` must carry the
    /// `SHRINK_TO_END` full-axis sentinel on the symbolic bystander axis
    /// (never a guessed concrete extent).
    #[test]
    fn stride_adjoint_symbolic_bystander_axis_uses_sentinel_trim() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            sym_batch_ty(&[4]),
            None,
        );
        let strided = dag.add_node(
            RiscOp::Stride {
                strides: vec![RtDim::Lit(1), RtDim::Lit(2)],
            },
            vec![x],
            sym_batch_ty(&[2]),
            None,
        );
        let out = reduce_rank2_to_scalar(&mut dag, strided, vec![DimInfo::Lit(2)]);

        let grad_result = grad_dag(&dag, out, &[x]).expect("symbolic bystander must construct");
        let sentinel_trim = grad_result.dag.nodes().iter().any(|node| {
            matches!(
                &node.op,
                RiscOp::Shrink { bounds }
                    if bounds.first() == Some(&(RtDim::Lit(0), RtDim::ToEnd))
                        && bounds.get(1) == Some(&(RtDim::Lit(0), RtDim::Lit(4)))
            )
        });
        assert!(
            sentinel_trim,
            "stride adjoint must trim via a SHRINK_TO_END sentinel on the \
             symbolic bystander axis and the concrete source extent on the \
             strided axis"
        );
    }

    /// chelis#616: a stride along the SYMBOLIC axis itself now constructs a
    /// RUNTIME adjoint cascade — the trim shrink's end bound is a
    /// node-valued Shape read of the forward input's extent (`n_a`), and the
    /// merge reshape's target extent is runtime `m_a * step` arithmetic —
    /// instead of the pre-#616 loud panic.
    #[test]
    fn stride_adjoint_symbolic_strided_axis_builds_runtime_trim() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Named("n".into(), None)],
                precision: chelis_types::types::Prim::F32,
            },
            None,
        );
        let strided = dag.add_node(
            RiscOp::Stride {
                strides: vec![RtDim::Lit(2)],
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Named("m".into(), None)],
                precision: chelis_types::types::Prim::F32,
            },
            None,
        );
        let out = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![strided],
            scalar_f32(),
            None,
        );
        let grad_result =
            grad_dag(&dag, out, &[x]).expect("runtime strided-axis adjoint must construct");
        // The trim shrink's end bound is node-valued (a Shape read of the
        // forward input), never a guessed literal.
        let runtime_trim = grad_result.dag.nodes().iter().any(|node| {
            matches!(
                &node.op,
                RiscOp::Shrink { bounds }
                    if bounds.first().is_some_and(|(s, e)| {
                        s.as_lit() == Some(0) && e.node_input().is_some()
                    })
            )
        });
        assert!(
            runtime_trim,
            "runtime strided-axis adjoint must trim to a node-valued end bound"
        );
        // The gradient of sum(stride(x, 2)) at n = 5 is the upsample mask
        // [1, 0, 1, 0, 1].
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        );
        let grad_node = grad_result.grad_nodes[&x];
        let values =
            crate::eval::eval_tensor_roots_with_strict(&grad_result.dag, &[grad_node], |name| {
                inputs.get(name).cloned()
            })
            .expect("runtime strided-axis adjoint must evaluate");
        assert_eq!(values[&grad_node].shape, vec![5]);
        assert_eq!(
            values[&grad_node].to_f64_lossy_vec(),
            vec![1.0, 0.0, 1.0, 0.0, 1.0]
        );
    }

    /// ProdReduce along the concrete axis of a `[batch, 3]` input: backward
    /// construction must succeed and the per-slice Shrinks must carry the
    /// sentinel on the symbolic bystander axis.
    #[test]
    fn prod_reduce_adjoint_symbolic_bystander_axis_uses_sentinel_slices() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            sym_batch_ty(&[3]),
            None,
        );
        let prod = dag.add_node(
            RiscOp::ProdReduce { axis: 1 },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Named("batch".into(), None)],
                precision: chelis_types::types::Prim::F32,
            },
            None,
        );
        let out = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![prod],
            scalar_f32(),
            None,
        );

        let grad_result = grad_dag(&dag, out, &[x]).expect("symbolic bystander must construct");
        let sentinel_slices = grad_result
            .dag
            .nodes()
            .iter()
            .filter(|node| {
                matches!(
                    &node.op,
                    RiscOp::Shrink { bounds }
                        if bounds.first() == Some(&(RtDim::Lit(0), RtDim::ToEnd))
                )
            })
            .count();
        assert_eq!(
            sentinel_slices, 3,
            "prod_reduce adjoint must emit one sentinel-bounded slice per \
             element of the concrete reduced axis"
        );
    }

    /// NEGATIVE PARITY: prod_reduce along the SYMBOLIC axis needs one slice
    /// per runtime element; stays fail-closed loud.
    #[test]
    #[should_panic(expected = "prod_reduce adjoint requires a concrete axis size")]
    fn prod_reduce_adjoint_symbolic_reduced_axis_fails_loud() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Named("n".into(), None)],
                precision: chelis_types::types::Prim::F32,
            },
            None,
        );
        let prod = dag.add_node(RiscOp::ProdReduce { axis: 0 }, vec![x], scalar_f64(), None);
        let _ = grad_dag(&dag, prod, &[x]);
    }

    /// The Shrink adjoint of a full-axis `(0, SHRINK_TO_END)` sentinel bound
    /// is exactly no padding on that axis: `(0, 0)`, no axis size needed.
    /// (This is the double-grad consumer of the sentinel the Pad and Stride
    /// adjoints emit on symbolic bystander axes.)
    #[test]
    fn shrink_adjoint_full_axis_sentinel_pads_zero() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            sym_batch_ty(&[3]),
            None,
        );
        let shrunk = dag.add_node(
            RiscOp::Shrink {
                bounds: vec![
                    (RtDim::Lit(0), RtDim::ToEnd),
                    (RtDim::Lit(1), RtDim::Lit(3)),
                ],
            },
            vec![x],
            sym_batch_ty(&[2]),
            None,
        );
        let out = reduce_rank2_to_scalar(&mut dag, shrunk, vec![DimInfo::Lit(2)]);

        let grad_result = grad_dag(&dag, out, &[x]).expect("sentinel shrink must construct");
        let pad_zero = grad_result.dag.nodes().iter().any(|node| {
            matches!(
                &node.op,
                RiscOp::Pad { padding, .. }
                    if padding.first() == Some(&(RtDim::Lit(0), RtDim::Lit(0)))
                        && padding.get(1) == Some(&(RtDim::Lit(1), RtDim::Lit(0)))
            )
        });
        assert!(
            pad_zero,
            "shrink adjoint must lower the full-axis sentinel to (0, 0) \
             padding and the concrete sub-range to its exact padding"
        );
    }

    /// NEGATIVE PARITY: a sentinel with a nonzero start is a producing-pass
    /// bug and must fail loud, never wrap `dim_size - end`.
    #[test]
    #[should_panic(expected = "malformed ToEnd sentinel")]
    fn shrink_adjoint_malformed_sentinel_fails_loud() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: chelis_types::types::Prim::F32,
            },
            None,
        );
        let shrunk = dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(1), RtDim::ToEnd)],
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(3)],
                precision: chelis_types::types::Prim::F32,
            },
            None,
        );
        let out = dag.add_node(
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![shrunk],
            scalar_f64(),
            None,
        );
        let _ = grad_dag(&dag, out, &[x]);
    }
}
