//! Reverse-mode automatic differentiation for RISC DAGs.
//!
//! Given a forward DAG computing `f(inputs) -> output`, produces a backward DAG
//! computing gradients of the output with respect to specified input nodes.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::dag::{Dag, DagNode, DimExpr, DimInfo, NodeId, RiscOp, TensorType};
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
) -> Result<GradResult, String> {
    if forward.is_empty() {
        return Err("cannot differentiate an empty DAG".to_string());
    }
    let out_node = forward
        .get(output)
        .ok_or_else(|| format!("output node {output:?} does not exist in forward DAG"))?;
    if !is_scalar_float(&out_node.output_type) {
        return Err(format!(
            "grad: output node {} must be a scalar float, got {:?}",
            output.0, out_node.output_type
        ));
    }

    // Walk the subgraph of nodes reachable from `output` and look for ops
    // whose adjoint is intentionally undefined.
    let mut live = vec![false; forward.len()];
    live[output.0] = true;
    for i in (0..forward.len()).rev() {
        if live[i] {
            for input in &forward.nodes()[i].inputs {
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
                return Err(format!(
                    "grad: Argmax at node {} is non-differentiable (integer-index output); \
                     remove it from the gradient path or wrap it in a stop-gradient",
                    node.id.0
                ));
            }
            RiscOp::Argmin { .. } => {
                return Err(format!(
                    "grad: Argmin at node {} is non-differentiable (integer-index output); \
                     remove it from the gradient path or wrap it in a stop-gradient",
                    node.id.0
                ));
            }
            RiscOp::Floor => {
                return Err(format!(
                    "grad: Floor at node {} is non-differentiable (piecewise constant); \
                     remove it from the gradient path or wrap it in a stop-gradient",
                    node.id.0
                ));
            }
            RiscOp::Ceil => {
                return Err(format!(
                    "grad: Ceil at node {} is non-differentiable (piecewise constant); \
                     remove it from the gradient path or wrap it in a stop-gradient",
                    node.id.0
                ));
            }
            _ => {}
        }
    }

    grad_dag(forward, output, wrt).ok_or_else(|| {
        "grad: failed to construct backward DAG (unsupported op or verification failure)"
            .to_string()
    })
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
    if forward.is_empty() {
        return None;
    }
    let output_ty = forward.get(output)?.output_type.clone();
    if !is_scalar_float(&output_ty) {
        return None;
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
    let seed = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], output_ty, None);
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
        let input_grads = compute_adjoints(&node, grad_out, forward, &mut dag)?;
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

    let (dag, output_node, grad_nodes) = prune_to_requested_outputs(&dag, output, &grad_nodes);

    if !crate::verify::verify(&dag).is_empty() {
        return None;
    }

    Some(GradResult {
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
        RiscOp::CmpLt => {
            let a = node.inputs[0];
            let b = node.inputs[1];
            let ty_a = forward.get(a).unwrap().output_type.clone();
            let ty_b = forward.get(b).unwrap().output_type.clone();
            let za = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], ty_a, None);
            let zb = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], ty_b, None);
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
            let one = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], ty.clone(), None);
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
                RiscOp::Const {
                    value: std::f64::consts::FRAC_PI_2,
                },
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
            let two = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], ty.clone(), None);
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
            let one = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], ty.clone(), None);
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
            let zero = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], ty.clone(), None);
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
            let zero = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], ty, None);
            Some(vec![(x, zero)])
        }
        RiscOp::Ceil => {
            // ceil is non-differentiable — grad_dag_checked will have already
            // rejected this; this arm is a safety net returning zero gradient.
            let x = node.inputs[0];
            let ty = forward.get(x).unwrap().output_type.clone();
            let zero = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], ty, None);
            Some(vec![(x, zero)])
        }
        RiscOp::UniformLike { .. } => {
            let x = node.inputs[0];
            let ty = forward.get(x).unwrap().output_type.clone();
            let zero = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], ty, None);
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
            let mut slices: Vec<NodeId> = Vec::with_capacity(axis_size);
            for i in 0..axis_size {
                let bounds: Vec<(usize, usize)> = (0..rank)
                    .map(|d| {
                        if d == *axis {
                            (i, i + 1)
                        } else {
                            (0, dim_size(&input_ty.dims[d]))
                        }
                    })
                    .collect();
                let s = dag.add_node(RiscOp::Shrink { bounds }, vec![x], slice_ty.clone(), None);
                slices.push(s);
            }

            // Prefix products: prefix[i] = prod_{j<i} slices[j], with prefix[0] = 1.
            let one = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], slice_ty.clone(), None);
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
                let padding: Vec<(usize, usize)> = (0..rank)
                    .map(|d| {
                        if d == *axis {
                            (i, axis_size - i - 1)
                        } else {
                            (0, 0)
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
            let original_shape = input_ty.dims.clone();
            let dx = dag.add_node(
                RiscOp::Reshape {
                    new_shape: original_shape,
                },
                vec![g],
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
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            // The gradient sum runs over the operand precision; use the
            // spec-default accumulator so the AD path tracks WS-A0 §5.7.1.
            let acc = RiscOp::default_reduce_sum_accumulator(input_ty.precision)
                .unwrap_or(input_ty.precision);
            let dx = dag.add_node(
                RiscOp::Sum {
                    axis: *axis,
                    accumulator: acc,
                },
                vec![g],
                input_ty,
                None,
            );
            Some(vec![(x, dx)])
        }
        RiscOp::OneHot { .. } => Some(vec![]),
        RiscOp::Pad { padding, .. } => {
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            // Shrink: bounds = [(before, before + dim_size), ...] for each axis
            let bounds: Vec<(usize, usize)> = padding
                .iter()
                .zip(input_ty.dims.iter())
                .map(|((before, _after), dim)| (*before, *before + dim_size(dim)))
                .collect();
            let dx = dag.add_node(RiscOp::Shrink { bounds }, vec![g], input_ty, None);
            Some(vec![(x, dx)])
        }
        RiscOp::Shrink { bounds } => {
            let x = node.inputs[0];
            let input_ty = forward.get(x).unwrap().output_type.clone();
            // Pad: for each axis, before = start, after = original_size - end
            let padding: Vec<(usize, usize)> = bounds
                .iter()
                .zip(input_ty.dims.iter())
                .map(|((start, end), dim)| (*start, dim_size(dim) - end))
                .collect();
            let dx = dag.add_node(RiscOp::Pad { padding, fill: 0.0 }, vec![g], input_ty, None);
            Some(vec![(x, dx)])
        }
        RiscOp::Stride { strides } => {
            // Phase 0 does not have a scatter/upsample primitive, so exact stride adjoints
            // cannot be represented soundly in the current RISC set.
            let _ = strides;
            None
        }

        // --- Memory ---
        RiscOp::Const { .. } => Some(vec![]),
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
                let zero = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], input_ty, None);
                Some(vec![(x, zero)])
            }
        }
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
                RiscOp::Const { value: 0.0 },
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
        RiscOp::BlasMatmul { .. } => None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::eval_scalar;
    use std::collections::HashMap;

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    /// Build a unary DAG: Load("x") -> op -> output.
    fn build_unary_dag(
        op_fn: impl FnOnce(&mut Dag, NodeId, &TensorType) -> NodeId,
    ) -> (Dag, NodeId, NodeId) {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let y = op_fn(&mut dag, x, &scalar_f32());
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
            scalar_f32(),
            None,
        );
        let y = dag.add_node(
            RiscOp::Load { name: "y".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let out = op_fn(&mut dag, x, y, &scalar_f32());
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
            scalar_f32(),
            None,
        );
        let x_sq = dag.add_node(RiscOp::Mul, vec![x, x], scalar_f32(), None);

        let (a, n) = finite_diff(&dag, x_sq, x, "x", &[], 3.0, 1e-5);
        assert_grad_close(a, n);
        assert!((a - 6.0).abs() < 1e-4); // 2 * 3 = 6
    }

    #[test]
    fn grad_relu_positive() {
        // relu(x) = max(x, 0), d/dx = 1 when x > 0
        let (dag, x, out) = build_unary_dag(|dag, a, ty| {
            let zero = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], ty.clone(), None);
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
            let zero = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], ty.clone(), None);
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
            scalar_f32(),
            None,
        );
        let y = dag.add_node(
            RiscOp::Load { name: "y".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let z = dag.add_node(
            RiscOp::Load { name: "z".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let xy = dag.add_node(RiscOp::Mul, vec![x, y], scalar_f32(), None);
        let out = dag.add_node(RiscOp::Add, vec![xy, z], scalar_f32(), None);

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
            scalar_f32(),
            None,
        );
        let sum1 = dag.add_node(RiscOp::Add, vec![x, x], scalar_f32(), None);
        let out = dag.add_node(RiscOp::Add, vec![sum1, x], scalar_f32(), None);

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
        let c = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32(), None);
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let out = dag.add_node(RiscOp::Mul, vec![x, c], scalar_f32(), None);

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
        assert_eq!(grad.data, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn grad_second_order() {
        // f(x) = x*x, f'(x) = 2x, f''(x) = 2
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let x_sq = dag.add_node(RiscOp::Mul, vec![x, x], scalar_f32(), None);

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
            scalar_f32(),
            None,
        );
        let exp_x = dag.add_node(RiscOp::Exp, vec![x], scalar_f32(), None);
        let out = dag.add_node(
            RiscOp::Store { name: "out".into() },
            vec![exp_x],
            scalar_f32(),
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
            scalar_f32(),
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
        let out = dag.add_node(RiscOp::Exp, vec![casted], scalar_f32(), None);

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
                new_shape: mat23_ty.dims.clone(),
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
        assert_eq!(grad.data, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
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
        assert_eq!(grad.data, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
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
        assert_eq!(grad.data, vec![2.0, 2.0, 2.0]);
        assert_eq!(grad.shape, vec![3]);
    }

    #[test]
    fn grad_mul_by_const() {
        // f(x) = 5 * x, df/dx = 5
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let five = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32(), None);
        let out = dag.add_node(RiscOp::Mul, vec![five, x], scalar_f32(), None);

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
            scalar_f32(),
            None,
        );
        let x_sq = dag.add_node(RiscOp::Mul, vec![x, x], scalar_f32(), None);
        let out = dag.add_node(RiscOp::Log, vec![x_sq], scalar_f32(), None);

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
            scalar_f32(),
            None,
        );
        let x_sq = dag.add_node(RiscOp::Mul, vec![x, x], scalar_f32(), None);
        let out = dag.add_node(RiscOp::Sin, vec![x_sq], scalar_f32(), None);

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
            scalar_f32(),
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
            scalar_f32(),
            None,
        );
        let y = dag.add_node(
            RiscOp::Load { name: "y".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let add_node = dag.add_node(RiscOp::Add, vec![x, y], scalar_f32(), None);
        let out = dag.add_node(RiscOp::Exp, vec![add_node], scalar_f32(), None);

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
            let one = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], ty.clone(), None);
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
            scalar_f32(),
            None,
        );
        let c = dag.add_node(RiscOp::Const { value: 42.0 }, vec![], scalar_f32(), None);
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
            scalar_f32(),
            None,
        );
        let s1 = dag.add_node(RiscOp::Add, vec![x, x], scalar_f32(), None);
        let s2 = dag.add_node(RiscOp::Add, vec![s1, x], scalar_f32(), None);
        let s3 = dag.add_node(RiscOp::Add, vec![s2, x], scalar_f32(), None);
        let out = dag.add_node(RiscOp::Add, vec![s3, x], scalar_f32(), None);

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
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![maxr],
            scalar_f32(),
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
        for (i, (actual, expected)) in grad.data.iter().zip(expected_grad.iter()).enumerate() {
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
            let f_plus = vals_plus[&out].data[0];

            let mut inputs_minus = HashMap::new();
            inputs_minus.insert(
                "x".to_string(),
                TensorValue::from_vec(vec![2, 3], data_minus),
            );
            let vals_minus = eval_tensor(&dag, &inputs_minus).unwrap();
            let f_minus = vals_minus[&out].data[0];

            let numerical = (f_plus - f_minus) / (2.0 * h);
            let analytical = grad.data[elem_idx];
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
        let grad_col0 = grad.data[0] + grad.data[2]; // should be 1.0 but will be 2.0
        eprintln!(
            "FINDING: max_reduce with ties: gradient = {:?}, column 0 sum = {grad_col0}",
            grad.data
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
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![maxr],
            scalar_f32(),
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
        for (i, (actual, expected)) in grad.data.iter().zip(expected_grad.iter()).enumerate() {
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
            let f_plus = vals_plus[&out].data[0];

            let mut inputs_minus = HashMap::new();
            inputs_minus.insert(
                "x".to_string(),
                TensorValue::from_vec(vec![2, 3], data_minus),
            );
            let vals_minus = eval_tensor(&dag, &inputs_minus).unwrap();
            let f_minus = vals_minus[&out].data[0];

            let numerical = (f_plus - f_minus) / (2.0 * h);
            let analytical = grad.data[elem_idx];
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
        // div(a, b) = a * exp(neg(log(b)))
        // d(a/b)/da = 1/b
        // d(a/b)/db = -a/b^2
        // Test at a=6, b=3: d/da=1/3, d/db=-6/9=-2/3
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::Load { name: "a".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::Load { name: "b".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let out = crate::tier2::lower_div(&mut dag, a, b, &scalar_f32(), None);

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
                padding: vec![(1, 1)],
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
        assert_eq!(grad.data, vec![1.0, 1.0, 1.0]);
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
                bounds: vec![(1, 4)],
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
            grad.data,
            vec![0.0, 1.0, 1.0, 1.0, 0.0],
            "shrink gradient mismatch: got {:?}",
            grad.data
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
            scalar_f32(),
            None,
        );
        let out = dag.add_node(RiscOp::Log, vec![x], scalar_f32(), None);

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
    fn adv_stride_gradient_is_rejected_until_supported() {
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
            RiscOp::Stride { strides: vec![2] },
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
        assert!(
            grad_dag(&dag, out, &[x]).is_none(),
            "stride gradients should fail closed until the RISC set can express them soundly"
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
        for (i, (got, want)) in grad.data.iter().zip(expected.iter()).enumerate() {
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
        for (i, (got, want)) in grad.data.iter().zip(expected.iter()).enumerate() {
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
        for v in &grad.data {
            assert!(v.is_finite(), "prod_reduce grad must be finite; got {v}");
        }
        let expected = [0.0, 40.0, 0.0, 0.0];
        for (i, (got, want)) in grad.data.iter().zip(expected.iter()).enumerate() {
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
            scalar_f32(),
            None,
        );

        let err = match grad_dag_checked(&dag, out, &[x]) {
            Err(e) => e,
            Ok(_) => panic!("argmax on the gradient path must error, not succeed"),
        };
        assert!(
            err.contains("Argmax") && err.contains("non-differentiable"),
            "error message must identify the non-differentiable op; got: {err}"
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
            scalar_f32(),
            None,
        );

        let err = match grad_dag_checked(&dag, out, &[x]) {
            Err(e) => e,
            Ok(_) => panic!("argmin on the gradient path must error, not succeed"),
        };
        assert!(
            err.contains("Argmin") && err.contains("non-differentiable"),
            "error message must identify the non-differentiable op; got: {err}"
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
            err.contains("Floor") && err.contains("non-differentiable"),
            "error message must identify Floor as non-differentiable; got: {err}"
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
            err.contains("Ceil") && err.contains("non-differentiable"),
            "error message must identify Ceil as non-differentiable; got: {err}"
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
            scalar_f32(),
            None,
        );
        let half = dag.add_node(RiscOp::Const { value: 0.5 }, vec![], scalar_f32(), None);
        let xp = dag.add_node(RiscOp::Add, vec![x, half], scalar_f32(), None);
        let out = dag.add_node(RiscOp::Cos, vec![xp], scalar_f32(), None);

        let x0 = 0.7_f64;
        let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
        assert_grad_close(a, n);
        let expected = -(x0 + 0.5).sin();
        assert!(
            (a - expected).abs() < 1e-4,
            "grad of cos(x+0.5) at x=0.7 should be -sin(1.2) ≈ {expected}, got {a}"
        );
    }

    /// floor must give a CLEAN error (containing "Floor" and "non-differentiable"),
    /// not a silent zero gradient. Double-checks grad_dag_checked, not just grad_dag.
    #[test]
    fn adv_floor_grad_dag_checked_error_is_not_silent_zero() {
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Floor, vec![a], ty.clone(), None));
        // grad_dag (unchecked) returns a zero gradient silently — that is the safety net.
        // grad_dag_checked must be the gate that rejects it.
        let result = grad_dag_checked(&dag, out, &[x]);
        match result {
            Err(msg) => {
                assert!(msg.contains("Floor"), "error must name 'Floor'; got: {msg}");
                assert!(
                    msg.contains("non-differentiable"),
                    "error must say 'non-differentiable'; got: {msg}"
                );
            }
            Ok(_) => panic!("floor must be rejected by grad_dag_checked. Got Ok"),
        }
    }

    /// ceil must give a CLEAN error, not succeed.
    #[test]
    fn adv_ceil_grad_dag_checked_error_is_not_silent_zero() {
        let (dag, x, out) =
            build_unary_dag(|dag, a, ty| dag.add_node(RiscOp::Ceil, vec![a], ty.clone(), None));
        let result = grad_dag_checked(&dag, out, &[x]);
        match result {
            Err(msg) => {
                assert!(msg.contains("Ceil"), "error must name 'Ceil'; got: {msg}");
                assert!(
                    msg.contains("non-differentiable"),
                    "error must say 'non-differentiable'; got: {msg}"
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
}
