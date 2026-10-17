//! Storage recipes for existing float contraction graphs.
//!
//! A recipe changes no operation, guard, or arithmetic tree. Its consumers
//! read checked metadata at the original node and ordered scalar leaves at
//! reductions. Exposed values and unsupported payload readers stay dense.

use crate::dag::{Dag, DagNode, NodeId, RiscOp, RtDim};

/// A lane-independent storage decision. Indices are original DAG node IDs.
#[derive(Debug, Clone)]
pub struct ContractionViews {
    deferred: Vec<bool>,
    last_read: Vec<usize>,
    base_last_read: Vec<Option<usize>>,
}

impl ContractionViews {
    pub fn is_deferred(&self, node: NodeId) -> bool {
        self.deferred[node.0]
    }

    /// Last logical reader of this node's metadata or payload.
    pub fn last_read(&self, node: NodeId) -> usize {
        self.last_read[node.0]
    }

    /// A dense base stays immutable and owned through this deferred read.
    pub fn base_last_read(&self, node: NodeId) -> Option<usize> {
        self.base_last_read[node.0]
    }
}

fn recipe(node: &DagNode) -> bool {
    node.output_type.precision.is_float()
        && node.owner.activation.is_none()
        && matches!(
            node.op,
            RiscOp::Expand { .. } | RiscOp::Permute { .. } | RiscOp::Reshape { .. } | RiscOp::Mul
        )
}

/// `InputAxis` auxiliary slots supply extents, never tensor elements. A
/// scalar bound is a payload read even when another bound names its shape.
pub(crate) fn payload_slot(node: &DagNode, slot: usize) -> bool {
    if matches!(node.op, RiscOp::Shape { .. } | RiscOp::Drop) {
        return false;
    }
    if slot == 0 {
        return true;
    }
    let bounds: &[RtDim] = match &node.op {
        RiscOp::Expand { size, .. } => std::slice::from_ref(size),
        RiscOp::Reshape { new_shape } => new_shape,
        _ => return true,
    };
    !bounds
        .iter()
        .any(|bound| matches!(bound, RtDim::InputAxis { tensor, .. } if *tensor == slot))
        || bounds
            .iter()
            .any(|bound| matches!(bound, RtDim::Node(input) if *input == slot))
}

/// Plan deferred recipes without changing the graph. A lane must retain
/// original admission/agreement/claim checks, implement every admitted
/// recipe and keep each dense base immutable through `base_last_read`.
///
/// Shared numeric expressions are materialization boundaries, reached through
/// movement chains too. Arithmetic feeding a broadcast is cached before that
/// broadcast, so caching never requires its replicated movement payload.
pub fn contraction_views(dag: &Dag, exposed: &[NodeId]) -> ContractionViews {
    contraction_views_excluding(dag, exposed, &[])
}

pub(crate) fn contraction_views_excluding(
    dag: &Dag,
    exposed: &[NodeId],
    opaque_consumers: &[NodeId],
) -> ContractionViews {
    contraction_views_with_products(dag, exposed, opaque_consumers, &[])
}

/// [`contraction_views_excluding`], where each inlined product's kernel
/// also reads its expansions' private reshapes as views of their dense
/// operands (chelis#3802). Such a reshape is deferred and read by the
/// kernel, at the kernel's position; its operand stays dense, so the read
/// is the operand's own element order.
pub(crate) fn contraction_views_with_products(
    dag: &Dag,
    exposed: &[NodeId],
    opaque_consumers: &[NodeId],
    products: &[crate::fuse::InlinedProduct],
) -> ContractionViews {
    let mut opaque = vec![false; dag.len()];
    for node in opaque_consumers {
        opaque[node.0] = true;
    }
    let candidate: Vec<_> = dag
        .nodes()
        .iter()
        .map(|node| {
            recipe(node)
                && !opaque[node.id.0]
                && (!matches!(node.op, RiscOp::Mul)
                    || (node.inputs.len() == 2
                        && node.inputs.iter().all(|input| {
                            dag.get(*input).is_some_and(|source| {
                                source.output_type.dims.len() == node.output_type.dims.len()
                                    && source.output_type.precision == node.output_type.precision
                            })
                        })))
        })
        .collect();
    let mut supported = vec![true; dag.len()];
    let mut last_read: Vec<_> = (0..dag.len()).collect();
    let mut reaches_sum = vec![false; dag.len()];
    for node in dag.nodes() {
        for (slot, input) in node.inputs.iter().enumerate() {
            last_read[input.0] = last_read[input.0].max(node.id.0);
            if payload_slot(node, slot) {
                supported[input.0] &= candidate[node.id.0]
                    || (matches!(node.op, RiscOp::Sum { .. }) && node.owner.activation.is_none());
            }
        }
        for input in node
            .shape_deps
            .iter()
            .chain(&node.result_claim_deps)
            .chain(&node.owner.activation)
        {
            last_read[input.0] = last_read[input.0].max(node.id.0);
        }
        if matches!(node.op, RiscOp::Sum { .. }) && node.owner.activation.is_none() {
            reaches_sum[node.inputs[0].0] = true;
        }
    }
    for node in dag.nodes().iter().rev() {
        if reaches_sum[node.id.0] && candidate[node.id.0] {
            for (slot, input) in node.inputs.iter().enumerate() {
                if payload_slot(node, slot) {
                    reaches_sum[input.0] = true;
                }
            }
        }
    }
    let mut kernel_view = vec![false; dag.len()];
    let mut dense = vec![false; dag.len()];
    for product in products {
        for reshape in product.view_reshapes.iter().flatten() {
            if candidate[reshape.0] {
                kernel_view[reshape.0] = true;
                dense[dag.nodes()[reshape.0].inputs[0].0] = true;
                last_read[reshape.0] = last_read[reshape.0].max(product.output().0);
            }
        }
    }
    for root in dag.roots().iter().chain(exposed) {
        supported[root.0] = false;
        kernel_view[root.0] = false;
        last_read[root.0] = dag.len();
    }
    let eligible: Vec<_> = dag
        .nodes()
        .iter()
        .map(|node| {
            !dense[node.id.0]
                && (kernel_view[node.id.0]
                    || (candidate[node.id.0] && reaches_sum[node.id.0] && supported[node.id.0]))
        })
        .collect();
    let movement: Vec<_> = dag
        .nodes()
        .iter()
        .map(|node| eligible[node.id.0] && !matches!(node.op, RiscOp::Mul))
        .collect();
    // Demand saturates at "many": only whether arithmetic can be repeated
    // matters. Each node/edge is visited a fixed number of times.
    let mut demand = vec![0u8; dag.len()];
    let mut cache = vec![false; dag.len()];
    for node in dag.nodes() {
        if movement[node.id.0] {
            continue;
        }
        for (slot, input) in node.inputs.iter().enumerate() {
            if payload_slot(node, slot) {
                demand[input.0] = (demand[input.0] + 1).min(2);
            }
        }
    }
    for node in dag.nodes().iter().rev() {
        if movement[node.id.0] {
            let source = node.inputs[0];
            demand[source.0] = (demand[source.0] + demand[node.id.0]).min(2);
            cache[source.0] |= cache[node.id.0] || matches!(node.op, RiscOp::Expand { .. });
        }
    }
    let deferred: Vec<_> = dag
        .nodes()
        .iter()
        .map(|node| {
            eligible[node.id.0]
                && (!matches!(node.op, RiscOp::Mul)
                    || (demand[node.id.0] <= 1 && !cache[node.id.0]))
        })
        .collect();
    let mut through = last_read.clone();
    let mut base_last_read = vec![None; dag.len()];
    for node in dag.nodes().iter().rev() {
        if !deferred[node.id.0] {
            continue;
        }
        // Bounds and extent carriers are read when the original node checks
        // its recipe, never when a later consumer reads an element.
        let operands = if matches!(node.op, RiscOp::Mul) { 2 } else { 1 };
        for input in node.inputs.iter().take(operands) {
            if deferred[input.0] {
                through[input.0] = through[input.0].max(through[node.id.0]);
            } else {
                base_last_read[input.0] = Some(
                    base_last_read[input.0]
                        .unwrap_or(input.0)
                        .max(through[node.id.0]),
                );
            }
        }
    }
    ContractionViews {
        deferred,
        last_read,
        base_last_read,
    }
}
