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
fn payload_slot(node: &DagNode, slot: usize) -> bool {
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
/// Shared numeric expressions are materialization boundaries. This applies
/// through movement chains too: duplicating a shared reshape of a product
/// would otherwise expand a cached DAG into an exponential expression tree.
pub fn contraction_views(dag: &Dag, exposed: &[NodeId]) -> ContractionViews {
    let candidate: Vec<_> = dag
        .nodes()
        .iter()
        .map(|node| {
            recipe(node)
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
    let mut payload_readers = vec![0usize; dag.len()];
    let mut supported = vec![true; dag.len()];
    let mut last_read: Vec<_> = (0..dag.len()).collect();
    let mut reaches_sum = vec![false; dag.len()];
    for node in dag.nodes() {
        for (slot, input) in node.inputs.iter().enumerate() {
            last_read[input.0] = last_read[input.0].max(node.id.0);
            if payload_slot(node, slot) {
                payload_readers[input.0] += 1;
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
    for root in dag.roots().iter().chain(exposed) {
        supported[root.0] = false;
        last_read[root.0] = dag.len();
    }
    let mut deferred = vec![false; dag.len()];
    let mut numeric_ancestry = vec![false; dag.len()];
    for node in dag.nodes() {
        let arithmetic = matches!(node.op, RiscOp::Mul)
            || node
                .inputs
                .iter()
                .enumerate()
                .any(|(slot, input)| payload_slot(node, slot) && numeric_ancestry[input.0]);
        deferred[node.id.0] = candidate[node.id.0]
            && reaches_sum[node.id.0]
            && supported[node.id.0]
            && (!arithmetic || payload_readers[node.id.0] <= 1);
        numeric_ancestry[node.id.0] = deferred[node.id.0] && arithmetic;
    }
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
