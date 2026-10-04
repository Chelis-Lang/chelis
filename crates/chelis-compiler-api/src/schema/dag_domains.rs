//! Numeric field domains and reference scopes, before a WireDag is admitted.
use super::{
    NonnegativeExtent, WireDag, WireDagContractError, WireDagNode, WireDimExpr, WireDimInfo,
    WireExtentWitnessSite, WireFusedInput, WireKeyBranch, WireLogicalKind, WireRiscOp, WireRtAxis,
    WireRtDim, host_index, wire_dim_info_equal,
};
use chelis_ir::dag::{DimInfo, KeyBranch};
use chelis_ir::verify::{
    BoundInput, ExtentSlot, KeyGraph, KeyRole, SlotRead, SplitCount, bound_slot_read,
    is_const_false, is_const_true, operand_extent_read, verify_key_rules, verify_random_operands,
};
use chelis_types::types::Prim;
use std::borrow::Cow;

type Result<T> = std::result::Result<T, WireDagContractError>;
fn reject(message: impl Into<String>) -> WireDagContractError {
    WireDagContractError::new(message)
}

fn input<'a>(dag: &'a WireDag, node: &WireDagNode) -> Result<&'a WireDagNode> {
    let id = node
        .inputs
        .first()
        .ok_or_else(|| reject("operation requires an input"))?;
    let index =
        usize::try_from(*id).map_err(|_| reject("input reference exceeds host capacity"))?;
    dag.nodes
        .get(index)
        .ok_or_else(|| reject("input reference is outside the owning DAG"))
}

fn axis(dag: &WireDag, node: &WireDagNode, axis: i32) -> Result<()> {
    let axis = usize::try_from(axis)
        .map_err(|_| reject("wire axis must be a normalized nonnegative int32"))?;
    if axis >= input(dag, node)?.output_type.dims.len() {
        return Err(reject("wire axis is outside its input rank"));
    }
    Ok(())
}

fn sparse_batch_prefix(
    dag: &WireDag,
    node: &WireDagNode,
    axis: i32,
    batch_rank: u32,
) -> Result<()> {
    let base = input(dag, node)?;
    let indices = node
        .inputs
        .get(1)
        .and_then(|id| usize::try_from(*id).ok())
        .and_then(|id| dag.nodes.get(id))
        .ok_or_else(|| reject("wire sparse operation requires an indices input"))?;
    let axis = usize::try_from(axis).map_err(|_| reject("wire sparse axis is negative"))?;
    let batch_rank = batch_rank as usize;
    if batch_rank > axis
        || batch_rank > indices.output_type.dims.len()
        || !base.output_type.dims[..batch_rank]
            .iter()
            .zip(&indices.output_type.dims[..batch_rank])
            .all(|(left, right)| wire_dim_info_equal(left, right))
    {
        return Err(reject(
            "wire sparse operation has invalid paired batch prefix",
        ));
    }
    Ok(())
}

fn extent(value: i64) -> Result<()> {
    if value < 0 {
        Err(reject("wire extent must be a nonnegative int64"))
    } else {
        Ok(())
    }
}
fn expression(value: &WireDimExpr) -> Result<()> {
    match value {
        WireDimExpr::Concrete { value } => extent(value.get()),
        WireDimExpr::Sym { .. } => Ok(()),
        WireDimExpr::Mul { lhs, rhs } | WireDimExpr::Div { lhs, rhs } => {
            expression(lhs)?;
            expression(rhs)
        }
    }
}
fn bound(value: &WireRtDim) -> Result<()> {
    if let WireRtDim::Lit { value } = value {
        extent(value.get())?;
    }
    Ok(())
}

/// spec/10 §3.2's random operand rules and key rules, applied by the IR
/// verifier's own implementations over the decoded graph: the operand rules
/// first, and the key rules over a graph whose operands hold. Every
/// reference has already been checked to name an earlier node, and every
/// root a node of this graph.
fn key_rules(dag: &WireDag) -> Result<()> {
    let mut errors = Vec::new();
    verify_random_operands(&DecodedKeys(dag), &mut errors);
    if errors.is_empty() {
        verify_key_rules(&DecodedKeys(dag), &mut errors);
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(reject(errors.join("; ")))
    }
}

fn wire_position(id: u64) -> Option<usize> {
    usize::try_from(id).ok()
}

/// The decoded graph as the key rules read it. A private view, so the rules
/// add nothing to `WireDag`'s public surface.
struct DecodedKeys<'a>(&'a WireDag);

impl KeyGraph for DecodedKeys<'_> {
    fn node_count(&self) -> usize {
        self.0.nodes.len()
    }

    fn role(&self, node: usize) -> KeyRole {
        match self.0.nodes.get(node).map(|node| &node.op) {
            Some(WireRiscOp::KeyFromSeed {}) => KeyRole::KeyFromSeed,
            Some(WireRiscOp::Split { branch }) => KeyRole::Split {
                branch: match branch {
                    WireKeyBranch::Left => KeyBranch::Left,
                    WireKeyBranch::Right => KeyBranch::Right,
                },
            },
            Some(WireRiscOp::FoldIn {}) => KeyRole::FoldIn,
            Some(WireRiscOp::KeySelect {}) => KeyRole::KeySelect,
            Some(WireRiscOp::SplitN { count }) => KeyRole::SplitN {
                count: match count {
                    WireRtDim::Lit { value } => {
                        usize::try_from(value.get()).map_or(SplitCount::Other, SplitCount::Lit)
                    }
                    WireRtDim::Node { input } => {
                        usize::try_from(*input).map_or(SplitCount::Other, SplitCount::Input)
                    }
                    _ => SplitCount::Other,
                },
            },
            Some(WireRiscOp::Load { .. }) => KeyRole::Load,
            Some(WireRiscOp::Store { .. }) => KeyRole::Store,
            Some(WireRiscOp::Dropout {}) => KeyRole::Dropout,
            Some(WireRiscOp::UniformLike {}) => KeyRole::UniformLike,
            Some(WireRiscOp::DropoutReplay {}) => KeyRole::DropoutReplay,
            Some(WireRiscOp::UniformBoundAdjoint { .. }) => KeyRole::UniformBoundAdjoint,
            Some(WireRiscOp::Drop) => KeyRole::Drop,
            Some(WireRiscOp::Logical {
                logical: WireLogicalKind::And,
            }) => KeyRole::And,
            Some(WireRiscOp::Logical {
                logical: WireLogicalKind::Not,
            }) => KeyRole::Not,
            Some(WireRiscOp::Const { value }) if is_const_false(value) => KeyRole::ConstFalse,
            Some(WireRiscOp::Const { value }) if is_const_true(value) => KeyRole::ConstTrue,
            _ => KeyRole::Other,
        }
    }

    fn slot_read(&self, node: usize, slot: usize) -> SlotRead {
        self.0
            .nodes
            .get(node)
            .map_or(SlotRead::Value, |node| wire_slot_read(&node.op, slot))
    }

    fn dtype(&self, node: usize) -> Option<Prim> {
        Prim::parse_interchange_name(&self.0.nodes.get(node)?.output_type.precision)
    }

    fn same_type(&self, left: usize, right: usize) -> bool {
        match (self.0.nodes.get(left), self.0.nodes.get(right)) {
            (Some(left), Some(right)) => {
                left.output_type.precision == right.output_type.precision
                    && left.output_type.dims.len() == right.output_type.dims.len()
                    && left
                        .output_type
                        .dims
                        .iter()
                        .zip(&right.output_type.dims)
                        .all(|(a, b)| wire_dim_info_equal(a, b))
            }
            _ => false,
        }
    }

    fn input(&self, node: usize, slot: usize) -> Option<usize> {
        wire_position(*self.0.nodes.get(node)?.inputs.get(slot)?)
    }

    fn activation(&self, node: usize) -> Option<usize> {
        wire_position(self.0.nodes.get(node)?.activation?)
    }

    fn roots(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.roots.iter().copied().filter_map(wire_position)
    }

    fn load_name(&self, node: usize) -> Option<&str> {
        match &self.0.nodes.get(node)?.op {
            WireRiscOp::Load { name } => Some(name),
            _ => None,
        }
    }

    fn declaration(&self, node: usize) -> &str {
        self.0
            .nodes
            .get(node)
            .and_then(|node| usize::try_from(node.declaration).ok())
            .and_then(|row| self.0.declarations.get(row))
            .map_or("", String::as_str)
    }

    /// Two nodes' declarations are one when they name one row: a name
    /// alone is not an identity, since two rows may share it.
    fn same_declaration(&self, left: usize, right: usize) -> bool {
        match (self.0.nodes.get(left), self.0.nodes.get(right)) {
            (Some(left), Some(right)) => left.declaration == right.declaration,
            _ => false,
        }
    }

    /// Each wire dim as the IR dim it decodes to. An extent beyond the
    /// host's `usize` has no IR reading, so its node has no dims and every
    /// rule that reads them rejects it.
    fn dims(&self, node: usize) -> Option<Cow<'_, [DimInfo]>> {
        let host = |size: &NonnegativeExtent| usize::try_from(size.get()).ok();
        self.0
            .nodes
            .get(node)?
            .output_type
            .dims
            .iter()
            .map(|dim| match dim {
                WireDimInfo::Lit { size } => host(size).map(DimInfo::Lit),
                WireDimInfo::Named { name, size } => match size {
                    Some(size) => host(size).map(|size| DimInfo::Named(name.clone(), Some(size))),
                    None => Some(DimInfo::Named(name.clone(), None)),
                },
            })
            .collect::<Option<Vec<_>>>()
            .map(Cow::Owned)
    }
}

/// `chelis_ir::verify::slot_read` of the operation `op` encodes: the same
/// rows, exhaustive with no wildcard arm, each bound read by the shared
/// `bound_slot_read`.
pub(crate) fn wire_slot_read(op: &WireRiscOp, slot: usize) -> SlotRead {
    let of = |bound: &WireRtDim| match bound {
        WireRtDim::Node { input } => {
            usize::try_from(*input).map_or(BoundInput::None, BoundInput::Value)
        }
        WireRtDim::InputAxis { tensor, .. } => {
            usize::try_from(*tensor).map_or(BoundInput::None, BoundInput::Extent)
        }
        WireRtDim::Lit { .. } | WireRtDim::ToEnd | WireRtDim::Sym { .. } => BoundInput::None,
    };
    let bounds = |kind, bounds: &mut dyn Iterator<Item = &WireRtDim>| {
        bound_slot_read(kind, bounds.map(of), slot)
    };
    match op {
        WireRiscOp::Shape { .. } => operand_extent_read(ExtentSlot::Shape, slot),
        WireRiscOp::ExtentWitness { .. } => operand_extent_read(ExtentSlot::ExtentWitness, slot),
        WireRiscOp::Expand { size, .. } => {
            bounds(ExtentSlot::ExpandSize, &mut std::iter::once(size))
        }
        WireRiscOp::Reshape { new_shape } => {
            bounds(ExtentSlot::ReshapeTarget, &mut new_shape.iter())
        }
        WireRiscOp::Pad { padding, .. } => bounds(
            ExtentSlot::PadBound,
            &mut padding.iter().flat_map(|(before, after)| [before, after]),
        ),
        WireRiscOp::Shrink { bounds: pairs } => bounds(
            ExtentSlot::ShrinkBound,
            &mut pairs.iter().flat_map(|(start, end)| [start, end]),
        ),
        WireRiscOp::Stride { strides } => bounds(ExtentSlot::StrideStep, &mut strides.iter()),
        WireRiscOp::SplitN { count } => bounds(ExtentSlot::SplitCount, &mut std::iter::once(count)),
        WireRiscOp::ListMapCapture { .. }
        | WireRiscOp::OrderedAdjointSum { .. }
        | WireRiscOp::Iota
        | WireRiscOp::Add
        | WireRiscOp::Sub
        | WireRiscOp::Mul
        | WireRiscOp::Div
        | WireRiscOp::FloorDiv
        | WireRiscOp::TruncDiv
        | WireRiscOp::Mod
        | WireRiscOp::Bitwise { .. }
        | WireRiscOp::Compare { .. }
        | WireRiscOp::Logical { .. }
        | WireRiscOp::Where {}
        | WireRiscOp::GuardedFail { .. }
        | WireRiscOp::MaxElem
        | WireRiscOp::MinElem
        | WireRiscOp::ExtremaAdjoint { .. }
        | WireRiscOp::Relu
        | WireRiscOp::Softmax { .. }
        | WireRiscOp::ReluAdjoint
        | WireRiscOp::Neg
        | WireRiscOp::Recip
        | WireRiscOp::Exp
        | WireRiscOp::Log
        | WireRiscOp::Sin
        | WireRiscOp::Sqrt
        | WireRiscOp::Cos
        | WireRiscOp::Tan
        | WireRiscOp::Atan
        | WireRiscOp::Tanh
        | WireRiscOp::Abs
        | WireRiscOp::Floor
        | WireRiscOp::Ceil
        | WireRiscOp::Round
        | WireRiscOp::UniformLike {}
        | WireRiscOp::Dropout {}
        | WireRiscOp::DropoutReplay {}
        | WireRiscOp::UniformBoundAdjoint { .. }
        | WireRiscOp::KeyFromSeed {}
        | WireRiscOp::Split { .. }
        | WireRiscOp::FoldIn {}
        | WireRiscOp::KeySelect {}
        | WireRiscOp::Sum { .. }
        | WireRiscOp::Count { .. }
        | WireRiscOp::MaxReduce { .. }
        | WireRiscOp::MinReduce { .. }
        | WireRiscOp::ProdReduce { .. }
        | WireRiscOp::ReduceWindow { .. }
        | WireRiscOp::ReduceWindowGrad { .. }
        | WireRiscOp::Argmax { .. }
        | WireRiscOp::Argmin { .. }
        | WireRiscOp::Permute { .. }
        | WireRiscOp::OneHot { .. }
        | WireRiscOp::CheckedReshapeExtent { .. }
        | WireRiscOp::CheckedUnitAxis { .. }
        | WireRiscOp::Const { .. }
        | WireRiscOp::ConstTensor { .. }
        | WireRiscOp::Load { .. }
        | WireRiscOp::Store { .. }
        | WireRiscOp::Copy
        | WireRiscOp::Drop
        | WireRiscOp::Realize
        | WireRiscOp::Cast { .. }
        | WireRiscOp::NamedCast { .. }
        | WireRiscOp::FusedElem { .. }
        | WireRiscOp::BlasMatmul { .. }
        | WireRiscOp::Gather { .. }
        | WireRiscOp::ScatterAdd { .. }
        | WireRiscOp::Scatter { .. }
        | WireRiscOp::ScatterElements { .. } => SlotRead::Value,
    }
}

fn input_rank(dag: &WireDag, node: &WireDagNode, slot: usize) -> Option<usize> {
    node.inputs
        .get(slot)
        .and_then(|id| usize::try_from(*id).ok())
        .and_then(|id| dag.nodes.get(id))
        .map(|input| input.output_type.dims.len())
}

fn rt_dim_supplies_extent(dim: &WireRtDim) -> bool {
    matches!(
        dim,
        WireRtDim::Lit { .. } | WireRtDim::Node { .. } | WireRtDim::InputAxis { .. }
    )
}

fn is_same_shape_result_op(op: &WireRiscOp) -> bool {
    matches!(
        op,
        WireRiscOp::Add
            | WireRiscOp::Sub
            | WireRiscOp::Mul
            | WireRiscOp::Div
            | WireRiscOp::FloorDiv
            | WireRiscOp::TruncDiv
            | WireRiscOp::Mod
            | WireRiscOp::Bitwise { .. }
            | WireRiscOp::Compare { .. }
            | WireRiscOp::Logical { .. }
            | WireRiscOp::Where { .. }
            | WireRiscOp::MaxElem
            | WireRiscOp::MinElem
            | WireRiscOp::ExtremaAdjoint { .. }
            | WireRiscOp::Relu
            | WireRiscOp::Softmax { .. }
            | WireRiscOp::ReluAdjoint
            | WireRiscOp::Neg
            | WireRiscOp::Recip
            | WireRiscOp::Exp
            | WireRiscOp::Log
            | WireRiscOp::Sin
            | WireRiscOp::Sqrt
            | WireRiscOp::Cos
            | WireRiscOp::Tan
            | WireRiscOp::Atan
            | WireRiscOp::Tanh
            | WireRiscOp::Abs
            | WireRiscOp::Floor
            | WireRiscOp::Ceil
            | WireRiscOp::Round
            | WireRiscOp::UniformLike {}
            | WireRiscOp::Dropout {}
            | WireRiscOp::DropoutReplay {}
            | WireRiscOp::Cast { .. }
            | WireRiscOp::NamedCast { .. }
            | WireRiscOp::FusedElem { .. }
    )
}

/// Validate the complete positive-rank agreement relation carried by a
/// same-shape producer. Rank-zero operands are scalar values, identical node
/// references deduplicate naturally, and every distinct positive-rank member
/// must have the result rank.
fn same_shape_result_relation_is_supported(dag: &WireDag, node: &WireDagNode) -> bool {
    if !is_same_shape_result_op(&node.op) {
        return false;
    }
    let rank = node.output_type.dims.len();
    if rank == 0 {
        return false;
    }
    let mut members = std::collections::BTreeSet::new();
    // A draw's data operand is its only same-shape operand; its controls, key
    // and activation are shaped like leading parts of the data.
    let operands = match node.op {
        WireRiscOp::UniformLike {} | WireRiscOp::Dropout {} | WireRiscOp::DropoutReplay {} => {
            &node.inputs[..node.inputs.len().min(1)]
        }
        _ => &node.inputs[..],
    };
    for input in operands {
        let Some(input) = usize::try_from(*input)
            .ok()
            .and_then(|input| dag.nodes.get(input))
        else {
            return false;
        };
        let input_rank = input.output_type.dims.len();
        if input_rank == 0 {
            continue;
        }
        if input_rank != rank {
            return false;
        }
        members.insert(input.id);
    }
    !members.is_empty()
}

/// Match lowering's admission rule for a declaration token retained by this
/// producing axis. An op-computed extent is usable only where the existing
/// guard derivation can compute it before the producer allocates its result.
fn literal_result_axis_is_supported(dag: &WireDag, node: &WireDagNode, axis: usize) -> bool {
    let rank = node.output_type.dims.len();
    if axis >= rank {
        return false;
    }
    if is_same_shape_result_op(&node.op) {
        return same_shape_result_relation_is_supported(dag, node);
    }
    match &node.op {
        WireRiscOp::Copy | WireRiscOp::Drop | WireRiscOp::Realize | WireRiscOp::Store { .. } => {
            node.inputs.iter().any(|id| {
                usize::try_from(*id)
                    .ok()
                    .and_then(|id| dag.nodes.get(id))
                    .is_some_and(|input| input.output_type.dims.len() == rank)
            })
        }
        WireRiscOp::Sum { axis: reduced, .. }
        | WireRiscOp::MaxReduce { axis: reduced }
        | WireRiscOp::MinReduce { axis: reduced }
        | WireRiscOp::ProdReduce { axis: reduced }
        | WireRiscOp::Argmax { axis: reduced }
        | WireRiscOp::Argmin { axis: reduced } => {
            let Some(input_rank) = input_rank(dag, node, 0) else {
                return false;
            };
            usize::try_from(*reduced)
                .ok()
                .is_some_and(|reduced| reduced < input_rank && input_rank - 1 == rank)
        }
        WireRiscOp::Count { axes } => {
            let Some(input_rank) = input_rank(dag, node, 0) else {
                return false;
            };
            let normalized = axes
                .iter()
                .map(|axis| usize::try_from(*axis))
                .collect::<std::result::Result<std::collections::BTreeSet<_>, _>>();
            normalized.is_ok_and(|axes| {
                axes.iter().all(|axis| *axis < input_rank)
                    && input_rank.saturating_sub(axes.len()) == rank
            })
        }
        WireRiscOp::ReduceWindow { window_shape, .. } => {
            input_rank(dag, node, 0).is_some_and(|input_rank| {
                window_shape.len() <= input_rank
                    && input_rank == rank
                    && axis < input_rank - window_shape.len()
            })
        }
        WireRiscOp::ReduceWindowGrad { .. } => {
            input_rank(dag, node, 0).is_some_and(|input_rank| input_rank == rank)
        }
        WireRiscOp::Reshape { new_shape } => {
            new_shape.get(axis).is_some_and(rt_dim_supplies_extent)
        }
        WireRiscOp::Permute { axes } => axes.len() == rank,
        WireRiscOp::Expand {
            axis: expanded,
            size,
        } => {
            let Some(input_rank) = input_rank(dag, node, 0) else {
                return false;
            };
            let Some(expanded) = usize::try_from(*expanded).ok() else {
                return false;
            };
            ((rank == input_rank && expanded < input_rank)
                || (input_rank.checked_add(1) == Some(rank) && expanded <= input_rank))
                && (axis != expanded || rt_dim_supplies_extent(size))
        }
        WireRiscOp::OneHot { .. } => {
            input_rank(dag, node, 0).and_then(|input_rank| input_rank.checked_add(1)) == Some(rank)
        }
        WireRiscOp::Pad { padding, .. } | WireRiscOp::Shrink { bounds: padding } => {
            axis < padding.len()
        }
        WireRiscOp::Stride { strides } => matches!(
            strides.get(axis),
            Some(WireRtDim::Lit { value }) if value.get() == 1
        ),
        WireRiscOp::Shape { .. }
        | WireRiscOp::ExtentWitness { .. }
        | WireRiscOp::CheckedReshapeExtent { .. } => false,
        WireRiscOp::CheckedUnitAxis { .. } => true,
        WireRiscOp::Const { .. } | WireRiscOp::ConstTensor { .. } => {
            match node.output_type.dims.get(axis) {
                Some(WireDimInfo::Lit { .. } | WireDimInfo::Named { size: Some(_), .. }) => true,
                Some(WireDimInfo::Named { name, size: None }) => {
                    let sibling_shaped = node
                        .shape_deps
                        .first()
                        .and_then(|id| usize::try_from(*id).ok())
                        .and_then(|id| dag.nodes.get(id))
                        .is_some_and(|sibling| sibling.output_type.dims.len() == rank);
                    (!name.is_empty() && name != "*") || sibling_shaped
                }
                None => false,
            }
        }
        WireRiscOp::Load { .. } => true,
        WireRiscOp::BlasMatmul { batch_dims, .. } => {
            rank >= 2
                && batch_dims.len() == rank - 2
                && node.inputs.iter().any(|id| {
                    usize::try_from(*id)
                        .ok()
                        .and_then(|id| dag.nodes.get(id))
                        .is_some_and(|input| input.output_type.dims.len() == rank)
                })
                && axis < batch_dims.len()
        }
        WireRiscOp::Gather {
            axis: gathered_axis,
            batch_rank,
        } => {
            let Some(values_rank) = input_rank(dag, node, 0) else {
                return false;
            };
            let Some(indices_rank) = input_rank(dag, node, 1) else {
                return false;
            };
            usize::try_from(*gathered_axis)
                .ok()
                .is_some_and(|gathered_axis| {
                    gathered_axis < values_rank
                        && (*batch_rank as usize) <= gathered_axis
                        && (*batch_rank as usize) <= indices_rank
                        && values_rank
                            .checked_sub(1)
                            .and_then(|rank| rank.checked_add(indices_rank))
                            .and_then(|rank| rank.checked_sub(*batch_rank as usize))
                            == Some(rank)
                })
        }
        WireRiscOp::ScatterAdd { .. }
        | WireRiscOp::Scatter { .. }
        | WireRiscOp::ScatterElements { .. } => {
            input_rank(dag, node, 0).is_some_and(|input_rank| input_rank == rank)
        }
        _ => false,
    }
}

pub(super) fn validate(dag: &WireDag) -> Result<()> {
    let mut declared = vec![false; dag.declarations.len()];
    for node in &dag.nodes {
        let row = usize::try_from(node.declaration)
            .ok()
            .filter(|row| *row < declared.len())
            .ok_or_else(|| {
                reject("a node's declaration must be a row of the owning DAG's declaration table")
            })?;
        declared[row] = true;
    }
    if declared.contains(&false) {
        return Err(reject(
            "every row of the declaration table must be the declaration of some node",
        ));
    }
    for (index, node) in dag.nodes.iter().enumerate() {
        if node.id != host_index(index) {
            return Err(reject(
                "node id must equal its zero-based position in the owning DAG",
            ));
        }
        if node.inputs.iter().any(|id| *id >= host_index(index)) {
            return Err(reject(
                "input references must resolve to earlier nodes in the owning DAG",
            ));
        }
        if node.shape_deps.iter().any(|id| *id >= host_index(index)) {
            return Err(reject(
                "shape dependencies must resolve to earlier nodes in the owning DAG",
            ));
        }
        if let Some(activation) = node.activation {
            let bool_node = usize::try_from(activation)
                .ok()
                .filter(|_| activation < host_index(index))
                .and_then(|activation| dag.nodes.get(activation))
                .is_some_and(|activation| activation.output_type.precision == "bool");
            if !bool_node {
                return Err(reject(
                    "a node's activation must be an earlier bool node of the owning DAG",
                ));
            }
        }
        for dependency in &node.shape_deps {
            let required = &dag.nodes[*dependency as usize];
            if let WireRiscOp::ExtentWitness {
                site: WireExtentWitnessSite::LiteralResultClaim,
                axis: WireRtAxis::Lit { value },
                ..
            } = &required.op
            {
                if usize::try_from(*value)
                    .ok()
                    .is_none_or(|axis| !literal_result_axis_is_supported(dag, node, axis))
                {
                    return Err(reject(
                        "literal result dependency requires a supported producing axis",
                    ));
                }
                continue;
            }
            if let WireRiscOp::ExtentWitness {
                site:
                    WireExtentWitnessSite::LocalAscriptionClaim {
                        axis: WireRtAxis::Lit { value },
                        ..
                    },
                ..
            } = &required.op
            {
                if usize::try_from(*value)
                    .ok()
                    .is_none_or(|axis| !literal_result_axis_is_supported(dag, node, axis))
                {
                    return Err(reject(
                        "local ascription dependency requires a supported producing axis",
                    ));
                }
                continue;
            }
            let WireRiscOp::ExtentWitness {
                site:
                    WireExtentWitnessSite::ResultClaim {
                        axis: WireRtAxis::Lit { value },
                        ..
                    },
                ..
            } = &required.op
            else {
                continue;
            };
            let result_axis = usize::try_from(*value)
                .map_err(|_| reject("result claim axis must be normalized int32"))?;
            let supported = match &node.op {
                WireRiscOp::Expand { axis, size } => {
                    usize::try_from(*axis).ok() == Some(result_axis)
                        && matches!(size, WireRtDim::Node { .. } | WireRtDim::InputAxis { .. })
                }
                WireRiscOp::Reshape { new_shape } => matches!(
                    new_shape.get(result_axis),
                    Some(WireRtDim::Node { .. } | WireRtDim::InputAxis { .. })
                ),
                WireRiscOp::Shrink { .. } | WireRiscOp::Pad { .. } => true,
                // [05-OP-71]: a split's count axis, as an expansion's size;
                // the split checks it before any key exists.
                WireRiscOp::SplitN { .. } => result_axis + 1 == node.output_type.dims.len(),
                WireRiscOp::Iota | WireRiscOp::ListMapCapture { .. } => result_axis == 0,
                _ => same_shape_result_relation_is_supported(dag, node),
            };
            if !supported || result_axis >= node.output_type.dims.len() {
                return Err(reject(
                    "result claim dependency requires a supported producing axis",
                ));
            }
        }
        i32::try_from(node.output_type.dims.len())
            .map_err(|_| reject("tensor rank exceeds int32"))?;
        for dim in &node.output_type.dims {
            match dim {
                WireDimInfo::Lit { size }
                | WireDimInfo::Named {
                    size: Some(size), ..
                } => extent(size.get())?,
                WireDimInfo::Named { size: None, .. } => {}
            }
        }
        Prim::parse_interchange_name(&node.output_type.precision)
            .ok_or_else(|| reject("unknown output dtype"))?;
        match &node.op {
            WireRiscOp::Sum { axis: a, .. }
            | WireRiscOp::MaxReduce { axis: a }
            | WireRiscOp::MinReduce { axis: a }
            | WireRiscOp::ProdReduce { axis: a }
            | WireRiscOp::Argmax { axis: a }
            | WireRiscOp::Argmin { axis: a }
            | WireRiscOp::Shape { axis: a }
            | WireRiscOp::ScatterElements { axis: a } => axis(dag, node, *a)?,
            WireRiscOp::Gather {
                axis: a,
                batch_rank,
            }
            | WireRiscOp::ScatterAdd {
                axis: a,
                batch_rank,
            }
            | WireRiscOp::Scatter {
                axis: a,
                batch_rank,
            } => {
                axis(dag, node, *a)?;
                sparse_batch_prefix(dag, node, *a, *batch_rank)?;
            }
            WireRiscOp::ExtentWitness {
                site: WireExtentWitnessSite::LiteralResultClaim,
                parameter,
                axis: WireRtAxis::Lit { value },
                requirements,
                claims,
            } => {
                if *value < 0
                    || !parameter.is_empty()
                    || !node.inputs.is_empty()
                    || !node.shape_deps.is_empty()
                    || !claims.is_empty()
                    || requirements.len() != 1
                    || !node.output_type.dims.is_empty()
                    || node.output_type.precision != "int64"
                {
                    return Err(reject(
                        "literal result claim requires one nonnegative int64 literal, normalized axis, scalar output, and no observing or entry dependencies",
                    ));
                }
                extent(requirements[0].get())?;
                if dag
                    .nodes
                    .iter()
                    .filter(|owner| owner.shape_deps.contains(&node.id))
                    .count()
                    != 1
                {
                    return Err(reject(
                        "literal result claim requires exactly one producing owner",
                    ));
                }
            }
            WireRiscOp::ExtentWitness {
                site:
                    WireExtentWitnessSite::LocalAscriptionClaim {
                        binding,
                        claim,
                        axis:
                            WireRtAxis::Lit {
                                value: claimed_axis,
                            },
                        ..
                    },
                parameter,
                axis: WireRtAxis::Lit { value },
                requirements,
                claims,
            } => {
                if binding.is_empty()
                    || claim.is_empty()
                    || *claimed_axis < 0
                    || *claimed_axis != *value
                    || !claims.is_empty()
                    || !node.output_type.dims.is_empty()
                    || node.output_type.precision != "int64"
                {
                    return Err(reject(
                        "local ascription claim requires nonempty provenance, one normalized axis, scalar int64 output, and no entry claims",
                    ));
                }
                let literal = node.inputs.is_empty()
                    && node.shape_deps.is_empty()
                    && parameter.is_empty()
                    && requirements.len() == 1;
                if literal {
                    extent(requirements[0].get())?;
                } else {
                    if parameter.is_empty() || !requirements.is_empty() || node.inputs.len() != 1 {
                        return Err(reject(
                            "local ascription claim must carry exactly one literal or one declaring witness",
                        ));
                    }
                    axis(dag, node, *value)?;
                    let declared = node
                        .shape_deps
                        .first()
                        .and_then(|required| dag.nodes.get(*required as usize));
                    let same_observation = node.shape_deps.len() == 1 && declared.is_some_and(|declared| {
                        matches!(declared.op, WireRiscOp::ExtentWitness { site: WireExtentWitnessSite::Caller, axis: WireRtAxis::Lit { value: observed }, .. } if observed == *value)
                            && declared.inputs.first() == node.inputs.first()
                            && declared.id < node.id
                    });
                    if !same_observation {
                        return Err(reject(
                            "named local ascription claim requires its exact earlier declaring observation",
                        ));
                    }
                }
                if dag
                    .nodes
                    .iter()
                    .filter(|owner| owner.shape_deps.contains(&node.id))
                    .count()
                    != 1
                {
                    return Err(reject(
                        "local ascription claim requires exactly one initializer owner",
                    ));
                }
            }
            WireRiscOp::ExtentWitness {
                site,
                axis: WireRtAxis::Lit { value },
                requirements,
                claims,
                ..
            } => {
                if let WireExtentWitnessSite::ResultClaim {
                    claim,
                    axis: WireRtAxis::Lit { value },
                } = site
                    && (claim.is_empty()
                        || *value < 0
                        || !requirements.is_empty()
                        || !claims.is_empty())
                {
                    return Err(reject(
                        "result claim witness requires a nonempty label, normalized axis, and no entry obligations",
                    ));
                }
                if matches!(site, WireExtentWitnessSite::ResultClaim { .. }) {
                    let declared = node
                        .shape_deps
                        .first()
                        .and_then(|required| dag.nodes.get(*required as usize));
                    let same_observation = node.shape_deps.len() == 1 && declared.is_some_and(|declared| {
                        matches!(declared.op, WireRiscOp::ExtentWitness { site: WireExtentWitnessSite::Caller, axis: WireRtAxis::Lit { value: observed }, .. } if observed == *value)
                            && declared.inputs.first() == node.inputs.first()
                            && declared.id < node.id
                    });
                    if !same_observation {
                        return Err(reject(
                            "result claim witness requires its exact earlier declaring observation",
                        ));
                    }
                }
                // wire v11: `inputs[0]` is the observed tensor; each named
                // claim adds one requirement edge naming an earlier witness.
                if node.inputs.len() != claims.len() + 1 {
                    return Err(reject(
                        "extent witness requires one earlier tensor input and one earlier witness input per named claim",
                    ));
                }
                axis(dag, node, *value)?;
                if !node.output_type.dims.is_empty() || node.output_type.precision != "int64" {
                    return Err(reject(
                        "extent witness output must be a rank-0 int64 tensor",
                    ));
                }
                for requirement in requirements {
                    extent(requirement.get())?;
                }
                for (claim, edge) in claims.iter().zip(node.inputs.iter().skip(1)) {
                    if claim.claim.is_empty() {
                        return Err(reject(
                            "an extent witness named claim requires a nonempty dimension binder",
                        ));
                    }
                    let required = usize::try_from(*edge)
                        .ok()
                        .and_then(|edge| dag.nodes.get(edge))
                        .filter(|required| required.id == *edge)
                        .ok_or_else(|| {
                            reject("an extent witness named claim requires an earlier node")
                        })?;
                    let witness = matches!(required.op, WireRiscOp::ExtentWitness { ref site, .. } if !matches!(site, WireExtentWitnessSite::LiteralResultClaim))
                        && required.output_type.dims.is_empty()
                        && required.output_type.precision == "int64";
                    if *edge >= node.id || !witness {
                        return Err(reject(
                            "an extent witness named claim requires an earlier rank-0 int64 extent witness",
                        ));
                    }
                }
            }
            WireRiscOp::Expand { axis: a, size } => {
                // The IR shares this variant between source expand and insert.
                // Insertion admits the trailing slot (and axis 0 at rank 0).
                let input_rank = input(dag, node)?.output_type.dims.len();
                let output_rank = node.output_type.dims.len();
                let a = usize::try_from(*a)
                    .map_err(|_| reject("wire axis must be a normalized nonnegative int32"))?;
                let valid = if output_rank == input_rank {
                    a < input_rank
                } else if input_rank.checked_add(1) == Some(output_rank) {
                    a <= input_rank
                } else {
                    false
                };
                if !valid {
                    return Err(reject(
                        "wire Expand axis or rank is outside its broadcast or insertion layout",
                    ));
                }
                bound(size)?;
            }
            WireRiscOp::Count { axes } => {
                for a in axes {
                    axis(dag, node, *a)?;
                }
            }
            WireRiscOp::Permute { axes } => {
                let rank = input(dag, node)?.output_type.dims.len();
                if axes.len() != rank
                    || axes
                        .iter()
                        .copied()
                        .collect::<std::collections::BTreeSet<_>>()
                        .len()
                        != rank
                {
                    return Err(reject(
                        "permutation must name every input axis exactly once",
                    ));
                }
                for a in axes {
                    axis(dag, node, *a)?;
                }
            }
            WireRiscOp::Reshape { new_shape } => {
                for value in new_shape {
                    bound(value)?;
                }
            }
            WireRiscOp::Pad { padding, .. } | WireRiscOp::Shrink { bounds: padding } => {
                for (start, end) in padding {
                    bound(start)?;
                    bound(end)?;
                }
            }
            WireRiscOp::Stride { strides } => {
                for value in strides {
                    bound(value)?;
                    if matches!(value, WireRtDim::Lit { value } if value.get() == 0) {
                        return Err(reject("stride must be positive"));
                    }
                }
            }
            WireRiscOp::ReduceWindow {
                window_shape,
                strides,
                ..
            }
            | WireRiscOp::ReduceWindowGrad {
                window_shape,
                strides,
                ..
            } => {
                let rank = input(dag, node)?.output_type.dims.len();
                if window_shape.len() != rank
                    || strides.len() != rank
                    || window_shape
                        .iter()
                        .chain(strides)
                        .any(|value| value.get() <= 0)
                {
                    return Err(reject(
                        "window shapes and strides must be positive int64 vectors matching rank",
                    ));
                }
            }
            WireRiscOp::OneHot { vocab } if vocab.get() <= 0 => {
                return Err(reject("one-hot vocabulary must be a positive int64"));
            }
            WireRiscOp::FusedElem { ops } => {
                for (step_index, step) in ops.iter().enumerate() {
                    for reference in &step.input_indices {
                        let valid = match reference {
                            WireFusedInput::External { index } => {
                                *index < host_index(node.inputs.len())
                            }
                            WireFusedInput::PreviousStep { index } => {
                                *index < host_index(step_index)
                            }
                        };
                        if !valid {
                            return Err(reject(
                                "fused reference does not resolve in its owning input or earlier-step scope",
                            ));
                        }
                    }
                }
            }
            WireRiscOp::BlasMatmul {
                batch_dims,
                m,
                n,
                k,
                ..
            } => {
                for value in batch_dims.iter().chain([m, n, k]) {
                    expression(value)?;
                }
            }
            // The random and key operations' operand rules are the IR
            // verifier's, applied by `key_rules`; only the wire's own
            // encodings are checked here.
            WireRiscOp::SplitN { count } => bound(count)?,
            _ => {}
        }
    }
    if dag
        .roots
        .iter()
        .any(|id| *id >= host_index(dag.nodes.len()))
    {
        return Err(reject("root reference is outside the owning DAG"));
    }
    key_rules(dag)
}
