//! Numeric field domains and reference scopes, before a WireDag is admitted.
use super::{
    WireDag, WireDagContractError, WireDagNode, WireDimExpr, WireDimInfo, WireExtentWitnessSite,
    WireFusedInput, WireRiscOp, WireRtAxis, WireRtDim, host_index, wire_dim_info_equal,
};
use chelis_ir::dag::RandomDraw;
use chelis_ir::verify::{KeyGraph, KeyRole, verify_key_rules};
use chelis_types::types::Prim;

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
fn input_at<'a>(dag: &'a WireDag, node: &WireDagNode, slot: usize) -> Result<&'a WireDagNode> {
    let id = node
        .inputs
        .get(slot)
        .ok_or_else(|| reject("random operation is missing an input"))?;
    let index =
        usize::try_from(*id).map_err(|_| reject("input reference exceeds host capacity"))?;
    dag.nodes
        .get(index)
        .ok_or_else(|| reject("input reference is outside the owning DAG"))
}

fn is_rank_zero(node: &WireDagNode, precision: &str) -> bool {
    node.output_type.dims.is_empty() && node.output_type.precision == precision
}

/// A random node's operand layout (spec/10 §3.2, v17): the data or template,
/// the controls, the key and one optional rank-zero Bool activation for the
/// primitives and their adjoints; the optional literal seed, the controls and
/// the optional activation for a draw key. Control values are checked at
/// execution under [05-OP-8]/[05-OP-37], never here.
fn random_node(dag: &WireDag, node: &WireDagNode, dtype: Prim) -> Result<()> {
    let float = |prim: Prim| matches!(prim, Prim::F16 | Prim::Bf16 | Prim::F32 | Prim::F64);
    let control = |slot: usize, draw_dtype: Prim, uniform: bool| -> Result<()> {
        let value = input_at(dag, node, slot)?;
        let admitted = value.output_type.dims.is_empty()
            && Prim::parse_interchange_name(&value.output_type.precision)
                .is_some_and(|prim| prim == draw_dtype || (uniform && prim == Prim::F32));
        if admitted {
            Ok(())
        } else {
            Err(reject(
                "random control must be a rank-zero value of the draw's dtype (f32 bounds admitted)",
            ))
        }
    };
    let activation = |slot: usize| -> Result<()> {
        match node.inputs.len() {
            count if count == slot => Ok(()),
            count if count == slot + 1 && is_rank_zero(input_at(dag, node, slot)?, "bool") => {
                Ok(())
            }
            _ => Err(reject(
                "random operation may end with exactly one rank-zero Bool activation",
            )),
        }
    };
    let key = |slot: usize| -> Result<()> {
        let key = input_at(dag, node, slot)?;
        if is_rank_zero(key, "key") && matches!(key.op, WireRiscOp::DrawKey { .. }) {
            Ok(())
        } else {
            Err(reject(
                "random operation requires a key produced by a draw key",
            ))
        }
    };
    let same_as_data = |data: &WireDagNode| {
        data.output_type.precision == node.output_type.precision
            && data.output_type.dims.len() == node.output_type.dims.len()
            && data
                .output_type
                .dims
                .iter()
                .zip(&node.output_type.dims)
                .all(|(a, b)| wire_dim_info_equal(a, b))
    };
    match &node.op {
        WireRiscOp::UniformLike {} => {
            if !float(dtype) || !same_as_data(input_at(dag, node, 0)?) {
                return Err(reject(
                    "uniform_like must preserve its float template's exact shape and dtype",
                ));
            }
            control(1, dtype, true)?;
            control(2, dtype, true)?;
            if input_at(dag, node, 1)?.output_type.precision
                != input_at(dag, node, 2)?.output_type.precision
            {
                return Err(reject("uniform_like bounds must share one dtype"));
            }
            key(3)?;
            activation(4)
        }
        WireRiscOp::Dropout {} | WireRiscOp::DropoutReplay {} => {
            if !float(dtype) || !same_as_data(input_at(dag, node, 0)?) {
                return Err(reject(
                    "dropout must preserve its float data input's exact shape and dtype",
                ));
            }
            control(1, dtype, false)?;
            key(2)?;
            activation(3)
        }
        WireRiscOp::UniformBoundAdjoint { .. } => {
            let template = input_at(dag, node, 0)?;
            if !float(dtype)
                || !node.output_type.dims.is_empty()
                || template.output_type.precision != node.output_type.precision
                || !same_as_data_shape(template, input_at(dag, node, 1)?)
            {
                return Err(reject(
                    "a uniform bound adjoint is a rank-zero value of its template's dtype over a same-shaped cotangent",
                ));
            }
            key(2)?;
            activation(3)
        }
        WireRiscOp::DrawKey {
            handler,
            draw,
            dtype: draw_dtype,
        } => {
            let draw_dtype = Prim::parse_interchange_name(draw_dtype)
                .filter(|prim| float(*prim))
                .ok_or_else(|| reject("draw key requires an active float draw dtype"))?;
            if !is_rank_zero(node, "key") {
                return Err(reject("draw key produces one rank-zero key"));
            }
            let seed_slots = match handler {
                super::WireRandomHandler::Inherited => 0,
                super::WireRandomHandler::Scoped { .. } => {
                    let seed = input_at(dag, node, 0)?;
                    if !is_rank_zero(seed, "int64") || !matches!(seed.op, WireRiscOp::Const { .. })
                    {
                        return Err(reject(
                            "a scoped draw key's first input is its rank-zero int64 literal seed",
                        ));
                    }
                    1
                }
            };
            let (controls, uniform) = match draw {
                super::WireRandomDraw::Dropout => (1, false),
                super::WireRandomDraw::UniformLike => (2, true),
            };
            for slot in seed_slots..seed_slots + controls {
                control(slot, draw_dtype, uniform)?;
            }
            activation(seed_slots + controls)
        }
        _ => unreachable!("random_node validates only random operations"),
    }
}

fn same_as_data_shape(template: &WireDagNode, cotangent: &WireDagNode) -> bool {
    template.output_type.dims.len() == cotangent.output_type.dims.len()
        && template
            .output_type
            .dims
            .iter()
            .zip(&cotangent.output_type.dims)
            .all(|(a, b)| wire_dim_info_equal(a, b))
}

/// spec/10 §3.2's key rules, applied by the IR verifier's own
/// implementation over the decoded graph. Every reference has already been
/// checked to name an earlier node, and every root a node of this graph.
fn key_rules(dag: &WireDag) -> Result<()> {
    let mut errors = Vec::new();
    verify_key_rules(&DecodedKeys(dag), &mut errors);
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
            Some(WireRiscOp::DrawKey {
                handler,
                draw,
                dtype,
            }) => {
                // `random_node` has already rejected a draw key whose dtype
                // is not an active float. Were one to reach the rules, it
                // would read as no draw key, and its key output would fail.
                let Some(dtype) = Prim::parse_interchange_name(dtype) else {
                    return KeyRole::Other;
                };
                KeyRole::DrawKey {
                    scoped: matches!(handler, super::WireRandomHandler::Scoped { .. }),
                    draw: match draw {
                        super::WireRandomDraw::Dropout => RandomDraw::Dropout,
                        super::WireRandomDraw::UniformLike => RandomDraw::UniformLike,
                    },
                    dtype,
                }
            }
            Some(WireRiscOp::Dropout {}) => KeyRole::Dropout,
            Some(WireRiscOp::UniformLike {}) => KeyRole::UniformLike,
            Some(WireRiscOp::DropoutReplay {}) => KeyRole::DropoutReplay,
            Some(WireRiscOp::UniformBoundAdjoint { .. }) => KeyRole::UniformBoundAdjoint,
            _ => KeyRole::Other,
        }
    }

    fn dtype(&self, node: usize) -> Option<Prim> {
        Prim::parse_interchange_name(&self.0.nodes.get(node)?.output_type.precision)
    }

    fn same_type(&self, left: usize, right: usize) -> bool {
        match (self.0.nodes.get(left), self.0.nodes.get(right)) {
            (Some(left), Some(right)) => {
                left.output_type.precision == right.output_type.precision
                    && same_as_data_shape(left, right)
            }
            _ => false,
        }
    }

    fn input(&self, node: usize, slot: usize) -> Option<usize> {
        wire_position(*self.0.nodes.get(node)?.inputs.get(slot)?)
    }

    fn dependencies(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.0
            .nodes
            .get(node)
            .into_iter()
            .flat_map(|node| node.shape_deps.iter().copied())
            .filter_map(wire_position)
    }

    fn roots(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.roots.iter().copied().filter_map(wire_position)
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
            | WireRiscOp::Compare { .. }
            | WireRiscOp::Logical { .. }
            | WireRiscOp::Where { .. }
            | WireRiscOp::MaxElem
            | WireRiscOp::MinElem
            | WireRiscOp::ExtremaAdjoint { .. }
            | WireRiscOp::Relu
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
            | WireRiscOp::Abs
            | WireRiscOp::Floor
            | WireRiscOp::Ceil
            | WireRiscOp::Round
            | WireRiscOp::UniformLike {}
            | WireRiscOp::Dropout {}
            | WireRiscOp::DropoutReplay {}
            | WireRiscOp::Cast { .. }
            | WireRiscOp::CastTrunc { .. }
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
    for input in &node.inputs {
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
                        && values_rank
                            .checked_sub(1)
                            .and_then(|rank| rank.checked_add(indices_rank))
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
                _ => same_shape_result_relation_is_supported(dag, node),
            };
            if !supported || result_axis >= node.output_type.dims.len() {
                return Err(reject(
                    "result claim dependency requires a supported producing axis",
                ));
            }
        }
        let owns_local_ascription = node.shape_deps.iter().any(|dependency| {
            dag.nodes
                .get(*dependency as usize)
                .is_some_and(|dependency| {
                    matches!(
                        dependency.op,
                        WireRiscOp::ExtentWitness {
                            site: WireExtentWitnessSite::LocalAscriptionClaim { .. },
                            ..
                        }
                    )
                })
        });
        if owns_local_ascription {
            let activation_count = node
                .shape_deps
                .iter()
                .filter_map(|dependency| dag.nodes.get(*dependency as usize))
                .filter(|dependency| {
                    dependency.output_type.dims.is_empty()
                        && dependency.output_type.precision == "bool"
                })
                .count();
            if activation_count > 1 {
                return Err(reject(
                    "local ascription owner has multiple runtime branch activations",
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
        let dtype = Prim::parse_interchange_name(&node.output_type.precision)
            .ok_or_else(|| reject("unknown output dtype"))?;
        match &node.op {
            WireRiscOp::Sum { axis: a, .. }
            | WireRiscOp::MaxReduce { axis: a }
            | WireRiscOp::MinReduce { axis: a }
            | WireRiscOp::ProdReduce { axis: a }
            | WireRiscOp::Argmax { axis: a }
            | WireRiscOp::Argmin { axis: a }
            | WireRiscOp::Shape { axis: a }
            | WireRiscOp::Gather { axis: a }
            | WireRiscOp::ScatterAdd { axis: a }
            | WireRiscOp::Scatter { axis: a }
            | WireRiscOp::ScatterElements { axis: a } => axis(dag, node, *a)?,
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
            WireRiscOp::UniformLike {}
            | WireRiscOp::Dropout {}
            | WireRiscOp::DropoutReplay {}
            | WireRiscOp::UniformBoundAdjoint { .. }
            | WireRiscOp::DrawKey { .. } => random_node(dag, node, dtype)?,
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
