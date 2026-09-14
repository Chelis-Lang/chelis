//! Numeric field domains and reference scopes, before a WireDag is admitted.
use super::{
    WireDag, WireDagContractError, WireDagNode, WireDimExpr, WireDimInfo, WireExtentWitnessSite,
    WireFusedInput, WireRiscOp, WireRtAxis, WireRtDim, host_index, wire_dim_info_equal,
};
use chelis_types::{ScalarValue, types::Prim};

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
fn float(value: ScalarValue, dtype: Prim) -> Result<f64> {
    if value.prim() != dtype || !matches!(dtype, Prim::F16 | Prim::Bf16 | Prim::F32 | Prim::F64) {
        return Err(reject(
            "random parameter must have the exact active float dtype",
        ));
    }
    let number = value.as_f64_lossy(); // Every admitted float widens exactly here.
    if !number.is_finite() {
        return Err(reject("random parameter must be finite"));
    }
    Ok(number)
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
                if usize::try_from(*value).map_or(true, |axis| axis >= node.output_type.dims.len())
                {
                    return Err(reject(
                        "literal result dependency requires a valid producing axis",
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
                _ => false,
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
        let dtype = Prim::parse_name(&node.output_type.precision)
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
            WireRiscOp::UniformLike { low, .. } | WireRiscOp::Dropout { rate: low, seed: _ } => {
                let template = input(dag, node)?;
                if matches!(&node.op, WireRiscOp::UniformLike { .. }) {
                    if !matches!(node.inputs.len(), 1 | 2) {
                        return Err(reject(
                            "uniform_like expects one template and at most one activation",
                        ));
                    }
                    if let [_, activation] = node.inputs.as_slice() {
                        let index = usize::try_from(*activation)
                            .map_err(|_| reject("input reference exceeds host capacity"))?;
                        let activation = dag
                            .nodes
                            .get(index)
                            .ok_or_else(|| reject("input reference is outside the owning DAG"))?;
                        // The IR's path-sensitive Random form adds a scalar
                        // Bool activation after the template (spec/10 §3.2).
                        if activation.output_type.precision != "bool"
                            || !activation.output_type.dims.is_empty()
                        {
                            return Err(reject(
                                "uniform_like requires a scalar Bool path activation",
                            ));
                        }
                    }
                } else if node.inputs.len() != 1 {
                    return Err(reject("dropout expects exactly one data input"));
                }
                if template.output_type.precision != node.output_type.precision
                    || template.output_type.dims.len() != node.output_type.dims.len()
                    || template
                        .output_type
                        .dims
                        .iter()
                        .zip(&node.output_type.dims)
                        .any(|(a, b)| !wire_dim_info_equal(a, b))
                {
                    return Err(reject(
                        "random operation must preserve its data input's exact shape and dtype",
                    ));
                }
                let low = float(*low, dtype)?;
                if let WireRiscOp::UniformLike { high, .. } = &node.op {
                    let high = float(*high, dtype)?;
                    let finite_difference = if dtype == Prim::F64 {
                        (high - low).is_finite()
                    } else {
                        ((high as f32) - (low as f32)).is_finite()
                    };
                    if low > high || !finite_difference {
                        return Err(reject(
                            "uniform bounds must be ordered with finite difference at the arithmetic width",
                        ));
                    }
                } else if !(0.0..1.0).contains(&low) {
                    return Err(reject("dropout rate must satisfy 0 <= rate < 1"));
                }
            }
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
    Ok(())
}
