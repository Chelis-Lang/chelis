//! The identity a compiled lane gives an anonymous extent.
//!
//! A declared type may spell an extent as `*` (or leave it empty), which
//! names no identity. The C backend's `rename_anonymous_dims` gives every
//! such axis one before emission: a node that keeps its operand's extent
//! takes the operand's identity, and a node that makes an extent of its own
//! gets a fresh `_anon_dim_{node}_{axis}`. Two axes with one identity are
//! one extent to the C lane, which is how it reads `where`'s "exactly
//! matching shape". [`anonymous_axis_names`] is that rule, applied by the C
//! backend; [`anonymous_axis_identity`] follows it from one axis to the
//! identity it ends at, which the runtime `if` join reads
//! (`lower.rs`, `join_condition_extents`) so the lanes agree on which arms
//! share an extent.

use crate::axis_sources::{AxisSource, output_axis_sources};
use crate::dag::{Dag, DimInfo, NodeId, RiscOp, RtAxis, RtDim};

/// How one output axis of a node with an anonymous extent is named.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnonymousAxisName {
    /// The axis keeps its own dimension, which is not anonymous.
    Own,
    /// The static extent.
    Literal(usize),
    /// The dimension `node`'s axis `axis` is given.
    Of { node: NodeId, axis: usize },
    /// A fresh identity ([`fresh_anonymous_dim`]).
    Fresh,
}

fn is_anonymous(name: &str) -> bool {
    name.is_empty() || name == "*"
}

/// `dim` given the fresh identity of `node`'s axis `axis`, keeping its size.
pub fn fresh_anonymous_dim(node: NodeId, axis: usize, dim: &DimInfo) -> DimInfo {
    match dim {
        DimInfo::Named(name, size) if is_anonymous(name) => {
            DimInfo::Named(format!("_anon_dim_{}_{}", node.0, axis), *size)
        }
        other => other.clone(),
    }
}

/// chelis#616 (soundness): a movement op with any NON-IDENTITY axis (a
/// node-valued bound, a non-sentinel shrink, a stride step other than
/// literal 1, or a non-zero pad) produces a FRESH output extent on that
/// axis, which is NOT the input axis extent. The "copy first-input dims"
/// rule would clobber such an axis with the input's dim (e.g. propagate a
/// shrink's `_anon_dim` onto a stride's output, making two different extents
/// share one C variable), so it does not apply, and each anonymous axis gets
/// a fresh `_anon_dim_{id}_{axis}` that `emit_shrink`/`emit_stride`/`emit_pad`
/// size from its own bounds. Mirrors the identity-only pass-through rule in
/// `chelis_ir::dag::shape_source_for_axis`.
fn movement_alters_extents(op: &RiscOp) -> bool {
    match op {
        RiscOp::Shrink { bounds } => bounds
            .iter()
            .any(|(s, e)| !(s.as_lit() == Some(0) && matches!(e, RtDim::ToEnd))),
        RiscOp::Pad { padding, .. } => padding
            .iter()
            .any(|(b, a)| !(b.as_lit() == Some(0) && a.as_lit() == Some(0))),
        RiscOp::Stride { strides } => strides.iter().any(|s| s.as_lit() != Some(1)),
        // A runtime reshape target's extent comes from its scalar, never from
        // the input's dims.
        RiscOp::Reshape { new_shape } => new_shape.iter().any(|d| d.node_input().is_some()),
        _ => false,
    }
}

/// How each output axis of `id` is named, or `None` when its type spells no
/// anonymous extent. Every `Of` names a node the naming reads as it stands
/// in `dag`, so a caller naming a whole DAG does so in node order.
pub fn anonymous_axis_names(dag: &Dag, id: NodeId) -> Option<Vec<AnonymousAxisName>> {
    let node = dag.get(id)?;
    let dims = &node.output_type.dims;
    let anonymous = |dim: &DimInfo| matches!(dim, DimInfo::Named(name, _) if is_anonymous(name));
    if !dims.iter().any(anonymous) {
        return None;
    }
    let fresh_or_own = |dim: &DimInfo| {
        if anonymous(dim) {
            AnonymousAxisName::Fresh
        } else {
            AnonymousAxisName::Own
        }
    };
    // The axis `read` of the tensor in input slot `slot`, when it has one.
    let input_axis = |slot: usize, read: i32| {
        let input = *node.inputs.get(slot)?;
        let read = usize::try_from(read).ok()?;
        (read < dag.get(input)?.output_type.dims.len()).then_some(AnonymousAxisName::Of {
            node: input,
            axis: read,
        })
    };
    // Each anonymous axis from its structural source; an axis spelled with a
    // required number is that literal claim.
    let by_source = |literal_sources: bool| {
        let sources = output_axis_sources(dag, id);
        dims.iter()
            .enumerate()
            .map(|(axis, dim)| match dim {
                DimInfo::Named(name, Some(required)) if is_anonymous(name) => {
                    AnonymousAxisName::Literal(*required)
                }
                DimInfo::Named(name, None) if is_anonymous(name) => {
                    let named = match sources.get(axis) {
                        Some(AxisSource::Literal { value }) if literal_sources => {
                            usize::try_from(*value).ok().map(AnonymousAxisName::Literal)
                        }
                        Some(AxisSource::InputAxis {
                            input,
                            axis: RtAxis::Lit(read),
                        }) => input_axis(*input, *read),
                        _ => None,
                    };
                    named.unwrap_or(AnonymousAxisName::Fresh)
                }
                _ => AnonymousAxisName::Own,
            })
            .collect()
    };
    let same_rank = |source: NodeId| {
        dag.get(source)
            .is_some_and(|source| source.output_type.dims.len() == dims.len())
            .then(|| {
                (0..dims.len())
                    .map(|axis| AnonymousAxisName::Of { node: source, axis })
                    .collect::<Vec<_>>()
            })
    };
    if let RiscOp::Expand {
        axis: expanded_axis,
        ..
    } = &node.op
        && !matches!(
            dims.get(*expanded_axis),
            Some(DimInfo::Named(name, _)) if !is_anonymous(name)
        )
    {
        // [05-MOV-1], #1619: the replaced/inserted axis reads the size
        // carrier; kept axes read their own operand positions. Rank equality
        // does not prove pass-through. Numeric result claims remain
        // independent of their sources. An explicit name ON THE EXPANDED AXIS
        // stays on the existing path: preserving one without its unread
        // signature witness can newly execute an unchecked wrong shape, and
        // B2b-1 owns that scoped claim-transport repair. chelis#1822: a real
        // name on a BYSTANDER axis is not that case. Testing every axis sent
        // an `expand` whose kept axis carries a signature binder to the
        // pass-through rule below, which copies the operand's PRE-EXPAND
        // extent onto the expanded axis, so `spec/05` section 2.4's
        // replacement was undone: the consumer then failed ownership
        // verification, or with no consumer the wrong type reached codegen
        // and the binary trapped while eval returned the right answer.
        return Some(by_source(true));
    }
    if matches!(node.op, RiscOp::Permute { .. }) {
        // A permutation preserves extents but not their positions. The
        // same-rank pass-through rule below copies the first input's
        // dimensions positionally, which silently undoes every non-identity
        // permutation as soon as one output axis is anonymous. Resolve each
        // anonymous output axis through the IR's structural axis-source
        // mapping; explicitly named axes already carry their destination
        // identity and stay untouched.
        return Some(by_source(false));
    }
    if let RiscOp::Gather { axis } = &node.op
        && let &[values, indices] = node.inputs.as_slice()
        && let (Some(values_node), Some(indices_node)) = (dag.get(values), dag.get(indices))
        && *axis < values_node.output_type.dims.len()
    {
        let of = |node: NodeId, range: std::ops::Range<usize>| {
            range.map(move |axis| AnonymousAxisName::Of { node, axis })
        };
        return Some(
            of(values, 0..*axis)
                .chain(of(indices, 0..indices_node.output_type.dims.len()))
                .chain(of(values, *axis + 1..values_node.output_type.dims.len()))
                .collect(),
        );
    }
    if !movement_alters_extents(&node.op)
        && let Some(names) = node.inputs.first().and_then(|input| same_rank(*input))
    {
        return Some(names);
    }
    if node.inputs.is_empty()
        && let Some(names) = node.shape_deps.first().and_then(|dep| same_rank(*dep))
    {
        // chelis#616: an input-less node (a `lower_if` mask Const) shaped
        // like a sibling records the relation as a shape-dep; tie its
        // anonymous dims to the sibling's instead of fragmenting them into a
        // sourceless fresh identity.
        return Some(names);
    }
    Some(dims.iter().map(fresh_or_own).collect())
}

/// The dimension the naming gives `node`'s axis `axis`, and the node axis
/// where that identity is decided: an axis that keeps its own dimension, a
/// static extent, or a fresh identity. A named node earlier in `dag` than
/// the one that reads it is read as named; a later one, as it is spelled.
pub(crate) fn anonymous_axis_identity(
    dag: &Dag,
    node: NodeId,
    axis: usize,
) -> Option<(DimInfo, NodeId, usize)> {
    let (mut node, mut axis) = (node, axis);
    for _ in 0..=dag.len() {
        let dim = dag.get(node)?.output_type.dims.get(axis)?.clone();
        let name = match anonymous_axis_names(dag, node) {
            Some(names) => *names.get(axis)?,
            None => AnonymousAxisName::Own,
        };
        match name {
            AnonymousAxisName::Of {
                node: source,
                axis: read,
            } if source.0 < node.0 => (node, axis) = (source, read),
            AnonymousAxisName::Of {
                node: source,
                axis: read,
            } => {
                let dim = dag.get(source)?.output_type.dims.get(read)?.clone();
                return Some((dim, source, read));
            }
            AnonymousAxisName::Own => return Some((dim, node, axis)),
            AnonymousAxisName::Literal(extent) => return Some((DimInfo::Lit(extent), node, axis)),
            AnonymousAxisName::Fresh => {
                return Some((fresh_anonymous_dim(node, axis, &dim), node, axis));
            }
        }
    }
    None
}
