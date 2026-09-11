//! One checked extent source per realized output axis (chelis#1277 Slice B).
//!
//! `spec/design/runtime_extents.md` C4 replaces the backward search
//! [`crate::dag::shape_source_for_axis`] performs - walk the operands of an
//! output axis until a `Load` carrying a matching dimension NAME turns up -
//! with a forward, exhaustive, per-output-axis derivation plus a cardinality
//! check. The difference matters because an exhaustive match over `RiscOp`
//! catches a new operation but not a missing flow through an existing one:
//! chelis#665 is the proof, since the operation is already `Expand` and the
//! missing fact is that an op-declared axis on its input must flow through a
//! kept output axis with an index shift.
//!
//! Three properties follow from deriving forward instead of searching
//! backward, and each is what a later Slice B commit consumes:
//!
//! - **Names group, sources locate.** `ExternalAxis` names the exact
//!   declaring `Load` by [`NodeId`]. Two `Load`s that share a dimension name
//!   still name themselves, so a cross-tensor read can no longer bind to
//!   whichever `Load` the search happened to reach first.
//! - **Every axis is accounted for.** [`output_axis_sources`] yields exactly
//!   one entry per output axis. [`check_axis_sources`] runs on every
//!   production path - eval, the C, HIP and Metal codegen entries, and
//!   `verify` - so a missing flow is a typed chelis#730 receipt rather than
//!   the occurrence pass's internal-compiler-error panic, and never an
//!   input extent substituted for the real one.
//! - **Identity stays a typed proof.** Only the identity movement axes that
//!   `spec/04` section 4.7 names, `Stride` with a literal step of one and
//!   `Pad` with literal zero padding, pass an input axis through. Every
//!   symbolic `Shrink` axis and every other non-identity movement axis is
//!   `OpComputed` with a fresh extent, a full-axis `(Lit(0), ToEnd)` slice
//!   included, because equality of extent formulas never proves identity.
//!
//! Sources are derived, never stored: C4.5 requires them to be computed from
//! the DAG a lane actually consumes, after vmap, grad, specialization,
//! fusion, cloning, and DCE, so a stale [`NodeId`] cannot outlive a
//! mutation. Nothing here is serialized or carried across a transform.
//!
//! This module changes no acceptance decision on its own. The equality
//! classes, the guard placement, and the deletion of the provenance walk are
//! the second half of Slice B.

use chelis_types::types::Prim;
use chelis_types::unimplemented_rejection;
use chelis_types::unsupported::{Stage, Unsupported, UnsupportedKind};

use crate::dag::{Dag, DagNode, DimInfo, NodeId, RiscOp, RtAxis, RtDim};

/// The two distinct primitives represented by a verified `RiscOp::Expand`.
/// Spec/10 section 3.4 selects the form by the verified input/output rank relation;
/// mutable runtime metadata supplies no source-operation authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpansionKind {
    Expand,
    Insert,
}
impl ExpansionKind {
    pub fn primitive_name(self) -> &'static str {
        match self {
            Self::Expand => "expand",
            Self::Insert => "insert",
        }
    }
}
/// An invalid or non-expansion node has no form; verification rejects invalid ranks.
pub fn expansion_kind(dag: &Dag, id: NodeId) -> Option<ExpansionKind> {
    let node = dag.get(id)?;
    if !matches!(node.op, RiscOp::Expand { .. }) {
        return None;
    }
    let input = dag.get(*node.inputs.first()?)?;
    match node
        .output_type
        .dims
        .len()
        .checked_sub(input.output_type.dims.len())
    {
        Some(0) => Some(ExpansionKind::Expand),
        Some(1) => Some(ExpansionKind::Insert),
        _ => None,
    }
}

/// Where one realized output axis gets its extent.
///
/// The variant set is C4's, and it is deliberately closed: an extent is a
/// compile-time literal, an axis of an external `Load`, an axis of one of
/// this node's own tensor inputs, a rank-0 `int64` scalar input, or a value
/// the operation computes by its own output-shape rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxisSource {
    /// A compile-time-constant extent.
    Literal { value: i64 },
    /// An axis of an external `Load`, named by the exact declaring node.
    ExternalAxis { load: NodeId, axis: usize },
    /// An axis of the tensor in this node's absolute input slot `input`.
    InputAxis { input: usize, axis: RtAxis },
    /// A rank-0 exact-`int64` extent value in absolute input slot `input`.
    ScalarInput { input: usize },
    /// The operation computes this extent by its own output-shape rule, so
    /// it is a fresh extent rather than any input's runtime dimension.
    OpComputed { op: NodeId, axis: usize },
    /// The extent is supplied BY the claim this axis carries, rather than
    /// determined by the operation: a `Const`'s declared named dimension, or
    /// a chelis#616 sibling-shaped fill, is sized by whatever the class
    /// resolves to.
    ///
    /// This kind exists because the axis is neither a witness of its claim
    /// nor a fresh extent. Guarding it against the class's canonical member
    /// would compare a value with itself, and calling it `OpComputed` says
    /// the operation decided an extent it actually consumed - which is what
    /// [`declared_shape_sources`] meant by "as far as this derivation is
    /// concerned" before there was a kind for it.
    ClassSupplied { op: NodeId, axis: usize },
}

/// An anonymous output dimension is not a referenceable symbol: nothing
/// renders it and distinct runtime extents share the spelling, so it can
/// neither name a class nor be located from a sibling.
fn is_anonymous(name: &str) -> bool {
    name.is_empty() || name == "*"
}

/// An axis of the tensor in `slot`. Ranks are bounded by addressable memory,
/// so the `i32` conversion cannot fail in practice; a hypothetical failure
/// degrades to the operation's own rule rather than fabricating an axis.
fn pass_through(node: NodeId, slot: usize, axis: usize) -> AxisSource {
    match i32::try_from(axis) {
        Ok(axis) => AxisSource::InputAxis {
            input: slot,
            axis: RtAxis::Lit(axis),
        },
        Err(_) => AxisSource::OpComputed { op: node, axis },
    }
}

fn op_computed(node: NodeId, rank: usize) -> Vec<AxisSource> {
    (0..rank)
        .map(|axis| AxisSource::OpComputed { op: node, axis })
        .collect()
}

fn literal(value: usize) -> AxisSource {
    AxisSource::Literal {
        value: i64::try_from(value).unwrap_or(i64::MAX),
    }
}

/// The rank of the tensor in absolute input slot `slot`, or `None` when the
/// slot is absent.
fn input_rank(dag: &Dag, node: &DagNode, slot: usize) -> Option<usize> {
    node.inputs
        .get(slot)
        .and_then(|id| dag.get(*id))
        .map(|source| source.output_type.dims.len())
}

/// Degrade an operand-derived mapping that does not fit the declared output
/// rank to the operation's own rule.
///
/// The arms that read an operand's rank - the reductions, `Count`, the
/// windowed pair, `OneHot`, `Gather`, and the scatter family - can disagree
/// with the declared output rank only when the node is already malformed: a
/// reduced axis past the operand's rank, a window arity above it, an absent
/// operand. Each of those is an arity rule that `verify` and the evaluator
/// already reject with their own diagnostic, so reporting cardinality here
/// would replace a specific error with a vaguer one on a lane that used to
/// give the specific one.
///
/// The cardinality check keeps its teeth where the defect is genuinely a
/// missing flow: the operations whose own declared fields fix the source
/// count (`Reshape`, `Permute`, `Pad`, `Shrink`, `Stride`) and the
/// input-less nodes that declare a shape they cannot supply.
fn or_op_computed(sources: Vec<AxisSource>, node: NodeId, rank: usize) -> Vec<AxisSource> {
    if sources.len() == rank {
        sources
    } else {
        op_computed(node, rank)
    }
}

/// The source a typed [`RtDim`] carrier denotes.
///
/// `Sym` is the reshape-only carrier for a dimension declared elsewhere
/// (C1.7). The `AxisSource` set has no `Sym` variant because a stamped name
/// GROUPS members into an equality class (C2.4) and does not LOCATE the
/// value; the reshape target binds to its class's canonical value exactly as
/// it does today. From this operation's point of view the extent is
/// therefore its own, which is what `OpComputed` states.
fn rt_dim_source(node: NodeId, axis: usize, dim: &RtDim) -> AxisSource {
    match dim {
        RtDim::Lit(value) => literal(*value),
        RtDim::Node(slot) => AxisSource::ScalarInput { input: *slot },
        RtDim::InputAxis { tensor, axis } => AxisSource::InputAxis {
            input: *tensor,
            axis: *axis,
        },
        RtDim::Sym(_) | RtDim::ToEnd => AxisSource::OpComputed { op: node, axis },
    }
}

/// The shape-preserving map: every output axis reads the same axis of the
/// operand that carries the output's rank.
///
/// Chelis has no implicit broadcasting, so a positive-rank elementwise
/// operand has the output's shape. Searching for the operand whose rank
/// matches rather than assuming slot 0 covers the fused chain, whose first
/// external input may be a rank-0 scalar.
fn shape_preserving(dag: &Dag, node: &DagNode) -> Vec<AxisSource> {
    let rank = node.output_type.dims.len();
    let slot = node.inputs.iter().position(|id| {
        dag.get(*id)
            .is_some_and(|source| source.output_type.dims.len() == rank)
    });
    match slot {
        Some(slot) => (0..rank)
            .map(|axis| pass_through(node.id, slot, axis))
            .collect(),
        None => op_computed(node.id, rank),
    }
}

/// The sources of a node that declares its own shape with no operand to read
/// it from: `Const` (a uniform fill at positive rank) and `ConstTensor`.
///
/// A statically known extent is the literal it declares. A named dimension
/// is supplied by the equality class that name groups, which is the
/// operation's own extent as far as this derivation is concerned. An
/// ANONYMOUS extent is neither: no literal, no operand, no class, and no
/// shape dependency supplies it, so the axis has NO source. That is
/// chelis#1482 exactly - a `relu` over a runtime-bound `shrink` gives its
/// zero fill a wildcard extent, and the C emitter's anonymous-dimension
/// rename then mints an identifier no `Load` declares and panics.
fn declared_shape_sources(dag: &Dag, node: &DagNode) -> Vec<AxisSource> {
    // chelis#616: an input-less node shaped like a sibling records the
    // relation as a shape dependency, and the C emitter ties its wildcard
    // dimensions to that sibling. Such an axis does have a source.
    let sibling_shaped = node
        .shape_deps
        .first()
        .and_then(|dep| dag.get(*dep))
        .is_some_and(|sibling| sibling.output_type.dims.len() == node.output_type.dims.len());
    node.output_type
        .dims
        .iter()
        .enumerate()
        .filter_map(|(axis, dim)| match dim {
            DimInfo::Lit(value) | DimInfo::Named(_, Some(value)) => Some(literal(*value)),
            DimInfo::Named(name, None) if !is_anonymous(name) || sibling_shaped => {
                Some(AxisSource::ClassSupplied { op: node.id, axis })
            }
            DimInfo::Named(_, None) => None,
        })
        .collect()
}

/// Every realized output axis of `node`, in output-axis order.
///
/// One exhaustive match over `RiscOp`, forward from each operation's own
/// output-shape rule. The result has one entry per output axis for a
/// well-formed node; [`check_axis_sources`] is what turns any other
/// cardinality into a typed receipt.
pub fn output_axis_sources(dag: &Dag, node: NodeId) -> Vec<AxisSource> {
    let Some(node) = dag.get(node) else {
        return Vec::new();
    };
    let id = node.id;
    let rank = node.output_type.dims.len();
    match &node.op {
        // --- Binary and unary elementwise: shape preserving ---
        RiscOp::Add
        | RiscOp::Sub
        | RiscOp::Mul
        | RiscOp::Div
        | RiscOp::FloorDiv
        | RiscOp::TruncDiv
        | RiscOp::Mod
        | RiscOp::CmpLt
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
        | RiscOp::UniformLike { .. }
        | RiscOp::Dropout { .. }
        | RiscOp::Copy
        | RiscOp::Drop
        | RiscOp::Realize
        | RiscOp::Cast { .. }
        | RiscOp::CastTrunc { .. }
        | RiscOp::FusedElem { .. }
        | RiscOp::Store { .. } => shape_preserving(dag, node),

        // --- Reductions: the reduced axis is removed, so output axis `a`
        // maps back to input axis `a` before it and `a + 1` at or after it.
        RiscOp::Sum {
            axis: reduced,
            accumulator: _,
        }
        | RiscOp::MaxReduce { axis: reduced }
        | RiscOp::MinReduce { axis: reduced }
        | RiscOp::ProdReduce { axis: reduced }
        | RiscOp::Argmax { axis: reduced }
        | RiscOp::Argmin { axis: reduced } => match input_rank(dag, node, 0) {
            Some(operand_rank) => or_op_computed(
                (0..operand_rank)
                    .filter(|axis| axis != reduced)
                    .map(|axis| pass_through(id, 0, axis))
                    .collect(),
                id,
                rank,
            ),
            None => op_computed(id, rank),
        },

        // `Count` removes every axis it names; the kept axes stay in order.
        RiscOp::Count { axes } => match input_rank(dag, node, 0) {
            Some(operand_rank) => or_op_computed(
                (0..operand_rank)
                    .filter(|axis| !axes.contains(axis))
                    .map(|axis| pass_through(id, 0, axis))
                    .collect(),
                id,
                rank,
            ),
            None => op_computed(id, rank),
        },

        // The leading axes pass through; each windowed axis has extent
        // `floor((input - window) / stride) + 1`, a value the operation
        // computes (`spec/05` section 2.3.1).
        //
        // The window arity above the operand's rank is a malformed node, and
        // it takes the fallback arm rather than a saturating fold. Clamping
        // `operand_rank - window_shape.len()` to zero would be a silent
        // saturation in the IR, which is chelis#888's class and which the
        // runtime-representation inventory records as deletion debt; the
        // explicit guard is behaviorally identical and adds no seam.
        RiscOp::ReduceWindow { window_shape, .. } => match input_rank(dag, node, 0) {
            Some(operand_rank) if window_shape.len() <= operand_rank => {
                let leading = operand_rank - window_shape.len();
                or_op_computed(
                    (0..operand_rank)
                        .map(|axis| {
                            if axis < leading {
                                pass_through(id, 0, axis)
                            } else {
                                AxisSource::OpComputed { op: id, axis }
                            }
                        })
                        .collect(),
                    id,
                    rank,
                )
            }
            _ => op_computed(id, rank),
        },

        // The adjoint's output is the forward INPUT's shape, which is its
        // own first operand.
        RiscOp::ReduceWindowGrad { .. } => match input_rank(dag, node, 0) {
            Some(operand_rank) => or_op_computed(
                (0..operand_rank)
                    .map(|axis| pass_through(id, 0, axis))
                    .collect(),
                id,
                rank,
            ),
            None => op_computed(id, rank),
        },

        // --- Movement ---
        // Each target is its own typed carrier (C1.7); nothing is read off
        // the operand's shape.
        RiscOp::Reshape { new_shape } => new_shape
            .iter()
            .enumerate()
            .map(|(axis, dim)| rt_dim_source(id, axis, dim))
            .collect(),

        // Output axis `a` is input axis `axes[a]`. The backward search
        // returned input axis `a` here, which is only correct for the
        // identity permutation.
        RiscOp::Permute { axes } => axes
            .iter()
            .map(|source_axis| pass_through(id, 0, *source_axis))
            .collect(),

        // C4.2: same-rank `Expand` replaces one axis and leaves the rest on
        // their own input axis; rank-increasing `Expand` inserts one, so
        // every later output axis reads input axis `output_axis - 1`.
        RiscOp::Expand {
            axis: expand_axis,
            size,
        } => {
            let inserts = input_rank(dag, node, 0).is_some_and(|operand| rank == operand + 1);
            (0..rank)
                .map(|axis| {
                    if axis == *expand_axis {
                        rt_dim_source(id, axis, size)
                    } else if inserts && axis > *expand_axis {
                        pass_through(id, 0, axis - 1)
                    } else {
                        pass_through(id, 0, axis)
                    }
                })
                .collect()
        }

        // `indices.dims ++ [vocab]`.
        RiscOp::OneHot { vocab } => match input_rank(dag, node, 0) {
            Some(operand_rank) => or_op_computed(
                (0..operand_rank)
                    .map(|axis| pass_through(id, 0, axis))
                    .chain(std::iter::once(literal(*vocab)))
                    .collect(),
                id,
                rank,
            ),
            None => op_computed(id, rank),
        },

        // Only literal zero padding is an identity on an axis.
        RiscOp::Pad { padding, .. } => padding
            .iter()
            .enumerate()
            .map(|(axis, (before, after))| {
                if before.as_lit() == Some(0) && after.as_lit() == Some(0) {
                    pass_through(id, 0, axis)
                } else {
                    AxisSource::OpComputed { op: id, axis }
                }
            })
            .collect(),

        // No `Shrink` axis passes an input axis through: `spec/04` section
        // 4.7 names only stride-one and zero-pad as identity movement, and
        // C1.2 requires every symbolic shrink axis to be fresh, so a
        // full-axis `(Lit(0), ToEnd)` slice mints a fresh extent too.
        RiscOp::Shrink { bounds } => bounds
            .iter()
            .enumerate()
            .map(|(axis, _)| AxisSource::OpComputed { op: id, axis })
            .collect(),

        // Only a literal step of one is an identity on an axis; every other
        // step computes `ceil(extent / step)`.
        RiscOp::Stride { strides } => strides
            .iter()
            .enumerate()
            .map(|(axis, step)| {
                if step.as_lit() == Some(1) {
                    pass_through(id, 0, axis)
                } else {
                    AxisSource::OpComputed { op: id, axis }
                }
            })
            .collect(),

        // --- Shape query: a rank-0 scalar has no output axis ---
        RiscOp::Shape { .. }
        | RiscOp::ExtentWitness { .. }
        | RiscOp::CheckedReshapeExtent { .. } => Vec::new(),
        RiscOp::CheckedUnitAxis { .. } => (0..rank).map(|axis| pass_through(id, 0, axis)).collect(),

        // --- Memory ---
        RiscOp::Const { .. } | RiscOp::ConstTensor { .. } => declared_shape_sources(dag, node),

        // A `Load` axis is external: the caller supplies it, and the exact
        // declaring node is this one.
        RiscOp::Load { .. } => (0..rank)
            .map(|axis| AxisSource::ExternalAxis { load: id, axis })
            .collect(),

        // --- Backend specialization ---
        // The output shape is `batch_dims ++ [m, n]`. Only `m` and `n` are
        // computed by the contraction: the batch axes are the operands' own
        // leading axes, because `specialize.rs`'s `matmul_dims` admits the
        // pattern only when BOTH operands have rank `batch_dims.len() + 2`
        // and their leading dims equal `batch_dims`. Calling them
        // `OpComputed` would say they are fresh extents no input supplies,
        // which is false and would cost them their equality-class guard.
        RiscOp::BlasMatmul { batch_dims, .. } => {
            let batch = batch_dims.len();
            let operand = [0usize, 1]
                .into_iter()
                .find(|slot| input_rank(dag, node, *slot) == Some(rank));
            match operand {
                Some(slot) if rank >= 2 && batch == rank - 2 => (0..batch)
                    .map(|axis| pass_through(id, slot, axis))
                    .chain((batch..rank).map(|axis| AxisSource::OpComputed { op: id, axis }))
                    .collect(),
                _ => op_computed(id, rank),
            }
        }

        // `values.dims[..axis] ++ indices.dims ++ values.dims[axis + 1..]`
        // (`spec/05` section 3.5): every output axis is an input axis of one
        // of the two operands.
        RiscOp::Gather { axis } => match (input_rank(dag, node, 0), input_rank(dag, node, 1)) {
            (Some(values_rank), Some(indices_rank)) if *axis < values_rank => or_op_computed(
                (0..*axis)
                    .map(|leading| pass_through(id, 0, leading))
                    .chain((0..indices_rank).map(|index| pass_through(id, 1, index)))
                    .chain((*axis + 1..values_rank).map(|trailing| pass_through(id, 0, trailing)))
                    .collect(),
                id,
                rank,
            ),
            _ => op_computed(id, rank),
        },

        // Every scatter family writes into a copy of its target, so the
        // output shape is the target's shape.
        RiscOp::ScatterAdd { .. } | RiscOp::Scatter { .. } | RiscOp::ScatterElements { .. } => {
            match input_rank(dag, node, 0) {
                Some(target_rank) => or_op_computed(
                    (0..target_rank)
                        .map(|axis| pass_through(id, 0, axis))
                        .collect(),
                    id,
                    rank,
                ),
                None => op_computed(id, rank),
            }
        }
    }
}

/// Every node's output-axis sources are complete and refer to real edges.
///
/// C4.1: the derivation must yield exactly the output rank with no omitted
/// or duplicated axis; `ExternalAxis` must name a real `Load`; `InputAxis`
/// must validate the tensor slot and its normalized `int32` axis;
/// `ScalarInput` must validate C2.1's rank-0 exact-`int64` contract. A
/// well-typed mapping this resolver cannot yet supply is the registered
/// chelis#730 typed receipt (C4.3), never the occurrence pass's panic and
/// never an input extent substituted for the missing one.
///
/// `stage` names the refusing lane so the rendered receipt says which
/// production path stopped, per the section C2 diagnostic shape.
pub fn check_axis_sources(dag: &Dag, stage: Stage) -> Result<(), Unsupported> {
    for node in dag.nodes() {
        let sources = output_axis_sources(dag, node.id);
        check_node_axis_sources(dag, node, &sources, stage)?;
    }
    Ok(())
}

fn receipt(what: String, node: &DagNode, stage: Stage, sourceless: bool) -> Unsupported {
    let authority = if sourceless {
        unimplemented_rejection!(
            1482,
            "give the axis a real extent source: size the consuming operation's output from the operand that declares the extent, or declare the extent on the producing node"
        )
    } else {
        unimplemented_rejection!(
            1277,
            "derive one checked source per output axis from the operation's own shape rule; see runtime_extents.md C4"
        )
    };
    Unsupported::new(
        UnsupportedKind::Construct(what),
        format!(
            "node {} output-axis extent sources (op {})",
            node.id.0,
            crate::grad::risc_op_name(&node.op)
        ),
        stage,
        authority,
    )
}

/// The per-node half of [`check_axis_sources`]. Crate-private, and taking
/// the source vector by reference rather than deriving it, so this module's
/// own unit tests can present a duplicated or misdirected vector that a
/// correct exhaustive match would never produce.
pub(crate) fn check_node_axis_sources(
    dag: &Dag,
    node: &DagNode,
    sources: &[AxisSource],
    stage: Stage,
) -> Result<(), Unsupported> {
    let rank = node.output_type.dims.len();
    if sources.len() != rank {
        let sourceless = sources.len() < rank
            && matches!(node.op, RiscOp::Const { .. } | RiscOp::ConstTensor { .. });
        return Err(receipt(
            format!(
                "{} extent source(s) for {rank} output axis(es)",
                sources.len()
            ),
            node,
            stage,
            sourceless,
        ));
    }
    for (axis, source) in sources.iter().enumerate() {
        match source {
            AxisSource::Literal { value } => {
                if *value < 0 {
                    return Err(receipt(
                        format!("negative literal extent {value} on output axis {axis}"),
                        node,
                        stage,
                        false,
                    ));
                }
            }
            AxisSource::ExternalAxis { load, axis: read } => {
                let declaring = dag
                    .get(*load)
                    .filter(|n| matches!(n.op, RiscOp::Load { .. }));
                let Some(declaring) = declaring else {
                    return Err(receipt(
                        format!(
                            "output axis {axis} names node {} as its declaring Load, which is not a Load",
                            load.0
                        ),
                        node,
                        stage,
                        false,
                    ));
                };
                if *read >= declaring.output_type.dims.len() {
                    return Err(receipt(
                        format!(
                            "output axis {axis} reads axis {read} of a rank {} Load",
                            declaring.output_type.dims.len()
                        ),
                        node,
                        stage,
                        false,
                    ));
                }
            }
            AxisSource::InputAxis {
                input,
                axis: RtAxis::Lit(read),
            } => {
                let Some(source_node) = node.inputs.get(*input).and_then(|id| dag.get(*id)) else {
                    return Err(receipt(
                        format!(
                            "output axis {axis} reads input slot {input}, but the node has {} input(s)",
                            node.inputs.len()
                        ),
                        node,
                        stage,
                        false,
                    ));
                };
                let Ok(read) = usize::try_from(*read) else {
                    return Err(receipt(
                        format!("output axis {axis} reads a non-normalized axis {read}"),
                        node,
                        stage,
                        false,
                    ));
                };
                if read >= source_node.output_type.dims.len() {
                    return Err(receipt(
                        format!(
                            "output axis {axis} reads axis {read} of a rank {} tensor in input slot {input}",
                            source_node.output_type.dims.len()
                        ),
                        node,
                        stage,
                        false,
                    ));
                }
            }
            AxisSource::ScalarInput { input } => {
                if *input == 0 {
                    return Err(receipt(
                        format!(
                            "output axis {axis} reads input slot 0, which is the tensor operand"
                        ),
                        node,
                        stage,
                        false,
                    ));
                }
                let Some(source_node) = node.inputs.get(*input).and_then(|id| dag.get(*id)) else {
                    return Err(receipt(
                        format!(
                            "output axis {axis} reads input slot {input}, but the node has {} input(s)",
                            node.inputs.len()
                        ),
                        node,
                        stage,
                        false,
                    ));
                };
                if !source_node.output_type.dims.is_empty() {
                    return Err(receipt(
                        format!(
                            "output axis {axis} reads a rank {} extent value in input slot {input}, which must be rank 0",
                            source_node.output_type.dims.len()
                        ),
                        node,
                        stage,
                        false,
                    ));
                }
                if source_node.output_type.precision != Prim::Int64 {
                    return Err(receipt(
                        format!(
                            "output axis {axis} reads a `{}` extent value in input slot {input}, which must be int64",
                            source_node.output_type.precision.name()
                        ),
                        node,
                        stage,
                        false,
                    ));
                }
            }
            // `ClassSupplied` carries the same self-reference contract as
            // `OpComputed`: both name this node and this axis, and differ
            // only in whether the operation decided the extent or consumed
            // one the claim supplies.
            AxisSource::OpComputed { op, axis: computed }
            | AxisSource::ClassSupplied { op, axis: computed } => {
                if *op != node.id || *computed != axis {
                    return Err(receipt(
                        format!(
                            "output axis {axis} is computed by node {} axis {computed} rather than by itself",
                            op.0
                        ),
                        node,
                        stage,
                        false,
                    ));
                }
            }
        }
    }
    Ok(())
}

/// The stamped extent claim a class groups by.
///
/// Grouping is by the CLAIM, "which is the output of the typed identity proof
/// (C1.2) and is what `symbolic_bindings` groups by today for names"
/// (`spec/design/runtime_extents.md` C2.4). A [`DimInfo`] is the wrong key
/// because `Named("n", Some(4))` and `Named("n", None)` are one claim and a
/// bare `Lit(4)` is a different one.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum DimClaim {
    /// A binder name, whether or not its extent is also statically known.
    Name(String),
    /// A literal extent. "A literal claim is the class's canonical value
    /// itself" (C2.4), so a literal class guards every member against the
    /// literal rather than against a first member.
    Literal(usize),
}

/// One output axis carrying a class's claim.
///
/// `source` is the axis's [`output_axis_sources`] entry, carried rather than
/// re-derived: the guard emitters need it to build the comparison expression,
/// and re-deriving per member at emission would let the value a guard compares
/// disagree with the source that grouping saw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassMember {
    pub node: NodeId,
    pub axis: usize,
    pub source: AxisSource,
}

/// Where C1.3 places a class's guards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardPlacement {
    /// Every operand is an interface value, so the guard runs at function
    /// entry in assigned-slot order, before any other operation of the
    /// function (`spec/04-type-system.md` section 4.7).
    Entry,
    /// At least one operand is locally computed, so the guard takes the
    /// source position of the operation that introduces the guarded extent.
    Local,
}

/// One derived runtime-dimension equality class.
///
/// Derived, never stored: computed from the DAG a lane consumes, after the
/// last rewrite, at the same point as [`output_axis_sources`] (C4.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeDimClass {
    pub claim: DimClaim,
    /// The output axes carrying the claim. For a [`DimClaim::Name`] the first
    /// member is canonical and every later member is one equality guard
    /// against it; for a [`DimClaim::Literal`] the literal is the canonical
    /// value and every member is one guard against it.
    pub members: Vec<ClassMember>,
}

/// The `Load` whose axis this operand names, and that axis, when the operand
/// is an input tensor's axis at all.
///
/// `spec/04-type-system.md` section 4.7 lists "an input tensor's axis" as an
/// interface value, and the DAG spells one two ways: a `Load`'s own output
/// axis, recorded as [`AxisSource::ExternalAxis`], and a folded
/// `shape(t, k)` read of that same tensor, recorded as
/// [`AxisSource::InputAxis`]. `spec/05` section 2.4.1 admits the second as an
/// extent read "directly from that tensor's shape metadata", so the two are
/// one category and every consumer asking "is this operand an input tensor's
/// axis" must accept both.
///
/// Three consumers ask it - guard placement, the interface bindings the C and
/// HIP prologues declare and guard from, and the C emitter's entry sites - and
/// they ask it HERE so they cannot drift. Answering it separately at each site
/// is what produced four instances of one defect: `placement` classified a
/// folded read as local, then `symbolic_bindings_interface` dropped the
/// declaration for a name whose only interface witness is a folded read,
/// which stopped the emitted C compiling.
///
/// "A `cast` takes the placement of the value it casts" (section 4.7, same
/// paragraph), so the walk looks through `Cast`/`CastTrunc` to the value cast:
/// an `int32` parameter reaching an extent through `cast(m, int64)` lands its
/// carrier at the Cast, not at the `Load`, and classifying by the immediate
/// producer would place one claim two ways depending on a width conversion.
/// The walk is bounded by the node count, so a malformed graph cannot spin.
///
/// `None` means the operand is not an input tensor's axis: a computed
/// producer, an extent an operation computes, or a literal.
pub fn member_load_axis(dag: &Dag, member: &ClassMember) -> Option<(NodeId, usize)> {
    match member.source {
        AxisSource::ExternalAxis { load, axis } => {
            matches!(dag.get(load)?.op, RiscOp::Load { .. }).then_some((load, axis))
        }
        AxisSource::InputAxis { input, axis } => {
            let RtAxis::Lit(read_axis) = axis;
            let load = load_through_casts(dag, member.node, input)?;
            Some((load, read_axis as usize))
        }
        _ => None,
    }
}

/// The `Load` reachable from `node`'s input `slot` through zero or more
/// width conversions, if any. Shared by [`member_load_axis`] and by
/// `RuntimeDimClass::placement`'s `ScalarInput` arm, which asks the same
/// question of a scalar rather than of an axis.
pub(crate) fn load_through_casts(dag: &Dag, node: NodeId, slot: usize) -> Option<NodeId> {
    let mut current = *dag.get(node)?.inputs.get(slot)?;
    for _ in 0..dag.nodes().len() {
        let producer = dag.get(current)?;
        match producer.op {
            RiscOp::Load { .. } => return Some(current),
            RiscOp::Cast { .. } | RiscOp::CastTrunc { .. } => {
                current = *producer.inputs.first()?;
            }
            _ => return None,
        }
    }
    None
}

impl RuntimeDimClass {
    /// C1.3's placement for this class.
    ///
    /// `spec/04-type-system.md` section 4.7: "A guard whose operands are all
    /// interface values (an input tensor's axis, a scalar parameter, or a
    /// literal) is evaluated at function entry". Every other class compares
    /// at least one locally computed value and takes the source position of
    /// the operation that introduces the guarded extent.
    pub fn placement(&self, dag: &Dag) -> GuardPlacement {
        // Section 4.7 keys on the guard's OPERANDS, not on whether the axis
        // belongs to a `Load`: "a guard whose operands are all interface
        // values (an input tensor's axis, a scalar parameter, or a literal)
        // is evaluated at function entry", against "a guard that compares a
        // locally computed value (checked integer arithmetic, a
        // user-function result, or an extent an operation computes)".
        //
        // So a folded `shape(y, k)` read is an INTERFACE value: `spec/05`
        // section 2.4.1 admits `InputAxis` as an `expand` extent read
        // "directly from that tensor's shape metadata", so the quantity the
        // guard compares is an input tensor's axis, exactly the first item in
        // section 4.7's list. Treating only `ExternalAxis` as interface
        // confused "is this axis a `Load`'s own" with "is this operand an
        // input's axis", and made every folded cross-tensor read a local
        // guard.
        // `ScalarInput` straddles section 4.7's line: it records only WHICH
        // SLOT holds the extent, and that slot is either a scalar PARAMETER
        // (interface) or the arithmetic that produced one (locally computed).
        // The DAG keeps the difference - the slot names a node and that node
        // has an op - so this is decided by reading it, not by adding a
        // second representation to carry it.
        // A slot holds an interface value only when its producer is a `Load`.
        // Section 4.7's list says "an input TENSOR's axis" and "a scalar
        // PARAMETER": a folded read of a COMPUTED tensor's axis, like a
        // computed scalar, does not exist until its producer runs, so its
        // guard cannot be evaluated at entry "before any other operation of
        // the function".
        // "A `cast` takes the placement of the value it casts" (section 4.7,
        // the same paragraph as the interface list). A cast is how a scalar
        // parameter of the wrong width reaches an extent - an `int32`
        // parameter `m` in `reshape(x, [cast(m, int64)])` lands its
        // `RtDim::Node` at the Cast, not at the `Load` - so classifying by
        // the slot's IMMEDIATE producer would place the same claim two
        // different ways depending on a width conversion. Look through the
        // cast to the value cast: a cast of a parameter is interface, a cast
        // of arithmetic is local.
        //
        // The walk is bounded by the node count, so a malformed graph cannot
        // spin here.
        if self
            .members
            .iter()
            .all(|member| member_is_interface(dag, member))
        {
            GuardPlacement::Entry
        } else {
            GuardPlacement::Local
        }
    }
}

/// Whether the quantity a guard reads at `member` is a section 4.7 INTERFACE
/// value.
///
/// Extracted from [`RuntimeDimClass::placement`] rather than copied, because
/// two derivations now ask it: the equality classes above and the unit-extent
/// claims below. The reasoning is the block comment in `placement`, and it
/// stays there; splitting it in two is how the four instances of one defect
/// that `member_load_axis` records came about.
fn member_is_interface(dag: &Dag, member: &ClassMember) -> bool {
    match member.source {
        AxisSource::Literal { .. } => true,
        AxisSource::ExternalAxis { .. } | AxisSource::InputAxis { .. } => {
            member_load_axis(dag, member).is_some()
        }
        // `ScalarInput` asks the same question of a scalar rather than of an
        // axis, so it shares the walk but not the axis it resolves to.
        AxisSource::ScalarInput { input } => load_through_casts(dag, member.node, input).is_some(),
        AxisSource::OpComputed { .. } | AxisSource::ClassSupplied { .. } => false,
    }
}

/// The ABI input slot of each `Load` name, assigned by first occurrence in
/// node order.
///
/// This is the key `spec/04-type-system.md` section 4.7 names: "an entry that
/// declares no signature orders those guards by its ABI input-slot order
/// instead", closing with "Whatever rule assigns the slots, the guard order
/// follows the assigned slots, and never a separate traversal by binding
/// name, hash iteration, or node identity."
///
/// Both lowering paths land on this one key. `lower_fn` registers parameter
/// `Load`s in declared order, so first occurrence IS declared signature
/// order there. Helpers extracted from a declared function receive that same
/// parameter order. Only signatureless subexpressions assign name-sorted
/// slots, so the ABI branch governs those entries. This is also what the C
/// emitter's `input_labels` assigns, so a guard's order here and its slot there cannot
/// disagree.
fn abi_input_slot(dag: &Dag, load: NodeId) -> Option<usize> {
    let name = match &dag.get(load)?.op {
        RiscOp::Load { name } => name.as_str(),
        _ => return None,
    };
    let mut slot = 0usize;
    let mut seen: Vec<&str> = Vec::new();
    for node in dag.nodes() {
        if let RiscOp::Load { name: other } = &node.op {
            if other.as_str() == name {
                return Some(slot);
            }
            if !seen.contains(&other.as_str()) {
                seen.push(other.as_str());
                slot += 1;
            }
        }
    }
    None
}

/// The claim stamped on one output axis, or `None` when the axis carries no
/// referenceable claim.
///
/// An ANONYMOUS dimension is not a claim: nothing renders it, distinct
/// runtime extents share the spelling, and grouping by it would identify
/// unrelated axes, which is the string-matching defect this module removes.
fn axis_claim(dim: &DimInfo) -> Option<DimClaim> {
    match dim {
        DimInfo::Lit(value) => Some(DimClaim::Literal(*value)),
        DimInfo::Named(name, _) if !is_anonymous(name) => Some(DimClaim::Name(name.clone())),
        DimInfo::Named(_, _) => None,
    }
}

/// Whether the operation SETS this output axis rather than forwarding an
/// input axis through it.
///
/// C2.4: "Only an output axis that C4.2 maps to an unchanged input axis is
/// pass-through and not a member; the axis an operation sets or inserts is a
/// member whatever slot its `InputAxis` names."
///
/// A set axis and a pass-through axis can both carry `AxisSource::InputAxis`,
/// because a folded `shape()` read is exactly "this axis's extent is that
/// tensor's axis". They are distinguished here rather than by inspecting the
/// variant, and the match is deliberately small: `Expand` and `Reshape` are
/// the only owners whose `RtDim` may be `InputAxis` (C1.7's owner matrix), so
/// every other `InputAxis` source is a forwarded axis.
fn sets_axis(op: &RiscOp, axis: usize) -> bool {
    match op {
        RiscOp::Expand { axis: set, .. } => axis == *set,
        // A `Reshape` target mints a fresh extent only when it computes one.
        // C4.2 lists the four target carriers, and C2.4 says a reshape-only
        // `Sym` target "keeps binding to its class's canonical value exactly
        // as it does today" - it RESTATES a symbol declared elsewhere rather
        // than declaring one, so it is no more a witness than a passed-through
        // axis. A `Lit` target is static and equally not a witness.
        RiscOp::Reshape { new_shape } => matches!(
            new_shape.get(axis),
            Some(RtDim::Node(_) | RtDim::InputAxis { .. })
        ),
        _ => false,
    }
}

/// The `RtDim` carrier an operation computes this output axis's extent from,
/// when the axis is one [`sets_axis`] admits as a witness.
///
/// This is the same C1.7 owner matrix `sets_axis` reads, returning the carrier
/// rather than a bit. It is PRIVATE and has exactly one caller, the class loop
/// in [`local_dim_guard_sites`], which stores what it returns in the site's
/// [`LocalGuardObservation`]. It was briefly public, so the DAG evaluator could
/// ask an operation which carrier held its guarded extent; that made the
/// evaluator answer a question the derivation had already answered, and it got
/// a unit-extent site wrong, because that site's node is the operand and its
/// operation carries no such axis. The derivation states the answer now, which
/// is what C2.7 asks for.
///
/// A `Sym` or `Lit` carrier is deliberately absent: neither computes an
/// extent, and `sets_axis` does not make either a witness on a `Reshape`.
fn expand_or_reshape_carrier(op: &RiscOp, axis: usize) -> Option<&RtDim> {
    let carrier = match op {
        RiscOp::Expand { axis: set, size } if axis == *set => size,
        RiscOp::Reshape { new_shape } => new_shape.get(axis)?,
        _ => return None,
    };
    matches!(carrier, RtDim::Node(_) | RtDim::InputAxis { .. }).then_some(carrier)
}

/// Whether this axis is a member of its claim's class.
///
/// Three rules, each from a normative sentence:
///
/// - A pass-through axis is not a member (C2.4, above). Its extent IS the
///   input's, so there is nothing to compare.
/// - An axis whose source is the literal its claim states is statically
///   proved, and section 4.7.2 conditions the guard on a claim "that is not
///   statically proven equal to `size`". No claim survives, so no member.
/// - Under a LITERAL claim an external `Load` axis is not a member. A
///   declared literal input extent is validated against the caller at the C
///   ABI boundary by the input shape preamble, which is a different
///   obligation from an extent class and covers programs containing no
///   runtime extent at all. Treating it as a member would mint a class for
///   every literal-shaped input.
fn is_member(op: &RiscOp, axis: usize, claim: &DimClaim, source: &AxisSource) -> bool {
    match source {
        AxisSource::InputAxis { .. } if !sets_axis(op, axis) => false,
        AxisSource::Literal { value } => !matches!(
            claim,
            DimClaim::Literal(claimed) if i64::try_from(*claimed) == Ok(*value)
        ),
        AxisSource::ExternalAxis { .. } => matches!(claim, DimClaim::Name(_)),
        // Sized by the claim, so there is nothing to compare it against.
        AxisSource::ClassSupplied { .. } => false,
        _ => true,
    }
}

/// Every stamped claim witness in `dag`, before the guard rule filters it.
///
/// [`derive_runtime_dim_classes`] is this list filtered to the claims that owe
/// a guard. Consumers that need the DECLARATION rather than the guard - the C
/// prologue's `int64_t n = inputs[s]->shape[a];`, `symbolic_params`, and
/// `bind_symbolic_dims`' exemption - need the unfiltered list, because a claim
/// with one witness still has to be declared even though it has nothing to
/// disagree with.
///
/// Unlike [`derive_runtime_dim_classes`] this keeps a name whose extent is
/// already statically bound, since the legacy occurrence pass distinguishes
/// `Named(n, None)` from `Named(n, Some(k))` and the C prologue declares only
/// the former.
/// Which scope each node belongs to, as one bit-set per node.
///
/// C2.4: a claim's identity is the name TOGETHER WITH the scope that
/// introduced it. A binder is scoped to the signature that declares it, and
/// the DAG carries that scope only sometimes.
///
/// A kernel lowered from ONE signature has exactly one scope, and every
/// interface witness belongs to it regardless of data use, because there is
/// nothing else it could belong to. `spec/04-type-system.md` section 4.7's
/// guard checks that a declared extent agrees with the value observed, and a
/// signature declaring `f(x: tensor[n, f32], p: tensor[n, f32])` is violated
/// by a caller whose `p` disagrees whether or not the body reads `p`.
///
/// A MERGED kernel - the `__global__` top-level kind, where independent
/// top-level results are lowered into one function - does not carry that
/// scope: two signatures' binders coexist in it, and grouping by name alone
/// identifies extents from different signatures. Root reachability
/// approximates the scope there. It is exact across independent results,
/// approximate under inlining, since a callee inlined into one result brings
/// its binders with it, and blind to an interface witness no result reaches,
/// which forms no class; that last case is a recorded residual.
///
/// **The two are NOT told apart, and that is the shipped limit.** Measured:
/// the merged kernel `named_axis__global__tensor_2` reports one explicit
/// root, eight `Load`s and twelve nodes, so by result count it is identical
/// to a single-signature kernel. Every discriminator the graph offers puts it
/// on the single-signature side, and the mechanism that does separate its two
/// `seq` claims - reachability leaving an unreached witness with an empty
/// scope - is the same mechanism that drops an interface witness no operation
/// reads. They are one rule seen from two sides, so this derivation cannot
/// honour section 4.7's "regardless of data use" for an unread witness and
/// separate two signatures at the same time.
///
/// Reachability is therefore applied everywhere. `Name` claims group among
/// root-reachable witnesses only, and a claim whose witness no result reaches
/// forms no class. That gap is a recorded residual, owned by B2b, whose fix
/// is scope carried on the dimension; the alternative traps correct programs,
/// which is the one thing this slice must not ship.
fn root_reach(dag: &Dag) -> Vec<u128> {
    // A DAG that declares no results carries no result-scoping information,
    // so it is ONE scope - the behavior every caller had before scoping
    // existed. Widening a class rather than splitting it is the conservative
    // direction: the derivation can then only guard more, never silently
    // guard less.
    //
    // Measured, and it is why there is no sink-based fallback here: every
    // kernel the C emitter lowers carries an explicit root, including the
    // merged `__global__` one this scoping exists for
    // (`named_axis__global__tensor_2`: one root, eight `Load`s, twelve
    // nodes), and the eval lane passes its roots explicitly. Inventing
    // "results" from sinks for a graph that declares none only reaches
    // hand-built fixtures, where it split classes the fixture meant as one
    // signature.
    let results = dag.roots();
    if results.is_empty() || results.len() > 128 {
        return vec![u128::MAX; dag.len()];
    }
    let mut reach = vec![0u128; dag.len()];
    for (index, result) in results.iter().enumerate() {
        let bit = 1u128 << index;
        let mut stack = vec![*result];
        while let Some(id) = stack.pop() {
            let Some(slot) = reach.get_mut(id.0) else {
                continue;
            };
            if *slot & bit != 0 {
                continue;
            }
            *slot |= bit;
            if let Some(node) = dag.get(id) {
                stack.extend(node.inputs.iter().copied());
                stack.extend(node.shape_deps.iter().copied());
            }
        }
    }
    reach
}

/// Split each claim's members by SCOPE, so a name spelled by two signatures
/// becomes two claims rather than one class.
///
/// Two members belong to one scope when their root reach overlaps, and the
/// merge is transitive so an A-B-C chain stays one scope. This is the ONE
/// implementation of C2.4's scoping: both `derive_dim_witnesses`, which the C
/// and HIP prologues read, and `derive_runtime_dim_classes`, which the entry
/// and local guard sites and the eval lane read, call it. Round 2 found the
/// scoping applied to the first and not the second, which is two derivations
/// that can disagree - the defect class this slice exists to remove, in the
/// slice's own code.
///
/// Measured on `rank_poly_tier3::named_axis_eval_parity_corners`, where
/// `total(x: &tensor[seq, f32])` and `use2(x: &tensor[batch, seq, f32])` are
/// merged into one global kernel: grouping by name alone identified a
/// 3-element axis with a 2-element one and made a correct program trap.
fn split_by_scope(
    dag: &Dag,
    grouped: Vec<(DimClaim, Vec<OrderedMember>)>,
) -> Vec<(DimClaim, Vec<OrderedMember>)> {
    let reach = root_reach(dag);
    grouped
        .into_iter()
        .flat_map(|(claim, members)| {
            let mut buckets: Vec<(u128, Vec<OrderedMember>)> = Vec::new();
            for entry in members {
                let mask = reach.get(entry.node).copied().unwrap_or(u128::MAX);
                let mut merged: Vec<OrderedMember> = vec![entry];
                let mut merged_mask = mask;
                buckets.retain_mut(|(bucket_mask, bucket)| {
                    if *bucket_mask & merged_mask != 0 {
                        merged_mask |= *bucket_mask;
                        merged.append(bucket);
                        false
                    } else {
                        true
                    }
                });
                buckets.push((merged_mask, merged));
            }
            buckets
                .into_iter()
                .map(move |(_, members)| (claim.clone(), members))
        })
        .collect()
}

pub fn derive_dim_witnesses(dag: &Dag) -> Vec<RuntimeDimClass> {
    let mut grouped: Vec<(DimClaim, Vec<OrderedMember>)> = Vec::new();
    for node in dag.nodes() {
        let sources = output_axis_sources(dag, node.id);
        for (axis, dim) in node.output_type.dims.iter().enumerate() {
            let DimInfo::Named(name, None) = dim else {
                continue;
            };
            if is_anonymous(name) {
                continue;
            }
            let Some(source) = sources.get(axis) else {
                continue;
            };
            // A pass-through axis is not a witness: its extent IS the input's,
            // so it neither declares nor disagrees. Same rule as `is_member`;
            // the two derivations differ only in the guard filter and in
            // whether a statically bound name counts.
            if matches!(source, AxisSource::InputAxis { .. }) && !sets_axis(&node.op, axis) {
                continue;
            }
            // An axis sized BY the claim consumes it rather than witnessing
            // it, so it is neither a declaration site nor a guard site. The
            // rule is on the SOURCE kind, so "what is a witness" is decided in
            // one place rather than by a second list of operations.
            if matches!(source, AxisSource::ClassSupplied { .. }) {
                continue;
            }
            let claim = DimClaim::Name(name.clone());
            let entry = OrderedMember::new(
                dag,
                ClassMember {
                    node: node.id,
                    axis,
                    source: source.clone(),
                },
            );
            match grouped.iter_mut().find(|(existing, _)| *existing == claim) {
                Some((_, members)) => members.push(entry),
                None => grouped.push((claim, vec![entry])),
            }
        }
    }
    let scoped = split_by_scope(dag, grouped);
    let mut out: Vec<(OrderKey, RuntimeDimClass)> = scoped
        .into_iter()
        .map(|(claim, mut members)| {
            members.sort_by_key(OrderedMember::key);
            let order = members[0].key();
            (
                order,
                RuntimeDimClass {
                    claim,
                    members: members.into_iter().map(|entry| entry.member).collect(),
                },
            )
        })
        .collect();
    out.sort_by_key(|(order, _)| *order);
    out.into_iter().map(|(_, class)| class).collect()
}

/// The equality classes of `dag`, in guard-evaluation order.
///
/// Derived, never stored (C4.5): computed from the DAG a lane actually
/// consumes, after the last rewrite, at the same point as
/// [`output_axis_sources`], so a stale [`NodeId`] cannot outlive a mutation.
///
/// A `Name` class needs at least two members, because one witness has nothing
/// to disagree with. A `Literal` class needs only one: C2.4 makes the literal
/// the canonical VALUE rather than a first member, so a single runtime-sourced
/// axis claiming a literal already owes a guard.
/// The sort key C2.4 rule 1 defines: interface members before local ones,
/// interface members by assigned ABI input slot, every remaining tie by node
/// position.
type OrderKey = (bool, Option<(usize, usize)>, usize);

/// A member together with the key that orders it.
///
/// `slot` is the declaring `Load`'s assigned ABI input slot and axis for an interface
/// member and `None` for a local one, so sorting on `(slot.is_none(), slot,
/// node)` puts every interface member ahead of every local one, orders the
/// interface group by assigned slot, and breaks every remaining tie by node
/// position. That is C2.4 rule 1 in one key.
struct OrderedMember {
    slot: Option<(usize, usize)>,
    node: usize,
    member: ClassMember,
}

impl OrderedMember {
    fn new(dag: &Dag, member: ClassMember) -> Self {
        // A named class takes its canonical value from an axis declaring
        // that name, not from a foreign value that merely claims equality.
        let slot = match member.source {
            AxisSource::ExternalAxis { load, axis } => {
                abi_input_slot(dag, load).map(|slot| (slot, axis))
            }
            _ => None,
        };
        Self {
            slot,
            node: member.node.0,
            member,
        }
    }

    fn literal_guard_key(&self, dag: &Dag) -> OrderKey {
        // A literal is its own canonical value. Its guard order therefore
        // follows the input being checked, even when a later operation
        // consumes a folded shape read or scalar parameter from that input.
        let input_axis = member_load_axis(dag, &self.member).or_else(|| {
            if let AxisSource::ScalarInput { input } = self.member.source {
                load_through_casts(dag, self.member.node, input).map(|load| (load, 0))
            } else {
                None
            }
        });
        let slot =
            input_axis.and_then(|(load, axis)| abi_input_slot(dag, load).map(|slot| (slot, axis)));
        (slot.is_none(), slot, self.node)
    }

    fn key(&self) -> OrderKey {
        (self.slot.is_none(), self.slot, self.node)
    }
}

pub fn derive_runtime_dim_classes(dag: &Dag) -> Vec<RuntimeDimClass> {
    let mut grouped: Vec<(DimClaim, Vec<OrderedMember>)> = Vec::new();

    for node in dag.nodes() {
        let sources = output_axis_sources(dag, node.id);
        for (axis, dim) in node.output_type.dims.iter().enumerate() {
            let Some(claim) = axis_claim(dim) else {
                continue;
            };
            // A cardinality failure is `check_axis_sources`' typed receipt to
            // report (C4.1), not this derivation's to guess at.
            let Some(source) = sources.get(axis) else {
                continue;
            };
            if !is_member(&node.op, axis, &claim, source) {
                continue;
            }
            let entry = OrderedMember::new(
                dag,
                ClassMember {
                    node: node.id,
                    axis,
                    source: source.clone(),
                },
            );
            match grouped.iter_mut().find(|(existing, _)| *existing == claim) {
                Some((_, members)) => members.push(entry),
                None => grouped.push((claim, vec![entry])),
            }
        }
    }

    let mut classes: Vec<(OrderKey, RuntimeDimClass)> = Vec::new();
    for (claim, mut members) in split_by_scope(dag, grouped) {
        let order_key = |member: &OrderedMember| match claim {
            DimClaim::Literal(_) => member.literal_guard_key(dag),
            DimClaim::Name(_) => member.key(),
        };
        members.sort_by_key(order_key);
        // A `Name` class needs two witnesses: one has nothing to disagree
        // with. A `Literal` class needs one, because C2.4 makes the literal
        // the canonical VALUE rather than a first member, so a single
        // runtime-sourced axis claiming a literal already owes a guard.
        let needed = match claim {
            DimClaim::Name(_) => 2,
            DimClaim::Literal(_) => 1,
        };
        if members.len() < needed {
            continue;
        }
        let order = order_key(&members[0]);
        classes.push((
            order,
            RuntimeDimClass {
                claim,
                members: members.into_iter().map(|entry| entry.member).collect(),
            },
        ));
    }
    // Class order preserves canonical witness identity. Individual entry
    // comparisons are scheduled separately by `entry_extent_guards`.
    classes.sort_by_key(|(order, _)| *order);
    classes.into_iter().map(|(_, class)| class).collect()
}

/// One `expand` node's claim that its operand's extent at `axis` is 1.
///
/// `spec/05-risc-primitives.md` section 2.4.1: the same-rank form "is well
/// formed only when the operand's extent at `axis` is 1 [...] the operation is
/// a claim that the operand's extent at `axis` is 1. A literal operand extent
/// at `axis` other than 1 is a type error. A symbolic or runtime operand
/// extent at `axis` other than 1 fails that claim's runtime extent guard and
/// traps `Domain`."
///
/// This is a PRECONDITION on an operand, not an identity between output axes,
/// which is why it is derived beside [`derive_runtime_dim_classes`] instead of
/// inside it. Forcing it in would go wrong two ways. The claim would be filed
/// under whatever the operand's own output dim states, `Name("n")` for a
/// symbolic operand, which is a different claim about a different quantity.
/// And an operand that IS a `Load` axis is excluded from a literal claim by
/// `is_member` on purpose: "Under a LITERAL claim an external `Load` axis is
/// not a member. A declared literal input extent is validated against the
/// caller at the C ABI boundary by the input shape preamble, which is a
/// different obligation from an extent class and covers programs containing no
/// runtime extent at all. Treating it as a member would mint a class for every
/// literal-shaped input." Relaxing that to admit this claim would put a guard
/// on every literal-shaped input in the repository.
///
/// Everything else is shared: the source comes from [`output_axis_sources`],
/// the placement from [`member_is_interface`] through [`Self::placement`], and
/// the rendering from the same [04-NUM-9] path the classes use. One more claim
/// KIND for the same three consumers, not a second answer to one question.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitExtentClaim {
    /// The `Expand` node making the claim.
    pub node: NodeId,
    /// Its operand, `node.inputs[0]`.
    pub operand: NodeId,
    /// The axis the operation broadcasts, in the operand's own numbering.
    pub axis: usize,
    /// The operand axis's own source, carried rather than re-derived for the
    /// reason [`ClassMember`] carries its own: re-deriving it at emission lets
    /// the value a guard compares disagree with the source the derivation saw.
    pub source: AxisSource,
}

impl UnitExtentClaim {
    /// The claim's guarded quantity, in the shape the shared placement and
    /// `Load`-resolution helpers take.
    ///
    /// The member names the OPERAND's axis, because that is the extent the
    /// guard reads. The claim's other side is the literal 1 and needs no
    /// member: there is nothing to read it from.
    pub fn member(&self) -> ClassMember {
        ClassMember {
            node: self.operand,
            axis: self.axis,
            source: self.source.clone(),
        }
    }

    /// C1.3's placement, by the same rule the equality classes take.
    pub fn placement(&self, dag: &Dag) -> GuardPlacement {
        if member_is_interface(dag, &self.member()) {
            GuardPlacement::Entry
        } else {
            GuardPlacement::Local
        }
    }

    /// The `<op>` slot of this claim's [04-NUM-9] line.
    ///
    /// `spec/04-type-system.md` section 4.7 fixes it by operand class: "for a
    /// guard whose operands are all interface values, the `load` primitive of
    /// the later witness in signature order", and otherwise "the source
    /// position of the operation that introduces the guarded extent", which
    /// for this claim is the `expand` that makes it.
    pub fn trap_op(&self, dag: &Dag) -> &'static str {
        match self.placement(dag) {
            GuardPlacement::Entry => "load",
            GuardPlacement::Local => "expand",
        }
    }
}

/// Every unit-extent claim in `dag` that still owes a runtime guard.
///
/// Derived, never stored, from the DAG a lane consumes after the last rewrite,
/// at the same point as [`output_axis_sources`] and
/// [`derive_runtime_dim_classes`] (C4.5).
///
/// Two exclusions, each from a normative sentence:
///
/// - Only the same-rank form makes this claim at all. The two forms of
///   `RiscOp::Expand` are told apart by the output rank against the operand's,
///   which is how `verify.rs`, `eval.rs`, `host.rs` and the C emitter already
///   tell them apart; a fifth spelling of that test is how they would drift.
/// - An operand extent of literal 1 satisfies the claim by construction, and
///   section 4.7.2 conditions a guard on a claim "not statically proven equal"
///   to the value. A literal other than 1 cannot reach here: the checker
///   refuses it, so this returns no claim for it rather than guarding it.
pub fn derive_unit_extent_claims(dag: &Dag) -> Vec<UnitExtentClaim> {
    let mut claims = Vec::new();
    for node in dag.nodes() {
        let RiscOp::Expand { axis, .. } = &node.op else {
            continue;
        };
        let Some(&operand_id) = node.inputs.first() else {
            continue;
        };
        let Some(operand) = dag.get(operand_id) else {
            continue;
        };
        if node.output_type.dims.len() != operand.output_type.dims.len() {
            continue;
        }
        let axis = *axis;
        match operand.output_type.dims.get(axis) {
            // Proved: the operand states the extent the claim asserts.
            Some(DimInfo::Lit(1)) => continue,
            Some(DimInfo::Lit(_)) | None => continue,
            Some(_) => {}
        }
        let Some(source) = output_axis_sources(dag, operand_id).get(axis).cloned() else {
            continue;
        };
        claims.push(UnitExtentClaim {
            node: node.id,
            operand: operand_id,
            axis,
            source,
        });
    }
    claims
}

/// One interface comparison, independent of the class that established its
/// witnesses. Both host lanes consume this schedule without regrouping it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntryExtentGuard {
    /// Compare two actual input axes from the same scoped claim.
    Named {
        claim: String,
        canonical: (NodeId, usize),
        observed: (NodeId, usize),
    },
    /// Compare an actual input axis with its literal claim or unit precondition.
    Literal {
        required: usize,
        observed: (NodeId, usize),
    },
}

/// The named witness claims an entry guard already checks, as
/// `(witness node, claim index)`.
///
/// chelis#1374 gave `ExtentWitness` named claims so a declared result's named
/// extent and a binder repeated across parameters are checked in every form:
/// an inlined root has no `Load` for [`entry_extent_guards`] to group, and a
/// declared-but-unread parameter has no class member until its witness is
/// retained. Where both witnesses DO read `Load`s, the class derivation
/// reaches the same pair and the entry schedule is the better owner: it runs
/// ahead of the evaluator's symbolic-dim binding, in assigned ABI-slot order.
/// `spec/04-type-system.md` §4.7 evaluates each guard "exactly once", so the
/// witness yields there and its claim is left carrying only the retention
/// that keeps the interface witness, and its ABI slot, alive.
pub fn entry_covered_witness_claims(dag: &Dag) -> Vec<(NodeId, usize)> {
    let observed_pair = |node: &crate::dag::DagNode| {
        let RiscOp::ExtentWitness {
            axis: RtAxis::Lit(axis),
            ..
        } = node.op
        else {
            return None;
        };
        let load = load_through_casts(dag, node.id, 0).or_else(|| node.inputs.first().copied())?;
        Some((load, usize::try_from(axis).ok()?))
    };
    let guards = entry_extent_guards(dag);
    let mut covered = Vec::new();
    for node in dag.nodes() {
        let RiscOp::ExtentWitness { claims, .. } = &node.op else {
            continue;
        };
        let Some(here) = observed_pair(node) else {
            continue;
        };
        for (index, (recorded, edge)) in claims.iter().zip(node.inputs.iter().skip(1)).enumerate() {
            let Some(there) = dag.get(*edge).and_then(observed_pair) else {
                continue;
            };
            if guards.iter().any(|guard| {
                matches!(guard,
                    EntryExtentGuard::Named { claim, canonical, observed }
                        if claim == &recorded.claim
                            && ((*canonical == here && *observed == there)
                                || (*canonical == there && *observed == here)))
            }) {
                covered.push((node.id, index));
            }
        }
    }
    covered
}

/// Section 4.7's individual entry checks in assigned input-slot/axis order.
/// A named check becomes due at the later of its two witnesses; its canonical
/// witness remains the declaring one even when that declaration is later.
/// Literal and unit checks become due at the observed input. Equal-position
/// checks retain derivation order. Exact duplicate comparisons are emitted once.
pub fn entry_extent_guards(dag: &Dag) -> Vec<EntryExtentGuard> {
    let position = |(load, axis)| {
        (
            abi_input_slot(dag, load).expect("entry witness is an input"),
            axis,
        )
    };
    let mut guards = Vec::new();
    // This is the same interface projection used by symbolic bindings: local
    // members do not prevent two input witnesses from disagreeing at entry.
    for class in derive_dim_witnesses(dag) {
        let DimClaim::Name(claim) = class.claim else {
            continue;
        };
        let mut reads = class
            .members
            .iter()
            .filter_map(|member| member_load_axis(dag, member));
        let Some(canonical) = reads.next() else {
            continue;
        };
        for observed in reads {
            if position(canonical) != position(observed) {
                guards.push(EntryExtentGuard::Named {
                    claim: claim.clone(),
                    canonical,
                    observed,
                });
            }
        }
    }
    for class in derive_runtime_dim_classes(dag) {
        if class.placement(dag) != GuardPlacement::Entry {
            continue;
        }
        let DimClaim::Literal(required) = class.claim else {
            continue;
        };
        for observed in class
            .members
            .iter()
            .filter_map(|member| member_load_axis(dag, member))
        {
            guards.push(EntryExtentGuard::Literal { required, observed });
        }
    }
    for claim in derive_unit_extent_claims(dag) {
        if claim.placement(dag) == GuardPlacement::Entry
            && let Some(observed) = member_load_axis(dag, &claim.member())
        {
            guards.push(EntryExtentGuard::Literal {
                required: 1,
                observed,
            });
        }
    }
    guards.sort_by_key(|guard| match guard {
        EntryExtentGuard::Named {
            canonical,
            observed,
            ..
        } => position(*canonical).max(position(*observed)),
        EntryExtentGuard::Literal { observed, .. } => position(*observed),
    });
    let mut unique = Vec::new();
    for guard in guards {
        if !unique.contains(&guard) {
            unique.push(guard);
        }
    }
    unique
}

/// A local guard's position: the node that introduces the extent, and the
/// output axis carrying the claim.
pub type LocalGuardSite = (usize, usize);

/// The value a local guard compares an observed extent against.
///
/// The derivation names the value; each lane resolves it through its own
/// declaration, which is why this is not a rendered string. C reads a
/// [`CanonicalExtent::Binder`] as the variable its prologue declared for that
/// claim; the DAG evaluator reads the same binder out of the bindings it
/// resolved before the first node ran.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalExtent {
    /// The claim's binder name.
    Binder(String),
    /// A size the checker already resolved for the claim. Keying the
    /// comparison on the resolved value rather than on whether a lane happens
    /// to declare a variable keeps the derivation the authority: a claim
    /// resolved to a literal over a RUNTIME read still owes the comparison
    /// `spec/04-type-system.md` section 4.7 requires between the claimed
    /// extent and the value actually observed, and the entry path already
    /// emits exactly that for a `Literal` claim (chelis#1377).
    Resolved(usize),
}

impl std::fmt::Display for CanonicalExtent {
    /// The canonical value as a lane reads it: the claim's binder name, or the
    /// resolved size. Both consumers render it through this one impl, so a
    /// guard's comparison and a diagnostic naming that comparison cannot
    /// describe the value two different ways.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CanonicalExtent::Binder(binder) => f.write_str(binder),
            CanonicalExtent::Resolved(value) => write!(f, "{value}"),
        }
    }
}

/// How a consumer reads the extent a local guard observes.
///
/// The derivation states it, because C2.7 puts one answer to one question in
/// one place. The two kinds of local claim observe different quantities: an
/// equality class compares the extent an operation is ABOUT TO produce, read
/// from the carrier it was given, and a unit-extent claim compares the extent
/// its operand ALREADY produced, read from that operand's realized shape. A
/// consumer that re-derives which of those to read from the site's own `op`
/// can only get one of them right, which is exactly the divergence C2.7
/// forbids.
///
/// The variants also fix WHEN each is readable, and that is not incidental.
/// [`Self::Carrier`] is readable before the site's node runs, which is where
/// `spec/04-type-system.md` section 4.7 puts a class guard: at "the source
/// position of the operation that introduces the guarded extent", so a wrong
/// claim is reported instead of the operation's own downstream failure.
/// [`Self::RealizedExtent`] is readable only after the site's node runs, which
/// is still "after its producers and before the first allocation or element
/// access whose shape depends on the guarded extent", because the site's node
/// IS the producer and the allocation belongs to its consumer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalGuardObservation {
    /// Evaluate this carrier against the site's node, before that node runs.
    Carrier(RtDim),
    /// Read the site node's realized output extent at the site's axis, after
    /// that node runs.
    RealizedExtent,
}

/// What a local guard reports, what it compares against, and how it reads the
/// value it compares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalGuardClaim {
    /// The claim as reported in the guard's context line.
    pub claim: String,
    /// The class's canonical value.
    pub canonical: CanonicalExtent,
    /// The operation [04-NUM-9]'s `<op>` slot names.
    pub op: &'static str,
    /// How to read the extent this guard observes.
    pub observed: LocalGuardObservation,
}

/// C1.3's local guard sites: `(node id, axis)` paired with the claim each
/// site guards against.
///
/// A `Local` class's guard "takes the source position of the operation that
/// introduces the guarded extent" (`spec/04-type-system.md` section 4.7), so
/// unlike the entry classes these are keyed by node.
///
/// Two lanes read this one function, which is what C2.7's single derivation
/// point means for a local guard: the C emitter places its guard at the
/// operation it names, and the DAG evaluator checks the same site when that
/// node produces its value. A second answer computed in either lane could
/// disagree with the first, and the point of deriving it here is that it
/// cannot.
pub fn local_dim_guard_sites(dag: &Dag) -> Vec<(LocalGuardSite, LocalGuardClaim)> {
    let mut sites = Vec::new();
    for class in derive_runtime_dim_classes(dag) {
        if class.placement(dag) != GuardPlacement::Local {
            continue;
        }
        // A numeric result annotation supplies the required value, never
        // evidence that the operation's independent carrier produces it.
        // Literal classes need no binder witness; resolved named classes keep
        // their existing class identity and compare against the required number.
        let (name, resolved) = match &class.claim {
            DimClaim::Literal(value) => (value.to_string(), Some(*value)),
            DimClaim::Name(name) => {
                let required = class.members.iter().find_map(|member| {
                    match dag.get(member.node)?.output_type.dims.get(member.axis)? {
                        DimInfo::Named(_, Some(value)) | DimInfo::Lit(value) => Some(*value),
                        DimInfo::Named(_, None) => None,
                    }
                });
                (name.clone(), required)
            }
        };
        for member in &class.members {
            // Only the independent extent source can discharge a claim.
            // Runtime carriers remain observable even when result metadata
            // contains a number. Literal-source proofs were handled by
            // is_member; unsupported observation kinds remain outside this
            // carrier consumer.
            if !matches!(
                member.source,
                AxisSource::InputAxis { .. } | AxisSource::ScalarInput { .. }
            ) {
                // C2.4's literal proof is about the axis SOURCE, not about the
                // claim: a member whose extent is a literal performs no runtime
                // read, so there is nothing to observe and nothing to compare.
                // A resolved claim over a runtime read is a different thing and
                // still owes its guard.
                //
                // The two admitted sources are the two C1.7 carriers an
                // operation can SET an axis from, which is what `sets_axis`
                // already says: a folded `shape(t, k)` read (`InputAxis`) and a
                // rank-0 computed scalar (`ScalarInput`). Section 4.7's local
                // sentence names "an extent an operation computes" among the
                // locally computed values, and a `Node` carrier is that extent
                // exactly - chelis#1375 is the case where the class formed with
                // both members and no site existed to compare them, so the
                // claim executed unguarded on every lane.
                //
                // External declarations retain their existing entry checks.
                // OpComputed sources need independent observations beyond
                // these carriers; B2b-0b still owns that separate extension.
                continue;
            }
            let Some(node) = dag.get(member.node) else {
                continue;
            };
            // The carrier this member's operation was given for the axis, which
            // is the value a consumer compares BEFORE the operation runs.
            //
            // The `else` is unreachable rather than defensive, and the two
            // filters above are why. `rt_dim_source` mints `ScalarInput` only
            // from `RtDim::Node` and `InputAxis` only from `RtDim::InputAxis`,
            // and the source filter just above admits no other source; a
            // member's source and its carrier are therefore the same fact read
            // two ways. `expand_or_reshape_carrier` re-reads the carrier from
            // the operation, so the two cannot drift apart silently:
            // `every_local_class_site_carries_the_carrier_its_source_names`
            // fails if a future source widens the admitted set without
            // widening this.
            let Some(carrier) = expand_or_reshape_carrier(&node.op, member.axis) else {
                continue;
            };
            // [04-NUM-9]'s `<op>` names the operation that introduces the
            // guarded extent, in the same vocabulary every other trap on this
            // lane uses.
            let op = if matches!(node.op, RiscOp::Expand { .. }) {
                // A malformed unverified node supplies no operation claim. The
                // verifier owns rejecting it; every executable node has a form.
                let Some(kind) = expansion_kind(dag, member.node) else {
                    continue;
                };
                kind.primitive_name()
            } else {
                crate::grad::risc_op_name(&node.op)
            };
            sites.push((
                (member.node.0, member.axis),
                LocalGuardClaim {
                    claim: name.clone(),
                    canonical: match resolved {
                        Some(value) => CanonicalExtent::Resolved(value),
                        None => CanonicalExtent::Binder(name.clone()),
                    },
                    op,
                    observed: LocalGuardObservation::Carrier(carrier.clone()),
                },
            ));
        }
    }

    // chelis#1277 S2b: the unit-extent claims section 4.7 places LOCAL, beside
    // the class members above and through the same emission.
    //
    // The site is keyed on the OPERAND's axis, because that is the extent the
    // guard reads: the claim asserts something about the operand, not about the
    // `expand`'s own output axis, and keying it on the `expand` would hand the
    // emitter the width being broadcast TO rather than the extent being
    // claimed. The claimed value is the literal 1, so both the reported claim
    // and the canonical value are `1`.
    //
    // `op` comes from the claim rather than from the operand's own operation.
    // Section 4.7's `<op>` names "the operation that introduces the guarded
    // extent", which is the `expand` making the claim, not whichever operation
    // happened to produce the operand.
    //
    // It is derived HERE rather than in the ownership view because C2.7 puts
    // the local site derivation in one place that both lanes read. S2b added
    // this loop beside the class loop when both lived in the view; the loop is
    // unchanged, and it moves with the function it was appended to.
    for claim in derive_unit_extent_claims(dag) {
        if claim.placement(dag) != GuardPlacement::Local {
            continue;
        }
        sites.push((
            (claim.operand.0, claim.axis),
            LocalGuardClaim {
                claim: "1".to_string(),
                canonical: CanonicalExtent::Resolved(1),
                op: claim.trap_op(dag),
                // The claim is about the extent the OPERAND produced, and no
                // carrier states it: the operand's own operation was not given
                // this axis, it computed it. So it is read from the realized
                // shape, which is why the read instruction is data rather than
                // something a consumer infers from the site's `op`.
                observed: LocalGuardObservation::RealizedExtent,
            },
        ));
    }
    sites
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::TensorType;

    fn ty(dims: Vec<DimInfo>, precision: Prim) -> TensorType {
        TensorType { dims, precision }
    }

    #[test]
    fn a_sym_reshape_target_is_the_operations_own_extent() {
        let mut dag = Dag::new();
        let operand = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![DimInfo::Named("n".into(), None)], Prim::F32),
            None,
        );
        let reshaped = dag.add_node(
            RiscOp::Reshape {
                new_shape: vec![RtDim::Sym("n".into())],
            },
            vec![operand],
            ty(vec![DimInfo::Named("n".into(), None)], Prim::F32),
            None,
        );
        assert_eq!(
            output_axis_sources(&dag, reshaped),
            vec![AxisSource::OpComputed {
                op: reshaped,
                axis: 0
            }],
        );
    }

    #[test]
    fn a_permute_reads_the_permuted_input_axis_not_the_output_index() {
        let mut dag = Dag::new();
        let operand = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![DimInfo::Lit(2), DimInfo::Lit(3)], Prim::F32),
            None,
        );
        let permuted = dag.add_node(
            RiscOp::Permute { axes: vec![1, 0] },
            vec![operand],
            ty(vec![DimInfo::Lit(3), DimInfo::Lit(2)], Prim::F32),
            None,
        );
        assert_eq!(
            output_axis_sources(&dag, permuted),
            vec![
                AxisSource::InputAxis {
                    input: 0,
                    axis: RtAxis::Lit(1)
                },
                AxisSource::InputAxis {
                    input: 0,
                    axis: RtAxis::Lit(0)
                },
            ],
        );
    }

    #[test]
    fn a_gather_composes_both_operands_shapes() {
        let mut dag = Dag::new();
        let values = dag.add_node(
            RiscOp::Load { name: "v".into() },
            vec![],
            ty(
                vec![DimInfo::Lit(2), DimInfo::Lit(5), DimInfo::Lit(7)],
                Prim::F32,
            ),
            None,
        );
        let indices = dag.add_node(
            RiscOp::Load { name: "i".into() },
            vec![],
            ty(vec![DimInfo::Lit(4)], Prim::Int64),
            None,
        );
        let gathered = dag.add_node(
            RiscOp::Gather { axis: 1 },
            vec![values, indices],
            ty(
                vec![DimInfo::Lit(2), DimInfo::Lit(4), DimInfo::Lit(7)],
                Prim::F32,
            ),
            None,
        );
        assert_eq!(
            output_axis_sources(&dag, gathered),
            vec![
                AxisSource::InputAxis {
                    input: 0,
                    axis: RtAxis::Lit(0)
                },
                AxisSource::InputAxis {
                    input: 1,
                    axis: RtAxis::Lit(0)
                },
                AxisSource::InputAxis {
                    input: 0,
                    axis: RtAxis::Lit(2)
                },
            ],
        );
        assert!(check_axis_sources(&dag, Stage::Lowering).is_ok());
    }

    #[test]
    fn a_blas_matmul_reads_its_batch_axes_off_the_operands() {
        use crate::dag::DimExpr;

        let mut dag = Dag::new();
        let lhs = dag.add_node(
            RiscOp::Load { name: "a".into() },
            vec![],
            ty(
                vec![DimInfo::Lit(2), DimInfo::Lit(4), DimInfo::Lit(3)],
                Prim::F32,
            ),
            None,
        );
        let rhs = dag.add_node(
            RiscOp::Load { name: "b".into() },
            vec![],
            ty(
                vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(5)],
                Prim::F32,
            ),
            None,
        );
        let product = dag.add_node(
            RiscOp::BlasMatmul {
                batch_dims: vec![DimExpr::Concrete(2)],
                m: DimExpr::Concrete(4),
                n: DimExpr::Concrete(5),
                k: DimExpr::Concrete(3),
                accumulator: Prim::F32,
            },
            vec![lhs, rhs],
            ty(
                vec![DimInfo::Lit(2), DimInfo::Lit(4), DimInfo::Lit(5)],
                Prim::F32,
            ),
            None,
        );
        assert_eq!(
            output_axis_sources(&dag, product),
            vec![
                AxisSource::InputAxis {
                    input: 0,
                    axis: RtAxis::Lit(0)
                },
                AxisSource::OpComputed {
                    op: product,
                    axis: 1
                },
                AxisSource::OpComputed {
                    op: product,
                    axis: 2
                },
            ],
            "the batch axis is the operand's axis; only m and n are contracted"
        );
        assert!(check_axis_sources(&dag, Stage::Lowering).is_ok());

        // Negative parity: an unbatched matmul has no pass-through axis.
        let mut flat = Dag::new();
        let lhs = flat.add_node(
            RiscOp::Load { name: "a".into() },
            vec![],
            ty(vec![DimInfo::Lit(4), DimInfo::Lit(3)], Prim::F32),
            None,
        );
        let rhs = flat.add_node(
            RiscOp::Load { name: "b".into() },
            vec![],
            ty(vec![DimInfo::Lit(3), DimInfo::Lit(5)], Prim::F32),
            None,
        );
        let product = flat.add_node(
            RiscOp::BlasMatmul {
                batch_dims: Vec::new(),
                m: DimExpr::Concrete(4),
                n: DimExpr::Concrete(5),
                k: DimExpr::Concrete(3),
                accumulator: Prim::F32,
            },
            vec![lhs, rhs],
            ty(vec![DimInfo::Lit(4), DimInfo::Lit(5)], Prim::F32),
            None,
        );
        assert_eq!(
            output_axis_sources(&flat, product),
            vec![
                AxisSource::OpComputed {
                    op: product,
                    axis: 0
                },
                AxisSource::OpComputed {
                    op: product,
                    axis: 1
                },
            ],
        );
    }

    #[test]
    fn a_shape_dep_sibling_supplies_an_input_less_nodes_wildcard_axis() {
        let mut dag = Dag::new();
        let sibling = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![DimInfo::Named("n".into(), None)], Prim::F32),
            None,
        );
        let mask = dag.add_node(
            RiscOp::synth_const(Prim::F32, 1.0),
            vec![],
            ty(vec![DimInfo::Named("*".into(), None)], Prim::F32),
            None,
        );
        assert_eq!(
            output_axis_sources(&dag, mask),
            vec![],
            "with no shape dependency the wildcard axis is sourceless"
        );
        dag.add_shape_dep(mask, sibling);
        assert_eq!(
            output_axis_sources(&dag, mask),
            vec![AxisSource::ClassSupplied { op: mask, axis: 0 }],
            "the recorded sibling relation SUPPLIES the axis, so it is              class-supplied rather than computed by this node"
        );
    }

    /// C4.1's duplicated direction, and the misdirected `OpComputed` case.
    /// A correct exhaustive match cannot produce either, so the vector is
    /// presented directly. `omitted_or_duplicated_output_axis_source_fails_before_emission`
    /// is the same rule reached through real operations.
    #[test]
    fn a_duplicated_or_misdirected_source_vector_is_rejected() {
        let mut dag = Dag::new();
        let operand = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![DimInfo::Lit(2)], Prim::F32),
            None,
        );
        let negated = dag.add_node(
            RiscOp::Neg,
            vec![operand],
            ty(vec![DimInfo::Lit(2)], Prim::F32),
            None,
        );
        let node = dag.get(negated).expect("node");
        let one = AxisSource::InputAxis {
            input: 0,
            axis: RtAxis::Lit(0),
        };

        assert!(
            check_node_axis_sources(&dag, node, std::slice::from_ref(&one), Stage::Lowering)
                .is_ok(),
            "exactly one source per output axis is the accepted cardinality"
        );

        let duplicated = check_node_axis_sources(&dag, node, &[one.clone(), one], Stage::Lowering)
            .expect_err("two sources for a rank 1 output must fail");
        assert!(duplicated.to_string().starts_with("unsupported: "));
        assert!(duplicated.to_string().contains("chelis#1277"));

        // `OpComputed` must name this node and this axis: an extent another
        // node computes is that node's axis, not a source for this one.
        let misdirected = check_node_axis_sources(
            &dag,
            node,
            &[AxisSource::OpComputed {
                op: operand,
                axis: 0,
            }],
            Stage::Lowering,
        )
        .expect_err("OpComputed must name the node whose axis it is");
        assert!(misdirected.to_string().contains("rather than by itself"));
    }

    #[test]
    fn an_external_axis_source_must_name_a_real_load_and_a_real_axis() {
        let mut dag = Dag::new();
        let source = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![DimInfo::Lit(2)], Prim::F32),
            None,
        );
        let negated = dag.add_node(
            RiscOp::Neg,
            vec![source],
            ty(vec![DimInfo::Lit(2)], Prim::F32),
            None,
        );
        let node = dag.get(negated).expect("node");

        assert!(
            check_node_axis_sources(
                &dag,
                node,
                &[AxisSource::ExternalAxis {
                    load: source,
                    axis: 0
                }],
                Stage::Lowering,
            )
            .is_ok()
        );

        let not_a_load = check_node_axis_sources(
            &dag,
            node,
            &[AxisSource::ExternalAxis {
                load: negated,
                axis: 0,
            }],
            Stage::Lowering,
        )
        .expect_err("a non-Load declaring node must fail");
        assert!(not_a_load.to_string().contains("which is not a Load"));

        let out_of_range = check_node_axis_sources(
            &dag,
            node,
            &[AxisSource::ExternalAxis {
                load: source,
                axis: 4,
            }],
            Stage::Lowering,
        )
        .expect_err("an axis past the Load's rank must fail");
        assert!(out_of_range.to_string().contains("rank 1 Load"));
    }

    #[test]
    fn an_input_axis_source_must_be_a_normalized_in_range_axis_of_a_real_slot() {
        let mut dag = Dag::new();
        let source = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![DimInfo::Lit(2)], Prim::F32),
            None,
        );
        let negated = dag.add_node(
            RiscOp::Neg,
            vec![source],
            ty(vec![DimInfo::Lit(2)], Prim::F32),
            None,
        );
        let node = dag.get(negated).expect("node");

        for (source_vector, needle) in [
            (
                AxisSource::InputAxis {
                    input: 3,
                    axis: RtAxis::Lit(0),
                },
                "the node has 1 input(s)",
            ),
            (
                AxisSource::InputAxis {
                    input: 0,
                    axis: RtAxis::Lit(-1),
                },
                "non-normalized axis",
            ),
            (
                AxisSource::InputAxis {
                    input: 0,
                    axis: RtAxis::Lit(7),
                },
                "of a rank 1 tensor",
            ),
        ] {
            let error = check_node_axis_sources(&dag, node, &[source_vector], Stage::Lowering)
                .expect_err("an invalid InputAxis must fail");
            assert!(error.to_string().contains(needle), "{error}");
        }
    }

    #[test]
    fn a_scalar_input_source_must_be_a_rank_zero_int64_outside_slot_zero() {
        let mut dag = Dag::new();
        let value = dag.add_node(
            RiscOp::Load {
                name: "value".into(),
            },
            vec![],
            ty(vec![], Prim::F32),
            None,
        );
        let extent = dag.add_node(
            RiscOp::Load {
                name: "extent".into(),
            },
            vec![],
            ty(vec![], Prim::Int64),
            None,
        );
        let tensor = dag.add_node(
            RiscOp::Load {
                name: "tensor".into(),
            },
            vec![],
            ty(vec![DimInfo::Lit(2)], Prim::Int64),
            None,
        );
        let expanded = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: RtDim::Node(1),
            },
            vec![value, extent, tensor],
            ty(vec![DimInfo::Named("n".into(), None)], Prim::F32),
            None,
        );
        let node = dag.get(expanded).expect("node");

        assert!(
            check_node_axis_sources(
                &dag,
                node,
                &[AxisSource::ScalarInput { input: 1 }],
                Stage::Lowering,
            )
            .is_ok()
        );

        for (slot, needle) in [
            (0, "which is the tensor operand"),
            (2, "must be rank 0"),
            (9, "the node has 3 input(s)"),
        ] {
            let error = check_node_axis_sources(
                &dag,
                node,
                &[AxisSource::ScalarInput { input: slot }],
                Stage::Lowering,
            )
            .expect_err("an invalid ScalarInput must fail");
            assert!(error.to_string().contains(needle), "{error}");
        }
    }

    #[test]
    fn a_negative_literal_extent_is_rejected() {
        let mut dag = Dag::new();
        let source = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![DimInfo::Lit(2)], Prim::F32),
            None,
        );
        let node = dag.get(source).expect("node");
        let error = check_node_axis_sources(
            &dag,
            node,
            &[AxisSource::Literal { value: -1 }],
            Stage::Lowering,
        )
        .expect_err("a negative extent is a static type error, never a source");
        assert!(error.to_string().contains("negative literal extent"));
    }

    // The four exact-vector controls below close the coverage gap that
    // `or_op_computed` opens. That helper normalizes any length mismatch to
    // exactly `rank` sources, so the table-driven cardinality test in
    // `runtime_extent_slice_b_sources.rs` structurally CANNOT see a wrong
    // count in an arm it wraps. Every narrowed arm therefore needs a test
    // that asserts the exact source vector, not just its length. The
    // reductions arm, `Count`, and `Gather` already had one; these are the
    // remaining four.

    #[test]
    fn a_reduce_window_passes_leading_axes_and_computes_the_windowed_ones() {
        let mut dag = Dag::new();
        let operand = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![DimInfo::Lit(2), DimInfo::Lit(3)], Prim::F32),
            None,
        );
        let windowed = dag.add_node(
            RiscOp::ReduceWindow {
                reducer: crate::dag::ReduceWindowKind::Sum,
                window_shape: vec![2],
                strides: vec![1],
            },
            vec![operand],
            ty(vec![DimInfo::Lit(2), DimInfo::Lit(2)], Prim::F32),
            None,
        );
        assert_eq!(
            output_axis_sources(&dag, windowed),
            vec![
                AxisSource::InputAxis {
                    input: 0,
                    axis: RtAxis::Lit(0)
                },
                AxisSource::OpComputed {
                    op: windowed,
                    axis: 1
                },
            ],
            "the leading axis passes through; the windowed extent is the op's own"
        );
        assert!(check_axis_sources(&dag, Stage::Lowering).is_ok());
    }

    #[test]
    fn a_reduce_window_grad_restores_the_forward_inputs_shape() {
        let mut dag = Dag::new();
        let forward_input = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![DimInfo::Lit(2), DimInfo::Lit(3)], Prim::F32),
            None,
        );
        let cotangent = dag.add_node(
            RiscOp::Load { name: "g".into() },
            vec![],
            ty(vec![DimInfo::Lit(2), DimInfo::Lit(2)], Prim::F32),
            None,
        );
        let adjoint = dag.add_node(
            RiscOp::ReduceWindowGrad {
                reducer: crate::dag::ReduceWindowKind::Sum,
                window_shape: vec![2],
                strides: vec![1],
            },
            vec![forward_input, cotangent],
            ty(vec![DimInfo::Lit(2), DimInfo::Lit(3)], Prim::F32),
            None,
        );
        assert_eq!(
            output_axis_sources(&dag, adjoint),
            vec![
                AxisSource::InputAxis {
                    input: 0,
                    axis: RtAxis::Lit(0)
                },
                AxisSource::InputAxis {
                    input: 0,
                    axis: RtAxis::Lit(1)
                },
            ],
            "the adjoint's output is the forward INPUT's shape, never the cotangent's"
        );
        assert!(check_axis_sources(&dag, Stage::Lowering).is_ok());
    }

    #[test]
    fn a_one_hot_appends_the_vocab_literal_to_the_index_axes() {
        let mut dag = Dag::new();
        let indices = dag.add_node(
            RiscOp::Load { name: "i".into() },
            vec![],
            ty(vec![DimInfo::Lit(4)], Prim::Int64),
            None,
        );
        let dense = dag.add_node(
            RiscOp::OneHot { vocab: 5 },
            vec![indices],
            ty(vec![DimInfo::Lit(4), DimInfo::Lit(5)], Prim::F32),
            None,
        );
        assert_eq!(
            output_axis_sources(&dag, dense),
            vec![
                AxisSource::InputAxis {
                    input: 0,
                    axis: RtAxis::Lit(0)
                },
                AxisSource::Literal { value: 5 },
            ],
        );
        assert!(check_axis_sources(&dag, Stage::Lowering).is_ok());
    }

    #[test]
    fn the_scatter_family_reads_its_output_axes_off_the_target() {
        for op in [
            RiscOp::ScatterAdd { axis: 1 },
            RiscOp::Scatter { axis: 1 },
            RiscOp::ScatterElements { axis: 1 },
        ] {
            let mut dag = Dag::new();
            let target = dag.add_node(
                RiscOp::Load { name: "t".into() },
                vec![],
                ty(
                    vec![DimInfo::Lit(2), DimInfo::Lit(5), DimInfo::Lit(7)],
                    Prim::F32,
                ),
                None,
            );
            let indices = dag.add_node(
                RiscOp::Load { name: "i".into() },
                vec![],
                ty(vec![DimInfo::Lit(4)], Prim::Int64),
                None,
            );
            let updates = dag.add_node(
                RiscOp::Load { name: "u".into() },
                vec![],
                ty(
                    vec![DimInfo::Lit(2), DimInfo::Lit(4), DimInfo::Lit(7)],
                    Prim::F32,
                ),
                None,
            );
            let scattered = dag.add_node(
                op.clone(),
                vec![target, indices, updates],
                ty(
                    vec![DimInfo::Lit(2), DimInfo::Lit(5), DimInfo::Lit(7)],
                    Prim::F32,
                ),
                None,
            );
            assert_eq!(
                output_axis_sources(&dag, scattered),
                vec![
                    AxisSource::InputAxis {
                        input: 0,
                        axis: RtAxis::Lit(0)
                    },
                    AxisSource::InputAxis {
                        input: 0,
                        axis: RtAxis::Lit(1)
                    },
                    AxisSource::InputAxis {
                        input: 0,
                        axis: RtAxis::Lit(2)
                    },
                ],
                "every scatter writes into a copy of its target, so the shape is the target's: {op:?}"
            );
            assert!(check_axis_sources(&dag, Stage::Lowering).is_ok());
        }
    }
}
