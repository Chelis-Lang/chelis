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
/// this node's own tensor inputs, a rank-0 `i64` scalar input, or a value
/// the operation computes by its own output-shape rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxisSource {
    /// A compile-time-constant extent.
    Literal { value: i64 },
    /// An axis of an external `Load`, named by the exact declaring node.
    ExternalAxis { load: NodeId, axis: usize },
    /// An axis of the tensor in this node's absolute input slot `input`.
    InputAxis { input: usize, axis: RtAxis },
    /// A rank-0 exact-`i64` extent value in absolute input slot `input`.
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

/// Whether a dimension name was minted by the compiler rather than written in
/// a signature.
///
/// Three minting vocabularies exist and all three name a FRESH extent that no
/// signature declares: the lowerer's `_rt_<operation>_dim_<node>_<axis>` for
/// an extent an operation computes, the C emitter's and DAG's
/// `_anon_dim_<node>_<axis>` for an axis with no name at all, and the
/// checker's `d<N>` for an unresolved dimension variable. An anonymous
/// spelling is included, since it is the same fact with no spelling.
///
/// The distinction this draws is ownership, not cosmetics. A user-spelled
/// name on an axis is a claim some signature makes about it, with its own
/// declaring witness and its own guard; a synthesized name is the compiler's
/// placeholder for an extent nothing has claimed yet. Consumers that must not
/// overwrite one signature's claim with another's ask this question.
///
/// `d<N>` is the one spelling a user could also write. Nothing distinguishes
/// them at this layer, so a signature declaring `tensor[d0, f32]` is read as
/// synthesized. That is the checker's existing display vocabulary rather than
/// a choice made here.
pub fn is_synthesized_dim_name(name: &str) -> bool {
    is_anonymous(name)
        || name.starts_with("_rt_")
        || name.starts_with("_anon_dim_")
        || name
            .strip_prefix('d')
            .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
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

/// Whether this operation semantically produces a result with the shape of
/// all of its positive-rank operands.
///
/// Administrative identity nodes stay outside this set: a `Copy`, `Drop`,
/// `Realize`, or `Store` forwards an existing value rather than becoming the
/// primitive named by a declared-result guard. Casts remain in the set because
/// spec/04 §4.7 gives a cast the placement of its input while retaining the
/// cast as the primitive that produced the returned value.
pub fn is_same_shape_result_op(op: &RiscOp) -> bool {
    matches!(
        op,
        RiscOp::Add
            | RiscOp::Sub
            | RiscOp::Mul
            | RiscOp::Div
            | RiscOp::FloorDiv
            | RiscOp::TruncDiv
            | RiscOp::Mod
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
            | RiscOp::Cast { .. }
            | RiscOp::CastTrunc { .. }
            | RiscOp::FusedElem { .. }
    )
}

/// The complete positive-rank operand relation for a same-shape result.
///
/// Members are node identities rather than input slots. Repeated edges to the
/// same value therefore form one agreement member, while distinct operand
/// paths remain distinct. Source order is retained only for deterministic
/// comparison order; no member is the result claim's semantic owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SameShapeAgreement {
    members: Vec<NodeId>,
}

impl SameShapeAgreement {
    pub fn members(&self) -> &[NodeId] {
        &self.members
    }
}

/// Derive the complete agreement relation for a positive-rank same-shape
/// result.
///
/// `Ok(None)` means the operation is not a same-shape producer, or that its
/// result is rank zero and therefore has no result axis to observe. A
/// positive-rank producer must have at least one positive-rank operand, and
/// every such operand must have the result rank. Returning an error rather
/// than an empty set prevents a malformed result claim from disappearing.
pub fn same_shape_result_agreement(
    dag: &Dag,
    node: NodeId,
) -> Result<Option<SameShapeAgreement>, String> {
    let owner = dag
        .get(node)
        .ok_or_else(|| format!("same-shape result references missing node {}", node.0))?;
    if !is_same_shape_result_op(&owner.op) || owner.output_type.dims.is_empty() {
        return Ok(None);
    }
    let result_rank = owner.output_type.dims.len();
    let mut members = Vec::new();
    // A random primitive's data operand is its only same-shape operand. Its
    // key, controls and activation are shaped like leading parts of the data
    // (spec/10 §3.2), whose agreement with the data's leading axes the draw
    // checks itself.
    let operands = match owner.op {
        RiscOp::UniformLike | RiscOp::Dropout | RiscOp::DropoutReplay => {
            &owner.inputs[..owner.inputs.len().min(1)]
        }
        _ => &owner.inputs[..],
    };
    for input in operands {
        let operand = dag.get(*input).ok_or_else(|| {
            format!(
                "same-shape result at node {} references missing operand {}",
                node.0, input.0
            )
        })?;
        let operand_rank = operand.output_type.dims.len();
        if operand_rank == 0 {
            continue;
        }
        if operand_rank != result_rank {
            return Err(format!(
                "same-shape result at node {} has positive-rank operand {} of rank {}, expected rank {}",
                node.0, input.0, operand_rank, result_rank
            ));
        }
        if !members.contains(input) {
            members.push(*input);
        }
    }
    if members.is_empty() {
        return Err(format!(
            "same-shape result at node {} has rank {} but no positive-rank agreement member",
            node.0, result_rank
        ));
    }
    Ok(Some(SameShapeAgreement { members }))
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
        | RiscOp::Compare(_)
        | RiscOp::Logical(_)
        | RiscOp::Where
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
        | RiscOp::Copy
        | RiscOp::Drop
        | RiscOp::Realize
        | RiscOp::Cast { .. }
        | RiscOp::CastTrunc { .. }
        | RiscOp::FusedElem { .. }
        | RiscOp::Store { .. }
        | RiscOp::KeyFromSeed
        | RiscOp::Split { .. }
        | RiscOp::FoldIn
        | RiscOp::KeySelect => shape_preserving(dag, node),

        // [05-OP-71]: the key's axes pass through and the new last axis is
        // the count's own typed carrier.
        RiscOp::SplitN { count } => match input_rank(dag, node, 0) {
            Some(key_rank) if key_rank + 1 == rank => (0..key_rank)
                .map(|axis| pass_through(id, 0, axis))
                .chain(std::iter::once(rt_dim_source(id, key_rank, count)))
                .collect(),
            _ => op_computed(id, rank),
        },

        // Rank-0 result: a bound adjoint is a scalar sum with no axis.
        RiscOp::UniformBoundAdjoint { .. } => op_computed(id, rank),

        // chelis#1464 / [05-OP-68]: the result IS the fallback, so every
        // output axis comes from input slot 1.
        //
        // Deliberately NOT `shape_preserving`, which picks the FIRST
        // rank-matching input: under `vmap` the condition is batched to
        // rank 1 alongside the fallback, so it would select slot 0, the
        // condition. `schema.rs`'s wire-side axis origin already reads
        // `inputs.get(1)` directly; this keeps the two in agreement.
        RiscOp::GuardedFail { .. } => match input_rank(dag, node, 1) {
            Some(fallback_rank) if fallback_rank == rank => {
                (0..rank).map(|axis| pass_through(id, 1, axis)).collect()
            }
            _ => op_computed(id, rank),
        },

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
/// must validate the tensor slot and its normalized `i32` axis;
/// `ScalarInput` must validate C2.1's rank-0 exact-`i64` contract. A
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

/// Every name a lane renders as an identifier resolves to an origin.
///
/// This is what replaced the occurrence walk's `panic!` (chelis#665).
/// [`check_axis_sources`] asks whether each AXIS has a source; this asks
/// whether each NAME the emitted text mentions has one place that assigns it.
/// The two are not the same question: a name can be carried by an axis whose
/// own source is a pass-through into an operand whose axis has none, and the
/// emitted text would then mention an identifier nothing declares.
///
/// Refusing is the whole point. Emitting the name anyway produces C that does
/// not compile at best and a mis-sized allocation at worst, and guessing an
/// extent from another axis that happens to share the spelling is the
/// string-matching defect this module exists to remove.
///
/// This is an EMISSION obligation and not a lowering one, which is why it is
/// not folded into [`check_axis_sources`]. Only a lane that renders a name as
/// an identifier owes a declaration for it, and a graph can be perfectly
/// well-formed for ownership, capacity planning or evaluation while carrying
/// a name no lane has to render: the evaluator computes every extent from
/// actual values and refuses a name it genuinely needs with its own
/// missing-binding error.
///
/// The C emitter is the only lane that calls this today, and that is a stated
/// boundary rather than an oversight. HIP declares from the interface
/// bindings alone and Metal consumes none of this derivation, so neither is
/// migrated onto origins and neither owes the check yet; adding it to a lane
/// whose declarations come from elsewhere would refuse programs that lane
/// emits correctly. Migrating them is the residual, tracked with the rest of
/// the declaration work.
///
/// The scan walks nodes rather than names, so the node the receipt blames is
/// the carrier the scan found, never a positional fallback: a name reaches
/// [`unresolved_dim_names`] only by way of a node that carries it, so there
/// is nothing to fall back to and an empty graph reports nothing.
pub fn check_rendered_dim_origins(dag: &Dag, stage: Stage) -> Result<(), Unsupported> {
    let unresolved = unresolved_dim_names(dag);
    if unresolved.is_empty() {
        return Ok(());
    }
    for node in dag.nodes() {
        let carried = node
            .output_type
            .dims
            .iter()
            .filter_map(|dim| match dim {
                DimInfo::Named(name, None) => Some(name.clone()),
                _ => None,
            })
            .chain(crate::dag::op_internal_symbolic_dims(&node.op))
            .find(|name| unresolved.contains(name));
        if let Some(name) = carried {
            return Err(receipt(
                format!("extent `{name}` resolves to no source"),
                node,
                stage,
                true,
            ));
        }
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
                            "output axis {axis} reads a `{}` extent value in input slot {input}, which must be i64",
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

// ===========================================================================
// C4.4's remaining half: where a named extent's value is PRODUCED.
//
// `output_axis_sources` answers "what determines this axis", one hop, in the
// operation's own terms. A declaration consumer needs the terminal answer
// instead: the emitted C allocates by NAME (`chelis_alloc(1, (int64_t[]){
// _anon_dim_2_1 })`), so every name it renders needs one place that assigns
// it. Resolving the one-hop chain to its terminal origin is what lets a
// declaration come from the axis SOURCE rather than from a search for a
// `Load` carrying the same string, which is what chelis#665 and chelis#1556
// both fail.
// ===========================================================================

/// Where the value of one output axis's extent is PRODUCED, after resolving
/// every pass-through hop.
///
/// This is the terminal form of [`AxisSource`]: `InputAxis` is a hop rather
/// than an origin (the extent belongs to the operand's axis, which has a
/// source of its own), and `ClassSupplied` states that the claim supplies the
/// extent without saying which member produces it. Both resolve here;
/// everything else is already terminal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtentOrigin {
    /// A compile-time-constant extent.
    Literal(i64),
    /// An axis of an external `Load`, named by the exact declaring node. The
    /// C and HIP prologues can declare this one from input shape metadata
    /// before any operation runs.
    ExternalAxis { load: NodeId, axis: usize },
    /// A rank-0 exact-`i64` extent value produced by `value`, read by the
    /// operation at output axis `axis` of `at`. The operation is where a lane
    /// renders the read, so it is also where a declaration goes; `value`
    /// names the node the extent comes out of.
    ScalarInput {
        value: NodeId,
        at: NodeId,
        axis: usize,
    },
    /// An extent the operation computes by its own output-shape rule, so it
    /// exists only once that operation has run and must be declared there.
    OpComputed { op: NodeId, axis: usize },
}

/// The origin of `node`'s output axis `axis`, resolving pass-through hops.
///
/// `InputAxis { input, Lit(a) }` recurses into the operand in that absolute
/// input slot at axis `a`: a kept axis's extent IS the operand's, which is
/// why it is not a witness of its own claim (`derive_dim_witnesses` skips it
/// for exactly that reason) and why a declaration for it has to come from
/// wherever the operand's axis is produced.
///
/// `ClassSupplied` resolves through the claim's class: the axis is sized by
/// whatever the class resolves to, so its origin is the origin of the class's
/// canonical member. A class with no member that resolves anywhere has no
/// origin, which is a typed receipt rather than a guess.
///
/// The walk is bounded by the node count, so a malformed graph cannot spin.
/// `None` means the axis has no resolvable origin.
pub fn resolve_axis_extent(dag: &Dag, node: NodeId, axis: usize) -> Option<ExtentOrigin> {
    resolve_axis_extent_bounded(dag, node, axis, dag.nodes().len())
}

fn resolve_axis_extent_bounded(
    dag: &Dag,
    node: NodeId,
    axis: usize,
    fuel: usize,
) -> Option<ExtentOrigin> {
    if fuel == 0 {
        return None;
    }
    let source = output_axis_sources(dag, node).into_iter().nth(axis)?;
    match source {
        AxisSource::Literal { value } => Some(ExtentOrigin::Literal(value)),
        AxisSource::ExternalAxis { load, axis } => Some(ExtentOrigin::ExternalAxis { load, axis }),
        AxisSource::OpComputed { op, axis } => Some(ExtentOrigin::OpComputed { op, axis }),
        AxisSource::ScalarInput { input } => {
            let value = *dag.get(node)?.inputs.get(input)?;
            Some(ExtentOrigin::ScalarInput {
                value,
                at: node,
                axis,
            })
        }
        AxisSource::InputAxis { input, axis } => {
            let RtAxis::Lit(read_axis) = axis;
            let operand = *dag.get(node)?.inputs.get(input)?;
            let read_axis = usize::try_from(read_axis).ok()?;
            resolve_axis_extent_bounded(dag, operand, read_axis, fuel - 1)
        }
        AxisSource::ClassSupplied { op, axis } => {
            resolve_class_supplied_extent(dag, op, axis, fuel - 1)
        }
    }
}

/// The origin a `ClassSupplied` axis takes from its class.
///
/// The axis consumes its claim rather than witnessing it, so its extent is
/// the extent the class resolves to. Take the first member of that class
/// whose own origin resolves, in the class's canonical order, skipping the
/// axis itself so a one-member class cannot recurse into itself.
fn resolve_class_supplied_extent(
    dag: &Dag,
    op: NodeId,
    axis: usize,
    fuel: usize,
) -> Option<ExtentOrigin> {
    if fuel == 0 {
        return None;
    }
    let claim = axis_claim(dag.get(op)?.output_type.dims.get(axis)?)?;
    derive_runtime_dim_classes(dag)
        .into_iter()
        .filter(|class| class.claim == claim)
        .flat_map(|class| class.members)
        .filter(|member| member.node != op || member.axis != axis)
        .find_map(|member| resolve_axis_extent_bounded(dag, member.node, member.axis, fuel - 1))
}

/// Every name a lane RENDERS as a C identifier, with the origin that
/// produces its value, in first-reference (node-id) order.
///
/// Two carriers render a name, and both are here because a declaration
/// consumer must cover both or emit an undeclared identifier: an unbound
/// `DimInfo::Named(name, None)` in an output type, which `emit_dim_info`
/// prints verbatim, and an op-internal reference, which is a `Reshape`
/// target's `RtDim::Sym` or one of `BlasMatmul`'s dimension expressions. A
/// statically bound `Named(name, Some(4))` renders as its literal and needs
/// no declaration, but it is still a CANDIDATE for locating the name's
/// origin, exactly as the legacy walk's `bind_symbol_from_any_load` accepted
/// one.
///
/// Selection among a name's candidate axes has one rule beyond node order,
/// and it is ordered by where a lane can actually put the declaration. An
/// `ExternalAxis` wins outright: it is an input tensor's axis, readable from
/// shape metadata before any operation runs, so declaring from it dominates
/// every reference to the name. That is the split the legacy walk made by
/// asking whether the name was "also Load-carried", and the derivation's
/// guards still compare it against any operation that stamps the same name on
/// a fresh axis. An `OpComputed` or `ScalarInput` comes next, because it
/// names the operation that produces the extent and so names a site. A
/// `Literal` comes LAST despite being the most certain answer, because it
/// names no site at all: no lane declares an entry literal today, so choosing
/// one over an available site would leave the name undeclared.
///
/// A name with no resolvable origin is absent here and listed by
/// [`unresolved_dim_names`] instead, so a consumer fails closed with a
/// receipt rather than panicking or guessing an extent.
pub fn dim_extent_origins(dag: &Dag) -> Vec<(String, ExtentOrigin)> {
    rendered_dim_names(dag)
        .into_iter()
        .filter_map(|name| {
            let origin = resolve_named_dim_origin(dag, &name)?;
            Some((name, origin))
        })
        .collect()
}

/// Every name a lane renders as a C identifier that [`dim_extent_origins`]
/// could NOT resolve to an origin. A consumer turns each into a typed
/// receipt rather than a panic or a guessed extent.
pub fn unresolved_dim_names(dag: &Dag) -> Vec<String> {
    rendered_dim_names(dag)
        .into_iter()
        .filter(|name| resolve_named_dim_origin(dag, name).is_none())
        .collect()
}

/// The names a lane renders as identifiers, deduplicated, in first-reference
/// node-id order. Within one node the output axes come before the op-internal
/// references, which is the order the emitter writes them in.
///
/// This is the complete set a declaration consumer owes a declaration for,
/// and the complete set [`crate::dag::bind_symbolic_dims`] can refuse.
pub fn rendered_dim_names(dag: &Dag) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for node in dag.nodes() {
        let axis_names = node.output_type.dims.iter().filter_map(|dim| match dim {
            DimInfo::Named(name, None) => Some(name.clone()),
            _ => None,
        });
        for name in axis_names.chain(crate::dag::op_internal_symbolic_dims(&node.op)) {
            if !is_anonymous(&name) && !out.contains(&name) {
                out.push(name);
            }
        }
    }
    out
}

impl ExtentOrigin {
    /// The `(node, axis)` where a lane renders this extent, for an origin the
    /// function entry cannot supply. `None` means the entry supplies it: a
    /// literal, or an input tensor's axis the prologue reads from shape
    /// metadata before any operation runs.
    pub fn local_site(&self) -> Option<(NodeId, usize)> {
        match self {
            ExtentOrigin::Literal(_) | ExtentOrigin::ExternalAxis { .. } => None,
            ExtentOrigin::OpComputed { op, axis } => Some((*op, *axis)),
            ExtentOrigin::ScalarInput { at, axis, .. } => Some((*at, *axis)),
        }
    }
}

/// The origin of one name, over every output axis that carries it, in the
/// preference order [`dim_extent_origins`] documents.
fn resolve_named_dim_origin(dag: &Dag, name: &str) -> Option<ExtentOrigin> {
    let mut local: Option<ExtentOrigin> = None;
    let mut literal: Option<ExtentOrigin> = None;
    for node in dag.nodes() {
        for (axis, dim) in node.output_type.dims.iter().enumerate() {
            if !matches!(dim, DimInfo::Named(other, _) if other == name) {
                continue;
            }
            let Some(origin) = resolve_axis_extent(dag, node.id, axis) else {
                continue;
            };
            match origin {
                ExtentOrigin::ExternalAxis { .. } => return Some(origin),
                ExtentOrigin::Literal(_) if literal.is_none() => literal = Some(origin),
                ExtentOrigin::OpComputed { .. } | ExtentOrigin::ScalarInput { .. }
                    if local.is_none() =>
                {
                    local = Some(origin)
                }
                _ => {}
            }
        }
    }
    local.or(literal)
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
/// an `i32` parameter reaching an extent through `cast(m, i64)` lands its
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
    /// Named classes compare their declaring witnesses. A literal result
    /// class instead constrains each produced axis, so its ownership must
    /// agree with an explicit literal-result token on the same axis (§4.7).
    pub fn placement(&self, dag: &Dag) -> GuardPlacement {
        if self.members.iter().all(|member| match self.claim {
            DimClaim::Literal(_) => {
                literal_result_interface_observation(dag, member.node, member.axis).is_some()
            }
            DimClaim::Name(_) => member_is_interface(dag, member),
        }) {
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
/// named witness classes and operand unit-extent claims both ask it. Literal
/// result classes use their output owner's interface observation instead:
/// an available operand does not turn its consumer into an interface value.
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
pub fn ordered_interface_loads(nodes: &[DagNode]) -> Vec<&DagNode> {
    let mut seen = Vec::new();
    let mut loads = Vec::new();
    for node in nodes {
        if let RiscOp::Load { name } = &node.op
            && !seen.contains(&name.as_str())
        {
            seen.push(name.as_str());
            loads.push(node);
        }
    }
    loads
}

/// Locate a witness in the single ordered interface-input set.
fn abi_input_slot(dag: &Dag, load: NodeId) -> Option<usize> {
    let name = match &dag.get(load)?.op {
        RiscOp::Load { name } => name.as_str(),
        _ => return None,
    };
    ordered_interface_loads(dag.nodes())
        .iter()
        .position(|node| matches!(&node.op, RiscOp::Load { name: other } if other.as_str() == name))
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
/// is what C2.5 asks for.
///
/// A `Sym` or `Lit` carrier is deliberately absent: neither computes an
/// extent, and `sets_axis` does not make either a witness on a `Reshape`.
pub(crate) fn expand_or_reshape_carrier(op: &RiscOp, axis: usize) -> Option<&RtDim> {
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
                stack.extend(node.result_claim_deps.iter().copied());
            }
        }
    }
    reach
}

/// Every node's SCOPE index, where two nodes share a scope when one root
/// reaches both of them, transitively through any node a chain of roots shares.
/// `None` marks a node no root reaches.
///
/// This is the node-level form of C2.4's scoping, over the same `root_reach`
/// masks [`split_by_scope`] buckets a claim's members with. It is deliberately
/// COARSER than that bucketing wherever two roots share a node, and it has to
/// be: a node two roots reach cannot carry a different dimension name per root,
/// so a consumer that RENAMES per scope may only split scopes that share
/// nothing. Scopes are numbered by their lowest node id, so the numbering is a
/// property of the graph rather than of iteration order.
pub fn node_scopes(dag: &Dag) -> Vec<Option<usize>> {
    let reach = root_reach(dag);
    let mut masks: Vec<u128> = Vec::new();
    for &mask in &reach {
        if mask == 0 {
            continue;
        }
        let mut merged = mask;
        let mut kept: Vec<u128> = Vec::new();
        for existing in masks.drain(..) {
            if existing & merged != 0 {
                merged |= existing;
            } else {
                kept.push(existing);
            }
        }
        kept.push(merged);
        masks = kept;
    }
    let mut ranked: Vec<(usize, usize)> = masks
        .iter()
        .enumerate()
        .map(|(index, mask)| {
            let first = reach
                .iter()
                .position(|node| *node != 0 && node & mask != 0)
                .unwrap_or(usize::MAX);
            (first, index)
        })
        .collect();
    ranked.sort_unstable();
    let mut rank = vec![0usize; masks.len()];
    for (position, (_, index)) in ranked.into_iter().enumerate() {
        rank[index] = position;
    }
    reach
        .iter()
        .map(|&mask| {
            if mask == 0 {
                return None;
            }
            masks
                .iter()
                .position(|scope| scope & mask != 0)
                .map(|index| rank[index])
        })
        .collect()
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
                // Absorb every intersecting bucket, then append the new entry,
                // so a merged bucket stays in derivation order. Pushing the
                // entry first and appending the buckets after it REVERSED the
                // group, which the sort below could not undo through a tie:
                // three axes of one node came out [2, 1, 0]. The sort key now
                // carries the axis, so this no longer decides anything on its
                // own, and it is kept order-preserving because the next tie to
                // be introduced would silently inherit the reversal.
                let mut merged: Vec<OrderedMember> = Vec::new();
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
                merged.push(entry);
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
/// position and then by the member's own axis.
///
/// The axis component is not decoration. `slot` is `Some` only for an
/// [`AxisSource::ExternalAxis`] member, and it already carries that member's
/// axis; every OTHER kind sorts with `slot = None`, so two axes of ONE node
/// tied on `(true, None, node)`. Measured before it was added: a `shrink`
/// whose two axes both claim `n` produced members `[(node, 1), (node, 0)]`,
/// and a `reshape` with three such axes produced `[2, 1, 0]` - the canonical
/// member of the class was the LAST axis, and the guards ran in reverse
/// declaration order, which `spec/04-type-system.md` section 4.7 fixes:
/// "Guards ready at the same source position are evaluated in declaration
/// order."
type OrderKey = (bool, Option<(usize, usize)>, usize, usize);

/// A member together with the key that orders it.
///
/// `slot` is the declaring `Load`'s assigned ABI input slot and axis for an interface
/// member and `None` for a local one, so sorting on `(slot.is_none(), slot,
/// node, axis)` puts every interface member ahead of every local one, orders
/// the interface group by assigned slot, and breaks every remaining tie by
/// node position and then by declaration order within that node. That is C2.4
/// rule 1 in one key.
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
        // A literal result claim follows its output owner, even when the
        // extent's value can be read from an earlier interface witness.
        let input_axis =
            literal_result_interface_observation(dag, self.member.node, self.member.axis)
                .and_then(|observation| observation.entry_axis(dag));
        let slot =
            input_axis.and_then(|(load, axis)| abi_input_slot(dag, load).map(|slot| (slot, axis)));
        (slot.is_none(), slot, self.node, self.member.axis)
    }

    fn key(&self) -> OrderKey {
        (self.slot.is_none(), self.slot, self.node, self.member.axis)
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
        //
        // One `Name` class needs only one, and for the same reason the
        // `Literal` rule gives: a RESOLVED claim is its own canonical value.
        // A single member whose source is `OpComputed` and whose dim carries
        // `Named(_, Some(v))` has both a number to compare against and an
        // extent the operation computes to compare, so it is a complete guard
        // with nothing missing. chelis#1800 is the case: `f(w: tensor[n, f32],
        // x) -> tensor[n, f32]` called with a graph-fixed `w` puts `Lit(v)` on
        // the declaring axis, which belongs to the `Literal(v)` class rather
        // than to `n`'s, leaving `n` with the op-computed member alone;
        // lowering resolves that member's claim so the number travels with it.
        //
        // The rule is deliberately this narrow. An unresolved single member
        // still forms no class, because nothing supplies its canonical value,
        // and a resolved member with any OTHER source is not admitted: an
        // entry class resolved this way would mint a guard for every
        // literal-shaped input, which is the mint `is_member`'s `ExternalAxis`
        // rule exists to prevent.
        let needed = match claim {
            DimClaim::Name(_) => {
                let resolved_op_computed = members.len() == 1
                    && matches!(members[0].member.source, AxisSource::OpComputed { .. })
                    && dag
                        .get(members[0].member.node)
                        .and_then(|node| node.output_type.dims.get(members[0].member.axis))
                        .is_some_and(|dim| matches!(dim, DimInfo::Named(_, Some(_))));
                if resolved_op_computed { 1 } else { 2 }
            }
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

/// One ordered entry check, shared by the DAG evaluator and direct C emitter.
/// Input metadata checks precede every read of that input's elements. Extent
/// comparisons retain the independent witness schedule from §4.7.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntryValidationStep {
    DType {
        load: NodeId,
    },
    Rank {
        load: NodeId,
    },
    LiteralAxis {
        load: NodeId,
        axis: usize,
        required: usize,
    },
    Extent(EntryExtentGuard),
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

/// Is this `ExtentWitness` retained purely as an ENTRY OBLIGATION - that is,
/// does no node ever read its VALUE?
///
/// `LowerCtx::retain_invocation_witnesses` keeps a claim-bearing witness alive
/// by listing it in a `Copy` carrier's `shape_deps`, never by feeding it to
/// anything, so a witness minted for a declared-but-unread parameter or for a
/// binder repeated across parameters carries an obligation without carrying a
/// number anyone computes with. Section 4.7 discharges such an obligation at
/// function entry, which is host work on every target.
///
/// The one data edge that does NOT count is a claim requirement: a witness's
/// inputs after the first are the earlier witnesses its named claims compare
/// against, and reading the requirement is part of discharging the same entry
/// obligation rather than a device computation.
///
/// A witness that IS read - the runtime `shape` value read of
/// `insert(b, 0, shape(y, 0))` - is not an entry obligation, and the HIP
/// target still refuses it under [05-SHAPE-1].
pub fn witness_is_entry_obligation(dag: &Dag, id: NodeId) -> bool {
    if !matches!(
        dag.get(id).map(|node| &node.op),
        Some(RiscOp::ExtentWitness {
            site: crate::dag::ExtentWitnessSite::Caller
                | crate::dag::ExtentWitnessSite::LocalExpand,
            ..
        })
    ) {
        // A producer-owned ResultClaim is not itself an entry witness. Target
        // adapters may still lower the complete producer relation to a host
        // guard when every observation is available from interface metadata.
        return false;
    }
    if dag.roots().contains(&id) {
        return false;
    }
    dag.nodes().iter().all(|node| {
        node.inputs.iter().enumerate().all(|(index, input)| {
            *input != id || (index > 0 && matches!(node.op, RiscOp::ExtentWitness { .. }))
        })
    })
}

/// One record inside a [04-NUM-9] extent diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtentRecord {
    /// `claimed = N`, the compile-time side of a literal requirement.
    Claimed(i64),
    /// `<parameter> axis <axis> = <runtime shape>`, read off an input tensor.
    Read {
        load: NodeId,
        axis: usize,
        parameter: String,
    },
}

/// One section 4.7 obligation carried by an entry-obligation `ExtentWitness`,
/// reduced to input reads so a host prologue can render it from ABI slots
/// alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessEntryObligation {
    /// The label between backticks: a binder for a named claim, the decimal
    /// extent for a literal requirement.
    pub label: String,
    /// The two records in [04-NUM-9] order, declaring side first.
    pub records: (ExtentRecord, ExtentRecord),
    /// The operation the `Domain` trap names: `load` or `expand`.
    pub operation: &'static str,
}

/// Every obligation an entry-obligation witness owes, or `None` when one of
/// them cannot be rendered from input reads.
///
/// `None` is the fail-closed answer. A target whose device lane cannot host a
/// witness ([05-SHAPE-1] on HIP) may discharge the witness in its host
/// prologue only when every obligation reduces to input tensors; otherwise it
/// refuses, exactly as it did before the witness carried claims.
///
/// Claims that [`entry_covered_witness_claims`] already names are omitted:
/// the entry schedule owns them on every lane, and section 4.7 evaluates each
/// guard exactly once.
pub fn witness_entry_obligations(
    dag: &Dag,
    witness: NodeId,
) -> Option<Vec<WitnessEntryObligation>> {
    let node = dag.get(witness)?;
    let RiscOp::ExtentWitness {
        site,
        requirements,
        claims,
        ..
    } = &node.op
    else {
        return None;
    };
    let operation = match site {
        crate::dag::ExtentWitnessSite::Caller => "load",
        crate::dag::ExtentWitnessSite::LocalExpand => "expand",
        crate::dag::ExtentWitnessSite::ResultClaim { .. }
        | crate::dag::ExtentWitnessSite::LiteralResultClaim
        | crate::dag::ExtentWitnessSite::LocalAscriptionClaim { .. } => return None,
    };
    let read_for = |id: NodeId| -> Option<ExtentRecord> {
        let observed = dag.get(id)?;
        let RiscOp::ExtentWitness {
            parameter,
            axis: RtAxis::Lit(axis),
            ..
        } = &observed.op
        else {
            return None;
        };
        let load = load_through_casts(dag, id, 0)?;
        abi_input_slot(dag, load)?;
        Some(ExtentRecord::Read {
            load,
            axis: usize::try_from(*axis).ok()?,
            parameter: parameter.clone(),
        })
    };
    let here = read_for(witness)?;
    let covered = entry_covered_witness_claims(dag);
    let mut obligations = Vec::new();
    for required in requirements {
        let required = required.as_i64_exact()?;
        obligations.push(WitnessEntryObligation {
            label: required.to_string(),
            records: (ExtentRecord::Claimed(required), here.clone()),
            operation,
        });
    }
    for (index, (claim, edge)) in claims.iter().zip(node.inputs.iter().skip(1)).enumerate() {
        if covered.contains(&(witness, index)) {
            continue;
        }
        let there = read_for(*edge)?;
        let records = if claim.requirement_declares {
            (there, here.clone())
        } else {
            (here.clone(), there)
        };
        obligations.push(WitnessEntryObligation {
            label: claim.claim.clone(),
            records,
            operation,
        });
    }
    Some(obligations)
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
    // An authored literal result whose extent is an exact parameter witness
    // is ready at that witness's signature position. The token identifies
    // the obligation; a matching physical dimension is not ownership proof.
    for (_, source, required) in literal_result_interface_claims(dag) {
        let Some(observed) = source.entry_axis(dag) else {
            continue;
        };
        guards.push(EntryExtentGuard::Literal {
            required: usize::try_from(required.as_i64_exact().expect("literal requirement"))
                .expect("nonnegative literal requirement"),
            observed,
        });
    }
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
        let DimClaim::Literal(required) = class.claim else {
            continue;
        };
        for member in &class.members {
            let observed = literal_result_interface_observation(dag, member.node, member.axis)
                .and_then(|observation| observation.entry_axis(dag))
                .or_else(|| {
                    // A body result cannot invent an input obligation. Keep
                    // the independent input check only when that input's own
                    // declaration already states this exact literal.
                    let (load, axis) = member_load_axis(dag, member)?;
                    let declared = dag.get(load)?.output_type.dims.get(axis)?;
                    matches!(declared,
                        DimInfo::Lit(value) | DimInfo::Named(_, Some(value))
                            if *value == required
                    )
                    .then_some((load, axis))
                });
            if let Some(observed) = observed {
                guards.push(EntryExtentGuard::Literal { required, observed });
            }
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
    // Physical dimension classes establish the comparison, while a typed
    // result claim identifies which observation declared its requirement.
    // A foreign producer can precede that declaring parameter. Preserve this
    // exact edge's diagnostic orientation without changing the comparison's
    // readiness position or joining any additional dimension classes.
    let witness_axis = |id: NodeId| {
        let node = dag.get(id)?;
        let RiscOp::ExtentWitness {
            site: crate::dag::ExtentWitnessSite::Caller,
            axis: RtAxis::Lit(axis),
            ..
        } = node.op
        else {
            return None;
        };
        Some((load_through_casts(dag, id, 0)?, usize::try_from(axis).ok()?))
    };
    for guard in &mut guards {
        let EntryExtentGuard::Named {
            claim,
            canonical,
            observed,
        } = guard
        else {
            continue;
        };
        let declaring_pair = dag.nodes().iter().find_map(|node| {
            let RiscOp::ExtentWitness { claims, .. } = &node.op else {
                return None;
            };
            let here = witness_axis(node.id)?;
            claims
                .iter()
                .zip(node.inputs.iter().skip(1))
                .find_map(|(recorded, edge)| {
                    let there = witness_axis(*edge)?;
                    (recorded.claim == *claim
                        && ((here == *canonical && there == *observed)
                            || (there == *canonical && here == *observed)))
                        .then_some(if recorded.requirement_declares {
                            (there, here)
                        } else {
                            (here, there)
                        })
                })
        });
        if let Some((declaring, producing)) = declaring_pair {
            *canonical = declaring;
            *observed = producing;
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

/// Validate interface inputs in their assigned ABI slot order (authored
/// signature order when one exists), then by axis within each input. A named
/// comparison is due at the later witness, exactly as `entry_extent_guards`
/// orders it; this plan interleaves that schedule with dtype/rank and literal
/// shape admission rather than regrouping checks by kind or label.
pub fn entry_validation_plan(dag: &Dag) -> Vec<EntryValidationStep> {
    entry_validation_plan_for_resolved_loads(dag, &[])
}

/// Eval can select a later occurrence of a name when an earlier Load belongs
/// only to an unselected root. Keep the name's ABI slot, but validate the
/// declaration that actually supplied the selected value.
pub fn entry_validation_plan_for_resolved_loads(
    dag: &Dag,
    resolved_loads: &[NodeId],
) -> Vec<EntryValidationStep> {
    let due = |guard: &EntryExtentGuard| match guard {
        EntryExtentGuard::Named {
            canonical,
            observed,
            ..
        } => {
            let position = |(load, axis): (NodeId, usize)| {
                (
                    abi_input_slot(dag, load).expect("entry witness is an input"),
                    axis,
                )
            };
            position(*canonical).max(position(*observed))
        }
        EntryExtentGuard::Literal { observed, .. } => (
            abi_input_slot(dag, observed.0).expect("entry witness is an input"),
            observed.1,
        ),
    };
    let guards = entry_extent_guards(dag);
    let mut steps = Vec::new();
    for (slot, first) in ordered_interface_loads(dag.nodes()).into_iter().enumerate() {
        let RiscOp::Load { name } = &first.op else {
            unreachable!()
        };
        let load = resolved_loads
            .iter()
            .filter_map(|id| dag.get(*id))
            .find(|node| matches!(&node.op, RiscOp::Load { name: resolved } if resolved == name))
            .unwrap_or(first);
        steps.push(EntryValidationStep::DType { load: load.id });
        steps.push(EntryValidationStep::Rank { load: load.id });
        for (axis, dim) in load.output_type.dims.iter().enumerate() {
            if let DimInfo::Lit(required) | DimInfo::Named(_, Some(required)) = dim
                && !guards.iter().any(|guard| {
                    matches!(guard, EntryExtentGuard::Literal { required: claim, observed }
                        if claim == required && *observed == (load.id, axis))
                })
            {
                steps.push(EntryValidationStep::LiteralAxis {
                    load: load.id,
                    axis,
                    required: *required,
                });
            }
            steps.extend(
                guards
                    .iter()
                    .filter(|guard| due(guard) == (slot, axis))
                    .cloned()
                    .map(EntryValidationStep::Extent),
            );
        }
        // A scalar interface witness has no tensor axis to enumerate. Its
        // due comparison still runs after the input's dtype and rank check.
        steps.extend(
            guards
                .iter()
                .filter(|guard| {
                    let (guard_slot, axis) = due(guard);
                    guard_slot == slot && axis >= load.output_type.dims.len()
                })
                .cloned()
                .map(EntryValidationStep::Extent),
        );
    }
    steps
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
    /// The exact declaring activation's rank-0 i64 observation.
    Witness(NodeId),
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
            CanonicalExtent::Witness(node) => write!(f, "extent witness {}", node.0),
        }
    }
}

/// How a consumer reads the extent a local guard observes.
///
/// The derivation states it, because C2.5 puts one answer to one question in
/// one place. The two kinds of local claim observe different quantities: an
/// equality class compares the extent an operation is ABOUT TO produce, read
/// from the carrier it was given, and a unit-extent claim compares the extent
/// its operand ALREADY produced, read from that operand's realized shape. A
/// consumer that re-derives which of those to read from the site's own `op`
/// can only get one of them right, which is exactly the divergence C2.5
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
    /// Compute the extent the site's node is ABOUT to produce, from that
    /// node's own bounds, before it runs.
    ///
    /// This is the third quantity, and it is neither of the first two. An
    /// [`AxisSource::OpComputed`] axis has no carrier: no `RtDim` on the
    /// operation states the extent, because the operation derives it from its
    /// own output-shape rule. [`Self::RealizedExtent`] does state it, but only
    /// after the node has run, and for an op-computed axis the node IS the
    /// allocation - C2.5 rejects that read in terms: "An extent computed by
    /// the operation must be computed/validated before its first
    /// shape-dependent allocation/access, not recovered from a tensor
    /// allocated using the unvalidated claim. Merely checking after a wrong
    /// allocation is not a conforming implementation of C1.3."
    ///
    /// So the derivation states HOW to compute it, once, and both lanes read
    /// that one answer rather than each asking the operation again.
    ComputedExtent(ComputedAxisExtent),
    /// Validate every distinct positive-rank operand shape, then read this
    /// axis from their one agreed shape.
    ///
    /// The relation is complete and nonempty by construction. It is observed
    /// before the producer runs: operand disagreement is reported first, and
    /// only an agreed result extent reaches the producer-owned claim.
    SameShapeAgreement(SameShapeAgreement),
    /// A malformed same-shape producer is retained as an explicit failed
    /// observation instead of degrading to `RealizedExtent` or disappearing.
    /// Verification rejects it before execution; raw evaluator entrypoints
    /// still return this reason if handed an unverified graph.
    MalformedSameShapeAgreement(String),
}

/// The extent an admitted [`AxisSource::OpComputed`] axis will produce,
/// expressed from the operation's own carriers.
///
/// One variant per admitted owner, because the arithmetic is the owner's own
/// output-shape rule and there is no rule shared across owners to factor out.
/// `spec/04-type-system.md` section 4.7's movement paragraph makes `shrink`
/// the owner whose symbolic axes "always mint fresh extents"; a `pad` with
/// non-zero padding and a `stride` with a non-unit step mint fresh extents
/// under the same sentence.
///
/// The admission test is whether the extent is readable BEFORE the owner
/// runs, which is what C2.5 requires of a computed-extent observation: "An
/// extent computed by the operation must be computed/validated before its
/// first shape-dependent allocation/access". Every admitted owner passes it
/// from their own carriers and the operand's realized shape, and both of
/// those are the owner's producers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputedAxisExtent {
    /// `shrink`'s half-open span on one axis: the extent is `end - start`
    /// (`spec/05-risc-primitives.md` section 2.4).
    ///
    /// `RtDim::ToEnd` resolves to the OPERAND's realized extent at
    /// `operand_axis`, which is readable before the shrink runs because the
    /// operand is one of the shrink's producers. `operand_axis` is carried
    /// rather than re-derived from the site key, so a consumer reading this
    /// variant needs nothing but the variant.
    ///
    /// A span whose `start` is not below its `end` computes no extent. The
    /// operation's own domain rejection owns that failure on both lanes and
    /// runs first (the C runtime's movement plan rejects it before the guard
    /// site is reached), so the guard yields nothing rather than comparing a
    /// fabricated number.
    ShrinkSpan {
        start: RtDim,
        end: RtDim,
        operand_axis: usize,
    },
    /// `pad`'s widened axis: the extent is the operand's extent at
    /// `operand_axis` plus `before` plus `after`
    /// (`spec/05-risc-primitives.md` section 2.4).
    ///
    /// Every `pad` bound is readable before the pad runs. `RtDim::ToEnd` is a
    /// `Shrink` `end` only, so a padding bound is a literal, a rank-0 integer
    /// input of this node, or a folded read of one of its inputs' shape
    /// metadata, and all three are this node's own producers. `operand_axis`
    /// is carried rather than re-derived, as [`Self::ShrinkSpan`] carries it,
    /// so a consumer reading this variant needs nothing but the variant;
    /// `pad` preserves rank, so it equals the output axis.
    ///
    /// chelis#1837 is why this owner is admitted. `concat` has no `RiscOp`:
    /// `tensor_concat_from_nodes` lowers a Pad+Add cascade, and the first Pad
    /// of that cascade is the origin of the result's concat axis. Without an
    /// admission here a declared concat-axis extent had no guard site on the
    /// DAG path, so `def main() -> tensor[100, 3, f32] = probe(...)` returned
    /// eight rows at exit zero on both host lanes.
    ///
    /// A sum that does not fit a host extent computes no extent. The owner's
    /// own allocation owns that failure, so the guard yields rather than
    /// comparing a wrapped number.
    ///
    /// There is deliberately NO `> 0` filter here, where [`Self::ShrinkSpan`]
    /// has one, and the asymmetry is the two quantities rather than an
    /// oversight. A shrink span of zero selects nothing and computes no
    /// extent, so the operation's own domain rejection owns it; a pad extent
    /// of zero is a real extent, and a claim of some other number over it is
    /// a mismatch the guard still owes. Filtering it would be a silent hole
    /// rather than parity.
    ///
    /// Measured, and the zero extent is REACHABLE, which is what decides it.
    /// A runtime bound resolving to zero is guarded correctly today:
    /// `pad(x, [[sub(shape(y, 0i32), shape(y, 0i32)), 1i64]], 0.0f32)` over
    /// three elements reports `pad axis 0 = 4`. A zero pad EXTENT is reached by
    /// giving that spelling both bounds and an empty operand, and it behaves
    /// correctly on both lanes: a declared `tensor[2, f32]` traps
    /// ``extent `2`: claimed = 2, pad axis 0 = 0``, and a declared
    /// `tensor[0, f32]` returns `shape=[0]` at exit zero. A `> 0` filter would
    /// silence the first of those, so it would introduce a defect rather than
    /// close one.
    ///
    /// An earlier version of this comment called the zero extent unreachable,
    /// on the strength of one spelling the checker refuses (`to_tensor([])`
    /// needs a resolved element dtype, which a typed parameter position
    /// supplies). Round 1's verification measured it; the decision is unchanged
    /// and its reason is now the measurement rather than an absence.
    PadSpan {
        before: RtDim,
        after: RtDim,
        operand_axis: usize,
    },
    /// `stride`'s narrowed axis: the extent is
    /// `ceil(operand_extent / step)` (`spec/05-risc-primitives.md` section
    /// 2.4.1).
    ///
    /// The signed runtime step is retained as its exact carrier so each lane
    /// can validate it before converting to an unsigned host extent or
    /// performing this division. A literal step of one is an identity and
    /// therefore never produces this variant; a runtime carrier remains
    /// op-computed even when its realized value is one.
    StrideSpan { step: RtDim, operand_axis: usize },
}

/// The op-computed origin a declared result axis reaches through
/// PASS-THROUGH hops alone, as `(node, axis)`.
///
/// This answers "which operation introduces the extent this axis carries",
/// which is the question `spec/04-type-system.md` section 4.7 asks when it
/// places a local guard at "the source position of the operation that
/// introduces the guarded extent". [`resolve_axis_extent`] answers a
/// different one, "where does this axis's VALUE come from", and the two
/// diverge on exactly the axes C2.4 separates: a set axis and a forwarded
/// axis both carry [`AxisSource::InputAxis`], and only the forwarded one
/// passes an extent through.
///
/// Measured, which is why the distinction is drawn here rather than left to
/// the value resolver. `expand(x, 0i32, shape(small, 0i32))` under a declared
/// `tensor[3, f32]`, with `small` a runtime `shrink`, has an `InputAxis`
/// source on the axis the `expand` SETS. The value resolver walks that hop
/// into the `shrink` and reports an op-computed origin there; a claim stamped
/// on it moves the guard off the `expand`, whose own axis is the class
/// witness [`is_member`] admits, and the trap renames itself from `expand` to
/// `shrink`. `issue_616_runtime_movement_c_parity`'s identity row measures
/// that rename.
///
/// So the walk crosses a hop only where [`is_member`] would refuse the axis
/// as a pass-through, and stops at the first axis the operation sets, whose
/// own claim is the one its guard checks. The walk is bounded by the node
/// count, so a malformed graph cannot spin.
pub fn op_computed_axis_origin(dag: &Dag, node: NodeId, axis: usize) -> Option<(NodeId, usize)> {
    op_computed_axis_origin_bounded(dag, node, axis, dag.nodes().len())
}

fn op_computed_axis_origin_bounded(
    dag: &Dag,
    node: NodeId,
    axis: usize,
    fuel: usize,
) -> Option<(NodeId, usize)> {
    if fuel == 0 {
        return None;
    }
    let owner = dag.get(node)?;
    if is_same_shape_result_op(&owner.op) {
        return None;
    }
    match output_axis_sources(dag, node).into_iter().nth(axis)? {
        AxisSource::OpComputed {
            op: origin,
            axis: computed,
        } => Some((origin, computed)),
        AxisSource::InputAxis {
            input,
            axis: RtAxis::Lit(read),
        } if !sets_axis(&owner.op, axis) => {
            let operand = *owner.inputs.get(input)?;
            op_computed_axis_origin_bounded(dag, operand, usize::try_from(read).ok()?, fuel - 1)
        }
        _ => None,
    }
}

/// The computed extent of `axis`, when `op` is an admitted op-computed owner.
///
/// ONE admission answer for two callers: [`local_dim_guard_sites`], which
/// turns it into a guard site, and lowering's declared-result stamp, which
/// may write a claim onto an op-computed result axis only where this function
/// produces the guard that enforces it. Splitting the two would let a claim
/// be stamped with no site to check it, which is a silently wrong shape on
/// the evaluator and the legacy movement failure on C.
pub fn op_computed_axis_extent(op: &RiscOp, axis: usize) -> Option<ComputedAxisExtent> {
    match op {
        RiscOp::Shrink { bounds } => {
            bounds
                .get(axis)
                .map(|(start, end)| ComputedAxisExtent::ShrinkSpan {
                    start: start.clone(),
                    end: end.clone(),
                    operand_axis: axis,
                })
        }
        RiscOp::Pad { padding, .. } => {
            padding
                .get(axis)
                .map(|(before, after)| ComputedAxisExtent::PadSpan {
                    before: before.clone(),
                    after: after.clone(),
                    operand_axis: axis,
                })
        }
        RiscOp::Stride { strides } => match strides.get(axis)? {
            RtDim::Lit(0 | 1) => None,
            RtDim::Lit(_) | RtDim::Node(_) => Some(ComputedAxisExtent::StrideSpan {
                step: strides[axis].clone(),
                operand_axis: axis,
            }),
            RtDim::ToEnd | RtDim::Sym(_) | RtDim::InputAxis { .. } => None,
        },
        _ => None,
    }
}

/// The extent an admitted op-computed axis produces, when every quantity its
/// output-shape rule reads is a compile-time constant.
///
/// `None` means the extent is a runtime value, which is the case
/// `spec/04-type-system.md` §4.7.2's guard exists for: it conditions the check
/// on a claim "that is not statically proven equal to `size`".
///
/// `Some(v)` means the operation's own rule PROVES the extent, so a
/// declaration claiming a different value is statically refuted rather than
/// runtime-checkable, and the graph cannot state it: [`crate::verify`]'s
/// per-owner static size check compares the same arithmetic and rejects an
/// output dim that disagrees. Measured, which is why this exists.
/// `concat([v, v], 0i32)` over four concrete rows under a declared
/// `tensor[100, 3, f32]` lowers to a Pad+Add cascade whose pads state
/// `Lit(8)`; stamping `Lit(100)` onto the origin produced
/// `pad at node 1: output axis 0 has size 100, expected 8` and
/// `binary op at node 3 has mismatched dimension at axis 0: Lit(100) vs
/// Lit(8)`, refusing the build on the C lane while the DAG evaluator, which
/// does not run the verifier, trapped at run time. A lane divergence with a
/// refused build on one side is not a repair.
///
/// So a statically refuted claim is not stamped: it is REJECTED. Lowering
/// does have an error channel - `LowerDiagnostic::fatal`, raised through
/// `lower::raise_fatal_lowering_error` and forwarded on both lanes by
/// `host::try_lower_compiled_program_with_lane_overrides` - and
/// `lower::LowerCtx::reject_refuted_result_axis` uses it, so a declaration
/// this function proves wrong ends the program before anything executes,
/// identically on eval and on C. That is the numbered spec's own division
/// reaching its conclusion rather than stopping short of one: §4.4 and §4.5
/// make a statically refuted declared result a type error and §4.7.2 makes an
/// unprovable one a runtime guard, and the checker reaches the first verdict
/// wherever it can see the extent. An extent that becomes literal only when a
/// call is inlined is one the checker cannot see, which is why the rejection
/// sits here as well (chelis#1837, chelis#1930).
pub fn static_op_computed_axis_extent(dag: &Dag, node: NodeId, axis: usize) -> Option<usize> {
    let owner = dag.get(node)?;
    let operand_extent = |operand_axis: usize| -> Option<usize> {
        let operand = *owner.inputs.first()?;
        match dag.get(operand)?.output_type.dims.get(operand_axis)? {
            DimInfo::Lit(size) | DimInfo::Named(_, Some(size)) => Some(*size),
            DimInfo::Named(_, None) => None,
        }
    };
    match op_computed_axis_extent(&owner.op, axis)? {
        ComputedAxisExtent::ShrinkSpan {
            start,
            end,
            operand_axis,
        } => {
            let extent = operand_extent(operand_axis);
            let start = static_bound(&start, extent)?;
            let end = static_bound(&end, extent)?;
            end.checked_sub(start)
        }
        ComputedAxisExtent::PadSpan {
            before,
            after,
            operand_axis,
        } => {
            let extent = operand_extent(operand_axis)?;
            let before = static_bound(&before, Some(extent))?;
            let after = static_bound(&after, Some(extent))?;
            extent.checked_add(before)?.checked_add(after)
        }
        ComputedAxisExtent::StrideSpan { step, operand_axis } => {
            let extent = operand_extent(operand_axis)?;
            let step = static_bound(&step, Some(extent))?;
            (step > 0).then(|| extent.div_ceil(step))
        }
    }
}

/// A movement bound's compile-time value, or `None` when only run time has it.
///
/// `RtDim::ToEnd` resolves to the operand's extent, which is a compile-time
/// constant exactly when that extent is. Every other runtime carrier - a
/// rank-0 node, a folded shape read, a symbol bound at entry - is a value no
/// compile-time arithmetic has.
fn static_bound(bound: &RtDim, operand_extent: Option<usize>) -> Option<usize> {
    match bound {
        RtDim::Lit(value) => Some(*value),
        RtDim::ToEnd => operand_extent,
        RtDim::Node(_) | RtDim::Sym(_) | RtDim::InputAxis { .. } => None,
    }
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
    /// Runtime branch activation for a path-local authored ascription.
    ///
    /// The lowering owner carries this scalar Bool beside its claim token as
    /// a non-value dependency. `None` is the ordinary unconditional guard.
    pub activation: Option<NodeId>,
}

/// A returned axis and the operation where its inherited claim becomes ready.
/// This is derived from the current graph, never serialized or attached to a
/// tensor's physical shape. Each caller supplies its own literal obligations.
#[derive(Clone, Debug)]
pub struct ResultExtentSite {
    output_axis: RtAxis,
    producer: NodeId,
    producer_axis: RtAxis,
    observation: LocalGuardObservation,
    runtime_carrier_origin: Option<(NodeId, RtAxis)>,
    operation: &'static str,
}

impl ResultExtentSite {
    pub fn output_axis(&self) -> RtAxis {
        self.output_axis
    }
    pub fn producer(&self) -> NodeId {
        self.producer
    }
    pub fn producer_axis(&self) -> RtAxis {
        self.producer_axis
    }
    pub fn observation(&self) -> &LocalGuardObservation {
        &self.observation
    }
    /// The operation/axis that introduced a runtime extent carrier, before
    /// any later forwarding. A pure pass-through from an input has none.
    pub fn runtime_carrier_origin(&self) -> Option<(NodeId, RtAxis)> {
        self.runtime_carrier_origin
    }
    pub fn operation(&self) -> &'static str {
        self.operation
    }
}

/// Follow the same typed axis relation used for observation. Stop at an
/// introducing carrier instead of resolving through it to input metadata:
/// the carrier's producer owns its declared result even when its size is
/// available at entry. No new DAG annotation or execution dependency is made.
fn runtime_carrier_origin(
    dag: &Dag,
    mut node: NodeId,
    mut axis: usize,
) -> Option<(NodeId, RtAxis)> {
    for _ in 0..dag.len() {
        let producer = dag.get(node)?;
        if expand_or_reshape_carrier(&producer.op, axis).is_some() {
            return Some((node, RtAxis::Lit(i32::try_from(axis).ok()?)));
        }
        match output_axis_sources(dag, node).get(axis)? {
            AxisSource::InputAxis {
                input,
                axis: RtAxis::Lit(read),
            } => {
                node = *producer.inputs.get(*input)?;
                axis = usize::try_from(*read).ok()?;
            }
            _ => return None,
        }
    }
    None
}

/// Locate inherited result checks without changing graph or execution identity.
/// A graph's own declaration guards remain separate obligations.
pub fn result_extent_sites(dag: &Dag, root: NodeId) -> Vec<ResultExtentSite> {
    let Some(result) = dag.get(root) else {
        return Vec::new();
    };
    (0..result.output_type.dims.len())
        .map(|output_axis| {
            // Copy is administrative; an actual conversion still owns the
            // returned value's diagnostic. §4.7 gives a cast the placement of
            // its input, independently of that primitive attribution.
            let mut attributed = root;
            while let Some(node) = dag.get(attributed) {
                if !matches!(node.op, RiscOp::Copy) {
                    break;
                }
                let Some(input) = node.inputs.first() else {
                    break;
                };
                attributed = *input;
            }
            let mut producer = attributed;
            while let Some(node) = dag.get(producer) {
                if !matches!(
                    node.op,
                    RiscOp::Copy | RiscOp::Cast { .. } | RiscOp::CastTrunc { .. }
                ) {
                    break;
                }
                let Some(input) = node.inputs.first() else {
                    break;
                };
                producer = *input;
            }
            let axis = output_axis;
            let node = dag.get(producer).expect("result source belongs to graph");
            let attributed_node = dag
                .get(attributed)
                .expect("result producer belongs to graph");
            let observation = match same_shape_result_agreement(dag, producer) {
                Ok(Some(agreement)) => LocalGuardObservation::SameShapeAgreement(agreement),
                Err(reason) => LocalGuardObservation::MalformedSameShapeAgreement(reason),
                Ok(None) => {
                    if let Some(carrier) = expand_or_reshape_carrier(&node.op, axis) {
                        LocalGuardObservation::Carrier(carrier.clone())
                    } else if let Some(computed) = op_computed_axis_extent(&node.op, axis) {
                        LocalGuardObservation::ComputedExtent(computed)
                    } else if let Some(AxisSource::InputAxis { input, axis }) =
                        output_axis_sources(dag, producer).get(axis)
                    {
                        LocalGuardObservation::Carrier(RtDim::InputAxis {
                            tensor: *input,
                            axis: *axis,
                        })
                    } else {
                        LocalGuardObservation::RealizedExtent
                    }
                }
            };
            ResultExtentSite {
                output_axis: RtAxis::Lit(i32::try_from(output_axis).expect("rank fits i32")),
                producer,
                producer_axis: RtAxis::Lit(i32::try_from(axis).expect("rank fits i32")),
                observation,
                runtime_carrier_origin: runtime_carrier_origin(dag, producer, axis),
                operation: expansion_kind(dag, attributed).map_or_else(
                    || crate::grad::risc_op_name(&attributed_node.op),
                    ExpansionKind::primitive_name,
                ),
            }
        })
        .collect()
}

/// Check the exact axis relationship before following administrative edges.
/// Raw evaluator graphs can reach this derivation before ownership verification;
/// a missing axis, missing input or cyclic alias must be an error, not a panic.
fn checked_result_extent_site(
    dag: &Dag,
    root: NodeId,
    axis: usize,
) -> Result<ResultExtentSite, String> {
    let invalid = || format!("missing or invalid producer axis {axis} at node {}", root.0);
    let mut current = root;
    loop {
        let node = dag.get(current).ok_or_else(invalid)?;
        if axis >= node.output_type.dims.len() {
            return Err(invalid());
        }
        if !matches!(
            node.op,
            RiscOp::Copy | RiscOp::Cast { .. } | RiscOp::CastTrunc { .. }
        ) {
            break;
        }
        let [input] = node.inputs.as_slice() else {
            return Err(invalid());
        };
        if input.0 >= current.0 {
            return Err(invalid());
        }
        current = *input;
    }
    let output_axis = i32::try_from(axis).map_err(|_| invalid())?;
    result_extent_sites(dag, root)
        .into_iter()
        .find(|site| site.output_axis == RtAxis::Lit(output_axis))
        .ok_or_else(invalid)
}

/// Nodes any producer-owned extent claim requires semantic rewrites to preserve.
///
/// Named `ResultClaim`, authored `LiteralResultClaim`, and local
/// `LocalAscriptionClaim` tokens make their owner potentially trapping.
/// Claims can be attached to an administrative Copy/Cast carrier while §4.7
/// attribution remains at the primitive behind that carrier, so the protected
/// set includes that complete administrative chain.
///
/// Every rewrite that can replace, merge, fold, fuse, specialize, or eliminate
/// a producer consumes this one classification. Keeping the token forms behind
/// this API prevents a new rewrite from accidentally protecting only one.
pub(crate) fn claimed_producers(dag: &Dag) -> Vec<bool> {
    claim_producers_matching(dag, directly_owns_producer_claim)
}

/// Literal-only ownership carriers are a lowering representation detail, not
/// the semantic rewrite barrier. Sparse helper recognition uses this narrower
/// query only to peel those exact administrative carriers.
pub(crate) fn literal_result_claim_producers(dag: &Dag) -> Vec<bool> {
    claim_producers_matching(dag, directly_owns_literal_result_claim)
}

fn claim_producers_matching(dag: &Dag, directly_owns: impl Fn(&Dag, NodeId) -> bool) -> Vec<bool> {
    let mut protected = vec![false; dag.len()];
    let mut pending = dag
        .nodes()
        .iter()
        .filter(|owner| directly_owns(dag, owner.id))
        .map(|owner| owner.id)
        .collect::<Vec<_>>();

    while let Some(current) = pending.pop() {
        let Some(slot) = protected.get_mut(current.0) else {
            continue;
        };
        if *slot {
            continue;
        }
        *slot = true;
        let Some(node) = dag.get(current) else {
            continue;
        };
        if matches!(
            node.op,
            RiscOp::Copy | RiscOp::Cast { .. } | RiscOp::CastTrunc { .. }
        ) && let Some(input) = node.inputs.first()
        {
            pending.push(*input);
        }
    }

    protected
}

pub(crate) fn directly_owns_producer_claim(dag: &Dag, owner: NodeId) -> bool {
    dag.get(owner).is_some_and(|node| {
        node.shape_deps
            .iter()
            .chain(&node.result_claim_deps)
            .any(|dependency| {
                matches!(
                    dag.get(*dependency).map(|node| &node.op),
                    Some(RiscOp::ExtentWitness {
                        site: crate::dag::ExtentWitnessSite::ResultClaim { .. }
                            | crate::dag::ExtentWitnessSite::LiteralResultClaim
                            | crate::dag::ExtentWitnessSite::LocalAscriptionClaim { .. },
                        ..
                    })
                )
            })
    })
}

pub(crate) fn directly_owns_literal_result_claim(dag: &Dag, owner: NodeId) -> bool {
    dag.get(owner).is_some_and(|node| {
        node.shape_deps
            .iter()
            .chain(&node.result_claim_deps)
            .any(|dependency| {
                matches!(
                    dag.get(*dependency).map(|node| &node.op),
                    Some(RiscOp::ExtentWitness {
                        site: crate::dag::ExtentWitnessSite::LiteralResultClaim,
                        ..
                    })
                )
            })
    })
}

/// The latest `Caller` witness at or before `not_after` that reads this exact
/// tensor axis.
///
/// A returned value can be passed to another invocation later in the DAG.
/// Such a downstream witness observes the same physical quantity but does not
/// own the earlier result claim, so the upper bound is part of the identity.
fn caller_witness_for_axis(
    dag: &Dag,
    tensor: NodeId,
    axis: usize,
    not_after: NodeId,
) -> Option<NodeId> {
    dag.nodes().iter().rev().find_map(|node| {
        let RiscOp::ExtentWitness {
            site: crate::dag::ExtentWitnessSite::Caller,
            axis: RtAxis::Lit(observed),
            ..
        } = node.op
        else {
            return None;
        };
        (node.id.0 <= not_after.0
            && usize::try_from(observed).ok() == Some(axis)
            && node.inputs.first() == Some(&tensor))
        .then_some(node.id)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiteralResultInterfaceObservation {
    Witness(NodeId),
    InputAxis { load: NodeId, axis: usize },
}

impl LiteralResultInterfaceObservation {
    fn entry_axis(self, dag: &Dag) -> Option<(NodeId, usize)> {
        match self {
            Self::Witness(witness) => {
                let node = dag.get(witness)?;
                let RiscOp::ExtentWitness { parameter, .. } = &node.op else {
                    return None;
                };
                let observed = interface_witness_axis(dag, witness)?;
                let RiscOp::Load { name } = &dag.get(observed.0)?.op else {
                    return None;
                };
                (parameter == name.as_str()).then_some(observed)
            }
            Self::InputAxis { load, axis } => {
                abi_input_slot(dag, load)?;
                Some((load, axis))
            }
        }
    }
}

/// The exact parameter observation supplying a returned axis, when the
/// output shape directly reads it.
///
/// This intentionally follows the one-hop [`AxisSource`] path instead of the
/// terminal [`ExtentOrigin`]. Resolving to an origin loses which invocation's
/// witness supplied an `InputAxis`, which changed alias diagnostics and could
/// select the wrong parameter when two inputs had the same runtime extent.
/// Only an actual parameter or its administrative copy/cast is an interface
/// result. A body operation owns its result claim even when its extent can be
/// read from parameter metadata before that operation runs (spec/04 §4.7).
fn literal_result_interface_observation(
    dag: &Dag,
    owner: NodeId,
    axis: usize,
) -> Option<LiteralResultInterfaceObservation> {
    fn walk(
        dag: &Dag,
        node_id: NodeId,
        axis: usize,
        not_after: NodeId,
        remaining: usize,
    ) -> Option<LiteralResultInterfaceObservation> {
        if remaining == 0 {
            return None;
        }
        if let Some(witness) = caller_witness_for_axis(dag, node_id, axis, not_after) {
            return Some(LiteralResultInterfaceObservation::Witness(witness));
        }
        let node = dag.get(node_id)?;
        match output_axis_sources(dag, node_id).get(axis)? {
            AxisSource::ExternalAxis { load, axis } => {
                caller_witness_for_axis(dag, *load, *axis, not_after)
                    .map(LiteralResultInterfaceObservation::Witness)
                    .or(Some(LiteralResultInterfaceObservation::InputAxis {
                        load: *load,
                        axis: *axis,
                    }))
            }
            AxisSource::InputAxis {
                input,
                axis: RtAxis::Lit(read),
            } if matches!(
                node.op,
                RiscOp::Copy | RiscOp::Cast { .. } | RiscOp::CastTrunc { .. }
            ) =>
            {
                walk(
                    dag,
                    *node.inputs.get(*input)?,
                    usize::try_from(*read).ok()?,
                    not_after,
                    remaining - 1,
                )
            }
            AxisSource::InputAxis { .. } | AxisSource::ScalarInput { .. } => None,
            AxisSource::Literal { .. }
            | AxisSource::OpComputed { .. }
            | AxisSource::ClassSupplied { .. } => None,
        }
    }

    walk(dag, owner, axis, owner, dag.len())
}

fn interface_witness_axis(dag: &Dag, witness: NodeId) -> Option<(NodeId, usize)> {
    let node = dag.get(witness)?;
    let RiscOp::ExtentWitness {
        axis: RtAxis::Lit(axis),
        ..
    } = node.op
    else {
        return None;
    };
    let load = load_through_casts(dag, witness, 0)?;
    abi_input_slot(dag, load)?;
    Some((load, usize::try_from(axis).ok()?))
}

/// A named equality can discharge a literal restatement only when its exact
/// other observation is fixed by the graph. An ABI shape promise is still
/// an obligation, not such a proof (the #1782 distinction).
fn literal_result_is_entailed(dag: &Dag, witness: NodeId, required: i64) -> bool {
    let fixed = |id: NodeId| {
        let node = dag.get(id)?;
        let RiscOp::ExtentWitness {
            axis: RtAxis::Lit(axis),
            ..
        } = node.op
        else {
            return None;
        };
        match resolve_axis_extent(dag, *node.inputs.first()?, usize::try_from(axis).ok()?)? {
            ExtentOrigin::Literal(value) => Some(value),
            _ => None,
        }
    };
    dag.nodes().iter().any(|node| {
        let RiscOp::ExtentWitness { claims, .. } = &node.op else {
            return false;
        };
        claims
            .iter()
            .zip(node.inputs.iter().skip(1))
            .any(|(_, edge)| {
                (node.id == witness && fixed(*edge) == Some(required))
                    || (*edge == witness && fixed(node.id) == Some(required))
            })
    })
}

fn literal_result_interface_claims(
    dag: &Dag,
) -> Vec<(
    NodeId,
    LiteralResultInterfaceObservation,
    chelis_types::ScalarValue,
)> {
    let mut claims = Vec::new();
    for owner in dag.nodes() {
        for token in owner.shape_deps.iter().chain(&owner.result_claim_deps) {
            let Some(crate::dag::DagNode {
                op:
                    RiscOp::ExtentWitness {
                        site: crate::dag::ExtentWitnessSite::LiteralResultClaim,
                        axis: RtAxis::Lit(axis),
                        requirements,
                        ..
                    },
                ..
            }) = dag.get(*token)
            else {
                continue;
            };
            let Some(observed) =
                literal_result_interface_observation(dag, owner.id, *axis as usize)
            else {
                continue;
            };
            let required = requirements[0];
            let entailed = match observed {
                LiteralResultInterfaceObservation::Witness(witness) => literal_result_is_entailed(
                    dag,
                    witness,
                    required.as_i64_exact().expect("literal requirement"),
                ),
                LiteralResultInterfaceObservation::InputAxis { .. } => false,
            };
            if !entailed {
                claims.push((*token, observed, required));
            }
        }
    }
    claims
}

/// Literal result obligations executed at an invocation's exact parameter
/// witness.
///
/// A witness retains the callee parameter identity even when it reads an ABI
/// `Load`: two calls can pass different outer parameters through the same
/// inner name. Collapsing it into the outer entry schedule loses that identity
/// and also moves an invocation-local obligation ahead of the call that owns
/// it. Only a raw input-axis observation with no witness enters the common
/// entry schedule above.
pub fn literal_result_witness_requirements(
    dag: &Dag,
    witness: NodeId,
) -> Vec<chelis_types::ScalarValue> {
    literal_result_interface_claims(dag)
        .into_iter()
        .filter_map(|(_, observed, required)| {
            (observed == LiteralResultInterfaceObservation::Witness(witness)
                && observed.entry_axis(dag).is_none())
            .then_some(required)
        })
        .collect()
}

/// C1.3's local guard sites: `(node id, axis)` paired with the claim each
/// site guards against.
///
/// A `Local` class's guard "takes the source position of the operation that
/// introduces the guarded extent" (`spec/04-type-system.md` section 4.7), so
/// unlike the entry classes these are keyed by node.
///
/// Two lanes read this one function, which is what C2.5's single derivation
/// point means for a local guard: the C emitter places its guard at the
/// operation it names, and the DAG evaluator checks the same site when that
/// node produces its value. A second answer computed in either lane could
/// disagree with the first, and the point of deriving it here is that it
/// cannot.
pub fn local_dim_guard_sites(dag: &Dag) -> Result<Vec<(LocalGuardSite, LocalGuardClaim)>, String> {
    let mut sites = Vec::new();
    // Explicit result obligations have graph identity, independently of any
    // labels the checked caller retains on its result. Their producer owns
    // the observation and their dependency owns the canonical value.
    for node in dag.nodes() {
        for required in node.shape_deps.iter().chain(&node.result_claim_deps) {
            if let Some(crate::dag::DagNode {
                op:
                    RiscOp::ExtentWitness {
                        site:
                            crate::dag::ExtentWitnessSite::LocalAscriptionClaim {
                                claim,
                                axis: RtAxis::Lit(axis),
                                ..
                            },
                        ..
                    },
                ..
            }) = dag.get(*required)
            {
                let axis = usize::try_from(*axis)
                    .map_err(|_| format!("invalid producer axis {axis} at node {}", node.id.0))?;
                let site = checked_result_extent_site(dag, node.id, axis)?;
                let (producer, observed) = if required.0 < site.producer.0 {
                    (site.producer, site.observation)
                } else {
                    (
                        node.id,
                        LocalGuardObservation::Carrier(RtDim::InputAxis {
                            tensor: 0,
                            axis: RtAxis::Lit(axis as i32),
                        }),
                    )
                };
                sites.push((
                    (producer.0, axis),
                    LocalGuardClaim {
                        claim: claim.clone(),
                        canonical: CanonicalExtent::Witness(*required),
                        op: site.operation,
                        observed,
                        activation: local_ascription_guard_activation(dag, node.id, *required)?,
                    },
                ));
                continue;
            }
            if let Some(crate::dag::DagNode {
                op:
                    RiscOp::ExtentWitness {
                        site: crate::dag::ExtentWitnessSite::LiteralResultClaim,
                        axis: RtAxis::Lit(axis),
                        requirements,
                        ..
                    },
                ..
            }) = dag.get(*required)
            {
                let axis = usize::try_from(*axis)
                    .map_err(|_| format!("invalid producer axis {axis} at node {}", node.id.0))?;
                if literal_result_interface_observation(dag, node.id, axis).is_some() {
                    // This exact token is an entry obligation or a proven
                    // named restatement, never a second local producer guard.
                    continue;
                }
                let site = checked_result_extent_site(dag, node.id, axis)?;
                // A claim captured after an existing value's production belongs
                // to this invocation's carrier; never add a backward dependency.
                let (producer, observed) = if required.0 < site.producer.0 {
                    (site.producer, site.observation)
                } else {
                    (
                        node.id,
                        LocalGuardObservation::Carrier(RtDim::InputAxis {
                            tensor: 0,
                            axis: RtAxis::Lit(axis as i32),
                        }),
                    )
                };
                let literal = requirements[0]
                    .as_i64_exact()
                    .expect("verified literal requirement");
                sites.push((
                    (producer.0, axis),
                    LocalGuardClaim {
                        claim: literal.to_string(),
                        canonical: CanonicalExtent::Witness(*required),
                        op: site.operation,
                        observed,
                        activation: None,
                    },
                ));
                continue;
            }
            let Some(crate::dag::DagNode {
                op:
                    RiscOp::ExtentWitness {
                        site:
                            crate::dag::ExtentWitnessSite::ResultClaim {
                                claim,
                                axis: RtAxis::Lit(axis),
                            },
                        ..
                    },
                ..
            }) = dag.get(*required)
            else {
                continue;
            };
            let axis = usize::try_from(*axis)
                .map_err(|_| format!("invalid producer axis {axis} at node {}", node.id.0))?;
            let site = checked_result_extent_site(dag, node.id, axis)?;
            let RtAxis::Lit(producer_axis) = site.producer_axis;
            let producer_axis =
                usize::try_from(producer_axis).expect("verified result producer axis");
            sites.push((
                (site.producer.0, producer_axis),
                LocalGuardClaim {
                    claim: claim.clone(),
                    canonical: CanonicalExtent::Witness(*required),
                    op: site.operation,
                    observed: site.observation,
                    activation: None,
                },
            ));
        }
    }
    // A physical literal annotation can restate an explicit token. Coalesce
    // only the exact comparison at the same producer and observation; another
    // required value or a named obligation remains independent.
    let explicit_literals = sites
        .iter()
        .filter_map(|(site, claim)| {
            let CanonicalExtent::Witness(witness) = claim.canonical else {
                return None;
            };
            let RiscOp::ExtentWitness {
                site: crate::dag::ExtentWitnessSite::LiteralResultClaim,
                requirements,
                ..
            } = &dag.get(witness)?.op
            else {
                return None;
            };
            Some((
                *site,
                claim.observed.clone(),
                requirements.first()?.as_i64_exact()?,
            ))
        })
        .collect::<Vec<_>>();
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
            if let DimClaim::Literal(value) = class.claim {
                let site = checked_result_extent_site(dag, member.node, member.axis)?;
                if explicit_literals
                    .iter()
                    .any(|(key, observation, required)| {
                        *key == (site.producer.0, member.axis)
                            && *observation == site.observation
                            && usize::try_from(*required).ok() == Some(value)
                    })
                {
                    continue;
                }
            }

            // Captured named tokens and these physical/literal contracts
            // are additive. One agreeing requirement cannot discharge another.
            // Only the independent extent source can discharge a claim.
            // Runtime carriers remain observable even when result metadata
            // contains a number. Literal-source proofs were handled by
            // is_member; unsupported observation kinds remain outside this
            // carrier consumer.
            // An op-computed axis is the third observation kind, and it is
            // admitted HERE rather than by widening the carrier filter below,
            // because it has no carrier: the operation derives the extent from
            // its own output-shape rule. [`op_computed_axis_extent`] is the one
            // admission answer, shared with lowering's declared-result stamp so
            // a claim cannot be stamped onto an axis with no site to check it.
            if let AxisSource::OpComputed { axis: computed, .. } = &member.source {
                let Some(node) = dag.get(member.node) else {
                    continue;
                };
                // `OpComputed` names this node and this axis (C4.1 rejects any
                // other pairing), so the member's axis IS the axis to compute.
                let Some(observed) = op_computed_axis_extent(&node.op, *computed) else {
                    // An unadmitted owner - `pad` with non-zero padding,
                    // `stride` with a non-unit step, a `reshape` binding a
                    // symbol it does not compute - keeps the disposition it
                    // had: no site before this arm existed and none after.
                    continue;
                };
                sites.push((
                    (member.node.0, member.axis),
                    LocalGuardClaim {
                        claim: name.clone(),
                        canonical: match resolved {
                            Some(value) => CanonicalExtent::Resolved(value),
                            None => CanonicalExtent::Binder(name.clone()),
                        },
                        op: crate::grad::risc_op_name(&node.op),
                        observed: LocalGuardObservation::ComputedExtent(observed),
                        activation: None,
                    },
                ));
                continue;
            }
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
                // OpComputed sources are admitted by the arm above, which
                // supplies the independent observation these carriers cannot.
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
                    activation: None,
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
    // It is derived HERE rather than in the ownership view because C2.5 puts
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
                activation: None,
            },
        ));
    }
    Ok(sites)
}

/// The one scalar Bool dependency that activates a path-local ascription.
///
/// Claim tokens and ordinary shape sources share `shape_deps`; the activation
/// is structurally distinct because it is rank-0 Bool. More than one such
/// dependency is ambiguous and therefore malformed rather than ordered or
/// guessed.
pub(crate) fn local_ascription_guard_activation(
    dag: &Dag,
    owner: NodeId,
    claim: NodeId,
) -> Result<Option<NodeId>, String> {
    let owner = dag
        .get(owner)
        .ok_or_else(|| "local ascription owner is missing".to_string())?;
    let activations = owner
        .shape_deps
        .iter()
        .copied()
        .filter(|dependency| *dependency != claim)
        .filter(|dependency| {
            dag.get(*dependency).is_some_and(|node| {
                node.output_type.dims.is_empty() && node.output_type.precision == Prim::Bool
            })
        })
        .collect::<Vec<_>>();
    match activations.as_slice() {
        [] => Ok(None),
        [activation] => Ok(Some(*activation)),
        _ => Err(format!(
            "local ascription owner {} has multiple runtime branch activations",
            owner.id.0
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::TensorType;

    fn ty(dims: Vec<DimInfo>, precision: Prim) -> TensorType {
        TensorType { dims, precision }
    }

    #[test]
    fn claimed_producers_cover_named_literal_local_and_the_owned_admin_chain() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let tensor = ty(vec![DimInfo::Lit(4)], Prim::F32);
        let input = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            tensor.clone(),
            None,
        );
        let producer = dag.add_node(decl, RiscOp::Mul, vec![input, input], tensor.clone(), None);
        let cast = dag.add_node(
            decl,
            RiscOp::Cast {
                new_precision: Prim::F32,
            },
            vec![producer],
            tensor.clone(),
            None,
        );
        let claim = dag.add_node(
            decl,
            RiscOp::ExtentWitness {
                site: crate::dag::ExtentWitnessSite::LiteralResultClaim,
                parameter: String::new(),
                axis: RtAxis::Lit(0),
                requirements: vec![chelis_types::scalar_from_i64("test", Prim::Int64, 4).unwrap()],
                claims: Vec::new(),
            },
            Vec::new(),
            ty(Vec::new(), Prim::Int64),
            None,
        );
        let owner = dag.add_node(decl, RiscOp::Copy, vec![cast], tensor.clone(), None);
        dag.add_shape_dep(owner, claim);
        let named_claim = dag.add_node(
            decl,
            RiscOp::ExtentWitness {
                site: crate::dag::ExtentWitnessSite::ResultClaim {
                    claim: "n".into(),
                    axis: RtAxis::Lit(0),
                },
                parameter: "x".into(),
                axis: RtAxis::Lit(0),
                requirements: Vec::new(),
                claims: Vec::new(),
            },
            vec![input],
            ty(Vec::new(), Prim::Int64),
            None,
        );
        let named_owner = dag.add_node(decl, RiscOp::Add, vec![input, input], tensor.clone(), None);
        dag.add_result_claim_dep(named_owner, named_claim);
        let local_claim = dag.add_node(
            decl,
            RiscOp::ExtentWitness {
                site: crate::dag::ExtentWitnessSite::LocalAscriptionClaim {
                    ascription_id: 0,
                    binding: "local".into(),
                    claim: "4".into(),
                    axis: RtAxis::Lit(0),
                },
                parameter: String::new(),
                axis: RtAxis::Lit(0),
                requirements: vec![chelis_types::scalar_from_i64("test", Prim::Int64, 4).unwrap()],
                claims: Vec::new(),
            },
            Vec::new(),
            ty(Vec::new(), Prim::Int64),
            None,
        );
        let local_owner = dag.add_node(decl, RiscOp::Exp, vec![input], tensor.clone(), None);
        dag.add_shape_dep(local_owner, local_claim);
        let unrelated = dag.add_node(decl, RiscOp::Copy, vec![input], tensor, None);

        let protected = claimed_producers(&dag);
        assert!(protected[owner.0]);
        assert!(protected[cast.0]);
        assert!(protected[producer.0]);
        assert!(protected[named_owner.0]);
        assert!(protected[local_owner.0]);
        assert!(!protected[input.0]);
        assert!(!protected[claim.0]);
        assert!(!protected[named_claim.0]);
        assert!(!protected[local_claim.0]);
        assert!(!protected[unrelated.0]);
    }

    #[test]
    fn a_sym_reshape_target_is_the_operations_own_extent() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let operand = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![DimInfo::Named("n".into(), None)], Prim::F32),
            None,
        );
        let reshaped = dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let operand = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![DimInfo::Lit(2), DimInfo::Lit(3)], Prim::F32),
            None,
        );
        let permuted = dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let values = dag.add_node(
            decl,
            RiscOp::Load { name: "v".into() },
            vec![],
            ty(
                vec![DimInfo::Lit(2), DimInfo::Lit(5), DimInfo::Lit(7)],
                Prim::F32,
            ),
            None,
        );
        let indices = dag.add_node(
            decl,
            RiscOp::Load { name: "i".into() },
            vec![],
            ty(vec![DimInfo::Lit(4)], Prim::Int64),
            None,
        );
        let gathered = dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let lhs = dag.add_node(
            decl,
            RiscOp::Load { name: "a".into() },
            vec![],
            ty(
                vec![DimInfo::Lit(2), DimInfo::Lit(4), DimInfo::Lit(3)],
                Prim::F32,
            ),
            None,
        );
        let rhs = dag.add_node(
            decl,
            RiscOp::Load { name: "b".into() },
            vec![],
            ty(
                vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(5)],
                Prim::F32,
            ),
            None,
        );
        let product = dag.add_node(
            decl,
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
        let flat_decl = flat.declare("test");
        let lhs = flat.add_node(
            flat_decl,
            RiscOp::Load { name: "a".into() },
            vec![],
            ty(vec![DimInfo::Lit(4), DimInfo::Lit(3)], Prim::F32),
            None,
        );
        let rhs = flat.add_node(
            flat_decl,
            RiscOp::Load { name: "b".into() },
            vec![],
            ty(vec![DimInfo::Lit(3), DimInfo::Lit(5)], Prim::F32),
            None,
        );
        let product = flat.add_node(
            flat_decl,
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
        let decl = dag.declare("test");
        let sibling = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![DimInfo::Named("n".into(), None)], Prim::F32),
            None,
        );
        let mask = dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let operand = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![DimInfo::Lit(2)], Prim::F32),
            None,
        );
        let negated = dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let source = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![DimInfo::Lit(2)], Prim::F32),
            None,
        );
        let negated = dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let source = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![DimInfo::Lit(2)], Prim::F32),
            None,
        );
        let negated = dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let value = dag.add_node(
            decl,
            RiscOp::Load {
                name: "value".into(),
            },
            vec![],
            ty(vec![], Prim::F32),
            None,
        );
        let extent = dag.add_node(
            decl,
            RiscOp::Load {
                name: "extent".into(),
            },
            vec![],
            ty(vec![], Prim::Int64),
            None,
        );
        let tensor = dag.add_node(
            decl,
            RiscOp::Load {
                name: "tensor".into(),
            },
            vec![],
            ty(vec![DimInfo::Lit(2)], Prim::Int64),
            None,
        );
        let expanded = dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let source = dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let operand = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![DimInfo::Lit(2), DimInfo::Lit(3)], Prim::F32),
            None,
        );
        let windowed = dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let forward_input = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![DimInfo::Lit(2), DimInfo::Lit(3)], Prim::F32),
            None,
        );
        let cotangent = dag.add_node(
            decl,
            RiscOp::Load { name: "g".into() },
            vec![],
            ty(vec![DimInfo::Lit(2), DimInfo::Lit(2)], Prim::F32),
            None,
        );
        let adjoint = dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let indices = dag.add_node(
            decl,
            RiscOp::Load { name: "i".into() },
            vec![],
            ty(vec![DimInfo::Lit(4)], Prim::Int64),
            None,
        );
        let dense = dag.add_node(
            decl,
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
            let decl = dag.declare("test");
            let target = dag.add_node(
                decl,
                RiscOp::Load { name: "t".into() },
                vec![],
                ty(
                    vec![DimInfo::Lit(2), DimInfo::Lit(5), DimInfo::Lit(7)],
                    Prim::F32,
                ),
                None,
            );
            let indices = dag.add_node(
                decl,
                RiscOp::Load { name: "i".into() },
                vec![],
                ty(vec![DimInfo::Lit(4)], Prim::Int64),
                None,
            );
            let updates = dag.add_node(
                decl,
                RiscOp::Load { name: "u".into() },
                vec![],
                ty(
                    vec![DimInfo::Lit(2), DimInfo::Lit(4), DimInfo::Lit(7)],
                    Prim::F32,
                ),
                None,
            );
            let scattered = dag.add_node(
                decl,
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
