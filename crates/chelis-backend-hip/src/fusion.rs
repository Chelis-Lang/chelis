//! HIP in-place fused-elementwise alias proof.
//!
//! Perf-F2(b) — the HIP port of the C backend's
//! `binder_equivalent_tensor_type` predicate and `fused_in_place_spec`
//! gate (`crates/chelis-backend-c/src/emit.rs`). When the upstream
//! linearity analyzer marks a FusedElem's reusable input AND the
//! reusable input is single-consumer AND its tensor type is
//! binder-equivalent to the FusedElem output, the HIP backend admits
//! the in-place alias: at runtime, if the reusable input's storage is
//! contiguous, the FusedElem's output view is aliased onto the input's
//! device buffer instead of allocated from the slot.
//!
//! This duplicates the C-side predicate intentionally (the W2-B brief
//! explicitly admits "acceptable code duplication for two backends"
//! over lifting into a shared crate). Both backends must keep the
//! predicate semantically identical; any change here should be
//! mirrored in `chelis-backend-c/src/emit.rs` under the same name.

use chelis_ir::dag::{DagNode, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::ownership::VerifiedDagView;

/// Pinned alias-proof spec for an in-place FusedElem.
///
/// `reusable_input` is the external input NodeId whose device buffer
/// the FusedElem's output is allowed to alias when contiguity holds at
/// runtime. `slot_has_later_owner` is true iff the FusedElem's slot
/// has a downstream owner that will consume the slot independently of
/// the runtime-contiguous path; when false, the fall-back slot
/// allocation can be deferred to inside the non-contiguous branch.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FusedInPlaceSpec {
    pub reusable_input: NodeId,
    pub slot_has_later_owner: bool,
}

/// Decide whether a FusedElem node admits in-place aliasing onto its
/// `reusable_input`. Returns `Some(reusable_input)` iff:
///   * the node has a `reusable_input` hint from upstream linearity,
///   * the node's op is `FusedElem`,
///   * the reusable input appears exactly once in the node's inputs
///     (no fan-in to the same buffer through multiple slots),
///   * the reusable input's output_type is binder-equivalent to the
///     FusedElem's output_type (see `binder_equivalent_tensor_type`),
///   * the reusable input has exactly one consumer in the DAG (the
///     FusedElem itself); multi-consumer inputs cannot be safely
///     aliased because mutating the buffer would corrupt the second
///     consumer's read.
pub(crate) fn fused_in_place_spec(node: &DagNode, dag: VerifiedDagView<'_>) -> Option<NodeId> {
    let reusable_input = node.reusable_input?;
    if !matches!(node.op, RiscOp::FusedElem { .. }) {
        return None;
    }
    if node
        .inputs
        .iter()
        .filter(|&&input| input == reusable_input)
        .count()
        != 1
    {
        return None;
    }
    let input_node = dag.get(reusable_input)?;
    if !binder_equivalent_tensor_type(&input_node.output_type, &node.output_type) {
        return None;
    }
    let consumer_count = dag
        .nodes()
        .iter()
        .flat_map(|candidate| candidate.inputs.iter())
        .filter(|&&input| input == reusable_input)
        .count()
        + dag
            .roots()
            .iter()
            .filter(|&&root| root == reusable_input)
            .count();
    if consumer_count != 1 {
        return None;
    }
    Some(reusable_input)
}

/// Conservative binder-equivalent equality for `TensorType` shape
/// comparisons in the HIP in-place fused-elementwise aliasing gate.
///
/// Mirrors the C-side `binder_equivalent_tensor_type` exactly. Two
/// tensor types are binder-equivalent iff:
///   * precisions match exactly,
///   * ranks match exactly,
///   * each pair of dim descriptors is binder-equivalent per
///     `binder_equivalent_dim_info`.
///
/// Strictly weaker than `DimExprKey::normalized_key` (which
/// alpha-renames symbolic dims by shape alone, an unsound expansion
/// per `chelis_ir::dag::DimExprKey`'s rustdoc warning), strictly
/// stronger than ignoring binder names.
pub(crate) fn binder_equivalent_tensor_type(a: &TensorType, b: &TensorType) -> bool {
    if a.precision != b.precision {
        return false;
    }
    if a.dims.len() != b.dims.len() {
        return false;
    }
    a.dims
        .iter()
        .zip(b.dims.iter())
        .all(|(da, db)| binder_equivalent_dim_info(da, db))
}

/// Two `DimInfo`s are binder-equivalent under the same forall scope
/// when their known-or-binder identity provably matches:
///   * `Lit(n)` ≡ `Lit(n)` — identical concrete sizes.
///   * `Named(n1, _)` ≡ `Named(n2, _)` — identical binder names
///     **and** consistent known sizes when both are known.
///   * `Lit(n)` ≡ `Named(_, Some(n))` and vice versa — a concrete
///     literal matches a named binder that has been resolved to the
///     same size.
///   * Everything else is rejected. `Lit` vs `Named(_, None)` is
///     intentionally rejected: a binder with unresolved size has no
///     evidence it matches a specific literal — `n` may differ.
///
/// Mirrors the C-side `binder_equivalent_dim_info` exactly.
fn binder_equivalent_dim_info(a: &DimInfo, b: &DimInfo) -> bool {
    match (a, b) {
        (DimInfo::Lit(la), DimInfo::Lit(lb)) => la == lb,
        (DimInfo::Named(na, sa), DimInfo::Named(nb, sb)) => {
            if na != nb {
                return false;
            }
            match (sa, sb) {
                (Some(la), Some(lb)) => la == lb,
                _ => true,
            }
        }
        (DimInfo::Lit(la), DimInfo::Named(_, Some(lb)))
        | (DimInfo::Named(_, Some(la)), DimInfo::Lit(lb)) => la == lb,
        _ => false,
    }
}
