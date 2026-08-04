//! WI-3 graph-extraction producer: build box/range [`Goal`]s with a
//! populated [`IrHandle`] from real Chelis source.
//!
//! This is the keystone the verification dispatch layer (WI-9) and the
//! out-of-tree Beacon shell build on. It produces the
//! [`GoalShape::BoxRange`] form against the frozen seam contract
//! (`docs/design/phase2_seam_contract.md`), populating the goal's
//! [`IrHandle`] with the content hash + root index that addresses the
//! serialized `WireDag` v1 artifact a consumer resolves.
//!
//! ## Why content addressing, not a Dag handle
//!
//! Beacon consumes the SERIALIZED `WireDag` JSON bytes out of process:
//! it parses the slice, validates `schema_version <= WIRE_DAG_SCHEMA_VERSION`
//! (a lower version is forward-compatible via additive defaults, a higher
//! one fails closed), sha256s the bytes,
//! and selects the output by root index. So this producer addresses that
//! artifact by its content hash (lowercase hex sha256) plus a root index,
//! and [`ExtractedGoal`] also carries the serialized bytes so nothing
//! downstream is lost. Content addressing IS the back-reference: if
//! [`chelis_compiler_api::compiler::lower`] is deterministic, the WireDag
//! this producer hashes is byte-identical to the build's, so the verified
//! thing is the lowered thing. (Cross-invocation hash-match with the build
//! is a property to be aware of; the shape tests here verify within-run.)
//!
//! ## Finite-float precondition
//!
//! The content-address path requires FINITE floats. `serde_json` serializes
//! a non-finite f64 (NaN / +inf / -inf) as the JSON token `null`, which both
//! fails a consumer's round-trip parse and collapses the three non-finite
//! values to one byte sequence (one hash). [`check_finite_floats`] rejects a
//! DAG carrying any non-finite node-op float at the producer boundary, before
//! serialize + hash, with [`GraphExtractError::NonFiniteValue`] — never
//! hashing an artifact a consumer cannot parse. A canonical non-finite
//! representation (to support content-addressing such DAGs) is a tracked
//! follow-up requiring a coordinated `WireDag`-JSON-format change with Beacon.
//!
//! ## What this is NOT
//!
//! This producer is standalone machinery, not a rewire of the SMT path.
//! The obligation/property emit path still yields [`Goal::smt`] and stays
//! that way; deciding when to route a goal to box/range vs SMT is the WI-9
//! dispatcher's job. The box/range BOUNDS are inputs here (from fixtures
//! today; the authoring syntax was deferred in Phase 1). No in-tree engine
//! discharges `BoxRange` yet (cvc5 rejects it; Beacon is out-of-tree), so
//! this is exercised by SHAPE.
//!
//! ## Crate-boundary note
//!
//! This reaches a real IR DAG through the PUBLIC
//! [`chelis_compiler_api::compiler::lower`] API (source in, `WireDag` +
//! `named_roots` out), the same source-in/value-out shape the tier-A/C
//! paths already use. It pulls no `chelis-ir` dependency into this crate
//! and changes no `chelis-compiler-api` API surface; the handle holds only
//! a hash + index.

use chelis_compiler_api::compiler;
#[cfg(debug_assertions)]
use chelis_compiler_api::schema::WIRE_DAG_SCHEMA_VERSION;
use chelis_compiler_api::schema::{
    LowerRequest, LowerResult, SourceKind, WireDag, WireDagSchemaError, WireRiscOp,
};
use sha2::{Digest, Sha256};

use crate::discharge::{Goal, GoalError, IntervalBox, IrHandle, OutputRange};

/// A box/range [`Goal`] produced from real source, plus the serialized
/// `WireDag` v1 artifact its [`IrHandle`] addresses.
///
/// The goal's [`IrHandle`] carries the content hash (lowercase-hex sha256)
/// of [`wire_dag_bytes`](Self::wire_dag_bytes) and the root index its single
/// scalar output selects. The bytes are kept here so a downstream consumer
/// (WI-9 / Beacon) has the exact artifact the hash addresses; how those
/// bytes are conveyed or stored is the dispatcher's call, but the producer
/// must not drop them.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedGoal {
    /// The box/range goal, with a populated [`IrHandle`].
    pub goal: Goal,
    /// The serialized `WireDag` v1 JSON bytes the handle's hash addresses.
    /// A consumer recomputes sha256 over exactly these bytes and compares
    /// for byte-identity before trusting the DAG.
    pub wire_dag_bytes: Vec<u8>,
    /// The lowercase-hex sha256 of [`wire_dag_bytes`](Self::wire_dag_bytes),
    /// equal to the goal's `IrHandle::dag_hash`. Surfaced directly so a
    /// caller need not re-extract it from the handle.
    pub dag_hash: String,
}

/// An error producing a box/range goal from source.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GraphExtractError {
    /// `chelis_compiler_api::compiler::lower` failed to lower the source.
    /// Carries the joined diagnostic messages.
    #[error("lowering failed: {0}")]
    LowerFailed(String),

    /// The lowered `WireDag` carries a schema version this build cannot
    /// interpret. The producer fails CLOSED at its boundary rather than
    /// hashing an untrusted surface (the cross-process consume gate Beacon
    /// relies on: a future-version DAG never silently becomes a hash).
    #[error("WireDag schema rejected at producer boundary: {0}")]
    SchemaRejected(WireDagSchemaError),

    /// The requested output name is not a root of the lowered DAG, so no
    /// root index addresses it. Carries the missing name and the available
    /// root names (sorted) for diagnosis.
    #[error(
        "output `{output}` is not a named root of the lowered DAG; available roots: {available:?}"
    )]
    UnknownOutput {
        output: String,
        available: Vec<String>,
    },

    /// The box/range bounds are ill-formed (an inverted or NaN interval).
    /// Carries the underlying [`GoalError`].
    #[error("ill-formed box/range bounds: {0}")]
    IllFormedGoal(#[from] GoalError),

    /// A node op carries a NON-FINITE float (NaN / +inf / -inf), so the DAG
    /// cannot be content-addressed: `serde_json` serializes a non-finite f64
    /// as the JSON token `null`, which (a) does NOT parse back as an f64 (a
    /// consumer's deserialize fails), and (b) collapses +inf / -inf / NaN to
    /// ONE byte sequence, so three distinct DAGs would collide on one hash.
    /// The producer fails CLOSED here (same posture as the schema-version
    /// check) rather than hashing an artifact a consumer cannot parse. The
    /// content-address path requires finite floats; see
    /// `docs/design/phase2_seam_contract.md`.
    #[error(
        "node {node} op field `{field}` is a non-finite float (NaN/inf); the content-address path requires finite floats"
    )]
    NonFiniteValue { node: usize, field: &'static str },
}

/// Reject a [`WireDag`] that carries any non-finite float in a node op,
/// failing closed BEFORE serialize + hash.
///
/// `serde_json` serializes a non-finite f64 (NaN / +inf / -inf) as the JSON
/// token `null`. That breaks content addressing two ways: the bytes do not
/// parse back as a `WireDag` (a consumer's deserialize fails on
/// `null`-where-f64-expected, AFTER the self-consistent hash already matched,
/// so it is silent at the producer), and +inf / -inf / NaN all collapse to
/// the same `null`, so three distinct DAGs would share one hash. `to_vec`
/// returns `Ok(null)` rather than `Err`, so the serialize `.expect` never
/// fires — this guard is the only thing that catches it.
///
/// The match is EXHAUSTIVE with no wildcard, so a future `WireRiscOp` variant
/// forces a compile error here rather than silently slipping the guard; the
/// f64-bearing variants are checked and the f64-free ones are listed
/// explicitly. The `every_f64_bearing_op_field_is_guarded` test pins the
/// f64-bearing field set so the list cannot drift unnoticed.
fn check_finite_floats(wire_dag: &WireDag) -> Result<(), GraphExtractError> {
    fn reject_if_non_finite(
        node: usize,
        field: &'static str,
        value: f64,
    ) -> Result<(), GraphExtractError> {
        if value.is_finite() {
            Ok(())
        } else {
            Err(GraphExtractError::NonFiniteValue { node, field })
        }
    }

    for n in &wire_dag.nodes {
        let id = n.id;
        match &n.op {
            // --- f64-bearing ops: check EVERY f64 field ---
            WireRiscOp::UniformLike { low, high, .. } => {
                reject_if_non_finite(id, "low", *low)?;
                reject_if_non_finite(id, "high", *high)?;
            }
            WireRiscOp::Dropout { rate, .. } => reject_if_non_finite(id, "rate", *rate)?,
            WireRiscOp::Pad { fill, .. } => reject_if_non_finite(id, "fill", *fill)?,
            // Wire v4 (chelis#856): the constant payloads are sealed
            // dtype-tagged values. Prove's real-envelope reading takes
            // the f64 image (integer payloads are always finite; the
            // exact-env swap is the chelis#688 Phase 2 work).
            WireRiscOp::Const { value } => reject_if_non_finite(id, "value", value.as_f64_lossy())?,
            WireRiscOp::ConstTensor { data } => {
                for v in data.to_f64_lossy_vec() {
                    if !v.is_finite() {
                        return Err(GraphExtractError::NonFiniteValue {
                            node: id,
                            field: "data",
                        });
                    }
                }
            }

            // --- f64-free ops: no float to check. Listed explicitly (no
            // wildcard) so a new variant breaks the build until someone
            // decides whether it carries an f64. ---
            WireRiscOp::Add
            | WireRiscOp::Mul
            | WireRiscOp::Div
            | WireRiscOp::FloorDiv
            | WireRiscOp::TruncDiv
            | WireRiscOp::CmpLt
            | WireRiscOp::MaxElem
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
            | WireRiscOp::Sum { .. }
            | WireRiscOp::MaxReduce { .. }
            | WireRiscOp::MinReduce { .. }
            | WireRiscOp::ProdReduce { .. }
            | WireRiscOp::ReduceWindow { .. }
            | WireRiscOp::ReduceWindowGrad { .. }
            | WireRiscOp::Argmax { .. }
            | WireRiscOp::Argmin { .. }
            | WireRiscOp::Reshape { .. }
            | WireRiscOp::Permute { .. }
            | WireRiscOp::Expand { .. }
            | WireRiscOp::OneHot { .. }
            | WireRiscOp::Shrink { .. }
            | WireRiscOp::Stride { .. }
            | WireRiscOp::Shape { .. }
            | WireRiscOp::Load { .. }
            | WireRiscOp::Store { .. }
            | WireRiscOp::Copy
            | WireRiscOp::Drop
            | WireRiscOp::Realize
            | WireRiscOp::Cast { .. }
            | WireRiscOp::CastTrunc { .. }
            | WireRiscOp::FusedElem { .. }
            | WireRiscOp::BlasMatmul { .. }
            | WireRiscOp::Gather { .. }
            | WireRiscOp::ScatterAdd { .. }
            | WireRiscOp::Scatter { .. }
            | WireRiscOp::ScatterElements { .. } => {}
        }
    }
    Ok(())
}

/// Serialize a [`WireDag`] to canonical JSON bytes for hashing.
///
/// `serde_json::to_vec` over `WireDag`'s derived `Serialize` is
/// deterministic: struct fields serialize in declaration order, and
/// `WireDag.roots` / `WireDagNode.inputs` are ordered `Vec`s, so equal DAGs
/// produce byte-identical output. This is the canonical artifact a consumer
/// recomputes its sha256 over.
///
/// PRECONDITION: every float in the DAG's node ops is FINITE. A non-finite
/// f64 serializes as the JSON token `null` (not a number), which both breaks
/// round-trip parse and collapses +inf / -inf / NaN to one byte sequence, so
/// the byte-identical-for-equal-DAGs property holds ONLY for finite floats.
/// [`check_finite_floats`] enforces this precondition at the producer
/// boundary before this is called, so by the time a DAG reaches here it
/// carries only finite floats and the canonical claim is true.
fn serialize_wire_dag(wire_dag: &WireDag) -> Vec<u8> {
    serde_json::to_vec(wire_dag).expect("WireDag serializes to JSON")
}

/// Lowercase-hex sha256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write;
        write!(hex, "{byte:02x}").expect("writing to a String never fails");
    }
    hex
}

/// Build a deterministically-ordered (name-sorted) input box from the given
/// `(name, lo, hi)` dimensions. Beacon's Load seeding and split tie-break
/// are name-addressed; sorting by name keeps the emitted form stable across
/// runs regardless of caller insertion order.
pub fn name_sorted_input_box(mut dims: Vec<(String, f64, f64)>) -> IntervalBox {
    dims.sort_by(|a, b| a.0.cmp(&b.0));
    IntervalBox { dims }
}

/// Build a single box/range [`Goal`] from an already-lowered `WireDag` and
/// its `named_roots`, addressing the goal's ONE scalar output by name.
///
/// This is the pure core of the producer: it asserts the DAG's schema
/// version AND that every node-op float is finite at the boundary (failing
/// closed on an unsupported version or a non-finite float), serializes +
/// hashes the canonical bytes, resolves the root index for
/// `output_range.output` via `named_roots`, and builds the box/range goal
/// with a populated [`IrHandle`]. The input box is emitted name-sorted.
///
/// One scalar output per goal: a property with multiple/vector outputs is
/// fanned out into separate goals (see [`box_range_goals_from_source`]),
/// each addressed by its own root index.
pub fn box_range_goal_from_wire_dag(
    wire_dag: &WireDag,
    named_roots: &std::collections::BTreeMap<String, usize>,
    input_box: IntervalBox,
    output_range: OutputRange,
) -> Result<ExtractedGoal, GraphExtractError> {
    // Fail closed at the producer boundary on an unsupported schema version
    // before hashing: a future-version DAG must never silently become a
    // content hash a consumer would then trust.
    wire_dag
        .validate_schema_version()
        .map_err(GraphExtractError::SchemaRejected)?;

    // Fail closed on a non-finite float before hashing: serde_json emits a
    // non-finite f64 as `null`, which would produce a self-consistent hash
    // over bytes a consumer cannot parse (and would collapse +inf/-inf/NaN to
    // one hash). Same boundary posture as the schema-version check.
    check_finite_floats(wire_dag)?;

    // Resolve the goal's single output to a root index by NAME (Beacon's
    // Load seeding is name-addressed; a positional index would force a
    // binding-integrity remap downstream).
    let root_index = match named_roots.get(&output_range.output) {
        Some(&idx) => idx as u64,
        None => {
            let mut available: Vec<String> = named_roots.keys().cloned().collect();
            available.sort();
            return Err(GraphExtractError::UnknownOutput {
                output: output_range.output.clone(),
                available,
            });
        }
    };

    let wire_dag_bytes = serialize_wire_dag(wire_dag);
    let dag_hash = sha256_hex(&wire_dag_bytes);

    // box_range validates lo <= hi on every dimension and the output range;
    // an inverted/NaN bound surfaces as IllFormedGoal rather than a goal
    // asserting an empty region.
    let goal = Goal::box_range(name_sorted_input_box(input_box.dims), output_range)?
        .with_ir(IrHandle::from_wire_dag(dag_hash.clone(), root_index));

    Ok(ExtractedGoal {
        goal,
        wire_dag_bytes,
        dag_hash,
    })
}

/// Lower `source` to a `WireDag` v1 via the public
/// [`chelis_compiler_api::compiler::lower`] API, mapping a lowering failure to
/// [`GraphExtractError::LowerFailed`]. When `entry` is `Some`, lowering is
/// scoped to the defs reachable from that named entry (the WI-3
/// entrypoint-isolation path): unrelated top-level functions that reference
/// unresolved imports are pruned before the type checker runs, so they cannot
/// block the target's extraction. An unresolved symbol in the entry's OWN
/// reachable closure still surfaces here as a lowering failure (pruning drops
/// only genuinely-unreachable defs).
fn lower_source(
    source: &str,
    source_kind: SourceKind,
    entry: Option<&str>,
) -> Result<LowerResult, GraphExtractError> {
    compiler::lower(LowerRequest {
        source_kind,
        source: source.to_string(),
        entry: entry.map(str::to_string),
    })
    .map_err(|err| {
        let messages: Vec<String> = err.errors.iter().map(|d| d.message.clone()).collect();
        GraphExtractError::LowerFailed(messages.join("; "))
    })
}

/// Build a single box/range [`Goal`] from Chelis source, addressing the
/// goal's ONE scalar output by name.
///
/// Lowers the WHOLE `source` to a `WireDag` v1, then delegates to
/// [`box_range_goal_from_wire_dag`]. The `output_range.output` name must be a
/// named root of the lowered program. Use
/// [`box_range_goal_from_source_entry`] to extract one entry from a module
/// that also defines unrelated, unlowerable functions.
pub fn box_range_goal_from_source(
    source: &str,
    source_kind: SourceKind,
    input_box: IntervalBox,
    output_range: OutputRange,
) -> Result<ExtractedGoal, GraphExtractError> {
    let lowered = lower_source(source, source_kind, None)?;
    box_range_goal_from_wire_dag(&lowered.dag, &lowered.named_roots, input_box, output_range)
}

/// Build a single box/range [`Goal`] from Chelis source, scoping lowering to
/// the defs reachable from `entry`.
///
/// This is the entrypoint-isolation form: a real source module may define an
/// unrelated function whose body references an unresolved import (so the
/// whole-program lowering [`box_range_goal_from_source`] fails), while
/// the target entry's own closure lowers cleanly. Pruning to `entry`'s
/// reachable defs drops the unrelated function before the type checker runs,
/// so the target extracts. The `entry` is the def whose result becomes a
/// named root; `output_range.output` is then resolved against that root set
/// (it is typically `entry` itself, but the producer addresses by output
/// name, not by the entry name, so a multi-root entry stays addressable).
///
/// The safety property (locked by a negative test): if `entry`'s OWN closure
/// references an unresolved symbol, that error is NOT hidden — pruning keeps
/// the target's real dependencies, so the lowering still fails with that
/// error rather than silently extracting a partial graph.
pub fn box_range_goal_from_source_entry(
    source: &str,
    source_kind: SourceKind,
    entry: &str,
    input_box: IntervalBox,
    output_range: OutputRange,
) -> Result<ExtractedGoal, GraphExtractError> {
    let lowered = lower_source(source, source_kind, Some(entry))?;
    box_range_goal_from_wire_dag(&lowered.dag, &lowered.named_roots, input_box, output_range)
}

/// Build MULTIPLE box/range [`Goal`]s from Chelis source, one per requested
/// output, each addressed by its OWN root index.
///
/// A property with vector or multiple scalar outputs fans out into separate
/// goals here rather than packing a multi-output range into one goal: each
/// `(output_range)` in `outputs` becomes its own [`ExtractedGoal`] addressed
/// by that output's root index. The source is lowered ONCE; every goal
/// shares the same DAG hash (the artifact is the same) but carries a
/// distinct root index. Every goal shares `input_box` (the same input
/// region), emitted name-sorted.
///
/// Returns the goals in the same order as `outputs`. An unknown output name
/// fails the whole fan-out (no partial goal set), so a caller never gets a
/// silently-dropped output.
pub fn box_range_goals_from_source(
    source: &str,
    source_kind: SourceKind,
    input_box: IntervalBox,
    outputs: Vec<OutputRange>,
) -> Result<Vec<ExtractedGoal>, GraphExtractError> {
    box_range_goals_from_source_scoped(source, source_kind, None, input_box, outputs)
}

/// Entrypoint-scoped form of [`box_range_goals_from_source`]: when `entry` is
/// `Some`, lowering is restricted to that entry's reachable defs (see
/// [`box_range_goal_from_source_entry`] for the isolation rationale and the
/// safety property). Every output is still resolved against the (pruned)
/// program's named roots and fans out into its own goal.
pub fn box_range_goals_from_source_scoped(
    source: &str,
    source_kind: SourceKind,
    entry: Option<&str>,
    input_box: IntervalBox,
    outputs: Vec<OutputRange>,
) -> Result<Vec<ExtractedGoal>, GraphExtractError> {
    let lowered = lower_source(source, source_kind, entry)?;

    let mut goals = Vec::with_capacity(outputs.len());
    for output_range in outputs {
        goals.push(box_range_goal_from_wire_dag(
            &lowered.dag,
            &lowered.named_roots,
            input_box.clone(),
            output_range,
        )?);
    }
    Ok(goals)
}

#[cfg(debug_assertions)]
const _: () = {
    // The producer hashes against the version this build supports; if the
    // schema ceiling ever moves, the boundary assertion + the contract doc
    // must move with it. This is a compile-time tripwire on the constant.
    //
    // Moved to `2` for chelis#178: the wire surface gained
    // `WireRiscOp::FloorDiv` / `TruncDiv` (integer floor/truncating
    // division). Both are added to the f64-free op group in
    // `reject_non_finite_floats` above — they carry no float field, so they
    // do not change the box/range float-bound extraction contract; the
    // version moves only because a producer may now emit the new ops.
    //
    // Moved to `3` for chelis#616: `WireRiscOp::Reshape::new_shape` changed
    // from `Vec<WireDimInfo>` to `Vec<WireRtDim>` (runtime reshape target
    // extents) and `WireRtDim` gained `Sym`. `Reshape` stays in the f64-free
    // op group (`WireRtDim` carries no float field), so the box/range
    // float-bound extraction contract is unchanged; the version moves
    // because the reshape payload shape itself changed.
    // Moved to `4` for the chelis#729 rework (chelis#856): the constant
    // payloads (`Const::value`, `ConstTensor::data`) became sealed
    // dtype-tagged values with finalize-on-decode. The float-bound
    // extraction reads their f64 images through the named-lossy
    // accessors, so the box/range contract is unchanged in value terms;
    // integer constants above 2^53 are now representable on the wire and
    // still enter the real envelope through the same lossy image (the
    // exact-envelope swap is the chelis#688 Phase 2 work).
    assert!(WIRE_DAG_SCHEMA_VERSION == 4);
};

#[cfg(test)]
mod tests;
