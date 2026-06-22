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
//! Beacon consumes the SERIALIZED `WireDag` v1 JSON bytes out of process:
//! it parses the slice, asserts `schema_version == 1`, sha256s the bytes,
//! and selects the output by root index. So this producer addresses that
//! artifact by its content hash (lowercase hex sha256) plus a root index,
//! and [`ExtractedGoal`] also carries the serialized bytes so nothing
//! downstream is lost. Content addressing IS the back-reference: if
//! [`chelis_compiler_api::compiler::lower`] is deterministic, the WireDag
//! this producer hashes is byte-identical to the build's, so the verified
//! thing is the lowered thing. (Cross-invocation hash-match with the build
//! is a property to be aware of; the shape tests here verify within-run.)
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
use chelis_compiler_api::schema::{
    LowerRequest, SourceKind, WIRE_DAG_SCHEMA_VERSION, WireDag, WireDagSchemaError,
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
}

/// Serialize a [`WireDag`] to canonical JSON bytes for hashing.
///
/// `serde_json::to_vec` over `WireDag`'s derived `Serialize` is
/// deterministic: struct fields serialize in declaration order, and
/// `WireDag.roots` / `WireDagNode.inputs` are ordered `Vec`s, so equal DAGs
/// produce byte-identical output. This is the canonical artifact a consumer
/// recomputes its sha256 over.
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
/// version at the boundary (failing closed on an unsupported version),
/// serializes + hashes the canonical bytes, resolves the root index for
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

/// Build a single box/range [`Goal`] from Chelis source, addressing the
/// goal's ONE scalar output by name.
///
/// Lowers `source` to a `WireDag` v1 via the public
/// [`chelis_compiler_api::compiler::lower`] API, then delegates to
/// [`box_range_goal_from_wire_dag`]. The `output_range.output` name must be
/// a named root of the lowered program.
pub fn box_range_goal_from_source(
    source: &str,
    source_kind: SourceKind,
    input_box: IntervalBox,
    output_range: OutputRange,
) -> Result<ExtractedGoal, GraphExtractError> {
    let lowered = compiler::lower(LowerRequest {
        source_kind,
        source: source.to_string(),
    })
    .map_err(|err| {
        let messages: Vec<String> = err.errors.iter().map(|d| d.message.clone()).collect();
        GraphExtractError::LowerFailed(messages.join("; "))
    })?;
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
    let lowered = compiler::lower(LowerRequest {
        source_kind,
        source: source.to_string(),
    })
    .map_err(|err| {
        let messages: Vec<String> = err.errors.iter().map(|d| d.message.clone()).collect();
        GraphExtractError::LowerFailed(messages.join("; "))
    })?;

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
    assert!(WIRE_DAG_SCHEMA_VERSION == 1);
};

#[cfg(test)]
mod tests;
