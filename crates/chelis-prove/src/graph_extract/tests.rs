//! WI-3 graph-extraction producer SHAPE tests.
//!
//! No in-tree engine discharges `BoxRange` yet (cvc5 rejects it; Beacon is
//! out-of-tree; the WI-9 dispatcher is next), so these verify the producer
//! by SHAPE, with positive cases paired with negative twins:
//!
//! - a real source yields a box/range goal whose `IrHandle` carries a
//!   populated sha256 hash + a NAME-resolved root index, whose box is
//!   name-keyed + deterministically ordered, with a single scalar output,
//!   and whose serialized bytes hash to exactly the handle's hash;
//! - a multi-output case fans out into N goals with DISTINCT root indices
//!   sharing one DAG hash;
//! - a non-v1 `WireDag` is REJECTED at the producer boundary, not silently
//!   hashed (the negative twin for the cross-process consume gate).

use std::collections::BTreeMap;

use chelis_compiler_api::schema::{
    SourceKind, WIRE_DAG_SCHEMA_VERSION, WireDag, WireDagNode, WireDimInfo, WireRiscOp,
    WireTensorType,
};
use sha2::{Digest, Sha256};

use super::*;
use crate::discharge::{GoalShape, IntervalBox, OutputRange};

/// A single-scalar-output program: `out = mul(x, x)`. Lowers to a `WireDag`
/// whose `named_roots` include the scalar root `out`.
const SINGLE_OUTPUT_SOURCE: &str = "x = (x : tensor[f32])\n\
                                    out = (mul(x, x) : tensor[f32])\n";

fn output_range(name: &str, lo: f64, hi: f64) -> OutputRange {
    OutputRange {
        output: name.to_string(),
        lo,
        hi,
    }
}

fn input_box(dims: &[(&str, f64, f64)]) -> IntervalBox {
    IntervalBox {
        dims: dims
            .iter()
            .map(|(n, lo, hi)| (n.to_string(), *lo, *hi))
            .collect(),
    }
}

fn expected_sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::new();
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

// ===========================================================================
// Positive: a real source produces a populated, name-addressed goal.
// ===========================================================================

#[test]
fn real_source_yields_box_range_goal_with_populated_handle() {
    let extracted = box_range_goal_from_source(
        SINGLE_OUTPUT_SOURCE,
        SourceKind::Surf,
        input_box(&[("x", -10.0, 10.0)]),
        output_range("out", 0.0, 100.0),
    )
    .expect("a well-formed single-output source extracts a goal");

    // Shape: it is a BoxRange goal, not an SMT goal.
    assert!(matches!(extracted.goal.shape, GoalShape::BoxRange { .. }));

    // The IrHandle is populated with a hash + a root index.
    assert!(extracted.goal.ir.is_populated());
    let hash = extracted
        .goal
        .ir
        .dag_hash()
        .expect("populated handle carries a dag hash");
    let root_index = extracted
        .goal
        .ir
        .root_index()
        .expect("populated handle carries a root index");

    // The hash is a lowercase-hex sha256 (64 hex chars).
    assert_eq!(hash.len(), 64, "sha256 is 32 bytes = 64 hex chars");
    assert!(
        hash.chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "hash must be lowercase hex, got {hash}"
    );

    // The handle's hash addresses EXACTLY the returned bytes.
    assert_eq!(extracted.dag_hash, hash);
    assert_eq!(
        expected_sha256_hex(&extracted.wire_dag_bytes),
        hash,
        "the handle's hash must be the sha256 of the serialized WireDag bytes"
    );

    // The serialized artifact is a v1 WireDag that round-trips.
    let parsed: WireDag = serde_json::from_slice(&extracted.wire_dag_bytes)
        .expect("the serialized bytes parse back as a WireDag");
    assert_eq!(parsed.schema_version, WIRE_DAG_SCHEMA_VERSION);
    parsed
        .validate_schema_version()
        .expect("the produced artifact is a supported version");

    // The root index is NAME-resolved: it is the `out` root of the parsed
    // DAG, not a positional guess. `out` indexes a real root.
    assert!(
        (root_index as usize) < parsed.nodes.len(),
        "root index addresses a node in the DAG"
    );
    assert!(
        parsed.roots.contains(&(root_index as usize)),
        "the resolved index is one of the DAG's roots"
    );
}

#[test]
fn input_box_is_name_keyed_and_deterministically_ordered() {
    // Pass the dims out of name order; the produced box must be sorted by
    // name (Beacon's seeding is name-addressed, so the form must be stable).
    let extracted = box_range_goal_from_source(
        // A two-input program so the box has two named dims.
        "a = (a : tensor[f32])\n\
         b = (b : tensor[f32])\n\
         out = (add(mul(a, a), b) : tensor[f32])\n",
        SourceKind::Surf,
        input_box(&[("b", 1.0, 2.0), ("a", -1.0, 1.0)]),
        output_range("out", 0.0, 100.0),
    )
    .expect("two-input source extracts a goal");

    let GoalShape::BoxRange { inputs, output } = &extracted.goal.shape else {
        panic!("expected a BoxRange goal");
    };
    let names: Vec<&str> = inputs.dims.iter().map(|(n, _, _)| n.as_str()).collect();
    assert_eq!(names, vec!["a", "b"], "input box dims are sorted by name");
    assert_eq!(output.output, "out");
    // Single scalar output: the goal carries exactly one OutputRange, not a
    // packed multi-output range.
    assert_eq!(output.lo, 0.0);
    assert_eq!(output.hi, 100.0);
}

#[test]
fn name_sorted_input_box_orders_by_name() {
    let boxed = name_sorted_input_box(vec![
        ("z".to_string(), 0.0, 1.0),
        ("a".to_string(), 0.0, 1.0),
        ("m".to_string(), 0.0, 1.0),
    ]);
    let names: Vec<&str> = boxed.dims.iter().map(|(n, _, _)| n.as_str()).collect();
    assert_eq!(names, vec!["a", "m", "z"]);
}

// ===========================================================================
// Positive: multi-output fans out into N goals with distinct root indices.
// ===========================================================================

#[test]
fn multi_output_fans_out_into_distinct_root_indexed_goals() {
    // Two scalar outputs from one program: each becomes its own goal,
    // addressed by its OWN root index, sharing one DAG hash.
    let goals = box_range_goals_from_source(
        "x = (x : tensor[f32])\n\
         lo = (mul(x, x) : tensor[f32])\n\
         hi = (add(x, x) : tensor[f32])\n",
        SourceKind::Surf,
        input_box(&[("x", -5.0, 5.0)]),
        vec![
            output_range("lo", 0.0, 25.0),
            output_range("hi", -10.0, 10.0),
        ],
    )
    .expect("a two-output source fans out into two goals");

    assert_eq!(goals.len(), 2, "one goal per requested output");

    // Each goal is a populated BoxRange goal with a single scalar output.
    for goal in &goals {
        assert!(matches!(goal.goal.shape, GoalShape::BoxRange { .. }));
        assert!(goal.goal.ir.is_populated());
    }

    // The outputs are preserved in order.
    let outputs: Vec<&str> = goals
        .iter()
        .map(|g| match &g.goal.shape {
            GoalShape::BoxRange { output, .. } => output.output.as_str(),
            _ => unreachable!(),
        })
        .collect();
    assert_eq!(outputs, vec!["lo", "hi"]);

    // DISTINCT root indices: `lo` and `hi` are different roots.
    let lo_root = goals[0].goal.ir.root_index().expect("lo root index");
    let hi_root = goals[1].goal.ir.root_index().expect("hi root index");
    assert_ne!(
        lo_root, hi_root,
        "each output is addressed by its OWN root index"
    );

    // SHARED DAG hash: both goals reference the same artifact (one lowering).
    assert_eq!(
        goals[0].dag_hash, goals[1].dag_hash,
        "the two goals address the same lowered DAG artifact"
    );
    assert_eq!(goals[0].wire_dag_bytes, goals[1].wire_dag_bytes);
}

#[test]
fn lowering_is_deterministic_within_run() {
    // Two extractions of the same source within this run produce the same
    // artifact bytes + hash (content addressing is the back-reference).
    let a = box_range_goal_from_source(
        SINGLE_OUTPUT_SOURCE,
        SourceKind::Surf,
        input_box(&[("x", -1.0, 1.0)]),
        output_range("out", 0.0, 1.0),
    )
    .expect("first extraction");
    let b = box_range_goal_from_source(
        SINGLE_OUTPUT_SOURCE,
        SourceKind::Surf,
        input_box(&[("x", -1.0, 1.0)]),
        output_range("out", 0.0, 1.0),
    )
    .expect("second extraction");
    assert_eq!(a.wire_dag_bytes, b.wire_dag_bytes);
    assert_eq!(a.dag_hash, b.dag_hash);
}

// ===========================================================================
// Negative twins.
// ===========================================================================

/// Build a future-version `WireDag` (schema_version above the supported
/// ceiling) over a single trivial root, to exercise the producer's
/// fail-closed boundary check.
fn future_version_wire_dag() -> WireDag {
    WireDag {
        schema_version: WIRE_DAG_SCHEMA_VERSION + 1,
        nodes: vec![WireDagNode {
            id: 0,
            op: WireRiscOp::Load {
                name: "x".to_string(),
            },
            inputs: vec![],
            output_type: WireTensorType {
                dims: vec![WireDimInfo::Lit { size: 1 }],
                precision: "f32".to_string(),
            },
        }],
        roots: vec![0],
    }
}

#[test]
fn non_v1_wire_dag_is_rejected_at_the_producer_boundary() {
    let mut named_roots = BTreeMap::new();
    named_roots.insert("out".to_string(), 0usize);
    let err = box_range_goal_from_wire_dag(
        &future_version_wire_dag(),
        &named_roots,
        input_box(&[("x", -1.0, 1.0)]),
        output_range("out", 0.0, 1.0),
    )
    .expect_err("a future-version WireDag must be rejected, not silently hashed");
    assert!(
        matches!(err, GraphExtractError::SchemaRejected(_)),
        "expected a boundary schema rejection, got {err:?}"
    );
}

#[test]
fn v1_wire_dag_passes_the_boundary_and_hashes() {
    // The positive twin of the boundary check: a supported-version DAG is
    // hashed and produces a populated goal.
    let mut named_roots = BTreeMap::new();
    named_roots.insert("out".to_string(), 0usize);
    let dag = WireDag {
        schema_version: WIRE_DAG_SCHEMA_VERSION,
        nodes: vec![WireDagNode {
            id: 0,
            op: WireRiscOp::Load {
                name: "x".to_string(),
            },
            inputs: vec![],
            output_type: WireTensorType {
                dims: vec![WireDimInfo::Lit { size: 1 }],
                precision: "f32".to_string(),
            },
        }],
        roots: vec![0],
    };
    let extracted = box_range_goal_from_wire_dag(
        &dag,
        &named_roots,
        input_box(&[("x", -1.0, 1.0)]),
        output_range("out", 0.0, 1.0),
    )
    .expect("a supported-version DAG passes the boundary");
    assert!(extracted.goal.ir.is_populated());
    assert_eq!(extracted.goal.ir.root_index(), Some(0));
}

#[test]
fn unknown_output_name_is_rejected_with_available_roots() {
    let err = box_range_goal_from_source(
        SINGLE_OUTPUT_SOURCE,
        SourceKind::Surf,
        input_box(&[("x", -1.0, 1.0)]),
        // `nope` is not a root of the program.
        output_range("nope", 0.0, 1.0),
    )
    .expect_err("an output name that is not a named root must be rejected");
    match err {
        GraphExtractError::UnknownOutput { output, available } => {
            assert_eq!(output, "nope");
            assert!(
                available.contains(&"out".to_string()),
                "the available roots must list the real root `out`, got {available:?}"
            );
            // Available roots are reported sorted for stable diagnosis.
            let mut sorted = available.clone();
            sorted.sort();
            assert_eq!(available, sorted, "available roots are sorted");
        }
        other => panic!("expected UnknownOutput, got {other:?}"),
    }
}

#[test]
fn inverted_output_range_is_rejected_as_ill_formed() {
    // A real lowering, but the output bounds are inverted: the producer must
    // surface the GoalError rather than build a goal asserting an empty
    // region.
    let err = box_range_goal_from_source(
        SINGLE_OUTPUT_SOURCE,
        SourceKind::Surf,
        input_box(&[("x", -1.0, 1.0)]),
        output_range("out", 10.0, 1.0),
    )
    .expect_err("an inverted output range must be rejected");
    assert!(
        matches!(err, GraphExtractError::IllFormedGoal(_)),
        "expected an ill-formed-goal error, got {err:?}"
    );
}

#[test]
fn unlowerable_source_surfaces_a_lower_failure() {
    // A source that does not lower (a type error) must surface LowerFailed,
    // not panic and not a fabricated goal.
    let err = box_range_goal_from_source(
        "out = (mul(x, x) : tensor[f32])\n", // x is undeclared
        SourceKind::Surf,
        input_box(&[("x", -1.0, 1.0)]),
        output_range("out", 0.0, 1.0),
    )
    .expect_err("an unlowerable source must surface a lower failure");
    assert!(
        matches!(err, GraphExtractError::LowerFailed(_)),
        "expected a lowering failure, got {err:?}"
    );
}

// ===========================================================================
// Non-finite float guard (the critical silent-corruption fix).
//
// serde_json serializes a non-finite f64 (NaN / +inf / -inf) as `null`,
// which would (a) produce a self-consistent hash over bytes a consumer
// cannot parse back as a WireDag, and (b) collapse +inf / -inf / NaN to one
// hash. The producer must REJECT such a DAG at the boundary, before hashing.
// ===========================================================================

/// A single-node `WireDag` v1 carrying `op`, rooted at node 0, output `out`.
fn single_op_dag(op: WireRiscOp) -> WireDag {
    WireDag {
        schema_version: WIRE_DAG_SCHEMA_VERSION,
        nodes: vec![WireDagNode {
            id: 0,
            op,
            inputs: vec![],
            output_type: WireTensorType {
                dims: vec![WireDimInfo::Lit { size: 1 }],
                precision: "f32".to_string(),
            },
        }],
        roots: vec![0],
    }
}

fn extract_single_op(op: WireRiscOp) -> Result<ExtractedGoal, GraphExtractError> {
    let mut named_roots = BTreeMap::new();
    named_roots.insert("out".to_string(), 0usize);
    box_range_goal_from_wire_dag(
        &single_op_dag(op),
        &named_roots,
        input_box(&[("x", -1.0, 1.0)]),
        output_range("out", 0.0, 1.0),
    )
}

#[test]
fn non_finite_const_from_real_source_is_rejected_not_corrupted() {
    // `1.0e400` overflows f64 to +inf and lowers to Const(inf). Before the
    // guard, this produced an artifact whose bytes serialize the inf as
    // `null` -- a self-consistent hash over UNPARSEABLE bytes (silent
    // corruption). It must now be rejected with the typed error.
    let err = box_range_goal_from_source(
        "out = (1.0e400 : tensor[f32])\n",
        SourceKind::Surf,
        input_box(&[("x", -1.0, 1.0)]),
        output_range("out", 0.0, 1.0),
    )
    .expect_err("a non-finite Const from real source must be rejected, not corrupted");
    assert!(
        matches!(err, GraphExtractError::NonFiniteValue { .. }),
        "expected NonFiniteValue, got {err:?}"
    );
}

#[test]
fn finite_extreme_const_from_real_source_still_passes_and_hashes() {
    // The positive twin: a finite extreme (1e300) is NOT non-finite, so it
    // serializes as a real JSON number and produces a populated goal.
    let extracted = box_range_goal_from_source(
        "out = (1.0e300 : tensor[f32])\n",
        SourceKind::Surf,
        input_box(&[("x", -1.0, 1.0)]),
        output_range("out", 0.0, 1.0),
    )
    .expect("a finite extreme const passes the boundary");
    assert!(extracted.goal.ir.is_populated());
    // The bytes round-trip as a WireDag (the corruption the guard prevents).
    let parsed: WireDag = serde_json::from_slice(&extracted.wire_dag_bytes)
        .expect("a finite-float artifact parses back as a WireDag");
    parsed
        .validate_schema_version()
        .expect("the artifact is a supported version");
}

#[test]
fn each_non_finite_const_variant_is_rejected() {
    // +inf, -inf, and NaN would all serialize to the same `null` (the
    // collision). Each must be rejected so the collision is never reachable.
    for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let err = extract_single_op(WireRiscOp::Const { value })
            .expect_err("a non-finite Const must be rejected");
        match err {
            GraphExtractError::NonFiniteValue { node, field } => {
                assert_eq!(node, 0);
                assert_eq!(field, "value");
            }
            other => panic!("expected NonFiniteValue for {value}, got {other:?}"),
        }
    }
}

#[test]
fn every_f64_bearing_op_field_is_guarded() {
    // EXHAUSTIVE over the f64-bearing WireRiscOp variants: each f64 field,
    // when non-finite, must be rejected and must NAME the offending field.
    // This is the systemic lock -- if a new op adds an f64 field, the
    // production match (no wildcard) stops compiling AND this list must grow.
    let cases: Vec<(WireRiscOp, &str)> = vec![
        (
            WireRiscOp::UniformLike {
                low: f64::NAN,
                high: 1.0,
                seed: 0,
            },
            "low",
        ),
        (
            WireRiscOp::UniformLike {
                low: 0.0,
                high: f64::INFINITY,
                seed: 0,
            },
            "high",
        ),
        (
            WireRiscOp::Dropout {
                rate: f64::NEG_INFINITY,
                seed: 0,
            },
            "rate",
        ),
        (
            WireRiscOp::Pad {
                padding: vec![(0, 0)],
                fill: f64::NAN,
            },
            "fill",
        ),
        (
            WireRiscOp::Const {
                value: f64::INFINITY,
            },
            "value",
        ),
    ];

    // Pin the count so a future f64 field cannot silently shrink this list.
    assert_eq!(
        cases.len(),
        5,
        "5 f64-bearing op fields: UniformLike.low, UniformLike.high, \
         Dropout.rate, Pad.fill, Const.value. Update this test AND the \
         production guard if a new op carries an f64."
    );

    for (op, expected_field) in cases {
        let err = extract_single_op(op).expect_err("a non-finite op field must be rejected");
        match err {
            GraphExtractError::NonFiniteValue { node, field } => {
                assert_eq!(node, 0);
                assert_eq!(
                    field, expected_field,
                    "the rejection must name the offending field"
                );
            }
            other => panic!("expected NonFiniteValue naming `{expected_field}`, got {other:?}"),
        }
    }
}

#[test]
fn finite_f64_bearing_ops_pass_the_finite_guard() {
    // The positive twin: the same op variants with FINITE floats pass the
    // guard (they are rejected later only if some other check fails, but the
    // finite guard itself must not reject them).
    let finite_ops = [
        WireRiscOp::UniformLike {
            low: -1.0,
            high: 1.0,
            seed: 0,
        },
        WireRiscOp::Dropout { rate: 0.5, seed: 0 },
        WireRiscOp::Pad {
            padding: vec![(0, 0)],
            fill: 0.0,
        },
        WireRiscOp::Const { value: 3.5 },
    ];
    for op in finite_ops {
        let extracted =
            extract_single_op(op.clone()).expect("a finite-float op passes the finite guard");
        assert!(
            extracted.goal.ir.is_populated(),
            "finite op {op:?} should produce a populated goal"
        );
    }
}
