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
//! - a non-current `WireDag` is REJECTED at the producer boundary, not silently
//!   hashed (the negative twin for the cross-process consume gate).

use chelis_compiler_api::schema::numbers::NonnegativeExtent;
use std::collections::BTreeMap;

use chelis_compiler_api::schema::{
    SourceKind, WIRE_DAG_SCHEMA_VERSION, WireDag, WireDagNode, WireDimInfo, WireRiscOp, WireRtDim,
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
fn result_claim_dependencies_remain_outside_the_scalar_proof_envelope() {
    use chelis_compiler_api::schema::{WireExtentWitnessSite, WireRtAxis};
    let mut dag = single_op_dag(WireRiscOp::Load { name: "x".into() });
    let mut witness = dag.nodes[0].clone();
    witness.id = 1;
    witness.op = WireRiscOp::ExtentWitness {
        site: WireExtentWitnessSite::Caller,
        parameter: "x".into(),
        axis: WireRtAxis::Lit { value: 0 },
        requirements: vec![],
        claims: vec![],
    };
    witness.inputs = vec![0];
    witness.output_type = WireTensorType {
        dims: vec![],
        precision: "int64".into(),
    };
    dag.nodes.push(witness.clone());
    witness.id = 2;
    witness.shape_deps = vec![1];
    let WireRiscOp::ExtentWitness { site, .. } = &mut witness.op else {
        unreachable!()
    };
    *site = WireExtentWitnessSite::ResultClaim {
        claim: "n".into(),
        axis: WireRtAxis::Lit { value: 0 },
    };
    dag.nodes.push(witness);
    dag.roots = vec![2];
    dag.validate_wire_contract()
        .expect("current transport admits the exact discrete obligation");
    let error = scalar_root_closure(&dag, 2)
        .expect_err("a discrete obligation has no float proof encoding");
    assert!(error.contains("no shape dependencies"), "{error}");
}

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

    // The serialized artifact is an exact-version v6 WireDag that round-trips.
    let parsed: WireDag = serde_json::from_slice(&extracted.wire_dag_bytes)
        .expect("the serialized bytes parse back as a WireDag");
    assert_eq!(parsed.schema_version, WIRE_DAG_SCHEMA_VERSION);
    parsed
        .validate_schema_version()
        .expect("the produced artifact is a supported version");

    // The root index is NAME-resolved: it is the `out` root of the parsed
    // DAG, not a positional guess. `out` indexes a real root.
    assert!(
        usize::try_from(root_index).unwrap() < parsed.nodes.len(),
        "root index addresses a node in the DAG"
    );
    assert!(
        parsed.roots.contains(&root_index),
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
            shape_deps: vec![],
            span_id: None,
            merged_spans: vec![],
            id: 0,
            op: WireRiscOp::Load {
                name: "x".to_string(),
            },
            inputs: vec![],
            output_type: WireTensorType {
                dims: vec![WireDimInfo::Lit {
                    size: NonnegativeExtent::new(1).unwrap(),
                }],
                precision: "f32".to_string(),
            },
        }],
        roots: vec![0],
    }
}

#[test]
fn non_current_wire_dag_is_rejected_at_the_producer_boundary() {
    let mut named_roots = BTreeMap::new();
    named_roots.insert("out".to_string(), 0_u64);
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
fn exact_v6_wire_dag_passes_the_boundary_and_hashes() {
    // The positive twin of the boundary check: a supported-version DAG is
    // hashed and produces a populated goal.
    let mut named_roots = BTreeMap::new();
    named_roots.insert("out".to_string(), 0_u64);
    let dag = WireDag {
        schema_version: WIRE_DAG_SCHEMA_VERSION,
        nodes: vec![WireDagNode {
            shape_deps: vec![],
            span_id: None,
            merged_spans: vec![],
            id: 0,
            op: WireRiscOp::Load {
                name: "x".to_string(),
            },
            inputs: vec![],
            output_type: WireTensorType {
                dims: vec![WireDimInfo::Lit {
                    size: NonnegativeExtent::new(1).unwrap(),
                }],
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
fn named_output_references_must_select_nodes_in_the_enclosed_dag() {
    let dag = WireDag {
        schema_version: WIRE_DAG_SCHEMA_VERSION,
        nodes: vec![
            WireDagNode {
                shape_deps: vec![],
                span_id: None,
                merged_spans: vec![],
                id: 0,
                op: WireRiscOp::Load { name: "x".into() },
                inputs: vec![],
                output_type: WireTensorType {
                    dims: vec![],
                    precision: "f32".into(),
                },
            },
            WireDagNode {
                shape_deps: vec![],
                span_id: None,
                merged_spans: vec![],
                id: 1,
                op: WireRiscOp::Copy,
                inputs: vec![0],
                output_type: WireTensorType {
                    dims: vec![],
                    precision: "f32".into(),
                },
            },
        ],
        roots: vec![1],
    };
    let extract = |named| {
        box_range_goal_from_wire_dag(
            &dag,
            &named,
            input_box(&[("x", -1.0, 1.0)]),
            output_range("out", -1.0, 1.0),
        )
    };
    for valid in [0_u64, 1] {
        assert_eq!(
            extract(BTreeMap::from([("out".into(), valid)]))
                .unwrap()
                .goal
                .ir
                .root_index(),
            Some(valid)
        );
    }
    for invalid in [2, u64::MAX] {
        let error = extract(BTreeMap::from([("out".into(), invalid)]))
            .expect_err("out-of-bounds reference");
        assert!(error.to_string().contains("owning WireDag"), "{error}");
        let error = extract(BTreeMap::from([
            ("out".into(), 1),
            ("unused".into(), invalid),
        ]))
        .expect_err("every named reference belongs to this DAG");
        assert!(error.to_string().contains("unused"), "{error}");
    }
}

#[test]
fn invalid_exact_v6_count_is_rejected_without_panicking() {
    let mut named_roots = BTreeMap::new();
    named_roots.insert("out".to_string(), 1_u64);
    let dag = WireDag {
        schema_version: WIRE_DAG_SCHEMA_VERSION,
        nodes: vec![
            WireDagNode {
                shape_deps: vec![],
                span_id: None,
                merged_spans: vec![],
                id: 0,
                op: WireRiscOp::Load {
                    name: "x".to_string(),
                },
                inputs: vec![],
                output_type: WireTensorType {
                    dims: vec![WireDimInfo::Lit {
                        size: NonnegativeExtent::new(4).unwrap(),
                    }],
                    precision: "f32".to_string(),
                },
            },
            WireDagNode {
                shape_deps: vec![],
                span_id: None,
                merged_spans: vec![],
                id: 1,
                op: WireRiscOp::Count { axes: vec![0] },
                inputs: vec![0],
                output_type: WireTensorType {
                    dims: vec![],
                    precision: "int64".to_string(),
                },
            },
        ],
        roots: vec![1],
    };
    let err = box_range_goal_from_wire_dag(
        &dag,
        &named_roots,
        input_box(&[("x", -1.0, 1.0)]),
        output_range("out", 0.0, 4.0),
    )
    .expect_err("Count over a non-bool input must return a typed wire-contract rejection");
    assert!(
        matches!(err, GraphExtractError::WireContractRejected(_)),
        "expected a typed wire-contract rejection, got {err:?}"
    );
    assert!(
        err.to_string()
            .contains("input dtype must be bool, found f32"),
        "the interchange dtype must validate before the intended Count rejection: {err}"
    );
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
// BoxRange retains finite-only proof support independently of the exact wire
// codec's ability to transport every stored nonfinite bit pattern.
// ===========================================================================

/// A graph with the operands and output dtype required by the tested operation.
fn single_op_dag(op: WireRiscOp) -> WireDag {
    let precision = match &op {
        WireRiscOp::Const { value } => value.prim(),
        WireRiscOp::ConstTensor { data } => data.prim(),
        WireRiscOp::Pad { fill, .. } => fill.prim(),
        _ => chelis_types::types::Prim::F32,
    };
    let size = match &op {
        WireRiscOp::ConstTensor { data } => i64::try_from(data.len()).unwrap(),
        _ => 1,
    };
    let output_type = WireTensorType {
        dims: vec![WireDimInfo::Lit {
            size: NonnegativeExtent::new(size).unwrap(),
        }],
        precision: precision.interchange_name().into(),
    };
    let mut nodes = Vec::new();
    let inputs = if matches!(op, WireRiscOp::Pad { .. }) {
        nodes.push(WireDagNode {
            shape_deps: vec![],
            span_id: None,
            merged_spans: vec![],
            id: 0,
            op: WireRiscOp::Load { name: "x".into() },
            inputs: vec![],
            output_type: output_type.clone(),
        });
        vec![0]
    } else {
        vec![]
    };
    let root = u64::try_from(nodes.len()).unwrap();
    nodes.push(WireDagNode {
        shape_deps: vec![],
        span_id: None,
        merged_spans: vec![],
        id: root,
        op,
        inputs,
        output_type,
    });
    WireDag {
        schema_version: WIRE_DAG_SCHEMA_VERSION,
        nodes,
        roots: vec![root],
    }
}

fn extract_single_op(op: WireRiscOp) -> Result<ExtractedGoal, GraphExtractError> {
    let mut named_roots = BTreeMap::new();
    let dag = single_op_dag(op);
    named_roots.insert("out".to_string(), dag.roots[0]);
    box_range_goal_from_wire_dag(
        &dag,
        &named_roots,
        input_box(&[("x", -1.0, 1.0)]),
        output_range("out", 0.0, 1.0),
    )
}

#[test]
fn non_finite_const_from_real_source_is_rejected_not_corrupted() {
    // `1e400` overflows f64 to +inf. Canonical Surf rejects non-finite
    // literals before lowering, so this real-source path must fail closed
    // without producing corrupt artifact bytes. The direct WireDag sibling
    // above still locks GraphExtractError::NonFiniteValue at its boundary.
    let err = box_range_goal_from_source(
        "out = (1e400 : tensor[f32])\n",
        SourceKind::Surf,
        input_box(&[("x", -1.0, 1.0)]),
        output_range("out", 0.0, 1.0),
    )
    .expect_err("a non-finite Const from real source must be rejected, not corrupted");
    assert!(
        matches!(&err, GraphExtractError::LowerFailed(message) if message.contains("non-finite")),
        "expected the canonical Surf non-finite rejection, got {err:?}"
    );

    // chelis#729 rework: the sealed payload finalizes at the ascribed
    // dtype, so a value that overflows ITS OWN dtype (1e300 at f32 is
    // +inf per [04-NUM-2]) is honestly non-finite and takes the same
    // rejection; the pre-sealed payload carried the finite f64 fiction
    // and slipped past this guard.
    let err = box_range_goal_from_source(
        "out = (1e300 : tensor[f32])\n",
        SourceKind::Surf,
        input_box(&[("x", -1.0, 1.0)]),
        output_range("out", 0.0, 1.0),
    )
    .expect_err("an f32-overflowing const finalizes to inf and must be rejected");
    assert!(
        matches!(err, GraphExtractError::NonFiniteValue { .. }),
        "expected NonFiniteValue for the finalized f32 inf, got {err:?}"
    );
}

#[test]
fn finite_extreme_const_from_real_source_still_passes_and_hashes() {
    // The positive twin: a finite extreme (1e300) is NOT non-finite, so it
    // serializes as a real JSON number and produces a populated goal.
    // RE-AUTHORED at the chelis#729 rework (chelis#856): the fixture was
    // `1.0e300 : tensor[f32]`, which only passed because the pre-sealed
    // payload carried the un-finalized f64 image; the honest f32 value
    // of 1e300 is +inf ([04-NUM-2] overflow), which the guard now
    // correctly rejects (see the rejected twin below). A finite extreme
    // needs a dtype that can hold it, so the fixture moves to f64.
    let extracted = box_range_goal_from_source(
        "out = (1e300 : tensor[f64])\n",
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
    // The wire admits all three classes exactly; BoxRange rejects them.
    for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let op = WireRiscOp::Const {
            value: chelis_types::scalar_from_f64("test", chelis_types::types::Prim::F64, value)
                .expect("float finalize is total"),
        };
        let bytes = serde_json::to_vec(&single_op_dag(op.clone())).unwrap();
        let decoded: WireDag = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(serde_json::to_vec(&decoded).unwrap(), bytes);
        let err = extract_single_op(op).expect_err("a non-finite Const must be rejected");
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
fn every_embedded_numeric_field_is_guarded() {
    // Keep the private finite-proof guard exhaustive; the public boundary
    // rejects invalid Random parameters earlier in WireDag admission.
    let cases: Vec<(WireRiscOp, &str)> = vec![
        (
            WireRiscOp::Pad {
                padding: vec![(
                    WireRtDim::Lit {
                        value: NonnegativeExtent::new(0).unwrap(),
                    },
                    WireRtDim::Lit {
                        value: NonnegativeExtent::new(0).unwrap(),
                    },
                )],
                fill: chelis_types::scalar_from_f64(
                    "test",
                    chelis_types::types::Prim::F32,
                    f64::NAN,
                )
                .expect("f32 accepts NaN"),
            },
            "fill",
        ),
        (
            WireRiscOp::Const {
                value: chelis_types::scalar_from_f64(
                    "test",
                    chelis_types::types::Prim::F64,
                    f64::INFINITY,
                )
                .expect("float finalize is total"),
            },
            "value",
        ),
        (
            WireRiscOp::ConstTensor {
                data: serde_json::from_value(
                    serde_json::json!({"dtype":"f32","bits":["3f800000","7fc00001"]}),
                )
                .unwrap(),
            },
            "data",
        ),
    ];

    // Pin the field set so an existing payload cannot silently disappear.
    assert_eq!(cases.len(), 3, "Pad.fill, Const.value and ConstTensor.data");

    for (op, expected_field) in cases {
        let dag = single_op_dag(op.clone());
        let err = check_finite_floats(&dag).expect_err("private finite-proof guard");
        match err {
            GraphExtractError::NonFiniteValue { node, field } => {
                assert_eq!(node, dag.nodes.len() - 1);
                assert_eq!(
                    field, expected_field,
                    "the rejection must name the offending field"
                );
            }
            other => panic!("expected NonFiniteValue naming `{expected_field}`, got {other:?}"),
        }
        let error = extract_single_op(op.clone()).expect_err("public proof boundary");
        assert!(
            matches!(error, GraphExtractError::NonFiniteValue { .. }),
            "{error}"
        );
    }
}

#[test]
fn finite_numeric_payloads_pass_the_proof_boundary() {
    // The positive twin: the same op variants with FINITE floats pass the
    // guard (they are rejected later only if some other check fails, but the
    // finite guard itself must not reject them).
    let finite_ops = [
        WireRiscOp::Pad {
            padding: vec![(
                WireRtDim::Lit {
                    value: NonnegativeExtent::new(0).unwrap(),
                },
                WireRtDim::Lit {
                    value: NonnegativeExtent::new(0).unwrap(),
                },
            )],
            fill: chelis_types::scalar_from_f64("test", chelis_types::types::Prim::F32, 0.0)
                .expect("finite f32"),
        },
        WireRiscOp::Const {
            value: chelis_types::scalar_from_f64("test", chelis_types::types::Prim::F64, 3.5)
                .expect("finite f64"),
        },
        WireRiscOp::ConstTensor {
            data: serde_json::from_value(
                serde_json::json!({"dtype":"int64","values":[9007199254740993_i64]}),
            )
            .unwrap(),
        },
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

// ===========================================================================
// Entrypoint isolation (chelis#440): extract one entry from a module that
// also defines an unrelated function which fails to lower.
//
// A real source module (Shoals pricing.ch) imports a symbol that is unbound
// without reef package context and uses it in ONE function, which blocks the
// WHOLE-program lowering of an unrelated, self-contained target. This mirrors
// that shape with a minimal but realistic pure-tensor fixture: a target entry
// `priced` (a pure-elementwise-tensor function -> a real WireDag root) that
// uses a shared helper `scaled`, plus an `unrelated` function whose body
// references an unresolved import `missing_sym`. Pruning to `priced` drops
// `unrelated` (and its unresolvable reference) before the type checker runs,
// so the target extracts and is addressable; the WHOLE-program form fails.
// ===========================================================================

/// A module with a self-contained pure-tensor target (`priced`, using helper
/// `scaled`) plus an `unrelated` function that references an unresolved
/// import. The whole program does not lower; pruning to `priced` does.
const ENTRY_ISOLATION_SOURCE: &str = "module Demo.Pricer\n\
    import Other.Pkg (missing_sym)\n\
    def scaled[n](v: tensor[n, f32], k: tensor[n, f32]) -> tensor[n, f32] = mul(v, k)\n\
    def priced[n](v: tensor[n, f32], k: tensor[n, f32], b: tensor[n, f32]) -> tensor[n, f32] = add(scaled(v, k), b)\n\
    def unrelated[n](v: tensor[n, f32]) -> tensor[n, f32] = missing_sym(v)\n";

/// chelis#506 decision fixture: scalar host entries are not WireDag roots.
/// Beacon's seam is tensor-DAG-root based, so the scalar path must fail with a
/// clear named-root diagnostic rather than fabricating a rank-0 root.
const SCALAR_HOST_ENTRY_SOURCE: &str = "module Demo.ScalarHost\n\
    def scalar_price(s: f32, k: f32) -> f32 = add(s, k)\n";

const ISSUE_506_TENSOR_ENTRY_SOURCE: &str = "module Demo.TensorEntry\n\
    def tensor_price(x: tensor[3, f32]) -> tensor[3, f32] = add(x, x)\n";

#[test]
fn whole_program_lowering_fails_when_an_unrelated_fn_is_unlowerable() {
    // The negative baseline: WITHOUT entry scoping, the unrelated
    // `missing_sym` reference makes the whole-program lowering fail, even
    // though `priced` itself is self-contained. This is the chelis#440 bug
    // the entry-scoped path fixes.
    let err = box_range_goal_from_source(
        ENTRY_ISOLATION_SOURCE,
        SourceKind::Surf,
        input_box(&[("v", -10.0, 10.0)]),
        output_range("priced", 0.0, 100.0),
    )
    .expect_err("the unrelated unlowerable fn blocks whole-program lowering");
    match err {
        GraphExtractError::LowerFailed(message) => {
            assert!(
                message.contains("missing_sym"),
                "the failure must be the unrelated import, got {message:?}"
            );
        }
        other => panic!("expected LowerFailed naming missing_sym, got {other:?}"),
    }
}

#[test]
fn entry_scoped_extraction_prunes_the_unrelated_fn_and_yields_a_populated_goal() {
    // The positive case: scoping to `priced` prunes `unrelated`, so the target
    // extracts with a populated, name-addressed handle.
    let extracted = box_range_goal_from_source_entry(
        ENTRY_ISOLATION_SOURCE,
        SourceKind::Surf,
        "priced",
        input_box(&[("v", -10.0, 10.0)]),
        output_range("priced", 0.0, 100.0),
    )
    .expect("scoping to `priced` prunes the unrelated fn and extracts the target");

    // It is a populated BoxRange goal addressed by a name-resolved root index.
    assert!(matches!(extracted.goal.shape, GoalShape::BoxRange { .. }));
    assert!(extracted.goal.ir.is_populated());
    let hash = extracted
        .goal
        .ir
        .dag_hash()
        .expect("a populated handle carries a dag hash");
    let root_index = extracted
        .goal
        .ir
        .root_index()
        .expect("a populated handle carries a root index");

    // The hash is a lowercase-hex sha256 of EXACTLY the serialized bytes
    // (the content-address convention Beacon hashes; no reformat step).
    assert_eq!(hash.len(), 64, "sha256 is 32 bytes = 64 hex chars");
    assert!(
        hash.chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "hash must be lowercase hex, got {hash}"
    );
    assert_eq!(extracted.dag_hash, hash);
    assert_eq!(
        expected_sha256_hex(&extracted.wire_dag_bytes),
        hash,
        "the handle's hash must be the sha256 of the serialized WireDag bytes"
    );

    // The serialized artifact is an exact-version v6 WireDag that round-trips, and the
    // name-resolved `priced` root indexes a real root of it.
    let parsed: WireDag = serde_json::from_slice(&extracted.wire_dag_bytes)
        .expect("the serialized bytes parse back as a WireDag");
    assert_eq!(parsed.schema_version, WIRE_DAG_SCHEMA_VERSION);
    parsed
        .validate_schema_version()
        .expect("the produced artifact is a supported version");
    assert!(
        usize::try_from(root_index).unwrap() < parsed.nodes.len(),
        "the root index addresses a node in the DAG"
    );
    assert!(
        parsed.roots.contains(&root_index),
        "the resolved index is one of the DAG's roots"
    );
}

#[test]
fn scalar_host_entry_is_not_a_wire_dag_root_and_reports_available_roots() {
    let err = box_range_goal_from_source_entry(
        SCALAR_HOST_ENTRY_SOURCE,
        SourceKind::Surf,
        "scalar_price",
        input_box(&[("k", 90.0, 110.0), ("s", 80.0, 120.0)]),
        output_range("scalar_price", 0.0, 250.0),
    )
    .expect_err("scalar host entries stay non-root; Shoals must provide a tensor entry");
    let message = err.to_string();
    assert!(
        message.contains("output `scalar_price` is not a named root"),
        "UnknownOutput diagnostic must name the missing scalar output, got {message:?}"
    );
    assert!(
        message.contains("available roots: []"),
        "UnknownOutput diagnostic must make the empty tensor-root set explicit, got {message:?}"
    );
    match err {
        GraphExtractError::UnknownOutput { output, available } => {
            assert_eq!(output, "scalar_price");
            assert!(
                available.is_empty(),
                "a scalar host entry should not be silently exposed as a WireDag root; available roots were {available:?}"
            );
        }
        other => panic!("expected UnknownOutput for scalar host non-root, got {other:?}"),
    }
}

#[test]
fn tensor_entry_is_a_wire_dag_root_for_issue_506_positive_control() {
    let extracted = box_range_goal_from_source_entry(
        ISSUE_506_TENSOR_ENTRY_SOURCE,
        SourceKind::Surf,
        "tensor_price",
        input_box(&[("x", -10.0, 10.0)]),
        output_range("tensor_price", 0.0, 250.0),
    )
    .expect("tensor/root-shaped entries should produce a WireDag root");

    assert!(matches!(extracted.goal.shape, GoalShape::BoxRange { .. }));
    assert!(extracted.goal.ir.is_populated());
    let root_index = extracted
        .goal
        .ir
        .root_index()
        .expect("tensor entry carries a root index");
    let parsed: WireDag = serde_json::from_slice(&extracted.wire_dag_bytes)
        .expect("the serialized bytes parse back as a WireDag");
    assert_eq!(
        parsed.roots.len(),
        1,
        "the tensor entry should produce exactly one root"
    );
    assert!(
        parsed.roots.contains(&root_index),
        "the name-resolved root index must address the WireDag root list"
    );
}

#[test]
fn entry_scoping_does_not_change_an_already_lowerable_program() {
    // The control: on a program that ALREADY lowers whole, scoping to the
    // sole entry produces a populated goal equivalent to the unscoped path.
    // (The bytes need not be identical to the unscoped form -- pruning may
    // drop unreachable roots -- but the target entry must still extract.)
    let scoped = box_range_goal_from_source_entry(
        "x = (x : tensor[f32])\n\
         out = (mul(x, x) : tensor[f32])\n",
        SourceKind::Surf,
        "out",
        input_box(&[("x", -1.0, 1.0)]),
        output_range("out", 0.0, 1.0),
    )
    .expect("scoping to the sole entry of a lowerable program extracts it");
    assert!(scoped.goal.ir.is_populated());
}

#[test]
fn entry_scoping_does_not_hide_an_error_in_the_targets_own_closure() {
    // The safety negative twin: when the ENTRY's OWN reachable closure
    // references the unresolved symbol, pruning must NOT hide it. Scoping to
    // `unrelated` (whose body IS `missing_sym(v)`) keeps that reference, so
    // the lowering still fails with the real error -- pruning drops only
    // genuinely-unreachable defs, never a real error in the target's deps.
    let err = box_range_goal_from_source_entry(
        ENTRY_ISOLATION_SOURCE,
        SourceKind::Surf,
        "unrelated",
        input_box(&[("v", -10.0, 10.0)]),
        output_range("unrelated", 0.0, 100.0),
    )
    .expect_err("scoping to an entry whose own closure is unlowerable must still fail");
    match err {
        GraphExtractError::LowerFailed(message) => {
            assert!(
                message.contains("missing_sym"),
                "the entry's own unresolved reference must surface, got {message:?}"
            );
        }
        other => panic!("expected LowerFailed naming missing_sym, got {other:?}"),
    }
}

#[test]
fn entry_scoped_extraction_is_deterministic_within_run() {
    // Two scoped extractions of the same source + entry within this run
    // produce byte-identical artifacts (content addressing is the
    // back-reference Beacon relies on; a reformat/re-serialize step would
    // break the hash match).
    let a = box_range_goal_from_source_entry(
        ENTRY_ISOLATION_SOURCE,
        SourceKind::Surf,
        "priced",
        input_box(&[("v", -1.0, 1.0)]),
        output_range("priced", 0.0, 1.0),
    )
    .expect("first scoped extraction");
    let b = box_range_goal_from_source_entry(
        ENTRY_ISOLATION_SOURCE,
        SourceKind::Surf,
        "priced",
        input_box(&[("v", -1.0, 1.0)]),
        output_range("priced", 0.0, 1.0),
    )
    .expect("second scoped extraction");
    assert_eq!(a.wire_dag_bytes, b.wire_dag_bytes);
    assert_eq!(a.dag_hash, b.dag_hash);
}
