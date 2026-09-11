//! [10] §3: exact claim, invocation dependency and provenance transport.
use chelis_compiler_api::schema::*;
use chelis_types::{scalar_from_i64, types::Prim};

fn integer(value: i64) -> chelis_types::ScalarValue {
    scalar_from_i64("load", Prim::Int64, value).unwrap()
}

fn extent(value: i64) -> numbers::NonnegativeExtent {
    numbers::NonnegativeExtent::new(value).unwrap()
}

fn fixture() -> WireDag {
    WireDag {
        schema_version: WIRE_DAG_SCHEMA_VERSION,
        roots: vec![2],
        nodes: vec![
            WireDagNode {
                id: 0,
                op: WireRiscOp::Load { name: "x".into() },
                inputs: vec![],
                output_type: WireTensorType {
                    dims: vec![WireDimInfo::Named {
                        name: "rows".into(),
                        size: None,
                    }],
                    precision: "f32".into(),
                },
                shape_deps: vec![],
                span_id: None,
                merged_spans: vec![],
            },
            WireDagNode {
                id: 1,
                op: WireRiscOp::ExtentWitness {
                    site: WireExtentWitnessSite::Caller,
                    parameter: "x".into(),
                    axis: WireRtAxis::Lit { value: 0 },
                    requirements: vec![extent(4), extent(4), extent(9)],
                    claims: vec![],
                },
                inputs: vec![0],
                output_type: WireTensorType {
                    dims: vec![],
                    precision: "int64".into(),
                },
                shape_deps: vec![],
                span_id: Some("call-f".into()),
                merged_spans: vec!["inlined-g".into()],
            },
            WireDagNode {
                id: 2,
                op: WireRiscOp::Const { value: integer(9) },
                inputs: vec![],
                output_type: WireTensorType {
                    dims: vec![],
                    precision: "int64".into(),
                },
                shape_deps: vec![1],
                span_id: None,
                merged_spans: vec![],
            },
        ],
    }
}
#[test]
fn witness_roundtrip_retains_claim_dependency_and_provenance() {
    let dag = fixture();
    let json = serde_json::to_value(&dag).unwrap();
    assert_eq!(
        json["nodes"][1]["op"]["requirements"],
        serde_json::json!([4, 4, 9])
    );
    assert_eq!(json["nodes"][2]["shape_deps"], serde_json::json!([1]));
    let decoded = WireDag::from_validated_json(&json.to_string()).unwrap();
    assert_eq!(serde_json::to_value(decoded).unwrap(), json);
}

#[test]
fn witness_site_is_explicit_and_survives_wire_transport() {
    for (site, spelling) in [
        (WireExtentWitnessSite::Caller, "caller"),
        (WireExtentWitnessSite::LocalExpand, "local_expand"),
    ] {
        let mut dag = fixture();
        if let WireRiscOp::ExtentWitness { site: target, .. } = &mut dag.nodes[1].op {
            *target = site;
        }
        let mut json = serde_json::to_value(dag).unwrap();
        assert_eq!(json["nodes"][1]["op"]["site"], spelling);
        let decoded = WireDag::from_validated_json(&json.to_string()).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), json);
        json["nodes"][1]["op"]["site"] = serde_json::json!("inferred");
        assert!(WireDag::from_validated_json(&json.to_string()).is_err());
    }
}

#[test]
fn wire_dag_v8_rejects_before_witness_body_decode() {
    let legacy_with_malformed_body = r#"{
        "schema_version": 8,
        "nodes": [{"op": {"kind": "not_a_real_operation"}}],
        "roots": []
    }"#;
    assert!(matches!(
        WireDag::from_validated_json(legacy_with_malformed_body),
        Err(WireDagDecodeError::Schema(
            WireDagSchemaError::UnsupportedSchemaVersion {
                found: 8,
                supported: WIRE_DAG_SCHEMA_VERSION
            }
        ))
    ));
}

#[test]
fn malformed_claims_and_invocation_edges_are_not_decoded_or_encoded() {
    for mutation in 0..5 {
        let mut dag = fixture();
        match mutation {
            0 => {
                if let WireRiscOp::ExtentWitness { axis, .. } = &mut dag.nodes[1].op {
                    *axis = WireRtAxis::Lit { value: 1 };
                }
            }
            1 => dag.nodes[1].inputs.clear(),
            2 => dag.nodes[1].output_type.precision = "f32".into(),
            3 => dag.nodes[2].shape_deps[0] = 2,
            4 => dag.nodes[2].shape_deps[0] = u64::MAX,
            _ => unreachable!(),
        }
        assert!(dag.validate_wire_contract().is_err(), "mutation {mutation}");
        assert!(
            serde_json::to_value(dag).is_err(),
            "encode mutation {mutation}"
        );
    }
    for requirement in [serde_json::json!(-1), serde_json::json!(4.0)] {
        let mut json = serde_json::to_value(fixture()).unwrap();
        json["nodes"][1]["op"]["requirements"][0] = requirement;
        assert!(
            WireDag::from_validated_json(&json.to_string()).is_err(),
            "accepted malformed requirement {json}"
        );
    }
    for field in ["site", "requirements", "parameter", "axis"] {
        let mut json = serde_json::to_value(fixture()).unwrap();
        json["nodes"][1]["op"]
            .as_object_mut()
            .unwrap()
            .remove(field);
        assert!(
            WireDag::from_validated_json(&json.to_string()).is_err(),
            "missing {field}"
        );
    }
    for field in ["shape_deps", "span_id", "merged_spans"] {
        let mut json = serde_json::to_value(fixture()).unwrap();
        json["nodes"][1].as_object_mut().unwrap().remove(field);
        assert!(
            WireDag::from_validated_json(&json.to_string()).is_err(),
            "missing {field}"
        );
    }
}

fn checked_fixture() -> WireDag {
    let mut dag = fixture();
    if let WireRiscOp::ExtentWitness { requirements, .. } = &mut dag.nodes[1].op {
        *requirements = vec![extent(1)];
    }
    dag.nodes[2].op = WireRiscOp::CheckedUnitAxis {
        axis: WireRtAxis::Lit { value: 0 },
    };
    dag.nodes[2].inputs = vec![0, 1];
    dag.nodes[2].shape_deps.clear();
    dag.nodes[2].output_type = WireTensorType {
        dims: vec![WireDimInfo::Lit { size: extent(1) }],
        precision: "f32".into(),
    };
    for id in [3, 4] {
        dag.nodes.push(WireDagNode {
            id,
            op: WireRiscOp::Const { value: integer(2) },
            inputs: vec![],
            output_type: WireTensorType {
                dims: vec![],
                precision: "int64".into(),
            },
            shape_deps: vec![],
            span_id: None,
            merged_spans: vec![],
        });
    }
    dag.nodes.push(WireDagNode {
        id: 5,
        op: WireRiscOp::CheckedReshapeExtent {
            claims: vec!["rows".into()],
            axis: WireRtAxis::Lit { value: 0 },
        },
        inputs: vec![3, 4],
        output_type: WireTensorType {
            dims: vec![],
            precision: "int64".into(),
        },
        shape_deps: vec![2],
        span_id: Some("reshape-call".into()),
        merged_spans: vec![],
    });
    dag.roots = vec![5];
    dag
}

#[test]
fn checked_extent_edges_roundtrip_exactly_and_legacy_versions_are_rejected() {
    let dag = checked_fixture();
    let json = serde_json::to_value(&dag).unwrap();
    let decoded = WireDag::from_validated_json(&json.to_string()).unwrap();
    assert_eq!(serde_json::to_value(decoded).unwrap(), json);
    let mut old = json.clone();
    old["schema_version"] = serde_json::json!(WIRE_DAG_SCHEMA_VERSION - 1);
    assert!(WireDag::from_validated_json(&old.to_string()).is_err());
    for (node, field) in [(5, "claims"), (5, "axis"), (2, "axis")] {
        let mut missing = json.clone();
        missing["nodes"][node]["op"]
            .as_object_mut()
            .unwrap()
            .remove(field);
        assert!(
            WireDag::from_validated_json(&missing.to_string()).is_err(),
            "missing node {node} field {field}"
        );
    }
}

#[test]
fn multiple_claim_edges_are_bijective_with_labels_on_the_wire() {
    let mut dag = checked_fixture();
    let WireRiscOp::CheckedReshapeExtent { claims, .. } = &mut dag.nodes[5].op else {
        unreachable!()
    };
    claims.push("outer".into());
    dag.nodes[5].inputs.push(3);
    let json = serde_json::to_value(&dag).unwrap();
    assert_eq!(
        serde_json::to_value(WireDag::from_validated_json(&json.to_string()).unwrap()).unwrap(),
        json
    );
    let mut missing = dag.clone();
    missing.nodes[5].inputs.pop();
    assert!(missing.validate_wire_contract().is_err());
    let WireRiscOp::CheckedReshapeExtent { claims, .. } = &mut dag.nodes[5].op else {
        unreachable!()
    };
    claims.clear();
    dag.nodes[5].inputs.truncate(1);
    assert!(dag.validate_wire_contract().is_err());
}

#[test]
fn checked_extent_wire_never_accepts_an_unproved_refinement() {
    for mutation in 0..10 {
        let mut dag = checked_fixture();
        match mutation {
            0 => {
                dag.nodes[2].inputs.pop();
            }
            1 => dag.nodes[2].inputs[1] = 0,
            2 => dag.nodes[2].output_type.dims[0] = WireDimInfo::Lit { size: extent(2) },
            3 => dag.nodes[2].output_type.precision = "f64".into(),
            4 => {
                if let WireRiscOp::ExtentWitness { requirements, .. } = &mut dag.nodes[1].op {
                    requirements.clear();
                }
            }
            5 => {
                dag.nodes[2].op = WireRiscOp::CheckedUnitAxis {
                    axis: WireRtAxis::Lit { value: 1 },
                }
            }
            6 => {
                dag.nodes[5].inputs.pop();
            }
            7 => dag.nodes[5].inputs[0] = 2,
            8 => dag.nodes[5].output_type.precision = "f32".into(),
            9 => dag.nodes[5].inputs[1] = 5,
            _ => unreachable!(),
        }
        assert!(dag.validate_wire_contract().is_err(), "mutation {mutation}");
        assert!(
            serde_json::to_value(dag).is_err(),
            "encode mutation {mutation}"
        );
    }
}

#[test]
fn remainder_wire_roundtrip_requires_exact_integer_operands() {
    for prim in ["int8", "int16", "int32", "int64"] {
        let mut dag = fixture();
        for (index, node) in dag.nodes.iter_mut().enumerate() {
            node.op = if index == 2 {
                WireRiscOp::Mod
            } else {
                WireRiscOp::Load {
                    name: format!("x{index}"),
                }
            };
            node.inputs = if index == 2 { vec![0, 1] } else { vec![] };
            node.shape_deps.clear();
            node.output_type = WireTensorType {
                dims: vec![],
                precision: prim.into(),
            };
        }
        let json = serde_json::to_value(&dag).unwrap();
        let decoded = WireDag::from_validated_json(&json.to_string()).unwrap();
        assert!(matches!(decoded.nodes[2].op, WireRiscOp::Mod));
        assert_eq!(serde_json::to_value(decoded).unwrap(), json);
        let mut invalid = dag.clone();
        invalid.nodes[2].output_type.precision = "f32".into();
        assert!(serde_json::to_value(invalid).is_err());
        for mutation in 0..5 {
            let mut bad = json.clone();
            match mutation {
                0 => bad["nodes"][2]["inputs"] = serde_json::json!([0]),
                1 => bad["nodes"][2]["output_type"]["precision"] = "f32".into(),
                2 => bad["nodes"][0]["output_type"]["precision"] = "bool".into(),
                3 => {
                    bad["nodes"][1]["output_type"]["dims"] =
                        serde_json::json!([{"kind":"lit", "size":2}])
                }
                4 => bad["nodes"][2]["inputs"] = serde_json::json!([0, 2]),
                _ => unreachable!(),
            }
            assert!(
                WireDag::from_validated_json(&bad.to_string()).is_err(),
                "{bad}"
            );
        }
    }
}

/// wire v11 / chelis#1374: a witness's NAMED claim survives transport with its
/// binder, its role and its requirement edge intact.
///
/// The claim sits on the LATER witness and its edge names the earlier one,
/// which is what `spec/04-type-system.md` §4.7's "due at the later of its two
/// witnesses" means on the wire. `requirement_declares` is carried rather than
/// re-derived because either side can be the later one: a result
/// `-> tensor[rows, cols]` whose set axis reads `shape(x, 0)` declares `cols`
/// on the later parameter.
///
/// EVIDENTIARY STATUS: regression test for the transport; the field did not
/// exist before this change, so the wire could not represent the obligation.
fn named_claim_fixture() -> WireDag {
    let node =
        |id: u64, op: WireRiscOp, inputs: Vec<u64>, dims, precision: &str, shape_deps: Vec<u64>| {
            WireDagNode {
                id,
                op,
                inputs,
                output_type: WireTensorType {
                    dims,
                    precision: precision.into(),
                },
                shape_deps,
                span_id: None,
                merged_spans: vec![],
            }
        };
    let named = |name: &str| {
        vec![WireDimInfo::Named {
            name: name.into(),
            size: None,
        }]
    };
    let witness = |parameter: &str, claims| WireRiscOp::ExtentWitness {
        site: WireExtentWitnessSite::Caller,
        parameter: parameter.into(),
        axis: WireRtAxis::Lit { value: 0 },
        requirements: vec![],
        claims,
    };
    WireDag {
        schema_version: WIRE_DAG_SCHEMA_VERSION,
        roots: vec![4],
        nodes: vec![
            node(
                0,
                WireRiscOp::Load { name: "x".into() },
                vec![],
                named("rows"),
                "f32",
                vec![],
            ),
            node(1, witness("x", vec![]), vec![0], vec![], "int64", vec![]),
            node(
                2,
                WireRiscOp::Load { name: "y".into() },
                vec![],
                named("cols"),
                "f32",
                vec![],
            ),
            node(
                3,
                witness(
                    "y",
                    vec![WireExtentClaim {
                        claim: "rows".into(),
                        requirement_declares: true,
                    }],
                ),
                vec![2, 1],
                vec![],
                "int64",
                vec![],
            ),
            node(
                4,
                WireRiscOp::Const { value: integer(9) },
                vec![],
                vec![],
                "int64",
                vec![3],
            ),
        ],
    }
}

#[test]
fn a_named_witness_claim_round_trips_with_its_binder_role_and_edge() {
    let dag = named_claim_fixture();
    let json = serde_json::to_value(&dag).expect("named claim must encode");
    assert_eq!(
        json["nodes"][3]["op"]["claims"],
        serde_json::json!([{"claim": "rows", "requirement_declares": true}]),
        "{json}"
    );
    assert_eq!(json["nodes"][3]["inputs"], serde_json::json!([2, 1]));
    let text = json.to_string();
    let decoded = WireDag::from_validated_json(&text).expect("named claim must decode");
    assert_eq!(
        serde_json::to_value(&decoded).unwrap(),
        json,
        "transport must be exact in both directions"
    );

    // Negative parity, one mutation per rule the decoder enforces.
    for (name, mutate) in [
        (
            "an empty binder identifies no obligation",
            Box::new(|dag: &mut WireDag| {
                if let WireRiscOp::ExtentWitness { claims, .. } = &mut dag.nodes[3].op {
                    claims[0].claim.clear();
                }
            }) as Box<dyn Fn(&mut WireDag)>,
        ),
        (
            "a claim without its requirement edge",
            Box::new(|dag: &mut WireDag| dag.nodes[3].inputs.truncate(1)),
        ),
        (
            "a requirement edge that is not a witness",
            Box::new(|dag: &mut WireDag| dag.nodes[3].inputs = vec![2, 0]),
        ),
        (
            "a requirement edge that is not earlier",
            Box::new(|dag: &mut WireDag| dag.nodes[3].inputs = vec![2, 3]),
        ),
    ] {
        let mut bad = named_claim_fixture();
        mutate(&mut bad);
        // The encoder validates too, so a malformed claim is refused on the
        // way out as well as on the way in; either refusal satisfies the rule.
        let refused = match serde_json::to_value(&bad) {
            Err(_) => true,
            Ok(value) => WireDag::from_validated_json(&value.to_string()).is_err(),
        };
        assert!(refused, "{name} must be refused, not transported");
    }

    // A missing `claims` field has no default and fails before the body.
    let mut raw: serde_json::Value = serde_json::from_str(&text).unwrap();
    raw["nodes"][3]["op"]
        .as_object_mut()
        .unwrap()
        .remove("claims");
    assert!(
        WireDag::from_validated_json(&raw.to_string()).is_err(),
        "a payload without `claims` must be rejected, never defaulted"
    );
}
