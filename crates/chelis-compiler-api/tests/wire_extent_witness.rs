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
                    parameter: "x".into(),
                    axis: WireRtAxis::Lit { value: 0 },
                    requirements: vec![extent(4), extent(4), extent(9)],
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
    for field in ["requirements", "parameter", "axis"] {
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
        *requirements = vec![integer(1)];
    }
    dag.nodes[2].op = WireRiscOp::CheckedUnitAxis {
        axis: WireRtAxis::Lit { value: 0 },
    };
    dag.nodes[2].inputs = vec![0, 1];
    dag.nodes[2].shape_deps.clear();
    dag.nodes[2].output_type = WireTensorType {
        dims: vec![WireDimInfo::Lit { size: 1 }],
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
            claim: "rows".into(),
            axis: WireRtAxis::Lit { value: 0 },
        },
        inputs: vec![3, 4],
        output_type: WireTensorType {
            dims: vec![],
            precision: "int64".into(),
        },
        shape_deps: vec![integer(2)],
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
    for (node, field) in [(5, "claim"), (5, "axis"), (2, "axis")] {
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
fn checked_extent_wire_never_accepts_an_unproved_refinement() {
    for mutation in 0..10 {
        let mut dag = checked_fixture();
        match mutation {
            0 => {
                dag.nodes[2].inputs.pop();
            }
            1 => dag.nodes[2].inputs[1] = 0,
            2 => dag.nodes[2].output_type.dims[0] = WireDimInfo::Lit { size: 2 },
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
