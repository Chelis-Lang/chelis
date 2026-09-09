//! [10] §3: exact claim, invocation dependency and provenance transport.
use chelis_compiler_api::schema::*;
use chelis_types::{scalar_from_i64, types::Prim};
fn integer(value: i64) -> chelis_types::ScalarValue {
    scalar_from_i64("load", Prim::Int64, value).unwrap()
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
                    requirements: vec![integer(4)],
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
                shape_deps: vec![integer(1)],
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
    let decoded = WireDag::from_validated_json(&json.to_string()).unwrap();
    assert_eq!(serde_json::to_value(decoded).unwrap(), json);
}
#[test]
fn malformed_claims_and_invocation_edges_are_not_decoded_or_encoded() {
    for mutation in 0..7 {
        let mut dag = fixture();
        match mutation {
            0 => {
                if let WireRiscOp::ExtentWitness { requirements, .. } = &mut dag.nodes[1].op {
                    requirements[0] = integer(-1);
                }
            }
            1 => {
                if let WireRiscOp::ExtentWitness { requirements, .. } = &mut dag.nodes[1].op {
                    requirements[0] = scalar_from_i64("load", Prim::Int32, 4).unwrap();
                }
            }
            2 => {
                if let WireRiscOp::ExtentWitness { axis, .. } = &mut dag.nodes[1].op {
                    *axis = WireRtAxis::Lit { value: 1 };
                }
            }
            3 => dag.nodes[1].inputs.clear(),
            4 => dag.nodes[1].output_type.precision = "f32".into(),
            5 => dag.nodes[2].shape_deps[0] = integer(2),
            6 => dag.nodes[2].shape_deps[0] = integer(-1),
            _ => unreachable!(),
        }
        assert!(dag.validate_wire_contract().is_err(), "mutation {mutation}");
        assert!(
            serde_json::to_value(dag).is_err(),
            "encode mutation {mutation}"
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
