use chelis_compiler_api::schema::{
    WIRE_DAG_SCHEMA_VERSION, WireDag, WireDagDecodeError, WireDagNode, WireDagSchemaError,
    WireDimInfo, WireRiscOp, WireTensorType,
};

fn bool_input() -> WireDagNode {
    WireDagNode {
        id: 0,
        op: WireRiscOp::Load {
            name: "mask".to_string(),
        },
        inputs: vec![],
        output_type: WireTensorType {
            dims: vec![
                WireDimInfo::Lit { size: 2 },
                WireDimInfo::Lit { size: 3 },
                WireDimInfo::Lit { size: 4 },
            ],
            precision: "bool".to_string(),
        },
    }
}

fn count_dag(axes: Vec<usize>) -> WireDag {
    WireDag {
        schema_version: WIRE_DAG_SCHEMA_VERSION,
        nodes: vec![
            bool_input(),
            WireDagNode {
                id: 1,
                op: WireRiscOp::Count { axes },
                inputs: vec![0],
                output_type: WireTensorType {
                    dims: vec![WireDimInfo::Lit { size: 3 }],
                    precision: "int64".to_string(),
                },
            },
        ],
        roots: vec![1],
    }
}

#[test]
fn wire_dag_v6_count_round_trips_canonical_axes() {
    assert_eq!(WIRE_DAG_SCHEMA_VERSION, 6);
    let dag = count_dag(vec![2, 0]);
    let json = serde_json::to_string(&dag).expect("canonical Count must encode");
    assert!(json.contains(r#""schema_version":6"#));
    assert!(json.contains(r#""kind":"count","axes":[2,0]"#));

    let decoded = WireDag::from_validated_json(&json).expect("canonical Count must decode");
    match &decoded.nodes[1].op {
        WireRiscOp::Count { axes } => assert_eq!(axes, &[2, 0]),
        other => panic!("expected Count, got {other:?}"),
    }
}

#[test]
fn wire_dag_v6_rejects_missing_older_and_future_versions_before_op_decode() {
    let cases = [
        (
            r#"{"nodes":[{"id":0,"op":{"kind":"not_an_op"},"inputs":[],"output_type":{"dims":[],"precision":"bool"}}],"roots":[0]}"#,
            None,
        ),
        (
            r#"{"schema_version":5,"nodes":[{"id":0,"op":{"kind":"not_an_op"},"inputs":[],"output_type":{"dims":[],"precision":"bool"}}],"roots":[0]}"#,
            Some(5),
        ),
        (
            r#"{"schema_version":7,"nodes":[{"id":0,"op":{"kind":"not_an_op"},"inputs":[],"output_type":{"dims":[],"precision":"bool"}}],"roots":[0]}"#,
            Some(7),
        ),
    ];

    for (json, found) in cases {
        match (WireDag::from_validated_json(json), found) {
            (
                Err(WireDagDecodeError::Schema(WireDagSchemaError::MissingSchemaVersion {
                    supported,
                })),
                None,
            ) => assert_eq!(supported, 6),
            (
                Err(WireDagDecodeError::Schema(WireDagSchemaError::UnsupportedSchemaVersion {
                    found: actual,
                    supported,
                })),
                Some(expected),
            ) => {
                assert_eq!(actual, expected);
                assert_eq!(supported, 6);
            }
            (other, _) => panic!("schema mismatch must win before op decode, got {other:?}"),
        }

        let direct = serde_json::from_str::<WireDag>(json)
            .expect_err("direct WireDag decode must enforce the same version gate");
        assert!(
            direct.to_string().contains("WireDag schema version"),
            "version rejection must precede the unknown op error: {direct}"
        );
    }

    let current_unknown = r#"{"schema_version":6,"nodes":[{"id":0,"op":{"kind":"not_an_op"},"inputs":[],"output_type":{"dims":[],"precision":"bool"}}],"roots":[0]}"#;
    assert!(matches!(
        WireDag::from_validated_json(current_unknown),
        Err(WireDagDecodeError::Parse(_))
    ));
}

#[test]
fn wire_dag_v6_rejects_every_older_explicit_version_and_legacy_pad() {
    for version in 1..=5 {
        let mut dag = count_dag(vec![2, 0]);
        dag.schema_version = version;
        assert!(
            serde_json::to_string(&dag).is_err(),
            "encoder must reject stale schema version {version}"
        );

        let json = format!(r#"{{"schema_version":{version},"nodes":[],"roots":[]}}"#);
        assert!(matches!(
            WireDag::from_validated_json(&json),
            Err(WireDagDecodeError::Schema(
                WireDagSchemaError::UnsupportedSchemaVersion {
                    found,
                    supported: 6
                }
            )) if found == version
        ));
    }

    let legacy_pad = r#"{
        "schema_version": 4,
        "nodes": [{
            "id": 0,
            "op": {"kind": "pad", "padding": [], "fill": 1.5},
            "inputs": [],
            "output_type": {"dims": [], "precision": "f32"}
        }],
        "roots": [0]
    }"#;
    assert!(matches!(
        WireDag::from_validated_json(legacy_pad),
        Err(WireDagDecodeError::Schema(
            WireDagSchemaError::UnsupportedSchemaVersion {
                found: 4,
                supported: 6
            }
        ))
    ));
}

#[test]
fn wire_dag_v6_rejects_noncanonical_count_axes_on_encode_and_decode() {
    for axes in [vec![], vec![0, 2], vec![2, 2], vec![3]] {
        let dag = count_dag(axes.clone());
        let encoded = serde_json::to_string(&dag);
        assert!(
            encoded.is_err(),
            "encoder must reject noncanonical Count axes {axes:?}"
        );

        let value = serde_json::json!({
            "schema_version": 6,
            "nodes": [
                {
                    "id": 0,
                    "op": {"kind": "load", "name": "mask"},
                    "inputs": [],
                    "output_type": {
                        "dims": [
                            {"kind": "lit", "size": 2},
                            {"kind": "lit", "size": 3},
                            {"kind": "lit", "size": 4}
                        ],
                        "precision": "bool"
                    }
                },
                {
                    "id": 1,
                    "op": {"kind": "count", "axes": axes},
                    "inputs": [0],
                    "output_type": {
                        "dims": [{"kind": "lit", "size": 3}],
                        "precision": "int64"
                    }
                }
            ],
            "roots": [1]
        });
        assert!(matches!(
            WireDag::from_validated_json(&value.to_string()),
            Err(WireDagDecodeError::Contract(_))
        ));
    }
}

#[test]
fn wire_dag_v6_rejects_count_without_one_resolvable_input() {
    for inputs in [vec![], vec![0, 0], vec![99]] {
        let mut dag = count_dag(vec![2, 0]);
        dag.nodes[1].inputs = inputs;
        assert!(serde_json::to_string(&dag).is_err());
    }
}

#[test]
fn wire_dag_v6_requires_accumulator_fields_in_current_ops() {
    for op in [
        r#"{"kind":"sum","axis":0}"#,
        r#"{"kind":"blas_matmul","batch_dims":[],"m":{"kind":"concrete","value":1},"n":{"kind":"concrete","value":1},"k":{"kind":"concrete","value":1}}"#,
    ] {
        let json = format!(
            r#"{{"schema_version":6,"nodes":[{{"id":0,"op":{op},"inputs":[],"output_type":{{"dims":[],"precision":"f32"}}}}],"roots":[0]}}"#
        );
        assert!(matches!(
            WireDag::from_validated_json(&json),
            Err(WireDagDecodeError::Parse(_))
        ));
    }

    for op in [
        r#"{"kind":"sum","axis":0,"accumulator":"f32"}"#,
        r#"{"kind":"blas_matmul","batch_dims":[],"m":{"kind":"concrete","value":1},"n":{"kind":"concrete","value":1},"k":{"kind":"concrete","value":1},"accumulator":"f32"}"#,
    ] {
        let json = format!(
            r#"{{"schema_version":6,"nodes":[{{"id":0,"op":{op},"inputs":[],"output_type":{{"dims":[],"precision":"f32"}}}}],"roots":[0]}}"#
        );
        WireDag::from_validated_json(&json).expect("explicit v6 accumulator fields must decode");
    }
}
