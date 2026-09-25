use chelis_compiler_api::schema::numbers::NonnegativeExtent;
use chelis_compiler_api::schema::{
    WIRE_DAG_SCHEMA_VERSION, WireDag, WireDagDecodeError, WireDagNode, WireDagSchemaError,
    WireDimInfo, WireRiscOp, WireTensorType,
};
use chelis_types::{scalar_from_i64, types::Prim};

fn bool_input() -> WireDagNode {
    WireDagNode {
        shape_deps: vec![],
        span_id: None,
        merged_spans: vec![],
        id: 0,
        op: WireRiscOp::Load {
            name: "mask".to_string(),
        },
        inputs: vec![],
        output_type: WireTensorType {
            dims: vec![
                WireDimInfo::Lit {
                    size: NonnegativeExtent::new(2).unwrap(),
                },
                WireDimInfo::Lit {
                    size: NonnegativeExtent::new(3).unwrap(),
                },
                WireDimInfo::Lit {
                    size: NonnegativeExtent::new(4).unwrap(),
                },
            ],
            precision: "bool".to_string(),
        },
    }
}

fn count_dag(axes: Vec<i32>) -> WireDag {
    WireDag {
        schema_version: WIRE_DAG_SCHEMA_VERSION,
        nodes: vec![
            bool_input(),
            WireDagNode {
                shape_deps: vec![],
                span_id: None,
                merged_spans: vec![],
                id: 1,
                op: WireRiscOp::Count { axes },
                inputs: vec![0],
                output_type: WireTensorType {
                    dims: vec![WireDimInfo::Lit {
                        size: NonnegativeExtent::new(3).unwrap(),
                    }],
                    precision: "int64".to_string(),
                },
            },
        ],
        roots: vec![1],
    }
}

fn assert_contract_rejects_encode_and_decode(dag: &WireDag, expected: &str) {
    let encode_error = serde_json::to_string(dag).expect_err("encoder must reject invalid WireDag");
    assert!(
        encode_error.to_string().contains(expected),
        "{encode_error}"
    );

    let raw_json = serde_json::json!({
        "schema_version": dag.schema_version,
        "nodes": &dag.nodes,
        "roots": &dag.roots,
    })
    .to_string();
    let decode_error = WireDag::from_validated_json(&raw_json)
        .expect_err("validated decode must reject invalid WireDag");
    assert!(matches!(decode_error, WireDagDecodeError::Contract(_)));
    assert!(
        decode_error.to_string().contains(expected),
        "{decode_error}"
    );

    let direct_error = serde_json::from_str::<WireDag>(&raw_json)
        .expect_err("direct decode must enforce the same WireDag contract");
    assert!(
        direct_error.to_string().contains(expected),
        "{direct_error}"
    );
}

#[test]
fn current_wire_dag_count_round_trips_canonical_axes() {
    assert_eq!(WIRE_DAG_SCHEMA_VERSION, 18);
    let dag = count_dag(vec![2, 0]);
    let json = serde_json::to_string(&dag).expect("canonical Count must encode");
    assert!(json.contains(&format!(r#""schema_version":{WIRE_DAG_SCHEMA_VERSION}"#)));
    assert!(json.contains(r#""kind":"count","axes":[2,0]"#));

    let decoded = WireDag::from_validated_json(&json).expect("canonical Count must decode");
    match &decoded.nodes[1].op {
        WireRiscOp::Count { axes } => assert_eq!(axes, &[2, 0]),
        other => panic!("expected Count, got {other:?}"),
    }
}

#[test]
fn current_wire_dag_rejects_missing_older_and_future_versions_before_op_decode() {
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
            r#"{"schema_version":17,"nodes":[{"id":0,"op":{"kind":"not_an_op"},"inputs":[],"output_type":{"dims":[],"precision":"bool"}}],"roots":[0]}"#,
            Some(17),
        ),
        (
            r#"{"schema_version":19,"nodes":[{"id":0,"op":{"kind":"not_an_op"},"inputs":[],"output_type":{"dims":[],"precision":"bool"}}],"roots":[0]}"#,
            Some(19),
        ),
    ];

    for (json, found) in cases {
        match (WireDag::from_validated_json(json), found) {
            (
                Err(WireDagDecodeError::Schema(WireDagSchemaError::MissingSchemaVersion {
                    supported,
                })),
                None,
            ) => assert_eq!(supported, WIRE_DAG_SCHEMA_VERSION),
            (
                Err(WireDagDecodeError::Schema(WireDagSchemaError::UnsupportedSchemaVersion {
                    found: actual,
                    supported,
                })),
                Some(expected),
            ) => {
                assert_eq!(actual, expected);
                assert_eq!(supported, WIRE_DAG_SCHEMA_VERSION);
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

    let current_unknown = r#"{"schema_version":18,"nodes":[{"id":0,"shape_deps":[],"span_id":null,"merged_spans":[],"op":{"kind":"not_an_op"},"inputs":[],"output_type":{"dims":[],"precision":"bool"}}],"roots":[0]}"#;
    assert!(matches!(
        WireDag::from_validated_json(current_unknown),
        Err(WireDagDecodeError::Parse(_))
    ));
}

#[test]
fn current_wire_dag_rejects_every_older_explicit_version_and_legacy_pad() {
    for version in 1..WIRE_DAG_SCHEMA_VERSION {
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
                    supported: WIRE_DAG_SCHEMA_VERSION
                }
            )) if found == version
        ));
    }

    let legacy_pad = r#"{
        "schema_version": 4,
        "nodes": [{
            "id": 0, "shape_deps": [], "span_id": null, "merged_spans": [],
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
                supported: WIRE_DAG_SCHEMA_VERSION
            }
        ))
    ));
}

#[test]
fn current_wire_dag_rejects_noncanonical_count_axes_on_encode_and_decode() {
    for axes in [vec![], vec![0, 2], vec![2, 2], vec![3]] {
        let dag = count_dag(axes.clone());
        let encoded = serde_json::to_string(&dag);
        assert!(
            encoded.is_err(),
            "encoder must reject noncanonical Count axes {axes:?}"
        );

        let value = serde_json::json!({
            "schema_version": WIRE_DAG_SCHEMA_VERSION,
            "nodes": [
                {
                    "id": 0, "shape_deps": [], "span_id": null, "merged_spans": [],
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
                    "id": 1, "shape_deps": [], "span_id": null, "merged_spans": [],
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
fn current_wire_dag_rejects_count_semantic_dtype_and_shape_corruption() {
    let mut wrong_input_dtype = count_dag(vec![2, 0]);
    wrong_input_dtype.nodes[0].output_type.precision = "f32".to_string();
    assert_contract_rejects_encode_and_decode(
        &wrong_input_dtype,
        "Count node 1 input dtype must be bool, found f32",
    );

    let mut wrong_output_dtype = count_dag(vec![2, 0]);
    wrong_output_dtype.nodes[1].output_type.precision = "int32".to_string();
    assert_contract_rejects_encode_and_decode(
        &wrong_output_dtype,
        "Count node 1 output dtype must be int64, found int32",
    );

    let mut wrong_output_shape = count_dag(vec![2, 0]);
    wrong_output_shape.nodes[1].output_type.dims = vec![WireDimInfo::Lit {
        size: NonnegativeExtent::new(4).unwrap(),
    }];
    assert_contract_rejects_encode_and_decode(
        &wrong_output_shape,
        "Count node 1 output dimensions must equal input dimensions with axes removed",
    );
}

#[test]
fn current_wire_dag_rejects_pad_fill_dtype_mismatch_on_encode_and_decode() {
    let fill = scalar_from_i64("wire_pad_mismatch", Prim::Int64, 7).expect("exact int64 Pad fill");
    let dag = WireDag {
        schema_version: WIRE_DAG_SCHEMA_VERSION,
        nodes: vec![WireDagNode {
            shape_deps: vec![],
            span_id: None,
            merged_spans: vec![],
            id: 0,
            op: WireRiscOp::Pad {
                padding: vec![],
                fill,
            },
            inputs: vec![],
            output_type: WireTensorType {
                dims: vec![],
                precision: "f32".to_string(),
            },
        }],
        roots: vec![0],
    };

    let encode_error = serde_json::to_string(&dag)
        .expect_err("encoder must reject a Pad fill whose dtype differs from the output dtype");
    assert!(encode_error.to_string().contains("Pad fill dtype int64"));
    assert!(encode_error.to_string().contains("output dtype f32"));

    let json = serde_json::json!({
        "schema_version": WIRE_DAG_SCHEMA_VERSION,
        "nodes": [{
            "id": 0, "shape_deps": [], "span_id": null, "merged_spans": [],
            "op": {"kind": "pad", "padding": [], "fill": fill},
            "inputs": [],
            "output_type": {"dims": [], "precision": "f32"}
        }],
        "roots": [0]
    })
    .to_string();
    let decode_error = WireDag::from_validated_json(&json)
        .expect_err("validated decode must reject mismatched Pad fill dtype");
    assert!(matches!(decode_error, WireDagDecodeError::Contract(_)));
    assert!(decode_error.to_string().contains("Pad fill dtype int64"));
    assert!(decode_error.to_string().contains("output dtype f32"));

    let direct_error = serde_json::from_str::<WireDag>(&json)
        .expect_err("direct decode must enforce the same Pad fill dtype contract");
    assert!(direct_error.to_string().contains("Pad fill dtype int64"));
    assert!(direct_error.to_string().contains("output dtype f32"));
}

#[test]
fn current_wire_dag_rejects_count_without_one_resolvable_input() {
    for inputs in [vec![], vec![0, 0], vec![99]] {
        let mut dag = count_dag(vec![2, 0]);
        dag.nodes[1].inputs = inputs;
        assert!(serde_json::to_string(&dag).is_err());
    }
}

#[test]
fn current_wire_dag_requires_accumulator_fields_in_current_ops() {
    for op in [
        r#"{"kind":"sum","axis":0,"accumulator":"f32"}"#,
        r#"{"kind":"blas_matmul","batch_dims":[],"m":{"kind":"concrete","value":1},"n":{"kind":"concrete","value":1},"k":{"kind":"concrete","value":1},"accumulator":"f32"}"#,
    ] {
        let op: serde_json::Value = serde_json::from_str(op).unwrap();
        let matmul = op["kind"] == "blas_matmul";
        let dim = serde_json::json!({"kind":"lit","size":1});
        let input_dims = if matmul {
            vec![dim.clone(), dim.clone()]
        } else {
            vec![dim.clone()]
        };
        let output_dims = if matmul { input_dims.clone() } else { vec![] };
        let mut nodes = vec![serde_json::json!({
            "shape_deps":[],"span_id":null,"merged_spans":[],
            "id":0,"op":{"kind":"load","name":"x"},"inputs":[],
            "output_type":{"dims":input_dims,"precision":"f32"}
        })];
        let inputs = if matmul {
            nodes.push(serde_json::json!({
                "shape_deps":[],"span_id":null,"merged_spans":[],
                "id":1,"op":{"kind":"load","name":"y"},"inputs":[],
                "output_type":{"dims":input_dims,"precision":"f32"}
            }));
            vec![0, 1]
        } else {
            vec![0]
        };
        let root = nodes.len();
        nodes.push(serde_json::json!({
            "shape_deps":[],"span_id":null,"merged_spans":[],
            "id":root,"op":op,"inputs":inputs,
            "output_type":{"dims":output_dims,"precision":"f32"}
        }));
        let mut json = serde_json::json!({"schema_version":WIRE_DAG_SCHEMA_VERSION,"nodes":nodes,"roots":[root]});
        WireDag::from_validated_json(&json.to_string())
            .expect("explicit current-version accumulator fields must decode");
        json["nodes"][root]["op"]
            .as_object_mut()
            .unwrap()
            .remove("accumulator");
        assert!(matches!(
            WireDag::from_validated_json(&json.to_string()),
            Err(WireDagDecodeError::Parse(_))
        ));
    }
}
