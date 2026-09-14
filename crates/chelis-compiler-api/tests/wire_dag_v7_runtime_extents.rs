use chelis_compiler_api::schema::numbers::NonnegativeExtent;
use chelis_compiler_api::schema::{
    WIRE_DAG_SCHEMA_VERSION, WireDag, WireDagDecodeError, WireDagNode, WireDagSchemaError,
    WireDimInfo, WireRiscOp, WireRtAxis, WireRtDim, WireTensorType,
};

fn ty(dims: &[i64], precision: &str) -> WireTensorType {
    WireTensorType {
        dims: dims
            .iter()
            .copied()
            .map(|size| WireDimInfo::Lit {
                size: NonnegativeExtent::new(size).unwrap(),
            })
            .collect(),
        precision: precision.to_string(),
    }
}

fn load(id: u64, name: &str, dims: &[i64], precision: &str) -> WireDagNode {
    WireDagNode {
        shape_deps: vec![],
        span_id: None,
        merged_spans: vec![],
        id,
        op: WireRiscOp::Load {
            name: name.to_string(),
        },
        inputs: vec![],
        output_type: ty(dims, precision),
    }
}

fn expand_dag(size: WireRtDim, bound: WireDagNode) -> WireDag {
    WireDag {
        schema_version: WIRE_DAG_SCHEMA_VERSION,
        nodes: vec![
            load(0, "value", &[1], "f32"),
            bound,
            WireDagNode {
                shape_deps: vec![],
                span_id: None,
                merged_spans: vec![],
                id: 2,
                op: WireRiscOp::Expand { axis: 0, size },
                inputs: vec![0, 1],
                output_type: ty(&[4], "f32"),
            },
        ],
        roots: vec![2],
    }
}

fn assert_contract_rejects(dag: &WireDag, expected: &str) {
    let encode = serde_json::to_string(dag).expect_err("invalid WireDag must not encode");
    assert!(encode.to_string().contains(expected), "{encode}");

    let raw = serde_json::json!({
        "schema_version": dag.schema_version,
        "nodes": &dag.nodes,
        "roots": &dag.roots,
    })
    .to_string();
    let decode = WireDag::from_validated_json(&raw)
        .expect_err("invalid WireDag must not pass validated decode");
    assert!(matches!(decode, WireDagDecodeError::Contract(_)));
    assert!(decode.to_string().contains(expected), "{decode}");
}

#[test]
fn v7_input_axis_round_trips_as_typed_structure() {
    assert_eq!(WIRE_DAG_SCHEMA_VERSION, 13);
    let dag = expand_dag(
        WireRtDim::InputAxis {
            tensor: 1,
            axis: WireRtAxis::Lit { value: 0 },
        },
        load(1, "witness", &[4], "f32"),
    );

    let json = serde_json::to_string(&dag).expect("valid InputAxis must encode");
    assert!(json.contains(r#""schema_version":12"#), "{json}");
    assert!(
        json.contains(
            r#""size":{"bound":"input_axis","tensor":1,"axis":{"axis":"lit","value":0}}"#
        ),
        "{json}"
    );

    let decoded = WireDag::from_validated_json(&json).expect("valid InputAxis must decode");
    assert!(matches!(
        decoded.nodes[2].op,
        WireRiscOp::Expand {
            size: WireRtDim::InputAxis {
                tensor: 1,
                axis: WireRtAxis::Lit { value: 0 },
            },
            ..
        }
    ));
}

#[test]
fn v7_node_extent_round_trips_only_from_rank_zero_int64() {
    let dag = expand_dag(
        WireRtDim::Node { input: 1 },
        load(1, "extent", &[], "int64"),
    );
    let json = serde_json::to_string(&dag).expect("valid Node extent must encode");
    let decoded = WireDag::from_validated_json(&json).expect("valid Node extent must decode");
    assert!(matches!(
        decoded.nodes[2].op,
        WireRiscOp::Expand {
            size: WireRtDim::Node { input: 1 },
            ..
        }
    ));

    for bad_bound in [
        load(1, "wrong_dtype", &[], "int32"),
        load(1, "wrong_rank", &[1], "int64"),
    ] {
        assert_contract_rejects(
            &expand_dag(WireRtDim::Node { input: 1 }, bad_bound),
            "rank-0 int64",
        );
    }
}

#[test]
fn v7_expand_rejects_forbidden_carriers_slots_and_cardinality() {
    for forbidden in [
        WireRtDim::Sym {
            name: "n".to_string(),
        },
        WireRtDim::ToEnd,
    ] {
        assert_contract_rejects(
            &expand_dag(forbidden, load(1, "witness", &[4], "f32")),
            "forbids",
        );
    }

    let invalid_slot = expand_dag(
        WireRtDim::InputAxis {
            tensor: 2,
            axis: WireRtAxis::Lit { value: 0 },
        },
        load(1, "witness", &[4], "f32"),
    );
    assert_contract_rejects(&invalid_slot, "invalid input slot 2");

    let mut extra_input = expand_dag(
        WireRtDim::Node { input: 1 },
        load(1, "extent", &[], "int64"),
    );
    extra_input.nodes[2].inputs.push(0);
    assert_contract_rejects(&extra_input, "has 3 inputs");
}

#[test]
fn v7_movement_ops_reject_unowned_runtime_extent_inputs() {
    let dag = WireDag {
        schema_version: WIRE_DAG_SCHEMA_VERSION,
        nodes: vec![
            load(0, "value", &[3], "f32"),
            load(1, "extent", &[], "int64"),
            load(2, "unowned", &[], "int64"),
            WireDagNode {
                shape_deps: vec![],
                span_id: None,
                merged_spans: vec![],
                id: 3,
                op: WireRiscOp::Reshape {
                    new_shape: vec![WireRtDim::Node { input: 1 }],
                },
                inputs: vec![0, 1, 2],
                output_type: ty(&[3], "f32"),
            },
        ],
        roots: vec![3],
    };

    assert_contract_rejects(&dag, "unowned runtime extent input slot 2");
}

#[test]
fn v7_input_axis_rejects_negative_or_out_of_range_axes_and_forbidden_owners() {
    for value in [-1, 1] {
        assert_contract_rejects(
            &expand_dag(
                WireRtDim::InputAxis {
                    tensor: 1,
                    axis: WireRtAxis::Lit { value },
                },
                load(1, "witness", &[4], "f32"),
            ),
            if value < 0 {
                "not normalized"
            } else {
                "out of range"
            },
        );
    }

    let pad = WireDag {
        schema_version: WIRE_DAG_SCHEMA_VERSION,
        nodes: vec![
            load(0, "value", &[4], "f32"),
            load(1, "witness", &[4], "f32"),
            WireDagNode {
                shape_deps: vec![],
                span_id: None,
                merged_spans: vec![],
                id: 2,
                op: WireRiscOp::Pad {
                    padding: vec![(
                        WireRtDim::InputAxis {
                            tensor: 1,
                            axis: WireRtAxis::Lit { value: 0 },
                        },
                        WireRtDim::Lit {
                            value: NonnegativeExtent::new(0).unwrap(),
                        },
                    )],
                    fill: chelis_types::scalar_from_f64(
                        "wire_runtime_extent_test",
                        chelis_types::types::Prim::F32,
                        0.0,
                    )
                    .expect("f32 zero"),
                },
                inputs: vec![0, 1],
                output_type: ty(&[8], "f32"),
            },
        ],
        roots: vec![2],
    };
    assert_contract_rejects(&pad, "forbids the input_axis carrier");
}

#[test]
fn v6_display_string_expand_payload_is_rejected_before_op_decode() {
    let old = r#"{"schema_version":6,"nodes":[{"shape_deps":[],"span_id":null,"merged_spans":[],"id":0,"op":{"kind":"expand","axis":0,"size":"4"},"inputs":[],"output_type":{"dims":[],"precision":"f32"}}],"roots":[0]}"#;
    assert!(matches!(
        WireDag::from_validated_json(old),
        Err(WireDagDecodeError::Schema(
            WireDagSchemaError::UnsupportedSchemaVersion {
                found: 6,
                supported: WIRE_DAG_SCHEMA_VERSION,
            }
        ))
    ));

    let stale_spelling = old.replace("\"schema_version\":6", "\"schema_version\":12");
    assert!(matches!(
        WireDag::from_validated_json(&stale_spelling),
        Err(WireDagDecodeError::Parse(_))
    ));
}

/// chelis#1480 / `spec/05` section 2.4.1: a `to_end` end is well formed only
/// beside a literal zero start, and the decoder is one of the stages that
/// validates a bound. The per-carrier check sees one bound at a time and
/// cannot see the pairing, so a `(Lit(1), ToEnd)` shrink decoded cleanly
/// before this.
#[test]
fn v7_shrink_rejects_a_to_end_end_over_a_non_zero_start() {
    let shrink_dag = |start: WireRtDim| WireDag {
        schema_version: WIRE_DAG_SCHEMA_VERSION,
        nodes: vec![
            load(0, "value", &[4], "f32"),
            WireDagNode {
                shape_deps: vec![],
                span_id: None,
                merged_spans: vec![],
                id: 1,
                op: WireRiscOp::Shrink {
                    bounds: vec![(start, WireRtDim::ToEnd)],
                },
                inputs: vec![0],
                output_type: ty(&[4], "f32"),
            },
        ],
        roots: vec![1],
    };

    // The one well-formed spelling still round trips.
    let valid = shrink_dag(WireRtDim::Lit {
        value: NonnegativeExtent::new(0).unwrap(),
    });
    let raw = serde_json::to_string(&valid).expect("the identity slice encodes");
    let decoded = WireDag::from_validated_json(&raw).expect("the identity slice decodes");
    assert_eq!(decoded.nodes.len(), 2);

    assert_contract_rejects(
        &shrink_dag(WireRtDim::Lit {
            value: NonnegativeExtent::new(1).unwrap(),
        }),
        "pairs the to_end carrier with a start that is not literal 0",
    );

    // A `to_end` START keeps its own rejection: the two rules are separate.
    assert_contract_rejects(
        &WireDag {
            schema_version: WIRE_DAG_SCHEMA_VERSION,
            nodes: vec![
                load(0, "value", &[4], "f32"),
                WireDagNode {
                    shape_deps: vec![],
                    span_id: None,
                    merged_spans: vec![],
                    id: 1,
                    op: WireRiscOp::Shrink {
                        bounds: vec![(
                            WireRtDim::ToEnd,
                            WireRtDim::Lit {
                                value: NonnegativeExtent::new(4).unwrap(),
                            },
                        )],
                    },
                    inputs: vec![0],
                    output_type: ty(&[4], "f32"),
                },
            ],
            roots: vec![1],
        },
        "forbids the to_end carrier",
    );
}
