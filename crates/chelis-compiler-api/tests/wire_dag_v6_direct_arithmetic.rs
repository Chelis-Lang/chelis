use chelis_compiler_api::schema::{
    WIRE_DAG_SCHEMA_VERSION, WireDag, WireDagDecodeError, WireRiscOp,
};

fn direct_sub_payload(version: Option<u32>) -> String {
    let mut payload = serde_json::json!({
        "nodes": [
            {
                "id": 0,
                "op": {"kind": "load", "name": "left"},
                "inputs": [],
                "shape_deps": [], "span_id": null, "merged_spans": [],
                "output_type": {"dims": [], "precision": "f32"}
            },
            {
                "id": 1,
                "op": {"kind": "load", "name": "right"},
                "inputs": [],
                "shape_deps": [], "span_id": null, "merged_spans": [],
                "output_type": {"dims": [], "precision": "f32"}
            },
            {
                "id": 2,
                "op": {"kind": "sub"},
                "inputs": [0, 1],
                "shape_deps": [], "span_id": null, "merged_spans": [],
                "output_type": {"dims": [], "precision": "f32"}
            }
        ],
        "roots": [2]
    });
    if let Some(version) = version {
        payload["schema_version"] = serde_json::Value::from(version);
    }
    payload.to_string()
}

fn count_payload(axes: &[i32]) -> String {
    serde_json::json!({
        "schema_version": WIRE_DAG_SCHEMA_VERSION,
        "nodes": [
            {
                "id": 0,
                "op": {"kind": "load", "name": "mask"},
                "inputs": [],
                "shape_deps": [], "span_id": null, "merged_spans": [],
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
                "shape_deps": [], "span_id": null, "merged_spans": [],
                "output_type": {
                    "dims": [{"kind": "lit", "size": 3}],
                    "precision": "int64"
                }
            }
        ],
        "roots": [1]
    })
    .to_string()
}

fn assert_version_rejected_before_node_decode(payload: &str, expected: &str) {
    let error = WireDag::from_validated_json(payload)
        .expect_err("an inexact WireDag version must fail before node decode");
    assert!(
        matches!(error, WireDagDecodeError::Schema(_)),
        "version rejection must use the schema error channel, got {error:?}"
    );
    assert!(
        error.to_string().contains(expected),
        "version error `{error}` did not identify `{expected}`"
    );
    assert!(
        !error.to_string().contains("not_an_op"),
        "the decoder inspected an operation before rejecting its version: {error}"
    );

    let direct_error = serde_json::from_str::<WireDag>(payload)
        .expect_err("direct WireDag deserialization must enforce the same exact-version gate");
    assert!(
        direct_error.to_string().contains(expected),
        "direct decode error `{direct_error}` did not identify `{expected}`"
    );
    assert!(
        !direct_error.to_string().contains("not_an_op"),
        "direct deserialization inspected an operation before rejecting its version: {direct_error}"
    );
}

#[test]
fn current_exact_wire_round_trips_direct_sub_identity() {
    assert_eq!(WIRE_DAG_SCHEMA_VERSION, 17);
    let payload = direct_sub_payload(Some(WIRE_DAG_SCHEMA_VERSION));
    let decoded = WireDag::from_validated_json(&payload).expect("exact current Sub must decode");
    assert!(matches!(decoded.nodes[2].op, WireRiscOp::Sub));

    let encoded = serde_json::to_value(&decoded).expect("exact current Sub must re-encode");
    assert_eq!(
        encoded["schema_version"],
        serde_json::json!(WIRE_DAG_SCHEMA_VERSION)
    );
    assert_eq!(encoded["nodes"][2]["op"]["kind"], "sub");
}

#[test]
fn missing_older_and_future_versions_fail_before_node_decode() {
    let unknown_op = |version: Option<u32>| {
        let mut payload = serde_json::json!({
            "nodes": [{
                "id": 0,
                "op": {"kind": "not_an_op"},
                "inputs": [],
                "output_type": {"dims": [], "precision": "bool"}
            }],
            "roots": [0]
        });
        if let Some(version) = version {
            payload["schema_version"] = serde_json::Value::from(version);
        }
        payload.to_string()
    };

    assert_version_rejected_before_node_decode(&unknown_op(None), "missing");
    for version in 1..=11 {
        assert_version_rejected_before_node_decode(
            &unknown_op(Some(version)),
            &version.to_string(),
        );
    }
    let future = WIRE_DAG_SCHEMA_VERSION + 1;
    assert_version_rejected_before_node_decode(&unknown_op(Some(future)), &future.to_string());
}

#[test]
fn v5_payload_cannot_smuggle_the_v6_only_sub_identity() {
    assert_version_rejected_before_node_decode(&direct_sub_payload(Some(5)), "5");
}

#[test]
fn current_wire_includes_the_canonical_count_form_owned_by_issue_1287() {
    let payload = count_payload(&[2, 0]);
    let decoded = WireDag::from_validated_json(&payload).expect("canonical Count must decode");
    let encoded = serde_json::to_value(&decoded).expect("canonical Count must re-encode");
    assert_eq!(encoded["nodes"][1]["op"]["kind"], "count");
    assert_eq!(encoded["nodes"][1]["op"]["axes"], serde_json::json!([2, 0]));

    for axes in [vec![], vec![0, 2], vec![2, 2], vec![3]] {
        let error = WireDag::from_validated_json(&count_payload(&axes))
            .expect_err("noncanonical or out-of-range Count axes must reject");
        let message = error.to_string();
        let names_axis_failure = message.contains("axis") || message.contains("axes");
        assert!(
            names_axis_failure
                && if axes == [3] {
                    message.contains("input rank")
                } else {
                    message.contains("Count")
                },
            "Count axes {axes:?} produced the wrong rejection: {error}"
        );
    }
}

#[test]
fn current_wire_has_no_legacy_pad_migration_or_raw_fill_spelling() {
    let legacy = r#"{
        "schema_version": 4,
        "nodes": [{
            "id": 0,
            "op": {"kind": "pad", "padding": [], "fill": 1.5},
            "inputs": [],
            "output_type": {"dims": [], "precision": "f32"}
        }],
        "roots": [0]
    }"#;
    assert_version_rejected_before_node_decode(legacy, "4");

    let mut raw_current: serde_json::Value = serde_json::from_str(legacy).unwrap();
    raw_current["schema_version"] = WIRE_DAG_SCHEMA_VERSION.into();
    raw_current["nodes"][0]["shape_deps"] = serde_json::json!([]);
    raw_current["nodes"][0]["span_id"] = serde_json::Value::Null;
    raw_current["nodes"][0]["merged_spans"] = serde_json::json!([]);
    let raw_current = raw_current.to_string();
    let error = WireDag::from_validated_json(&raw_current)
        .expect_err("current Pad.fill must use an exact ScalarValue payload");
    assert!(matches!(error, WireDagDecodeError::Parse(_)));
    assert!(error.to_string().contains("floating point"), "{error}");
    assert!(serde_json::from_str::<WireDag>(&raw_current).is_err());
}
