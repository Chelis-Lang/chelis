use chelis_compiler_api::schema::{
    WIRE_DAG_SCHEMA_VERSION, WireComparisonKind, WireDag, WireDagDecodeError, WireLogicalKind,
    WireRiscOp,
};

fn payload(op: serde_json::Value, inputs: Vec<u64>, precision: &str) -> String {
    let mut nodes = vec![
        serde_json::json!({
            "id": 0, "op": {"kind": "load", "name": "a"}, "inputs": [],
            "shape_deps": [], "span_id": null, "merged_spans": [],
            "output_type": {"dims": [{"kind": "lit", "size": 2}], "precision": precision}
        }),
        serde_json::json!({
            "id": 1, "op": {"kind": "load", "name": "b"}, "inputs": [],
            "shape_deps": [], "span_id": null, "merged_spans": [],
            "output_type": {"dims": [{"kind": "lit", "size": 2}], "precision": precision}
        }),
    ];
    if inputs.len() == 3 {
        nodes.insert(
            0,
            serde_json::json!({
                "id": 0, "op": {"kind": "load", "name": "condition"}, "inputs": [],
                "shape_deps": [], "span_id": null, "merged_spans": [],
                "output_type": {"dims": [{"kind": "lit", "size": 2}], "precision": "bool"}
            }),
        );
        nodes[1]["id"] = 1.into();
        nodes[2]["id"] = 2.into();
    }
    let id = nodes.len() as u64;
    nodes.push(serde_json::json!({
        "id": id, "op": op, "inputs": inputs,
        "shape_deps": [], "span_id": null, "merged_spans": [],
        "output_type": {
            "dims": [{"kind": "lit", "size": 2}],
            "precision": if precision == "bool" || inputs.len() == 3 { precision } else { "bool" }
        }
    }));
    serde_json::json!({
        "schema_version": WIRE_DAG_SCHEMA_VERSION,
        "nodes": nodes,
        "roots": [id]
    })
    .to_string()
}

#[test]
fn wire_v15_round_trips_direct_comparison_logical_and_where_vocabulary() {
    assert_eq!(WIRE_DAG_SCHEMA_VERSION, 15);
    for comparison in ["cmp_lt", "lt", "eq", "neq", "gt", "gte", "lte"] {
        let decoded = WireDag::from_validated_json(&payload(
            serde_json::json!({"kind": "compare", "comparison": comparison}),
            vec![0, 1],
            "f32",
        ))
        .unwrap();
        assert!(matches!(decoded.nodes[2].op, WireRiscOp::Compare { .. }));
        assert_eq!(
            serde_json::to_value(&decoded).unwrap()["nodes"][2]["op"],
            serde_json::json!({"kind": "compare", "comparison": comparison})
        );
    }

    for logical in ["and", "or", "not"] {
        let inputs = if logical == "not" {
            vec![0]
        } else {
            vec![0, 1]
        };
        let decoded = WireDag::from_validated_json(&payload(
            serde_json::json!({"kind": "logical", "logical": logical}),
            inputs,
            "bool",
        ))
        .unwrap();
        assert!(matches!(decoded.nodes[2].op, WireRiscOp::Logical { .. }));
    }

    let decoded = WireDag::from_validated_json(&payload(
        serde_json::json!({"kind": "where"}),
        vec![0, 1, 2],
        "f32",
    ))
    .unwrap();
    assert!(matches!(decoded.nodes[3].op, WireRiscOp::Where {}));
}

#[test]
fn wire_enums_are_closed_and_have_exact_spellings() {
    for (kind, spelling) in [
        (WireComparisonKind::CmpLt, "cmp_lt"),
        (WireComparisonKind::Lt, "lt"),
        (WireComparisonKind::Eq, "eq"),
        (WireComparisonKind::Neq, "neq"),
        (WireComparisonKind::Gt, "gt"),
        (WireComparisonKind::Gte, "gte"),
        (WireComparisonKind::Lte, "lte"),
    ] {
        assert_eq!(serde_json::to_value(kind).unwrap(), spelling);
    }
    for (kind, spelling) in [
        (WireLogicalKind::And, "and"),
        (WireLogicalKind::Or, "or"),
        (WireLogicalKind::Not, "not"),
    ] {
        assert_eq!(serde_json::to_value(kind).unwrap(), spelling);
    }
    for malformed in [
        serde_json::json!({"kind": "cmp_lt"}),
        serde_json::json!({"kind": "compare", "comparison": "ne"}),
        serde_json::json!({"kind": "logical", "logical": "xor"}),
        serde_json::json!({"kind": "where", "comparison": "eq"}),
        serde_json::json!({
            "kind": "fused_elem",
            "ops": [{
                "op": "cmp_lt",
                "input_indices": [
                    {"kind": "external", "index": 0},
                    {"kind": "external", "index": 1}
                ]
            }]
        }),
    ] {
        assert!(
            serde_json::from_value::<WireRiscOp>(malformed.clone()).is_err(),
            "malformed WireRiscOp unexpectedly decoded: {malformed}"
        );
    }
}

#[test]
fn wire_contract_rejects_bad_arity_domains_shapes_and_outputs() {
    for (op, inputs, precision, needle) in [
        (
            serde_json::json!({"kind": "compare", "comparison": "eq"}),
            vec![0],
            "f32",
            "inputs",
        ),
        (
            serde_json::json!({"kind": "logical", "logical": "and"}),
            vec![0, 1],
            "f32",
            "bool",
        ),
        (
            serde_json::json!({"kind": "compare", "comparison": "eq"}),
            vec![0, 1],
            "string",
            "numeric or bool",
        ),
        (
            serde_json::json!({"kind": "compare", "comparison": "eq"}),
            vec![0, 1],
            "f8e4m3",
            "active numeric",
        ),
        (
            serde_json::json!({"kind": "where"}),
            vec![0, 1],
            "f32",
            "inputs",
        ),
    ] {
        let error = WireDag::from_validated_json(&payload(op, inputs, precision)).unwrap_err();
        assert!(
            matches!(error, WireDagDecodeError::Contract(_)),
            "expected contract error, got {error:?}"
        );
        assert!(
            error
                .to_string()
                .to_ascii_lowercase()
                .contains(&needle.to_ascii_lowercase()),
            "{error}"
        );
    }
}

#[test]
fn version_14_rejects_before_inspecting_the_new_operation() {
    let mut encoded: serde_json::Value = serde_json::from_str(&payload(
        serde_json::json!({"kind": "compare", "comparison": "not_a_comparison"}),
        vec![0, 1],
        "f32",
    ))
    .unwrap();
    encoded["schema_version"] = 14.into();
    let error = WireDag::from_validated_json(&encoded.to_string()).unwrap_err();
    assert!(matches!(error, WireDagDecodeError::Schema(_)));
    assert!(!error.to_string().contains("not_a_comparison"));
}
