use chelis_compiler_api::schema::{
    WIRE_DAG_SCHEMA_VERSION, WireComparisonKind, WireDag, WireDagDecodeError, WireLogicalKind,
    WireRiscOp,
};
use chelis_types::types::Prim;

fn payload(op: serde_json::Value, inputs: Vec<u64>, precision: &str) -> String {
    let mut nodes = vec![
        serde_json::json!({
            "id": 0, "op": {"kind": "load", "name": "a"}, "inputs": [],
            "shape_deps": [], "span_id": null, "merged_spans": [], "declaration": 0, "activation": null,
            "output_type": {"dims": [{"kind": "lit", "size": 2}], "precision": precision}
        }),
        serde_json::json!({
            "id": 1, "op": {"kind": "load", "name": "b"}, "inputs": [],
            "shape_deps": [], "span_id": null, "merged_spans": [], "declaration": 0, "activation": null,
            "output_type": {"dims": [{"kind": "lit", "size": 2}], "precision": precision}
        }),
    ];
    if inputs.len() == 3 {
        nodes.insert(
            0,
            serde_json::json!({
                "id": 0, "op": {"kind": "load", "name": "condition"}, "inputs": [],
                "shape_deps": [], "span_id": null, "merged_spans": [], "declaration": 0, "activation": null,
                "output_type": {"dims": [{"kind": "lit", "size": 2}], "precision": "bool"}
            }),
        );
        nodes[1]["id"] = 1.into();
        nodes[2]["id"] = 2.into();
    }
    let id = nodes.len() as u64;
    nodes.push(serde_json::json!({
        "id": id, "op": op, "inputs": inputs,
        "shape_deps": [], "span_id": null, "merged_spans": [], "declaration": 0, "activation": null,
        "output_type": {
            "dims": [{"kind": "lit", "size": 2}],
            "precision": if precision == "bool" || inputs.len() == 3 { precision } else { "bool" }
        }
    }));
    serde_json::json!({
        "schema_version": WIRE_DAG_SCHEMA_VERSION,
        "declarations": ["entry"],
        "nodes": nodes,
        "roots": [id]
    })
    .to_string()
}

fn payload_value(op: serde_json::Value, inputs: Vec<u64>, precision: &str) -> serde_json::Value {
    serde_json::from_str(&payload(op, inputs, precision)).unwrap()
}

fn dims(size: u64) -> serde_json::Value {
    serde_json::json!([{"kind": "lit", "size": size}])
}

fn named_dims(name: &str, size: Option<u64>) -> serde_json::Value {
    serde_json::json!([{"kind": "named", "name": name, "size": size}])
}

fn assert_valid(payload: &serde_json::Value) {
    WireDag::from_validated_json(&payload.to_string()).unwrap();
}

fn assert_contract_error(payload: &serde_json::Value, expected: &str) {
    match WireDag::from_validated_json(&payload.to_string()).unwrap_err() {
        WireDagDecodeError::Contract(error) => assert_eq!(error.to_string(), expected),
        other => panic!("expected contract error, got {other:?}"),
    }
}

fn wire_node(
    id: u64,
    name: &str,
    dims: serde_json::Value,
    precision: &str,
    shape_deps: Vec<u64>,
) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "op": {"kind": "load", "name": name},
        "inputs": [],
        "shape_deps": shape_deps,
        "span_id": null,
        "merged_spans": [], "declaration": 0, "activation": null,
        "output_type": {"dims": dims, "precision": precision}
    })
}

fn wire_operation_node(
    id: u64,
    op: serde_json::Value,
    inputs: Vec<u64>,
    dims: serde_json::Value,
    precision: &str,
) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "op": op,
        "inputs": inputs,
        "shape_deps": [],
        "span_id": null,
        "merged_spans": [], "declaration": 0, "activation": null,
        "output_type": {"dims": dims, "precision": precision}
    })
}

fn wire_dag_payload(nodes: Vec<serde_json::Value>, root: u64) -> serde_json::Value {
    serde_json::json!({
        "schema_version": WIRE_DAG_SCHEMA_VERSION,
        "declarations": ["entry"],
        "nodes": nodes,
        "roots": [root]
    })
}

#[test]
fn wire_v15_round_trips_direct_comparison_logical_and_where_vocabulary() {
    assert_eq!(WIRE_DAG_SCHEMA_VERSION, 23);
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
fn wire_v15_comparison_validation_has_direct_positive_negative_parity() {
    const CONTRACT: &str = "WireDag Compare node 2 requires two same-shape, same-precision active numeric or bool operands and a same-shape Bool output; ordered comparisons require active numeric operands";

    let valid = payload_value(
        serde_json::json!({"kind": "compare", "comparison": "eq"}),
        vec![0, 1],
        "f32",
    );
    assert_valid(&valid);

    let mut resolved_shape_spellings = valid.clone();
    resolved_shape_spellings["nodes"][0]["output_type"]["dims"] = named_dims("left", Some(2));
    resolved_shape_spellings["nodes"][2]["output_type"]["dims"] = named_dims("output", Some(2));
    assert_valid(&resolved_shape_spellings);

    let mut producer_actualized_shape = valid.clone();
    producer_actualized_shape["nodes"][0]["output_type"]["dims"] = named_dims("runtime", None);
    producer_actualized_shape["nodes"][1]["output_type"]["dims"] = named_dims("", None);
    producer_actualized_shape["nodes"][1]["shape_deps"] = serde_json::json!([0]);
    producer_actualized_shape["nodes"][2]["output_type"]["dims"] = named_dims("runtime", None);
    assert_valid(&producer_actualized_shape);

    let mut distinct_unresolved_symbols = valid.clone();
    distinct_unresolved_symbols["nodes"][0]["output_type"]["dims"] = named_dims("batch", None);
    distinct_unresolved_symbols["nodes"][1]["output_type"]["dims"] = named_dims("sequence", None);
    distinct_unresolved_symbols["nodes"][2]["output_type"]["dims"] = named_dims("batch", None);
    assert_contract_error(&distinct_unresolved_symbols, CONTRACT);

    let mut output_shape = valid.clone();
    output_shape["nodes"][2]["output_type"]["dims"] = dims(3);
    assert_contract_error(&output_shape, CONTRACT);

    let mut output_dtype = valid.clone();
    output_dtype["nodes"][2]["output_type"]["precision"] = "f32".into();
    assert_contract_error(&output_dtype, CONTRACT);

    let equality_f8e4m3 = payload_value(
        serde_json::json!({"kind": "compare", "comparison": "eq"}),
        vec![0, 1],
        "f8e4m3",
    );
    assert_contract_error(&equality_f8e4m3, CONTRACT);

    let ordered_f8e4m3 = payload_value(
        serde_json::json!({"kind": "compare", "comparison": "gte"}),
        vec![0, 1],
        "f8e4m3",
    );
    assert_contract_error(&ordered_f8e4m3, CONTRACT);

    let bad_arity = payload_value(
        serde_json::json!({"kind": "compare", "comparison": "eq"}),
        vec![0],
        "f32",
    );
    assert_contract_error(
        &bad_arity,
        "WireDag Compare node 2 requires exactly two inputs",
    );
}

#[test]
fn wire_v15_logical_validation_has_direct_positive_negative_parity() {
    const AND_CONTRACT: &str =
        "WireDag Logical node 2 requires exactly 2 same-shape bool input(s) and a Bool output";
    const NOT_CONTRACT: &str =
        "WireDag Logical node 2 requires exactly 1 same-shape bool input(s) and a Bool output";

    let valid_and = payload_value(
        serde_json::json!({"kind": "logical", "logical": "and"}),
        vec![0, 1],
        "bool",
    );
    assert_valid(&valid_and);

    let valid_not = payload_value(
        serde_json::json!({"kind": "logical", "logical": "not"}),
        vec![0],
        "bool",
    );
    assert_valid(&valid_not);

    let mut resolved_shape_spellings = valid_and.clone();
    resolved_shape_spellings["nodes"][0]["output_type"]["dims"] = named_dims("left", Some(2));
    resolved_shape_spellings["nodes"][2]["output_type"]["dims"] = named_dims("output", Some(2));
    assert_valid(&resolved_shape_spellings);

    let mut distinct_unresolved_symbols = valid_and.clone();
    distinct_unresolved_symbols["nodes"][0]["output_type"]["dims"] = named_dims("batch", None);
    distinct_unresolved_symbols["nodes"][1]["output_type"]["dims"] = named_dims("sequence", None);
    distinct_unresolved_symbols["nodes"][2]["output_type"]["dims"] = named_dims("batch", None);
    assert_contract_error(&distinct_unresolved_symbols, AND_CONTRACT);

    let mut input_shape = valid_and.clone();
    input_shape["nodes"][0]["output_type"]["dims"] = dims(3);
    assert_contract_error(&input_shape, AND_CONTRACT);

    let mut input_dtype = valid_and.clone();
    input_dtype["nodes"][0]["output_type"]["precision"] = "f32".into();
    assert_contract_error(&input_dtype, AND_CONTRACT);

    let mut output_shape = valid_and.clone();
    output_shape["nodes"][2]["output_type"]["dims"] = dims(3);
    assert_contract_error(&output_shape, AND_CONTRACT);

    let mut output_dtype = valid_and.clone();
    output_dtype["nodes"][2]["output_type"]["precision"] = "f32".into();
    assert_contract_error(&output_dtype, AND_CONTRACT);

    let and_bad_arity = payload_value(
        serde_json::json!({"kind": "logical", "logical": "and"}),
        vec![0],
        "bool",
    );
    assert_contract_error(&and_bad_arity, AND_CONTRACT);

    let not_bad_arity = payload_value(
        serde_json::json!({"kind": "logical", "logical": "not"}),
        vec![0, 1],
        "bool",
    );
    assert_contract_error(&not_bad_arity, NOT_CONTRACT);
}

#[test]
fn wire_v15_where_validation_has_direct_positive_negative_parity() {
    const CONTRACT: &str = "WireDag Where node 3 requires a same-shape Bool condition and exactly matching branch/output types";

    let valid = payload_value(serde_json::json!({"kind": "where"}), vec![0, 1, 2], "f32");
    assert_valid(&valid);

    let mut resolved_shape_spellings = valid.clone();
    resolved_shape_spellings["nodes"][0]["output_type"]["dims"] = named_dims("condition", Some(2));
    resolved_shape_spellings["nodes"][1]["output_type"]["dims"] = named_dims("then", Some(2));
    resolved_shape_spellings["nodes"][3]["output_type"]["dims"] = named_dims("output", Some(2));
    assert_valid(&resolved_shape_spellings);

    let mut producer_actualized_anonymous_branch = valid.clone();
    producer_actualized_anonymous_branch["nodes"][0]["output_type"]["dims"] =
        named_dims("runtime", None);
    producer_actualized_anonymous_branch["nodes"][1]["output_type"]["dims"] =
        named_dims("runtime", None);
    producer_actualized_anonymous_branch["nodes"][2]["output_type"]["dims"] = named_dims("", None);
    producer_actualized_anonymous_branch["nodes"][2]["shape_deps"] = serde_json::json!([1]);
    producer_actualized_anonymous_branch["nodes"][3]["output_type"]["dims"] = named_dims("", None);
    assert_valid(&producer_actualized_anonymous_branch);

    for precision in ["f8e4m3", "string"] {
        let inactive = payload_value(
            serde_json::json!({"kind": "where"}),
            vec![0, 1, 2],
            precision,
        );
        assert_contract_error(&inactive, CONTRACT);
    }

    let mut distinct_unresolved_symbols = valid.clone();
    distinct_unresolved_symbols["nodes"][0]["output_type"]["dims"] = named_dims("condition", None);
    distinct_unresolved_symbols["nodes"][1]["output_type"]["dims"] = named_dims("branch", None);
    distinct_unresolved_symbols["nodes"][2]["output_type"]["dims"] =
        named_dims("other_branch", None);
    distinct_unresolved_symbols["nodes"][3]["output_type"]["dims"] = named_dims("branch", None);
    assert_contract_error(&distinct_unresolved_symbols, CONTRACT);

    let mut condition_shape = valid.clone();
    condition_shape["nodes"][0]["output_type"]["dims"] = dims(3);
    assert_contract_error(&condition_shape, CONTRACT);

    let mut condition_dtype = valid.clone();
    condition_dtype["nodes"][0]["output_type"]["precision"] = "f32".into();
    assert_contract_error(&condition_dtype, CONTRACT);

    let mut branch_shape = valid.clone();
    branch_shape["nodes"][2]["output_type"]["dims"] = dims(3);
    assert_contract_error(&branch_shape, CONTRACT);

    let mut branch_dtype = valid.clone();
    branch_dtype["nodes"][2]["output_type"]["precision"] = "f64".into();
    assert_contract_error(&branch_dtype, CONTRACT);

    let mut output_shape = valid.clone();
    output_shape["nodes"][3]["output_type"]["dims"] = dims(3);
    assert_contract_error(&output_shape, CONTRACT);

    let mut output_dtype = valid.clone();
    output_dtype["nodes"][3]["output_type"]["precision"] = "f64".into();
    assert_contract_error(&output_dtype, CONTRACT);

    let bad_arity = payload_value(serde_json::json!({"kind": "where"}), vec![0, 1], "f32");
    assert_contract_error(
        &bad_arity,
        "WireDag Where node 2 requires exactly three inputs",
    );
}

#[test]
fn wire_v15_rejects_unrelated_shape_dependency_authority() {
    const COMPARE_CONTRACT: &str = "WireDag Compare node 3 requires two same-shape, same-precision active numeric or bool operands and a same-shape Bool output; ordered comparisons require active numeric operands";
    const LOGICAL_CONTRACT: &str =
        "WireDag Logical node 3 requires exactly 2 same-shape bool input(s) and a Bool output";
    const WHERE_CONTRACT: &str = "WireDag Where node 4 requires a same-shape Bool condition and exactly matching branch/output types";

    let compare = wire_dag_payload(
        vec![
            wire_node(0, "decoy", named_dims("runtime", None), "f32", vec![]),
            wire_node(1, "left", named_dims("runtime", None), "f32", vec![]),
            wire_node(2, "right", named_dims("", None), "f32", vec![0]),
            wire_operation_node(
                3,
                serde_json::json!({"kind": "compare", "comparison": "eq"}),
                vec![1, 2],
                named_dims("runtime", None),
                "bool",
            ),
        ],
        3,
    );
    assert_contract_error(&compare, COMPARE_CONTRACT);

    let logical = wire_dag_payload(
        vec![
            wire_node(0, "decoy", named_dims("runtime", None), "bool", vec![]),
            wire_node(1, "left", named_dims("runtime", None), "bool", vec![]),
            wire_node(2, "right", named_dims("", None), "bool", vec![0]),
            wire_operation_node(
                3,
                serde_json::json!({"kind": "logical", "logical": "and"}),
                vec![1, 2],
                named_dims("runtime", None),
                "bool",
            ),
        ],
        3,
    );
    assert_contract_error(&logical, LOGICAL_CONTRACT);

    let where_dag = wire_dag_payload(
        vec![
            wire_node(0, "decoy", named_dims("runtime", None), "f32", vec![]),
            wire_node(1, "condition", named_dims("runtime", None), "bool", vec![]),
            wire_node(2, "then", named_dims("runtime", None), "f32", vec![]),
            wire_node(3, "else", named_dims("", None), "f32", vec![0]),
            wire_operation_node(
                4,
                serde_json::json!({"kind": "where"}),
                vec![1, 2, 3],
                named_dims("runtime", None),
                "f32",
            ),
        ],
        4,
    );
    assert_contract_error(&where_dag, WHERE_CONTRACT);
}

#[test]
fn wire_v15_rejects_unresolved_anonymous_shape_authority() {
    const COMPARE_CONTRACT: &str = "WireDag Compare node 2 requires two same-shape, same-precision active numeric or bool operands and a same-shape Bool output; ordered comparisons require active numeric operands";
    const LOGICAL_CONTRACT: &str =
        "WireDag Logical node 2 requires exactly 2 same-shape bool input(s) and a Bool output";

    let producerless_compare = wire_dag_payload(
        vec![
            wire_node(0, "left", named_dims("", None), "f32", vec![]),
            wire_node(1, "right", named_dims("", None), "f32", vec![]),
            wire_operation_node(
                2,
                serde_json::json!({"kind": "compare", "comparison": "eq"}),
                vec![0, 1],
                named_dims("", None),
                "bool",
            ),
        ],
        2,
    );
    assert_contract_error(&producerless_compare, COMPARE_CONTRACT);

    let producerless_logical = wire_dag_payload(
        vec![
            wire_node(0, "left", named_dims("", None), "bool", vec![]),
            wire_node(1, "right", named_dims("", None), "bool", vec![]),
            wire_operation_node(
                2,
                serde_json::json!({"kind": "logical", "logical": "and"}),
                vec![0, 1],
                named_dims("", None),
                "bool",
            ),
        ],
        2,
    );
    assert_contract_error(&producerless_logical, LOGICAL_CONTRACT);

    let mut laundered_compare = producerless_compare;
    laundered_compare["nodes"][2]["shape_deps"] = serde_json::json!([0]);
    assert_contract_error(&laundered_compare, COMPARE_CONTRACT);

    let mut laundered_logical = producerless_logical;
    laundered_logical["nodes"][2]["shape_deps"] = serde_json::json!([0]);
    assert_contract_error(&laundered_logical, LOGICAL_CONTRACT);
}

#[test]
fn wire_v15_rejects_operation_provenance_laundering() {
    let malformed_add = wire_dag_payload(
        vec![
            wire_node(0, "left", named_dims("", None), "f32", vec![]),
            wire_node(1, "unrelated", named_dims("", None), "f32", vec![]),
            wire_operation_node(
                2,
                serde_json::json!({"kind": "add"}),
                vec![0, 1],
                named_dims("", None),
                "f32",
            ),
            serde_json::json!({
                "id": 3,
                "op": {"kind": "compare", "comparison": "eq"},
                "inputs": [2, 0],
                "shape_deps": [0],
                "span_id": null,
                "merged_spans": [], "declaration": 0, "activation": null,
                "output_type": {"dims": named_dims("", None), "precision": "bool"}
            }),
        ],
        3,
    );
    assert_contract_error(
        &malformed_add,
        "WireDag Compare node 3 requires two same-shape, same-precision active numeric or bool operands and a same-shape Bool output; ordered comparisons require active numeric operands",
    );

    let fill = chelis_types::scalar_from_f64("wire Pad provenance", Prim::F32, 0.0).unwrap();
    let nonidentity_pad = wire_dag_payload(
        vec![
            wire_node(0, "input", named_dims("", None), "f32", vec![]),
            serde_json::json!({
                "id": 1,
                "op": {
                    "kind": "pad",
                    "padding": [[
                        {"bound": "lit", "value": 1},
                        {"bound": "lit", "value": 0}
                    ]],
                    "fill": fill
                },
                "inputs": [0],
                "shape_deps": [0],
                "span_id": null,
                "merged_spans": [], "declaration": 0, "activation": null,
                "output_type": {"dims": named_dims("", None), "precision": "f32"}
            }),
            serde_json::json!({
                "id": 2,
                "op": {"kind": "compare", "comparison": "eq"},
                "inputs": [1, 0],
                "shape_deps": [0],
                "span_id": null,
                "merged_spans": [], "declaration": 0, "activation": null,
                "output_type": {"dims": named_dims("", None), "precision": "bool"}
            }),
        ],
        2,
    );
    assert_contract_error(
        &nonidentity_pad,
        "WireDag Compare node 2 requires two same-shape, same-precision active numeric or bool operands and a same-shape Bool output; ordered comparisons require active numeric operands",
    );
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
