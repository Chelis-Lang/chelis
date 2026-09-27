//! spec/10 §§3.2–3.4: exact numeric domains and references in their owners.
use chelis_compiler_api::schema::{WIRE_DAG_SCHEMA_VERSION, WireDag};
use serde_json::{Value, json};

fn load() -> Value {
    json!({"shape_deps":[],"span_id":null,"merged_spans":[],"declaration":0,"activation":null,"id":0,"op":{"kind":"load","name":"x"},"inputs":[],"output_type":{"dims":[{"kind":"lit","size":2}],"precision":"f32"}})
}
fn graph(op: Value) -> Value {
    let mut value = json!({"schema_version":WIRE_DAG_SCHEMA_VERSION,"declarations":["entry"],"nodes":[load(),{"shape_deps":[],"span_id":null,"merged_spans":[],"declaration":0,"activation":null,"id":1,"op":op,"inputs":[0],"output_type":{"dims":[{"kind":"lit","size":2}],"precision":"f32"}}],"roots":[1]});
    if value["nodes"][1]["op"]["kind"] == "shape" {
        value["nodes"][1]["output_type"] = json!({"dims":[],"precision":"int64"});
    }
    if value["nodes"][1]["op"]["kind"] == "one_hot" {
        value["nodes"][0]["output_type"]["precision"] = json!("int64");
        value["nodes"][1]["output_type"]["dims"] =
            json!([{"kind":"lit","size":2},{"kind":"lit","size":2}]);
    }
    value
}
fn admits(value: &Value) -> bool {
    serde_json::from_value::<WireDag>(value.clone()).is_ok()
}
fn float(bits: &str) -> Value {
    json!({"dtype":"f32","bits":bits})
}

#[test]
fn in_memory_reference_and_numeric_mutations_cannot_be_serialized() {
    use chelis_compiler_api::schema::WireRiscOp;
    let mut wire: WireDag =
        serde_json::from_value(graph(json!({"kind":"shape","axis":0}))).unwrap();
    wire.nodes[1].op = WireRiscOp::Shape { axis: -1 };
    assert!(serde_json::to_string(&wire).is_err());
    wire.nodes[1].op = WireRiscOp::Shape { axis: 0 };
    wire.roots = vec![u64::MAX];
    assert!(serde_json::to_string(&wire).is_err());
}

#[test]
fn live_lowering_emits_exact_control_carriers_as_operands() {
    use chelis_compiler_api::schema::{LowerRequest, SourceKind};
    let lower = |body: &str| {
        chelis_compiler_api::compiler::lower(LowerRequest {
            source_kind: SourceKind::Surf,
            source: format!("def sample(k: key, x: tensor[2, f32]) -> tensor[2, f32] = {body}\n"),
            entry: Some("sample".into()),
        })
    };
    // The bounds are exact f32 carriers on ordinary constant operands of the
    // draw, never fields of the random node.
    let lowered = lower("uniform_like(k, x, 0.1f32, 1.0f32)").unwrap();
    let wire = serde_json::to_value(lowered).unwrap();
    let nodes = wire["dag"]["nodes"].as_array().unwrap();
    let uniform = nodes
        .iter()
        .find(|node| node["op"]["kind"] == "uniform_like")
        .unwrap();
    assert_eq!(uniform["op"], json!({"kind":"uniform_like"}));
    let operand = |slot: usize| {
        let id = usize::try_from(uniform["inputs"][slot].as_u64().unwrap()).unwrap();
        nodes[id]["op"]["value"].clone()
    };
    assert_eq!(operand(1), float("3dcccccd"));
    assert_eq!(operand(2), float("3f800000"));
    // [05-OP-37]: a rate outside [0, 1) is refused by the draw at execution,
    // before it draws; lowering carries it as an operand.
    lower("dropout(k, x, 1.0f32)").unwrap();
}

#[test]
fn graph_references_are_positions_and_resolve_only_to_earlier_nodes() {
    let valid = graph(json!({"kind":"copy"}));
    assert!(admits(&valid));
    for (path, value) in [
        ("id", json!(7)),
        ("inputs", json!([1])),
        ("inputs", json!([u64::MAX])),
    ] {
        let mut changed = valid.clone();
        changed["nodes"][1][path] = value;
        assert!(!admits(&changed), "accepted {changed}");
    }
    let mut roots = valid;
    roots["roots"] = json!([2]);
    assert!(!admits(&roots));
}

#[test]
fn axes_and_extents_enforce_fixed_widths_before_consumption() {
    let valid = graph(json!({"kind":"shape","axis":0}));
    assert!(admits(&valid));
    for axis in [json!(-1), json!(1), json!(2147483648_i64), json!(0.0)] {
        let mut changed = valid.clone();
        changed["nodes"][1]["op"]["axis"] = axis;
        assert!(!admits(&changed), "accepted {changed}");
    }
    for size in [json!(-1), json!(u64::MAX), json!(2.0)] {
        let mut changed = valid.clone();
        changed["nodes"][0]["output_type"]["dims"][0]["size"] = size;
        assert!(!admits(&changed), "accepted {changed}");
    }
}

fn typed_load(id: u64, name: &str, sizes: &[i64], dtype: &str) -> Value {
    let dims = sizes
        .iter()
        .map(|size| json!({"kind":"lit","size":size}))
        .collect::<Vec<_>>();
    json!({"shape_deps":[],"span_id":null,"merged_spans":[],"declaration":0,"activation":null,
        "id":id,"op":{"kind":"load","name":name},"inputs":[],
        "output_type":{"dims":dims,"precision":dtype}})
}

fn axis_graph(kind: &str) -> Value {
    let indexed = matches!(
        kind,
        "gather" | "scatter" | "scatter_add" | "scatter_elements"
    );
    let mut nodes = vec![typed_load(
        0,
        "x",
        if indexed { &[2, 3] } else { &[2] },
        "f32",
    )];
    if indexed {
        nodes.push(typed_load(
            1,
            "indices",
            if kind == "scatter_elements" {
                &[2, 3]
            } else {
                &[1]
            },
            "int64",
        ));
        if kind != "gather" {
            nodes.push(typed_load(
                2,
                "updates",
                if kind == "scatter_elements" {
                    &[2, 3]
                } else {
                    &[1, 3]
                },
                "f32",
            ));
        }
    }
    let mut op = json!({"kind":kind,"axis":0});
    if kind == "sum" {
        op["accumulator"] = json!("f32");
    }
    let id = nodes.len() as u64;
    let output_sizes: &[i64] = match kind {
        "gather" => &[1, 3],
        "scatter" | "scatter_add" | "scatter_elements" => &[2, 3],
        _ => &[],
    };
    let dtype = if matches!(kind, "argmax" | "argmin" | "shape") {
        "int64"
    } else {
        "f32"
    };
    let mut output = typed_load(id, "unused", output_sizes, dtype);
    output["op"] = op;
    output["inputs"] = json!((0..id).collect::<Vec<_>>());
    nodes.push(output);
    json!({"schema_version":WIRE_DAG_SCHEMA_VERSION,"declarations":["entry"],"nodes":nodes,"roots":[id]})
}

#[test]
fn every_registered_single_axis_operation_checks_its_own_input_rank() {
    // spec/10 §3.4 and each operation's exact [05-OP] authority: these
    // are wire parameter admission checks, not complete IR type checking.
    use chelis_compiler_api::schema::{WireDagDecodeError, WireRiscOp};
    for kind in [
        "sum",
        "max_reduce",
        "min_reduce",
        "prod_reduce",
        "argmax",
        "argmin",
        "shape",
        "gather",
        "scatter_add",
        "scatter",
        "scatter_elements",
    ] {
        let valid = axis_graph(kind);
        let mut decoded = WireDag::from_validated_json(&valid.to_string()).unwrap();
        assert_eq!(serde_json::to_value(&decoded).unwrap(), valid, "{kind}");
        let last = decoded.nodes.len() - 1;
        let rank = decoded.nodes[0].output_type.dims.len() as i32;
        for axis in [-1, rank] {
            let mut changed = valid.clone();
            changed["nodes"][last]["op"]["axis"] = json!(axis);
            let error = WireDag::from_validated_json(&changed.to_string()).unwrap_err();
            assert!(
                matches!(error, WireDagDecodeError::Contract(_)),
                "{kind}: {error}"
            );
            assert!(error.to_string().contains("wire axis"), "{kind}: {error}");
            assert!(!admits(&changed), "{kind}: direct decode accepted {axis}");
        }
        for axis in [json!(2147483648_i64), json!(0.0)] {
            let mut changed = valid.clone();
            changed["nodes"][last]["op"]["axis"] = axis;
            assert!(!admits(&changed), "{kind}: accepted non-int32 axis");
        }
        let axis = match &mut decoded.nodes[last].op {
            WireRiscOp::Sum { axis, .. }
            | WireRiscOp::MaxReduce { axis }
            | WireRiscOp::MinReduce { axis }
            | WireRiscOp::ProdReduce { axis }
            | WireRiscOp::Argmax { axis }
            | WireRiscOp::Argmin { axis }
            | WireRiscOp::Shape { axis }
            | WireRiscOp::Gather { axis }
            | WireRiscOp::ScatterAdd { axis }
            | WireRiscOp::Scatter { axis }
            | WireRiscOp::ScatterElements { axis } => axis,
            other => panic!("wrong fixture: {other:?}"),
        };
        *axis = rank;
        let error = serde_json::to_value(&decoded).unwrap_err();
        assert!(error.to_string().contains("wire axis"), "{kind}: {error}");
    }
}

#[test]
fn forward_and_adjoint_window_fields_reject_bad_domains_at_both_edges() {
    // [05-OP-39]: the adjoint takes (x, g), retaining the forward window
    // and stride domains. Its result has x's shape.
    use chelis_compiler_api::schema::{WireDagDecodeError, WireRiscOp, numbers::NonnegativeExtent};
    for kind in ["reduce_window", "reduce_window_grad"] {
        for reducer in ["sum", "mean", "max", "min"] {
            let mut nodes = vec![typed_load(0, "x", &[3], "f32")];
            if kind == "reduce_window_grad" {
                nodes.push(typed_load(1, "g", &[2], "f32"));
            }
            let id = nodes.len() as u64;
            let mut output = typed_load(
                id,
                "unused",
                if kind == "reduce_window_grad" {
                    &[3]
                } else {
                    &[2]
                },
                "f32",
            );
            output["inputs"] = json!((0..id).collect::<Vec<_>>());
            output["op"] = json!({"kind":kind,"reducer":reducer,"window_shape":[2],"strides":[1]});
            nodes.push(output);
            let value = json!({"schema_version":WIRE_DAG_SCHEMA_VERSION,"declarations":["entry"],"nodes":nodes,"roots":[id]});
            let decoded = WireDag::from_validated_json(&value.to_string()).unwrap();
            assert_eq!(serde_json::to_value(&decoded).unwrap(), value);
            let last = decoded.nodes.len() - 1;
            for field in ["window_shape", "strides"] {
                for bad in [json!([0]), json!([]), json!([1, 1])] {
                    let mut changed = value.clone();
                    changed["nodes"][last]["op"][field] = bad;
                    let error = WireDag::from_validated_json(&changed.to_string()).unwrap_err();
                    assert!(matches!(error, WireDagDecodeError::Contract(_)), "{error}");
                    assert!(
                        error.to_string().contains("window shapes and strides"),
                        "{error}"
                    );
                    assert!(!admits(&changed));
                }
                for bad in [json!([-1]), json!([u64::MAX]), json!([1.0])] {
                    let mut changed = value.clone();
                    changed["nodes"][last]["op"][field] = bad;
                    assert!(!admits(&changed), "{kind}/{reducer}/{field}: {changed}");
                }
                let mut changed = decoded.clone();
                let (window, strides) = match &mut changed.nodes[last].op {
                    WireRiscOp::ReduceWindow {
                        window_shape,
                        strides,
                        ..
                    }
                    | WireRiscOp::ReduceWindowGrad {
                        window_shape,
                        strides,
                        ..
                    } => (window_shape, strides),
                    other => panic!("wrong fixture: {other:?}"),
                };
                let values = if field == "window_shape" {
                    window
                } else {
                    strides
                };
                values[0] = NonnegativeExtent::new(0).unwrap();
                let error = serde_json::to_value(&changed).unwrap_err();
                assert!(
                    error.to_string().contains("window shapes and strides"),
                    "{error}"
                );
            }
        }
    }
}

fn expand_graph(input_dims: &[i64], output_dims: &[i64], axis: i32) -> Value {
    let dims = |sizes: &[i64]| {
        sizes
            .iter()
            .map(|size| json!({"kind":"lit","size":size}))
            .collect::<Vec<_>>()
    };
    let mut value = graph(json!({"kind":"expand","axis":axis,"size":{"bound":"lit","value":3}}));
    value["nodes"][0]["output_type"]["dims"] = json!(dims(input_dims));
    value["nodes"][1]["output_type"]["dims"] = json!(dims(output_dims));
    value
}

#[test]
fn shared_expand_ir_accepts_broadcast_and_insert_axis_domains() {
    // spec/10 §3.4: the shared IR variant retains both source operations.
    for (input, output, axis) in [
        (vec![1, 2], vec![3, 2], 0),
        (vec![2, 1], vec![2, 3], 1),
        (vec![], vec![3], 0),
        (vec![2], vec![3, 2], 0),
        (vec![2], vec![2, 3], 1),
    ] {
        let value = expand_graph(&input, &output, axis);
        let decoded: WireDag = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), value);
    }
}

#[test]
fn shared_expand_ir_rejects_axes_outside_its_selected_layout() {
    for (input, output, axis) in [
        (vec![1, 2], vec![3, 2], -1),
        (vec![1, 2], vec![3, 2], 2),
        (vec![], vec![3], -1),
        (vec![], vec![3], 1),
        (vec![2], vec![2, 3], 2),
        (vec![], vec![], 0),
        (vec![2], vec![2, 3, 3], 0),
        (vec![2, 3], vec![3], 0),
    ] {
        let value = expand_graph(&input, &output, axis);
        assert!(!admits(&value), "accepted {value}");
    }
    let mut valid: WireDag = serde_json::from_value(expand_graph(&[], &[3], 0)).unwrap();
    valid.nodes[1].output_type.dims.clear();
    assert!(serde_json::to_value(valid).is_err());
}

#[test]
fn fused_references_reject_nonexistent_inputs_and_future_steps() {
    let step = |input: Value| json!({"op":"neg","input_indices":[input]});
    let valid = graph(
        json!({"kind":"fused_elem","ops":[step(json!({"kind":"external","index":0})),step(json!({"kind":"previous_step","index":0}))]}),
    );
    assert!(admits(&valid));
    for (step_index, input) in [
        (0, json!({"kind":"external","index":1})),
        (0, json!({"kind":"previous_step","index":0})),
        (1, json!({"kind":"previous_step","index":1})),
    ] {
        let mut changed = valid.clone();
        changed["nodes"][1]["op"]["ops"][step_index]["input_indices"][0] = input;
        assert!(!admits(&changed), "accepted {changed}");
    }
}

#[test]
fn window_and_onehot_extents_are_positive_int64() {
    for op in [
        json!({"kind":"reduce_window","reducer":"sum","window_shape":[1],"strides":[1]}),
        json!({"kind":"one_hot","vocab":2}),
    ] {
        assert!(admits(&graph(op.clone())), "rejected {op}");
    }
    for op in [
        json!({"kind":"reduce_window","reducer":"sum","window_shape":[0],"strides":[1]}),
        json!({"kind":"reduce_window","reducer":"sum","window_shape":[1],"strides":[0]}),
        json!({"kind":"one_hot","vocab":0}),
        json!({"kind":"one_hot","vocab":u64::MAX}),
    ] {
        assert!(!admits(&graph(op.clone())), "accepted {op}");
    }
}

#[test]
fn standalone_dimension_carriers_reject_negative_extents_before_dag_admission() {
    use chelis_compiler_api::schema::{WireDimExpr, WireDimInfo, WireRtDim};
    for value in [0_i64, 1, i64::MAX] {
        let dim = json!({"kind":"lit","size":value});
        let parsed: WireDimInfo = serde_json::from_value(dim.clone()).unwrap();
        assert_eq!(serde_json::to_value(parsed).unwrap(), dim);
        let named = json!({"kind":"named","name":"n","size":value});
        let parsed: WireDimInfo = serde_json::from_value(named.clone()).unwrap();
        assert_eq!(serde_json::to_value(parsed).unwrap(), named);
        let expression = json!({"kind":"concrete","value":value});
        let parsed: WireDimExpr = serde_json::from_value(expression.clone()).unwrap();
        assert_eq!(serde_json::to_value(parsed).unwrap(), expression);
        let bound = json!({"bound":"lit","value":value});
        let parsed: WireRtDim = serde_json::from_value(bound.clone()).unwrap();
        assert_eq!(serde_json::to_value(parsed).unwrap(), bound);
    }
    for value in [json!(-1), json!(u64::MAX), json!(1.0), json!(true)] {
        assert!(serde_json::from_value::<WireDimInfo>(json!({"kind":"lit","size":value})).is_err());
        assert!(
            serde_json::from_value::<WireDimInfo>(json!({"kind":"named","name":"n","size":value}))
                .is_err()
        );
        assert!(
            serde_json::from_value::<WireDimExpr>(json!({"kind":"concrete","value":value}))
                .is_err()
        );
        assert!(serde_json::from_value::<WireRtDim>(json!({"bound":"lit","value":value})).is_err());
    }
}
