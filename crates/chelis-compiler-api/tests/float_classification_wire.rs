use chelis_compiler_api::schema::{WIRE_DAG_SCHEMA_VERSION, WireDag, WireRiscOp};
use chelis_numeric_core::dtype_semantics::FloatClassify;
use serde_json::{Value, json};

fn graph(precision: &str, kind: &str) -> Value {
    json!({"schema_version": WIRE_DAG_SCHEMA_VERSION, "declarations": ["entry"], "roots": [1], "nodes": [
        {"id":0,"declaration":0,"activation":null,"shape_deps":[],"span_id":null,"merged_spans":[],"inputs":[],"op":{"kind":"load","name":"x"},"output_type":{"dims":[{"kind":"lit","size":2}],"precision":precision}},
        {"id":1,"declaration":0,"activation":null,"shape_deps":[],"span_id":null,"merged_spans":[],"inputs":[0],"op":{"kind":"float_classify","classification":kind},"output_type":{"dims":[{"kind":"lit","size":2}],"precision":"bool"}}
    ]})
}

#[test]
fn exact_predicate_vocabulary_round_trips_at_every_float_width() {
    for precision in ["f16", "bf16", "f32", "f64"] {
        for (name, kind) in [
            ("nan", FloatClassify::Nan),
            ("finite", FloatClassify::Finite),
            ("infinite", FloatClassify::Infinite),
        ] {
            let source = graph(precision, name);
            let decoded = WireDag::from_validated_json(&source.to_string()).unwrap();
            assert!(
                matches!(decoded.nodes[1].op, WireRiscOp::FloatClassify { classification } if classification == kind)
            );
            assert_eq!(serde_json::to_value(decoded).unwrap(), source);
        }
    }
}

#[test]
fn wrong_dtype_shape_arity_and_discriminator_fail_closed() {
    for precision in [
        "int8", "int16", "int32", "int64", "bool", "string", "f8e4m3",
    ] {
        assert!(WireDag::from_validated_json(&graph(precision, "nan").to_string()).is_err());
    }
    let good = graph("f32", "nan");
    for (pointer, value) in [
        ("/nodes/1/output_type/precision", json!("f32")),
        ("/nodes/1/output_type/dims/0/size", json!(3)),
        ("/nodes/1/inputs", json!([])),
        ("/nodes/1/inputs", json!([0, 0])),
        ("/nodes/1/op/classification", json!("is_nan")),
        ("/schema_version", json!(WIRE_DAG_SCHEMA_VERSION - 1)),
    ] {
        let mut bad = good.clone();
        *bad.pointer_mut(pointer).unwrap() = value;
        assert!(
            WireDag::from_validated_json(&bad.to_string()).is_err(),
            "{pointer}: {bad}"
        );
    }
    let mut missing = good.clone();
    missing["nodes"][1]["op"]
        .as_object_mut()
        .unwrap()
        .remove("classification");
    assert!(WireDag::from_validated_json(&missing.to_string()).is_err());
}
