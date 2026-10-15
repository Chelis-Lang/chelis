use chelis_compiler_api::schema::{WIRE_DAG_SCHEMA_VERSION, WireDag};
use serde_json::{Value, json};
fn graph(p: &str, direction: &str) -> Value {
    json!({"schema_version":WIRE_DAG_SCHEMA_VERSION,"declarations":["scan"],"roots":[1],"nodes":[
{"id":0,"declaration":0,"activation":null,"shape_deps":[],"span_id":null,"merged_spans":[],"inputs":[],"op":{"kind":"load","name":"x"},"output_type":{"dims":[{"kind":"lit","size":3}],"precision":p}},
{"id":1,"declaration":0,"activation":null,"shape_deps":[],"span_id":null,"merged_spans":[],"inputs":[0],"op":{"kind":"cumsum","axis":0,"direction":direction},"output_type":{"dims":[{"kind":"lit","size":3}],"precision":p}}]})
}
#[test]
fn scan_wire_roundtrip_and_version_rejection() {
    assert_eq!(WIRE_DAG_SCHEMA_VERSION, 30);
    for p in ["f16", "bf16", "f32", "f64"] {
        for d in ["increasing", "decreasing"] {
            let value = graph(p, d);
            let decoded = WireDag::from_validated_json(&value.to_string()).unwrap();
            assert_eq!(serde_json::to_value(decoded).unwrap(), value);
        }
    }
    let mut missing = graph("f32", "increasing");
    missing.as_object_mut().unwrap().remove("schema_version");
    assert!(WireDag::from_validated_json(&missing.to_string()).is_err());
    for version in [29, WIRE_DAG_SCHEMA_VERSION + 1] {
        let mut value = graph("f32", "increasing");
        value["schema_version"] = json!(version);
        assert!(WireDag::from_validated_json(&value.to_string()).is_err());
    }
}
#[test]
fn scan_wire_rejects_wrong_axis_dtype_shape_arity_and_direction() {
    let good = graph("f32", "increasing");
    for (pointer, value) in [
        ("/nodes/1/op/axis", json!(-1)),
        ("/nodes/1/op/axis", json!(1)),
        ("/nodes/1/op/direction", json!("sideways")),
        ("/nodes/1/inputs", json!([])),
        ("/nodes/1/output_type/precision", json!("f64")),
        ("/nodes/1/output_type/dims/0/size", json!(4)),
    ] {
        let mut bad = good.clone();
        *bad.pointer_mut(pointer).unwrap() = value;
        assert!(
            WireDag::from_validated_json(&bad.to_string()).is_err(),
            "{bad}"
        );
    }
    assert!(WireDag::from_validated_json(&graph("bool", "increasing").to_string()).is_err());
    assert!(WireDag::from_validated_json(&graph("i32", "decreasing").to_string()).is_err());
}
