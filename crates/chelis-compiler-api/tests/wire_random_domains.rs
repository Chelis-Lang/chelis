//! spec/10 §§3.2–3.4 and [05-OP-8/37]: random wire parameters are admitted
//! before consumption, with exact float dtype, finite domain and complete seed.
//! These tests exercise codecs and object admission, not random kernel output.
use chelis_compiler_api::schema::{
    CheckRequest, LowerRequest, SourceKind, WIRE_DAG_SCHEMA_VERSION, WireDag, WireDagDecodeError,
    WireDagNode,
};
use serde_json::{Value, json};

#[derive(Clone, Copy)]
struct FloatCase {
    dtype: &'static str,
    zero: &'static str,
    negative_zero: &'static str,
    half: &'static str,
    negative_half: &'static str,
    below_one: &'static str,
    one: &'static str,
    infinity: &'static str,
    negative_infinity: &'static str,
    nan: &'static str,
}

const FLOATS: [FloatCase; 4] = [
    FloatCase {
        dtype: "f64",
        zero: "0000000000000000",
        negative_zero: "8000000000000000",
        half: "3fe0000000000000",
        negative_half: "bfe0000000000000",
        below_one: "3fefffffffffffff",
        one: "3ff0000000000000",
        infinity: "7ff0000000000000",
        negative_infinity: "fff0000000000000",
        nan: "7ff8000000000001",
    },
    FloatCase {
        dtype: "f32",
        zero: "00000000",
        negative_zero: "80000000",
        half: "3f000000",
        negative_half: "bf000000",
        below_one: "3f7fffff",
        one: "3f800000",
        infinity: "7f800000",
        negative_infinity: "ff800000",
        nan: "7fc00001",
    },
    FloatCase {
        dtype: "f16",
        zero: "0000",
        negative_zero: "8000",
        half: "3800",
        negative_half: "b800",
        below_one: "3bff",
        one: "3c00",
        infinity: "7c00",
        negative_infinity: "fc00",
        nan: "7e01",
    },
    FloatCase {
        dtype: "bf16",
        zero: "0000",
        negative_zero: "8000",
        half: "3f00",
        negative_half: "bf00",
        below_one: "3f7f",
        one: "3f80",
        infinity: "7f80",
        negative_infinity: "ff80",
        nan: "7fc1",
    },
];

fn scalar(dtype: &str, bits: &str) -> Value {
    json!({"dtype":dtype,"bits":bits})
}

fn graph(float: FloatCase, kind: &str, seed: u64) -> Value {
    let op = if kind == "uniform_like" {
        json!({"kind":kind,"low":scalar(float.dtype,float.negative_zero),"high":scalar(float.dtype,float.one),"seed":seed})
    } else {
        assert_eq!(kind, "dropout");
        json!({"kind":kind,"rate":scalar(float.dtype,float.negative_zero),"seed":seed})
    };
    let ty =
        json!({"dims":[{"kind":"lit","size":2},{"kind":"lit","size":3}],"precision":float.dtype});
    json!({"schema_version":WIRE_DAG_SCHEMA_VERSION,"nodes":[
        {"shape_deps":[],"span_id":null,"merged_spans":[],"id":0,"op":{"kind":"load","name":"x"},"inputs":[],"output_type":ty},
        {"shape_deps":[],"span_id":null,"merged_spans":[],"id":1,"op":op,"inputs":[0],"output_type":ty}
    ],"roots":[1]})
}

fn construct(value: &Value) -> WireDag {
    WireDag {
        schema_version: WIRE_DAG_SCHEMA_VERSION,
        nodes: serde_json::from_value::<Vec<WireDagNode>>(value["nodes"].clone()).unwrap(),
        roots: serde_json::from_value(value["roots"].clone()).unwrap(),
    }
}

fn activated_graph(float: FloatCase, active: bool) -> Value {
    let mut value = graph(float, "uniform_like", 7);
    let nodes = value["nodes"].as_array_mut().unwrap();
    nodes.insert(
        1,
        json!({"shape_deps":[],"span_id":null,"merged_spans":[],
            "id":1,"op":{"kind":"const","value":{"dtype":"bool","value":active}},
            "inputs":[],"output_type":{"dims":[],"precision":"bool"}}),
    );
    nodes[2]["id"] = json!(2);
    nodes[2]["inputs"] = json!([0, 1]);
    value["roots"] = json!([2]);
    value
}

fn accepts(value: &Value) {
    let text = value.to_string();
    let direct: WireDag = serde_json::from_str(&text).unwrap();
    let admitted = WireDag::from_validated_json(&text).unwrap();
    for wire in [direct, admitted, construct(value)] {
        assert_eq!(serde_json::to_value(wire).unwrap(), *value);
    }
}

fn rejects_domain(value: &Value, reason: &str) {
    let text = value.to_string();
    let direct = serde_json::from_str::<WireDag>(&text).unwrap_err();
    assert!(direct.to_string().contains(reason), "{direct}: {text}");
    let admitted = WireDag::from_validated_json(&text).unwrap_err();
    assert!(matches!(admitted, WireDagDecodeError::Contract(_)));
    assert!(admitted.to_string().contains(reason), "{admitted}: {text}");
    let serialization = serde_json::to_value(construct(value)).unwrap_err();
    assert!(
        serialization.to_string().contains(reason),
        "{serialization}: {text}"
    );
}

fn rejects_codec(text: &str) {
    assert!(serde_json::from_str::<WireDag>(text).is_err(), "{text}");
    assert!(
        matches!(
            WireDag::from_validated_json(text),
            Err(WireDagDecodeError::Parse(_))
        ),
        "{text}"
    );
}

const PARAMETERS: [(&str, &str); 3] = [
    ("uniform_like", "low"),
    ("uniform_like", "high"),
    ("dropout", "rate"),
];

#[test]
fn each_random_seed_preserves_all_bits_and_rejects_alternate_or_missing_encodings() {
    for kind in ["uniform_like", "dropout"] {
        for seed in [0, 1, 1_u64 << 63, u64::MAX] {
            accepts(&graph(FLOATS[1], kind, seed));
        }
        let valid = graph(FLOATS[1], kind, 0);
        let text = valid.to_string();
        assert_eq!(text.matches("\"seed\":0").count(), 1);
        for bad in [
            "-1",
            "18446744073709551616",
            "0.0",
            "0e0",
            "null",
            "true",
            "\"0\"",
        ] {
            rejects_codec(&text.replace("\"seed\":0", &format!("\"seed\":{bad}")));
        }
        let mut absent = valid;
        absent["nodes"][1]["op"]
            .as_object_mut()
            .unwrap()
            .remove("seed");
        rejects_codec(&absent.to_string());
    }
}

#[test]
fn each_random_parameter_requires_its_exact_active_float_dtype() {
    for float in FLOATS {
        for (kind, field) in PARAMETERS {
            let valid = graph(float, kind, u64::MAX);
            accepts(&valid);
            let mut wrong = vec![
                json!({"dtype":"int64","value":0}),
                json!({"dtype":"bool","value":false}),
            ];
            wrong.extend(
                FLOATS
                    .iter()
                    .filter(|other| other.dtype != float.dtype)
                    .map(|other| scalar(other.dtype, other.zero)),
            );
            for scalar in wrong {
                let mut changed = valid.clone();
                changed["nodes"][1]["op"][field] = scalar;
                rejects_domain(&changed, "exact active float dtype");
            }
        }
    }
}

#[test]
fn each_random_parameter_rejects_nonfinite_values_at_object_admission() {
    for float in FLOATS {
        for (kind, field) in PARAMETERS {
            let valid = graph(float, kind, 0);
            accepts(&valid);
            for bits in [float.infinity, float.negative_infinity, float.nan] {
                // The canonical codec permits these exact execution-value bits.
                let number = scalar(float.dtype, bits);
                let value: chelis_types::ScalarValue =
                    serde_json::from_value(number.clone()).unwrap();
                assert_eq!(serde_json::to_value(value).unwrap(), number);
                let mut changed = valid.clone();
                changed["nodes"][1]["op"][field] = number;
                rejects_domain(&changed, "must be finite");
            }
        }
    }
}

#[test]
fn each_random_parameter_rejects_malformed_tags_and_payloads() {
    for float in FLOATS {
        for (kind, field) in PARAMETERS {
            let valid = graph(float, kind, 0);
            accepts(&valid);
            for bad in [
                json!(0.0),
                json!(null),
                json!({"dtype":"future","bits":float.zero}),
                json!({"dtype":float.dtype}),
                json!({"dtype":float.dtype,"value":0}),
                json!({"dtype":float.dtype,"bits":"0"}),
            ] {
                let mut changed = valid.clone();
                changed["nodes"][1]["op"][field] = bad;
                rejects_codec(&changed.to_string());
            }
        }
    }
}

#[test]
fn dropout_has_closed_zero_and_open_one_bounds_in_every_float_dtype() {
    for float in FLOATS {
        for bits in [float.zero, float.negative_zero, float.half, float.below_one] {
            let mut valid = graph(float, "dropout", 0);
            valid["nodes"][1]["op"]["rate"] = scalar(float.dtype, bits);
            accepts(&valid);
        }
        for bits in [float.negative_half, float.one] {
            let mut changed = graph(float, "dropout", 0);
            changed["nodes"][1]["op"]["rate"] = scalar(float.dtype, bits);
            rejects_domain(&changed, "0 <= rate < 1");
        }
    }
}

#[test]
fn uniform_accepts_equal_bounds_and_rejects_independently_reversed_low_or_high() {
    for float in FLOATS {
        let mut valid = graph(float, "uniform_like", 0);
        valid["nodes"][1]["op"]["low"] = scalar(float.dtype, float.half);
        valid["nodes"][1]["op"]["high"] = scalar(float.dtype, float.half);
        accepts(&valid);
        for (field, bits) in [("low", float.one), ("high", float.zero)] {
            let mut changed = valid.clone();
            changed["nodes"][1]["op"][field] = scalar(float.dtype, bits);
            rejects_domain(&changed, "bounds must be ordered");
        }
    }
}

#[test]
fn random_operations_preserve_the_data_inputs_exact_shape_and_dtype() {
    for float in FLOATS {
        for kind in ["uniform_like", "dropout"] {
            let valid = graph(float, kind, 0);
            accepts(&valid);
            let mut absent = valid.clone();
            absent["nodes"][1]["inputs"] = json!([]);
            rejects_domain(&absent, "requires an input");
            for field in ["dims", "precision"] {
                for index in [0, 1] {
                    let mut changed = valid.clone();
                    changed["nodes"][index]["output_type"][field] = if field == "dims" {
                        json!([{"kind":"lit","size":2},{"kind":"lit","size":4}])
                    } else {
                        json!(if float.dtype == "f64" { "f32" } else { "f64" })
                    };
                    rejects_domain(&changed, "input's exact shape and dtype");
                }
            }
            let mut rank = valid;
            rank["nodes"][1]["output_type"]["dims"] = json!([{"kind":"lit","size":6}]);
            rejects_domain(&rank, "input's exact shape and dtype");
        }
    }
}

#[test]
fn uniform_accepts_only_an_optional_scalar_bool_activation() {
    // spec/10 §3.2 follows the IR's path-sensitive Random input grammar.
    // Both Bool values, and a runtime Bool input, are valid activations.
    for float in FLOATS {
        accepts(&graph(float, "uniform_like", 7));
        for active in [false, true] {
            accepts(&activated_graph(float, active));
        }
        let mut runtime = activated_graph(float, false);
        runtime["nodes"][1]["op"] = json!({"kind":"load","name":"active"});
        accepts(&runtime);

        for dtype in [
            "f16", "bf16", "f32", "f64", "int8", "int16", "int32", "int64",
        ] {
            let mut wrong = runtime.clone();
            wrong["nodes"][1]["output_type"]["precision"] = json!(dtype);
            rejects_domain(&wrong, "scalar Bool path activation");
        }
        for dims in [
            json!([{"kind":"lit","size":0}]),
            json!([{"kind":"lit","size":1}]),
        ] {
            let mut rankful = runtime.clone();
            rankful["nodes"][1]["output_type"]["dims"] = dims;
            rejects_domain(&rankful, "scalar Bool path activation");
        }
        let mut third = runtime;
        third["nodes"][2]["inputs"] = json!([0, 1, 1]);
        rejects_domain(
            &third,
            "uniform_like expects one template and at most one activation",
        );
    }
}

#[test]
fn uniform_activation_references_must_resolve_to_earlier_nodes() {
    let mut valid = activated_graph(FLOATS[1], true);
    valid["nodes"].as_array_mut().unwrap().swap(0, 1);
    valid["nodes"][0]["id"] = json!(0);
    valid["nodes"][1]["id"] = json!(1);
    valid["nodes"][2]["inputs"] = json!([1, 0]);
    // Absolute node zero is a valid activation, not an absent-input marker.
    accepts(&valid);
    for activation in [2, 3, u64::MAX] {
        let mut wrong = valid.clone();
        wrong["nodes"][2]["inputs"] = json!([1, activation]);
        rejects_domain(&wrong, "input references must resolve to earlier nodes");
    }
    let mut swapped = valid;
    swapped["nodes"][2]["inputs"] = json!([0, 1]);
    rejects_domain(&swapped, "scalar Bool path activation");
}

#[test]
fn uniform_activation_does_not_bypass_template_or_parameter_domains() {
    for float in FLOATS {
        for active in [false, true] {
            let valid = activated_graph(float, active);
            accepts(&valid);
            for index in [0, 2] {
                for field in ["dims", "precision"] {
                    let mut wrong = valid.clone();
                    wrong["nodes"][index]["output_type"][field] = if field == "dims" {
                        json!([{"kind":"lit","size":6}])
                    } else {
                        json!(if float.dtype == "f64" { "f32" } else { "f64" })
                    };
                    rejects_domain(&wrong, "input's exact shape and dtype");
                }
            }
            for field in ["low", "high"] {
                for (number, reason) in [
                    (
                        json!({"dtype":"int64","value":0}),
                        "exact active float dtype",
                    ),
                    (scalar(float.dtype, float.infinity), "must be finite"),
                    (scalar(float.dtype, float.nan), "must be finite"),
                ] {
                    let mut wrong = valid.clone();
                    wrong["nodes"][2]["op"][field] = number;
                    rejects_domain(&wrong, reason);
                }
            }
            let mut reversed = valid;
            reversed["nodes"][2]["op"]["high"] = scalar(float.dtype, float.negative_half);
            rejects_domain(&reversed, "bounds must be ordered");
        }
    }
}

#[test]
fn dropout_rejects_an_activation_input() {
    for float in FLOATS {
        accepts(&graph(float, "dropout", 7));
        for active in [false, true] {
            let mut wrong = activated_graph(float, active);
            wrong["nodes"][2]["op"] = graph(float, "dropout", 7)["nodes"][1]["op"].clone();
            rejects_domain(&wrong, "dropout expects exactly one data input");
        }
    }
}

#[test]
fn gradient_random_lowering_preserves_scalar_bool_activation() {
    // spec/06 §§2.10.1/2.11 and [05-OP-8]: differentiation keeps the
    // handled forward stream, including the IR's Bool path activation.
    let source = concat!(
        "def loss(x: tensor[2, f32]) -> f32 ! { Random } = tensor_to_scalar(sum(mul(copy(x), uniform_like(x, 0.0f32, 1.0f32)), 0))\n",
        "def derivative(x: tensor[2, f32]) -> tensor[2, f32] = with seed(7i64) { grad(loss)(x) }\n"
    );
    let checked = chelis_compiler_api::compiler::check(CheckRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
    })
    .unwrap();
    assert!(checked.errors.is_empty(), "{:?}", checked.errors);
    assert_eq!(serde_json::to_value(checked).unwrap()["score"], json!(1.0));
    let lowered = chelis_compiler_api::compiler::lower(LowerRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        entry: Some("derivative".into()),
    })
    .unwrap();
    let result = serde_json::to_value(lowered).unwrap();
    let dag = &result["dag"];
    assert_eq!(dag["schema_version"], json!(WIRE_DAG_SCHEMA_VERSION));
    let nodes = dag["nodes"].as_array().unwrap();
    let random = nodes
        .iter()
        .find(|node| {
            node["op"]["kind"] == "uniform_like"
                && node["inputs"]
                    .as_array()
                    .is_some_and(|inputs| inputs.len() == 2)
        })
        .expect("gradient lowering must emit a UniformLike with a path activation");
    assert_eq!(random["inputs"].as_array().unwrap().len(), 2);
    let activation = random["inputs"][1].as_u64().unwrap();
    assert!(activation < random["id"].as_u64().unwrap());
    let activation = usize::try_from(activation).unwrap();
    assert_eq!(
        nodes[activation]["output_type"],
        json!({"dims":[],"precision":"bool"})
    );
    accepts(dag);

    // The actual producer's activation still owes wire admission on every
    // encode/decode path; a consumer cannot replace it with a numeric scalar.
    let mut malformed = dag.clone();
    malformed["nodes"][activation]["op"] = json!({"kind":"load","name":"bad_activation"});
    malformed["nodes"][activation]["output_type"]["precision"] = json!("f32");
    rejects_domain(&malformed, "scalar Bool path activation");
}
