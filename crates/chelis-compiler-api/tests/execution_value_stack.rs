//! Deep public execution values must survive embedding input decoding and drop.

use chelis_compiler_api::schema::{DictEntryValue, ExecutionValue};
use chelis_compiler_api::{DecodeError, try_decode_adt_value};
use serde::Deserialize;
use std::process::Command;

const CHILD_ENV: &str = "CHELIS_2601_EXECUTION_VALUE_CHILD";

fn nested_list(depth: usize, leaf: ExecutionValue) -> ExecutionValue {
    let mut value = leaf;
    for _ in 0..depth {
        value = ExecutionValue::List { value: vec![value] };
    }
    value
}

fn nested_adt(depth: usize, leaf: ExecutionValue) -> ExecutionValue {
    let mut value = leaf;
    for _ in 0..depth {
        value = ExecutionValue::Adt {
            ctor: "Link".to_string(),
            fields: vec![value],
        };
    }
    value
}

fn nested_list_wire(depth: usize, leaf: &str) -> String {
    format!(
        "{}{leaf}{}",
        "{\"type\":\"list\",\"value\":[".repeat(depth),
        "]}".repeat(depth)
    )
}

fn chain_program() -> Vec<chelis_deep::ast::Expr> {
    let source = "type Chain =\n  | End { value: i64 }\n  | Link { next: Chain }\n";
    let decls = chelis_surf::parser::parse_str(source).expect("parse Chain declaration");
    chelis_surf::desugar::desugar_program(&decls).expect("desugar Chain declaration")
}

fn opaque_carrier_program() -> Vec<chelis_deep::ast::Expr> {
    let source = "type Chain =\n  | End { value: f32 }\n  | Link { next: Chain }\n\
@opaque\n@invariant(c) c.seed >= 0.0\ntype Carrier =\n  | Carrier { seed: f32, next: Chain }\n";
    let decls = chelis_surf::parser::parse_str(source).expect("parse Carrier declaration");
    chelis_surf::desugar::desugar_program(&decls).expect("desugar Carrier declaration")
}

fn scalar_f32(value: f32) -> ExecutionValue {
    serde_json::from_value(serde_json::json!({
        "type": "scalar", "value": {"dtype": "f32", "bits": format!("{:08x}", value.to_bits())}
    }))
    .expect("f32 scalar")
}

fn run_on_small_stack_in_child(test_name: &str, probe: impl FnOnce() + Send + 'static) {
    if std::env::var_os(CHILD_ENV).is_some() {
        std::thread::Builder::new()
            .stack_size(2 * 1024 * 1024)
            .spawn(probe)
            .expect("spawn 2 MiB embedding worker")
            .join()
            .expect("embedding worker returned");
        return;
    }

    let result = Command::new(std::env::current_exe().expect("test binary"))
        .args(["--exact", test_name, "--nocapture"])
        .env(CHILD_ENV, "1")
        .output()
        .expect("run isolated execution-value probe");
    assert!(result.status.success(), "{result:?}");
}

#[test]
fn standalone_execution_value_drops_all_recursive_variants_on_small_stack() {
    run_on_small_stack_in_child(
        "standalone_execution_value_drops_all_recursive_variants_on_small_stack",
        || {
            let mut value = ExecutionValue::Unit;
            for index in 0..20_000 {
                value = match index % 4 {
                    0 => ExecutionValue::List { value: vec![value] },
                    1 => ExecutionValue::Tuple { value: vec![value] },
                    2 => ExecutionValue::Adt {
                        ctor: "Link".to_string(),
                        fields: vec![value],
                    },
                    _ => ExecutionValue::Dict {
                        entries: vec![DictEntryValue {
                            key: ExecutionValue::Unit,
                            value,
                        }],
                    },
                };
            }
            drop(value);
        },
    );
}

#[test]
fn shallow_list_has_the_canonical_recursive_wire_shape() {
    let value = nested_list(1, ExecutionValue::Unit);
    let encoded = serde_json::to_value(&value).expect("serialize shallow value");
    assert_eq!(
        encoded,
        serde_json::json!({"type":"list","value":[{"type":"unit"}]})
    );
}

#[test]
fn custom_serializer_preserves_every_execution_variant_wire_shape() {
    let fixtures = [
        serde_json::json!({"type":"tensor","value":{"shape":[1],"data":{"dtype":"f32","bits":["3f800000"]}}}),
        serde_json::json!({"type":"scalar","value":{"dtype":"int64","value":7}}),
        serde_json::json!({"type":"bool","value":true}),
        serde_json::json!({"type":"key","bits":"0000000000000007"}),
        serde_json::json!({"type":"string","value":"hello"}),
        serde_json::json!({"type":"list","value":[{"type":"unit"}]}),
        serde_json::json!({"type":"dict","entries":[{"key":{"type":"unit"},"value":{"type":"bool","value":false}}]}),
        serde_json::json!({"type":"tuple","value":[{"type":"unit"}]}),
        serde_json::json!({"type":"adt","ctor":"Link","fields":[{"type":"unit"}]}),
        serde_json::json!({"type":"unit"}),
    ];
    for expected in fixtures {
        let value: ExecutionValue =
            serde_json::from_value(expected.clone()).expect("decode canonical value");
        assert_eq!(
            serde_json::to_value(&value).expect("encode canonical value"),
            expected
        );
    }
}

#[test]
fn ordinary_serde_serialization_survives_a_deep_value_on_a_small_stack() {
    run_on_small_stack_in_child(
        "ordinary_serde_serialization_survives_a_deep_value_on_a_small_stack",
        || {
            let value = nested_list(5_000, ExecutionValue::Unit);
            let wire = serde_json::to_vec(&value).expect("serialize deep value");
            assert_eq!(wire.iter().filter(|byte| **byte == b'[').count(), 5_000);
            assert_eq!(wire.iter().filter(|byte| **byte == b']').count(), 5_000);
            assert!(
                wire.windows(b"\"type\":\"unit\"".len())
                    .any(|slice| slice == b"\"type\":\"unit\"")
            );
            drop(value);
        },
    );
}

#[test]
fn ordinary_serde_deserialization_rejects_excessive_depth_without_aborting() {
    run_on_small_stack_in_child(
        "ordinary_serde_deserialization_rejects_excessive_depth_without_aborting",
        || {
            let wire = nested_list_wire(5_000, "{\"type\":\"unit\"}");
            let error = serde_json::from_str::<ExecutionValue>(&wire)
                .expect_err("ordinary JSON decoder must bound excessive depth");
            assert!(
                error.to_string().contains("recursion limit exceeded"),
                "{error}"
            );
        },
    );
}

#[test]
fn unbounded_serde_deserialization_survives_deep_valid_and_invalid_values() {
    run_on_small_stack_in_child(
        "unbounded_serde_deserialization_survives_deep_valid_and_invalid_values",
        || {
            for (leaf, accepted) in [
                ("{\"type\":\"unit\"}", true),
                ("{\"type\":\"missing\"}", false),
            ] {
                let wire = nested_list_wire(5_000, leaf);
                let mut decoder = serde_json::Deserializer::from_str(&wire);
                decoder.disable_recursion_limit();
                let result = ExecutionValue::deserialize(&mut decoder);
                if accepted {
                    let value = result.expect("decode deep valid value");
                    decoder.end().expect("complete deep JSON");
                    drop(value);
                } else {
                    let error = result.expect_err("reject deep invalid variant");
                    assert!(error.to_string().contains("unknown variant"), "{error}");
                }
            }
        },
    );
}

#[test]
fn structural_decode_accepts_a_deep_list_on_a_small_stack() {
    run_on_small_stack_in_child(
        "structural_decode_accepts_a_deep_list_on_a_small_stack",
        || {
            let value = nested_list(5_000, ExecutionValue::Unit);
            let decoded = try_decode_adt_value(&[], &value).expect("decode deep list");
            drop(decoded);
            drop(value);
        },
    );
}

#[test]
fn structural_decode_rejects_a_deep_invalid_leaf_without_aborting() {
    run_on_small_stack_in_child(
        "structural_decode_rejects_a_deep_invalid_leaf_without_aborting",
        || {
            let value = nested_list(
                20_000,
                ExecutionValue::Adt {
                    ctor: "Missing".to_string(),
                    fields: vec![],
                },
            );
            let error = try_decode_adt_value(&[], &value).expect_err("unknown ctor rejected");
            assert!(matches!(error, DecodeError::Structural(_)), "{error}");
            drop(value);
        },
    );
}

#[test]
fn nested_adt_decode_accepts_and_rejects_deep_values_on_a_small_stack() {
    run_on_small_stack_in_child(
        "nested_adt_decode_accepts_and_rejects_deep_values_on_a_small_stack",
        || {
            let program = chain_program();
            let valid = nested_adt(
                5_000,
                ExecutionValue::Adt {
                    ctor: "End".to_string(),
                    fields: vec![
                        serde_json::from_value(serde_json::json!({
                            "type": "scalar", "value": {"dtype": "int64", "value": 0}
                        }))
                        .expect("int64 leaf"),
                    ],
                },
            );
            let decoded = try_decode_adt_value(&program, &valid).expect("decode deep Chain");
            drop(decoded);
            drop(valid);

            let invalid = nested_adt(
                5_000,
                ExecutionValue::Adt {
                    ctor: "Missing".to_string(),
                    fields: vec![],
                },
            );
            let error = try_decode_adt_value(&program, &invalid).expect_err("reject leaf");
            assert!(matches!(error, DecodeError::Structural(_)), "{error}");
            drop(invalid);
        },
    );
}

#[test]
fn opaque_carrier_checks_deep_adt_representation_without_overflow() {
    run_on_small_stack_in_child(
        "opaque_carrier_checks_deep_adt_representation_without_overflow",
        || {
            let program = opaque_carrier_program();
            let chain = nested_adt(
                5_000,
                ExecutionValue::Adt {
                    ctor: "End".to_string(),
                    fields: vec![scalar_f32(0.0)],
                },
            );
            let good = ExecutionValue::Adt {
                ctor: "Carrier".to_string(),
                fields: vec![scalar_f32(0.25), chain],
            };
            let decoded = try_decode_adt_value(&program, &good).expect("accept finite Carrier");
            drop(decoded);
            drop(good);

            let bad = ExecutionValue::Adt {
                ctor: "Carrier".to_string(),
                fields: vec![
                    scalar_f32(-0.25),
                    nested_adt(
                        5_000,
                        ExecutionValue::Adt {
                            ctor: "End".to_string(),
                            fields: vec![scalar_f32(0.0)],
                        },
                    ),
                ],
            };
            let error = try_decode_adt_value(&program, &bad).expect_err("reject Carrier");
            assert!(matches!(error, DecodeError::Invariant(_)), "{error}");
            drop(bad);

            let nonfinite = ExecutionValue::Adt {
                ctor: "Carrier".to_string(),
                fields: vec![
                    scalar_f32(0.25),
                    nested_adt(
                        5_000,
                        ExecutionValue::Adt {
                            ctor: "End".to_string(),
                            fields: vec![scalar_f32(f32::NAN)],
                        },
                    ),
                ],
            };
            let error = try_decode_adt_value(&program, &nonfinite)
                .expect_err("reject non-finite representation");
            let DecodeError::Invariant(message) = error else {
                panic!("expected invariant error, got {error}");
            };
            let field_path = format!("next.{}value", "next.".repeat(5_000));
            assert!(message.contains("opaque type `Carrier`"), "{message}");
            assert!(message.contains(&field_path), "{message}");
            drop(nonfinite);
        },
    );
}
