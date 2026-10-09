//! Deep public execution values must survive embedding input decoding and drop.

use chelis_compiler_api::schema::{DictEntryValue, ExecutionValue};
use chelis_compiler_api::{DecodeError, try_decode_adt_value};
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
