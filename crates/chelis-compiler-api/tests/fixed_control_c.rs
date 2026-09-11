//! Native execution of the sealed, unfused fixed-control backend entry.
//! These are not compiler-API source admission or certificate-transport tests.
#[allow(dead_code)]
mod ownership_support;

use chelis_ir::dag::{DimInfo, TensorType};
use chelis_ir::evaluation::{EvaluationPlan, RandomExecutionContext};
use chelis_ir::host::RandomLoweringState;
use chelis_types::types::Prim;
use chelis_unord::UnordMap;

fn plan(source: &str, prim: Prim, count: usize) -> EvaluationPlan {
    let inputs = [(
        "x".into(),
        TensorType {
            dims: vec![DimInfo::Lit(count)],
            precision: prim,
        },
    )]
    .into_iter()
    .collect();
    plan_with_inputs(source, inputs)
}

fn plan_with_inputs(source: &str, inputs: UnordMap<String, TensorType>) -> EvaluationPlan {
    let expr = chelis_deep::parser::parse_str(source)
        .unwrap()
        .pop()
        .unwrap();
    chelis_ir::lower::try_lower_subexpr_evaluation_plan(
        &expr,
        inputs,
        UnordMap::new(),
        UnordMap::new(),
        &RandomExecutionContext::new(RandomLoweringState {
            seed: Some(42),
            counter: 0,
        }),
    )
    .unwrap()
}

#[test]
fn native_empty_and_zero_rate_calls_consume_one_ordinal_before_the_next_draw() {
    // Independent [05-RNG-1] reference: neither the evaluator nor its sampler
    // supplies these expected bits. The unused first result must still draw.
    fn mix(mut word: u64) -> u64 {
        word = word.wrapping_add(0x9e37_79b9_7f4a_7c15);
        word = (word ^ (word >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        word = (word ^ (word >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        word ^ (word >> 31)
    }
    for seed in [42_i64, -1, i64::MIN] {
        let expected = (0..32_u64)
            .map(|index| {
                let word = mix((seed as u64) ^ mix(1).rotate_left(17) ^ mix(index).rotate_left(41));
                let unit = ((word >> 11) as f64 / ((1_u64 << 53) as f64)) as f32;
                if unit < 0.5 { "0u" } else { "0x40000000u" }
            })
            .collect::<Vec<_>>()
            .join(",");
        for count in [0, 32] {
            for rate in ["0.0", "0.5"] {
                let source = format!(
                    "(handle-effect {{effect: random}} (lit {{type: (t-prim {{}} int64)}} {seed}) \
                     (let {{}} (bind {{}} discarded (app {{}} (var {{}} dropout) \
                     (var {{}} prefix) (lit {{type: (t-prim {{}} f32)}} {rate}))) \
                     (app {{}} (var {{}} dropout) (var {{}} x) \
                     (lit {{type: (t-prim {{}} f32)}} 0.5))))"
                );
                let inputs = [("prefix", count), ("x", 32)]
                    .into_iter()
                    .map(|(name, extent)| {
                        (
                            name.into(),
                            TensorType {
                                dims: vec![DimInfo::Lit(extent)],
                                precision: Prim::F32,
                            },
                        )
                    })
                    .collect();
                let artifact = chelis_backend_c::codegen_evaluation_with_options(
                    plan_with_inputs(&source, inputs)
                        .verify_ownership()
                        .unwrap(),
                    "sample",
                    Default::default(),
                )
                .unwrap();
                assert_eq!(artifact.input_labels, ["prefix", "x"]);
                let driver = format!(
                    r#"
int main(void) {{
    int64_t prefix_n = {count}, n = 32;
    chelis_tensor *inputs[] = {{
        chelis_alloc(1, &prefix_n, CHELIS_DTYPE_F32),
        chelis_alloc(1, &n, CHELIS_DTYPE_F32)
    }};
    for (int i = 0; i < 2; ++i) {{
        chelis_tensor_write *write = chelis_tensor_begin_write(inputs[i]);
        chelis_fill_scalar(write, chelis_scalar_from_bits(CHELIS_DTYPE_F32, 0x3f800000u));
        chelis_tensor_end_write(write);
    }}
    const uint32_t expected[] = {{{expected}}};
    for (int repeat = 0; repeat < 4; ++repeat) {{
        chelis_tensor *outputs[1];
        sample(inputs, 2, outputs, 1);
        chelis_read_view view = chelis_tensor_read_view(outputs[0]);
        assert(view.dtype == CHELIS_DTYPE_F32 && view.count == n);
        assert(memcmp(view.data, expected, sizeof(expected)) == 0);
        chelis_tensor_release(outputs[0]);
    }}
    chelis_tensor_release(inputs[0]);
    chelis_tensor_release(inputs[1]);
    return 0;
}}
"#
                );
                ownership_support::balanced(&ownership_support::run(&artifact.c_source, &driver));
            }
        }
    }
}

#[test]
fn sealed_native_dropout_matches_evaluator_and_restarts_each_public_invocation() {
    for (dtype, prim, tag, ctype) in [
        ("f32", Prim::F32, "CHELIS_DTYPE_F32", "uint32_t"),
        ("f64", Prim::F64, "CHELIS_DTYPE_F64", "uint64_t"),
        ("f16", Prim::F16, "CHELIS_DTYPE_F16", "uint16_t"),
        ("bf16", Prim::Bf16, "CHELIS_DTYPE_BF16", "uint16_t"),
    ] {
        for count in [0, 1, 32] {
            let source = format!(
                "(handle-effect {{effect: random}} (lit {{type: (t-prim {{}} int64)}} 42) (app {{}} (var {{}} dropout) (var {{}} x) (lit {{type: (t-prim {{}} {dtype})}} 0.1)))"
            );
            let plan = plan(&source, prim, count);
            let input = chelis_types::finalize_tensor(
                "test",
                prim,
                chelis_types::RawTensor::Float(
                    (0..count).map(|i| (i as f64 + 1.0) / 7.0).collect(),
                ),
            )
            .unwrap();
            let mut context = RandomExecutionContext::new(RandomLoweringState {
                seed: Some(42),
                counter: 0,
            });
            let values = chelis_ir::eval::eval_tensor_plan_with_strict(&plan, &mut context, |_| {
                Some(chelis_ir::eval::TensorValue::from_storage(
                    vec![count],
                    input.clone(),
                ))
            })
            .unwrap();
            let output = &values[&plan.dag_for_inspection().roots()[0]];
            let bits = |values: Vec<f64>| {
                let words = values
                    .into_iter()
                    .map(|x| {
                        let bits = match prim {
                            Prim::F32 => (x as f32).to_bits() as u64,
                            Prim::F64 => x.to_bits(),
                            Prim::F16 => half::f16::from_f64(x).to_bits() as u64,
                            Prim::Bf16 => half::bf16::from_f64(x).to_bits() as u64,
                            _ => unreachable!(),
                        };
                        format!("0x{bits:x}ULL")
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                if words.is_empty() { "0".into() } else { words }
            };
            let expected = bits(output.to_f64_lossy_vec());
            let input_bits = bits(input.to_f64_lossy_vec());
            let artifact = chelis_backend_c::codegen_evaluation_with_options(
                plan.verify_ownership().unwrap(),
                "sample",
                Default::default(),
            )
            .unwrap();
            let driver = format!(
                r#"
int main(void) {{
    int64_t n = {count};
    const {ctype} original[] = {{{input_bits}}};
    chelis_scalar elements[{capacity}];
    for (int64_t i = 0; i < n; ++i)
        elements[i] = chelis_scalar_from_bits({tag}, original[i]);
    chelis_tensor *x = chelis_alloc(1, &n, {tag});
    chelis_tensor_write *write = chelis_tensor_begin_write(x);
    chelis_tensor_write_literal(write, chelis_scalar_from_bits(CHELIS_DTYPE_I64, n), elements);
    chelis_tensor_end_write(write);
    const {ctype} expected[] = {{{expected}}};
    for (int repeat = 0; repeat < 4; ++repeat) {{
        chelis_tensor *outputs[1];
        sample(&x, 1, outputs, 1);
        chelis_read_view view = chelis_tensor_read_view(outputs[0]);
        assert(view.dtype == {tag} && view.count == n);
        if (n) assert(memcmp(view.data, expected, n * sizeof({ctype})) == 0);
        chelis_tensor_release(outputs[0]);
        if (n) assert(memcmp(chelis_tensor_read_view(x).data, original, n * sizeof({ctype})) == 0);
    }}
    chelis_tensor_release(x);
    return 0;
}}
"#,
                capacity = count.max(1),
            );
            ownership_support::balanced(&ownership_support::run(&artifact.c_source, &driver));
            if prim == Prim::F32 && count == 32 {
                let reciprocal = artifact.c_source.replace(
                    " / chelis_f32_from_bits(0x3f666666u)",
                    " * (1.0f / chelis_f32_from_bits(0x3f666666u))",
                );
                assert_ne!(reciprocal, artifact.c_source);
                assert_native_value_failure(&reciprocal, &driver);
            }
        }
    }
}

#[test]
fn native_replay_nested_restore_and_next_uniform_follow_source_steps() {
    for body in [
        "with seed(42i64) { grad(loss)(x) }",
        "with seed(42i64) {\n dead = dropout(x, 0.0f32)\n _ = drop(dead)\n grad(loss)(x)\n}",
        "with seed(42i64) {\n inner = with seed(99i64) { dropout(x, 0.5f32) }\n _ = drop(inner)\n g = grad(loss)(x)\n add(g, uniform_like(x, 0.0f32, 1.0f32))\n}",
        "with seed(42i64) {\n g = grad(loss)(x)\n inner = with seed(42i64) { dropout(x, 0.5f32) }\n _ = drop(inner)\n add(g, dropout(x, 0.5f32))\n}",
    ] {
        let source = format!(
            "def loss(x: tensor[32, f32]) -> tensor[f32] ! {{ Random }} = sum(dropout(x, 0.5f32), 0)\ndef sample(x: tensor[32, f32]) -> tensor[32, f32] = {body}\n"
        );
        let parsed = chelis_surf::parser::parse_str(&source).unwrap();
        let checked =
            chelis_types::check_ir_program(&chelis_surf::desugar::desugar_program(&parsed))
                .unwrap();
        let mut context = RandomExecutionContext::new(RandomLoweringState {
            seed: Some(7),
            counter: 13,
        });
        // The ordinary host selector intentionally cuts at lexical handlers.
        // This backend test lowers the checked body itself; host/API transport
        // is a separate integration obligation, not silently tested here.
        fn children(expr: &chelis_deep::Expr) -> &[chelis_deep::Expr] {
            match expr {
                chelis_deep::Expr::Node(node, _) => node.children_slice(),
                chelis_deep::Expr::List(list, _) => &list.elements[2..],
                _ => panic!("tagged checked expression"),
            }
        }
        let defs: UnordMap<String, chelis_deep::Expr> = checked
            .exprs()
            .iter()
            .filter_map(|expr| {
                if expr.tag() != Some(chelis_deep::tag::DeepTag::Def) {
                    return None;
                }
                let children = children(expr);
                let chelis_deep::Expr::Atom(chelis_deep::ast::Atom::Name(name), _) = &children[0]
                else {
                    panic!("def name");
                };
                Some((name.clone(), children[1].clone()))
            })
            .collect();
        let sample = defs
            .to_sorted()
            .into_iter()
            .find(|(name, _)| name.as_str() == "sample" || name.ends_with(".sample"))
            .expect("checked sample definition")
            .1;
        assert_eq!(sample.tag(), Some(chelis_deep::tag::DeepTag::Fn));
        let plan = chelis_ir::lower::try_lower_subexpr_evaluation_plan(
            &children(sample)[1],
            [(
                "x".into(),
                TensorType {
                    dims: vec![DimInfo::Lit(32)],
                    precision: Prim::F32,
                },
            )]
            .into_iter()
            .collect(),
            UnordMap::new(),
            defs.clone(),
            &context,
        )
        .unwrap();
        assert!(
            plan.dag_for_inspection()
                .nodes()
                .iter()
                .filter(|node| matches!(node.op, chelis_ir::dag::RiscOp::Dropout { .. }))
                .count()
                >= 2
        );
        let values = chelis_ir::eval::eval_tensor_plan_with_strict(&plan, &mut context, |_| {
            Some(chelis_ir::eval::TensorValue::from_vec(
                vec![32],
                vec![1.0; 32],
            ))
        })
        .unwrap();
        assert_eq!(context.state().seed, Some(7));
        assert_eq!(context.state().counter, 13);
        let expected = values[&plan.dag_for_inspection().roots()[0]]
            .to_f64_lossy_vec()
            .into_iter()
            .map(|value| format!("0x{:x}u", (value as f32).to_bits()))
            .collect::<Vec<_>>()
            .join(",");
        let artifact = chelis_backend_c::codegen_evaluation_with_options(
            plan.verify_ownership().unwrap(),
            "sample",
            Default::default(),
        )
        .unwrap();
        let driver = format!(
            r#"
int main(void) {{
    int64_t n = 32;
    chelis_tensor *x = chelis_alloc(1, &n, CHELIS_DTYPE_F32);
    chelis_tensor_write *write = chelis_tensor_begin_write(x);
    chelis_fill_scalar(write, chelis_scalar_from_bits(CHELIS_DTYPE_F32, 0x3f800000u));
    chelis_tensor_end_write(write);
    const uint32_t expected[] = {{{expected}}};
    for (int repeat = 0; repeat < 4; ++repeat) {{
        chelis_tensor *outputs[1];
        sample(&x, 1, outputs, 1);
        chelis_read_view view = chelis_tensor_read_view(outputs[0]);
        assert(view.dtype == CHELIS_DTYPE_F32 && view.count == n);
        assert(memcmp(view.data, expected, sizeof(expected)) == 0);
        chelis_tensor_release(outputs[0]);
    }}
    chelis_tensor_release(x);
    return 0;
}}
"#
        );
        ownership_support::balanced(&ownership_support::run(&artifact.c_source, &driver));
        // These native corruptions must be caught by complete value comparison
        // and by the existing ownership ledger, not by emitted-text assertions.
        if body.contains("dead =") {
            let wrong_draw = artifact.c_source.replace(
                "__chelis_fixed_counter++",
                "(__chelis_fixed_counter++ + 1ULL)",
            );
            assert_ne!(wrong_draw, artifact.c_source);
            assert_native_value_failure(&wrong_draw, &driver);
            let release = artifact
                .c_source
                .lines()
                .find(|line| line.contains("chelis_tensor_release(t"))
                .expect("actual owned intermediate cleanup");
            let missing_cleanup = artifact.c_source.replacen(release, "", 1);
            let leaked = ownership_support::run(&missing_cleanup, &driver);
            assert!(leaked["live_owners"].as_u64().unwrap() > 0);
        }
    }
}

fn assert_native_value_failure(source: &str, driver: &str) {
    let failure = std::panic::catch_unwind(|| ownership_support::run(source, driver))
        .expect_err("mutation must fail the executed C value assertion");
    let message = failure
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| failure.downcast_ref::<&str>().copied())
        .expect("failure message");
    assert!(
        message.starts_with("C status"),
        "must fail execution, not compilation: {message}"
    );
}

#[test]
fn native_dropout_preserves_signed_zero_and_nonfinite_classes() {
    // [05-OP-37]: a dropped value is +0, even for NaN/infinity; kept
    // values use division, preserving -0. This is IEEE conformance evidence,
    // not part of the rational pathwise derivative theorem.
    let input: Vec<f32> = (0..32)
        .map(|i| {
            [
                0.0,
                -0.0,
                -1.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                1.0,
            ][i % 7]
        })
        .collect();
    for rate in ["0.0", "0.5"] {
        let source = format!(
            "(handle-effect {{effect: random}} (lit {{type: (t-prim {{}} int64)}} 42) (app {{}} (var {{}} dropout) (var {{}} x) (lit {{type: (t-prim {{}} f32)}} {rate})))"
        );
        let plan = plan(&source, Prim::F32, input.len());
        let mut context = RandomExecutionContext::new(RandomLoweringState {
            seed: Some(42),
            counter: 13,
        });
        let values = chelis_ir::eval::eval_tensor_plan_with_strict(&plan, &mut context, |_| {
            Some(chelis_ir::eval::TensorValue::from_vec(
                vec![input.len()],
                input.iter().map(|&value| f64::from(value)).collect(),
            ))
        })
        .unwrap();
        let expected: Vec<f32> = values[&plan.dag_for_inspection().roots()[0]]
            .to_f64_lossy_vec()
            .into_iter()
            .map(|value| value as f32)
            .collect();
        assert_eq!(context.state().seed, Some(42));
        assert_eq!(context.state().counter, 13);
        if rate == "0.0" {
            assert_eq!(expected[1].to_bits(), (-0.0_f32).to_bits());
            assert!(expected[3].is_infinite() && expected[5].is_nan());
        } else {
            assert!(
                input
                    .iter()
                    .zip(&expected)
                    .any(|(before, after)| { !before.is_finite() && after.to_bits() == 0 })
            );
        }
        let artifact = chelis_backend_c::codegen_evaluation_with_options(
            plan.verify_ownership().unwrap(),
            "sample",
            Default::default(),
        )
        .unwrap();
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| format!("0x{:x}u", value.to_bits()))
                .collect::<Vec<_>>()
                .join(",")
        };
        let driver = format!(
            r#"
int main(void) {{
    int64_t n = 32;
    const uint32_t input[] = {{{input}}};
    const uint32_t expected[] = {{{expected}}};
    chelis_scalar elements[32];
    for (int i = 0; i < 32; ++i)
        elements[i] = chelis_scalar_from_bits(CHELIS_DTYPE_F32, input[i]);
    chelis_tensor *x = chelis_alloc(1, &n, CHELIS_DTYPE_F32);
    chelis_tensor_write *write = chelis_tensor_begin_write(x);
    chelis_tensor_write_literal(write, chelis_scalar_from_bits(CHELIS_DTYPE_I64, 32), elements);
    chelis_tensor_end_write(write);
    for (int repeat = 0; repeat < 4; ++repeat) {{
        chelis_tensor *outputs[1];
        sample(&x, 1, outputs, 1);
        chelis_read_view view = chelis_tensor_read_view(outputs[0]);
        assert(view.dtype == CHELIS_DTYPE_F32 && view.count == n);
        const uint32_t *actual = view.data;
        for (int i = 0; i < 32; ++i) {{
            if ((expected[i] & 0x7fffffffu) > 0x7f800000u)
                assert((actual[i] & 0x7fffffffu) > 0x7f800000u);
            else
                assert(actual[i] == expected[i]);
        }}
        chelis_tensor_release(outputs[0]);
    }}
    chelis_tensor_release(x);
    return 0;
}}
"#,
            input = bits(&input),
            expected = bits(&expected),
        );
        ownership_support::balanced(&ownership_support::run(&artifact.c_source, &driver));
    }
}

#[test]
fn public_native_entry_does_not_invent_an_inherited_random_context() {
    let plan = plan(
        "(app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.5))",
        Prim::F32,
        32,
    );
    let error = chelis_backend_c::codegen_evaluation_with_options(
        plan.verify_ownership().unwrap(),
        "sample",
        Default::default(),
    )
    .err()
    .expect("unhandled export must reject");
    assert!(error.to_string().contains("inherited Random"), "{error}");
}

#[test]
fn native_entry_rejects_invalid_stored_rates_even_for_empty_inputs() {
    for (dtype, prim) in [
        ("f16", Prim::F16),
        ("bf16", Prim::Bf16),
        ("f32", Prim::F32),
        ("f64", Prim::F64),
    ] {
        for rate in ["-0.5", "1.0"] {
            let source = format!(
                "(handle-effect {{effect: random}} (lit {{type: (t-prim {{}} int64)}} 42) \
                 (app {{}} (var {{}} dropout) (var {{}} x) \
                 (lit {{type: (t-prim {{}} {dtype})}} {rate})))"
            );
            let error = chelis_backend_c::codegen_evaluation_with_options(
                plan(&source, prim, 0).verify_ownership().unwrap(),
                "sample",
                Default::default(),
            )
            .err()
            .expect("invalid static rate cannot produce an artifact");
            assert!(error.to_string().contains("domain in dropout"), "{error}");
        }
    }
}
